"""
object_memory.py — what the robot knows about each object, kept across scans,
its own actions and restarts.

A scan says what is where NOW. It does not say which of the things it sees is
the one the arm put down a minute ago, which one was measured last time, or
what the user said a thing was. This keeps one record per physical object — a
track — and matches every scan against the tracks, so what was learned about an
object stays with the object when it moves, and "put it back" has a "back".

Nothing here knows about any task. A deployment hands observe() detections —
where something stands on the work surface, plus whatever was measured about it
— and optionally the name of the place it stands on (a chess square, a tray
slot). It gets back the same objects with stable ids and everything known about
them. Labels and attributes are free text: "knight" and {"colour": "black"}
for chess, "mug" and {"handle": "left"} for something else.

How a scan is matched to the tracks (observe)
  Every way a detection could be a remembered object is priced, and the
  cheapest explanations are believed first:
    stayed   the detection is within NEAR_M of where the object was: costs only
             how different it looks (signature_distance), up to SAME_PLACE_MAX.
             The foot is judged loosely here — one outline that took in a patch
             of table reads 10 mm wide, and that is not a new object — but
             anything measured up close is only trusted while the foot still
             agrees (Track.shape_current).
    moved    it is somewhere else: costs MOVE_COST more, and only an object
             that looks the same (up to MOVE_MAX) can have moved. Objects
             already missing can come back the same way. Not between an
             object something has measured and a sighting nothing has (see
             Detection.confirmed), either way round: a vision model's guess
             carries a colour at most, which matches any object of that colour.
  Pricing both together is what gets a capture right. When a pawn takes the
  knight on e5, e5 is still occupied; "the knight stayed but now looks like a
  pawn" costs more than "the pawn that vanished from d4 moved to e5".
  A move is only believed when it is clearly better than any rival — unless
  the rivals are indistinguishable anyway (two white pawns), when which is
  which makes no difference, and the move is marked uncertain.
  Detections left over are new objects. Objects left over become "missing":
  kept, with their last place, not forgotten — a piece taken off the board is
  still something the user can ask about, and comes back as itself.
  What a scan took by look alone — a new object, a move, a return — is a
  guess (guessed()): two blocks of one size and colour look the same to it,
  and only the letter on top tells them apart. Something that can read it
  (the vision model) can say the sighting is another remembered object, and
  reidentify() gives it to that one.

The arm's own actions are certain and are recorded directly: picked(),
placed(), measured(). A held object is not looked for on the table.

Labels come from several sources that deserve different trust. A higher one is
never overwritten by a lower one; within the vision model's guesses, the label
it gave most often wins (its answer for the same piece varies scan to scan).

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import copy
import functools
import json
import threading
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from mrcf import flight_recorder

# Who said what an object is, least to most trusted. A label from a source
# below the current one's is ignored.
LABEL_SOURCES = ("vlm", "shape", "taught", "user")
_VOTING_SOURCES = {"vlm"}

# A detection this close to where a track was last seen is that track.
NEAR_M = 0.015
PLACED_NEAR_M = 0.030         # where the arm has just put it down: a place lands up to ~2 cm off

# How different two measurements may be and still be the same object, per
# feature. Distances are in units of these: 1.0 is "at the edge of plausible".
# Feet repeat to ~1 mm scan to scan on this rig (2-3 mm for some pieces);
# heights from the gripper camera to a few mm.
FOOT_TOL_MM = 4.0
SAME_PLACE_FOOT_TOL_MM = 10.0  # standing where it stood: only a gross change counts
HEIGHT_TOL_MM = 6.0
APPEARANCE_TOL = 12.0         # Lab units over whatever channels both carry
ATTRIBUTE_MISMATCH = 1.5      # one attribute (colour) disagreeing
LABEL_MISMATCH = 2.0          # two trusted labels disagreeing

# Stayed: one attribute misread plus ordinary noise is still the same object;
# that and a foot 13 mm wider is not.
SAME_PLACE_MAX = ATTRIBUTE_MISMATCH + 0.25
MOVE_COST = 1.0               # believing an object moved, over believing it stayed
MOVE_MAX = 1.0                # moved: signature alone, so be strict
AMBIGUITY_MARGIN = 0.35       # a move must beat any rival by this much
MISSING_PENALTY = 0.1         # an object gone before this scan is a slightly worse fit

# A measured shape (height, widths) belongs to the object whose foot it was
# measured with. If the foot a scan sees now differs by more than this, the
# shape is not trusted for motion: the object may have been swapped in place.
SHAPE_FOOT_TOL_MM = 4.0

_KEEP_OBSERVATIONS = 12       # per track, for the running medians
_ATTRIBUTE_WINDOW = 3         # attributes: majority of this many recent scans
_KEEP_HISTORY = 50            # events per track
_MISSING_SHOWN = 6            # objects not seen now, named to the planner (describe)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _median(values) -> "float | None":
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def _consensus(values, tol: float) -> "tuple[float | None, int]":
    """
    What most of `values` agree on: (the median of the largest group within
    tol of one of them, how many are in it); ties go to the group round the
    most recent. (None, 0) when there are none. Unlike a median of everything,
    one wild value among two good ones does not move it, and between one good
    and one wild value it says so (1 agreeing) instead of splitting the
    difference — a 41 mm pawn measured 59.6 and then 41 has a median of 50.
    """
    vals = [float(v) for v in values if v is not None]
    best = None
    for i, v in enumerate(vals):
        group = [w for w in vals if abs(w - v) <= tol]
        if best is None or (len(group), i) > best[0]:
            best = ((len(group), i), group)
    return (None, 0) if best is None else (_median(best[1]), len(best[1]))


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass
class Detection:
    """One object as a scan saw it."""
    position: tuple                          # (x, y) on the work surface, robot frame, m
    place: "str | None" = None               # the deployment's name for where it stands
    foot_mm: "float | None" = None           # diameter of what it stands on
    height_mm: "float | None" = None
    appearance: dict = field(default_factory=dict)   # numeric colour summary, e.g. Lab {"L", "a", "b"}
    attributes: dict = field(default_factory=dict)   # e.g. {"colour": "black"}
    label: "str | None" = None
    label_source: "str | None" = None
    extra: dict = field(default_factory=dict)        # kept on the track as the latest
    # False: only a source that can report what is not there (a vision model)
    # saw it, and one that measures what is there (an outline) looked and found
    # nothing. A vision model can put objects where there are none; taken as
    # real, each would block picks as a neighbour, and match as a "move" of a
    # real object, which would then lose its identity to it.
    confirmed: bool = True


@dataclass
class Track:
    """One physical object, as remembered."""
    id: str
    state: str = "present"
    position: "list | None" = None
    place: "str | None" = None
    label: "str | None" = None
    label_source: "str | None" = None
    label_votes: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)
    attribute_seen: dict = field(default_factory=dict)   # key -> recent observed values
    pinned: list = field(default_factory=list)       # attributes a person or a lesson set
    feet_mm: list = field(default_factory=list)
    heights_mm: list = field(default_factory=list)
    widths_mm: dict = field(default_factory=dict)    # the measured profile in use, height -> width
    shapes: list = field(default_factory=list)       # the arm's looks: [{"height_mm", "widths_mm"}]
    measured_foot_mm: "float | None" = None          # the scan's foot when it was measured
    refoot: bool = False                             # the arm put it down: the next scan finds it
                                                     # there whatever its foot reads, and that foot
                                                     # is the one to compare with (placed)
    # Something since its measurement says what stands here may be another
    # object: it moved, not by the arm, or was out of sight and came back, or
    # its colour or label changed.
    # The foot alone cannot tell (a 42 mm pawn and a 52 mm piece differ by 2 mm),
    # so shape_current() is False until the arm measures it again.
    shape_doubted: bool = False
    appearances: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    # Ever seen by a confirmed sighting (Detection.confirmed), or held by the
    # arm. Kept once true: a piece the outline misses on one scan is still there.
    # True for tracks saved before the flag existed.
    confirmed: bool = True
    # Standing on another object, where the arm put it (a block on a block):
    # that track's id, and how high its foot is above the work surface. A scan
    # sees a stack as one object, as tall as all of it, where the one on top is.
    on: "str | None" = None
    raised_mm: float = 0.0
    first_seen: str = ""
    last_seen: str = ""
    history: list = field(default_factory=list)

    @property
    def foot_mm(self) -> "float | None":
        return _median(self.feet_mm)

    @property
    def height_mm(self) -> "float | None":
        return _consensus(self.heights_mm, HEIGHT_TOL_MM)[0]

    @property
    def appearance(self) -> dict:
        keys = set().union(*self.appearances) if self.appearances else set()
        return {k: _median(a.get(k) for a in self.appearances) for k in keys}

    @property
    def measured(self) -> bool:
        """Has the arm measured its height (not just the camera its foot)?"""
        return bool(self.heights_mm)

    def shape(self) -> "dict | None":
        """
        The measured shape to move by: {"height_mm", "widths_mm", "looks"} —
        the height most measurements agree on, the profile of the arm's look
        nearest it, and how many agree (a policy looks again while that is
        few). None when never measured.
        """
        if not self.measured:
            return None
        height, looks = _consensus(self.heights_mm, HEIGHT_TOL_MM)
        widths = self.widths_mm
        if self.shapes:
            widths = min(self.shapes, key=lambda sh: abs(sh["height_mm"] - height))["widths_mm"]
        return {"height_mm": height, "widths_mm": widths, "looks": looks}

    def shape_current(self) -> bool:
        """
        Measured, and still the object it was measured as: nothing has cast
        doubt on it since (shape_doubted), and the latest scan's foot agrees
        with the foot it had then. Motion should not trust a shape otherwise —
        a different object of a similar look may stand there now.
        """
        if not self.measured or self.shape_doubted:
            return False
        if self.measured_foot_mm is None or not self.feet_mm:
            return True
        return abs(self.feet_mm[-1] - self.measured_foot_mm) <= SHAPE_FOOT_TOL_MM

    def label_rank(self) -> int:
        return LABEL_SOURCES.index(self.label_source) if self.label_source in LABEL_SOURCES else -1

    def unstack(self) -> None:
        """Standing on nothing: on the surface, or not on the table at all."""
        self.on, self.raised_mm = None, 0.0

    def _event(self, event: str, **details) -> None:
        self.history.append({"time": _now(), "event": event, **_plain(details)})
        del self.history[:-_KEEP_HISTORY]


def signature_distance(a, b, foot_tol_mm: float = FOOT_TOL_MM) -> "float | None":
    """
    How different two objects look, from whatever both carry: 0 identical, 1 at
    the edge of plausible, more is a different object. None when they share
    nothing to compare. Differences add as a root sum of squares: an object
    can't make up for the wrong colour by having the right foot, and one
    feature at the edge of plausible is believable where two at once are not.

    a and b are Tracks or Detections.
    """
    terms = []
    fa, fb = a.foot_mm, b.foot_mm
    if fa is not None and fb is not None:
        terms.append(abs(fa - fb) / foot_tol_mm)
    ha, hb = a.height_mm, b.height_mm
    if ha is not None and hb is not None:
        terms.append(abs(ha - hb) / HEIGHT_TOL_MM)
    look = _look_apart(a, b)
    if look is not None:
        terms.append(look)
    for k in set(a.attributes) & set(b.attributes):
        if a.attributes[k] is not None and b.attributes[k] is not None:
            terms.append(0.0 if a.attributes[k] == b.attributes[k] else ATTRIBUTE_MISMATCH)
    # Only labels someone vouches for count: the vision model's guesses are
    # too often wrong to split one object into two.
    trusted = LABEL_SOURCES.index("shape")
    if (a.label and b.label and a.label_source in LABEL_SOURCES and b.label_source in LABEL_SOURCES
            and min(LABEL_SOURCES.index(a.label_source), LABEL_SOURCES.index(b.label_source)) >= trusted):
        terms.append(0.0 if a.label == b.label else LABEL_MISMATCH)
    return float(np.linalg.norm(terms)) if terms else None


def _look_apart(a, b) -> "float | None":
    """How different two objects' colours are to the camera, 1 at the edge of plausible; None: nothing to compare."""
    pa, pb = a.appearance, b.appearance
    common = [k for k in pa if k in pb and pa[k] is not None and pb[k] is not None]
    return float(np.linalg.norm([pa[k] - pb[k] for k in common])) / APPEARANCE_TOL if common else None


def _alike(a, b) -> bool:
    """
    Nothing tells these two apart (Tracks, or Detections): the same attributes,
    and the same label where anyone better than the vision model gave one.
    """
    def vouched(x):
        trusted = x.label_source in LABEL_SOURCES and LABEL_SOURCES.index(x.label_source) >= 1
        return x.label if trusted else None
    return vouched(a) == vouched(b) and a.attributes == b.attributes


def _as_top_of_stack(d: Detection, track: Track) -> Detection:
    """
    d as the object on top of a stack would be by itself: a scan measures the
    stack's height, the foot raised_mm up adds that much under it. d itself
    for anything standing on the surface.
    """
    if not track.on or d.height_mm is None:
        return d
    return replace(d, height_mm=max(float(d.height_mm) - track.raised_mm, 0.0))


def default_namer(track: Track) -> str:
    parts = [str(v) for v in track.attributes.values() if v] + [track.label or "object"]
    if track.state == "present" and track.place:
        parts.append(track.place)
    else:
        parts.append(track.id.replace("-", ""))
    return "_".join(p.replace(" ", "_") for p in parts)


def _locked(method):
    """Run an ObjectMemory method holding the memory's lock."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class ObjectMemory:
    """
    Every object the robot has seen, by stable id. Saved to `path` (JSON) after
    every change when a path is given, and loaded from it on construction
    unless load is False: a task whose surface is rearranged between runs (the
    tabletop's) starts from what its first scan sees, and the file is written
    over.

    namer(track) -> str gives each track the name the planner sees; the
    deployment's naming convention, so plans and memory agree on names.

    Safe to share between threads — a scan on one, the arm's own picks and
    places on another: every method runs under one lock, so a scan's matching
    sees the tracks either before an action's update or after it, not halfway.
    """

    def __init__(self, path: "str | Path | None" = None, namer=None, load: bool = True):
        self.path = Path(path) if path else None
        self.namer = namer or default_namer
        self.tracks: dict = {}
        self._next = 1
        self._lock = threading.RLock()
        # The latest scan's guesses: track id -> (the track as it was before
        # that scan, None for a new one; the sighting it was given).
        self._guessed: dict = {}
        self._lost_now: set = set()          # tracks that scan left missing
        if load and self.path is not None and self.path.is_file():
            self._load()

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        assert self.path is not None        # called only when there is a file to read
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            print(f"  [Memory] Could not read {self.path} ({exc}) — starting empty.")
            return
        known = set(Track.__dataclass_fields__)
        for t in data.get("tracks", []):
            track = Track(**{k: v for k, v in t.items() if k in known})
            track.widths_mm = {int(k): v for k, v in track.widths_mm.items()}   # JSON keys are strings
            for sh in track.shapes:
                sh["widths_mm"] = {int(k): v for k, v in sh.get("widths_mm", {}).items()}
            self.tracks[track.id] = track
        self._next = int(data.get("next", len(self.tracks) + 1))

    @_locked
    def save(self) -> None:
        if self.path is None:
            return
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_plain({"saved": _now(), "next": self._next,
                                              "tracks": [asdict(t) for t in self.tracks.values()]}),
                                      indent=1))
            tmp.replace(self.path)
        except OSError as exc:
            print(f"  [Memory] Could not save {self.path} ({exc}).")

    @_locked
    def clear(self) -> None:
        """Forget every object — the scene was reset or rearranged wholesale."""
        self.tracks.clear()
        self._guessed, self._lost_now = {}, set()
        self.save()

    # ── Lookup ──────────────────────────────────────────────────────────────

    @_locked
    def get(self, track_id: "str | None") -> "Track | None":
        return self.tracks.get(track_id) if track_id else None

    @_locked
    def name(self, track: Track) -> str:
        return self.namer(track)

    @_locked
    def in_state(self, state: str) -> list:
        return [t for t in self.tracks.values() if t.state == state]

    # ── Scans ───────────────────────────────────────────────────────────────

    @_locked
    def observe(self, detections: list, hidden=None) -> "tuple[list, dict]":
        """
        Match one scan against the tracks (see the module docstring) and
        update them. Returns (tracks, report): tracks[i] is detections[i]'s
        track; report lists what changed — {"new", "moved", "returned",
        "missing", "hidden", "replaced"}, each a list of (track, detail) pairs.
        hidden(xy) -> bool: where this scan could not see (the arm in the
        camera's way): a present object not seen there stays present, not
        missing.
        """
        report: dict[str, list] = {"new": [], "moved": [], "returned": [], "missing": [], "hidden": [],
                                   "replaced": []}
        self._guessed, self._lost_now = {}, set()
        out: list = [None] * len(detections)
        open_dets = set(range(len(detections)))
        present = [t for t in self.tracks.values() if t.state == "present" and t.position is not None]

        # Every plausible explanation of every detection, with its price.
        pairs = []
        for t in present + self.in_state("missing"):
            for i, d in enumerate(detections):
                here = (t.state == "present"
                        and np.hypot(t.position[0] - d.position[0], t.position[1] - d.position[1])
                        <= (PLACED_NEAR_M if t.refoot else NEAR_M))
                if here:
                    # On top of a stack, what is seen there is as tall as the
                    # stack: this one is that, less what it stands on.
                    d = _as_top_of_stack(d, t)
                    # Just put there by the arm (placed): it is there — up to
                    # PLACED_NEAR_M off, whatever its foot reads from that spot.
                    # Judged like any other sighting, a placed object can read
                    # as a new one, its old self left "missing" on that spot
                    # and blocking it, or its name go to another object.
                    sig = signature_distance(t, d, foot_tol_mm=float("inf") if t.refoot
                                             else SAME_PLACE_FOOT_TOL_MM)
                    if sig is None or sig <= SAME_PLACE_MAX:
                        pairs.append((sig or 0.0, t.id, i, False))
                    continue
                if t.confirmed != d.confirmed:
                    continue
                if (hidden is not None and t.state == "present" and t.position is not None
                        and hidden(t.position[:2])):
                    continue                     # where it stands is hidden this scan: not seen elsewhere either
                sig = signature_distance(t, d)
                if sig is not None and sig <= MOVE_MAX:
                    # Of two equally good fits, the object that was on the
                    # table a moment ago, not one that left it earlier.
                    cost = MOVE_COST + sig + (MISSING_PENALTY if t.state == "missing" else 0.0)
                    pairs.append((cost, t.id, i, True))
        pairs.sort(key=lambda p: p[0])

        taken: set = set()
        moved_dets: set = set()
        guessed: set = set()
        guessed_dets: set = set()
        for cost, tid, i, moved in pairs:
            if tid in taken or i not in open_dets:
                continue
            track, d = self.tracks[tid], detections[i]
            if moved:
                # Anything else about as good — another object for this
                # detection, or another detection for this object — makes the
                # move a guess, unless the two are indistinguishable anyway.
                near = [(r, j) for c, r, j, _ in pairs
                        if c < cost + AMBIGUITY_MARGIN and (r, j) != (tid, i)
                        and r not in taken and j in open_dets]
                rival_tracks = [self.tracks[r] for r, j in near if j == i]
                rival_dets = [detections[j] for r, j in near if r == tid]
                if (any(not _alike(r, track) for r in rival_tracks)
                        or any(not _alike(r, d) for r in rival_dets)):
                    continue
                # Which of two identical objects went where is a guess, for the
                # second pairing as much as the first.
                uncertain = bool(rival_tracks or rival_dets) or tid in guessed or i in guessed_dets
                self._guessed[tid] = (copy.deepcopy(track), d)
                guessed.update(r.id for r in rival_tracks)
                guessed_dets.update(j for r, j in near if r == tid)
                kind = "returned" if track.state == "missing" else "moved"
                track.shape_doubted = True         # not moved by the arm: what is there may be another
                report[kind].append((track, {"from": track.place, "to": d.place, "uncertain": uncertain}))
                track._event(kind, by="camera", **{"from": track.place, "to": d.place}, uncertain=uncertain)
            out[i] = track
            if moved:
                moved_dets.add(i)
            taken.add(tid)
            open_dets.discard(i)
        # What stands under an object still seen where it was — all the way
        # down its stack — is out of sight, not gone.
        covered: set = set()
        for i, track in enumerate(out):
            below = track.on if track is not None and i not in moved_dets else None
            while below and below not in covered and below in self.tracks:
                covered.add(below)
                below = self.tracks[below].on
        open_tracks = {t.id for t in present} - taken - covered

        # New.
        for i in sorted(open_dets):
            track = self._new_track()
            out[i] = track
            self._guessed[track.id] = (None, detections[i])
            report["new"].append((track, {"at": detections[i].place}))

        # Fold every detection into its track.
        stamp = _now()
        for i, (d, track) in enumerate(zip(detections, out)):
            if track.on and i in moved_dets:
                track.unstack()      # moved off its stack, by hand: on the surface now
            self._absorb(track, _as_top_of_stack(d, track), stamp)
            if any(track is t for t, _ in report["new"]):
                # Something new where a remembered object stood: say so, that
                # is how a swap made by hand shows up.
                was = next((t for t in present if t.id in open_tracks and t.place and t.place == d.place), None)
                if was is not None:
                    report["replaced"].append((track, {"replaced": was.id, "at": d.place}))
                track._event("first seen", at=d.place)

        # Gone — or where the scan could not see: there still, as far as anyone knows.
        unplaced = []
        for tid in list(open_tracks):
            track = self.tracks[tid]
            if hidden is not None and track.position is not None and hidden(track.position[:2]):
                track._event("hidden", at=track.place)
                report["hidden"].append((track, {"at": track.place}))
                open_tracks.discard(tid)
        for tid in open_tracks:
            track = self.tracks[tid]
            track.state = "missing"
            track.unstack()
            track._event("not seen", last_at=track.place)
            self._lost_now.add(tid)
            report["missing"].append((track, {"last_at": track.place}))
            if track.refoot:
                unplaced.append(track)

        self.save()
        flight_recorder.event("memory", "scan", **{k: [f"{self.namer(t)} {_plain(d)}" for t, d in v]  # type: ignore[arg-type]  # keys are report's kinds, never "image"
                                                   for k, v in report.items() if v})
        for track in unplaced:
            # The arm put it down there, and the first look since does not see
            # it: it fell, rolled, or was never let go of where the arm said.
            # A place can report success for an object that fell out on the
            # way, so this is where that shows.
            flight_recorder.RECORDER.incident(f"{self.namer(track)} not seen where it was put ({track.place})",
                                              flight_recorder.RECORDER.last_action("place"),
                                              object=self.namer(track), place=track.place)
        return out, report

    @_locked
    def guessed(self) -> set:
        """Ids of the tracks the latest scan gave a sighting by look alone: new, moved by hand, or back."""
        return set(self._guessed)

    @_locked
    def checked(self, track_id: str) -> None:
        """Something that can tell objects apart has looked at it, and it is who the scan said: no longer a guess."""
        self._guessed.pop(track_id, None)

    @_locked
    def reidentify(self, track_id: str, label: str, why: str = "", alike=None) -> "Track | None":
        """
        The latest scan gave track_id a sighting by look alone (guessed()), and
        something that can read what the scan cannot (the vision model, the
        letter on a block) calls it `label`. If an object that scan did not see
        is called that, and looks like the sighting to the camera, that one
        takes the sighting, and track_id is as it was before that scan, not
        seen there (a new one is forgotten). The camera's colour must agree as
        well as the label: the model misreads letters (a U on its side read
        as D), and the blocks of a set carry a different one on each face.

        alike(track) -> bool: for a sighting that is new (no remembered object
        taken for it), the objects it may be when none is called `label` — of
        its kind and colour (a block's letter misread): one of them, and the
        camera's colour agreeing, it is that one. Not for a sighting the scan
        gave to an object: that one stays itself unless the label says else.

        Returns the track that has the sighting now; None when there is no
        such object, or more than one alike, or track_id was not a guess.
        """
        got = self._guessed.get(track_id)
        if got is None or self.tracks.get(track_id) is None or self.tracks[track_id].state != "present":
            return None
        before, d = got
        missing = [t for t in self.in_state("missing") if t.id != track_id]
        named = [t for t in missing if t.label == label]
        if not named and alike is not None and before is None:
            named = [t for t in missing if alike(t)]
        fits = []
        for t in named:
            apart = _look_apart(t, d)
            if apart is None or apart <= 1.0:
                fits.append((apart or 0.0, t))
        fits.sort(key=lambda x: x[0])
        if not fits or (len(fits) > 1 and fits[1][0] - fits[0][0] < AMBIGUITY_MARGIN):
            return None
        really = fits[0][1]
        really_id = really.id
        del self._guessed[track_id]
        if before is None:
            del self.tracks[track_id]
        else:
            before.state = "missing"
            before.unstack()
            before._event("not seen", last_at=before.place)
            self.tracks[track_id] = before
            self._lost_now.add(track_id)
        # Missing only since this scan: it was on the table a moment ago, and
        # has been moved by hand; missing before it, it is back.
        kind = "moved" if really_id in self._lost_now else "returned"
        if kind == "moved" and really.history and really.history[-1]["event"] == "not seen":
            really.history.pop()
        self._lost_now.discard(really_id)
        was = really.place
        really.shape_doubted = True
        self._absorb(really, d, _now())
        really._event(kind, by="camera", **{"from": was, "to": d.place}, uncertain=False, why=why)
        self.save()
        flight_recorder.event("memory", "reidentified", track=self.namer(really), was=track_id, why=why)
        return really

    def _new_track(self) -> Track:
        track = Track(id=f"obj-{self._next}", first_seen=_now(), confirmed=False)    # _absorb says
        self._next += 1
        self.tracks[track.id] = track
        return track

    def _absorb(self, track: Track, d: Detection, stamp: str) -> None:
        track.state = "present"
        track.position = [float(d.position[0]), float(d.position[1])]
        track.place = d.place
        track.last_seen = stamp
        track.confirmed = track.confirmed or d.confirmed
        if d.foot_mm is not None:
            track.feet_mm = (track.feet_mm + [float(d.foot_mm)])[-_KEEP_OBSERVATIONS:]
            if track.refoot:
                if track.measured_foot_mm is not None:
                    track.measured_foot_mm = float(d.foot_mm)
                track.refoot = False
        if d.height_mm is not None:
            track.heights_mm = (track.heights_mm + [float(d.height_mm)])[-_KEEP_OBSERVATIONS:]
        if d.appearance:
            track.appearances = (track.appearances + [dict(d.appearance)])[-_KEEP_OBSERVATIONS:]
        # Attributes go by the majority of the last few scans: one misreading
        # does not flip an object's colour, a real change wins within two.
        for k, v in d.attributes.items():
            if v is None or k in track.pinned:
                continue
            seen = (track.attribute_seen.get(k, []) + [v])[-_ATTRIBUTE_WINDOW:]
            track.attribute_seen[k] = seen
            was = track.attributes.get(k)
            track.attributes[k] = max(seen, key=lambda x: (seen.count(x), x == v))
            if was is not None and track.attributes[k] != was:
                track.shape_doubted = True
        if d.label:
            was = track.label
            self._offer_label(track, d.label, d.label_source or "vlm")
            if was is not None and track.label != was:
                track.shape_doubted = True
        if d.extra:
            track.extra.update(d.extra)

    @_locked
    def add(self, d: Detection, why: str = "") -> Track:
        """
        One object a scan did not outline, found another way (by what it is
        called: a deployment's look_around), as a track of its own. The next
        scan matches it as any other; one that cannot see it leaves it missing.
        """
        track = self._new_track()
        self._absorb(track, d, _now())
        track._event("found", why=why)
        self.save()
        return track

    @_locked
    def part_of(self, track_id: str, of: str, why: str) -> None:
        """A track that was a part of another object all along (a plug's pin): gone, said why."""
        track = self.get(track_id)
        if track is None or track.state == "gone":
            return
        track.state = "gone"
        track._event("part of", of=of, why=why)
        self.save()

    # ── What the arm did, and what people said ──────────────────────────────

    @_locked
    def picked(self, track_id: str) -> None:
        track = self.get(track_id)
        if track is None:
            return
        self._guessed.pop(track_id, None)         # the arm has acted on it since: not the scan's guess now
        track.state = "held"
        track.unstack()
        track.confirmed = True                 # the jaws closed on it
        track._event("picked", by="arm", **{"from": track.place})
        self.save()

    @_locked
    def restack(self) -> None:
        """
        Each object on a stack, its foot at the top of what is under it as tall
        as that is known to be now: a height learned since it was put there (a
        known size corrected, a measurement) moves everything above it.
        """
        def top_mm(tid, seen=()):
            t = self.tracks.get(tid)
            if t is None or t.height_mm is None or tid in seen:
                return None
            under = top_mm(t.on, seen + (tid,)) if t.on else 0.0
            return None if under is None else under + float(t.height_mm)

        changed = False
        for t in self.tracks.values():
            if t.on and t.state == "present":
                foot = top_mm(t.on)
                if foot is not None and abs(foot - t.raised_mm) > 0.5:
                    t.raised_mm, changed = foot, True
        if changed:
            self.save()

    @_locked
    def placed(self, track_id: str, position, place: "str | None" = None, on: "str | None" = None,
               raised_mm: float = 0.0) -> None:
        """Put down by the arm at position (x, y); on another track (stacked), its foot raised_mm up."""
        track = self.get(track_id)
        if track is None:
            return
        self._guessed.pop(track_id, None)         # the arm has acted on it since: not the scan's guess now
        last = next((e.get("from") for e in reversed(track.history) if e["event"] == "picked"), None)
        track.state = "present"
        track.position = [float(position[0]), float(position[1])]
        track.place = place
        track.on, track.raised_mm = (on, float(raised_mm)) if on else (None, 0.0)
        track.last_seen = _now()
        # The arm carried it here, so it is still the object it measured; a
        # scan reads its foot differently from another spot (a 69 mm queen:
        # 27 mm on one square, 17 on another), and that difference alone must
        # not drop its shape. Its foot here, as the next scan reads it, is the
        # one to tell a swapped object by; and that scan finds it here (observe).
        track.refoot = True
        track._event("placed", by="arm", **{"from": last, "to": place})
        self.save()

    @_locked
    def moved(self, track_id: str, position, how: str = "") -> None:
        """
        The arm moved it without holding it — a push, a nudge: where it stands
        now. Which place that is, the next scan says.
        """
        track = self.get(track_id)
        if track is None:
            return
        self._guessed.pop(track_id, None)         # the arm has acted on it since: not the scan's guess now
        was = track.place
        track.position = [float(position[0]), float(position[1])]
        track.place = None
        track.unstack()
        track.last_seen = _now()
        track._event("moved", by="arm", how=how, **{"from": was})
        self.save()

    @_locked
    def superseded(self, track_id: str, by: str, why: str) -> None:
        """
        A missing object that was the one `by` all along — seen since under
        that track, on the same spot: no longer kept clear of (state "gone").
        """
        track = self.get(track_id)
        if track is None or track.state != "missing":
            return
        track.state = "gone"
        track._event("superseded", by=by, why=why)
        self.save()

    @_locked
    def released(self, track_id: str, why: str) -> None:
        """A held object is no longer known to be held, and not known to be anywhere."""
        track = self.get(track_id)
        if track is None or track.state != "held":
            return
        self._guessed.pop(track_id, None)         # the arm has acted on it since: not the scan's guess now
        track.state = "missing"
        track.unstack()
        track._event("lost", why=why)
        self.save()

    @_locked
    def measured(self, track_id: str, height_mm: "float | None" = None,
                 widths_mm: "dict | None" = None, foot_mm: "float | None" = None, looks: int = 1) -> None:
        """
        A close measurement by the arm (gripper camera) — better than a scan's.
        looks: how many separate looks agreed on it (a height looked for from
        several sides), each counted toward the consensus (at most 3).
        """
        track = self.get(track_id)
        if track is None:
            return
        if height_mm is not None:
            if not track.shape_current():
                track.heights_mm, track.shapes = [], []   # those were of whatever stood here before
            track.heights_mm = (track.heights_mm + [float(height_mm)] * max(1, min(int(looks), 3)))[-_KEEP_OBSERVATIONS:]
            track.measured_foot_mm = track.feet_mm[-1] if track.feet_mm else None
            track.shape_doubted = False
            if widths_mm:
                look = {"height_mm": float(height_mm),
                        "widths_mm": {int(k): float(v) for k, v in widths_mm.items()}}
                track.shapes = (track.shapes + [look])[-_KEEP_OBSERVATIONS:]
        if widths_mm:
            # The profile that goes with the height most looks agree on — not
            # simply the latest, which may be the one wild look.
            shape = track.shape()
            track.widths_mm = (shape["widths_mm"] if shape and shape["widths_mm"]
                               else {int(k): float(v) for k, v in widths_mm.items()})
        track.confirmed = True                 # the wrist camera outlined it
        track._event("measured", height_mm=height_mm, foot_mm=foot_mm)
        self.save()

    @_locked
    def set_label(self, track_id: str, label: "str | None", source: str = "user",
                  attributes: "dict | None" = None) -> bool:
        """
        Say what an object is. From "user" or "taught", attributes given here
        are pinned: later scans don't change them. False if a more trusted
        source already labelled it differently.
        """
        track = self.get(track_id)
        if track is None:
            return False
        changed = self._offer_label(track, label, source) if label else False
        for k, v in (attributes or {}).items():
            track.attributes[k] = v
            if source in ("user", "taught") and k not in track.pinned:
                track.pinned.append(k)
            changed = True
        if changed:
            track._event("labelled", label=track.label, attributes=attributes, source=source)
            self.save()
        return changed

    def _offer_label(self, track: Track, label: str, source: str) -> bool:
        rank = LABEL_SOURCES.index(source) if source in LABEL_SOURCES else 0
        if rank < track.label_rank():
            return False
        if source in _VOTING_SOURCES:
            track.label_votes[label] = track.label_votes.get(label, 0) + 1
            label = max(track.label_votes, key=lambda x: (track.label_votes[x], x == label))
        else:
            track.label_votes = {}
        changed = (label, source) != (track.label, track.label_source)
        track.label, track.label_source = label, source
        return changed

    # ── For the planner ─────────────────────────────────────────────────────

    @_locked
    def certainty(self, track: Track) -> str:
        """How the label is known, in words, for descriptions."""
        return {"user": "named by the user", "taught": "taught", "shape": "recognised by shape",
                "vlm": "vision model's guess"}.get(track.label_source, "not identified")  # type: ignore[arg-type]  # None gives the default

    @_locked
    def describe(self, max_events: int = 6) -> str:
        """
        What the planner should know beyond the current scene: what the arm
        did recently, what changed by hand, and what is no longer seen.
        """
        lines = []
        events = []
        for t in self.tracks.values():
            for e in t.history:
                if e["event"] in ("placed", "moved", "returned", "not seen", "labelled", "lost"):
                    events.append((e["time"], t, e))
        events.sort(key=lambda x: x[0])
        if events:
            lines.append("Recent events (oldest first):")
            for stamp, t, e in events[-max_events:]:
                lines.append(f"  {stamp[11:16]} {self._event_text(t, e)}")
        # The most recently seen few: every planning prompt carries this, and
        # the list only grows (a day's blocks, gone since the morning).
        missing = sorted(self.in_state("missing"), key=lambda t: t.last_seen, reverse=True)
        if missing:
            shown = missing[:_MISSING_SHOWN]
            lines.append("Remembered but not seen now: " + ", ".join(
                f"{self.name(t)} (last on {t.place})" if t.place else self.name(t) for t in shown)
                + (f", and {len(missing) - len(shown)} more" if len(missing) > len(shown) else ""))
        return "\n".join(lines)

    def _event_text(self, t: Track, e: dict) -> str:
        n = self.name(t)
        ev = e["event"]
        if ev == "placed":
            return f"the arm moved {n} from {e.get('from') or '?'} to {e.get('to') or '?'}"
        if ev in ("moved", "returned"):
            guess = " (or an identical one)" if e.get("uncertain") else ""
            verb = "came back, on" if ev == "returned" else f"was moved by hand from {e.get('from') or '?'} to"
            return f"{n}{guess} {verb} {e.get('to') or '?'}"
        if ev == "not seen":
            return f"{n} disappeared from {e.get('last_at') or '?'}"
        if ev == "lost":
            return f"{n} was no longer in the gripper ({e.get('why')})"
        return f"{n} was identified ({e.get('source')})"
