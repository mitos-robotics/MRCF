"""
containers.py — things to put objects into (a tub, a bowl, a tray), found by
what someone calls them, as the policies' Opening.

The surface scan (scene_scan.py) finds objects, things the size of an object;
a container is often bigger, thin-walled or see-through, and never among them
(2026-11-17: the SO-101's clear plastic tub). Here one is found by name in a
fixed camera's picture (perception/grounding.py: a vision model's box, the
segmenter's outline inside it), and its outline carried down onto the work
surface.

Where its rim is
----------------
A camera above at an angle sees a container's outline as the rim on the side
away from it and the foot of the near wall on the side toward it. Neither side
alone says how high the walls are. So the outline is carried onto the surface
twice: at the surface's own height, where its near side is (the wall's foot),
and at the rim's (rim_m — measured, or the tallest the walls may be), where its
far side is. Each carried the wrong way lands further out: the far side at the
surface, beyond the rim; the near side up at the rim, toward the camera — and
the camera can be outside the container on that side (the SO-101's overhead
over the tub, 2026-11-17). The opening is where the two agree, their overlap
(opening_xy): it errs inward, never out — what a place that must let go inside
it needs. The same holds for an object the scan outlined that is a container
(a bowl): its outline from the scan, its height as the rim.

Nothing here moves the arm or knows the task. Who keeps the containers, names
them for the planner and looks again is the deployment's (tabletop/skills.py).

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from mrcf.policies.base import Opening

# The outline kept in a camera's pixels is simplified to within this (px): a
# few dozen corners, not a thousand.
OUTLINE_SIMPLIFY_PX = 2.0


@dataclass
class Container:
    """A container, as found: its outline in one fixed camera, and how high its walls are."""
    name: str                         # the scene's name for it ("plastic_tub")
    what: str                         # what it was looked for as ("the plastic tub")
    camera: str                       # the fixed camera that outlined it
    size: tuple                       # (w, h) of that camera's picture
    outline_px: np.ndarray            # (N, 2) its outline in that picture
    rim_m: float                      # its walls above the surface: measured, or the tallest they may be
    rim_measured: bool = False
    found_at: float = field(default_factory=time.time)
    box: "tuple | None" = None        # the vision model's box round it, px
    score: float = 0.0                # the segmenter's estimate of the outline

    def outline_xy(self, model: dict, at_m: "float | None" = None) -> "np.ndarray | None":
        """Its outline on the surface's plane (world x, y, metres), each pixel carried to at_m above it (rim_m)."""
        return outline_on_plane(model, self.outline_px, self.rim_m if at_m is None else at_m)

    def opening_xy(self, model: dict) -> "np.ndarray | None":
        """Its opening on the surface's plane, inward of its rim everywhere: the outline carried at the surface
        and at the rim, where they overlap (see the module docstring)."""
        low, high = self.outline_xy(model, at_m=0.0), self.outline_xy(model)
        if low is None or high is None or len(low) < 3 or len(high) < 3:
            return None
        return overlap(low, high)

    def opening(self, model: dict) -> "Opening | None":
        """As the policies take it (policies.base.Opening); None when the camera's model cannot carry it down."""
        xy = self.opening_xy(model)
        if xy is None or len(xy) < 3:
            return None
        return Opening(xy, float(self.rim_m), self.rim_measured, label=self.name)

    def contains(self, model: dict, xy, margin_m: float = 0.0) -> bool:
        """Whether a foot at xy stands inside its outline on the surface's own plane (by margin_m): generously."""
        from mrcf.policies.place_in import inside_by

        poly = self.outline_xy(model, at_m=0.0)
        return poly is not None and len(poly) >= 3 and inside_by(poly, xy) >= margin_m

    def centre_xy(self, model: dict) -> "np.ndarray | None":
        poly = self.opening_xy(model)
        return None if poly is None or not len(poly) else poly.mean(axis=0)


# The overlap of two outlines is drawn at this resolution (m a pixel).
OVERLAP_RES_M = 0.001


def overlap(a, b, res_m: float = OVERLAP_RES_M) -> "np.ndarray | None":
    """The overlap of two outlines on a plane ((N, 2) metres each): its own outline, or None when they do not meet."""
    import cv2

    a, b = np.asarray(a, float).reshape(-1, 2), np.asarray(b, float).reshape(-1, 2)
    lo = np.minimum(a.min(axis=0), b.min(axis=0)) - 2 * res_m
    hi = np.maximum(a.max(axis=0), b.max(axis=0)) + 2 * res_m
    w, h = (np.ceil((hi - lo) / res_m).astype(int) + 1)
    if w * h > 4_000_000:                     # more than 2 m square at 1 mm: not a container on a table
        return None
    canvas = []
    for poly in (a, b):
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round((poly - lo) / res_m).astype(np.int32)], 1)
        canvas.append(m)
    both = (canvas[0] & canvas[1]).astype(np.uint8)
    found, _ = cv2.findContours(both, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return None
    biggest = max(found, key=cv2.contourArea)
    simple = cv2.approxPolyDP(biggest, 2.0, True).reshape(-1, 2).astype(float)
    return lo + simple * res_m if len(simple) >= 3 else None


def outline_on_plane(model: dict, px, height_m: float) -> "np.ndarray | None":
    """(N, 2) world (x, y) where each pixel's ray meets the plane height_m above the surface; None if none does."""
    from mrcf.perception.plane_geometry import PlaneView

    view = PlaneView.from_camera_model(model)
    raised = PlaneView(view.K, view.D, view.R, view.t + view.R @ np.array([0.0, 0.0, float(height_m) * 1000.0]),
                       scale=view.scale, origin_m=view.origin_m)
    xy = raised.to_plane(np.asarray(px, float).reshape(-1, 2))
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not len(xy):
        return None
    origin = np.zeros(2) if raised.origin_m is None else np.asarray(raised.origin_m, float)[:2]
    return origin[None, :] + xy / 1000.0


def outline_px(mask: np.ndarray) -> "np.ndarray | None":
    """The outer outline of the biggest part of a mask, simplified (OUTLINE_SIMPLIFY_PX): (N, 2) px."""
    import cv2

    m = np.asarray(mask, bool).astype(np.uint8)
    found, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not found:
        return None
    biggest = max(found, key=cv2.contourArea)
    simple = cv2.approxPolyDP(biggest, OUTLINE_SIMPLIFY_PX, True).reshape(-1, 2)
    return simple.astype(float) if len(simple) >= 3 else None


def open_top(frame_rgb: np.ndarray, mask: np.ndarray) -> "np.ndarray | None":
    """
    The opening of an open box within its outline from above: the part in its
    own shade — the inside walls and floor, darker than the flaps and rim that
    face the light — the darker of the outline's two lightnesses (Otsu), its
    largest piece, its convex hull. None when that is not a believable opening
    (under a tenth of the outline, or nearly all of it): the outline itself
    then. 2026-11-20, the UR5's cardboard box, its four flaps folded out: the
    whole of it taken for the opening, a screwdriver let go over a flap
    (user: "it should realise not to drop it on the flap").
    """
    import cv2

    m = np.asarray(mask, bool)
    if m.sum() < 400:
        return None
    L = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2LAB)[:, :, 0]
    t, _ = cv2.threshold(L[m].reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = ((L <= t) & m).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(out, [cv2.convexHull(max(cs, key=cv2.contourArea))], 1)
    share = float(out.sum()) / float(m.sum())
    return out.astype(bool) if 0.10 <= share <= 0.85 else None


def find(frame_rgb: np.ndarray, camera: str, name: str, what: str, *, model: str, rim_m: float,
         judge_model: "str | None" = None, convention: "str | None" = None, ask=None,
         segment_box=None, opening_by_shade: bool = False) -> "tuple[Container | None, str]":
    """
    The container called `what` in one fixed camera's frame: (Container, "")
    or (None, why not). model: the vision model that boxes it; judge_model:
    the one that says whether it is a container at all (None: not asked —
    found again, it was asked when first found). rim_m: how high its walls may
    be, at most (not measured). ask, segment_box: grounding.find's, for tests.
    """
    from mrcf.perception import grounding

    if ask is None:
        from mrcf.vlm import call_vlm as ask
    found, why = grounding.find(frame_rgb, what, model=model, convention=convention, ask=ask, segment_box=segment_box,
                                judge_model=judge_model)
    if found is None or found.mask is None:
        return None, why or "nothing was outlined"
    opening = open_top(frame_rgb, found.mask) if opening_by_shade else None
    px = outline_px(opening if opening is not None else found.mask)
    if px is None:
        return None, "its outline has no shape"
    h, w = frame_rgb.shape[:2]
    return Container(name, what, camera, (w, h), px, float(rim_m), False, box=found.box, score=found.score), ""


def from_outline(name: str, camera: str, outline: dict, rim_m: float, rim_measured: bool) -> "Container | None":
    """
    A container from an object's outline as the scan kept it
    ({"size": (w, h), "px": [[u, v], ...]}, Target.outlines' entry for one
    camera): a bowl the scan found as an object. rim_m: its height.
    """
    px = np.asarray(outline.get("px") or [], float).reshape(-1, 2)
    if len(px) < 3:
        return None
    return Container(name, name.replace("_", " "), camera, tuple(outline.get("size") or (0, 0)), px,
                     float(rim_m), rim_measured)


# ── Keeping them for a deployment ─────────────────────────────────────────────

_ARTICLES = {"the", "a", "an", "my", "that", "this", "into", "in", "inside", "to"}


def name_for(what) -> str:
    """The scene's name for a container the user calls `what`: "the clear plastic tub" -> clear_plastic_tub."""
    import re

    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", str(what).lower().replace("_", " ")).split() if w not in _ARTICLES]
    return "_".join(words) or "container"


class Keeper:
    """
    The containers a deployment has found, by name, for the session: found by
    what the user calls them in a fixed camera's picture (find), found again
    at each look (refresh), offered to the planner as places to put things
    INTO (targets), told what stands in which (holds), and handed to the
    policies (opening). Their walls' height, once measured (measured), is
    kept on disk (rims_path) and given back when the same container is found
    again, after a restart too — a container's walls do not change. One kept
    on disk is looked for again at each look until found (remembered,
    refresh): 2026-11-18, a session's first place into the tub parked the
    arm high to look for it, and measured its walls on the way.

    look(env) -> (frame, camera model): the fixed camera's picture now, at its
    size; the deployment's, with the arm where it wants it for the look.
    """

    def __init__(self, camera: str, look, *, grounding_model: str, judge_model: "str | None",
                 tallest_m: float, table_z: float, rims_path=None, keep_picture=None, ask=None,
                 opening_by_shade: bool = False):
        self.camera, self.look = camera, look
        self.grounding_model, self.judge_model = grounding_model, judge_model
        self.tallest_m, self.table_z = float(tallest_m), float(table_z)
        self.opening_by_shade = bool(opening_by_shade)   # an open box's opening by its shade (open_top)
        self.kept: dict = {}                     # name -> Container
        self.model: "dict | None" = None         # the fixed camera's model, as last read
        self.rims_path = rims_path
        self.keep_picture, self.ask = keep_picture, ask

    # ── what is kept ──────────────────────────────────────────────────────────

    def known(self, name) -> "Container | None":
        """The kept container called name: exactly, else the one whose name's words hold all of name's ("tub")."""
        key = name_for(name)
        if key in self.kept:
            return self.kept[key]
        words = set(key.split("_"))
        hits = [c for n, c in self.kept.items() if words <= set(n.split("_")) or set(n.split("_")) <= words]
        return hits[0] if len(hits) == 1 else None

    def _rims(self) -> dict:
        import json
        from pathlib import Path

        if self.rims_path is None:
            return {}
        try:
            return dict(json.loads(Path(self.rims_path).read_text()))
        except (OSError, ValueError, TypeError):
            return {}

    def remembered(self) -> dict:
        """
        The containers kept on disk from an earlier session and not found yet
        in this one: {name: what it was looked for as}. Looked for again at
        each look (refresh), their walls as measured then: a place into one
        never stops to look for it, nor to measure its walls again.
        """
        return {n: str(r.get("what") or n.replace("_", " ")) for n, r in self._rims().items()
                if n not in self.kept and isinstance(r, dict)}

    def _save_rim(self, name: str, rim_m: float, what: "str | None" = None) -> None:
        import json
        import time as _time
        from pathlib import Path

        if self.rims_path is None:
            return
        rims = self._rims()
        rims[name] = {**(rims.get(name) or {}), "rim_m": round(float(rim_m), 4),
                      "measured": _time.strftime("%Y-%m-%d %H:%M")}
        if what:
            rims[name]["what"] = str(what)
        Path(self.rims_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.rims_path).write_text(json.dumps(rims, indent=1) + "\n")

    def _keep(self, found: "Container", old: "Container | None") -> None:
        if old is not None and old.rim_measured:
            found.rim_m, found.rim_measured = old.rim_m, True
        else:
            kept = self._rims().get(found.name)
            # Measured only where it was (_save_rim stamps "measured"): _save_where keeps the walls' height
            # beside the outline whatever it is, the tallest they may be too. 2026-11-20, the UR5's box: its
            # 300 mm upper bound, saved with its outline, came back as measured, and the hammer was lowered
            # in among its walls rather than let go from over them.
            if kept is not None and "rim_m" in kept and kept.get("measured"):
                found.rim_m, found.rim_measured = float(kept["rim_m"]), True
        self.kept[found.name] = found

    # ── looking ───────────────────────────────────────────────────────────────

    def find(self, env, scene_state, what) -> "tuple[Container | None, str]":
        """Look for `what` (look(env)), and keep it if the vision model finds it and takes it for a container."""
        name, words = name_for(what), str(what).replace("_", " ")
        frame, model = self.look(env)
        if model is None:
            return None, "the fixed camera is not calibrated: nothing can be found with it"
        self.model = model
        kw = {} if self.ask is None else {"ask": self.ask}
        found, why = find(frame, self.camera, name, words, model=self.grounding_model, rim_m=self.tallest_m,
                          opening_by_shade=self.opening_by_shade,
                          judge_model=self.judge_model, **kw)
        if found is None:
            return None, why
        self._keep(found, self.kept.get(name))
        if self.keep_picture is not None:
            self.keep_picture(frame, found)
        self.targets(scene_state)
        self._save_where()
        return found, ""

    def refresh(self, env, scene_state, covered=None) -> list:
        """
        Every kept container found again where it is now; one not seen is
        forgotten. covered(container) -> bool: one the arm is over in the
        picture keeps the outline it had (found through the arm, the outline
        would take the arm in). [(name, what happened)].
        """
        said: "list[tuple[str, str]]" = []
        remembered = self.remembered()
        if not self.kept and not remembered:
            return said
        frame, model = self.look(env)
        if model is None:
            return said
        self.model = model
        kw = {} if self.ask is None else {"ask": self.ask}
        for name, old in list(self.kept.items()):
            if covered is not None and covered(old):
                said.append((name, "the arm is over it — kept where it was"))
                continue
            found, why = find(frame, self.camera, name, old.what, model=self.grounding_model, rim_m=old.rim_m,
                              opening_by_shade=self.opening_by_shade, **kw)
            if found is None:
                said.append((name, f"not seen this time ({why}) — forgotten until it is asked for again"))
                del self.kept[name]
                continue
            self._keep(found, old)
        for name, what in remembered.items():
            # Taken for a container when first found: the judge is not asked again.
            found, why = find(frame, self.camera, name, what, model=self.grounding_model, rim_m=self.tallest_m,
                              opening_by_shade=self.opening_by_shade, **kw)
            if found is None:
                said.append((name, f"kept from before, not seen now ({why})"))
                continue
            if covered is not None and covered(found):
                said.append((name, "kept from before; the arm is over it — looked for again at the next look"))
                continue
            self._keep(found, None)
            said.append((name, "found where it is now" + (f", its walls {found.rim_m * 1000:.0f} mm as measured before"
                                                          if found.rim_measured else "")))
        self.targets(scene_state)
        self._save_where()
        return said

    def _save_where(self) -> None:
        """
        Where each kept container stands now, its outline at the rim on the
        surface (world xy), kept on disk beside its walls' height: for a tool
        with no camera that must keep clear of it (tools/so101/rest.py — on
        2026-11-19 it folded the arm into the tub it knew nothing of).
        """
        import json
        import time as _time
        from pathlib import Path

        if self.rims_path is None or self.model is None or not self.kept:
            return
        rims = self._rims()
        for name, c in self.kept.items():
            xy = c.outline_xy(self.model)
            if xy is None or len(xy) < 3:
                continue
            entry = dict(rims.get(name) or {"rim_m": round(float(c.rim_m), 4)})
            entry["outline_xy"] = [[round(float(x), 4), round(float(y), 4)] for x, y in np.asarray(xy)[::max(1, len(xy) // 64)]]
            entry["seen"] = _time.strftime("%Y-%m-%d %H:%M")
            rims[name] = entry
        try:
            Path(self.rims_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.rims_path).write_text(json.dumps(rims, indent=1) + "\n")
        except OSError:
            pass

    def measured(self, scene_state, name, rim_m: float) -> None:
        """How high its walls stand (Scenario.opening_measured): kept for the session, and on disk."""
        c = self.known(name)
        if c is None:
            return
        c.rim_m, c.rim_measured = float(rim_m), True
        self._save_rim(c.name, rim_m, c.what)
        self._save_where()                     # its outline carried to the walls' measured height, not the guess
        self.targets(scene_state)

    # ── for the planner and the policies ──────────────────────────────────────

    def object_for(self, scene_state, container: "Container"):
        """An object the scan listed that is this container itself (a bowl): its foot inside, and nearly as wide."""
        from mrcf.policies.place_in import inside_by

        if self.model is None:
            return None
        floor = container.outline_xy(self.model, at_m=0.0)
        if floor is None or len(floor) < 3:
            return None
        least = float(np.min(floor.max(axis=0) - floor.min(axis=0)))
        for o in scene_state.objects:
            if o.position is None:
                continue
            foot = (o.geometry or {}).get("base_diameter_m") or 0.0
            if foot >= 0.5 * least and inside_by(floor, o.position[:2]) > 0:
                return o
        return None

    def holds(self, scene_state, obj) -> "str | None":
        """Scenario.holds: the kept container obj stands inside, if any."""
        from mrcf.policies.place_in import inside_by

        if obj.position is None or self.model is None:
            return None
        for name, c in self.kept.items():
            if self.object_for(scene_state, c) is obj:
                continue
            floor = c.outline_xy(self.model, at_m=0.0)
            if floor is not None and len(floor) >= 3 and inside_by(floor, obj.position[:2]) > 0:
                return name
        return None

    def targets(self, scene_state) -> None:
        """The kept containers as place targets, and each object inside one said to be."""
        from mrcf.vlm import SceneObject

        if self.model is None:
            return
        targets = [t for t in scene_state.targets if t.name not in self.kept]
        for name, c in self.kept.items():
            centre = c.centre_xy(self.model)
            if centre is None:
                continue
            rim = (f"walls {c.rim_m * 1000:.0f} mm tall" if c.rim_measured
                   else "how tall its walls are not yet measured")
            targets.append(SceneObject(name, f"a container ({c.what}): put things INTO it with place_in; {rim}",
                                       interactable=False, position=np.array([centre[0], centre[1], self.table_z])))
        scene_state.targets = targets
        for o in scene_state.objects:
            inside = self.holds(scene_state, o)
            if inside and f"— in {inside}" not in (o.description or ""):
                o.description = f"{o.description} — in {inside} already"

    def opening(self, scene_state, name):
        """Scenario.opening: the kept container called name as the policies' Opening, and its scene entry; or why not."""
        c = self.known(name)
        if c is None:
            return None
        if self.model is None:
            return f"{c.name}: the fixed camera's model is not known, so its outline cannot be carried onto the table"
        opening = c.opening(self.model)
        if opening is None:
            return f"{c.name}'s outline cannot be carried onto the table"
        entry = self.object_for(scene_state, c) or next((t for t in scene_state.targets if t.name == c.name), None)
        if entry is None:
            return f"{c.name} is not among the places now — look again"
        return opening, entry

    def walls(self) -> list:
        """
        Every kept container as an Opening of its outline at the rim (the walls
        themselves, not the room inside them): what a move across keeps the
        jaws above (ParallelJaw.walls, spec.transit_over_walls).
        """
        if self.model is None:
            return []
        out = []
        for c in self.kept.values():
            xy = c.outline_xy(self.model)
            if xy is not None and len(xy) >= 3:
                out.append(Opening(xy, float(c.rim_m), c.rim_measured, label=c.name))
        return out

    def floors(self) -> list:
        """Every kept container's outline on the surface (world xy): where else not to put or look."""
        if self.model is None:
            return []
        return [c.outline_xy(self.model, at_m=0.0) for c in self.kept.values()]
