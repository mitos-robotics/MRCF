"""
catalogue.py — the kinds of object a deployment works with, as the arm has
learned them: what each looks like to the fixed camera, how it rests, and the
grip that held it.

Why: a deployment's objects are mostly the same ones, put down in other
places. The chessboard told the arm what each piece was and where; a bare
table tells it nothing, so without this every pick is a first meeting — the
object looked at from the side for its height (25-48 s), a grip planned on
what that made of it, a miss when it was 1-2 cm out, and the same again next
time. 2026-11-18, the SO-101's table: four objects, about ten such surveys in
a day, a different answer each time; and the vision model's name for each
changed from scan to scan (one wooden figure: "bottle cap", "bottle",
"shaker", "ball", "red bottle"), so nothing kept by name was found again.

What is kept, per kind (Kind), per way it rests (Pose: "upright", "lying"):

  signature   what the fixed camera sees of it from above — its foot's width,
              its outline's length and width on the surface, its colour
              (signature()) — which a sighting is matched by (match()), not
              by any name
  recipe      the grips tried on it: where the jaws closed, the top the hover
              cleared, how often each held and missed (learn()); best() is the
              one to pick by next time
  height      what the wrist camera measured of it, when it did

A kind is an object type, not one object: two figures alike are one kind, and
what is learned on either serves both. Kept on disk (path), across restarts.
Nothing here moves the arm, names a task or knows a camera; the deployment
gives it signatures and outcomes (mrcf/tabletop/skills.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# How near a sighting's signature must be to a pose's to be it: its size —
# foot, and its outline's length and width on the surface — each within its
# tolerance (the larger of the absolute and the share), their sum under
# MATCH_SUM. Colour only weighs between kinds alike in size (COLOUR_WEIGHT),
# never rules one out: the scan's colour is its outline's middle, which on a
# two-coloured thing is whichever part is there — 2026-11-18, two wooden
# figures alike, red bodies and green heads, read a 14 b 5 and a 41 b 12, and
# the second was not taken for the first. Two kinds alike in size grip alike.
FOOT_TOL = (4.0, 0.15)            # mm, share
LONG_TOL = (6.0, 0.20)
SHORT_TOL = (5.0, 0.20)
COLOUR_TOL = 15.0                 # Lab a, b distance ...
LIGHT_TOL = 40.0                  # ... and L, each counted as a term over this
COLOUR_WEIGHT = 0.25              # ... weighted so, and not a gate ...
COLOUR_GATE = 2.5                 # ... unless it is this many tolerances off: then not it, however alike in
                                  # size (2026-11-19: a blue scoop upside down, plug-sized, taken for the white
                                  # plug — 45 a/b and 150 L apart; the two-tone figures 28 a/b and 38 L)
MATCH_SUM = 3.0


def signature(mask: np.ndarray, view, foot_mm: float, appearance: "dict | None") -> dict:
    """
    What the fixed camera sees of an object from above: {"foot_mm", "long_mm",
    "short_mm", "L", "a", "b"} — its outline (mask, in the camera's frame)
    carried onto the surface through view (plane_geometry.PlaneView), the
    smallest rectangle round it; its colour (scan's appearance, Lab medians).
    """
    import cv2

    ys, xs = np.nonzero(np.asarray(mask, bool))
    long_mm = short_mm = float(foot_mm)
    if len(xs) >= 10:
        cs, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            px = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
            on = view.to_plane(px)
            on = on[np.isfinite(on).all(1)]
            if len(on) >= 3:
                (_, _), (w, h), _ = cv2.minAreaRect(on.astype(np.float32))
                long_mm, short_mm = float(max(w, h)), float(min(w, h))
    a = appearance or {}
    return {"foot_mm": round(float(foot_mm), 1), "long_mm": round(long_mm, 1), "short_mm": round(short_mm, 1),
            "L": a.get("L"), "a": a.get("a"), "b": a.get("b")}


def _tol(x, y, tol) -> float:
    return max(tol[0], tol[1] * max(abs(float(x)), abs(float(y))))


SIZES = (("foot_mm", FOOT_TOL), ("long_mm", LONG_TOL), ("short_mm", SHORT_TOL))


def _off(v: float, ref: float, span, tol) -> float:
    """How far v is from ref — or from the range span (lo, hi) it has been seen at, 0 inside it — over its tolerance."""
    lo, hi = (float(span[0]), float(span[1])) if span else (float(ref), float(ref))
    near = min(max(float(v), lo), hi)
    return abs(float(v) - near) / _tol(v, near, tol)


def distance(s1: dict, s2: dict, span: "dict | None" = None) -> "float | None":
    """
    How far apart two signatures are (sum of each term over its tolerance);
    None when any term is out. span: the range each size of s2's has been
    seen at ({key: [lo, hi]}): inside it, that term counts nothing.
    """
    span = span or {}
    size = [_off(s1[k], s2[k], span.get(k), tol) for k, tol in SIZES]
    if max(size) > 1.0 or sum(size) > MATCH_SUM:
        return None
    colour = 0.0
    if None not in (s1.get("a"), s1.get("b"), s2.get("a"), s2.get("b")):
        ab = float(np.hypot(s1["a"] - s2["a"], s1["b"] - s2["b"])) / COLOUR_TOL
        if ab > COLOUR_GATE:
            return None
        colour += ab
    if s1.get("L") is not None and s2.get("L") is not None:
        light = abs(s1["L"] - s2["L"]) / LIGHT_TOL
        if light > COLOUR_GATE:
            return None
        colour += light
    return float(sum(size) + COLOUR_WEIGHT * colour)


@dataclass
class Pose:
    """One way a kind rests, as learned: its signature, what was measured of it, the grips tried."""
    name: str                                  # "upright", "lying", ...
    sig: dict
    height_mm: "float | None" = None
    widths_mm: dict = field(default_factory=dict)
    grips: list = field(default_factory=list)  # {"grip_at_m", "top_m", "held", "missed", "last"}
    seen: int = 1
    # The range each size has been seen at ({key: [lo, hi]}): one object's
    # outline is not one size scan to scan (2026-11-19, the plug 42-57 mm
    # across, the scoop 65-125 as its handle was or was not taken in).
    span: dict = field(default_factory=dict)

    def widen(self, sig: dict) -> None:
        """Its size ranges, widened to take in sig."""
        for k, _ in SIZES:
            v = sig.get(k)
            if v is None:
                continue
            lo, hi = self.span.get(k, [self.sig.get(k, v), self.sig.get(k, v)])
            self.span[k] = [round(min(float(lo), float(v)), 1), round(max(float(hi), float(v)), 1)]

    def best(self) -> "dict | None":
        """The grip to pick by: the one that held most, net of its misses; None until one has held."""
        held = [g for g in self.grips if g.get("held", 0) > 0]
        return max(held, key=lambda g: (g["held"] - g.get("missed", 0), g["held"])) if held else None


@dataclass
class Kind:
    """A kind of object the deployment works with."""
    id: str
    name: str                                  # what it is called: the first name it was given
    colour: str = ""
    poses: list = field(default_factory=list)  # Pose
    learned: str = ""
    picked_by: str = ""                        # the action that held it ("pick", "pick_by_rim"), as last held

    def pose(self, name: str) -> "Pose | None":
        return next((p for p in self.poses if p.name == name), None)

    def longest_mm(self) -> "float | None":
        """Its longest size seen, any way it rested: how tall it can be resting any way at all."""
        sizes = [float(v) for p in self.poses
                 for v in [p.sig.get("long_mm"), (p.span.get("long_mm") or [None, None])[1], p.height_mm]
                 if v is not None]
        return max(sizes) if sizes else None


def _merge(into: "Kind", other: "Kind") -> None:
    """other's poses and grips taken into into: one object kept as two."""
    for op in other.poses:
        p = into.pose(op.name)
        if p is None:
            into.poses.append(op)
            continue
        p.seen += op.seen
        p.widen(op.sig)
        for k, (lo, hi) in (op.span or {}).items():
            p.widen({k: lo})
            p.widen({k: hi})
        for g in op.grips:
            same = next((x for x in p.grips if abs(x["grip_at_m"] - g["grip_at_m"]) < 0.003), None)
            if same is None:
                p.grips.append(dict(g))
            else:
                same["held"] = same.get("held", 0) + g.get("held", 0)
                same["missed"] = same.get("missed", 0) + g.get("missed", 0)


class Catalogue:
    """A deployment's kinds of object, kept at path (None: for the session only)."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.kinds: list = []
        if self.path is not None and self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                for k in data.get("kinds", []):
                    poses = [Pose(**p) for p in k.pop("poses", [])]
                    for p in poses:
                        p.widths_mm = {int(h): float(w) for h, w in (p.widths_mm or {}).items()}
                    self.kinds.append(Kind(**k, poses=poses))
            except (OSError, ValueError, TypeError) as exc:
                print(f"  [Catalogue] Could not read {self.path} ({exc}) — starting empty.")
                self.kinds = []
        # One object, one kind: kinds called the same and coloured the same, kept apart before, merged.
        merged: list = []
        for k in self.kinds:
            same = next((m for m in merged if m.name.lower() == k.name.lower() and m.colour == k.colour), None)
            if same is None:
                merged.append(k)
            else:
                _merge(same, k)
        self.kinds = merged

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   "kinds": [asdict(k) for k in self.kinds]}, indent=1, default=float))
        tmp.replace(self.path)

    def get(self, kind_id: "str | None") -> "Kind | None":
        return next((k for k in self.kinds if k.id == kind_id), None)

    def match(self, sig: dict) -> "tuple[Kind, Pose, float] | None":
        """The kind and pose whose signature this sighting's is nearest (distance()), or None: none near enough."""
        best = None
        for k in self.kinds:
            for p in k.poses:
                d = distance(sig, p.sig, p.span)
                if d is not None and (best is None or d < best[2]):
                    best = (k, p, d)
        return best

    def named(self, name: "str | None", colour: "str | None") -> "Kind | None":
        """The kind called name and coloured colour, if there is one: the same object, whatever its outline now."""
        if not name:
            return None
        return next((k for k in self.kinds if k.name.lower() == str(name).lower()
                     and (k.colour or "") == (colour or "")), None)

    def add(self, name: str, colour: str, sig: dict, pose: str = "upright") -> "tuple[Kind, Pose]":
        """
        A kind first met: named as the vision model or the user named it,
        resting as `pose` — or, called and coloured as one already kept, a new
        way that one rests (it is the same object: 2026-11-19, the scoop kept
        twice, its better grip under the other).
        """
        same = self.named(name, colour)
        if same is not None:
            p = same.pose(pose)
            if p is None:
                p = Pose(pose if not same.pose(pose) else f"{pose}-{len(same.poses) + 1}", dict(sig))
                same.poses.append(p)
            p.widen(sig)
            return same, p
        kind = Kind(f"kind-{len(self.kinds) + 1}", str(name), str(colour or ""), [Pose(pose, dict(sig))],
                    time.strftime("%Y-%m-%d"))
        self.kinds.append(kind)
        return kind, kind.poses[0]

    def learn(self, kind: Kind, pose_name: str, sig: dict, *, grip_at_m: "float | None", top_m: "float | None",
              held: bool, height_mm: "float | None" = None, widths_mm: "dict | None" = None) -> Pose:
        """
        What a pick of it showed: the grip it closed at (grip_at_m, above the
        surface), the top its hover cleared (top_m), whether it held; what the
        wrist camera measured, if anything. The pose learned too when new (a
        figure found lying), its signature drawn toward this one. Saved.
        """
        pose = kind.pose(pose_name)
        if pose is None:
            pose = Pose(pose_name, dict(sig))
            kind.poses.append(pose)
        else:
            pose.seen += 1
            pose.widen(sig)
            for key, v in sig.items():
                old = pose.sig.get(key)
                if v is not None and old is not None:
                    pose.sig[key] = round(0.8 * float(old) + 0.2 * float(v), 1)   # drawn toward what is seen now
        if height_mm is not None:
            pose.height_mm = round(float(height_mm), 1)
            pose.widths_mm = {int(h): float(w) for h, w in (widths_mm or {}).items()} or pose.widths_mm
        if grip_at_m is not None:
            g = next((g for g in pose.grips if abs(g["grip_at_m"] - grip_at_m) < 0.003), None)
            if g is None:
                g = {"grip_at_m": round(float(grip_at_m), 4), "top_m": None, "held": 0, "missed": 0}
                pose.grips.append(g)
            g["held" if held else "missed"] += 1
            if top_m is not None and held:
                g["top_m"] = round(max(float(top_m), float(g["top_m"] or 0.0)), 4)
            g["last"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.save()
        return pose
