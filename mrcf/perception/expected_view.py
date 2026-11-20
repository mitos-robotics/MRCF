"""
expected_view.py — what a located camera should see of objects another camera
has already found: where each one's outline falls in its frame, which stands in
front of which, and so which outline in the frame is whose.

A wrist camera close over a crowd sees several objects at once, the taller and
nearer ones in front of the rest. Without knowing who stands where, it takes
the outline nearest where it expects its object, and on a full chessboard
(2026-11-03) that was the e1 king in front of the e2 pawn from over e2, or the
c1 bishop from over b1. The overhead scan knows every object's foot and size;
with the wrist camera located (RobotEnv.camera_model), each can be drawn in
the frame as the camera should see it — a column standing on its foot — and
each pixel given to the nearest column that covers it. Then:

  - the prompts for the target go where the target, and nothing in front of
    it, should be (prompts());
  - an outline belongs to the object owning most of it (owner_of()), so a
    neighbour's outline is not taken for the target's;
  - a target with its foot hidden behind others says so (foot_visible()),
    before anything is measured from a foot that is not there to see.

Geometry only: columns as wide as the foot all the way up (wider than most
objects above their foot: in front of more than they are, never less), up to
the height the object is known or allowed to have.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mrcf.perception.plane_geometry import PlaneView


@dataclass
class KnownObject:
    """An object another camera found: who, where its foot is (the view's world x, y, metres), how big."""
    key: str
    foot_xy_m: np.ndarray
    radius_m: float
    height_m: float


class ExpectedView:
    """The objects as `view` should see them, in a frame of `shape` (h, w)."""

    RINGS = 6           # circles up each column, foot to top: its outline is their hull

    def __init__(self, view: PlaneView, objects: "list[KnownObject]", shape):
        if view.origin_m is None:
            raise ValueError("ExpectedView needs a view in the world frame (PlaneView.from_camera_model)")
        self.view, self.shape = view, tuple(shape[:2])
        self.keys = [o.key for o in objects]
        self.objects = {o.key: o for o in objects}
        h, w = self.shape
        self.masks: dict = {}
        self.feet: dict = {}
        depth: dict = {}
        for o in objects:
            mask, foot, d = self._column(o, w, h)
            if mask is not None:
                self.masks[o.key], self.feet[o.key], depth[o.key] = mask, foot, d
        # Each pixel to the nearest column covering it: what is in front wins.
        self.owner = np.full((h, w), -1, np.int32)
        nearest = np.full((h, w), np.inf)
        for i, k in enumerate(self.keys):
            if k not in self.masks:
                continue
            closer = self.masks[k] & (depth[k] < nearest)
            self.owner[closer], nearest[closer] = i, depth[k]
        self.depth = depth

    def _plane_mm(self, xy_m) -> np.ndarray:
        return (np.asarray(xy_m, float)[:2] - self.view.origin_m) * 1000.0

    def _column(self, o: KnownObject, w: int, h: int):
        """(mask, foot mask, distance from the camera) of o's column in the frame; (None, None, inf) when behind it."""
        v = self.view
        foot = self._plane_mm(o.foot_xy_m)
        r = o.radius_m * 1000.0 / v.scale
        a = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
        ring = np.c_[r * np.cos(a), r * np.sin(a)]
        pts = np.vstack([np.c_[foot + ring, np.zeros(len(a))] + v.up * (z / v.scale)
                         for z in np.linspace(0.0, o.height_m * 1000.0, self.RINGS)])
        cam = (v.R @ pts.T).T + v.t
        if (cam[:, 2] <= 1.0).any():               # part of it behind the camera: not drawn
            return None, None, np.inf
        px = v.project(pts)
        if not np.isfinite(px).all() or np.abs(px).max() > 10 * max(w, h):
            return None, None, np.inf
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(px.astype(np.int32)), 1)
        foot_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(foot_mask, cv2.convexHull(px[:24].astype(np.int32)), 1)
        mid = v.lifted(foot, o.height_m * 500.0)
        return mask.astype(bool), foot_mask.astype(bool), float(np.linalg.norm(mid - v.centre))

    # ── What the frame should show ─────────────────────────────────────────────

    def drawn(self, key: str) -> bool:
        """Whether key's column falls in front of the camera at all."""
        return key in self.masks

    def visible(self, key: str) -> np.ndarray:
        """The pixels where key should be seen: its column, less what stands in front of it."""
        return self.owner == self.keys.index(key)

    def visible_fraction(self, key: str) -> float:
        if key not in self.masks or not self.masks[key].any():
            return 0.0
        return float(self.visible(key).sum()) / float(self.masks[key].sum())

    def foot_visible(self, key: str) -> float:
        """The share of key's foot (the circle it stands on) not hidden by anything in front of it."""
        foot = self.feet.get(key)
        if foot is None or not foot.any():
            return 0.0
        return float((foot & self.visible(key)).sum()) / float(foot.sum())

    def in_front(self, key: str) -> list:
        """The keys of what stands in front of key's column, most of it first."""
        mask = self.masks.get(key)
        if mask is None:
            return []
        ids, counts = np.unique(self.owner[mask], return_counts=True)
        mine = self.keys.index(key)
        return [self.keys[i] for i, _ in sorted(zip(ids, counts), key=lambda p: -p[1]) if i not in (-1, mine)]

    def prompts(self, key: str, n: int = 6, lift_mm: float = 15.0) -> list:
        """
        Pixels to ask the segmenter about for key: up its axis from a little
        above the foot, those it should be seen at, nearest the foot first —
        the lower body decides where the foot is. [] when none of it is seen.
        """
        o = self.objects[key]
        vis = self.visible(key)
        if not vis.any():
            return []
        foot = self._plane_mm(o.foot_xy_m)
        top = max(o.height_m * 1000.0, lift_mm + 1.0)
        pts = np.array([self.view.lifted(foot, z) for z in np.linspace(lift_mm, 0.8 * top, 8)])
        h, w = self.shape
        out = []
        for u, v in self.view.project(pts):
            if 0 <= u < w and 0 <= v < h and vis[int(v), int(u)]:
                out.append((float(u), float(v)))
        if not out:                                 # the axis is hidden: where it shows at all
            ys, xs = np.nonzero(vis)
            out.append((float(xs.mean()), float(ys.mean())))
        return out[:n]

    def owner_of(self, mask: np.ndarray) -> "tuple[str | None, float]":
        """
        (key, share): the object owning most of an outline's pixels that fall
        on any expected column, and that share of them. (None, 0.0) when it
        covers none.
        """
        ids = self.owner[np.asarray(mask, bool)]
        ids = ids[ids >= 0]
        if not len(ids):
            return None, 0.0
        vals, counts = np.unique(ids, return_counts=True)
        best = int(np.argmax(counts))
        return self.keys[int(vals[best])], float(counts[best]) / float(len(ids))
