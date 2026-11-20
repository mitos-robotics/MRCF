"""
jaw_view.py — where a gripper's own jaws are in its wrist camera's picture,
learned from pictures of them rather than drawn from a model.

A camera on the gripper is fixed to the gripper's body and to its fixed jaw:
wherever the arm goes, they are at the same pixels, and the moving jaw at
pixels that depend only on how far it is open. So where they are is learned
once, per opening, from the pictures themselves: the jaws are the dark shapes
that come in from the bottom of the picture and stay put while the scene
behind them changes. Pictures at a few openings from a few places over the
table (learn's frames): at each opening, what is dark and joined to the
picture's bottom edge in EVERY place is the jaws — an object or a shadow is
dark in one place, not in all. Each shape kept is filled out to its convex
hull (a jaw's light marks, its pads' edges, come with it).

Why not from the camera's model: a mount fitted to see the table right at
8-17 cm put the jaws, 5-8 cm from the lens, ~11 mm from where they are
(2026-11-03, robot_config GRIPPER_CAM_POSE; 2026-11-17, drawn 50-90 px off in
every look for an object's height, which then outlined a jaw as the object
and planned views the jaws hid). The pictures need no model.

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# A pixel is dark when its lightness is under this share of the picture's
# bright lightness (its 75th percentile: the table). The SO-101's black jaws
# read under a third of the white table's.
DARK_SHARE = 0.45
# A dark shape smaller than this share of the picture is a speck, not a jaw.
MIN_SHAPE = 0.004
# Gaps narrower than this (px, at 640 wide) are closed before shapes are told
# apart: a jaw's light mark (the SO-101's moving jaw has a white disc) splits its
# tip off from the rest, and the tip no longer touches the picture's bottom.
CLOSE_GAP_PX = 31


def dark_from_bottom(frame_rgb: np.ndarray) -> np.ndarray:
    """The dark shapes in a picture that touch its bottom edge, bool (h, w)."""
    import cv2

    L = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2LAB)[..., 0].astype(float)
    h, w = L.shape
    dark = (L < DARK_SHARE * np.percentile(L, 75)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    k = max(3, int(round(CLOSE_GAP_PX * w / 640.0)) | 1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    keep = np.zeros((h, w), bool)
    for i in range(1, n):
        _, y, _, bh, area = stats[i]
        if y + bh >= h - 2 and area >= MIN_SHAPE * h * w:
            keep |= labels == i
    return keep


def jaws_in(frames: list) -> np.ndarray:
    """The jaws, from pictures at one opening taken from different places: dark from the bottom in all of them."""
    import cv2

    if not frames:
        raise ValueError("no pictures to find the jaws in")
    every = np.all(np.stack([dark_from_bottom(f) for f in frames]), axis=0)
    m = every.astype(np.uint8)
    h, w = m.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        _, y, _, bh, area = stats[i]
        if y + bh < h - 2 or area < MIN_SHAPE * h * w:
            continue
        pts = np.column_stack(np.nonzero(labels == i)[::-1]).astype(np.int32)
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


@dataclass
class JawView:
    """Where the jaws are in the wrist camera's picture, at each opening learned."""
    openings_m: list                  # the openings learned, ascending
    masks: np.ndarray                 # (n, h, w) bool: the jaws at each
    learned_at: float = field(default_factory=time.time)

    @property
    def size(self) -> tuple:
        """(w, h) of the pictures it was learned from."""
        return int(self.masks.shape[2]), int(self.masks.shape[1])

    def at(self, gap_m: float, shape=None) -> np.ndarray:
        """
        The jaws open gap_m, bool (h, w) at shape (the learned size if None):
        those learned at the openings either side of it, together — the moving
        jaw is somewhere between.
        """
        import cv2

        gaps = np.asarray(self.openings_m, float)
        hi = int(np.searchsorted(gaps, float(gap_m)))
        idx = sorted({min(max(hi - 1, 0), len(gaps) - 1), min(hi, len(gaps) - 1)})
        m = np.any(np.stack([self.masks[i] for i in idx]), axis=0)
        if shape is not None and tuple(shape[:2]) != m.shape:
            m = cv2.resize(m.astype(np.uint8), (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST) > 0
        return m

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, openings_m=np.asarray(self.openings_m, float), masks=self.masks.astype(bool),
                            learned_at=np.float64(self.learned_at))

    @classmethod
    def load(cls, path) -> "JawView | None":
        """The view saved at path; None when there is none, or it cannot be read."""
        try:
            with np.load(Path(path)) as d:
                return cls([float(v) for v in d["openings_m"]], d["masks"].astype(bool), float(d["learned_at"]))
        except (OSError, KeyError, ValueError):
            return None


def learn(frames_by_opening: dict) -> JawView:
    """A JawView from {opening_m: [pictures from different places]}."""
    gaps = sorted(frames_by_opening)
    return JawView([float(g) for g in gaps], np.stack([jaws_in(frames_by_opening[g]) for g in gaps]))
