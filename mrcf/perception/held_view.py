"""
held_view.py — whether something is held, from two wrist-camera pictures
taken as the arm moved with the jaws shut.

A camera on the gripper sees what the jaws hold where it was: it moves with
the camera. The table and everything standing on it move in the picture as the
arm does — and by how much is known, from where the arm put the camera each
time (its camera models): a level surface maps from one picture to the other
by a homography. So, of the pixels with texture enough to tell (edges), those
that stayed where they were and did NOT move as the surface did are carried
by the gripper; the jaws themselves (the learned jaw view, perception/
jaw_view.py) are left out. Enough of them: something is held. None, where the
surface's own edges showed it moving: nothing is. Neither — a blank table, an
arm that hardly moved — says nothing, and the gripper's own reading decides.

2026-11-17: a scoop held by its 1 mm wall, the jaws squeezed to 0.1 mm, read
empty by the motor's width; the wrist camera had its bowl in the same place
in every frame of the lift, and the table sliding away under it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mrcf.perception.plane_geometry import PlaneView

WIDTH = 320                 # pictures compared this wide
TEXTURE = 18.0              # a pixel counts with an edge this strong (Sobel, 0-255 grey, blurred) in both
SAME = 14.0                 # ... where it was, within this (grey levels, blurred)
MOVED = 28.0                # ... and the surface's prediction off by more than this
JAWS_GROW_PX = 9            # the jaws' learned outline, grown by this (at WIDTH) and left out
STILL_PX = 1                # "where it was": within this many pixels (the servos settle a little)
SURFACE_PX = 3              # "as the surface": within this many (the wrist camera's model is a few px out)
HELD_SHARE = 0.004          # held: this share of the picture carried (~300 px at 320x240) ...
SURFACE_SHARE = 0.01        # ... nothing held only with this share of it seen moving as the surface


@dataclass
class HeldView:
    held: "bool | None"     # True: something carried; False: nothing, the surface seen moving; None: cannot tell
    carried: float          # share of the picture carried by the gripper (not the jaws)
    surface: float          # share seen moving as the surface does
    why: str = ""


def _grey(frame: np.ndarray) -> np.ndarray:
    import cv2

    f = np.ascontiguousarray(frame)
    h, w = f.shape[:2]
    small = cv2.resize(f, (WIDTH, int(round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)


def _nearest(b: np.ndarray, a: np.ndarray, px: int) -> np.ndarray:
    """Per pixel of b, the least |b - a| over a shifted by up to px either way."""
    best = np.full(b.shape, np.inf, np.float32)
    h, w = b.shape
    for dy in range(-px, px + 1):
        for dx in range(-px, px + 1):
            shifted = np.full_like(a, 1e4)
            shifted[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
                a[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
            best = np.minimum(best, np.abs(b - shifted))
    return best


def surface_homography(model_a: dict, model_b: dict, size_a, size_b, height_m: float = 0.0) -> "np.ndarray | None":
    """3x3: picture A's pixels of the surface (height_m above it) to where picture B sees them, at WIDTH; None if it cannot."""
    import cv2

    va, vb = PlaneView.from_camera_model(model_a), PlaneView.from_camera_model(model_b)
    wa, ha = size_a
    us, vs = np.meshgrid(np.linspace(0.1 * wa, 0.9 * wa, 7), np.linspace(0.1 * ha, 0.9 * ha, 7))
    px = np.c_[us.ravel(), vs.ravel()]
    lift = np.array([0.0, 0.0, height_m * 1000.0 / va.scale])
    raised = PlaneView(va.K, va.D, va.R, va.t + va.R @ lift, scale=va.scale, origin_m=va.origin_m)
    xy = raised.to_plane(px)
    ok = np.isfinite(xy).all(axis=1)
    if ok.sum() < 8:
        return None
    pts = np.c_[xy[ok], np.full(int(ok.sum()), height_m * 1000.0 / vb.scale)]
    if ((pts @ vb.R.T + vb.t)[:, 2] <= 1.0).any():
        return None
    qb = vb.project(pts)
    sa, sb = WIDTH / float(wa), WIDTH / float(size_b[0])
    H, _ = cv2.findHomography((px[ok] * sa).astype(np.float32), (qb * sb).astype(np.float32))
    return H


def held_view(frame_a: np.ndarray, model_a: dict, frame_b: np.ndarray, model_b: dict,
              jaws_b: "np.ndarray | None" = None, surface_height_m: float = 0.0) -> HeldView:
    """
    Whether the gripper carries something from picture A to picture B (the
    jaws shut through both, the arm moved between): see the module docstring.
    jaws_b: where the jaws are in B, bool, any size (JawView.at); None: cannot
    tell the jaws from what they hold.
    """
    import cv2

    if jaws_b is None:
        return HeldView(None, 0.0, 0.0, "the jaws' place in the picture is not learned")
    a, b = _grey(frame_a), _grey(frame_b)
    if a.shape != b.shape:
        return HeldView(None, 0.0, 0.0, "the two pictures are not the same size")
    h, w = b.shape
    H = surface_homography(model_a, model_b, frame_a.shape[1::-1], frame_b.shape[1::-1], surface_height_m)
    if H is None:
        return HeldView(None, 0.0, 0.0, "where the surface went between the pictures is not known")
    predicted = cv2.warpPerspective(a, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=-1000.0)
    covered = predicted > -500.0
    texture = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    texture_a = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    edged = (texture >= TEXTURE) & (texture_a >= TEXTURE)
    small_jaws = cv2.resize(np.asarray(jaws_b, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    jaws: np.ndarray = cv2.dilate(small_jaws, np.ones((2 * JAWS_GROW_PX + 1, 2 * JAWS_GROW_PX + 1), np.uint8)) > 0
    stayed = _nearest(b, a, STILL_PX) <= SAME
    off_surface = _nearest(b, np.where(covered, predicted, np.float32(1e4)), SURFACE_PX)
    as_surface = covered & (off_surface <= SAME)
    moved_off = ~covered | (off_surface > MOVED)
    carried = edged & ~jaws & stayed & moved_off
    surface = edged & ~jaws & as_surface & ~stayed
    n = float(h * w)
    c, sfc = float(carried.sum()) / n, float(surface.sum()) / n
    if c >= HELD_SHARE:
        return HeldView(True, c, sfc)
    if sfc >= SURFACE_SHARE:
        return HeldView(False, c, sfc)
    return HeldView(None, c, sfc, "too little to see by: the surface plain, or the arm hardly moved")
