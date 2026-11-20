"""
placed_check.py — a quick look, once the arm has let go, at whether what it
put down is where it put it: one frame from a fixed camera, and the colour of
the patch where the object's top should now be, against the colour it is
remembered to have and the colour of what it was put on. No segmenter, no
vision model: a fraction of a second, where a scan is ten and a look with the
arm parked is forty.

Colour alone cannot say it: pale blocks under a camera with a cast are the
table's colour to within a block's own spread (2026-11-12: a patch of bare
table 6 cm from each of 9 blocks read as that block 5 times). So the patch is
also compared with a frame from before the place — the last scan's, the arm
parked out of view: where nothing changed, nothing was put down.

It can say yes (that patch changed, and to its colour, not the one under it),
no (it did not change, or it shows what it was put on: it fell, or slid
off), or that it cannot tell (the arm in the camera's way; the object the
colour of what it was put on). A no is worth a full look; a cannot-tell is
not a failure.

    look(frame, model, xy_m, top_z_m, radius_m, want_ab, under_ab=None, covered=None, before=None)
        -> (True | False | None, why)
    look_volume(frame, model, xy_m, bottom_z_m, top_z_m, half_m, covered=None, before=None, covered_before=None)
        -> (True | False | None, why)

look is for a camera looking down on the object's top. look_volume is for one
off to the side, low (2026-11-13: a RealSense's infrared view along the UR5's
table): the space the object now fills, its inside points, project onto
whatever the camera sees there — the object's own side when it stands there,
what is behind it when it does not — so a change from before says it is
there, and none that it is not. It goes by change alone: that camera is
greyscale, and its view from the side sees under a gripper the camera above
cannot see past.
"""

from __future__ import annotations

import cv2
import numpy as np

from mrcf.perception import camera_from_arm as cfa

# Lab chroma (a, b) apart, as object_memory.APPEARANCE_TOL measures it: 1 at
# the edge of the same colour.
APPEARANCE_TOL = 12.0
SAME = 1.0                 # this near its colour: it is there
NOT_IT = 1.6               # this far from it: it is not
CLEARER_BY = 0.4           # nearer its own colour than the one under it by this much, to say which
MIN_SEEN = 0.5             # of the patch's points, the share the arm must leave in view
CHANGED_LAB = 15.0         # the patch's change from before, the whole frame's shift aside, that is something
                           # new there (motion_guard's diff_lab)
# look_volume's, from the side: 2026-11-13, the UR5's RealSense, blocks that
# had come since read 43-81; the same spots with nothing new, against 32
# pictures taken while the arm was pushed round by hand (it left out), at most 27.
VOLUME_CHANGED = 35.0


def _patch_points(xy_m, top_z_m: float, radius_m: float, n: int = 5) -> np.ndarray:
    """Points of a disc of radius_m round xy_m at height top_z_m (world frame, m)."""
    xs = np.linspace(-radius_m, radius_m, n)
    pts = [(xy_m[0] + dx, xy_m[1] + dy, top_z_m) for dx in xs for dy in xs if dx * dx + dy * dy <= radius_m ** 2]
    return np.asarray(pts, float)


def chroma(frame_rgb: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Median Lab (a, b) of frame_rgb at the pixels px (N x 2), as surface_scan.appearance gives it (centred on 0)."""
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    cols = np.clip(np.round(px[:, 0]).astype(int), 0, frame_rgb.shape[1] - 1)
    rows = np.clip(np.round(px[:, 1]).astype(int), 0, frame_rgb.shape[0] - 1)
    vals = lab[rows, cols].astype(float)
    return np.array([np.median(vals[:, 1]) - 128.0, np.median(vals[:, 2]) - 128.0])


def change(frame_rgb: np.ndarray, before_rgb: np.ndarray, px: np.ndarray) -> float:
    """How much the pixels px changed from before_rgb to frame_rgb (Lab, median), the whole frame's shift aside."""
    now = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    was = cv2.cvtColor(np.ascontiguousarray(before_rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    shift = np.median((now - was).reshape(-1, 3), axis=0)          # the light, the exposure
    cols = np.clip(np.round(px[:, 0]).astype(int), 0, frame_rgb.shape[1] - 1)
    rows = np.clip(np.round(px[:, 1]).astype(int), 0, frame_rgb.shape[0] - 1)
    d = now[rows, cols] - was[rows, cols] - shift
    return float(np.median(np.linalg.norm(d, axis=1)))


def look(frame_rgb: np.ndarray, model: dict, xy_m, top_z_m: float, radius_m: float, want_ab,
         under_ab=None, covered: "np.ndarray | None" = None,
         before: "np.ndarray | None" = None) -> "tuple[bool | None, str]":
    """
    Is the object whose colour is want_ab (Lab a, b) standing with its top at
    (xy_m, top_z_m)? model: the camera's (K, D, T_world_cam) for frame's size.
    under_ab: the colour of what it was put on (None: the bare surface).
    covered: where the arm is in the frame (bool, frame's size), left out.
    before: a frame of the same view from before it was put there, the arm
    out of it (None: by colour alone, which can mistake the surface for it).
    """
    h, w = frame_rgb.shape[:2]
    pts = _patch_points(xy_m, top_z_m, radius_m)
    px, ok = cfa.project(pts, np.asarray(model["T_world_cam"], float), np.asarray(model["K"], float),
                         np.asarray(model["D"], float))
    ok &= (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    if covered is not None:
        inside = px[ok].astype(int)
        free = ~covered[np.clip(inside[:, 1], 0, h - 1), np.clip(inside[:, 0], 0, w - 1)]
        idx = np.flatnonzero(ok)
        ok[idx[~free]] = False
    if ok.sum() < max(3, MIN_SEEN * len(pts)):
        return None, "the arm is in the camera's way there"
    ab = chroma(frame_rgb, px[ok])
    if before is not None and before.shape == frame_rgb.shape:
        moved = change(frame_rgb, before, px[ok])
        if moved < CHANGED_LAB:
            return False, f"where its top should be looks as it did before it was put there (changed {moved:.0f})"
    d_want = float(np.linalg.norm(ab - np.asarray(want_ab, float))) / APPEARANCE_TOL
    d_under = (float(np.linalg.norm(ab - np.asarray(under_ab, float))) / APPEARANCE_TOL
               if under_ab is not None else None)
    seen = f"(a {ab[0]:+.0f}, b {ab[1]:+.0f})"
    if d_want <= SAME and (d_under is None or d_want + CLEARER_BY <= d_under):
        return True, f"its colour is where its top should be {seen}"
    if d_want > NOT_IT:
        what = "what it was put on" if d_under is not None and d_under <= SAME else "something else"
        return False, f"where its top should be looks like {what} {seen}, not it"
    if d_under is not None and d_under <= SAME:
        return None, f"it is the colour of what it was put on {seen}: cannot tell them apart from here"
    return None, f"its colour there is not clear {seen}"


def _volume_points(xy_m, bottom_z_m: float, top_z_m: float, half_m: float, n: int = 3) -> np.ndarray:
    """Points inside an upright box round xy_m, bottom_z_m to top_z_m, half_m either side, kept off its faces."""
    across = np.linspace(-0.5 * half_m, 0.5 * half_m, n)
    up = np.linspace(bottom_z_m + 0.25 * (top_z_m - bottom_z_m), top_z_m - 0.25 * (top_z_m - bottom_z_m), n)
    return np.asarray([(xy_m[0] + dx, xy_m[1] + dy, z) for dx in across for dy in across for z in up], float)


def look_volume(frame_rgb: np.ndarray, model: dict, xy_m, bottom_z_m: float, top_z_m: float, half_m: float,
                covered: "np.ndarray | None" = None, before: "np.ndarray | None" = None,
                covered_before: "np.ndarray | None" = None) -> "tuple[bool | None, str]":
    """
    Does something new stand where an object put at xy_m now should, its foot
    at bottom_z_m and its top at top_z_m, half_m from its axis to its sides —
    as a camera that sees it from the side (model: K, D, T_world_cam for the
    frame's size) shows it? By the change from before (the same view before
    it was put there) at the pixels its inside projects to, the arm left out
    of both (covered, covered_before). None: no before to go by, or the arm
    hides it.
    """
    if before is None or before.shape != frame_rgb.shape:
        return None, "no picture from before it was put there to compare with"
    h, w = frame_rgb.shape[:2]
    pts = _volume_points(xy_m, bottom_z_m, top_z_m, half_m)
    px, ok = cfa.project(pts, np.asarray(model["T_world_cam"], float), np.asarray(model["K"], float),
                         np.asarray(model["D"], float))
    ok &= (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    for mask in (covered, covered_before):
        if mask is not None and ok.any():
            inside = px[ok].astype(int)
            hidden = mask[np.clip(inside[:, 1], 0, h - 1), np.clip(inside[:, 0], 0, w - 1)]
            ok[np.flatnonzero(ok)[hidden]] = False
    if ok.sum() < max(3, MIN_SEEN * len(pts)):
        return None, "the arm is in this camera's way there"
    moved = change(frame_rgb, before, px[ok])
    if moved >= VOLUME_CHANGED:
        return True, f"something new stands where it should (changed {moved:.0f})"
    return False, f"where it should stand looks as it did before it was put there (changed {moved:.0f})"
