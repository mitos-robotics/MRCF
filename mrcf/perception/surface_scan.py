"""
surface_scan.py — find and measure every object standing on the work surface,
from a camera the backend has located. No board, no list of objects.

The segmenter (segment.py) outlines whatever is there and plane_geometry
measures each outline — where its foot is in the arm's frame, how wide, and its
colour. Nothing here knows what the objects are, nor keeps them: scene_scan.py
(at the framework's top level) turns a scan into object-memory Detections and
the world model's objects.

Telling objects from the surface
--------------------------------
On a plain table there is nothing to tell: the segmenter's outlines of bare
table are large, touch the frame's edge, or are not object-sized. A patterned
surface — tiles, a printed mat, a chessboard — has parts it outlines as
readily as an object. Two ways to say what the surface looks like:

  reference          a photo of the EMPTY surface from the same fixed camera
                     (save_reference). Of each outline, the share that differs
                     from it decides: mostly unchanged is surface. One photo
                     per surface, nothing to know about the pattern.
  surface_fraction   a deployment's own rule, mask -> share that is surface
                     (chess: how much of the outline runs along grid lines).

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np

from mrcf.perception import outlines
from mrcf.perception import plane_geometry as pg
from mrcf import privacy

# A pixel differs from the empty surface when its Lab distance from the
# reference is above this, after evening out the frame's overall brightness.
# An outline is an object when at least MIN_CHANGED of it differs.
#
# Cream on maple is only 7-14 apart: a coarser threshold does not see it. At
# 10 and 0.4, 0.60-0.97 of a cream pawn's outline differs from the empty board,
# and no bare square comes out at all.
CHANGED_LAB = 10.0
MIN_CHANGED = 0.4
# The light now is not the light the reference was taken in: cameras run on
# auto exposure and white balance, and daylight moves across the surface
# through the day. So both images are levelled before they are compared — each
# pixel's lightness as a share of its surroundings' (LIGHT_SCALE_PX across:
# about two 40 mm squares from the overhead camera), its colour as the
# difference from theirs (_levelled). Whatever changes over that scale —
# exposure, white balance, a lamp, the sun across the board — drops out; an
# object a few centimetres across does not. Compared raw, a bare board reads
# changed over 22-58% of its area hours to days after the photo; levelled,
# 0-9%, with the pieces at 67-99%.
LIGHT_SCALE_PX = 121
MAX_SURFACE = 0.4
# A shadow an object casts beside itself is the surface, darker: levelled, its
# lightness drops (by less than SHADOW_DEEPEST: a dark object drops further)
# and its colour stays within SHADOW_AB of the empty surface's. An outline
# whose changed pixels are SHADOW_SHARE or more of that is a shadow, not an
# object. UR5 tabletop, 2026-11-13: the shadow beside each block was outlined
# and named after the block, 72-84% of it so; the blocks 0-1% (their paint),
# the arm's black gripper 7-13% (darker still).
SHADOW_AB = 4.0
SHADOW_DEEPEST = 40.0
SHADOW_SHARE = 0.5
# The camera checked against the reference (drift): the surface's own features
# matched between the two, at least DRIFT_MATCHES agreeing on one slide and
# turn of the picture for it to say. UR5 tabletop, 2026-11-13: an overhead
# camera on a mount that crept 12 px (13 mm on the table) in 15 minutes after
# its calibration, and a pick came down on its block; still, the same
# measurement read 0.3-0.6 px.
DRIFT_FEATURES = 4000
DRIFT_MATCHES = 20


class CameraMoved(RuntimeError):
    """The camera's picture has slid from where it was when the reference was taken (scan's max_drift_px)."""

    def __init__(self, message: str, px: float, mm: "float | None"):
        super().__init__(message)
        self.px, self.mm = px, mm
FOOT_MM = (8.0, 120.0)          # what a graspable thing's foot can measure


def save_reference(frame_rgb: np.ndarray, path) -> None:
    """Keep a photo of the empty surface (RGB) for changed_fraction."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    privacy.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def load_reference(path, size=None) -> "np.ndarray | None":
    """The saved empty-surface photo, RGB, resized to size (w, h) if given; None if absent."""
    img = cv2.imread(str(path)) if path and Path(path).is_file() else None
    if img is None:
        return None
    if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        img = cv2.resize(img, tuple(size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def reference_from_frames(frames_rgb: list) -> np.ndarray:
    """
    The empty surface from photos with objects in different places: per pixel,
    the median. Right where no spot held an object in half of them or more —
    no need to clear the surface to take one.
    """
    return np.median(np.stack([np.asarray(f) for f in frames_rgb]), axis=0).astype(np.uint8)


def realign_reference(reference_rgb: np.ndarray, then_px, now_px) -> "tuple[np.ndarray, float] | None":
    """
    An empty-surface photo moved to where the surface is in the frame now —
    after the camera shifted — from points on the surface seen in both: the
    photo's (then_px) and now (now_px), (N, 2) pixels, matched as nearest
    neighbours (each must find a different one). The surface is a plane, so a
    homography from them moves every part of it. (lined-up photo, median shift
    in px), or None when the points do not match one to one.
    """
    then, now = np.asarray(then_px, float), np.asarray(now_px, float)
    if len(then) < 4 or len(then) != len(now):
        return None
    apart = np.linalg.norm(then[:, None] - now[None], axis=2)
    match = apart.argmin(1)
    if len(set(match.tolist())) != len(now):
        return None
    H, _ = cv2.findHomography(then, now[match], 0)
    if H is None:
        return None
    ref = np.asarray(reference_rgb)
    lined_up = cv2.warpPerspective(ref, H, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    return lined_up, float(np.median(apart.min(1)))


def _levelled(rgb: np.ndarray) -> np.ndarray:
    """Lab with the light over it taken out (see LIGHT_SCALE_PX): L as 100 x its share of the local mean, a and b less theirs."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    k = (LIGHT_SCALE_PX, LIGHT_SCALE_PX)
    L = 100.0 * lab[..., 0] / (cv2.blur(lab[..., 0], k) + 1.0)
    return np.dstack([L, lab[..., 1:] - cv2.blur(lab[..., 1:], k)])


def compare(frame_rgb: np.ndarray, reference_rgb: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """
    Per pixel, against the empty surface, whatever the light: (changed — it
    differs; shadowed — it differs only as the surface in a shadow does, see
    SHADOW_SHARE). Bool, frame-sized; shadowed is within changed.
    """
    a = cv2.GaussianBlur(_levelled(frame_rgb), (5, 5), 0)
    b = cv2.GaussianBlur(_levelled(reference_rgb), (5, 5), 0)
    diff = a - b
    changed = np.linalg.norm(diff, axis=2) > CHANGED_LAB
    dark = diff[..., 0]
    shadowed = changed & (dark < 0) & (dark > -SHADOW_DEEPEST) & (np.linalg.norm(diff[..., 1:], axis=2) < SHADOW_AB)
    return changed, shadowed


def change_map(frame_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """Per pixel: does the frame differ from the empty surface, whatever the light? Bool, frame-sized."""
    return compare(frame_rgb, reference_rgb)[0]


def is_shadow(mask: np.ndarray, changed: np.ndarray, shadowed: np.ndarray) -> bool:
    """Is the outline mask a shadow on the surface (compare's maps): SHADOW_SHARE of what changed in it is shadowed."""
    n = int(changed[mask].sum())
    return n > 0 and int(shadowed[mask].sum()) >= SHADOW_SHARE * n


def drift(frame_rgb: np.ndarray, reference_rgb: np.ndarray) -> "tuple[float, float, float, int] | None":
    """
    How far the camera's picture has moved since reference_rgb was taken, from
    the features of the surface and whatever else stands still in both: (dx,
    dy) px that the middle of the picture has slid, the turn (degrees), and how
    many features agree on it. None when too few do to say (a covered surface,
    another camera). Things on the surface now, and the arm, do not fit one
    slide of the whole picture and are left out by it.
    """
    a = cv2.cvtColor(np.asarray(reference_rgb), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2GRAY)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    orb = cv2.ORB.create(DRIFT_FEATURES)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < DRIFT_MATCHES or len(kb) < DRIFT_MATCHES:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(pairs) < DRIFT_MATCHES:
        return None
    pa = np.array([ka[p.queryIdx].pt for p in pairs], dtype=np.float32)
    pb = np.array([kb[p.trainIdx].pt for p in pairs], dtype=np.float32)
    M, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inliers is None or int(inliers.sum()) < DRIFT_MATCHES:
        return None
    h, w = a.shape
    moved = M @ np.array([w / 2.0, h / 2.0, 1.0]) - np.array([w / 2.0, h / 2.0])
    turn = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return float(moved[0]), float(moved[1]), turn, int(inliers.sum())


def check_drift(frame_rgb: np.ndarray, reference_rgb: np.ndarray, view: "pg.PlaneView", max_px: float) -> None:
    """
    Raise CameraMoved when the camera has moved more than max_px since the
    reference was taken: everything the scan measured would be that far off.
    Says so and goes on when it cannot tell (drift None).
    """
    got = drift(frame_rgb, reference_rgb)
    if got is None:
        print("  [Scan] Could not check the camera against the empty-surface photo (too few features matched).")
        return
    dx, dy, turn, _ = got
    px = float(np.hypot(dx, dy))
    if px <= max_px:
        return
    h, w = np.asarray(frame_rgb).shape[:2]
    ends = view.to_plane(np.array([[w / 2.0, h / 2.0], [w / 2.0 + dx, h / 2.0 + dy]]))
    mm = float(np.linalg.norm(ends[1] - ends[0]) * view.scale) if np.isfinite(ends).all() else None
    raise CameraMoved(
        f"the overhead camera has moved since its photo of the empty surface: {px:.0f} px"
        + (f" (about {mm:.0f} mm on the surface)" if mm is not None else "") + f", turned {turn:+.1f} deg. "
        "Everything it sees would be that far off. Secure it, calibrate it again "
        "(tools/calibrate_camera_by_hand.py), then retake the empty-surface photo "
        "(tools/capture_surface_reference.py).", px, mm)


def without_shadow(mask: np.ndarray, shadowed: np.ndarray, min_px: int = 30) -> np.ndarray:
    """
    The outline mask less the shadow in it (compare's shadowed): its largest
    piece left. An object's outline takes in some of the shadow beside it, and
    measured with it the object stands toward its shadow: UR5 tabletop,
    2026-11-13, a block's centre 9 mm off with it, 3 mm without. The mask as it
    was when too little is left to measure.
    """
    left = (np.asarray(mask, bool) & ~shadowed).astype(np.uint8)
    left = cv2.morphologyEx(left, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))     # no threads of it left
    n, labels, stats, _ = cv2.connectedComponentsWithStats(left, connectivity=8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < min_px:
        return mask
    return labels == biggest


def scan(env, camera: str, width: int, height: int, x_range_m, y_range_m, *,
         reference: "np.ndarray | None" = None, surface_fraction=None,
         step_mm: float = 12.0, foot_mm=FOOT_MM,
         max_drift_px: "float | None" = None) -> "tuple[np.ndarray, list] | None":
    """
    Every object standing in the rectangle x_range_m x y_range_m (arm frame)
    as `camera` sees it now: (frame, [found]), each found a dict with
    "measurement" (plane_geometry.ObjectMeasurement, base_world_m set),
    "mask", "score", "changed" (share differing from the reference, or None)
    and "appearance" ({"L", "a", "b"}: the median colour of its middle).
    None when the backend cannot locate this camera or there is no segmenter.
    Raises CameraMoved, before anything is outlined, when max_drift_px is given
    and the camera has moved further than that since the reference was taken.
    """
    from mrcf.perception import segment

    model = env.camera_model(camera, width, height)
    if model is None or not segment.available():
        return None
    frame = env.render_image(camera, width=width, height=height)
    view = pg.PlaneView.from_camera_model(model)
    if reference is not None and max_drift_px is not None:
        check_drift(frame, reference, view, max_drift_px)         # before the segmenter: a moved camera costs nothing
    prompts = pg.surface_grid_px(view, np.asarray(x_range_m) * 1000, np.asarray(y_range_m) * 1000, step_mm)
    h, w = frame.shape[:2]
    prompts = prompts[(prompts[:, 0] >= 2) & (prompts[:, 0] < w - 2)
                      & (prompts[:, 1] >= 2) & (prompts[:, 1] < h - 2)]
    changed, shadowed = compare(frame, reference) if reference is not None else (None, None)

    def object_share(m):
        share = 1.0
        if changed is not None:
            share = min(share, float(changed[m].mean()))
        if surface_fraction is not None:
            share = min(share, 1.0 - float(surface_fraction(m)))
        return share

    def reject(m):
        if changed is not None and shadowed is not None and (changed[m].mean() < MIN_CHANGED
                                                             or is_shadow(m, changed, shadowed)):
            return True
        return surface_fraction is not None and surface_fraction(m) > MAX_SURFACE

    def accept(meas):
        return foot_mm[0] <= 2 * meas.base_radius_mm <= foot_mm[1]

    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    found = []
    trim = functools.partial(without_shadow, shadowed=shadowed) if shadowed is not None else None
    for score, m, meas in pg.outline_objects(frame, view, prompts, reject_mask=reject, accept=accept,
                                             weigh=object_share, prefer="score", trim=trim):
        found.append({"measurement": meas, "mask": m, "score": score,
                      "changed": None if changed is None else float(changed[m].mean()),
                      "appearance": appearance(lab, m)})
    return frame, found


def appearance(lab: np.ndarray, mask: np.ndarray) -> dict:
    """Median Lab colour of the middle of an outline, away from its edge (where surface creeps in)."""
    middle = outlines.core(mask)
    L, a, b = (float(np.median(lab[..., i][middle])) for i in range(3))
    return {"L": L, "a": a - 128.0, "b": b - 128.0}
