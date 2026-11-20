"""
fiducials.py — find printed markers in a camera frame and turn them into poses.

The rule they serve: store only what is rigid, measure what passes through a
servo. A marker glued to something rigid is a stored constant that cannot
drift; where it is in a frame is a measurement, every frame.

Two kinds:

  AprilTags (tag36h11)  square tags of known size — on a work surface's border,
            on a gripper body. Each gives its pose relative to the camera:
            precise across the image (sub-pixel corners), weaker along the
            camera ray, where a small tag's distance comes only from its
            apparent size. A small flat tag also has the classic two-fold
            ambiguity: two poses, near mirror images about the line of sight,
            fit its corners almost equally well, and noise can make the wrong
            one win. In simulation a 24 mm tag at 0.55 m jumped 35-160 mm that
            way at the grasp point on about one frame in twenty. A caller that
            knows roughly how the thing is turned (an arm's own kinematics, to a
            few degrees) passes that, and the pose that agrees with it is used.

  Bullseyes  a white disc with a darker centre dot, where a tag will not fit —
            on the jaw tips, for a wrist camera. Their centre is the measurement.

No knowledge of what the markers are stuck on: ids, sizes and layouts come
from the backend and the deployment.

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11

# Tags are looked for in a copy of the frame enlarged this much. On the rig a
# 24 mm tag ~46 px across and a 26 mm board tag ~31 px were missed at native
# resolution and found at 2x: the detector's border search needs a few pixels
# per cell, and a tag seen at a slant has fewer along one side.
DETECT_UPSCALE = 2.0

# A sheet of tags to lay flat on any surface, as a calibration target where
# there is no board (tools/make_tag_sheets.py prints it; tools/so101/
# calibrate_gripper_pose.py --target tags): ids clear of the board's (0-5) and
# the gripper's (10-12), black squares this size, in a 3 x 4 grid.
TABLE_TARGET_IDS = tuple(range(20, 32))
TABLE_TARGET_TAG_M = 0.030

_detectors: dict = {}


def _detector() -> "cv2.aruco.ArucoDetector":
    if "tags" not in _detectors:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _detectors["tags"] = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(TAG_DICTIONARY), params)
    return _detectors["tags"]


def _gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def detect_tags(frame: np.ndarray, upscale: float = DETECT_UPSCALE) -> dict[int, np.ndarray]:
    """
    Every tag in the frame: {id: (4, 2) corner pixels in the frame's own
    coordinates}, corners in OpenCV's order — top-left, top-right,
    bottom-right, bottom-left of the tag as printed.
    """
    gray = _gray(frame)
    if upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None:
        return {}
    # Pixel centres scale about the half-pixel offset, not the origin.
    return {int(i): (c.reshape(4, 2) + 0.5) / upscale - 0.5 for c, i in zip(corners, ids.ravel())}


def tag_object_points(size_m: float) -> np.ndarray:
    """A tag's black-square corners in its own frame (x right, y up, z out of the face)."""
    h = size_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=float)


@dataclass
class TagPose:
    """One tag's pose: R, t map tag-frame points into the camera frame (metres)."""
    R: np.ndarray
    t: np.ndarray
    err_px: float

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T


# A second pose is a real alternative only if it fits nearly as well as the best.
ALTERNATIVE_FIT_PX = 2.0


def tag_poses(corners: np.ndarray, size_m: float, K, D) -> list[TagPose]:
    """
    The poses that fit one square tag — both, where it is ambiguous — best
    reprojection first; [] if nothing fits.

    Several solvers, each scored by its own reprojection rather than trusted:
    OpenCV's square-tag IPPE returns NaN for a tag tilted exactly on the
    optical axis, and a 180-degree flip with ~30 px error for one seen exactly
    square-on, where SQPNP is fine. Real frames are rarely that symmetric, but
    near it the same arithmetic goes unstable.
    """
    obj = tag_object_points(size_m)
    img = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
    K, D = np.asarray(K, float), np.asarray(D, float)
    candidates = []
    for flag in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP):
        try:
            _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, D, flags=flag)
        except cv2.error:
            continue
        for r, t in zip(rvecs, tvecs):
            if not (np.isfinite(r).all() and np.isfinite(t).all()) or t.ravel()[2] <= 0:
                continue
            proj, _ = cv2.projectPoints(obj, r, t, K, D)
            err = float(np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1).mean())
            candidates.append(TagPose(cv2.Rodrigues(r)[0], t.ravel(), err))
    if not candidates:
        return []
    best = min(p.err_px for p in candidates)
    kept: list[TagPose] = []
    for p in sorted(candidates, key=lambda p: p.err_px):
        if p.err_px <= best + ALTERNATIVE_FIT_PX and \
                all(rotation_angle_deg(p.R, k.R) > 2.0 for k in kept):
            kept.append(p)
    return kept


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """The angle of the rotation between two orientations."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pick_pose(poses: list[TagPose], R_expected: "np.ndarray | None" = None) -> TagPose:
    """
    The pose to use: the one turned most like R_expected (tag -> camera) when
    the caller knows roughly how the tag is turned, else the best fit.
    """
    if R_expected is None or len(poses) == 1:
        return poses[0]
    return min(poses, key=lambda p: rotation_angle_deg(p.R, R_expected))


def position_with_known_rotation(corners: np.ndarray, size_m: float, R_cam_tag: np.ndarray, K, D,
                                 t_guess: np.ndarray, iters: int = 10) -> "tuple[np.ndarray, float]":
    """
    Where a tag is (its centre in the camera frame, metres) when how it is
    turned is already known — from an arm's kinematics, say — and the mean
    reprojection error, px. Three unknowns instead of six: a small tag's own
    tilt is its noisiest part, and through a lever (a tag ~90 mm from the jaws)
    it dominates the error. On the rig, tag position with the arm's rotation
    agreed with the arm to 0.6 mm median, the full tag pose to 2.2.
    """
    obj = tag_object_points(size_m)
    img = np.asarray(corners, float).reshape(4, 2)
    K, D = np.asarray(K, float), np.asarray(D, float)
    rvec = cv2.Rodrigues(np.asarray(R_cam_tag, float))[0]

    def residual(t):
        px = cv2.projectPoints(obj, rvec, t, K, D)[0].reshape(4, 2)
        return (px - img).ravel()

    t = np.asarray(t_guess, float).copy()
    for _ in range(iters):
        r = residual(t)
        J = np.stack([(residual(t + e) - r) / 1e-6 for e in np.eye(3) * 1e-6], axis=1)
        with np.errstate(all="ignore"):           # Accelerate's spurious matmul warnings
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
        t = t + step
        if np.abs(step).max() < 1e-7:
            break
    return t, float(np.linalg.norm(residual(t).reshape(4, 2), axis=1).mean())


def layout_pose(found: dict[int, np.ndarray], layout: dict[int, np.ndarray], K, D
                ) -> "tuple[np.ndarray, int, float] | None":
    """
    The pose of a rigid set of tags from every one of them in view at once:
    (4x4 layout frame -> camera, tags used, mean reprojection px), or None.
    layout: {id: (4, 3) corner positions in the layout's frame, OpenCV order}.
    One tag is enough to solve; more pin down the depth and the tilt.
    """
    ids = [i for i in found if i in layout]
    if not ids:
        return None
    obj = np.vstack([np.asarray(layout[i], float) for i in ids])
    img = np.vstack([found[i] for i in ids]).astype(np.float64)
    K, D = np.asarray(K, float), np.asarray(D, float)
    flag = cv2.SOLVEPNP_IPPE if len(ids) == 1 else cv2.SOLVEPNP_SQPNP
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=flag)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return T, len(ids), float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())


# ── Bullseyes ─────────────────────────────────────────────────────────────────

@dataclass
class Bullseye:
    centre: tuple[float, float]       # the dot's centroid, px
    disc_area_px: float
    ellipse: tuple                     # cv2.fitEllipse of the disc


# The disc is the brightest thing round it (white paper on a dark jaw) but not
# always brighter than everything in the frame: a light board square can be as
# bright as an unevenly lit sticker. So it is looked for at several levels, and
# a candidate must have a dot in its middle, be convex, and be brighter than
# most — not all — of a ring round it: a sticker can run into a light square
# along one edge, or its dot be half washed out by glare.
BULLSEYE_LEVELS = (250, 240, 225, 210, 190)


def find_bullseyes(frame: np.ndarray, min_area_px: float = 40, max_area_px: float = 8000,
                   levels=BULLSEYE_LEVELS, merge_px: float = 10.0) -> list[Bullseye]:
    """White discs with a darker centre dot, one per mark, left to right."""
    gray = _gray(frame)
    found: list[tuple[float, Bullseye]] = []
    for level in levels:
        bw = (gray >= level).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        for i, c in enumerate(contours):
            if hierarchy[0][i][3] != -1 or len(c) < 8:
                continue
            area = cv2.contourArea(c)
            if not min_area_px <= area <= max_area_px:
                continue
            holes, child = [], hierarchy[0][i][2]
            while child != -1:
                if cv2.contourArea(contours[child]) >= 4:
                    holes.append(contours[child])
                child = hierarchy[0][child][0]
            if not holes:
                continue
            dot = max(holes, key=cv2.contourArea)
            if not 0.03 * area <= cv2.contourArea(dot) <= 0.5 * area:
                continue
            ellipse = cv2.fitEllipse(c)
            (ex, ey), axes, _ = ellipse
            m = cv2.moments(dot)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            if np.hypot(cx - ex, cy - ey) > 0.3 * max(axes):
                continue
            if area / max(cv2.contourArea(cv2.convexHull(c)), 1.0) < 0.8:
                continue
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [c], -1, 255, -1)
            ring = cv2.dilate(mask, np.ones((7, 7), np.uint8)) & ~mask
            inside = float(gray[mask > 0].mean())
            darker = float((gray[ring > 0] < inside - 40).mean())
            if darker < 0.5:
                continue
            found.append((darker * min(area, 3000.0), Bullseye((cx, cy), area, ellipse)))
    kept: list[Bullseye] = []
    for _, b in sorted(found, key=lambda s: -s[0]):
        if all(np.hypot(b.centre[0] - k.centre[0], b.centre[1] - k.centre[1]) > merge_px for k in kept):
            kept.append(b)
    return sorted(kept, key=lambda b: b.centre[0])
