"""Measuring an upright object on a plane from a located camera — no board, no segmenter.

Silhouettes are rendered from solids of revolution with the gripper camera's
real lens, from where a hover puts it.
"""

import numpy as np
import pytest

try:
    import cv2
except ImportError:          # the framework's maths needs OpenCV; skip where it is absent
    pytest.skip("needs OpenCV", allow_module_level=True)

from mrcf.perception import plane_geometry as pg

WRIST_K = np.array([[343.0, 0, 341], [0, 342.0, 263], [0, 0, 1]])
OVERHEAD_K = np.array([[430.0, 0, 320], [0, 430.0, 240], [0, 0, 1]])
TARGET = np.array([0.20, 0.02, 0.0])

PAWN = (lambda h: 11.0 if h < 28 else (6.0 if h < 32 else np.sqrt(max(36 - (h - 38) ** 2, 0.0))), 44.0, 11.0)
CONE = (lambda h: 11.0 * (1 - h / 50.0), 50.0, 11.0)
CYLINDER = (lambda h: 11.0, 44.0, 11.0)
MUG = (lambda h: 40.0, 95.0, 40.0)


def look_at(eye, target) -> np.ndarray:
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]
    T[:3, 3] = eye
    return T


def view_from(eye_m, look_m, K=WRIST_K, scale=1.0, plane_z=0.0) -> pg.PlaneView:
    return pg.PlaneView.from_camera_model({"K": K, "D": np.zeros(5), "plane_z_m": plane_z, "scale": scale,
                                           "T_world_cam": look_at(np.asarray(eye_m, float), look_m)})


def silhouette(view, shape, at=TARGET, size=(480, 640)) -> np.ndarray:
    """Mask of a solid of revolution (radius(h), height, real mm) standing at `at`."""
    radius, height, _ = shape
    a = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    base = (np.asarray(at, float)[:2] - (view.origin_m if view.origin_m is not None else 0)) * 1000
    rings = []
    for h in np.linspace(0, height, 80):
        r = radius(h) / view.scale
        pts = np.c_[base[0] + r * np.cos(a), base[1] + r * np.sin(a), np.zeros(len(a))] + view.up * h / view.scale
        rings.append(view.project(pts))
    mask = np.zeros(size, np.uint8)
    for r0, r1 in zip(rings[:-1], rings[1:]):
        cv2.fillConvexPoly(mask, cv2.convexHull(np.r_[r0, r1].astype(np.float32)).astype(np.int32), 1)
    return mask.astype(bool)


def wrist_view(**kw) -> pg.PlaneView:
    return view_from([0.14, 0.02, 0.11 + kw.get("plane_z", 0.0)], TARGET + [0, 0, 0.03 + kw.get("plane_z", 0.0)], **kw)


def measured(view, shape):
    mask = silhouette(view, shape)
    assert not pg.touches_frame(mask)
    return pg.measure(mask, view)


def test_rounded_and_pointed_tops_measure_true():
    for shape in (PAWN, CONE):
        m = measured(wrist_view(), shape)
        assert abs(m.height_mm - shape[1]) < 3.0
        assert abs(2 * m.base_radius_mm - 2 * shape[2]) < 2.5
        assert np.linalg.norm(m.base_world_m - TARGET[:2]) * 1000 < 2.0


def test_a_flat_top_reads_high_never_low():
    # The outline's highest point is the far rim of a flat top, traced back to
    # the axis: high by up to the top's radius x tan(depression). Safe for a
    # hover; documented, not hidden.
    m = measured(wrist_view(), CYLINDER)
    assert CYLINDER[1] <= m.height_mm <= CYLINDER[1] + CYLINDER[2] * 1.1 + 3.0


def test_overhead_view_has_foot_but_no_height():
    view = view_from([0.25, -0.10, 0.52], TARGET, K=OVERHEAD_K)
    m = measured(view, PAWN)
    assert m.height_mm is None
    assert abs(2 * m.base_radius_mm - 22.0) < 2.5
    assert np.linalg.norm(m.base_world_m - TARGET[:2]) * 1000 < 2.0


def test_a_wide_tall_object_close_up_is_not_read_by_its_body():
    view = view_from([0.10, 0.02, 0.16], TARGET + [0, 0, 0.03])
    m = measured(view, MUG)
    assert abs(2 * m.base_radius_mm - 80.0) < 8.0
    assert np.linalg.norm(m.base_world_m - TARGET[:2]) * 1000 < 4.0


def test_a_frame_that_reads_short_still_gives_real_sizes():
    # The arm's frame reads 9% short: positions stay in it, sizes come out real.
    m = measured(wrist_view(scale=1.0979), PAWN)
    assert abs(m.height_mm - 44.0) < 3.0
    assert abs(2 * m.base_radius_mm - 22.0) < 2.5
    assert np.linalg.norm(m.base_world_m - TARGET[:2]) * 1000 < 2.0


def test_the_surface_need_not_be_at_zero():
    m = measured(wrist_view(plane_z=0.012), PAWN)
    assert abs(m.height_mm - 44.0) < 3.0
    assert np.linalg.norm(m.base_world_m - TARGET[:2]) * 1000 < 2.0


# ── A wrist camera's sight line at another tilt ───────────────────────────────

def _leaned(deg):
    """A gripper straight down (its z = world -z), leaned deg about its own y."""
    a = np.radians(deg)
    return np.diag([1.0, -1.0, -1.0]) @ np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                                                  [-np.sin(a), 0, np.cos(a)]])


DOWN = np.array([0.0, 0.0, -1.0])
CAMERA = np.array([-0.04, 0.033, -0.053])        # behind, beside and above the jaws


def test_aligned_at_its_own_tilt_and_depth_the_servo_is_straight_over_the_point():
    ref = _leaned(-5)
    assert np.allclose(pg.aligned_offset(ref, ref.T @ DOWN, CAMERA, 0.049, 0.049), 0, atol=1e-12)


def test_with_the_camera_at_the_jaws_a_lean_swings_the_sight_line_by_its_angle():
    ref = _leaned(0)
    off = pg.aligned_offset(_leaned(30), ref.T @ DOWN, np.zeros(3), 0.05, 0.05)
    assert np.linalg.norm(off) == pytest.approx(0.05 * np.tan(np.radians(30)))


def test_the_aligned_point_lies_on_the_calibrated_sight_line_at_its_own_depth():
    ref, now = _leaned(-5), _leaned(-35)
    off = pg.aligned_offset(now, ref.T @ DOWN, CAMERA, 0.049, 0.045)
    point = np.array([off[0], off[1], -0.045])                # from the grasp point, world
    seen_now = now.T @ point - CAMERA                         # camera to point, gripper frame
    seen_then = 0.049 * (ref.T @ DOWN) - CAMERA
    assert np.allclose(np.cross(seen_now, seen_then), 0, atol=1e-12) and seen_now @ seen_then > 0
    # Rank 8 against a calibration in mid-board: the jaws' slack several times over.
    assert 0.018 < np.linalg.norm(off) < 0.032
