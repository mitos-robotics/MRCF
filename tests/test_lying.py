"""
mrcf/perception/lying.py: an object on its side, from its outline in a fixed camera
— which way it lies, which end it stands on, where the knob at its top is.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs OpenCV: pip install -e '.[measure]'")

from mrcf.perception import lying  # noqa: E402
from mrcf.perception import plane_geometry as pg  # noqa: E402

# A camera placed as the rig's overhead one is: half a metre up, behind the
# board's far side, looking down at ~15-20 deg where the pieces are.
CAMERA_M = np.array([0.35, 0.01, 0.535])


def overhead_view() -> pg.PlaneView:
    R_wc = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # looking straight down
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_wc, CAMERA_M
    K = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
    return pg.PlaneView.from_camera_model({"K": K, "D": np.zeros(5), "T_world_cam": T, "plane_z_m": 0.0})


def pawn_radius_mm(d: float) -> float:
    """A 40 mm pawn's profile, d mm from its foot: foot, body, collar, neck, head."""
    if d < 6:
        return 12.0
    if d < 24:
        return 12.0 - (d - 6) * 0.25
    if d < 27:
        return 9.0
    if d < 28:
        return 4.0
    return np.sqrt(max(0.0, 6.5 ** 2 - (d - 33.5) ** 2))


def lying_pawn_mask(view, foot_xy_m, along_deg: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The outline of a pawn on its side (axis 12 mm up, foot at foot_xy_m), and where its head's middle is."""
    a = np.radians(along_deg)
    axis = np.array([np.cos(a), np.sin(a), 0.0])
    side = np.array([-axis[1], axis[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    foot = np.array([foot_xy_m[0] * 1000, foot_xy_m[1] * 1000, 12.0])
    pts = [foot + axis * d + pawn_radius_mm(d) * (np.cos(t) * side + np.sin(t) * up)
           for d in np.arange(0.0, 40.01, 0.25) for t in np.linspace(0, 2 * np.pi, 72, endpoint=False)]
    px = view.project(np.array(pts))
    mask = np.zeros((720, 1280), np.uint8)
    for p in px.round().astype(int):
        if 0 <= p[0] < 1280 and 0 <= p[1] < 720:
            mask[p[1], p[0]] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    head = (foot[:2] + axis[:2] * 33.5) / 1000
    return mask, head, np.asarray(foot_xy_m, float)


@pytest.mark.parametrize("along_deg", [0.0, 20.0, 90.0, 200.0, -60.0])
def test_a_pawn_on_its_side_is_measured_from_above(along_deg):
    view = overhead_view()
    mask, head, foot = lying_pawn_mask(view, (0.21, -0.01), along_deg)
    m = lying.measure_lying(mask, view)
    assert m is not None and m.ends_certain
    a = np.radians(along_deg)
    assert m.axis @ [np.cos(a), np.sin(a)] > 0.99                     # foot to head, the right way round
    assert m.length_mm == pytest.approx(40.0, abs=3.0)
    assert np.linalg.norm(m.top_m - head) < 0.003                       # the head's middle, within 3 mm
    assert np.linalg.norm(m.foot_end_m - foot) < 0.004
    assert m.top_width_mm == pytest.approx(13.0, abs=3.0) and m.foot_width_mm == pytest.approx(24.0, abs=3.0)
    assert m.elongation > lying.MIN_ELONGATION


def test_a_round_blob_has_no_ends_to_tell_apart():
    view = overhead_view()
    mask = np.zeros((720, 1280), np.uint8)
    centre = view.project(np.array([[230.0, 0.0, 0.0]]))[0].round().astype(int)
    cv2.circle(mask, tuple(int(v) for v in centre), 30, 1, -1)
    m = lying.measure_lying(mask.astype(bool), view)
    assert m is None or not m.ends_certain or m.elongation < lying.MIN_ELONGATION


def test_too_small_to_measure_is_none():
    assert lying.measure_lying(np.zeros((720, 1280), bool), overhead_view()) is None


def standing_pawn_mask(view, foot_xy_m) -> np.ndarray:
    """The outline of an upright 40 mm pawn at foot_xy_m, as the slanted camera sees it."""
    foot = np.array([foot_xy_m[0] * 1000, foot_xy_m[1] * 1000, 0.0])
    pts = [foot + [pawn_radius_mm(d) * np.cos(t), pawn_radius_mm(d) * np.sin(t), d]
           for d in np.arange(0.0, 40.01, 0.25) for t in np.linspace(0, 2 * np.pi, 72, endpoint=False)]
    px = view.project(np.array(pts)).round().astype(int)
    mask = np.zeros((720, 1280), np.uint8)
    mask[px[:, 1], px[:, 0]] = 1
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)


def test_an_upright_pawn_seen_at_a_slant_is_not_taken_for_one_lying():
    view = overhead_view()
    m = lying.measure_lying(standing_pawn_mask(view, (0.15, -0.01)), view)     # far from the camera: slanted
    assert m is not None and lying.looks_upright(m, view)


def test_a_pawn_lying_with_its_top_toward_the_camera_cannot_be_upright():
    view = overhead_view()
    toward = lying.measure_lying(lying_pawn_mask(view, (0.21, -0.01), 10.0)[0], view)    # top toward the camera
    across = lying.measure_lying(lying_pawn_mask(view, (0.21, -0.01), 90.0)[0], view)
    assert not lying.looks_upright(toward, view) and not lying.looks_upright(across, view)
    # Lying straight away from it, it might be either: taken for upright (find_lying says so).
    away = lying.measure_lying(lying_pawn_mask(view, (0.25, -0.01), 180.0)[0], view)
    assert lying.looks_upright(away, view)


def test_in_a_wrist_frame_a_pawn_outlined_in_parts_is_put_together_and_a_square_is_not_it(monkeypatch):
    from mrcf.perception import segment
    # A camera straight down, 0.3 mm a pixel: pixel (u, v) is world (0.2 + (v - 240) * 3e-4, (u - 320) * 3e-4).
    def to_surface(px, height_m=0.0):
        px = np.atleast_2d(px)
        return np.c_[0.2 + (px[:, 1] - 240) * 3e-4, (px[:, 0] - 320) * 3e-4]

    def to_px(x_mm, y_mm):
        return np.array([320 + y_mm / 0.3, 240 + (x_mm - 200) / 0.3])

    u, v = np.meshgrid(np.arange(640), np.arange(480))
    def region(inside):
        return inside(*np.moveaxis(to_surface(np.c_[u.ravel(), v.ravel()]).reshape(480, 640, 2) * 1000, -1, 0))
    # Lying along +x, foot end at x = 190 mm: the foot and body one part, the head another (as SAM gives them).
    radius = np.vectorize(pawn_radius_mm)
    body = region(lambda x, y: (x >= 190) & (x < 218) & (np.abs(y) <= radius(np.clip(x - 190, 0, 40))))
    head = region(lambda x, y: (x >= 218) & (x <= 230) & (np.abs(y) <= radius(np.clip(x - 190, 0, 40))))
    square = region(lambda x, y: (x >= 175) & (x < 215) & (y > 14) & (y < 54))
    monkeypatch.setattr(segment, "available", lambda: True)
    monkeypatch.setattr(segment, "candidates", lambda frame, prompts, **_: [(0.98, square, 0), (0.95, body, 1),
                                                                             (0.9, head, 2)])
    before = lying.LyingMeasurement(foot_end_m=np.array([0.192, 0.002]), top_end_m=np.array([0.232, 0.002]),
                                    top_m=np.array([0.2255, 0.002]), length_mm=40.0, top_width_mm=13.0,
                                    foot_width_mm=24.0, widths_mm=[], ends_certain=True, elongation=1.7)
    m, why = lying.find_lying_in_frame(np.zeros((480, 640, 3), np.uint8), lying.SurfaceMap(to_surface),
                                       to_px(212, 0), before)
    assert m is not None, why
    assert m.length_mm == pytest.approx(40.0, abs=2.0) and m.axis @ [1.0, 0.0] > 0.99
    assert np.linalg.norm(m.top_m - [0.2235, 0.0]) < 0.0015
    assert m.foot_width_mm == pytest.approx(24.0, abs=2.0)
