"""tiles: objects cut from one frame upright, numbered on one sheet for a vision model to name at once."""

import numpy as np

try:
    import cv2  # noqa: F401 — the tiles are cut with it
except ImportError:
    import pytest
    pytest.skip("needs OpenCV", allow_module_level=True)

from mrcf.perception import plane_geometry as pg
from mrcf.perception import tiles
from mrcf.perception.expected_view import ExpectedView, KnownObject

SHAPE = (720, 1280)


def _view(roll_deg=0.0):
    """A camera 0.4 m up and 0.25 m back, looking down at a slant, turned roll_deg about its axis."""
    a = np.radians(40.0)
    z = np.array([0.0, np.sin(a), -np.cos(a)])
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    r = np.radians(roll_deg)
    x, y = np.cos(r) * x + np.sin(r) * y, -np.sin(r) * x + np.cos(r) * y
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = [0.0, -0.25, 0.40]
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    return pg.PlaneView.from_camera_model({"K": K, "D": np.zeros(5), "T_world_cam": T, "plane_z_m": 0.0})


def _frame_with(view, obj):
    """A white frame with obj's column drawn dark, as the camera would see it."""
    frame = np.full((*SHAPE, 3), 255, np.uint8)
    frame[ExpectedView(view, [obj], SHAPE).masks[obj.key]] = 40
    return frame


def _axis_deg(tile):
    """The dark shape's long axis, degrees from the tile's vertical."""
    ys, xs = np.nonzero(tile[..., 0] < 128)
    pts = np.c_[xs - xs.mean(), ys - ys.mean()]
    _, vecs = np.linalg.eigh(pts.T @ pts)
    vx, vy = vecs[:, -1]
    return float(np.degrees(np.arctan2(abs(vx), abs(vy))))


def test_an_object_leaning_across_the_frame_stands_upright_in_its_tile():
    tall = KnownObject("king", np.array([0.05, 0.10]), 0.012, 0.080)
    for roll in (0.0, 35.0, -60.0):                 # however the camera is turned
        view = _view(roll)
        tile = tiles.upright(_frame_with(view, tall), view, tall.foot_xy_m * 1000.0, 12.0, 80.0)
        assert tile.shape == (160, 112, 3)
        ys, xs = np.nonzero(tile[..., 0] < 128)
        assert _axis_deg(tile) < 4.0, roll
        assert abs(xs.mean() - 56) < 3                                # in the middle, across
        assert ys.max() > 120 and ys.min() < 50                      # foot near the bottom, top near the top


def test_a_short_object_stands_on_the_same_foot_line_as_a_tall_one():
    # Every tile is framed to the tallest object's height, so a short one
    # reads short beside a tall one on the sheet.
    view = _view()
    pawn = KnownObject("pawn", np.array([0.05, 0.10]), 0.012, 0.040)
    king = KnownObject("king", np.array([0.05, 0.10]), 0.012, 0.080)
    short, tall = (tiles.upright(_frame_with(view, o), view, o.foot_xy_m * 1000.0, 12.0, 80.0) for o in (pawn, king))
    rows = [np.nonzero((t[..., 0] < 128).any(axis=1))[0] for t in (short, tall)]
    assert abs(rows[0].max() - rows[1].max()) <= 2                   # the same foot line
    assert rows[0].min() > rows[1].min() + 25                        # the pawn's top well below the king's


def test_the_sheet_numbers_tiles_in_rows():
    tile = np.zeros((160, 112, 3), np.uint8)
    out = tiles.sheet([tile] * 10)
    assert out.shape == (2 * (160 + 22), 8 * 112, 3)
    assert tiles.sheet([tile] * 3).shape == (160 + 22, 3 * 112, 3)
    assert isinstance(tiles.png_b64(out), str)
