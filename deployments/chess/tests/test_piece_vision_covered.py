"""piece_vision: a piece that covers its whole square is still a piece (a rook from above, 2026-10-19)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from deployments.chess import piece_vision as pv  # noqa: E402

LIGHT, DARK, WOOD, HOLLOW = (232, 220, 196), (112, 62, 42), (128, 74, 48), (55, 30, 20)   # RGB
SQUARE_PX = 40          # texture pixels per square
N = 30                  # squares each way: more than the camera sees


def _camera():
    """World (board, units of squares) to camera, from the servo's hover: ~3 squares up, looking ~35 deg off vertical."""
    c, target = np.array([15.0, 17.5, -2.8]), np.array([15.0, 15.5, 0.0])     # board z points away from it
    z = (target - c) / np.linalg.norm(target - c)
    x = np.array([1.0, 0.0, 0.0]) - z * z[0]
    x /= np.linalg.norm(x)
    r = np.array([x, np.cross(z, x), z])
    return r, -r @ c


def _view(piece_at=None):
    k = np.asarray(pv.cfg.GRIPPER_CAM_K, float)
    tex = np.zeros((N * SQUARE_PX, N * SQUARE_PX, 3), np.uint8)          # board to the frame's edges
    for i in range(N):
        for j in range(N):
            tex[i * SQUARE_PX:(i + 1) * SQUARE_PX, j * SQUARE_PX:(j + 1) * SQUARE_PX] = LIGHT if (i + j) % 2 == 0 else DARK
    r, t = _camera()
    h = k @ np.column_stack([r[:, 0], r[:, 1], t]) @ np.diag([1 / SQUARE_PX, 1 / SQUARE_PX, 1.0])
    img = cv2.warpPerspective(tex, h, (640, 480), flags=cv2.INTER_AREA, borderValue=(90, 90, 90))
    if piece_at is None:
        return img, None
    # A disc 0.9 of a square across over the light square piece_at: ~85% of the
    # square hidden, as a rook's top hid its square from the hover. With a
    # hollow, as the rook's top has: a real piece is not one flat colour (the
    # squares pieces covered on the rig were 0.30-0.41 uniform, glare 1.0).
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for radius, colour in ((0.45, WOOD), (0.30, HOLLOW)):
        rim = np.c_[piece_at[0] + 0.5 + radius * np.cos(a), piece_at[1] + 0.5 + radius * np.sin(a), np.zeros(64)]
        px, _ = cv2.projectPoints(rim, cv2.Rodrigues(r)[0], t, k, np.zeros(5))
        cv2.fillPoly(img, [px.reshape(-1, 2).astype(np.int32)], colour)
    return img, px.reshape(-1, 2).mean(0)


@pytest.fixture(autouse=True)
def _undistorted_lens(monkeypatch):
    from mrcf.backends.so101 import robot_config
    monkeypatch.setattr(robot_config, "GRIPPER_CAM_DIST", [0.0] * 5)
    monkeypatch.setattr(pv, "_rays_cache", {})


def _near(blobs, where, px=25.0):
    return [b for b in blobs if np.linalg.norm(np.asarray(b[:2]) - where) < px]


def test_a_piece_hiding_its_square_is_found(monkeypatch):
    img, centre = _view(piece_at=(15, 15))               # (15 + 15) even: a light square
    assert _near(pv._blobs(img), centre)
    # Without judging covered squares it is invisible: the square was taken for off the board.
    monkeypatch.setattr(pv, "_COVERED_BESIDE", 5)
    assert not _near(pv._blobs(img), centre)


def test_a_square_of_glare_is_board_not_a_piece():
    # A dark square mirroring the window: one even, bluish grey, between the
    # two board colours — flagged whole, it pulled the piece's centre 65 px.
    img, _ = _view()
    r, t = _camera()
    k = np.asarray(pv.cfg.GRIPPER_CAM_K, float)
    corners = np.array([[14, 15, 0], [15, 15, 0], [15, 16, 0], [14, 16, 0]], float)   # (14 + 15) odd: dark
    px, _ = cv2.projectPoints(corners, cv2.Rodrigues(r)[0], t, k, np.zeros(5))
    cv2.fillPoly(img, [px.reshape(-1, 2).astype(np.int32)], (140, 135, 150))
    assert pv._blobs(img) == []


def test_the_board_alone_shows_nothing():
    img, _ = _view()
    assert pv._blobs(img) == []
