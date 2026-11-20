"""board_tags: which colour a1 is, judged with the pieces set up."""

import numpy as np

try:
    import cv2
except ImportError:
    import pytest
    pytest.skip("needs OpenCV", allow_module_level=True)

from deployments.chess import board_tags
from deployments.chess import config as cfg

# A camera 0.6 m straight above the board's centre: the board frame's x to the
# image's right, its y (toward rank 8) up the image. 800 x 800, no distortion.
K = np.array([[1000.0, 0, 400], [0, 1000.0, 400], [0, 0, 1]])
D = np.zeros(5)
RVEC = np.array([np.pi, 0.0, 0.0])
TVEC = np.array([0.0, 0.0, 0.6])


def _px(x, y):
    return cv2.projectPoints(np.array([[x, y, 0.0]]), RVEC, TVEC, K, D)[0].ravel()


def _board(a1_dark=True, pieces=True):
    """Grey image of the board; with pieces, the starting position (cream ranks 1-2, brown 7-8)."""
    p = cfg.CHESSBOARD_SQUARE_SIZE_M
    gray = np.full((800, 800), 120, np.uint8)
    for r in range(8):
        for f in range(8):
            x0, y0 = (f - 4) * p, (r - 4) * p
            quad = np.array([_px(x0, y0), _px(x0 + p, y0), _px(x0 + p, y0 + p), _px(x0, y0 + p)], np.int32)
            dark = ((f + r) % 2 == 0) == a1_dark
            cv2.fillConvexPoly(gray, quad, 60 if dark else 200)
            if pieces and r in (0, 1, 6, 7):
                u, v = _px((f - 3.5) * p, (r - 3.5) * p)
                cv2.circle(gray, (int(u), int(v)), 20, 230 if r < 2 else 40, -1)
    return gray


def test_a1_is_dark_with_a_rook_on_a1_and_h1():
    # Both corner squares are under cream rooks: a1 against h1 alone reads them, not the squares.
    assert board_tags._a1_dark(_board(), RVEC, TVEC, K, D)


def test_a_board_turned_a_quarter_turn_reads_a1_light():
    assert not board_tags._a1_dark(_board(a1_dark=False), RVEC, TVEC, K, D)
    assert not board_tags._a1_dark(_board(a1_dark=False, pieces=False), RVEC, TVEC, K, D)


def test_the_empty_board_reads_a1_dark():
    assert board_tags._a1_dark(_board(pieces=False), RVEC, TVEC, K, D)
