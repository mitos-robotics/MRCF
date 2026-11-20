"""chessboard's corner grid: one found a square off is not taken for a board turned round."""

import numpy as np

try:
    import cv2  # noqa: F401 — chessboard needs it
except ImportError:
    import pytest
    pytest.skip("needs OpenCV", allow_module_level=True)

from deployments.chess import chessboard

# An overhead view, straight down: 60 px squares, the arm at the bottom of the
# frame, its left (+y, the a-file) on the image's left. Pixels at RENDER_W x
# RENDER_H map to the arm's frame by this homography.
PITCH, X0, Y0 = 60, 400, 100            # board's far-left corner, CAPTURE pixels


def _overhead():
    img = np.full((chessboard.CAPTURE_H, chessboard.CAPTURE_W, 3), 90, np.uint8)
    cv2.rectangle(img, (X0 - 30, Y0 - 30), (X0 + 8 * PITCH + 30, Y0 + 8 * PITCH + 30), (60, 35, 25), -1)
    for i in range(8):                  # image rows, far (rank 8) to near (rank 1)
        for j in range(8):              # image columns, a to h
            dark = (i + j) % 2 == 1      # so a1, bottom left, is dark
            colour = (90, 50, 35) if dark else (235, 225, 200)
            cv2.rectangle(img, (X0 + j * PITCH, Y0 + i * PITCH),
                          (X0 + (j + 1) * PITCH - 1, Y0 + (i + 1) * PITCH - 1), colour, -1)
    return img


def _corners(first_row):
    """The 7 x 7 grid starting at lattice row first_row (1: the real internal corners)."""
    return np.array([[X0 + j * PITCH, Y0 + i * PITCH]
                     for i in range(first_row, first_row + 7) for j in range(1, 8)], float)


def _homography():
    sx, sy = chessboard.CAPTURE_W / 640, chessboard.CAPTURE_H / 480   # render -> capture
    m = 0.040 / PITCH                                                 # metres per capture px
    # x = (near edge - v) * m: away from the arm; y = (centre - u) * m: to its left.
    return np.array([[0.0, -m * sy, (Y0 + 8 * PITCH) * m + 0.10],
                     [-m * sx, 0.0, (X0 + 4 * PITCH) * m],
                     [0.0, 0.0, 1.0]])


def test_the_real_grid_fits_and_one_a_row_off_does_not(monkeypatch):
    # The arm's setting, which the chess config serves live.
    from mrcf.backends.so101 import robot_config
    monkeypatch.setattr(robot_config, "OVERHEAD_HOMOGRAPHY", _homography())
    frame = _overhead()
    assert chessboard._lattice_fits(frame, _corners(1))
    squares, pitch, _ = chessboard.square_map(_corners(1), _homography(), (chessboard.CAPTURE_W, chessboard.CAPTURE_H),
                                              (640, 480))
    assert abs(pitch - 0.040) < 1e-6
    assert squares["a1"]["pixel"][0] < squares["h1"]["pixel"][0]            # a-file on the arm's left
    assert squares["a1"]["pixel"][1] > squares["a8"]["pixel"][1]            # rank 1 nearest the arm
    # Toward the arm by a row: the near row's "squares" sit on the border, every
    # name is a rank out, and a1 comes out light.
    assert not chessboard._lattice_fits(frame, _corners(2))


def test_the_lattice_comes_from_the_tags_first_and_a_scrambled_one_is_never_used_or_kept(monkeypatch, tmp_path):
    # 2026-11-04: with the e-pawn moved, the corner finder found all 49 corners
    # but in the wrong order — 41.5 mm squares, 10.2 mm off a regular grid — and
    # that lattice, cached, named every square of the scans after it.
    from mrcf.backends.so101 import robot_config
    monkeypatch.setattr(robot_config, "OVERHEAD_HOMOGRAPHY", _homography())
    monkeypatch.setattr(chessboard, "_P", lambda *_: tmp_path / "chessboard.py")
    frame = _overhead()
    good = _corners(1)
    scrambled = good.copy()
    scrambled[[3, 30]] = scrambled[[30, 3]]
    scrambled[[10, 45]] = scrambled[[45, 10]]
    assert not chessboard._lattice_ok(frame, scrambled)

    monkeypatch.setattr(chessboard, "lattice_from_tags", lambda f: good)
    monkeypatch.setattr(chessboard, "detect_corners", lambda f: scrambled)
    assert np.array_equal(chessboard.board_lattice(frame), good)
    assert np.array_equal(np.load(tmp_path / chessboard._CORNER_CACHE), good)

    # No tags in view: the corner finder's scrambled lattice is refused, the last good one used.
    monkeypatch.setattr(chessboard, "lattice_from_tags", lambda f: None)
    assert np.array_equal(chessboard.board_lattice(frame), good)
    assert np.array_equal(np.load(tmp_path / chessboard._CORNER_CACHE), good)

    # And a scrambled lattice in the cache is not used either.
    np.save(tmp_path / chessboard._CORNER_CACHE, scrambled)
    assert chessboard.board_lattice(frame) is None
