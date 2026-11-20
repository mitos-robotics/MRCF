"""
piece_geometry.py — the board as a ruler, for measuring pieces.

The measuring itself is the framework's (plane_geometry): a calibrated view of
a flat surface turns an object's outline into its foot, height and widths. This
adds what a chessboard gives on top:

  fixed_camera_view   a view calibrated from ONE photo of the board's 49
                      corners — the overhead camera needs no separate
                      calibration step, and keeps up if it is nudged
  BoardView           a view in the board's own frame: origin at the first
                      internal corner OpenCV reports, x and y along its rows
                      and columns, mm — the grid chessboard.square_map uses, so
                      square (R, C) is centred at ((C - 0.5) * pitch, (R - 0.5) * pitch)
  grid_edge_fraction  telling a piece from a square: an outline whose edge runs
                      along grid lines is part of the board

A backend that knows where its cameras are (RobotEnv.camera_model) needs none
of this to measure; chess uses it where the board is in view, and to filter.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mrcf.perception.plane_geometry import (  # noqa: F401 — re-exported for chess callers
    MIN_HEIGHT_PX_PER_MM,
    ObjectMeasurement as PieceMeasurement,
    PlaneView,
    clean_mask,
    measure,
    touches_frame,
)

# An outline whose edge runs along grid lines this much of the time is part of
# the board (a square, or a strip of one), not something standing on it.
GRID_EDGE_FRACTION = 0.4
# ... unless it is small, only partly on the grid, and not the colour of the
# square it stands on: a piece near a square's edge or corner, its outline cut
# where its colour meets a square of its own shade. A cream pawn near a
# square's corner: 0.35 of a square, 0.47 on the grid, 13.6 from the square's
# colour (Lab, 0-255). Quarters of bare squares cut from the same frame: 0.25 of
# a square, 0.56-0.58 on the grid, 1.0-3.6 from their own colour.
GRID_EDGE_PIECE_MAX = 0.6
PIECE_MAX_SQUARE_FRACTION = 0.5
UNLIKE_SQUARE_LAB = 8.0
GRID_EDGE_BAND = 0.06      # of a square either side of a grid line


@dataclass
class BoardView(PlaneView):
    """A PlaneView in the board's own frame, which also knows the square size."""
    pitch_mm: float = 40.0

    def to_board(self, px: np.ndarray) -> np.ndarray:
        return self.to_plane(px)


def fixed_camera_view(corners_px: np.ndarray, image_wh, pitch_mm: float,
                      cols: int = 7) -> "tuple[BoardView, float]":
    """
    Calibrate a fixed camera from ONE photo of the board: (view, rms_px).

    The board's 49 internal corners are a known flat grid, and one view of a
    flat grid seen at a slant pins down the focal length and the lens's barrel
    distortion along with the pose — so the overhead camera needs no separate
    calibration step. Principal point at the image centre and square pixels
    are assumed; on this rig the fit reprojects to 0.3 px.
    """
    w, h = image_wh
    obj = np.array([[(i % cols) * pitch_mm, (i // cols) * pitch_mm, 0.0]
                    for i in range(len(corners_px))], np.float32)
    k0 = np.array([[0.7 * w, 0, w / 2], [0, 0.7 * w, h / 2], [0, 0, 1]], float)
    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
             | cv2.CALIB_USE_INTRINSIC_GUESS)
    rms, k, d, rv, tv = cv2.calibrateCamera([obj], [np.asarray(corners_px, np.float32)],
                                            (int(w), int(h)), k0, np.zeros(5), flags=flags)
    return BoardView(k, d.ravel(), cv2.Rodrigues(rv[0])[0], tv[0].ravel(), pitch_mm=pitch_mm), float(rms)


# ── Telling a piece from the board ─────────────────────────────────────────────

def grid_edge_fraction(mask: np.ndarray, view: BoardView) -> float:
    """How much of the outline's edge runs along the board's grid lines."""
    edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(edge)
    if len(xs) < 20:
        return 1.0
    g = view.to_plane(np.c_[xs, ys]) / view.pitch_mm
    g = g[np.isfinite(g).all(1)]
    if len(g) < 20:
        return 1.0
    f = g - np.floor(g)
    near = np.minimum.reduce([f[:, 0], 1 - f[:, 0], f[:, 1], 1 - f[:, 1]]) < GRID_EDGE_BAND
    return float(near.mean())


def _its_square(mask: np.ndarray, view: BoardView) -> "np.ndarray | None":
    """The square the outline's middle is on, as a pixel polygon (4, 2); None off the plane."""
    ys, xs = np.nonzero(mask)
    c = view.to_plane(np.array([[xs.mean(), ys.mean()]], float))[0] / view.pitch_mm
    if not np.isfinite(c).all():
        return None
    i, j = np.floor(c)
    cell = np.array([[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]]) * view.pitch_mm
    return view.project(np.array([view.lifted(p, 0.0) for p in cell]))


def unlike_its_square(lab: np.ndarray, mask: np.ndarray, view: BoardView) -> bool:
    """
    Is the outline's middle a different colour from the bare part of the square
    it stands on (Lab, 0-255 scale, > UNLIKE_SQUARE_LAB)? A patch of the square
    itself is not.
    """
    poly = _its_square(mask, view)
    if poly is None:
        return False
    cell = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(cell, [poly.astype(np.int32)], 1)
    cell = cv2.erode(cell, np.ones((5, 5), np.uint8)).astype(bool)          # off the grid lines' blur
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    bare = cell & ~grown
    if bare.sum() < 50:
        return False
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    core = dist > 0.5 * dist.max()
    core = core if core.sum() > 30 else mask.astype(bool)
    lab = np.asarray(lab, float)
    return float(np.linalg.norm(np.median(lab[core], 0) - np.median(lab[bare], 0))) > UNLIKE_SQUARE_LAB


def on_the_grid(mask: np.ndarray, view: BoardView, lab: np.ndarray, differs: bool = False) -> bool:
    """
    Is this outline part of the board rather than something on it? Along the
    grid more than GRID_EDGE_FRACTION of its edge, unless it is a piece cut by
    a grid line: small, at most GRID_EDGE_PIECE_MAX on the grid, and unlike
    its square's colour — or, `differs`, known by the caller to differ from
    the empty board (its middle changed from a photo of it): a washed-out
    cream pawn on a light square read 7 Lab units from the square, under the
    colour test's 8, while 79% of its middle had changed from the photo.
    """
    on = grid_edge_fraction(mask, view)
    if on <= GRID_EDGE_FRACTION:
        return False
    if on > GRID_EDGE_PIECE_MAX:
        return True
    poly = _its_square(mask, view)
    if poly is None or mask.sum() >= PIECE_MAX_SQUARE_FRACTION * abs(cv2.contourArea(poly.astype(np.float32))):
        return True
    return not (differs or unlike_its_square(lab, mask, view))
