"""
board_tags.py — the chess board's pose in one overhead frame, from its tags
and its lattice: chess's own perception of the board as the overhead camera's
calibration target (skills.BoardTarget), and what finds the lattice when a
piece hides corners (chessboard.lattice_from_tags). The arm's tag tour
(tools/tag_tour.py) measures the board with it too.

Board frame: origin at the board's centre, x toward the h-file, y toward rank
8, z up toward the camera (metres).
"""

import cv2
import numpy as np

from mrcf.perception import fiducials

from . import config as cfg


def lattice_board_points() -> np.ndarray:
    """The 49 inner corners in the board frame (metres): origin at the board's
    centre, x toward the h-file, y toward rank 8, z up toward the camera."""
    p = cfg.CHESSBOARD_SQUARE_SIZE_M
    n = cfg.CHESSBOARD_INTERNAL_COLS
    return np.array([[(c - (n - 1) / 2) * p, (r - (n - 1) / 2) * p, 0.0]
                     for r in range(cfg.CHESSBOARD_INTERNAL_ROWS) for c in range(n)])


def _is_x_corner(gray: np.ndarray, px: np.ndarray, axes_px: np.ndarray) -> bool:
    """A lattice corner: the squares diagonally across it alike, the two pairs unlike."""
    vals = []
    for sx, sy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        u, v = px + 0.3 * (sx * axes_px[0] + sy * axes_px[1])
        if not (0 <= int(v) < gray.shape[0] and 0 <= int(u) < gray.shape[1]):
            return False
        vals.append(float(gray[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2].mean()))
    same = max(abs(vals[0] - vals[1]), abs(vals[2] - vals[3]))
    across = abs((vals[0] + vals[1]) - (vals[2] + vals[3])) / 2
    return across > 25 and same < 0.5 * across


def measure_board(frame_rgb: np.ndarray, K, D) -> "dict | None":
    """
    The board's pose in one clean frame, tied to its squares: origin at the
    board's centre, x toward the h-file, y toward rank 8, z up (metres).

    Tags first, roughly: their centres against where they were stuck
    (BOARD_TAG_ROUGH_XY_M) give the pose to the few mm a hand placement is
    good to. Then the lattice: each of the 49 inner corners is predicted from
    that, snapped to the corner nearby, checked to be a corner, and a robust
    fit over those found drops any a piece or the arm hides (OpenCV's finder
    needs the whole pattern, so one piece on the board defeats it). The
    lattice's reprojection is the lens check; each tag's corners traced to the
    board's plane are where it really is.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    found = fiducials.detect_tags(frame_rgb)
    rough = [i for i in cfg.BOARD_TAG_ROUGH_XY_M if i in found]
    if len(rough) < 4:
        return None
    obj = np.array([[*cfg.BOARD_TAG_ROUGH_XY_M[i], 0.0] for i in rough])
    img = np.array([found[i].mean(0) for i in rough])
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_SQPNP)
    grid = lattice_board_points()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    snapped, keep = None, None
    for win_frac in (0.3, 0.15):                       # coarse, then again from the better pose
        pred = cv2.projectPoints(grid, rvec, tvec, K, D)[0].reshape(-1, 2)
        sq_px = float(np.median(np.linalg.norm(np.diff(pred.reshape(7, 7, 2), axis=1), axis=2)))
        win = max(3, int(sq_px * win_frac))
        snapped = cv2.cornerSubPix(gray, pred.astype(np.float32).reshape(-1, 1, 2), (win, win), (-1, -1),
                                   crit).reshape(-1, 2).astype(np.float64)
        # one square along each board axis, in pixels, at every corner (for the corner check)
        step = np.array([[cfg.CHESSBOARD_SQUARE_SIZE_M, 0, 0], [0, cfg.CHESSBOARD_SQUARE_SIZE_M, 0]])
        ax = np.stack([cv2.projectPoints(g + step, rvec, tvec, K, D)[0].reshape(2, 2) - pr
                       for g, pr in zip(grid, pred)])
        keep = np.array([np.linalg.norm(snapped[k] - pred[k]) < 0.35 * sq_px and _is_x_corner(gray, snapped[k], ax[k])
                         for k in range(len(grid))])
        # Fit, drop corners far off the fit (a piece's edge mistaken for one), fit
        # again. Not solvePnPRansac: on clean synthetic corners under this lens it
        # kept 10 of 49, whichever solver it was given.
        for _ in range(3):
            if keep.sum() < 12:
                return None
            rvec, tvec = cv2.solvePnPRefineLM(grid[keep], snapped[keep], K, D, rvec, tvec)
            res = np.linalg.norm(cv2.projectPoints(grid, rvec, tvec, K, D)[0].reshape(-1, 2) - snapped, axis=1)
            limit = max(1.5, 4 * float(np.median(res[keep])))
            if not (keep & (res > limit)).any():
                break
            keep &= res <= limit
    proj = cv2.projectPoints(grid[keep], rvec, tvec, K, D)[0].reshape(-1, 2)
    err = np.linalg.norm(proj - snapped[keep], axis=1)
    R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
    C = -R.T @ t
    tags = {}
    for i, q in found.items():
        und = cv2.undistortPoints(q.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        rays = (R.T @ np.c_[und, np.ones(4)].T).T
        tags[i] = C + rays * (-C[2] / rays[:, 2])[:, None]
    return {"T_cam_board": np.vstack([np.c_[R, t], [0, 0, 0, 1]]), "lattice_px": snapped[keep],
            "lattice_all_px": cv2.projectPoints(grid, rvec, tvec, K, D)[0].reshape(-1, 2),
            "lattice_used": int(keep.sum()), "lattice_err_px": (float(err.mean()), float(err.max())),
            "tags": tags, "a1_dark": _a1_dark(gray, rvec, tvec, K, D)}


def _a1_dark(gray: np.ndarray, rvec, tvec, K, D) -> bool:
    """
    Whether a1's squares are the dark ones (the board is set up for a player
    at the arm): every square of a1's colour against every square of the
    other, by their medians. Not a1 against h1: with the pieces set up, a
    rook stands on each, and the answer is the rooks'. Pieces on the back
    ranks cover as many squares of each colour, so the empty middle decides.
    """
    p = cfg.CHESSBOARD_SQUARE_SIZE_M
    centres = np.array([[(f - 3.5) * p, (r - 3.5) * p, 0.0] for r in range(8) for f in range(8)])
    px = cv2.projectPoints(centres, rvec, tvec, K, D)[0].reshape(-1, 2)
    a1_colour, other = [], []
    for (u, v), (f, r) in zip(px, ((f, r) for r in range(8) for f in range(8))):
        patch = gray[max(int(v) - 3, 0):int(v) + 4, max(int(u) - 3, 0):int(u) + 4]
        if patch.size:
            (a1_colour if (f + r) % 2 == 0 else other).append(float(np.median(patch)))
    return bool(a1_colour and other and np.median(a1_colour) < np.median(other))
