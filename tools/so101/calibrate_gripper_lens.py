"""
calibrate_gripper_lens.py — measure the gripper camera's lens (intrinsics).

Produces GRIPPER_CAM_K and GRIPPER_CAM_DIST for robot_config.py: the camera
matrix and distortion coefficients at RENDER_W x RENDER_H. The gripper-camera
piece detector uses them to model the board plane across the whole frame.

Re-run after refocusing the lens: focal length moves with focus.

What it does
------------
Drives the camera over the chessboard through a grid of positions at two
heights, with wrist tilts and twists at some of them, and photographs the
board from each. Fully automatic — nothing to place, nothing to confirm.
Pieces may stay on the board.

Every view only sees part of the board, so corners come from
findChessboardCornersSB with CALIB_CB_LARGER, each view with its own local
grid. Some of those corners are wrong — the detector invents them on a piece
or the jaw standing in front of the lattice — and a straight calibrateCamera
over everything came out at 28 px RMS, i.e. nonsense. So it is robust in two
stages:

  1. calibrate on the views whose corners already fit a flat grid well
  2. with that, solvePnPRansac each view, keep only its inlier corners, and
     recalibrate on what survives; repeat until stable

First run on this rig: 34 views, 481 of 685 corners kept, 0.53 px RMS.

Usage:
    python tools/so101/calibrate_gripper_lens.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2            # noqa: E402
import numpy as np    # noqa: E402

from tools.so101._common import make_env, rc, task_config  # noqa: E402
from mrcf import privacy  # noqa: E402

cfg = task_config()
HERE = rc.RIG_DIR             # this rig's folder (rig_state.py)
FRAMES = HERE / "lens_frames"

_FLAGS = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE

# Positions over the near half of the board, at heights that keep the jaws
# clear of a standing piece. Wrist (flex, roll) offsets in degrees add the
# angular variety intrinsics need, at the outer positions only.
XS, YS, HEIGHTS = (0.19, 0.225, 0.26), (-0.05, 0.0, 0.05), (0.08, 0.12)
WIGGLES = [(8, 0), (-8, 0), (0, 15), (0, -15)]

CLEAN_FIT_PX = 10.0     # stage 1: views whose corners fit a flat grid this well
INLIER_PX = 3.0         # stage 2: reprojection error to keep a corner


def capture(env) -> list:
    FRAMES.mkdir(exist_ok=True)
    frames = []

    def grab(tag):
        time.sleep(0.8)
        env.render_image(rc.CAMERA_NAME)      # drop a possibly stale frame
        f = env.render_image(rc.CAMERA_NAME, width=rc.RENDER_W, height=rc.RENDER_H)
        privacy.imwrite(str(FRAMES / f"{len(frames):02d}_{tag}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        frames.append(f)
        print(f"  {len(frames):2d} {tag}")

    env.go_home()
    env.set_gripper(cfg.PREGRASP_OPEN_M, n_steps=10)
    for h in HEIGHTS:
        for x in XS:
            for y in YS:
                tag = f"x{x:.3f}_y{y:+.2f}_h{h:.2f}"
                if not env.move_ee_to(np.array([x, y, cfg.TABLE_Z + h]), tol=0.01):
                    print(f"     skip {tag} (did not converge)")
                    continue
                grab(tag)
                if y == 0.0 and x != XS[1]:
                    base = env.get_joint_degrees()
                    for dflex, droll in WIGGLES:
                        q = dict(base)
                        q["wrist_flex"] += dflex
                        q["wrist_roll"] += droll
                        if env.goto_joint_degrees(q, tol_deg=2.5, timeout_s=4.0):
                            grab(f"{tag}_f{dflex:+d}r{droll:+d}")
                    env.goto_joint_degrees(base, tol_deg=2.5, timeout_s=4.0)
    return frames


def views_from(frames: list) -> list:
    """(object_points, image_points, flat_grid_fit_px) per view with a board grid."""
    out = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        cv2.setRNGSeed(0)
        ok, corners, meta = cv2.findChessboardCornersSBWithMeta(gray, (3, 3), _FLAGS)
        if not ok:
            continue
        c = corners.reshape(-1, 2).astype(np.float32)
        cols = meta.shape[1]
        grid = np.array([[i % cols, i // cols] for i in range(len(c))], np.float32)
        h, _ = cv2.findHomography(grid, c, 0)
        fit = float(np.sqrt(((cv2.perspectiveTransform(grid[None], h)[0] - c) ** 2).sum(1).mean()))
        # Square size is irrelevant to intrinsics; units of squares will do.
        obj = np.hstack([grid, np.zeros((len(grid), 1), np.float32)])
        out.append((obj, c, fit))
    return out


def calibrate(views: list, size) -> "tuple[float, np.ndarray, np.ndarray, int]":
    clean = [v for v in views if v[2] < CLEAN_FIT_PX]
    if len(clean) < 5:
        raise SystemExit(f"Only {len(clean)} clean views — not enough to start from.")
    rms, k, d, _, _ = cv2.calibrateCamera([v[0] for v in clean], [v[1] for v in clean], size, None, None)
    print(f"\n  stage 1: {len(clean)} clean views, RMS {rms:.2f} px")
    kept = clean
    for it in range(3):
        kept = []
        for obj, img, _ in views:
            ok, _, _, inl = cv2.solvePnPRansac(obj, img, k, d, reprojectionError=INLIER_PX,
                                               iterationsCount=500)
            if ok and inl is not None and len(inl) >= 8:
                inl = inl.ravel()
                kept.append((obj[inl], img[inl]))
        rms, k, d, _, _ = cv2.calibrateCamera([v[0] for v in kept], [v[1] for v in kept], size, None, None)
        n = sum(len(v[0]) for v in kept)
        print(f"  stage 2, pass {it + 1}: {len(kept)} views, {n} corners kept, RMS {rms:.3f} px")
    return rms, k, d, sum(len(v[0]) for v in kept)


def main() -> None:
    env = make_env()
    print(f"Connecting to {rc.FOLLOWER_ID} on {rc.FOLLOWER_PORT}...")
    env.connect()
    try:
        print("Capturing views (keep clear of the arm)...")
        frames = capture(env)
    finally:
        env.go_home()
        env.stop()

    views = views_from(frames)
    print(f"\n{len(views)} of {len(frames)} views show a board grid.")
    rms, k, d, _ = calibrate(views, (rc.RENDER_W, rc.RENDER_H))
    if rms > 1.5:
        print(f"\nRMS {rms:.2f} px is too high to trust — check the saved frames in {FRAMES}.")
        return

    print("\n" + "=" * 70)
    print("Paste into mrcf/backends/so101/robot_config.py:\n")
    print("GRIPPER_CAM_K: \"list[list[float]] | None\" = [")
    for row in k:
        print(f"    [{row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f}],")
    print("]")
    print("GRIPPER_CAM_DIST: \"list[float] | None\" = ["
          + ", ".join(f"{v:.5f}" for v in d.ravel()) + "]")
    print("=" * 70)


if __name__ == "__main__":
    main()
