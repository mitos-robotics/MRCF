"""
capture_piece_frames.py — record labelled gripper-camera frames for testing piece_vision.

Each run records one piece (or an empty board) from the exact situation the
servo is in: open jaws at hover height, the piece somewhere below, off-centre by
up to the scan's typical error.

Labels are automatic, not clicked:

  1. you stand the piece centred between the open jaws at grasp height, by eye
  2. the arm reads the open gap's centre — the piece's position
  3. the arm visits hover poses at known offsets from it
  4. at each, the calibrated gripper-camera model (GRIPPER_CAM_GRASP_PX and
     GRIPPER_CAM_JACOBIAN) predicts where the piece must appear, and that
     pixel is the label

So a label is as good as the servo calibration, which is the right standard:
the test asks "would the servo have converged on this frame?".

For an empty-board run, the arm visits the same poses with nothing there; the
right answer at every pose is "no piece".

The hover, the grasp height and the jaw opening are the piece's own (the arm's
policy sizing config.PIECE_SIZES_M for the type in the label), so a king is
framed from where a king pick would look, not from a pawn's hover through its
crown. Put the colour in
the label too — black and white are scored separately.

Usage (from the repo root):
    python deployments/chess/tests/capture_piece_frames.py --label white_pawn
    python deployments/chess/tests/capture_piece_frames.py --label black_knight --at 0.20 0.05
    python deployments/chess/tests/capture_piece_frames.py --label black_king --at 0.24 0.04
    python deployments/chess/tests/capture_piece_frames.py --label empty

The runs behind its values and fixes, with their dates: deployments/chess/HISTORY.md.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np

from tools._common import make_env, rc, task_config
from mrcf import privacy

cfg = task_config("chess")
OUT = Path(__file__).parent / "piece_frames"

# Offsets (mm) of the hover pose from the piece. Out to 45 mm because a scan
# error of a full square (36 mm) has happened, and the servo's first look has
# to find the piece from there.
OFFSETS_MM = [(0, 0), (15, 0), (-15, 0), (0, 15), (0, -15),
              (30, 30), (30, -30), (-30, 30), (-30, -30),
              (45, 0), (-45, 0), (0, 45), (0, -45)]


def predicted_px(env, piece_xy) -> "list[float]":
    """Where the piece should appear in the gripper camera from the arm's current pose."""
    q6 = np.append(env._q_arm_deg(), 0.0)
    t = env._kinematics.forward_kinematics(q6)
    here = t[:3, 3] + t[:3, :3] @ env._tcp()
    scale = max(float(here[2]) - cfg.TABLE_Z, 1e-3) / rc.GRIPPER_CAM_REF_HEIGHT_M
    jac = np.asarray(rc.GRIPPER_CAM_JACOBIAN, float) * scale
    # The servo moves the arm by d = jac @ -(px - grasp_px) to centre the
    # piece, and the arm must move by (piece - here) to get there. Invert.
    d_gripper = (t[:3, :3].T @ np.array([piece_xy[0] - here[0], piece_xy[1] - here[1], 0.0]))[:2]
    px = np.asarray(rc.GRIPPER_CAM_GRASP_PX, float) - np.linalg.solve(jac, d_gripper)
    return [float(px[0]), float(px[1])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="colour_type, e.g. black_knight, or 'empty'")
    ap.add_argument("--at", nargs=2, type=float, default=(0.24, 0.0), metavar=("X", "Y"))
    args = ap.parse_args()

    run = OUT / f"{datetime.now():%Y%m%d_%H%M%S}_{args.label}"
    run.mkdir(parents=True)
    from tools._common import make_policies
    from mrcf.policies.base import Target
    size = cfg.PIECE_SIZES_M.get(cfg.piece_type(args.label))
    profile = make_policies().plan(
        Target(np.zeros(3), foot_m=size[1], height_m=size[0]) if size
        else Target(np.zeros(3), max_height_m=cfg.TALLEST_PIECE_M))
    labels = {}
    env = make_env()
    env.connect()
    try:
        env.go_home()
        env.set_gripper(profile.pregrasp_open_m, n_steps=15)
        x, y = args.at
        hover_z = cfg.TABLE_Z + profile.hover_height_m
        env.move_ee_to(np.array([x, y, hover_z]))

        if args.label == "empty":
            input("Make sure nothing is on the board near the arm, then press Enter...")
            piece = None
            centre = np.array([x, y])
        else:
            env.move_ee_to(np.array([x, y, cfg.TABLE_Z + profile.grasp_height_m]),
                           tol=0.008, max_step_deg=cfg.GRASP_STEP_DEG)
            input(f"Stand the {args.label} on the board EXACTLY centred between the open "
                  "jaw pads (by eye), take your hand away, then press Enter...")
            # The open gap's centre, read without moving the jaws: closing
            # them to centre the piece would lean on get_gripper_width, which
            # reads ~0.7 mm for a held 20 mm piece.
            centre = env.get_ee_pos()[:2].copy()
            piece = centre
            env.move_ee_to(np.array([centre[0], centre[1], hover_z]), tol=0.006, max_step_deg=3.0)
            print(f"Piece centre: [{centre[0]:.4f} {centre[1]:+.4f}]")

        for dx, dy in OFFSETS_MM:
            goal = np.array([centre[0] + dx / 1000, centre[1] + dy / 1000, hover_z])
            if not env.move_ee_to(goal, tol=0.006, max_step_deg=3.0):
                print(f"  {dx:+3d},{dy:+3d} mm: did not converge, skipped")
                continue
            time.sleep(0.5)
            for k in range(2):
                frame = env.render_image(rc.CAMERA_NAME, width=rc.RENDER_W, height=rc.RENDER_H)
                name = f"o{dx:+03d}_{dy:+03d}_{k}.png"
                privacy.imwrite(str(run / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                labels[name] = None if piece is None else predicted_px(env, piece)
            lab = labels[name]
            print(f"  {dx:+3d},{dy:+3d} mm: saved" + ("" if lab is None else
                  f", piece expected at ({lab[0]:.0f}, {lab[1]:.0f})"))
        (run / "labels.json").write_text(json.dumps(
            {"piece": args.label, "at": list(map(float, centre)), "hover_z": hover_z,
             "grasp_px": list(map(float, rc.GRIPPER_CAM_GRASP_PX)), "frames": labels}, indent=2))
        print(f"\nSaved {len(labels)} frames to {run}")
    finally:
        env.go_home()
        env.stop()


if __name__ == "__main__":
    main()
