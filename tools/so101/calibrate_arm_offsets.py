"""
calibrate_arm_offsets.py — correct the arm's joint readings against a chessboard.

LeRobot's calibration sets each joint's zero from a pose set by eye, and the
URDF's kinematics then turn those angles into positions. A few degrees off at
the shoulder or elbow is invisible in the joints and plain in the positions:
touching the corners of a flat board, this arm read the near corners at -5 mm
and the far one at -28 mm, and a 40 mm square as 32 mm along one edge and
38 mm along the other. Everything that compares a camera's positions with
the arm's inherits that — the overhead calibration rejected itself over it.

This finds the corrections. You touch board corners with the closed jaw tips,
torque off; the arm's raw joint angles are recorded at each. A board is rigid,
flat and square, with known 40 mm squares, so every touch must land on one
plane in one grid: what is fitted is the joint offsets (shoulder_lift,
elbow_flex, wrist_flex — shoulder_pan just turns the whole board, and the roll
barely moves the tips) together with where the board is, and a scale. The
board is held LEVEL in the corrected frame: a tilt and a shoulder offset look
alike from touches on one plane, and level is what grasp heights need.

Prints JOINT_OFFSETS_DEG for robot_config.py, TABLE_Z from the board's height,
and — since the board's corners are then known in the corrected arm frame and
the overhead camera sees all 49 of them — OVERHEAD_HOMOGRAPHY, OVERHEAD_CAM_MODEL
and ARM_FRAME_SCALE, replacing calibrate_overhead_camera.py's three-touch fit.

Usage:
    python tools/so101/calibrate_arm_offsets.py --deployment chess
    python tools/so101/calibrate_arm_offsets.py --deployment chess --fit touches.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2            # noqa: E402
import numpy as np    # noqa: E402

from tools import calibrate_overhead_camera as co   # noqa: E402  (same board detection and target images)
from mrcf.backends.so101.kinematics import JOINT_NAMES  # noqa: E402
from tools.so101._common import make_env, rc  # noqa: E402
from mrcf import privacy  # noqa: E402

cfg = co.cfg
HERE = rc.RIG_DIR             # this rig's folder (rig_state.py)

# A 3 x 3 spread over the 7 x 7 inner corners: near and far, both sides and the
# middle, so the arm is measured folded and stretched, left and right.
TARGETS = [(0, 0), (0, 3), (0, 6), (3, 6), (3, 3), (3, 0), (6, 0), (6, 3), (6, 6)]

FITTED = ["shoulder_lift", "elbow_flex", "wrist_flex"]


class Model:
    """Tip position from raw joints + offsets, and board corner positions from a board pose."""

    def __init__(self, env, pitch_m: float):
        self.kin = getattr(env, "_kinematics_raw", env._kinematics)   # fitting the corrections: uncorrected
        self.tcp = env._tcp
        self.pitch = pitch_m
        self.tip = np.array([0.0, 0.0, float(getattr(rc, "JAW_TIP_BELOW_GRASP_M", 0.007))])

    def tips(self, touches, offsets: np.ndarray) -> np.ndarray:
        out = []
        for t in touches:
            q = np.array([t["joints"][n] for n in JOINT_NAMES] + [0.0])
            for k, name in enumerate(FITTED):
                q[JOINT_NAMES.index(name)] += offsets[k]
            T = self.kin.forward_kinematics(q)
            out.append(T[:3, 3] + T[:3, :3] @ (self.tcp(t["gap_m"]) + self.tip))
        return np.array(out)

    def corners(self, rows_cols, pose: np.ndarray, mirror: bool) -> np.ndarray:
        """Board corners for pose [x, y, z, yaw, scale]: LEVEL in the corrected arm frame."""
        o, yaw, k = pose[:3], pose[3], pose[4]
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        g = np.array([[cc, rr, 0.0] if not mirror else [rr, cc, 0.0] for rr, cc in rows_cols]) * self.pitch
        return o + k * (g @ R.T)


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Rotation + uniform scale + translation (no reflection) taking src to dst."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, s, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    D = np.diag([1.0, 1.0, d])
    R = u @ D @ vt
    k = float((s * np.diag(D)).sum() / (a ** 2).sum())
    return R, k, md - k * R @ ms


# How firmly the elbow and wrist corrections are held at zero: touches on one
# plane barely tell them apart from the board's pose and scale, so without
# this they wander by degrees on millimetres of hand error. 1 deg of either
# costs as much as 0.5 mm of touch residual. The shoulder is left free: it is
# what makes a level board read level.
_PRIOR_M_PER_DEG = 0.0005


def fit(model: Model, touches) -> dict:
    """
    Least squares over [shoulder, elbow, wrist offsets (deg), board x, y, z (m),
    yaw (rad), scale], the board held LEVEL in the corrected frame — so the
    corrections are what make it read level, as a flat board on a level table
    must, and grasp heights hold across it.
    """
    rows_cols = [tuple(t["grid"]) for t in touches]
    best = None
    for mirror in (False, True):
        tips0 = model.tips(touches, np.zeros(3))
        g = np.array([[c, r, 0.0] if not mirror else [r, c, 0.0] for r, c in rows_cols]) * model.pitch
        R, k, o = _umeyama(g, tips0)
        x = np.concatenate([np.zeros(3), o, [np.arctan2(R[1, 0], R[0, 0]), k]])

        def residual(x):
            fit_err = (model.tips(touches, x[:3]) - model.corners(rows_cols, x[3:], mirror)).ravel()
            return np.concatenate([fit_err, _PRIOR_M_PER_DEG * x[1:3]])

        steps = np.array([0.01] * 3 + [1e-5] * 3 + [1e-5, 1e-5])
        lam = 1e-3
        r0 = residual(x)
        for _ in range(200):
            J = np.column_stack([(residual(x + e) - residual(x - e)) / (2 * e[i])
                                 for i, e in enumerate(np.diag(steps))])
            A = J.T @ J
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -J.T @ r0)
            if np.abs((x + step)[:3]).max() > 20.0:     # a joint zero is not 20 deg out
                lam *= 10
                if lam > 1e8:
                    break
                continue
            r1 = residual(x + step)
            if r1 @ r1 < r0 @ r0:
                x, r0, lam = x + step, r1, lam * 0.3
                if np.abs(step).max() < 1e-9:
                    break
            else:
                lam *= 10
                if lam > 1e8:
                    break
        per = np.linalg.norm(r0[:-2].reshape(-1, 3), axis=1)
        rms = float(np.sqrt((per ** 2).mean()))
        sigma2 = float(r0 @ r0) / max(1, len(r0) - len(x))
        try:
            std = np.sqrt(np.clip(np.diag(np.linalg.inv(J.T @ J) * sigma2), 0, None))
        except np.linalg.LinAlgError:
            std = np.full(len(x), np.nan)
        if best is None or rms < best["rms"]:
            best = {"x": x, "rms": rms, "std": std, "mirror": mirror, "per_touch_mm": per * 1000}
    return best


def collect(env) -> list:
    print("Moving to CALIBRATION_PARK_DEGREES so the arm isn't occluding the board...")
    env.goto_joint_degrees(rc.CALIBRATION_PARK_DEGREES or rc.HOME_DEGREES)
    frame = env.render_image(cfg.OVERHEAD_CAMERA_NAME, width=co.CAPTURE_W, height=co.CAPTURE_H)
    corners = co._detect_chessboard(frame)
    if corners is None:
        raise SystemExit("Could not find the board's 49 corners in the overhead camera — is it clear?")
    print(f"Found {len(corners)} corners.\n")
    np.save(HERE / "calibrate_arm_offsets_corners.npy", corners)
    privacy.imwrite(str(HERE / "calibrate_arm_offsets_frame.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    target = HERE / "calibrate_overhead_target.png"
    cols = cfg.CHESSBOARD_INTERNAL_COLS

    print("Disabling torque — the arm is free to move by hand. For each corner: close the")
    print("jaws, put the point where the tips meet on the circled corner, touching the board.\n")
    env.disable_torque()
    touches = []
    try:
        for n, (r, c) in enumerate(TARGETS, start=1):
            co._save_single_target(frame, corners, r * cols + c, target)
            co._open_image(target)
            input(f"Touch {n}/{len(TARGETS)} — grid ({r},{c}), circled; press Enter> ")
            joints = {k: float(v) for k, v in env._motor_pos().items()}
            touches.append({"grid": [r, c], "joints": joints, "gap_m": float(env.get_gripper_width())})
            p = env.get_ee_pos()
            print(f"  ({r},{c}): arm reads ({p[0]:.3f}, {p[1]:.3f}, {p[2]:+.3f})\n", flush=True)
    finally:
        env.enable_torque()
    out = HERE / f"arm_offsets_touches_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"touches": touches}, indent=1))
    print(f"Saved the touches to {out} (re-fit with --fit).")
    return touches


def report(env, touches) -> None:
    model = Model(env, cfg.CHESSBOARD_SQUARE_SIZE_M)
    before = fit_without_offsets(model, touches)
    best = fit(model, touches)
    x, std = best["x"], best["std"]
    print("\n" + "=" * 64)
    print(f"Touch residual: {before:.1f} mm RMS with the readings as they are, "
          f"{best['rms'] * 1000:.1f} mm after correcting them.")
    print("Per touch (mm): " + ", ".join(f"{v:.1f}" for v in best["per_touch_mm"]))
    for k, name in enumerate(FITTED):
        print(f"  {name:13} {x[k]:+6.2f} deg  (+/- {std[k]:.2f})")
    k = x[7]
    print(f"  scale         {k:.4f}  (board squares read {k * cfg.CHESSBOARD_SQUARE_SIZE_M * 1000:.1f} mm "
          "in the corrected arm frame)")
    rows_cols = [(r, c) for r in range(cfg.CHESSBOARD_INTERNAL_ROWS) for c in range(cfg.CHESSBOARD_INTERNAL_COLS)]
    board = model.corners(rows_cols, x[3:8], best["mirror"])
    centre = board[len(board) // 2]
    print(f"  board centre  ({centre[0]:.3f}, {centre[1]:.3f}), surface at z {centre[2]:+.4f} "
          "(level, in the corrected frame)")
    worst = int(np.argmax(best["per_touch_mm"]))
    if best["per_touch_mm"][worst] > 6.0:
        print(f"  WARNING: touch {tuple(touches[worst]['grid'])} is {best['per_touch_mm'][worst]:.0f} mm off the "
              "fit — the wrong corner, or the tips not on the board? Re-touch it (--fit after editing).")
    print("=" * 64)
    print("\nPaste into mrcf/backends/so101/robot_config.py:\n")
    print("JOINT_OFFSETS_DEG = {")
    for i, name in enumerate(FITTED):
        print(f"    \"{name}\": {x[i]:.2f},")
    print("}")
    print(f"TABLE_Z = {centre[2]:.4f}")

    corners_px = np.load(HERE / "calibrate_arm_offsets_corners.npy")
    world_xy = board[:, :2]
    scale = np.array([cfg.RENDER_W / co.CAPTURE_W, cfg.RENDER_H / co.CAPTURE_H])
    H, _ = cv2.findHomography(corners_px * scale, world_xy, method=0)
    proj = cv2.perspectiveTransform((corners_px * scale).reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - world_xy, axis=1) * 1000
    print(f"\n# Overhead fit: 49 corners, mean {err.mean():.1f} mm, max {err.max():.1f} mm")
    print("OVERHEAD_HOMOGRAPHY = [")
    for row in H:
        print(f"    [{float(row[0])!r}, {float(row[1])!r}, {float(row[2])!r}],")
    print("]")
    square_m = k * cfg.CHESSBOARD_SQUARE_SIZE_M
    model_cam, rep = co.overhead_camera_model(corners_px, world_xy, (co.CAPTURE_W, co.CAPTURE_H),
                                              cfg.CHESSBOARD_SQUARE_SIZE_M, square_m)
    print(f"# camera: lens {rep['lens_rms_px']:.2f} px RMS, pose {rep['pose_mean_px']:.2f} px mean, "
          f"{rep['height_mm']:.0f} mm above the board")
    print("OVERHEAD_CAM_MODEL: \"dict | None\" = {")
    print(f"    \"size\": {model_cam['size']},")
    print(f"    \"K\": {model_cam['K']},")
    print(f"    \"D\": {model_cam['D']},")
    print("    \"T_world_cam\": [")
    for row in model_cam["T_world_cam"]:
        print(f"        {row},")
    print("    ],")
    print("}")
    print(f"ARM_FRAME_SCALE = {1.0 / k:.4f}")


def fit_without_offsets(model: Model, touches) -> float:
    """RMS (mm) of the best rigid board + scale with the joint readings uncorrected."""
    rows_cols = [tuple(t["grid"]) for t in touches]
    tips = model.tips(touches, np.zeros(3))
    best = np.inf
    for mirror in (False, True):
        g = np.array([[c, r, 0.0] if not mirror else [r, c, 0.0] for r, c in rows_cols]) * model.pitch
        R, k, o = _umeyama(g, tips)
        best = min(best, float(np.sqrt(((k * g @ R.T + o - tips) ** 2).sum(1).mean())) * 1000)
    return best


def main() -> None:
    env = make_env(overhead_camera_width=co.CAPTURE_W, overhead_camera_height=co.CAPTURE_H,
                   overhead_camera_fps=co.CAPTURE_FPS)
    if "--fit" in sys.argv:
        touches = json.loads(Path(sys.argv[sys.argv.index("--fit") + 1]).read_text())["touches"]
        report(env, touches)
        return
    print(f"Connecting to {rc.FOLLOWER_ID} on {rc.FOLLOWER_PORT}...")
    env.connect()
    try:
        touches = collect(env)
        env.go_home()
    finally:
        env.stop()
    report(env, touches)


if __name__ == "__main__":
    main()
