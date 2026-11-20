"""
teleop_keyboard.py — drive the arm by hand from the keyboard.

Why this exists
---------------
Every other tool here closes its loop on forward kinematics, so none of them can
answer "is this grasp physically possible at all?" — they can only tell you
whether the arm went where the model asked. This takes the model out of the loop
and puts you in it.

Two things it is for:

  1. Feasibility. Jog the jaws onto a piece yourself and close them. If you
     cannot pick it up by hand either, the problem is the gripper, the grasp
     height or the piece — not the perception or the calibration, and no amount
     of re-calibrating will fix it.

  2. Ground truth. Press 'k' when the jaws are correctly around a piece and the
     position is recorded. On exit it prints every mark, so you can compare what
     the arm ACTUALLY needed against what board_squares.json claimed — which is
     a direct measurement of the targeting error, more precise than eyeballing
     offsets against grid lines.

Controls (press '?' in the session for this list)
------------------------------------------------
  Cartesian mode (default)
    up / down  or  w / s     x  — away from / toward the base
    left/right or  a / d     y
    r / f                    z  — up / down
    [ / ]                    step size smaller / larger
  Joint mode ('m' toggles)
    1 2 3 4 5                select joint
    w / s                    move it by the step angle
  Gripper
    o                        open to PREGRASP_OPEN_M
    O                        open fully
    c                        close (gently, GRASP_CLOSE_STEP_PCT)
    , / .                    nudge open / closed
  Other
    k                        mark this position (prints at exit)
    p                        print full state
    h                        go home
    g                        drop to grasp height, keeping x/y
    v                        rise to hover height, keeping x/y
    ?                        help
    x  or  Ctrl-C            quit (homes first)

Usage:
    python tools/so101/teleop_keyboard.py
    python tools/so101/teleop_keyboard.py --square d4
    python tools/so101/teleop_keyboard.py --xy 0.24 -0.05
    python tools/so101/teleop_keyboard.py --step 0.002
"""

import json
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np   # noqa: E402

from mrcf.backends.so101.kinematics import JOINT_NAMES                        # noqa: E402
from tools.so101._common import make_env, rc, task_config  # noqa: E402
from tools._keys import check_terminal, single_keys  # noqa: E402  (keys as pressed, on any OS)

cfg = task_config()

STEP_CHOICES_M   = [0.001, 0.002, 0.005, 0.010, 0.020]
STEP_CHOICES_DEG = [0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_STEP_IDX = 2     # 5 mm / 2 deg

# Jog moves are small, so they want small per-cycle joint steps too — a 5 mm
# nudge commanded at the arm-wide 8 deg/cycle arrives as one snap.
JOG_MAX_STEP_DEG = 3.0
JOG_TOL_M        = 0.005


def load_square(name: str) -> "np.ndarray | None":
    path = Path(cfg.__file__).parent / "board_squares.json"
    if not path.is_file():
        print(f"  No board map at {path} — ignoring --square.")
        return None
    raw = json.loads(path.read_text())
    if name not in raw:
        print(f"  '{name}' not in the board map — ignoring.")
        return None
    return np.asarray(raw[name]["world"], float)


def clamp_to_workspace(p: np.ndarray) -> "tuple[np.ndarray, str]":
    """Keep a jog inside the measured workspace, and never below the table."""
    b = rc.WORKSPACE_BOUNDS
    out = p.copy()
    hits = []
    for i, (axis, lo, hi) in enumerate((("x", b.x_min, b.x_max),
                                        ("y", b.y_min, b.y_max),
                                        ("z", max(b.z_min, cfg.TABLE_Z), b.z_max))):
        if out[i] < lo:
            out[i] = lo; hits.append(f"{axis}min")
        elif out[i] > hi:
            out[i] = hi; hits.append(f"{axis}max")
    return out, ",".join(hits)


def show(env, target, step_i, mode, joint_i, note="") -> None:
    here = env.get_ee_pos()
    w = env.get_gripper_width()
    if mode == "cartesian":
        unit = f"step {STEP_CHOICES_M[step_i] * 1000:.0f}mm"
    else:
        unit = f"joint {joint_i + 1}:{JOINT_NAMES[joint_i]} step {STEP_CHOICES_DEG[step_i]:.1f}deg"
    print(f"  ee [{here[0]:+.4f} {here[1]:+.4f} {here[2]:+.4f}]  "
          f"jaws {w * 1000:4.1f}mm  {mode[:4]}  {unit}"
          + (f"   {note}" if note else ""))


def main() -> None:
    # Before anything connects: single-key input needs a real terminal, and
    # discovering that after opening the serial port would leak it.
    check_terminal()

    argv = sys.argv
    step_i = DEFAULT_STEP_IDX
    start_xy = None
    for i, arg in enumerate(argv):
        if arg == "--square" and i + 1 < len(argv):
            start_xy = load_square(argv[i + 1])
        elif arg == "--xy" and i + 2 < len(argv):
            start_xy = np.array([float(argv[i + 1]), float(argv[i + 2])])
        elif arg == "--step" and i + 1 < len(argv):
            want = float(argv[i + 1])
            step_i = int(np.argmin([abs(s - want) for s in STEP_CHOICES_M]))

    env = make_env()
    print(f"Connecting to {rc.FOLLOWER_ID} on {rc.FOLLOWER_PORT}...")
    env.connect()
    print("Connected.")

    marks: list[tuple[np.ndarray, float]] = []
    terminal = ExitStack()          # single-key mode, once it is on: undone first, whatever happens
    try:
        print()
        env.go_home()
        env.set_gripper(cfg.PREGRASP_OPEN_M, n_steps=10)

        if start_xy is not None:
            start = np.array([start_xy[0], start_xy[1], cfg.TABLE_Z + cfg.HOVER_HEIGHT])
            print(f"Moving to {np.round(start, 4).tolist()}...")
            if not env.move_ee_to(start, tol=JOG_TOL_M):
                print("  Could not get there — staying where the arm stopped.")
        target = env.get_ee_pos()
        mode, joint_i = "cartesian", 0

        print(__doc__.split("Controls")[1].split("Usage:")[0].rstrip())
        print("\nReady. Keys act immediately — no Enter.\n")
        show(env, target, step_i, mode, joint_i)

        # cbreak, not raw: output post-processing and Ctrl-C both keep working,
        # so a hung move is still interruptible and printing stays readable.
        keys = terminal.enter_context(single_keys(arrows=True))
        while True:
            key = keys.wait()
            note = ""

            if key in ("x", "X", "\x03", "\x04"):
                break

            elif key == "?":
                print(__doc__.split("Controls")[1].split("Usage:")[0].rstrip())
                continue

            elif key == "m":
                mode = "joint" if mode == "cartesian" else "cartesian"
                if mode == "cartesian":
                    target = env.get_ee_pos()   # resync: joints moved under us
                note = f"-> {mode} mode"

            elif key == "p":
                deg = env._q_arm_deg()
                print("  joints (deg): " + "  ".join(
                    f"{n}={v:+.1f}" for n, v in zip(JOINT_NAMES, deg)))
                print(f"  target  [{target[0]:+.4f} {target[1]:+.4f} {target[2]:+.4f}]")
                print(f"  table_z {cfg.TABLE_Z:.4f}   grasp_z "
                      f"{cfg.TABLE_Z + cfg.GRASP_HEIGHT_M:.4f}   hover_z "
                      f"{cfg.TABLE_Z + cfg.HOVER_HEIGHT:.4f}")
                continue

            elif key == "k":
                here = env.get_ee_pos()
                marks.append((here.copy(), env.get_gripper_width()))
                note = f"marked #{len(marks)}"

            elif key == "h":
                env.go_home()
                target = env.get_ee_pos()
                note = "home"

            elif key in ("[", "]"):
                step_i = int(np.clip(step_i + (1 if key == "]" else -1),
                                     0, len(STEP_CHOICES_M) - 1))

            # ── gripper ──────────────────────────────────────────────────────
            elif key == "o":
                env.set_gripper(cfg.PREGRASP_OPEN_M, n_steps=10)
                note = "open (pregrasp)"
            elif key == "O":
                env.set_gripper(cfg.GRIPPER_MAX_OPEN_M, n_steps=10)
                note = "open (full)"
            elif key == "c":
                env.set_gripper(0.0, n_steps=10,
                                max_step_pct=getattr(cfg, "GRASP_CLOSE_STEP_PCT", None))
                note = "closed"
            elif key in (",", "."):
                w = env.get_gripper_width() + (0.003 if key == "," else -0.003)
                env.set_gripper(float(np.clip(w, 0.0, cfg.GRIPPER_MAX_OPEN_M)), n_steps=4,
                                max_step_pct=getattr(cfg, "GRASP_CLOSE_STEP_PCT", None))

            # ── height presets ───────────────────────────────────────────────
            elif key in ("g", "v") and mode == "cartesian":
                z = cfg.TABLE_Z + (cfg.GRASP_HEIGHT_M if key == "g" else cfg.HOVER_HEIGHT)
                cand = np.array([target[0], target[1], z])
                if env.move_ee_to(cand, tol=JOG_TOL_M, max_step_deg=JOG_MAX_STEP_DEG):
                    target = cand
                    note = "grasp height" if key == "g" else "hover height"
                else:
                    target = env.get_ee_pos()
                    note = "could not reach that height"

            # ── motion ───────────────────────────────────────────────────────
            elif mode == "cartesian":
                d = STEP_CHOICES_M[step_i]
                delta = {"UP": (0, +d), "w": (0, +d), "DOWN": (0, -d), "s": (0, -d),
                         "LEFT": (1, +d), "a": (1, +d), "RIGHT": (1, -d), "d": (1, -d),
                         "r": (2, +d), "f": (2, -d)}.get(key)
                if delta is None:
                    continue
                axis, amount = delta
                cand = target.copy()
                cand[axis] += amount
                cand, hit = clamp_to_workspace(cand)
                if hit:
                    note = f"clamped at {hit}"
                if env.move_ee_to(cand, tol=JOG_TOL_M, max_step_deg=JOG_MAX_STEP_DEG):
                    target = cand
                else:
                    # Resync so a refused jog does not leave the target drifting
                    # somewhere the arm never went — the next jog would then be
                    # measured from a fiction.
                    target = env.get_ee_pos()
                    note = (note + "  " if note else "") + "jog refused (IK/limit)"

            else:  # joint mode
                if key in "12345":
                    joint_i = int(key) - 1
                    note = f"-> {JOINT_NAMES[joint_i]}"
                elif key in ("w", "s", "UP", "DOWN"):
                    sign = 1.0 if key in ("w", "UP") else -1.0
                    deg = env._q_arm_deg()
                    goal = {n: float(v) for n, v in zip(JOINT_NAMES, deg)}
                    goal[JOINT_NAMES[joint_i]] += sign * STEP_CHOICES_DEG[step_i]
                    if not env.goto_joint_degrees(goal, tol_deg=1.0, timeout_s=2.0):
                        note = "joint did not reach (limit?)"
                else:
                    continue

            show(env, target, step_i, mode, joint_i, note)

    except KeyboardInterrupt:
        pass
    finally:
        # Restore the terminal FIRST: if homing raises, a half-reset terminal
        # leaves the shell unusable, which is a worse failure than a stuck arm.
        terminal.close()
        print("\nHoming...")
        try:
            env.go_home()
        finally:
            env.stop()
        print("Disconnected.")

    if marks:
        print("\n" + "=" * 68)
        print("Marked positions")
        print("=" * 68)
        for i, (p, w) in enumerate(marks, 1):
            print(f"  {i}.  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}   "
                  f"jaws {w * 1000:.1f} mm")
        print("\nCompare a mark against what the scan claimed for that square:")
        print("  grep -A4 '\"<square>\"' deployments/chess/board_squares.json")
        print("The difference is the targeting error at that square, measured")
        print("rather than eyeballed — which is what check_targeting.py fits.")


if __name__ == "__main__":
    main()
