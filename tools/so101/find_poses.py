"""
find_poses.py — interactive helper to measure the physical constants in config.py.

Connects to the follower arm and disables torque so you can move it BY HAND.
For each pose you want to capture, type a label and press Enter; it prints
the joint degrees and FK end-effector position, and remembers it under that
label. At the end it prints a summary next to the config.py constant it
belongs in — no more guessing which reading was which afterwards.

Four labels are special (home / table / observe / calib) and map to the
matching config.py constants. Any OTHER label (e.g. "left", "right") is
treated as an extra look-around vantage point and prints a ready-to-paste
LOOK_AROUND_POSES entry instead — capture "table" at some point too, since
that's needed to compute each look-around pose's height_m.

Does not touch the gripper or send any motion command — it only reads.

Usage:
    python tools/so101/find_poses.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.so101._common import make_env, rc as cfg  # noqa: E402
from mrcf import privacy  # noqa: E402


def main() -> None:
    # The overhead camera is included so each labelled capture can save what
    # that camera sees — the only practical way to confirm a 'calib' pose
    # really does keep the arm out of frame, which is hard to judge by eye.
    env = make_env()

    print(f"Connecting to {cfg.FOLLOWER_ID} on {cfg.FOLLOWER_PORT}...")
    env.connect()
    print("Connected. Disabling torque — the arm is now free to move by hand.\n")
    env.disable_torque()

    print("You need at least three poses — move the arm by hand, label, and press Enter:")
    print("  home    — safe pose raised CLEARLY (15-20+ cm) above the table, pulled back")
    print("            from anything it could hit when it returns here between commands.")
    print("  table   — gripper fingertips just touching the table. Gives TABLE_Z.")
    print("  observe — gripper-mounted camera looking down at the whole work area. Gives")
    print("            OBSERVE_DEGREES; combined with 'table', also gives OBSERVE_HEIGHT_M.")
    print("  calib   — (optional) arm fully clear of the overhead camera's view of the board,")
    print("            for calibrate_overhead_camera.py. Falls back to 'home' if not captured.")
    print("Any OTHER label (e.g. left, right) captures an extra LOOK_AROUND_POSES vantage")
    print("point instead — point the camera at a different part of the table first.")
    print("Ctrl-C when done.\n")

    captured: dict[str, tuple[dict, "object"]] = {}

    try:
        while True:
            label = input("Label for this pose (home/table/observe/calib/anything else), or blank to just peek> ").strip().lower()
            degrees = env.get_joint_degrees()
            print("  Joint degrees:", {k: round(v, 1) for k, v in degrees.items()})
            ee = None
            try:
                ee = env.get_ee_pos()
                print(f"  FK end-effector (m, base frame): [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
            except NotImplementedError as exc:
                print(f"  (FK unavailable: {exc})")
            if label:
                captured[label] = (degrees, ee)
                if cfg.OVERHEAD_CAMERA_NAME is not None:
                    try:
                        frame = env.render_image(cfg.OVERHEAD_CAMERA_NAME,
                                                 width=cfg.RENDER_W, height=cfg.RENDER_H)
                        shot = cfg.RIG_DIR / f"find_poses_{label}.png"
                        privacy.save_rgb(shot, frame)
                        print(f"  Overhead view at this pose saved to {shot}")
                        if label == "calib":
                            print("  ^ open it and confirm the ARM IS NOT IN FRAME over the board.")
                    except Exception as exc:
                        print(f"  (overhead snapshot failed: {exc})")
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nDone.")
    finally:
        env.enable_torque()
        env.stop()

    if not captured:
        return

    table_ee = captured["table"][1] if "table" in captured else None

    print("\n" + "=" * 60)
    print("Captured poses — paste into config.py:")
    print("=" * 60)
    if "home" in captured:
        degrees, _ = captured["home"]
        print("\nHOME_DEGREES = {")
        for k, v in degrees.items():
            print(f'    "{k}": {round(v, 1)},')
        print("}")
    if "observe" in captured:
        degrees, _ = captured["observe"]
        print("\nOBSERVE_DEGREES = {")
        for k, v in degrees.items():
            print(f'    "{k}": {round(v, 1)},')
        print("}")
    if "calib" in captured:
        degrees, _ = captured["calib"]
        print("\nCALIBRATION_PARK_DEGREES = {")
        for k, v in degrees.items():
            print(f'    "{k}": {round(v, 1)},')
        print("}")
    if "table" in captured:
        if table_ee is not None:
            print(f"\nTABLE_Z = {table_ee[2]:.3f}")
        else:
            print("\nTABLE_Z = <FK was unavailable for the 'table' capture — rerun with placo/URDF set up>")
    if "observe" in captured and table_ee is not None:
        _, ee_obs = captured["observe"]
        if ee_obs is not None:
            print(f"OBSERVE_CENTER_XY = ({ee_obs[0]:.3f}, {ee_obs[1]:.3f})")
            print(f"OBSERVE_HEIGHT_M  = {ee_obs[2] - table_ee[2]:.3f}")

    extra_labels = [l for l in captured if l not in ("home", "table", "observe", "calib")]
    if extra_labels:
        print("\n# LOOK_AROUND_POSES entries:")
        print("LOOK_AROUND_POSES = [")
        for label in extra_labels:
            degrees, ee = captured[label]
            degrees_repr = ", ".join(f'"{k}": {round(v, 1)}' for k, v in degrees.items())
            if ee is not None and table_ee is not None:
                center_xy = f"({ee[0]:.3f}, {ee[1]:.3f})"
                height_m = f"{ee[2] - table_ee[2]:.3f}"
                print(f'    {{"degrees": {{{degrees_repr}}}, "center_xy": {center_xy}, "height_m": {height_m}}},  # {label}')
            else:
                missing = "FK unavailable" if ee is None else "capture a 'table' pose too for center_xy/height_m"
                print(f'    {{"degrees": {{{degrees_repr}}}, "center_xy": ..., "height_m": ...}},  # {label} — {missing}')
        print("]")


if __name__ == "__main__":
    main()
