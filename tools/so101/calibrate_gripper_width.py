"""
calibrate_gripper_width.py — measure how wide the jaws really open.

The gripper motor reports 0-100%, and everything that sizes a grasp works in
millimetres: the opening an approach needs, whether a close caught anything,
where the centre of the gap is (only one jaw moves). Without a measurement the
backend maps 0-100% straight onto 0-GRIPPER_MAX_OPEN_M, which for this hinged
jaw is badly wrong: a ~20 mm piece reads 2-6 mm, so every "30 mm" approach is
in fact the jaws nearly wide open.

This steps the jaws through a few motor positions with the arm holding the
gripper up in the air, and you measure the gap between the pads — at the
middle of the pads, where an object is gripped — with a ruler or calipers.
It prints GRIPPER_WIDTH_TABLE for robot_config.py.

Usage:
    python tools/so101/calibrate_gripper_width.py            # type each gap
    python tools/so101/calibrate_gripper_width.py --hold 30  # timed: hold each
                                                                      # 30 s, note the gaps
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from tools.so101._common import make_env, rc  # noqa: E402

# Dense where grasps happen: a hinged jaw opens fastest near closed, and pieces
# are gripped in the first fraction of its travel.
PCTS = [5.0, 10.0, 15.0, 25.0, 40.0, 70.0]

# In the air over the middle of the board, clear of anything on it, where the
# jaws are easy to reach from the side.
PRESENT_AT = (0.22, 0.0, 0.12)


def main() -> None:
    hold = float(sys.argv[sys.argv.index("--hold") + 1]) if "--hold" in sys.argv else None
    env = make_env()
    print(f"Connecting to {rc.FOLLOWER_ID} on {rc.FOLLOWER_PORT}...")
    env.connect()
    table = [(0.0, 0.0)]
    try:
        env.go_home()
        if not env.move_ee_to(np.array(PRESENT_AT), tol=0.01):
            print("  Could not reach the measuring pose; measuring wherever the arm stopped.")
        for i, pct in enumerate(PCTS, start=1):
            env.drive_gripper_pct(pct, n_steps=10)
            at = float(env._motor_pos()["gripper"])
            if hold is not None:
                print(f"  {i}/{len(PCTS)}: motor {at:.1f}% — measure the gap now ({hold:.0f} s)", flush=True)
                time.sleep(hold)
                continue
            reply = input(f"  {i}/{len(PCTS)}: motor {at:.1f}% — gap between the pads in mm "
                          "(blank to skip): ").strip()
            if reply:
                table.append((round(at, 1), float(reply) / 1000.0))
        if hold is None and len(table) > 2:
            print("\nPaste into mrcf/backends/so101/robot_config.py:\n")
            print("GRIPPER_WIDTH_TABLE = [")
            for pct, width in table:
                print(f"    ({pct:5.1f}, {width:.4f}),")
            print("]")
    finally:
        env.drive_gripper_pct(10.0, n_steps=5)
        env.go_home()
        env.stop()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
