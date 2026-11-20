"""
check_gripper.py — standalone gripper sanity check for the SO-101 deployment.

Cycles the gripper open/closed a few times and prints the measured width
after each move, so you can confirm the gripper motor, GRIPPER_MAX_OPEN_M
calibration, and get_gripper_width()/set_gripper() round-trip all work
without going through pick/place or any VLM/planning code. Does not move
the arm joints.

Usage:
    python tools/so101/check_gripper.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.so101._common import make_env, rc as cfg  # noqa: E402


def main() -> None:
    env = make_env()
    print(f"Connecting to {cfg.FOLLOWER_ID} on {cfg.FOLLOWER_PORT}...")
    env.connect()
    print(f"Connected. GRIPPER_MAX_OPEN_M = {cfg.GRIPPER_MAX_OPEN_M * 1000:.0f}mm. Cycling gripper...\n")

    try:
        for cycle in range(3):
            print(f"Cycle {cycle + 1}/3:")

            env.set_gripper(cfg.GRIPPER_MAX_OPEN_M)
            width = env.get_gripper_width()
            print(f"  opened  -> width={width * 1000:.1f}mm "
                  f"({'OK' if width > cfg.GRIPPER_MAX_OPEN_M * 0.7 else 'LOW — check calibration/linkage'})")

            env.set_gripper(0.0)
            width = env.get_gripper_width()
            print(f"  closed  -> width={width * 1000:.1f}mm "
                  f"({'OK' if width < cfg.MIN_HOLD_WIDTH_M * 3 else 'HIGH — gripper may be obstructed'})")

        env.set_gripper(cfg.GRIPPER_MAX_OPEN_M)
        print("\nLeft the gripper open.")
    finally:
        env.stop()


if __name__ == "__main__":
    main()
