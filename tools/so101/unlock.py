"""
unlock.py — free a "locked" SO-101 arm and drive it back to HOME_DEGREES.

The arm holds torque after every script exits (see disable_torque_on_disconnect
in mrcf/backends/so101/follower.py) so it doesn't go limp and drop — but that means it
resists being moved by hand until torque is explicitly disabled. This script:

  1. Connects and disables torque — the arm goes slack, move it by hand if
     it's in an awkward/stalled pose.
  2. Waits for you to press Enter.
  3. Re-enables torque and drives to HOME_DEGREES under motor power.
  4. Disconnects holding home (torque stays enabled on exit).

Usage:
    python tools/so101/unlock.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.so101._common import make_env, rc as cfg  # noqa: E402


def main() -> None:
    env = make_env()
    print(f"Connecting to {cfg.FOLLOWER_ID} on {cfg.FOLLOWER_PORT}...")
    env.connect()

    print("Disabling torque — the arm is now free to move by hand.")
    print("Reposition it if it's stuck or in an awkward pose.\n")
    env.disable_torque()

    drive_home = True
    try:
        input("Press Enter when ready to re-enable torque and drive to HOME_DEGREES> ")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled — re-enabling torque to hold the current (repositioned) pose, "
              "without driving to home.")
        drive_home = False

    print("\nRe-enabling torque...")
    env.enable_torque()

    if drive_home:
        print("Driving to HOME_DEGREES...")
        reached = env.go_home()
        print(f"  -> {'reached' if reached else 'DID NOT REACH within tolerance/timeout'}")

    env.stop()
    print("Disconnected. Arm is holding position under torque.")


if __name__ == "__main__":
    main()
