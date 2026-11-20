"""
check_motion.py — standalone motion sanity check for the SO-101 deployment.

Drives go_home() and go_observe() directly and reports whether each pose was
actually reached (within tolerance) — no VLM, no planner, no perception.
Useful for telling "the arm can't physically reach this pose" apart from
"the VLM/planner never asked it to", which look identical from the REPL.

Refuses to attempt go_observe() if OBSERVE_DEGREES in config.py is still the
unmeasured all-zero placeholder, since that's a joint pose nobody has
confirmed is safe yet — run find_poses.py first.

Usage:
    python tools/so101/check_motion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.so101._common import make_env, rc as cfg  # noqa: E402


def _print_degrees(label: str, degrees: dict[str, float], target: "dict[str, float] | None" = None) -> None:
    print(f"  {label}:")
    for name, value in degrees.items():
        if target is not None:
            print(f"    {name:14s} {value:7.1f}deg  (target {target[name]:.1f}, "
                  f"off by {abs(value - target[name]):.1f})")
        else:
            print(f"    {name:14s} {value:7.1f}deg")


def main() -> None:
    env = make_env()
    print(f"Connecting to {cfg.FOLLOWER_ID} on {cfg.FOLLOWER_PORT}...")
    env.connect()
    print("Connected.\n")

    try:
        _print_degrees("Current pose on connect", env.get_joint_degrees())

        print("\nMoving to HOME_DEGREES...")
        reached = env.go_home()
        print(f"  -> {'reached' if reached else 'DID NOT REACH within tolerance/timeout'}")
        _print_degrees("Pose after go_home()", env.get_joint_degrees(), cfg.HOME_DEGREES)

        if all(abs(v) < 1e-9 for v in cfg.OBSERVE_DEGREES.values()):
            print("\nOBSERVE_DEGREES in config.py is still the all-zero placeholder — skipping "
                  "go_observe(). Run find_poses.py, move the arm so the camera looks at the "
                  "table, label it 'observe', and paste the result into config.py first.")
        else:
            print("\nMoving to OBSERVE_DEGREES...")
            reached = env.go_observe()
            print(f"  -> {'reached' if reached else 'DID NOT REACH within tolerance/timeout'}")
            _print_degrees("Pose after go_observe()", env.get_joint_degrees(), cfg.OBSERVE_DEGREES)

        print("\nReturning home...")
        env.go_home()
    finally:
        env.stop()


if __name__ == "__main__":
    main()
