"""
check_hold.py — can the arm actually REACH and HOLD a near-board hover pose?

Pure joint-space. No IK loop, no VLM, no planner. It commands the exact joint
pose move_ee_to converges to when hovering over a near-rank square, then
reports what each joint actually reached. That separates two things the REPL
cannot tell apart from a bare "IK stalled" message:

  - the solver asking for something unreachable  (an IK/calibration problem)
  - the servo being unable to drive there under gravity  (a torque problem)

Why this is the suspect: offline (tools/so101/visualise_ik.py, no
gravity modelled) both e4 (0.227, 0.024) and c4 (0.233, -0.048) converge
cleanly to the SAME shape of pose — shoulder_lift near 0 deg, elbow_flex in
the single digits, i.e. the arm stretched almost horizontally at ~23 cm
reach. On real hardware both of those squares produced an "IK stalled ~1 cm
short" failure. A calibration error moves WHERE the target is, not whether
the arm can complete a given reach — so two different targets stalling by
the same ~1 cm, in the same posture, argues for a torque limit at this
reach/height combination rather than a bad homography.

It walks shoulder_lift in from a folded pose so the angle at which tracking
starts to fail is visible, rather than just pass/fail at the endpoint.

Usage:
    python tools/so101/check_hold.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.so101._common import make_env, rc as cfg  # noqa: E402

# Representative converged hover pose (see module docstring) — both e4 and c4
# land within a couple of degrees of this, so one test covers the class of
# reach rather than one specific square.
HOVER = {
    "shoulder_pan": -7.4,
    "shoulder_lift": -2.1,
    "elbow_flex": 9.0,
    "wrist_flex": 83.1,
    "wrist_roll": -7.4,
}

# Walk shoulder_lift in from folded to extended, holding the rest at the
# hover solution, so the angle where the servo starts losing the fight with
# gravity is visible rather than just the endpoint.
LIFT_STEPS = [-60.0, -45.0, -30.0, -20.0, -12.0, -6.0, -2.1]


def report(env, target: dict) -> float:
    pos = env.get_joint_degrees()
    worst = 0.0
    for name, want in target.items():
        got = pos[name]
        off = abs(got - want)
        worst = max(worst, off)
        flag = "  <-- NOT REACHED" if off > 2.0 else ""
        print(f"      {name:14s} want {want:7.1f}  got {got:7.1f}  off {off:5.1f}{flag}")
    return worst


def main() -> None:
    env = make_env()
    print(f"Connecting to {cfg.FOLLOWER_ID} on {cfg.FOLLOWER_PORT}...")
    env.connect()
    print("Connected.\n")
    try:
        print("Homing...")
        env.go_home()
        time.sleep(0.5)

        print("\nWalking shoulder_lift toward the hover pose.")
        print("Everything else held at the hover solution.\n")
        for lift in LIFT_STEPS:
            target = dict(HOVER, shoulder_lift=lift)
            reached = env.goto_joint_degrees(target, tol_deg=2.0, timeout_s=8.0)
            print(f"  shoulder_lift -> {lift:6.1f} deg   "
                  f"{'reached' if reached else 'TIMED OUT'}")
            worst = report(env, target)
            # Hold it a moment and re-read: a servo that arrives and then sags
            # under load is a different failure from one that never gets there.
            time.sleep(1.0)
            after = env.get_joint_degrees()["shoulder_lift"]
            sag = abs(after - lift)
            print(f"      after 1 s holding: shoulder_lift {after:7.1f} "
                  f"({'sagged ' + format(sag, '.1f') + ' deg' if sag > 2.0 else 'holding'})")
            print(f"      ee now at {env.get_ee_pos().round(3).tolist()}\n")
            if worst > 5.0:
                print("  Stopping: the arm is more than 5 deg off what was asked.\n")
                break

        print("Returning home.")
        env.go_home()
    finally:
        env.stop()
        print("Disconnected.")


if __name__ == "__main__":
    main()
