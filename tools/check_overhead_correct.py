"""
check_overhead_correct.py — does the overhead camera put the gripper over a square? MOVES THE ARM.

For each square: hover over it as the kinematics would, see how far off the
overhead camera says the gripper is, let the pick's correction
(ParallelJaw._overhead_correct) move it until the camera sees it over the
square, then look again, independently, to see where it really ended up.
Hovers only — nothing is picked or placed.

The squares are where the last board scan put them (board_squares.json), so
the target comes through the same camera as the gripper's position, as a
pick's does.

Usage:
    python tools/check_overhead_correct.py
    python tools/check_overhead_correct.py --squares c3 f5 e7 --height 0.07
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tools._common import FRAMEWORK_ROOT, connecting_to, make_env, make_policies, rc, task_config  # noqa: E402

cfg = task_config()
SQUARES = ["c3", "f3", "d4", "g4", "b5", "e5", "c6", "f6", "d7", "g7"]


def main() -> None:
    argv = sys.argv
    names = argv[argv.index("--squares") + 1:] if "--squares" in argv else SQUARES
    names = [n for n in names if not n.startswith("--")]
    height = float(argv[argv.index("--height") + 1]) if "--height" in argv else 0.07
    squares = json.loads((FRAMEWORK_ROOT / "deployments/chess/board_squares.json").read_text())
    pol = make_policies()
    env = make_env()
    print(f"Connecting to {connecting_to()}...")
    env.connect()
    rows = []
    try:
        env.go_home()
        for sq in names:
            target = np.array([*squares[sq]["world"], rc.TABLE_Z + height])
            rot = pol.hold_orientation(env, [("hover", target, pol.spec.travel_tol_m)], f"[{sq}]",
                                       tilt_deg=cfg.approach_tilt(sq))
            if not env.move_ee_to(target, tol=pol.spec.travel_tol_m, orientation=rot):
                print(f"  {sq}: hover not reached — skipped")
                continue
            env.wait_still()
            off = env.overhead_gripper_offset()
            if off is None:
                print(f"  {sq}: gripper tag not in view — skipped")
                continue
            before = (target - (env.get_ee_pos() + off))[:2]
            corr = pol._overhead_correct(env, target, pol.spec.travel_tol_m, rot, f"  [{sq}]")
            env.wait_still()
            off = env.overhead_gripper_offset()            # an independent look, after
            after = None if off is None else (target - (env.get_ee_pos() + off))[:2]
            rows.append((sq, before, corr, after))
            print(f"  {sq}: camera said {np.linalg.norm(before) * 1000:5.1f} mm off before, corrected "
                  f"{np.linalg.norm(corr) * 1000:4.1f} mm, then "
                  + ("tag not seen" if after is None else f"{np.linalg.norm(after) * 1000:4.1f} mm off")
                  + f"  (fit {env.last_tag_fit_px:.2f} px)")
    finally:
        try:
            env.go_home()
        finally:
            env.stop()
    done = [r for r in rows if r[3] is not None]
    if done:
        b = np.array([np.linalg.norm(r[1]) for r in done]) * 1000
        a = np.array([np.linalg.norm(r[3]) for r in done]) * 1000
        print(f"\n{len(done)} squares: before {np.median(b):.1f} mm median ({b.max():.1f} max); "
              f"after {np.median(a):.1f} mm median ({a.max():.1f} max)")


if __name__ == "__main__":
    main()
