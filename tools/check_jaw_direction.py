"""
check_jaw_direction.py — which way does this arm's gripper really open? Check
it once on the rig, by eye, before any pick turns the jaws by it.

Every pick turns the jaws to the thing it grips — across a handle, square to
a block's sides — by the arm's JAW_SLIDE_AXIS: which way the fingers slide in
the flange's frame, as the gripper is mounted. Written down wrong, every
pick of anything not square comes down along it, onto it. 2026-11-20, the
UR5: a guess, never looked at, and the jaws came down on a hammer's handle at
50 N. Until this has checked it, a pick on the real arm is refused
(mrcf/arm_facts.py).

THE ARM MOVES: home; its wrist turned in place so the jaws should open
toward and away from the arm's base; the jaws closed and opened twice; home
again. Nothing else moves. A hand near the emergency stop.

    MRCF_ARM=ur5 python tools/check_jaw_direction.py

You say what you saw: y (toward and away from the base) — recorded as checked,
rig/<arm>/checked_facts.json; s (sideways) — the axis a quarter turn round,
printed to set in robot_config, then run this again (any earlier check of the
value set now is dropped); anything else — nothing recorded.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mrcf import arm_facts  # noqa: E402
from mrcf.capabilities import JawDirection, OrientationSearch  # noqa: E402
from tools._common import connecting_to, make_env, rc  # noqa: E402


def quarter_turned(axis) -> tuple:
    """The flange-frame axis a quarter turn round the flange's z from `axis`."""
    x, y, z = (float(v) for v in axis)
    return (round(-y, 6) + 0.0, round(x, 6) + 0.0, round(z, 6) + 0.0)


def radial_rotation(env, at) -> "np.ndarray | None":
    """
    The straight-down gripper rotation at `at` that opens its jaws, as the arm's
    settings have them, toward and away from its base (BASE_XY); None: none
    the arm reaches.
    """
    got = env.approach_orientation_for([at])
    if got is None:
        return None
    jaw = env.jaw_direction(got[0])
    if jaw is None:
        return None
    radial = np.asarray(at, float)[:2] - np.asarray(getattr(rc, "BASE_XY", (0.0, 0.0)), float)[:2]
    radial = radial / (np.linalg.norm(radial) or 1.0)
    ang = float(np.degrees(np.arctan2(jaw[0] * radial[1] - jaw[1] * radial[0], float(jaw @ radial))))
    turned = env.approach_orientation_for([at], turn_deg=(ang + 90.0) % 180.0 - 90.0)
    return None if turned is None else np.asarray(turned[0], float)


def main() -> None:
    axis = tuple(getattr(rc, "JAW_SLIDE_AXIS"))
    print(f"Which way the jaws open — {connecting_to()}. Set now: JAW_SLIDE_AXIS = {axis} "
          f"({'checked' if arm_facts.checked(rc, 'JAW_SLIDE_AXIS') else 'NOT checked on the rig'}).")
    env = make_env()
    if not (isinstance(env, OrientationSearch) and isinstance(env, JawDirection)):
        raise SystemExit("This arm's env cannot turn its gripper to a given way (capabilities.OrientationSearch, "
                         "JawDirection): nothing to check here.")
    env.connect()
    try:
        input("\nTHE ARM MOVES: home, then its wrist turns in place and the jaws close and open. "
              "Hand near the emergency stop. [Enter] ")
        if not env.go_home():
            raise SystemExit("Could not go home: nothing checked.")
        at = np.asarray(env.get_ee_pos(), float)
        rot = radial_rotation(env, at)
        if rot is None:
            raise SystemExit("No way to turn the wrist there: nothing checked.")
        env.set_gripper(0.0, n_steps=15)
        if not env.move_ee_to(at, orientation=rot):
            raise SystemExit("Could not turn the wrist there: nothing checked.")
        print("\nWatch the two fingers.")
        opening = float(getattr(rc, "GRIPPER_MAX_OPEN_M", 0.05))
        for _ in range(2):
            env.set_gripper(opening, n_steps=15)
            time.sleep(0.8)
            env.set_gripper(0.0, n_steps=15)
            time.sleep(0.5)
        env.set_gripper(opening, n_steps=15)
        said = input("\nDid the fingers move TOWARD and AWAY from the arm's base — along the line from the base out "
                     "to the gripper? [y = yes / s = no, sideways / n = not sure] ").strip().lower()
        if said in ("y", "yes"):
            where = arm_facts.record(rc, "JAW_SLIDE_AXIS", how="watched: tools/check_jaw_direction.py")
            print(f"\nChecked: JAW_SLIDE_AXIS = {axis}, kept in {where}. Picks may turn the jaws by it.")
        elif said in ("s", "sideways"):
            arm_facts.forget(rc, "JAW_SLIDE_AXIS")          # seen wrong: no longer checked, if it was
            other = quarter_turned(axis)
            print(f"\nThe jaws open a quarter turn from where the arm's settings say. Set, in the arm's "
                  f"robot_config.py:\n    JAW_SLIDE_AXIS = {other}\nthen run this again to check it. "
                  "Nothing recorded.")
        else:
            print("\nNothing recorded: picks stay refused until it is checked.")
    finally:
        try:
            env.go_home()
        finally:
            env.stop()


if __name__ == "__main__":
    main()
