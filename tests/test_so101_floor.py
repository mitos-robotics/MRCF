"""
The SO-101 keeps its gripper out of the table: no command is sent whose path
would take the jaw tips or the wrist below its floor (and lower than they
are), and no goal held to an orientation it cannot reach is gone for — on the
arm's own kinematics, its servos stood in for (tools/so101/visualise_ik).

2026-11-17 14:27: a height look rose toward a transit 40 cm up that the arm
could not reach with the gripper held as the look wanted, stalled, and the
swing on down to the look put the gripper into the table.
"""

import numpy as np
import pytest

from arm_baselines import Skipped, offline_env

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
FLOOR = 0.0007


@pytest.fixture
def arm():
    from mrcf import backends

    so101 = backends.arm("so101")
    try:
        env = offline_env(so101)
    except Skipped as why:
        pytest.skip(str(why))
    from tools.so101.visualise_ik import _FakeFollower

    rc = so101.config
    cameras = env._robot.cameras
    env._robot = _FakeFollower(rc.HOME_DEGREES, rc.MAX_RELATIVE_TARGET_DEG)
    env._robot.cameras, env._robot.calibration = cameras, {}
    env._floor_z = FLOOR
    return env


def _degrees(env) -> dict:
    return {n: float(v) for n, v in zip(JOINTS, env._q_arm_deg())}


def _into_the_table(env) -> dict:
    """Joint goals from here, inside the joints' range, that put the gripper's lowest point well under the floor."""
    start = _degrees(env)
    limits = env._joint_deg_limits
    goals = ({**start, "shoulder_lift": start["shoulder_lift"] + lift, "elbow_flex": start["elbow_flex"] + elbow,
              "wrist_flex": start["wrist_flex"] + wrist}
             for lift in np.arange(0.0, 181.0, 10.0) for elbow in np.arange(-130.0, 131.0, 10.0)
             for wrist in (0.0, -40.0, 40.0))
    for goal in goals:
        if any(not limits[n][0] + 3 <= goal[n] <= limits[n][1] - 3 for n in JOINTS if n in limits):
            continue
        if env._lowest_z(np.array([goal[n] for n in JOINTS]), 0.02) < FLOOR - 0.03:
            return goal
    raise AssertionError("no joint goal reached the table")


def test_a_joint_move_into_the_table_is_not_sent(arm):
    arm.go_home()
    start, sent = _degrees(arm), len(arm._robot.commanded)
    assert arm.goto_joint_degrees(_into_the_table(arm)) is False
    assert arm.move_refusals == 1 and "below the table" in arm.last_refusal
    moved = max(abs(_degrees(arm)[n] - start[n]) for n in JOINTS)
    assert moved < 1e-6 and all(set(c) <= set(JOINTS) for c in arm._robot.commanded[sent:])  # held, nothing else


def test_up_out_of_a_low_spot_is_let_through(arm):
    arm.go_home()
    here = arm._lowest_z(arm._q_arm_deg(), 0.02)
    arm._floor_z = here + 0.05                      # the gripper below its floor already
    goal = _degrees(arm)
    goal["shoulder_lift"] -= 10.0                   # up
    assert arm._lowest_z(np.array([goal[n] for n in JOINTS]), 0.02) > here
    assert arm.goto_joint_degrees(goal) is True


def test_a_goal_held_to_an_orientation_it_cannot_reach_is_not_gone_for(arm):
    arm.go_home()
    found = arm.approach_orientation_for([np.array([0.25, 0.05, 0.10])])
    assert found is not None
    rot = np.asarray(found[0], float)
    sent = len(arm._robot.commanded)
    assert arm.move_ee_to(np.array([0.25, 0.05, 0.45]), orientation=rot) is False
    assert len(arm._robot.commanded) == sent and "cannot be reached" in arm.last_refusal


def test_a_pick_low_over_the_table_still_goes(arm):
    arm.go_home()
    found = arm.approach_orientation_for([np.array([0.25, 0.05, FLOOR + 0.06]), np.array([0.25, 0.05, FLOOR + 0.016])])
    assert found is not None
    rot = np.asarray(found[0], float)
    assert arm.move_ee_to(np.array([0.25, 0.05, FLOOR + 0.06]), orientation=rot)
    assert arm.move_ee_to(np.array([0.25, 0.05, FLOOR + 0.016]), tol=0.004, orientation=rot)
    assert arm.move_refusals == 0


def test_a_move_that_hangs_just_short_stops_within_a_second_or_two(arm, monkeypatch):
    """
    2026-11-18 16:15, 36 cm out: a descent hung 4-8 mm off its goal — inside
    the stall check's 2 x tol, outside tol — and streamed for the whole 20 s
    backstop, the arm shaking. Here the servos stop moving 7 mm over the goal.
    """
    import time

    arm.go_home()
    found = arm.approach_orientation_for([np.array([0.25, 0.05, FLOOR + 0.06])])
    assert found is not None
    rot = np.asarray(found[0], float)
    goal = np.array([0.25, 0.05, FLOOR + 0.03])
    assert arm.move_ee_to(goal + np.array([0.0, 0.0, 0.007]), tol=0.001, orientation=rot)

    def stuck(action):                               # the servos hold where they are, whatever is sent
        arm._robot.commanded.append({k.removesuffix(".pos"): v for k, v in action.items()})
        return action
    monkeypatch.setattr(arm._robot, "send_action", stuck)
    sent, t0 = len(arm._robot.commanded), time.monotonic()
    assert arm.move_ee_to(goal, tol=0.004, max_step_deg=3.0, orientation=rot) is False
    assert len(arm._robot.commanded) - sent < 120 and time.monotonic() - t0 < 8.0
