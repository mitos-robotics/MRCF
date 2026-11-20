"""Plan and execution bounds, including relative heights and direct dispatch."""

import dataclasses

import numpy as np
import pytest

from mrcf import manipulation
from mrcf import safety
from mrcf import vlm
from fake_arm import FakeArm
from mrcf.interlock import MotionStopped
from mrcf.policies.parallel_jaw import ParallelJaw
from mrcf.skill_executor import run_policy
from test_manipulation import SPEC


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(vlm, "call_vlm", lambda **kw: "")
    monkeypatch.setattr(safety, "_bounds", safety.WorkspaceBounds(z_min=0, z_max=0.2))
    monkeypatch.setattr(safety, "_sequence_validators", [])
    for attr in ("_policies", "_vision", "_scenario"):
        monkeypatch.setattr(manipulation, attr, getattr(manipulation, attr))
    manipulation.install(ParallelJaw(SPEC))
    return FakeArm(), vlm.SceneState(objects=[vlm.SceneObject("block", "block", position=np.array([0.2, 0, 0]))])


@pytest.mark.parametrize("height", [0.3, 0.201, -0.01, float("nan"), float("inf"), "-inf", "bad", []])
@pytest.mark.parametrize("nested", [False, True])
def test_invalid_heights_are_blocked_before_any_motion(world, height, nested):
    arm, state = world
    params = {"object_name": "block", "height": height}
    action = {"action": "move_above", **({"parameters": params} if nested else params)}
    filtered, violations = safety.validate_plan([action], state)
    assert filtered == [] and any(v.severity == "block" for v in violations)
    assert vlm.dispatch(action, arm, state) is False
    assert arm.cycles == 0


@pytest.mark.parametrize("height", [0.0, 0.1, 0.2, "0.15", None])
def test_valid_and_default_heights_remain_usable(world, height):
    arm, state = world
    action = {"action": "move_above", "object_name": "block", "height": height}
    filtered, violations = safety.validate_plan([action], state)
    assert filtered == [action] and not violations
    assert vlm.dispatch(action, arm, state) is True


def test_height_is_relative_to_a_nonzero_work_plane(world):
    arm, state = world
    state.objects[0].position = np.array([0.2, 0, 0.1])
    manipulation.install(ParallelJaw(dataclasses.replace(SPEC, table_z=0.1)))
    action = {"action": "move_above", "object_name": "block", "height": 0.15}
    assert safety.validate_plan([action], state)[0] == []
    assert vlm.dispatch(action, arm, state) is False
    assert arm.cycles == 0
    action["height"] = 0.1
    assert vlm.dispatch(action, arm, state) is True
    assert arm.ee[2] == pytest.approx(0.2)


def test_explicit_bounds_override_global_limits(world):
    _, state = world
    action = {"action": "move_above", "object_name": "block", "height": 0.3}
    assert safety.validate_plan([action], state, bounds=safety.WorkspaceBounds(z_max=0.4))[0] == [action]


@pytest.mark.parametrize("position", [[float("nan"), 0, 0], [0, float("inf"), 0], [0, 0, float("nan")]])
def test_nonfinite_object_coordinates_are_blocked(world, position):
    _, state = world
    state.objects[0].position = np.array(position)
    assert safety.validate_plan([{"action": "pick", "object_name": "block"}], state)[0] == []


def test_execution_monitor_stops_a_policy_that_crosses_the_ceiling(world):
    arm, _ = world
    with pytest.raises(MotionStopped, match="ceiling"):
        run_policy(lambda env: env.move_ee_to(np.array([0.2, 0, 0.4])), "test_move", arm)
    assert arm.holds == 1 and 0.2 <= arm.ee[2] <= 0.211
    # The failed policy leaves no stale monitor registered on the interlock.
    assert safety._monitor is None
    assert not arm.interlock.tripped()


@pytest.mark.parametrize("z, halted", [(0.2, False), (0.201, True), (-0.004, False), (-0.006, True),
                                       (float("nan"), True), (float("inf"), True)])
def test_monitor_height_boundaries(world, z, halted):
    arm, _ = world
    arm.ee[2] = z
    monitor = safety.ExecutionMonitor(arm, "test")
    assert monitor.check() is halted
    if halted:
        arm.ee[2] = 0.1
        assert monitor.check() is True  # latched for this action


@pytest.mark.parametrize("force, halted, near_misses", [(0.0, False, 0), (3.5, False, 1), (5.0, True, 0)])
def test_force_limits_still_work_with_height_monitoring(world, monkeypatch, force, halted, near_misses):
    arm, _ = world
    monkeypatch.setattr(arm, "get_contact_force", lambda: force)
    monitor = safety.ExecutionMonitor(arm, "test")
    assert monitor.check() is halted
    assert len(monitor.near_misses) == near_misses
    if near_misses:
        assert monitor.near_misses[0].margin_pct() == pytest.approx(30)


def test_the_final_approach_stops_at_the_arms_own_tighter_limit(world, monkeypatch):
    # 2026-11-12, the UR5: a finger landing on a block went 19 -> 70 -> 140 N in 0.2 s; its
    # descent read 1-10 N touching nothing, so the final approach stops at 30 N, not 100.
    arm, _ = world
    arm.contact_force_limits = {"max_contact_force_n": 100.0, "near_miss_force_n": 80.0,
                                "approach_contact_force_n": 30.0}
    monkeypatch.setattr(arm, "get_contact_force", lambda: 40.0)
    assert safety.ExecutionMonitor(arm, "test").check() is False          # in a move: under 100 N
    monitor = safety.ExecutionMonitor(arm, "test")
    with safety.approach_contact_limit(arm):
        assert monitor.check() is True and "final approach" in monitor.halt_reason
    assert safety._approach_limit_n is None                               # and only while it lasts


def test_an_arm_without_an_approach_limit_keeps_its_own(world, monkeypatch):
    arm, _ = world
    monkeypatch.setattr(arm, "get_contact_force", lambda: 4.0)            # under the 5 N default
    monitor = safety.ExecutionMonitor(arm, "test")
    with safety.approach_contact_limit(arm):
        assert monitor.check() is False
