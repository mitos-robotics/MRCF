"""Physical outcomes survive failures, stops and retries through dispatch."""

import numpy as np
import pytest

from mrcf import manipulation
from mrcf import safety
from mrcf import vlm
from fake_arm import Cylinder, FakeArm
from mrcf.interlock import MotionStopped
from mrcf.policies.parallel_jaw import ParallelJaw
from test_manipulation import SPEC, scene_with


@pytest.fixture(autouse=True)
def isolated_actions(monkeypatch):
    monkeypatch.setattr(vlm, "call_vlm", lambda **kw: "")
    monkeypatch.setattr(safety, "_bounds", safety.WorkspaceBounds())
    monkeypatch.setattr(safety, "_sequence_validators", [])
    for attr in ("_policies", "_vision", "_scenario"):
        monkeypatch.setattr(manipulation, attr, getattr(manipulation, attr))
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda obj: 0.05))


class FaultArm(FakeArm):
    fault = None
    released = False

    def move_ee_to(self, target, **kw):
        if self.held is not None and target[2] > self.ee[2] and self.fault in ("lift", "lift_exception"):
            if self.fault == "lift_exception":
                raise RuntimeError("lift drive failed")
            return False
        if self.released and self.fault in ("retreat", "retreat_exception", "retreat_stop"):
            if self.fault == "retreat_exception":
                raise RuntimeError("retreat drive failed")
            if self.fault == "retreat_stop":
                raise MotionStopped("hand at retreat")
            return False
        return super().move_ee_to(target, **kw)

    def set_gripper(self, width, **kw):
        releasing = self.held is not None and width > self.width
        if releasing and self.fault == "jam":
            return
        if releasing and self.fault == "before_release_stop":
            raise MotionStopped("hand before release")
        super().set_gripper(width, **kw)
        if releasing:
            self.released = True
            if self.fault == "partial_open":
                self.width = 0.035  # released, but below open_gripper's success threshold
            if self.fault == "partial_release":
                self.width = 0.02   # released, but not enough clearance to retreat
            if self.fault == "open_exception":
                raise RuntimeError("bus lost after opening")
            if self.fault == "open_stop":
                raise MotionStopped("hand after opening")


def setup_pick(fault=None, memory=True):
    state = scene_with(("block", (0.2, 0), 0.021))
    if not memory:
        state.memory = None
    block = state.objects[0]
    arm = FaultArm([Cylinder((0.2, 0), 0.021, 0.04)])
    arm.fault = fault
    return arm, state, block


def pick(arm, state):
    return vlm.dispatch({"action": "pick", "object_name": "block"}, arm, state)


def place(arm, state):
    return vlm.dispatch({"action": "place", "target_name": "spot"}, arm, state)


@pytest.mark.parametrize("fault", ["lift", "lift_exception"])
@pytest.mark.parametrize("memory", [True, False])
def test_failed_lift_blocks_another_pick_but_allows_recovery_place(fault, memory):
    arm, state, block = setup_pick(fault, memory)
    assert pick(arm, state) is False
    assert arm.held is not None and state.held_object() is block
    if memory:
        assert state.memory.get(block.track_id).state == "held"
    if fault == "lift":
        assert block.grasp is not None
    cycles = arm.cycles
    assert pick(arm, state) is False
    assert arm.cycles == cycles and arm.held is not None
    arm.fault = None
    assert place(arm, state) is True
    assert arm.held is None and state.held_object() is None
    assert np.allclose(block.position, state.targets[0].position)


def test_a_missed_pick_does_not_invent_a_held_object():
    arm, state, block = setup_pick()
    arm.objects.clear()
    assert pick(arm, state) is False
    assert state.held_object() is None and block.grasp is None
    assert state.memory.get(block.track_id).state == "present"


def test_jammed_release_fails_without_retreat_or_forgetting_the_hold():
    arm, state, block = setup_pick()
    assert pick(arm, state)
    arm.fault = "jam"
    assert place(arm, state) is False
    assert arm.held is not None and state.held_object() is block
    assert state.memory.get(block.track_id).state == "held"
    assert arm.ee[2] == pytest.approx(block.grasp.release_height_m)
    arm.fault = None
    assert place(arm, state) is True


@pytest.mark.parametrize("fault", ["retreat", "retreat_exception", "retreat_stop"])
def test_failed_retreat_preserves_the_confirmed_placement(fault):
    arm, state, block = setup_pick()
    assert pick(arm, state)
    arm.fault = fault
    if fault == "retreat_stop":
        with pytest.raises(MotionStopped):
            place(arm, state)
    else:
        assert place(arm, state) is False
    assert arm.held is None and state.held_object() is None
    assert block.grasp is None
    assert np.allclose(block.position, state.targets[0].position)
    assert state.memory.get(block.track_id).state == "present"


def test_a_stop_before_release_keeps_the_object_held():
    arm, state, block = setup_pick()
    assert pick(arm, state)
    arm.fault = "before_release_stop"
    with pytest.raises(MotionStopped):
        place(arm, state)
    assert arm.held is not None and state.held_object() is block
    assert state.memory.get(block.track_id).state == "held"


def test_partial_release_records_placement_but_does_not_retreat():
    arm, state, block = setup_pick()
    assert pick(arm, state)
    release_height = block.grasp.release_height_m
    arm.fault = "partial_release"
    assert place(arm, state) is False
    assert arm.ee[2] == pytest.approx(release_height)
    assert arm.held is None and state.held_object() is None
    assert np.allclose(block.position, state.targets[0].position)
    assert state.memory.get(block.track_id).state == "present"


@pytest.mark.parametrize("fault", [None, "partial_open", "open_exception", "open_stop"])
@pytest.mark.parametrize("memory", [True, False])
def test_open_reconciles_releases_even_on_partial_failure_or_stop(fault, memory):
    arm, state, block = setup_pick(memory=memory)
    assert pick(arm, state)
    arm.fault = fault
    if fault == "open_stop":
        with pytest.raises(MotionStopped):
            vlm.dispatch({"action": "open_gripper"}, arm, state)
    else:
        assert vlm.dispatch({"action": "open_gripper"}, arm, state) is (fault is None)
    assert arm.held is None and state.held_object() is None
    assert block not in state.objects and block.grasp is None
    if memory:
        assert state.memory.get(block.track_id).state == "missing"
        # A later scan can match the original identity instead of duplicating it.
        from mrcf.object_memory import Detection
        tracks, _ = state.memory.observe([Detection(position=arm.ee[:2], foot_mm=21)])
        assert tracks[0].id == block.track_id and tracks[0].state == "present"


def test_jammed_open_keeps_the_hold_and_blocks_a_new_pick():
    arm, state, block = setup_pick()
    assert pick(arm, state)
    arm.fault = "jam"
    assert vlm.dispatch({"action": "open_gripper"}, arm, state) is False
    assert state.held_object() is block and arm.held is not None
    assert state.memory.get(block.track_id).state == "held"
    assert pick(arm, state) is False


def test_release_checks_load_even_if_the_width_claims_it_opened(monkeypatch):
    arm, state, block = setup_pick()
    assert pick(arm, state)
    arm.fault = "jam"
    monkeypatch.setattr(arm, "get_gripper_width", lambda: SPEC.max_open_m)
    assert place(arm, state) is False
    assert state.held_object() is block and arm.held is not None


def test_an_unavailable_hold_sensor_does_not_discard_the_held_identity(monkeypatch):
    arm, state, block = setup_pick()
    assert pick(arm, state)
    arm.fault = "jam"

    def unreadable():
        raise OSError("load read failed")

    monkeypatch.setattr(arm, "gripper_loaded", unreadable)
    assert vlm.dispatch({"action": "open_gripper"}, arm, state) is False
    assert state.held_object() is block
    assert state.memory.get(block.track_id).state == "held"


def test_place_also_works_on_a_backend_without_a_load_sensor():
    class WidthOnlyArm(FaultArm):
        gripper_loaded = None          # no load sensor: not a capabilities.GripSensing

    _, state, _ = setup_pick()
    arm = WidthOnlyArm([Cylinder((0.2, 0), 0.021, 0.04)])
    assert pick(arm, state) is True
    assert place(arm, state) is True
    assert arm.held is None and state.held_object() is None


def test_sequence_validation_accounts_for_an_explicit_open():
    arm, state, _ = setup_pick()
    assert pick(arm, state)
    actions = [{"action": "open_gripper"}, {"action": "pick", "object_name": "another"}]
    assert manipulation.gripper_sequence_validator(actions, state) == []


def test_an_object_dropped_on_the_way_is_not_recorded_as_placed():
    # On the way to e3 (2026-10-28) a queen fell out of the jaws and rolled off
    # the board; the place opened the jaws on nothing and called it placed.
    arm, state, block = setup_pick()
    assert pick(arm, state)
    moves = arm.move_ee_to

    def drops_it(target, **kw):
        if arm.held is not None and np.linalg.norm(np.asarray(target)[:2] - arm.ee[:2]) > 0.02:
            arm.held = None                       # out of the jaws mid-carry
        return moves(target, **kw)
    arm.move_ee_to = drops_it
    assert place(arm, state) is False
    assert state.held_object() is None and block not in state.objects
    assert state.memory.get(block.track_id).state == "missing"
