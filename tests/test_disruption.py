"""
Something knocked, mid-task: a hand over the table, a stack the arm built
knocked down. The sentry pauses the arm while it lasts; after it, the arm does
not come down on what may no longer be there, and a plan stops between steps
to look again. Offline: the fake arm, its sentry's sightings given by hand.
"""

import numpy as np
import pytest

from fake_arm import Cylinder, FakeArm
from mrcf import controller, manipulation, safety
from mrcf.object_memory import Detection, ObjectMemory
from mrcf.policies.base import Target, Vision
from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw
from mrcf.vlm import dispatch, register_action

from test_manipulation import scene_with

SPEC = GripperSpec(max_open_m=0.045, tip_below_grasp_m=0.007, free_depth_m=0.055,
                   finger_thickness_m=0.009, finger_half_width_m=0.008)


class Sentried(FakeArm):
    """A fake arm whose sentry saw, by hand, something move at `moved` (x, y, radius) from `at` (monotonic) on."""

    def __init__(self, objects=(), moved=(), at=0.0):
        super().__init__(objects)
        self.moved, self.at = list(moved), at

    def motion_seen_since(self, t):
        return list(self.moved) if t <= self.at or self.at == 0.0 else []

    def sentry_sources(self):
        return {"motion"}


def test_moved_near_says_how_near_and_only_within_reach():
    arm = Sentried(moved=[(0.50, 0.30, 0.04)])
    assert safety.moved_near(arm, (0.55, 0.30), 0.0) == pytest.approx(0.01)        # 5 cm off, 4 cm across
    assert safety.moved_near(arm, (0.80, 0.30), 0.0) is None                # 30 cm off
    assert safety.moved_near(FakeArm(), (0.50, 0.30), 0.0) is None          # no sentry: nothing to say


def test_a_pick_does_not_come_down_where_something_moved_on_its_way():
    """No wrist camera, no way to find it again: it stops and asks for a look, rather than land on what is there."""
    arm = Sentried([Cylinder((0.20, 0.0), 0.021, 0.04)], moved=[(0.23, 0.02, 0.04)])
    got = ParallelJaw(SPEC).pick(arm, Target(np.array([0.20, 0.0, 0.0]), height_m=0.04), vision=Vision())
    assert not got.ok and "moved near it" in got.message and "Scan again" in got.message and arm.held is None


def test_a_pick_that_can_find_it_again_does_so_and_carries_on():
    """Chess: a fresh look (Vision.relocate) finds the piece where it is now, and the pick goes on."""
    arm = Sentried([Cylinder((0.215, 0.0), 0.021, 0.04)], moved=[(0.23, 0.02, 0.04)], at=1e12)
    looked = []

    def relocate(env, near_xy):
        looked.append(tuple(near_xy))
        env.moved = []                                        # it has settled: nothing moving now
        return np.array([0.215, 0.0])

    got = ParallelJaw(SPEC).pick(arm, Target(np.array([0.20, 0.0, 0.0]), height_m=0.04),
                                 vision=Vision(relocate=relocate))
    assert looked and got.ok and arm.held is not None


def test_a_place_keeps_hold_where_something_moved_on_its_way():
    """A stack knocked down while the arm brought the next block: it does not let go over where the top was."""
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.05))
    state = scene_with(("block", (0.20, 0.0), 0.021))
    arm = Sentried([Cylinder((0.20, 0.0), 0.021, 0.04)])
    assert dispatch({"action": "pick", "object_name": "block"}, arm, state) is True
    arm.moved, arm.at = [(0.27, -0.04, 0.05)], 1e12          # a hand at the spot, from here on
    assert dispatch({"action": "place", "target_name": "spot"}, arm, state) is False
    assert arm.held is not None and state.find("block").position is None     # still in the gripper


_RAN: list = []


@register_action("_disruption_step", description="test", parameters={"object_name": "what it works on"},
                 always=True)
def _step(env, scene_state, **_):
    _RAN.append(1)
    return True


def test_a_plan_stops_between_steps_when_something_moved_near_what_is_on_the_table():
    from mrcf import skill_executor

    _RAN.clear()
    ran = _RAN
    state = scene_with(("red_block_u", (0.57, 0.38), 0.03))
    arm = Sentried()
    controller.execute_plan([{"action": "_disruption_step", "object_name": "red_block_u"}], arm, state, controller.Deployment())
    assert ran == [1]
    arm.moved, arm.at = [(0.60, 0.40, 0.05)], 1e12          # the tower knocked over, after the first step
    skill_executor.clear_report()
    failed = controller.execute_plan([{"action": "_disruption_step", "object_name": "red_block_u"}, {"action": "_disruption_step", "object_name": "red_block_u"}], arm, state, controller.Deployment())
    assert failed and ran == [1, 1]                           # the first ran; the second waits for a look
    said = skill_executor.last_report()
    assert said["skill"] == "plan" and "red_block_u" in said["message"] and not said["refused"]

    arm.moved = [(0.10, 0.75, 0.05)]                          # only far off, at the table's edge
    two = [{"action": "_disruption_step", "object_name": "red_block_u"}] * 2
    assert not controller.execute_plan(two, arm, state, controller.Deployment())
    arm.moved = [(0.25, -0.04, 0.03)]                         # at the spot: on chess, any empty square
    assert not controller.execute_plan(two, arm, state, controller.Deployment())    # the plan does not use it


def test_a_block_knocked_off_a_stack_and_misread_keeps_its_name_by_its_colour():
    """Seen anew where it fell, its letter read wrong: the one missing green block it looks like is it."""
    m = ObjectMemory()
    (g, r), _ = m.observe([Detection(position=(0.35, 0.42), foot_mm=34.0, appearance={"a": -13.0, "b": -18.0}),
                           Detection(position=(0.57, 0.38), foot_mm=34.0, appearance={"a": 9.0, "b": -14.0})])
    m.set_label(g.id, "block 7", "vlm", {"colour": "green"})
    m.set_label(r.id, "block u", "vlm", {"colour": "red"})
    m.picked(g.id)
    m.placed(g.id, (0.57, 0.38), on=r.id, raised_mm=30.0)
    tracks, report = m.observe([Detection(position=(0.57, 0.38), foot_mm=34.0, appearance={"a": 9.0, "b": -14.0}),
                                Detection(position=(0.70, 0.20), foot_mm=22.0, appearance={"a": -13.0, "b": -17.0})])
    fell = tracks[1]
    assert [t.id for t, _ in report["new"]] == [fell.id] and m.get(g.id).state == "missing"

    def green_block(t):
        return t.attributes.get("colour") == "green" and (t.label or "").split()[:1] == ["block"]

    really = m.reidentify(fell.id, "block p", "the vision model calls it green block p", alike=green_block)
    assert really is m.get(g.id) and really.position == [0.70, 0.20] and really.on is None
    assert fell.id not in m.tracks
