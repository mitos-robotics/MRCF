"""The bridge: planner names in, policy geometry out, what happened written to the world model and memory."""

import importlib.util
import json

import numpy as np
import pytest

from mrcf import manipulation
from fake_arm import Cylinder, FakeArm
from mrcf.interlock import MotionStopped
from mrcf.object_memory import Detection, ObjectMemory
from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw
from mrcf.vlm import SceneObject, SceneState, dispatch

SPEC = GripperSpec(max_open_m=0.045, tip_below_grasp_m=0.007, free_depth_m=0.055,
                   finger_thickness_m=0.009, finger_half_width_m=0.008)


def scene_with(*pieces):
    """A world model and memory holding these (name, xy, foot_m) objects, as a scan would leave them."""
    memory = ObjectMemory()
    tracks, _ = memory.observe([Detection(position=np.array(xy), foot_mm=foot * 1000) for _, xy, foot in pieces])
    objects = []
    for (name, xy, foot), track in zip(pieces, tracks):
        obj = SceneObject(name, name, position=np.array([xy[0], xy[1], 0.0]))
        obj.track_id = track.id
        obj.geometry = {"base_diameter_m": foot}
        objects.append(obj)
    spot = SceneObject("spot", "an empty spot", position=np.array([0.26, -0.04, 0.0]))
    return SceneState(objects=objects, targets=[spot], memory=memory)


def test_an_object_becomes_a_target_and_its_neighbours_obstacles():
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.09))
    state = scene_with(("cup", (0.2, 0.0), 0.03), ("box", (0.25, 0.0), 0.05))
    cup = state.find("cup")
    cup.measured_shape = {"height_mm": 80.0, "widths_mm": {0: 30, 80: 35}}
    t = manipulation.target_of(cup)
    assert (t.foot_m, t.height_m, t.max_height_m) == (0.03, 0.08, 0.09)
    (box,) = manipulation.obstacles_around(state, exclude=cup)
    assert box.label == "box" and box.radius_m == 0.025



def test_the_scenarios_approach_rule_reaches_the_pick_and_the_place():
    leans, placed_with = {"cup": 25.0, "spot": 15.0}, []
    pol = ParallelJaw(SPEC)
    original_place = pol.place

    def place(env, at, **kw):
        placed_with.append(kw.get("tilt_deg"))
        return original_place(env, at, **kw)

    pol.place = place
    manipulation.install(pol, scenario=manipulation.Scenario(max_height=lambda o: 0.05,
                                                            approach_tilt=lambda o: leans.get(o.name)))
    state = scene_with(("cup", (0.20, 0.0), 0.021))
    assert manipulation.target_of(state.find("cup")).tilt_deg == 25.0
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04)])
    assert dispatch({"action": "pick", "object_name": "cup"}, arm, state) is True
    assert dispatch({"action": "place", "target_name": "spot"}, arm, state) is True
    assert placed_with == [15.0]

def test_pick_and_place_through_the_planner_update_world_model_and_memory():
    seen = []
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.05,
        describe=lambda track: "a small block",
        picked=lambda obj, start, outcome: seen.append(("picked", outcome.ok)),
        placed=lambda obj, target, outcome: seen.append(("placed", target.name, outcome.ok))))
    state = scene_with(("block", (0.20, 0.0), 0.021))
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04)])
    block = state.find("block")

    assert dispatch({"action": "pick", "object_name": "block"}, arm, state) is True
    assert block.position is None and block.grasp is not None
    assert state.memory.get(block.track_id).state == "held"

    assert dispatch({"action": "place", "target_name": "spot"}, arm, state) is True
    assert np.allclose(block.position[:2], (0.26, -0.04))
    assert block.grasp is None
    track = state.memory.get(block.track_id)
    assert track.state == "present" and track.place == "spot"
    assert block.description == "a small block on spot"
    assert seen == [("picked", True), ("placed", "spot", True)]


def test_an_older_track_of_the_object_picked_is_not_kept_clear_of_where_it_is_put_back():
    # 2026-11-04: the b8 knight, picked under a new track, could not be put
    # back on b8 — its older track, "missing, last seen there", filled it.
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.05))
    state = scene_with(("knight", (0.20, 0.0), 0.021))
    old = state.memory._new_track()
    old.state, old.position, old.last_seen = "missing", [0.201, 0.0], state.memory.get(state.objects[0].track_id).last_seen
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04)])
    assert dispatch({"action": "pick", "object_name": "knight"}, arm, state) is True
    assert old.state == "gone"
    assert manipulation.obstacles_around(state) == []


def test_a_picks_measurement_is_what_the_objects_next_pick_plans_by(monkeypatch):
    # 2026-10-30: a pawn the start's scan had in doubt was measured at every
    # pick, and every pick planned it as never measured: only a scan set it.
    from mrcf.policies.base import Measured, Outcome
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.09))
    state = scene_with(("pawn", (0.20, 0.0), 0.021))
    pawn = state.find("pawn")
    assert manipulation.target_of(pawn).height_m is None
    looked = Measured(height_mm=43.0, widths_mm={20: 15.0}, foot_mm=21.0, slant_deg=0.0)
    monkeypatch.setattr(manipulation.policies(), "pick",
                        lambda env, **kw: Outcome(False, "closed on nothing", measured=looked))
    assert dispatch({"action": "pick", "object_name": "pawn"}, FakeArm(), state) is False
    assert manipulation.target_of(pawn).height_m == pytest.approx(0.043)


def test_neighbours_a_pick_measured_are_kept_and_size_the_next_picks_room(monkeypatch):
    # 2026-11-03: the d1 queen was refused on the c1 bishop's foot. A pick that
    # measures a neighbour from over it keeps that, as the neighbour's own:
    # the next pick beside it knows how slim it is where the fingers pass.
    from mrcf.policies.base import Measured, Outcome
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.09))
    state = scene_with(("queen", (0.20, 0.0), 0.027), ("bishop", (0.20, 0.038), 0.024))
    bishop = state.find("bishop")
    assert manipulation.obstacles_around(state, exclude=state.find("queen"))[0].widths_mm is None
    seen = Measured(height_mm=62.0, widths_mm={4: 26.0, 32: 18.5, 44: 18.8}, foot_mm=26.0, slant_deg=20.0)
    monkeypatch.setattr(manipulation.policies(), "pick",
                        lambda env, **kw: Outcome(False, "no room", refused=True, around={"bishop": seen}))
    assert dispatch({"action": "pick", "object_name": "queen"}, FakeArm(), state) is False
    assert bishop.measured_shape["height_mm"] == pytest.approx(62.0)
    (ob,) = manipulation.obstacles_around(state, exclude=state.find("queen"))
    assert ob.label == "bishop" and ob.widths_mm == {4: 26.0, 32: 18.5, 44: 18.8} and ob.height_m == pytest.approx(0.062)


def test_the_scenario_says_how_short_and_slim_an_object_not_yet_measured_could_be():
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.08, min_height=lambda o: 0.045, assumed_shape=lambda o: {0: 30.0, 80: 25.0}))
    state = scene_with(("queen", (0.20, 0.0), 0.027), ("bishop", (0.20, 0.038), 0.024))
    queen, bishop = state.find("queen"), state.find("bishop")
    t = manipulation.target_of(queen)
    assert (t.min_height_m, t.assumed_widths_mm) == (0.045, {0: 30.0, 80: 25.0})
    (ob,) = manipulation.obstacles_around(state, exclude=queen)
    assert ob.widths_mm == {0: 30.0, 80: 25.0}
    bishop.measured_shape = {"height_mm": 62.0, "widths_mm": {4: 26.0, 32: 18.5}}
    queen.measured_shape = {"height_mm": 69.0, "widths_mm": {4: 28.0, 40: 22.0}}
    assert manipulation.obstacles_around(state, exclude=queen)[0].widths_mm == {4: 26.0, 32: 18.5}
    t = manipulation.target_of(queen)
    assert t.min_height_m is None and t.assumed_widths_mm is None          # measured: its own shape


def test_a_verb_for_a_piece_the_scan_missed_finds_it_by_its_name_or_where_it_was_last_seen(monkeypatch):
    # 2026-11-03: the f1 bishop fell over on c4; lying, the scan no longer
    # outlined it, and standing it up needed it by name.
    from mrcf.policies.base import Outcome
    manipulation.install(ParallelJaw(SPEC))
    state = scene_with(("bishop", (0.20, 0.0), 0.024), ("pawn", (0.25, 0.04), 0.021))
    bishop = state.find("bishop")
    state.memory.observe([Detection(position=np.array([0.25, 0.04]), foot_mm=21.0)])   # the bishop not seen
    state.objects.remove(bishop)
    state.targets.append(SceneObject("c4", "empty square c4", position=np.array([0.205, 0.004, 0.0])))
    seen = []
    manipulation.object_action("_test_tip_up", lambda pol, env, target, obstacles, **_:
                               seen.append(target.position[:2].copy()) or Outcome(True, "up", at_xy=np.array([0.21, 0.0])),
                               description="test", moves_object=True, finds_missing=True)
    assert dispatch({"action": "_test_tip_up", "object_name": "c4"}, FakeArm(), state) is True
    assert np.allclose(seen[0], (0.20, 0.0))                              # where it was last seen
    found = next(o for o in state.objects if o.track_id == bishop.track_id)
    assert np.allclose(found.position[:2], (0.21, 0.0))                   # and where it stands now
    assert dispatch({"action": "_test_tip_up", "object_name": "e8"}, FakeArm(), state) is False   # nothing missing there


def test_a_refused_pick_reports_failure_and_changes_nothing():
    manipulation.install(ParallelJaw(SPEC))
    # Radial jaws at (0.2, 0) open along x: neighbours 21.5 mm either side along x.
    state = scene_with(("a", (0.20, 0.0), 0.021), ("b", (0.2215, 0.0), 0.021), ("c", (0.1785, 0.0), 0.021))
    arm = FakeArm()
    assert dispatch({"action": "pick", "object_name": "a"}, arm, state) is False
    assert state.find("a").position is not None and arm.cycles == 0


def test_the_scenario_hears_where_a_pick_took_an_object_from_and_only_when_it_did():
    vacated = []
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.05, vacated=lambda state, obj, xy: vacated.append((obj.name, tuple(xy)))))
    state = scene_with(("block", (0.2, 0.0), 0.021))
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04)])
    assert dispatch({"action": "pick", "object_name": "block"}, arm, state) is True
    assert vacated == [("block", (0.2, 0.0))]
    vacated.clear()
    state = scene_with(("a", (0.2, 0.0), 0.06))                # wider than the jaws open: refused, nothing taken
    assert dispatch({"action": "pick", "object_name": "a"}, FakeArm(), state) is False
    assert vacated == []


def test_a_pick_stopped_after_the_grasp_leaves_the_object_held_in_the_world_model():
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.05))
    state = scene_with(("block", (0.20, 0.0), 0.021))
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04)])
    # Stopped on the way up, with the block already in the jaws.
    arm.interlock.add_check("hands", lambda: "hand near the arm"
                            if arm.held is not None and arm.ee[2] > 0.04 else None)
    with pytest.raises(MotionStopped):
        dispatch({"action": "pick", "object_name": "block"}, arm, state)
    block = state.find("block")
    assert block.position is None and state.held_object() is block
    assert state.memory.get(block.track_id).state == "held"


def test_an_object_the_last_scan_missed_is_still_kept_clear_of_for_a_while():
    state = scene_with(("a", (0.20, 0.0), 0.021), ("b", (0.20, -0.075), 0.027))
    b = state.find("b")
    state.memory.observe([Detection(position=np.array((0.20, 0.0)), foot_mm=21.0)])   # b not seen
    state.objects = [o for o in state.objects if o is not b]
    (missed,) = manipulation.obstacles_around(state, exclude=state.find("a"))
    assert np.allclose(missed.position, (0.20, -0.075)) and "missed" in missed.label


def test_an_object_only_the_vision_model_reported_is_not_kept_clear_of(capsys):
    # 2026-10-21: a piece invented on e1 refused the pick on d1 next to it.
    state = scene_with(("d1", (0.14, 0.0075), 0.021), ("e1", (0.14, -0.0325), 0.021))
    state.find("e1").confirmed = False
    assert manipulation.obstacles_around(state, exclude=state.find("d1")) == []
    assert "e1 (only the vision model reported it)" in capsys.readouterr().out


def test_a_missed_object_is_not_kept_clear_of_where_something_now_stands():
    # 2026-10-21 12:53: the pawn's own stale track (missed, its identity lost)
    # refused the pick of the pawn standing where it had stood, "0 mm" of room.
    state = scene_with(("pawn", (0.1613, -0.0135), 0.0206), ("b", (0.20, -0.075), 0.027))
    stale = state.memory.observe([Detection(position=np.array((0.1613, -0.0135)), foot_mm=20.6),
                                  Detection(position=np.array((0.20, -0.075)), foot_mm=27.0),
                                  Detection(position=np.array((0.1695, -0.0088)), foot_mm=20.1)])[0][2]
    state.memory.observe([Detection(position=np.array((0.1613, -0.0135)), foot_mm=20.6),
                          Detection(position=np.array((0.20, -0.075)), foot_mm=27.0)])
    assert stale.state == "missing"
    assert [o.label for o in manipulation.obstacles_around(state, exclude=state.find("pawn"))] == ["b"]


def test_a_missed_object_only_the_vision_model_ever_reported_is_not_kept_clear_of():
    state = scene_with(("a", (0.20, 0.0), 0.021))
    ghost = state.memory.observe([Detection(position=np.array((0.20, 0.0)), foot_mm=21.0),
                                  Detection(position=np.array((0.20, -0.04)), confirmed=False)])[0][1]
    state.memory.observe([Detection(position=np.array((0.20, 0.0)), foot_mm=21.0)])
    assert ghost.state == "missing" and not ghost.confirmed
    assert manipulation.obstacles_around(state, exclude=state.find("a")) == []


def test_a_pick_never_goes_for_an_empty_square_that_merely_shares_the_name():
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.09))
    state = scene_with(("black_piece_f4", (0.21, -0.03), 0.023))
    state.targets.append(SceneObject("e4", "empty square e4", position=np.array([0.222, -0.034, 0.0])))
    # The scan renamed the piece; the old name now matches only the square.
    arm = FakeArm()
    assert manipulation._pick(arm, state, "white_piece_e4") is False
    assert arm.path == []                                    # never moved
    assert state.find("white_piece_e4") is not None          # a place could still use it


def test_a_verb_of_your_own_acts_on_a_named_object_and_the_world_model_follows():
    from mrcf.policies.base import Outcome

    class Pushing(ParallelJaw):
        def push(self, env, target, distance_m, obstacles=()):
            to = np.asarray(target.position[:2], float) + [float(distance_m), 0.0]
            env.objects[0].xy = to.copy()                    # the fake arm's block slides there
            return Outcome(True, "pushed", at_xy=to)

    manipulation.install(Pushing(SPEC))
    manipulation.object_action(
        "push", lambda pol, env, target, obstacles, distance_m, **_: pol.push(env, target, distance_m, obstacles),
        description="Push an object away from the arm.", parameters={"distance_m": "How far, metres."},
        moves_object=True)
    state = scene_with(("block", (0.20, 0.0), 0.021))
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04)])

    assert dispatch({"action": "push", "object_name": "block", "distance_m": 0.03}, arm, state) is True
    block = state.find("block")
    assert np.allclose(block.position[:2], (0.23, 0.0))
    track = state.memory.get(block.track_id)
    assert np.allclose(track.position, (0.23, 0.0)) and track.history[-1]["event"] == "moved"
    # Like the standard actions: a name not in the world model does nothing.
    assert dispatch({"action": "push", "object_name": "nothing_here", "distance_m": 0.03}, arm, state) is False


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None,
                    reason="needs OpenCV: the flight recorder writes frames with it")
def test_a_verb_of_your_own_that_fails_leaves_an_incident_with_what_the_policy_saw(tmp_path, monkeypatch):
    from mrcf import flight_recorder as fr
    from mrcf.policies.base import Outcome

    rec = fr.FlightRecorder().start(tmp_path)
    monkeypatch.setattr(fr, "RECORDER", rec)

    class Standing(ParallelJaw):
        def stand_up(self, env, target, obstacles=()):
            o = Outcome(False, "closed on nothing", at_xy=target.position[:2])
            o.views = [("at_grip", np.full((48, 64, 3), 90, np.uint8))]
            return o

    looked = []
    manipulation.install(Standing(SPEC))
    manipulation.object_action("stand_up", lambda pol, env, target, obstacles, **_: pol.stand_up(env, target),
                               description="Stand it up.",
                               after=lambda *a: looked.append((fr.RECORDER.last_action().name,
                                                               fr.RECORDER.last_action().ended)))
    try:
        assert dispatch({"action": "stand_up", "object_name": "pawn"}, FakeArm(), scene_with(("pawn", (0.2, 0.0), 0.02))) \
            is False
    finally:
        rec.stop()
    # The look afterwards is part of the action, so the incident shows it too.
    assert looked == [("stand_up", None)]
    (incident,) = (rec.directory / "incidents").iterdir()
    summary = json.loads((incident / "summary.json").read_text())
    assert summary["why"] == "stand_up failed: closed on nothing"
    assert any(p.name.startswith("event_") and p.name.endswith("_policy.jpg") for p in (incident / "frames").iterdir())


def test_a_task_that_stacks_nothing_never_puts_one_object_on_another(capsys):
    """Chess: e4 occupied is no place target, and "e4" matches white_pawn_e4 loosely — no piece goes on the pawn."""
    from mrcf.action_registry import ACTIONS

    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.05))
    assert "place_on" not in ACTIONS                          # not offered to the planner
    state = scene_with(("black_knight_g8", (0.20, 0.0), 0.021), ("white_pawn_e4", (0.25, 0.05), 0.021))
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04), Cylinder((0.25, 0.05), 0.021, 0.04)])
    assert dispatch({"action": "pick", "object_name": "black_knight_g8"}, arm, state) is True
    assert manipulation._place(arm, state, "e4") is False      # refused: the pawn stands there
    assert "on top of" not in capsys.readouterr().out
    assert manipulation._place_on(arm, state, "white_pawn_e4") is False
    assert state.find("black_knight_g8").position is None      # still held

    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.05,
                                                                           carries=lambda o: True))
    assert "place_on" in ACTIONS


def test_a_place_at_a_spot_that_has_gone_is_not_taken_for_the_object_it_was_named_for():
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.05,
                                                                           carries=lambda o: True))
    state = scene_with(("red_block", (0.20, 0.0), 0.021), ("blue_block", (0.25, 0.05), 0.021))
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04), Cylinder((0.25, 0.05), 0.021, 0.04)])
    assert dispatch({"action": "pick", "object_name": "red_block"}, arm, state) is True
    assert manipulation._place(arm, state, "where_blue_block_was") is False     # gone: not onto blue_block
    assert state.find("red_block").position is None


def test_a_pick_is_refused_while_the_grippers_own_sensor_says_it_holds_something(capsys):
    """The world model thought the jaws empty (a stop mid-pick, Ctrl-C): a pick opens them first, and drops it."""
    class Loaded(FakeArm):
        def gripper_loaded(self):
            return True

    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.05))
    state = scene_with(("block", (0.20, 0.0), 0.021))
    arm = Loaded([Cylinder((0.20, 0.0), 0.021, 0.04)])
    arm.set_gripper(0.02)                                   # closed on something
    assert dispatch({"action": "pick", "object_name": "block"}, arm, state) is False
    assert "already holds something" in capsys.readouterr().out
    arm.set_gripper(SPEC.max_open_m)                        # open: loaded or not, no hold
    assert dispatch({"action": "pick", "object_name": "block"}, arm, state) is True


# ── stack_sequence_validator ──────────────────────────────────────────────────

def _blocks(*on):
    """Three blocks on the table; on: (name, under) pairs already stacked."""
    state = SceneState()
    for name, xy in (("blue", (0.43, 0.26)), ("green", (0.55, 0.45)), ("red", (0.29, 0.54))):
        state.objects.append(SceneObject(name, f"{name} block", position=np.array([xy[0], xy[1], 0.0])))
    for name, under in on:
        state.find(name).on = under
    return state


def _steps(*steps):
    return [{"action": a, "parameters": {"object_name": n}} for a, n in steps]


def test_a_plan_that_picks_what_it_has_just_stacked_on_is_not_run():
    # UR5, 2026-11-13: blue onto green, then green onto red. The pick of green was
    # refused after blue was on it, and the plan made again took blue off.
    plan = _steps(("pick", "blue"), ("place_on", "green"), ("pick", "green"), ("place_on", "red"))
    found = manipulation.stack_sequence_validator(plan, _blocks())
    assert len(found) == 1 and found[0].action is plan[2] and found[0].severity == "block"
    assert "blue is on top of it (step 2 puts it there)" in found[0].reason and "bottom up" in found[0].reason


def test_a_tower_built_from_the_bottom_up_passes():
    plan = _steps(("pick", "green"), ("place_on", "red"), ("pick", "blue"), ("place_on", "green"))
    assert manipulation.stack_sequence_validator(plan, _blocks()) == []
    # Onto the base again: its stack's top is green, so blue goes there; then nothing picks under it.
    plan = _steps(("pick", "green"), ("place_on", "red"), ("pick", "blue"), ("place_on", "red"))
    assert manipulation.stack_sequence_validator(plan, _blocks()) == []


def test_picking_under_what_stands_on_it_now_is_not_run_but_the_top_is():
    state = _blocks(("blue", "green"))
    found = manipulation.stack_sequence_validator(_steps(("pick", "green")), state)
    assert found and "blue is on top of it (it is there now)" in found[0].reason
    assert manipulation.stack_sequence_validator(_steps(("pick", "blue"), ("place_on", "red"), ("pick", "green")),
                                                 state) == []


def test_a_pick_refused_for_what_stands_on_it_tells_the_planner_why():
    from mrcf import skill_executor
    skill_executor.clear_report()
    state = _blocks(("blue", "green"))
    assert manipulation._pick(FakeArm(), state, "green") is False
    said = skill_executor.last_report()
    assert said["refused"] and said["message"] == "blue is on top of green: move blue off first"


def test_a_place_beside_lets_go_of_what_was_held():
    # 2026-11-13: a plan putting a block beside another, then picking again, was
    # blocked as "pick planned while already holding an object".
    plan = _steps(("pick", "blue"), ("place_beside", "red"), ("pick", "green"), ("place_on", "red"))
    assert [v for v in manipulation.gripper_sequence_validator(plan, _blocks()) if v.severity == "block"] == []


def test_a_place_in_lets_go_and_a_pick_by_its_wall_holds():
    # 2026-11-18: "put the four things into the tub" — pick, place_in, pick, ... —
    # blocked at every pick after the first, and planned again into the same block.
    plan = [{"action": "pick", "parameters": {"object_name": "blue"}},
            {"action": "place_in", "parameters": {"container": "tub"}},
            {"action": "pick_by_rim", "parameters": {"object_name": "green"}},
            {"action": "place_in", "parameters": {"container": "tub"}},
            {"action": "pick", "parameters": {"object_name": "red"}}]
    assert [v for v in manipulation.gripper_sequence_validator(plan, _blocks()) if v.severity == "block"] == []
    held = [{"action": "pick_by_rim", "parameters": {"object_name": "green"}},
            {"action": "pick", "parameters": {"object_name": "red"}}]
    assert [v.severity for v in manipulation.gripper_sequence_validator(held, _blocks())] == ["block"]



def test_what_the_scan_put_inside_the_object_being_picked_is_part_of_it(monkeypatch):
    """2026-11-19: a figure outlined as two, its head 1 cm from its body: "multicoloured_toy leaves 0 mm"."""
    import numpy as np

    from mrcf import manipulation
    from mrcf.vlm import SceneObject, SceneState

    body = SceneObject("red_ball", "a red figure", interactable=True, position=np.array([0.20, 0.17, 0.0]))
    body.geometry = {"base_diameter_m": 0.022}
    head = SceneObject("multicoloured_toy", "its head", interactable=True, position=np.array([0.195, 0.175, 0.0]))
    head.geometry = {"base_diameter_m": 0.016}
    other = SceneObject("white_plug", "a plug", interactable=True, position=np.array([0.30, 0.17, 0.0]))
    other.geometry = {"base_diameter_m": 0.054}
    state = SceneState()
    state.objects = [body, head, other]
    names = lambda: [o.label for o in manipulation.obstacles_around(state, exclude=body, quiet=True)]   # noqa: E731
    assert names() == ["multicoloured_toy", "white_plug"]                     # off: kept clear of, as before
    monkeypatch.setattr(manipulation, "PARTS_INSIDE_ARE_ITS_OWN", True)
    assert names() == ["white_plug"]                                          # on: part of what is picked


def test_a_pick_by_its_kinds_recipe_opens_the_jaws_past_what_the_kind_is_across(monkeypatch, capsys):
    """2026-11-20 09:13: the plug's foot read 44 mm, its kind 59 across; the jaws opened 52 and one landed on it."""
    from mrcf.policies.base import Outcome
    pol = ParallelJaw(SPEC)
    seen = []
    monkeypatch.setattr(pol, "pick", lambda env, target, **k: seen.append(target) or Outcome(False, "stop here"))
    for across in (0.059, 0.030):
        manipulation.install(pol, scenario=manipulation.Scenario(
            max_height=lambda o: 0.09,
            recipe=lambda st, o, a=across: {"grip_at_m": 0.018, "max_height_m": 0.08, "across_m": a}))
        state = scene_with(("plug", (0.22, 0.05), 0.044))
        dispatch({"action": "pick", "object_name": "plug"}, FakeArm([]), state)
    assert seen[0].foot_m == pytest.approx(0.059) and seen[0].grip_at_m == pytest.approx(0.018)
    assert seen[1].foot_m == pytest.approx(0.044)                       # narrower than the scan's foot: the foot
    assert "its kind is 59 mm across, the scan made its foot 44 — the jaws open for 59." in capsys.readouterr().out
