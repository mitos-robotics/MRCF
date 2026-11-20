"""Pushing an object with the closed jaws (mrcf/policies/nudge.py), and making room for a pick (manipulation)."""

import numpy as np
import pytest

from fake_arm import Cylinder, FakeArm
from mrcf import manipulation
from mrcf import safety
from mrcf import vlm
from mrcf.policies.base import Obstacle, Target
from mrcf.policies.parallel_jaw import ParallelJaw
from mrcf.vlm import SceneObject, dispatch
from test_manipulation import SPEC, scene_with

PUSHER_R = 0.0085          # the closed fingers' block, as the simulation pushes with it


class PushArm(FakeArm):
    """
    FakeArm whose shut jaws push what they run into: with the jaws closed and
    the tips below a cylinder's top, a cylinder closer to the grasp point than
    its radius plus PUSHER_R is pushed along the line between them until it
    only touches. Enough to see where a push leaves things.
    """

    def move_ee_to(self, target, **kw):
        arrived = super().move_ee_to(target, **kw)
        self._push()
        return arrived

    def _push(self):
        if self.width > 0.001 or self.held is not None:
            return
        tips = self.ee[2] - SPEC.tip_below_grasp_m
        for c in self.objects:
            if tips >= c.height_m:
                continue
            v = c.xy - self.ee[:2]
            d = float(np.linalg.norm(v))
            touch = c.diameter_m / 2 + PUSHER_R
            if 0 < d < touch:
                c.xy = self.ee[:2] + v / d * touch


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(vlm, "call_vlm", lambda **kw: "")
    monkeypatch.setattr(safety, "_bounds", safety.WorkspaceBounds())
    monkeypatch.setattr(safety, "_sequence_validators", [])
    for attr in ("_policies", "_vision", "_scenario"):
        monkeypatch.setattr(manipulation, attr, getattr(manipulation, attr))
    monkeypatch.setitem(manipulation._ROOM, "restore", [])
    monkeypatch.setitem(manipulation._ROOM, "busy", False)


def cylinder_target(c: Cylinder) -> Target:
    return Target(position=np.array([c.xy[0], c.xy[1], 0.0]), foot_m=c.diameter_m, max_height_m=c.height_m)


# ── The push ───────────────────────────────────────────────────────────────────

def test_a_nudge_pushes_an_object_to_where_it_is_asked():
    pawn = Cylinder((0.20, 0.03), 0.024, 0.045)
    arm, pol = PushArm([pawn]), ParallelJaw(SPEC)
    out = pol.nudge(arm, cylinder_target(pawn), (0.20, 0.012))
    assert out.ok, out.message
    assert np.linalg.norm(pawn.xy - (0.20, 0.012)) < 0.003
    assert np.allclose(out.at_xy, (0.20, 0.012))
    assert arm.ee[2] > pawn.height_m                         # up and away afterwards


def test_a_check_moves_nothing():
    pawn = Cylinder((0.20, 0.03), 0.024, 0.045)
    arm, pol = PushArm([pawn]), ParallelJaw(SPEC)
    assert pol.nudge(arm, cylinder_target(pawn), (0.20, 0.012), check_only=True).ok
    assert not arm.path and np.allclose(pawn.xy, (0.20, 0.03))


@pytest.mark.parametrize("neighbour, why", [
    ((0.20, 0.058), "no room for the fingers"),              # where the fingers come down behind it
    ((0.20, -0.006), "pushed into"),                         # in its way
])
def test_a_nudge_is_refused_before_moving_when_it_cannot_be_done(neighbour, why):
    pawn = Cylinder((0.20, 0.03), 0.024, 0.045)
    arm, pol = PushArm([pawn]), ParallelJaw(SPEC)
    out = pol.nudge(arm, cylinder_target(pawn), (0.20, 0.012),
                    obstacles=[Obstacle(np.array(neighbour), 0.012, "rook", 0.045)])
    assert out.refused and why in out.message
    assert not arm.path


def test_too_far_for_a_push_is_refused():
    pawn = Cylinder((0.20, 0.03), 0.024, 0.045)
    out = ParallelJaw(SPEC).nudge(PushArm([pawn]), cylinder_target(pawn), (0.20, -0.03))
    assert out.refused and "pick it up instead" in out.message


def test_a_tall_object_is_pushed_higher_up_so_the_gripper_body_clears_its_top():
    king = Cylinder((0.20, 0.03), 0.030, 0.082)
    arm, pol = PushArm([king]), ParallelJaw(SPEC)
    out = pol.nudge(arm, cylinder_target(king), (0.20, 0.015))
    assert out.ok, out.message
    pushing = [p for p in arm.path if p[2] < 0.05]
    lowest_tips = min(p[2] for p in pushing) - SPEC.tip_below_grasp_m
    # The body is tip_below + free_depth above the tips: over the king's top.
    assert lowest_tips + SPEC.tip_below_grasp_m + SPEC.free_depth_m >= king.height_m
    assert np.linalg.norm(king.xy - (0.20, 0.015)) < 0.003


def test_a_push_that_would_have_to_go_too_high_up_is_refused():
    # Something tall beside the way: the push would rise over it, and a short
    # object pushed that high up tips over.
    stub = Cylinder((0.20, 0.03), 0.024, 0.020)
    out = ParallelJaw(SPEC).nudge(PushArm([stub]), cylinder_target(stub), (0.20, 0.015),
                                  obstacles=[Obstacle(np.array([0.225, 0.045]), 0.012, "vase", 0.15)])
    assert out.refused and "tip it over" in out.message


# ── The nudge action ───────────────────────────────────────────────────────────

def test_the_nudge_action_moves_the_object_in_the_world_model_and_memory():
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.045))
    state = scene_with(("pawn", (0.20, 0.03), 0.024))
    state.targets.append(SceneObject("middle", "the middle of its square", position=np.array([0.20, 0.012, 0.0])))
    pawn = Cylinder((0.20, 0.03), 0.024, 0.045)
    assert dispatch({"action": "nudge", "object_name": "pawn", "target_name": "middle"}, PushArm([pawn]), state)
    obj = state.find("pawn")
    assert np.allclose(obj.position[:2], (0.20, 0.012))
    assert np.allclose(state.memory.get(obj.track_id).position[:2], (0.20, 0.012))


def test_the_nudge_action_with_no_spot_pushes_it_back_where_it_belongs():
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.045, home_of=lambda state, obj, xy: ("its square", np.array([0.20, 0.012]))))
    state = scene_with(("pawn", (0.20, 0.03), 0.024))
    pawn = Cylinder((0.20, 0.03), 0.024, 0.045)
    assert dispatch({"action": "nudge", "object_name": "pawn"}, PushArm([pawn]), state)
    assert np.linalg.norm(pawn.xy - (0.20, 0.012)) < 0.003


# ── Making room for a pick ─────────────────────────────────────────────────────
# Radial jaws at (0.20, 0): they open along x. A neighbour 32 mm along x leaves
# the fingers too little room; one beside them, along y, does not matter.

def crowded(aside=None, home_of=None):
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.045, aside_spots=aside, home_of=home_of))
    state = scene_with(("pawn", (0.20, 0.0), 0.024), ("king", (0.232, 0.0), 0.024))
    pawn, king = Cylinder((0.20, 0.0), 0.024, 0.045), Cylinder((0.232, 0.0), 0.024, 0.045)
    return state, pawn, king, PushArm([pawn, king])


def test_a_pick_refused_for_room_names_what_leaves_too_little():
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(max_height=lambda o: 0.045))
    state = scene_with(("pawn", (0.20, 0.0), 0.024), ("king", (0.232, 0.0), 0.024), ("rook", (0.20, 0.05), 0.024))
    out = manipulation.policies().pick_check(FakeArm(), manipulation.target_of(state.find("pawn")),
                                             manipulation.obstacles_around(state, state.find("pawn")))
    assert out.refused and out.blocked_by == ["king"]


def test_a_crowded_object_is_pushed_clear_and_picked(capsys):
    state, pawn, king, arm = crowded()
    assert dispatch({"action": "pick", "object_name": "pawn"}, arm, state) is True
    assert arm.held is pawn
    assert np.allclose(king.xy, (0.232, 0.0))                 # the cheapest way: the pawn itself, not the king
    assert "Making room: pushing pawn" in capsys.readouterr().out
    assert not manipulation._ROOM["restore"]


def test_a_neighbour_is_lifted_aside_and_put_back_after_the_place(monkeypatch):
    # Two upright cylinders crowding each other's jaws always crowd both ways
    # (each leaves the other what it is left), so a neighbour that can be lifted
    # when the object cannot comes from jaw turns or shapes the fake arm does not
    # have: the object's first pick is refused as the arm would refuse it.
    from mrcf.policies.base import Outcome
    monkeypatch.setattr(manipulation, "ROOM_PUSHES_M", ())    # no pushes: the lift is all there is
    spot = SceneObject("spare", "a free spot", interactable=False, position=np.array([0.20, -0.10, 0.0]))
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.045, aside_spots=lambda state, obj, for_obj, headed_for=None: [spot]))
    state = scene_with(("pawn", (0.20, 0.0), 0.024), ("king", (0.20, 0.06), 0.024))
    state.targets.append(spot)
    pawn, king = Cylinder((0.20, 0.0), 0.024, 0.045), Cylinder((0.20, 0.06), 0.024, 0.045)
    arm, pol = PushArm([pawn, king]), manipulation.policies()
    real, calls = pol.pick, []

    def crowded_first(env, target, obstacles=(), vision=None):
        calls.append(target.position[:2].copy())
        if len(calls) == 1:
            return Outcome(False, "jaws need 32 mm open here but king leaves 6 mm", refused=True, blocked_by=["king"])
        return real(env, target, obstacles, vision)

    monkeypatch.setattr(pol, "pick", crowded_first)
    assert dispatch({"action": "pick", "object_name": "pawn"}, arm, state) is True
    assert arm.held is pawn and np.linalg.norm(king.xy - (0.20, -0.10)) < 0.002
    assert dispatch({"action": "place", "target_name": "spot"}, arm, state) is True
    # The king back where it stood, and the world model agrees.
    assert np.linalg.norm(king.xy - (0.20, 0.06)) < 0.002
    assert np.linalg.norm(state.find("king").position[:2] - (0.20, 0.06)) < 0.002
    assert not manipulation._ROOM["restore"]


def test_a_neighbour_is_pushed_clear_and_put_back_after_the_place(monkeypatch):
    state, pawn, king, arm = crowded(home_of=lambda s, obj, xy: ("king_home", np.array([0.232, 0.0])))
    real = manipulation.policies().nudge

    def pawn_wont_slide(env, target, to_xy, obstacles=(), check_only=False):
        if np.allclose(target.position[:2], (0.20, 0.0)):
            from mrcf.policies.base import Outcome
            return Outcome(False, "wedged", refused=True)
        return real(env, target, to_xy, obstacles, check_only)

    monkeypatch.setattr(manipulation.policies(), "nudge", pawn_wont_slide)
    state.targets.append(SceneObject("far", "far off", interactable=False, position=np.array([0.20, 0.12, 0.0])))
    assert dispatch({"action": "pick", "object_name": "pawn"}, arm, state) is True
    assert arm.held is pawn and np.linalg.norm(king.xy - (0.232, 0.0)) > 0.008
    assert len(manipulation._ROOM["restore"]) == 1
    assert dispatch({"action": "place", "target_name": "far"}, arm, state) is True
    assert np.linalg.norm(king.xy - (0.232, 0.0)) < 0.002
    assert not manipulation._ROOM["restore"]


def test_with_no_way_to_make_room_the_refusal_stands(monkeypatch, capsys):
    monkeypatch.setattr(manipulation, "ROOM_PUSHES_M", ())
    state, pawn, king, arm = crowded()
    assert dispatch({"action": "pick", "object_name": "pawn"}, arm, state) is False
    assert "No way to make room" in capsys.readouterr().out
    assert arm.held is None and not arm.path


def test_an_object_of_unknown_height_is_pushed_no_higher_than_is_safe_for_the_shortest_it_could_be():
    # Kept clear of as the tallest it could be (a king), the push would go up
    # it by that; but it could be a pawn, which pushed that high tips over.
    pawn = Cylinder((0.20, 0.03), 0.024, 0.045)
    unknown = Target(position=np.array([0.20, 0.03, 0.0]), foot_m=0.024, max_height_m=0.082, min_height_m=0.045)
    out = ParallelJaw(SPEC).nudge(PushArm([pawn]), unknown, (0.20, 0.015))
    assert out.refused and "tip it over" in out.message
    known = Target(position=np.array([0.20, 0.03, 0.0]), foot_m=0.024, max_height_m=0.082, min_height_m=0.082)
    assert ParallelJaw(SPEC).nudge(PushArm([Cylinder((0.20, 0.03), 0.024, 0.082)]), known, (0.20, 0.015)).ok


def _lifted_aside(monkeypatch, spots):
    """The pawn's first pick refused for the king, which can only be lifted aside onto one of spots."""
    from mrcf.policies.base import Outcome
    monkeypatch.setattr(manipulation, "ROOM_PUSHES_M", ())
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.045, aside_spots=lambda state, obj, for_obj, headed_for=None: list(spots)))
    state = scene_with(("pawn", (0.20, 0.0), 0.024), ("king", (0.20, 0.06), 0.024))
    state.targets.extend(spots)
    pawn, king = Cylinder((0.20, 0.0), 0.024, 0.045), Cylinder((0.20, 0.06), 0.024, 0.045)
    arm, pol = PushArm([pawn, king]), manipulation.policies()
    real, calls = pol.pick, []

    def crowded_first(env, target, obstacles=(), vision=None):
        calls.append(1)
        if len(calls) == 1:
            return Outcome(False, "jaws need 32 mm open here but king leaves 6 mm", refused=True, blocked_by=["king"])
        return real(env, target, obstacles, vision)

    monkeypatch.setattr(pol, "pick", crowded_first)
    return state, pawn, king, arm


def test_nothing_is_set_aside_on_the_spot_the_picked_object_is_going_to(monkeypatch):
    # 1. e4 e5 2. Bd3 with the bishop crowded by the king: e3 is the king's best
    # free square, and the king set down there would crowd the bishop off d3.
    dest = SceneObject("dest", "where the pawn goes", interactable=False, position=np.array([0.20, -0.10, 0.0]))
    spare = SceneObject("spare", "a free spot", interactable=False, position=np.array([0.26, -0.10, 0.0]))
    state, pawn, king, arm = _lifted_aside(monkeypatch, [dest, spare])
    state.headed_for = "dest"
    assert dispatch({"action": "pick", "object_name": "pawn"}, arm, state) is True
    assert arm.held is pawn and np.linalg.norm(king.xy - (0.26, -0.10)) < 0.002


def test_a_plan_tells_the_pick_where_its_place_goes():
    from mrcf.controller import _headed_for
    plan = [{"action": "pick", "object_name": "a"}, {"action": "move_above", "object_name": "b"},
            {"action": "place", "parameters": {"target_name": "e4"}}, {"action": "pick", "object_name": "c"}]
    assert _headed_for(plan, 1) == "e4"
    assert _headed_for(plan, 3) is None                       # a place step itself
    assert _headed_for(plan, 4) is None                       # no place after it
    assert _headed_for([{"action": "pick"}, {"action": "pick"}, {"action": "place", "target_name": "x"}], 1) is None


def test_an_object_set_aside_and_moved_since_is_not_put_back(monkeypatch, capsys):
    # In a game the person may move the piece the arm set aside: putting it
    # back would undo their move.
    spare = SceneObject("spare", "a free spot", interactable=False, position=np.array([0.20, -0.10, 0.0]))
    far = SceneObject("far", "far off", interactable=False, position=np.array([0.26, 0.10, 0.0]))
    state, pawn, king, arm = _lifted_aside(monkeypatch, [spare])
    state.targets.append(far)
    seen_king = state.find("king")                            # renamed for where it stands once set down
    assert dispatch({"action": "pick", "object_name": "pawn"}, arm, state) is True
    king.xy = np.array([0.20, -0.14])                         # moved by hand, and a look saw it
    seen_king.position = np.array([0.20, -0.14, 0.0])
    assert dispatch({"action": "place", "target_name": "far"}, arm, state) is True
    assert np.allclose(king.xy, (0.20, -0.14))
    assert "moved since it was set aside" in capsys.readouterr().out
    assert not manipulation._ROOM["restore"]
