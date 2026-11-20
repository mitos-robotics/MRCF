"""
A thing lying along the table, its height not known (mrcf/policies/lying_pick.py):
a hammer gripped by its handle next to its heavy head, a screwdriver at its
balance point (in its handle), closed from no higher than it is wide there and
lower until held, lifted a little first and the grip checked; a neighbour in
the jaws' way passed over; anything not lying along the table left to the
plain pick. And how long and narrow across the scan finds it
(plane_geometry.long_extent_mm).
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fake_arm import FakeArm  # noqa: E402
from mrcf.perception.plane_geometry import long_extent_mm  # noqa: E402
from mrcf.policies.base import Obstacle, Target  # noqa: E402
from mrcf.policies.lying_pick import grips, is_lying, lying_pick  # noqa: E402
from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw  # noqa: E402

SPEC = GripperSpec(max_open_m=0.052, tip_below_grasp_m=0.011, free_depth_m=0.040,
                   finger_thickness_m=0.009, finger_half_width_m=0.010, jaw_axis="tangential",
                   hover_height_m=0.06, lift_height_m=0.10, lift_clearance_m=0.05, min_grasp_height_m=0.016,
                   lying_pick=True, handle_first=True)
OUTLINES = {"overhead": {"size": [1280, 720], "px": []}}


def hammer(x0=0.20, y=0.30, handle_len=0.28, handle_w=0.028, head_w=0.110, head_t=0.025):
    """A claw hammer lying on its side, from above: its handle along +x, its head across the far end."""
    x1, h, H = x0 + handle_len, handle_w / 2, head_w / 2
    return np.array([[x0, y - h], [x1, y - h], [x1, y - H], [x1 + head_t, y - H], [x1 + head_t, y + H],
                     [x1, y + H], [x1, y + h], [x0, y + h]])


def screwdriver(x0=0.10, y=0.40, shaft=0.10, shaft_w=0.006, handle=0.10, handle_w=0.028):
    """A screwdriver lying along +x: a thin steel shaft, then its handle."""
    a, b = shaft_w / 2, handle_w / 2
    x1, x2 = x0 + shaft, x0 + shaft + handle
    return np.array([[x0, y - a], [x1, y - a], [x1, y - b], [x2, y - b], [x2, y + b], [x1, y + b], [x1, y + a],
                     [x0, y + a]])


def _rig(monkeypatch, poly, closes, holding=True, spec=SPEC):
    from mrcf.policies import rim_pick as rp

    pol = ParallelJaw(spec)
    arm = FakeArm()
    monkeypatch.setattr(rp, "rim_outline", lambda env, target, h: poly)
    seq = iter(closes)
    monkeypatch.setattr(arm, "squeeze_gripper", lambda **k: next(seq))
    downs = []
    monkeypatch.setattr(pol, "descend_onto", lambda env, goal, **k: downs.append(np.asarray(goal, float)) or True)
    checks = iter(holding if isinstance(holding, list) else [holding] * 4)
    monkeypatch.setattr(pol, "holding", lambda env: next(checks))
    monkeypatch.setattr(pol, "_can_overhead", lambda env: False)
    return pol, arm, downs


def test_a_hammer_is_gripped_by_its_handle_next_to_its_heavy_head():
    """User 2026-11-20: "the hammer is one side heavy so account for that in the pick"."""
    poly = hammer()
    assert is_lying(poly, SPEC.lying_min_aspect)
    centre, across, width, how = grips(poly, SPEC)[0]
    assert how == "handle"
    assert width == pytest.approx(0.028, abs=0.003)
    join = 0.20 + 0.28
    assert join - 0.035 < centre[0] < join - SPEC.finger_half_width_m      # beside the head, not mid-handle
    assert abs(centre[1] - 0.30) < 0.003
    assert abs(across[1]) > 0.99                                           # the jaws open across the handle


def test_a_screwdriver_with_no_handle_to_join_is_gripped_at_its_balance_point_in_its_handle():
    poly = screwdriver()
    assert is_lying(poly, SPEC.lying_min_aspect)
    centre, _, width, how = grips(poly, SPEC)[0]
    assert how == "balance"
    assert 0.20 < centre[0] < 0.30                                         # in the handle, not on the shaft
    assert width == pytest.approx(0.028, abs=0.003)


def test_its_height_unknown_it_is_closed_from_below_its_width_down_until_held(monkeypatch):
    """2026-11-20 14:34: planned for 150 mm, the UR5's jaws closed 116 mm above the hammer, twice."""
    pol, arm, downs = _rig(monkeypatch, hammer(), [(0.0003, False), (0.026, True)])
    target = Target(np.array([0.46, 0.30, 0.0]), outlines=OUTLINES, max_height_m=0.15)
    out = pol.pick(arm, target)
    assert out.ok, out.message
    assert "handle" in out.message
    assert [a["result"] for a in out.attempts] == ["closed_on_nothing", "ok"]
    heights = [round(float(d[2] - SPEC.table_z), 4) for d in downs]
    assert heights == [pytest.approx(0.028 - SPEC.handle_step_m, abs=0.003), pytest.approx(0.016)]
    assert max(heights) < 0.03                                             # never in the air over it


def test_a_grip_that_turns_out_of_the_jaws_is_caught_a_little_way_up(monkeypatch):
    pol, arm, _ = _rig(monkeypatch, hammer(), [(0.026, True)], holding=[False])
    out = lying_pick(pol, arm, Target(np.array([0.46, 0.30, 0.0]), outlines=OUTLINES))
    assert not out.ok and "turned out of the jaws" in out.message
    assert out.attempts[-1]["result"] == "grip_lost"
    # Found not held at the test lift, 2 cm over the grasp: holding() answers once only, so no full lift came
    # first. Then opened, and back up over it.
    lifted = [p[2] for p in arm.path]
    grasp_z = SPEC.table_z + out.grasp.grasp_height_m
    test_at = lifted.index(min(lifted))
    assert lifted[test_at] == pytest.approx(grasp_z + SPEC.lying_test_lift_m, abs=0.002)
    assert arm.width > 0.02


def test_a_neighbour_in_the_jaws_way_is_passed_over_and_all_of_it_blocked_is_refused_unmoved(monkeypatch):
    poly = hammer()
    first = grips(poly, SPEC)[0][0]
    beside = Obstacle(position=first + np.array([0.0, 0.030]), radius_m=0.01, label="tape")
    pol, arm, _ = _rig(monkeypatch, poly, [(0.026, True)])
    out = lying_pick(pol, arm, Target(np.array([0.46, 0.30, 0.0]), outlines=OUTLINES), [beside])
    assert out.ok, out.message
    assert "balance point" in out.message                                  # not the handle, the tape beside it
    pol2, arm2, _ = _rig(monkeypatch, poly, [])
    everywhere = Obstacle(position=np.array([0.34, 0.33]), radius_m=0.20, label="the box")
    out = lying_pick(pol2, arm2, Target(np.array([0.46, 0.30, 0.0]), outlines=OUTLINES), [everywhere])
    assert out.refused and "the box" in out.message
    assert arm2.path == []
    part = Obstacle(position=np.array([0.30, 0.30]), radius_m=0.01, label="its own shaft")
    pol3, arm3, _ = _rig(monkeypatch, poly, [(0.026, True)])
    assert lying_pick(pol3, arm3, Target(np.array([0.46, 0.30, 0.0]), outlines=OUTLINES), [part]).ok


def test_what_does_not_lie_along_the_table_is_left_to_the_plain_pick(monkeypatch):
    box = np.array([[0.30, 0.30], [0.365, 0.30], [0.365, 0.358], [0.30, 0.358]])    # a tape measure
    pol, arm, _ = _rig(monkeypatch, box, [])
    assert lying_pick(pol, arm, Target(np.array([0.33, 0.33, 0.0]), outlines=OUTLINES)) is None
    pol, arm, _ = _rig(monkeypatch, hammer(), [], spec=dataclasses.replace(SPEC, lying_pick=False))
    assert lying_pick(pol, arm, Target(np.array([0.46, 0.30, 0.0]), outlines=OUTLINES)) is None
    assert arm.path == []


def _mask(poly_mm, size=(400, 400)):
    m = np.zeros(size, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly_mm)).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m


def test_the_scan_finds_how_long_a_tool_is_and_how_narrow_across_not_its_round_foot():
    one_px_a_mm = SimpleNamespace(to_plane=lambda px: np.asarray(px, float))
    h = _mask(hammer(x0=40, y=200, handle_len=280, handle_w=28, head_w=110, head_t=25))
    got = long_extent_mm(h, one_px_a_mm)
    assert got["length_mm"] == pytest.approx(305, abs=3)
    assert got["widest_mm"] == pytest.approx(110, abs=3)
    assert got["narrowest_mm"] == pytest.approx(28, abs=3)                # its handle; the head is at an end
    s = _mask(screwdriver(x0=40, y=200, shaft=100, shaft_w=6, handle=100, handle_w=28))
    got = long_extent_mm(s, one_px_a_mm)
    assert got["length_mm"] == pytest.approx(200, abs=3) and got["narrowest_mm"] < 10
    assert long_extent_mm(np.zeros((50, 50), np.uint8), one_px_a_mm) is None


def test_on_the_ur5_itself_a_hammer_is_picked_by_its_handle_with_the_arms_own_moves(monkeypatch):
    """
    The UR5 offline (its own env: moves, orientation search, the descent's
    safety), not a fake arm: 2026-11-20 14:56, every lying pick on the rig
    raised before moving — the UR5's approach_orientation_for takes no
    prefer_tilt_deg, and the fake arm searches no orientations at all.
    """
    from mrcf.backends import ur5
    from mrcf.perception.plane_geometry import PlaneView

    env = ur5.make_env(host=None)
    pol = ur5.make_policies(lying_pick=True, handle_first=True)
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    T[:3, 3] = [0.35, 0.40, 1.0]                                  # straight down, a metre over the table
    model = {"K": np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]), "D": np.zeros(5), "T_world_cam": T,
             "plane_z_m": 0.0}
    monkeypatch.setattr(env, "camera_model", lambda camera, w, h: model)
    poly = hammer(x0=0.20, y=0.40)
    px = PlaneView.from_camera_model(model).project(np.c_[poly * 1000.0, np.full(len(poly), 14.0)])
    target = Target(np.array([0.46, 0.40, 0.0]), max_height_m=0.15,
                    outlines={"overhead": {"size": [1280, 720], "px": px.tolist()}})
    closes = iter([(0.0, False), (0.026, True)])
    monkeypatch.setattr(env, "squeeze_gripper", lambda **k: next(closes))
    monkeypatch.setattr(pol, "holding", lambda e: True)
    monkeypatch.setattr(pol, "_can_overhead", lambda e: False)
    out = pol.pick(env, target)
    assert out.ok, out.message
    assert [a["result"] for a in out.attempts] == ["closed_on_nothing", "ok"]
    assert [a["approach"] for a in out.attempts] == ["lying_handle"] * 2
    grip = out.at_xy
    assert 0.44 < grip[0] < 0.475 and abs(grip[1] - 0.40) < 0.006          # beside the head, on the handle
    assert env.get_ee_pos()[2] == pytest.approx(SPEC.table_z + out.grasp.lift_height_m, abs=0.01)


def test_a_flared_end_to_the_hammers_grip_is_part_of_its_handle_not_where_it_joins():
    """2026-11-20, the UR5's hammer: one 3 mm slice of its grip's end read 42 mm, and it had "no handle"."""
    from mrcf.policies.handle_pick import find_handle

    plain = hammer()
    x0, y, h, lip = 0.20, 0.30, 0.014, 0.021
    flared = np.vstack([[[x0, y - lip], [x0 + 0.006, y - lip], [x0 + 0.006, y - h]], plain[1:-1],
                        [[x0 + 0.006, y + h], [x0 + 0.006, y + lip], [x0, y + lip]]])
    centre, _, width, how = grips(flared, SPEC)[0]
    assert how == "handle" and centre[0] > 0.44                           # beside the head, as without the lip
    s = SPEC
    assert find_handle(flared, s.handle_min_m, s.handle_max_m, s.handle_min_len_m, s.handle_from_join_m,
                       s.finger_half_width_m) is None                     # the scoop's way: the lip is the join


def test_jaws_the_arm_would_open_along_the_handle_are_refused_before_anything_moves(monkeypatch):
    """
    2026-11-20 15:26: the UR5's jaws came down along the hammer's handle, on it,
    at 50 N (user: "there should be some basic common sense"). Asked of the
    rotation the arm will really take: here one turned a quarter off.
    """
    from mrcf.backends import ur5
    from mrcf.perception.plane_geometry import PlaneView
    from mrcf.policies import lying_pick as lp
    from mrcf.policies.grasp_sense import fingers_on_it

    env = ur5.make_env(host=None)
    pol = ur5.make_policies(lying_pick=True, handle_first=True)
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    T[:3, 3] = [0.35, 0.40, 1.0]
    model = {"K": np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]), "D": np.zeros(5), "T_world_cam": T,
             "plane_z_m": 0.0}
    monkeypatch.setattr(env, "camera_model", lambda camera, w, h: model)
    poly = hammer(x0=0.20, y=0.40)
    px = PlaneView.from_camera_model(model).project(np.c_[poly * 1000.0, np.full(len(poly), 14.0)])
    target = Target(np.array([0.46, 0.40, 0.0]), outlines={"overhead": {"size": [1280, 720], "px": px.tolist()}})
    real = lp.straight_down

    def a_quarter_off(env_, targets, turn):
        rot, why = real(env_, targets, turn)
        quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        return quarter @ rot, why
    monkeypatch.setattr(lp, "straight_down", a_quarter_off)
    start = env.get_ee_pos().copy()
    out = lp.lying_pick(pol, env, target)
    assert out.refused and "off square across it" in out.message
    assert np.allclose(env.get_ee_pos(), start)                             # nothing moved
    # A finger's footprint on the thing is refused too, square or not.
    centre, across = np.array([0.30, 0.40]), np.array([0.0, 1.0])
    assert fingers_on_it(poly, centre, across, across, 0.020, 0.009, 0.010) == ("a finger would come down on it, "
                                                                                "not beside it")
    assert fingers_on_it(poly, centre, across, across, 0.052, 0.009, 0.010) is None


def test_on_a_real_arm_a_pick_waits_until_its_jaws_way_of_opening_has_been_checked(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from mrcf import arm_facts

    cfg = SimpleNamespace(RIG_DIR=tmp_path, JAW_SLIDE_AXIS=(1.0, 0.0, 0.0))
    assert arm_facts.unchecked(cfg) == ("JAW_SLIDE_AXIS",)
    pol = ParallelJaw(SPEC)
    arm = FakeArm()
    names = arm_facts.unchecked(cfg)
    arm.unchecked_facts = lambda: names
    out = pol.pick(arm, Target(np.array([0.46, 0.30, 0.0]), outlines=OUTLINES))
    assert out.refused and "tools/check_jaw_direction.py" in out.message and arm.path == []
    arm_facts.record(cfg, "JAW_SLIDE_AXIS", how="watched")
    assert arm_facts.unchecked(cfg) == ()
    cfg.JAW_SLIDE_AXIS = (0.0, 1.0, 0.0)                                     # changed since: a guess again
    assert arm_facts.unchecked(cfg) == ("JAW_SLIDE_AXIS",)


def _held_hammer():
    """A 251 mm hammer as the lying pick grips it, by its handle next to its head: its outline in the jaws' frame."""
    poly = hammer(x0=0.20, y=0.30, handle_len=0.226, head_w=0.110, head_t=0.025)
    centre, across, _, _ = grips(poly, dataclasses.replace(SPEC, max_open_m=0.052))[0]
    square = np.array([-across[1], across[0]])
    return tuple((float((p - centre) @ across), float((p - centre) @ square)) for p in poly)


def test_on_the_ur5_a_hammer_is_laid_into_a_box_longer_than_it_not_refused_for_a_circle(monkeypatch):
    """
    2026-11-20 16:1x: held, the 251 mm hammer was refused — "nowhere 145 mm in
    from its walls" — in a box longer than it (user: "there is enough room in
    the box"; "it can angle it diagonally if it wants more space"). The box
    as it stood, its rim's corners from the overhead camera; the UR5 offline,
    its own orientation search (which takes no lean).
    """
    from mrcf.backends import ur5
    from mrcf.backends.ur5 import robot_config as rc
    from mrcf.policies import place_in as pi
    from mrcf.policies.base import Opening
    from mrcf.policies.parallel_jaw.spec import GraspPlan

    monkeypatch.setattr(rc, "GRIPPER_BOX", ((0.0, 0.90), (0.0, 0.75), (0.0, 0.80)))
    env = ur5.make_env(host=None)
    pol = ur5.make_policies(place_in=True)
    monkeypatch.setattr(pol, "holding", lambda e: True)
    monkeypatch.setattr(pi, "look_at_rim", lambda *a, **k: (None, "no wrist camera"))
    env.move_ee_to(np.array([0.45, 0.30, 0.20]))
    box = Opening(np.array([[0.596, 0.405], [0.607, 0.136], [0.829, 0.035], [0.822, 0.326]]), 0.30, False, "box")
    held = _held_hammer()
    plan = GraspPlan(height_m=0.028, width_m=0.251, grasp_height_m=0.016, grip_width_m=0.028, pregrasp_open_m=0.052,
                     hover_height_m=0.13, lift_height_m=0.13, release_height_m=0.026, held_outline=held)
    out = pi.place_in(pol, env, box, grasp=plan)
    assert out.ok, out.message
    jaw = env.jaw_direction(env.get_ee_rotation())
    square = np.array([-jaw[1], jaw[0]])
    world = [out.at_xy + u * jaw + v * square for u, v in held]
    assert min(pi.inside_by(box.outline_xy, p) for p in world) >= pol.spec.container_outline_clearance_m - 0.003
    # As before, with no outline of what is held: a circle as wide as it is long, and refused.
    env.move_ee_to(np.array([0.45, 0.30, 0.20]))
    out = pi.place_in(pol, env, box, grasp=dataclasses.replace(plan, held_outline=None))
    assert out.refused and "in from its walls" in out.message


def test_what_the_ur5_carries_is_part_of_its_shape_for_the_sentry_until_let_go():
    """2026-11-20 16:24: a carried hammer paused the arm every few cm; 16:31 its shape growing crashed the sentry."""
    from mrcf.backends import ur5
    from mrcf.motion_guard import arm_mask, arm_moved

    env = ur5.make_env(host=None)
    bare = env.arm_shape()
    env.carrying(_held_hammer(), 0.024)
    held = env.arm_shape()
    assert len(held) > len(bare)
    shapes = [(0.0, np.array([p for p, _ in bare]), np.array([r for _, r in bare])),
              (0.1, np.array([p for p, _ in held]), np.array([r for _, r in held]))]
    model = {"K": np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]), "D": np.zeros(5),
             "T_world_cam": np.array([[1, 0, 0, 0.35], [0, -1, 0, 0.40], [0, 0, -1, 1.0], [0, 0, 0, 1.0]])}
    assert arm_mask(shapes, model, (720, 1280), 0.05).any()                   # no crash as the shape grows
    assert not arm_moved(shapes[1:] + shapes[1:], 0.01)
    env.set_gripper(0.05)                                                     # let go: no longer carried
    assert len(env.arm_shape()) == len(bare)


def test_an_open_boxs_opening_is_its_shaded_inside_not_its_flaps_folded_out():
    """2026-11-20 16:47: the UR5's box outlined with its four flaps out; a screwdriver let go over a flap."""
    from mrcf.containers import open_top

    img = np.full((400, 400, 3), 200, np.uint8)                   # the table
    box = np.zeros((400, 400), bool)
    box[100:300, 40:360] = True                                   # flaps out left and right ...
    box[40:360, 120:280] = True                                   # ... and front and back: a cross
    img[box] = (190, 150, 100)                                    # cardboard in the light
    img[100:300, 120:280] = (95, 70, 45)                          # the inside, in its own shade
    got = open_top(img, box)
    assert got is not None
    assert got[200, 200] and not got[200, 60] and not got[60, 200]      # the middle, not a flap
    assert abs(int(got.sum()) - 200 * 160) < 0.1 * 200 * 160
    assert open_top(np.full((400, 400, 3), 150, np.uint8), box) is None  # no shade to go by: the outline itself


def test_a_hammer_whose_head_lies_at_the_boxs_edge_is_gripped_further_along_its_handle(monkeypatch):
    """2026-11-20 17:01: the hammer's head 4 cm from the gripper box's x = 0 edge; "did not get over it", handle unused."""
    from mrcf.backends import ur5
    from mrcf.perception.plane_geometry import PlaneView

    env = ur5.make_env(host=None)
    pol = ur5.make_policies(lying_pick=True, handle_first=True)
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    T[:3, 3] = [0.20, 0.45, 1.0]
    model = {"K": np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]), "D": np.zeros(5), "T_world_cam": T,
             "plane_z_m": 0.0}
    monkeypatch.setattr(env, "camera_model", lambda camera, w, h: model)
    head_near_edge = np.array([0.497, 0.0]) + hammer(x0=0.20, y=0.45) * np.array([-1.0, 1.0])     # head at x -0.01-0.02
    px = PlaneView.from_camera_model(model).project(np.c_[head_near_edge * 1000.0, np.full(len(head_near_edge), 14.0)])
    target = Target(np.array([0.15, 0.45, 0.0]), outlines={"overhead": {"size": [1280, 720], "px": px.tolist()}})
    closes = iter([(0.0, False), (0.024, True), (0.024, True)])
    monkeypatch.setattr(env, "squeeze_gripper", lambda **k: next(closes))
    monkeypatch.setattr(pol, "holding", lambda e: True)
    monkeypatch.setattr(pol, "_can_overhead", lambda e: False)
    env.move_ee_to(np.array([0.30, 0.40, 0.25]))                          # inside its box, as from home
    out = pol.pick(env, target)
    assert out.ok, out.message
    assert out.at_xy[0] > 0.06                                             # not at the head, against the edge


def test_a_tool_goes_into_a_box_that_already_has_tools_in_it(monkeypatch):
    """2026-11-20 17:34: a hammer and a screwdriver in the box; the second screwdriver refused, every spot kept clear."""
    from mrcf.backends import ur5
    from mrcf.backends.ur5 import robot_config as rc
    from mrcf.policies import place_in as pi
    from mrcf.policies.base import Obstacle, Opening
    from mrcf.policies.parallel_jaw.spec import GraspPlan

    monkeypatch.setattr(rc, "GRIPPER_BOX", ((0.0, 0.90), (0.0, 0.75), (0.0, 0.80)))
    env = ur5.make_env(host=None)
    pol = ur5.make_policies(place_in=True, place_in_over_contents=True)
    monkeypatch.setattr(pol, "holding", lambda e: True)
    env.move_ee_to(np.array([0.35, 0.40, 0.25]))
    box = Opening(np.array([[0.49, 0.17], [0.69, 0.17], [0.69, 0.40], [0.49, 0.40]]), 0.26, True, "box")
    sd = screwdriver(x0=0.10, y=0.40, shaft=0.05, handle=0.10)
    c, acr = np.array([0.18, 0.40]), np.array([0.0, 1.0])
    held = tuple(((p - c) @ acr, (p - c) @ np.array([-1.0, 0.0])) for p in sd)
    plan = GraspPlan(height_m=0.02, width_m=0.15, grasp_height_m=0.016, grip_width_m=0.02, pregrasp_open_m=0.044,
                     hover_height_m=0.1, lift_height_m=0.1, release_height_m=0.026, held_outline=held)
    already = [Obstacle(position=np.array([x, y]), radius_m=0.04, label="a tool in the box")
               for x in (0.53, 0.59, 0.65) for y in (0.22, 0.28, 0.34)]
    out = pi.place_in(pol, env, box, grasp=plan, obstacles=already)
    assert out.ok, out.message
