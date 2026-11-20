"""How the two-finger policies come at an object by where it stands: near_base, top_down, reach_out."""

import dataclasses
import threading

import numpy as np
import pytest

from fake_arm import Cylinder, FakeArm
from mrcf.policies.base import Target, Vision
from mrcf.policies.parallel_jaw import NEAR_BASE, REACH_OUT, TOP_DOWN, GripperSpec, ParallelJaw

# The SO-101's gripper, with this rig's zones and base, the wrist servo set at
# 35 mm and -5 deg from a 70 mm hover — but radial jaws, where a lean tips them
# relative to the object (the rig's open across the arm: JAW_AXIS).
SPEC = GripperSpec(max_open_m=0.045, tip_below_grasp_m=0.007, free_depth_m=0.055,
                   finger_thickness_m=0.009, finger_half_width_m=0.008, jaw_axis="radial",
                   hover_height_m=0.06, lift_height_m=0.07, lift_clearance_m=0.05,
                   camera="wrist", servo_height_m=0.07, servo_open_m=0.035, servo_tilt_deg=-5.0,
                   near_base_m=0.15, far_reach_m=0.29, base_keep_out_m=0.06)

CEILING_R_M, CEILING_Z_M = 0.13, 0.065     # nearer the base than this, nothing higher reaches


def on_table(xy, z=0.0):
    return np.array([xy[0], xy[1], z])


def _lean(rot) -> float:
    z = np.asarray(rot, float)[:, 2]
    return float(np.degrees(np.arctan2(z[0], -z[2])))


class ReachArm(FakeArm):
    """
    A FakeArm with this rig's reach, roughly, along +x from a base at the
    origin: close in, no point above CEILING_Z_M is reachable; beyond 0.28 m
    the gripper must lean forward 5 deg per 10 mm, and no more than 60. A
    joint margin of 3 deg or more costs another 5 deg of lean. A fixed camera
    sees the gripper exactly where it is (visible), and a wrist servo is
    calibrated, aligning wherever it is asked to.
    """
    wrist_servo_target_px = (380.0, 260.0)

    def __init__(self, objects=(), visible=True):
        super().__init__(objects)
        self.visible = visible
        self.servo_heights = []
        self.asked = []                 # every orientation request: (targets, margin_deg)

    def approach_orientation_for(self, targets, tols=None, flip=False, prefer_tilt_deg=None, margin_deg=None):
        self.asked.append(([np.asarray(t, float).copy() for t in targets], margin_deg))
        need = 0.0
        for t in targets:
            r = float(np.linalg.norm(np.asarray(t[:2], float)))
            if r < CEILING_R_M and t[2] > CEILING_Z_M:
                return None
            need = max(need, 5.0 * np.ceil(max(0.0, r - 0.28) * 100))
        if margin_deg is not None and margin_deg >= 3:
            need += 5.0
        if need > 60.0:
            return None
        a = np.radians(need)
        x = np.array([np.cos(a), 0.0, np.sin(a)]) * (-1 if flip else 1)
        z = np.array([np.sin(a), 0.0, -np.cos(a)])
        return np.column_stack([x, np.cross(z, x), z]), -need

    def gripper_tilt_deg(self, rot=None):
        return -_lean(rot)

    def overhead_gripper_offset(self):
        return np.zeros(3) if self.visible else None

    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        self.servo_heights.append(round(float(self.ee[2]), 3))
        return True, "aligned"


def _finds_it():
    return Vision(find=lambda frame, expected_px, radius_px: expected_px)


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


# ── Which approach where ───────────────────────────────────────────────────────

def test_a_spot_straight_down_would_need_a_lean_for_is_the_leaning_approachs():
    """2026-11-19: a plug at 32 cm, "mid-reach" by chess's distance, went top_down and stalled three times."""
    arm = ReachArm()
    plan = ParallelJaw(SPEC).plan(Target(on_table((0.30, 0.0)), foot_m=0.05, height_m=0.08))
    by_distance = ParallelJaw(dataclasses.replace(SPEC, far_reach_m=0.32))
    assert by_distance.approach_order(arm, (0.30, 0.0), plan)[0].name == "top_down"      # 10 deg: mid all the same
    by_lean = ParallelJaw(dataclasses.replace(SPEC, far_reach_m=0.32, far_from_lean_deg=5.0))
    assert by_lean.approach_order(arm, (0.30, 0.0), plan)[0].name == "reach_out"
    assert by_lean.approach_order(arm, (0.30, 0.0))[0].name == "top_down"               # no plan: by distance
    assert by_lean.approach_order(arm, (0.25, 0.0), plan)[0].name == "top_down"         # straight down reaches


class _BelowArm(ReachArm):
    """ReachArm, its jaw tips 10 mm under the grasp point leaning, 7 straight down."""
    def below_grasp_m(self, rot, gap_m):
        return 0.010 if abs(_lean(rot)) >= 5 else 0.007


def test_a_grasp_held_leaning_is_raised_clear_of_the_table():
    """2026-11-19: a plug 30 cm out gripped 18 mm up leaning 20-30 deg; the floor guard stopped every descent."""
    from types import SimpleNamespace

    arm = _BelowArm()
    leaning, _ = arm.approach_orientation_for([np.array([0.33, 0.0, 0.02])])
    upright, _ = arm.approach_orientation_for([np.array([0.22, 0.0, 0.02])])

    from mrcf.policies.base import Diagnosis

    def grasp_z(spec, rot, tilt, height=0.055, diagnosis=None):
        ctx = SimpleNamespace(rot=rot, tilt=tilt, tag="", grasp=np.array([0.33, 0.0, SPEC.table_z + 0.018]),
                              gp=SimpleNamespace(pregrasp_open_m=0.06, grip_width_m=0.053, height_m=height),
                              diagnosis=diagnosis)
        ParallelJaw(spec)._clear_floor(arm, ctx)
        return float(ctx.grasp[2]) - SPEC.table_z

    raised = dataclasses.replace(SPEC, lean_clears_floor=True, lean_floor_clear_m=0.016)
    assert grasp_z(SPEC, leaning, -25.0) == pytest.approx(0.018)                 # off: as planned
    assert grasp_z(raised, leaning, -30.0) == pytest.approx(0.010 + 0.016)       # the tips' reach, and room
    assert grasp_z(raised, leaning, -20.0) == pytest.approx(0.010 + 0.010)       # half the lean: half the room
    assert grasp_z(raised, upright, 0.0) == pytest.approx(0.018)                 # straight down: as planned
    assert grasp_z(raised, leaning, -30.0, height=0.030) == pytest.approx(0.018) # too short to grip that high
    # 2026-11-19 12:06: lowered after a miss above it, then raised again for the lean — and missed again.
    assert grasp_z(raised, leaning, -30.0, diagnosis=Diagnosis("too_high", "")) == pytest.approx(0.018)


def test_each_zone_starts_with_its_own_approach_and_keeps_the_others_as_fallbacks():
    pol, arm = ParallelJaw(SPEC), ReachArm()
    # Near the base top_down still goes first: near_base is for where it cannot reach.
    assert [a.name for a in pol.approach_order(arm, (0.12, 0.0))] == ["top_down", "near_base", "reach_out"]
    assert [a.name for a in pol.approach_order(arm, (0.22, 0.0))] == ["top_down", "near_base", "reach_out"]
    assert [a.name for a in pol.approach_order(arm, (0.33, 0.0))] == ["reach_out", "top_down", "near_base"]
    # Without zones, everywhere is the middle.
    plain = ParallelJaw(dataclasses.replace(SPEC, near_base_m=None, far_reach_m=None))
    assert plain.zone(arm, (0.05, 0.0)) == plain.zone(arm, (0.40, 0.0)) == "mid"


def test_the_zone_is_measured_from_where_the_arm_stands():
    class Offset(ReachArm):
        base_xy = np.array([0.04, 0.0])
    pol = ParallelJaw(SPEC)
    assert pol.zone(ReachArm(), (0.16, 0.0)) == "mid"
    assert pol.zone(Offset(), (0.16, 0.0)) == "near"       # 0.12 m from its axis


def test_nothing_that_reaches_is_refused_before_moving():
    # A king's hover (~0.1 m) over a square next to the base: out of reach every way.
    king = Cylinder((0.10, 0.0), 0.030, 0.090)
    arm, pol = ReachArm([king]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(king.xy), foot_m=0.030, height_m=0.090), vision=_finds_it())
    assert not got.ok and got.refused
    assert arm.cycles == 0
    assert all(name in got.message for name in ("near_base", "top_down", "reach_out"))


# ── Near the base ──────────────────────────────────────────────────────────────

def test_near_the_base_the_servo_aligns_wherever_top_down_reaches():
    # 0.14 m is "near", but its 70 mm servo hover and lift reach (the ceiling is at 0.13).
    # No fixed camera sees the gripper here, so the servo aligns on the first attempt.
    pawn = Cylinder((0.14, 0.0), 0.022, 0.045)
    arm, pol = ReachArm([pawn], visible=False), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok, got.message
    assert got.approach == "top_down" and arm.servo_heights == [0.07]
    assert any("Approach: top_down (close to the arm's base)" in n for n in got.notes)


def test_near_the_base_a_missed_close_is_retried_by_an_approach_that_looks_again():
    # 2026-10-21, d2: near_base's retry came down blind on the spot the first
    # close had pushed the pawn off. After a miss, only one that looks.
    arm, pol = ReachArm(), ParallelJaw(SPEC)                  # nothing there: every close misses
    got = pol.pick(arm, Target(on_table((0.14, 0.0)), foot_m=0.021, height_m=0.045), vision=_finds_it())
    assert not got.ok
    assert [a["approach"] for a in got.attempts] == ["top_down", "top_down", "top_down"]
    assert any("which looks for it with the wrist camera" in n for n in got.notes)


class _Pushes(ReachArm):
    """
    The rig's close: the first one misses and pushes the piece `push` along x,
    toward the base. Its wrist servo aligns over the piece wherever it is now,
    if it is within `sees` of where it looks — else it does not find it.
    """
    push, sees = -0.015, 0.03

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.closes = 0

    def squeeze_gripper(self):
        self.closes += 1
        if self.closes == 1:
            self.objects[0].xy = self.objects[0].xy + np.array([self.push, 0.0])
            self.width = 0.0
            return 0.0, False
        return super().squeeze_gripper()

    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        self.servo_heights.append(round(float(self.ee[2]), 3))
        piece = self.objects[0].xy
        if np.linalg.norm(piece - self.ee[:2]) > self.sees:
            return False, "target lost while acquiring (not found)"
        self.move_ee_to(np.array([piece[0], piece[1], self.ee[2]]))
        return True, "aligned"


def test_a_piece_found_again_after_a_miss_is_come_at_the_same_way_again():
    # 2026-10-31, d3: the retry after a miss turned to near_base, which the
    # grasp correction is not measured for, and missed at the fresh position.
    pawn = Cylinder((0.20, 0.0), 0.022, 0.045)
    arm, pol = _Pushes([pawn]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.022, height_m=0.045),
                   vision=Vision(find=lambda f, e, r: e, relocate=lambda env, near_xy: pawn.xy.copy()))
    assert got.ok and arm.held is pawn, got.message
    assert [(a["approach"], a["result"]) for a in got.attempts] == [("top_down", "closed_on_nothing"),
                                                                     ("top_down", "ok")]
    assert any("Again by top_down: it reached it; the aim was off." in n for n in got.notes)


def test_after_a_miss_the_wrist_camera_finds_where_the_piece_went_and_the_pick_goes_there():
    pawn = Cylinder((0.20, 0.0), 0.022, 0.045)
    arm, pol = _Pushes([pawn]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok and arm.held is pawn, got.message
    assert [a["result"] for a in got.attempts] == ["closed_on_nothing", "ok"]
    assert not any("descending at the scan's estimate" in n for n in got.notes[got.notes.index(
        next(n for n in got.notes if n.startswith("[Pick 2/3]"))):])


def test_after_a_miss_a_piece_the_wrist_camera_cannot_find_is_not_looked_for_where_it_was():
    pawn = Cylinder((0.20, 0.0), 0.022, 0.045)
    arm, pol = _Pushes([pawn]), ParallelJaw(SPEC)
    arm.push, arm.sees = -0.04, 0.03
    got = pol.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert not got.ok
    assert [a["result"] for a in got.attempts] == ["closed_on_nothing", "lost_after_miss", "lost_after_miss"]
    assert arm.closes == 1                                   # never came down on the empty spot again


def test_after_a_miss_where_nothing_can_look_the_pick_stops():
    # Next to the base only near_base reaches, and it does not use the wrist camera.
    pawn = Cylinder((0.12, 0.0), 0.022, 0.045)
    arm, pol = _Pushes([pawn]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table((0.12, 0.0)), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert not got.ok and len(got.attempts) == 1 and arm.closes == 1
    assert any("not coming down where it was. Scan again" in n for n in got.notes)


class _ServoFindsIt(ReachArm):
    """A wrist servo that aligns over the piece wherever it stands."""

    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        self.servo_heights.append(round(float(self.ee[2]), 3))
        self.move_ee_to(np.array([*self.objects[0].xy, self.ee[2]]))
        return True, "aligned"


def test_a_first_attempt_aligned_by_the_wrist_camera_too_goes_where_the_piece_stands():
    # 2026-10-30: a pawn the arm had put down stood 9-15 mm from where it was
    # put. The overhead camera put the jaws over that spot, and the close
    # found nothing: it checks the jaws, not the piece under them.
    def pick(overhead_first, off_m):
        pawn = Cylinder((0.20 + off_m, 0.0), 0.021, 0.045)
        arm = _ServoFindsIt([pawn])
        pol = ParallelJaw(dataclasses.replace(SPEC, overhead_first=overhead_first, servo_agree_m=0.008))
        got = pol.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), vision=_finds_it())
        return got, arm.held is pawn

    got, held = pick(True, 0.011)
    assert got.attempts[0]["result"] == "closed_on_nothing"
    got, held = pick(False, 0.011)
    assert held and [a["result"] for a in got.attempts] == ["ok"]
    # A few mm apart, the scan's estimate wins: the fixed camera checks the
    # jaws against it, where the servo can hold a bias of its own.
    got, held = pick(False, 0.003)
    assert held and any("descending at the scan's estimate" in n for n in got.notes)


def test_near_base_hovers_only_as_high_as_the_object_needs():
    pol = ParallelJaw(SPEC)
    plan = pol.plan_from_shape(0.041, {}, 0.021)
    assert plan.hover_height_m == pytest.approx(0.06)                    # the floor
    low = pol._own_hover(NEAR_BASE, plan)
    assert low.hover_height_m == pytest.approx(0.041 + 0.007 + 0.008)     # top, tips, clearance
    assert pol._own_hover(TOP_DOWN, plan) is plan


def test_near_the_base_it_comes_in_low_aligned_by_the_fixed_camera_and_lifts_as_high_as_reaches():
    pawn = Cylinder((0.12, 0.0), 0.022, 0.045)
    arm, pol = ReachArm([pawn]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok, got.message
    assert got.approach == "near_base" and arm.held is pawn
    # No climb to the servo's 70 mm, which is out of reach here; the fixed camera aligned.
    assert arm.servo_heights == []
    assert any("overhead camera" in n for n in got.notes)
    # Up no higher than reaches straight above it: the hover's 60 mm, not the plan's 70.
    grasped = int(np.argmin([p[2] for p in arm.path]))
    assert max(p[2] for p in arm.path[grasped:]) <= CEILING_Z_M


def test_near_the_base_it_does_not_come_down_without_the_fixed_cameras_sighting():
    pawn = Cylinder((0.12, 0.0), 0.022, 0.045)
    arm, pol = ReachArm([pawn], visible=False), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert not got.ok and arm.held is None
    assert min(p[2] for p in arm.path) > 0.05                 # never went down to grasp
    # top_down cannot reach here (its climb to the servo's height); the others, low and
    # aligned by the fixed camera, each stopped for want of its sighting.
    assert [a["approach"] for a in got.attempts] == ["near_base", "reach_out", "near_base"]
    assert {a["result"] for a in got.attempts} == {"blind"}


def test_near_the_base_told_to_risk_it_it_comes_down_at_the_scans_position_unsighted():
    pawn = Cylinder((0.12, 0.0), 0.022, 0.045)
    arm = ReachArm([pawn], visible=False)
    got = ParallelJaw(dataclasses.replace(SPEC, descend_unconfirmed=True)).pick(
        arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok and arm.held is pawn, got.message
    assert [a["approach"] for a in got.attempts] == ["near_base"]
    assert any("descending at the scan's position anyway" in n for n in got.notes)


def test_the_arms_base_keeps_the_fingers_and_the_wrist_camera_off_it():
    pol, arm = ParallelJaw(dataclasses.replace(SPEC, camera_body_m=(-0.075, 0.0, 0.02))), ReachArm()
    base = pol._with_base(arm, [])
    room, limit = pol.jaw_room(arm, (0.09, 0.0), base)
    assert limit == "the arm's own base"
    assert room == pytest.approx(2 * (0.09 - 0.06 - 0.009 - 0.003))
    # Unturned, the camera hangs 75 mm toward the base from 0.10 m: inside it. Turned, outward.
    at = np.array([0.10, 0.0, 0.06])
    rot = pol.hold_orientation(arm, [("hover", at, 0.01)], obstacles=base)
    assert rot[0, 0] < 0
    assert pol.body_clearance(arm, rot, at, base)[0] > 0.09     # 175 mm out, less the base and the camera


# ── In the middle ──────────────────────────────────────────────────────────────

def test_in_the_middle_it_is_the_pick_as_it_was_aligned_by_the_servo_from_its_height():
    pawn = Cylinder((0.22, 0.0), 0.022, 0.045)
    arm, pol = ReachArm([pawn], visible=False), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok, got.message
    assert got.approach == "top_down" and arm.servo_heights == [0.07]
    assert got.grasp.grasp_height_m == pytest.approx(ParallelJaw(SPEC).plan_from_shape(0.045, {}, 0.022).grasp_height_m)


def test_a_retry_tries_the_next_approach_then_the_zones_own_again():
    # Nothing there: every attempt closes on nothing. (Retried blind, for the retries' own sake:
    # by default a miss with nothing to look again stops the pick — test_ur5.)
    arm, pol = FakeArm(), ParallelJaw(dataclasses.replace(SPEC, blind_retry_after_miss=True))
    got = pol.pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.021, height_m=0.045))
    assert not got.ok
    assert [a["approach"] for a in got.attempts] == ["top_down", "near_base", "top_down"]
    assert {a["result"] for a in got.attempts} == {"closed_on_nothing"}
    assert any("This attempt: near_base." in n for n in got.notes)


# ── Far out ────────────────────────────────────────────────────────────────────

def test_far_out_the_fixed_camera_aligns_when_the_lean_is_far_from_the_servos():
    pawn = Cylinder((0.33, 0.0), 0.022, 0.045)
    arm, pol = ReachArm([pawn]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok, got.message
    assert got.approach == "reach_out"
    assert arm.servo_heights == []                       # far from the -5 deg the servo was set at
    # 35 to reach (the grasp 15 mm further out included), 5 more for the joint margin it asked for.
    assert got.attempts[0]["tilt_deg"] == pytest.approx(-40.0)
    assert any(m == 3.0 for _, m in arm.asked)
    assert any("leaves every joint 3 deg" in n for n in got.notes)


def test_far_out_but_near_the_servos_lean_the_servo_still_aligns():
    pawn = Cylinder((0.27, 0.0), 0.022, 0.045)           # 10 deg, the margin's 5 included
    arm, pol = ReachArm([pawn], visible=False), ParallelJaw(dataclasses.replace(SPEC, far_reach_m=0.25))
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok and got.approach == "reach_out"
    assert arm.servo_heights == [0.07]


def test_reach_out_grasps_higher_so_the_tipped_jaws_clear_the_top():
    pol = ParallelJaw(SPEC)
    widths = {0: 22, 10: 18, 20: 16, 30: 13, 38: 12, 45: 8}
    plan = pol._widened(pol.plan_from_shape(0.045, widths, 0.022))
    assert pol.top_overlap(plan, widths, -45.0) > 0       # grasped low, the head is past the outer jaw
    high = pol._top_clear(plan, widths, -45.0)
    assert high.grasp_height_m > plan.grasp_height_m
    assert pol.top_overlap(high, widths, -45.0) <= 0
    assert high.release_height_m == pytest.approx(high.grasp_height_m + SPEC.release_clearance_m)
    # A small lean needs nothing; tangential jaws lean along their pads.
    assert pol._top_clear(plan, widths, -5.0) is plan
    tangential = ParallelJaw(dataclasses.replace(SPEC, jaw_axis="tangential"))
    assert tangential._top_clear(plan, widths, -45.0) is plan


def test_a_far_pick_grasps_where_its_lean_needs(monkeypatch):
    pawn = Cylinder((0.37, 0.0), 0.022, 0.045)
    arm, pol = ReachArm([pawn]), ParallelJaw(SPEC)
    widths = {0: 22, 10: 18, 20: 16, 30: 13, 38: 12, 45: 8}
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045, widths_mm=widths),
                   vision=_finds_it())
    assert got.ok, got.message
    assert got.approach == "reach_out"
    assert got.grasp.grasp_height_m > pol.plan_from_shape(0.045, widths, 0.022).grasp_height_m
    assert pol.top_overlap(got.grasp, widths, got.attempts[0]["tilt_deg"]) <= 0


def test_without_a_servo_opening_the_plan_keeps_its_own():
    pol = ParallelJaw(dataclasses.replace(SPEC, servo_open_m=None))
    plan = pol.plan_from_shape(0.045, {}, 0.022)
    assert pol._widened(plan) is plan
    # With one, opened to it — within the room the neighbours leave.
    pol = ParallelJaw(SPEC)
    assert pol._widened(plan).pregrasp_open_m == pytest.approx(0.035)
    assert pol._widened(plan, room=0.031).pregrasp_open_m == pytest.approx(0.031)


# ── Placing ────────────────────────────────────────────────────────────────────

def _holding(arm, plan):
    piece = Cylinder(arm.ee[:2], 0.022, 0.045)
    arm.objects.append(piece)
    arm.held = piece
    return piece


def test_near_the_base_a_place_carries_as_high_as_reaches():
    arm, pol = ReachArm(), ParallelJaw(SPEC)
    plan = pol.plan_from_shape(0.045, {}, 0.022)
    piece = _holding(arm, plan)
    put = pol.place(arm, on_table((0.12, 0.0)), grasp=plan)
    assert put.ok, put.message
    assert put.approach == "near_base"
    assert np.allclose(piece.xy, (0.12, 0.0), atol=0.002)


def test_a_place_nothing_reaches_is_refused_still_holding():
    arm, pol = ReachArm(), ParallelJaw(SPEC)
    plan = pol.plan_from_shape(0.045, {}, 0.022)
    piece = _holding(arm, plan)
    put = pol.place(arm, on_table((0.45, 0.0)), grasp=plan)
    assert not put.ok and put.refused
    assert arm.held is piece and arm.cycles == 0


# ── The backend's side: a joint margin ─────────────────────────────────────────

def test_the_orientation_probe_can_insist_on_joint_margin():
    from mrcf.backends.so101.kinematics import JOINT_NAMES
    from mrcf.backends.so101.env import SO101Env

    class Point:
        """The first three joints, in degrees, are the grasp point's x, y, z in mm."""
        joint_names = JOINT_NAMES + ["gripper"]

        def forward_kinematics(self, q):
            t = np.eye(4)
            t[:3, 3] = np.asarray(q, float)[:3] * 1e-3
            return t

        def inverse_kinematics(self, current, desired, **_):
            q = np.asarray(current, float).copy()
            q[:3] = np.asarray(desired, float)[:3, 3] * 1e3
            return q

    env = SO101Env.__new__(SO101Env)
    env._bus_lock = threading.RLock()
    env._kinematics = Point()
    env._tcp = lambda gap_m=None: np.zeros(3)
    env._ik_limits = {"shoulder_pan": (-300.0, 203.0)}       # x within 203 mm
    target, seed = np.array([0.200, 0.0, 0.05]), np.zeros(6)
    assert env._probe(np.eye(3), target, np.zeros(3), 0.0, seed, 0.001) is not None            # 1 deg default
    assert env._probe(np.eye(3), target, np.zeros(3), 0.0, seed, 0.001, margin_deg=5.0) is None


def test_the_shipped_approaches_are_what_the_reach_study_found():
    # near the base: no servo (its height is out of reach), low lift, margin toward the base;
    # far out: the servo only near its own lean, a joint margin, the top kept clear.
    assert (NEAR_BASE.servo, NEAR_BASE.lift, NEAR_BASE.reach_margin) == (False, "highest", "in")
    assert (TOP_DOWN.servo, TOP_DOWN.lift, TOP_DOWN.keep_top_clear) == (True, "full", False)
    assert REACH_OUT.keep_top_clear and REACH_OUT.lean_margin_deg and REACH_OUT.servo_max_lean_change_deg


# ── After a miss ───────────────────────────────────────────────────────────────

class _Recording(ReachArm):
    """A ReachArm that notes the height of the jaws whenever they open."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.opened_at = []

    def set_gripper(self, width, n_steps=15, max_step_pct=None, **kw):
        if width > self.width:
            self.opened_at.append(round(float(self.ee[2]), 3))
        super().set_gripper(width, n_steps, max_step_pct, **kw)


def test_a_missed_close_backs_straight_up_before_the_jaws_reopen_or_the_arm_moves_on():
    # 2026-10-21, d2: three missed closes, each followed by reopening at the
    # pawn's height and a move to the next attempt's hover, left the pawn
    # 19.5 mm nearer the base; the failed pick left the jaws down beside it.
    arm, pol = _Recording(), ParallelJaw(SPEC)                 # nothing there: every close misses
    got = pol.pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.021, height_m=0.045), vision=_finds_it())
    assert not got.ok and len(got.attempts) == 3
    hover = got.grasp.hover_height_m
    assert min(arm.opened_at) >= hover - SPEC.travel_tol_m           # never opened down by the object
    lowest = int(np.argmin([p[2] for p in arm.path]))
    up = next(i for i in range(lowest, len(arm.path)) if arm.path[i][2] >= hover - SPEC.travel_tol_m)
    assert all(np.allclose(p[:2], arm.path[lowest][:2]) for p in arm.path[lowest:up + 1])   # straight up
    assert arm.ee[2] >= hover - SPEC.travel_tol_m                    # and left clear of it
    assert sum("Backing straight up" in n for n in got.notes) == 3


class _Drifting(ReachArm):
    """
    Over the object at the hover; once down, the jaws really are `drift`
    further out than the kinematics say, and the fixed camera sees it.
    """
    drift = 0.004

    def _off(self):
        return np.array([self.drift, 0.0, 0.0]) if self.ee[2] < 0.04 else np.zeros(3)

    def overhead_gripper_offset(self):
        return self._off()

    def _between_jaws(self):
        real = self.ee[:2] + self._off()[:2]
        return next((o for o in self.objects
                     if np.linalg.norm(o.xy - real) < 0.003 and self.ee[2] < o.height_m), None)


def test_the_fixed_camera_puts_the_jaws_on_their_aim_before_they_close():
    # 2026-10-21, d2: 4-6 mm left over after the hover's correction and the
    # descent, and the closing jaw cammed the pawn out toward the base.
    pawn = Cylinder((0.22, 0.0), 0.022, 0.045)
    arm, pol = _Drifting([pawn]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok, got.message
    seen = got.attempts[0]["overhead_at_grasp"]
    assert seen["drift_mm"] == pytest.approx([4.0, 0.0])
    assert seen["from_scan_mm"] == seen["aim_before_mm"] == pytest.approx([-4.0, 0.0])
    assert seen["aim_after_mm"] == pytest.approx([0.0, 0.0])
    assert any("the descent moved them 4.0 mm off the kinematics (+4.0 along" in n for n in got.notes)
    assert any("moved the jaws 4.0 mm onto their aim" in n for n in got.notes)


def test_jaws_far_from_their_aim_at_the_grasp_are_not_dragged_round_the_object():
    arm, pol = _Drifting([Cylinder((0.22, 0.0), 0.022, 0.045)]), ParallelJaw(SPEC)
    arm.drift = 0.012
    got = pol.pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.022, height_m=0.045), vision=_finds_it())
    seen = got.attempts[0]["overhead_at_grasp"]
    assert seen["aim_before_mm"] == seen["aim_after_mm"] == pytest.approx([-12.0, 0.0])
    assert any("more than 8 mm round the object; closing where they are" in n for n in got.notes)


class _ServoOff(ReachArm):
    """A wrist servo that 'aligns' `servo_off` along x from where it was asked to look."""
    servo_off = 0.005

    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        self.servo_heights.append(round(float(self.ee[2]), 3))
        self.move_ee_to(self.ee + np.array([self.servo_off, 0.0, 0.0]))
        return True, "aligned"


def test_a_servo_a_few_mm_from_a_scan_the_fixed_camera_checks_is_overruled():
    # 2026-10-21, d2: the servo moved the jaws 4-8 mm outward of the pawn, the
    # scan (and a person looking) had it right, and the close pushed it away.
    pawn = Cylinder((0.22, 0.0), 0.022, 0.045)
    arm, pol = _ServoOff([pawn]), ParallelJaw(dataclasses.replace(SPEC, overhead_first=False))
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok and arm.held is pawn and len(got.attempts) == 1, got.message
    assert any("descending at the scan's estimate" in n for n in got.notes)


def test_with_the_fixed_camera_over_it_the_first_attempt_does_not_wait_for_the_servo():
    # 2026-10-22: the overhead camera, calibrated, aimed every pick that gripped; the
    # servo was 4-17 mm off, and twice took the jaws 14-17 mm to an empty spot.
    pawn = Cylinder((0.22, 0.0), 0.022, 0.045)
    arm, pol = _ServoOff([pawn]), ParallelJaw(SPEC)
    arm.servo_off = 0.015
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok and arm.held is pawn and len(got.attempts) == 1, got.message
    assert arm.servo_heights == []
    assert any("the wrist camera looks again only after a miss" in n for n in got.notes)


def test_a_servo_that_finds_the_object_well_away_from_the_scan_is_followed():
    pawn = Cylinder((0.235, 0.0), 0.022, 0.045)                 # 15 mm from where the scan saw it
    arm, pol = _ServoOff([pawn]), ParallelJaw(SPEC)
    arm.servo_off = 0.015
    got = pol.pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok and arm.held is pawn, got.message
    assert not any("descending at the scan's estimate" in n for n in got.notes)


class _ServoStopsShort(ReachArm):
    """A wrist servo that leads the jaws 90 % of the way to `sees` and gives up there, unless already over it."""

    def __init__(self, objects, sees):
        super().__init__(objects)
        self.sees = np.asarray(sees, float)

    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        self.servo_heights.append(round(float(self.ee[2]), 3))
        gap = self.sees - self.ee[:2]
        if np.linalg.norm(gap) < 0.002:
            return True, "aligned"
        self.move_ee_to(np.r_[self.ee[:2] + 0.9 * gap, self.ee[2]])
        return False, "still 15.6 px off after 7 corrections"


def _relocating(found, asked):
    return Vision(find=lambda f, e, r: e, relocate=lambda env, near_xy: (asked.append(1), found())[1])


def test_a_servo_that_leads_well_away_and_cannot_settle_has_the_object_found_again_first():
    # 2026-10-31, d3: a pawn stood ~10 mm further out than the arm had put it;
    # the servo led the jaws 13-15 mm towards it and gave up, and the close at
    # the remembered spot missed.
    pawn = Cylinder((0.235, 0.0), 0.022, 0.045)
    arm, asked = _ServoStopsShort([pawn], sees=pawn.xy), []
    pol = ParallelJaw(dataclasses.replace(SPEC, overhead_first=False))
    got = pol.pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.022, height_m=0.045),
                   vision=_relocating(lambda: pawn.xy.copy(), asked))
    assert got.ok and arm.held is pawn, got.message
    assert [(a["approach"], a["result"]) for a in got.attempts] == [("top_down", "servo_disagrees"),
                                                                     ("top_down", "ok")]
    assert asked == [1]
    assert any("Found it again 15 mm from where it was" in n for n in got.notes)


def test_a_servo_that_disagrees_again_after_the_object_was_found_again_is_overruled():
    # Found where it was thought to be: it is the servo that is off, and the
    # fixed camera's sighting stands, as for a servo that cannot settle at all.
    pawn = Cylinder((0.22, 0.0), 0.022, 0.045)
    arm, asked = _ServoStopsShort([pawn], sees=(0.235, 0.0)), []
    pol = ParallelJaw(dataclasses.replace(SPEC, overhead_first=False))
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045),
                   vision=_relocating(lambda: pawn.xy.copy(), asked))
    assert got.ok and arm.held is pawn, got.message
    assert [a["result"] for a in got.attempts] == ["servo_disagrees", "ok"] and asked == [1]
    assert any("Descending where the overhead camera put the jaws" in n for n in got.notes)


def test_a_servo_that_settles_well_off_a_scan_that_was_right_is_overruled_once_a_scan_has_decided():
    # 2026-10-31, c2: the servo settled 11-13 mm outward of a pawn the scan had
    # right, three attempts running, and every close missed.
    pawn = Cylinder((0.22, 0.0), 0.022, 0.045)
    arm, asked = _ServoOff([pawn]), []
    arm.servo_off = 0.012
    pol = ParallelJaw(dataclasses.replace(SPEC, overhead_first=False))
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045),
                   vision=_relocating(lambda: pawn.xy.copy(), asked))
    assert got.ok and arm.held is pawn, got.message
    assert [a["result"] for a in got.attempts] == ["servo_disagrees", "ok"] and asked == [1]
    assert any("going by the scan, not the servo's 12 mm" in n for n in got.notes)


def test_a_servo_that_settles_on_the_object_where_the_scan_was_wrong_is_borne_out_by_the_scan():
    pawn = Cylinder((0.235, 0.0), 0.022, 0.045)                 # 15 mm from where it was thought to be
    arm, asked = _ServoOff([pawn]), []
    arm.servo_off = 0.015
    pol = ParallelJaw(dataclasses.replace(SPEC, overhead_first=False))
    got = pol.pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.022, height_m=0.045),
                   vision=_relocating(lambda: pawn.xy.copy(), asked))
    assert got.ok and arm.held is pawn, got.message
    assert [a["result"] for a in got.attempts] == ["servo_disagrees", "ok"] and asked == [1]
    assert any("Found it again 15 mm from where it was" in n for n in got.notes)


def test_without_the_fixed_cameras_sighting_the_servo_is_trusted_as_before():
    pawn = Cylinder((0.225, 0.0), 0.022, 0.045)
    arm, pol = _ServoOff([pawn], visible=False), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.022, height_m=0.045), vision=_finds_it())
    assert got.ok and arm.held is pawn, got.message


def test_a_correction_the_arm_cannot_make_at_its_limits_is_undone_not_closed_on():
    # 2026-10-22, d6: three stalled 3 mm corrections at full reach left the
    # jaws 8.5 mm off where they had been 3.4, and it closed there.
    class Lurches(_Drifting):
        def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None, orientation=None):
            target = np.asarray(target, float)
            sideways = self.ee[2] < 0.04 and abs(target[2] - self.ee[2]) < 1e-6 and \
                np.linalg.norm(target[:2] - self.ee[:2]) < 0.006 and not getattr(self, "undoing", False)
            if sideways:
                self.undoing = True                     # the next small move is the undo: let it through
                return super().move_ee_to(self.ee + np.array([0.0, 0.006, 0.0]), tol, max_iters) and False
            self.undoing = False
            return super().move_ee_to(target, tol, max_iters, step_size, max_step_deg, orientation)

    arm, pol = Lurches([Cylinder((0.22, 0.0), 0.022, 0.045)]), ParallelJaw(SPEC)
    arm.drift = 0.003
    got = pol.pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.022, height_m=0.045), vision=_finds_it())
    seen = got.attempts[0]["overhead_at_grasp"]
    assert seen["aim_after_mm"] == pytest.approx(seen["aim_before_mm"])
    assert any("back to where they were 3.0 mm off" in n for n in got.notes)


class _Sags(ReachArm):
    """Reaches below 4 cm only leaning at least 39 deg — far out the real arm reaches less than its model."""
    def approach_orientation_for(self, targets, tols=None, flip=False, prefer_tilt_deg=None, margin_deg=None):
        got = super().approach_orientation_for(targets, tols, flip, None, margin_deg)
        if got is None or prefer_tilt_deg is None or abs(prefer_tilt_deg) <= abs(got[1]):
            return got
        a = np.radians(abs(prefer_tilt_deg))                # it takes the lean asked for, when more
        x = np.array([np.cos(a), 0.0, np.sin(a)]) * (-1 if flip else 1)
        z = np.array([np.sin(a), 0.0, -np.cos(a)])
        return np.column_stack([x, np.cross(z, x), z]), -abs(prefer_tilt_deg)

    def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None, orientation=None):
        target = np.asarray(target, float)
        if orientation is not None and target[2] < 0.04 and _lean(orientation) < 39.0:
            super().move_ee_to(target + [0.0, 0.0, 0.05], tol, max_iters)    # stalls 5 cm short
            return False
        return super().move_ee_to(target, tol, max_iters, step_size, max_step_deg, orientation)


def test_a_far_pick_whose_descent_stalls_leans_further_and_comes_down_again():
    # 2026-10-27, f7 at 0.38 m: the hover reached at -40 deg, the descent
    # stalled 6-12 cm above the grasp, and each attempt ended there.
    rook = Cylinder((0.30, 0.0), 0.024, 0.056)
    arm, pol = _Sags([rook]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(rook.xy), foot_m=0.024, height_m=0.056), vision=_finds_it())
    assert got.ok, got.message
    assert arm.held is rook and len(got.attempts) == 1
    assert any("down again leaning" in n for n in got.notes)
    assert abs(got.attempts[0]["tilt_deg"]) >= 39.0


def test_a_far_place_whose_descent_stalls_leans_further_and_comes_down_again():
    # 2026-10-22, e7: the model's -35 deg stalled 7.7 cm above the release.
    class Sags(ReachArm):
        def approach_orientation_for(self, targets, tols=None, flip=False, prefer_tilt_deg=None, margin_deg=None):
            got = super().approach_orientation_for(targets, tols, flip, None, margin_deg)
            if got is None or prefer_tilt_deg is None or abs(prefer_tilt_deg) <= abs(got[1]):
                return got
            a = np.radians(abs(prefer_tilt_deg))                # it takes the lean asked for, when more
            x = np.array([np.cos(a), 0.0, np.sin(a)]) * (-1 if flip else 1)
            z = np.array([np.sin(a), 0.0, -np.cos(a)])
            return np.column_stack([x, np.cross(z, x), z]), -abs(prefer_tilt_deg)

        def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None, orientation=None):
            target = np.asarray(target, float)
            if orientation is not None and target[2] < 0.04 and _lean(orientation) < 39.0:
                super().move_ee_to(target + [0.0, 0.0, 0.05], tol, max_iters)    # stalls 5 cm short
                return False
            return super().move_ee_to(target, tol, max_iters, step_size, max_step_deg, orientation)

    arm, pol = Sags(), ParallelJaw(SPEC)
    plan = pol.plan_from_shape(0.045, {}, 0.022)
    piece = _holding(arm, plan)
    put = pol.place(arm, on_table((0.33, 0.0)), grasp=plan)
    assert put.ok, put.message
    assert np.allclose(piece.xy, (0.33, 0.0), atol=0.002)
    assert sum("down again leaning" in n for n in pol._notes or []) == 0      # notes belong to picks


class _GoesOffLine(_ServoFindsIt):
    """
    The SO-101 on 2026-11-19 at 15:28 (run 37): a descent that turns as it
    goes, 3 deg a step, goes 4 cm up and 3 cm out and stops; one that sets out
    more than 15 mm to the side of its goal stops where it is.
    """
    def __init__(self, objects=()):
        super().__init__(objects)
        self.now_rot, self.went_off = None, 0

    def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None, orientation=None, **k):
        target = np.asarray(target, float)
        stepped_down = max_step_deg is not None and target[2] < self.ee[2] - 0.01
        turning = (orientation is not None and self.now_rot is not None
                   and abs(_lean(orientation) - _lean(self.now_rot)) > 3.0)
        aside = float(np.linalg.norm(self.ee[:2] - target[:2])) > 0.015
        if orientation is not None:
            self.now_rot = np.asarray(orientation, float)
        if stepped_down and turning:
            self.went_off += 1
            super().move_ee_to(self.ee + [0.03, 0.0, 0.04], tol, max_iters)
            return False
        if stepped_down and aside:
            return False
        return super().move_ee_to(target, tol, max_iters, step_size, max_step_deg, orientation)


def _leans_more_once_measured(pol):
    """The descent's lean re-chosen 10 deg further than the hover's (_descend's re-hold: start= given)."""
    held_for = pol._held_for

    def again(env, approach, xy, plan, *a, **k):
        rot, tilt, checked, lift = held_for(env, approach, xy, plan, *a, **k)
        if k.get("start") is None or rot is None:
            return rot, tilt, checked, lift
        t = np.radians(abs(tilt) + 10.0)
        x, z = np.array([np.cos(t), 0.0, np.sin(t)]), np.array([np.sin(t), 0.0, -np.cos(t)])
        return np.column_stack([x, np.cross(z, x), z]), -(abs(tilt) + 10.0), checked, lift
    pol._held_for = again


def test_a_descent_leaning_other_than_its_hover_turns_over_the_object_first():
    """2026-11-19 15:28 (run 37, the user's film): turning on the way down, the arm went 4 cm up and 3 cm out."""
    def pick(on):
        rook = Cylinder((0.303, 0.0), 0.024, 0.056)
        arm = _GoesOffLine([rook])
        pol = ParallelJaw(dataclasses.replace(SPEC, descent_turn_first=on, overhead_first=False,
                                              servo_agree_m=0.004))
        _leans_more_once_measured(pol)
        got = pol.pick(arm, Target(on_table((0.296, 0.0)), foot_m=0.024, height_m=0.056), vision=_finds_it())
        return got, arm
    got, arm = pick(False)
    assert arm.went_off >= 1                                            # as it was: turned going down
    got, arm = pick(True)
    assert got.ok and arm.went_off == 0, got.message
    assert [a["result"] for a in got.attempts][-1] == "ok"


def test_a_descent_that_went_off_its_line_goes_back_over_the_grasp_not_up_from_where_it_went(capsys):
    from mrcf.policies.parallel_jaw.pick import PickContext

    def ctx_for(pol, arm):
        rot, tilt = arm.approach_orientation_for([on_table((0.303, 0.0), 0.07)])
        ctx = PickContext(Target(on_table((0.303, 0.0)), foot_m=0.024, height_m=0.056), [], Vision(), 0.04,
                          pol.plan_from_shape(0.056, {}, 0.024), np.array([0.303, 0.0]), False)
        ctx.rot, ctx.tilt, ctx.tag = rot, tilt, "[Pick 1/3]"
        ctx.hover, ctx.hover_tol = on_table((0.303, 0.0), 0.071), SPEC.travel_tol_m
        ctx.grasp, ctx.grasp_tol, ctx.grasp_step = on_table((0.303, 0.0), 0.025), 0.004, 3.0
        ctx.descent_from = ctx.hover.copy()
        return ctx

    for on in (False, True):
        arm = _GoesOffLine()
        pol = ParallelJaw(dataclasses.replace(SPEC, descent_turn_first=on, descent_lean_retries=0))
        pol._attempts = [{"attempt": 1}]
        ctx = ctx_for(pol, arm)
        arm.now_rot = ctx.rot
        arm.ee = np.array([0.333, 0.0, 0.111])                      # where the turning descent left it
        assert pol._lean_down_again(arm, ctx) is on
        if on:
            assert np.allclose(arm.ee, ctx.grasp, atol=0.005)
            assert "went off its line: 9.1 cm from the grasp, from 4.6" in capsys.readouterr().out
