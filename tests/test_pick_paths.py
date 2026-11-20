"""
The pick's less travelled paths, one test each: a wrist camera that locates the
object's foot, and every reason an attempt is retried or a pick given up on.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from fake_arm import Cylinder, FakeArm
from mrcf.policies.base import Target, Vision
from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw
from test_approaches import SPEC as ZONED, ReachArm

SPEC = GripperSpec(max_open_m=0.045, tip_below_grasp_m=0.007, free_depth_m=0.055,
                   finger_thickness_m=0.009, finger_half_width_m=0.008, jaw_axis="tangential",
                   hover_height_m=0.06, lift_height_m=0.07, lift_clearance_m=0.05)


def on_table(xy, z=0.0):
    return np.array([xy[0], xy[1], z])


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


def _looks(height_mm=45.0, foot_xy=None, foot_mm=21.0, ok=True, times=None):
    """Vision.measure: the object as the wrist camera measures it (ok=False: it cannot)."""
    calls = []

    def measure(frame, expected_px=None, model=None, expected_xy_m=None):
        calls.append(expected_xy_m)
        if not ok or (times is not None and len(calls) > times):
            return {"ok": False, "why": "nothing outlined"}
        return {"ok": True, "measurement": SimpleNamespace(
            height_mm=height_mm, widths_mm={0: foot_mm, 10: 15, 20: 16, int(height_mm): 8},
            base_radius_mm=foot_mm / 2, slant_deg=30.0,
            base_world_m=None if foot_xy is None else np.asarray(foot_xy, float))}
    return Vision(measure=measure, find=lambda frame, expected_px, radius_px: expected_px)


# ── A wrist camera that knows where it is ─────────────────────────────────────

class LocatedArm(FakeArm):
    """A wrist camera with a pose (camera_model), so the pick looks for the object's foot."""

    def camera_model(self, camera, width, height):
        return {"K": np.eye(3), "D": np.zeros(5), "T_world_cam": np.eye(4), "plane_z_m": 0.0, "scale": 1.0}


@pytest.fixture
def segmenter(monkeypatch):
    from mrcf.perception import segment
    monkeypatch.setattr(segment, "available", lambda: True)


def _located(**spec):
    return ParallelJaw(dataclasses.replace(SPEC, camera="wrist", **spec))


def test_a_located_wrist_camera_aims_at_the_foot_it_finds(segmenter):
    pawn = Cylinder((0.205, 0.0), 0.021, 0.045)          # 5 mm from where the scan saw it
    arm = LocatedArm([pawn])
    got = _located().pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045),
                          vision=_looks(foot_xy=pawn.xy))
    assert got.ok and arm.held is pawn, got.message
    assert any("the overhead camera's aim moved by that" in n for n in got.notes)
    assert np.allclose(got.at_xy, pawn.xy)


def test_a_located_wrist_camera_can_be_told_to_aim_by_the_scan(segmenter):
    pawn = Cylinder((0.201, 0.0), 0.021, 0.045)
    arm = LocatedArm([pawn])
    got = _located(aim_at_measured_foot=False).pick(
        arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), vision=_looks(foot_xy=pawn.xy))
    assert got.ok, got.message
    assert any("aiming by the scan" in n for n in got.notes)
    assert np.allclose(got.at_xy, (0.20, 0.0))


@pytest.mark.parametrize("vision, code", [
    (_looks(foot_xy=None), "foot_not_found"),
    (_looks(foot_xy=(0.23, 0.0)), "foot_too_far"),
    (_looks(height_mm=80.0, foot_xy=(0.20, 0.0)), "other_object"),
])
def test_a_located_wrist_camera_does_not_descend_without_the_right_foot(segmenter, vision, code):
    arm = LocatedArm([Cylinder((0.20, 0.0), 0.021, 0.045)])
    got = _located().pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), vision=vision)
    assert not got.ok and arm.held is None
    assert [a["result"] for a in got.attempts] == [code] * 3
    assert min(p[2] for p in arm.path) > 0.05           # never went down


def test_a_height_off_with_nothing_near_is_a_part_of_it_not_another_object(segmenter):
    # 2026-11-17: a figure 44 mm from the side read 21 from over it in a dim room, nothing within 11 cm.
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm = LocatedArm([pawn])
    got = _located(other_object_near_m=0.04).pick(
        arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), vision=_looks(height_mm=21.0, foot_xy=pawn.xy))
    assert got.ok and arm.held is pawn, got.message
    assert any("a part of it" in n for n in got.notes)


def test_a_height_off_with_a_neighbour_near_is_still_another_object(segmenter):
    from mrcf.policies.base import Obstacle

    arm = LocatedArm([Cylinder((0.20, 0.0), 0.021, 0.045)])
    beside = [Obstacle(np.array([0.20, 0.048]), 0.011, "another figure", 0.021)]   # room for the jaws; within 4 cm
    got = _located(other_object_near_m=0.04).pick(
        arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), obstacles=beside,
        vision=_looks(height_mm=21.0, foot_xy=(0.20, 0.0)))
    assert not got.ok and arm.held is None
    assert "other_object" in [a["result"] for a in got.attempts]


def test_told_to_risk_it_a_foot_not_found_comes_down_at_the_scans_position(segmenter):
    # 2026-11-16, Qa4: the d1 queen's foot was hidden from the wrist camera and nothing else
    # could see the jaws; descend_unconfirmed takes the scan's word for it instead of retrying.
    pawn = Cylinder((0.201, 0.0), 0.021, 0.045)
    arm = LocatedArm([pawn])
    got = _located(descend_unconfirmed=True).pick(
        arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), vision=_looks(foot_xy=None))
    assert got.ok and arm.held is pawn, got.message
    assert [a["result"] for a in got.attempts] == ["ok"]
    assert any("descending at the scan's position anyway" in n for n in got.notes)
    assert np.allclose(got.at_xy, (0.20, 0.0))


def test_told_to_risk_it_it_still_does_not_come_down_blind_after_a_miss(segmenter):
    # A close on nothing may have pushed it: the scan's position is stale, and down there
    # again blind is how a piece gets knocked over.
    arm = LocatedArm([Cylinder((0.235, 0.0), 0.021, 0.045)])      # beyond where the jaws close
    got = _located(descend_unconfirmed=True).pick(
        arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), vision=_looks(foot_xy=None))
    assert not got.ok and arm.held is None
    assert [a["result"] for a in got.attempts] == ["closed_on_nothing", "foot_not_found", "foot_not_found"]
    lows = [i for i, p in enumerate(arm.path) if p[2] < 0.03]
    first_up = next(i for i in range(lows[0], len(arm.path)) if arm.path[i][2] > 0.05)
    assert all(i < first_up for i in lows)               # down once, the first attempt only


class SilentWristArm(LocatedArm):
    """2026-11-14: the wrist camera dropped off USB mid-game; lerobot then raises on every frame."""

    def __init__(self, objects=(), age_s=float("inf")):
        super().__init__(objects)
        self.age_s = age_s

    def frame_age_s(self, camera):
        return self.age_s

    def render_image(self, camera, width=640, height=480):
        if camera == "wrist":
            raise RuntimeError("OpenCVCamera(1) has not captured any frames yet.")
        return super().render_image(camera, width, height)


def test_a_wrist_camera_that_has_stopped_sending_is_not_waited_for(segmenter):
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm = SilentWristArm([pawn])
    got = _located().pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045),
                          vision=_looks(foot_xy=pawn.xy))
    assert got.ok and arm.held is pawn, got.message


def test_a_wrist_frame_that_does_not_come_is_a_foot_not_found_not_a_crash(segmenter):
    arm = SilentWristArm([Cylinder((0.20, 0.0), 0.021, 0.045)], age_s=0.1)   # it stops between the age and the read
    got = _located().pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045),
                          vision=_looks(foot_xy=(0.20, 0.0)))
    assert not got.ok and arm.held is None
    assert [a["result"] for a in got.attempts] == ["foot_not_found"] * 3
    assert min(p[2] for p in arm.path) > 0.05           # not down blind: no overhead camera put the jaws there


# ── Retried, and why ──────────────────────────────────────────────────────────

class _Faulty(FakeArm):
    """A FakeArm with one fault: `short` stops moves to the hover 5 cm out, `stall` stops descents
    3 cm up, `wide` closes reading 35 mm, `drop` lets go during the lift."""

    def __init__(self, objects=(), fault=""):
        super().__init__(objects)
        self.fault = fault
        self.held_widths = []

    def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None, orientation=None):
        target = np.asarray(target, float)
        if self.fault == "short" and target[2] > 0.05:
            super().move_ee_to(target + [0.05, 0.0, 0.0], tol=tol)
            return False
        if self.fault == "stall" and target[2] < 0.03:
            super().move_ee_to(np.array([target[0], target[1], 0.06]), tol=tol)
            return False
        if self.fault == "drop" and self.held is not None and target[2] > self.ee[2]:
            self.held = None
        return super().move_ee_to(target, tol=tol, max_iters=max_iters, max_step_deg=max_step_deg,
                                  orientation=orientation)

    def squeeze_gripper(self):
        got = super().squeeze_gripper()
        return (0.035, True) if self.fault == "wide" else got

    def hold_width(self, width_m):
        self.held_widths.append(width_m)


@pytest.mark.parametrize("fault, code", [
    ("short", "approach_short"), ("stall", "descent_stalled"), ("wide", "too_wide"), ("drop", "grip_lost"),
])
def test_each_failed_phase_is_retried_under_its_own_code(fault, code):
    arm = _Faulty([Cylinder((0.20, 0.0), 0.021, 0.045)], fault)
    got = ParallelJaw(dataclasses.replace(SPEC, blind_retry_after_miss=True)).pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045))
    assert not got.ok
    assert [a["result"] for a in got.attempts] == [code] * 3


def test_a_grip_tells_the_backend_how_wide_the_held_object_is():
    arm = _Faulty([Cylinder((0.20, 0.0), 0.021, 0.045)])
    got = ParallelJaw(SPEC).pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045))
    assert got.ok, got.message
    # Last, once it held it; first, where to come down (against the fixed jaw: _aim_at_anvil).
    assert arm.held_widths[-1] == got.grasp.grip_width_m
    assert arm.held_widths[:-1] == [pytest.approx(got.grasp.width_m + 2 * SPEC.anvil_margin_m)]


class _BalksOnce(FakeArm):
    """Its first descent below 3 cm does not move at all; every one after it arrives."""

    def __init__(self, objects=()):
        super().__init__(objects)
        self.balked = 0

    def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None, orientation=None):
        if np.asarray(target, float)[2] < 0.03 and not self.balked:
            self.balked += 1
            return False
        return super().move_ee_to(target, tol=tol, max_iters=max_iters, max_step_deg=max_step_deg,
                                  orientation=orientation)


def test_a_descent_that_never_got_going_is_tried_again_where_it_was_before_anything_else():
    # 2026-11-04, e4: the descent at -15 deg stalled 7.3 of 7.2 cm short; leaning
    # to -25 it reached, and the close squeezed the pawn out of the jaws.
    arm = _BalksOnce([Cylinder((0.20, 0.0), 0.021, 0.045)])
    got = ParallelJaw(SPEC).pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045))
    assert got.ok, got.message
    assert arm.balked == 1 and len(got.attempts) == 1


class _LosesReach(ReachArm):
    """Its orientation search finds nothing once the pick has started moving."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.started = False

    def set_gripper(self, width, **kw):
        self.started = True
        return super().set_gripper(width, **kw)

    def approach_orientation_for(self, targets, **kw):
        return None if self.started else super().approach_orientation_for(targets, **kw)


def test_no_orientation_from_where_the_arm_is_is_retried_then_given_up():
    arm = _LosesReach([Cylinder((0.22, 0.0), 0.021, 0.045)])
    got = ParallelJaw(ZONED).pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.021, height_m=0.045),
                                  vision=Vision(find=lambda f, e, r: e))
    assert not got.ok and [a["result"] for a in got.attempts] == ["no_orientation"] * 3
    assert arm.path == []


# ── The wrist servo's own height ──────────────────────────────────────────────

class _NoServoHeight(ReachArm):
    """Reaches everything, except — once moving — down to the servo's 70 mm (orientation) or (move)."""

    def __init__(self, *a, how="orientation", **k):
        super().__init__(*a, **k)
        self.how = how

    def approach_orientation_for(self, targets, **kw):
        if self.how == "orientation" and self.path and any(abs(t[2] - 0.07) < 1e-3 for t in targets):
            return None
        return super().approach_orientation_for(targets, **kw)

    def move_ee_to(self, target, *a, **k):
        if self.how == "move" and abs(np.asarray(target)[2] - 0.07) < 1e-3:
            return False
        return super().move_ee_to(target, *a, **k)


@pytest.mark.parametrize("how", ["orientation", "move"])
def test_a_servo_height_out_of_reach_is_retried(how):
    arm = _NoServoHeight([Cylinder((0.22, 0.0), 0.030, 0.082)], how=how)
    pol = ParallelJaw(dataclasses.replace(ZONED, overhead_first=False))
    got = pol.pick(arm, Target(on_table((0.22, 0.0)), foot_m=0.030, max_height_m=0.082),
                   vision=_looks(ok=False))
    # The next approach along does not climb to the servo's height, and grips.
    assert got.ok, got.message
    assert [(a["approach"], a["result"]) for a in got.attempts] == [("top_down", "servo_height"),
                                                                     ("near_base", "ok")]


def test_the_far_look_stands_when_the_one_from_the_servo_height_fails():
    king = Cylinder((0.22, 0.0), 0.030, 0.082)
    arm = ReachArm([king])
    pol = ParallelJaw(dataclasses.replace(ZONED, overhead_first=False))
    got = pol.pick(arm, Target(on_table(king.xy), foot_m=0.030, max_height_m=0.082),
                   vision=_looks(height_mm=82.0, foot_mm=30.0, times=2))
    assert got.ok and arm.held is king, got.message
    assert got.measured.height_mm == 82.0


# ── Wrist servo against the fixed camera ──────────────────────────────────────

class _LostServo(ReachArm):
    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        self.servo_heights.append(round(float(self.ee[2]), 3))
        return False, "target lost while acquiring (iteration 1: not found)"


def test_a_lost_wrist_servo_descends_where_the_fixed_camera_put_the_jaws():
    pawn = Cylinder((0.22, 0.0), 0.021, 0.045)
    arm = _LostServo([pawn])
    pol = ParallelJaw(dataclasses.replace(ZONED, overhead_first=False))
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, height_m=0.045), vision=Vision(find=lambda f, e, r: e))
    assert got.ok and arm.held is pawn, got.message
    assert arm.servo_heights == [0.07]
    assert any(n.endswith("Descending where the overhead camera put the jaws (0.0 mm from the object).")
               for n in got.notes)


def test_jaws_the_servo_left_narrow_are_opened_to_the_plan_before_descending():
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm = ReachArm([pawn])
    pol = ParallelJaw(dataclasses.replace(ZONED, overhead_first=False, servo_open_m=0.025, servo_height_m=None))
    widths = []
    real = arm.set_gripper
    arm.set_gripper = lambda w, **k: (widths.append(round(w, 4)), real(w, **k))[1]
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.045), vision=Vision(find=lambda f, e, r: e))
    assert got.ok, got.message
    assert widths[:3] == [0.036, 0.025, 0.036]            # planned, the servo's, planned again


# ── Re-planned once measured ──────────────────────────────────────────────────

def test_a_measured_object_is_fitted_again_and_refused_if_nothing_reaches_it_as_it_is():
    # Near the base nothing reaches above 65 mm. Planned as at most 45 mm tall, it is
    # measured at 82 from near_base's low hover: the retry has no approach that hovers above it.
    arm = ReachArm([])                                    # nothing there to grip
    pol = ParallelJaw(ZONED)
    got = pol.pick(arm, Target(on_table((0.12, 0.0)), foot_m=0.030, max_height_m=0.045),
                   vision=_looks(height_mm=82.0, foot_mm=30.0))
    assert not got.ok and len(got.attempts) == 1
    assert any("As measured, it is out of reach here for every approach" in n for n in got.notes)


def test_a_lean_far_out_grasps_a_tall_measured_object_higher_so_its_top_stays_between_the_jaws():
    king = Cylinder((0.33, 0.0), 0.030, 0.082)
    arm = ReachArm([king])
    pol = ParallelJaw(ZONED)
    got = pol.pick(arm, Target(on_table(king.xy), foot_m=0.030, max_height_m=0.082),
                   vision=_looks(height_mm=82.0, foot_mm=30.0))
    assert got.approach == "reach_out"
    assert any("so its top stays between the tipped jaws" in n for n in got.notes)


# ── Extending: a new approach, a new aligner ──────────────────────────────────

def test_a_new_approach_can_line_up_the_jaws_its_own_way():
    from mrcf.policies.aligners import ALIGNERS, Aligner, register_aligner
    from mrcf.policies.approaches import Approach

    class Probe(Aligner):
        """Finds the object by touch: here, the true position the test knows."""
        name = "probe"

        def applies(self, pol, env, ctx):
            return True

        def run(self, pol, env, ctx):
            pol._say(ctx.tag, "Probed it.")
            ctx.target_xy = env.objects[0].xy.copy()

    register_aligner(Probe())
    try:
        pawn = Cylinder((0.206, 0.0), 0.021, 0.045)          # 6 mm from the scan: the scan alone misses
        probing = Approach("probing", zones=("mid",), servo=False, align=("probe",))
        pol = ParallelJaw(dataclasses.replace(SPEC, approaches=(probing,)))
        got = pol.pick(FakeArm([pawn]), Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045))
        assert got.ok and got.approach == "probing", got.message
        assert "[Pick 1/3] Probed it." in got.notes
    finally:
        ALIGNERS.pop("probe", None)


def test_an_approach_naming_an_aligner_that_does_not_exist_says_so():
    from mrcf.policies.approaches import Approach
    # When the policy is built, not on the arm mid-pick.
    with pytest.raises(ValueError, match="no aligner typo: wrist_srevo"):
        ParallelJaw(dataclasses.replace(SPEC, approaches=(Approach("typo", align=("wrist_srevo",)),)))


# ── Carrying ──────────────────────────────────────────────────────────────────

class _Slips(FakeArm):
    """Reads a held piece's own width, and lets it go `drop_after` cycles into a carry."""
    drop_after = 5

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.carrying: "int | None" = None

    def get_gripper_width(self):
        return self.held.diameter_m if self.held is not None else self.width

    def step(self, n=1):
        super().step(n)
        if self.carrying is not None and self.held is not None:
            self.carrying += n
            if self.carrying >= self.drop_after:
                self.held, self.width = None, 0.0


def test_a_piece_that_slips_out_on_the_way_stops_the_arm_there():
    # 2026-10-30: a pawn slipped out 0.2-0.4 s into the carry; the place found
    # out at the release, 9 s and a whole carry later.
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm, pol = _Slips([pawn]), ParallelJaw(SPEC)
    assert pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, height_m=0.045)).ok
    trips, arm.carrying = arm.interlock.trip_count, 0
    got = pol.place(arm, on_table((0.20, 0.08)))
    assert not got.ok and "slipped out of the jaws on the way: they closed from 21 to 0 mm" in got.message
    # Stopped on the way, not carried on over the target and down to the release.
    assert np.linalg.norm(arm.ee[:2] - (0.20, 0.08)) > 0.03 and arm.ee[2] > 0.045
    # Cleared, but counted: the controller looks at the scene again.
    assert not arm.interlock.tripped() and arm.interlock.trip_count == trips + 1


def test_a_piece_carried_all_the_way_is_not_taken_for_a_slip():
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm, pol = _Slips([pawn]), ParallelJaw(SPEC)
    assert pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, height_m=0.045)).ok
    trips = arm.interlock.trip_count
    got = pol.place(arm, on_table((0.20, 0.08)))
    assert got.ok, got.message
    assert arm.interlock.trip_count == trips and "parallel_jaw:grip" not in arm.interlock._checks


class _TouchesDown(_Slips):
    """Its held piece hangs LOW_M lower than the jaws' model has it (leaning, on the rig): its foot meets
    the table above the release height and it pivots in the jaws, which close PIVOT_M further."""
    LOW_M, PIVOT_M = 0.018, 0.006

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.grasp_z = None
        self.placing = False

    def squeeze_gripper(self):
        got = super().squeeze_gripper()
        self.grasp_z = float(self.ee[2])
        return got

    def get_gripper_width(self):
        if self.held is not None and self.placing and self.ee[2] < self.grasp_z + self.LOW_M:
            return self.held.diameter_m - self.PIVOT_M
        return super().get_gripper_width()


def test_jaws_that_give_as_the_piece_touches_down_let_it_go_there():
    # 2026-11-04: a pawn onto e7 at a -40 deg lean — its foot met the board 8 mm
    # above the release height, it pivoted between the pads and the jaws closed
    # 6 mm. Taken for a slip, the arm stopped holding it; it went with the arm
    # and stood on e6.
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm, pol = _TouchesDown([pawn]), ParallelJaw(SPEC)
    assert pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, height_m=0.045)).ok
    arm.placing = True
    got = pol.place(arm, on_table((0.20, 0.08)))
    assert got.ok and got.released and "touched down" in got.message, got.message
    assert arm.held is None and np.allclose(pawn.xy, (0.20, 0.08), atol=0.002)
    assert "parallel_jaw:grip" not in arm.interlock._checks


# ── After a miss: found again where it was pushed (Vision.relocate) ──────────────

class _PushedOnMiss(FakeArm):
    """A close that misses pushes the object PUSH_M toward the base (at the origin here)."""
    PUSH_M = 0.010

    def squeeze_gripper(self):
        width, held = super().squeeze_gripper()
        if not held:
            for o in self.objects:
                o.xy = o.xy - self.PUSH_M * o.xy / np.linalg.norm(o.xy)
        return width, held


def test_a_miss_that_pushed_the_object_is_followed_by_finding_it_again_and_aiming_there():
    # 2026-10-30: each miss pushed a free-standing pawn 2-27 mm toward the
    # base; retries looking for it from the old spot chased it 42 mm and lost it.
    pawn = Cylinder((0.208, 0.0), 0.021, 0.045)            # 8 mm further out than the scan saw it
    arm = _PushedOnMiss([pawn])
    asked = []

    def relocate(env, near_xy):
        asked.append(np.round(near_xy, 3).tolist())
        return pawn.xy.copy()

    got = ParallelJaw(SPEC).pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045),
                                 vision=Vision(relocate=relocate))
    assert got.ok and arm.held is pawn, got.message
    assert asked == [[0.2, 0.0]] and [a["result"] for a in got.attempts][:1] == ["closed_on_nothing"]
    assert any("Found it again 2 mm from where it was (-2 along the arm" in n for n in got.notes)


def test_a_miss_that_left_the_object_where_it_was_steps_the_aim_out_along_the_arm():
    # 2026-11-04, e4: three closes on nothing at one spot, the pawn never
    # touched, the jaws short of it toward the base. Found again where it was,
    # each retry aimed there again and missed the same way.
    pawn = Cylinder((0.207, 0.0), 0.021, 0.045)        # 7 mm further out than the aim, and it stays there
    arm = FakeArm([pawn])
    spec = dataclasses.replace(SPEC, miss_steps_along_m=(0.005, 0.010))
    got = ParallelJaw(spec).pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045),
                                 vision=Vision(relocate=lambda env, near_xy: np.asarray(near_xy, float)))
    assert got.ok and arm.held is pawn, got.message
    assert [a["result"] for a in got.attempts] == ["closed_on_nothing", "ok"]
    assert any("this attempt aims +5 mm along the arm" in n for n in got.notes)
    # Pushed aside by the miss, with no look saying the jaws closed short: aimed
    # where it was found, not stepped out blind (2026-11-04, e2: stepped out,
    # two retries came down 6-8 mm beyond a pawn the miss pushed toward the base).
    class _PushedFar(_PushedOnMiss):
        PUSH_M = 0.020
    pawn.xy, arm = np.array([0.207, 0.0]), _PushedFar([pawn])
    got = ParallelJaw(spec).pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045),
                                 vision=Vision(relocate=lambda env, near_xy: pawn.xy.copy()))
    assert not any("the aim was off — this attempt aims" in n for n in got.notes), got.notes
    # Without the steps: the same spot three times.
    pawn.xy, arm = np.array([0.207, 0.0]), FakeArm([pawn])
    got = ParallelJaw(SPEC).pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045),
                                 vision=Vision(relocate=lambda env, near_xy: np.asarray(near_xy, float)))
    assert not got.ok and [a["result"] for a in got.attempts] == ["closed_on_nothing"] * 3


def test_a_miss_after_which_the_object_cannot_be_found_ends_the_pick_without_chasing_it():
    pawn = Cylinder((0.208, 0.0), 0.021, 0.045)
    arm = _PushedOnMiss([pawn])
    got = ParallelJaw(SPEC).pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045),
                                 vision=Vision(relocate=lambda env, near_xy: None))
    assert not got.ok and arm.held is None
    assert len(got.attempts) == 1 and "cannot be found near where it was" in got.notes[-3] + got.notes[-2]


# ── Looking from where a pick looks, without picking (tools/look_tour.py) ──────

def test_go_over_stops_at_the_picks_hover_with_the_jaws_open_and_closes_on_nothing():
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.045)])
    pol = ParallelJaw(SPEC)
    ctx = pol.go_over(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045))
    assert not hasattr(ctx, "ok"), ctx.message
    assert np.allclose(arm.get_ee_pos(), ctx.hover, atol=0.01)
    assert ctx.hover[2] >= SPEC.table_z + 0.045 + SPEC.tip_below_grasp_m      # its top clear of the open jaws
    assert arm.width == pytest.approx(ctx.gp.pregrasp_open_m) and arm.held is None
    assert pol.look_from(arm, ctx, 0.05)
    assert arm.get_ee_pos()[2] == pytest.approx(SPEC.table_z + 0.05, abs=0.01)
    assert arm.held is None


def test_go_over_says_why_when_a_pick_would_be_refused():
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.045)])
    from mrcf.policies.base import Obstacle
    crowded = [Obstacle(np.array([0.20, 0.025]), 0.012, "a neighbour"), Obstacle(np.array([0.20, -0.025]), 0.012, "another")]
    got = ParallelJaw(SPEC).go_over(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), crowded)
    assert got.refused and not got.ok and "leaves" in got.message
    assert arm.path == []                                                     # nothing moved


# ── A located wrist camera among other objects (expected_view) ─────────────────

def _looks_among(foot_xy, to_camera=lambda xy: np.asarray(xy, float)):
    """Vision.measure that takes the objects around the target: its foot from straight above, no height."""
    calls = []

    def measure(frame, expected_px=None, model=None, expected_xy_m=None, known=None, target=None, need_height=True):
        calls.append({"known": known, "target": target, "need_height": need_height, "expected": expected_xy_m})
        if foot_xy is None:
            return {"ok": False, "why": "it is hidden behind a neighbour"}
        if need_height:
            return {"ok": False, "why": "its outline was found but the view cannot tell its height"}
        return {"ok": True, "measurement": SimpleNamespace(
            height_mm=None, widths_mm={}, base_radius_mm=10.5, slant_deg=30.0, base_world_m=to_camera(foot_xy))}
    return Vision(measure=measure), calls


class _ShiftedCamera(LocatedArm):
    """
    Its camera placed through corrected joints: the arm's own model is 10 mm
    further along x (to_arm), and the scan's positions 6 mm short of the
    camera's frame along x the other way (scan_to_camera).
    """

    def camera_model(self, camera, width, height):
        model = super().camera_model(camera, width, height)
        model["to_arm"] = np.eye(4)
        model["to_arm"][0, 3] = 0.010
        model["scan_to_camera"] = np.array([-0.006, 0.0, 0.0])
        return model


def test_the_wrist_camera_is_told_who_stands_around_and_aims_by_a_foot_without_a_height(segmenter):
    from mrcf.policies.base import Obstacle
    pawn = Cylinder((0.204, 0.0), 0.021, 0.045)
    arm = LocatedArm([pawn])
    vision, calls = _looks_among(pawn.xy)
    beside = Obstacle(np.array([0.20, 0.045]), 0.0105, "the pawn beside it", 0.045)
    got = _located().pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), [beside], vision=vision)
    assert got.ok and arm.held is pawn, got.message
    assert np.allclose(got.at_xy, pawn.xy)
    first = calls[0]
    assert first["target"] == "target" and first["need_height"] is False
    keys = {k.key: k for k in first["known"]}
    assert "target" in keys and any("beside" in k for k in keys)
    assert not any("base" in k for k in keys)             # the arm's own base is not something it sees
    assert np.allclose(keys["target"].foot_xy_m, (0.20, 0.0))


def test_a_foot_the_wrist_camera_cannot_find_is_descended_on_where_the_overhead_camera_put_the_jaws(segmenter, monkeypatch):
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm = LocatedArm([pawn])
    pol = _located()

    def sighted(env, at, tol, rot, tag):
        pol._overhead_err_m = 0.0015
        return np.zeros(2)
    monkeypatch.setattr(pol, "_overhead_correct", sighted)
    got = pol.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), vision=_looks_among(None)[0])
    assert got.ok and arm.held is pawn, got.message
    assert any("cannot find its foot" in n and "overhead camera put the jaws" in n for n in got.notes)


def test_the_jaws_go_where_the_scan_was_moved_by_how_far_the_camera_sees_the_piece_from_it(segmenter):
    # The scan has the pawn at (0.20, 0) and the camera's frame is 6 mm short
    # of the scan's (scan_to_camera): it is looked for at (0.194, 0). It stands
    # 5 mm further out: the camera sees it at (0.199, 0), 5 mm from the scan in
    # the scan's frame, and the aim moves 5 mm. The camera's own jaw model
    # (to_arm) is only said, not aimed by.
    pawn = Cylinder((0.205, 0.0), 0.021, 0.045)
    arm = _ShiftedCamera([pawn])
    vision, calls = _looks_among(pawn.xy, to_camera=lambda xy: np.asarray(xy, float) - [0.006, 0.0])
    got = _located().pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, height_m=0.045), vision=vision)
    assert got.ok and arm.held is pawn, got.message
    target = next(k for k in calls[0]["known"] if k.key == "target")
    assert np.allclose(target.foot_xy_m, (0.194, 0.0))     # looked for in the camera's frame
    assert np.allclose(got.at_xy, pawn.xy)                 # aimed 5 mm on from the scan
    assert any("+5.0 along the arm" in n and "its own jaw model" in n for n in got.notes)


# ── A crowded spot: measured from over it before refusing ─────────────────────
# The full chess board, 2026-11-03: the d1 queen between the c1 bishop and the
# e1 king, where the scan saw them. By their feet c1 leaves 27 mm and the
# queen needs 35; measured, each is slimmer where the finger tips come down to
# (on the SO-101, from 34 mm up: the queen needs 30, the bishop leaves 33).

QUEEN = {4: 28.9, 8: 28.1, 12: 26.9, 16: 25.5, 20: 23.7, 24: 22.5, 28: 21.9, 32: 22.4, 36: 22.3, 40: 21.6,
         44: 20.8, 48: 21.0, 52: 22.1, 56: 21.4, 60: 20.4, 64: 16.7}           # a queen picked 2026-10-28
BISHOP = {4: 26.3, 8: 25.6, 12: 25.1, 16: 24.3, 20: 22.8, 24: 21.0, 28: 18.9, 32: 18.5, 36: 18.4, 40: 18.7,
          44: 18.8, 48: 18.4, 52: 17.1, 56: 14.9}                               # 2026-10-15
D1, C1, E1 = (0.1334, 0.0175), (0.1315, 0.0554), (0.1388, -0.0253)


def _back_rank():
    from mrcf.policies.base import Obstacle
    return [Obstacle(np.array(C1), 0.01215, "white_piece_c1", 0.082),
            Obstacle(np.array(E1), 0.0129, "white_piece_e1", 0.082)]


def _measures(shapes):
    """Vision.measure among the objects around: {key fragment: (height_mm, widths_mm, foot_xy)}; else not found."""
    asked = []

    def measure(frame, expected_px=None, model=None, expected_xy_m=None, known=None, target=None, need_height=True):
        asked.append(target)
        for part, (height, widths, xy) in shapes.items():
            if part in (target or "target"):
                return {"ok": True, "measurement": SimpleNamespace(
                    height_mm=height, widths_mm=dict(widths), base_radius_mm=widths[min(widths)] / 2,
                    slant_deg=20.0, base_world_m=np.asarray(xy, float))}
        return {"ok": False, "why": "nothing of its own outlined where it should be seen"}
    return Vision(measure=measure), asked


def test_a_piece_its_neighbours_feet_crowd_is_measured_from_over_it_and_picked_when_it_fits(segmenter):
    queen = Cylinder(D1, 0.0224, 0.0686)
    arm = LocatedArm([queen])
    vision, asked = _measures({"target": (68.6, QUEEN, D1), "c1": (61.7, BISHOP, C1)})
    got = _located().pick(arm, Target(on_table(D1), foot_m=0.0266, max_height_m=0.082), _back_rank(), vision=vision)
    assert got.ok and arm.held is queen, got.message
    assert any("By the feet the scan saw" in n and "white_piece_c1 leaves 27 mm" in n for n in got.notes)
    assert any("Measured white_piece_c1 from over it: 62 mm tall" in n for n in got.notes)
    assert any("Room for the jaws from" in n for n in got.notes)
    assert min(p[2] for p in arm.path) < 0.04           # went down to it
    assert "0:white_piece_c1" in asked and not any("e1" in (a or "") for a in asked)   # e1 leaves room by its foot
    assert set(got.around) == {"white_piece_c1"} and got.around["white_piece_c1"].height_mm == 61.7


def test_a_piece_still_crowded_once_measured_is_refused_without_going_down(segmenter):
    fat = {k: 26.0 for k in BISHOP}                     # as wide all the way up as at its foot
    arm = LocatedArm([Cylinder(D1, 0.0224, 0.0686)])
    vision, _ = _measures({"target": (68.6, QUEEN, D1), "c1": (61.7, fat, C1)})
    got = _located().pick(arm, Target(on_table(D1), foot_m=0.0266, max_height_m=0.082), _back_rank(), vision=vision)
    assert not got.ok and got.refused and arm.held is None
    assert "white_piece_c1 leaves" in got.message and "as measured from over it" in got.message
    assert min(p[2] for p in arm.path) > 0.082          # looked from above everything; never went down
    assert got.measured is not None and "white_piece_c1" in got.around   # both kept, for the next try


def test_a_neighbour_the_camera_cannot_measure_is_kept_clear_of_by_its_foot(segmenter):
    arm = LocatedArm([Cylinder(D1, 0.0224, 0.0686)])
    vision, _ = _measures({"target": (68.6, QUEEN, D1)})
    got = _located().pick(arm, Target(on_table(D1), foot_m=0.0266, max_height_m=0.082), _back_rank(), vision=vision)
    assert not got.ok and got.refused and arm.held is None
    assert any("Could not measure white_piece_c1 from here" in n for n in got.notes)


def test_feet_that_would_leave_no_gap_however_slim_above_are_refused_before_moving(segmenter):
    from mrcf.policies.base import Obstacle
    arm = LocatedArm([Cylinder((0.20, 0.0), 0.021, 0.045)])
    crowded = [Obstacle(np.array([0.20, 0.025]), 0.012, "a neighbour"),
               Obstacle(np.array([0.20, -0.025]), 0.012, "another")]
    got = _located().pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, max_height_m=0.045), crowded,
                          vision=_measures({})[0])
    assert got.refused and "leaves" in got.message and arm.path == []


def test_a_measured_piece_crowded_by_neighbours_no_camera_can_measure_is_refused_before_moving():
    arm = FakeArm([Cylinder(D1, 0.0224, 0.0686)])       # its wrist camera has no pose: it sees no neighbours
    got = _located().pick(arm, Target(on_table(D1), foot_m=0.0266, height_m=0.0686, widths_mm=QUEEN),
                          _back_rank(), vision=_measures({})[0])
    assert got.refused and "white_piece_c1 leaves" in got.message and arm.path == []


def test_a_neighbour_counts_as_wide_as_its_measured_shape_from_the_finger_tips_up():
    from mrcf.policies.base import Obstacle
    pol = ParallelJaw(SPEC)
    foot = Obstacle(np.array(C1), 0.01215, "c1")
    shaped = dataclasses.replace(foot, widths_mm=BISHOP)
    short = dataclasses.replace(foot, widths_mm={4: 12.0, 8: 12.0, 12: 11.0})   # a part's outline: its head alone
    assert pol.half_width_at(foot, 0.034) == pol.half_width_at(shaped) == 0.01215     # no shape, or no height
    assert pol.half_width_at(shaped, 0.034) == pytest.approx(0.0094)                    # 18.8 mm from 32 mm up
    assert pol.half_width_at(short, 0.034) == 0.01215                                   # stops short: its foot
    by_feet, _ = pol.jaw_room(FakeArm(), D1, [foot], 0.034)
    by_shape, _ = pol.jaw_room(FakeArm(), D1, [shaped], 0.034)
    assert by_shape - by_feet == pytest.approx(2 * (0.01215 - 0.0094))


# ── Looking from beside it, for what cannot be measured from over it ──────────
# Live, 2026-11-03: from over the d1 queen the wrist camera could tell neither
# her height nor the c1 bishop's (a part of it, 13 mm off).

def _seen_from_beside(arm, shapes, beside_m=0.02):
    """
    Vision.measure that measures each of shapes ({key fragment: (height_mm,
    widths_mm, foot_xy)}) only with the gripper at least beside_m from where
    it stands; from over it (the arm's pick hover), as the rig did: no height.
    """
    looks = []

    def measure(frame, expected_px=None, model=None, expected_xy_m=None, known=None, target=None, need_height=True):
        here = arm.get_ee_pos()[:2].copy()
        looks.append((target, here))
        for part, (height, widths, xy) in shapes.items():
            if part not in (target or "target"):
                continue
            if target == "target" and not need_height:     # where its foot is: found from over it
                return {"ok": True, "measurement": SimpleNamespace(
                    height_mm=None, widths_mm={}, base_radius_mm=13.0, slant_deg=20.0,
                    base_world_m=np.asarray(xy, float))}
            if np.linalg.norm(here - np.asarray(D1)) < beside_m:
                return {"ok": False, "why": "its outline was found but the view cannot tell its height"}
            return {"ok": True, "measurement": SimpleNamespace(
                height_mm=height, widths_mm=dict(widths), base_radius_mm=widths[min(widths)] / 2,
                slant_deg=20.0, base_world_m=np.asarray(xy, float))}
        return {"ok": False, "why": "nothing of its own outlined where it should be seen"}
    return Vision(measure=measure), looks


def test_what_cannot_be_measured_from_over_it_is_looked_at_from_beside_it_and_then_picked(segmenter):
    queen = Cylinder(D1, 0.0224, 0.0686)
    arm = LocatedArm([queen])
    vision, looks = _seen_from_beside(arm, {"target": (68.6, QUEEN, D1), "c1": (61.7, BISHOP, C1)})
    got = _located().pick(arm, Target(on_table(D1), foot_m=0.0266, max_height_m=0.082), _back_rank(), vision=vision)
    assert got.ok and arm.held is queen, got.message
    said = next(n for n in got.notes if "Looked from" in n)
    assert "measured it, white_piece_c1" in said, said
    assert any("Measured white_piece_c1 from beside it: 62 mm tall" in n for n in got.notes)
    assert got.measured is not None and got.measured.height_mm == 68.6
    away = [p for p in arm.path if np.linalg.norm(p[:2] - np.asarray(D1)) > 0.03 and p[2] > 0.08]
    assert away, "never looked from beside it"
    beside = [here for target, here in looks if target == "target" and np.linalg.norm(here - np.asarray(D1)) > 0.03]
    assert beside                                          # the queen measured from a spot beside her
    low = next(i for i, p in enumerate(arm.path) if p[2] < 0.05)
    assert np.linalg.norm(arm.path[low][:2] - np.asarray(D1)) < 0.005   # and went down over her


def test_what_cannot_be_measured_from_anywhere_is_refused_after_looking_around_without_going_down(segmenter):
    arm = LocatedArm([Cylinder(D1, 0.0224, 0.0686)])
    vision, looks = _seen_from_beside(arm, {"target": (68.6, QUEEN, D1)}, beside_m=1.0)   # no height anywhere
    pol = _located()
    got = pol.pick(arm, Target(on_table(D1), foot_m=0.0266, max_height_m=0.082), _back_rank(), vision=vision)
    assert got.refused and not got.ok and arm.held is None
    assert any("Looked from" in n and "could not measure it" in n for n in got.notes)
    assert min(p[2] for p in arm.path) > 0.082                          # never went down
    spots = {tuple(np.round(here, 3)) for target, here in looks if np.linalg.norm(here - np.asarray(D1)) > 0.03}
    assert 0 < len(spots) <= pol.spec.look_around_max
    assert np.linalg.norm(arm.get_ee_pos()[:2] - np.asarray(D1)) < 0.005   # back over it before it ends


def test_a_pick_told_not_to_look_around_refuses_from_over_it():
    arm = LocatedArm([Cylinder(D1, 0.0224, 0.0686)])
    vision, _ = _seen_from_beside(arm, {"target": (68.6, QUEEN, D1), "c1": (61.7, BISHOP, C1)})
    from mrcf.perception import segment
    segment.available, was = (lambda: True), segment.available
    try:
        got = _located(look_around_max=0).pick(arm, Target(on_table(D1), foot_m=0.0266, max_height_m=0.082),
                                               _back_rank(), vision=vision)
    finally:
        segment.available = was
    assert got.refused and not any("Looked from" in n for n in got.notes)


def test_an_unmeasured_object_is_grasped_as_high_as_the_shortest_it_could_be_and_opened_by_how_slim_it_is():
    pol = ParallelJaw(SPEC)
    t = Target(on_table(D1), foot_m=0.026, max_height_m=0.082)
    low = pol.plan(t)                                               # nothing more said: as low as the tallest allows
    assert low.grasp_height_m == pytest.approx(0.082 - SPEC.free_depth_m) and low.width_m == 0.026
    slim = pol.plan(dataclasses.replace(t, min_height_m=0.045,
                                        assumed_widths_mm={0: 26.0, 16: 26.0, 20: 23.0, 24: 22.6, 82: 22.6}))
    assert slim.grasp_height_m == pytest.approx(0.035)              # the pads still on a 45 mm one
    assert slim.width_m == pytest.approx(0.0226)                    # from its tips (28 mm) up
    assert slim.hover_height_m == low.hover_height_m                # still hovered clear of the tallest
    wide = pol.plan(dataclasses.replace(t, assumed_widths_mm={0: 40.0, 82: 40.0}))
    assert wide.width_m == 0.026                                    # never wider than its foot


# ── Turning the jaws where the neighbours stand diagonally to them ────────────
# 2026-11-03, the full chess board: the f1 bishop, 21 deg round from the arm's
# straight-ahead line, between the e1 king and the g1 knight (the 17:3x scan).

F1, E1, G1, F2 = (0.1660, -0.0500), (0.1580, -0.0130), (0.1680, -0.0890), (0.2020, -0.0450)


class _TurningArm(FakeArm):
    """A FakeArm whose orientation search turns the jaws when asked (OrientationSearch), and says what it was asked."""

    def __init__(self, objects=()):
        super().__init__(objects)
        self.turns = []

    def approach_orientation_for(self, targets, gap_m=None, tols=None, flip=False, prefer_tilt_deg=None,
                                 margin_deg=None, turn_deg=0.0):
        self.turns.append(turn_deg)
        return np.eye(3), 0.0


def _back_rank_f1():
    from mrcf.policies.base import Obstacle
    slim = {0: 1.0, 16: 1.0, 20: 0.89, 24: 0.87, 82: 0.87}
    return [Obstacle(np.array(xy), foot / 2, name, 0.082, widths_mm={h: s * foot * 1000 for h, s in slim.items()})
            for name, xy, foot in (("e1", E1, 0.027), ("g1", G1, 0.024), ("f2", F2, 0.021))]


def test_neighbours_diagonal_to_the_jaws_are_cleared_by_turning_them():
    bishop = Cylinder(F1, 0.020, 0.062)
    arm = _TurningArm([bishop])
    target = Target(on_table(F1), foot_m=0.027, max_height_m=0.082, min_height_m=0.045,
                    assumed_widths_mm={0: 27.0, 16: 27.0, 20: 24.0, 24: 23.4, 82: 23.4})
    pol = ParallelJaw(SPEC)
    assert pol.fit(arm, pol.plan(target), F1, _back_rank_f1())[1] is not None      # as the arm opens them: no room
    got = pol.pick(arm, target, _back_rank_f1())
    assert got.ok and arm.held is bishop, got.message
    said = next(n for n in got.notes if "turning the jaws" in n)
    turn = float(said.split("turning the jaws ")[1].split(" deg")[0])
    assert turn in SPEC.jaw_turns_deg and arm.turns and set(arm.turns) == {turn}   # every hold, turned so


def test_an_arm_that_cannot_turn_its_jaws_is_refused_as_before():
    arm = FakeArm([Cylinder(F1, 0.020, 0.062)])                                  # no orientation search
    target = Target(on_table(F1), foot_m=0.027, max_height_m=0.082, min_height_m=0.045,
                    assumed_widths_mm={0: 27.0, 16: 27.0, 20: 24.0, 24: 23.4, 82: 23.4})
    got = ParallelJaw(SPEC).pick(arm, target, _back_rank_f1())
    assert got.refused and arm.path == []


# ── Leaning far over: what the gripper passes above ───────────────────────────

def _leaning(tilt_deg):
    """A gripper orientation leaning tilt_deg out along +x (the wrist toward the base at the origin)."""
    t = np.radians(tilt_deg)
    z = np.array([np.sin(t), 0.0, -np.cos(t)])          # the approach axis, at the jaws
    y = np.array([0.0, 1.0, 0.0])
    return np.column_stack([np.cross(y, z), y, z])


def test_a_steep_lean_over_the_piece_in_front_is_seen_to_hit_it_and_a_gentler_one_to_clear_it():
    # 2026-11-04: rank 8 reaches at 60-80 deg, the rank-7 pawn 40 mm in front.
    from mrcf.policies.base import Obstacle
    pol = ParallelJaw(dataclasses.replace(SPEC, jaw_axis="radial"))
    at = np.array([0.40, 0.0, 0.03])
    pawn = Obstacle(np.array([0.36, 0.0]), 0.011, "the pawn in front", 0.045)
    gap, near = pol.sweep_clearance(FakeArm(), _leaning(70), at, [pawn])
    assert near == "the pawn in front" and gap < -0.010
    gap, _ = pol.sweep_clearance(FakeArm(), _leaning(45), at, [pawn])
    assert gap > 0.003
    # Straight down passes over nothing; beside the jaws is jaw_room's, not this.
    assert pol.sweep_clearance(FakeArm(), _leaning(0), at, [pawn]) is None
    beside = Obstacle(np.array([0.40, 0.04]), 0.011, "beside", 0.045)
    assert pol.sweep_clearance(FakeArm(), _leaning(70), at, [beside]) is None
    # A height nobody knows is not guessed here (the deployment gives the tallest it may be).
    assert pol.sweep_clearance(FakeArm(), _leaning(70), at, [Obstacle(np.array([0.36, 0.0]), 0.011, "?")]) is None
    # Tangential jaws open across the line: a pawn on it, narrower than the open
    # fingers and within their length, passes between them; one off the line does not.
    tangential = ParallelJaw(SPEC)
    assert tangential.sweep_clearance(FakeArm(), _leaning(70), at, [pawn]) is None
    off_line = Obstacle(np.array([0.36, 0.015]), 0.011, "off the line", 0.045)
    assert tangential.sweep_clearance(FakeArm(), _leaning(70), at, [off_line])[0] < 0


def test_an_object_a_place_needs_held_higher_is_planned_gripped_that_high_within_its_top():
    # A place refused because, leaning, the gripper would pass too low over the
    # piece in front (Outcome.regrip_above_m): the next pick grips it higher.
    pol = ParallelJaw(SPEC)
    low = pol.plan(Target(on_table((0.4, 0.0)), foot_m=0.022, height_m=0.055))
    high = pol.plan(Target(on_table((0.4, 0.0)), foot_m=0.022, height_m=0.055, grasp_above_m=0.040))
    assert low.grasp_height_m < 0.040 and high.grasp_height_m == pytest.approx(0.040)
    capped = pol.plan(Target(on_table((0.4, 0.0)), foot_m=0.022, height_m=0.055, grasp_above_m=0.060))
    assert capped.grasp_height_m == pytest.approx(0.055 - SPEC.grip_over_top_m)


def test_leaning_tangential_jaws_grip_higher_by_how_far_the_pads_low_ends_drop():
    # 2026-11-04, e7 at -55 deg: the pads' low ends closed on a pawn's base.
    pol = ParallelJaw(SPEC)                                   # tangential
    plan = pol.plan(Target(on_table((0.4, 0.0)), foot_m=0.022, height_m=0.041))
    leaned = pol.pads_clear_at(plan, -55.0)
    drop = SPEC.finger_half_width_m * np.sin(np.radians(55.0))
    assert leaned.grasp_height_m == pytest.approx(min(plan.grasp_height_m + drop, 0.041 - SPEC.grip_over_top_m))
    assert pol.pads_clear_at(plan, 0.0) is plan and pol.pads_clear_at(plan, None) is plan
    radial = ParallelJaw(dataclasses.replace(SPEC, jaw_axis="radial"))
    assert radial.pads_clear_at(plan, -55.0) is plan          # _top_clear's, for radial jaws
