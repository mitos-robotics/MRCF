"""The two-finger policies on a simulated arm: sizing, pick, place, refusing, missing, stopping."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from fake_arm import Cylinder, FakeArm
from mrcf.interlock import MotionStopped
from mrcf.policies.base import Obstacle, Target, Vision
from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw

# The SO-101's gripper (robot_config), without a wrist camera.
SPEC = GripperSpec(max_open_m=0.045, tip_below_grasp_m=0.007, free_depth_m=0.055,
                   finger_thickness_m=0.009, finger_half_width_m=0.008, jaw_axis="tangential",
                   hover_height_m=0.06, lift_height_m=0.07, lift_clearance_m=0.05)


def on_table(xy, z=0.0):
    return np.array([xy[0], xy[1], z])


# A 41 mm pawn as the wrist camera measured it on d2 (2026-10-21): a base
# flaring out below 20 mm, a straight neck above.
PAWN_D2 = {4: 22.1, 8: 21.1, 12: 19.1, 16: 17.2, 20: 15.5, 24: 15.5, 28: 15.4, 32: 14.3, 36: 12.7}
# A king as a Staunton set's is shaped: a long taper, a collar ridge at 48 mm,
# the crown above it.
KING = {4: 32, 8: 31, 12: 28, 16: 25, 20: 23, 24: 21, 28: 19, 32: 18, 36: 17, 40: 16.5, 44: 17, 48: 20,
        52: 17, 56: 16, 60: 18, 64: 20, 68: 19, 72: 9, 76: 9}


def test_where_the_sides_are_straight_the_grasp_is_at_the_preferred_height():
    pol = ParallelJaw(SPEC)
    straight = {z: 20.0 for z in range(4, 44, 4)}
    g, _ = pol.grasp_height_for(0.044, straight)
    assert g == pytest.approx(SPEC.grasp_height_fraction * 0.044, abs=0.001)


def test_a_flared_base_is_not_where_the_jaws_close():
    pol = ParallelJaw(SPEC)
    # 0.45 of it is 18 mm, on the flare, where the closing jaws squeezed it out.
    g, why = pol.grasp_height_for(0.0409, PAWN_D2)
    assert 0.022 <= g <= 0.026, why
    plan = pol.plan_from_shape(0.0409, PAWN_D2, 0.021)
    assert plan.grasp_height_m == pytest.approx(g)
    assert plan.grip_width_m == pytest.approx(0.0155, abs=0.0005)
    assert (plan.hover_height_m, plan.lift_height_m) == pytest.approx((0.06, 0.07 + g - 0.02))


def test_a_ridge_is_not_where_the_jaws_close():
    pol = ParallelJaw(GripperSpec(**{**SPEC.__dict__, "grasp_height_fraction": 0.6}))
    # 0.6 of the king is 49 mm: on its collar. Below it, on the neck.
    g, why = pol.grasp_height_for(0.082, KING)
    assert 0.036 <= g <= 0.042, why
    plan = pol.plan_from_shape(0.082, KING, 0.032)
    assert plan.hover_height_m == pytest.approx(0.097)   # jaw tips clear its top
    assert plan.height_m - plan.grasp_height_m <= SPEC.free_depth_m


def test_with_nothing_straight_the_straightest_spot_within_reach():
    pol = ParallelJaw(SPEC)
    cone = {z: 30.0 - 0.4 * z for z in range(4, 52, 4)}
    g, why = pol.grasp_height_for(0.052, cone)
    assert "straightest" in why
    assert SPEC.min_grasp_height_m <= g <= 0.052 - 0.010


def test_an_outline_cut_short_does_not_shape_the_grasp():
    pol = ParallelJaw(SPEC)
    assert pol.usable_widths({8: 0.7, 12: 2.8, 16: 30.9, 20: 31.2, 24: 29.9}) == [
        (16.0, 30.9), (20.0, 31.2), (24.0, 29.9)]
    # Too little left to choose by: the preferred height.
    g, why = pol.grasp_height_for(0.044, {8: 0.7, 12: 2.8, 16: 30.9})
    assert g == pytest.approx(SPEC.grasp_height_fraction * 0.044) and "no profile" in why


def test_a_short_object_never_has_the_jaw_tips_put_into_the_surface():
    pol = ParallelJaw(SPEC)
    g, _ = pol.grasp_height_for(0.012, {4: 30.0, 8: 30.0})
    assert g - SPEC.tip_below_grasp_m > 0


def test_an_unmeasured_object_is_approached_as_the_tallest_it_could_be():
    pol = ParallelJaw(SPEC)
    plan = pol.plan(Target(on_table((0.2, 0.0)), foot_m=0.021, max_height_m=0.082))
    assert plan.hover_height_m == pytest.approx(0.082 + 0.007 + 0.008)
    # As low as an object that tall allows: its top stays out of the gripper.
    assert plan.grasp_height_m == pytest.approx(0.082 - 0.055)
    assert plan.pregrasp_open_m == pytest.approx(0.036)


def test_pick_then_place_moves_the_object():
    pawn = Cylinder((0.20, 0.02), 0.021, 0.045)
    arm, pol = FakeArm([pawn]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.082))
    assert got.ok, got.message
    assert arm.held is pawn and got.grasp is not None
    put = pol.place(arm, on_table((0.25, -0.03)), grasp=got.grasp)
    assert put.ok, put.message
    assert arm.held is None
    assert np.allclose(pawn.xy, (0.25, -0.03), atol=1e-6)


def test_no_room_between_neighbours_is_refused_without_moving():
    arm, pol = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.045)]), ParallelJaw(SPEC)
    across = pol.jaw_axis(arm, (0.20, 0.0))
    neighbour = Obstacle(np.array([0.20, 0.0]) + across * 0.022, 0.0105, "black_pawn_e5")
    got = pol.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021), obstacles=[neighbour])
    assert not got.ok and got.refused
    assert "black_pawn_e5" in got.message
    assert arm.cycles == 0


def test_a_neighbour_off_the_fingers_path_does_not_count():
    arm, pol = FakeArm(), ParallelJaw(SPEC)
    across = pol.jaw_axis(arm, (0.20, 0.0))
    beside = np.array([-across[1], across[0]])          # perpendicular to the jaws
    room, limit = pol.jaw_room(arm, (0.20, 0.0), [Obstacle(np.array([0.20, 0.0]) + beside * 0.03, 0.0105)])
    assert limit is None and room == SPEC.max_open_m


def test_nothing_there_retries_then_fails():
    arm, pol = FakeArm(), ParallelJaw(dataclasses.replace(SPEC, blind_retry_after_miss=True))     # retried blind: the retries are the test
    got = pol.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, max_height_m=0.05))
    assert not got.ok and not got.refused
    assert "after 3 attempts" in got.message


def test_a_failed_pick_says_why_each_attempt_was_retried():
    arm, pol = FakeArm(), ParallelJaw(dataclasses.replace(SPEC, blind_retry_after_miss=True))     # retried blind: the retries are the test
    got = pol.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, max_height_m=0.05))
    retried = [n for n in got.notes if "retrying" in n]
    assert [n[:10] for n in retried] == ["[Pick 1/3]", "[Pick 2/3]", "[Pick 3/3]"]
    assert all("Closed on nothing" in n for n in retried)
    assert got.notes[-1].startswith("[Pick] ")               # and where the time went
    # Nothing carried over into the next pick's account.
    again = pol.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.021, max_height_m=0.05))
    assert len([n for n in again.notes if "retrying" in n]) == 3


def test_a_measurement_at_the_hover_resizes_the_grasp(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm = FakeArm([pawn])
    pol = ParallelJaw(GripperSpec(**{**SPEC.__dict__, "camera": "wrist"}))
    looked = SimpleNamespace(height_mm=43.0, widths_mm={0: 21, 10: 15, 20: 16, 43: 8},
                             base_radius_mm=10.5, slant_deg=30.0, base_world_m=None)
    vision = Vision(measure=lambda frame, expected_px=None, model=None, expected_xy_m=None:
                    {"ok": True, "measurement": looked})
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.082), vision=vision)
    assert got.ok, got.message
    assert got.measured.height_mm == 43.0 and len(got.measured.frames) == 2
    assert got.grasp.height_m == pytest.approx(0.043)        # sized from what it saw


class _StopsShortOnce(FakeArm):
    """The first long move from home stops 5 cm short, as c5's and d5's hovers did (2026-10-28)."""
    def __init__(self, objects=()):
        super().__init__(objects)
        self.stopped = 0

    def move_ee_to(self, target, *args, **kwargs):
        target = np.asarray(target, float)
        if not self.stopped and np.linalg.norm(target - self.ee) > 0.06:
            self.stopped += 1
            short = target + (self.ee - target) / np.linalg.norm(self.ee - target) * 0.05
            super().move_ee_to(short, *args, **kwargs)
            return False
        return super().move_ee_to(target, *args, **kwargs)


def test_a_hover_that_stops_well_short_is_asked_for_once_more_before_the_attempt_is_given_up():
    pawn = Cylinder((0.30, 0.0), 0.021, 0.045)
    arm = _StopsShortOnce([pawn])
    got = ParallelJaw(SPEC).pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.05))
    assert got.ok, got.message
    assert arm.stopped == 1 and [a["result"] for a in got.attempts] == ["ok"]


class _OneJawArm(FakeArm):
    """Tells the policy where an object of a given width would be held (HeldWidth), and records it."""
    def __init__(self, objects=()):
        super().__init__(objects)
        self.told = []                          # (width, grasp-point height) at each hold_width

    def hold_width(self, width_m):
        self.told.append((width_m, float(self.ee[2])))


def test_the_object_comes_down_against_the_jaw_that_does_not_move():
    # Centred, the moving jaw pushed a rook 11 mm across before holding it; it
    # tipped and slid out under the fixed jaw, twice on d3 (2026-10-28).
    rook = Cylinder((0.20, 0.0), 0.020, 0.054)
    arm = _OneJawArm([rook])
    got = ParallelJaw(SPEC).pick(arm, Target(on_table(rook.xy), foot_m=0.020, height_m=0.054))
    assert got.ok, got.message
    (aimed, at_z), (held, _) = arm.told[0], arm.told[-1]
    assert aimed == pytest.approx(got.grasp.width_m + 2 * SPEC.anvil_margin_m)
    assert at_z > got.grasp.grasp_height_m + 0.02            # told before coming down
    assert held == pytest.approx(got.grasp.grip_width_m)     # and the real width once it held it
    centred = ParallelJaw(dataclasses.replace(SPEC, anvil_margin_m=None))
    arm = _OneJawArm([Cylinder((0.20, 0.0), 0.020, 0.054)])
    assert centred.pick(arm, Target(on_table((0.20, 0.0)), foot_m=0.020, height_m=0.054)).ok
    assert len(arm.told) == 1                                 # only once it held it


def test_a_look_whose_foot_the_scan_does_not_bear_out_is_not_grasped_by(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    rook = Cylinder((0.20, 0.0), 0.020, 0.050)
    arm = FakeArm([rook])
    pol = ParallelJaw(GripperSpec(**{**SPEC.__dict__, "camera": "wrist"}))
    # f7, leaning 40 deg (2026-10-28): both frames read 29 mm tall on an 8 mm
    # foot, where the scan measured 18; gripped at 17 mm it fell over when set down.
    junk = SimpleNamespace(height_mm=29.0, widths_mm={4: 9, 12: 8, 20: 8, 28: 6},
                           base_radius_mm=4.0, slant_deg=30.0, base_world_m=None)
    vision = Vision(measure=lambda frame, expected_px=None, model=None, expected_xy_m=None:
                    {"ok": True, "measurement": junk})
    got = pol.pick(arm, Target(on_table(rook.xy), foot_m=0.018, max_height_m=0.082), vision=vision)
    assert got.ok, got.message
    assert got.measured is None                               # not kept, not remembered
    assert got.grasp.grasp_height_m == pytest.approx(0.082 - SPEC.free_depth_m)   # as for the unmeasured
    assert any("not trusting it" in n for n in got.notes)


def test_a_stop_mid_descent_holds_the_arm_and_ends_the_pick():
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm, pol = FakeArm([pawn]), ParallelJaw(SPEC)
    # As a hand detector would, the moment the jaws come down near the board.
    arm.interlock.add_check("hands", lambda: "hand near the arm" if arm.ee[2] < 0.05 else None)
    with pytest.raises(MotionStopped):
        pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.05))
    assert arm.holds >= 1
    assert arm.held is None and arm.ee[2] >= 0.04       # stopped on the way down, not after


class _ServoArm(FakeArm):
    """A FakeArm with a calibrated wrist-camera servo that records where it ran."""
    wrist_servo_target_px = (381.0, 263.0)

    def __init__(self, objects=()):
        super().__init__(objects)
        self.servo_heights = []
        self.servo_openings = []
        self.servo_points = []

    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        self.servo_heights.append(round(float(self.ee[2]), 3))
        self.servo_openings.append(round(float(self.width), 3))
        self.servo_points.append(None if point_height_m is None else round(point_height_m, 4))
        return True, "aligned"


def _servoing(spec_overrides=None):
    return ParallelJaw(GripperSpec(**{**SPEC.__dict__, "camera": "wrist", "servo_height_m": 0.06,
                                      **(spec_overrides or {})}))


def _measuring(height_mm, foot_mm=21.0):
    looked = SimpleNamespace(height_mm=height_mm, widths_mm={0: 21, 10: 15, 20: 16, int(height_mm): 8},
                             base_radius_mm=foot_mm / 2, slant_deg=30.0, base_world_m=None)
    return Vision(measure=lambda frame, expected_px=None, model=None, expected_xy_m=None:
                  {"ok": True, "measurement": looked},
                  find=lambda frame, expected_px, radius_px: expected_px)


def test_the_servo_aims_only_from_the_height_it_was_calibrated_at(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.021, 0.043)
    arm = _ServoArm([pawn])
    got = _servoing().pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.082),
                           vision=_measuring(43.0))
    assert got.ok, got.message
    # Found from the hover that clears the tallest piece, then aligned for real at 60 mm.
    assert arm.servo_heights == [0.097, 0.06]


def test_a_tall_piece_is_aligned_from_the_servo_height_with_its_top_between_the_jaws(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    king = Cylinder((0.20, 0.0), 0.030, 0.082)
    arm = _ServoArm([king])
    got = _servoing().pick(arm, Target(on_table(king.xy), foot_m=0.030, max_height_m=0.082),
                           vision=_measuring(82.0, foot_mm=30.0))
    assert got.ok, got.message
    # Found and measured from the high hover, aligned from the servo's own height.
    assert arm.servo_heights == [0.097, 0.06]
    assert arm.held is king


class _TwoWayArm(FakeArm):
    """Reaches anything straight down, jaws along world y, either way round."""
    def approach_orientation_for(self, targets, tols=None, flip=False):
        x = np.array([0.0, -1.0, 0.0]) if flip else np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, -1.0])
        return np.column_stack([x, np.cross(z, x), z]), 0.0


def test_the_wrist_turns_to_keep_the_camera_off_a_neighbour():
    pol = ParallelJaw(GripperSpec(**{**SPEC.__dict__, "camera_body_m": (-0.075, 0.0, 0.02)}))
    arm, at = _TwoWayArm(), np.array([0.20, 0.0, 0.06])
    # Unturned, the camera hangs out 75 mm toward -y: right where this piece stands.
    beside = [Obstacle(np.array([0.20, -0.075]), 0.012, "black_piece_g3")]
    rot = pol.hold_orientation(arm, [("hover", at, 0.01)], obstacles=beside)
    assert rot[1, 0] < 0                                  # jaw axis turned to -y: camera to +y
    assert pol.body_clearance(arm, rot, at, beside)[0] > 0.1
    # With nothing beside it, it stays as it is.
    assert pol.hold_orientation(arm, [("hover", at, 0.01)])[1, 0] > 0


def test_a_look_at_some_other_object_is_not_descended_on(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.022, 0.041)
    arm = _ServoArm([pawn])
    # Sent for a 41 mm pawn with a 22 mm foot; the camera keeps seeing a 54 mm, 28 mm neighbour.
    got = _servoing({"remeasure_known": True}).pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.041),
                                                    vision=_measuring(54.0, foot_mm=28.0))
    assert not got.ok and arm.held is None
    assert min(p[2] for p in arm.path) > 0.05                # never went down to grasp


def test_a_shape_one_look_made_hovers_clear_of_the_tallest_it_could_be():
    pol = ParallelJaw(SPEC)
    one = Target(on_table((0.2, 0.0)), foot_m=0.021, height_m=0.0195, widths_mm={4: 21.5, 8: 20.1, 12: 17.9},
                 max_height_m=0.082, shape_looks=1)
    assert pol.plan(one).hover_height_m == pytest.approx(0.082 + 0.007 + 0.008)
    settled = dataclasses.replace(one, shape_looks=2)
    assert pol.plan(settled).hover_height_m == pytest.approx(0.06)
    assert pol.plan(dataclasses.replace(one, shape_looks=None)).hover_height_m == pytest.approx(0.06)


def test_a_shape_one_look_made_is_looked_at_again_and_a_disagreeing_look_wins(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.022, 0.041)
    arm = _ServoArm([pawn])
    # Remembered as 59.6 mm from one bad look; the camera now sees 41.
    got = _servoing().pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.0596,
                                       widths_mm={16: 30.9, 20: 31.2, 24: 29.9, 28: 28.8}, max_height_m=0.082,
                                       shape_looks=1),
                           vision=_measuring(41.0, foot_mm=22.0))
    assert got.ok, got.message
    assert got.measured is not None and got.measured.height_mm == pytest.approx(41.0)
    assert got.grasp.height_m == pytest.approx(0.041)
    assert not any("not the one sent for" in n for n in got.notes)


def test_a_disagreeing_look_is_looked_at_a_third_time_and_without_two_that_agree_the_earlier_stands(monkeypatch):
    # 2026-10-30: the coarse look from high up read a 39 mm pawn as 17; the
    # look at the servo's height had the gripper over its top and refused; the
    # pick went by the coarse 17 and closed on the pawn's foot.
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.022, 0.039)
    calls = []

    def measure(frame, expected_px=None, model=None, expected_xy_m=None):
        calls.append(1)
        h = 17.0
        return {"ok": True, "occluded": len(calls) > 2,
                "measurement": SimpleNamespace(height_mm=h, widths_mm={0: 22, 10: 15, int(h): 8},
                                               base_radius_mm=11.0, slant_deg=30.0, base_world_m=None)}
    got = _servoing().pick(_ServoArm([pawn]), Target(on_table(pawn.xy), foot_m=0.022, height_m=0.039,
                                                     widths_mm=PAWN_D2, max_height_m=0.082, shape_looks=1),
                           vision=Vision(measure=measure, find=lambda frame, expected_px, radius_px: expected_px))
    assert got.ok, got.message
    assert any("no two agree; going by the earlier" in n for n in got.notes)
    assert got.grasp.height_m == pytest.approx(0.039)           # planned by the 39 mm it was, not the 17


def test_a_third_look_that_sides_with_the_earlier_wins(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.022, 0.039)
    looks = iter([17.0, 17.0, 17.0, 17.0] + [39.0] * 20)      # coarse, then at the servo's height, then the third

    def measure(frame, expected_px=None, model=None, expected_xy_m=None):
        h = next(looks)
        return {"ok": True, "measurement": SimpleNamespace(height_mm=h, widths_mm={0: 22, 10: 15, int(h): 8},
                                                           base_radius_mm=11.0, slant_deg=30.0, base_world_m=None)}
    got = _servoing().pick(_ServoArm([pawn]), Target(on_table(pawn.xy), foot_m=0.022, height_m=0.039,
                                                     widths_mm=PAWN_D2, max_height_m=0.082, shape_looks=1),
                           vision=Vision(measure=measure, find=lambda frame, expected_px, radius_px: expected_px))
    assert got.ok, got.message
    assert any("going by the earlier and the third" in n for n in got.notes)
    assert got.measured.height_mm == pytest.approx(39.0)


def test_a_settled_shape_is_not_looked_at_again(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.022, 0.041)
    arm = _ServoArm([pawn])

    def no_look(*a, **k):
        raise AssertionError("looked at a settled shape")

    got = _servoing().pick(arm, Target(on_table(pawn.xy), foot_m=0.022, height_m=0.041, widths_mm=PAWN_D2,
                                       shape_looks=3),
                           vision=Vision(measure=no_look, find=lambda frame, expected_px, radius_px: expected_px))
    assert got.ok, got.message
    assert got.measured is None


def test_the_servo_runs_at_the_jaw_opening_it_was_calibrated_at(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.021, 0.043)
    arm = _ServoArm([pawn])
    # Unmeasured, so the plan opens the jaws 36 mm; the servo was set at 30.
    got = _servoing({"servo_open_m": 0.030}).pick(
        arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.082), vision=_measuring(43.0))
    assert got.ok, got.message
    assert arm.servo_openings == [0.030, 0.030]


def test_a_measured_object_goes_straight_to_the_servo_height(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    piece = Cylinder((0.20, 0.0), 0.025, 0.050)
    arm = _ServoArm([piece])
    got = _servoing().pick(arm, Target(on_table(piece.xy), foot_m=0.025, height_m=0.050),
                           vision=_measuring(50.0, foot_mm=25.0))
    assert got.ok, got.message
    # No look from the higher hover first: its height is already known.
    assert arm.servo_heights == [0.06]


def test_the_servo_is_told_how_high_up_the_object_its_point_is(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.021, 0.043)
    arm = _ServoArm([pawn])
    got = _servoing().pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.082),
                           vision=_measuring(43.0))
    assert got.ok, got.message
    # Height unknown for the first, coarse look; measured from it (43 mm) by the time it aligns.
    assert arm.servo_points == [None, pytest.approx(0.4 * 0.043)]


class _CorrectingServoArm(_ServoArm):
    """A servo that finds the object `error` from where the scan put it, and moves there."""
    def __init__(self, objects=(), error=(0.0, 0.0)):
        super().__init__(objects)
        self.error = np.array([*error, 0.0])

    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        # Every attempt re-approaches the scan's estimate, so it corrects by the same again.
        self.servo_heights.append(round(float(self.ee[2]), 3))
        self.ee = self.ee + self.error
        return True, "aligned"


def test_the_servo_is_trusted_to_correct_the_scan_by_a_couple_of_centimetres(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # The scan puts it 24 mm short of where it stands: what this rig's does near mid-board.
    piece = Cylinder((0.244, 0.0), 0.025, 0.050)
    arm = _CorrectingServoArm([piece], error=(0.024, 0.0))
    got = _servoing().pick(arm, Target(on_table((0.220, 0.0)), foot_m=0.025, height_m=0.050),
                           vision=_measuring(50.0, foot_mm=25.0))
    assert got.ok, got.message
    assert arm.held is piece


def test_a_servo_that_ends_nearer_a_neighbour_is_not_descended_on(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    piece = Cylinder((0.220, 0.0), 0.025, 0.050)
    neighbour = Cylinder((0.220, 0.045), 0.025, 0.050)
    # 30 mm: inside the limit, but nearer the neighbour's estimate than this one's.
    arm = _CorrectingServoArm([piece, neighbour], error=(0.0, 0.030))
    got = _servoing().pick(arm, Target(on_table(piece.xy), foot_m=0.025, height_m=0.050),
                           obstacles=[Obstacle(on_table(neighbour.xy), 0.0125, "white_pawn_d5")],
                           vision=_measuring(50.0, foot_mm=25.0))
    assert not got.ok and arm.held is None
    assert min(p[2] for p in arm.path) > 0.05                # never went down to grasp



class _LeaningArm(FakeArm):
    """Reaches anything at whatever lean it is asked to prefer; records what it was asked."""
    def __init__(self, objects=()):
        super().__init__(objects)
        self.asked = []

    def approach_orientation_for(self, targets, tols=None, flip=False, prefer_tilt_deg=None):
        self.asked.append(prefer_tilt_deg)
        a = np.radians(-(prefer_tilt_deg or 0.0))
        x, z = np.array([np.cos(a), 0.0, np.sin(a)]), np.array([np.sin(a), 0.0, -np.cos(a)])
        return np.column_stack([x, np.cross(z, x), z]), float(np.degrees(a))


def test_an_approach_rules_lean_is_where_the_arms_search_starts():
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm, pol = _LeaningArm([pawn]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.05, tilt_deg=30.0))
    assert got.ok, got.message
    assert arm.asked and set(arm.asked) == {30.0}
    assert any("approach rule" in n and "-30 deg" in n for n in got.notes)
    put = pol.place(arm, on_table((0.25, -0.03)), grasp=got.grasp, tilt_deg=20.0)
    assert put.ok and arm.asked[-1] == 20.0



class _TiltReadingArm(_LeaningArm):
    """A _LeaningArm that can say how far a gripper orientation leans."""
    def gripper_tilt_deg(self, rot=None):
        z = np.asarray(rot, float)[:, 2]
        return float(np.degrees(np.arctan2(z[0], -z[2])))


def test_a_place_leans_as_the_object_was_picked():
    # The object hangs below the jaws: picked at one lean and set down at
    # another, its foot swings — 7-9 mm out on d5, short on e4 (2026-10-20).
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm, pol = _TiltReadingArm([pawn]), ParallelJaw(SPEC)
    got = pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.05, tilt_deg=30.0))
    assert got.ok, got.message
    put = pol.place(arm, on_table((0.25, -0.03)), grasp=got.grasp, tilt_deg=20.0)
    assert put.ok and arm.asked[-1] == pytest.approx(-30.0)

def test_without_a_rule_the_arm_is_not_asked_for_a_lean():
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm, pol = _LeaningArm([pawn]), ParallelJaw(SPEC)
    assert pol.pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.05)).ok
    assert set(arm.asked) == {None}


class _NoLeanArm(FakeArm):
    """A backend whose orientation search knows nothing of a preferred lean."""
    def approach_orientation_for(self, targets, tols=None, flip=False):
        return np.column_stack([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]), 0.0


def test_a_backend_without_preferred_leans_still_gets_one_orientation():
    pol = ParallelJaw(SPEC)
    rot = pol.hold_orientation(_NoLeanArm(), [("hover", np.array([0.2, 0.0, 0.06]), 0.01)], tilt_deg=25.0)
    assert rot is not None


class _SidewaysServoArm(_LeaningArm):
    """Servo finds the object `error` sideways of the scan's estimate; records orientation requests and jaw widths."""
    wrist_servo_target_px = (381.0, 263.0)

    def __init__(self, objects=(), error=(0.0, 0.0)):
        super().__init__(objects)
        self.error = np.array([*error, 0.0])
        self.requests = []          # the grasp target of every orientation request
        self.widths_down = []       # jaw opening at each move that goes below 3 cm

    def approach_orientation_for(self, targets, tols=None, flip=False, prefer_tilt_deg=None):
        self.requests.append(np.asarray(targets[1], float).copy())
        return super().approach_orientation_for(targets, tols, flip, prefer_tilt_deg)

    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        self.ee = self.ee + self.error
        return True, "aligned"

    def move_ee_to(self, target, *args, **kwargs):
        if target[2] < 0.03:
            self.widths_down.append(round(float(self.width), 4))
        return super().move_ee_to(target, *args, **kwargs)


def test_after_a_sideways_servo_the_orientation_is_chosen_again_for_where_it_is(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # The scan put it 20 mm to one side of where it stands.
    piece = Cylinder((0.25, 0.02), 0.022, 0.045)
    arm = _SidewaysServoArm([piece], error=(0.0, 0.02))
    got = _servoing().pick(arm, Target(on_table((0.25, 0.0)), foot_m=0.022, height_m=0.045), vision=_measuring(45.0))
    assert got.ok, got.message
    assert np.allclose(arm.requests[-1][:2], (0.25, 0.02), atol=1e-6)     # the last choice was for the real spot


def test_the_jaws_go_down_at_the_opening_the_servo_aligned_at(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.0), 0.021, 0.043)
    arm = _SidewaysServoArm([pawn])
    # Unmeasured: measured at the hover, which sizes the grasp's opening below the servo's 35 mm.
    got = _servoing({"servo_open_m": 0.033}).pick(
        arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.082), vision=_measuring(43.0))
    assert got.ok, got.message
    shaped = _servoing().plan_from_shape(0.043, got.measured.widths_mm, 0.021)
    assert shaped.pregrasp_open_m < 0.033 - 0.002                 # its shape wanted narrower...
    assert arm.widths_down and arm.widths_down[0] == pytest.approx(0.033)   # ...but it went down as aligned
    assert got.grasp.pregrasp_open_m == pytest.approx(0.033)      # and the plan it hands on says so


class OffArm(FakeArm):
    """
    An arm whose kinematics are off by `delta`: it is really at ee, it thinks it
    is at ee - delta, and a fixed camera that sees its gripper says so.
    """

    def __init__(self, objects=(), delta=(0.012, -0.008, 0.0), visible=True):
        super().__init__(objects)
        self.delta = np.asarray(delta, float)
        self.visible = visible
        self.looks = 0

    def get_ee_pos(self) -> np.ndarray:
        return self.ee - self.delta

    def move_ee_to(self, target, *args, **kwargs):
        return super().move_ee_to(np.asarray(target, float) + self.delta, *args, **kwargs)

    def overhead_gripper_offset(self):
        self.looks += 1
        return self.delta.copy() if self.visible else None


def _holding(arm, pol, xy=(0.20, 0.02)):
    pawn = Cylinder(np.asarray(xy) + arm.delta[:2], 0.021, 0.045)     # where it really is
    arm.objects.append(pawn)
    arm.ee = np.array([pawn.xy[0], pawn.xy[1], 0.02])
    arm.set_gripper(0.0)
    arm.held = pawn
    return pawn, pol.plan(Target(on_table(xy), foot_m=0.021, height_m=0.045))


def test_place_lands_where_the_camera_sees_the_target_not_where_the_kinematics_think():
    arm = OffArm()
    pawn, plan = _holding(arm, ParallelJaw(SPEC))
    put = ParallelJaw(SPEC).place(arm, on_table((0.25, -0.03)), grasp=plan)
    assert put.ok, put.message
    assert np.allclose(pawn.xy, (0.25, -0.03), atol=0.002)
    # without the camera it lands off by the kinematics' error
    arm = OffArm()
    pawn, plan = _holding(arm, ParallelJaw(SPEC))
    blind = ParallelJaw(dataclasses.replace(SPEC, overhead_correct=False))
    assert blind.place(arm, on_table((0.25, -0.03)), grasp=plan).ok
    assert np.allclose(pawn.xy, np.array([0.25, -0.03]) + arm.delta[:2], atol=0.002)


class _DriftingDescentArm(OffArm):
    """An OffArm whose error grows 6 mm outward on a descent with something held, as a leaning one did."""
    def move_ee_to(self, target, *args, **kwargs):
        target = np.asarray(target, float)
        if self.held is not None and target[2] < self.ee[2] - 0.02:
            self.delta = self.delta + np.array([0.006, 0.0, 0.0])
        return super().move_ee_to(target, *args, **kwargs)


class _GraspDriftArm(OffArm):
    """An OffArm whose descent onto an object (nothing held) drifts by `drift`."""
    def __init__(self, objects=(), drift=(0.0, 0.0, 0.0)):
        super().__init__(objects)
        self.drift = np.asarray(drift, float)

    def move_ee_to(self, target, *args, **kwargs):
        target = np.asarray(target, float)
        if self.held is None and target[2] < self.ee[2] - 0.02:
            self.delta = self.delta + self.drift
        return super().move_ee_to(target, *args, **kwargs)


@pytest.mark.parametrize("drift, corrected", [((-0.0022, 0.0, 0.0), True), ((0.0, 0.0022, 0.0), False)])
def test_at_the_grasp_a_small_error_along_the_arm_is_corrected_and_one_across_is_not(drift, corrected):
    # Every queen close 2.5-2.7 mm toward the base of the pads missed, wedged
    # out that way (2026-10-28); across, the open jaws leave 6-10 mm of slack.
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)               # straight out from the base: along = x
    arm = _GraspDriftArm([pawn], drift=drift)
    got = ParallelJaw(SPEC).pick(arm, Target(on_table(pawn.xy), foot_m=0.021, height_m=0.045))
    assert got.ok, got.message
    at = got.attempts[0]["overhead_at_grasp"]
    assert max(abs(v) for v in at["aim_before_mm"]) == pytest.approx(2.2, abs=0.1)
    assert (max(abs(v) for v in at["aim_after_mm"]) < 1.0) is corrected


class _StallsAtGraspArm(_GraspDriftArm):
    """A _GraspDriftArm whose small sideways moves down at the grasp stall where they start (the SO-101 near the table)."""
    def move_ee_to(self, target, *args, **kwargs):
        target = np.asarray(target, float)
        here = self.get_ee_pos()
        if target[2] < 0.03 and abs(target[2] - here[2]) < 0.002 and np.linalg.norm(target[:2] - here[:2]) < 0.008:
            return False
        return super().move_ee_to(target, *args, **kwargs)


@pytest.mark.parametrize("reseat, held", [(0.003, True), (None, False)])
def test_jaws_left_off_its_middle_along_the_arm_go_up_over_and_down_again_rather_than_close(reseat, held):
    # 2026-11-17, a wooden figure: the correction at the grasp stalled 4.8 mm off,
    # the jaws closed anyway and squeezed the round body out toward the base.
    pawn = Cylinder((0.20, 0.0), 0.021, 0.045)
    arm = _StallsAtGraspArm([pawn], drift=(-0.005, 0.0, 0.0))
    spec = dataclasses.replace(SPEC, grasp_reseat_along_m=reseat, retries=1)
    got = ParallelJaw(spec).pick(arm, Target(on_table(pawn.xy), foot_m=0.021, height_m=0.045))
    assert got.ok is held, got.message
    if held:
        assert abs(got.attempts[0]["overhead_at_grasp"]["aim_after_mm"][0]) <= 3.0


def test_a_place_puts_the_jaws_back_over_the_target_when_the_descent_drifts():
    # Leaning 20 deg onto d5 (2026-10-28) the descent drifted out and the queen
    # landed 16 mm past the centre; nothing checked it before the jaws opened.
    arm = _DriftingDescentArm()
    pawn, plan = _holding(arm, ParallelJaw(SPEC))
    put = ParallelJaw(SPEC).place(arm, on_table((0.25, -0.03)), grasp=plan)
    assert put.ok, put.message
    assert np.linalg.norm(pawn.xy - np.array([0.25, -0.03])) <= SPEC.grasp_settle_tol_m


class _StaleLoadArm(OffArm):
    """Its gripper's load reads high once more just after the jaws open, as the SO-101's did."""
    def set_gripper(self, width, *args, **kwargs):
        super().set_gripper(width, *args, **kwargs)
        self.stale = width > 0.0

    def gripper_loaded(self) -> bool:
        stale, self.stale = getattr(self, "stale", False), False
        return stale or super().gripper_loaded()


def test_a_load_read_just_after_the_jaws_opened_is_read_again(monkeypatch):
    # On d4 (2026-10-28) the jaws stood at 35 mm round a 21 mm queen and the
    # load still read high: the place refused to retreat, and the next move
    # swept the open jaws through the queen and knocked it over.
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    arm = _StaleLoadArm()
    pawn, plan = _holding(arm, ParallelJaw(SPEC))
    put = ParallelJaw(SPEC).place(arm, on_table((0.25, -0.03)), grasp=plan)
    assert put.ok and put.released, put.message
    assert arm.ee[2] >= plan.hover_height_m - 0.011              # and it went back up


def test_pick_grasps_an_object_its_kinematics_would_miss():
    pawn = Cylinder((0.20, 0.02), 0.021, 0.045)
    arm = OffArm([pawn])
    got = ParallelJaw(SPEC).pick(arm, Target(on_table(pawn.xy), foot_m=0.021, max_height_m=0.082))
    assert got.ok, got.message
    assert arm.held is pawn and arm.looks >= 1
    arm = OffArm([Cylinder((0.20, 0.02), 0.021, 0.045)])
    blind = ParallelJaw(dataclasses.replace(SPEC, overhead_correct=False, retries=1))
    assert not blind.pick(arm, Target(on_table((0.20, 0.02)), foot_m=0.021, max_height_m=0.082)).ok



class _LostServoOffArm(OffArm):
    """Kinematics off by delta, a fixed camera that sees it, a wrist camera that cannot find the object."""
    wrist_servo_target_px = _ServoArm.wrist_servo_target_px

    def visual_servo_xy(self, detect, tol_px=5.0, orientation=None, point_height_m=None):
        return False, "target lost while acquiring (iteration 1: not found)"


def test_a_wrist_camera_that_loses_the_object_descends_where_the_overhead_camera_put_the_jaws(monkeypatch):
    # At the board's edge and on a square of its own colour the wrist camera
    # loses the piece; the fixed camera, which saw the jaws over it, has
    # already measured where they are.
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pawn = Cylinder((0.20, 0.02), 0.021, 0.045)
    arm = _LostServoOffArm([pawn])
    got = _servoing().pick(arm, Target(on_table(pawn.xy), foot_m=0.021, height_m=0.045), vision=_measuring(45.0))
    assert got.ok, got.message
    assert arm.held is pawn
    # Without that sighting it would be blind: it does not go down.
    arm = _LostServoOffArm([Cylinder((0.20, 0.02), 0.021, 0.045)], visible=False)
    got = _servoing({"retries": 1}).pick(arm, Target(on_table((0.20, 0.02)), foot_m=0.021, height_m=0.045),
                                         vision=_measuring(45.0))
    assert not got.ok and arm.held is None


class _ShortFirstMoveArm(OffArm):
    """Its first move stops 12 mm short and says so, as the servos do near the wrist's limit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.short = True

    def move_ee_to(self, target, *args, **kwargs):
        if self.short:
            self.short = False
            super().move_ee_to(np.asarray(target, float) + [0.012, 0.0, 0.0], *args, **kwargs)
            return False
        return super().move_ee_to(target, *args, **kwargs)


def test_a_hover_that_stops_short_goes_on_once_the_camera_has_the_jaws_over_the_object():
    pawn = Cylinder((0.20, 0.02), 0.021, 0.045)
    arm = _ShortFirstMoveArm([pawn])
    got = ParallelJaw(SPEC).pick(arm, Target(on_table(pawn.xy), foot_m=0.021, height_m=0.045))
    assert got.ok, got.message
    assert arm.held is pawn and not any("[Pick 2/" in n for n in got.notes)
    # Without the camera's sighting a short hover is not trusted.
    arm = _ShortFirstMoveArm([Cylinder((0.20, 0.02), 0.021, 0.045)], visible=False)
    got = ParallelJaw(dataclasses.replace(SPEC, retries=1)).pick(
        arm, Target(on_table((0.20, 0.02)), foot_m=0.021, height_m=0.045))
    assert not got.ok and any("could not confirm" in n for n in got.notes)

def test_an_implausibly_large_correction_is_not_made():
    arm = OffArm(delta=(0.08, 0.0, 0.0))
    pawn, plan = _holding(arm, ParallelJaw(SPEC))
    put = ParallelJaw(SPEC).place(arm, on_table((0.25, -0.03)), grasp=plan)
    assert put.ok
    assert np.allclose(pawn.xy, np.array([0.25, -0.03]) + arm.delta[:2], atol=0.002)


def test_without_the_tag_in_view_the_kinematics_are_used():
    arm = OffArm(visible=False)
    pawn, plan = _holding(arm, ParallelJaw(SPEC))
    assert ParallelJaw(SPEC).place(arm, on_table((0.25, -0.03)), grasp=plan).ok
    assert arm.looks == 2                     # over the target, and at the release
    assert np.allclose(pawn.xy, np.array([0.25, -0.03]) + arm.delta[:2], atol=0.002)


def test_the_descent_is_fast_down_to_just_above_a_known_top_then_gentle(monkeypatch):
    # UR5, 2026-11-13: 15 cm at the gentle grasp step was 8.6 s of every pick.
    from mrcf import safety
    tight, descend = [], safety.descend_onto
    monkeypatch.setattr(safety, "descend_onto", lambda env, target, **kw: tight.append(float(target[2])) or
                        descend(env, target, **kw))
    spec = GripperSpec(**{**SPEC.__dict__, "hover_height_m": 0.15, "fast_descent_above_top_m": 0.015})
    block = Cylinder((0.20, 0.02), 0.015, 0.030)
    arm, pol = FakeArm([block]), ParallelJaw(spec)
    moves, move = [], arm.move_ee_to

    def watched(target, *a, **kw):
        moves.append((float(np.asarray(target)[2]), kw.get("max_step_deg")))
        return move(target, *a, **kw)
    arm.move_ee_to = watched
    got = pol.pick(arm, Target(on_table(block.xy), foot_m=0.030, height_m=0.030, widths_mm={4: 30.0, 26: 30.0}))
    assert got.ok, got.message
    grasp_z = got.grasp.grasp_height_m
    fast = 0.030 + spec.tip_below_grasp_m + 0.015                # the tips 15 mm above the top
    downs = [(z, step) for z, step in moves if z < 0.15 - 1e-6]
    assert downs[0] == (pytest.approx(fast, abs=1e-6), None)     # full speed to there ...
    assert downs[1][0] == pytest.approx(grasp_z, abs=1e-6) and downs[1][1] is not None   # ... then gently
    # Only the gentle stretch under the approach's tight limit: a CB3's estimate read
    # 24-54 N at full speed, touching nothing, and stopped a pick 17 mm above its block.
    assert tight[0] == pytest.approx(grasp_z, abs=1e-6) and fast not in [pytest.approx(t, abs=1e-6) for t in tight]

    unknown = Cylinder((0.20, -0.06), 0.015, 0.030)              # its height not known: all of it gently
    arm, pol = FakeArm([unknown]), ParallelJaw(spec)
    moves.clear()
    move = arm.move_ee_to
    arm.move_ee_to = watched
    pol.pick(arm, Target(on_table(unknown.xy), foot_m=0.030, max_height_m=0.15))   # planned as 150 mm tall
    hover = 0.15 + spec.tip_below_grasp_m + spec.top_clearance_m
    first_down = next(step for z, step in moves if z < hover - 1e-6)
    assert first_down is not None                                # no full-speed stretch down


# ── Closing across something lying long ───────────────────────────────────────

def _overhead(xyz=(0.36, 0.04, 0.52)):
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    T[:3, 3] = xyz
    return {"K": np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]), "D": np.zeros(5), "T_world_cam": T,
            "plane_z_m": 0.0}


def _outline_of(model, pts_m):
    """The outline px a camera sees of the 3-D points pts_m (their hull)."""
    import cv2
    from mrcf.perception.plane_geometry import PlaneView

    px = PlaneView.from_camera_model(model).project(np.asarray(pts_m, float) * 1000.0)
    return cv2.convexHull(px.astype(np.float32)).reshape(-1, 2).tolist()


def _cylinder(centre, axis_deg, length, radius, lying):
    """Points over a cylinder's surface: lying along axis_deg on the table, or standing (length its height)."""
    c = np.asarray(centre, float)
    ring = [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    if not lying:
        return [(c[0] + radius * u, c[1] + radius * v, z) for u, v in ring for z in (0.0, length)]
    d = np.array([np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg))])
    n = np.array([-d[1], d[0]])
    return [tuple(np.r_[c + d * s + n * radius * u, radius + radius * v]) for u, v in ring for s in (-length / 2, length / 2)]


class _SeenFromAbove(FakeArm):
    def camera_model(self, camera, width, height):
        return _overhead()


@pytest.mark.parametrize("axis_deg", [0.0, 35.0, 90.0, 140.0])
def test_a_thing_lying_on_its_side_is_closed_on_across_its_length(axis_deg):
    spec = dataclasses.replace(SPEC, close_across_long=True)
    pol, arm = ParallelJaw(spec), _SeenFromAbove()
    xy = (0.22, 0.17)
    t = Target(np.r_[xy, 0.0], foot_m=0.022, max_height_m=0.05,
               outlines={"overhead": {"size": [1280, 720], "px": _outline_of(_overhead(), _cylinder(xy, axis_deg, 0.045, 0.011, True))}})
    long = pol.long_way_deg(arm, t)
    assert long is not None and abs((long - axis_deg + 90.0) % 180.0 - 90.0) < 6.0
    jaws = pol.jaw_axis(arm, xy, pol.across_turn(arm, xy, long))
    assert abs(float(jaws @ (np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg))))) < 0.1


def test_a_standing_thing_seen_slantwise_is_not_taken_for_a_lying_one():
    pol, arm = ParallelJaw(dataclasses.replace(SPEC, close_across_long=True)), _SeenFromAbove()
    xy = (0.10, 0.25)                                     # far off the camera's nadir: its outline stretches out
    pts = _cylinder(xy, 0.0, 0.045, 0.011, False)
    t = Target(np.r_[xy, 0.0], foot_m=0.022, max_height_m=0.05,
               outlines={"overhead": {"size": [1280, 720], "px": _outline_of(_overhead(), pts)}})
    assert pol.long_way_deg(arm, t) is None
