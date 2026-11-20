"""
Standing a fallen object up (mrcf/policies/stand_up.py): grip the knob at its top
with the jaws square across it, lift that end straight up, let it down.
"""

import dataclasses

import numpy as np
import pytest

from fake_arm import FakeArm
from mrcf.policies.base import Lying, Obstacle, Target
from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw
from mrcf.policies.stand_up import jaws_land_on, turn_to_square

SPEC = GripperSpec(max_open_m=0.045, tip_below_grasp_m=0.007, free_depth_m=0.055,
                   finger_thickness_m=0.009, finger_half_width_m=0.008, jaw_axis="tangential",
                   hover_height_m=0.06, lift_height_m=0.07, lift_clearance_m=0.05)


def pawn_lying(top_xy=(0.235, -0.001), along_deg=20.0, length=0.040, knob=0.013):
    """A 40 mm pawn on its side, its head's middle at top_xy, lying along_deg from +x (foot to top)."""
    a = np.radians(along_deg)
    axis = np.array([np.cos(a), np.sin(a)])
    top = np.asarray(top_xy, float)
    foot = top - axis * (length - knob / 2)
    widths = tuple((d, 0.024 if d < 0.01 else 0.018 if 0.026 < d < 0.03 else 0.015 if d < 0.026 else knob)
                   for d in np.arange(0.001, length, 0.002))
    return Lying(foot_xy=foot, top_xy=top, length_m=length, top_width_m=knob, widths=widths)


class LyingArm(FakeArm):
    """
    A FakeArm that turns its jaws when asked (OrientationSearch with turn_deg)
    and grips a lying pawn's head when the jaws close low over it: within
    GRIP_TOL_M of its middle, the tips below its middle.
    """
    GRIP_TOL_M = 0.004

    def __init__(self, lying: Lying, reach=True):
        super().__init__()
        self.lying, self.reach = lying, reach
        self.asked, self.widths_held, self.closed_at = [], [], None
        self.gripping = False
        self.released_at = None

    def approach_orientation_for(self, targets, tols=None, flip=False, prefer_tilt_deg=None, margin_deg=None,
                                 turn_deg=0.0):
        self.asked.append((turn_deg, flip))
        return (np.eye(3), 0.0) if self.reach else None

    def hold_width(self, width_m):
        self.widths_held.append(round(width_m, 4))

    def set_gripper(self, width, n_steps=15, max_step_pct=None, **_):
        self.guard()
        if width > self.width and self.gripping:
            self.gripping, self.released_at = False, self.ee.copy()
        if width < self.width:
            near = np.linalg.norm(self.ee[:2] - self.lying.top_xy) < self.GRIP_TOL_M
            low = self.ee[2] - SPEC.tip_below_grasp_m < self.lying.top_width_m / 2
            self.closed_at = self.ee.copy()
            self.gripping = near and low
        self.width = width
        self.step(1)

    def gripper_loaded(self):
        return self.gripping

    def get_gripper_width(self):
        return 0.013 if self.gripping else self.width


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_the_jaws_turn_the_short_way_round_to_close_square_across_it():
    assert turn_to_square([0.0, 1.0], [1.0, 0.0]) == pytest.approx(0.0)        # already square
    assert turn_to_square([0.0, 1.0], [0.0, 1.0]) == pytest.approx(90.0)       # a quarter turn, either way
    a = np.radians(20.0)
    assert turn_to_square([0.0, 1.0], [np.cos(a), np.sin(a)]) == pytest.approx(20.0)
    assert turn_to_square([0.0, 1.0], [-np.cos(a), -np.sin(a)]) == pytest.approx(20.0)   # foot the other end
    assert turn_to_square([0.0, 1.0], [np.cos(-a), np.sin(-a)]) == pytest.approx(-20.0)


def test_it_grips_the_head_low_lifts_it_upright_and_lets_go():
    lying = pawn_lying()
    arm = LyingArm(lying)
    got = ParallelJaw(SPEC).stand_up(arm, Target(np.r_[lying.top_xy, 0.0], lying=lying))
    assert got.ok, got.message
    assert np.allclose(got.at_xy, lying.top_xy)
    s = SPEC
    # The jaws turned square across a pawn lying 20 deg from the arm's line (tangential jaws).
    assert arm.asked and arm.asked[0][0] == pytest.approx(20.0, abs=0.5)
    # Closed with the tips stand_up_tip_clearance_m off the surface, under the head's middle.
    assert arm.closed_at[2] == pytest.approx(s.tip_below_grasp_m + s.stand_up_tip_clearance_m, abs=0.002)
    # Came down beside the head (and the collar within a finger's width of it), then held the head.
    assert arm.widths_held == [round(0.018 + 2 * s.stand_up_margin_m, 4), 0.013]
    # Let go with the head as high as it is with the pawn stood up, plus the overshoot.
    stood = arm.closed_at[2] + (0.040 - 0.013) + s.stand_up_overshoot_m
    assert arm.released_at[2] == pytest.approx(stood, abs=0.0025)
    assert np.allclose(arm.released_at[:2], lying.top_xy, atol=1e-3)       # straight up, never sideways
    assert arm.ee[2] >= s.hover_height_m - s.travel_tol_m                   # and back up, jaws open
    assert not arm.gripping


def test_where_the_wrist_turns_no_further_it_grips_as_near_square_as_it_does():
    # 2026-11-03, a bishop fallen on c4: square needed the wrist roll past where
    # it meets the camera's cable; it stopped 8 deg short, from where it could
    # have gripped. Here the jaws turn no further than 14 deg.
    class StiffWrist(LyingArm):
        def approach_orientation_for(self, targets, tols=None, flip=False, prefer_tilt_deg=None, margin_deg=None,
                                     turn_deg=0.0):
            self.asked.append((turn_deg, flip))
            return (np.eye(3), 0.0) if abs(turn_deg) <= 14.0 else None

    lying = pawn_lying(along_deg=20.0)
    arm = StiffWrist(lying)
    got = ParallelJaw(SPEC).stand_up(arm, Target(np.r_[lying.top_xy, 0.0], lying=lying))
    assert got.ok, got.message
    assert any("-8 off square" in n for n in got.notes) or any("-7 off square" in n for n in got.notes), got.notes
    assert abs(arm.asked[-1][0]) <= 14.0
    stiffer = StiffWrist(pawn_lying(along_deg=40.0))                          # 26 deg off: too far
    got = ParallelJaw(SPEC).stand_up(stiffer, Target(np.r_[stiffer.lying.top_xy, 0.0], lying=stiffer.lying))
    assert not got.ok and "within 15 deg of square" in got.message


def test_a_close_on_nothing_backs_off_open_and_says_it_has_likely_moved():
    lying = pawn_lying()
    arm = LyingArm(lying)
    off = dataclasses.replace(lying, top_xy=np.asarray(lying.top_xy) + [0.0, 0.008])   # told 8 mm off
    got = ParallelJaw(SPEC).stand_up(arm, Target(np.r_[off.top_xy, 0.0], lying=off))
    assert not got.ok and not got.refused and "closed on nothing" in got.message
    assert got.attempts[-1]["result"] == "closed_on_nothing"
    assert arm.width > 0.02 and arm.ee[2] >= SPEC.hover_height_m - SPEC.travel_tol_m


@pytest.mark.parametrize("why, target, arm_kwargs, says", [
    ("not lying", Target(np.zeros(3)), {}, "not known to be lying"),
    ("knob too wide", Target(np.zeros(3), lying=pawn_lying(knob=0.030)), {}, "jaws would need"),
    ("no orientation", Target(np.zeros(3), lying=pawn_lying()), {"reach": False}, "no one gripper orientation"),
])
def test_it_refuses_before_moving(why, target, arm_kwargs, says):
    arm = LyingArm(pawn_lying(), **arm_kwargs)
    got = ParallelJaw(SPEC).stand_up(arm, target)
    assert got.refused and says in got.message, (why, got.message)
    assert arm.path == []


def test_an_arm_that_cannot_turn_its_jaws_is_refused():
    class Unturned(LyingArm):
        def approach_orientation_for(self, targets, tols=None, flip=False, prefer_tilt_deg=None, margin_deg=None):
            return np.eye(3), 0.0
    lying = pawn_lying()
    got = ParallelJaw(SPEC).stand_up(Unturned(lying), Target(np.zeros(3), lying=lying))
    assert got.refused and "cannot turn its jaws" in got.message


def test_a_neighbour_where_a_finger_comes_down_refuses_it():
    lying = pawn_lying(along_deg=0.0)                        # lies along x: the jaws close along y
    beside = Obstacle(np.asarray(lying.top_xy) + [0.0, 0.025], 0.012, "black_pawn_d4")
    got = ParallelJaw(SPEC).stand_up(LyingArm(lying), Target(np.zeros(3), lying=lying), obstacles=[beside])
    assert got.refused and "black_pawn_d4" in got.message
    # The same neighbour off the end of the head, past the fingers' width: room.
    ahead = Obstacle(np.asarray(lying.top_xy) + [0.030, 0.0], 0.012, "ahead")
    assert jaws_land_on(lying.top_xy, lying.axis, 0.03, 0.008, [ahead], 0.003, 0.016) is None


class WristArm(LyingArm):
    """A LyingArm with a wrist camera: what it sees is up to the test's Vision.lying; records where it looked from."""

    def __init__(self, lying, **kw):
        super().__init__(lying, **kw)
        self.looked_from = []

    def render_image(self, camera, width=640, height=480):
        self.looked_from.append(self.ee.copy())
        return np.zeros((height, width, 3), np.uint8)

    def wrist_to_surface(self, px, height_m=0.0):
        return np.zeros((len(np.atleast_2d(px)), 2))


WRIST = dataclasses.replace(SPEC, camera="wrist", servo_height_m=0.07)


def test_after_a_miss_the_wrist_camera_looks_again_from_over_it_and_the_next_try_grips():
    from mrcf.policies.base import Vision
    truth = pawn_lying()
    told = dataclasses.replace(truth, top_xy=np.asarray(truth.top_xy) + [0.0, 0.008])     # the first sighting: 8 mm off
    arm = WristArm(truth)
    asked = []

    def sees(frame, to_surface, near_px, was):
        asked.append(was)
        return truth, ""

    got = ParallelJaw(WRIST).stand_up(arm, Target(np.r_[told.top_xy, 0.0], lying=told), vision=Vision(lying=sees))
    assert got.ok and "on try 2" in got.message, got.message
    assert [a["result"] for a in got.attempts] == ["closed_on_nothing", "ok"]
    assert [a["seen_by"] for a in got.attempts] == ["given", "wrist"]
    assert asked == [told]                                   # compared with what it was told before
    # It looked once, from the servo's height over where the jaws had gone — never off to be parked.
    looks = [p for p in arm.looked_from if abs(p[2] - WRIST.servo_height_m) < 0.003]
    assert len(looks) == 1 and np.allclose(looks[0][:2], told.top_xy, atol=1e-3)
    first_grip = next(i for i, p in enumerate(arm.path) if p[2] < 0.03)
    assert max(p[2] for p in arm.path[first_grip:]) < 0.1
    assert np.allclose(got.at_xy, truth.top_xy)


@pytest.mark.parametrize("sees, says", [
    (lambda *a: (None, "nothing outlined there"), "could not find it: nothing outlined there"),
    (lambda f, t, p, was: (dataclasses.replace(was, top_xy=np.asarray(was.top_xy) + [0.05, 0.0]), ""),
     "further than a miss moves one"),
])
def test_when_the_wrist_camera_cannot_find_it_again_it_stops_there(sees, says):
    from mrcf.policies.base import Vision
    truth = pawn_lying()
    told = dataclasses.replace(truth, top_xy=np.asarray(truth.top_xy) + [0.0, 0.008])
    got = ParallelJaw(WRIST).stand_up(WristArm(truth), Target(np.zeros(3), lying=told), vision=Vision(lying=sees))
    assert not got.ok and not got.refused and "closed on nothing" in got.message and says in got.message
    assert len(got.attempts) == 1


def test_it_gives_up_after_its_attempts():
    from mrcf.policies.base import Vision
    truth = pawn_lying()
    told = dataclasses.replace(truth, top_xy=np.asarray(truth.top_xy) + [0.0, 0.008])
    got = ParallelJaw(WRIST).stand_up(WristArm(truth), Target(np.zeros(3), lying=told),
                                      vision=Vision(lying=lambda f, t, p, was: (was, "")))   # sees it where it was
    assert not got.ok and [a["result"] for a in got.attempts] == ["closed_on_nothing"] * WRIST.stand_up_attempts
