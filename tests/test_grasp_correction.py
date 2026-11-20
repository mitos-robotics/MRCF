"""
Where the jaws really close, by lean and reach (grasp_correction.py), and the
arm putting that point — not its model's — where it is sent.
"""

import numpy as np
import pytest

from mrcf.grasp_correction import GraspCorrection, Point
from test_so101_env import _moving_so101, _perfect_servo


def test_no_table_changes_nothing():
    table = GraspCorrection()
    assert not table and table.at(-15.0, 0.25) == (0.0, 0.0)
    assert np.array_equal(table.offset(-15.0, (0.25, 0.0)), np.zeros(3))


def test_a_measured_point_is_its_own_value_and_between_them_it_blends():
    table = GraspCorrection([Point(-5.0, 0.25, along_m=0.000), Point(-15.0, 0.25, along_m=-0.010, across_m=0.002)])
    assert table.at(-15.0, 0.25) == (-0.010, 0.002)
    along, across = table.at(-10.0, 0.25)
    assert along == pytest.approx(-0.005) and across == pytest.approx(0.001)
    # Past what was measured: the last value measured, not a guess beyond it.
    assert table.at(-30.0, 0.25)[0] == pytest.approx(-0.010)
    # Between reaches, linear in reach too.
    table.points.append(Point(-15.0, 0.35, along_m=-0.020))
    assert table.at(-15.0, 0.30)[0] == pytest.approx(-0.015)
    assert table.at(-15.0, 0.45)[0] == pytest.approx(-0.020)


def test_points_from_one_object_pushed_along_as_it_was_measured_are_one_reach():
    # 2026-10-30: one pawn, pushed toward the base by the probes, gave points
    # from 140 mm out down to 117; split in two, lean 0 near the base read
    # the -10 deg value of 6 mm instead of the 0 measured there.
    table = GraspCorrection([Point(0.5, 0.138, along_m=0.0), Point(-4.3, 0.140, along_m=0.004),
                             Point(-9.8, 0.134, along_m=0.006), Point(-14.4, 0.119, along_m=0.002),
                             Point(-19.7, 0.117, along_m=0.004)])
    assert len(table._bands()) == 1
    assert table.at(0.0, 0.120)[0] == pytest.approx(0.0, abs=0.0005)


def test_along_points_away_from_the_base_and_across_a_quarter_turn_from_it():
    table = GraspCorrection([Point(-10.0, 0.20, along_m=0.010, across_m=0.004)])
    assert np.allclose(table.offset(-10.0, (0.20, 0.0)), [0.010, 0.004, 0.0])
    assert np.allclose(table.offset(-10.0, (0.0, 0.20)), [-0.004, 0.010, 0.0])
    assert np.allclose(table.offset(-10.0, (0.05, 0.20), base_xy=(0.05, 0.0)), [-0.004, 0.010, 0.0])


def test_it_is_kept_on_disk(tmp_path):
    table = GraspCorrection([Point(-15.0, 0.26, along_m=-0.009, span_m=[-0.018, 0.0], note="e4")])
    table.save(tmp_path / "grasp_correction.json")
    again = GraspCorrection.load(tmp_path / "grasp_correction.json")
    assert again.points == table.points and again.measured
    assert not GraspCorrection.load(tmp_path / "none.json")


def test_the_arm_puts_where_its_jaws_really_close_on_the_target():
    # 2026-10-30: at a lean of -15 deg the jaws closed ~1 cm short of a pawn
    # both cameras had them over.
    env = _moving_so101(_perfect_servo)
    env._pan_centre = np.zeros(2)
    env.grasp_lean_deg = lambda rot, at: -15.0
    env._grasp_table = GraspCorrection([Point(-15.0, 0.05, along_m=-0.010)])
    goal = np.array([0.03, 0.04, 0.0])                  # 0.05 m out
    assert env.move_ee_to(goal, tol=0.001, orientation=np.eye(3), max_step_deg=3.0) is True
    assert np.linalg.norm(env.get_ee_pos() - goal) < 0.001
    model = env.gripper_frame()[:3, 3]                  # this stand-in's tcp is zero
    assert np.allclose(model - goal, [0.006, 0.008, 0.0], atol=0.001)   # 10 mm further out, along the arm


# ── Measuring it: tools/calibrate_grasp.py on an arm whose jaws drift with the lean ──

class _LeaningArm:
    """
    Jaws that really close MM_PER_DEG mm further out along the arm per degree
    of lean than the model has them (the model is all the arm and the
    overhead camera know), over an object held down at `obj`. They catch it
    when it is within PAD_M of the true grasp point along the arm, and a miss
    pushes it PUSH_M toward the base. Out to 0.22 m it reaches at leans 0 to
    -10, further out -10 to -30.
    """
    MM_PER_DEG, PAD_M, PUSH_M = 0.6, 0.007, 0.0

    def __init__(self, obj):
        self.obj, self.ee, self.lean, self.closes = np.asarray(obj, float), np.zeros(3), 0.0, 0
        self.base_xy = np.zeros(2)
        self._tilts: dict = {}

    def _u(self):
        return self.obj / np.linalg.norm(self.obj)

    def turned(self, tilt):
        """An orientation at this tilt, which the arm leans at once moved with it."""
        rot = np.eye(3)
        self._tilts[id(rot)] = (rot, tilt)
        return rot

    def approach_orientation_for(self, targets, tols=None, prefer_tilt_deg=None, **_):
        reach = float(np.linalg.norm(np.asarray(targets[0])[:2]))
        lo, hi = (-10.0, 0.0) if reach < 0.22 else (-30.0, -10.0)
        tilt = float(np.clip(prefer_tilt_deg or 0.0, lo, hi))
        return (self.turned(tilt), tilt)

    def move_ee_to(self, target, tol=0.01, orientation=None, max_step_deg=None):
        self.ee = np.asarray(target, float).copy()
        if id(orientation) in self._tilts:
            self.lean = self._tilts[id(orientation)][1]
        return True

    def set_gripper(self, width, n_steps=15, max_step_pct=None):
        self.width = width
        if width == 0.0:
            self.close_step_pct = max_step_pct

    def get_gripper_width(self):
        return self.width

    def squeeze_gripper(self):
        self.closes += 1
        true = self.ee[:2] - self.MM_PER_DEG / 1000 * self.lean * self._u()   # leaning -15: 9 mm further out
        along = float((true - self.obj) @ self._u())
        if abs(along) <= self.PAD_M:
            return 0.016, True
        self.obj = self.obj - self.PUSH_M * self._u()
        return 0.0, False

    def get_ee_pos(self):
        return self.ee.copy()

    def get_ee_rotation(self):
        return np.eye(3)

    def grasp_lean_deg(self, rot, at):
        return self.lean

    def overhead_gripper_offset(self):
        return np.zeros(3)


def test_the_calibration_finds_where_the_jaws_really_close_at_each_lean():
    from mrcf.backends.so101 import make_policies
    from tools import calibrate_grasp

    pol = make_policies()
    found = []
    for obj in ((0.18, 0.0), (0.26, 0.05)):
        arm = _LeaningArm(obj)
        real = arm.approach_orientation_for
        arm.approach_orientation_for = lambda t, **kw: (lambda r: (setattr(arm, "lean", r[1]), r)[1])(real(t, **kw))
        found += calibrate_grasp.calibrate(arm, pol, {"x": obj}, 0.043, leans=(0.0, -10.0, -20.0, -30.0),
                                           say=lambda *a: None)
        # Closed as a pick closes: at full speed a near miss threw a pawn off the board (2026-10-30).
        assert arm.close_step_pct == pol.spec.close_step_pct
    assert [(p.lean_deg, round(p.reach_m, 2)) for p in found] == [(0.0, 0.18), (-10.0, 0.18), (-10.0, 0.26),
                                                                   (-20.0, 0.26), (-30.0, 0.26)]
    for p in found:
        # True minus model, within half a 4 mm step: at -20, the jaws 12 mm further out.
        assert p.along_m == pytest.approx(-_LeaningArm.MM_PER_DEG / 1000 * p.lean_deg, abs=0.0021)


def test_a_span_that_runs_off_the_sweep_is_not_used():
    from tools import calibrate_grasp
    probe = {"lean_deg": -30.0, "tried": [-0.008, -0.004, 0.0, 0.004, 0.008], "caught": [0.004, 0.008]}
    assert calibrate_grasp.point_from(probe, 0.30) is None
    probe["caught"] = [-0.004, 0.0, 0.004]
    assert calibrate_grasp.point_from(probe, 0.30).along_m == 0.0


def test_measuring_again_replaces_what_was_measured_there():
    from tools import calibrate_grasp
    old = GraspCorrection([Point(-15.0, 0.26, along_m=-0.009), Point(-5.0, 0.18, along_m=0.0)])
    new = calibrate_grasp.merged(old, [Point(-15.2, 0.265, along_m=-0.011)])
    assert sorted((p.lean_deg, p.along_m) for p in new.points) == [(-15.2, -0.011), (-5.0, 0.0)]


def test_the_jaws_must_open_before_anything_rises():
    # 2026-10-30: lifted with the jaws still shut on a pawn held down on e2,
    # the probes pulled it off and left it on e3.
    from tools import calibrate_grasp
    arm = _LeaningArm((0.2, 0.0))
    arm.set_gripper = lambda width, n_steps=15: None            # stuck shut
    arm.width = 0.016
    with pytest.raises(calibrate_grasp.JawsShut):
        calibrate_grasp.open_jaws(arm, 0.035)
    rose = []
    arm.move_ee_to = lambda *a, **k: rose.append(a)
    calibrate_grasp.leave_clear(arm, 0.035)
    assert rose == []                                           # stays put, holding on


def test_an_object_a_miss_pushes_along_the_arm_is_found_again_and_probed_from_where_it_stands():
    # 2026-10-30: "the pawns move and that's realistic" — nothing held down,
    # and every lean pushed the pawn 3-14 mm toward the base.
    from mrcf.backends.so101 import make_policies
    from tools import calibrate_grasp
    arm = _LeaningArm((0.26, 0.0))
    arm.PUSH_M = 0.005
    real = arm.approach_orientation_for
    arm.approach_orientation_for = lambda t, **kw: (lambda r: (setattr(arm, "lean", r[1]), r)[1])(real(t, **kw))
    asked = []

    def recheck(xy):
        asked.append(np.asarray(xy, float).copy())
        return arm.obj.copy()

    found = calibrate_grasp.calibrate(arm, make_policies(), {"e4": (0.26, 0.0)}, 0.043, leans=(-10.0, -20.0),
                                      say=lambda *a: None, recheck=recheck)
    assert [p.lean_deg for p in found] == [-10.0, -20.0]
    for p in found:
        assert p.along_m == pytest.approx(-_LeaningArm.MM_PER_DEG / 1000 * p.lean_deg, abs=0.0021)
    assert arm.obj[0] < 0.26 - 0.02 and len(asked) >= 4          # pushed well along, found after each miss


def test_without_finding_it_again_a_pushed_object_gives_a_wrong_answer():
    # What finding it again is for: the same arm, the object taken not to move.
    from mrcf.backends.so101 import make_policies
    from tools import calibrate_grasp
    arm = _LeaningArm((0.26, 0.0))
    arm.PUSH_M = 0.005
    real = arm.approach_orientation_for
    arm.approach_orientation_for = lambda t, **kw: (lambda r: (setattr(arm, "lean", r[1]), r)[1])(real(t, **kw))
    found = calibrate_grasp.calibrate(arm, make_policies(), {"e4": (0.26, 0.0)}, 0.043, leans=(-20.0,),
                                      say=lambda *a: None)
    assert not found or abs(found[0].along_m - 0.012) > 0.0021


def test_each_lean_is_turned_to_high_above_the_object_not_beside_it():
    # 2026-10-30: turned to the next lean at the probes' hover, 2 cm over a
    # pawn's top, the swinging jaws knocked it off its square twice.
    from mrcf.backends.so101 import make_policies
    from tools import calibrate_grasp
    arm = _LeaningArm((0.26, 0.05))
    moves = []
    real_move = arm.move_ee_to
    arm.move_ee_to = lambda target, **kw: (moves.append((round(float(target[2]), 3), arm.lean)), real_move(target, **kw))[1]
    real = arm.approach_orientation_for
    arm.approach_orientation_for = lambda t, **kw: (lambda r: (setattr(arm, "lean", r[1]), r)[1])(real(t, **kw))
    calibrate_grasp.calibrate(arm, make_policies(), {"e4": (0.26, 0.05)}, 0.043, leans=(-10.0, -20.0),
                              say=lambda *a: None)
    top = 0.043 + calibrate_grasp.CLEAR_M + calibrate_grasp.TURN_ABOVE_M[0]
    turns = [i for i in range(1, len(moves)) if moves[i][1] != moves[i - 1][1]]
    assert turns and all(moves[i][0] >= top - 1e-6 for i in turns)


def test_a_lean_that_does_not_reach_from_high_up_is_turned_to_as_high_as_it_does():
    # 2026-10-30: turned to 6 cm over the probes' hover, the SO-101 reached e6
    # at no lean at all, and e2 only at -25 and -30.
    from mrcf.backends.so101 import make_policies
    from tools import calibrate_grasp
    arm, pol = _LeaningArm((0.26, 0.05)), make_policies()
    hover = pol.spec.table_z + 0.043 + calibrate_grasp.CLEAR_M
    ceiling = hover + 0.03                     # this arm reaches no higher at a lean, only straight down
    moves = []
    real_move = arm.move_ee_to
    arm.move_ee_to = lambda target, **kw: (real_move(target, **kw), moves.append((round(float(target[2]), 3),
                                                                                  arm.lean)))[0]
    real = arm.approach_orientation_for

    def reach(t, **kw):
        if max(float(p[2]) for p in t) > ceiling + 1e-6:
            return (arm.turned(0.0), 0.0)
        return real(t, **kw)
    arm.approach_orientation_for = reach
    found = calibrate_grasp.calibrate(arm, pol, {"e4": (0.26, 0.05)}, 0.043, leans=(-20.0,),
                                      say=lambda *a: None)
    assert [p.lean_deg for p in found] == [-20.0]
    # 2026-10-30: straight in from home to 2 cm over a pawn, the arm knocked it
    # over. Over it high up first, however it reaches there; turned as high as
    # the lean reaches; then down.
    top = hover + calibrate_grasp.TURN_ABOVE_M[0]
    assert moves[0][0] == pytest.approx(top, abs=0.001) and moves[0][1] == 0.0
    assert moves[1][0] == pytest.approx(ceiling, abs=0.001) and moves[1][1] == -20.0
