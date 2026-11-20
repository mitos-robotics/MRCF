"""
A pick of an object nobody has measured looks for its height first
(parallel_jaw/height_looks.py): 2026-11-16 on the SO-101 tabletop every pick of a
25 mm ball planned for 150 mm and closed in the air above it.

The arm here is fake_arm's, with a wrist camera bolted behind the jaws looking
down and ahead (as the SO-101's), a fixed camera overhead, and a segmenter that
outlines whatever true object is under each point it is asked about.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fake_arm import FakeArm  # noqa: E402
from mrcf.perception import plane_geometry as pg  # noqa: E402
from mrcf.policies.base import Measured, Obstacle, Target, Vision  # noqa: E402
from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw  # noqa: E402
from mrcf.policies.parallel_jaw.height_looks import HeightLooks  # noqa: E402

W, H = 640, 480
K = np.array([[480.0, 0, 320], [0, 480.0, 240], [0, 0, 1]])

SPEC = GripperSpec(max_open_m=0.045, tip_below_grasp_m=0.007, free_depth_m=0.055,
                   finger_thickness_m=0.009, finger_half_width_m=0.008, jaw_axis="tangential",
                   hover_height_m=0.06, lift_height_m=0.07, lift_clearance_m=0.05, camera="wrist",
                   default_max_height_m=0.15, height_from_views=True, approach_look=True)


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


def _look_at(cam, target, up=(0.0, 0.0, 1.0)):
    cam, target = np.asarray(cam, float), np.asarray(target, float)
    z = target - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]
    T[:3, 3] = cam
    return T


OVERHEAD_T = _look_at((0.36, 0.04, 0.52), (0.30, 0.06, 0.0), up=(1.0, 0.0, 0.0))
# The wrist camera in the gripper frame (x out along the arm, z down the jaws): 7 cm
# behind them toward the base, 5 cm up, looking down the jaws tipped 30 deg ahead.
CAM_ON_GRIPPER = np.eye(4)
CAM_ON_GRIPPER[:3, :3] = np.array([[0.0, -0.866, 0.5], [1.0, 0.0, 0.0], [0.0, 0.5, 0.866]])
CAM_ON_GRIPPER[:3, 3] = [-0.07, 0.0, -0.05]


class Ball:
    """A ball standing on the table: what the fake arm grasps (xy, diameter_m, height_m) and the cameras see."""

    def __init__(self, xy, diameter_m: float):
        self.xy, self.diameter_m, self.height_m = np.asarray(xy, float), diameter_m, diameter_m

    def points_mm(self) -> np.ndarray:
        u, v = np.meshgrid(np.linspace(0, np.pi, 30), np.linspace(0, 2 * np.pi, 40))
        r, c = self.diameter_m * 500.0, self.xy * 1000.0
        return np.c_[c[0] + r * (np.sin(u) * np.cos(v)).ravel(), c[1] + r * (np.sin(u) * np.sin(v)).ravel(),
                     r - r * np.cos(u).ravel()]


def _down(at, flip=False):
    r = np.asarray(at, float)[:2]
    r = r / (np.linalg.norm(r) or 1.0)
    x, y, z = np.r_[r, 0.0], np.array([r[1], -r[0], 0.0]), np.array([0.0, 0.0, -1.0])
    if flip:
        x, y = -x, -y
    return np.c_[x, y, z]


def _lean(at, rot, lean_deg):
    r = np.asarray(at, float)[:2]
    r = r / (np.linalg.norm(r) or 1.0)
    a = np.array([-r[1], r[0], 0.0])
    t = np.radians(lean_deg)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return (np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)) @ rot


class SeeingArm(FakeArm):
    """fake_arm's arm with a located wrist camera, an overhead one, and the kinematics a look is planned by."""

    def __init__(self, objects=(), max_lean_deg=80.0, min_z=0.0):
        super().__init__(objects)
        self.rot = _down(self.ee)
        self.max_lean_deg, self.min_z = max_lean_deg, min_z
        self.looks = 0

    # Capabilities the looks are planned by
    def down_rotation(self, at, flip=False, turn_deg=0.0):
        return _down(at, flip)

    def approach_orientation_for(self, targets, gap_m=None, tols=None, flip=False, prefer_tilt_deg=None,
                                 margin_deg=None, turn_deg=0.0):
        for t in targets:
            reach = float(np.linalg.norm(np.asarray(t, float)[:2]))
            if not (0.10 <= reach <= 0.42 and self.min_z <= t[2] <= 0.35):
                return None
        tilt = float(np.clip(prefer_tilt_deg or 0.0, -self.max_lean_deg, self.max_lean_deg))
        return _lean(targets[0], _down(targets[0], flip), tilt), tilt

    def get_ee_rotation(self):
        return self.rot.copy()

    def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None, orientation=None):
        if orientation is not None:
            self.rot = np.asarray(orientation, float)
        return super().move_ee_to(target, tol, max_iters, step_size, max_step_deg, orientation)

    # Cameras
    def _model(self, T):
        return {"K": K.copy(), "D": np.zeros(5), "T_world_cam": T, "plane_z_m": 0.0, "scale": 1.0}

    def camera_model(self, camera, width, height):
        if camera == "overhead":
            return self._model(OVERHEAD_T)
        if camera == "wrist":
            return self.camera_model_at(camera, width, height, self.ee, self.rot)
        return None

    def camera_model_at(self, camera, width, height, at, rotation, gap_m=None):
        if camera != "wrist":
            return self.camera_model(camera, width, height)
        flange = np.eye(4)
        flange[:3, :3], flange[:3, 3] = rotation, at
        return self._model(flange @ CAM_ON_GRIPPER)

    def render_image(self, camera, width=640, height=480):
        self.looks += camera == "wrist"
        return np.zeros((height, width, 3), np.uint8)

    def outline(self, obj, camera="wrist"):
        view = pg.PlaneView.from_camera_model(self.camera_model(camera, W, H))
        pts = obj.points_mm()
        with np.errstate(all="ignore"):
            behind = ((pts @ view.R.T + view.t)[:, 2] <= 1.0).any()
        if behind:
            return np.zeros((H, W), bool)
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(view.project(pts).astype(np.int32)), 1)
        return m.astype(bool)


@pytest.fixture
def seeing(monkeypatch):
    """The segmenter, as the arm's wrist camera would outline its true objects."""
    from mrcf.perception import segment
    arms = []

    def segment_points(frame, prompts):
        arm = arms[0]
        out = []
        for u, v in prompts:
            hit = next((o for o in arm.objects if arm.outline(o)[int(v), int(u)]), None)
            if hit is None:
                blob = np.zeros((H, W), bool)
                blob[max(0, int(v) - 40):int(v) + 40, max(0, int(u) - 40):int(u) + 40] = True
                out.append(([blob] * 3, np.array([0.3, 0.3, 0.3])))
            else:
                m = arm.outline(hit)
                out.append(([m, m, m], np.array([0.95, 0.9, 0.85])))
        return out

    monkeypatch.setattr(segment, "available", lambda: True)
    monkeypatch.setattr(segment, "segment", segment_points)
    return arms


def _feet(arm):
    """Vision.measure for the pick's foot finder: the true foot of the object nearest where it is looked for."""
    def measure(frame, expected_px=None, model=None, expected_xy_m=None, known=None, target=None, need_height=True):
        if need_height:
            # Asked for its height from over it, as the hover's look does: 2026-11-16, 0.21 px a mm.
            return {"ok": False, "why": "its outline was found but the view cannot tell its height"}
        want = next(k for k in known if k.key == target).foot_xy_m
        o = min(arm.objects, key=lambda ob: np.linalg.norm(ob.xy - want))
        return {"ok": True, "occluded": False, "measurement": SimpleNamespace(
            height_mm=None, widths_mm={}, base_radius_mm=o.diameter_m * 500, slant_deg=30.0, base_world_m=o.xy.copy())}
    return Vision(measure=measure)


def _target(arm, ball, measured_from_above=True):
    outlines = None
    if measured_from_above:
        cs, _ = cv2.findContours(arm.outline(ball, "overhead").astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        outlines = {"overhead": {"size": [W, H], "px": max(cs, key=cv2.contourArea).reshape(-1, 2).tolist()}}
    return Target(np.r_[ball.xy, 0.0], foot_m=ball.diameter_m, max_height_m=0.15, outlines=outlines)


def test_a_ball_nobody_measured_is_looked_at_from_the_side_and_picked(seeing):
    ball = Ball((0.27, 0.12), 0.025)
    arm = SeeingArm([ball])
    seeing.append(arm)
    got = ParallelJaw(SPEC).pick(arm, _target(arm, ball), vision=_feet(arm))
    assert got.ok and arm.held is ball, "\n".join(got.notes)
    assert any("mm tall (ball" in n for n in got.notes), "\n".join(got.notes)
    assert isinstance(got.measured, Measured) and abs(got.measured.height_mm - 25.0) <= 3.0
    assert got.measured.looks >= 2                       # the memory takes it as settled
    grip = min(p[2] for p in arm.path)
    assert 0.010 < grip < 0.025                          # closed on it, not 117 mm up


def test_without_looking_the_same_pick_closes_in_the_air(seeing):
    # The run of 2026-11-16 again: planned for the tallest it could be.
    ball = Ball((0.27, 0.12), 0.025)
    arm = SeeingArm([ball])
    seeing.append(arm)
    got = ParallelJaw(dataclasses.replace(SPEC, height_from_views=False, approach_look=False)).pick(
        arm, _target(arm, ball), vision=_feet(arm))
    assert not got.ok and arm.held is None
    assert "closed_on_nothing" in [a["result"] for a in got.attempts]
    assert min(p[2] for p in arm.path) > 0.08


def test_an_arm_that_can_only_look_down_at_it_refuses_rather_than_guess(seeing):
    # Right over it, and allowed nowhere lower than 30 cm, straight down: no look sees its
    # height. Refused, not a pick planned on the tallest it could be.
    ball = Ball((0.27, 0.12), 0.025)
    arm = SeeingArm([ball], max_lean_deg=0.0, min_z=0.30)
    arm.ee = np.array([0.27, 0.12, 0.32])
    arm.rot = _down(arm.ee)
    seeing.append(arm)
    got = ParallelJaw(SPEC).pick(arm, _target(arm, ball), vision=_feet(arm))
    assert not got.ok and got.refused
    assert got.message.startswith("could not tell how tall it is"), got.message
    assert all(p[2] >= 0.29 for p in arm.path)
    assert arm.held is None


def test_its_neighbour_is_kept_clear_of_while_looking(seeing):
    # A neighbour 5 cm toward the base, as tall as it could be (not measured): no look is taken
    # with the jaws in its column below its top.
    ball = Ball((0.27, 0.12), 0.025)
    other = Ball((0.27 - 0.05 * 0.914, 0.12 - 0.05 * 0.406), 0.03)
    arm = SeeingArm([ball, other])
    seeing.append(arm)
    obstacles = [Obstacle(np.r_[other.xy, 0.0], 0.015, "other", None)]
    got = ParallelJaw(SPEC).pick(arm, _target(arm, ball), obstacles=obstacles, vision=_feet(arm))
    assert got.ok, "\n".join(got.notes)
    for p in arm.path:
        near = np.linalg.norm(p[:2] - other.xy) - 0.015 < SPEC.height_view_clearance_m - 0.005
        if near and not np.linalg.norm(p[:2] - ball.xy) < 0.01:
            assert p[2] - SPEC.tip_below_grasp_m >= 0.15            # over the top it could have


def test_a_height_two_sides_bound_but_do_not_settle_is_gripped_as_the_shortest(monkeypatch):
    from mrcf.perception import shape_fit as sf

    jaw = ParallelJaw(SPEC)
    fit = sf.ShapeFit(height_mm=30.0, profile="straight", footprint=sf.Footprint.circle((0.27, 0.12), 0.04),
                      widths_mm={z: 40.0 for z in range(0, 42, 2)}, score=0.95, ious=[0.99, 0.93, 0.94],
                      px_per_mm=[0.2, 1.9, 2.4], spread_mm=(28.0, 40.0), resolved=False, why="spread")
    looks = HeightLooks(None, [SimpleNamespace(fixed=True), SimpleNamespace(fixed=False), SimpleNamespace(fixed=False)],
                        [], np.zeros(2), (4.0, 150.0), fit=fit)
    assert jaw._height_bounded(looks)
    looks.fit = dataclasses.replace(fit, spread_mm=(20.0, 60.0))
    assert not jaw._height_bounded(looks)                 # 40 mm apart: no grip right for all of them
    looks.fit = dataclasses.replace(fit, px_per_mm=[0.2, 0.4, 0.45])
    assert not jaw._height_bounded(looks)                 # no look saw its height


def test_the_scan_keeps_each_objects_outline_for_the_pick(monkeypatch):
    from mrcf import manipulation, scene_scan

    monkeypatch.setattr(manipulation, "_scenario", manipulation.Scenario())   # whatever a test before installed

    mask = np.zeros((H, W), bool)
    mask[100:140, 200:260] = True
    px = scene_scan._outline_px(mask)
    assert px is not None and len(px) >= 4
    assert min(u for u, _ in px) == 200 and max(v for _, v in px) == 139
    obj = SimpleNamespace(measured_shape=None, position=np.array([0.27, 0.12, 0.0]), on=None,
                          geometry={"base_diameter_m": 0.025, "outlines": {"overhead": {"size": [W, H], "px": px}}})
    target = manipulation.target_of(obj)
    assert target.outlines == {"overhead": {"size": [W, H], "px": px}}


def test_a_survey_counts_each_side_that_saw_it_in_the_memory():
    from mrcf.object_memory import Detection, ObjectMemory

    memory = ObjectMemory()
    (track,), _ = memory.observe([Detection(position=(0.27, 0.12), foot_mm=25.0, height_mm=None,
                                            appearance={"a": 0.0, "b": 0.0})])
    memory.measured(track.id, 25.0, {0: 18.0, 12: 25.0}, 18.0, looks=2)
    assert track.shape()["looks"] == 2


def test_a_height_remembered_wrong_is_put_right_from_close_above_on_the_way_down(seeing):
    # The memory has the ball 80 mm tall (another object's track, say): planned by that, the jaws
    # close 36 mm up, over it. The look from just above what it may be, fitted with the scan's
    # outline, finds 25 and the grasp is planned again before the jaws get there.
    ball = Ball((0.27, 0.12), 0.025)
    arm = SeeingArm([ball])
    seeing.append(arm)
    wrong = dataclasses.replace(_target(arm, ball), height_m=0.080, widths_mm={0: 25.0, 40: 25.0, 80: 25.0},
                                shape_looks=2)
    got = ParallelJaw(SPEC).pick(arm, wrong, vision=_feet(arm))
    assert got.ok and arm.held is ball, "\n".join(got.notes)
    assert any("From close above it is 2" in n and "the grasp planned again" in n for n in got.notes), \
        "\n".join(got.notes)
    assert min(p[2] for p in arm.path) < 0.025
    assert got.measured is not None and abs(got.measured.height_mm - 25.0) <= 3.0
    # Without the look, the same pick closes over it.
    arm = SeeingArm([ball := Ball((0.27, 0.12), 0.025)])
    seeing[:] = [arm]
    got = ParallelJaw(dataclasses.replace(SPEC, approach_look=False)).pick(
        arm, dataclasses.replace(wrong, outlines=_target(arm, ball).outlines), vision=_feet(arm))
    assert arm.held is None and "closed_on_nothing" in [a["result"] for a in got.attempts]


def test_a_target_with_no_outline_from_above_is_not_looked_at_so(seeing):
    # Chess's board scan outlines no piece from above: its picks are as they were.
    ball = Ball((0.27, 0.12), 0.025)
    arm = SeeingArm([ball])
    seeing.append(arm)
    got = ParallelJaw(SPEC).pick(arm, _target(arm, ball, measured_from_above=False), vision=_feet(arm))
    assert not any(n.startswith("[Height]") for n in got.notes)


def test_each_look_is_kept_with_the_run_and_its_numbers_recorded(seeing, monkeypatch):
    from mrcf import flight_recorder

    kept, events = [], []
    monkeypatch.setattr(flight_recorder, "keep_picture", lambda stem, image: kept.append((stem, image.shape)))
    monkeypatch.setattr(flight_recorder, "event", lambda source, kind, image=None, **data: events.append((kind, data)))
    ball = Ball((0.27, 0.12), 0.025)
    arm = SeeingArm([ball])
    seeing.append(arm)
    got = ParallelJaw(SPEC).pick(arm, _target(arm, ball), vision=_feet(arm))
    assert got.ok
    stems = [s for s, _ in kept]
    assert stems[:2] == ["height_1", "height_2"] and all(shape == (H, W, 3) for _, shape in kept)
    looks = [d for k, d in events if k == "height_look"]
    assert looks and {"height_mm", "spread_mm", "ious", "px_per_mm"} <= set(looks[0])


def test_after_a_miss_it_backs_off_no_higher_than_the_attempt_came_from():
    # 2026-11-16: 32 cm out, backing up to a hover clear of 150 mm ground on 24 s each retry.
    from mrcf.policies.parallel_jaw.spec import GraspPlan

    plan = GraspPlan(height_m=0.03, width_m=0.03, grasp_height_m=0.016, grip_width_m=0.03, pregrasp_open_m=0.04,
                     hover_height_m=0.165, lift_height_m=0.18, release_height_m=0.02)
    came_from = np.array([0.32, 0.10, 0.062])
    for switch, top in ((False, 0.165), (True, 0.062)):
        arm = SeeingArm([])
        arm.ee = np.array([0.32, 0.10, 0.02])
        ParallelJaw(dataclasses.replace(SPEC, back_off_to_attempt_hover=switch))._back_off(
            arm, plan, "[Pick 2/3]", reached=came_from)
        assert arm.ee[2] == pytest.approx(top, abs=0.011)


def _fit(spread, bound, score=0.75, ious=(0.7, 0.75, 0.76), px=(1.0, 1.1, 1.2)):
    return SimpleNamespace(score=score, spread_mm=spread, bound_mm=lambda drop: bound, ious=list(ious),
                           px_per_mm=list(px), resolved=False)


@pytest.mark.parametrize("spread, bound, on, capped", [
    ((5.0, 31.0), 28.0, True, 41.0),        # the wooden figures, 2026-11-17: gripped as low as the jaws go
    ((5.0, 31.0), 28.0, False, None),       # not on this arm: refused, as before
    ((5.0, 70.0), 60.0, True, 80.0),        # gripped 25 mm up its tallest clears the body (a miss were it short)
    ((28.0, 62.0), 62.0, True, 72.0),       # a 3-pin plug: gripped at 17 mm its tallest clears, its shortest is held
    ((24.0, 59.0), 77.0, True, 87.0),       # the same plug, 13:26: gripped at 32 mm (a miss were it under 42 mm)
    ((5.0, 30.0), 120.0, True, None),       # its body clears only gripped above even the tallest that fits: refused
])
def test_a_height_known_only_from_above_is_gripped_low_when_short_enough(spread, bound, on, capped):
    spec = dataclasses.replace(SPEC, grip_low_when_capped=on, min_grasp_height_m=0.016)
    pol = ParallelJaw(spec)
    looks = SimpleNamespace(fit=_fit(spread, bound), seeing=lambda: 2)
    assert pol._height_capped(looks) == capped
    if capped is not None:
        plan = pol.plan(Target(np.array([0.2, 0.0, 0.0]), foot_m=0.022, max_height_m=capped / 1000.0))
        assert plan.grasp_height_m == pytest.approx(max(0.016, capped / 1000.0 - spec.free_depth_m))
        assert plan.hover_height_m >= capped / 1000.0 + spec.tip_below_grasp_m


def test_one_its_outline_from_above_caps_low_enough_is_gripped_low_without_side_looks(monkeypatch):
    # 2026-11-18: side looks made a plug 36-51 mm tall; gripped by that it was closed on nothing
    # twice and pushed away, where gripped low it had held.
    spec = dataclasses.replace(SPEC, grip_low_from_above=True, min_grasp_height_m=0.016)
    pol = ParallelJaw(spec)
    target = Target(np.array([0.25, 0.05, 0.0]), foot_m=0.05, max_height_m=0.15, outlines={"overhead": {}})
    monkeypatch.setattr(pol, "could_be_mm", lambda *a: 63.0)
    low = pol.low_from_above(None, target)
    assert low is not None and low.max_height_m == pytest.approx(0.063) and low.height_m is None
    plan = pol.plan(low)
    assert plan.grasp_height_m == pytest.approx(0.016)                 # as low as the jaws go
    monkeypatch.setattr(pol, "could_be_mm", lambda *a: 90.0)          # the body would not clear a top that tall
    assert pol.low_from_above(None, target) is None
    monkeypatch.setattr(pol, "could_be_mm", lambda *a: 150.0)         # no bound from above
    assert pol.low_from_above(None, target) is None


def test_a_ball_capped_from_above_is_picked_low_with_no_side_looks(seeing):
    ball = Ball((0.20, -0.10), 0.025)              # its outline from above: no taller than ~53 mm
    arm = SeeingArm([ball])
    seeing.append(arm)
    spec = dataclasses.replace(SPEC, grip_low_from_above=True)
    got = ParallelJaw(spec).pick(arm, _target(arm, ball), vision=_feet(arm))
    assert got.ok and arm.held is ball, "\n".join(got.notes)
    assert any("no looks from the side" in n for n in got.notes), "\n".join(got.notes)
    assert not any("From " in n and "cm off" in n for n in got.notes), "\n".join(got.notes)
    assert min(p[2] for p in arm.path) < 0.025


# ── A grip that held it before, and why a close found nothing ─────────────────

def test_a_grip_that_held_it_before_is_planned_by_with_no_looks_for_its_height(seeing):
    ball = Ball((0.27, 0.12), 0.025)
    arm = SeeingArm([ball])
    seeing.append(arm)
    target = dataclasses.replace(_target(arm, ball), grip_at_m=0.016, max_height_m=0.05)
    got = ParallelJaw(SPEC).pick(arm, target, vision=_feet(arm))
    assert got.ok and arm.held is ball, "\n".join(got.notes)
    assert any("Gripped as before, 16 mm up" in n for n in got.notes), "\n".join(got.notes)
    assert not any("looking at it from the side" in n for n in got.notes), "\n".join(got.notes)


def test_a_close_above_it_is_asked_why_and_the_next_one_grips_lower(seeing):
    # 2026-11-18: three closes on nothing over a plug on its back, each "re-aimed", the grip
    # 21 mm up over its body. Asked why (the wrist camera's pictures), the next attempt grips low.
    from mrcf.policies.base import Diagnosis

    ball = Ball((0.27, 0.12), 0.025)
    spec = dataclasses.replace(SPEC, measure_at_hover=False, approach_look=False)
    asked = []

    def diagnose(frames, what):
        asked.append([label for label, _ in frames])
        return Diagnosis("too_high", "the jaws shut over its top")
    arm = SeeingArm([ball])
    seeing.append(arm)
    high = dataclasses.replace(_target(arm, ball), grip_at_m=0.040, max_height_m=0.06)
    got = ParallelJaw(spec).pick(arm, high, vision=dataclasses.replace(_feet(arm), diagnose=diagnose))
    assert got.ok and arm.held is ball, "\n".join(got.notes)
    assert [a["result"] for a in got.attempts][:1] == ["closed_on_nothing"] and got.attempts[0]["diagnosis"] == "too_high"
    assert asked and asked[0][-1].endswith("_shut")
    assert any("Why: the jaws shut over its top" in n for n in got.notes), "\n".join(got.notes)
    # Not asked: the same grip, three times over it.
    arm = SeeingArm([ball := Ball((0.27, 0.12), 0.025)])
    seeing[:] = [arm]
    got = ParallelJaw(spec).pick(arm, dataclasses.replace(high, outlines=_target(arm, ball).outlines), vision=_feet(arm))
    assert arm.held is None and [a["result"] for a in got.attempts].count("closed_on_nothing") >= 2


def test_one_lying_on_its_side_is_not_picked_as_it_stands(seeing):
    from mrcf.policies.base import Diagnosis

    ball = Ball((0.27, 0.12), 0.025)
    spec = dataclasses.replace(SPEC, measure_at_hover=False, approach_look=False)
    arm = SeeingArm([ball])
    seeing.append(arm)
    high = dataclasses.replace(_target(arm, ball), grip_at_m=0.040, max_height_m=0.06)
    got = ParallelJaw(spec).pick(arm, high, vision=dataclasses.replace(
        _feet(arm), diagnose=lambda frames, what: Diagnosis("lying", "its head points at the jaws")))
    assert not got.ok and "lying on its side" in got.message and got.diagnosis.cause == "lying"
    assert len(got.attempts) == 1                                       # not tried again as it stands


def test_a_close_on_nothing_with_it_seen_in_the_gap_is_too_high_whatever_the_model_says(seeing):
    from mrcf.policies.base import Diagnosis

    ball = Ball((0.27, 0.12), 0.025)
    spec = dataclasses.replace(SPEC, measure_at_hover=False, approach_look=False)
    arm = SeeingArm([ball])
    seeing.append(arm)
    pol = ParallelJaw(spec)
    real = pol._close

    def close_seen(env, ctx):                        # the wrist camera at the grasp: centred in the gap
        pol._attempts[-1]["wrist_at_grasp_mm"] = [[0.6, -0.4]]
        return real(env, ctx)
    pol._close = close_seen
    high = dataclasses.replace(_target(arm, ball), grip_at_m=0.040, max_height_m=0.06)
    got = pol.pick(arm, high, vision=dataclasses.replace(
        _feet(arm), diagnose=lambda frames, what: Diagnosis("aim_off", "left of it")))
    assert got.ok and arm.held is ball, "\n".join(got.notes)
    assert got.attempts[0]["diagnosis"] == "too_high"
    assert any("between the jaws" in n and "closed on nothing" in n for n in got.notes)
    assert not any("Found it again" in n or "cannot be found" in n for n in got.notes)   # not looked for afresh


def test_jaws_stopped_short_of_shut_with_no_load_are_lifted_and_the_hold_decides(seeing, monkeypatch):
    # 2026-11-19: a plug on its back, the jaws shut on its smooth pins at 28.5 mm, "no load" — a miss, then.
    ball = Ball((0.27, 0.12), 0.025)
    arm = SeeingArm([ball])
    seeing.append(arm)
    spec = dataclasses.replace(SPEC, stopped_short_holds=True, measure_at_hover=False, approach_look=False)
    pol = ParallelJaw(spec)
    real = arm.squeeze_gripper

    def unfelt(**k):
        w, _ = real(**k)
        return (0.025 if arm.held is not None else w), False   # round the 25 mm ball, the servo feeling nothing
    monkeypatch.setattr(arm, "squeeze_gripper", unfelt)
    got = pol.pick(arm, dataclasses.replace(_target(arm, ball), grip_at_m=0.016, max_height_m=0.05), vision=_feet(arm))
    assert got.ok and arm.held is ball, "\n".join(got.notes)
    assert any("short of shut" in n for n in got.notes)
    pol = ParallelJaw(dataclasses.replace(spec, stopped_short_holds=False))
    arm2 = SeeingArm([Ball((0.27, 0.12), 0.025)])
    seeing[:] = [arm2]
    real2 = arm2.squeeze_gripper
    monkeypatch.setattr(arm2, "squeeze_gripper", lambda **k: (real2(**k)[0], False))
    got = pol.pick(arm2, dataclasses.replace(_target(arm2, arm2.objects[0]), grip_at_m=0.016, max_height_m=0.05),
                   vision=_feet(arm2))
    assert "closed_on_nothing" in [a["result"] for a in got.attempts]          # off: a miss, as before


def test_gripped_as_low_as_the_jaws_go_they_open_as_wide_as_the_neighbours_let_them():
    from mrcf.policies.parallel_jaw.pick import PickContext
    from mrcf.policies.base import Target, Vision

    spec = dataclasses.replace(SPEC, max_open_m=0.08, open_wide_when_low=True, min_grasp_height_m=0.016)
    pol = ParallelJaw(spec)
    target = Target(np.array([0.25, 0.1, 0.0]), foot_m=0.055, max_height_m=0.06)
    plan = pol.plan(target)
    ctx = PickContext(target=target, obstacles=[], vision=Vision(), room=0.07, plan=plan,
                      object_xy=np.array([0.25, 0.1]), servo_ready=False)
    ctx.grip_at_m = 0.016
    got = pol._gripped_at(ctx, dataclasses.replace(plan, pregrasp_open_m=0.06))
    assert got.grasp_height_m == pytest.approx(0.016) and got.pregrasp_open_m == pytest.approx(0.07)  # the room's
    ctx.grip_at_m = 0.03
    assert pol._gripped_at(ctx, dataclasses.replace(plan, pregrasp_open_m=0.06)).pregrasp_open_m == pytest.approx(0.06)


def test_after_a_close_on_nothing_it_is_looked_for_afresh_when_the_deployment_wants(seeing):
    """2026-11-19 14:03: a figure "seen between the jaws" had been tipped; the next try came down blind."""
    from mrcf.policies.base import Diagnosis

    for relook, looked_for in ((False, []), (True, ["relocate"])):
        ball = Ball((0.27, 0.12), 0.025)
        spec = dataclasses.replace(SPEC, measure_at_hover=False, approach_look=False, relook_after_miss=relook)
        arm = SeeingArm([ball])
        seeing[:] = [arm]
        calls: list = []
        vision = dataclasses.replace(_feet(arm), diagnose=lambda frames, what: Diagnosis("too_high", "over its top"),
                                     relocate=lambda env, was: calls.append("relocate") or np.asarray(was, float)[:2])
        high = dataclasses.replace(_target(arm, ball), grip_at_m=0.040, max_height_m=0.06)
        got = ParallelJaw(spec).pick(arm, high, vision=vision)
        assert got.ok and arm.held is ball, "\n".join(got.notes)
        assert calls[:1] == looked_for, (relook, calls)
