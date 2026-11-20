"""
Putting things into containers: finding one by name (grounding), its outline
on the table (containers), how high its walls stand from two looks
(plane_sweep), and the place into it (policies/place_in) — on the fake arm.
"""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("cv2")

import cv2  # noqa: E402

from fake_arm import Cylinder, FakeArm  # noqa: E402
from mrcf import containers, manipulation  # noqa: E402
from mrcf.object_memory import Detection, ObjectMemory  # noqa: E402
from mrcf.perception import grounding, plane_sweep  # noqa: E402
from mrcf.perception.plane_geometry import PlaneView  # noqa: E402
from mrcf.policies.base import Opening, Target  # noqa: E402
from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw  # noqa: E402
from mrcf.policies.place_in import inside_by, nearest_wall, spots  # noqa: E402
from mrcf.vlm import SceneObject, SceneState, dispatch  # noqa: E402

SPEC = GripperSpec(max_open_m=0.045, tip_below_grasp_m=0.007, free_depth_m=0.055,
                   finger_thickness_m=0.009, finger_half_width_m=0.008, jaw_axis="tangential",
                   hover_height_m=0.06, lift_height_m=0.07, lift_clearance_m=0.05, place_in=True)
# A tub 25 x 35 cm beside the arm, walls 90 mm.
TUB = np.array([[0.10, -0.30], [0.45, -0.30], [0.45, -0.05], [0.10, -0.05]])
K = np.array([[460.0, 0, 320], [0, 460.0, 240], [0, 0, 1]])


def down_camera(xyz, tilt_deg: float = 0.0) -> dict:
    """A camera model looking straight down from xyz (tilted about its x axis), as RobotEnv.camera_model gives one."""
    t = np.radians(tilt_deg)
    tilt = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float) @ tilt
    T[:3, 3] = xyz
    return {"K": K, "D": np.zeros(5), "T_world_cam": T, "plane_z_m": 0.0}


def rim_mask(model: dict, rim_m: float, poly=TUB, size=(480, 640)) -> np.ndarray:
    """The tub's outline from above in a camera: its rim, rim_m up."""
    view = PlaneView.from_camera_model(model)
    px = view.project(np.c_[np.asarray(poly) * 1000.0, np.full(len(poly), rim_m * 1000.0)])
    out = np.zeros(size, np.uint8)
    cv2.fillPoly(out, [px.astype(np.int32)], 1)
    return out.astype(bool)


# ── How high a rim stands, from two looks ─────────────────────────────────────

@pytest.mark.parametrize("rim", [0.05, 0.09, 0.13])
@pytest.mark.parametrize("tilt", [0.0, 20.0, -15.0])
def test_two_looks_across_a_wall_say_how_high_its_rim_stands(rim, tilt):
    a, b = down_camera([0.08, -0.15, 0.22], tilt), down_camera([0.12, -0.15, 0.22], tilt)
    got = plane_sweep.sweep(rim_mask(a, rim), a, rim_mask(b, rim), b, np.arange(0.0, 0.151, 0.002))
    assert got.height_m == pytest.approx(rim, abs=0.005)
    assert got.spread_m[1] - got.spread_m[0] <= 0.012


def test_two_looks_along_a_wall_cannot_tell_how_high_it_is():
    # A straight edge seen from two points along it looks the same at every height.
    a, b = down_camera([0.10, -0.17, 0.22]), down_camera([0.10, -0.13, 0.22])
    got = plane_sweep.sweep(rim_mask(a, 0.09), a, rim_mask(b, 0.09), b, np.arange(0.0, 0.151, 0.002))
    assert got.spread_m[1] - got.spread_m[0] > 0.05


def test_an_outline_that_is_only_the_pictures_border_says_nothing():
    a, b = down_camera([0.25, -0.17, 0.22]), down_camera([0.29, -0.17, 0.22])
    full = np.ones((480, 640), bool)
    got = plane_sweep.sweep(full, a, full, b, np.arange(0.0, 0.151, 0.002))
    assert got.height_m is None and "edge" in got.why


def tub_picture(model: dict, rim_m: float, poly=TUB, size=(480, 640), inside=()) -> np.ndarray:
    """
    What a camera sees of the tub: a table with a grain of short marks every
    way, the edge of the tub's floor (2 cm in, 3 mm up), its rim's lip (the rim
    and, 4 mm out, 2 mm lower), and the round tops of what stands in it
    (inside: [((x, y), top)]).
    """
    view = PlaneView.from_camera_model(model)
    img = np.full((*size, 3), 215, np.uint8)
    rng = np.random.default_rng(3)

    def draw(pts_m, z, colour, closed=True, thick=2):
        pts = np.c_[np.asarray(pts_m, float) * 1000.0, np.full(len(pts_m), z * 1000.0)]
        if ((pts @ view.R.T + view.t)[:, 2] <= 1.0).any():
            return
        cv2.polylines(img, [np.round(view.project(pts)).astype(np.int32)], closed, colour, thick, cv2.LINE_AA)
    for xy in rng.uniform([-0.1, -0.5], [0.6, 0.3], (900, 2)):
        draw([xy, xy + rng.normal(0.0, 0.004, 2)], 0.0, (165, 165, 165), closed=False, thick=1)
    lo, hi = np.asarray(poly).min(axis=0), np.asarray(poly).max(axis=0)
    box = lambda d: np.array([[lo[0] - d, lo[1] - d], [hi[0] + d, lo[1] - d], [hi[0] + d, hi[1] + d], [lo[0] - d, hi[1] + d]])
    draw(box(-0.02), 0.003, (185, 185, 185))
    draw(box(0.0), rim_m, (120, 120, 120))
    draw(box(0.004), rim_m - 0.002, (150, 150, 150))
    for (x, y), top in inside:
        draw([(x + 0.011 * np.cos(t), y + 0.011 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)],
             top, (90, 60, 60))
    return img


def wall_band(model: dict, y_wall: float, size=(480, 640), px: int = 40) -> np.ndarray:
    """Where the tub's wall along x at y_wall can be in a view, at any height up to 17 cm."""
    view = PlaneView.from_camera_model(model)
    band = np.zeros(size, np.uint8)
    for z in np.linspace(0.0, 0.17, 8):
        pts = np.array([[x * 1000.0, y_wall * 1000.0, z * 1000.0] for x in np.linspace(0.18, 0.34, 9)])
        cv2.polylines(band, [np.round(view.project(pts)).astype(np.int32)], False, 1, thickness=2 * px)
    return band.astype(bool)


@pytest.mark.parametrize("rim", [0.05, 0.07, 0.11])
@pytest.mark.parametrize("tilt", [0.0, 12.0])
def test_the_edges_of_two_looks_across_a_wall_say_how_high_its_rim_stands(rim, tilt):
    # Through the near wall (y = -0.05): the table's grain at 0, the floor at 3 mm, a figure's top at 40, the rim.
    a, b = down_camera([0.26, -0.03, 0.23], tilt), down_camera([0.26, -0.07, 0.23], tilt)
    figure = [((0.28, -0.09), 0.04)]
    pa, pb = tub_picture(a, rim, inside=figure), tub_picture(b, rim, inside=figure)
    # Round the wall's line, as place_in looks: within a few mm.
    banded = plane_sweep.sweep_edges(pa, a, pb, b, np.arange(0.0, 0.171, 0.002),
                                     keep_a=wall_band(a, -0.05), keep_b=wall_band(b, -0.05))
    assert banded.height_m is not None and rim - 0.004 <= banded.height_m <= rim + 0.012, banded
    # The whole picture: a higher alignment nearly as strong is taken instead — high, not low.
    got = plane_sweep.sweep_edges(pa, a, pb, b, np.arange(0.0, 0.171, 0.002))
    assert got.height_m is not None and rim - 0.004 <= got.height_m <= rim + 0.03, got


def test_edges_two_looks_along_a_wall_see_say_nothing_of_its_height():
    # Along the wall, its edges land on themselves at every height: none of them count.
    a, b = down_camera([0.24, -0.05, 0.23]), down_camera([0.28, -0.05, 0.23])
    got = plane_sweep.sweep_edges(tub_picture(a, 0.08), a, tub_picture(b, 0.08), b, np.arange(0.0, 0.171, 0.002),
                                  keep_a=wall_band(a, -0.05, px=12), keep_b=wall_band(b, -0.05, px=12))
    assert got.height_m is None or abs(got.height_m - 0.08) > 0.01


def test_a_blank_picture_says_nothing():
    a, b = down_camera([0.26, -0.03, 0.23]), down_camera([0.26, -0.07, 0.23])
    blank = np.full((480, 640, 3), 200, np.uint8)
    got = plane_sweep.sweep_edges(blank, a, blank, b, np.arange(0.0, 0.171, 0.002))
    assert got.height_m is None and "edges" in got.why


# ── Finding it by name ────────────────────────────────────────────────────────

def _asker(reply):
    asked = []

    def ask(prompt, image, _tag=None, model=None, schema=None):
        asked.append({"prompt": prompt, "tag": _tag, "model": model, "schema": schema})
        return reply(prompt) if callable(reply) else reply
    return ask, asked


def test_a_qwen_box_is_read_in_pixels_and_a_gemma_box_as_thousandths_y_first():
    frame = np.zeros((720, 1280, 3), np.uint8)
    ask, asked = _asker({"found": True, "bbox_2d": [167, 38, 515, 567]})
    assert grounding.locate(frame, "the tub", model="qwen2.5vl:7b", ask=ask) == ((167, 38, 515, 567), "")
    assert "bbox_2d" in asked[0]["schema"]["properties"]
    ask, asked = _asker({"found": True, "box_2d": [42, 131, 793, 402]})
    box, why = grounding.locate(frame, "the tub", model="gemma4:31b", ask=ask)
    assert np.allclose(box, (167.68, 30.24, 514.56, 570.96)) and why == ""
    assert "box_2d" in asked[0]["schema"]["properties"]


@pytest.mark.parametrize("reply, why", [
    ({"found": False, "bbox_2d": [0, 0, 0, 0]}, "does not see"),
    ({"found": True, "bbox_2d": [5000, 10, 6000, 20]}, "no area"),
    ({"found": True, "bbox_2d": "near the top"}, "not four numbers"),
    ({}, "no answer"),
])
def test_no_box_is_no_answer(reply, why):
    ask, _ = _asker(reply)
    box, said = grounding.locate(np.zeros((720, 1280, 3), np.uint8), "the tub", model="qwen2.5vl:7b", ask=ask)
    assert box is None and why in said


def test_the_outline_is_the_best_candidate_that_fills_the_box():
    frame = np.zeros((100, 100, 3), np.uint8)
    speck, whole, spill = (np.zeros((100, 100), bool) for _ in range(3))
    speck[40:45, 40:45] = True                # inside, but no more than a speck of it
    whole[20:80, 20:80] = True                # the thing
    spill[:, :] = True                        # everything: out of the box

    def segment_box(f, box):
        return [speck, whole, spill], [0.99, 0.9, 0.95]
    mask, score, why = grounding.outline(frame, (20, 20, 80, 80), segment_box=segment_box)
    assert mask is whole and score == 0.9 and why == ""


def test_whether_it_is_a_container_is_the_models_answer_on_a_crop():
    frame = np.zeros((720, 1280, 3), np.uint8)

    def reply(prompt):
        return {"found": True, "bbox_2d": [100, 100, 300, 300]} if prompt.startswith("Find") else {"container": False}
    ask, asked = _asker(reply)
    found, why = grounding.find(frame, "the red figure", model="qwen2.5vl:7b", ask=ask, judge_model="gemma4:31b",
                                segment_box=lambda f, b: ([np.ones((720, 1280), bool)], [0.9]))
    assert found is None and "not take it for something to put things into" in why
    assert asked[1]["model"] == "gemma4:31b" and "container" in asked[1]["schema"]["properties"]


def test_a_container_outline_is_carried_onto_the_table_inward_never_out():
    overhead = down_camera([0.36, 0.04, 0.52], -8.0)
    rim = 0.09
    mask = rim_mask(overhead, rim) | rim_mask(overhead, 0.0)      # rim on the far side, foot on the near
    found = containers.Container("tub", "the tub", "overhead", (640, 480), containers.outline_px(mask), rim)
    raised = found.outline_xy(overhead)        # the near side carried up toward the camera: out past the rim
    assert min(inside_by(TUB, p) for p in raised) < -0.01
    xy = found.opening_xy(overhead)
    inside = [inside_by(TUB, p) for p in xy]
    assert min(inside) > -0.004                # never further out than the rim (a few mm of pixels)
    assert max(inside) < 0.03                  # nor far in
    assert found.contains(overhead, (0.25, -0.15)) and not found.contains(overhead, (0.25, 0.05))


# ── The place into it, on the fake arm ────────────────────────────────────────

def picked(arm, pol, xy=(0.25, 0.05)):
    piece = Cylinder(xy, 0.021, 0.045)
    arm.objects.append(piece)
    got = pol.pick(arm, Target(np.r_[xy, 0.0], foot_m=0.021, height_m=0.045,
                               widths_mm={4: 21.0, 20: 21.0, 40: 21.0}))
    assert got.ok, got.message
    return piece, got.grasp


def test_it_goes_over_the_walls_and_lets_go_inside():
    arm, pol = FakeArm(), ParallelJaw(SPEC)
    piece, grasp = picked(arm, pol)
    start = len(arm.path)
    out = pol.place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=grasp)
    assert out.ok, out.message
    assert arm.held is None and inside_by(TUB, piece.xy) > 0.04
    hang = grasp.grasp_height_m
    for p in arm.path[start:]:
        if abs(inside_by(TUB, p[:2])) < 0.01:            # crossing the wall line
            assert p[2] - hang >= 0.09 + SPEC.container_wall_clearance_m - 1e-6, p


def test_walls_not_measured_are_passed_as_high_as_they_may_be():
    arm, pol = FakeArm(), ParallelJaw(SPEC)
    _, grasp = picked(arm, pol)
    start = len(arm.path)
    out = pol.place_in(arm, Opening(TUB, 0.15, False, "tub"), grasp=grasp)
    assert out.ok and out.measured is None, out.message          # the fake arm has no wrist camera to look with
    crossing = [p[2] for p in arm.path[start:] if abs(inside_by(TUB, p[:2])) < 0.01]
    assert crossing and min(crossing) - grasp.grasp_height_m >= 0.15


def test_a_spot_keeps_clear_of_the_walls_and_of_what_is_in_it():
    from mrcf.policies.base import Obstacle

    arm, pol = FakeArm(), ParallelJaw(SPEC)
    inside = [Obstacle(np.array([0.27, -0.17]), 0.03, "ball", 0.03)]
    got = spots(pol, arm, Opening(TUB, 0.09, True), 0.04, inside, held_half_m=0.0105)
    assert got
    for xy in got:
        assert inside_by(TUB, xy) >= 0.04
        assert np.linalg.norm(np.asarray(xy) - (0.27, -0.17)) >= 0.0105 + 0.03 + SPEC.container_wall_clearance_m


@pytest.mark.parametrize("opening, spec, why", [
    (Opening(np.array([[0.2, -0.1], [0.23, -0.1], [0.23, -0.07]]), 0.05, True), SPEC, "no room inside"),
    (Opening(TUB, 0.09, True), dataclasses.replace(SPEC, place_in=False), "not been set up"),
    (Opening(TUB[:2], 0.09, True), SPEC, "three points"),
])
def test_refused_without_letting_go(opening, spec, why):
    arm = FakeArm()
    piece, grasp = picked(arm, ParallelJaw(SPEC))
    out = ParallelJaw(spec).place_in(arm, opening, grasp=grasp)
    assert out.refused and why in out.message
    assert arm.held is piece


def test_no_clear_room_it_is_let_go_over_the_walls_onto_what_is_in_it():
    """2026-11-19: the scoop refused — nowhere clear of the walls and of the figure in the tub's middle."""
    from mrcf.policies.base import Obstacle

    arm = FakeArm()
    piece, grasp = picked(arm, ParallelJaw(SPEC))
    crowded = [Obstacle(np.array([0.275, -0.175]), 0.12, "pile", 0.05)]       # fills the middle
    out = ParallelJaw(SPEC).place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=grasp, obstacles=crowded)
    assert out.refused and "no room inside" in out.message and arm.held is piece
    spec = dataclasses.replace(SPEC, place_in_over_contents=True)
    start = len(arm.path)
    out = ParallelJaw(spec).place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=grasp, obstacles=crowded)
    assert out.ok and arm.held is None, out.message
    inside = [p for p in arm.path[start:] if inside_by(TUB, p[:2]) > 0.01]
    assert inside and min(p[2] for p in inside) - grasp.grasp_height_m >= 0.09    # let go from over the walls


class _LeaningArm(FakeArm):
    """Reaches anywhere, leaning 35 deg past x 0.30 (the tub's far end) and 10 deg nearer; stopped at 80 mm there."""
    def __init__(self):
        super().__init__()
        self.downs = 0

    def approach_orientation_for(self, targets, gap_m=None, tols=None, flip=False, prefer_tilt_deg=None,
                                 margin_deg=None, turn_deg=0.0):
        return np.eye(3), (-35.0 if max(float(t[0]) for t in targets) > 0.30 else -10.0)

    def move_ee_to(self, target, tol=0.01, orientation=None, **k):
        target = np.asarray(target, float)
        if target[2] < 0.08 and self.ee[0] > 0.30:            # something in the way down there
            self.downs += 1
            target = np.r_[target[:2], 0.08]
        return super().move_ee_to(target, tol=tol, orientation=orientation, **k)


def test_no_spot_reached_leaning_far_out_and_a_stopped_way_down_is_not_pushed_again():
    """2026-11-19 11:18: the far corner reached leaning 35 deg, the descent stopped by the wall and tried again."""
    from mrcf.policies.base import Obstacle

    near_full = [Obstacle(np.array([0.20, -0.175]), 0.09, "pile", 0.05)]     # the clear room is the far end
    arm = _LeaningArm()
    _, grasp = picked(arm, ParallelJaw(SPEC), xy=(0.25, 0.05))
    out = ParallelJaw(SPEC).place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=grasp, obstacles=near_full)
    assert out.ok and out.at_xy[0] > 0.30 and arm.downs == 2              # as it was: far out, pushed twice

    arm = _LeaningArm()
    _, grasp = picked(arm, ParallelJaw(SPEC), xy=(0.25, 0.05))
    no_push = dataclasses.replace(SPEC, container_descent_retry=False)
    start = len(arm.path)
    out = ParallelJaw(no_push).place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=grasp, obstacles=near_full)
    assert out.ok and arm.downs == 1 and "dropped into" in out.message   # once, then up and let go over the walls
    assert arm.path[-1][2] - grasp.grasp_height_m >= 0.09

    arm = _LeaningArm()
    _, grasp = picked(arm, ParallelJaw(SPEC), xy=(0.25, 0.05))
    upright = dataclasses.replace(no_push, container_max_lean_deg=20.0, place_in_over_contents=True)
    out = ParallelJaw(upright).place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=grasp, obstacles=near_full)
    assert out.ok and out.at_xy[0] <= 0.30 and arm.downs == 0, out.message   # nearer, onto what is there
    assert start > 0
    # Only far out is there room, and dropping from over the walls may lean that far: let go there, not lowered.
    arm = _LeaningArm()
    _, grasp = picked(arm, ParallelJaw(SPEC), xy=(0.25, 0.05))
    drops = dataclasses.replace(no_push, container_max_lean_deg=20.0, container_max_drop_lean_deg=45.0)
    out = ParallelJaw(drops).place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=grasp, obstacles=near_full)
    assert out.ok and out.at_xy[0] > 0.30 and arm.downs == 0 and "dropped into" in out.message, out.message



def _lean(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


class _HighReachLeansArm(FakeArm):
    """
    The SO-101 over its tub (2026-11-19, runs 36-37): over a spot high up it
    reaches only leaning 35 deg; with the jaws among the walls 10 deg will do
    — 25 deg past x `far_x`. Every move's goal and orientation kept.
    """
    def __init__(self, among_z: float, far_x: float = 9.0):
        super().__init__()
        self.among_z, self.far_x, self.moves = among_z, far_x, []

    def approach_orientation_for(self, targets, gap_m=None, tols=None, flip=False, prefer_tilt_deg=None,
                                 margin_deg=None, turn_deg=0.0):
        high = max(float(t[2]) for t in targets) > self.among_z + 1e-6
        far = max(float(t[0]) for t in targets) > self.far_x
        deg = -35.0 if high else (-25.0 if far else -10.0)
        return _lean(deg), deg

    def move_ee_to(self, target, tol=0.01, orientation=None, **k):
        self.moves.append((np.asarray(target, float).copy(), None if orientation is None else np.asarray(orientation)))
        return super().move_ee_to(target, tol=tol, orientation=orientation, **k)


def test_going_down_inside_leans_as_far_as_over_the_spot_takes_only_above_the_walls():
    """2026-11-19, runs 36-37: 6 of 9 let go 13-18 cm up — one lean all the way down, the top's (30-40 deg)."""
    tub = Opening(TUB, 0.09, True, "tub")
    among = SPEC.table_z + 0.09 + SPEC.container_wall_clearance_m + SPEC.tip_below_grasp_m
    leans = dataclasses.replace(SPEC, container_descent_retry=False, container_max_lean_deg=20.0,
                                container_max_drop_lean_deg=45.0)
    arm = _HighReachLeansArm(among)
    _, grasp = picked(arm, ParallelJaw(leans))
    out = ParallelJaw(leans).place_in(arm, tub, grasp=grasp)
    assert out.ok and "dropped into" in out.message, out.message        # as it was: one lean, the top's
    assert arm.path[-1][2] > among

    stepped = dataclasses.replace(leans, container_lean_above_walls=True)
    arm = _HighReachLeansArm(among)
    _, grasp = picked(arm, ParallelJaw(stepped))
    start = len(arm.moves)
    out = ParallelJaw(stepped).place_in(arm, tub, grasp=grasp)
    assert out.ok and "put into" in out.message, out.message
    moves = arm.moves[start:]
    low = min(m[0][2] for m in moves)
    assert low < among - 0.03                                            # down inside
    for goal, rot in moves:
        if goal[2] < among - 1e-6:
            assert np.allclose(rot, _lean(-10.0))                        # among the walls: 10 deg
    assert any(np.allclose(rot, _lean(-35.0)) for goal, rot in moves if goal[2] > among + 1e-6)
    i_low = next(i for i, (g, _) in enumerate(moves) if g[2] == low)
    assert any(np.isclose(g[2], among) for g, _ in moves[i_low:])        # back up the way it went down


def test_a_spot_it_can_go_down_inside_is_chosen_before_one_it_can_only_drop_into():
    tub = Opening(TUB, 0.09, True, "tub")
    among = SPEC.table_z + 0.09 + SPEC.container_wall_clearance_m + SPEC.tip_below_grasp_m
    spec = dataclasses.replace(SPEC, container_descent_retry=False, container_max_lean_deg=20.0,
                               container_max_drop_lean_deg=45.0, container_lean_above_walls=True)
    arm = _HighReachLeansArm(among, far_x=0.20)                          # the roomiest spots all past x 0.20
    _, grasp = picked(arm, ParallelJaw(spec))
    out = ParallelJaw(spec).place_in(arm, tub, grasp=grasp)
    assert out.ok and "put into" in out.message and out.at_xy[0] <= 0.20, out.message
    arm = _HighReachLeansArm(among, far_x=0.0)                           # nowhere among the walls within 20 deg
    _, grasp = picked(arm, ParallelJaw(spec))
    out = ParallelJaw(spec).place_in(arm, tub, grasp=grasp)
    assert out.ok and "dropped into" in out.message, out.message        # over the walls, as before


def test_a_pick_that_starts_over_the_tub_goes_up_over_its_walls_before_crossing_them():
    """2026-11-19 11:18: from the tub, 112 mm up, straight to the next hover at 61 mm: the jaws dragged the tub."""
    def run(spec):
        arm = FakeArm()
        pol = ParallelJaw(spec)
        pol.walls = lambda: [Opening(TUB, 0.09, True, "tub")]
        arm.ee = np.array([0.30, -0.12, 0.05])              # just let go of something inside, low
        arm.objects.append(Cylinder((0.30, 0.06), 0.021, 0.045))
        start = len(arm.path)
        got = pol.pick(arm, Target(np.r_[0.30, 0.06, 0.0], foot_m=0.021, height_m=0.045,
                                   widths_mm={4: 21.0, 20: 21.0, 40: 21.0}))
        assert got.ok, got.message
        reach = spec.container_wall_clearance_m + spec.finger_thickness_m
        return [p for p in arm.path[start:] if inside_by(TUB, p[:2]) > -reach]

    over = run(SPEC)                                         # off: straight there, low across the wall
    assert min(p[2] for p in over) - SPEC.tip_below_grasp_m < 0.09
    over = run(dataclasses.replace(SPEC, transit_over_walls=True))
    first_out = next(i for i, p in enumerate(over) if p[2] - SPEC.tip_below_grasp_m >= 0.09)
    assert all(p[2] - SPEC.tip_below_grasp_m >= 0.09 for p in over[first_out:])         # up, then over
    assert all(abs(p[0] - 0.30) < 1e-6 and abs(p[1] + 0.12) < 1e-6 for p in over[:first_out])   # straight up


class _SettlingArm(FakeArm):
    """Records whether each move was waited on to settle."""
    def __init__(self):
        super().__init__()
        self.settled = []

    def move_ee_to(self, target, tol=0.01, orientation=None, settle=True, **k):
        self.settled.append((round(float(np.asarray(target)[2]), 3), settle))
        return super().move_ee_to(target, tol=tol, orientation=orientation, **k)


def test_moves_that_only_travel_go_on_without_settling():
    """User 2026-11-19: "its very jerky" — every move ended creeping onto its goal for 1-1.5 s."""
    arm = _SettlingArm()
    _, grasp = picked(arm, ParallelJaw(SPEC))
    arm.settled.clear()
    out = ParallelJaw(dataclasses.replace(SPEC, transit_settle=False)).place_in(arm, Opening(TUB, 0.09, True, "tub"),
                                                                                grasp=grasp)
    assert out.ok, out.message
    lowest = min(z for z, _ in arm.settled)
    assert [settle for z, settle in arm.settled if z == lowest] == [True]           # the way down inside settles
    assert not any(settle for z, settle in arm.settled if z > lowest)               # up, across, back up do not


def test_held_by_its_wall_it_crosses_into_the_tub_higher_for_its_swing():
    """2026-11-19 12:05: a scoop held by the very tip of its rim swung on the way across and hit the tub's wall."""
    from mrcf.policies.parallel_jaw.spec import GraspPlan

    wall = GraspPlan(height_m=0.05, width_m=0.11, grasp_height_m=0.03, grip_width_m=0.004, pregrasp_open_m=0.024,
                     hover_height_m=0.08, lift_height_m=0.09, release_height_m=0.04)

    def crossing(spec):
        arm = FakeArm()
        arm.ee = np.array([0.25, 0.05, 0.10])
        arm.held = Cylinder((0.25, 0.05), 0.05, 0.05)
        pol = ParallelJaw(spec)
        pol.holding = lambda env: True
        start = len(arm.path)
        out = pol.place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=wall)
        assert out.ok, out.message
        return min(p[2] for p in arm.path[start:] if abs(inside_by(TUB, p[:2])) < 0.01)

    plain = crossing(SPEC)
    swung = crossing(dataclasses.replace(SPEC, rim_swing_share=0.33))
    assert swung - plain == pytest.approx(0.33 * 0.11 - 0.0, abs=0.005)


def test_how_high_it_crosses_the_walls_depends_on_where_it_was_gripped():
    """User 2026-11-19: "based on where it grips the object it needs to lift it a different height"."""
    from mrcf.policies.parallel_jaw.spec import GraspPlan
    from mrcf.policies.place_in import _turned_hang

    plug = GraspPlan(height_m=0.055, width_m=0.054, grasp_height_m=0.020, grip_width_m=0.053, pregrasp_open_m=0.062,
                     hover_height_m=0.08, lift_height_m=0.09, release_height_m=0.03)
    assert _turned_hang(plug, 0.053) == pytest.approx(0.020)                     # through its middle: as gripped
    assert _turned_hang(plug, 0.027) == pytest.approx(np.hypot(0.035, 0.027))    # off it: its far corner
    scoop = dataclasses.replace(plug, height_m=0.05, width_m=0.11, grasp_height_m=0.031, grip_width_m=0.004)
    assert _turned_hang(scoop, 0.001) == pytest.approx(np.hypot(0.031, 0.055))   # by its wall

    def crossing(spec, jaws):
        arm = FakeArm()
        arm.ee = np.array([0.25, 0.05, 0.10])
        arm.held = Cylinder((0.25, 0.05), jaws, 0.055)
        arm.get_gripper_width = lambda: jaws                    # the fake reads 3 mm whatever it holds
        pol = ParallelJaw(spec)
        pol.holding = lambda env: True
        start = len(arm.path)
        out = pol.place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=plug)
        assert out.ok, out.message
        return min(p[2] for p in arm.path[start:] if abs(inside_by(TUB, p[:2])) < 0.01)

    turns = dataclasses.replace(SPEC, held_turns=True)
    assert crossing(turns, 0.053) == pytest.approx(crossing(SPEC, 0.053), abs=0.002)      # gripped well: as before
    assert crossing(turns, 0.027) - crossing(SPEC, 0.027) == pytest.approx(np.hypot(0.035, 0.027) - 0.020, abs=0.003)


class _ShutOnAWall:
    """Jaws stopped 1.1 mm apart on a scoop's wall: the servo feels no load."""
    def __init__(self):
        self.gap = 0.0011

    def gripper_loaded(self):
        return False

    def get_gripper_width(self):
        return self.gap


def test_a_wall_felt_under_load_is_held_while_the_jaws_stay_where_they_stopped():
    """2026-11-19, three times: the scoop held by its wall, called "left the jaws", carried home in them."""
    env = _ShutOnAWall()
    pol = ParallelJaw(dataclasses.replace(SPEC, held_by_camera=True, held_while_still=True))
    pol._asked_holding = lambda e: False                       # the model says no, as it did
    pol._held_at(env, 0.0011)
    assert pol.holding(env)                                    # no picture needed: the jaws say so
    env.gap = 0.0003
    assert not pol.holding(env)                                # shut on: it slipped out


def test_seen_carried_it_is_held_while_the_jaws_stand_still():
    """2026-11-19 11:09: the scoop seen carried at the lift, then "the jaws are empty" from home."""
    env = _ShutOnAWall()
    seen = []

    def run(spec):
        pol = ParallelJaw(spec)
        pol._grip_view = {"frame": None, "model": None, "gap": 0.0011, "said": False}
        pol._seen_holding = lambda e: bool(seen) and seen.pop(0)
        pol._asked_holding = lambda e: False
        return pol

    spec = dataclasses.replace(SPEC, held_by_camera=True)
    pol = run(spec)
    seen[:] = [True]
    assert pol.holding(env) and not pol.holding(env)              # off: each look decides afresh
    pol = run(dataclasses.replace(spec, held_while_still=True))
    seen[:] = [True]
    assert pol.holding(env) and pol.holding(env)                  # seen once; the jaws have not moved
    env.gap = 0.0004
    assert not pol.holding(env)                                   # closed on: it slipped out
    env.gap, pol._grip_view["held_gap"] = 0.0006, 0.0006
    assert not pol.holding(env)                                   # shut that far, an empty close looks the same


def test_the_nearest_wall_and_the_way_inside_from_it():
    q, n = nearest_wall(TUB, (0.30, -0.06))
    assert np.allclose(q, (0.30, -0.05)) and np.allclose(n, (0.0, -1.0))


def test_with_its_walls_measured_it_is_lowered_inside_the_gripper_kept_over_the_rim():
    arm, pol = FakeArm(), ParallelJaw(SPEC)
    piece, grasp = picked(arm, pol)
    start = len(arm.path)
    out = pol.place_in(arm, Opening(TUB, 0.09, True, "tub"), grasp=grasp)
    assert out.ok and out.released and out.message.startswith("put into"), out.message
    low = min(arm.path[start:], key=lambda p: p[2])
    hang = grasp.grasp_height_m
    assert inside_by(TUB, low[:2]) > 0.03
    # what is held just over the floor, or the gripper's body over the rim: whichever is higher
    assert low[2] == pytest.approx(max(SPEC.container_floor_m + hang + SPEC.release_clearance_m,
                                       0.09 + SPEC.container_wall_clearance_m - SPEC.free_depth_m), abs=0.003)
    assert arm.ee[2] - hang >= 0.09 + SPEC.container_wall_clearance_m - 1e-6     # and straight back up


class _LookingArm(FakeArm):
    """The fake arm with a wrist camera looking straight down from 12 cm above its grasp point."""

    def approach_orientation_for(self, targets, **_):
        return np.eye(3), 0.0

    def camera_model_at(self, camera, width, height, at, rotation, gap_m=None):
        return down_camera(np.asarray(at, float) + [0.0, 0.0, 0.12])

    def camera_model(self, camera, width, height):
        return self.camera_model_at(camera, width, height, self.ee, None)


def test_the_wrist_camera_finds_how_high_its_walls_stand_and_passes_them_only_that_high():
    spec = dataclasses.replace(SPEC, camera="wrist", render_w=640, render_h=480, container_look_settle_s=0.0)
    arm, pol = _LookingArm(), ParallelJaw(spec)
    looked = []

    def frame(env, tag):
        looked.append(arm.ee.copy())
        return tub_picture(arm.camera_model("wrist", 640, 480), 0.07)
    pol._wrist_frame = frame
    _, grasp = picked(arm, ParallelJaw(SPEC))
    start = len(arm.path)
    out = pol.place_in(arm, Opening(TUB, 0.15, False, "tub"), grasp=grasp)
    assert out.ok, out.message
    assert out.measured is not None and 66.0 <= out.measured.height_mm <= 82.0, out.measured
    # two looks a turn of the base apart: as far from it, 4 cm between them
    a, b = (np.asarray(p[:2]) - np.asarray(arm.base_xy, float)[:2] for p in looked)
    assert np.linalg.norm(a) == pytest.approx(np.linalg.norm(b), abs=1e-3)
    assert np.linalg.norm(a - b) == pytest.approx(0.04, abs=2e-3)
    rim = out.measured.height_mm / 1000.0 + spec.container_rim_margin_m
    after = arm.path[start + [i for i, p in enumerate(arm.path[start:]) if np.allclose(p, looked[-1])][-1]:]
    crossing = [p[2] for p in after if abs(inside_by(TUB, p[:2])) < 0.01]
    assert not crossing or min(crossing) - grasp.grasp_height_m >= rim + spec.container_wall_clearance_m - 1e-6
    assert min(p[2] for p in after) < 0.09                  # lowered inside, not dropped from over 15 cm walls


# ── Through the planner's actions ─────────────────────────────────────────────

def _scene(arm):
    memory = ObjectMemory()
    (track,), _ = memory.observe([Detection(position=np.array([0.25, 0.05]), foot_mm=21.0)])
    ball = SceneObject("ball", "a ball", position=np.array([0.25, 0.05, 0.0]))
    ball.track_id, ball.geometry = track.id, {"base_diameter_m": 0.021}
    ball.measured_shape = {"height_mm": 45.0, "widths_mm": {4: 21.0, 20: 21.0, 40: 21.0}}
    arm.objects.append(Cylinder((0.25, 0.05), 0.021, 0.045))
    tub = SceneObject("tub", "a container", interactable=False, position=np.array([0.27, -0.17, 0.0]))
    return SceneState(objects=[ball], targets=[tub], memory=memory)


def _install(kept):
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.15,
        opening=lambda state, name: ((Opening(TUB, 0.09, True, "tub"), state.find("tub"))
                                     if name == "tub" else None),
        holds=lambda state, obj: "tub" if obj.position is not None and inside_by(TUB, obj.position[:2]) > 0 else None,
        opening_measured=lambda state, name, rim: kept.append((name, rim))))
    manipulation.install_place_in()


@pytest.mark.parametrize("action", [{"action": "place_in", "container": "tub"},
                                    {"action": "place", "target_name": "tub"}])
def test_into_or_at_a_container_both_put_it_inside(action):
    arm = FakeArm()
    _install([])
    state = _scene(arm)
    ball = state.find("ball")
    assert dispatch({"action": "pick", "object_name": "ball"}, arm, state) is True
    assert dispatch(action, arm, state) is True
    assert ball.position is not None and inside_by(TUB, ball.position[:2]) > 0.03
    assert "in tub" in ball.description


def test_a_pick_by_its_wall_outlines_it_again_first():
    arm, asked = FakeArm(), []
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.15, outline_again=lambda env, state, obj: asked.append(obj.name) or True))
    manipulation.install_pick_by_rim()
    state = _scene(arm)
    dispatch({"action": "pick_by_rim", "object_name": "ball"}, arm, state)
    assert asked == ["ball"]


def test_a_pick_out_of_a_container_is_refused_before_anything_moves():
    arm = FakeArm()
    _install([])
    state = _scene(arm)
    state.find("ball").position = np.array([0.27, -0.17, 0.0])
    moved = len(arm.path)
    assert dispatch({"action": "pick", "object_name": "ball"}, arm, state) is False
    assert len(arm.path) == moved


# ── The wrist camera's view of its own jaws ───────────────────────────────────

from mrcf.perception import jaw_view  # noqa: E402


def _jaws(gap_m: float, size=(480, 640)) -> np.ndarray:
    """Where synthetic jaws are: a fixed one on the left, a moving one further right the wider they open."""
    h, w = size
    m = np.zeros(size, np.uint8)
    cv2.fillPoly(m, [np.array([[150, h], [220, h - 160], [270, h - 150], [260, h]])], 1)
    x = 300 + int(gap_m * 4000)
    cv2.fillPoly(m, [np.array([[x, h], [x + 10, h - 150], [x + 60, h - 160], [x + 130, h]])], 1)
    return m.astype(bool)


def _picture(gap_m: float, spot: int, size=(480, 640)) -> np.ndarray:
    """A white table, the jaws in black with a white mark, and dark things that are elsewhere at each spot."""
    rng = np.random.default_rng(spot)
    frame = np.full(size + (3,), 225, np.uint8) + rng.integers(0, 8, size + (3,), dtype=np.uint8)
    frame[_jaws(gap_m, size)] = 25
    x = 300 + int(gap_m * 4000)
    cv2.circle(frame, (x + 40, size[0] - 120), 9, (240, 240, 240), -1)           # the moving jaw's white disc
    cv2.circle(frame, (60 + 180 * spot, 120 + 50 * spot), 35, (40, 30, 30), -1)  # an object, moved each time
    cv2.rectangle(frame, (20 + 120 * spot, size[0] - 60), (90 + 120 * spot, size[0]), (60, 60, 60), -1)  # a shadow at the bottom
    return frame


def test_the_jaws_are_what_is_dark_from_the_bottom_in_every_place():
    view = jaw_view.learn({g: [_picture(g, spot) for spot in range(3)] for g in (0.0, 0.02, 0.04)})
    for g, m in zip(view.openings_m, view.masks):
        truth = _jaws(g)
        assert (m & truth).sum() / truth.sum() > 0.97                   # all of the jaws, the disc's gap closed
        assert (m & ~truth).sum() / truth.sum() < 0.1                   # and next to nothing else
    between = view.at(0.03)
    assert between[_jaws(0.02)].mean() > 0.97 and between[_jaws(0.04)].mean() > 0.97


def test_one_place_is_not_enough_to_tell_the_jaws_from_a_shadow():
    one = jaw_view.jaws_in([_picture(0.02, 1)])
    assert (one & ~_jaws(0.02)).sum() > 1000


def test_a_learned_view_is_kept_and_read_back(tmp_path):
    view = jaw_view.learn({g: [_picture(g, spot) for spot in range(3)] for g in (0.0, 0.04)})
    view.save(tmp_path / "jaws.npz")
    back = jaw_view.JawView.load(tmp_path / "jaws.npz")
    assert back.openings_m == view.openings_m and (back.masks == view.masks).all()
    assert jaw_view.JawView.load(tmp_path / "none.npz") is None


class _JawArm(FakeArm):
    """The fake arm with a wrist camera that sees the synthetic jaws, the scene behind them changing with the spot."""

    def render_image(self, camera, width=640, height=480):
        spot = int(round(self.ee[0] * 100)) % 7
        return _picture(self.width, spot, (height, width))


def test_the_arm_learns_its_jaws_over_three_spots_and_the_looks_use_them(tmp_path):
    spec = dataclasses.replace(SPEC, camera="wrist", render_w=640, render_h=480,
                               jaw_view_openings_m=(0.0, 0.02, 0.04))
    pol = ParallelJaw(spec, jaw_view_path=tmp_path / "jaws.npz")
    arm = _JawArm()
    assert pol.jaw_view() is None
    out = pol.learn_jaw_view(arm, [(0.20, 0.05), (0.22, -0.05), (0.24, 0.10)])
    assert out.ok, out.message
    view = pol.jaw_view()
    assert view is not None and view.openings_m == [0.0, 0.02, 0.04]
    assert (view.masks[1] & _jaws(0.02)).sum() / _jaws(0.02).sum() > 0.97
    arm.width = 0.02
    zone = pol._jaw_zone(arm, None, None, (480, 640))
    assert zone[_jaws(0.02)].all()


def test_learning_the_jaws_is_refused_while_holding_something(tmp_path):
    spec = dataclasses.replace(SPEC, camera="wrist", render_w=640, render_h=480)
    arm = _JawArm()
    _, grasp = picked(arm, ParallelJaw(SPEC))
    out = ParallelJaw(spec, jaw_view_path=tmp_path / "j.npz").learn_jaw_view(arm, [(0.2, 0.0)] * 3)
    assert out.refused and "holds" in out.message and not (tmp_path / "j.npz").exists()


def test_a_container_not_known_yet_is_looked_for_by_what_the_user_calls_it():
    arm, looked = FakeArm(), []
    state = _scene(arm)
    state.targets = []                                  # nobody has asked for the tub yet

    def find_opening(env, scene_state, name):
        looked.append(name)
        scene_state.targets.append(SceneObject("plastic_tub", "a container", interactable=False,
                                               position=np.array([0.27, -0.17, 0.0])))
        return True
    manipulation.install(ParallelJaw(SPEC), scenario=manipulation.Scenario(
        max_height=lambda o: 0.15, find_opening=find_opening,
        opening=lambda st, name: ((Opening(TUB, 0.09, True, "plastic_tub"), st.find("plastic_tub"))
                                  if st.find("plastic_tub", among="targets") is not None and "tub" in name else None)))
    manipulation.install_place_in()
    ball = state.find("ball")
    assert dispatch({"action": "pick", "object_name": "ball"}, arm, state) is True
    assert dispatch({"action": "place_in", "container": "the plastic tub"}, arm, state) is True
    assert looked == ["the plastic tub"] and inside_by(TUB, ball.position[:2]) > 0.03


# ── Touching the surface ──────────────────────────────────────────────────────

class _TableArm(FakeArm):
    """The fake arm over a table at surface_z: its jaw tips (7 mm under the grasp point) stop on it, a little short."""

    def __init__(self, surface_z, **kw):
        super().__init__(**kw)
        self.surface_z = surface_z

    def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None, orientation=None):
        target = np.asarray(target, float).copy()
        floor = self.surface_z + 0.007 - 0.0004                 # pressed 0.4 mm into it
        target[2] = max(target[2], floor)
        reached = super().move_ee_to(target, tol, max_iters, step_size, max_step_deg, orientation)
        if target[2] == floor:
            self.ee[2] += 0.0012 * (np.random.default_rng(len(self.path)).random() - 0.5)   # its readings wobble
        return reached


@pytest.mark.parametrize("surface", [-0.008, 0.0167])
def test_the_shut_jaws_lowered_onto_the_table_say_how_high_it_is(surface):
    spec = dataclasses.replace(SPEC, touch_surface=True)
    arm = _TableArm(surface)
    out = ParallelJaw(spec).touch_surface(arm, (0.25, 0.05), 0.06, -0.04)
    assert out.ok, out.message
    assert out.measured.height_mm / 1000.0 == pytest.approx(surface, abs=0.0025)
    assert arm.width == 0.0 and arm.ee[2] == pytest.approx(0.06)          # shut, and back up


def test_touching_is_refused_where_the_arm_has_not_been_set_up_for_it_and_fails_above_nothing():
    arm = _TableArm(-0.008)
    assert ParallelJaw(SPEC).touch_surface(arm, (0.25, 0.05), 0.06, -0.04).refused
    out = ParallelJaw(dataclasses.replace(SPEC, touch_surface=True)).touch_surface(arm, (0.25, 0.05), 0.06, 0.02)
    assert not out.ok and "nothing stopped" in out.message


def test_dropped_in_from_over_the_walls_when_asked():
    arm, pol = FakeArm(), ParallelJaw(SPEC)
    piece, grasp = picked(arm, pol)
    start = len(arm.path)
    out = pol.place_in(arm, Opening(TUB, 0.15, False, "tub"), grasp=grasp, drop=True)
    assert out.ok and out.released, out.message
    assert arm.held is None and inside_by(TUB, piece.xy) > 0.03
    lowest_over_wall = min(p[2] for p in arm.path[start:] if abs(inside_by(TUB, p[:2])) < 0.01)
    assert lowest_over_wall - grasp.grasp_height_m >= 0.15


# ── Picking by a wall ─────────────────────────────────────────────────────────

def test_a_walls_points_have_outward_normals_whichever_way_the_outline_runs():
    from mrcf.policies.rim_pick import wall_points

    square = np.array([[0.0, 0.0], [0.04, 0.0], [0.04, 0.04], [0.0, 0.04]])
    for poly in (square, square[::-1]):
        pts = wall_points(poly)
        assert pts
        for p, n in pts:
            assert float((p - (0.02, 0.02)) @ n) > 0              # pointing away from the middle


def test_a_handles_edge_has_no_hollow_behind_it_and_a_bowls_wall_has():
    from mrcf.policies.rim_pick import hollow_behind, wall_points

    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    bowl = np.c_[0.30 + 0.03 * np.cos(t), 0.10 + 0.03 * np.sin(t)]          # 60 mm across
    handle = np.array([[0.20, 0.095], [0.272, 0.095], [0.272, 0.105], [0.20, 0.105]])   # 10 mm wide, off its side
    scoop = np.vstack([bowl[(np.abs(bowl[:, 1] - 0.10) > 0.006) | (bowl[:, 0] > 0.30)], handle])
    import cv2
    hull = cv2.convexHull(scoop.astype(np.float32)).reshape(-1, 2)          # an outline round both
    pts = [(np.array([0.24, 0.0951]), np.array([0.0, -1.0])), (np.array([0.33, 0.10]), np.array([1.0, 0.0]))]
    square = np.array([[0.20, 0.095], [0.33, 0.095], [0.33, 0.13], [0.20, 0.13]])
    assert hollow_behind(square, pts[:1])[0] > 0.03
    slim = np.array([[0.20, 0.095], [0.33, 0.095], [0.33, 0.105], [0.20, 0.105]])
    assert hollow_behind(slim, pts[:1])[0] < 0.012                            # a handle: 10 mm
    assert hollow_behind(bowl, [(np.array([0.33, 0.10]), np.array([1.0, 0.0]))])[0] > 0.05
    assert len(wall_points(hull)) > 0


def test_picking_by_a_wall_is_refused_without_an_outline_or_where_the_arm_is_not_set_up_for_it():
    target = Target(np.array([0.3, 0.1, 0.0]), foot_m=0.06, height_m=0.04)
    assert ParallelJaw(SPEC).pick_by_rim(FakeArm(), target).refused
    spec = dataclasses.replace(SPEC, rim_pick=True)
    out = ParallelJaw(spec).pick_by_rim(FakeArm(), target)
    assert out.refused and "outline" in out.message


# ── Kept for a deployment ─────────────────────────────────────────────────────

def test_a_containers_walls_measured_once_are_known_after_a_restart(tmp_path, monkeypatch):
    model = down_camera([0.36, 0.04, 0.52], -8.0)
    mask = rim_mask(model, 0.09)

    def found(frame, camera, name, what, **_):
        return containers.Container(name, what, camera, (640, 480), containers.outline_px(mask), 0.15), ""
    monkeypatch.setattr(containers, "find", found)
    look = lambda env: (np.zeros((480, 640, 3), np.uint8), model)
    state = SceneState(objects=[], targets=[])
    first = containers.Keeper("overhead", look, grounding_model="g", judge_model=None, tallest_m=0.15, table_z=0.0,
                              rims_path=tmp_path / "rims.json")
    tub, _ = first.find(None, state, "the plastic tub")
    assert tub is not None and tub.name == "plastic_tub" and not tub.rim_measured
    first.measured(state, "tub", 0.082)
    again = containers.Keeper("overhead", look, grounding_model="g", judge_model=None, tallest_m=0.15, table_z=0.0,
                              rims_path=tmp_path / "rims.json")         # a restart
    tub, _ = again.find(None, state, "plastic tub")
    assert tub.rim_measured and tub.rim_m == pytest.approx(0.082)
    assert any(t.name == "plastic_tub" and "walls 82 mm" in t.description for t in state.targets)


def test_a_container_kept_from_an_earlier_session_is_found_at_the_first_look(tmp_path, monkeypatch):
    model = down_camera([0.36, 0.04, 0.52], -8.0)
    mask = rim_mask(model, 0.09)
    asked = []

    def found(frame, camera, name, what, **_):
        asked.append(what)
        return containers.Container(name, what, camera, (640, 480), containers.outline_px(mask), 0.15), ""
    monkeypatch.setattr(containers, "find", found)
    look = lambda env: (np.zeros((480, 640, 3), np.uint8), model)
    state = SceneState(objects=[], targets=[])
    first = containers.Keeper("overhead", look, grounding_model="g", judge_model=None, tallest_m=0.15, table_z=0.0,
                              rims_path=tmp_path / "rims.json")
    first.find(None, state, "the plastic tub")
    first.measured(state, "tub", 0.074)
    again = containers.Keeper("overhead", look, grounding_model="g", judge_model=None, tallest_m=0.15, table_z=0.0,
                              rims_path=tmp_path / "rims.json")         # a restart, nothing asked for yet
    assert again.remembered() == {"plastic_tub": "the plastic tub"}      # as it was asked for
    said = again.refresh(None, state)                                  # the scan's look
    tub = again.known("plastic_tub")
    assert tub is not None and tub.rim_measured and tub.rim_m == pytest.approx(0.074)
    assert asked[-1] == "the plastic tub" and "74 mm" in said[0][1] and again.remembered() == {}
    assert any(t.name == "plastic_tub" for t in state.targets)



def test_where_a_kept_container_stands_is_on_disk_for_a_tool_without_a_camera(tmp_path, monkeypatch):
    # 2026-11-19: tools/so101/rest.py folded the arm into the tub it knew nothing of.
    import json

    model = down_camera([0.36, 0.04, 0.52], -8.0)
    mask = rim_mask(model, 0.09)
    monkeypatch.setattr(containers, "find", lambda frame, camera, name, what, **_: (
        containers.Container(name, what, camera, (640, 480), containers.outline_px(mask), 0.15), ""))
    look = lambda env: (np.zeros((480, 640, 3), np.uint8), model)
    state = SceneState(objects=[], targets=[])
    keeper = containers.Keeper("overhead", look, grounding_model="g", judge_model=None, tallest_m=0.15, table_z=0.0,
                               rims_path=tmp_path / "rims.json")
    keeper.find(None, state, "the plastic tub")
    keeper.measured(state, "tub", 0.074)
    kept = json.loads((tmp_path / "rims.json").read_text())["plastic_tub"]
    xy = np.asarray(kept["outline_xy"])
    assert kept["rim_m"] == pytest.approx(0.074) and "seen" in kept and len(xy) >= 3
    assert np.allclose(xy.mean(axis=0), keeper.known("plastic_tub").outline_xy(model).mean(axis=0), atol=0.01)
    keeper.refresh(None, state)                                        # the next look keeps both
    again = json.loads((tmp_path / "rims.json").read_text())["plastic_tub"]
    assert again["rim_m"] == pytest.approx(0.074) and len(again["outline_xy"]) >= 3


def test_a_container_kept_from_before_and_not_on_the_table_is_not_kept(tmp_path, monkeypatch):
    (tmp_path / "rims.json").write_text('{"plastic_tub": {"rim_m": 0.074, "measured": "2026-11-18 16:07"}}')
    monkeypatch.setattr(containers, "find", lambda *a, **k: (None, "nothing like it there"))
    model = down_camera([0.36, 0.04, 0.52], -8.0)
    keeper = containers.Keeper("overhead", lambda env: (np.zeros((480, 640, 3), np.uint8), model),
                               grounding_model="g", judge_model=None, tallest_m=0.15, table_z=0.0,
                               rims_path=tmp_path / "rims.json")
    said = keeper.refresh(None, SceneState(objects=[], targets=[]))
    assert keeper.known("plastic_tub") is None and "not seen now" in said[0][1]
    assert keeper.remembered() == {"plastic_tub": "plastic tub"}      # looked for again next time


# ── The wrist camera at the grasp: the object against the gap between the jaws ─

def _disc(view, xy, h_m, r_mm, size=(480, 640)):
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.c_[xy[0] * 1000 + r_mm * np.cos(t), xy[1] * 1000 + r_mm * np.sin(t), np.full(24, h_m * 1000)]
    out = np.zeros(size, np.uint8)
    cv2.fillPoly(out, [view.project(ring).astype(np.int32)], 1)
    return out.astype(bool)


def test_the_jaws_are_reseated_onto_what_the_wrist_camera_sees_between_them(monkeypatch):
    from types import SimpleNamespace

    from mrcf.policies.parallel_jaw.pick import PickContext
    from mrcf.policies.base import Target, Vision

    grasp = np.array([0.25, 0.05, 0.016])
    model = down_camera([0.25, 0.05, 0.20])
    view = PlaneView.from_camera_model(model)
    jaws = _disc(view, (0.25, 0.069), 0.016, 5.0) | _disc(view, (0.25, 0.031), 0.016, 5.0)   # either side of the gap
    obj = _disc(view, (0.256, 0.05), 0.016, 9.0)                                            # 6 mm off in x
    arm = FakeArm()
    arm.ee = grasp.copy()
    moves = []
    arm.move_ee_to = lambda goal, **k: moves.append(np.asarray(goal, float)) or setattr(arm, "ee", np.asarray(goal, float)) or True
    arm.camera_model = lambda cam, w, h: model
    arm.render_image = lambda cam, width=None, height=None: np.zeros((480, 640, 3), np.uint8)
    spec = dataclasses.replace(SPEC, camera="wrist", render_w=640, render_h=480, wrist_at_grasp=True, grasp_reseats=1)
    pol = ParallelJaw(spec)
    monkeypatch.setattr(pol, "jaw_view", lambda: SimpleNamespace(at=lambda gap, shape: jaws))
    seen = iter([obj, _disc(view, (0.2501, 0.05), 0.016, 9.0)])     # then, reseated, between them
    between = lambda frame, jm, gap: next(seen)                      # noqa: E731
    target = Target(np.array([0.25, 0.05, 0.0]), foot_m=0.02)
    ctx = PickContext(target=target, obstacles=[], vision=Vision(between=between), room=0.045, plan=pol.plan(target),
                      object_xy=np.array([0.25, 0.05]), servo_ready=False)
    ctx.gp, ctx.grasp, ctx.target_xy, ctx.grasp_tol, ctx.grasp_step = pol.plan(target), grasp.copy(), grasp[:2].copy(), 0.003, 3.0
    ctx.gp = dataclasses.replace(ctx.gp, grasp_height_m=0.016)
    pol._begin_log()
    pol._attempts.append({"attempt": 1})
    pol._wrist_at_grasp(arm, ctx)
    assert ctx.grasp[0] == pytest.approx(0.256, abs=0.0015) and ctx.grasp[1] == pytest.approx(0.05, abs=0.0015)
    assert any(abs(m[0] - 0.256) < 0.002 for m in moves)                     # over onto it
    assert len(pol._attempts[0]["wrist_at_grasp_mm"]) == 2                    # looked at again: between them
    assert any("between the jaws" in n for n in pol._notes)



def test_the_gap_is_halfway_between_the_jaws_inner_edges():
    m = np.zeros((100, 200), bool)
    m[60:90, 20:70] = True                     # left jaw, inner edge x=69
    m[60:90, 130:180] = True                   # right jaw, inner edge x=130
    gap = ParallelJaw(SPEC).gap_px(m)
    assert gap[0] == pytest.approx(99.5, abs=1.0) and 60 <= gap[1] <= 90
    assert ParallelJaw(SPEC).gap_px(np.zeros((100, 200), bool)) is None


def test_what_lies_between_the_jaws_is_outlined_at_their_gap_not_the_jaws_nor_the_table(monkeypatch):
    from mrcf.perception import segment
    from mrcf.policies.base import Vision

    pol = ParallelJaw(SPEC)
    frame = np.zeros((480, 640, 3), np.uint8)
    jm = np.zeros((480, 640), bool)
    jm[300:480, 150:280] = True
    jm[300:480, 400:530] = True
    gap = np.array([340.0, 320.0])
    obj = np.zeros((480, 640), bool)
    obj[290:360, 300:380] = True
    table = np.ones((480, 640), bool)
    jaw = jm.copy()
    monkeypatch.setattr(segment, "available", lambda: True)
    monkeypatch.setattr(segment, "segment", lambda f, pts: [([table, jaw, obj], [0.9, 0.95, 0.8]) for _ in pts])
    got = pol._between_jaws(frame, jm, gap, Vision())
    assert got is not None and got[320, 340] and not got[400, 200] and got.mean() < 0.05


def test_a_wall_grip_that_held_is_used_and_a_close_on_nothing_is_tried_lower(monkeypatch):
    # 2026-11-18, a scoop: held once 36 mm up its wall, its looks wandering 48-150 mm on the next picks.
    from mrcf.policies import rim_pick as rp

    spec = dataclasses.replace(SPEC, rim_pick=True, min_grasp_height_m=0.016, rim_grip_depth_m=0.012,
                               height_from_views=False)
    pol = ParallelJaw(spec)
    arm = FakeArm()
    square = np.array([[0.25, 0.0], [0.31, 0.0], [0.31, 0.06], [0.25, 0.06]])
    monkeypatch.setattr(rp, "rim_outline", lambda env, target, lo: square)
    closes = iter([(0.001, False), (0.003, True)])
    monkeypatch.setattr(arm, "squeeze_gripper", lambda **k: next(closes))
    downs = []
    monkeypatch.setattr(pol, "descend_onto", lambda env, goal, **k: downs.append(float(goal[2])) or True)
    monkeypatch.setattr(pol, "holding", lambda env: True)
    monkeypatch.setattr(pol, "_can_overhead", lambda env: False)
    target = Target(np.array([0.28, 0.03, 0.0]), foot_m=0.06, outlines={"overhead": {"size": [640, 480], "px": []}},
                    grip_at_m=0.036, max_height_m=0.05)
    out = pol.pick_by_rim(arm, target)
    assert out.ok, out.message
    assert [a["result"] for a in out.attempts] == ["closed_on_nothing", "ok"]
    assert out.attempts[0]["diagnosis"] == "too_high"
    assert downs[0] == pytest.approx(spec.table_z + 0.036) and downs[1] == pytest.approx(spec.table_z + 0.026)
    assert out.attempts[1]["grip_m"] == pytest.approx(0.026) and out.grasp.grasp_height_m == pytest.approx(0.026)


def _scoop(at=(0.25, 0.05), bowl_r=0.035, handle_w=0.020, handle_len=0.065, n=48):
    """A scoop's outline from above: a round bowl and a straight handle out along +x."""
    cx, cy = at
    pts = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        x, y = cx + bowl_r * np.cos(a), cy + bowl_r * np.sin(a)
        if x > cx and abs(y - cy) < handle_w / 2:
            continue                                         # where the handle leaves the bowl
        pts.append((x, y))
    join = cx + np.sqrt(bowl_r ** 2 - (handle_w / 2) ** 2)
    tip = cx + bowl_r + handle_len
    handle = [(join, cy - handle_w / 2), (tip, cy - handle_w / 2), (tip, cy + handle_w / 2), (join, cy + handle_w / 2)]
    ang = np.arctan2(np.array([p[1] for p in pts]) - cy, np.array([p[0] for p in pts]) - cx) % (2 * np.pi)
    pts = [p for _, p in sorted(zip(ang, pts))]
    lower = [p for p in pts if p[1] < cy]
    upper = [p for p in pts if p[1] >= cy]
    return np.array(upper + lower[:0] + [handle[3], handle[2], handle[1], handle[0]] + lower)


def test_a_handle_is_the_long_narrow_part_leading_from_the_widest():
    """User 2026-11-19: "with the scoop why not grasp a more secure thing ... like maybe grab the handle??"."""
    from mrcf.policies.handle_pick import find_handle

    scoop = _scoop()
    got = find_handle(scoop, 0.008, 0.040, 0.025, 0.010, 0.008)
    assert got is not None
    centre, tip, width = got
    assert width == pytest.approx(0.020, abs=0.003)
    assert tip[0] > 0.9                                           # toward the handle's tip, out along +x
    assert 0.25 + 0.035 < centre[0] < 0.25 + 0.035 + 0.030        # on the handle, near where it joins
    assert abs(centre[1] - 0.05) < 0.003
    square = np.array([[0.25, 0.0], [0.31, 0.0], [0.31, 0.06], [0.25, 0.06]])
    assert find_handle(square, 0.008, 0.040, 0.025, 0.010, 0.008) is None          # no handle: a wall instead
    stick = np.array([[0.2, 0.0], [0.3, 0.0], [0.3, 0.015], [0.2, 0.015]])
    assert find_handle(stick, 0.008, 0.040, 0.025, 0.010, 0.008) is None           # all handle: nothing to join


def test_by_its_handle_the_jaws_close_from_high_down_until_they_hold_it(monkeypatch):
    """Its height is not known: closed, opened and lower, until the jaws stop on it under load."""
    from mrcf.policies import rim_pick as rp

    spec = dataclasses.replace(SPEC, rim_pick=True, handle_first=True, min_grasp_height_m=0.016,
                               height_from_views=False)
    pol = ParallelJaw(spec)
    arm = FakeArm()
    monkeypatch.setattr(rp, "rim_outline", lambda env, target, h: _scoop())
    closes = iter([(0.0003, False), (0.0003, False), (0.019, True)])
    monkeypatch.setattr(arm, "squeeze_gripper", lambda **k: next(closes))
    downs = []
    monkeypatch.setattr(pol, "descend_onto", lambda env, goal, **k: downs.append(float(goal[2])) or True)
    monkeypatch.setattr(pol, "holding", lambda env: True)
    monkeypatch.setattr(pol, "_can_overhead", lambda env: False)
    target = Target(np.array([0.25, 0.05, 0.0]), foot_m=0.07, outlines={"overhead": {"size": [640, 480], "px": []}},
                    min_height_m=0.045, max_height_m=0.055)
    out = pol.pick_by_rim(arm, target)
    assert out.ok and "handle" in out.message, out.message
    assert [a["result"] for a in out.attempts] == ["closed_on_nothing", "closed_on_nothing", "ok"]
    assert all(a["approach"] == "handle" for a in out.attempts)
    assert np.diff(downs) == pytest.approx([-spec.handle_step_m] * 2)                # 8 mm lower each time
    assert out.grasp.grip_width_m == pytest.approx(0.020, abs=0.003)                 # not a 4 mm wall

    pol2 = ParallelJaw(dataclasses.replace(spec, handle_first=False))                # off: by a wall, as before
    monkeypatch.setattr(pol2, "descend_onto", lambda env, goal, **k: True)
    monkeypatch.setattr(pol2, "holding", lambda env: True)
    monkeypatch.setattr(pol2, "_can_overhead", lambda env: False)
    arm2 = FakeArm()
    monkeypatch.setattr(arm2, "squeeze_gripper", lambda **k: (0.001, True))
    out = pol2.pick_by_rim(arm2, target)
    assert "wall" in out.message, out.message



def test_by_its_handle_or_its_wall_too_the_way_from_the_tub_goes_up_over_its_walls(monkeypatch):
    """2026-11-19 15:06: from the tub to the scoop's handle, the arm came down across the tub's wall and pushed it."""
    from mrcf.policies import rim_pick as rp

    spec = dataclasses.replace(SPEC, rim_pick=True, handle_first=True, min_grasp_height_m=0.016,
                               height_from_views=False, transit_over_walls=True)
    pol = ParallelJaw(spec)
    pol.walls = lambda: [Opening(TUB, 0.09, True, "tub")]
    arm = FakeArm()
    arm.ee = np.array([0.30, -0.12, 0.05])                        # just let go of something inside, low
    monkeypatch.setattr(rp, "rim_outline", lambda env, target, h: _scoop(at=(0.25, 0.10)))
    monkeypatch.setattr(arm, "squeeze_gripper", lambda **k: (0.019, True))
    monkeypatch.setattr(pol, "descend_onto", lambda env, goal, **k: True)
    monkeypatch.setattr(pol, "holding", lambda env: True)
    monkeypatch.setattr(pol, "_can_overhead", lambda env: False)
    target = Target(np.array([0.25, 0.10, 0.0]), foot_m=0.07, outlines={"overhead": {"size": [640, 480], "px": []}},
                    min_height_m=0.045, max_height_m=0.055)
    start = len(arm.path)
    out = pol.pick_by_rim(arm, target)
    assert out.ok, out.message
    reach = spec.container_wall_clearance_m + spec.finger_thickness_m
    near = [p for p in arm.path[start:] if inside_by(TUB, p[:2]) > -reach]
    first_out = next(i for i, p in enumerate(near) if p[2] - spec.tip_below_grasp_m >= 0.09)
    assert all(p[2] - spec.tip_below_grasp_m >= 0.09 for p in near[first_out:])



def test_by_its_handle_where_it_held_before_the_first_close_bites_a_little_lower(monkeypatch):
    """User 2026-11-19: "the pick of the scoop is also too precarious - move the jaw down slightly"."""
    from mrcf.policies import rim_pick as rp

    spec = dataclasses.replace(SPEC, rim_pick=True, handle_first=True, min_grasp_height_m=0.016,
                               height_from_views=False)
    pol = ParallelJaw(spec)
    arm = FakeArm()
    monkeypatch.setattr(rp, "rim_outline", lambda env, target, h: _scoop())
    monkeypatch.setattr(arm, "squeeze_gripper", lambda **k: (0.019, True))
    downs = []
    monkeypatch.setattr(pol, "descend_onto", lambda env, goal, **k: downs.append(float(goal[2])) or True)
    monkeypatch.setattr(pol, "holding", lambda env: True)
    monkeypatch.setattr(pol, "_can_overhead", lambda env: False)
    target = Target(np.array([0.25, 0.05, 0.0]), foot_m=0.07, outlines={"overhead": {"size": [640, 480], "px": []}},
                    grip_at_m=0.034, max_height_m=0.055)
    out = pol.pick_by_rim(arm, target)
    assert out.ok and len(out.attempts) == 1, out.message                         # first try
    assert downs[0] == pytest.approx(spec.table_z + 0.034 - spec.handle_bite_m)   # a little under where it held
    assert out.attempts[0]["grip_m"] == pytest.approx(0.034)                      # learned where it held: no creep


def test_jaws_that_closed_on_after_the_lift_hold_nothing_whatever_the_camera_seems_to_see():
    """2026-11-20 09:25: the plug slipped out on the lift, the jaws went 42.8 -> 28.8 mm, and was "put into" the tub."""
    arm = FakeArm()
    arm.gripper_loaded = lambda: False
    width = {"now": 0.0428}
    arm.get_gripper_width = lambda: width["now"]
    for closed_m, expect in ((None, True), (0.010, False)):
        pol = ParallelJaw(dataclasses.replace(SPEC, held_by_camera=True, held_closed_m=closed_m))
        pol._grip_view = {"frame": None, "model": None, "gap": 0.0428, "said": False}
        pol._seen_holding = lambda env: True                     # 0.5% of the picture "moving with them"
        width["now"] = 0.0288
        assert pol.holding(arm) is expect
        width["now"] = 0.0410                                    # a squeeze's few mm on: still held
        pol._grip_view = {"frame": None, "model": None, "gap": 0.0428, "said": False}
        assert pol.holding(arm) is True
