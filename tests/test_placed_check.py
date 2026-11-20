"""
perception/placed_check.py: the quick look after a place, on frames drawn
through a calibrated camera (tests/test_camera_from_arm.py's): a block's top
in its colour where it was put, or not; and manipulation's use of it.
"""

import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, "tests")

from mrcf.perception import camera_from_arm as cfa  # noqa: E402
from mrcf.perception import placed_check  # noqa: E402

import test_camera_from_arm as synth  # noqa: E402

W, H = 1280, 720
MODEL = {"K": synth.K, "D": synth.D, "T_world_cam": synth.CAMERA}
TABLE_RGB = (190, 170, 150)
BLUE_RGB, GREEN_RGB = (60, 90, 200), (70, 160, 90)


def _ab(rgb):
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    return lab[1] - 128.0, lab[2] - 128.0


def _frame(blocks=()):
    """The table, and each (xy, top_z, rgb) block's top face drawn as it projects: a 30 mm square."""
    img = np.full((H, W, 3), TABLE_RGB, np.uint8)
    for (x, y), z, rgb in blocks:
        corners = [(x - 0.015, y - 0.015, z), (x + 0.015, y - 0.015, z), (x + 0.015, y + 0.015, z), (x - 0.015, y + 0.015, z)]
        px, _ = cfa.project(corners, synth.CAMERA, synth.K, synth.D)
        cv2.fillConvexPoly(img, np.round(px).astype(np.int32), rgb)
    return img


AT = (0.10, 0.30)


def test_a_block_put_down_is_seen_there_and_one_that_is_not_is_not():
    before = _frame()
    after = _frame([(AT, 0.03, BLUE_RGB)])
    assert placed_check.look(after, MODEL, AT, 0.03, 0.008, _ab(BLUE_RGB), before=before)[0] is True
    assert placed_check.look(before, MODEL, AT, 0.03, 0.008, _ab(BLUE_RGB), before=before)[0] is False   # unchanged
    gone = _frame([((0.25, 0.30), 0.03, BLUE_RGB)])                                                    # it slid off
    assert placed_check.look(gone, MODEL, AT, 0.03, 0.008, _ab(BLUE_RGB), before=before)[0] is False


def test_on_a_stack_it_is_told_from_the_block_under_it_unless_they_are_alike():
    before = _frame([(AT, 0.03, GREEN_RGB)])
    on_top = _frame([(AT, 0.03, GREEN_RGB), (AT, 0.06, BLUE_RGB)])
    fell = _frame([(AT, 0.03, GREEN_RGB), ((0.25, 0.30), 0.03, BLUE_RGB)])
    assert placed_check.look(on_top, MODEL, AT, 0.06, 0.008, _ab(BLUE_RGB), _ab(GREEN_RGB), before=before)[0] is True
    assert placed_check.look(fell, MODEL, AT, 0.06, 0.008, _ab(BLUE_RGB), _ab(GREEN_RGB), before=before)[0] is False
    same = _frame([(AT, 0.03, BLUE_RGB), (AT, 0.06, BLUE_RGB)])                                       # blue on blue
    got, why = placed_check.look(same, MODEL, AT, 0.06, 0.008, _ab(BLUE_RGB), _ab(BLUE_RGB),
                                 before=_frame([(AT, 0.03, BLUE_RGB)]))
    assert got is None and "cannot tell" in why


def test_where_the_arm_hides_it_it_cannot_say():
    covered = np.ones((H, W), bool)
    got, why = placed_check.look(_frame([(AT, 0.03, BLUE_RGB)]), MODEL, AT, 0.03, 0.008, _ab(BLUE_RGB),
                                 covered=covered, before=_frame())
    assert got is None and "arm" in why


def test_a_place_the_check_says_did_not_land_is_a_failed_place(capsys):
    from fake_arm import Cylinder, FakeArm
    from mrcf import manipulation, skill_executor
    from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw
    from mrcf.vlm import dispatch
    from test_manipulation import scene_with

    spec = GripperSpec(max_open_m=0.045, tip_below_grasp_m=0.007, free_depth_m=0.055,
                       finger_thickness_m=0.009, finger_half_width_m=0.008)
    said = {"verdict": False}
    manipulation.install(ParallelJaw(spec), scenario=manipulation.Scenario(
        max_height=lambda o: 0.05, confirm_placed=lambda env, state, obj, at, on: said["verdict"]))
    for verdict, ok in ((False, False), (None, True), (True, True)):
        said["verdict"] = verdict
        state = scene_with(("block", (0.20, 0.0), 0.021))
        arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04)])
        assert dispatch({"action": "pick", "object_name": "block"}, arm, state) is True
        skill_executor.clear_report()
        assert dispatch({"action": "place", "target_name": "spot"}, arm, state) is ok
        if not ok:
            assert "not at spot as the camera sees it" in capsys.readouterr().out
            assert skill_executor.last_report()["ok"] is False
    manipulation.install(ParallelJaw(spec), scenario=manipulation.Scenario(
        max_height=lambda o: 0.05, confirm_placed=lambda *a: 1 / 0))           # a check that breaks: the place stands
    state = scene_with(("block", (0.20, 0.0), 0.021))
    arm = FakeArm([Cylinder((0.20, 0.0), 0.021, 0.04)])
    dispatch({"action": "pick", "object_name": "block"}, arm, state)
    assert dispatch({"action": "place", "target_name": "spot"}, arm, state) is True


# ── From the side (look_volume) ───────────────────────────────────────────────

SIDE = synth._look_at(np.array([-0.30, 0.30, 0.20]), [0.10, 0.30, 0.04])     # low, along the table
GREY = (120, 120, 120)


def _side(blocks=(), wall=(60, 60, 60)):
    """From SIDE: a dark wall above the horizon, the table, and each (xy, bottom_z, top_z) block's faces, grey."""
    img = np.full((H, W, 3), wall, np.uint8)
    img[H // 2:] = TABLE_RGB
    for (x, y), z0, z1 in blocks:
        corners = [(x + dx, y + dy, z) for dx in (-0.015, 0.015) for dy in (-0.015, 0.015) for z in (z0, z1)]
        px, _ = cfa.project(corners, SIDE, synth.K, synth.D)
        cv2.fillConvexPoly(img, cv2.convexHull(np.round(px).astype(np.int32)), GREY)
    return img


SIDE_MODEL = {"K": synth.K, "D": synth.D, "T_world_cam": SIDE}


def test_from_the_side_the_top_block_of_a_stack_is_seen_there_or_not():
    base = [(AT, 0.0, 0.03)]
    before = _side(base)                                          # the stack's first block, at the last look
    stacked = _side(base + [(AT, 0.03, 0.06)])
    fell = _side(base + [((0.25, 0.10), 0.0, 0.03)])               # it slid off and lies beside
    got, why = placed_check.look_volume(stacked, SIDE_MODEL, AT, 0.03, 0.06, 0.015, before=before)
    assert got is True, why
    got, why = placed_check.look_volume(fell, SIDE_MODEL, AT, 0.03, 0.06, 0.015, before=before)
    assert got is False and "as it did before" in why
    assert placed_check.look_volume(stacked, SIDE_MODEL, AT, 0.03, 0.06, 0.015)[0] is None   # nothing to compare with


def test_from_the_side_the_arm_in_either_frame_is_left_out():
    base = [(AT, 0.0, 0.03)]
    covered = np.ones((H, W), bool)
    got, why = placed_check.look_volume(_side(base + [(AT, 0.03, 0.06)]), SIDE_MODEL, AT, 0.03, 0.06, 0.015,
                                        before=_side(base), covered_before=covered)
    assert got is None and "arm" in why


def test_the_check_after_a_place_asks_every_view_and_a_no_from_any_is_a_no(monkeypatch, capsys):
    import importlib
    from types import SimpleNamespace

    importlib.import_module("deployments.tabletop.skills")      # the table task, bound to the UR5's letter blocks
    from mrcf.tabletop import skills

    monkeypatch.setattr(skills, "PLACED_SETTLE_S", 0.0)
    monkeypatch.setattr(skills, "boxed", lambda track: (30.0, 30.0))
    track = SimpleNamespace(id="t1", appearance={"a": -10.0, "b": -30.0}, height_mm=30.0, foot_mm=30.0)
    memory = SimpleNamespace(get=lambda tid: track if tid == "t1" else None)
    frame = np.zeros((6, 8, 3), np.uint8)
    state = SimpleNamespace(memory=memory, look_frame={"camera": "overhead", "frame": frame, "positions": {},
                                                       "views": {"side": frame}, "arm": None})
    env = SimpleNamespace(list_cameras=lambda: ["overhead", "side"], camera_model=lambda *a: {"K": 1},
                          render_image=lambda name, w, h: frame, arm_shape=lambda: None)
    obj = SimpleNamespace(name="blue_block_d", track_id="t1")
    for top, side, want in ((True, False, False), (None, True, True), (None, None, None), (True, None, True),
                            (False, True, False)):
        monkeypatch.setattr(placed_check, "look", lambda *a, top=top: (top, f"top {top}"))
        monkeypatch.setattr(placed_check, "look_volume", lambda *a, side=side: (side, f"side {side}"))
        assert skills.confirm_placed(env, state, obj, (0.5, 0.3, 0.06), None) is want, (top, side)
    out = capsys.readouterr().out
    assert "[Check] blue_block_d: overhead: top False; side: side True." in out
