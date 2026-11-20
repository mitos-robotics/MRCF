"""expected_view: objects another camera found, as a located camera should see them — and measuring one among them."""

import numpy as np

try:
    import cv2  # noqa: F401 — the masks are drawn with it
except ImportError:
    import pytest
    pytest.skip("needs OpenCV", allow_module_level=True)

from mrcf.perception import plane_geometry as pg
from mrcf.perception.expected_view import ExpectedView, KnownObject

SHAPE = (480, 640)


def _view(x=0.10, height=0.25, lean_deg=30.0):
    """A camera at (x, 0, height) looking down and forward along +x, leant lean_deg from straight down."""
    a = np.radians(lean_deg)
    z = np.array([np.sin(a), 0.0, -np.cos(a)])            # optical axis
    xc = np.array([0.0, -1.0, 0.0])
    yc = np.cross(z, xc)
    T = np.eye(4)
    T[:3, :3] = np.c_[xc, yc, z]
    T[:3, 3] = [x, 0.0, height]
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    return pg.PlaneView.from_camera_model({"K": K, "D": np.zeros(5), "T_world_cam": T, "plane_z_m": 0.0})


TARGET = KnownObject("target", np.array([0.22, 0.0]), 0.010, 0.045)
BESIDE = KnownObject("beside", np.array([0.22, 0.04]), 0.010, 0.045)


def test_what_stands_between_the_camera_and_the_target_hides_its_foot():
    tall_in_front = KnownObject("in front", np.array([0.18, 0.0]), 0.012, 0.12)
    ev = ExpectedView(_view(), [TARGET, tall_in_front, BESIDE], SHAPE)
    assert ev.in_front("target") == ["in front"]
    assert ev.foot_visible("target") < 0.5
    assert ev.foot_visible("beside") == 1.0 and ev.in_front("beside") == []
    # Asked where to look for the target: only where it, not what is in front, should show.
    for u, v in ev.prompts("target"):
        assert ev.visible("target")[int(v), int(u)]


def test_an_outline_belongs_to_whoever_owns_most_of_it():
    ev = ExpectedView(_view(), [TARGET, BESIDE], SHAPE)
    assert ev.foot_visible("target") == 1.0 and ev.visible_fraction("target") > 0.9
    assert ev.owner_of(ev.masks["beside"]) == ("beside", 1.0)
    assert ev.owner_of(np.zeros(SHAPE, bool)) == (None, 0.0)


def _segmenter(monkeypatch, masks):
    """segment.candidates offering these outlines, the biggest first, whatever the prompts."""
    from mrcf.perception import segment
    monkeypatch.setattr(segment, "available", lambda: True)
    monkeypatch.setattr(segment, "candidates", lambda frame, prompts: [(0.9, m, None) for m in masks])


def test_measuring_one_among_known_objects_takes_its_outline_not_a_bigger_neighbours(monkeypatch):
    view = _view()
    ev = ExpectedView(view, [TARGET, BESIDE], SHAPE)
    neighbour = ev.masks["beside"].copy()
    neighbour[:, :] |= np.roll(ev.masks["beside"], 3, axis=1)     # a little bigger, offered first
    _segmenter(monkeypatch, [neighbour, ev.masks["target"]])
    r = pg.measure_near(np.zeros((*SHAPE, 3), np.uint8), view, known=[TARGET, BESIDE], target="target",
                        need_height=False)
    assert r["ok"], r["why"]
    assert np.linalg.norm(np.asarray(r["measurement"].base_world_m) - TARGET.foot_xy_m) < 0.004
    assert r["hidden_by"] == [] and r["foot_visible"] == 1.0


def test_with_only_a_neighbours_outline_on_offer_it_measures_nothing(monkeypatch):
    view = _view()
    ev = ExpectedView(view, [TARGET, BESIDE], SHAPE)
    _segmenter(monkeypatch, [ev.masks["beside"]])
    r = pg.measure_near(np.zeros((*SHAPE, 3), np.uint8), view, known=[TARGET, BESIDE], target="target",
                        need_height=False)
    assert not r["ok"]


def test_an_outline_of_only_part_of_the_target_is_not_measured_as_it(monkeypatch):
    # The segmenter outlines a pawn's head on its own as readily as the whole
    # pawn; from the head alone its foot comes out well off and its height at
    # half. On its column and nearest its foot, it is still not the target's measure.
    view = _view()
    ev = ExpectedView(view, [TARGET, BESIDE], SHAPE)
    top = ev.masks["target"].copy()
    top[270:, :] = False                                         # the rows nearest its foot cut off
    _segmenter(monkeypatch, [top])
    r = pg.measure_near(np.zeros((*SHAPE, 3), np.uint8), view, known=[TARGET, BESIDE], target="target")
    assert not r["ok"] and "part of it" in r["why"]
    _segmenter(monkeypatch, [top, ev.masks["target"]])
    r = pg.measure_near(np.zeros((*SHAPE, 3), np.uint8), view, known=[TARGET, BESIDE], target="target",
                        need_height=False)
    assert r["ok"] and np.linalg.norm(np.asarray(r["measurement"].base_world_m) - TARGET.foot_xy_m) < 0.004


def test_an_outline_that_runs_off_the_frame_says_so(monkeypatch):
    # 2026-11-03: from beside the d1 queen a wrist camera outlined her only as
    # far as the frame's edge and read her 28 mm tall instead of 69.
    view = _view()
    ev = ExpectedView(view, [TARGET, BESIDE], SHAPE)
    whole = ev.masks["target"]
    _segmenter(monkeypatch, [whole])
    r = pg.measure_near(np.zeros((*SHAPE, 3), np.uint8), view, known=[TARGET, BESIDE], target="target",
                        need_height=False)
    assert r["ok"] and r["occluded"] is pg.cut_off(whole)
    edge = np.zeros(SHAPE, bool)
    edge[-1, 10:20] = True
    assert pg.cut_off(edge) and not pg.cut_off(np.pad(np.ones((4, 4), bool), 10))
