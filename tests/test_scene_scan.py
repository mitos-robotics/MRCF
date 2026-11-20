"""scene_scan.observe: a surface scan into the memory and the world model (the scan itself faked)."""

from types import SimpleNamespace

import numpy as np
import pytest

# scene_scan imports the surface scan, which is OpenCV's: a module-level import
# error stops the whole run, which conftest.py's skip cannot catch.
pytest.importorskip("cv2", reason="needs OpenCV: pip install -e '.[measure]'")

from mrcf import scene_scan  # noqa: E402
from mrcf.object_memory import ObjectMemory  # noqa: E402
from mrcf.perception import plane_geometry as pg  # noqa: E402
from mrcf.vlm import SceneState  # noqa: E402

BOX_OFF_M = 0.011          # where a box's top face puts it, from the round foot fitted to its outline


def _found(xy, a):
    meas = SimpleNamespace(base_world_m=np.array(xy, float), base_radius_mm=17.0, height_mm=None, base_px=(0.0, 0.0))
    return {"measurement": meas, "appearance": {"L": 60.0, "a": a, "b": -20.0}, "mask": np.ones((4, 4), bool)}


def _observe(monkeypatch, memory, state, found):
    monkeypatch.setattr(scene_scan, "scan", lambda *a, **k: (np.zeros((4, 4, 3), np.uint8), found))
    env = SimpleNamespace(camera_model=lambda *a: {"model": True})
    return scene_scan.observe(env, state, memory, "overhead", 4, 4, (0, 1), (0, 1),
                              boxed=lambda track: (30.0, 30.0))


def test_a_block_hidden_under_another_stays_where_its_top_face_put_it(monkeypatch):
    """Rebuilt from the memory while covered, a base went back to its round-foot fit, 11 mm off its middle."""
    monkeypatch.setattr(pg.PlaneView, "from_camera_model", staticmethod(lambda model: object()))
    monkeypatch.setattr(pg, "side_direction_deg", lambda mask, view: 10.0)
    monkeypatch.setattr(pg, "box_centre_xy", lambda mask, view, top_mm, side_mm, faces: (
        np.asarray(found_at[0], float) + [BOX_OFF_M, 0.0]))
    memory, state = ObjectMemory(), SceneState()
    found_at = [(0.50, 0.30)]
    assert _observe(monkeypatch, memory, state, [_found((0.50, 0.30), 10.0), _found((0.35, 0.42), -13.0)])
    base, top = state.objects
    assert np.allclose(base.position[:2], (0.50 + BOX_OFF_M, 0.30))

    memory.picked(top.track_id)
    memory.placed(top.track_id, (0.50 + BOX_OFF_M, 0.30), on=base.track_id, raised_mm=30.0)
    found_at[0] = (0.50, 0.30)
    assert _observe(monkeypatch, memory, state, [_found((0.50, 0.30), -13.0)])    # the stack: its top seen
    hidden = next(o for o in state.objects if o.track_id == base.track_id)
    assert "under what stands on it" in hidden.description
    assert np.allclose(hidden.position[:2], (0.50 + BOX_OFF_M, 0.30))            # its top face's, not the foot's


def test_a_scan_keeps_a_frame_from_every_other_calibrated_view(monkeypatch):
    """For the check after a place from the side (placed_check.look_volume): the same moment's view, the arm then."""
    monkeypatch.setattr(scene_scan, "scan", lambda *a, **k: (np.zeros((4, 4, 3), np.uint8), [_found((0.5, 0.3), 5.0)]))
    monkeypatch.setattr(pg.PlaneView, "from_camera_model", staticmethod(lambda model: object()))
    monkeypatch.setattr(pg, "side_direction_deg", lambda mask, view: 10.0)
    asked = []

    def render(name, w, h):
        asked.append((name, w, h))
        if name == "broken":
            raise RuntimeError("no frame")
        return np.full((h, w, 3), 7, np.uint8)

    env = SimpleNamespace(camera_model=lambda *a: {"model": True}, render_image=render,
                          fixed_views=lambda: {"side": (8, 6), "broken": (8, 6)},
                          arm_shape=lambda: [(np.zeros(3), 0.05)])
    state = SceneState()
    assert scene_scan.observe(env, state, ObjectMemory(), "overhead", 4, 4, (0, 1), (0, 1))
    views = state.look_frame["views"]
    assert list(views) == ["side"] and views["side"].shape == (6, 8, 3) and ("side", 8, 6) in asked
    assert state.look_frame["arm"][0][1] == 0.05


def test_past_the_arm_what_is_on_it_is_left_out_and_what_it_hides_is_kept(monkeypatch):
    found = [_found((0.50, 0.30), 10.0), _found((0.35, 0.42), -13.0)]
    found[0]["mask"] = np.zeros((4, 4), bool)
    found[0]["mask"][:, :2] = True                 # this one is clear of it
    found[1]["mask"] = np.zeros((4, 4), bool)
    found[1]["mask"][:, 2:] = True                 # this one lies on the arm in the picture
    arm = np.zeros((4, 4), bool)
    arm[:, 2:] = True
    monkeypatch.setattr(scene_scan, "scan", lambda *a, **k: (np.zeros((4, 4, 3), np.uint8), found))
    monkeypatch.setattr(scene_scan, "arm_in_view", lambda env, cam, w, h: arm)
    monkeypatch.setattr(pg.PlaneView, "from_camera_model", staticmethod(lambda model: object()))
    monkeypatch.setattr(pg, "side_direction_deg", lambda mask, view: None)
    hides = {}
    monkeypatch.setattr(scene_scan, "hidden_by", lambda mask, view, z: hides.setdefault("f", lambda xy: xy[0] < 0.4))
    memory, state = ObjectMemory(), SceneState()
    env = SimpleNamespace(camera_model=lambda *a: {"model": True})
    memory.observe([scene_scan.Detection(position=(0.30, 0.10), foot_mm=30.0)])     # remembered under the arm
    assert scene_scan.observe(env, state, memory, "overhead", 4, 4, (0, 1), (0, 1), past_arm=True)
    xs = sorted(round(float(o.position[0]), 2) for o in state.objects)
    assert xs == [0.30, 0.50]                       # the outline on the arm left out; the hidden one kept
    assert any("hidden by the arm" in o.description for o in state.objects)
