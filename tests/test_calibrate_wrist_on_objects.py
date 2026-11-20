"""
tools/calibrate_wrist_on_objects.py offline: which objects make the ruler, the
shift from the feet both cameras found, the config line it writes, and a fit
again from a saved capture — nothing connected, nothing moved.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import calibrate_wrist_on_objects as cw

BOUNDS = SimpleNamespace(x_min=0.10, x_max=0.40, y_min=-0.20, y_max=0.25)


def test_the_ruler_is_the_small_objects_standing_apart_within_reach():
    feet = [((0.33, 0.07), 22.0),                     # a figure, on its own
            ((0.22, 0.14), 41.0), ((0.227, 0.171), 14.0),   # a plug and its pin, 3 cm apart
            ((0.30, 0.15), 105.0),                    # a scoop: too wide
            ((0.18, 0.08), 21.0),                     # the other figure
            ((0.50, 0.00), 22.0)]                     # out of reach
    got = cw.ruler(feet, BOUNDS)
    assert [round(float(xy[0]), 2) for _, xy, _ in got] == [0.33, 0.18]
    assert [k for k, _, _ in cw.ruler(feet, BOUNDS, containers=[lambda xy: xy[0] > 0.3])] == [got[1][0]]


def test_the_shift_is_where_the_wrist_camera_puts_the_feet_less_the_scan():
    rng = np.random.default_rng(0)
    true = np.array([0.004, -0.009])
    pairs = []
    for k in range(20):
        scan = np.array([0.2 + 0.01 * k, 0.05])
        wrist = scan + true + rng.normal(0, 0.001, 2)
        pairs.append({"wrist": wrist.tolist(), "scan": scan.tolist(), "object": f"obj{k % 2}",
                      "height_m": 0.10 if k % 3 else 0.14, "lean": "+0/+0"})
    got = cw.fit_shift(pairs)
    assert np.allclose(got["shift_m"], true, atol=0.001)
    assert got["spread_mm"] < 2.0 and got["feet"] == 20 and set(got["by_object_mm"]) == {"obj0", "obj1"}


def test_it_writes_the_shift_into_the_arms_config_line_as_it_is(tmp_path):
    cfg = tmp_path / "robot_config.py"
    cfg.write_text('X = 1\nGRIPPER_CAM_SCAN_SHIFT_M: "tuple[float, float, float] | None" = (-0.0065, -0.0032, -0.0025)\n'
                   'GRIPPER_CAM_SCAN_SHIFT_AT: "str | None" = "2026-11-03T17:10:13"\n')
    cw.apply(cfg, [0.0041, -0.0089, -0.0025], "2026-11-16T10:27:02")
    text = cfg.read_text()
    assert '= (0.0041, -0.0089, -0.0025)\n' in text and "'2026-11-16T10:27:02'" in text and text.startswith("X = 1\n")


def test_a_saved_capture_is_fitted_again_without_the_arm(tmp_path, monkeypatch):
    from mrcf.perception import plane_geometry

    true = np.array([0.006, -0.004])
    views = []
    for i in range(8):
        views.append({"frame": f"view{i:03d}.png", "object": "obj0" if i < 4 else "obj1", "height_m": 0.10,
                      "lean": "+0/+0", "model": {"K": np.eye(3).tolist(), "D": [0.0] * 5, "T_world_cam": np.eye(4).tolist(),
                                                 "plane_z_m": 0.0007, "scale": 1.0, "scan_to_camera": [0.0, 0.0, 0.0]}})
    import cv2
    for v in views:
        cv2.imwrite(str(tmp_path / v["frame"]), np.zeros((48, 64, 3), np.uint8))
    feet = {"obj0": {"xy": [0.33, 0.07], "d_mm": 22.0}, "obj1": {"xy": [0.18, 0.08], "d_mm": 21.0}}
    (tmp_path / "views.json").write_text(json.dumps({"ruler": feet, "all_feet": [f["xy"] for f in feet.values()],
                                                     "views": views}))

    def found(frame, expected_px=None, model=None, expected_xy_m=None, **kw):
        at = np.asarray(feet[kw["target"]]["xy"]) + true
        return {"ok": True, "measurement": SimpleNamespace(base_world_m=at)}
    monkeypatch.setattr(plane_geometry, "measure_in_frame", found)
    got = cw.report(tmp_path, (-0.0065, -0.0032, -0.0025))
    assert np.allclose(got["shift_m"][:2], true, atol=1e-6) and got["shift_m"][2] == pytest.approx(-0.0025)
    assert json.loads((tmp_path / "result.json").read_text())["feet"] == 8


def test_an_object_whose_feet_sit_apart_from_the_others_is_left_out():
    true, pairs = np.array([-0.006, -0.008]), []
    for k, obj in enumerate(["a"] * 5 + ["b"] * 5 + ["in_tub"] * 5):
        scan = np.array([0.2, 0.01 * k])
        off = true + (np.array([0.0, -0.010]) if obj == "in_tub" else 0.0)
        pairs.append({"wrist": (scan + off).tolist(), "scan": scan.tolist(), "object": obj, "height_m": 0.1, "lean": "0"})
    got = cw.fit_shift(pairs)
    assert got["left_out"] == ["in_tub"] and np.allclose(got["shift_m"], true, atol=1e-6) and got["feet"] == 10
