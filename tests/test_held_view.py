"""
Whether the gripper carries something, from two wrist-camera pictures as the
arm moved (perception/held_view.py), and holding() asking it when the
gripper reads empty (ParallelJaw._seen_holding).
"""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("cv2")

import cv2  # noqa: E402

from mrcf.perception.held_view import HeldView, held_view  # noqa: E402
from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw  # noqa: E402

K = np.array([[460.0, 0, 320], [0, 460.0, 240], [0, 0, 1]])


def camera(z: float) -> dict:
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    T[:3, 3] = (0.25, 0.0, z)
    return {"K": K, "D": np.zeros(5), "T_world_cam": T, "plane_z_m": 0.0}


def table(model: dict) -> np.ndarray:
    """A table with a grain of marks, seen from model."""
    from mrcf.perception.plane_geometry import PlaneView

    view = PlaneView.from_camera_model(model)
    img = np.full((480, 640, 3), 200, np.uint8)
    rng = np.random.default_rng(1)
    for xy in rng.uniform([0.1, -0.2], [0.4, 0.2], (900, 2)):
        p = view.project(np.array([[xy[0] * 1000, xy[1] * 1000, 0.0]]))[0]
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (90, 90, 90), -1)
    return img


def held_thing(img: np.ndarray) -> np.ndarray:
    """A dark, edged thing between the jaws, at the same place in every picture: carried."""
    out = img.copy()
    cv2.rectangle(out, (260, 330), (380, 430), (40, 30, 120), -1)
    cv2.line(out, (260, 380), (380, 380), (220, 220, 220), 3)
    return out


JAWS = np.zeros((480, 640), bool)
JAWS[400:, 180:250] = True
JAWS[400:, 390:460] = True


def test_what_stays_put_while_the_table_moves_is_held():
    low, high = camera(0.12), camera(0.17)
    got = held_view(held_thing(table(low)), low, held_thing(table(high)), high, JAWS)
    assert got.held is True and got.carried > 0.01


def test_with_nothing_between_the_jaws_it_is_not_seen_held():
    low, high = camera(0.12), camera(0.17)
    got = held_view(table(low), low, table(high), high, JAWS)
    assert got.held is not True


def test_without_the_jaws_learned_it_cannot_tell():
    low, high = camera(0.12), camera(0.17)
    assert held_view(table(low), low, table(high), high, None).held is None


class _ThinWallArm:
    """Reads empty whatever it holds (a 1 mm wall squeezed thinner); has a wrist camera."""

    def __init__(self):
        self.gap = 0.0005

    def gripper_loaded(self):
        return False

    def get_gripper_width(self):
        return self.gap

    def render_image(self, camera, width=640, height=480):
        return np.zeros((height, width, 3), np.uint8)

    def camera_model(self, camera, w, h):
        return camera_model_stub


camera_model_stub = camera(0.12)


class _Jaws:
    def at(self, gap, shape):
        return np.zeros(shape, bool)


def test_holding_asks_the_wrist_camera_when_the_gripper_reads_empty(monkeypatch):
    import mrcf.perception.held_view as hv

    spec = GripperSpec(max_open_m=0.08, tip_below_grasp_m=0.007, free_depth_m=0.055, finger_thickness_m=0.009,
                       finger_half_width_m=0.008, camera="wrist", render_w=640, render_h=480)
    arm = _ThinWallArm()
    for on, sees, want in ((False, True, False), (True, True, True), (True, False, False)):
        pol = ParallelJaw(dataclasses.replace(spec, held_by_camera=on))
        pol.jaw_view = lambda: _Jaws()
        monkeypatch.setattr(hv, "held_view", lambda *a, **k: HeldView(sees, 0.02 if sees else 0.0, 0.0))
        pol._note_grip(arm)
        assert pol.holding(arm) is want, (on, sees)
    arm.gap = 0.03                                   # the jaws opened: whatever it held is let go
    assert pol.holding(arm) is False and pol._grip_view is None
