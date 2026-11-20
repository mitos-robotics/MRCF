"""
fake_arm.py — a simulated arm with a two-finger gripper, for testing policies
and the layers above them without hardware.

Not a physics simulation. The grasp point moves in straight steps of STEP_M per
control cycle, calling guard() before each one as a real backend must; objects
are upright cylinders on the table; the jaws hold one if they close around its
body (centred within CENTRE_TOL_M, below its top) and let it go where they
open. Enough to tell a pick that grasps from one that misses, to see where a
place puts things, and to stop mid-move.
"""

from __future__ import annotations

import time

import numpy as np

from mrcf.robot_env import RobotEnv

STEP_M = 0.01
CENTRE_TOL_M = 0.004
HOME = np.array([0.20, 0.0, 0.15])


class Cylinder:
    def __init__(self, xy, diameter_m: float, height_m: float):
        self.xy = np.asarray(xy, float)
        self.diameter_m = diameter_m
        self.height_m = height_m


class FakeArm(RobotEnv):
    def __init__(self, objects=(), cycle_s: float = 0.0):
        self.cycle_s = cycle_s   # real time per control cycle, for tests with threads
        self.ee = HOME.copy()
        self.width = 0.0
        self.objects = list(objects)
        self.held: "Cylinder | None" = None
        self.cycles = 0          # control cycles run
        self.holds = 0           # times hold() was called
        self.path: list = []     # every grasp-point position, in order

    # ── RobotEnv ────────────────────────────────────────────────────────────────

    def step(self, n: int = 1) -> None:
        self.cycles += n
        if self.cycle_s:
            time.sleep(n * self.cycle_s)

    def hold(self) -> None:
        self.holds += 1

    def get_ee_pos(self) -> np.ndarray:
        return self.ee.copy()

    def get_joint_angles(self) -> dict:
        return {}

    def get_body_pos(self, name):
        return None

    def get_movable_body_names(self) -> list:
        return []

    def get_static_body_names(self, exclude_keywords=None) -> list:
        return []

    def render_image(self, camera, width=640, height=480) -> np.ndarray:
        return np.zeros((height, width, 3), np.uint8)

    def pixel_to_world(self, u, v, z_world, camera, render_w=640, render_h=480) -> np.ndarray:
        return np.array([0.0, 0.0, z_world])

    def go_home(self) -> bool:
        return self.move_ee_to(HOME)

    def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None,
                   orientation=None) -> bool:
        target = np.asarray(target, float)
        for _ in range(max_iters):
            gap = target - self.ee
            dist = float(np.linalg.norm(gap))
            if dist < 1e-9:
                break
            self.guard()
            self.ee = self.ee + gap * min(1.0, STEP_M / dist)
            if self.held is not None:
                self.held.xy = self.ee[:2].copy()
            self.path.append(self.ee.copy())
            self.step(1)
        return bool(np.linalg.norm(target - self.ee) < tol)

    # ── Gripper ─────────────────────────────────────────────────────────────────

    def _between_jaws(self) -> "Cylinder | None":
        for o in self.objects:
            if np.linalg.norm(o.xy - self.ee[:2]) < CENTRE_TOL_M and self.ee[2] < o.height_m:
                return o
        return None

    def set_gripper(self, width: float, n_steps: int = 15, max_step_pct=None, **_) -> None:
        self.guard()
        if width > self.width and self.held is not None:
            self.held = None                   # let go: it stays where it was put down
        inside = self._between_jaws() if width < self.width else None
        self.width = max(width, inside.diameter_m) if inside is not None else width
        self.step(1)

    def squeeze_gripper(self) -> "tuple[float, bool]":
        self.guard()
        inside = self._between_jaws()
        if inside is not None and self.width <= inside.diameter_m + 1e-9:
            self.held = inside
            return 0.003, True                 # a held piece reads ~3 mm on the SO-101
        self.width = 0.0
        return 0.0, False

    def get_gripper_width(self) -> float:
        return 0.003 if self.held is not None else self.width

    def gripper_loaded(self) -> bool:
        return self.held is not None
