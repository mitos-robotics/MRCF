"""
touch_surface.py — how high the surface under a spot stands, found by touching
it: a two-finger gripper's jaws shut, lowered straight down in small steps
until they stop going down, and lifted back.

Every height the arm works by is measured from its work surface (spec.table_z,
an arm's TABLE_Z), and that is only the surface it was measured on: the
SO-101's was its chessboard's top, and with the board gone the table stood
20-30 mm lower — every grasp closed round the objects' heads (2026-11-17). So
the surface a task works on is touched, not assumed.

    shut     the jaws closed: their tips are the gripper's lowest point
    over     above the spot at start_z (the grasp point), one orientation held
             for the whole way down — as near straight down as reaches
    down     spec.touch_step_m at a time, slowly; the jaws have touched when two
             steps in a row brought them down less than half of what was asked
             (a single move of this arm can end a millimetre or two short in
             free air)
    up       straight back to start_z

Outcome.measured.height_mm: the surface's height where the tips touched, world z
in mm (the grasp point where it stopped, less spec.tip_below_grasp_m). Refused,
nothing moved, while something is held or no orientation reaches the whole way
down; failed, back up, when nothing stopped it above floor_z. Contact is told
by the arm stopping: a backend with no stall to show cannot use it, and an arm
that stops hard on contact (a protective stop) should not
(GripperSpec.touch_surface).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mrcf.capabilities import GripperTilt, OrientationSearch
from mrcf.policies.base import Measured, Outcome

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw

TAG = "[Touch]"


def touch_surface(policy: "ParallelJaw", env, xy, start_z: float, floor_z: float) -> Outcome:
    """Touch the surface under xy, from the grasp point at start_z down to floor_z at most: see the module docstring."""
    s = policy.spec
    if not s.touch_surface:
        return Outcome(False, "this arm does not touch the surface to measure it (GripperSpec.touch_surface)",
                       refused=True)
    if policy.holding(env):
        return Outcome(False, "the gripper holds something: its jaws cannot be shut to touch with", refused=True)
    x, y = float(xy[0]), float(xy[1])
    top, bottom = np.array([x, y, float(start_z)]), np.array([x, y, float(floor_z)])
    rot = None
    if isinstance(env, OrientationSearch):
        found = env.approach_orientation_for([top, bottom], prefer_tilt_deg=0.0)
        if found is None:
            return Outcome(False, f"no orientation reaches straight down at ({x:+.3f}, {y:+.3f}) to {floor_z * 1000:.0f} mm",
                           refused=True)
        rot = np.asarray(found[0], float)
    env.set_gripper(0.0, n_steps=15)
    env.move_ee_to(top, tol=s.travel_tol_m, orientation=rot)
    if not policy._near(env, top, 2 * s.travel_tol_m):
        env.move_ee_to(top, tol=s.travel_tol_m, orientation=rot)
        return Outcome(False, f"did not get over ({x:+.3f}, {y:+.3f})")
    heights = [float(env.get_ee_pos()[2])]
    z, touched = float(start_z), None
    while z > floor_z + 1e-9:
        z = max(z - s.touch_step_m, float(floor_z))
        env.move_ee_to(np.array([x, y, z]), tol=0.001, max_step_deg=1.0, orientation=rot)
        heights.append(float(env.get_ee_pos()[2]))
        if len(heights) >= 3 and heights[-3] - heights[-1] < s.touch_step_m:
            touched = heights[-1]                  # two steps asked for, less than one taken
            break
    env.move_ee_to(top, tol=s.travel_tol_m, orientation=rot)
    if touched is None:
        return Outcome(False, f"nothing stopped the jaws above {floor_z * 1000:.0f} mm at ({x:+.3f}, {y:+.3f})")
    tilt = env.gripper_tilt_deg(rot) if isinstance(env, GripperTilt) and rot is not None else 0.0
    surface = touched - s.tip_below_grasp_m * float(np.cos(np.radians(tilt)))
    policy._say(TAG, f"Touched at ({x:+.3f}, {y:+.3f}): the surface is at {surface * 1000:+.1f} mm "
                     f"(the jaws stopped {touched * 1000:.1f} mm up, held {tilt:+.0f} deg).")
    return Outcome(True, f"the surface is at {surface * 1000:+.1f} mm", at_xy=np.array([x, y]),
                   measured=Measured(height_mm=round(surface * 1000.0, 2), widths_mm={}, foot_mm=0.0,
                                     slant_deg=float(tilt)))
