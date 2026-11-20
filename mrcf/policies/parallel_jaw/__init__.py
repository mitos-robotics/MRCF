"""
mrcf/policies/parallel_jaw/ — pick and place for an arm with a two-finger gripper,
from above.

Works on any RobotEnv with move_ee_to, set_gripper, get_gripper_width and
squeeze_gripper; uses more where the backend has it (capabilities.py — SO101Env
has them all):

  OrientationSearch   hold one wrist orientation through a sequence
  GripSensing         sense a grip from the servo's load
  HeldWidth           tell the grasp-point model how wide the held thing is
  OverheadSighting    a fixed camera that sees the gripper corrects the arm
  WristServo          centre the jaws over the object from the wrist camera
  camera_model        a located wrist camera: measure the object at the hover
                      and aim at its foot

Everything about the gripper — its geometry, how fast to close, how to tell a
grip from a miss — comes from a GripperSpec, which the arm's backend fills in
(mrcf/backends/so101/__init__.py, make_policies). Nothing here knows what the
objects are or what the task is.

How a pick goes
---------------
  first      Refused, nothing moved, while a fact of the arm's the jaws are
             turned by is not checked on the rig (mrcf/arm_facts.py). A thing
             lying along the table, its height not known, goes to the lying
             pick instead (spec.lying_pick, mrcf/policies/lying_pick.py).
  height     Not measured, its kind's height not known, and outlined from above
             by the scan (spec.height_from_views, Target.outlines):
             the wrist camera is sent where the object's height shows, beside it
             and leaning in, and its shape fitted to its outlines in every view,
             the scan's from above among them (height_looks.py,
             mrcf/perception/shape_fit.py). Refused, with why, while it cannot
             tell — never planned on the tallest it could be.  (survey_height)
  size       From the Target: a measured object's height and widths, or, before
             it has been measured, the tallest it could be. Jaw opening checked
             against the obstacles' feet (and the arm's own base); refused,
             without moving, if a finger would land on one.
  approach   How to come at it, by where it stands from the arm's base
             (mrcf/policies/approaches.py): top_down with the wrist servo wherever
             it reaches short of far out, near_base close to the base where it
             does not, reach_out far out. Each is checked against the arm's
             reach before anything moves; refused if none can be done. A retry
             tries the next one along.
  hover      Above it, high enough that the open jaw tips clear its top; then
             over it as the fixed camera sees it.                    (_hover)
  align      Where the jaws close: the approach's aligners, in order
             (mrcf/policies/aligners.py) — the foot a located wrist camera finds,
             the fixed camera's sighting, the wrist servo.           (_align)
  look       The wrist camera measures it (two frames that must agree), and the
             grasp is re-sized from that.                            (_measure)
  descend    Slowly, onto the grasp height; with spec.approach_look, a stop just
             above its top for one more look, and the grasp planned again by it
             (a height looked for, or one remembered).   (_descend, _approach_look)
  close      Gently; the servo's load says whether anything is held. (_close)
  lift       Straight up from where the jaws are; still holding?    (_lift)
Up to spec.retries attempts; a phase that cannot go on ends its attempt with
aligners.Retry, under a short code kept in Outcome.attempts.

Adding a way to pick: a new Approach (approaches.py) in GripperSpec.approaches,
and, if it lines the jaws up some new way, an Aligner (aligners.py).

Where it lives
--------------
ParallelJaw (below) is assembled from parts, one file each, each building on
the one before:

  spec.py       GripperSpec (the gripper) and GraspPlan (one grasp)
  planning.py   the grasp from the object's shape; the policy's state
  reach.py      room, waypoints, the held orientation, the approach, small moves
  sighting.py   the wrist camera's measurement, the wrist servo, the fixed camera
  height_looks.py  looks for how tall the object is: where from, the outline in
                each, the fit, and what the gripper keeps clear of meanwhile
  pick.py       the pick's phases (above) and its retries
  place.py      the place
"""

from __future__ import annotations

import contextlib
import dataclasses

import numpy as np

from mrcf.policies.approaches import NEAR_BASE, REACH_OUT, TOP_DOWN, Approach, ApproachFit
from mrcf.policies.base import Obstacle, Opening, Outcome, Target, Vision
from mrcf.policies.parallel_jaw.pick import PickContext, PickMixin
from mrcf.policies.parallel_jaw.place import PlaceMixin
from mrcf.policies.parallel_jaw.spec import GraspPlan, GripperSpec

__all__ = ["GripperSpec", "GraspPlan", "ParallelJaw", "PickContext", "Approach", "ApproachFit", "TOP_DOWN",
           "NEAR_BASE", "REACH_OUT", "Obstacle", "Target", "Vision", "Outcome"]


class ParallelJaw(PlaceMixin, PickMixin):
    """Top-down pick and place with a two-finger gripper. See the package docstring."""

    # ── Policies ────────────────────────────────────────────────────────────────

    def pick(self, env, target: Target, obstacles=(), vision: "Vision | None" = None) -> Outcome:
        with self._on_surface(target.surface_z_m):
            return super().pick(env, target, obstacles, vision)

    def place(self, env, at, held: "Target | None" = None, grasp=None, obstacles=(),
              tilt_deg: "float | None" = None, surface_z: "float | None" = None,
              faces_deg: "float | None" = None) -> Outcome:
        with self._on_surface(surface_z):
            return super().place(env, at, held, grasp, obstacles, tilt_deg, surface_z, faces_deg)

    @contextlib.contextmanager
    def _on_surface(self, z: "float | None"):
        """
        Every height of a pick or a place is above the surface the object
        stands on (spec.table_z). On something raised — a block on a block —
        that surface is the top of what is under it, and for the call the spec
        says so. The neighbours on the table then read as taller than they are
        by as much, which keeps the jaws further from them, never nearer.
        """
        spec = self.spec
        if z is not None and float(z) > spec.table_z + 1e-4:
            self.spec = dataclasses.replace(spec, table_z=float(z))
        try:
            yield
        finally:
            self.spec = spec

    def place_in(self, env, opening: Opening, held: "Target | None" = None, grasp=None,
                 obstacles=(), drop: bool = False) -> Outcome:
        """Put what is held into a container (opening): place_in.py."""
        from mrcf.policies.place_in import place_in
        return place_in(self, env, opening, held, grasp, obstacles, drop)

    @staticmethod
    def descend_onto(env, target, **move_kw) -> bool:
        """A final descent under the arm's approach limit (safety.descend_onto), for the policies beside this one."""
        from mrcf import safety
        return safety.descend_onto(env, target, **move_kw)

    def pick_by_rim(self, env, target: Target, obstacles=(), vision: "Vision | None" = None) -> Outcome:
        """Pick target up by one of its walls (a cup, a bowl, a scoop): rim_pick.py."""
        from mrcf.policies.rim_pick import rim_pick
        unchecked = self.unchecked_refusal(env)
        if unchecked is not None:
            return unchecked
        return rim_pick(self, env, target, self._with_base(env, obstacles))

    def touch_surface(self, env, xy, start_z: float, floor_z: float) -> Outcome:
        """The surface's height under xy, found by lowering the shut jaws onto it: touch_surface.py."""
        from mrcf.policies.touch_surface import touch_surface
        return touch_surface(self, env, xy, start_z, floor_z)

    def learn_jaw_view(self, env, spots, height_m: "float | None" = None) -> Outcome:
        """The wrist camera learns where the jaws are in its picture, over spots [(x, y)]: learn_jaw_view.py."""
        from mrcf.policies.learn_jaw_view import learn_jaw_view
        return learn_jaw_view(self, env, spots, height_m)

    def open_gripper(self, env) -> Outcome:
        s = self.spec
        env.set_gripper(s.max_open_m, n_steps=15)
        width = env.get_gripper_width()
        if width >= 0.8 * s.max_open_m:
            return Outcome(True, f"open ({width * 1000:.0f} mm)")
        return Outcome(False, f"stalled at {width * 1000:.0f} mm — expected near {s.max_open_m * 1000:.0f} mm")

    def close_gripper(self, env) -> Outcome:
        env.set_gripper(0.0, n_steps=15)
        width, loaded = env.squeeze_gripper()
        if loaded:
            self._note_grip(env)
            return Outcome(True, f"closed on something (width={width * 1000:.0f} mm)")
        if width <= self.spec.min_hold_width_m:
            return Outcome(True, f"fully closed ({width * 1000:.0f} mm, nothing held)")
        return Outcome(False, f"stalled at {width * 1000:.0f} mm — nothing grasped")

    def home(self, env) -> Outcome:
        # A backend that knows whether it arrived says so (SO101Env returns a
        # bool); the RobotEnv default returns None, which is taken as arrived.
        if env.go_home() is False:
            return Outcome(False, "did not reach the home position — check for an obstruction")
        return Outcome(True, "arm at home position")

    def move_above(self, env, target: Target, height_m: "float | None" = None) -> Outcome:
        # Default to the object's own hover height rather than a literal: a
        # vertical gripper runs out of reach above some height, so a fixed
        # default drifts out of reach whenever that ceiling is re-tuned — and
        # a short object's hover puts the jaw tips through a tall one.
        s = self.spec
        if height_m is None:
            height_m = self.plan(target).hover_height_m
        goal = np.array([target.position[0], target.position[1], s.table_z + float(height_m)])
        arrived = env.move_ee_to(goal)
        if arrived and self._near(env, goal, s.approach_tol_m * 2):
            return Outcome(True, f"hovering at {height_m:.3f} m", at_xy=goal[:2])
        dist = float(np.linalg.norm(env.get_ee_pos() - goal))
        return Outcome(False, f"IK did not converge (residual {dist * 100:.1f} cm)")

    def nudge(self, env, target: Target, to_xy, obstacles=(), check_only: bool = False) -> Outcome:
        """Push target across the surface to to_xy with the closed jaws, not gripping it: nudge.py."""
        from mrcf.policies.nudge import nudge
        return nudge(self, env, target, to_xy, obstacles, check_only)

    def stand_up(self, env, target: Target, obstacles=(), vision: "Vision | None" = None) -> Outcome:
        """Stand target, fallen on its side (Target.lying), back up where it lies: stand_up.py."""
        from mrcf.policies.stand_up import stand_up
        return stand_up(self, env, target, obstacles, vision)
