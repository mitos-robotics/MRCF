"""
mrcf/policies/parallel_jaw/place.py — a place: carry what is held over the
target, let it down, release, and check it was let go.

The runs behind its values and fixes, with their dates: mrcf/policies/HISTORY.md.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import cast

import numpy as np

from mrcf import safety
from mrcf.capabilities import GripSensing, MoveRefusals
from mrcf.interlock import MotionStopped
from mrcf.policies.approaches import (  # noqa: F401 — re-exported
    DEFAULT_ALIGN,
    NEAR_BASE,
    REACH_OUT,
    TOP_DOWN,
    Approach,
    ApproachFit,
    _Choices,
)
from mrcf.policies.base import Outcome, Target
from mrcf.policies.parallel_jaw.sighting import SightingMixin
from mrcf.policies.parallel_jaw.spec import GraspPlan

# A release that reads loaded with the jaws open past the object is read again after this long.
_LOAD_SETTLE_S = 0.4

# The interlock source the grip watch trips (PlaceMixin._watching_grip).
_GRIP_CHECK = "parallel_jaw:grip"


class _Slipped(Exception):
    """The object left the jaws while they carried it; str() says how."""


class PlaceMixin(SightingMixin):
    """ParallelJaw's place (see the package docstring)."""

    @contextlib.contextmanager
    def _watching_grip(self, env):
        """
        While carrying: a per-cycle check on the arm's interlock that stops the
        arm the moment the jaws close more than spec.slip_m past where they
        held the object — it has slipped out — and raises _Slipped in place of
        that stop. Without it the place found out at the release, seconds and
        a whole carry later. The trip counts as one (Interlock.trip_count), so
        the controller looks at the scene again before the next command.
        Nothing is watched without an interlock or slip_m.
        """
        interlock, tol = env.interlock, self.spec.slip_m
        held = env.get_gripper_width()
        if interlock is None or tol is None or not np.isfinite(held):
            yield
            return
        why: list = []

        def check():
            width = env.get_gripper_width()
            if np.isfinite(width) and width < held - tol:
                why[:] = [f"the object slipped out of the jaws on the way: they closed from "
                          f"{held * 1000:.0f} to {width * 1000:.0f} mm"]
                return why[0]
            return None

        interlock.add_check(_GRIP_CHECK, check)
        try:
            yield
        except MotionStopped:
            if not why:
                raise
            raise _Slipped(why[0]) from None
        finally:
            interlock.remove_check(_GRIP_CHECK)     # and the trip it caused

    def _place_hold(self, env, xy, plan: GraspPlan, obstacles, tilt_deg) -> tuple:
        """
        How a place at xy comes down: (approach, rot, carry height, checked),
        the first of approach_order that reaches the transit, the release and
        the retreat. A "highest" approach may carry lower than the plan's lift,
        down to its hover: near the base the plan's carry height straight above
        the target is out of reach, and the release below it is not.
        (None, None, None, True): none reaches. checked False: the backend
        cannot say, and the plan's own heights are used.
        """
        swept = None                           # the least any lean would have swept too low, if that was why
        for approach in self.approach_order(env, xy):
            mine = self._own_hover(approach, plan)
            heights = self._lift_heights(mine) if approach.lift == "highest" else [mine.lift_height_m]
            for carry in heights:
                (_, transit, transit_tol, _), (_, descend, release_tol, _), (_, retreat, retreat_tol, _) = \
                    self.place_waypoints(xy, dataclasses.replace(mine, lift_height_m=carry))
                rot, tilt, checked = self._hold(
                    env, [("transit", transit, transit_tol), ("release", descend, release_tol),
                          ("retreat", retreat, retreat_tol)], "[Place]", obstacles,
                    tilt_deg, approach.lean_margin_deg, quiet=True, open_m=plan.grip_width_m,
                    turn_deg=self._place_turn_deg)
                if not checked:
                    return approach, None, plan.lift_height_m, False
                if rot is not None:
                    return approach, rot, carry, True
                if self._swept is not None and (swept is None or self._swept[0] < swept[0]):
                    swept = self._swept
        self._swept = swept
        return None, None, None, True

    def place(self, env, at, held: "Target | None" = None, grasp=None, obstacles=(),
              tilt_deg: "float | None" = None, surface_z: "float | None" = None,
              faces_deg: "float | None" = None) -> Outcome:
        """
        Put the held object down with its foot at `at`; on something raised
        (surface_z), ParallelJaw.place has already raised spec.table_z to it.
        grasp is how it was
        picked (pick()'s Outcome.grasp), which sets the carry and release
        heights; without it, a plan for `held`. tilt_deg: a lean the
        scenario's rules ask for at `at` (see hold_orientation). It comes
        down as pick()'s approaches do for that spot (see _place_hold), and is
        refused, still holding the object, where none reaches.
        """
        s = self.spec
        set_out = time.monotonic()
        plan = grasp if isinstance(grasp, GraspPlan) else self.plan(held or Target(np.zeros(3)))
        xy = np.asarray(at[:2], dtype=float)
        obstacles = self._with_base(env, obstacles)
        plan, problem, _ = self.fit(env, plan, xy, obstacles)
        if problem:
            return Outcome(False, problem, refused=True, grasp=plan)
        if not self.holding(env):
            return Outcome(False, "not holding an object — pick first", refused=True, grasp=plan)

        # Leaning as it was picked, where the arm can: the object hangs below
        # the jaws, so a different lean swings its foot by millimetres. Else the
        # scenario's rule for `at`.
        lean = self._held_tilt_deg if self._held_tilt_deg is not None else tilt_deg
        # Squared on the faces asked for (a block on a block): held across its own,
        # its sides turn with the jaws.
        turns = self.face_turns(env, xy, faces_deg)
        self._place_turn_deg = turns[0] if turns else 0.0
        approach, _, carry, checked = self._place_hold(env, xy, plan, obstacles, lean)
        if approach is None:
            swept = getattr(self, "_swept", None)
            if swept is not None:
                # Its gripper would sweep into what stands toward the base: the
                # object gripped higher would pass over it.
                above = plan.grasp_height_m + swept[0] + 0.004
                return Outcome(False, f"leaning to reach the target, the gripper would pass {swept[0] * 1000:.0f} mm "
                                      f"too low over {swept[1]} — still holding the object; gripped at "
                                      f"{above * 1000:.0f} mm it would clear it",
                               refused=True, grasp=plan, regrip_above_m=above)
            return Outcome(False, "the target is out of reach for every approach — still holding the object",
                           refused=True, grasp=plan)
        if carry < plan.lift_height_m - 1e-6:
            self._say("[Place]", f"{approach.name}: carrying at {carry * 1000:.0f} mm, not "
                                 f"{plan.lift_height_m * 1000:.0f} — as high as reaches straight above the target.")
        (_, transit, transit_tol, _), (_, descend, release_tol, release_step), \
            (_, retreat, retreat_tol, _) = self.place_waypoints(
                xy, dataclasses.replace(self._own_hover(approach, plan), lift_height_m=carry))
        # Chosen again, aloud, for the heights it found: the same search, now saying what it holds.
        rot, _, _ = self._hold(env, [("transit", transit, transit_tol), ("release", descend, release_tol),
                                     ("retreat", retreat, retreat_tol)], "[Place]", obstacles, lean,
                               approach.lean_margin_deg,
                               tilt_why=("it was picked at" if self._held_tilt_deg is not None
                                         else "the approach rule here asks for"), open_m=plan.grip_width_m,
                               turn_deg=self._place_turn_deg)

        # Carried watching the grip: a slip stops the arm where it happens.
        try:
            with self._watching_grip(env):
                refusals = env.move_refusals if isinstance(env, MoveRefusals) else 0
                arrived = self._move_again_if_short(env, transit, transit_tol, rot,
                                                    env.move_ee_to(transit, tol=transit_tol, orientation=rot))
                if isinstance(env, MoveRefusals) and env.move_refusals > refusals \
                        and not self._near(env, transit, s.approach_tol_m * 2):
                    return Outcome(False, ("the arm will not go there: "
                                           f"{env.last_refusal or 'it refused the move'} — still "
                                           "holding the object"), refused=True, grasp=plan, approach=approach.name)
                # The same rule as the pick's hover: a 3 cm proximity test alone approves
                # a move that stalled 1-2 cm out, and the object is then set down that far off.
                if not arrived or not self._near(env, transit, s.approach_tol_m * 2):
                    dist = float(np.linalg.norm(env.get_ee_pos() - transit))
                    return Outcome(False, f"transit to target did not converge (residual {dist * 100:.1f} cm)",
                                   grasp=plan, approach=approach.name)
                # Over the target as the fixed camera sees it: nothing else corrects a
                # place for the arm's own error, which can be a few centimetres.
                corr = self._overhead_correct(env, transit, transit_tol, rot, "[Place]")
                descend_base, retreat_base = descend.copy(), retreat.copy()
                if np.any(corr):
                    descend, retreat = descend.copy(), retreat.copy()
                    descend[:2] += corr
                    retreat[:2] += corr

                # Same gentle final approach as the pick: coming down fast with an
                # object in the jaws knocks over whatever stands next to the target.
                near = safety.moved_near(env, descend[:2], set_out)
                if near is not None:
                    # A hand, or something knocked over, came by on the way: what it was
                    # to go on (a stack's top) may not be there now. Kept hold of, to look.
                    return Outcome(False, (f"something moved {near * 100:.0f} cm from where it was going while the "
                                           "arm was on its way — not setting it down there; still holding the "
                                           "object"), grasp=plan, approach=approach.name)
                # Setting it down: a press is it landing (safety.descend_onto).
                descended = safety.descend_onto(env, descend, tol=release_tol, max_step_deg=release_step,
                                                orientation=rot)
                if not descended and self._stalled_at_start(env, transit[2], descend):
                    here = env.get_ee_pos()
                    env.move_ee_to(np.array([here[0], here[1], transit[2]]), tol=transit_tol, orientation=rot)
                    descended = safety.descend_onto(env, descend, tol=release_tol, max_step_deg=release_step,
                                                    orientation=rot)
                # Far out the arm reaches less than its model: a descent the model
                # approved can stall centimetres above the release. More lean reaches
                # further down there — up, lean further, down again.
                tilt = self._tilt_of(env, rot)
                for _ in range(s.descent_lean_retries if tilt is not None else 0):
                    if descended:
                        break
                    new_rot, new_tilt = self._leaned_further(                      # runs only when tilt is set
                        env, cast(float, tilt), [("transit", transit, transit_tol), ("release", descend, release_tol),
                                    ("retreat", retreat, retreat_tol)], descend, "[Place]", obstacles, approach.lean_margin_deg)
                    if new_rot is None:
                        break
                    here = env.get_ee_pos()
                    env.move_ee_to(np.array([here[0], here[1], transit[2]]), tol=transit_tol, orientation=rot)
                    rot, tilt = new_rot, new_tilt
                    env.move_ee_to(transit + np.r_[corr, 0.0], tol=transit_tol, orientation=rot)
                    again = self._overhead_correct(env, transit, transit_tol, rot, "[Place]")
                    if np.any(again):
                        corr = again
                    descend = descend_base + np.r_[corr, 0.0]
                    retreat = retreat_base + np.r_[corr, 0.0]
                    descended = safety.descend_onto(env, descend, tol=release_tol, max_step_deg=release_step,
                                                    orientation=rot)
                # Opening the jaws short of the release height drops the object from
                # wherever the arm stopped. Keep hold of it and report instead.
                if not descended:
                    dist = float(np.linalg.norm(env.get_ee_pos() - descend))
                    return Outcome(False, (f"descent to the release height stalled {dist * 100:.1f} cm short — "
                                           "still holding the object"), grasp=plan, approach=approach.name)
        except _Slipped as slip:
            here = env.get_ee_pos()
            above = float(here[2] - descend[2])
            if not (above <= s.touchdown_m and float(np.linalg.norm(here[:2] - descend[:2])) <= s.approach_tol_m):
                self._say("[Place]", f"{slip} — stopped there.")
                return Outcome(False, str(slip), grasp=plan, approach=approach.name)
            # The jaws gave in the last few millimetres down: its foot met the
            # surface before the release height — leaning, it hangs lower than
            # the jaws' model has it — and it pivoted between the pads. Not a
            # slip: let go of it there (2026-11-04, a pawn onto e7 at -40 deg,
            # 8 mm up; held on to, it went with the arm and stood on e6).
            self._say("[Place]", f"The jaws gave {above * 1000:.0f} mm above the release height — it touched "
                                 "down early; letting go there.")
            touched = True
        else:
            touched = False

        # Still holding it? An object can fall out of the jaws on the way, and
        # opening them on nothing is not a place.
        if not touched and isinstance(env, GripSensing) and not self.holding(env):
            self._move_through(env, retreat, retreat_tol, rot)
            return Outcome(False, "the object was no longer in the jaws at the release — dropped on the way",
                           grasp=plan, approach=approach.name)
        # Over the target as the fixed camera sees the jaws now, before they
        # open, as at a pick's grasp: nothing else checks the descent, and a
        # leaning one can drift out by more than a centimetre. Not once it
        # stands on the surface: moving the jaws would drag it.
        if not touched and self._can_overhead(env):
            self._overhead_at_grasp(env, xy, xy, descend, rot, "[Place]", what="release")

        env.set_gripper(plan.pregrasp_open_m, n_steps=15)

        # set_gripper may return on a stall or timeout as well as on arrival.
        # Require both an opening and no sensed grip before lifting away.
        width = env.get_gripper_width()
        loaded = self.holding(env) if isinstance(env, GripSensing) else None
        if touched and loaded is None:
            loaded = False
        if loaded and np.isfinite(width) and width >= plan.grip_width_m + 2 * s.jaw_slack_m:
            # Open wider than the object, and still loaded: the servo's load
            # can read high just after the jaws moved — read it again once
            # they have stopped, or the place refuses to retreat and the next
            # move sweeps the open jaws through the object. A jam still reads
            # loaded.
            time.sleep(_LOAD_SETTLE_S)
            loaded = self.holding(env)
        if not np.isfinite(width) or width < 0.8 * plan.pregrasp_open_m or loaded:
            released = loaded is False
            message = ("object released but jaws did not open far enough — not retreating" if released
                       else "gripper did not release the object — not retreating")
            return Outcome(False, message, grasp=plan, at_xy=xy if released else None, released=released,
                           approach=approach.name)

        # Up to the object's own hover height: the open jaw tips have to clear
        # its top before anything moves sideways.
        try:
            retreated = self._move_through(env, retreat, retreat_tol, rot)
        except Exception as exc:
            return Outcome(False, f"released object but retreat raised: {exc}",
                           grasp=plan, at_xy=xy, released=True, approach=approach.name)
        if not retreated:
            return Outcome(False, "released object but retreat did not converge",
                           grasp=plan, at_xy=xy, released=True, approach=approach.name)

        faces = (self.faces_at(env, xy, self._place_turn_deg)
                 if held is not None and held.faces_deg is not None else None)
        return Outcome(True, "placed object" + (" (let go where it touched down)" if touched else ""),
                       grasp=plan, at_xy=xy, released=True, approach=approach.name, faces_deg=faces)
