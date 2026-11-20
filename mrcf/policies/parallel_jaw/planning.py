"""
mrcf/policies/parallel_jaw/planning.py — the grasp, from the object's shape:
how high on it to close, how wide to open, whether the open jaws clear its top,
and what makes a plan unsafe to try. The base the other parts of ParallelJaw
build on.

The runs behind its values and fixes, with their dates: mrcf/policies/HISTORY.md.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import cast

import numpy as np

from mrcf import flight_recorder
from mrcf.policies.aligners import ALIGNERS
from mrcf.policies.approaches import (  # noqa: F401 — re-exported
    DEFAULT_ALIGN,
    NEAR_BASE,
    REACH_OUT,
    TOP_DOWN,
    Approach,
    ApproachFit,
    _Choices,
)
from mrcf.policies.base import Outcome, Policies, Target
from mrcf.policies.parallel_jaw.spec import GraspPlan, GripperSpec

# Room wanted between an upright object and a tipped jaw (see _top_clear).
_TOP_CLEAR_M = 0.001

def _takes(fn, name: str) -> bool:
    """Whether fn takes the keyword argument `name` by name (a backend's optional extra)."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# The pads close no higher than this under an object's top.
_PADS_UNDER_TOP_M = 0.010


class GraspPlanning(Policies):
    """ParallelJaw's grasp planning, and its state (see the package docstring)."""

    def __init__(self, spec: GripperSpec, jaw_view_path=None):
        unknown = [f"{a.name}: {n}" for a in spec.approaches for n in a.align if n not in ALIGNERS]
        if unknown:
            raise ValueError(f"no aligner {', '.join(unknown)} — known: {', '.join(ALIGNERS)} "
                             "(mrcf/policies/aligners.py; register_aligner() a new one before building the policy)")
        self.spec = spec
        # Where the wrist camera's view of its own jaws is kept (perception/jaw_view.py), learned on
        # the rig (learn_jaw_view); None: the jaws are drawn from the camera's model instead.
        self.jaw_view_path = None if jaw_view_path is None else Path(jaw_view_path)
        self._jaw_view_read: tuple = (None, None)      # (the file's mtime, the JawView read from it)
        self._servo_warned = False
        self._notes: "list | None" = None     # what the pick under way has said, for its Outcome
        self._views: list = []                # and the frames it took
        self._overhead_err_m: "float | None" = None   # how far off the fixed camera last saw the
                                                      # gripper, once corrected; None: not seen
        self._overhead_off: "np.ndarray | None" = None   # and the kinematics' error it saw then (xy)
        self._overhead_first_err_m: "float | None" = None  # and how far off it saw it before correcting
        self._held_tilt_deg: "float | None" = None    # the lean the held object was gripped at
        self._place_turn_deg = 0.0                      # the jaws' turn for the place under way (its faces)
        self._attempts: list = []                     # the pick under way, one entry per attempt
        self._approach_name: "str | None" = None      # the approach its latest attempt used
        self._around: dict = {}                       # neighbours measured on the way, by label
        self._grip_view: "dict | None" = None         # the wrist camera's picture as the jaws shut on something (_note_grip)

    def jaw_view(self):
        """The wrist camera's learned view of the jaws (perception.jaw_view.JawView), None without one."""
        path = self.jaw_view_path
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        if self._jaw_view_read[0] != mtime:
            from mrcf.perception.jaw_view import JawView
            self._jaw_view_read = (mtime, JawView.load(path))
        return self._jaw_view_read[1]

    def _say(self, tag: str, msg: str) -> None:
        """
        Print one line about the action under way, and keep it for its Outcome
        (Outcome.notes) while a pick is running: a pick that fails after three
        attempts has to be diagnosed from why each one was retried, and the
        console is gone by then.
        """
        if getattr(self, "_quiet", False):
            return                       # a check of what a pick would do (pick_check), not a pick
        line = f"{tag} {msg}" if tag else msg
        print(f"  {line}")
        flight_recorder.event("policy", "say", text=line)
        if self._notes is not None:
            self._notes.append(line)

    def _done(self, outcome: Outcome) -> Outcome:
        """outcome, with what the pick said, saw and tried on the way (see _say)."""
        outcome.notes, outcome.views, outcome.attempts = self._notes or [], self._views, self._attempts
        outcome.around = self._around
        if outcome.approach is None:
            outcome.approach = self._approach_name
        self._notes, self._views, self._attempts, self._around = None, [], [], {}
        return outcome

    # ── Sizing ──────────────────────────────────────────────────────────────────

    def _plan(self, height, width, grasp, grip, opening) -> GraspPlan:
        s = self.spec
        return GraspPlan(
            height_m=height, width_m=width, grasp_height_m=grasp, grip_width_m=grip,
            pregrasp_open_m=opening,
            # The jaw tips hang tip_below_grasp_m below the grasp point, and must
            # clear the object's top on the way in.
            hover_height_m=max(s.hover_height_m, height + s.tip_below_grasp_m + s.top_clearance_m),
            lift_height_m=max(s.lift_height_m, grasp + s.lift_clearance_m),
            # Released holding it at the height it was grasped at, so its foot
            # ends up just clear of the surface rather than driven into it.
            release_height_m=grasp + s.release_clearance_m)

    @staticmethod
    def usable_widths(widths_mm: "dict | None") -> list:
        """
        [(height_mm, width_mm)] up a measured profile, without the levels no
        upright object has: a width under 3 mm or under 0.3 of the profile's
        median is an outline cut short — by the jaws, the frame edge, or a
        base lost against a surface of its own colour — not the object.
        """
        levels = sorted((float(k), float(v)) for k, v in (widths_mm or {}).items())
        if not levels:
            return []
        floor = max(3.0, 0.3 * float(np.median([w for _, w in levels])))
        return [(z, w) for z, w in levels if w >= floor]

    def _unevenness_mm(self, levels: list, g_mm: float) -> float:
        """How much the width changes over the pads' height centred at g_mm (max - min, mm)."""
        z, w = np.array(levels).T
        half = self.spec.tip_below_grasp_m * 1000     # the pads reach as far below the grasp point as above
        lo, hi = max(g_mm - half, z[0]), min(g_mm + half, z[-1])
        inside = w[(z > lo) & (z < hi)]
        seen = np.r_[np.interp(lo, z, w), np.interp(hi, z, w), inside]
        return float(seen.max() - seen.min())

    def grasp_height_for(self, height_m: float, widths_mm: "dict | None") -> "tuple[float, str]":
        """
        Where up a measured object the pads should close: (metres above the
        surface, why).

        Within reach first: high enough that the jaw tips stay off the surface
        (min_grasp_height_m) and the top no more than free_depth_m into the
        gripper, low enough that the pads are on the object, 10 mm under its
        top. Within that, where its sides are straight over the pads' height:
        a taper or a ridge there is what the closing jaws cam the object out
        along. Of the straight enough heights (grasp_flat_mm, or within half a
        millimetre of the straightest when none is), the nearest
        grasp_height_fraction of the height. No profile: that fraction.
        """
        s = self.spec
        lo = max(s.min_grasp_height_m, height_m - s.free_depth_m)
        hi = height_m - _PADS_UNDER_TOP_M
        prefer = s.grasp_height_fraction * height_m
        # Never with the jaw tips into the surface, whatever the object.
        floor = s.tip_below_grasp_m + 0.002
        levels = self.usable_widths(widths_mm)
        if hi <= lo or len(levels) < 3:
            g = max(min(max(prefer, lo), hi), floor)
            return g, "no profile to choose by" if len(levels) < 3 else "short: as high as its top allows"
        heights = np.arange(lo * 1000, hi * 1000 + 1e-6, 1.0)
        uneven = np.array([self._unevenness_mm(levels, h) for h in heights])
        straightest = float(uneven.min())
        flat = uneven <= (s.grasp_flat_mm if straightest <= s.grasp_flat_mm else straightest + 0.5)
        options = heights[flat]
        best = int(np.argmin(np.abs(options - prefer * 1000)))
        g_mm = float(options[best])
        varies = float(uneven[flat][best])
        why = (f"its width changes {varies:.1f} mm over the pads there"
               + ("" if varies <= s.grasp_flat_mm else " — the straightest it has"))
        return max(g_mm / 1000, floor), why

    def plan_from_shape(self, height_m: float, widths_mm: dict, foot_m: "float | None" = None) -> GraspPlan:
        """
        A grasp from a measured shape: height, and {height_mm: width_mm} up the
        object. Widths are the outline's, so they are what the jaws meet.
        Grasped where grasp_height_for says.
        """
        s = self.spec
        g, _ = self.grasp_height_for(height_m, widths_mm)
        levels = self.usable_widths(widths_mm)
        if levels:
            hs, ws = np.array(levels).T
            grip = float(np.interp(g * 1000, hs, ws)) / 1000
            # The open jaws pass everything from their tips' height up.
            passed = ws[hs >= max(0.0, (g - s.tip_below_grasp_m) * 1000 - 2.0)]
            width = float(passed.max()) / 1000 if len(passed) else grip
        else:
            grip = width = foot_m or (s.unknown_open_m - 2 * s.jaw_slack_m)
        opening = min(s.max_open_m, max(width, grip) + 2 * s.jaw_slack_m)
        return self._plan(height_m, width, g, grip, opening)

    def plan(self, target: Target) -> GraspPlan:
        """
        _plan_for's plan, gripped no lower than target.grasp_above_m where the
        object is tall enough for that (spec.grip_over_top_m below its top).
        """
        return self.gripped_above(self._plan_for(target), target.grasp_above_m)

    def pads_clear_at(self, plan: GraspPlan, tilt_deg: "float | None") -> GraspPlan:
        """
        plan, gripped higher by how far a lean drops the low end of tangential
        jaws' pads below where they come straight down (finger_half_width_m x
        sin(lean)), so the pads still close above the object's base — capped
        spec.grip_over_top_m under its top. Radial jaws: _top_clear's.
        2026-11-04, e7 at -55 deg: the pads' low ends closed on a pawn's base
        (22 mm across, against 15 at its neck), and held by the edge of its
        base the pawn was put down 28 mm off its square.
        """
        s = self.spec
        if s.jaw_axis != "tangential" or tilt_deg is None or abs(float(tilt_deg)) < 1.0:
            return plan
        drop = s.finger_half_width_m * float(np.sin(np.radians(abs(float(tilt_deg)))))
        return self.gripped_above(plan, plan.grasp_height_m + drop)

    def gripped_above(self, plan: GraspPlan, above_m: "float | None") -> GraspPlan:
        """plan, its grasp raised to above_m (capped spec.grip_over_top_m under its top); as it was when None or lower."""
        if above_m is None or plan.grasp_height_m >= above_m:
            return plan
        g = min(float(above_m), plan.height_m - self.spec.grip_over_top_m)
        return plan if g <= plan.grasp_height_m else dataclasses.replace(plan, grasp_height_m=g)

    def _plan_for(self, target: Target) -> GraspPlan:
        """
        How to grasp target before anything has been checked against its
        neighbours. Measured: from its shape. Not yet: hover clear of the
        tallest it could be, and grasp as low as an object that tall allows —
        a low grasp suits a short object and a tall one alike; the pick
        measures it from the hover anyway. Opened by its foot as the scan saw
        it, or wider.

        Where the deployment says the shortest it could be (min_height_m), as
        high as that one still has the pads on it, as long as the tallest
        still fits in the gripper: higher up, objects of a kind are slimmer,
        and that is where crowded neighbours leave the fingers room. Where it
        says how slim (assumed_widths_mm), the jaws are opened by that from
        the tips up, never wider than the foot. On the full chess board
        (2026-11-03), grasped as low as a king allows and opened by its foot,
        no back-rank piece could be picked between two others.
        """
        s = self.spec
        tallest = target.max_height_m or s.default_max_height_m
        if target.measured:
            height_m = cast(float, target.height_m)     # measured: it has one
            plan = self.plan_from_shape(height_m, target.widths_mm or {}, target.foot_m)
            if self.shape_unsettled(target) and tallest > height_m:
                # One look, which may be wrong (Target.shape_looks): the pick
                # looks again from the hover, so that hover clears whatever
                # the object could be, as for one never measured.
                return dataclasses.replace(plan, hover_height_m=self._plan(
                    tallest, plan.width_m, plan.grasp_height_m, plan.grip_width_m,
                    plan.pregrasp_open_m).hover_height_m)
            return plan
        if self.height_known(target):
            # Its kind is known, so its height is: gripped where a measured
            # one of that height would be, by its kind's shape (the widest it
            # can be up its height, as the deployment has it), not 10 mm under
            # its top. 2026-11-11, the f1 bishop named as one: the shortest-
            # piece rule put the pads at 52 mm on a 62 mm bishop, on its
            # tapering mitre; a measured bishop is gripped at 31.
            return self.plan_from_shape(tallest, target.assumed_widths_mm or {}, target.foot_m)
        grasp = max(s.min_grasp_height_m, tallest - s.free_depth_m)
        if target.min_height_m is not None:
            grasp = max(grasp, target.min_height_m - _PADS_UNDER_TOP_M)
        width = grip = target.foot_m or (s.unknown_open_m - 2 * s.jaw_slack_m)
        levels = self.usable_widths(target.assumed_widths_mm)
        if levels:
            hs, ws = np.array(levels).T
            passed = ws[hs >= (grasp - s.tip_below_grasp_m) * 1000 - 2.0]
            if len(passed):
                width = min(width, float(passed.max()) / 1000)
            grip = min(width, float(np.interp(grasp * 1000, hs, ws)) / 1000)
        opening = min(s.max_open_m, max(s.unknown_open_m, width + 2 * s.jaw_slack_m))
        return self._plan(tallest, width, grasp, grip, opening)

    @staticmethod
    def height_known(target: Target) -> bool:
        """Not measured, but the deployment knows how tall it is: the shortest and tallest it could be agree."""
        return (not target.measured and target.min_height_m is not None and target.max_height_m is not None
                and abs(target.max_height_m - target.min_height_m) <= 0.001)

    def shape_unsettled(self, target: Target) -> bool:
        """target's measured shape rests on fewer agreeing looks than spec.trust_shape_after."""
        return (target.measured and target.shape_looks is not None
                and target.shape_looks < self.spec.trust_shape_after)

    def problems(self, plan: GraspPlan) -> list:
        """What is physically wrong with a plan, if anything."""
        s, out = self.spec, []
        if plan.height_m - plan.grasp_height_m > s.free_depth_m:
            out.append(f"top is {(plan.height_m - plan.grasp_height_m) * 1000:.0f} mm above the grasp "
                       f"point; the gripper has {s.free_depth_m * 1000:.0f} mm — grasp higher")
        if plan.grip_width_m >= plan.pregrasp_open_m:
            out.append(f"grip width {plan.grip_width_m * 1000:.0f} mm does not fit the "
                       f"{plan.pregrasp_open_m * 1000:.0f} mm opening")
        if plan.pregrasp_open_m <= plan.width_m:
            out.append(f"opening {plan.pregrasp_open_m * 1000:.0f} mm does not clear the "
                       f"{plan.width_m * 1000:.0f} mm width")
        if plan.pregrasp_open_m > s.max_open_m:
            out.append(f"opening {plan.pregrasp_open_m * 1000:.0f} mm exceeds the jaws' "
                       f"{s.max_open_m * 1000:.0f} mm")
        return out

    def _widened(self, plan: GraspPlan, room: "float | None" = None) -> GraspPlan:
        """
        plan, its jaws opened to at least the wrist servo's opening
        (spec.servo_open_m) where the neighbours leave room: what every pick
        that worked came down at, the servo having opened them to it. An
        approach without the servo would otherwise come down at the plan's
        own few millimetres of slack — little for a fixed camera's 1-3 mm,
        and less still for jaws tipped by a lean (_top_clear).
        """
        s = self.spec
        if s.servo_open_m is None or plan.pregrasp_open_m >= s.servo_open_m:
            return plan
        wide = min(s.servo_open_m, s.max_open_m if room is None else room)
        if wide <= plan.pregrasp_open_m:
            return plan
        return dataclasses.replace(plan, pregrasp_open_m=wide)

    @staticmethod
    def _levels(plan: GraspPlan, widths_mm: "dict | None") -> list:
        """[(height_m, width_m)] up the object, to its top: its measured widths, else its plan's width all the way."""
        levels = sorted((float(k) / 1000, float(v) / 1000) for k, v in (widths_mm or {}).items())
        if not levels:
            return [(0.0, plan.width_m), (plan.height_m, plan.width_m)]
        if levels[-1][0] < plan.height_m - 1e-6:
            levels.append((plan.height_m, levels[-1][1]))
        return levels

    def top_overlap(self, plan: GraspPlan, widths_mm: "dict | None", tilt_deg: "float | None",
                    grasp_m: "float | None" = None, levels: "list | None" = None) -> float:
        """
        How far an upright object pokes past radial jaws tipped by tilt_deg,
        grasped at grasp_m above the surface (default: the plan's) — metres,
        <= 0 when it clears them. Above the grasp it leans toward the outer
        pad, below it (down to the jaw tips) toward the inner one; see
        _top_clear. 0 for tangential jaws, which lean along their pads.
        """
        s = self.spec
        if s.jaw_axis != "radial" or tilt_deg is None:
            return 0.0
        lean = np.radians(abs(float(tilt_deg)))
        t, half_gap = float(np.tan(lean)), plan.pregrasp_open_m / 2 / float(np.cos(lean))
        g = plan.grasp_height_m if grasp_m is None else float(grasp_m)
        levels = levels if levels is not None else self._levels(plan, widths_mm)
        return max((abs(z - g) * t + w / 2 - half_gap for z, w in levels if z >= g - s.tip_below_grasp_m),
                   default=0.0)

    def _top_clear(self, plan: GraspPlan, widths_mm: "dict | None", tilt_deg: float) -> GraspPlan:
        """
        plan, grasping higher up the object if a lean would put the object
        against a jaw — the same plan when it would not, or when grasping
        higher does not help.

        Radial jaws open in the plane the arm leans in, so a lean tips the pads.
        The gap's centre line then runs up the gripper's axis, and an upright
        object between them sits further toward the outer pad the higher up it
        goes: (height above the grasp) x tan(lean), against the tipped pads'
        level half-gap of opening / 2 / cos(lean). At the 30-50 deg the far
        ranks need, the top of a pawn grasped low is past the outer pad, and
        that jaw comes down on it. Grasping higher shortens what stands above
        the pads. Tangential jaws lean along their pads: nothing to do.
        """
        s = self.spec
        if s.jaw_axis != "radial" or abs(float(tilt_deg)) < 1.0:
            return plan
        levels = self._levels(plan, widths_mm)

        def past(g: float) -> float:
            return self.top_overlap(plan, widths_mm, tilt_deg, grasp_m=g, levels=levels)

        if past(plan.grasp_height_m) <= -_TOP_CLEAR_M:
            return plan
        top = plan.height_m - 0.010            # plan_from_shape's highest grasp
        if top <= plan.grasp_height_m:
            return plan
        tries = np.arange(plan.grasp_height_m, top + 1e-9, 0.002)
        fits = [g for g in tries if past(g) <= -_TOP_CLEAR_M]
        g = float(fits[0]) if fits else float(min(tries, key=past))
        if g <= plan.grasp_height_m + 1e-6:
            return plan
        hs, ws = np.array(levels).T
        grip = float(np.interp(g, hs, ws))
        if grip + 2 * s.jaw_slack_m > plan.pregrasp_open_m:
            return plan
        passed = ws[hs >= g - s.tip_below_grasp_m - 0.002]
        return dataclasses.replace(
            plan, grasp_height_m=g, grip_width_m=grip,
            width_m=float(max(passed.max(), grip)) if len(passed) else grip,
            lift_height_m=max(s.lift_height_m, g + s.lift_clearance_m),
            release_height_m=g + s.release_clearance_m)
