"""
mrcf/policies/approaches.py — the ways a two-finger pick comes at an object.

Which one a pick uses depends on where the object stands from the arm's base
(ParallelJaw.zone): the approaches listed for its zone first, each worked out
against the arm's reach before anything moves (ParallelJaw.fit_approach). An
Approach is data — where it is a first choice, whether the wrist servo aligns,
how it lifts, how it lines the jaws up over the object (`align`, names of
mrcf/policies/aligners.py's strategies, tried in order) — so a new way of coming
at an object is a new Approach, and at most one new aligner.

    MY_WAY = Approach("my_way", zones=("far",), align=("fixed_camera", "kinematics"))
    make_policies(approaches=(TOP_DOWN, NEAR_BASE, REACH_OUT, MY_WAY))

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import GraspPlan

# How an approach lines the jaws up over the object once over it, in order of
# preference: the first whose conditions hold runs (mrcf/policies/aligners.py).
DEFAULT_ALIGN = ("wrist_camera_foot", "overhead_first", "wrist_servo", "fixed_camera", "kinematics")


@dataclass(frozen=True)
class Approach:
    """
    One way to come at an object: a pick (and place) strategy. Which one is
    used depends on where the object stands relative to the arm's base — the
    same gripper comes at an object next to the base, in the middle of its
    reach, and at the end of its reach in different ways, because a different
    thing runs out at each:

      near the base   holding a point high above it: the grasp itself is easy,
                      but the climb to the wrist servo's 70 mm and the lift
                      straight up may not be (on the SO-101 the hover over d1
                      is out of reach at any lean, the grasp is not). Top-down
                      with the servo still goes first where it reaches: the
                      fixed camera alone aligns the jaws over where the scan
                      put the object, and the scan's foot is the less sure
      mid-reach       nothing: top-down with the wrist servo, as calibrated
      far out         the arm's length: the grasp alone needs a lean of 20-50
                      deg, whatever the path to it, and that lean takes the
                      wrist servo far from the lean it was calibrated at, and
                      tips radial jaws relative to an upright object

    zones       where it is a first choice: "near", "mid" and/or "far" (see
                GripperSpec.near_base_m / far_reach_m); of two in one zone, the
                one listed first in GripperSpec.approaches
    servo       align with the wrist camera's servo, from the height it was
                calibrated at; else align by the fixed camera that sees the
                gripper, and never descend without that sighting
    servo_max_lean_change_deg
                ... and only while the held lean is within this of the lean the
                servo was calibrated at (GripperSpec.servo_tilt_deg)
    lift        "full": straight up to the plan's lift height; "highest": as high
                straight up as the held orientation reaches, down to the hover
    lean_margin_deg
                hold the least lean that leaves every joint this far from its
                limit, where the backend can say (else the least that reaches)
    keep_top_clear
                with radial jaws and a lean, grasp high enough up the object
                that its top stays inside the tipped jaws (see _top_clear)
    own_hover   hover only as high as the object needs (its top, the jaw tips,
                GripperSpec.top_clearance_m), not GripperSpec.hover_height_m's floor:
                close to the base the height above the spot is what runs out
    reach_margin
                which way the grasp must also reach GripperSpec.servo_reach_margin_m
                further, for the correction made after the orientation is chosen:
                "out" (away from the base, where reach runs out far from it) or
                "in" (toward it, where it runs out close to it)
    align       how the jaws are lined up over the object once over it: names of
                mrcf/policies/aligners.py's strategies, the first that applies runs
    """
    name: str
    zones: tuple = ("mid",)
    servo: bool = True
    servo_max_lean_change_deg: "float | None" = None
    lift: str = "full"
    lean_margin_deg: "float | None" = None
    keep_top_clear: bool = False
    own_hover: bool = False
    reach_margin: str = "out"
    align: tuple = DEFAULT_ALIGN


# Near the base, top_down before near_base wherever it reaches: near_base has
# only the fixed camera to align by, over where the scan put the foot, which is
# the less sure (see Approach).
TOP_DOWN = Approach("top_down", zones=("near", "mid"))
# Its hover is the object's own: close to the base a few millimetres of hover
# decide whether a spot is in reach at all.
NEAR_BASE = Approach("near_base", zones=("near",), servo=False, lift="highest", reach_margin="in",
                     own_hover=True)
REACH_OUT = Approach("reach_out", zones=("far",), servo_max_lean_change_deg=10.0, lift="highest",
                     lean_margin_deg=3.0, keep_top_clear=True)


@dataclass
class ApproachFit:
    """
    An Approach worked out for one object at one spot: what the pick will
    hold and do. rot None: the backend cannot check orientations, so each move
    chooses its own (checked: False).
    """
    approach: Approach
    plan: "GraspPlan"
    rot: "np.ndarray | None"
    tilt_deg: "float | None"
    lift_height_m: float                 # above the surface
    servo: bool                          # the wrist servo aligns, from its own height
    checked: bool = True
    servo_skipped: bool = False          # this rig has a wrist servo, and the approach does not use it here
    grasp_over_m: "float | None" = None  # grasped this high (above the surface) so the leaning gripper
                                         # passes over what stands toward the base (fit_approach)


class _Choices:
    """
    The approaches for one pick, in order, each worked out only when it is
    needed (fit_approach probes the arm's reach several times over): the
    first that can be done for the first attempt, the next ones only if an
    attempt fails.
    """

    def __init__(self, order: list, plan, fit_one):
        self.order, self.plan, self._fit_one = list(order), plan, fit_one
        self.fits: list = []
        self.why_not: list = []
        self._next = 0

    def feasible(self, k: int) -> "ApproachFit | None":
        """The k-th approach in order that can be done, or None."""
        while len(self.fits) <= k and self._next < len(self.order):
            fit = self._fit_one(self.order[self._next], self.plan)
            self._next += 1
            (self.why_not if isinstance(fit, str) else self.fits).append(fit)
        return self.fits[k] if k < len(self.fits) else None

    def for_attempt(self, attempt: int) -> "ApproachFit | None":
        """
        The approach for attempt `attempt` (from 0): the first, then the next
        one along, then the first again, then the one after... One that failed
        for a reason of its own (a servo that lost the object, a hover out of
        reach) is not simply run again three times; and the zone's own
        approach still gets every other attempt.
        """
        if attempt % 2 == 0:
            return self.feasible(0)
        return self.feasible(attempt // 2 + 1) or self.feasible(0)

    def named(self, name: str) -> "ApproachFit | None":
        """The approach called `name`, if it can be done here."""
        k = 0
        while (fit := self.feasible(k)) is not None:
            if fit.approach.name == name:
                return fit
            k += 1
        return None

    def reset(self, plan) -> None:
        """Work them out again, for a plan that has changed (the object was measured)."""
        self.plan, self.fits, self.why_not, self._next = plan, [], [], 0


__all__ = ["Approach", "ApproachFit", "DEFAULT_ALIGN", "TOP_DOWN", "NEAR_BASE", "REACH_OUT"]
