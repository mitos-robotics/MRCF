"""
mrcf/policies/parallel_jaw/pick.py — a pick, phase by phase (the package
docstring lists them), retried with the next approach when a phase cannot go on.

The runs behind its values and fixes, with their dates: mrcf/policies/HISTORY.md.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from mrcf import safety
from mrcf.capabilities import GripperBelowGrasp, GripperTilt, HeldWidth, MoveRefusals
from mrcf.policies.aligners import ALIGNERS, GiveUp, Refuse, Retry
from mrcf.policies.approaches import (  # noqa: F401 — re-exported
    DEFAULT_ALIGN,
    NEAR_BASE,
    REACH_OUT,
    TOP_DOWN,
    Approach,
    ApproachFit,
    _Choices,
)
from mrcf.policies.base import MISS_CAUSES, Diagnosis, Measured, Outcome, Target, Vision
from mrcf.policies.parallel_jaw.height_looks import SEEING_IOU, HeightLookMixin, HeightLooks
from mrcf.policies.parallel_jaw.sighting import _in_arm_frame
from mrcf.policies.parallel_jaw.spec import GraspPlan

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw
    from mrcf.robot_env import RobotEnv

# A close that found nothing, or slipped, may have pushed the object: from then
# on where the scan saw it is stale, and an attempt has to look again.
_MAY_HAVE_MOVED_IT = ("closed_on_nothing", "too_wide", "grip_lost")
# A descent re-chosen to lean further than this from how the gripper is held
# turns there first (spec.descent_turn_first).
TURN_FIRST_DEG = 3.0


def _angle_deg(a, b) -> float:
    """The angle between two rotations, in degrees."""
    c = (float(np.trace(np.asarray(a, float).T @ np.asarray(b, float))) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

@dataclass
class PickContext:
    """
    One pick under way, as its phases and aligners (mrcf/policies/aligners.py) share
    it: what every attempt works from, then what the attempt under way has
    worked out so far. World frame, metres.
    """
    target: Target
    obstacles: list                       # the object's neighbours, and the arm's own base
    vision: Vision
    room: float                           # the widest the neighbours let the jaws open
    plan: GraspPlan                       # re-planned once the object has been measured
    object_xy: np.ndarray                 # where the scan saw it, moved by the fixed camera's corrections
    servo_ready: bool                     # the wrist servo can run here (and no located camera aims instead)
    room_unsettled: "str | None" = None   # the scan's feet leave too little room, but a look from
                                          # over it may find more: what they said, until it has looked
    turn_deg: float = 0.0                 # the jaws turned about the approach, where the neighbours
                                          # leave more room so (_turned_for_room)
    widths: "dict | None" = None          # its shape up its height, once measured
    measured: "Measured | None" = None    # what the wrist camera measured of it
    missed: bool = False                  # a close may have moved it (_MAY_HAVE_MOVED_IT)
    doubted: bool = False                 # the wrist camera and where it was thought to be disagree
    disturbed: bool = False               # the sentry saw something move near it on the way (_descend)
    refound: bool = False                 # found again by a scan this pick: that decides from now on
    keep_approach: "str | None" = None    # found again: the approach before, which reached it
    aim_steps: int = 0                    # misses with the jaws on their aim (spec.miss_steps_along_m)
    jaw_along_m: "float | None" = None    # how far out along the arm the wrist camera's jaw model would aim
    grip_at_m: "float | None" = None      # grip at this height (Target.grip_at_m, or lowered after a miss)
    diagnosis: "Diagnosis | None" = None  # why the last close found nothing (Vision.diagnose)
    choices: "_Choices | None" = None     # the approaches, in the order attempts take them
    height: "HeightLooks | None" = None   # what the looks for its height saw (height_looks.py): looked
                                          # at once more on the way down (_approach_look)
    # ── The attempt under way ──────────────────────────────────────────────────
    tag: str = ""
    fit: "ApproachFit | None" = None
    gp: "GraspPlan | None" = None         # the plan as this attempt's approach grasps it
    lift_h: float = 0.0                   # the highest its lift was found to reach
    rot: "np.ndarray | None" = None       # the orientation held (None: each move chooses)
    tilt: "float | None" = None
    hover: "np.ndarray | None" = None     # the waypoints, with their tolerances
    hover_tol: float = 0.0
    grasp: "np.ndarray | None" = None
    grasp_tol: float = 0.0
    grasp_step: "float | None" = None
    lift: "np.ndarray | None" = None
    lift_tol: float = 0.0
    descent_from: "np.ndarray | None" = None   # where the gripper was when the descent set out
    target_xy: "np.ndarray | None" = None  # where the jaws close: the scan's, or what an aligner found
    aligned_by: "str | None" = None       # the aligner that decided it
    located: "Measured | None" = None     # a located wrist camera's look at the hover
    far: "Measured | None" = None         # a coarse look from above the servo's height
    # ── Where the time goes (said when the pick ends) ──────────────────────────
    started: float = field(default_factory=time.monotonic)
    last_mark: float = field(default_factory=time.monotonic)
    # Since when what moved near the object counts (_descend): the pick's
    # start, then each time it was looked for again.
    last_look: float = field(default_factory=time.monotonic)
    phases: list = field(default_factory=list)

    @property
    def approach(self) -> Approach:
        assert self.fit is not None         # set by _start_attempt
        return self.fit.approach

    @property
    def inward(self) -> bool:
        return self.approach.reach_margin == "in"

    def mark(self, name: str) -> None:
        now = time.monotonic()
        self.phases.append(f"{name} {now - self.last_mark:.1f}")
        self.last_mark = now


class PickMixin(HeightLookMixin):
    """ParallelJaw's pick (see the package docstring)."""

    def pick(self, env, target: Target, obstacles=(), vision: "Vision | None" = None) -> Outcome:
        """
        Pick the object at target, checking the room obstacles leave first.
        Up to spec.retries attempts, each: hover -> align -> measure -> descend
        -> close -> lift (_hover, _align, _measure, _descend, _close, _lift),
        verifying the grip after the close and after the lift. A phase that
        cannot go on ends the attempt with Retry, under a short code kept in
        Outcome.attempts; GiveUp ends the pick.

        How it comes at the object depends on where it stands (see Approach):
        the approaches for its zone first, each worked out against the arm's
        reach before anything moves, and the first that can be done is used. A
        retry tries the next one that can, then the first again. When none
        can, the pick is refused without moving, rather than going ahead with
        each move choosing its own orientation and failing partway. How the
        jaws are lined up over the object is the approach's list of aligners
        (Approach.align, mrcf/policies/aligners.py).

        The plan only has to get the arm to a hover that clears the object:
        once over it, the wrist camera measures it, and the grasp height, grip
        and opening come from that instead.
        """
        s = self.spec
        vision = vision or Vision()
        self._begin_log()
        unchecked = self.unchecked_refusal(env)
        if unchecked is not None:
            return self._done(unchecked)
        if s.lying_pick and target.outlines and target.grip_at_m is None and self.height_unknown(target):
            # A long thing lying along the table, its height not known: gripped across where it holds best,
            # from no higher than it is wide there (policies/lying_pick.py). None: not one, nothing moved.
            from mrcf.policies.lying_pick import lying_pick
            along = lying_pick(cast("ParallelJaw", self), env, target, self._with_base(env, obstacles))
            if along is not None:
                self._attempts = list(along.attempts or [])
                return self._done(along)
        looks = None
        if target.grip_at_m is not None and self.height_unknown(target):
            # A grip that held it before (a deployment's catalogue): planned by that, with no looks
            # for its height first. 2026-11-18: the same four objects looked at from the side,
            # 25-48 s each, about ten times in a day, a different answer each time.
            self._say("[Height]", f"Gripped as before, {target.grip_at_m * 1000:.0f} mm up"
                      + (f", the hover clear of {target.max_height_m * 1000:.0f} mm" if target.max_height_m else "")
                      + " — no looks for its height.")
        elif s.height_from_views and self.height_unknown(target) and target.outlines:
            # How tall it is, looked for before anything is planned by it: planned for
            # the tallest it could be, the jaws close in the air over a short object
            # (2026-11-16, the SO-101 tabletop: 117 mm up, over a 25 mm ball). For a
            # target the scan outlined from above (scene_scan gives each one), which
            # the fit starts from — not chess's, found on the board's squares.
            low = self.low_from_above(env, target) if s.grip_low_from_above else None
            if low is not None:
                target, why = low, ""
            else:
                target, looks, why = self.survey_height(env, target, obstacles)
            if why:
                self._say("[Height]", f"Could not tell how tall it is — {why}.")
                return self._done(Outcome(False, f"could not tell how tall it is — {why}", refused=True))
        elif s.approach_look and target.measured and target.outlines:
            # Its height remembered, not looked at this pick: what the scan's outline and
            # the look from close above make of it checks that on the way down.
            looks = self._remembered_height(env, target, obstacles)
        ctx = self._begin_pick(env, target, obstacles, vision, fresh=False)
        measured = self.height_measured(looks) if looks is not None and looks.wrist_looks() else None
        if isinstance(ctx, Outcome):
            ctx.measured = ctx.measured or measured
            return ctx
        ctx.height, ctx.measured = looks, measured
        ctx.grip_at_m = target.grip_at_m
        for attempt in range(s.retries):
            try:
                self._start_attempt(env, ctx, attempt)
                self._hover(env, ctx)
                self._align(env, ctx)
                ctx.mark(ctx.aligned_by or "align")
                self._measure(env, ctx)
                self._settle_room(env, ctx)
                self._descend(env, ctx)
                self._close(env, ctx)
                return self._lift(env, ctx)
            except Retry as retry:
                # Why this attempt is being retried: said, and kept as a short
                # code in Outcome.attempts, so the log has more than free text to go on.
                self._say(ctx.tag, retry.message)
                self._attempts[-1]["result"] = retry.code
                ctx.missed = ctx.missed or retry.code in _MAY_HAVE_MOVED_IT
                if retry.code in ("closed_on_nothing", "grip_lost"):
                    lying = self._diagnose(env, ctx)
                    if lying:
                        gave_up = lying
                        break
            except GiveUp as stop:
                self._say(ctx.tag, str(stop))
                if isinstance(stop, Refuse):
                    self._back_off(env, ctx.plan, "[Pick]", reached=ctx.hover)
                    self._say_timing(ctx)
                    return self._done(Outcome(False, str(stop), refused=True, measured=ctx.measured,
                                              grasp=ctx.plan))
                gave_up = str(stop)
                break
        else:
            gave_up = None
        self._back_off(env, ctx.plan, "[Pick]", reached=ctx.hover)
        self._say_timing(ctx)
        return self._done(Outcome(False, gave_up or f"pick failed after {s.retries} attempts",
                                  measured=ctx.measured, grasp=ctx.gp or ctx.plan, diagnosis=ctx.diagnosis))

    def unchecked_refusal(self, env) -> "Outcome | None":
        """
        A refusal, nothing moved, while a fact of the arm's the pick turns the
        jaws by has not been checked on the rig (env.unchecked_facts,
        mrcf/arm_facts.py); None: go on. 2026-11-20, the UR5: its jaws' way of
        opening, a guess, put them along a hammer's handle, onto it at 50 N.
        """
        from mrcf import arm_facts

        names = [n for n in env.unchecked_facts() if n in arm_facts.GRASP_FACTS]
        if not names:
            return None
        why = arm_facts.refusal(names)
        self._say("[Pick]", f"Refused: {why}")
        return Outcome(False, why, refused=True)

    def _diagnose(self, env, ctx: PickContext) -> "str | None":
        """
        Why the close just made found nothing, asked of the wrist camera's
        pictures (Vision.diagnose: the jaws open over it, then shut), and the
        next attempt changed by it: too high, gripped as low as the jaws go;
        aimed off or pushed, found again (ctx.missed); lying on its side, not
        picked as it stands — the reason returned, to give up with. None
        otherwise. 2026-11-18: three closes on nothing, each "re-aimed",
        while the pictures showed the jaws over a figure lying on its side;
        and over a plug lying on its back, gripped above its body.
        """
        s = self.spec
        if s.camera is None:
            return None
        n = len(self._attempts)
        seen = (self._attempts[-1].get("wrist_at_grasp_mm") or [None])[-1]
        if seen is not None and float(np.hypot(*seen)) <= 2 * s.wrist_at_grasp_tol_m * 1000:
            # The wrist camera saw it in the gap between the jaw tips as they closed, and they met on
            # nothing: they closed above it — whatever a model makes of the pictures. 2026-11-18, a plug
            # on its back and a figure lying down, each centred in the gap, gripped 21-28 mm up.
            got = Diagnosis("too_high", f"the wrist camera saw it between the jaws ({seen[0]:+.1f}, {seen[1]:+.1f} mm "
                                        "from their gap) as they closed on nothing")
        elif ctx.vision.diagnose is None:
            return None
        else:
            frames = [(label, f) for label, f in self._views if label == f"try{n}_at_grasp"]
            try:
                frames.append((f"try{n}_shut", env.render_image(s.camera, width=s.render_w, height=s.render_h)))
                got = ctx.vision.diagnose(frames, None)
            except Exception as exc:                  # noqa: BLE001 — a diagnosis is a help, never a stop
                self._say(ctx.tag, f"Could not ask why ({exc}).")
                return None
        if got is None or got.cause not in MISS_CAUSES:
            return None
        ctx.diagnosis = got
        self._attempts[-1]["diagnosis"] = got.cause
        self._say(ctx.tag, f"Why: {got.said or got.cause} — {MISS_CAUSES[got.cause]}.")
        if got.cause == "too_high":
            # The jaws met above it: it was not touched, so not moved — no looking for it afresh
            # (2026-11-18: a white plug, re-found by the overhead camera, which cannot see it, and given up).
            # Unless spec.relook_after_miss: 2026-11-19 14:03, a figure "seen between the jaws" as they
            # closed on nothing had been tipped by them; the next try came down on it again blind, and it
            # rolled off the table.
            ctx.missed = bool(s.relook_after_miss)
            here = (ctx.gp.grasp_height_m if ctx.gp is not None else None)
            if here is not None and here > s.min_grasp_height_m + 0.003:
                ctx.grip_at_m = s.min_grasp_height_m
            else:
                self._say(ctx.tag, "Already as low as the jaws go.")
        elif got.cause in ("aim_off", "moved"):
            ctx.missed = True
        elif got.cause == "lying":
            return f"it is lying on its side ({got.said or 'the wrist camera'}): not picked as it stands"
        return None

    def _gripped_at(self, ctx: PickContext, p: "GraspPlan | None") -> "GraspPlan | None":
        """p closed at ctx.grip_at_m (a recipe's, or lowered after a miss): no lower than the jaws go, under its top."""
        g = ctx.grip_at_m
        if g is None or p is None:
            return p
        s = self.spec
        g = max(s.min_grasp_height_m, float(g))
        g = min(g, max(s.min_grasp_height_m, float(p.height_m) - s.grip_over_top_m))
        q = p if abs(g - p.grasp_height_m) < 1e-4 else dataclasses.replace(p, grasp_height_m=g)
        if s.open_wide_when_low and g <= s.min_grasp_height_m + 1e-4:
            # As low as the jaws go: round a wide flat body (a plug on its back, 55 x 47 mm, its fingers come
            # down on it at 60 mm open — 2026-11-18), as wide as the neighbours let them.
            wide = min(s.max_open_m, max(float(ctx.room), q.pregrasp_open_m))
            if wide > q.pregrasp_open_m + 1e-4:
                q = dataclasses.replace(q, pregrasp_open_m=wide)
        return q

    # ── Looking from where a pick looks, without picking ────────────────────────

    def go_over(self, env, target: Target, obstacles=(), vision: "Vision | None" = None) -> "PickContext | Outcome":
        """
        As a pick's first attempt, as far as its hover: the approach chosen,
        the jaws opened, over the object as the fixed camera sees it. Nothing
        is aligned or closed. The PickContext, or the Outcome that refused or
        stopped it. For tools that look from where a pick looks (look_tour.py).
        """
        ctx = self._begin_pick(env, target, obstacles, vision or Vision())
        if isinstance(ctx, Outcome):
            return ctx
        try:
            self._start_attempt(env, ctx, 0)
            self._hover(env, ctx)
        except (Retry, GiveUp) as stop:
            return self._done(Outcome(False, str(stop), grasp=ctx.plan))
        return ctx

    def look_from(self, env, ctx: PickContext, height_m: float) -> bool:
        """
        From go_over's hover, down to height_m above the surface over the
        object, holding what reaches there — as the wrist servo comes down to
        its own height (aligners.WristServo). Whether it got there.
        """
        assert ctx.gp is not None               # set by go_over
        self._set_waypoints(ctx, ctx.object_xy, ctx.gp, self._lift_for(ctx.approach, ctx.gp, ctx.lift_h),
                            hover_height=height_m)
        ctx.rot, ctx.tilt, checked = self._hold(env, self._pick_reach_of(env, ctx), ctx.tag, ctx.obstacles,
                                                ctx.target.tilt_deg, ctx.approach.lean_margin_deg,
                                                turn_deg=ctx.turn_deg, open_m=ctx.gp.pregrasp_open_m)
        if checked and ctx.rot is None:
            return False
        return bool(env.move_ee_to(ctx.hover, tol=ctx.hover_tol, orientation=ctx.rot)
                    and self._near(env, ctx.hover, self.spec.approach_tol_m * 2))

    def known_around(self, ctx: PickContext) -> list:
        """
        The object, keyed "target", and what stands around it, as the wrist
        camera should find them (expected_view.KnownObject), where the scan
        measured them. Each as tall as it is known to be, else as the tallest
        it may be — a neighbour drawn too tall stands in front of more than it
        does, and the target is then left unmeasured rather than mistaken.
        """
        from mrcf.perception.expected_view import KnownObject
        from mrcf.policies.parallel_jaw.reach import BASE_LABEL

        s, t = self.spec, ctx.target
        tallest = t.max_height_m or s.default_max_height_m
        out = [KnownObject("target", np.asarray(t.position[:2], float), (t.foot_m or 2 * s.unmeasured_obstacle_radius_m) / 2,
                           t.height_m or tallest)]
        for i, ob in enumerate(ctx.obstacles):
            if ob.label == BASE_LABEL:
                continue
            out.append(KnownObject(f"{i}:{ob.label}", np.asarray(ob.position[:2], float),
                                   ob.radius_m or s.unmeasured_obstacle_radius_m, ob.height_m or tallest))
        return out

    def _begin_log(self) -> None:
        """A pick's notes, views and attempts, begun afresh (what its Outcome carries: _done)."""
        self._notes, self._views, self._attempts, self._around = [], [], [], {}
        self._approach_name = None
        self._held_tilt_deg = None
        self._height_looks = None

    def _begin_pick(self, env, target: Target, obstacles, vision: Vision,
                    fresh: bool = True) -> "PickContext | Outcome":
        """
        The pick's plan and approaches, before anything moves — or the Outcome
        refusing it. fresh: its notes begun here (_begin_log); False, pick()
        has begun them, and what it looked at on the way is kept.
        """
        if fresh:
            self._begin_log()
        obstacles = self._with_base(env, obstacles)
        # Flat sides (a block): the jaws across a pair of them, the least turn first.
        faces = self.face_turns(env, target.position[:2], target.faces_deg)
        # Longer one way than the other (lying on its side): across it, and only so.
        long = self.long_way_deg(env, target) if faces is None and self.spec.close_across_long else None
        if long is not None:
            faces = [self.across_turn(env, target.position[:2], long)]
        turn = faces[0] if faces else 0.0
        plan, problem, room = self.fit(env, self.plan(target), target.position[:2], obstacles, turn)
        if long is not None:
            self._say("", f"It lies longest {long:.0f} deg from x: the jaws turned {turn:+.0f} deg to close across it.")
        elif faces and turn:
            self._say("", f"Its sides face {target.faces_deg:.0f} deg: the jaws turned {turn:+.0f} deg to close across them.")
        if problem:
            turned = self._turned_for_room(env, target, obstacles, faces[1:] if faces else None)
            if turned is not None:
                turn, plan, room = turned
                self._say("", f"{problem[0].upper()}{problem[1:]} — turning the jaws {turn:+.0f} deg about the "
                              f"approach, where the neighbours leave {room * 1000:.0f} mm.")
                problem = None
        if problem:
            if not self._room_by_looking(env, vision, target, plan, obstacles):
                return self._done(Outcome(False, problem, refused=True, grasp=plan,
                                          blocked_by=self._blockers(env, target, plan, obstacles)))
            self._say("", f"By the feet the scan saw, the {problem} — going over it to measure "
                          "where the fingers pass before deciding.")
        self._say("", f"Height {plan.height_m * 1000:.0f} mm "
                      f"({'measured' if target.measured else 'not yet measured'}), "
                      f"jaws {(plan.pregrasp_open_m if not problem else min(plan.pregrasp_open_m, max(room, 0.0))) * 1000:.0f} mm.")
        # Where the wrist servo would run: a located camera aims by the foot it
        # measures instead (FootAligner), whatever the approach.
        ctx = PickContext(target=target, obstacles=obstacles, vision=vision, room=room, plan=plan,
                          room_unsettled=problem, turn_deg=turn,
                          object_xy=np.asarray(target.position[:2], dtype=float),
                          servo_ready=self._servo_ready(env, vision) and not self.camera_located(env),
                          widths=target.widths_mm if target.measured else None)
        ctx.choices = self._choices_for(env, ctx, plan)
        if ctx.choices.feasible(0) is None:
            return self._done(Outcome(
                False, "out of reach here for every approach — " + "; ".join(ctx.choices.why_not),
                refused=True, grasp=plan))
        zone = self.zone(env, ctx.object_xy)
        if zone == "mid" and self._leans_far(env, ctx.object_xy, plan):
            zone = "far"
        where = {"near": "close to the arm's base", "mid": "mid-reach", "far": "far out"}[zone]
        self._say("", f"Approach: {cast(ApproachFit, ctx.choices.feasible(0)).approach.name} ({where}).")  # cached, checked above
        # Where the time goes, said when the pick ends: "make it faster" starts
        # from knowing which phase is slow, not from guessing.
        ctx.started = ctx.last_mark = ctx.last_look = time.monotonic()
        return ctx

    def pick_check(self, env, target: Target, obstacles=(), vision: "Vision | None" = None,
                   quick: bool = False) -> Outcome:
        """
        Whether pick() would go ahead (Policies.pick_check): its start —
        sizing, the room the obstacles leave, the jaw turns that make more,
        an approach that reaches — with nothing moved or said. quick: the
        room alone, turns included, no reach worked out (that is the slow
        part). What make-room recovery weighs each rearrangement by.
        """
        from mrcf.capabilities import OrientationSearch
        from mrcf.policies.parallel_jaw.planning import _takes

        saved = (self._notes, self._views, self._attempts, self._around, self._approach_name,
                 self._held_tilt_deg)
        self._quiet = True
        try:
            if quick:
                obs = self._with_base(env, obstacles)
                first = self.plan(target)
                xy = np.asarray(target.position[:2], float)
                turns = [0.0]
                if isinstance(env, OrientationSearch) and _takes(env.approach_orientation_for, "turn_deg"):
                    turns += list(self.spec.jaw_turns_deg)
                refusal = None
                for turn in turns:
                    plan, problem, room = self.fit(env, first, xy, obs, turn)
                    if not problem:
                        return Outcome(True, f"the jaws have {room * 1000:.0f} mm here, turned {turn:+.0f} deg",
                                       grasp=plan)
                    refusal = refusal or Outcome(False, problem, refused=True, grasp=plan,
                                                 blocked_by=self._blockers(env, target, plan, obs))
                return cast(Outcome, refusal)
            ctx = self._begin_pick(env, target, obstacles, vision or Vision())
            if isinstance(ctx, Outcome):
                return ctx
            fit = ctx.choices.feasible(0) if ctx.choices is not None else None
            name = fit.approach.name if fit is not None else "an approach"
            turned = f", the jaws turned {ctx.turn_deg:+.0f} deg" if ctx.turn_deg else ""
            return Outcome(True, f"would pick it by {name}{turned}", grasp=ctx.plan, approach=name)
        finally:
            self._quiet = False
            (self._notes, self._views, self._attempts, self._around, self._approach_name,
             self._held_tilt_deg) = saved

    def _blockers(self, env, target: Target, plan: GraspPlan, obstacles) -> list:
        """The labels of the obstacles that each, alone, leave the jaws too little room at target (not the arm's base)."""
        from mrcf.policies.parallel_jaw.reach import BASE_LABEL

        s = self.spec
        xy = np.asarray(target.position[:2], float)
        tips = plan.grasp_height_m - s.tip_below_grasp_m
        need = min(plan.pregrasp_open_m, plan.width_m + 2 * s.jaw_slack_m)
        return [ob.label for ob in obstacles
                if ob.label and ob.label != BASE_LABEL and self.jaw_room(env, xy, [ob], tips)[0] < need]

    def _turned_for_room(self, env, target: Target, obstacles,
                         turns: "list | None" = None) -> "tuple[float, GraspPlan, float] | None":
        """
        Where the neighbours leave the jaws too little room as the arm opens
        them, the first of spec.jaw_turns_deg about the approach where they
        leave enough and some approach reaches: (turn_deg, plan, room_m), or
        None (none does, or the backend cannot turn them).

        Off the arm's straight-ahead line the jaws open at a slant to a row
        of objects, and the neighbours stand diagonally to them. On the full
        chess board (2026-11-03) the f1 bishop, 21 deg round, left 28 mm
        between the e1 king and the g1 knight against 31 needed; turned 15 deg,
        no neighbour stood in the fingers' way.
        """
        from mrcf.capabilities import OrientationSearch
        from mrcf.policies.parallel_jaw.planning import _takes

        if not isinstance(env, OrientationSearch) or not _takes(env.approach_orientation_for, "turn_deg"):
            return None
        xy = np.asarray(target.position[:2], float)
        first = self.plan(target)
        for turn in (self.spec.jaw_turns_deg if turns is None else turns):   # turns: a block's other faces
            plan, problem, room = self.fit(env, first, xy, obstacles, turn)
            if problem:
                continue
            fit, _ = self.choose_approach(env, xy, plan, obstacles, target.tilt_deg, room=room, turn_deg=turn)
            if fit is not None:
                return float(turn), plan, room
        return None

    def _room_by_looking(self, env, vision: Vision, target: Target, plan: GraspPlan, obstacles) -> bool:
        """
        Whether a pick the scan's feet crowd may find room once over it: the
        wrist camera measures the object, and the neighbours where it is
        located (measure_neighbours), and either can be slimmer where the
        fingers pass than at its foot. It may if they would leave room were
        each spec.slimmest_above_foot of its foot there. Not where they could
        not even then: feet that leave a finger no gap.

        2026-11-03, the full chess board: the d1 queen was refused on the
        c1 bishop's foot; from 34 mm up, where the finger tips come down to,
        a measured queen needs 31 mm and a bishop leaves 33.
        """
        from mrcf.policies.parallel_jaw.reach import BASE_LABEL

        s = self.spec
        if not s.measure_at_hover or s.camera is None:
            return False
        k = s.slimmest_above_foot
        around = self.can_measure_around(env, vision)
        slim = [dataclasses.replace(ob, radius_m=k * (ob.radius_m if ob.radius_m is not None
                                                      else s.unmeasured_obstacle_radius_m))
                if around and not ob.widths_mm and ob.label != BASE_LABEL else ob for ob in obstacles]
        width = plan.width_m if target.measured else k * plan.width_m
        room, _ = self.jaw_room(env, target.position[:2], slim, plan.grasp_height_m - s.tip_below_grasp_m)
        return room >= width + 2 * s.jaw_slack_m

    def _choices_for(self, env, ctx: PickContext, plan: GraspPlan) -> _Choices:
        """
        The approaches for where the object is, in order. Worked out where it
        is now and as it is now measured: the fixed camera's correction and a
        measurement both change them.
        """
        return _Choices(
            self.approach_order(env, ctx.object_xy, plan), plan,
            lambda a, p: self.fit_approach(env, a, ctx.object_xy, p, ctx.obstacles, ctx.target.tilt_deg,
                                           ctx.servo_ready, widths_mm=ctx.widths, measured=ctx.widths is not None,
                                           room=ctx.room, turn_deg=ctx.turn_deg))

    def _say_timing(self, ctx: PickContext) -> None:
        self._say("[Pick]", f"{time.monotonic() - ctx.started:.1f} s — {', '.join(ctx.phases)}")

    # ── The phases of an attempt ────────────────────────────────────────────────

    def _start_attempt(self, env, ctx: PickContext, attempt: int) -> None:
        """Choose this attempt's approach (backing off from the last one first)."""
        s = self.spec
        ctx.tag = f"[Pick {attempt + 1}/{s.retries}]"
        if attempt:
            ctx.mark(f"#{attempt + 1}")
            self._back_off(env, ctx.plan, ctx.tag, reached=ctx.hover)
            assert ctx.choices is not None          # set by _begin_pick
            if (ctx.missed or ctx.doubted) and ctx.vision.relocate is not None:
                self._found_again(env, ctx)
            elif ctx.disturbed and not (ctx.servo_ready or self.camera_located(env)):
                raise GiveUp("Something moved near it on the way, and nothing here can look for it again — "
                             "not coming down where it was. Scan again before another try.")
            elif (ctx.missed and not (ctx.servo_ready or self.camera_located(env))
                  and not s.blind_retry_after_miss):
                # Nothing here can look for it again — no wrist camera to servo or to
                # find its foot with, no way to find it afresh — and a close taken for
                # a miss may have moved it (HISTORY.md: the UR5's e-stop, 2026-11-12).
                raise GiveUp("The last close may have moved it, and nothing here can look for it again — "
                             "not coming down where it was. Scan again before another try.")
            elif ctx.choices.plan is not ctx.plan:
                ctx.choices.reset(ctx.plan)          # measured since: worked out again for its real size
        assert ctx.choices is not None
        fit = ctx.choices.for_attempt(attempt)
        if fit is None:
            raise GiveUp("As measured, it is out of reach here for every approach — " + "; ".join(ctx.choices.why_not))
        if ctx.keep_approach is not None:
            # Found again: it was the aim that was off, not the approach, and
            # the grasp correction is measured for the one that got there —
            # 2026-10-31, d3: the next one along, near_base, missed at the
            # position found afresh.
            same = ctx.choices.named(ctx.keep_approach)
            if same is not None and same is not fit:
                self._say(ctx.tag, f"Again by {same.approach.name}: it reached it; the aim was off.")
                fit = same
            ctx.keep_approach = None
        if ctx.missed and ctx.servo_ready and not fit.servo:
            # After a miss only an approach whose wrist camera looks again will
            # do: one that does not comes down on the spot the miss may have
            # pushed the object off. One that looks, if any can be done here;
            # else stop rather than push it further.
            k, looks = 0, None
            while looks is None and (f := ctx.choices.feasible(k)) is not None:
                looks, k = (f if f.servo else None), k + 1
            if looks is None:
                raise GiveUp("The last close may have moved it, and no approach here looks with the wrist camera — "
                             "not coming down where it was. Scan again before another try.")
            self._say(ctx.tag, f"The last close may have moved it: {looks.approach.name}, which looks for it "
                               "with the wrist camera.")
            fit = looks
        if attempt and fit.approach.name != self._approach_name:
            self._say(ctx.tag, f"This attempt: {fit.approach.name}.")
        self._approach_name = fit.approach.name
        self._attempts.append({"attempt": attempt + 1, "approach": fit.approach.name,
                               "tilt_deg": None, "result": None})
        ctx.fit, ctx.gp, ctx.lift_h = fit, fit.plan, fit.lift_height_m
        if ctx.grip_at_m is not None:
            # Closed where it held before, or lower after a miss — still over what it must pass over.
            ctx.gp = self.gripped_above(self._gripped_at(ctx, ctx.gp) or ctx.gp, fit.grasp_over_m)
        self._attempts[-1]["grip_m"] = round(float(ctx.gp.grasp_height_m), 4)
        ctx.located = ctx.far = None

    def _found_again(self, env, ctx: PickContext) -> None:
        """
        After a close that missed, or a wrist camera that saw the object
        elsewhere and could not settle: the object found again where it stands
        now (Vision.relocate), and the pick aimed there afresh, as its first
        attempt was at the scan's. A miss pushes a free-standing object: on
        the SO-101 (2026-10-30) 2-27 mm a time, mostly toward the base, and
        retries that looked for it from the old spot with the wrist camera
        chased it 42 mm and lost it.
        """
        ctx.last_look = time.monotonic()           # what moved before this look, it sees
        assert ctx.vision.relocate is not None
        was = np.asarray(ctx.target.position[:2], float)
        now = ctx.vision.relocate(env, was)
        ctx.mark("relocate")
        if now is None:
            if ctx.diagnosis is not None and ctx.diagnosis.cause == "too_high" and not ctx.disturbed:
                # The wrist camera saw it between the jaws as they closed above it; from above it is not
                # found (2026-11-19 15:29, a white plug on the white table): lower, where it was.
                self._say(ctx.tag, "Not found again from above, but the wrist camera had it between the jaws: "
                                   "lower, where it was.")
                ctx.missed = False
                return
            after = "Since something moved by it" if ctx.disturbed and not ctx.missed else "After the miss"
            raise GiveUp(f"{after} it cannot be found near where it was — it may have been knocked away. "
                         "Scan again before another try.")
        now = np.asarray(now, float)[:2]
        out = was - self._base_xy(env)
        out = out / (np.linalg.norm(out) or 1.0)
        moved = now - was
        self._say(ctx.tag, f"Found it again {np.linalg.norm(moved) * 1000:.0f} mm from where it was "
                           f"({float(moved @ out) * 1000:+.0f} along the arm, "
                           f"{float(moved @ np.array([-out[1], out[0]])) * 1000:+.0f} across).")
        unmoved = np.linalg.norm(moved) <= self.spec.unmoved_m
        jaw = ctx.jaw_along_m
        along, across = float(moved @ out), float(moved @ np.array([-out[1], out[0]]))
        # Pushed sideways, along the jaws' line, and not toward the base: the closing jaw caught
        # its edge — the jaws were beside it, not past it (GripperSpec.step_after_sideways_push).
        sideways = (self.spec.step_after_sideways_push and abs(across) >= 2.0 * abs(along) and along > -0.002
                    and np.linalg.norm(moved) <= self.spec.max_aim_shift_m)
        # Only after a close: found again because something moved by it on the way
        # (disturbed), the jaws never closed, and its being where it was says
        # nothing about the aim.
        if ctx.missed and self.spec.miss_steps_along_m and (unmoved or sideways
                                                            or (jaw is not None and 0.002 <= jaw <= 0.012)):
            # The jaws closed beside it, not on it: aiming at it again the same
            # way misses the same way (2026-11-04, e4, twice three closes on
            # nothing, the jaws short toward the base as the person watching
            # saw). Only with evidence they closed short: it had not moved, or
            # the wrist camera's jaw model put them short of it. A miss that
            # pushed it, with no such look, says nothing about which way —
            # stepped out blind, two retries on e2 came down 6-8 mm beyond a
            # pawn the first miss had pushed toward the base.
            ctx.aim_steps += 1
            step = self._aim_step(ctx)
            self._say(ctx.tag, f"{'It had not moved' if unmoved else 'Pushed sideways' if sideways else 'Found'}: "
                               f"the aim was off — this attempt aims {step * 1000:+.0f} mm along the arm from it"
                               + (" (where the wrist camera's jaw model puts the jaws)" if ctx.jaw_along_m else "")
                               + ".")
        ctx.target = dataclasses.replace(ctx.target, position=np.r_[now, np.asarray(ctx.target.position)[2:]],
                                         outlines=None)
        ctx.object_xy = now.copy()
        ctx.height = None                        # its outlines were of where it stood: the height stands
        ctx.missed = ctx.doubted = ctx.disturbed = False        # found where it is: any approach will do...
        ctx.refound = True
        ctx.keep_approach = self._approach_name  # ... the one that reached it first
        ctx.choices = self._choices_for(env, ctx, ctx.plan)

    def _aim_step(self, ctx: PickContext) -> float:
        """
        How far out along the arm a retry after ctx.aim_steps misses aims:
        where the wrist camera's jaw model put the jaws short of the object,
        when it said so within reason (2-12 mm out), else
        spec.miss_steps_along_m's step; 4 mm more for each miss after the first.
        """
        s = self.spec
        jaw = ctx.jaw_along_m
        if jaw is None or not 0.002 <= jaw <= 0.012:
            return s.miss_steps_along_m[min(ctx.aim_steps, len(s.miss_steps_along_m)) - 1]
        return jaw + 0.004 * (ctx.aim_steps - 1)

    def _hover(self, env, ctx: PickContext) -> None:
        """Open the jaws, choose the orientation to hold, and go over the object as the fixed camera sees it."""
        s = self.spec
        assert ctx.gp is not None               # set by _start_attempt
        opening = ctx.gp.pregrasp_open_m
        if ctx.room_unsettled is not None:
            opening = min(opening, max(ctx.room, 0.0))   # no wider than the feet allow, until it has looked
        env.set_gripper(opening, n_steps=15)
        self._set_waypoints(ctx, ctx.object_xy, ctx.gp, ctx.lift_h)
        ctx.rot, ctx.tilt, checked = self._hold(env, self._pick_reach_of(env, ctx), ctx.tag, ctx.obstacles,
                                                ctx.target.tilt_deg, ctx.approach.lean_margin_deg,
                                                turn_deg=ctx.turn_deg, open_m=ctx.gp.pregrasp_open_m)
        if checked and ctx.rot is None:
            raise Retry("no_orientation", "No single orientation reaches every phase from where the arm is "
                                          "now — retrying")
        self._attempts[-1]["tilt_deg"] = None if ctx.tilt is None else round(ctx.tilt, 1)

        # Take what move_ee_to reports, not just a proximity test: its
        # tolerance (2 x approach_tol_m = 3 cm) is looser than a hover that
        # stalls 1-2 cm out, and descending from 1.3 cm off puts a jaw on top
        # of the object instead of beside it.
        if not self._over_walls(env, ctx.hover, ctx.tag, ctx.rot):
            raise Refuse("could not get up over a container's walls to cross them")
        refusals = env.move_refusals if isinstance(env, MoveRefusals) else 0
        arrived = self._move_again_if_short(env, ctx.hover, ctx.hover_tol, ctx.rot,
                                            env.move_ee_to(ctx.hover, tol=ctx.hover_tol, orientation=ctx.rot))
        ctx.mark("hover")
        assert ctx.hover is not None            # set by _set_waypoints
        if isinstance(env, MoveRefusals) and env.move_refusals > refusals \
                and not self._near(env, ctx.hover, s.approach_tol_m * 2):
            # Not a move that fell short: one the arm would not make (its box).
            # Another try is the same move, and a plan made again the same pick.
            raise Refuse(f"the arm will not go there: {env.last_refusal or 'it refused the move'}")
        if not self._near(env, ctx.hover, s.approach_tol_m * 2):
            dist = float(np.linalg.norm(env.get_ee_pos() - ctx.hover))
            raise Retry("approach_short", f"Approach did not converge (residual {dist * 100:.1f} cm) — retrying")
        # Over the object as the fixed camera sees it — the scan saw the
        # object through the same camera. Everything after aims in the
        # kinematics' frame, so the object's position moves with the correction.
        corr = self._overhead_correct(env, ctx.hover, ctx.hover_tol, ctx.rot, ctx.tag)
        if self._overhead_first_err_m is not None:
            self._attempts[-1]["overhead_err_mm"] = round(self._overhead_first_err_m * 1000, 1)
        if not arrived and self._overhead_err_m is None:
            # A hover that stopped short is fine once the camera has put the
            # jaws over the object — near the wrist's limit the servos stop
            # 1-2 cm short of where they are sent. Without that sighting, it is
            # not: a stalled hover passed on proximity alone can put a jaw on
            # the object.
            dist = float(np.linalg.norm(env.get_ee_pos() - ctx.hover))
            raise Retry("approach_unconfirmed", f"Approach stopped {dist * 100:.1f} cm short and the camera could "
                                                "not confirm where the jaws are — retrying")
        if np.any(corr):
            ctx.object_xy = ctx.object_xy + corr
            self._set_waypoints(ctx, ctx.object_xy, ctx.gp, ctx.lift_h)

    def _align(self, env, ctx: PickContext) -> None:
        """
        Where the jaws close: the first of the approach's aligners that
        applies decides (mrcf/policies/aligners.py) — by default the scan's
        estimate, as the fixed camera has just put the jaws over it.
        """
        ctx.target_xy = ctx.object_xy.copy()
        ctx.aligned_by = None
        for name in ctx.approach.align:
            aligner = ALIGNERS.get(name)
            if aligner is None:
                raise ValueError(f"{ctx.approach.name}: no aligner '{name}' (known: {', '.join(ALIGNERS)})")
            if aligner.applies(self, env, ctx):
                ctx.aligned_by = name
                aligner.run(self, env, ctx)
                return

    def _measure(self, env, ctx: PickContext) -> None:
        """
        Measure the object with the wrist camera and re-plan the grasp from it.
        Not for an object already measured, unless asked or its shape rests on
        too few looks (shape_unsettled): the plan has its shape, and a look
        takes seconds (two segmented frames) to say the same again. What it
        adds is the height check against the servo having found some other
        object; for objects the scan knows about the servo's own check (nearer
        another's estimate) covers that.
        """
        s = self.spec
        unsettled = self.shape_unsettled(ctx.target)
        if not (s.measure_at_hover and ctx.measured is None
                and (s.remeasure_known or not ctx.target.measured or unsettled)):
            return
        ctx.measured = ctx.located or self.measure_at_hover(
            env, ctx.vision, ctx.tag, expected_xy=ctx.object_xy,
            known=self.known_around(ctx) if self.camera_located(env) else None)
        ctx.mark("measure")
        if ctx.measured is None and ctx.far is not None:
            ctx.measured = ctx.far
        unlike = self._unlike_scan(ctx.target, ctx.measured) if ctx.measured is not None else None
        if unlike:
            self._say(ctx.tag, f"The look makes its foot {unlike} — not trusting it; using what the scan gave.")
            ctx.measured = None
        short = self._too_short(ctx.target, ctx.measured) if ctx.measured is not None else None
        if short:
            self._say(ctx.tag, f"The look makes it {short} — not trusting it; using what the scan gave.")
            ctx.measured = None
        other = self._other_object(ctx.target, ctx.measured, ctx.obstacles) if ctx.measured is not None else None
        if other:
            ctx.measured = None
            raise Retry("other_object", f"The object under the jaws is {other} — not the one sent for; "
                                        "not descending, retrying")
        part = self._part_seen(ctx.target, ctx.measured, ctx.obstacles) if ctx.measured is not None else None
        if part:
            self._say(ctx.tag, f"The look makes it {part} — not trusting it; going by its height as known.")
            ctx.measured = None
        earlier = cast(float, ctx.target.height_m)            # unsettled: measured
        as_known = False
        if unsettled and ctx.measured is not None and abs(ctx.measured.height_mm / 1000 - earlier) > s.same_height_tol_m:
            ctx.measured = self._third_look(env, ctx, earlier, ctx.measured)
            as_known = ctx.measured is None
        if ctx.measured is not None:
            self._replan(env, ctx, ctx.measured)
        elif as_known:
            # The coarse look from higher up may have re-planned it already.
            ctx.plan = self.fit(env, self.plan(ctx.target), ctx.target.position[:2], ctx.obstacles, ctx.turn_deg)[0]
            ctx.widths = ctx.target.widths_mm
            ctx.gp = self._as_approach(ctx, ctx.approach, ctx.plan, ctx.tilt, ctx.widths)
        if ctx.measured is not None or as_known:
            assert ctx.lift is not None and ctx.grasp is not None and ctx.gp is not None
            ctx.lift = np.array([ctx.lift[0], ctx.lift[1], s.table_z + self._lift_for(ctx.approach, ctx.gp, ctx.lift_h)])
            ctx.grasp = np.array([ctx.grasp[0], ctx.grasp[1], s.table_z + ctx.gp.grasp_height_m])

    def _settle_room(self, env, ctx: PickContext) -> None:
        """
        For a pick the scan's feet crowded (PickContext.room_unsettled), the
        room the neighbours leave from where the finger tips come down to,
        now the wrist camera has measured the object: the neighbours that
        still keep the jaws from opening as wide as it needs, measured from
        here (measure_neighbours). Refused, having touched nothing, when it
        still does not fit; else the jaws planned within it. The jaws come
        down where the aim now is, so the room is measured there.
        """
        if ctx.room_unsettled is None:
            return
        from mrcf.policies.parallel_jaw.reach import BASE_LABEL

        s = self.spec
        plan = ctx.plan
        tips = plan.grasp_height_m - s.tip_below_grasp_m
        need = plan.width_m + 2 * s.jaw_slack_m
        at = np.asarray(ctx.target.position[:2], float)
        if ctx.target_xy is not None:
            at = at + (ctx.target_xy - ctx.object_xy)
        room, limit = self.jaw_room(env, at, ctx.obstacles, tips, ctx.turn_deg)

        def in_the_way(tips_m, need_m) -> dict:
            """The neighbours not yet measured that alone leave the jaws less than need_m from tips_m up."""
            return {f"{i}:{ob.label}": ob.label for i, ob in enumerate(ctx.obstacles)
                    if ob.label != BASE_LABEL and not ob.widths_mm
                    and self.jaw_room(env, at, [ob], tips_m, ctx.turn_deg)[0] < need_m}

        def keep(found: dict, where: str) -> None:
            for key, m in found.items():
                i = int(key.split(":", 1)[0])
                ob = ctx.obstacles[i] = dataclasses.replace(ctx.obstacles[i], widths_mm=m.widths_mm,
                                                            height_m=m.height_mm / 1000)
                self._around[ob.label] = m
                self._say(ctx.tag, f"Measured {ob.label} {where}: {m.height_mm:.0f} mm tall, "
                                   f"{2000 * self.half_width_at(ob, tips):.0f} mm across from {tips * 1000:.0f} mm "
                                   f"up (its foot {m.foot_mm:.0f}).")

        if room < need:
            which = in_the_way(tips, need)
            keep(self.measure_neighbours(env, ctx.vision, ctx.tag, self.known_around(ctx), which), "from over it")
            if which:
                ctx.mark("measure around")
            room, limit = self.jaw_room(env, at, ctx.obstacles, tips, ctx.turn_deg)
        if room < need and s.look_around_max > 0:
            want = {} if ctx.measured is not None or ctx.target.measured else {"target": "it"}
            want.update(in_the_way(tips, need))
            found = self._look_around(env, ctx, want)
            seen = found.pop("target", None)
            if seen is not None and self._use_measurement(env, ctx, seen):
                plan = ctx.plan
                tips = plan.grasp_height_m - s.tip_below_grasp_m
                need = plan.width_m + 2 * s.jaw_slack_m
            keep(found, "from beside it")
            room, limit = self.jaw_room(env, at, ctx.obstacles, tips, ctx.turn_deg)
        if room < need:
            raise Refuse(f"jaws need {need * 1000:.0f} mm open from {tips * 1000:.0f} mm up, where the finger "
                         f"tips come down to, but {limit} leaves {max(room, 0) * 1000:.0f} mm there, as measured "
                         "from over it")
        self._say(ctx.tag, f"Room for the jaws from {tips * 1000:.0f} mm up: {room * 1000:.0f} mm, "
                           f"{need * 1000:.0f} needed.")
        ctx.room, ctx.room_unsettled = room, None
        ctx.plan = dataclasses.replace(plan, pregrasp_open_m=min(max(plan.pregrasp_open_m, need), room))
        assert ctx.gp is not None
        ctx.gp = self._widened(dataclasses.replace(ctx.gp, pregrasp_open_m=min(max(ctx.gp.pregrasp_open_m, need),
                                                                                 room)), room)
        if env.get_gripper_width() > room + 0.001:
            env.set_gripper(ctx.gp.pregrasp_open_m, n_steps=10)

    def _look_around(self, env, ctx: PickContext, which: dict) -> dict:
        """
        What the wrist camera could not measure from over the object — which:
        {key in known_around: what to call it}, "target" for the object
        itself — measured from over the spots beside each instead:
        spec.look_aside_m either side of it, across the arm's reach, nearest
        first, then twice as far; not from over the object again; at most
        spec.look_around_max spots. Each spot is looked from as high as the
        hover, its jaw tips clear of the tallest of them. Then back over the
        object, held as it was. {key: Measured} for those measured; a frame
        from each spot goes with the pick's views.

        From straight over an object, a camera beside the jaws sees its top
        and not how tall it is, and the gripper hides what stands nearer the
        arm. From one spot across, it sees it from the side. On the full chess
        board (2026-11-03 look tour, real hover frames), pieces one square
        across from a hover, as far out, were measured in 5 of 6 views; pieces
        a rank nearer the arm in 4 of 17 (one of those a bishop read at 30 mm),
        and those straight nearer it were hidden by the gripper. Live, from
        over the d1 queen the camera could tell neither her height nor the c1
        bishop's.
        """
        s = self.spec
        if not which or not self.can_measure_around(env, ctx.vision):
            return {}
        known = self.known_around(ctx)
        feet = {k.key: np.asarray(k.foot_xy_m, float) for k in known}
        # Positions are the scan's; the arm moves by the kinematics, which the
        # fixed camera's correction at the hover moved the object by.
        moved = ctx.object_xy - np.asarray(ctx.target.position[:2], float)
        over = np.asarray(ctx.object_xy, float)
        here = env.get_ee_pos()[:2]
        ranked: list = []
        for times in (1.0, 2.0):
            for key in which:
                xy = feet[key] + moved
                out = xy - self._base_xy(env)
                out /= np.linalg.norm(out) or 1.0
                for side in (1.0, -1.0):
                    spot = xy + side * times * s.look_aside_m * np.array([-out[1], out[0]])
                    if (np.linalg.norm(spot - over) > s.look_aside_m / 2
                            and all(np.linalg.norm(spot - p) > 0.01 for _, _, p in ranked)):
                        ranked.append((times, float(np.linalg.norm(spot - here)), spot))
        spots = [p for *_, p in sorted(ranked, key=lambda r: (r[0], r[1]))]
        assert ctx.hover is not None
        tallest = max([ob.height_m for ob in ctx.obstacles if ob.height_m is not None]
                      + [ctx.plan.height_m, ctx.target.max_height_m or s.default_max_height_m])
        z = max(float(ctx.hover[2]), s.table_z + tallest + s.tip_below_grasp_m + s.top_clearance_m)
        found: dict = {}
        why_not: dict = {}
        left, looked = dict(which), 0
        for spot in spots:
            if not left or looked >= s.look_around_max:
                break
            point = np.array([spot[0], spot[1], z])
            rot, _, checked = self._hold(env, [("look", point, s.travel_tol_m)], quiet=True)
            if checked and rot is None:
                continue
            env.move_ee_to(point, tol=s.travel_tol_m, orientation=rot)
            if not self._near(env, point, s.approach_tol_m * 2):
                continue
            looked += 1
            frames: list = []
            got = self.measure_neighbours(env, ctx.vision, ctx.tag, known, left, why_not=why_not, frames=frames)
            if frames:
                self._views.append((f"look{looked}_{spot[0] * 1000:.0f}_{spot[1] * 1000:.0f}", frames[0]))
            found.update(got)
            for key in got:
                left.pop(key)
                why_not.pop(key, None)
        if looked:
            env.move_ee_to(ctx.hover, tol=ctx.hover_tol, orientation=ctx.rot)
            ctx.mark("look around")
        names = [which[k] for k in found]
        self._say(ctx.tag, f"Looked from {looked} spot{'s' if looked != 1 else ''} beside it: "
                           + (f"measured {', '.join(names)}" if names else "measured nothing")
                           + ("" if not left else "; could not measure "
                              + ", ".join(f"{label} ({why_not.get(k, 'not in view')})" for k, label in left.items()))
                           + ".")
        return found

    def _use_measurement(self, env, ctx: PickContext, m: Measured) -> bool:
        """
        A measurement of the object taken away from its hover (_look_around),
        planned by as _measure's is, after the same checks that it is of the
        object. Whether it was.
        """
        s = self.spec
        why = self._unlike_scan(ctx.target, m) or self._too_short(ctx.target, m) or self._other_object(ctx.target, m)
        if why:
            self._say(ctx.tag, f"The look from beside it makes it {why} — not trusting it.")
            return False
        ctx.measured = m
        self._replan(env, ctx, m)
        assert ctx.lift is not None and ctx.grasp is not None and ctx.gp is not None
        ctx.lift = np.array([ctx.lift[0], ctx.lift[1], s.table_z + self._lift_for(ctx.approach, ctx.gp, ctx.lift_h)])
        ctx.grasp = np.array([ctx.grasp[0], ctx.grasp[1], s.table_z + ctx.gp.grasp_height_m])
        return True

    def _third_look(self, env, ctx: PickContext, earlier_m: float, this: Measured) -> "Measured | None":
        """
        An earlier look and this one disagree on the object's height: a third
        look now, and the height two of them agree on — the third look when it
        sides with either, else the earlier (None: the plan as it was). Not
        this one by default: on the rig (2026-10-30) a look at a 39 mm pawn
        with the gripper over part of it read 17 mm, and the pick closed on
        its foot.
        """
        s = self.spec
        third = self.measure_at_hover(env, ctx.vision, ctx.tag, expected_xy=ctx.object_xy)
        ctx.mark("measure")
        said = (f"One earlier look made it {earlier_m * 1000:.0f} mm tall, this one {this.height_mm:.0f}; "
                + ("a third could not measure it" if third is None else f"a third makes it {third.height_mm:.0f}"))
        if third is not None and abs(third.height_mm / 1000 - earlier_m) <= s.same_height_tol_m:
            self._say(ctx.tag, f"{said} — going by the earlier and the third.")
            return third
        if third is not None and abs(third.height_mm - this.height_mm) / 1000 <= s.same_height_tol_m:
            self._say(ctx.tag, f"{said} — going by this one and the third.")
            return third
        self._say(ctx.tag, f"{said} — no two agree; going by the earlier.")
        return None

    def _descend(self, env, ctx: PickContext) -> None:
        """Down to the grasp at target_xy, holding an orientation chosen for where that is."""
        s = self.spec
        assert ctx.target_xy is not None and ctx.grasp is not None and ctx.fit is not None and ctx.gp is not None
        near = safety.moved_near(env, ctx.target_xy, ctx.last_look)
        if near is not None:
            # A hand, or something knocked over, came by since the arm set out:
            # what stands there now may not be what this pick was planned for.
            ctx.disturbed, ctx.doubted, ctx.last_look = True, True, time.monotonic()
            raise Retry("disturbed", f"Something moved {near * 100:.0f} cm from it while the arm was on its way "
                                     "— not coming down on what may no longer be there")
        if ctx.aim_steps and s.miss_steps_along_m:
            out = ctx.target_xy - self._base_xy(env)
            out = out / (np.linalg.norm(out) or 1.0)
            ctx.target_xy = ctx.target_xy + out * self._aim_step(ctx)
        ctx.grasp = np.array([ctx.target_xy[0], ctx.target_xy[1], ctx.grasp[2]])
        if float(np.linalg.norm(ctx.target_xy - ctx.object_xy)) > 0.003 or ctx.gp is not ctx.fit.plan:
            # The orientation was chosen for where the scan put the object,
            # and for the grasp as first planned. A lean lies in the arm's
            # own vertical plane, which turns with the base: 23 mm sideways
            # at 0.28 m is a plane 5 deg round, and holding the old one can
            # stall the descent 4-5 cm short.
            # A grasp re-planned from the measurement may be higher or lower,
            # and its lift with it. Chosen again for where it is, as it is.
            approach = ctx.approach
            again, again_tilt, _, again_lift = self._held_for(
                env, approach, ctx.target_xy, ctx.gp, ctx.obstacles, ctx.target.tilt_deg, tag=ctx.tag, quiet=False,
                start=env.get_ee_pos(), lift_h=None if approach.lift == "highest" else ctx.gp.lift_height_m,
                turn_deg=ctx.turn_deg)
            if again is not None:
                turned = ctx.rot is not None and _angle_deg(ctx.rot, again) > TURN_FIRST_DEG
                ctx.rot, ctx.tilt = again, again_tilt
                if turned and s.descent_turn_first:
                    # Turned to its lean over the object, then straight down: turning on the way down, 3 deg a
                    # step, the arm can go off the line (2026-11-19 15:28, 4 cm up and 3 cm out, stopped).
                    here = np.asarray(env.get_ee_pos(), float)
                    env.move_ee_to(np.array([ctx.target_xy[0], ctx.target_xy[1], here[2]]), tol=ctx.hover_tol,
                                   orientation=ctx.rot)
                if again_lift is not None:
                    ctx.lift = np.array([ctx.target_xy[0], ctx.target_xy[1], s.table_z + again_lift])
            elif (swept := getattr(self, "_swept", None)) is not None:
                raise Retry("no_orientation", f"Where it is, every lean would sweep the gripper "
                                              f"{swept[0] * 1000:.0f} mm too low over {swept[1]} — "
                                              "not coming down")
        # The servo may have left them narrower than the plan wants. target_xy
        # is a point on the table now, so changing the opening does not move it.
        if env.get_gripper_width() < ctx.gp.pregrasp_open_m - 0.002:
            env.set_gripper(ctx.gp.pregrasp_open_m, n_steps=10)
        self._clear_floor(env, ctx)
        self._aim_at_anvil(env, ctx)
        self._fast_to_above_top(env, ctx)
        if s.approach_look and ctx.height is not None:
            self._approach_look(env, ctx)
        # Gentler than the traverse: the last few centimetres happen with the
        # jaws straddling the object, where arriving fast knocks it over and
        # the grasp is lost no matter how good the target was. Under the arm's
        # tighter contact limit, if it has one (safety.descend_onto): pressing
        # here is a finger landing on the object.
        ctx.descent_from = np.asarray(env.get_ee_pos(), float).copy()
        descended = safety.descend_onto(env, ctx.grasp, tol=ctx.grasp_tol, max_step_deg=ctx.grasp_step,
                                        orientation=ctx.rot)
        if not descended:
            descended = self._lean_down_again(env, ctx)
        ctx.mark("descend")
        # Closing the jaws while still short of the grasp pose grips air, and
        # the width checks below cannot tell that apart from reaching for an
        # object that was never there.
        if not descended:
            dist = float(np.linalg.norm(env.get_ee_pos() - ctx.grasp))
            raise Retry("descent_stalled", f"Descent stalled {dist * 100:.1f} cm short — retrying")

    def _clear_floor(self, env, ctx: PickContext) -> None:
        """
        spec.lean_clears_floor: the grasp raised so the leaning gripper's
        lowest point stays lean_floor_clear_m over the table — no higher than
        grip_over_top_m under its top. A grip learned straight down is too low
        for a lean: 2026-11-19, the SO-101, a plug 30 cm out gripped 18 mm up as
        it had held before, leaning 20-30 deg; every descent was stopped by the
        floor guard, the jaw tips 3-8 mm under the table, and the pick failed.
        """
        s = self.spec
        if not s.lean_clears_floor or ctx.rot is None or not isinstance(env, GripperBelowGrasp):
            return
        if ctx.tilt is None or abs(float(ctx.tilt)) < s.lean_clears_floor_from_deg:
            return
        assert ctx.grasp is not None and ctx.gp is not None
        below = max(env.below_grasp_m(ctx.rot, g) for g in (ctx.gp.pregrasp_open_m, ctx.gp.grip_width_m))
        lean = abs(float(ctx.tilt))
        share = float(np.clip((lean - s.lean_clears_floor_from_deg)
                              / max(s.lean_floor_clear_full_deg - s.lean_clears_floor_from_deg, 1e-6), 0.0, 1.0))
        room = s.lean_floor_clear_min_m + share * max(s.lean_floor_clear_m - s.lean_floor_clear_min_m, 0.0)
        if ctx.diagnosis is not None and ctx.diagnosis.cause == "too_high":
            # The wrist camera saw it between the jaws as they closed above it: lower, as it said. 2026-11-19,
            # a plug knocked onto its back: lowered after the miss, raised again for the lean, and missed again.
            room = s.lean_floor_clear_min_m
        need = s.table_z + below + room
        if float(ctx.grasp[2]) >= need - 1e-4:
            return
        top = s.table_z + ctx.gp.height_m - s.grip_over_top_m
        was = (float(ctx.grasp[2]) - s.table_z) * 1000
        if need > top:
            self._say(ctx.tag, f"Leaning, the jaw tips reach {below * 1000:.0f} mm below the grip: it cannot be gripped "
                               f"high enough on it to clear the table — as planned, {was:.0f} mm up.")
            return
        self._say(ctx.tag, f"Leaning {lean:.0f} deg, the jaw tips reach {below * 1000:.0f} mm below the grip: gripping "
                           f"{(need - s.table_z) * 1000:.0f} mm up rather than {was:.0f}, clear of the table.")
        ctx.grasp = np.array([ctx.grasp[0], ctx.grasp[1], need])

    def _fast_to_above_top(self, env: "RobotEnv", ctx: PickContext) -> None:
        """
        Down the final descent's line at full speed to where the jaw tips are
        spec.fast_descent_above_top_m above the object's top (GripperSpec), when
        it is set; the rest, the jaws straddling it, gently as before, under the
        approach's contact limit from a fresh zero (safety.descend_onto). This
        stretch is a move like any other, under the arm's own limit: on the UR5
        (2026-11-13) a CB3's force estimate read 24-54 N moving down at 5 cm/s
        touching nothing, and the approach's 45 N stopped the pick 17 mm above
        the block. Nothing when that point is not below the arm by a
        centimetre: an object of unknown height is planned as the tallest
        there could be.
        """
        s, gp = self.spec, ctx.gp or ctx.plan
        if s.fast_descent_above_top_m is None or gp is None:
            return
        here, grasp = np.asarray(env.get_ee_pos(), float), np.asarray(ctx.grasp, float)
        z = grasp[2] - gp.grasp_height_m + gp.height_m + s.tip_below_grasp_m + s.fast_descent_above_top_m
        if not grasp[2] < z < here[2] - 0.01:
            return
        point = grasp + (here - grasp) * (z - grasp[2]) / (here[2] - grasp[2])      # on the line down
        env.move_ee_to(point, tol=s.travel_tol_m, orientation=ctx.rot)

    def _approach_look(self, env, ctx: PickContext) -> None:
        """
        Down to where the jaw tips are spec.approach_look_above_top_m over the
        object's top (the most the fit allows), one more look, and the fit
        again with it: the grasp re-planned when the height moved by more than
        2 mm, the aim moved when the fit's middle did by 1.5 mm to
        spec.approach_look_shift_m (further, it is not trusted). A look that
        matches the fit poorly is set aside, said.
        """
        s = self.spec
        looks = ctx.height
        if looks is None or looks.fit is None or ctx.grasp is None or ctx.gp is None:
            return
        before = looks.fit
        here = np.asarray(env.get_ee_pos(), float)
        grasp = np.asarray(ctx.grasp, float)
        # Above the tallest it may be: as planned, as tall as an unsettled shape could be, as the fit allows.
        top_mm = max(ctx.gp.height_m * 1000.0, before.spread_mm[1] if before.resolved else 0.0,
                     (ctx.target.max_height_m or 0.0) * 1000.0 if self.shape_unsettled(ctx.target) else 0.0)
        z = s.table_z + top_mm / 1000.0 + s.tip_below_grasp_m + s.approach_look_above_top_m
        if grasp[2] < z < here[2] - 0.01:
            stage = grasp + (here - grasp) * (z - grasp[2]) / (here[2] - grasp[2])
            env.move_ee_to(stage, tol=s.travel_tol_m, orientation=ctx.rot, max_step_deg=ctx.grasp_step)
        had = {id(x) for x in looks.silhouettes}
        lo, hi = looks.height_range_mm
        if before.resolved:
            looks.height_range_mm = (max(lo, before.spread_mm[0] - 10.0), min(hi, before.spread_mm[1] + 10.0))
        try:
            if not self._look_here(env, looks, ctx.tag, where="close above"):
                return
        finally:
            looks.height_range_mm = (lo, hi)
        after = looks.fit
        if not looks.silhouettes or id(looks.silhouettes[-1]) in had:      # set aside as an outlier
            looks.fit = before
            return
        if after is None or not after.resolved or after.ious[-1] < SEEING_IOU:
            self._say(ctx.tag, "The look from close above does not settle it — going by the looks before.")
            del looks.silhouettes[-1]
            del looks.models[-1]
            del looks.looked_from[-1]
            looks.fit = before
            return
        model = looks.models[-1]
        moved = (_in_arm_frame(model, after.centre_xy_m) - _in_arm_frame(model, before.centre_xy_m))
        shift = float(np.linalg.norm(moved))
        if 0.0015 <= shift <= s.approach_look_shift_m and ctx.target_xy is not None:
            ctx.target_xy = ctx.target_xy + moved
            ctx.grasp = np.array([ctx.target_xy[0], ctx.target_xy[1], ctx.grasp[2]])
            self._say(ctx.tag, f"From close above its middle is {shift * 1000:.1f} mm from where the jaws aim — "
                               "aiming there.")
        elif shift > s.approach_look_shift_m:
            self._say(ctx.tag, f"From close above its middle moved {shift * 1000:.0f} mm — not trusting that; "
                               "aiming as before.")
        planned = ctx.gp.height_m * 1000.0                 # a bounded height was planned as its shortest
        if abs(after.height_mm - planned) > 2.0:
            m = self.height_measured(looks)
            if m is not None:
                self._say(ctx.tag, f"From close above it is {after.height_mm:.0f} mm tall, not "
                                   f"{planned:.0f} — the grasp planned again.")
                ctx.measured = m
                self._replan(env, ctx, m)
                assert ctx.lift is not None and ctx.gp is not None
                ctx.lift = np.array([ctx.lift[0], ctx.lift[1], s.table_z + self._lift_for(ctx.approach, ctx.gp, ctx.lift_h)])
                ctx.grasp = np.array([ctx.grasp[0], ctx.grasp[1], s.table_z + ctx.gp.grasp_height_m])
        ctx.mark("look close")

    def _aim_at_anvil(self, env: "RobotEnv", ctx: PickContext) -> None:
        """
        Come down with the object spec.anvil_margin_m off the jaw that does
        not move, where the backend's grasp-point model knows one (HeldWidth):
        told the object between the jaws is its widest part plus a margin
        each side, the grasp point is where that would be held — against the
        fixed jaw on a gripper with one moving jaw; the middle on one whose
        jaws both move. The close then presses it there rather than shoving
        it across first.

        Centred in jaws opened wide for the camera's aim, the moving jaw
        pushes a piece (open - width) / 2 across the board before it is
        held, and it can tip and slide out under the fixed jaw. The wide
        opening stays: its slack is all on the moving jaw's side, where an
        aim error does no harm.
        """
        s = self.spec
        if s.anvil_margin_m is None or not isinstance(env, HeldWidth):
            return
        assert ctx.gp is not None
        env.hold_width(min(env.get_gripper_width(), ctx.gp.width_m + 2 * s.anvil_margin_m))

    def _off_its_line(self, env, ctx: PickContext, quiet: bool = False) -> bool:
        """
        Whether the descent that stopped ended further from the grasp than it
        set out (ctx.descent_from), by more than a centimetre: it went off its
        line, said aloud if so.
        """
        assert ctx.grasp is not None
        if ctx.descent_from is None:
            return False
        here = np.asarray(env.get_ee_pos(), float)
        now, was = float(np.linalg.norm(here - ctx.grasp)), float(np.linalg.norm(ctx.descent_from - ctx.grasp))
        if now <= was + 0.01:
            return False
        if not quiet:
            self._say(ctx.tag, f"The descent went off its line: {now * 100:.1f} cm from the grasp, from {was * 100:.1f} — "
                               f"back over it at {ctx.tilt or 0.0:+.0f} deg, and down again.")
        return True

    def _lean_down_again(self, env, ctx: PickContext) -> bool:
        """
        A pick's descent that stalled, tried again leaning further, as a
        place's is (PlaceMixin.place). Straight up to the hover, the
        orientation chosen again spec.descent_lean_step_deg further from
        straight down, over the object again as the fixed camera sees it, and
        down; up to spec.descent_lean_retries times. True once it arrives.

        Far out the real arm reaches less than its model: a lean that reaches
        the hover can stall centimetres above the grasp, and more lean reaches
        further down. But a descent that stalled before it had got going
        (_stalled_at_start) is tried once more at its own lean first: that is
        not the arm's reach, and a pawn gripped at the further lean can be
        squeezed out of the jaws (2026-11-04, e4). spec.descent_turn_first: one
        that went off its line (_off_its_line) goes back over the grasp at its
        lean and down from there first — "up" from where it went is no nearer
        (2026-11-19 15:28: 4.1 s up from 3 cm out, then 13.5 cm short).
        """
        assert ctx.hover is not None and ctx.grasp is not None
        if self.spec.descent_turn_first and self._off_its_line(env, ctx):
            # Back over the grasp at its lean, and down again, from there: "up" from where it went is no nearer.
            env.move_ee_to(np.array([ctx.grasp[0], ctx.grasp[1], ctx.hover[2]]), tol=ctx.hover_tol,
                           orientation=ctx.rot)
            ctx.descent_from = np.asarray(env.get_ee_pos(), float).copy()
            if safety.descend_onto(env, ctx.grasp, tol=ctx.grasp_tol, max_step_deg=ctx.grasp_step, orientation=ctx.rot):
                return True
        elif self._stalled_at_start(env, ctx.hover[2], ctx.grasp):
            here = env.get_ee_pos()
            env.move_ee_to(np.array([here[0], here[1], ctx.hover[2]]), tol=ctx.hover_tol, orientation=ctx.rot)
            if safety.descend_onto(env, ctx.grasp, tol=ctx.grasp_tol, max_step_deg=ctx.grasp_step, orientation=ctx.rot):
                return True
        tilt = self._tilt_of(env, ctx.rot)
        if tilt is None:
            return False
        for _ in range(self.spec.descent_lean_retries):
            new_rot, new_tilt = self._leaned_further(env, tilt, self._pick_reach_of(env, ctx), ctx.grasp, ctx.tag,
                                                     ctx.obstacles, ctx.approach.lean_margin_deg, ctx.turn_deg)
            if new_rot is None:
                return False
            here = env.get_ee_pos()
            assert ctx.hover is not None
            if self.spec.descent_turn_first and self._off_its_line(env, ctx, quiet=True):
                here = ctx.grasp                              # off its line: up over the grasp, not where it went
            env.move_ee_to(np.array([here[0], here[1], ctx.hover[2]]), tol=ctx.hover_tol, orientation=ctx.rot)
            ctx.rot, ctx.tilt, tilt = new_rot, new_tilt, new_tilt
            self._attempts[-1]["tilt_deg"] = round(new_tilt, 1)
            assert ctx.grasp is not None
            above = np.array([ctx.grasp[0], ctx.grasp[1], ctx.hover[2]])
            env.move_ee_to(above, tol=ctx.hover_tol, orientation=ctx.rot)
            # A new lean is a new pose, with its own kinematic error: the
            # object as the camera sees it, again. Both its estimate and the
            # aim move, so where it is in the camera's frame does not.
            corr = self._overhead_correct(env, above, ctx.hover_tol, ctx.rot, ctx.tag)
            if np.any(corr):
                ctx.object_xy = ctx.object_xy + corr
                assert ctx.target_xy is not None
                ctx.target_xy = ctx.target_xy + corr
                ctx.grasp = np.array([ctx.target_xy[0], ctx.target_xy[1], ctx.grasp[2]])
            ctx.descent_from = np.asarray(env.get_ee_pos(), float).copy()
            if safety.descend_onto(env, ctx.grasp, tol=ctx.grasp_tol, max_step_deg=ctx.grasp_step, orientation=ctx.rot):
                return True
        return False

    @staticmethod
    def gap_px(jaws_mask) -> "np.ndarray | None":
        """
        The middle of the gap between the jaw tips in a wrist-camera picture
        (jaws_mask: the jaws' learned view at this opening): halfway between
        the two jaws' inner edges, along the line between them. None when the
        view does not show two jaws.
        """
        import cv2

        n, labels, stats, centres = cv2.connectedComponentsWithStats(np.asarray(jaws_mask).astype(np.uint8), connectivity=8)
        if n < 3:
            return None
        two = sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA])[:2]
        c1, c2 = (np.asarray(centres[k], float) for k in two)
        axis = c2 - c1
        if float(np.linalg.norm(axis)) < 1.0:
            return None
        axis = axis / np.linalg.norm(axis)
        inner = []
        for k, toward in ((two[0], 1.0), (two[1], -1.0)):
            ys, xs = np.nonzero(labels == k)
            pts = np.c_[xs, ys].astype(float)
            proj = ((pts - (c1 + c2) / 2) * axis).sum(axis=1) * toward
            inner.append(pts[int(np.argmax(proj))])
        return (inner[0] + inner[1]) / 2

    # What is outlined at the jaws' gap is the object when: no more than this
    # share of it is jaw, and it is between these shares of the picture (less: a
    # speck; more: the table).
    BETWEEN_JAW_SHARE = 0.3
    BETWEEN_AREA = (0.003, 0.35)
    BETWEEN_PROBE_PX = 30

    def _between_jaws(self, frame, jm, gap, vision: Vision) -> "np.ndarray | None":
        """
        What lies at and round the gap between the jaw tips in a wrist-camera
        picture, outlined (the segmenter at the gap and round it): the
        outline that is not the jaws nor the table, nearest the gap; None.
        Not where the scan said it would be: at the grasp the camera is too
        close for that, and the jaws stand where it is.
        """
        from mrcf.perception import segment

        if vision.between is not None:
            return vision.between(frame, jm, gap)
        if not segment.available():
            return None
        d = self.BETWEEN_PROBE_PX
        points = [(float(gap[0]) + dx, float(gap[1]) + dy) for dx, dy in ((0, 0), (-d, 0), (d, 0), (0, -d), (0, -2 * d))]
        h, w = frame.shape[:2]
        points = [(u, v) for u, v in points if 0 <= u < w and 0 <= v < h and not jm[int(v), int(u)]]
        best = None
        for masks, scores in segment.segment(frame, points):
            for m, sc in zip(masks, scores):
                m = np.asarray(m, bool)
                area = float(m.mean())
                if not self.BETWEEN_AREA[0] <= area <= self.BETWEEN_AREA[1]:
                    continue
                if float((m & jm).sum()) > self.BETWEEN_JAW_SHARE * max(1.0, float(m.sum())):
                    continue
                ys, xs = np.nonzero(m & ~jm)
                if len(xs) < 30:
                    continue
                near = float(np.min(np.hypot(xs - gap[0], ys - gap[1])))
                key = (near, -float(sc))
                if best is None or key < best[0]:
                    best = (key, m & ~jm)
        return None if best is None else best[1]

    def _wrist_at_grasp(self, env, ctx: PickContext) -> None:
        """
        At the grasp, the jaws open round it: where the object is against the
        gap between the jaw tips, both found in the one wrist-camera picture —
        the object by the pick's own measure (its neighbours told, the jaws'
        pixels not taken for it), the gap from the jaws' learned view at this
        opening — and both carried onto the surface at the grip's height
        through the same camera, so the camera's own placing errs alike for
        both. Off by more than spec.wrist_at_grasp_tol_m: up a little, over by
        that, down again (as _reseat), and looked at once more; by more than
        spec.wrist_at_grasp_max_m, not trusted. 2026-11-18, the SO-101's table:
        the jaws came down ~1 cm to one side of a figure and of a plug, the
        overhead camera putting them "within 2 mm" of their aim — its model of
        where the jaws are, not where they are — and closed on nothing.
        """
        from mrcf.perception.plane_geometry import PlaneView
        from mrcf.policies.parallel_jaw.sighting import _in_arm_frame

        s = self.spec
        jaws = self.jaw_view()
        if s.camera is None or jaws is None or ctx.gp is None or ctx.grasp is None:
            return
        for look in range(s.grasp_reseats + 1):
            time.sleep(0.2)
            frame = self._wrist_frame(env, ctx.tag)
            model = env.camera_model(s.camera, s.render_w, s.render_h)
            if frame is None or model is None:
                return
            frame = np.asarray(frame)
            jm = jaws.at(float(env.get_gripper_width()), frame.shape[:2])
            gap = self.gap_px(jm)
            if gap is None:
                self._say(ctx.tag, "Wrist camera at the grasp: the jaws' view shows no gap — not checked.")
                return
            mask = self._between_jaws(frame, jm, gap, ctx.vision)
            if mask is None:
                self._say(ctx.tag, "Wrist camera at the grasp: nothing outlined at the jaws' gap — not checked.")
                return
            ys, xs = np.nonzero(mask)
            if len(xs) < 30:
                return
            # Both onto the plane at the grip's height, in the camera's frame, then the arm's.
            z = float(model.get("plane_z_m", 0.0)) + float(ctx.gp.grasp_height_m)
            view = PlaneView.from_camera_model(dict(model, plane_z_m=z))
            origin = np.zeros(2) if view.origin_m is None else np.asarray(view.origin_m, float)[:2]
            on = view.to_plane(np.c_[xs, ys].astype(float))
            on = on[np.isfinite(on).all(1)]
            gap_on = view.to_plane(np.asarray([gap], float))[0]
            if not len(on) or not np.isfinite(gap_on).all():
                return
            obj_xy = _in_arm_frame(model, origin + np.median(on, axis=0) / 1000.0)
            gap_xy = _in_arm_frame(model, origin + gap_on / 1000.0)
            off = np.asarray(obj_xy, float) - np.asarray(gap_xy, float)
            out = np.asarray(ctx.target.position[:2], float) - self._base_xy(env)
            out = out / (np.linalg.norm(out) or 1.0)
            along, across = float(off @ out) * 1000, float(off @ np.array([-out[1], out[0]])) * 1000
            self._attempts[-1].setdefault("wrist_at_grasp_mm", []).append([round(along, 1), round(across, 1)])
            dist = float(np.linalg.norm(off))
            if dist <= s.wrist_at_grasp_tol_m:
                self._say(ctx.tag, f"Wrist camera at the grasp: it is between the jaws ({along:+.1f} mm along the "
                                   f"arm, {across:+.1f} across).")
                return
            if dist > s.wrist_at_grasp_max_m:
                self._say(ctx.tag, f"Wrist camera at the grasp: its outline puts it {dist * 1000:.0f} mm from the "
                                   "jaws' gap — not trusted, not moved by it.")
                return
            if look >= s.grasp_reseats:
                self._say(ctx.tag, f"Wrist camera at the grasp: still {along:+.1f} mm along, {across:+.1f} across "
                                   "from the jaws' gap — closing as it is.")
                return
            self._say(ctx.tag, f"Wrist camera at the grasp: it is {along:+.1f} mm along the arm, {across:+.1f} across "
                               f"from the jaws' gap — up {s.grasp_reseat_lift_m * 1000:.0f} mm, over, and down onto it.")
            here = np.asarray(env.get_ee_pos(), float)
            up = here + np.array([0.0, 0.0, s.grasp_reseat_lift_m])
            ctx.grasp = np.array([ctx.grasp[0] + off[0], ctx.grasp[1] + off[1], ctx.grasp[2]])
            if ctx.target_xy is not None:
                ctx.target_xy = ctx.target_xy + off
            over = ctx.grasp + np.array([0.0, 0.0, s.grasp_reseat_lift_m])
            # Each settled to 2 mm: a reseat is a few millimetres, lost in a looser move (2026-11-19, run 38:
            # unsettled, the cap stayed 6.6 then 7.8 mm off the jaws' gap).
            for point in (up, over):
                env.move_ee_to(point, tol=0.002, max_step_deg=s.grasp_step_deg, orientation=ctx.rot)
            safety.descend_onto(env, ctx.grasp, tol=ctx.grasp_tol, max_step_deg=ctx.grasp_step, orientation=ctx.rot)
        return

    def _reseat(self, env, ctx: PickContext, aim, got: dict) -> dict:
        """
        The jaws still further from their aim along the arm than a grip holds
        (spec.grasp_reseat_along_m) after the fixed camera's correction at the
        grasp: up a little (spec.grasp_reseat_lift_m — the open pads slide along
        their own faces), over by what is left, down again, and looked at again;
        up to spec.grasp_reseats times. Still off: retried without closing
        (Retry "off_at_grasp"). 2026-11-17, a wooden figure 25 cm out: the
        correction stalled after 2.2 of 4.9 mm, the jaws closed anyway, and the
        round body was squeezed out toward the base.
        """
        s = self.spec
        if s.grasp_reseat_along_m is None or got.get("aim_after_mm") is None:
            return got
        out = np.asarray(ctx.target.position[:2], float) - self._base_xy(env)
        out = out / (np.linalg.norm(out) or 1.0)
        across = np.array([-out[1], out[0]])
        for _ in range(s.grasp_reseats + 1):
            along_mm, across_mm = got["aim_after_mm"]
            if abs(along_mm) <= s.grasp_reseat_along_m * 1000 and abs(across_mm) <= 2 * s.grasp_settle_tol_m * 1000:
                return got
            if _ >= s.grasp_reseats:
                break
            err = (along_mm * out + across_mm * across) / 1000.0
            self._say(ctx.tag, f"Jaws still {along_mm:+.1f} mm along the arm, {across_mm:+.1f} across from their aim "
                               f"— up {s.grasp_reseat_lift_m * 1000:.0f} mm, over, and down again rather than "
                               "closing off its middle.")
            here = np.asarray(env.get_ee_pos(), float)
            env.move_ee_to(here + np.array([0.0, 0.0, s.grasp_reseat_lift_m]), tol=0.002,
                           max_step_deg=s.grasp_step_deg, orientation=ctx.rot)
            assert ctx.grasp is not None and ctx.target_xy is not None      # set by the descent before it
            ctx.grasp = np.array([ctx.grasp[0] + err[0], ctx.grasp[1] + err[1], ctx.grasp[2]])
            ctx.target_xy = ctx.target_xy + err
            env.move_ee_to(ctx.grasp + np.array([0.0, 0.0, s.grasp_reseat_lift_m]), tol=0.002,
                           max_step_deg=s.grasp_step_deg, orientation=ctx.rot)
            safety.descend_onto(env, ctx.grasp, tol=ctx.grasp_tol, max_step_deg=ctx.grasp_step, orientation=ctx.rot)
            again = self._overhead_at_grasp(env, ctx.target.position[:2], aim, ctx.grasp, ctx.rot, ctx.tag)
            if again is None:
                return got
            got = again
        raise Retry("off_at_grasp", f"The jaws stayed {got['aim_after_mm'][0]:+.1f} mm along the arm from the object's "
                                    "middle — not closing off it; retrying")

    def _close(self, env, ctx: PickContext) -> None:
        """Check the jaws against their aim, close gently, and tell a grip from a miss."""
        s = self.spec
        # Where the jaws stopped against the aim, read before closing (the
        # grasp point is modelled from the opening, so it moves as they shut).
        off_mm = (env.get_ee_pos() - ctx.grasp) * 1000
        # Its aim in the fixed camera's frame: the scan's object, moved as
        # the wrist camera moved the jaws (object_xy is the scan's, in the
        # kinematics' frame since the hover's correction).
        aim = np.asarray(ctx.target.position[:2], float) + (ctx.target_xy - ctx.object_xy)
        at_grasp = self._overhead_at_grasp(env, ctx.target.position[:2], aim, ctx.grasp, ctx.rot, ctx.tag)
        if at_grasp is not None:
            at_grasp = self._reseat(env, ctx, aim, at_grasp)
            self._attempts[-1]["overhead_at_grasp"] = at_grasp
            ctx.mark("check")
        if s.wrist_at_grasp:
            self._wrist_at_grasp(env, ctx)
            ctx.mark("wrist_at_grasp")
        # What the wrist camera sees with the jaws round it, or not: the one
        # look that settles why a close found nothing.
        if s.camera is not None:
            try:
                self._views.append((f"try{len(self._attempts)}_at_grasp",
                                    env.render_image(s.camera, width=s.render_w, height=s.render_h)))
            except Exception:
                pass

        # Close gently: at the arm-wide step rate a jaw sweeps ~70 mm/s and
        # sends a light object skidding instead of gripping it.
        env.set_gripper(0.0, n_steps=20, max_step_pct=s.close_step_pct)
        width, loaded = env.squeeze_gripper()
        ctx.mark("close")
        if not loaded:
            if s.stopped_short_holds and width >= s.min_hold_width_m + s.stopped_short_m:
                # Stopped well short of shut: something is between them that the servo does not feel
                # (2026-11-18, a plug on its back, its smooth pins: 28.5 mm, "no load", taken for a miss).
                self._say(ctx.tag, f"The jaws stopped {width * 1000:.1f} mm apart, short of shut, with no load felt: "
                                   "something is between them — lifting, the hold to be seen.")
            else:
                raise Retry("closed_on_nothing", f"Closed on nothing ({width * 1000:.1f} mm, no load) — missed — "
                                                 f"retrying (the jaws stopped {off_mm[0]:+.0f}, {off_mm[1]:+.0f}, "
                                                 f"{off_mm[2]:+.0f} mm x/y/z from the aim)")
        if width >= s.miss_width_m:
            raise Retry("too_wide", f"Too wide ({width * 1000:.0f} mm) — missed — retrying")
        self._note_grip(env)
        if isinstance(env, HeldWidth):
            assert ctx.gp is not None
            env.hold_width(ctx.gp.grip_width_m)

    def _lift(self, env, ctx: PickContext) -> Outcome:
        """Straight up, still holding it: the pick's Outcome (or Retry when the grip is lost)."""
        s = self.spec
        # Straight up from where the grasp point now IS, not from target_xy.
        # Those differ as soon as the jaws close: the grasp point is modelled
        # from the opening, which changed, so aiming the lift at target_xy
        # drags the gripped object sideways across the surface while rising.
        here = env.get_ee_pos()
        assert ctx.lift is not None
        ctx.lift = np.array([here[0], here[1], ctx.lift[2]])
        lifted = self._move_through(env, ctx.lift, ctx.lift_tol, ctx.rot)
        ctx.mark("lift")
        if not self.holding(env):
            raise Retry("grip_lost", f"Grip lost during lift (width={env.get_gripper_width() * 1000:.0f} mm) — "
                                     "retrying")
        if not lifted:
            # Deliberately NOT a retry: still holding the object, and the
            # next attempt starts by reopening the jaws, which would drop it.
            # A stalled lift means the arm is against a joint limit, which
            # re-running the same approach from the same place cannot fix.
            dist = float(np.linalg.norm(env.get_ee_pos() - ctx.lift))
            self._attempts[-1]["result"] = "lift_stalled"
            self._say_timing(ctx)
            return self._done(Outcome(
                False, (f"grasped the object but the lift stalled {dist * 100:.1f} cm short "
                        f"of {ctx.lift[2] - s.table_z:.3f} m — arm is against a joint limit"),
                measured=ctx.measured, grasp=ctx.gp, at_xy=ctx.target_xy))
        if isinstance(env, GripperTilt):
            self._held_tilt_deg = env.gripper_tilt_deg(ctx.rot)
        self._attempts[-1]["result"] = "ok"
        self._say_timing(ctx)
        gripped = ctx.gp
        if ctx.gp is not None and ctx.grasp is not None:
            at = float(ctx.grasp[2]) - s.table_z
            if abs(at - ctx.gp.grasp_height_m) > 0.0005:
                # Gripped where it was (raised for a lean, _clear_floor): what the place reckons it hangs by.
                # 2026-11-19 13:35: planned 16 mm up, gripped 19, and carried over the tub's wall 3 mm low.
                gripped = dataclasses.replace(ctx.gp, grasp_height_m=at)
        return self._done(Outcome(True, f"grasped object (width={env.get_gripper_width() * 1000:.0f} mm)",
                                  measured=ctx.measured, grasp=gripped, at_xy=ctx.target_xy))

    # ── What the phases and aligners share ──────────────────────────────────────

    def _set_waypoints(self, ctx: PickContext, xy, p: GraspPlan, lift_height: float,
                       hover_height: "float | None" = None) -> None:
        """ctx's hover, grasp and lift over xy for plan p (hover_height: instead of the plan's)."""
        q = p if hover_height is None else dataclasses.replace(p, hover_height_m=hover_height)
        (_, ctx.hover, ctx.hover_tol, _), (_, ctx.grasp, ctx.grasp_tol, ctx.grasp_step), (_, _, ctx.lift_tol, _) = \
            self.pick_waypoints(xy, q)
        ctx.lift = np.array([xy[0], xy[1], self.spec.table_z + lift_height])

    def _pick_reach_of(self, env, ctx: PickContext) -> list:
        """The waypoints ctx's held orientation must reach (_pick_reach)."""
        return self._pick_reach(ctx.hover, ctx.hover_tol, ctx.grasp, ctx.grasp_tol, ctx.lift, ctx.lift_tol, env,
                                ctx.inward)

    @staticmethod
    def _lift_for(approach: Approach, p: GraspPlan, lift_h: float) -> float:
        """The lift height: a "highest" lift goes no higher than its approach found it can."""
        return p.lift_height_m if approach.lift == "full" else min(p.lift_height_m, lift_h)

    def _as_approach(self, ctx: PickContext, approach: Approach, p: GraspPlan, tilt, shape) -> GraspPlan:
        """The plan as this approach grasps it, once the object has been measured."""
        p = self._own_hover(approach, self._widened(p, ctx.room))
        p = self._gripped_at(ctx, p) or p               # where it held before, or lower after a miss
        leaned = self.pads_clear_at(p, tilt)
        if leaned is not p:
            self._say(ctx.tag, f"Leaning {abs(tilt):.0f} deg: grasping at {leaned.grasp_height_m * 1000:.0f} mm rather "
                               f"than {p.grasp_height_m * 1000:.0f}, so the pads' low ends stay above its base.")
            p = leaned
        over = ctx.fit.grasp_over_m if ctx.fit is not None else None
        # Gripped high to pass over what stands in front (fit_approach), or as
        # high as a place needs it held (Target.grasp_above_m): still, whatever
        # the measurement says is the straightest part.
        p = self.gripped_above(p, max((v for v in (over, ctx.target.grasp_above_m) if v is not None), default=None))
        if not approach.keep_top_clear or tilt is None:
            return p
        clear = self._top_clear(p, shape, tilt)
        if clear is not p:
            self._say(ctx.tag, f"Leaning {abs(tilt):.0f} deg: grasping at {clear.grasp_height_m * 1000:.0f} mm "
                               f"rather than {p.grasp_height_m * 1000:.0f}, so its top stays between the tipped jaws.")
        return clear

    def _replan(self, env, ctx: PickContext, measured: Measured) -> None:
        """The grasp re-planned from what the wrist camera measured, for the pick and this attempt."""
        ctx.plan = self._fit_measured(env, measured, ctx.room, ctx.hover, ctx.tag)
        ctx.widths = measured.widths_mm
        ctx.gp = self._as_approach(ctx, ctx.approach, ctx.plan, ctx.tilt, ctx.widths)
        if self._attempts:
            self._attempts[-1]["grip_m"] = round(float(ctx.gp.grasp_height_m), 4)
