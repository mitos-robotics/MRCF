"""
stand_up.py — stand an object that has fallen on its side back up, where it
lies, with a two-finger gripper working from above.

Grip the knob at its top end (a pawn's round head, a bottle's cap) with the
jaws pointing straight down and closing square across the object, then lift
that end straight up. The knob turns between the pads like a hinge; the foot,
the heavy end of most things made to stand, drags along the surface in under
it; and once the object hangs upright the jaws let it down onto its foot and
open. Nothing is turned in the air, so nothing needs the gripper on its side —
which a 5-joint arm reaches only far out from its base and well up (on the
SO-101, ~28 cm out and 6 cm up, while a fallen pawn lies on the board).

What it is given (Target.lying, policies.base.Lying): where the knob's middle
is, which way the object lies, how long it is and how wide the knob is — from
a fixed camera (mrcf/perception/lying.py) or the scenario. It looks no further
itself: what it is told is where it grips.

    hover    over the knob, the jaws open, turned square across the object,
             told the knob is GripperSpec.stand_up_margin_m wider each side
             than it is, so the fixed jaw comes down beside it (HeldWidth)
    grip     straight down until the jaw tips are stand_up_tip_clearance_m off
             the surface — below the knob's middle — and close gently
    stand    straight up, in steps, until the knob is where it is with the
             object stood up, plus stand_up_overshoot_m
    release  open, and straight back up to the hover

A miss, a slip or a stall opens the jaws and backs up; it has likely moved
the object, so the wrist camera looks again from over it (the backend's
WristSurface: where its pixels land on the surface from there) and the next
try aims at what it sees — up to GripperSpec.stand_up_attempts, the arm never
parked for a scan in between. Where the wrist camera cannot find it (or there
is none), it ends there (ok False), and the caller looks again. A stop
(MotionStopped) passes through, as everywhere.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mrcf.capabilities import HeldWidth, OrientationSearch, Settling, WristServo, WristSurface
from mrcf.policies.base import Lying, Outcome, Target

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw
    from mrcf.robot_env import RobotEnv

TAG = "[Stand up]"
# What a look from over it can help with: it was missed, or let go of, and
# has likely moved a little.
LOOK_AGAIN_AFTER = ("closed_on_nothing", "slipped", "lift_stalled", "descent_stalled")


def turn_to_square(jaws_open_along: np.ndarray, lies_along: np.ndarray) -> float:
    """
    Degrees, counter-clockwise from above, to turn jaws that open along
    jaws_open_along so they open square across something lying along
    lies_along — the smaller way round (the jaws are the same either way), so
    within (-90, 90].
    """
    want = np.array([-lies_along[1], lies_along[0]], float)
    have = np.asarray(jaws_open_along, float)[:2]
    turn = np.degrees(np.arctan2(have[0] * want[1] - have[1] * want[0], have @ want))
    turn = (turn + 90.0) % 180.0 - 90.0
    return 90.0 if turn == -90.0 else float(turn)


def jaws_land_on(knob_xy, lies_along, reach_m: float, half_width_m: float, obstacles, clearance_m: float,
                 unmeasured_radius_m: float) -> "str | None":
    """
    What the fingers would come down on: the label of an obstacle within
    clearance_m of the strip the jaws sweep — reach_m either way across the
    object from the knob, half_width_m along it — or None.
    """
    axis = np.asarray(lies_along, float)
    across = np.array([-axis[1], axis[0]])
    for ob in obstacles:
        d = np.asarray(ob.position, float)[:2] - np.asarray(knob_xy, float)[:2]
        a, c = abs(float(d @ axis)), abs(float(d @ across))
        r = ob.radius_m if ob.radius_m is not None else unmeasured_radius_m
        gap = np.hypot(max(0.0, a - half_width_m), max(0.0, c - reach_m))
        if gap < r + clearance_m:
            return ob.label or "an object"
    return None


@dataclass
class _Plan:
    """One try's geometry, from what is known of how it lies."""
    knob: np.ndarray
    knob_w: float
    beside: float
    open_w: float
    grip_z: float
    stood_z: float
    hover_z: float
    turn: float
    rot: "np.ndarray | None"

    @property
    def hover(self) -> np.ndarray:
        return np.array([self.knob[0], self.knob[1], self.hover_z])


def stand_up(pol: "ParallelJaw", env: "RobotEnv", target: Target, obstacles=(), vision=None) -> Outcome:
    """
    Stand target up (module docstring). Refused, nothing moved, when it is not
    known to be lying, its knob is too wide or narrow for the jaws, the fingers
    would land on a neighbour, or no one orientation of the gripper reaches
    the hover, the grip and the stood-up height with the jaws turned square.
    vision: policies.Vision, for its `lying` (how the wrist camera finds it
    after a miss); None, the framework's.
    """
    from mrcf.policies.parallel_jaw.planning import _takes

    s = pol.spec
    pol._notes, pol._views, pol._attempts = [], [], []
    lying = target.lying
    if lying is None:
        return pol._done(Outcome(False, "not known to be lying on its side (Target.lying) — nothing to stand up",
                                 refused=True))
    if not isinstance(env, OrientationSearch) or not _takes(env.approach_orientation_for, "turn_deg"):
        return pol._done(Outcome(False, "this arm cannot turn its jaws to a direction "
                                        "(approach_orientation_for takes no turn_deg)", refused=True))
    tries = max(1, int(s.stand_up_attempts))
    seen_by, n = "given", 0
    while True:
        n += 1
        plan = _plan(pol, env, lying, obstacles)
        if isinstance(plan, str):
            if n == 1:
                return pol._done(Outcome(False, plan, refused=True))
            return pol._done(Outcome(False, f"after a miss, as the wrist camera sees it: {plan}",
                                     at_xy=np.asarray(lying.top_xy, float)[:2]))
        attempt = {"attempt": n, "approach": "stand_up", "seen_by": seen_by, "turn_deg": round(plan.turn, 1),
                   "result": None}
        pol._attempts.append(attempt)
        code, why = _try(pol, env, plan, attempt)
        if code == "ok":
            again = f" on try {n}" if n > 1 else ""
            return pol._done(Outcome(True, f"stood it up{again}: gripped its top, lifted it "
                                           f"{(plan.stood_z - plan.grip_z) * 1000:.0f} mm and let it down onto "
                                           "its foot", at_xy=plan.knob, approach="stand_up"))
        if code not in LOOK_AGAIN_AFTER or n == tries:
            return pol._done(Outcome(False, why, at_xy=plan.knob))
        found, not_found = _look_again(pol, env, plan, lying, vision, n)
        if found is None:
            return pol._done(Outcome(False, f"{why}; then the wrist camera {not_found}", at_xy=plan.knob))
        moved = float(np.linalg.norm(np.asarray(found.top_xy, float)[:2] - plan.knob)) * 1000
        pol._say(TAG, f"{why} — the wrist camera sees it {found.length_m * 1000:.0f} mm long, its top "
                      f"{moved:.0f} mm from where the jaws went: trying again there.")
        lying, seen_by = found, "wrist"


def _plan(pol: "ParallelJaw", env, lying: Lying, obstacles) -> "_Plan | str":
    """A try's geometry and the one gripper orientation for it, or why there is none (nothing moved)."""
    s = pol.spec
    axis = lying.axis
    knob = np.asarray(lying.top_xy, float)[:2]
    knob_w = float(lying.top_width_m)
    # The jaws come down with the knob stand_up_margin_m off the fixed jaw, and
    # beside the widest thing within a finger's width of it (a collar).
    beside = lying.widest_near_top(s.finger_half_width_m) + 2 * s.stand_up_margin_m
    open_w = beside + 2 * s.jaw_slack_m
    if knob_w < s.stand_up_min_knob_m:
        return f"its top measured {knob_w * 1000:.0f} mm across — too small to be one to grip, or a bad measurement"
    if open_w > s.max_open_m:
        return (f"its top is {knob_w * 1000:.0f} mm across: the jaws would need {open_w * 1000:.0f} mm, "
                f"and open {s.max_open_m * 1000:.0f}")
    reach = open_w - beside / 2 + s.finger_thickness_m
    near = jaws_land_on(knob, axis, reach, s.finger_half_width_m, obstacles, s.neighbour_clearance_m,
                        s.unmeasured_obstacle_radius_m)
    if near is not None:
        return f"no room for the jaws round its top: they would come down on {near}"
    z0 = s.table_z
    grip_z = z0 + s.tip_below_grasp_m + s.stand_up_tip_clearance_m
    # The knob rises with the jaws from where they grip it: stood up, its
    # middle is the object's length less half the knob above the surface.
    stood_z = grip_z + (lying.length_m - knob_w) + s.stand_up_overshoot_m
    widest = max([knob_w, *(w for _, w in lying.widths)])
    hover_z = z0 + max(s.hover_height_m, widest + s.top_clearance_m + s.tip_below_grasp_m,
                       stood_z - z0 + s.travel_tol_m)
    square = turn_to_square(pol.jaw_axis(env, knob), axis)
    waypoints = [("hover", np.array([knob[0], knob[1], hover_z]), s.travel_tol_m),
                 ("grip", np.array([knob[0], knob[1], grip_z]), s.grasp_tol_m),
                 ("stand", np.array([knob[0], knob[1], stood_z]), s.travel_tol_m)]
    # Square to it if the wrist turns that far; else the nearest it does, up to
    # stand_up_turn_slack_deg off. 2026-11-03, a bishop fallen on c4: square
    # needed the wrist roll at -88 or +92 deg, past where it meets the camera's
    # cable; it stopped at -80, 8 deg off square, from where it could have gripped.
    turn = None
    for off in _offsets(s.stand_up_turn_slack_deg):
        if pol._hold(env, waypoints, TAG, obstacles, quiet=True, turn_deg=square + off)[0] is not None:
            turn = square + off
            break
    if turn is None:
        return (f"no one gripper orientation reaches its top at ({knob[0]:+.3f}, {knob[1]:+.3f}) from above "
                f"with the jaws within {s.stand_up_turn_slack_deg:.0f} deg of square to it ({square:+.0f})")
    off_square = "" if abs(turn - square) < 0.5 else f" ({turn - square:+.0f} off square: the wrist turns no further)"
    pol._say(TAG, f"Lying {lying.length_m * 1000:.0f} mm long, its top {knob_w * 1000:.0f} mm across at "
                  f"({knob[0]:+.3f}, {knob[1]:+.3f}): jaws turned {turn:+.0f} deg{off_square}, open "
                  f"{open_w * 1000:.0f} mm, lifting it to {(stood_z - z0) * 1000:.0f} mm.")
    rot = pol.hold_orientation(env, waypoints, TAG, obstacles, turn_deg=turn)
    if rot is None:
        return (f"no one gripper orientation reaches its top at ({knob[0]:+.3f}, {knob[1]:+.3f}) from above "
                f"with the jaws turned {turn:+.0f} deg")
    return _Plan(knob, knob_w, beside, open_w, grip_z, stood_z, hover_z, turn, rot)


def _offsets(slack_deg: float, step_deg: float = 2.5) -> list:
    """0, then +-step, +-2 step, ... out to slack_deg: the turns off square to try, nearest first."""
    out, k = [0.0], 1
    while k * step_deg <= slack_deg + 1e-9:
        out += [k * step_deg, -k * step_deg]
        k += 1
    return out


def _try(pol: "ParallelJaw", env, plan: _Plan, attempt: dict) -> "tuple[str, str]":
    """One grip, lift and release: ("ok", "") or (a result code, what happened), the jaws open and up."""
    from mrcf.policies.parallel_jaw.planning import _takes

    s = pol.spec
    knob, rot = plan.knob, plan.rot

    def failed(code: str, why: str) -> "tuple[str, str]":
        attempt["result"] = code
        env.set_gripper(plan.open_w)
        here = env.get_ee_pos()
        if here[2] < plan.hover_z - s.travel_tol_m:
            pol._move_through(env, np.array([here[0], here[1], plan.hover_z]), s.travel_tol_m, rot)
        return code, why

    env.set_gripper(plan.open_w)
    if isinstance(env, HeldWidth):
        env.hold_width(plan.beside)
    if not env.move_ee_to(plan.hover, tol=s.travel_tol_m, orientation=rot) and not pol._near(
            env, plan.hover, 2 * s.approach_tol_m):
        return failed("hover_short", "could not reach the hover over its top")
    # Asked again once if it stops high: with the tips 2-3 mm higher they are
    # level with a pawn head's middle, where the moving jaw squeezes it out
    # (GripperSpec.stand_up_tip_clearance_m).
    grip = np.array([knob[0], knob[1], plan.grip_z])
    for _ in range(2):
        env.move_ee_to(grip, tol=s.stand_up_grip_tol_m, max_step_deg=s.grasp_step_deg, orientation=rot)
        high = env.get_ee_pos()[2] - plan.grip_z
        if high <= 2 * s.stand_up_grip_tol_m:
            break
    else:
        return failed("descent_stalled", f"the descent stopped {high * 1000:.1f} mm above the grip")
    _keep_view(pol, env, f"stand_up_at_grip_{attempt['attempt']}")
    if _takes(env.set_gripper, "max_step_pct"):
        env.set_gripper(0.0, n_steps=20, max_step_pct=s.close_step_pct)
    else:
        env.set_gripper(0.0)
    if not pol.holding(env):
        return failed("closed_on_nothing", f"closed on nothing ({env.get_gripper_width() * 1000:.1f} mm) — "
                                           "missed its top")
    if isinstance(env, HeldWidth):
        env.hold_width(plan.knob_w)
    pol._say(TAG, "Gripped its top — lifting it upright.")
    here = env.get_ee_pos()
    steps = max(1, int(np.ceil((plan.stood_z - here[2]) / s.stand_up_step_m)))
    for z in np.linspace(here[2], plan.stood_z, steps + 1)[1:]:
        env.move_ee_to(np.array([here[0], here[1], z]), tol=0.002, max_step_deg=s.stand_up_step_deg,
                       orientation=rot)
    if not pol.holding(env):
        return failed("slipped", "it slipped out of the jaws on the way up")
    short = plan.stood_z - env.get_ee_pos()[2]
    if short > 0.006:
        return failed("lift_stalled", f"the lift stopped {short * 1000:.0f} mm short — let go of it there")
    env.set_gripper(plan.open_w)
    # Up from where the jaws are now: opening moves the grasp point the arm
    # reports (one jaw moves), and aiming at the old one swings the fingers
    # sideways past the piece just stood up.
    here = env.get_ee_pos()
    pol._move_through(env, np.array([here[0], here[1], plan.hover_z]), s.travel_tol_m, rot)
    attempt["result"] = "ok"
    return "ok", ""


def _look_again(pol: "ParallelJaw", env: "RobotEnv", plan: _Plan, was: Lying, vision, n: int) -> "tuple[Lying | None, str]":
    """
    After a miss, from over where the knob was: how the wrist camera sees it
    lying now, or (None, what went wrong). From the servo's own height, where
    the backend's pixel map was calibrated, with the jaws as they were.
    """
    s = pol.spec
    if s.camera is None or not isinstance(env, WristSurface):
        return None, "is not there to look again with"
    z = s.table_z + (s.servo_height_m if s.servo_height_m is not None else plan.hover_z - s.table_z)
    env.move_ee_to(np.array([plan.knob[0], plan.knob[1], z]), tol=s.travel_tol_m, orientation=plan.rot)
    if isinstance(env, Settling):
        env.wait_still()
    else:
        time.sleep(0.3)
    frame = env.render_image(s.camera, width=s.render_w, height=s.render_h)
    pol._views.append((f"stand_up_look_{n}", frame))
    near_px = (env.wrist_servo_target_px if isinstance(env, WristServo) else None) or (s.render_w / 2, s.render_h / 2)
    if vision is not None and getattr(vision, "lying", None) is not None:
        found, why = vision.lying(frame, env.wrist_to_surface, near_px, was)
    else:
        found, why = _find_lying(frame, env.wrist_to_surface, near_px, was, s.stand_up_relook_shift_m)
    if found is None:
        return None, f"could not find it: {why}"
    moved = float(np.linalg.norm(np.asarray(found.top_xy, float)[:2] - np.asarray(was.top_xy, float)[:2]))
    if moved > s.stand_up_relook_shift_m:
        return None, (f"found something lying with its top {moved * 1000:.0f} mm from where it was — "
                      "further than a miss moves one; not taken for it")
    return found, ""


def _find_lying(frame, to_surface, near_px, was: Lying, max_shift_m: float) -> "tuple[Lying | None, str]":
    """Vision.lying's default: perception.lying.find_lying_in_frame, with what it was expected to be."""
    from mrcf.perception import lying as perceive

    seen, why = perceive.find_lying_in_frame(frame, perceive.SurfaceMap(to_surface), near_px,
                                             _as_measurement(was), max_shift_m=max_shift_m,
                                             max_length_mm=was.length_m * 1000 * (1 + perceive.LENGTH_AGREE))
    if seen is None:
        return None, why
    return Lying(foot_xy=np.asarray(seen.foot_end_m, float), top_xy=np.asarray(seen.top_m, float),
                 length_m=seen.length_mm / 1000.0, top_width_m=seen.top_width_mm / 1000.0,
                 widths=tuple((at / 1000.0, w / 1000.0) for at, w in seen.widths_mm)), ""


def _as_measurement(was: Lying):
    """A Lying as perception.lying measures one, to compare a new sighting with."""
    from mrcf.perception import lying as perceive

    axis = was.axis
    top = np.asarray(was.top_xy, float)[:2]
    widths = [(at * 1000.0, w * 1000.0) for at, w in was.widths]
    foot_w = max([w for at, w in widths if at <= perceive.END_SPAN * was.length_m * 1000.0] or [was.top_width_m * 1000])
    return perceive.LyingMeasurement(
        foot_end_m=np.asarray(was.foot_xy, float)[:2], top_end_m=top + axis * was.top_width_m / 2, top_m=top,
        length_mm=was.length_m * 1000.0, top_width_mm=was.top_width_m * 1000.0, foot_width_mm=foot_w,
        widths_mm=widths, ends_certain=True,
        elongation=was.length_m * 1000.0 / max([w for _, w in widths] or [foot_w]))


def _keep_view(pol: "ParallelJaw", env, label: str) -> None:
    s = pol.spec
    if s.camera is not None:
        try:
            pol._views.append((label, env.render_image(s.camera, width=s.render_w, height=s.render_h)))
        except Exception:
            pass
