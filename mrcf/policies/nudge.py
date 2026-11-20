"""
nudge.py — push an object a short way across the surface with the closed jaws
of a two-finger gripper, without gripping it: back onto its spot after it has
been knocked, or out of the way of another object's pick.

    close    the jaws shut: together the fingers are a small blunt block
    hover    over a spot just short of the object, on the side away from where
             it is going (GripperSpec.push_gap_m off it)
    down     straight down until the jaw tips are push_tip_clearance_m off the
             surface — low on the object, where pushing slides it rather than
             tips it
    push     along the line to the goal in short steps (push_step_m), slowly,
             until the fingers stand where the object's near side will be
    off      back from it a little, then straight up

Pushed low, a tall object's top would meet the gripper body above the fingers
(free_depth_m above the pads), so the push is raised until the body clears it
and whatever tall stands beside the fingers' path — but never above
push_max_contact_share of the object's height, where a push tips it over
rather than slides it. Refused, nothing moved, when the push is longer than
push_max_m, the object would be pushed into a neighbour, the fingers have no
room to come down behind it or to pass, or no one gripper orientation reaches
the whole push. The object ends roughly where it was pushed: a push is not a
place, so whoever asked for it looks again before relying on where it stands.

What it is given is geometry, as every policy is (policies.base): the object
as a Target, its neighbours as Obstacles, and where its foot should end up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mrcf.capabilities import OrientationSearch
from mrcf.policies.base import Outcome, Target

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw
    from mrcf.robot_env import RobotEnv

TAG = "[Nudge]"


def _segment_distance(p, a, b) -> float:
    """Distance from point p to the segment a-b (surface plane)."""
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a
    t = 0.0 if float(ab @ ab) == 0.0 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _half_width_up(target: Target, lo_m: float, hi_m: float, default_m: float) -> float:
    """target's widest half-width between lo_m and hi_m above the surface, by its shape (or its kind's); else default_m."""
    widths = target.widths_mm or target.assumed_widths_mm
    if not widths:
        return default_m
    levels = sorted((float(h), float(w)) for h, w in widths.items())
    hs, ws = np.array(levels).T
    band = ws[(hs >= lo_m * 1000 - 2.0) & (hs <= hi_m * 1000 + 2.0)]
    at = [float(np.interp(lo_m * 1000, hs, ws)), float(np.interp(hi_m * 1000, hs, ws))]
    return max([*band, *at]) / 2000.0


def nudge(pol: "ParallelJaw", env: "RobotEnv", target: Target, to_xy, obstacles=(),
          check_only: bool = False) -> Outcome:
    """Push target's foot from where it is to to_xy (module docstring). check_only: plan it, move nothing."""
    s = pol.spec
    if not check_only:
        pol._notes, pol._views, pol._attempts = [], [], []
    plan = _plan(pol, env, target, np.asarray(to_xy, float)[:2], obstacles)
    if isinstance(plan, str):
        return pol._done(Outcome(False, plan, refused=True)) if not check_only else Outcome(False, plan, refused=True)
    if plan["dist"] < 0.001:
        return Outcome(True, "already there", at_xy=plan["to"])
    if check_only:
        return Outcome(True, plan["said"], at_xy=plan["to"], approach="nudge")
    if pol.holding(env):
        return pol._done(Outcome(False, "something is in the jaws — the fingers cannot close to push", refused=True))
    pol._say(TAG, plan["said"])
    rot, start_low, end_low, hover, lift = plan["rot"], plan["start_low"], plan["end_low"], plan["hover"], plan["lift"]
    env.set_gripper(0.0)
    if not env.move_ee_to(hover, tol=s.travel_tol_m, orientation=rot) and not pol._near(
            env, hover, 2 * s.approach_tol_m):
        return pol._done(Outcome(False, "could not reach the spot over where the fingers come down",
                                 at_xy=plan["from"]))
    env.move_ee_to(start_low, tol=s.grasp_tol_m, max_step_deg=s.grasp_step_deg, orientation=rot)
    if env.get_ee_pos()[2] - start_low[2] > 2 * s.grasp_tol_m:
        pol._move_through(env, hover, s.travel_tol_m, rot)
        return pol._done(Outcome(False, "the fingers stopped above where they push from — nothing pushed",
                                 at_xy=plan["from"]))
    d = plan["d"]
    steps = max(1, int(np.ceil(float(np.linalg.norm(end_low[:2] - start_low[:2])) / s.push_step_m)))
    for p in np.linspace(start_low, end_low, steps + 1)[1:]:
        env.move_ee_to(p, tol=0.002, max_step_deg=s.push_step_deg, orientation=rot)
    here = env.get_ee_pos()
    # How far the fingers got along the line, past where they first touch it.
    moved = float(np.clip((here[:2] - start_low[:2]) @ d - s.push_gap_m, 0.0, plan["dist"]))
    at = plan["from"] + d * moved
    back = here - np.array([d[0], d[1], 0.0]) * s.push_back_off_m
    env.move_ee_to(back, tol=s.grasp_tol_m, max_step_deg=s.push_step_deg, orientation=rot)
    pol._move_through(env, np.array([back[0], back[1], lift[2]]), s.travel_tol_m, rot)
    short = plan["dist"] - moved
    if short > s.push_short_tol_m:
        return pol._done(Outcome(False, f"the push stopped {short * 1000:.0f} mm short — it stands about "
                                        f"{moved * 1000:.0f} mm on from where it was", at_xy=at, approach="nudge"))
    return pol._done(Outcome(True, f"pushed it {plan['dist'] * 1000:.0f} mm", at_xy=plan["to"], approach="nudge"))


def _plan(pol: "ParallelJaw", env, target: Target, to: np.ndarray, obstacles) -> "dict | str":
    """The push's geometry and one gripper orientation for it, or why there is none (nothing moved)."""
    from mrcf.policies.parallel_jaw.planning import _takes
    from mrcf.policies.parallel_jaw.reach import BASE_LABEL

    s = pol.spec
    frm = np.asarray(target.position[:2], float)
    v = to - frm
    dist = float(np.linalg.norm(v))
    if dist < 0.001:
        return {"dist": 0.0, "to": frm}
    if dist > s.push_max_m:
        return (f"{dist * 1000:.0f} mm is further than one push moves anything ({s.push_max_m * 1000:.0f} mm) — "
                "pick it up instead")
    d = v / dist
    obstacles = pol._with_base(env, obstacles)
    height = target.height_m or target.max_height_m or s.default_max_height_m
    body_above_tips = s.tip_below_grasp_m + s.free_depth_m
    foot_r = (target.foot_m or 2 * s.unmeasured_obstacle_radius_m) / 2
    # Where the closed fingers' block is, to the side of the line and along it.
    finger_r = float(np.hypot(s.finger_thickness_m, s.finger_half_width_m))
    pusher_reach = s.finger_thickness_m + s.finger_half_width_m + finger_r     # the longest the line could need
    start_guess = frm - d * (foot_r + pusher_reach + s.push_gap_m)
    # Tall things the gripper body passes over while the fingers are down: the
    # object itself, and any neighbour beside the fingers' path.
    tall = [height] + [ob.height_m or s.default_max_height_m for ob in obstacles
                       if ob.label != BASE_LABEL and _segment_distance(ob.position, start_guess, to - d * foot_r)
                       < s.sweep_half_width_m + (ob.radius_m or s.unmeasured_obstacle_radius_m)]
    tip_z = max(s.push_tip_clearance_m, max(tall) + s.push_body_margin_m - body_above_tips)
    # Kept clear of as the tallest it could be, but pushed no higher up than
    # is safe for the shortest: an object not measured may be either.
    shortest = target.height_m or target.min_height_m or height
    if tip_z > s.push_max_contact_share * shortest:
        return (f"to keep the gripper over {'it' if max(tall) == height else 'what stands beside the way'}, the "
                f"fingers would push it {tip_z * 1000:.0f} mm up — high enough to tip it over")
    # The fingers come down past all of it above their tips, and push on its
    # widest there: a top wider than the foot would be landed on otherwise.
    contact_r = _half_width_up(target, tip_z, height, foot_r)
    for ob in obstacles:
        r = ob.radius_m if ob.radius_m is not None else s.unmeasured_obstacle_radius_m
        if _segment_distance(ob.position, frm, to) < foot_r + r + s.neighbour_clearance_m:
            return f"it would be pushed into {ob.label or 'an obstacle'}"
    turns = [0.0]
    if isinstance(env, OrientationSearch) and _takes(env.approach_orientation_for, "turn_deg"):
        turns += list(s.jaw_turns_deg)
    why_not = "no room"
    for turn in turns:
        axis = pol.jaw_axis(env, frm, turn)
        across = np.array([-axis[1], axis[0]])
        face = s.finger_thickness_m * abs(float(d @ axis)) + s.finger_half_width_m * abs(float(d @ across))
        start = frm - d * (contact_r + face + s.push_gap_m)
        end = to - d * (contact_r + face)
        blocked = None
        for ob in obstacles:
            r = pol.half_width_at(ob, tip_z)
            if float(np.linalg.norm(np.asarray(ob.position[:2], float) - start)) < finger_r + r + s.neighbour_clearance_m:
                blocked = f"no room for the fingers to come down behind it: {ob.label or 'an obstacle'} is there"
                break
            if _segment_distance(ob.position, start, end) < finger_r + r + s.neighbour_clearance_m:
                blocked = f"the fingers would run into {ob.label or 'an obstacle'} on the way"
                break
        if blocked:
            why_not = blocked
            continue
        z0 = s.table_z
        grasp_z = z0 + tip_z + s.tip_below_grasp_m
        hover_z = z0 + max(s.hover_height_m, height + s.tip_below_grasp_m + s.top_clearance_m)
        start_low = np.array([start[0], start[1], grasp_z])
        end_low = np.array([end[0], end[1], grasp_z])
        hover = np.array([start[0], start[1], hover_z])
        lift = np.array([end[0], end[1], hover_z])
        waypoints = [("hover", hover, s.travel_tol_m), ("down", start_low, s.grasp_tol_m),
                     ("push", end_low, s.grasp_tol_m), ("lift", lift, s.travel_tol_m)]
        rot, _, checked = pol._hold(env, waypoints, TAG, obstacles, quiet=True, turn_deg=turn)
        if checked and rot is None:
            why_not = "no one gripper orientation reaches the whole push"
            continue
        if rot is not None:
            near = [c for p in (start_low, end_low)
                    if (c := pol.body_clearance(env, rot, p, obstacles)) is not None and c[0] < s.camera_clearance_m]
            if near:
                why_not = f"the wrist camera would pass {max(near[0][0], 0) * 1000:.0f} mm from {near[0][1]}"
                continue
        turned = f", the jaws turned {turn:+.0f} deg" if turn else ""
        said = (f"Pushing it {dist * 1000:.0f} mm toward ({to[0]:+.3f}, {to[1]:+.3f}) with the jaws shut, "
                f"{tip_z * 1000:.0f} mm up it{turned}.")
        return {"dist": dist, "from": frm, "to": to, "d": d, "rot": rot, "start_low": start_low,
                "end_low": end_low, "hover": hover, "lift": lift, "said": said}
    return why_not


__all__ = ["nudge"]
