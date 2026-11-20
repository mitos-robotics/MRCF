"""
lying_pick.py — pick up a long thing lying along the table whose height is not
known — a tool: a hammer, a screwdriver, a spanner — gripped across where it
holds best, closed from as high as it can stand there down until the jaws
hold it.

    what     an outline from above (Target.outlines) at least
             spec.lying_min_aspect times as long as it is widest across: one
             lying along the table (else None: the plain pick); at 1 or less,
             any outline, compact things too (a tape measure)
    where    by its handle, where it has one (handle_first; handle_pick.
             find_handle: the long narrow stretch at its narrower end, gripped
             next to where it joins the wider part). The wider part of a
             hammer, an axe, a mallet is its heavy end: gripped beside it the
             tool hangs in the jaws instead of turning out of them (user,
             2026-11-20: "the hammer is one side heavy so account for that in
             the pick"). Else at its balance point — the middle of its outline's
             area along its length — or the stretch nearest it the jaws open
             across, spec.lying_side_clear_m clear each side. A stretch whose
             fingers would land on a neighbour (Obstacle) is passed over; one
             inside its own outline is a part of it
    how high a thing at rest lies on its broad side, so where it is gripped it
             stands no taller than it is wide across there. Its outline is put
             on the plane half that high, and the jaws closed from below that
             width, opened and spec.handle_step_m lower each time they close on
             nothing, down to the least grasp height. Its height need not be
             measured, and no size is written in: 2026-11-20, the UR5 (no wrist
             camera, its overhead camera too steep to see heights) planned the
             hammer for the tallest anything could be and closed 116 mm above it,
             twice
    up       spec.lying_test_lift_m first and the grip checked, then the full
             lift: a tool gripped off its balance turns out of the jaws a
             couple of centimetres up, not from the lift height

It returns None, nothing moved, when the thing is not lying along the table or
no stretch of it fits the jaws (taking compact things too, spec.lying_min_aspect
1 or less: an Outcome refused, said); an Outcome refused, nothing moved, when
every stretch that fits has a neighbour in the jaws' way; an Outcome once it
has tried.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mrcf.policies import grasp_sense
from mrcf.policies.base import Outcome, Target
from mrcf.policies.handle_pick import _sections, _turn_to, find_handle, straight_down

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw

TAG = "[Lying]"


def _axes(poly: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """(centre, unit along its length, unit across it) of an outline on the surface."""
    centre = poly.mean(axis=0)
    _, _, vt = np.linalg.svd(poly - centre, full_matrices=False)
    return centre, vt[0] / np.linalg.norm(vt[0]), vt[1] / np.linalg.norm(vt[1])


def is_lying(poly, min_aspect: float) -> bool:
    """Whether an outline on the surface is at least min_aspect times as long as it is widest across."""
    poly = np.asarray(poly, float).reshape(-1, 2)
    if len(poly) < 3:
        return False
    centre, along, across = _axes(poly)
    secs = _sections(poly, centre, along, across)
    if len(secs) < 6:
        return False
    if min_aspect <= 1.0:
        return True                                  # compact things too (a tape measure): gripped low the same way
    length = float(np.ptp((poly - centre) @ along))
    return length >= min_aspect * max(w for _, w, _ in secs)


def grips(poly, spec) -> list:
    """
    Where to grip an outline lying along the surface, best first: [(centre xy,
    unit across it — the way the jaws open, its width there, "handle" or
    "balance")]. By its handle next to the part it joins first (spec.
    handle_first), then the stretches the jaws open across, nearest its
    balance point first — no nearer an end than the fingers' half width.
    """
    s = spec
    poly = np.asarray(poly, float).reshape(-1, 2)
    out = []
    if s.handle_first:
        found = find_handle(poly, s.handle_min_m, s.handle_max_m, s.handle_min_len_m, s.handle_from_join_m,
                            s.finger_half_width_m, join_min_len=s.lying_join_min_m)
        if found is not None:
            centre, tip, width = found
            if width + 2 * s.lying_side_clear_m <= s.max_open_m:
                out.append((np.asarray(centre, float), np.array([-tip[1], tip[0]]), float(width), "handle"))
    centre, along, across = _axes(poly)
    secs = _sections(poly, centre, along, across)
    if not secs:
        return out
    widths = np.array([w for _, w, _ in secs])
    at = np.array([p for p, _, _ in secs])
    balance = float((at * widths).sum() / widths.sum())        # the middle of its area along it
    lo, hi = at.min() + s.finger_half_width_m, at.max() - s.finger_half_width_m
    for clear in (s.lying_side_clear_m, 0.002):
        # Room each side as the jaws come down; where nothing has that, a tighter fit, 2 mm a side (2026-11-20:
        # the UR5's tape measure, no stretch of it 5 mm clear each side of the 52 mm jaws).
        fit = [(p, w, m) for p, w, m in secs
               if lo <= p <= hi and s.handle_min_m <= w and w + 2 * clear <= s.max_open_m]
        if fit:
            break
    for p, w, m in sorted(fit, key=lambda r: abs(r[0] - balance)):
        out.append((centre + along * p + across * m, across.copy(), float(w), "balance"))
    return out


def _inside(poly: np.ndarray, xy) -> bool:
    import cv2
    return cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (float(xy[0]), float(xy[1])), False) >= 0


def _in_the_way(spec, poly, centre, across, opening, obstacles) -> "str | None":
    """The neighbour the open jaws would come down on, gripping at centre across `across`; None: clear."""
    along = np.array([across[1], -across[0]])
    for o in obstacles:
        if o.position is None:
            continue
        xy = np.asarray(o.position, float)[:2]
        if _inside(poly, xy):
            continue                                  # a part of it the scan outlined apart
        r = float(o.radius_m) if o.radius_m is not None else spec.unmeasured_obstacle_radius_m
        d = xy - centre
        if (abs(float(d @ along)) < spec.finger_half_width_m + r
                and abs(float(d @ across)) < opening / 2 + spec.finger_thickness_m + r):
            return o.label or "something"
    return None


def lying_pick(policy: "ParallelJaw", env, target: Target, obstacles=()) -> "Outcome | None":
    """Pick target up lying along the table (see the module docstring); None: not one, nothing moved."""
    from mrcf.policies.parallel_jaw.spec import GraspPlan
    from mrcf.policies.rim_pick import rim_outline

    s = policy.spec
    if not s.lying_pick or not target.outlines:
        return None
    plane = s.min_grasp_height_m
    poly = rim_outline(env, target, plane)
    if poly is None or not is_lying(poly, s.lying_min_aspect):
        return None
    found = grips(poly, s)
    if found and abs(found[0][2] / 2 - plane) > 0.003:
        # Put on the plane half as high as it is wide where it is gripped: where its sides are.
        again = rim_outline(env, target, max(plane, found[0][2] / 2))
        if again is not None and is_lying(again, s.lying_min_aspect):
            poly, found = again, grips(again, s)
    if not found:
        if s.lying_min_aspect <= 1.0:
            # Taking compact things too, and nothing on it fits the jaws: said, not handed to a pick that has no
            # height for it and closes in the air over it (2026-11-20 17:20, the tape measure, three times).
            narrowest = min(w for _, w, _ in _sections(poly, *_axes(poly))) if len(poly) >= 3 else 0.0
            return Outcome(False, f"nowhere on it narrow enough for the jaws: {narrowest * 1000:.0f} mm across at its "
                                  f"narrowest, as the overhead camera sees it, the jaws open "
                                  f"{s.max_open_m * 1000:.0f}", refused=True)
        return None
    _, along_all, across_all = _axes(poly)
    secs = _sections(poly, poly.mean(axis=0), along_all, across_all)
    length = float(np.ptp(poly @ along_all))
    # As tall as it can be anywhere: no taller than it is wide (lying on its broad side), nor than the
    # tallest the caller says anything is.
    tallest_all = max(w for _, w, _ in secs)
    if target.max_height_m is not None:
        tallest_all = min(tallest_all, float(target.max_height_m))
    high_h = max(s.lift_height_m, tallest_all + s.tip_below_grasp_m + s.top_clearance_m)
    # Each place it can be gripped, best first, until the arm gets over one: a grip it cannot come down to —
    # a neighbour in the jaws' way, no orientation that reaches, the gripper's box (2026-11-20 17:01: the
    # hammer's head 4 cm from the box's edge, its handle further out, and it "did not get over it") — is
    # passed over for the next, nothing moved.
    start = np.asarray(env.get_ee_pos(), float).copy()
    why_not: list = []
    got_over = False
    for centre, across, width, how in found:
        opening = min(width + 2 * s.handle_open_m, s.max_open_m)
        who = _in_the_way(s, poly, centre, across, opening, obstacles)
        if who is not None:
            why_not.append(f"{who} in the jaws' way")
            continue
        stands = min(width, tallest_all)                  # where it is gripped
        turn = _turn_to(policy, env, centre, across)
        heights = []
        h = stands - s.handle_step_m
        while h >= s.min_grasp_height_m - 1e-9:
            heights.append(h)
            h -= s.handle_step_m
        if not heights or heights[-1] > s.min_grasp_height_m + 0.001:
            heights.append(s.min_grasp_height_m)
        hover_h = stands + s.tip_below_grasp_m + s.top_clearance_m
        high = np.array([centre[0], centre[1], s.table_z + high_h])
        hover = np.array([centre[0], centre[1], s.table_z + hover_h])
        lowest = np.array([centre[0], centre[1], s.table_z + heights[-1]])
        rot, why = straight_down(env, [high, hover, lowest], turn)
        if why:
            why_not.append(f"{why} there")
            continue
        # Does it make sense, as the arm will really be turned: a finger either side of it, not on it
        # (policies/grasp_sense.py). Refused before anything moves.
        jaw = grasp_sense.jaw_direction(env, rot)
        if jaw is not None:
            wrong = grasp_sense.fingers_on_it(poly, centre, jaw, across, opening, s.finger_thickness_m,
                                              s.finger_half_width_m)
            if wrong:
                policy._say(TAG, f"Not coming down there: {wrong}.")
                why_not.append(wrong)
                continue
        where = "by its handle, next to the heavier part it joins" if how == "handle" else "at its balance point"
        policy._say(TAG, f"Lying along the table, {length * 1000:.0f} mm long: gripped {where} at "
                         f"({centre[0]:+.3f}, {centre[1]:+.3f}), {width * 1000:.0f} mm across — the jaws turned "
                         f"{turn:+.0f} deg, {opening * 1000:.0f} mm open, closed from {heights[0] * 1000:.0f} mm "
                         "down until they hold it.")
        env.set_gripper(opening, n_steps=15)
        if not policy._over_walls(env, high, TAG, rot):
            return Outcome(False, "could not get up over a container's walls to cross them", refused=True)
        env.move_ee_to(high, tol=s.travel_tol_m, orientation=rot)
        if policy._near(env, high, 2 * s.travel_tol_m):
            got_over = True
            break
        if not policy._near(env, start, 0.01):
            return Outcome(False, "did not get over it", refused=True)       # it moved, and stopped short
        why_not.append("the arm cannot get over it there")
    if not got_over:
        if why_not and all("jaws' way" in w for w in why_not):
            return Outcome(False, f"every place it can be gripped across has {why_not[0]}", refused=True)
        return Outcome(False, "did not get over it anywhere it can be gripped across — "
                              + "; ".join(dict.fromkeys(why_not)), refused=True)
    if policy._can_overhead(env):
        corr = policy._overhead_correct(env, high, s.travel_tol_m, rot, TAG)
        centre = centre + corr
        hover[:2] = centre
    env.move_ee_to(hover, tol=s.travel_tol_m, orientation=rot)      # straight down, over it alone
    attempts: list = []
    held = 0.0
    grasp = hover.copy()
    for h in heights:
        grasp = np.array([centre[0], centre[1], s.table_z + h])
        attempts.append({"attempt": len(attempts) + 1, "approach": f"lying_{how}", "grip_m": round(float(h), 4)})
        if not policy.descend_onto(env, grasp, tol=s.grasp_tol_m, max_step_deg=s.grasp_step_deg, orientation=rot):
            if not policy._near(env, grasp, 0.006):
                attempts[-1]["result"] = "descent_stalled"
                break                                 # stopped on it, not beside it: a close here grips nothing
        env.set_gripper(0.0, n_steps=20, max_step_pct=s.close_step_pct)
        held, loaded = env.squeeze_gripper()
        if loaded and held >= s.handle_min_m / 2:
            attempts[-1]["result"] = "ok"
            break
        attempts[-1]["result"] = "closed_on_nothing"
        attempts[-1]["diagnosis"] = "too_high"
        env.set_gripper(opening, n_steps=10)
    # What it holds, in the jaws' own frame (u along the way they open, v square to it) from where they close:
    # what a place lays inside a container, turned with the wrist (place_in.laid_in).
    opens = jaw if jaw is not None else np.asarray(across, float)
    square = np.array([-opens[1], opens[0]])
    rel = poly - np.asarray(grasp[:2], float)[None, :]
    held_uv = tuple((round(float(d @ opens), 4), round(float(d @ square), 4)) for d in rel)
    plan = GraspPlan(height_m=float(stands), width_m=length, grasp_height_m=float(grasp[2] - s.table_z),
                     grip_width_m=float(width), pregrasp_open_m=opening, hover_height_m=high_h,
                     lift_height_m=max(s.lift_height_m, high_h),
                     release_height_m=float(grasp[2] - s.table_z) + s.release_clearance_m, held_outline=held_uv)
    if not attempts or attempts[-1].get("result") != "ok":
        env.move_ee_to(high, tol=s.travel_tol_m, orientation=rot)
        return Outcome(False, f"closed on nothing across it, {heights[0] * 1000:.0f} mm down to "
                              f"{float(grasp[2] - s.table_z) * 1000:.0f}", grasp=plan, attempts=attempts)
    policy._note_grip(env)
    from mrcf.capabilities import Carries
    if isinstance(env, Carries):
        env.carrying(held_uv, float(held))            # its outline part of the arm's shape for the motion guard
    # A little way up first: gripped off its balance, it turns out of the jaws here, not from the lift height.
    env.move_ee_to(grasp + np.array([0.0, 0.0, s.lying_test_lift_m]), tol=s.travel_tol_m, orientation=rot)
    if not policy.holding(env):
        attempts[-1]["result"] = "grip_lost"
        env.set_gripper(opening, n_steps=10)
        env.move_ee_to(high, tol=s.travel_tol_m, orientation=rot)
        return Outcome(False, f"it turned out of the jaws {s.lying_test_lift_m * 1000:.0f} mm up — gripped off "
                              "its balance, or too lightly", grasp=plan, attempts=attempts)
    env.move_ee_to(np.array([grasp[0], grasp[1], s.table_z + plan.lift_height_m]), tol=s.travel_tol_m,
                   orientation=rot)
    if not policy.holding(env):
        attempts[-1]["result"] = "grip_lost"
        return Outcome(False, "it left the jaws on the way up", grasp=plan, attempts=attempts)
    return Outcome(True, f"picked up lying along the table, {where} ({held * 1000:.0f} mm, "
                         f"{plan.grasp_height_m * 1000:.0f} mm up)", grasp=plan, at_xy=grasp[:2].copy(),
                   attempts=attempts)
