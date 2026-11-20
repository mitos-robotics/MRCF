"""
rim_pick.py — pick up something with thin walls and an open top (a cup, a bowl,
a scoop) by one of its walls: one jaw inside it, one outside, closed on the wall
a little below its rim.

What is too wide to close round, or has a handle in the way, still has a wall
that is a few millimetres thick (user, 2026-11-17: "it can be lifted by just
grasping one wall").

    rim      its outline as the overhead camera saw it (Target.outlines), carried
             onto the plane of its rim — at the lowest it may stand (its height,
             or the height looks' lower bound)
    where    a point of that outline with a hollow behind it at least
             spec.rim_hollow_m across (a wall, not a handle's edge), where the
             wall faces along the jaws' own
             line (no turn of the wrist), within spec.rim_max_misalign_deg, the
             outer jaw's side clear of other objects; the best aligned first,
             nearest the arm's base among equals; the first the arm reaches
    over     above it, the jaws open spec.rim_jaw_gap_m either side of the wall,
             their tips clear of the rim; corrected by the fixed camera
    down     to spec.rim_grip_depth_m below the rim (no lower than the least
             grasp height)
    close    squeezed: something held — up; nothing — open, back up, failed

Refused, nothing moved, when no outline, no height, or no point reaches; the
height looks (height_looks.py) are asked when the height is not known, and
only their lower bound is needed. Outcome.grasp is a GraspPlan for the place:
the object hangs grasp_height_m below the jaws.
"""

from __future__ import annotations

import dataclasses

from typing import TYPE_CHECKING

import numpy as np

from mrcf.capabilities import OrientationSearch
from mrcf.policies.base import Outcome, Target
from mrcf.policies.parallel_jaw.planning import _takes

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw

TAG = "[Rim]"


def rim_outline(env, target: Target, height_m: float) -> "np.ndarray | None":
    """target's outline from a fixed camera, carried onto the plane height_m above the surface: (N, 2) world xy."""
    from mrcf.perception.plane_geometry import PlaneView

    for camera, outline in (target.outlines or {}).items():
        size = tuple(outline.get("size") or ())
        px = np.asarray(outline.get("px") or [], float).reshape(-1, 2)
        if len(size) != 2 or len(px) < 3:
            continue
        model = env.camera_model(camera, int(size[0]), int(size[1]))
        if model is None:
            continue
        view = PlaneView.from_camera_model(model)
        raised = PlaneView(view.K, view.D, view.R, view.t + view.R @ np.array([0.0, 0.0, height_m * 1000.0 / view.scale]),
                           scale=view.scale, origin_m=view.origin_m)
        xy = raised.to_plane(px)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) >= 3:
            origin = np.zeros(2) if raised.origin_m is None else np.asarray(raised.origin_m, float)[:2]
            return origin[None, :] + xy / 1000.0
    return None


def wall_points(poly: np.ndarray, step_m: float = 0.004) -> list:
    """[(point, outward unit normal)] every step_m along a closed outline."""
    poly = np.asarray(poly, float)
    area = 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        n = np.array([d[1], -d[0]]) / length            # outward for a counter-clockwise outline
        if area < 0:
            n = -n
        for f in np.arange(0.0, 1.0, step_m / length):
            out.append((a + f * d, n))
    return out


def hollow_behind(poly: np.ndarray, points: list, reach_m: float = 0.03) -> list:
    """
    For each (point, outward normal) of a closed outline (wall_points), how
    wide a space it encloses just inside there (metres): the widest circle
    inside the outline within reach_m of the point, along the normal inward,
    across. A cup's or a bowl's wall has its hollow behind it; the edge of a
    handle, a few millimetres from its other edge, has none.
    """
    import cv2

    poly = np.asarray(poly, float)
    lo = poly.min(axis=0) - 0.005
    mm = np.round((poly - lo) * 1000.0).astype(np.int32)
    size = mm.max(axis=0) + 6
    inside = np.zeros((int(size[1]), int(size[0])), np.uint8)
    cv2.fillPoly(inside, [mm], 1)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    out = []
    for p, n in points:
        best = 0.0
        for d in np.arange(0.002, reach_m + 1e-9, 0.002):
            q = np.round((np.asarray(p, float) - np.asarray(n, float) * d - lo) * 1000.0).astype(int)
            if 0 <= q[1] < dist.shape[0] and 0 <= q[0] < dist.shape[1]:
                best = max(best, float(dist[q[1], q[0]]))
        out.append(2.0 * best / 1000.0)
    return out


# A close on nothing at the wall is tried again this much lower, this many times (no lower than the jaws go).
RIM_LOWER_M = 0.010
RIM_LOWER_TRIES = 1


def rim_pick(policy: "ParallelJaw", env, target: Target, obstacles=()) -> Outcome:
    """Pick target up by one of its walls: see the module docstring."""
    from mrcf.policies.parallel_jaw.spec import GraspPlan

    s = policy.spec
    if not s.rim_pick:
        return Outcome(False, "this arm does not pick things up by a wall (GripperSpec.rim_pick)", refused=True)
    if not target.outlines:
        return Outcome(False, "no outline of it from above to find its wall by", refused=True)
    lo = hi = target.height_m
    if target.grip_at_m is not None:
        # A wall grip that held it before (a deployment's catalogue): its rim taken as that grip's depth above
        # it, the hover over the top it was planned by — no looks for its height. 2026-11-18, a scoop: its
        # looks put the rim at 48-56, 54-62, 66-150 mm and "4" on four picks; held once, gripped 36 mm up.
        lo = float(target.grip_at_m) + s.rim_grip_depth_m
        hi = max(lo, float(target.max_height_m or lo))
        policy._say(TAG, f"Gripped by its wall as before, {target.grip_at_m * 1000:.0f} mm up — no looks for its height.")
    if lo is None and s.height_from_views:
        _, looks, _ = policy.survey_height(env, target, obstacles)
        fit = None if looks is None else looks.fit
        if fit is not None:
            # Gripped by the best fit, not the least that fits: too high is a close on air beside the
            # wall, too low no lower than the jaws go. A scoop's looks put that least at 4, 22, 27 and
            # 13 mm on four tries (2026-11-17), its best at 40-odd every time. Hovered over the most.
            lo, hi = min(float(fit.height_mm), float(fit.spread_mm[1])) / 1000.0, float(fit.spread_mm[1]) / 1000.0
    if lo is None:
        lo, hi = target.min_height_m, target.max_height_m
    if lo is None or hi is None:
        return Outcome(False, "how high its rim stands is not known", refused=True)
    # Gripped rim_grip_depth_m below the rim — or as low as the jaws go, if that is still a bite of the
    # wall: 2026-11-17, a scoop's looks said 22-57 mm and it was refused for wanting 28.
    if lo < s.min_grasp_height_m + s.rim_min_bite_m:
        return Outcome(False, f"its walls may be only {lo * 1000:.0f} mm tall: too low to grip below the rim",
                       refused=True)
    from mrcf.policies.handle_pick import handle_pick

    by_handle = handle_pick(policy, env, target, obstacles, lo=lo, hi=hi)
    if by_handle is not None and not by_handle.refused:
        return by_handle                     # tried by its handle: held, or not (it moved nothing else)
    poly = rim_outline(env, target, lo)
    if poly is None:
        return Outcome(False, "its outline could not be carried onto its rim", refused=True)
    centre = poly.mean(axis=0)
    width = float(np.max(np.linalg.norm(poly - centre, axis=1)) * 2.0)
    base = policy._base_xy(env)
    gap = s.rim_jaw_gap_m
    opening = min(2 * gap, s.max_open_m)
    grasp_h = max(s.min_grasp_height_m, lo - s.rim_grip_depth_m)
    hover_h = hi + s.tip_below_grasp_m + s.top_clearance_m
    lift_h = max(s.lift_height_m, hover_h, grasp_h + s.lift_clearance_m)
    cos_ok = float(np.cos(np.radians(s.rim_max_misalign_deg)))
    others = [o for o in obstacles if o.position is not None]
    ranked = []
    points = wall_points(poly)
    # Only a wall with a hollow behind it: the edge of a handle has none (2026-11-17, a scoop's handle
    # was taken for a wall and the jaws shut on the table beside it).
    hollows = hollow_behind(poly, points)
    for (p, n), hollow in zip(points, hollows):
        if hollow < s.rim_hollow_m:
            continue
        axis = policy.jaw_axis(env, p)
        align = abs(float(n @ axis))
        if align < cos_ok:
            continue
        outer = p + n * (gap + s.finger_thickness_m)
        if any(float(np.linalg.norm(outer - np.asarray(o.position, float)[:2]))
               < (o.radius_m or s.unmeasured_obstacle_radius_m) + s.finger_thickness_m for o in others):
            continue
        ranked.append((-round(align, 2), float(np.linalg.norm(p - base)), p, n))
    ranked.sort(key=lambda r: (r[0], r[1]))
    if not ranked:
        return Outcome(False, f"no stretch of its wall with a hollow behind it ({s.rim_hollow_m * 1000:.0f} mm) faces "
                              f"along the jaws (within {s.rim_max_misalign_deg:.0f} deg) with room for the outer jaw",
                       refused=True)
    chosen: "tuple | None" = None
    for _, _, p, n in ranked[:s.rim_tries]:
        grasp = np.array([p[0], p[1], s.table_z + grasp_h])
        hover = np.array([p[0], p[1], s.table_z + hover_h])
        if isinstance(env, OrientationSearch):
            found = env.approach_orientation_for([hover, grasp], prefer_tilt_deg=0.0)
            if found is None:
                continue
            chosen = (grasp, hover, np.asarray(found[0], float), n)
        else:
            chosen = (grasp, hover, None, n)
        break
    if chosen is None:
        return Outcome(False, "no point of its wall the arm reaches straight enough down", refused=True)
    grasp, hover, rot, n = chosen
    plan = GraspPlan(height_m=hi, width_m=width, grasp_height_m=grasp_h, grip_width_m=0.004, pregrasp_open_m=opening,
                     hover_height_m=hover_h, lift_height_m=lift_h, release_height_m=grasp_h + s.release_clearance_m)
    policy._say(TAG, f"By its wall at ({grasp[0]:+.3f}, {grasp[1]:+.3f}): the jaws {opening * 1000:.0f} mm open across "
                     f"it, {grasp_h * 1000:.0f} mm up (its rim {lo * 1000:.0f}-{hi * 1000:.0f} mm).")
    env.set_gripper(opening, n_steps=15)
    if not policy._over_walls(env, hover, TAG, rot):           # as a plain pick's way to its hover
        return Outcome(False, "could not get up over a container's walls to cross them", refused=True, grasp=plan)
    env.move_ee_to(hover, tol=s.travel_tol_m, orientation=rot)
    if not policy._near(env, hover, 2 * s.travel_tol_m):
        return Outcome(False, "did not get over its wall", grasp=plan)
    if policy._can_overhead(env):
        corr = policy._overhead_correct(env, hover, s.travel_tol_m, rot, TAG)
        grasp = grasp + np.r_[corr, 0.0]
    attempts: list = []
    while True:
        attempts.append({"attempt": len(attempts) + 1, "approach": "rim", "grip_m": round(float(grasp[2] - s.table_z), 4)})
        if not policy.descend_onto(env, grasp, tol=s.grasp_tol_m, max_step_deg=s.grasp_step_deg, orientation=rot):
            if not policy._near(env, grasp, 0.006):
                env.move_ee_to(hover, tol=s.travel_tol_m, orientation=rot)
                attempts[-1]["result"] = "descent_stalled"
                return Outcome(False, "the descent onto its wall stalled short", grasp=plan, attempts=attempts)
        env.set_gripper(0.0, n_steps=20, max_step_pct=s.close_step_pct)
        # A wall is thinner than any piece: held by the load, down to rim_min_hold_m, where the backend can tell.
        thin = {"min_hold_m": s.rim_min_hold_m} if _takes(env.squeeze_gripper, "min_hold_m") else {}
        held, loaded = env.squeeze_gripper(**thin)
        if loaded:
            break
        attempts[-1]["result"] = "closed_on_nothing"
        env.set_gripper(opening, n_steps=10)
        lower = float(grasp[2] - s.table_z) - RIM_LOWER_M
        if len(attempts) > RIM_LOWER_TRIES or lower < s.min_grasp_height_m:
            env.move_ee_to(hover, tol=s.travel_tol_m, orientation=rot)
            return Outcome(False, f"closed on nothing at its wall ({held * 1000:.1f} mm)", grasp=plan, attempts=attempts)
        # Over the rim, most likely — the looks for its height are loose on a thin wall: lower, where the
        # open jaws still straddle it, and closed again.
        attempts[-1]["diagnosis"] = "too_high"
        policy._say(TAG, f"Closed on nothing {grasp[2] - s.table_z:.3f} m up — {RIM_LOWER_M * 1000:.0f} mm lower, "
                         "the jaws still either side of its wall.")
        grasp = np.array([grasp[0], grasp[1], s.table_z + lower])
        plan = dataclasses.replace(plan, grasp_height_m=lower, release_height_m=lower + s.release_clearance_m)
    attempts[-1]["result"] = "ok"
    policy._note_grip(env)                     # a thin wall can read empty once carried: the wrist camera sees it
    if s.held_while_still:
        # The jaws stopped on the wall under load: held, while they stay where they stopped (Sighting.
        # _still_gripped). 2026-11-19, three times: the scoop held by its wall, called "left the jaws on the
        # way up" by the camera's check and the model's, and carried home in the jaws.
        policy._held_at(env, held)
    env.move_ee_to(np.array([grasp[0], grasp[1], s.table_z + lift_h]), tol=s.travel_tol_m, orientation=rot)
    if not policy.holding(env):
        attempts[-1]["result"] = "grip_lost"
        return Outcome(False, "it left the jaws on the way up", grasp=plan, attempts=attempts)
    return Outcome(True, f"picked up by its wall (jaws {held * 1000:.1f} mm)", grasp=plan, at_xy=grasp[:2].copy(),
                   attempts=attempts)
