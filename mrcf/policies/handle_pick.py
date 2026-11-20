"""
handle_pick.py — pick something up by its handle: the long narrow part that
leads away from its widest part (a scoop's, a pan's, a ladle's), the jaws
turned to close across it.

User, 2026-11-19: "with the scoop why not grasp a more secure thing - its
right on the very very edge - like maybe grab the handle??". Held by its 1 mm
wall, the scoop was gripped at the very edge (its rim stood lower than its
looks said) and, carried, the jaws crept shut on the flexing wall (0.9 % of
their travel to 0.07 %), so nothing could tell it held from lost; three times
that day it was called lost with it in the jaws. A handle 2 cm across stops
the jaws on itself under load, and the servo feels it held.

    where   its outline from above (Target.outlines), carried onto the plane
            its handle may be at, along the outline's long axis: how wide it
            is across, every few millimetres. The handle is the stretch at the
            narrower end between spec.handle_min_m and handle_max_m wide, at
            least handle_min_len_m long; gripped handle_from_join_m out from
            where it joins the wide part (sturdiest there), the jaws' own width
            clear of that part (find_handle)
    turn    the jaws turned to open across the handle (OrientationSearch's
            turn_deg); without a turn the arm can make, not by the handle
    down    straddling it, open spec.handle_open_m wider than it each side;
            closed first at the top it may have — its height's lower bound, or
            where it held before (Target.grip_at_m) — and, closing on nothing,
            opened and handle_step_m lower each time, down to the least grasp
            height: how high a handle stands is not known, and need not be
    close   felt under load, at least handle_min_m / 2 apart: held — up

handle_pick returns None, nothing moved, when there is no handle to grip
(the caller grips a wall instead); an Outcome once it has tried.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mrcf.capabilities import OrientationSearch
from mrcf.policies.base import Outcome, Target
from mrcf.policies.parallel_jaw.planning import _takes

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw

TAG = "[Handle]"
STEP_M = 0.003                 # cross-sections this far apart along the outline's long axis


def _sections(poly: np.ndarray, centre, along, across, step_m: float = STEP_M) -> list:
    """[(s, width, mid)]: at each s along `along` from centre, how wide poly is across (and the middle of that)."""
    rel = np.asarray(poly, float) - centre
    s_all, t_all = rel @ along, rel @ across
    out = []
    n = len(rel)
    for s in np.arange(float(s_all.min()) + step_m / 2, float(s_all.max()), step_m):
        ts = []
        for i in range(n):
            s0, s1, t0, t1 = s_all[i], s_all[(i + 1) % n], t_all[i], t_all[(i + 1) % n]
            if (s0 - s) * (s1 - s) <= 0 and s0 != s1:
                ts.append(t0 + (s - s0) / (s1 - s0) * (t1 - t0))
        if len(ts) >= 2:
            out.append((float(s), float(max(ts) - min(ts)), float((max(ts) + min(ts)) / 2)))
    return out


def find_handle(poly, min_w: float, max_w: float, min_len: float, from_join: float,
                finger_half_m: float = 0.0, join_min_len: float = 0.0) -> "tuple[np.ndarray, np.ndarray, float] | None":
    """
    Where to grip poly's handle: (centre xy, unit vector along the handle
    toward its tip, its width there), or None when it has no handle — no
    stretch at its narrower end between min_w and max_w wide for min_len.
    Gripped from_join out from where the handle meets the wider part, and
    finger_half_m more (the jaws' own half width, along the handle).
    join_min_len: the wider part is wider than max_w for at least this long;
    a shorter wide stretch — the flared butt at the end of a hammer's grip, a
    lip — is passed over as part of the handle. 2026-11-20, the UR5's hammer:
    one 3 mm slice of its grip's end read 42 mm, the walk from its tip stopped
    there 9 mm in, and it had "no handle". 0: the first wider slice is the join.
    """
    poly = np.asarray(poly, float).reshape(-1, 2)
    if len(poly) < 3:
        return None
    centre = poly.mean(axis=0)
    _, _, vt = np.linalg.svd(poly - centre, full_matrices=False)
    along, across = vt[0] / np.linalg.norm(vt[0]), vt[1] / np.linalg.norm(vt[1])
    secs = _sections(poly, centre, along, across)
    if len(secs) < 6:
        return None
    third = max(1, len(secs) // 3)
    head, tail = np.mean([w for _, w, _ in secs[:third]]), np.mean([w for _, w, _ in secs[-third:]])
    if head > tail:                                   # walk from the narrower end: the handle's tip
        secs, along = [(-s, w, m) for s, w, m in reversed(secs)], -along
    run = []
    for i, (s, w, m) in enumerate(secs):
        if w > max_w:
            wide = i
            while wide < len(secs) and secs[wide][1] > max_w:
                wide += 1
            if wide < len(secs) and secs[wide][0] - s < join_min_len:
                continue                              # a short wider stretch of the handle itself: a lip
            break                                     # the wide part: where the handle joins it
        run.append((s, w, m))
    else:
        return None                                   # narrow all along: a stick, not a handle
    run = [r for r in run if r[1] >= min_w] if run else []
    if not run or run[-1][0] - run[0][0] < min_len:
        return None
    join = run[-1][0]
    want = join - from_join - finger_half_m
    s, w, m = min(run, key=lambda r: abs(r[0] - want))
    if join - s < finger_half_m:
        return None
    # `along` points from the tip toward the wide part here: the tip is the other way.
    return centre + along * s + across * m, -along, float(w)


def _turn_to(policy: "ParallelJaw", env, xy, across) -> float:
    """The turn (deg, within +-90) that makes the jaws open along `across` at xy."""
    axis = policy.jaw_axis(env, xy)
    ang = float(np.degrees(np.arctan2(axis[0] * across[1] - axis[1] * across[0], float(axis @ across))))
    return (ang + 90.0) % 180.0 - 90.0


def straight_down(env, targets: list, turn_deg: float) -> "tuple[np.ndarray | None, str]":
    """
    (rotation, "") of one gripper orientation, straight down, the jaws turned
    turn_deg, that reaches every target (OrientationSearch); (None, "") where
    the env does not search orientations (the arm's own); (None, why) where it
    cannot. Only the keyword arguments the arm's search takes are passed
    (capabilities.OrientationSearch: a backend may leave some out — the UR5's
    takes no prefer_tilt_deg; 2026-11-20 every lying pick raised on it).
    """
    if not isinstance(env, OrientationSearch):
        return None, ""
    fn = env.approach_orientation_for
    kw: dict = {}
    if _takes(fn, "prefer_tilt_deg"):
        kw["prefer_tilt_deg"] = 0.0
    if turn_deg and _takes(fn, "turn_deg"):
        kw["turn_deg"] = turn_deg
    elif abs(turn_deg) > 10.0:
        return None, "the jaws cannot be turned across it here"
    got = fn(list(targets), **kw)
    if got is None:
        return None, "no way straight down to it with the jaws across it"
    return np.asarray(got[0], float), ""


def handle_pick(policy: "ParallelJaw", env, target: Target, obstacles=(), lo: "float | None" = None,
                hi: "float | None" = None) -> "Outcome | None":
    """Pick target up by its handle (see the module docstring); None: no handle to grip, nothing moved."""
    from mrcf.policies.parallel_jaw.spec import GraspPlan
    from mrcf.policies.rim_pick import rim_outline

    s = policy.spec
    if not s.handle_first or not target.outlines:
        return None
    top = lo if lo is not None else (target.min_height_m or target.height_m)
    tallest = hi if hi is not None else (target.max_height_m or target.height_m or top)
    if top is None or tallest is None:
        return None
    poly = rim_outline(env, target, max(s.min_grasp_height_m, 0.6 * float(top)))
    if poly is None:
        return None
    found = find_handle(poly, s.handle_min_m, s.handle_max_m, s.handle_min_len_m, s.handle_from_join_m,
                        s.finger_half_width_m)
    if found is None:
        return None
    centre, tip, width = found
    across = np.array([-tip[1], tip[0]])
    turn = _turn_to(policy, env, centre, across)
    opening = min(width + 2 * s.handle_open_m, s.max_open_m)
    # Where it held before: closed there and handle_bite_m lower, first try — not a step above it (2026-11-19,
    # user: "the pick of the scoop is also too precarious - move the jaw down slightly to make sure you get it
    # first try"; it closed on air at 42 mm, then held 34 mm up, at its handle's top edge).
    held_at = target.grip_at_m
    start = float(held_at) - s.handle_bite_m if held_at is not None else float(top) - s.handle_step_m
    heights = []
    h = min(start, float(tallest) - s.handle_step_m)
    while h >= s.min_grasp_height_m - 1e-9:
        heights.append(h)
        h -= s.handle_step_m
    if not heights:
        heights = [s.min_grasp_height_m]
    hover_h = float(tallest) + s.tip_below_grasp_m + s.top_clearance_m
    hover = np.array([centre[0], centre[1], s.table_z + hover_h])
    lowest = np.array([centre[0], centre[1], s.table_z + heights[-1]])
    rot, why = straight_down(env, [hover, lowest], turn)
    if why:
        return None                                   # the jaws cannot be turned across it here
    policy._say(TAG, f"By its handle at ({centre[0]:+.3f}, {centre[1]:+.3f}), {width * 1000:.0f} mm across: the jaws "
                     f"turned {turn:+.0f} deg, {opening * 1000:.0f} mm open, closed from {heights[0] * 1000:.0f} mm "
                     f"down until they hold it.")
    env.set_gripper(opening, n_steps=15)
    # Over a container's walls first, if the way there crosses them (spec.transit_over_walls): 2026-11-19
    # 15:06, from the tub to the scoop's handle the arm came down across the tub's wall and pushed it.
    if not policy._over_walls(env, hover, TAG, rot):
        return Outcome(False, "could not get up over a container's walls to cross them", refused=True)
    env.move_ee_to(hover, tol=s.travel_tol_m, orientation=rot)
    if not policy._near(env, hover, 2 * s.travel_tol_m):
        return Outcome(False, "did not get over its handle", refused=True)
    if policy._can_overhead(env):
        corr = policy._overhead_correct(env, hover, s.travel_tol_m, rot, TAG)
        centre = centre + corr
    attempts: list = []
    held, loaded = 0.0, False
    for h in heights:
        grasp = np.array([centre[0], centre[1], s.table_z + h])
        # Learned as the height it held at before, the bite below it the pick's own: or each pick learns
        # its bite lower, and the next goes lower again.
        learned = float(held_at) if (held_at is not None and h == heights[0]) else float(h)
        attempts.append({"attempt": len(attempts) + 1, "approach": "handle", "grip_m": round(learned, 4)})
        if not policy.descend_onto(env, grasp, tol=s.grasp_tol_m, max_step_deg=s.grasp_step_deg, orientation=rot):
            if not policy._near(env, grasp, 0.006):
                attempts[-1]["result"] = "descent_stalled"
                break                                 # on it, not beside it: a close here grips nothing
        env.set_gripper(0.0, n_steps=20, max_step_pct=s.close_step_pct)
        held, loaded = env.squeeze_gripper()
        if loaded and held >= s.handle_min_m / 2:
            attempts[-1]["result"] = "ok"
            break
        attempts[-1]["result"] = "closed_on_nothing"
        attempts[-1]["diagnosis"] = "too_high"
        env.set_gripper(opening, n_steps=10)
    plan = GraspPlan(height_m=float(tallest), width_m=float(np.ptp(poly @ tip)), grasp_height_m=float(grasp[2] - s.table_z),
                     grip_width_m=float(width), pregrasp_open_m=opening, hover_height_m=hover_h,
                     lift_height_m=max(s.lift_height_m, hover_h), release_height_m=float(grasp[2] - s.table_z)
                     + s.release_clearance_m)
    if not attempts or attempts[-1].get("result") != "ok":
        env.move_ee_to(hover, tol=s.travel_tol_m, orientation=rot)
        return Outcome(False, f"closed on nothing across its handle, {heights[0] * 1000:.0f} mm down to "
                              f"{float(grasp[2] - s.table_z) * 1000:.0f}", grasp=plan, attempts=attempts)
    policy._note_grip(env)
    env.move_ee_to(np.array([grasp[0], grasp[1], s.table_z + plan.lift_height_m]), tol=s.travel_tol_m, orientation=rot)
    if not policy.holding(env):
        attempts[-1]["result"] = "grip_lost"
        return Outcome(False, "it left the jaws on the way up", grasp=plan, attempts=attempts)
    return Outcome(True, f"picked up by its handle ({held * 1000:.0f} mm, {plan.grasp_height_m * 1000:.0f} mm up)",
                   grasp=plan, at_xy=grasp[:2].copy(), attempts=attempts)
