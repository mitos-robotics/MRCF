"""
learn_jaw_view.py — a two-finger gripper's wrist camera learns where its own
jaws are in its picture (perception/jaw_view.py), by looking at them.

    for each spot     over a clear stretch of the surface, spec.jaw_view_height_m
                      up: the scene behind the jaws differs from spot to spot
    for each opening  the jaws open that far (spec.jaw_view_openings_m), a moment
                      to settle, one picture
    learn             at each opening, what is dark and joined to the bottom of
                      the picture in every spot's picture: the jaws
    keep              saved where the policy reads it (ParallelJaw.jaw_view_path)

Refused, nothing moved, while something is held (the jaws open) or there is no
wrist camera or nowhere to keep it. Fails, the old view kept, unless at least
spec.jaw_view_spots_min spots were reached and every opening's jaws found.
The spots are the caller's: clear of anything the gripper could touch at that
height, and of anything dark the camera would see joined to the jaws.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from mrcf import flight_recorder
from mrcf.capabilities import OrientationSearch
from mrcf.policies.base import Outcome

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw

TAG = "[JawView]"
SETTLE_S = 0.5             # after the jaws move, before the picture
# A view whose jaws cover more of the picture than this is not the jaws: what
# lay behind them in every picture (beyond the table's edge, the floor) came
# in with them. 2026-11-18, the SO-101's table from 18 cm, three spots along
# one edge: 41-46%, a dark third of the picture learned as jaw.
MAX_SHARE = 0.35


def learn_jaw_view(policy: "ParallelJaw", env, spots, height_m: "float | None" = None) -> Outcome:
    """
    Learn the wrist camera's view of the jaws over `spots` [(x, y)]: moved
    between them height_m above the surface (spec.jaw_view_height_m if None,
    or higher), and down to spec.jaw_view_height_m over each for the
    pictures — from lower the view keeps to the surface, and less lies behind
    the jaws that is the same in every picture. See the module docstring.
    """
    from mrcf.perception import jaw_view

    s = policy.spec
    if s.camera is None:
        return Outcome(False, "no wrist camera to learn the jaws with", refused=True)
    if policy.jaw_view_path is None:
        return Outcome(False, "nowhere to keep what the wrist camera learns of the jaws (jaw_view_path)", refused=True)
    if policy.holding(env):
        return Outcome(False, "the gripper holds something: its jaws cannot be opened to learn them", refused=True)
    openings = sorted({min(float(g), s.max_open_m) for g in s.jaw_view_openings_m})
    frames: dict = {g: [] for g in openings}
    reached = 0
    for xy in spots:
        goal = np.array([float(xy[0]), float(xy[1]), s.table_z + (s.jaw_view_height_m if height_m is None else height_m)])
        rot = None
        if isinstance(env, OrientationSearch):
            found = env.approach_orientation_for([goal])
            if found is None:
                policy._say(TAG, f"({goal[0]:+.3f}, {goal[1]:+.3f}) is out of reach — the next spot.")
                continue
            rot = np.asarray(found[0], float)
        env.move_ee_to(goal, tol=s.travel_tol_m, orientation=rot)
        if not policy._near(env, goal, 2 * s.travel_tol_m):
            policy._say(TAG, f"Did not get over ({goal[0]:+.3f}, {goal[1]:+.3f}) — the next spot.")
            continue
        low = np.array([goal[0], goal[1], s.table_z + s.jaw_view_height_m])
        if goal[2] > low[2] + s.travel_tol_m:
            env.move_ee_to(low, tol=s.travel_tol_m, orientation=rot)     # straight down for the pictures
        got = 0
        for gap in openings:
            env.set_gripper(gap, n_steps=15)
            time.sleep(SETTLE_S)
            frame = policy._wrist_frame(env, TAG)
            if frame is not None:
                frames[gap].append(np.asarray(frame))
                got += 1
        reached += got == len(openings)
        if reached >= s.jaw_view_spots_min:
            break                     # enough: the rest were spares for one out of reach
    if reached < s.jaw_view_spots_min:
        return Outcome(False, f"pictures from only {reached} spot(s), {s.jaw_view_spots_min} needed to tell the jaws "
                              "from what is behind them — the old view kept")
    view = jaw_view.learn({g: f for g, f in frames.items() if len(f) >= s.jaw_view_spots_min})
    empty = [g for g, m in zip(view.openings_m, view.masks) if not m.any()]
    if len(view.openings_m) < len(openings) or empty:
        return Outcome(False, "the jaws were not found in every opening's pictures"
                              + (f" (none at {', '.join(f'{g * 1000:.0f}' for g in empty)} mm)" if empty else "")
                              + " — the old view kept")
    share = [float(m.mean()) for m in view.masks]
    for gap, mask in zip(view.openings_m, view.masks):          # kept whatever is decided: what it saw
        shown = frames[gap][0].copy()
        shown[mask] = (0.5 * shown[mask] + np.array([127, 0, 127])).astype(shown.dtype)
        flight_recorder.keep_picture(f"jaw_view_{gap * 1000:.0f}mm", shown)
    if max(share) > MAX_SHARE:
        return Outcome(False, f"what it learned covers {max(share) * 100:.0f}% of the picture — more than the jaws "
                              "do; what lay behind them came in too — the old view kept")
    # Two jaws with a gap between them at every opening from 20 mm — but at the widest the moving jaw may be
    # out of the picture (the SO-101's at 80 mm, 2026-11-18): one jaw is right there, above the widest that shows two.
    two = [g for g, m in zip(view.openings_m, view.masks) if g >= 0.02 and policy.gap_px(m) is not None]
    one = [g for g, m in zip(view.openings_m, view.masks)
           if g >= 0.02 and policy.gap_px(m) is None and (not two or g < max(two))]
    if one or not two:
        return Outcome(False, "it does not show two jaws with a gap between them at "
                              + ", ".join(f"{g * 1000:.0f}" for g in one) + " mm — the old view kept")
    view.save(policy.jaw_view_path)
    policy._say(TAG, f"Learned where the jaws are at {len(openings)} openings, from {reached} spots: "
                     f"{min(share) * 100:.0f}-{max(share) * 100:.0f}% of the picture.")
    return Outcome(True, f"learned the jaws' view at {len(openings)} openings from {reached} spots")
