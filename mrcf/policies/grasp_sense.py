"""
grasp_sense.py — before the jaws come down, does the grasp make sense: will
they open across the thing, a finger either side of it, or come down on it?

Asked of the orientation the arm will really take (the rotation its search
gave, the jaws' way of opening as mounted, capabilities.JawDirection), not
of the plan's intent: an arm that cannot turn its wrist as far as asked, or a
turn worked out the wrong way round, shows here before anything touches the
table. 2026-11-20, the UR5: the jaws came down along a hammer's handle, on it,
at 50 N (user: "there should be some basic common sense - like how could a
hammer be picked up like this - this needs to be somewhere in the platform").

Geometry only: the thing's outline from above on the surface, where the jaws
are to close, which way they open, how wide, and the fingers' size.
"""

from __future__ import annotations

import numpy as np

BULGE_M = 0.003                  # an outline this much wider than the thing at a finger's inner edge: let be
MAX_OFF_ACROSS_DEG = 20.0        # the jaws open within this of square across the thing where it is gripped


def _inside(poly: np.ndarray, xy) -> bool:
    import cv2
    return cv2.pointPolygonTest(np.asarray(poly, np.float32).reshape(-1, 1, 2), (float(xy[0]), float(xy[1])),
                                False) >= 0


def jaw_direction(env, rot) -> "np.ndarray | None":
    """The way the jaws open on the surface (unit xy) with the gripper at rotation rot; None: the env does not say."""
    from mrcf.capabilities import JawDirection

    if rot is None or not isinstance(env, JawDirection):
        return None
    return env.jaw_direction(rot)


def fingers_on_it(poly, centre, jaw, across, opening_m: float, finger_thickness_m: float,
                  finger_half_width_m: float) -> "str | None":
    """
    Why the open jaws at `centre`, opening along `jaw`, would not close across
    the outline `poly` (world xy, m) — they are turned more than
    MAX_OFF_ACROSS_DEG off `across` (the way they should open there), or a
    finger's footprint lands on it; None: a finger either side of it.
    """
    jaw = np.asarray(jaw, float) / (np.linalg.norm(jaw) or 1.0)
    across = np.asarray(across, float) / (np.linalg.norm(across) or 1.0)
    off = float(np.degrees(np.arccos(min(1.0, abs(float(jaw @ across))))))
    if off > MAX_OFF_ACROSS_DEG:
        return f"the jaws would open {off:.0f} deg off square across it: they would come down on it"
    along = np.array([-jaw[1], jaw[0]])
    centre = np.asarray(centre, float)
    # Each finger's middle, and half its width along the thing: on the outline there, it would come down ON it.
    # Not its outer edges — an angled camera's outline takes in the thing's sides, a few mm wider than its top,
    # and a grip the jaws close round anyway was refused for it (2026-11-20 17:25, the UR5's tape measure).
    for side in (-1.0, 1.0):
        for out in (opening_m / 2 + min(BULGE_M, finger_thickness_m / 2), opening_m / 2 + finger_thickness_m / 2):
            mid = centre + jaw * side * out
            for a in np.linspace(-finger_half_width_m / 2, finger_half_width_m / 2, 3):
                if _inside(poly, mid + along * a):
                    return "a finger would come down on it, not beside it"
    return None
