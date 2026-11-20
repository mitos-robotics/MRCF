"""
outlines.py — from a segmenter's many candidate outlines to one per object,
and the part of an outline worth reading.

A segmenter prompted at many points finds the same object many times, as
variously good outlines of it, and with them patches of surface and shadow.
keep_outlines picks one per object; core gives an outline's middle, away from
its edge, where any surface or shadow it took in does not reach — what to read
an object's colour from, or whether it has changed from an empty surface.

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

import cv2
import numpy as np

# Two outlines overlapping by more than this share of the smaller are one object.
SAME_OBJECT_IOU = 0.5
# Of two outlines of one object, the smaller wins when it lies inside the other
# (this share of it), is at least this share of its area, and scored at least
# ALT_MIN_SCORE: the larger one carries a shadow or a patch of surface.
ALT_INSIDE = 0.9
ALT_MIN_AREA = 0.4
ALT_MIN_SCORE = 0.5
# ... and when it moves the foot no further than this, or reads it wider. A
# shadow moves it up to ~13 mm (0.4-10 mm on 2026-11-04's frames); the top of
# a tall object seen from the side also lies inside its outline, narrower, and
# moves it 20-40 mm.
ALT_MAX_FOOT_SHIFT_MM = 15.0
# An outline that took in two objects side by side: the surer outlines it
# swallowed have feet further apart than their two foot radii, and cover at
# least this share of it between them. It is replaced by them.
SPLIT_MIN_COVER = 0.6


def _foot_shift_mm(a, b) -> float:
    """How far apart two measurements put the foot (their base_mm); 0 when either does not say."""
    pa, pb = getattr(a, "base_mm", None), getattr(b, "base_mm", None)
    if pa is None or pb is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(pa, float) - np.asarray(pb, float)))


def core(mask: np.ndarray, min_px: int = 30) -> np.ndarray:
    """
    The middle of an outline: its pixels more than half its greatest depth from
    the edge. The whole outline when that is no more than min_px pixels.
    """
    m = np.asarray(mask, bool)
    dist: np.ndarray = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    middle = dist > 0.5 * dist.max()
    return middle if middle.sum() > min_px else m


def keep_outlines(found: list) -> list:
    """
    One outline per object, from candidates [(score, mask, measurement)] of
    which many are the same object found by different prompts.
    """
    # Many prompts find the same object, as variously good outlines of it. Keep
    # the one the segmenter is surest of. Not the largest: seen from above,
    # parts of an object are rare, but an outline that took in the object and a
    # patch of surface beside it is not, and it is always the largest — keeping
    # it makes the same object's foot read differently from scan to scan.
    #
    # But the surest outline can still carry the object's shadow. A shadow or a
    # patch of surface only ever ADDS to an object's outline, on the side it
    # falls, and the foot is read from the outline's edge nearest the camera —
    # so with the light behind the board and the camera behind the arm,
    # the surest outline can put an object up to ~13 mm nearer the camera than
    # it is, while a smaller outline of the object alone reads steady. Not
    # always with a narrower foot: the shadow may fall only on the near side.
    # So the smallest outline lying inside the kept one replaces it — when it
    # reads the foot near where the larger one did: the top of a tall object
    # seen from the side also lies inside its outline, and its "foot" is the
    # object's shoulder, centimetres off (2026-11-04: a queen's head put her
    # off the board, and the scan lost her).
    found.sort(key=lambda f: -f[0])
    groups: list = []                  # [the surest outline, the others it is the same object as]
    for cand in found:
        same = next((g for g in groups
                     if (cand[1] & g[0][1]).sum() > SAME_OBJECT_IOU * min(cand[1].sum(), g[0][1].sum())), None)
        if same is None:
            groups.append([cand, []])
        else:
            same[1].append(cand)
    kept: list = []
    for top, others in groups:
        parts = _parts(top, others)
        if parts:
            kept.extend(_refined(p, [o for o in others if o is not p]) for p in parts)
        else:
            kept.append(_refined(top, others))
    return kept


def _refined(top, others):
    """The outline that stands for top's object: a smaller one inside it, without its shadow (see keep_outlines)."""
    best = top
    for score, m, meas in others:
        _, k_m, k_meas = best
        if (score >= ALT_MIN_SCORE and m.sum() < k_m.sum()
                and (m & k_m).sum() >= ALT_INSIDE * m.sum()
                and m.sum() >= ALT_MIN_AREA * k_m.sum()
                and (_foot_shift_mm(meas, k_meas) <= ALT_MAX_FOOT_SHIFT_MM or _wider(meas, k_meas))):
            best = (score, m, meas)
    return best


def _parts(top, others) -> list:
    """
    The objects an outline took in, when it took in two or more side by side;
    [] when it is one (SPLIT_MIN_COVER). 2026-11-04: one outline took in the
    f1 bishop and the f2 pawn beside it, and the scan lost one of them. Each
    part must be a separate object by its foot — further from the others than
    their two foot radii — which one object's top, its foot read near its own,
    never is.
    """
    _, k_m, _ = top
    parts: list = []
    for cand in others:                                    # surest first
        score, m, meas = cand
        if score < ALT_MIN_SCORE or m.sum() >= k_m.sum() or getattr(meas, "base_mm", None) is None:
            continue
        if all(_apart(meas, p[2]) and (m & p[1]).sum() < 0.1 * min(m.sum(), p[1].sum()) for p in parts):
            parts.append(cand)
    if len(parts) < 2:
        return []
    union = np.zeros_like(k_m)
    for _, m, _ in parts:
        union |= m
    return parts if (union & k_m).sum() >= SPLIT_MIN_COVER * k_m.sum() else []


def _wider(a, b) -> bool:
    """Whether a reads a wider foot than b (2026-11-03: a 4 mm sliver of a pawn, replaced by its whole foot)."""
    ra, rb = getattr(a, "base_radius_mm", None), getattr(b, "base_radius_mm", None)
    return ra is not None and rb is not None and float(ra) > float(rb)


def _apart(a, b) -> bool:
    """Whether two measurements' feet are separate objects': further apart than their two radii."""
    reach = float(getattr(a, "base_radius_mm", 0.0) or 0.0) + float(getattr(b, "base_radius_mm", 0.0) or 0.0)
    return _foot_shift_mm(a, b) > reach
