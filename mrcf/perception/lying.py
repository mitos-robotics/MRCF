"""
lying.py — an object lying on its side, from its outline in a fixed camera:
which way it lies, how long it is, how wide along its length, and which end it
would stand on.

plane_geometry.measure reads an outline as an UPRIGHT object: a round foot, and
a height up the line where verticals converge. A piece that has fallen over has
neither. Seen from above it is a long shape on the surface: its length is the
object's height, and its width along that length is the object's profile.

The silhouette's edge is made of points at different heights — a lying piece's
sides at about half its width up, the rim of its foot at its full width — and
each pixel is put on a plane at ONE height, so where a point really is and
where it is put differ by (its height - the plane's) x tan(the camera's slant).
The whole outline goes on the plane at half its widest (the sides), and the top
end's pixels again on the plane at half the top's width (a knob's middle), the
part a gripper is aimed at. From the overhead camera (slant ~15-25 deg over the
board), a pawn's head comes out within a millimetre or two.

The end it would stand on is the wider one — a chess piece's felt foot, a
bottle's base. When the ends are within END_RATIO of each other, it cannot say.

Nothing here knows what the object is, or about any arm: geometry in, geometry
out. find_lying looks at a fixed camera's frame now (surface_scan.scan, over a
small patch round where the object was) and measures the outline nearest.
find_lying_in_frame does it in a wrist camera's frame, through where its
pixels land on the surface (SurfaceMap) — the camera need not be located.

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import cv2
import numpy as np

from mrcf.perception import plane_geometry as pg

BIN_MM = 2.0            # the width profile's step along the object
END_SPAN = 0.25         # each end's share of the length, for telling foot from top
END_RATIO = 1.15        # the foot end must be this much wider than the top
MIN_LENGTH_MM = 15.0
# A whole object on its side is this many times as long as it is wide, or more:
# a pawn 1.7, the rest of a chess set more. A board square outlined with part
# of a piece on it is ~1.0-1.1; an upright pawn from the overhead camera 1.5-1.9
# (its height laid along the camera's line of sight) — but its ends read alike.
MIN_ELONGATION = 1.3
# Of an outline taken for a lying object, at least this share differs from the
# empty surface (surface_scan.change_map) — more than a scan asks of an
# upright one (MIN_CHANGED). The longer outlines a segmenter offers round a
# lying piece are the piece with a patch of surface, and can be shaped like a
# lying piece themselves; such an outline differs over about half of it
# (0.57), the piece itself over 0.9.
LYING_MIN_CHANGED = 0.75
# An UPRIGHT object seen at a slant is long too — its height laid along the
# line from the camera's foot through it, its top outward — and a head over its
# foot makes one end the narrower. An outline within UPRIGHT_LINE_DEG of
# that line with its narrow end outward may be either, so it is taken for
# upright: a lying piece pointing that way is not stood up from this camera.
# A lying one's runs any way, and with its top inward it cannot be upright.
UPRIGHT_LINE_DEG = 25.0
UPRIGHT_MIN_ELONGATION = 1.2
# In a frame with nothing to tell object from surface by (find_lying_in_frame),
# an outline is taken for the object when its length is within this share of
# the length measured before; it may be put together from up to MAX_PARTS of
# the segmenter's outlines, of the first MAX_CANDIDATE_PARTS near it.
LENGTH_AGREE = 0.3
MAX_PARTS = 3
MAX_CANDIDATE_PARTS = 12


@dataclass
class SurfaceMap:
    """
    A camera known only by where its pixels land: to_surface(px (N, 2),
    height_m) -> (N, 2) world metres on the plane height_m above the surface,
    or None. A wrist camera with no located pose, through its servo's
    calibration (capabilities.WristSurface). Stands in for a PlaneView in
    measure_lying.
    """
    to_surface: Callable


@dataclass
class LyingMeasurement:
    """A lying object, world frame (x, y metres; sizes real mm)."""
    foot_end_m: np.ndarray               # the end it would stand on, at the surface
    top_end_m: np.ndarray                # the other end
    top_m: np.ndarray                    # the middle of the top's knob (what to grip): a top-width back from its end
    length_mm: float
    top_width_mm: float                  # widest within the knob
    foot_width_mm: float                 # widest over the foot end's END_SPAN
    widths_mm: list                      # [(mm from the foot end, width mm)], every BIN_MM
    ends_certain: bool                   # the foot end is clearly the wider one
    elongation: float                    # length / widest

    @property
    def axis(self) -> np.ndarray:
        """Unit (x, y) along it, from its foot end to its top."""
        d = np.asarray(self.top_end_m, float) - np.asarray(self.foot_end_m, float)
        return d / (np.linalg.norm(d) or 1.0)

    def widest_within(self, from_top_mm: float, span_mm: float) -> float:
        """The widest it is over span_mm either side of from_top_mm back from its top end."""
        centre = self.length_mm - from_top_mm
        near = [w for at, w in self.widths_mm if abs(at - centre) <= span_mm]
        return max(near) if near else self.top_width_mm

    def as_dict(self) -> dict:
        return {"foot_end_m": [round(float(v), 4) for v in self.foot_end_m],
                "top_end_m": [round(float(v), 4) for v in self.top_end_m],
                "top_m": [round(float(v), 4) for v in self.top_m],
                "length_mm": round(self.length_mm, 1), "top_width_mm": round(self.top_width_mm, 1),
                "foot_width_mm": round(self.foot_width_mm, 1), "ends_certain": self.ends_certain,
                "elongation": round(self.elongation, 2)}


def looks_upright(m: LyingMeasurement, view: pg.PlaneView) -> bool:
    """Whether the outline m was measured from could be an upright object seen at a slant (UPRIGHT_LINE_DEG)."""
    if m.elongation < UPRIGHT_MIN_ELONGATION:
        return False
    foot = view.world_xy_m(view.centre[:2])
    mid = (np.asarray(m.foot_end_m) + np.asarray(m.top_end_m)) / 2
    out = mid - foot
    out = out / (np.linalg.norm(out) or 1.0)
    return float(m.axis @ out) > np.cos(np.radians(UPRIGHT_LINE_DEG))


def at_height(view: pg.PlaneView, height_mm: float) -> pg.PlaneView:
    """The same camera, with its plane raised height_mm (real) off the surface: to_plane then lands there."""
    return dataclasses.replace(view, t=view.t + view.R @ (view.up * height_mm / view.scale))


def _plane_points(mask: np.ndarray, view: "pg.PlaneView | SurfaceMap",
                  height_mm: float) -> "tuple[np.ndarray, np.ndarray]":
    """The mask's pixels and where they are (real mm) on the plane height_mm up."""
    ys, xs = np.nonzero(mask)
    px = np.c_[xs, ys].astype(float)
    if isinstance(view, SurfaceMap):
        got = view.to_surface(px, height_mm / 1000.0)
        pts = np.full((len(px), 2), np.nan) if got is None else np.asarray(got, float) * 1000.0
    else:
        pts = at_height(view, height_mm).to_plane(px) * view.scale
    ok = np.isfinite(pts).all(1)
    return px[ok], pts[ok]


def _profile(pts: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray, list]":
    """Centre, unit axis (longest direction), each point's (along, across) and the width every BIN_MM along."""
    centre = np.median(pts, 0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0]
    along, across = (pts - centre) @ axis, (pts - centre) @ np.array([-axis[1], axis[0]])
    lo = np.percentile(along, 0.5)
    widths = []
    for a in np.arange(lo, np.percentile(along, 99.5), BIN_MM):
        band = (along >= a) & (along < a + BIN_MM)
        if band.sum() >= 3:
            widths.append((float(a - lo + BIN_MM / 2), float(np.percentile(across[band], 99)
                                                             - np.percentile(across[band], 1))))
    return centre, axis, np.c_[along, across], widths


def measure_lying(mask: np.ndarray, view: "pg.PlaneView | SurfaceMap") -> "LyingMeasurement | None":
    """
    The outlined object as one lying on its side, or None when the outline is
    too small or not long enough to have a direction. view: the camera's
    PlaneView in the world frame (PlaneView.from_camera_model), or a
    SurfaceMap.
    """
    mask = np.asarray(mask, bool)
    if mask.sum() < 30 or (not isinstance(view, SurfaceMap) and view.origin_m is None):
        return None
    # First pass at a guess at half a small object's width, then again at half
    # of what it measured: where its sides really are.
    height = 10.0
    for _ in range(2):
        _, pts = _plane_points(mask, view, height)
        if len(pts) < 30:
            return None
        centre, axis, aa, widths = _profile(pts)
        if not widths:
            return None
        height = max(w for _, w in widths) / 2
    along = aa[:, 0]
    lo, hi = float(np.percentile(along, 0.5)), float(np.percentile(along, 99.5))
    length = hi - lo
    widest = max(w for _, w in widths)
    if length < MIN_LENGTH_MM:
        return None
    span = END_SPAN * length
    first = max(w for at, w in widths if at <= span)
    last = max(w for at, w in widths if at >= length - span)
    if last > first:                     # the foot is the wider end: make the profile run from it
        axis, along, lo, hi = -axis, -along, -hi, -lo
        widths = [(length - at, w) for at, w in reversed(widths)]
        first, last = last, first
    foot_end = centre + axis * lo
    top_end = centre + axis * hi
    # The knob at the top end, and its middle: those pixels again, on the plane
    # at half the knob's width — where its middle really is.
    # The knob's width: the top END_SPAN's widest. Not over a knob's width back
    # from the end, which can reach a collar below the knob (on a pawn,
    # 16.5-17 mm for a 13 mm head) and put its middle short.
    top_w = last
    # Across the axis, the knob's own pixels say where its middle is; along it,
    # half its width back from the end (its pixels run on into the neck).
    _, pts = _plane_points(mask, view, top_w / 2)
    side = np.array([-axis[1], axis[0]])
    knob = ((pts - centre) @ axis) - hi >= -top_w
    off = float((pts[knob].mean(0) - top_end) @ side) if knob.sum() >= 5 else 0.0
    top = top_end - axis * top_w / 2 + side * off

    def world(p):
        return p / 1000.0 if isinstance(view, SurfaceMap) else view.world_xy_m(p / view.scale)

    return LyingMeasurement(foot_end_m=world(foot_end), top_end_m=world(top_end), top_m=world(top),
                            length_mm=float(length), top_width_mm=float(top_w), foot_width_mm=float(first),
                            widths_mm=widths, ends_certain=first >= END_RATIO * top_w,
                            elongation=float(length / widest))


def find_lying(env, camera: str, width: int, height: int, near_xy_m, *, radius_m: float = 0.05,
               reference=None, surface_fraction=None, max_length_mm: float = 150.0,
               ) -> "tuple[LyingMeasurement | None, str]":
    """
    The object lying at near_xy_m as `camera` sees it now (the arm out of its
    view): (measurement, "") or (None, why not).

    Every outline the segmenter finds over a square 2 x radius_m across round
    near_xy_m that is object, not surface (reference and surface_fraction as
    in surface_scan.scan), comes within 15 mm of near_xy_m and is shaped like
    a whole object on its side: at least MIN_ELONGATION as long as it is wide,
    no longer than max_length_mm (the tallest it could be), one end clearly the
    wider. Of those, the LONGEST. Not the surest, as a scan keeps: a lying
    object across two surfaces can come out surest as its half on one of
    them, and whole only as a less sure outline. A part is the usual mistake
    here, and a part is shorter.

    It can fail to see one: a pale end on a pale surface (cream on maple, 7-14
    apart in Lab) is lost by the segmenter and the change map both, and the
    object is never outlined whole. Then it says so (None) rather than return
    the part it did outline, whose axis is a guess. And when any outline there
    could be an upright object (looks_upright), it says it may be standing:
    an upright object's outline with its shadow can read as one lying, its
    ends unequal.
    """
    from mrcf.perception import segment, surface_scan

    model = env.camera_model(camera, width, height)
    if model is None or not segment.available():
        return None, f"cannot outline objects in the {camera} camera (no camera model or no segmenter)"
    view = pg.PlaneView.from_camera_model(model)
    x, y = float(near_xy_m[0]), float(near_xy_m[1])
    frame = env.render_image(camera, width=width, height=height)
    prompts = pg.surface_grid_px(view, np.array([x - radius_m, x + radius_m]) * 1000 / view.scale,
                                 np.array([y - radius_m, y + radius_m]) * 1000 / view.scale, 8.0, lift_mm=6.0)
    h, w = frame.shape[:2]
    prompts = prompts[(prompts[:, 0] >= 2) & (prompts[:, 0] < w - 2) & (prompts[:, 1] >= 2) & (prompts[:, 1] < h - 2)]
    changed = surface_scan.change_map(frame, reference) if reference is not None else None
    at = view.project(np.array([[x * 1000 / view.scale, y * 1000 / view.scale, 0.0]]))[0]
    best, longest, seen, unshaped, upright = None, 0.0, 0, "", False
    for _, mask, _ in segment.candidates(frame, prompts):
        m = pg.clean_mask(mask)
        if pg.touches_frame(m):
            continue
        if changed is not None and changed[m].mean() < LYING_MIN_CHANGED:
            continue
        if surface_fraction is not None and surface_fraction(m) > surface_scan.MAX_SURFACE:
            continue
        ys, xs = np.nonzero(m)
        if np.min(np.hypot(xs - at[0], ys - at[1])) * _mm_per_px(view, (x, y)) > 15.0:
            continue
        meas = measure_lying(m, view)
        if meas is None:
            continue
        seen += 1
        upright = upright or looks_upright(meas, view)
        why = (f"{meas.length_mm:.0f} mm long, longer than it could be" if meas.length_mm > max_length_mm
               else f"only {meas.elongation:.1f} times as long as it is wide" if meas.elongation < MIN_ELONGATION
               else "its two ends are as wide as each other" if not meas.ends_certain else "")
        if why:
            unshaped = unshaped or why
            continue
        if meas.length_mm > longest:
            best, longest = meas, meas.length_mm
    if upright:
        return None, ("it may be standing: its outline runs straight away from the camera, top outward, "
                      "as an upright object's does (or it lies pointing that way)")
    if best is None:
        if not seen:
            return None, f"nothing outlined at ({x:+.3f}, {y:+.3f}) that differs from the empty surface"
        return None, (f"no outline there looks like a whole object on its side ({unshaped}) — standing, or "
                      "part of it lost against the surface")
    return best, ""


def find_lying_in_frame(frame_rgb: np.ndarray, surface: SurfaceMap, near_px,
                        expected: "LyingMeasurement | None" = None, *, radius_px: float = 160.0,
                        step_px: float = 24.0, max_shift_m: float = 0.025, max_length_mm: float = 150.0,
                        exclude: "np.ndarray | None" = None) -> "tuple[LyingMeasurement | None, str]":
    """
    The object lying near pixel near_px in one frame from a camera known by
    where its pixels land (a wrist camera at a hover over it): (measurement,
    "") or (None, why not). Outlines that run off the frame or into `exclude`
    (the jaws, where a backend can say) are not it.

    There is no picture of the empty surface to tell object from surface by,
    as find_lying has, so what was measured before (expected) decides — a
    miss moves an object a little, not a lot. And close up, the segmenter
    can outline a lying object in parts — its head, its foot — and never
    whole: so the outlines near where it was
    are put together, up to MAX_PARTS touching ones, and each whole measured
    as one object on its side. Of those shaped like one (as find_lying), with
    its knob within max_shift_m of expected's and its length within
    LENGTH_AGREE, the one that agrees best with expected — length, then the
    widths of its knob and foot, then where it lies. A square of the board
    with it has two ends alike, and is not.

    Without expected, the longest whole outline shaped like one.
    """
    from mrcf.perception import segment

    if not segment.available():
        return None, segment.why_unavailable()
    h, w = frame_rgb.shape[:2]
    cx, cy = float(near_px[0]), float(near_px[1])
    gx, gy = np.meshgrid(np.arange(cx - radius_px, cx + radius_px + 1, step_px),
                         np.arange(cy - radius_px, cy + radius_px + 1, step_px))
    prompts = np.c_[gx.ravel(), gy.ravel()]
    prompts = prompts[(np.hypot(prompts[:, 0] - cx, prompts[:, 1] - cy) <= radius_px)
                      & (prompts[:, 0] >= 2) & (prompts[:, 0] < w - 2) & (prompts[:, 1] >= 2) & (prompts[:, 1] < h - 2)]
    if exclude is not None:
        prompts = prompts[~exclude[prompts[:, 1].astype(int), prompts[:, 0].astype(int)]]
    if expected is not None:
        middle = (np.asarray(expected.foot_end_m, float) + np.asarray(expected.top_end_m, float)) / 2
        reach = expected.length_mm / 2000.0 + max_shift_m
    parts = []
    for _, mask, _ in segment.candidates(frame_rgb, prompts, max_area_frac=0.08):
        m = pg.clean_mask(mask)
        if pg.touches_frame(m) or (exclude is not None and (m & exclude).sum() > 0.05 * m.sum()):
            continue
        if expected is not None:
            # A part of it lies wholly where it could be now.
            ys, xs = np.nonzero(m)
            pick = np.linspace(0, len(xs) - 1, min(len(xs), 200)).astype(int)
            at = surface.to_surface(np.c_[xs[pick], ys[pick]].astype(float), 0.005)
            if at is None or np.linalg.norm(np.asarray(at) - middle, axis=1).max() > reach:
                continue
        parts.append(m)
        if len(parts) >= MAX_CANDIDATE_PARTS:
            break
    wholes: list[tuple[np.ndarray, ...]] = [(p,) for p in parts]
    if expected is not None:
        wholes += [c for k in range(2, MAX_PARTS + 1) for c in combinations(parts, k)]
    best, best_score, seen, unshaped = None, np.inf, 0, ""
    grow = np.ones((5, 5), np.uint8)
    for group in wholes:
        m = np.logical_or.reduce(group)  # type: ignore[call-overload]  # numpy's stubs hide reduce()'s plain overload
        if len(group) > 1:
            n, _ = cv2.connectedComponents(cv2.dilate(m.astype(np.uint8), grow))
            if n > 2:                           # not touching: not one object
                continue
        meas = measure_lying(m, surface)
        if meas is None:
            continue
        seen += 1
        why = (f"{meas.length_mm:.0f} mm long, longer than it could be" if meas.length_mm > max_length_mm
               else f"only {meas.elongation:.1f} times as long as it is wide" if meas.elongation < MIN_ELONGATION
               else "its two ends are as wide as each other" if not meas.ends_certain else "")
        if not why and expected is not None:
            off = abs(meas.length_mm - expected.length_mm) / expected.length_mm
            shift = float(np.linalg.norm(np.asarray(meas.top_m) - np.asarray(expected.top_m, float)))
            why = (f"{meas.length_mm:.0f} mm long, not the {expected.length_mm:.0f} it was" if off > LENGTH_AGREE
                   else f"its top {shift * 1000:.0f} mm from where it was" if shift > max_shift_m else "")
            score = (off + 0.5 * abs(meas.top_width_mm - expected.top_width_mm) / expected.top_width_mm
                     + 0.5 * abs(meas.foot_width_mm - expected.foot_width_mm) / expected.foot_width_mm
                     + 0.5 * shift / max_shift_m)
        else:
            score = -meas.length_mm
        if why:
            unshaped = unshaped or why
            continue
        if score < best_score:
            best, best_score = meas, score
    if best is None:
        if not seen:
            return None, "nothing outlined there"
        return None, f"no outline there is the object on its side ({unshaped})"
    return best, ""


def _mm_per_px(view: pg.PlaneView, xy_m) -> float:
    """Real mm per pixel on the surface round world point xy_m."""
    p = np.array([xy_m[0], xy_m[1]]) * 1000 / view.scale
    a, b = view.project(np.array([[*p, 0.0], [p[0] + 10.0, p[1], 0.0]]))
    return 10.0 * view.scale / float(np.linalg.norm(b - a))
