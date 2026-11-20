"""
shape_fit.py — how tall an object is, and how wide up its height, from its
outlines in any number of calibrated views.

One view seldom settles it. Looking down at an object, a camera sees its top
and not how far that stands off the surface: on the SO-101 tabletop
(2026-11-16) the wrist camera over a 25 mm ball, 8 deg from straight down,
moved its top 0.21 px per mm of height, and every pick closed in the air
above it. From the side a mm of height spans a pixel or more, but one outline
still leaves the top's place along the line of sight open by the object's
depth: traced back to the vertical through the foot, a flat top 40 mm across
reads 20-30 mm too tall at 40 deg.

So the object is fitted, not traced. A solid stands on the surface: its
cross-section is the footprint (an ellipse, from where the scan saw it), scaled
up its height by a profile — straight sides, a rounded top, a dome, a ball, a
taper. Cast into each calibrated view (a wrist camera's at each look, the
overhead camera's at the scan), it must cover the outline seen there: each view
scores the cast by intersection over union, over the pixels where the object
could have been seen at all (not hidden by a neighbour, the gripper, the frame's
edge). The height, profile, footprint scale and centre that score best across
every view at once are the fit; the heights that score nearly as well are its
spread, and the height counts as resolved only when that spread is narrow. A
view that sees height poorly (from straight above) barely changes its score
with height, so it settles the footprint and leaves the height to the views
that can see it.

Geometry only: what draws the outlines (the segmenter) and where the cameras
are (their calibration) come from elsewhere.

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from mrcf.perception.plane_geometry import PlaneView

# ── The solids tried ───────────────────────────────────────────────────────────
# Cross-section scale up the height (u: 0 at the foot, 1 at the top), each
# reaching 1 (the footprint) somewhere: the overhead camera sees an object's
# widest cross-section as its outline, so the footprint is that one.
PROFILES = {
    "straight": lambda u: np.ones_like(u),                                     # a box, a can, a cup
    "rounded": lambda u: np.sqrt(np.clip(1.0 - np.clip((u - 0.6) / 0.4, 0.0, 1.0) ** 2, 0.0, 1.0)),
                                                                                # straight, then a rounded top
    "dome": lambda u: np.sqrt(np.clip(1.0 - u ** 2, 0.0, 1.0)),                # half an ellipsoid
    "ball": lambda u: np.sqrt(np.clip(1.0 - (2.0 * u - 1.0) ** 2, 0.0, 1.0)),  # widest halfway up
    "taper": lambda u: 1.0 - 0.55 * u,                                         # narrowing to the top
}
_MIN_SCALE = 0.08                       # no cross-section thinner than this share: the cast stays a solid
_LEVELS = np.unique(np.r_[np.linspace(0.0, 1.0, 11), 0.95, 0.98, 1.0, 0.02, 0.05])
_ANGLES = np.linspace(0.0, 2 * np.pi, 20, endpoint=False)

# ── What counts as settled ─────────────────────────────────────────────────────
MIN_FIT_IOU = 0.70          # below this the solid does not look like what was outlined
SPREAD_DROP = 0.015         # heights within this of the best score (plus the shortfall's share,
                            # below) fit as well as it: the spread
SPREAD_SHORTFALL = 0.06     # ... a worse best fit widens the spread by this share of what it lacks
RESOLVED_MM = 6.0           # a height is resolved when its spread is within this ...
RESOLVED_SHARE = 0.20       # ... or this share of it, whichever is more
SEES_HEIGHT_PX_PER_MM = 0.6 # ... and some view the solid fits well sees height: a mm of it moves
                            # its top this many px. From above (0.2-0.3) an outline's parallax still
                            # favours some heights, and an object unlike every solid (a bottle's neck)
                            # can then fit one confidently and wrongly: 50 mm for a 70 mm bottle
MAX_CROP_PX = 192           # each view is scored on a crop around the object, at most this many px a side
COARSE_STEP_MM = 3.0


@dataclass
class Footprint:
    """
    Where an object stands, and how wide: an ellipse in the fit's frame, the
    same at every height (what a pick knows of it without an outline from
    above: the scan's foot). centre_mm is plane units (the world frame's mm,
    PlaneView.from_camera_model); the half axes are real mm.
    """
    centre_mm: np.ndarray
    half_axes_mm: "tuple[float, float]"
    angle_rad: float = 0.0

    @classmethod
    def circle(cls, centre_xy_m, diameter_m: float) -> "Footprint":
        r = float(diameter_m) * 500.0
        return cls(np.asarray(centre_xy_m, float)[:2] * 1000.0, (r, r), 0.0)

    @property
    def centre_xy_m(self) -> np.ndarray:
        return np.asarray(self.centre_mm, float) / 1000.0

    @property
    def width_mm(self) -> float:
        return 2.0 * float(max(self.half_axes_mm))

    def section(self, scale: float, height_mm: float = 0.0, profile: str = "straight") -> "tuple[np.ndarray, np.ndarray]":
        """(centre in plane units, (N, 2) ring of its widest cross-section about it in real mm)."""
        a, b = self.half_axes_mm
        ca, sa = np.cos(self.angle_rad), np.sin(self.angle_rad)
        ex, ey = a * np.cos(_ANGLES), b * np.sin(_ANGLES)
        return np.asarray(self.centre_mm, float)[:2], np.c_[ca * ex - sa * ey, sa * ex + ca * ey]

    def moved(self, offset_mm, k: float, scale: float = 1.0) -> "Footprint":
        a, b = self.half_axes_mm
        return Footprint(np.asarray(self.centre_mm, float)[:2] + np.asarray(offset_mm, float) / scale,
                         (a * k, b * k), self.angle_rad)


# Where each solid is widest, as shares of its height: its outline seen from
# above is that cross-section's, and every height in the range must fit
# inside what the outline allows there.
WIDEST = {"straight": (0.0, 1.0), "rounded": (0.0, 0.6), "dome": (0.0, 0.0), "ball": (0.5, 0.5), "taper": (0.0, 0.0)}


def _across(ring_mm) -> float:
    """The widest a cross-section (N, 2, real mm) is, any way across."""
    r = np.asarray(ring_mm, float)
    if len(r) < 2:
        return 0.0
    d = r[:, None, :] - r[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())


class OutlineFootprint:
    """
    The footprint as a fixed camera's outline of the object says (the scan's),
    for each height and profile tried: the outline's rays carried to the
    heights where that solid is widest (WIDEST), and what lies inside all of
    them. For a straight-sided object that is its true cross-section: the
    outline's near edge is its foot's, its far edge its top's, and each carried
    to the surface alone is pushed away from the camera by the object's height
    (on the SO-101 tabletop 7 mm on a 30 mm box 0.12 m from below the camera).
    """

    def __init__(self, view: PlaneView, outline_px: np.ndarray, centre_mm=None):
        hull = cv2.convexHull(np.asarray(outline_px, np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
        if len(hull) > 32:
            hull = cv2.approxPolyDP(hull.reshape(-1, 1, 2), 0.7, True).reshape(-1, 2)
        self.view, self.px = view, hull.astype(float)
        self._cache: dict = {}
        got = self.section(1.0, 0.0, "dome")
        c = np.zeros(2) if got is None else got[0]
        self.centre_mm = np.asarray(c if centre_mm is None else centre_mm, float)[:2]

    def _at(self, z_mm: float) -> np.ndarray:
        """The outline's rays where they cross z_mm (real) above the surface, plane units."""
        v = self.view
        r = v.rays(self.px)
        rp = r @ v.R                         # ray directions in the plane frame
        o = -v.R.T @ v.t                     # the camera centre in the plane frame
        z0 = float(v.up[2]) * z_mm / v.scale
        with np.errstate(all="ignore"):
            s = (z0 - o[2]) / rp[:, 2]
        return o[:2] + s[:, None] * rp[:, :2]

    def section(self, scale: float, height_mm: float, profile: str) -> "tuple[np.ndarray, np.ndarray] | None":
        """(centre in plane units, (N, 2) cross-section about it in real mm), or None: no such solid fits the outline."""
        lo, hi = WIDEST[profile]
        key = (round(height_mm * lo, 1), round(height_mm * hi, 1))
        if key not in self._cache:
            poly = self._at(key[0]).astype(np.float32)
            if key[1] != key[0]:
                area, both = cv2.intersectConvexConvex(poly, self._at(key[1]).astype(np.float32))
                if area <= 1.0 or both is None or len(both) < 3:
                    self._cache[key] = None
                    return None
                poly = np.asarray(both, np.float32).reshape(-1, 2)
            if not np.isfinite(poly).all():
                self._cache[key] = None
                return None
            centre = poly.mean(0)
            m = cv2.moments(poly.reshape(-1, 1, 2))
            if m["m00"] > 1e-6:
                centre = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            self._cache[key] = (centre, (poly - centre) * self.view.scale)
        return self._cache[key]

    @property
    def centre_xy_m(self) -> np.ndarray:
        return self.centre_mm / 1000.0

    @property
    def width_mm(self) -> float:
        got = self.section(1.0, 0.0, "dome")
        return 0.0 if got is None else _across(got[1])


@dataclass
class Silhouette:
    """
    An object as one calibrated view saw it: the view, in the fit's frame
    (PlaneView.from_camera_model, shifted into it if need be: shifted()),
    candidate outlines of the object, best first (bool masks the frame's
    size), and where it could have been seen (valid; None: anywhere in the
    frame). Off valid, an outline and a cast neither agree nor disagree.
    """
    view: PlaneView
    masks: list
    valid: "np.ndarray | None" = None
    name: str = ""
    fixed: bool = False                 # a fixed camera's: sees the footprint, rarely the height


@dataclass
class ShapeFit:
    """The solid that matches the outlines best, and how sure that is."""
    height_mm: float
    profile: str
    footprint: Footprint                # as fitted: centre and scale moved from the one given
    widths_mm: dict                     # {height_mm: width_mm} up it, the widest way across
    score: float                        # mean intersection over union, over the views
    ious: list                          # per view, in the order given
    px_per_mm: list                     # per view: how far a mm of height moves in its image
    spread_mm: "tuple[float, float]"    # the heights that fit about as well
    resolved: bool
    why: str = ""                       # not resolved: why, in words
    views: int = 0
    names: list = field(default_factory=list)
    curve: "tuple | None" = None        # (heights mm, best score at each over the profiles): what bound_mm reads

    def bound_mm(self, drop: float = 0.10) -> float:
        """
        The tallest it can well be: the tallest height that still scores within
        drop of the best. Looser than the spread, and meant as a bound — for
        what a gripper must stay above, not what it grips by. From a camera
        looking down at it 8-20 deg off straight below, 0.10 was above the
        truth for every object tried (a 25 mm ball bounded 48-74 mm, a 140 mm
        bottle 142-150), where 0.05 once fell under it.
        """
        if self.curve is None:
            return self.spread_mm[1]
        hs, sc = (np.asarray(v, float) for v in self.curve)
        ok = hs[sc >= float(sc.max()) - drop]
        return float(ok.max()) if len(ok) else self.spread_mm[1]

    @property
    def centre_xy_m(self) -> np.ndarray:
        return self.footprint.centre_xy_m

    def widths_up_to(self, height_mm: float) -> dict:
        """widths_mm, the solid squeezed to height_mm: each width at the same share of the way up."""
        k = float(height_mm) / self.height_mm if self.height_mm > 0 else 1.0
        return {int(round(z * k)): w for z, w in sorted(self.widths_mm.items())}

    @property
    def foot_mm(self) -> float:
        """Its width at the surface — for a ball, the width a little above it (what stands on the surface is a point)."""
        low = [w for z, w in sorted(self.widths_mm.items()) if z <= max(4.0, 0.15 * self.height_mm)]
        return float(max(low)) if low else self.footprint.width_mm

    def as_dict(self) -> dict:
        return {"height_mm": round(self.height_mm, 1), "profile": self.profile,
                "centre_m": [round(float(v), 5) for v in self.centre_xy_m],
                "width_mm": round(self.footprint.width_mm, 1),
                "spread_mm": [round(float(v), 1) for v in self.spread_mm], "score": round(self.score, 3),
                "ious": [round(float(v), 3) for v in self.ious],
                "px_per_mm": [round(float(v), 2) for v in self.px_per_mm],
                "resolved": self.resolved, "why": self.why, "views": self.views, "names": list(self.names)}


# ── Views and footprints ───────────────────────────────────────────────────────

def shifted(view: PlaneView, shift_m) -> PlaneView:
    """view with its world frame moved by shift_m (x, y): a point at p in its old frame is at p + shift in the new one."""
    s = np.r_[np.asarray(shift_m, float)[:2] * 1000.0, 0.0]
    return PlaneView(view.K, view.D, view.R, view.t - view.R @ s, scale=view.scale, origin_m=view.origin_m)


def footprint_from_outline(view: PlaneView, mask: np.ndarray) -> "Footprint | None":
    """
    The ellipse with the same area and spread as an outline carried down to the
    surface (to_plane): for a camera looking down at the object, its widest
    cross-section, slightly enlarged and pushed away from the camera by its
    height (the fit moves and scales it). None when the outline is too small.
    """
    ys, xs = np.nonzero(np.asarray(mask, bool))
    if len(xs) < 20:
        return None
    step = max(1, len(xs) // 4000)
    plane = view.to_plane(np.c_[xs[::step], ys[::step]].astype(float))
    plane = plane[np.isfinite(plane).all(1)]
    if len(plane) < 20:
        return None
    centre = plane.mean(0)
    cov = np.cov((plane - centre).T) * view.scale ** 2
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    # A filled ellipse with half axes a, b has variance a^2/4, b^2/4 along them.
    b, a = 2.0 * np.sqrt(vals)
    angle = float(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return Footprint(centre, (float(a), float(b)), angle)


def px_per_mm_of_height(view: PlaneView, centre_mm) -> float:
    """How far a point over centre_mm moves in view's image per real mm it rises (PlaneView.px_per_mm_of_height)."""
    try:
        return float(view.px_per_mm_of_height(np.asarray(centre_mm, float)[:2]))
    except Exception:                    # noqa: BLE001 — behind the camera: it sees no height at all
        return 0.0


# ── Casting a solid into a view ────────────────────────────────────────────────

def solid_points(view: PlaneView, fp, height_mm: float, profile: str, scale: float = 1.0,
                 offset_mm=(0.0, 0.0)) -> "np.ndarray | None":
    """
    (N, 3) points on the solid's surface, in view's plane frame: fp's widest
    cross-section (Footprint or OutlineFootprint .section) scaled up its height
    by the profile, times scale, moved offset_mm (real). None when the footprint
    allows no such solid (an outline too narrow for its foot and its top both).
    """
    got = fp.section(view.scale, float(height_mm), profile)
    if got is None:
        return None
    centre, ring_mm = got
    s = np.maximum(PROFILES[profile](_LEVELS), _MIN_SCALE) * scale
    ring = np.asarray(ring_mm, float) / view.scale                              # plane units
    centre = np.asarray(centre, float)[:2] + np.asarray(offset_mm, float) / view.scale
    up = view.up / view.scale
    pts = centre[None, None, :] + s[:, None, None] * ring[None, :, :]
    z = (_LEVELS * float(height_mm))[:, None, None] * up[None, None, :]
    out = np.concatenate([pts, np.zeros(pts.shape[:2] + (1,))], axis=2) + z
    return out.reshape(-1, 3)


def cast(view: PlaneView, pts: np.ndarray, shape) -> "np.ndarray | None":
    """The outline the solid (solid_points) casts in a frame of shape (h, w): its points' convex hull. None: behind the camera."""
    h, w = shape[:2]
    hull = _hull_px(view, pts)
    if hull is None:
        return None
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    return mask.astype(bool)


def _hull_px(view: PlaneView, pts: np.ndarray) -> "np.ndarray | None":
    cam = pts @ view.R.T + view.t
    if (cam[:, 2] <= 1.0).any():
        return None
    px = view.project(pts)
    if not np.isfinite(px).all():
        return None
    return cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)


# ── Fitting ────────────────────────────────────────────────────────────────────

class _View:
    """One silhouette, cropped round the object and shrunk to at most MAX_CROP_PX a side, for scoring."""

    def __init__(self, sil: Silhouette, origin, size, factor: float):
        self.sil, self.origin, self.size, self.factor = sil, np.asarray(origin, float), size, factor
        x0, y0 = int(origin[0]), int(origin[1])
        w, h = size
        full_w = int(round(w * factor))
        full_h = int(round(h * factor))

        def crop(m):
            m = np.asarray(m, bool)[y0:y0 + full_h, x0:x0 + full_w].astype(np.uint8) * 255
            if m.shape[0] != full_h or m.shape[1] != full_w:     # at the frame's edge: what is beyond is not seen
                pad = np.zeros((full_h, full_w), np.uint8)
                pad[:m.shape[0], :m.shape[1]] = m
                m = pad
            return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) > 127

        frame_h, frame_w = np.asarray(sil.masks[0]).shape[:2]
        inside = np.zeros((frame_h, frame_w), bool)
        inside[:, :] = True
        valid = inside if sil.valid is None else (np.asarray(sil.valid, bool) & inside)
        self.valid = crop(valid)
        self.masks = [crop(m) & self.valid for m in sil.masks]
        self.choice = 0

    @property
    def mask(self) -> np.ndarray:
        return self.masks[self.choice]

    def score(self, pts: np.ndarray, which: "int | None" = None) -> float:
        hull = _hull_px(self.sil.view, pts)
        if hull is None:
            return 0.0
        local = (hull - self.origin) / self.factor
        m = np.zeros(self.valid.shape, np.uint8)
        cv2.fillConvexPoly(m, np.round(local * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
        cast_m = m.astype(bool) & self.valid
        seen = self.masks[self.choice if which is None else which]
        union = int((cast_m | seen).sum())
        return float((cast_m & seen).sum()) / union if union else 0.0


def _prepare(silhouettes: list, fp: Footprint, h_max_mm: float) -> list:
    views = []
    for sil in silhouettes:
        if not sil.masks:
            continue
        frame_h, frame_w = np.asarray(sil.masks[0]).shape[:2]
        boxes = []
        reach = solid_points(sil.view, fp, h_max_mm, "dome", scale=1.6)
        hull = None if reach is None else _hull_px(sil.view, reach)
        if hull is not None:
            boxes.append(hull)
        for m in sil.masks:
            ys, xs = np.nonzero(np.asarray(m, bool))
            if len(xs):
                boxes.append(np.array([[np.min(xs), np.min(ys)], [np.max(xs), np.max(ys)]], float))
        if not boxes:
            continue
        allp = np.vstack(boxes)
        lo = np.clip(np.floor(allp.min(0)) - 8, 0, [frame_w - 1, frame_h - 1])
        hi = np.clip(np.ceil(allp.max(0)) + 8, 1, [frame_w, frame_h])
        span = np.maximum(hi - lo, 8)
        factor = max(1.0, float(span.max()) / MAX_CROP_PX)
        size = (max(8, int(np.ceil(span[0] / factor))), max(8, int(np.ceil(span[1] / factor))))
        v = _View(sil, lo, size, factor)
        if any(m.sum() >= 6 for m in v.masks):
            views.append(v)
    return views


def fit_shape(silhouettes: list, footprint, height_range_mm=(4.0, 150.0),
              profiles: "tuple | None" = None) -> "ShapeFit | None":
    """
    The solid standing on footprint (moved and scaled a little) that casts the
    outlines seen in silhouettes best, over heights in height_range_mm and the
    profiles (PROFILES' names; None: all). None when no view has an outline
    to fit. Each view's best-matching outline among its candidates is chosen
    as the fit goes (a segmenter offers an object whole and in parts).
    """
    names = tuple(profiles or PROFILES)
    lo_h, hi_h = float(height_range_mm[0]), float(height_range_mm[1])
    views = _prepare(silhouettes, footprint, hi_h)
    if not views:
        return None

    def score(h, profile, k=1.0, off=(0.0, 0.0), choose=False) -> float:
        total = 0.0
        for v in views:
            pts = solid_points(v.sil.view, footprint, h, profile, k, off)
            if pts is None:
                return 0.0
            if choose and len(v.masks) > 1:
                best = max(range(len(v.masks)), key=lambda i: v.score(pts, i))
                v.choice = best
            total += v.score(pts)
        return total / len(views)

    # Coarse: every profile, the whole height range, the footprint as given.
    heights = np.arange(lo_h, hi_h + 1e-9, COARSE_STEP_MM)
    coarse = {(p, float(h)): score(h, p) for p in names for h in heights}
    ranked = sorted(coarse, key=lambda key: -coarse[key])
    # Refine the best few (distinct profiles first): centre, scale, height in turn.
    seeds, seen = [], set()
    for p, h in ranked:
        if p not in seen:
            seeds.append((p, h))
            seen.add(p)
        if len(seeds) >= 3:
            break
    best = None
    for p, h in seeds:
        k, off = 1.0, np.zeros(2)
        cur = score(h, p, k, off, choose=True)
        for _ in range(2):
            base = off.copy()
            for dx in (-4.0, -2.0, 0.0, 2.0, 4.0):
                for dy in (-4.0, -2.0, 0.0, 2.0, 4.0):
                    cand = base + np.array([dx, dy])
                    sc = score(h, p, k, cand)
                    if sc > cur:
                        cur, off = sc, cand
            for kk in (k * 0.88, k * 0.94, k * 1.06, k * 1.12):
                sc = score(h, p, kk, off)
                if sc > cur:
                    cur, k = sc, kk
            for hh in np.arange(max(lo_h, h - 4.0), min(hi_h, h + 4.0) + 1e-9, 1.0):
                sc = score(hh, p, k, off)
                if sc > cur:
                    cur, h = sc, float(hh)
            cur = max(cur, score(h, p, k, off, choose=True))
        if best is None or cur > best[0]:
            best = (cur, p, h, k, off.copy())
    assert best is not None
    top, profile, height, k, off = best
    score(height, profile, k, off, choose=True)

    # The spread: every height, each profile at the fitted centre and scale.
    fine = np.arange(lo_h, hi_h + 1e-9, 1.0)
    curve = np.array([max(score(h, p, k, off) for p in names) for h in fine])
    top = max(top, float(curve.max()))
    cut = top - (SPREAD_DROP + SPREAD_SHORTFALL * (1.0 - top))
    ok = fine[curve >= cut]
    spread = (float(np.min(ok)), float(np.max(ok))) if len(ok) else (height, height)

    centre, ring = footprint.section(views[0].sil.view.scale, height, profile)
    ring = np.asarray(ring, float) * k
    radii = np.linalg.norm(ring, axis=1)
    fitted = Footprint(np.asarray(centre, float)[:2] + off / views[0].sil.view.scale,
                       (float(np.max(radii)), float(np.min(radii))), 0.0)
    across = _across(ring)
    widths = {}
    for z in np.arange(0.0, height + 1e-9, 2.0):
        u = np.array([z / height]) if height > 0 else np.array([0.0])
        widths[int(round(z))] = round(float(across * max(PROFILES[profile](u)[0], _MIN_SCALE)), 1)
    ious = []
    sens = []
    for sil in silhouettes:
        v = next((x for x in views if x.sil is sil), None)
        pts = solid_points(sil.view, footprint, height, profile, k, off)
        ious.append(0.0 if v is None or pts is None else v.score(pts))
        sens.append(px_per_mm_of_height(sil.view, fitted.centre_mm))
    width = spread[1] - spread[0]
    seeing = [p for p, iou in zip(sens, ious) if iou >= MIN_FIT_IOU - 0.1]
    sees = max(seeing) if seeing else 0.0
    resolved, why = True, ""
    if top < MIN_FIT_IOU:
        resolved, why = False, (f"no solid matches what was outlined (best {top:.2f} of the outlines' area "
                                f"in common, {MIN_FIT_IOU:.2f} needed)")
    elif sees < SEES_HEIGHT_PX_PER_MM:
        resolved, why = False, (f"no view that sees it sees its height (the best moves {sees:.2f} px per mm of "
                                f"height, {SEES_HEIGHT_PX_PER_MM:.1f} needed)")
    elif width > max(RESOLVED_MM, RESOLVED_SHARE * height):
        resolved, why = False, (f"anything from {spread[0]:.0f} to {spread[1]:.0f} mm tall fits what was seen "
                                f"(the best view moves {sees:.2f} px per mm of height)")
    return ShapeFit(height_mm=round(float(height), 1), profile=profile, footprint=fitted, widths_mm=widths,
                    score=round(float(top), 4), ious=ious, px_per_mm=sens, spread_mm=spread, resolved=resolved,
                    why=why, views=len(views), names=[s.name for s in silhouettes],
                    curve=(fine.copy(), curve.copy()))
