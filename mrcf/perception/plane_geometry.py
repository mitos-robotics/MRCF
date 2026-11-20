"""
plane_geometry.py — measure upright objects on a flat work surface from any
calibrated camera.

A camera that knows where it is relative to the work surface can trace every
pixel back to a point on that surface. An object standing on the surface is a
solid on a plane. Those two facts turn an outline (segment.py) into
measurements, with no knowledge of what the object is or what the surface
looks like:

  base      where it stands, and how wide its foot is. Every pixel of the
            outline is projected down onto the surface along its camera ray. Its
            upper parts land BEHIND the object, away from the camera, so the
            near edge of that footprint is the foot's own rim, at true size.
            Exact from any viewpoint.
  height    where the outline's top is, traced back to the vertical line through
            the base. Needs a slanted view: the top only separates from the base
            in the image by (height x slant), so from straight above it cannot be
            seen at all. Reported only where the view gives enough pixels per mm.
  widths    the outline's width at each height up that vertical.

Where the camera is comes from the backend (RobotEnv.camera_model): a fixed
camera's calibrated pose, or a wrist camera's pose from the arm's joint angles.
A deployment with a target of its own in view — a chessboard — can build a
view from that instead (PlaneView takes any pose); the measuring is the same.

Units. Positions are in the view's plane frame (PlaneView docs); with a view
from the backend that is the robot's base frame, in mm, so a foot's position is
where to send the arm. Sizes — foot, height, widths — are always real mm: a
view carries `scale`, real mm per plane-frame unit, for an arm whose frame
reads short of the real world (the SO-101's reads ~9% short).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Below this many pixels of image movement per millimetre of height, a height is
# not reported: two pixels of outline noise would be several millimetres. A
# wrist camera's slanted view at a hover gives ~1.5 px/mm; an overhead camera
# looking almost straight down, ~0.3 near its centre.
MIN_HEIGHT_PX_PER_MM = 0.8

# How the foot is found from the footprint. "band" takes its width just behind
# its near edge: on the real chess pieces the steadier of the two (a pawn
# 21.3 mm over ten gripper views, 1 mm spread; from overhead, within a mm of
# what the gripper camera reads). "rim" fits a circle to the near edge itself
# (_near_rim_circle): ~2 mm wider than "band" on small pieces from overhead,
# but it is not fooled by a tall, wide object close to the camera, whose
# magnified upper body fills the band — a simulated 80 mm mug under the wrist
# camera read 117 mm by band, 82 by rim. "auto" takes band unless it reads
# more than BAND_OVER_RIM wider than rim, which is that failure.
FOOT_METHOD = "auto"
BAND_OVER_RIM = 1.25


@dataclass
class PlaneView:
    """
    A camera's pose relative to a work surface: X_cam = R @ X_plane + t.

    The plane frame has the surface at z = 0 and z pointing to the camera's
    side. Its x, y are whatever the view was built from: the robot's base frame
    (from_camera_model) or a target's own axes (a chessboard's corners).
    Plane-frame units are mm of that frame; `scale` turns them into real mm.
    """
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    t: np.ndarray
    scale: float = 1.0                        # real mm per plane-frame unit
    origin_m: "np.ndarray | None" = None      # world (x, y) of the plane origin, m — set
                                              # only when the plane frame is the world's

    @classmethod
    def from_camera_model(cls, model: dict) -> "PlaneView":
        """
        From a backend's camera model (RobotEnv.camera_model): K, D,
        T_world_cam (4x4, metres), plane_z_m (the surface's height in the world
        frame) and scale. The plane frame is then the world frame lowered by
        plane_z, in mm, so positions measured in it are world positions.
        """
        T = np.asarray(model["T_world_cam"], float)
        R_wc, p_wc = T[:3, :3], T[:3, 3]
        R = R_wc.T                                     # world -> camera
        plane_z = float(model.get("plane_z_m", 0.0))
        t = -R @ (p_wc - np.array([0.0, 0.0, plane_z])) * 1000.0
        return cls(np.asarray(model["K"], float), np.asarray(model["D"], float).ravel(), R, t,
                   scale=float(model.get("scale", 1.0)), origin_m=np.zeros(2))

    def world_xy_m(self, plane_xy) -> "np.ndarray | None":
        """Plane-frame (x, y) as world (x, y) in metres, if this view is in the world frame."""
        if self.origin_m is None:
            return None
        return self.origin_m + np.asarray(plane_xy, float)[:2] / 1000.0

    @property
    def centre(self) -> np.ndarray:
        """The camera's position in plane coordinates."""
        return -self.R.T @ self.t

    @property
    def up(self) -> np.ndarray:
        """Unit vector off the surface toward the camera's side."""
        return np.array([0.0, 0.0, 1.0 if self.centre[2] >= 0 else -1.0])

    def rays(self, px: np.ndarray) -> np.ndarray:
        r = cv2.undistortPoints(np.asarray(px, np.float64).reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        return np.c_[r, np.ones(len(r))]

    def to_plane(self, px: np.ndarray) -> np.ndarray:
        """(N, 2) plane points where each pixel's ray meets the surface."""
        r = self.rays(px)
        n = self.R[:, 2]
        # Rays that run parallel to the surface (pixels above its horizon) never
        # meet it: they come out non-finite, and callers drop them.
        with np.errstate(all="ignore"):
            s = (n @ self.t) / (r @ n)
            return ((r * s[:, None] - self.t) @ self.R)[:, :2]

    def project(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) pixels for (N, 3) plane-frame points."""
        p, _ = cv2.projectPoints(np.asarray(pts, float).reshape(-1, 3),
                                 cv2.Rodrigues(self.R)[0], self.t, self.K, self.D)
        return p.reshape(-1, 2)

    def lifted(self, xy, height_mm: float) -> np.ndarray:
        """The surface point xy raised height_mm (real mm) off the surface, as a 3-vector."""
        return np.array([xy[0], xy[1], 0.0]) + self.up * height_mm / self.scale

    def px_per_mm_of_height(self, xy) -> float:
        """How far a point at xy moves in the image per real mm it is raised."""
        a, b = self.project(np.array([self.lifted(xy, 0.0), self.lifted(xy, 10.0)]))
        return float(np.linalg.norm(b - a)) / 10.0

    def slant_deg(self, xy) -> float:
        """How far from straight down the camera looks at surface point xy."""
        c = self.centre
        return float(np.degrees(np.arctan2(np.linalg.norm(np.asarray(xy) - c[:2]), abs(c[2]))))

    def footprint_px(self, xy, radius_mm: float, n: int = 24) -> np.ndarray:
        """(n, 2) pixels of a circle of radius_mm (real) on the surface around xy."""
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = radius_mm / self.scale
        return self.project(np.c_[xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), np.zeros(n)])


# ── Masks ──────────────────────────────────────────────────────────────────────

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """The largest connected part of a mask, holes filled — one object, one outline."""
    m = np.asarray(mask, bool).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == keep).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def touches_frame(mask: np.ndarray) -> bool:
    """Runs off the image edge — cut off, so neither its foot nor its top can be trusted."""
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


# ── Measuring ──────────────────────────────────────────────────────────────────

@dataclass
class ObjectMeasurement:
    base_mm: np.ndarray                 # centre of the foot, plane coordinates
    base_radius_mm: float               # real mm, as are the sizes below
    height_mm: "float | None"           # None where the view cannot resolve it
    widths_mm: dict = field(default_factory=dict)   # height (mm) -> width (mm)
    base_px: "tuple[float, float]" = (0.0, 0.0)
    top_px: "tuple[float, float] | None" = None
    slant_deg: float = 0.0
    px_per_mm_height: float = 0.0
    area_px: int = 0
    base_world_m: "np.ndarray | None" = None    # world (x, y), when the view is in the world frame

    def as_dict(self) -> dict:
        return {
            "base_mm": [round(float(v), 1) for v in self.base_mm],
            "base_diameter_mm": round(2 * self.base_radius_mm, 1),
            "height_mm": None if self.height_mm is None else round(self.height_mm, 1),
            "widths_mm": {int(k): round(v, 1) for k, v in self.widths_mm.items()},
            "slant_deg": round(self.slant_deg, 1),
            "area_px": int(self.area_px),
        }


def measure(mask: np.ndarray, view: PlaneView, size_hint_mm: float = 12.0) -> "ObjectMeasurement | None":
    """
    Base, height and width profile of the upright object outlined by mask.
    None if the mask is too small to say anything. size_hint_mm is a first
    guess at the foot's radius; the fit moves away from it within a few steps.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    px = np.c_[xs, ys].astype(float)
    foot = view.to_plane(px)
    ok = np.isfinite(foot).all(1)
    px, foot = px[ok], foot[ok]
    if len(px) < 30:
        return None

    # The footprint, in the frame of the line from the camera's foot through it:
    # `along` away from the camera, `across` sideways. The base's near rim is the
    # footprint's nearest edge; the base is as wide as the footprint is across,
    # measured over the first base-width of it (further along, the projected
    # upper body widens it, magnified by its height). In plane units here.
    cam = view.centre[:2]
    d = np.median(foot, 0) - cam
    d = d / (np.linalg.norm(d) or 1.0)
    dp = np.array([-d[1], d[0]])
    with np.errstate(all="ignore"):          # Accelerate's matmul warns on finite input
        along, across = (foot - cam) @ d, (foot - cam) @ dp
    floor = 4.0 / view.scale
    near = float(np.percentile(along, 0.5))
    r = size_hint_mm / view.scale
    for _ in range(4):
        band = along < near + 2 * r
        r = max(floor, float(np.percentile(across[band], 98) - np.percentile(across[band], 2)) / 2)
    band = along < near + 2 * r
    base = cam + d * (near + r) + dp * float(np.median(across[band]))
    if FOOT_METHOD in ("rim", "auto"):
        rim = _near_rim_circle(along, across, floor)
        if rim is not None and (FOOT_METHOD == "rim" or r > BAND_OVER_RIM * rim[2]):
            base, r = cam + d * rim[0] + dp * rim[1], rim[2]

    b_px = view.project(view.lifted(base, 0.0)[None])[0]
    rate = view.px_per_mm_of_height(base)
    m = ObjectMeasurement(base_mm=base, base_radius_mm=r * view.scale, height_mm=None,
                          base_px=tuple(b_px), slant_deg=view.slant_deg(base),
                          px_per_mm_height=rate, area_px=int(mask.sum()),
                          base_world_m=view.world_xy_m(base))
    if rate < MIN_HEIGHT_PX_PER_MM:
        return m

    # Height: the outline's extreme toward where verticals converge, traced back
    # to the vertical through the base (closest approach of the two lines).
    a_px = view.project(view.lifted(base, 10.0)[None])[0]
    axis = (a_px - b_px) / (np.linalg.norm(a_px - b_px) or 1.0)
    prog = (px - b_px) @ axis
    top = px[prog >= np.percentile(prog, 99.7)].mean(0)
    ray = view.rays(top[None])[0]
    ray /= np.linalg.norm(ray)
    b_cam = view.R @ view.lifted(base, 0.0) + view.t
    u_cam = view.R @ view.up
    sol, *_ = np.linalg.lstsq(np.c_[ray, -u_cam], b_cam, rcond=None)
    m.height_mm, m.top_px = float(sol[1]) * view.scale, tuple(top)

    # Width at each height: the outline's extent across the axis there, scaled
    # by that point's distance from the camera.
    perp = np.array([-axis[1], axis[0]])
    for hz in range(4, int(m.height_mm) - 1, 4):
        p3 = view.lifted(base, hz)
        a = view.project(p3[None])[0]
        strip = np.abs((px - a) @ axis) < 1.5
        if strip.sum() < 3:
            continue
        ext = (px[strip] - a) @ perp
        dist = float(np.linalg.norm(view.R @ p3 + view.t))
        m.widths_mm[hz] = float((ext.max() - ext.min()) * dist / view.K[0, 0]) * view.scale
    return m


def box_centre_xy(mask: np.ndarray, view: "PlaneView", height_mm: float, side_mm: float,
                  faces_deg: float) -> "np.ndarray | None":
    """
    Where a box of known size (a block) stands, world (x, y) metres, from its
    outline: by the edges of its top face that face away from the camera. Every
    pixel carried up to the top face's height, the top face lands where it is
    and the sides the camera sees land beyond it on the camera's side; so on
    each of the box's two axes the edge away from the camera is the top face's
    own, and its centre half a side in from it. A foot fitted as a circle, as
    for a round object, is pulled toward the camera by those sides. None when
    the outline is too small.
    """
    ys, xs = np.nonzero(np.asarray(mask))
    if len(xs) < 30:
        return None
    rays = view.rays(np.c_[xs, ys].astype(float))
    up_cam = view.R @ view.up
    with np.errstate(all="ignore"):
        s = (height_mm / view.scale + up_cam @ view.t) / (rays @ up_cam)
        top = ((rays * s[:, None] - view.t) @ view.R)[:, :2]
    top = top[np.isfinite(top).all(1)]
    if len(top) < 30:
        return None
    a = np.radians(float(faces_deg))
    mid, cam, half = np.median(top, 0), np.asarray(view.centre, float)[:2], side_mm / view.scale / 2.0
    centre = np.zeros(2)
    for axis in (np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])):
        with np.errstate(all="ignore"):          # Accelerate's matmul warns on finite input
            along = top @ axis
        far = (np.percentile(along, 1.0) + half) if (cam - mid) @ axis > 0 else (np.percentile(along, 99.0) - half)
        centre += far * axis
    return view.world_xy_m(centre)


def side_direction_deg(mask: np.ndarray, view: "PlaneView") -> "float | None":
    """
    Which way the outline's sides face on the surface, as for a block: the
    smallest rectangle round the outline, one edge's direction carried onto the
    surface (degrees from the world x axis, mod 90). For a round object it
    means nothing: the caller decides which objects have sides. None when the
    outline is too small to say.
    """
    import cv2
    m = np.asarray(mask, np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None
    (cx, cy), _, angle = cv2.minAreaRect(c)
    d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    p = view.to_plane(np.array([[cx, cy], [cx + 10.0 * d[0], cy + 10.0 * d[1]]]))
    if not np.isfinite(p).all():
        return None
    e = p[1] - p[0]
    return float(np.degrees(np.arctan2(e[1], e[0])) % 90.0)


LONG_STEP_MM = 3.0            # long_extent_mm: cross-sections this far apart along the long axis
LONG_ENDS_SHARE = 0.15        # ... and the narrowest of them taken over the middle, this share off each end


def long_extent_mm(mask: np.ndarray, view: "PlaneView") -> "dict | None":
    """
    How long the outline is on the surface and how wide across it: its long
    axis (the principal axis of its boundary carried onto the surface), and
    across that, every LONG_STEP_MM, how wide it is — {"length_mm", "widest_mm",
    "narrowest_mm"}, the narrowest over its middle only (an end's taper is not
    where it is gripped). For a thing lying along the table — a tool — this is
    what the jaws close across; the round foot plane_geometry.measure fits is
    an upright object's. Each pixel is put on the surface itself, so a raised
    edge lands further from the camera than it is: the widths are, if anything,
    wider than the object. None when the outline is too small to say.
    """
    import cv2
    m = np.asarray(mask, np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None
    poly = view.to_plane(c.reshape(-1, 2).astype(float))
    poly = poly[np.isfinite(poly).all(axis=1)]
    if len(poly) < 6:
        return None
    centre = poly.mean(axis=0)
    with np.errstate(all="ignore"):          # Accelerate's matmul warns on finite input
        _, _, vt = np.linalg.svd(poly - centre, full_matrices=False)
        s_all, t_all = (poly - centre) @ vt[0], (poly - centre) @ vt[1]
    lo, hi = float(s_all.min()), float(s_all.max())
    widths = []
    n = len(poly)
    for s in np.arange(lo + LONG_STEP_MM / 2, hi, LONG_STEP_MM):
        ts = [t_all[i] + (s - s_all[i]) / (s_all[(i + 1) % n] - s_all[i]) * (t_all[(i + 1) % n] - t_all[i])
              for i in range(n) if (s_all[i] - s) * (s_all[(i + 1) % n] - s) <= 0 and s_all[i] != s_all[(i + 1) % n]]
        if len(ts) >= 2:
            widths.append((float(s), float(max(ts) - min(ts))))
    if len(widths) < 3:
        return None
    length = hi - lo
    middle = [w for s, w in widths if lo + LONG_ENDS_SHARE * length <= s <= hi - LONG_ENDS_SHARE * length]
    return {"length_mm": round(length, 1), "widest_mm": round(max(w for _, w in widths), 1),
            "narrowest_mm": round(min(middle or [w for _, w in widths]), 1)}


def _near_rim_circle(along, across, floor, bin_w: float = 0.5) -> "tuple[float, float, float] | None":
    """
    The foot as the circle through the footprint's near edge: (centre along,
    centre across, radius), plane units, or None when there is too little edge.

    For every sideways position, the footprint point nearest the camera is
    the foot's own rim — the body above it projects further away. Those points
    trace the near half of the foot's circle; anything off that circle, or
    beyond its centre, is body seen beside the foot, and is dropped before
    refitting. Unlike taking the footprint's width near its near edge, this
    does not grow with the object's height: a wide, tall object close to the
    camera (a mug under a wrist camera) projects its upper body across that
    band, and was read twice as wide as its foot.
    """
    lo, hi = np.percentile(across, 0.5), np.percentile(across, 99.5)
    idx = np.digitize(across, np.arange(lo, hi + bin_w, bin_w))
    order = np.lexsort((along, idx))
    first = np.r_[True, idx[order][1:] != idx[order][:-1]]
    env = np.c_[across[order][first], along[order][first]]
    if len(env) < 8:
        return None
    keep = np.ones(len(env), bool)
    for _ in range(6):
        x, y = env[keep, 0], env[keep, 1]
        sol, *_ = np.linalg.lstsq(np.c_[x, y, np.ones(len(x))], -(x ** 2 + y ** 2), rcond=None)
        cx, cy = -sol[0] / 2, -sol[1] / 2
        r = float(np.sqrt(max(cx ** 2 + cy ** 2 - sol[2], 0.0)))
        res = np.abs(np.hypot(env[:, 0] - cx, env[:, 1] - cy) - r)
        new = (res < max(1.0, 0.04 * r)) & (env[:, 1] <= cy + 0.2 * r)
        if new.sum() < 6 or (new == keep).all():
            break
        keep = new
    if not np.isfinite([cx, cy, r]).all() or r <= 0:
        return None
    return float(cy), float(cx), max(r, floor)


def outline_objects(frame_rgb: np.ndarray, view: PlaneView, prompts_px, *,
                    reject_mask=None, accept=None, weigh=None, prefer: str = "score",
                    size_hint_mm: float = 12.0, overlap: float = 0.5, trim=None) -> list:
    """
    Every object the segmenter outlines at the prompt pixels, measured:
    [(score, mask, ObjectMeasurement)], one per object.

      reject_mask(mask) -> bool   drop an outline before measuring (the
                                  gripper's own jaw, a patch of the surface)
      accept(measurement) -> bool keep only plausible measurements (size, height)
      trim(mask) -> mask          what of a kept outline is not the object taken
                                  out before it is measured (its shadow: the
                                  surface scan's), the mask returned in its place
      weigh(mask) -> float        how much of the outline is object rather than
                                  surface, 0-1: multiplies the segmenter's score
                                  when choosing among overlapping outlines, so an
                                  outline that took in a patch of patterned
                                  surface loses to the object's own
      prefer                      of overlapping outlines of one object, keep the
                                  one the segmenter is surest of ("score") or the
                                  largest ("area"). From far above, an outline
                                  that took in a patch of surface is the common
                                  mistake and is always the largest; close up,
                                  outlines of a PART (a head, a foot) are, and
                                  the whole is the largest.

    [] without the segmenter (segment.available()).
    """
    from mrcf.perception import segment

    if not segment.available():
        return []
    found = []
    for score, mask, _ in segment.candidates(frame_rgb, prompts_px):
        m = clean_mask(mask)
        if touches_frame(m) or (reject_mask is not None and reject_mask(m)):
            continue
        whole = m
        if trim is not None:
            m = trim(m)
        meas = measure(m, view, size_hint_mm)
        if meas is None or (accept is not None and not accept(meas)):
            continue
        if weigh is not None:
            score = score * float(weigh(whole))
        found.append((score, m, meas))
    found.sort(key=(lambda f: -f[0]) if prefer == "score" else (lambda f: -int(f[1].sum())))
    kept: list = []
    for score, m, meas in found:
        if any((m & k[1]).sum() > overlap * min(m.sum(), k[1].sum()) for k in kept):
            continue
        kept.append((score, m, meas))
    return kept


def surface_grid_px(view: PlaneView, x_range_mm, y_range_mm, step_mm: float,
                    lift_mm: float = 12.0) -> np.ndarray:
    """
    Prompt pixels for outline_objects: a grid over a rectangle of the surface,
    each point lifted lift_mm so it lands on an object's body rather than on
    the surface just in front of its foot.
    """
    xs = np.arange(x_range_mm[0], x_range_mm[1] + 1e-9, step_mm / view.scale)
    ys = np.arange(y_range_mm[0], y_range_mm[1] + 1e-9, step_mm / view.scale)
    pts = np.array([view.lifted((x, y), lift_mm) for y in ys for x in xs])
    px = view.project(pts)
    return px[np.isfinite(px).all(1)]


# ── The object nearest a point ─────────────────────────────────────────────────
#
# What a wrist camera does at a hover: outline whatever stands nearest where the
# object is expected, and measure it.

def cut_off(mask: np.ndarray, margin_px: int = 2) -> bool:
    """
    Whether an outline runs off the edge of the frame: what is beyond is not
    in it, so its height is no more than a lower bound. 2026-11-03: from
    beside the d1 queen a wrist camera outlined her only up to the frame's
    edge, and read her 28 mm tall instead of 69.
    """
    return bool(mask[:margin_px].any() or mask[-margin_px:].any()
                or mask[:, :margin_px].any() or mask[:, -margin_px:].any())


def measure_near(frame_rgb: np.ndarray, view: PlaneView, expected_px=None, expected_xy_m=None, *,
                 reject_mask=None, foot_radius_mm=(3.0, 60.0), height_mm=(5.0, 300.0),
                 search_px: float = 140.0, step_px: float = 20.0, prompt_lift_mm: float = 20.0,
                 known=None, target: "str | None" = None, need_height: bool = True, foot_tol_mm: float = 6.0) -> dict:
    """
    Measure the object standing nearest where it is expected, in one frame.

    It is expected at expected_xy_m — its foot, world (x, y) in metres — when
    the view is in the world frame (from_camera_model); else at expected_px;
    else mid-frame. Prompts cover search_px either side of that pixel, aimed
    prompt_lift_mm up the object, at its body rather than the surface in front
    of its foot. Outlines whose foot radius or height fall outside the given
    ranges are not objects; reject_mask(mask) -> True drops others (the
    gripper's own jaw, a surface pattern).

    The segmenter offers an object close up both whole and in parts (a pawn's
    base on its own, its head on its own), and a part's height is wrong — the
    base alone measured 20 mm on a 45 mm pawn — so of overlapping outlines the
    largest is kept, and of what is left, the one nearest the expected pixel.

    With known (expected_view.KnownObject, every object another camera found
    near here, in this view's world frame) and target (the key of the one
    wanted), it measures that one and no other (_measure_known): prompted only
    where it should be seen, and an outline kept only if it lies on its column
    and its foot is nearer the target's than any other's, and within
    foot_tol_mm of it — an outline of a part (a pawn's head alone) puts its
    foot ~10 mm off and its height at half (2026-11-03: 22 mm for a 40 mm
    pawn, and the grasp planned on its foot). need_height=False keeps an
    outline whose height the view cannot resolve (looking straight down at
    it): its foot still says where it is.

    Returns {"ok", "why", and when ok "measurement" (ObjectMeasurement),
    "mask", "score", "occluded" (it runs off the frame: cut_off)}; with known
    also "hidden_by" (what stands in front of the target here, most first) and
    "foot_visible" (the share of its foot that should show).
    """
    from mrcf.perception import segment

    if not segment.available():
        return {"ok": False, "why": segment.why_unavailable()}
    if known is not None and target is not None and view.origin_m is not None:
        return _measure_known(frame_rgb, view, known, target, reject_mask, foot_radius_mm, height_mm, need_height,
                              foot_tol_mm)
    h, w = frame_rgb.shape[:2]
    if expected_xy_m is not None and view.origin_m is not None:
        foot = (np.asarray(expected_xy_m, float)[:2] - view.origin_m) * 1000.0
        expected_px = view.project(view.lifted(foot, prompt_lift_mm)[None])[0]
    if expected_px is None or not np.isfinite(expected_px).all():
        expected_px = (w / 2, h / 2)
    ex, ey = float(expected_px[0]), float(expected_px[1])
    offs = np.arange(-search_px, search_px + 1, step_px)
    prompts = [(ex + dx, ey + dy) for dy in offs for dx in offs
               if 2 <= ex + dx < w - 2 and 2 <= ey + dy < h - 2]
    why = ["nothing object-like near the expected pixel"]

    def accept(meas):
        if not (foot_radius_mm[0] <= meas.base_radius_mm <= foot_radius_mm[1]):
            return False
        if meas.height_mm is None or not (height_mm[0] <= meas.height_mm <= height_mm[1]):
            why[0] = "an outline was found but its height made no sense"
            return False
        return True

    wholes = outline_objects(frame_rgb, view, prompts, reject_mask=reject_mask, accept=accept,
                             prefer="area")
    if not wholes:
        return {"ok": False, "why": why[0]}

    def distance(found):
        ys, xs = np.nonzero(found[1])
        return float(np.hypot(xs.mean() - ex, ys.mean() - ey))
    score, m, meas = min(wholes, key=distance)
    return {"ok": True, "why": "", "measurement": meas, "mask": m, "score": score, "occluded": cut_off(m)}


def _measure_known(frame_rgb, view: PlaneView, known, target: str, reject_mask, foot_radius_mm, height_mm,
                   need_height: bool, foot_tol_mm: float = 6.0) -> dict:
    """measure_near for one object among others another camera found (see there)."""
    from mrcf.perception.expected_view import ExpectedView

    ev = ExpectedView(view, list(known), frame_rgb.shape)
    if not ev.drawn(target):
        return {"ok": False, "why": "where it stands is out of the camera's view", "hidden_by": [], "foot_visible": 0.0}
    hidden_by, foot_visible = ev.in_front(target), ev.foot_visible(target)
    prompts = ev.prompts(target)
    if not prompts:
        return {"ok": False, "why": f"it is hidden behind {', '.join(hidden_by[:2]) or 'something'}",
                "hidden_by": hidden_by, "foot_visible": foot_visible}
    column = ev.masks[target]
    want = np.asarray(ev.objects[target].foot_xy_m, float)
    others = [np.asarray(o.foot_xy_m, float) for o in known if o.key != target]
    why = ["nothing of its own outlined where it should be seen"]

    def reject(m):
        if reject_mask is not None and reject_mask(m):
            return True
        return bool((m & column).sum() < 0.5 * m.sum())          # mostly off its column: something else

    def accept(meas):
        if not (foot_radius_mm[0] <= meas.base_radius_mm <= foot_radius_mm[1]):
            return False
        if meas.height_mm is None:
            if need_height:
                why[0] = "its outline was found but the view cannot tell its height"
                return False
        elif not (height_mm[0] <= meas.height_mm <= height_mm[1]):
            why[0] = "an outline was found but its height made no sense"
            return False
        foot = np.asarray(meas.base_world_m, float)[:2]
        mine = float(np.linalg.norm(foot - want))
        if any(float(np.linalg.norm(foot - o)) < mine for o in others):
            why[0] = "the outline found there stands nearer another object's foot than its own"
            return False
        if mine * 1000 > foot_tol_mm:
            why[0] = (f"the only outline of it puts its foot {mine * 1000:.0f} mm from where it stands — "
                      "a part of it, not all of it")
            return False
        return True

    wholes = outline_objects(frame_rgb, view, prompts, reject_mask=reject, accept=accept, prefer="area")
    if not wholes:
        return {"ok": False, "why": why[0], "hidden_by": hidden_by, "foot_visible": foot_visible}
    score, m, meas = min(wholes, key=lambda f: float(np.linalg.norm(np.asarray(f[2].base_world_m)[:2] - want)))
    return {"ok": True, "why": "", "measurement": meas, "mask": m, "score": score, "occluded": cut_off(m),
            "hidden_by": hidden_by, "foot_visible": foot_visible}


def measure_in_frame(frame_rgb: np.ndarray, expected_px=None, model: "dict | None" = None,
                     expected_xy_m=None, **known) -> dict:
    """
    measure_near for a frame from a camera the backend has located (model:
    RobotEnv.camera_model, read with the frame). The default way a policy
    measures what is under its wrist camera; a deployment can pass its own
    with the same signature (and an optional "occluded" in the result).
    known: measure_near's known, target and need_height.
    """
    if model is None:
        return {"ok": False, "why": "the camera is not located (no camera model)"}
    return measure_near(frame_rgb, PlaneView.from_camera_model(model), expected_px, expected_xy_m, **known)


# ── A wrist camera's fixed sight line, at another tilt ─────────────────────────

def aligned_offset(rot, down_ref_g, camera_g, ref_depth_m: float, depth_m: float) -> np.ndarray:
    """
    Where a point sits relative to the grasp point, horizontally in the world
    (metres), when a wrist camera sees it at the pixel a visual servo was
    calibrated with.

    A servo calibrated "the object appears HERE when the jaws are over it"
    learned one sight line, fixed in the gripper: from the camera to a point
    ref_depth_m straight below the grasp point, with the gripper at the tilt
    of the calibration. Aligning that pixel puts the object on that line
    whatever the gripper does — but straight below the jaws only at that same
    tilt and depth. Tilted further, the line swings away from the vertical,
    and a vertical descent from an "aligned" hover lands that far off. On the
    SO-101, a 30 deg difference was ~25 mm: the jaws' whole slack three times.

      rot          the gripper's orientation now (world from gripper)
      down_ref_g   at the calibration: straight down, in the gripper's frame
      camera_g     the camera's centre in the gripper's frame, from the grasp
                   point. Only matters away from the calibration and weakly:
                   20 mm wrong moved the answer ~1 mm at a 15 deg difference
      ref_depth_m  how far below the grasp point the point was at the calibration
      depth_m      and is now

    (0, 0) at the calibration's own tilt and depth. Moving the grasp point by
    the result puts it over the point.
    """
    rot = np.asarray(rot, float)
    camera_g = np.asarray(camera_g, float)
    sight_g = float(ref_depth_m) * np.asarray(down_ref_g, float) - camera_g
    camera, sight = rot @ camera_g, rot @ sight_g
    if sight[2] >= -1e-9:
        raise ValueError("the servo's sight line does not reach the surface at this orientation")
    s = (-float(depth_m) - camera[2]) / sight[2]
    return (camera + s * sight)[:2]
