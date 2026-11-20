"""
plane_sweep.py — how high an outline's edge stands, from two views of it by
one camera that moved between them.

Each view's outline of a thing — a tub, seen from above — runs along its
top edge (the rim) where it is not cut off by the picture's border. Carry the
edge of the first view onto a level plane h above the surface, through that
view's camera, and look at where it lands in the second view: only at the
rim's own height does it fall on the second view's edge everywhere. So h is
swept from the surface up, and the h whose carried edge lies closest to the
second edge (median distance, px) is how high the rim stands.

Only the two cameras' poses relative to each other and the surface matter, so
one camera moved by the arm serves: an error in where it sits on the gripper
moves both views alike and largely cancels — unlike measuring from one view,
where it does not. The views need to differ by a few centimetres sideways,
the further the better, and both to see the same stretch of edge.

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mrcf.perception.plane_geometry import PlaneView

# Edge pixels within this of the picture's border are where the picture ends,
# not where the thing does.
BORDER_PX = 6
# At most this many edge pixels are carried (evenly spaced along the edge).
MAX_EDGE_POINTS = 400
# A sweep that lands under half the carried edge inside the second view says nothing.
MIN_SHARED = 0.5


@dataclass
class Swept:
    """How high an edge stands, by plane sweep."""
    height_m: "float | None"         # the best height; None: could not tell
    error_px: float                  # how far the carried edge lies from the other, there (median)
    spread_m: tuple                  # the heights within error_px + 1 px of the best: how sure
    points: int                      # edge points that counted
    why: str = ""


def edge_px(mask: np.ndarray, border_px: int = BORDER_PX, keep: "np.ndarray | None" = None) -> np.ndarray:
    """
    (N, 2) pixels along a mask's outer edge, leaving out where it runs along the
    picture's border, and outside `keep` (bool (h, w)) when given.
    """
    import cv2

    m = np.asarray(mask, bool).astype(np.uint8)
    found, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not found:
        return np.zeros((0, 2))
    pts = np.vstack([c.reshape(-1, 2) for c in found]).astype(float)
    h, w = m.shape
    only = keep
    keep = ((pts[:, 0] >= border_px) & (pts[:, 0] < w - border_px)
            & (pts[:, 1] >= border_px) & (pts[:, 1] < h - border_px))
    pts = pts[keep]
    if only is not None:
        inside = only[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        pts = pts[inside]
    if len(pts) > MAX_EDGE_POINTS:
        pts = pts[np.linspace(0, len(pts) - 1, MAX_EDGE_POINTS).astype(int)]
    return pts


def _raised(view: PlaneView, height_m: float) -> PlaneView:
    """view, its plane lifted height_m: to_plane meets that level, not the surface."""
    return PlaneView(view.K, view.D, view.R, view.t + view.R @ np.array([0.0, 0.0, height_m * 1000.0 / view.scale]),
                     scale=view.scale, origin_m=view.origin_m)


def carry(px, view_a: PlaneView, view_b: PlaneView, height_m: float) -> np.ndarray:
    """Pixels of view A, as points height_m above the surface, where view B sees them ((N, 2), NaN where none)."""
    xy = _raised(view_a, height_m).to_plane(np.asarray(px, float).reshape(-1, 2))
    pts = np.c_[xy, np.full(len(xy), height_m * 1000.0 / view_b.scale)]
    out = np.full((len(xy), 2), np.nan)
    ok = np.isfinite(pts).all(axis=1)
    if ok.any():
        cam = pts[ok] @ view_b.R.T + view_b.t
        front = cam[:, 2] > 1.0
        idx = np.flatnonzero(ok)[front]
        if len(idx):
            out[idx] = view_b.project(pts[idx])
    return out


def sweep(mask_a: np.ndarray, model_a: dict, mask_b: np.ndarray, model_b: dict,
          heights_m, border_px: int = BORDER_PX, keep_a: "np.ndarray | None" = None,
          keep_b: "np.ndarray | None" = None) -> Swept:
    """
    How high the edge both masks share stands: mask_a and mask_b, the thing's
    outline in two views (camera models model_a, model_b: RobotEnv.camera_model,
    read with each frame), heights_m the levels to try (metres above the surface).
    keep_a, keep_b: where in each view its edge may count (where the edge
    looked for can be; not the gripper), bool (h, w); None: everywhere.
    """
    import cv2

    view_a, view_b = PlaneView.from_camera_model(model_a), PlaneView.from_camera_model(model_b)
    pts = edge_px(mask_a, border_px, keep_a)
    if len(pts) < 10:
        return Swept(None, float("inf"), (), 0, "the first view shows too little of its edge")
    edge_b = np.zeros(np.asarray(mask_b).shape, np.uint8)
    for p in edge_px(mask_b, border_px, keep_b).astype(int):
        edge_b[p[1], p[0]] = 1
    if edge_b.sum() < 10:
        return Swept(None, float("inf"), (), 0, "the second view shows too little of its edge")
    dist = cv2.distanceTransform((1 - edge_b).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = edge_b.shape
    scores = []
    for z in heights_m:
        q = carry(pts, view_a, view_b, float(z))
        inside = (np.isfinite(q).all(axis=1) & (q[:, 0] >= border_px) & (q[:, 0] < w - border_px)
                  & (q[:, 1] >= border_px) & (q[:, 1] < h - border_px))
        if inside.mean() < MIN_SHARED:
            scores.append((float(z), float("inf"), int(inside.sum())))
            continue
        d = dist[q[inside, 1].astype(int), q[inside, 0].astype(int)]
        scores.append((float(z), float(np.median(d)), int(inside.sum())))
    finite = [s for s in scores if np.isfinite(s[1])]
    if not finite:
        return Swept(None, float("inf"), (), 0, "the two views share no stretch of its edge at any height tried")
    best = min(finite, key=lambda s: s[1])
    near = [s[0] for s in finite if s[1] <= best[1] + 1.0]
    return Swept(best[0], best[1], (min(near), max(near)), best[2])


# ── From edges, not outlines ──────────────────────────────────────────────────
# A see-through container is outlined badly from close up (2026-11-17: the
# segmenter's outline of the clear tub in the wrist camera scored 0.66 and took
# in half of it), but its rim is a strong edge. So the picture itself is swept,
# at its edges: each edge pixel of view A, carried to a height, is compared
# with view B where it lands — the small patch of picture round it, by
# normalised correlation. Every edge at a height lines up there — the table's
# and the floor's at the surface, the rim's at its top — and the rim's, long
# and sharp the length of the wall, most of all above the floor.
#
# Only an edge that crosses the line its pixel moves along as the height
# changes (its epipolar line) says anything about height: one running along it
# lands on itself at every height. Such edges are left out, and the rest
# counted by how squarely they cross it. Comparing patches, not only whether an
# edge lands on an edge, keeps a rim's lip — two or three edges side by side —
# from lining up with itself a step higher or lower (synthetic tub,
# tests/test_place_in.py: the strongest peak within 2 mm, the edges alone up
# to 20 mm off).
#
# Replayed on the 12:01 look at the clear tub (2026-11-17), the peaks stood at
# 62 and 80 mm; the overhead camera put the rim at about 70 (63-75). The two
# looks there differed in every joint (a tub-floor sticker read 40 mm by the
# edges alone): two looks that differ only by a turn about the arm's base
# disagree far less in where the arm thinks the camera is.

PATCH_PX = 3               # patches (2 PATCH_PX + 1) pixels square
MIN_ACROSS = 0.5           # an edge counts only crossing its epipolar line at least this squarely (|cos|)
MIN_PROMINENCE = 0.04      # a peak stands at least this far over the median likeness ...
PEAK_SHARE = 0.5           # ... and at least this share of the strongest peak's prominence
PEAK_TIE = 0.9             # a higher peak this nearly as strong as the strongest is taken instead: the safer
PEAK_WINDOW_M = 0.004      # a peak is the likest within this either side
PEAK_HOLD = 0.9            # ... and it holds, upward, while the likeness stays this much of the peak's
MAX_EDGE_PTS = 3000


def edges(frame_rgb: np.ndarray, keep: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """A picture's strong edges, bool (h, w) — only within keep when given — and each pixel's gradient direction (rad, mod pi)."""
    import cv2

    gray = cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2GRAY), (5, 5), 0)
    e = cv2.Canny(gray, 30, 90) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    if keep is not None:
        e &= np.asarray(keep, bool)
    return e, np.mod(np.arctan2(gy, gx), np.pi)


def _patches(gray: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """(N, (2 PATCH_PX + 1)^2): the picture round each of xy (N, 2), read between pixels, each less its mean and of unit length."""
    r = np.arange(-PATCH_PX, PATCH_PX + 1, dtype=np.float32)
    offsets = np.stack(np.meshgrid(r, r), -1).reshape(-1, 2)
    at = np.asarray(xy, np.float32)[:, None, :] + offsets[None]
    h, w = gray.shape
    x0 = np.clip(np.floor(at[..., 0]).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(at[..., 1]).astype(int), 0, h - 2)
    fx, fy = at[..., 0] - x0, at[..., 1] - y0
    p = (gray[y0, x0] * (1 - fx) * (1 - fy) + gray[y0, x0 + 1] * fx * (1 - fy)
         + gray[y0 + 1, x0] * (1 - fx) * fy + gray[y0 + 1, x0 + 1] * fx * fy)
    p = p - p.mean(axis=1, keepdims=True)
    return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-3)


def edge_scores(frame_a: np.ndarray, model_a: dict, frame_b: np.ndarray, model_b: dict, heights_m,
                keep_a: "np.ndarray | None" = None, keep_b: "np.ndarray | None" = None) -> "tuple[list, int]":
    """
    [(height, likeness, edge pixels that counted)] for each height in heights_m:
    view A's edges (within keep_a) that cross their epipolar line, carried to
    that height, and how like view B is where they land (within keep_b) —
    their patches' correlation, below 0 as 0, weighted by how squarely each
    edge crosses. And how many of A's edge pixels were carried.
    """
    import cv2

    view_a, view_b = PlaneView.from_camera_model(model_a), PlaneView.from_camera_model(model_b)
    ea, turn_a = edges(frame_a, keep_a)
    gray_a = cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(frame_a), cv2.COLOR_RGB2GRAY), (3, 3), 0).astype(np.float32)
    gray_b = cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(frame_b), cv2.COLOR_RGB2GRAY), (3, 3), 0).astype(np.float32)
    h, w = gray_b.shape
    m = PATCH_PX + 1
    pts = np.column_stack(np.nonzero(ea)[::-1]).astype(float)
    pts = pts[(pts[:, 0] >= m) & (pts[:, 0] < ea.shape[1] - m - 1) & (pts[:, 1] >= m) & (pts[:, 1] < ea.shape[0] - m - 1)]
    if len(pts) > MAX_EDGE_PTS:
        pts = pts[np.linspace(0, len(pts) - 1, MAX_EDGE_PTS).astype(int)]
    heights = [float(z) for z in heights_m]
    if len(pts) < 30 or len(heights) < 2:
        return [(z, 0.0, 0) for z in heights], len(pts)
    with np.errstate(all="ignore"):
        lo, hi = carry(pts, view_a, view_b, min(heights)), carry(pts, view_a, view_b, max(heights))
    run = hi - lo
    length = np.linalg.norm(run, axis=1)
    moves = np.isfinite(length) & (length > 1e-6)
    along = np.zeros_like(run)
    along[moves] = run[moves] / length[moves, None]
    turn = turn_a[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    across = np.abs(np.cos(turn) * along[:, 0] + np.sin(turn) * along[:, 1])
    use = moves & (across >= MIN_ACROSS)
    pts, weight = pts[use], across[use]
    if len(pts) < 30:
        return [(z, 0.0, 0) for z in heights], len(pts)
    seen_a = _patches(gray_a, pts)
    keep_b_bool = None if keep_b is None else np.asarray(keep_b, bool)
    out = []
    for z in heights:
        with np.errstate(all="ignore"):
            q = carry(pts, view_a, view_b, z)
        inside = np.isfinite(q).all(axis=1)
        inside[inside] = (q[inside, 0] >= m) & (q[inside, 0] < w - m - 1) & (q[inside, 1] >= m) & (q[inside, 1] < h - m - 1)
        if keep_b_bool is not None and inside.any():
            idx = np.flatnonzero(inside)
            near = np.round(q[idx]).astype(int)
            inside[idx[~keep_b_bool[near[:, 1], near[:, 0]]]] = False
        if inside.sum() < 20:
            out.append((z, 0.0, int(inside.sum())))
            continue
        alike = np.clip((seen_a[inside] * _patches(gray_b, q[inside])).sum(axis=1), 0.0, 1.0)
        wgt = weight[inside]
        out.append((z, float((wgt * alike).sum() / wgt.sum()), int(inside.sum())))
    return out, len(pts)


def top_peak(scores: list, min_height_m: float = 0.02) -> "tuple[float | None, list]":
    """
    Where the top edge stands by edge_scores, at or above min_height_m: (the
    top of the peak or None, [(height, likeness)] of every clear peak). A peak
    is the likest within PEAK_WINDOW_M either side; clear, standing over the
    median likeness by at least MIN_PROMINENCE and PEAK_SHARE of the strongest
    peak's prominence. The strongest — or a higher one PEAK_TIE as strong — and
    its top: as high above it as the likeness holds at PEAK_HOLD of its own;
    two looks a few centimetres apart cannot tell heights within it apart, and
    the higher is the safer.
    """
    if not scores:
        return None, []
    zs = np.array([s[0] for s in scores], float)
    alike = np.array([s[1] for s in scores], float)
    prominence = alike - float(np.median(alike))
    above = zs >= min_height_m - 1e-9
    if not above.any():
        return None, []
    least = max(MIN_PROMINENCE, PEAK_SHARE * float(np.max(prominence[above])))
    peaks: "list[int]" = [int(i) for i in np.flatnonzero(above) if prominence[i] >= least
                          and all(alike[i] >= alike[j] for j in np.flatnonzero(np.abs(zs - zs[i]) <= PEAK_WINDOW_M + 1e-9))]
    if not peaks:
        return None, []
    strongest = max(peaks, key=lambda i: float(alike[i]))
    top = max((i for i in peaks if alike[i] >= PEAK_TIE * alike[strongest]), key=lambda i: float(zs[i]))
    held = top
    while held + 1 < len(zs) and alike[held + 1] >= PEAK_HOLD * alike[top]:
        held += 1
    return float(zs[held]), [(float(zs[i]), float(alike[i])) for i in peaks]


def sweep_edges(frame_a: np.ndarray, model_a: dict, frame_b: np.ndarray, model_b: dict, heights_m,
                keep_a: "np.ndarray | None" = None, keep_b: "np.ndarray | None" = None,
                min_height_m: float = 0.02) -> Swept:
    """
    How high the top edge two views share stands, from their edges: top_peak
    of edge_scores swept both ways — A onto B and B onto A — and averaged.
    Swept.spread_m: from the strongest peak to the height given (how sure);
    Swept.error_px: (1 - the likeness at the strongest peak) x 10, for the
    record.
    """
    ab, n_ab = edge_scores(frame_a, model_a, frame_b, model_b, heights_m, keep_a, keep_b)
    ba, n_ba = edge_scores(frame_b, model_b, frame_a, model_a, heights_m, keep_b, keep_a)
    if min(n_ab, n_ba) < 30:
        return Swept(None, float("inf"), (), 0, "too few edges where the wall should be")
    both = [(z, (x + y) / 2.0, min(n, k)) for (z, x, n), (_, y, k) in zip(ab, ba)]
    top, peaks = top_peak(both, min_height_m)
    if top is None:
        best = max(s[1] for s in both)
        return Swept(None, float("inf"), (), 0,
                     f"the two pictures are alike at no one height clearly (at best {best:.2f})")
    strongest = max(peaks, key=lambda p: p[1])
    return Swept(top, (1.0 - strongest[1]) * 10.0, (strongest[0], top), min(n_ab, n_ba))
