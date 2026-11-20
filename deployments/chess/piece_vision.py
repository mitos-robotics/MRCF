"""
piece_vision.py — find and measure a chess piece in the GRIPPER camera's view.

Two jobs. find_piece locates a piece for visual servoing
(SO101Env.visual_servo_xy) with a classical detector, described below.
measure_piece (at the end) outlines the piece under the jaws and measures it —
height, widths, and, once the camera is located, where its foot is — with the
framework's mrcf/perception/segment.py and mrcf/perception/plane_geometry.py.

How: a piece is whatever is not board
-------------------------------------
The camera is always looking at a chessboard, which is about the most
predictable background there is. So rather than describe a piece, this models
the board and reports what disagrees with it:

  1. Find whatever partial grid of board corners is visible (OpenCV's
     findChessboardCornersSB with CALIB_CB_LARGER — three or four squares is
     enough).
  2. Recover the board plane's pose from those corners with the calibrated
     lens (GRIPPER_CAM_K / GRIPPER_CAM_DIST) by a deterministic consensus, so
     corners a piece or the jaw has corrupted are thrown out instead of
     skewing the fit (see _board_pose and _pose_from_corners for why neither
     RANSAC nor a bare homography would do).
  3. Intersect every pixel's ray with that plane: every pixel now knows which
     square it lies in, and how close it is to a grid line.
  4. Compare each pixel with the median colour of its own square. A piece
     covering less than half a square cannot move that median much, and a
     per-square reference absorbs vignetting and uneven light, which a single
     frame-wide "light square colour" does not.
  5. What remains, cleaned up, is the piece.

No shape assumption, so a knight is as findable as a pawn.

Why not a shape or a colour rule
--------------------------------
"A piece seen from above is round" holds for a sharp pawn seen straight down
and for nothing else on this rig: the camera looks at an angle, so a pawn is a
head on a body, a knight is not round from anywhere, and out of focus nothing
is round. A colour rule fails because a wooden piece is roughly a blend of the
two board colours, so in colour space it sits ON the line between them.

Measured on 36 gripper-camera frames from this rig (34 lens-calibration views
at 8 and 12 cm with wrist tilts and twists, plus 2 servo-height frames; pawn
and knight):
    piece is the strongest detection          33
    nothing returned, board pose not found     2    (safe: servo falls back)
    a wrong detection                          1    (no piece in view; the table
                                                     past the board edge, ~190 px
                                                     from where a servo search
                                                     would look)
    answers that changed across repeat runs    0
~0.2 s per frame.

Being confidently wrong is the failure that matters: find_piece runs
UNATTENDED inside visual_servo_xy during a real pick, so a wrong position moves
the arm, while None just falls back to kinematics. The search radius around
the predicted pixel is the main defence; keep it.

Known weakness: a piece as dark and colourless as the gripper's jaw. The jaw
is masked out by colour (_JAW_L, _JAW_CHROMA), so an ebonised black piece
would be masked with it. jaw_mask(), which masks by place (connected to the
frame's bottom edge), is the fix measure_piece uses; tried here it found more
pieces (58 of 72 frames against 55) but also two pieces on empty-board frames
that had none, and a false detection moves the arm, so this detector keeps
the colour rule. The red-brown wooden "black" pieces of this rig's set are
well clear of it.

Known weakness: hard-edged sunlight patches. Sunlit wood is warmer as well as
brighter, so the edge of a patch crossing a square can read as a piece-sized
triangle. The piece still out-scored them on every frame tried, but draw the
curtain if the sun is on the board.

The point returned is the centroid of the piece's blob. It only has to be
consistent with how GRIPPER_CAM_GRASP_PX was measured, and
calibrate_gripper_servo.py measures that pixel with this same function.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from itertools import combinations

import cv2
import numpy as np

from . import config as cfg

# Board-pose fit. (3, 3) is the smallest grid asked for; CALIB_CB_LARGER lets
# the detector return however much more of the board is visible.
_CB_FLAGS = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE
_PNP_REPROJ_PX = 3.0     # corners further than this from the fitted plane are dropped
_MIN_INLIERS = 6
_CORNER_SEEDS = (0, 1, 2, 3)
_MAX_SUBSETS = 1500      # 4-corner subsets tried; exhaustive up to 17 corners

_EDGE_BAND = 0.10        # fraction of a square either side of a grid line to ignore:
                         # blur makes those pixels a blend of both colours
_MIN_SQUARE_PX = 150     # too few usable pixels to trust a square's median
_BOARD_TOL = 45.0        # Lab distance from its colour's typical square beyond which a
                         # "square" is off the board (table, background). Lenient on
                         # purpose: vignetting and a piece on it both shift a real
                         # square's median, and must not exclude it.
_NEAR_REF = 25.0         # Lab distance counting a pixel as its square's board colour
_MIN_OWN_COLOUR = 0.3    # share of a covered square that must still be board colour
_JAW_L, _JAW_CHROMA = 60, 12   # dark and colourless: the black jaw, not a dark square
# A square too covered to have a colour of its own is judged as covered when
# the board surrounds it (see _residual_and_pose): board on this many of its
# four sides, at least this many squares of its colour diagonally next to it
# to judge it against, and no more than this share of it the jaw.
_COVERED_BESIDE = 3
_COVERED_DIAGONAL = 2
_COVERED_MAX_JAW = 0.1
# A square off its colour's frame-wide reference but uniform (this share of it
# within _NEAR_REF of its own median), nearer its own colour than the other's,
# with board on this many sides, is board under different light: a sunlit
# patch can fail the reference by 49 and, rejected, leave the piece beside it
# without the board round it to be judged.
_UNIFORM_SHARE = 0.6
_LIT_BESIDE = 2
# And one this uniform is board whichever colour it is nearer: a varnished dark
# square mirroring the window can come out nearer the light squares, and judged
# as covered, the whole square is flagged and pulls the piece's centre off. The
# squares a piece really covers measured 0.30-0.41 uniform (piece and square
# both in them); the glare, 1.0. A covered square must be less uniform than this.
_PLAIN_UNIFORM = 0.8
# A covered square is judged against what shows of it, when enough does: its
# pixels within this distance of the neighbours' colour, at least this many and
# this share of it.
_REMNANT_NEAR = 50.0
_REMNANT_MIN_PX = 60
_REMNANT_MIN_SHARE = 0.03
# How much a pixel's lightness counts, against its colour, in how far it is from
# its square's. A shadow changes lightness and hardly colour, so weighting it
# down drops a piece's own shadow from its outline (cream at L 165 beside lit
# cream at 242, the same a and b — it pulled the centre ~13 px, ~4 mm). But the
# shaded side of a CREAM piece is that same colour: at 0.2 the labelled pawn
# went from 20 of 24 found to 4. So full weight; shadows are left to the
# servo's calibration, taken in the same light.
_LIGHTNESS_WEIGHT = 1.0
# An object with more than this share of it within this many px of the jaw is
# the jaw's own edge or shadow, which, alone in view, the servo would centre on. On
# the rig's frames they had 37-100% of themselves there; a knight standing
# right by the jaw tips, 22%.
_JAW_TOUCH_PX = 12
_JAW_TOUCH_MAX = 0.3

_RESIDUAL = 20.0         # Lab distance from its square's median that counts as "not board"
_MIN_FRAGMENT_PX = 150   # smaller specks are noise, not part of anything
_RING_PX = 20            # width of the ring checked around an object
_MIN_BOARD_SUPPORT = 0.6 # fraction of that ring that must be board squares
_MERGE_GAP_PX = 30       # fragments this close (bounding-box gap) are one object
# A whole piece at servo height is ~1500-11000 px; marks and grain on the board
# came out at up to ~950. Measured on tests/piece_frames.
_MIN_AREA_PX = 1200
_MAX_AREA_PX = 60000

DEFAULT_SEARCH_RADIUS_PX = 110

_KEY_OFFSET = 1000       # see residual_map's square index
_KEY_BASE = 100000

_rays_cache: dict = {}


def _rays(w: int, h: int) -> np.ndarray:
    """Undistorted, normalised camera ray (x, y) for every pixel, cached per size."""
    if (w, h) not in _rays_cache:
        k = np.asarray(cfg.GRIPPER_CAM_K, float)
        d = np.asarray(cfg.GRIPPER_CAM_DIST, float)
        px = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 1, 2).astype(np.float32)
        _rays_cache[(w, h)] = cv2.undistortPoints(px, k, d).reshape(h, w, 2)
    return _rays_cache[(w, h)]


def _board_pose(gray: np.ndarray):
    """
    (R, t, extent) of the board plane in camera coordinates, in units of
    squares, or None. `extent` is (u_min, u_max, v_min, v_max) of the inlier
    corners.

    findChessboardCornersSB is itself randomised: on the same frame it can
    return a 3x3, 3x6 or 6x3 grid from call to call, even single-threaded, so
    the servo would see a piece on one call and nothing on the next. Seeding
    OpenCV's RNG makes each call repeatable, but which grid comes back still
    depends on the seed, so this tries a few fixed seeds and keeps the
    best-supported pose.
    """
    best = _best_board_fit(gray)
    return None if best is None else (best[0], best[1], best[3])


_CORNER_POOL = None


def _corner_pool():
    global _CORNER_POOL
    if _CORNER_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _CORNER_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="board-corners")
    return _CORNER_POOL


def _find_corners(img: np.ndarray, seed: int):
    cv2.setRNGSeed(seed)
    return cv2.findChessboardCornersSBWithMeta(img, (3, 3), _CB_FLAGS)


def _best_board_fit(gray: np.ndarray):
    """(R, t, n_inliers, extent) of the best-supported board fit, or None."""
    best = None
    # A contrast-equalised copy often yields a larger patch of grid than the
    # raw frame (3x5 where raw gave 3x3), and with a piece covering part of a
    # small patch, too few corners agree to trust the pose.
    variants = (gray, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))
    # Each search on a thread of its own (OpenCV lets go of the GIL, and its
    # RNG is per thread, so each stays as repeatable as when run in turn);
    # taken in the same order, so ties go the same way.
    jobs = [(img, seed) for img in variants for seed in _CORNER_SEEDS]
    found = list(_corner_pool().map(lambda job: _find_corners(*job), jobs))
    # Different seeds often find the very same grid: its pose is worked out once.
    fits: dict = {}
    for ok, corners, meta in found:
        if not ok:
            continue
        # Corners are refined on the raw frame either way.
        corners = corners.reshape(-1, 2).astype(np.float64)
        key = (meta.shape[1], corners.tobytes())
        if key not in fits:
            fits[key] = _pose_from_corners(corners, meta.shape[1])
        fit = fits[key]
        if fit is not None and (best is None or fit[2] > best[2]):
            best = fit
    return best


def board_inliers(frame_rgb: np.ndarray) -> int:
    """
    How many board corners agree on the plane in this gripper-camera frame; 0
    when the board pose cannot be recovered at all.

    Everything the detector does starts from that pose, so this is the first
    thing that fails near the board's edge, where the camera sees the border
    and the table instead of a grid. Exposed so a reach survey can map where
    servoing is even possible before any piece is involved.
    """
    if cfg.GRIPPER_CAM_K is None or cfg.GRIPPER_CAM_DIST is None:
        return 0
    best = _best_board_fit(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))
    return 0 if best is None else int(best[2])


def _pose_from_corners(corners: np.ndarray, cols: int):
    """
    (R, t, n_inliers, extent) from one detected corner grid, or None.

    Some detected corners are wrong — a piece or the jaw standing in front of
    the lattice gets corners invented on it — so the pose is a consensus: solve
    from each 4-corner subset (IPPE, the planar solver), score by how many of
    all the corners it reprojects within _PNP_REPROJ_PX, keep the best, refine
    on its inliers.

    Not cv2.solvePnPRansac: it samples at random and stops early, so the same
    frame could give different answers, and a servo must give one answer per
    image. Not a bare homography either: it has more freedom than a real
    camera, and when a piece corrupts a whole row of corners it can find a
    self-consistent fit one full square off. The calibrated lens rules that out.
    """
    # Units of squares, not metres: the classification below only needs which
    # square a pixel is in, which is independent of the board's true scale.
    obj = np.array([[i % cols, i // cols, 0] for i in range(len(corners))], np.float64)
    k = np.asarray(cfg.GRIPPER_CAM_K, float)
    d = np.asarray(cfg.GRIPPER_CAM_DIST, float)

    subsets = list(combinations(range(len(obj)), 4))
    if len(subsets) > _MAX_SUBSETS:
        pick = np.random.default_rng(0).choice(len(subsets), _MAX_SUBSETS, replace=False)
        subsets = [subsets[i] for i in sorted(pick)]
    best, best_n, best_err = None, 0, np.inf
    for idx in subsets:
        idx = list(idx)
        # Three collinear lattice points do not pin down a plane.
        if abs(np.linalg.det(np.c_[obj[idx[:3], :2], np.ones(3)])) < 1e-6:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj[idx], corners[idx], k, d, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, k, d)
        err = np.linalg.norm(proj.reshape(-1, 2) - corners, axis=1)
        inl = err < _PNP_REPROJ_PX
        n_inl = int(inl.sum())
        if n_inl > best_n or (n_inl == best_n and n_inl and err[inl].mean() < best_err):
            best, best_n, best_err = inl, n_inl, float(err[inl].mean())
            if best_n == len(obj):
                # Every corner agrees: the refinement below uses them all,
                # whichever subset found it — nothing left to search for.
                break
    if best is None or best_n < _MIN_INLIERS:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj[best], corners[best], k, d, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    # Lattice extent of the corners that agreed: squares touching these are
    # certainly real board, which residual_map needs to know.
    gx, gy = obj[best, 0], obj[best, 1]
    extent = (gx.min(), gx.max(), gy.min(), gy.max())
    return cv2.Rodrigues(rvec)[0], tvec.ravel(), best_n, extent


def _residual(lab_px: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance of Lab pixels from a square's colour, lightness weighted down (_LIGHTNESS_WEIGHT)."""
    d = lab_px - ref
    return np.sqrt((_LIGHTNESS_WEIGHT * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def residual_map(frame_rgb: np.ndarray) -> "np.ndarray | None":
    """
    Per-pixel Lab distance from the pixel's own board square's median colour,
    0 where unjudged (grid lines, jaw, off-board). None if the board pose could
    not be recovered. Exposed so tools can show what the detector reacts to.
    """
    out = _residual_and_pose(frame_rgb)
    return None if out is None else out[0]


def _residual_and_pose(frame_rgb: np.ndarray):
    h, w = frame_rgb.shape[:2]
    pose = _board_pose(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))
    if pose is None:
        return None
    rot, t, (u_lo, u_hi, v_lo, v_hi) = pose

    # Every pixel's lens-corrected ray, intersected with the board plane Z=0.
    ray = np.dstack([_rays(w, h), np.ones((h, w))])
    normal = rot[:, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = (normal @ t) / (ray @ normal)
        on_plane = ray * s[..., None] - t
    board = on_plane @ rot                       # rot.T @ (X - t), per pixel
    u, v = board[..., 0], board[..., 1]
    in_front = np.isfinite(u) & np.isfinite(v) & (s > 0)
    u, v = np.where(in_front, u, 0.0), np.where(in_front, v, 0.0)

    fu, fv = u - np.floor(u), v - np.floor(v)
    near_line = np.minimum.reduce([fu, 1 - fu, fv, 1 - fv]) < _EDGE_BAND
    lab = cv2.cvtColor(cv2.GaussianBlur(frame_rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    jaw = (lab[..., 0] < _JAW_L) & (np.hypot(lab[..., 1] - 128, lab[..., 2] - 128) < _JAW_CHROMA)

    usable = in_front & ~near_line & ~jaw
    # Packed (u, v) square index. Offset so both halves are non-negative:
    # squares left of or above the detected corners have negative indices, and
    # unpacking a negative key with // and % gives the wrong square — with the
    # wrong light/dark parity, so it would be judged against the other colour.
    square = (np.floor(u).astype(np.int64) + _KEY_OFFSET) * _KEY_BASE \
        + (np.floor(v).astype(np.int64) + _KEY_OFFSET)
    keys, inv = np.unique(square[usable], return_inverse=True)
    lab_u = lab[usable]
    su, sv = keys // _KEY_BASE - _KEY_OFFSET, keys % _KEY_BASE - _KEY_OFFSET
    parity = (su + sv) % 2

    medians = np.full((len(keys), 3), np.nan, np.float32)
    for k in range(len(keys)):
        sel = inv == k
        if sel.sum() >= _MIN_SQUARE_PX:
            medians[k] = np.median(lab_u[sel], axis=0)

    out = np.zeros(len(lab_u), np.float32)
    accepted = []
    judged_as = {}          # (u, v) of each accepted square -> the colour it is judged against
    covered = []            # off-colour squares with little of their own colour left: (k, refs)
    parity_refs = {}        # each colour's references
    for p in (0, 1):
        judged = (parity == p) & ~np.isnan(medians[:, 0])
        # The typical colour comes only from squares touching corners the
        # detector found, which are certainly board. Taken over every square in
        # view it would include the virtual ones extended over the dark border
        # and the table; on a view with plenty of border that drags "typical
        # light square" so far that every light square fails the test below,
        # and a piece standing on one is never found.
        anchored = judged & (su >= u_lo - 1) & (su <= u_hi) & (sv >= v_lo - 1) & (sv <= v_hi)
        if not judged.any():
            continue
        # Either reference may be the trustworthy one: the anchored set can be
        # just two or three squares under one patch of light, and the all-square
        # set is fine whenever little border is in view. A square that matches
        # either is board.
        refs = [np.median(medians[judged], axis=0)]
        if anchored.any():
            refs.append(np.median(medians[anchored], axis=0))
        parity_refs[p] = refs

    def own_colour(k, p, pix):
        """
        The colour to judge square k by: its pixels nearer its own colour than
        the other's, when enough are. Its plain median is dragged toward a
        piece over part of it, and the square's own visible part, judged
        against that, comes out as piece: the "aligned" centre then sits
        beside the piece and the jaws close on nothing.
        """
        if 1 - p not in parity_refs:
            return medians[k]
        d_own = np.min([np.linalg.norm(pix - r, axis=-1) for r in parity_refs[p]], axis=0)
        d_other = np.min([np.linalg.norm(pix - r, axis=-1) for r in parity_refs[1 - p]], axis=0)
        mine = d_own < d_other
        return np.median(pix[mine], axis=0) if mine.mean() >= _MIN_OWN_COLOUR else medians[k]

    for p in (0, 1):
        if p not in parity_refs:
            continue
        refs = parity_refs[p]
        judged = (parity == p) & ~np.isnan(medians[:, 0])
        for k in np.nonzero(judged)[0]:
            sel = inv == k
            if min(np.linalg.norm(medians[k] - r) for r in refs) <= _BOARD_TOL:
                ref = own_colour(k, p, lab_u[sel])
            else:
                # The median is off — but a piece covering most of its square
                # drags the median toward itself, and skipping the square would
                # hide the very piece standing on it. A real square still
                # has a solid share of its own colour round the piece; the
                # border, the table and the cables past the edge do not.
                near = np.min([np.linalg.norm(lab_u[sel] - r, axis=-1) for r in refs], axis=0) < _NEAR_REF
                if near.mean() < _MIN_OWN_COLOUR:
                    covered.append((k, refs))
                    continue
                ref = np.median(lab_u[sel][near], axis=0)
            out[sel] = _residual(lab_u[sel], ref)
            accepted.append(keys[k])
            judged_as[(int(su[k]), int(sv[k]))] = ref

    # A square with almost none of its own colour left is the border, the table
    # — or a square a piece stands over. From the servo's hover a wide piece
    # seen from above covers 90% of its square, and skipping that square would
    # hide the piece completely. The board decides: a square with board on
    # three or four sides is inside it; one past the edge has board on one
    # side, two at a corner. Judged against the squares of its own colour
    # diagonally next to it — never a frame-wide colour, which reads the darker
    # corners of the frame as objects — and not where the jaw covers it, whose
    # edges and shadow would do the same.
    #
    # Light is not even across a frame, so the board is grown from the squares
    # already accepted: an off-colour square that is uniform and nearer its
    # own colour, with board on two sides, is board under other light (see
    # _UNIFORM_SHARE); one past the board's edge has board on one side only.
    # Each square taken lets its neighbours be judged, so round until nothing
    # changes.
    framed = in_front & ~near_line
    all_keys, all_n = np.unique(square[framed], return_counts=True)
    jaw_keys, jaw_n = np.unique(square[framed & jaw], return_counts=True)
    jaw_share = dict(zip(jaw_keys.tolist(), (jaw_n / all_n[np.searchsorted(all_keys, jaw_keys)]).tolist()))

    def uniformity(k) -> float:
        return float((np.linalg.norm(lab_u[inv == k] - medians[k], axis=-1) < _NEAR_REF).mean())

    def lit_differently(k) -> bool:
        p, uniform = int(parity[k]), uniformity(k)
        if uniform >= _PLAIN_UNIFORM:
            return True
        if 1 - p not in parity_refs:
            return False
        nearest = [min(np.linalg.norm(medians[k] - r) for r in parity_refs[q]) for q in (p, 1 - p)]
        return uniform >= _UNIFORM_SHARE and nearest[0] < nearest[1]

    for _ in range(6):
        still = []
        for k, refs in covered:
            u0, v0 = int(su[k]), int(sv[k])
            beside = sum((u0 + du, v0 + dv) in judged_as for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            diagonal = [judged_as[(u0 + du, v0 + dv)] for du in (-1, 1) for dv in (-1, 1)
                        if (u0 + du, v0 + dv) in judged_as]
            if beside >= _LIT_BESIDE and lit_differently(k):
                ref = medians[k]
            elif (beside >= _COVERED_BESIDE and len(diagonal) >= _COVERED_DIAGONAL
                    and jaw_share.get(int(keys[k]), 0.0) <= _COVERED_MAX_JAW
                    and uniformity(k) < _PLAIN_UNIFORM):
                ref = np.median(diagonal, axis=0)
                # What shows of the square itself is the better reference: the
                # neighbours' colour, 20+ off it under uneven light, flags the
                # visible corner of a covered square with the piece and pulls
                # the "aligned" centre off the piece. Its own pixels near that
                # colour are the square; the piece is far off it.
                pix = lab_u[inv == k]
                own = np.linalg.norm(pix - ref, axis=-1) < _REMNANT_NEAR
                if own.sum() >= max(_REMNANT_MIN_PX, _REMNANT_MIN_SHARE * len(pix)):
                    ref = np.median(pix[own], axis=0)
            else:
                still.append((k, refs))
                continue
            sel = inv == k
            out[sel] = _residual(lab_u[sel], ref)
            accepted.append(keys[k])
            judged_as[(u0, v0)] = ref
        if len(still) == len(covered):
            break
        covered = still

    res = np.zeros((h, w), np.float32)
    res[usable] = out
    # Pixels on squares judged to be real board, grid lines included.
    on_board = in_front & np.isin(square, np.asarray(accepted, dtype=np.int64))
    return res, on_board


def _blobs(frame_rgb: np.ndarray) -> list:
    """
    [(x, y, r_equiv, area), ...] piece-like OBJECTS, largest first; empty if
    the board was not found.

    One piece often comes out as several fragments: a pawn's neck is the same
    tone as the square behind it, so the head and the base separate. Returning
    one fragment or the other at random makes the reported point jump ~50 px
    between frames, and the base alone is small enough to lose to a mark on
    the board. So fragments closer than _MERGE_GAP_PX are grouped into one
    object, reported at the area-weighted centre of the group.
    """
    got = _residual_and_pose(frame_rgb)
    if got is None:
        return []
    res, on_board = got
    # The jaw by place (jaw_mask), not by colour: dark grain on the dark
    # squares has the jaw's colour too, and would reject nearly every piece.
    near_jaw = cv2.dilate(jaw_mask(frame_rgb).astype(np.uint8),
                          np.ones((2 * _JAW_TOUCH_PX + 1,) * 2, np.uint8)).astype(bool)
    mask = (res > _RESIDUAL).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Bridge the ignored grid-line bands so a piece straddling one stays one blob.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    frags = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= _MIN_FRAGMENT_PX]

    # Union-find on bounding-box gaps.
    parent = {i: i for i in frags}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def gap(i, j):
        xi, yi, wi, hi = stats[i, :4]
        xj, yj, wj, hj = stats[j, :4]
        dx = max(0, max(xi, xj) - min(xi + wi, xj + wj))
        dy = max(0, max(yi, yj) - min(yi + hi, yj + hj))
        return float(np.hypot(dx, dy))

    for ai, i in enumerate(frags):
        for j in frags[ai + 1:]:
            if gap(i, j) <= _MERGE_GAP_PX:
                parent[root(i)] = root(j)

    groups: dict = {}
    for i in frags:
        groups.setdefault(root(i), []).append(i)
    out = []
    for members in groups.values():
        areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in members], float)
        total = float(areas.sum())
        if not _MIN_AREA_PX <= total <= _MAX_AREA_PX:
            continue
        # A piece stands ON the board, so the ring of pixels around it is
        # board. Something on the border, the table or the cables past the
        # board's edge is surrounded by unjudged pixels instead.
        obj = np.isin(labels, members).astype(np.uint8)
        ring = cv2.dilate(obj, np.ones((2 * _RING_PX + 1, 2 * _RING_PX + 1), np.uint8)).astype(bool) & ~obj.astype(bool)
        if ring.any() and on_board[ring].mean() < _MIN_BOARD_SUPPORT:
            continue
        if near_jaw[obj.astype(bool)].mean() > _JAW_TOUCH_MAX:
            continue                     # the jaw's edge or shadow (see _JAW_TOUCH_PX)
        c = (np.array([cents[i] for i in members]) * areas[:, None]).sum(0) / total
        out.append((float(c[0]), float(c[1]), float(np.sqrt(total / np.pi)), total))
    return sorted(out, key=lambda b: -b[3])


def candidates(frame_rgb: np.ndarray) -> list:
    """
    Every piece-like region in frame: [(x, y, r_equiv, area_px), ...], largest
    first.

    Public so tools can say WHY a detection failed. "Cannot see the piece" and
    "I can see it, 169 px away, outside the 110 px search radius" are different
    problems with different fixes, and only the second tells you to move the
    piece rather than doubt the detector.
    """
    return _blobs(frame_rgb)


def find_piece(
    frame_rgb: np.ndarray,
    expected_px: "tuple[float, float]",
    search_radius_px: float = DEFAULT_SEARCH_RADIUS_PX,
    rank: str = "nearest",
) -> "tuple[float, float] | None":
    """
    Centre of the piece nearest `expected_px`, or None if nothing qualifies.

    `rank` picks between candidates once they are found, and the right choice
    depends on whether the caller has a real prior:

      "nearest"   (default) the one closest to expected_px. Correct during
                  servoing, where the kinematics already put the prediction
                  within a few millimetres, so proximity is strong evidence.
      "strongest" the largest. Correct when expected_px is a placeholder with
                  no predictive value, as in a calibration's first look.
    """
    if cfg.GRIPPER_CAM_K is None or cfg.GRIPPER_CAM_DIST is None:
        return None
    expected = np.asarray(expected_px, dtype=float)
    near = [b for b in _blobs(frame_rgb)
            if np.linalg.norm(np.array(b[:2]) - expected) <= search_radius_px]
    if not near:
        return None
    if rank == "strongest":
        best = max(near, key=lambda b: b[3])
    else:
        best = min(near, key=lambda b: float(np.linalg.norm(np.array(b[:2]) - expected)))
    return best[0], best[1]


def annotate(frame_rgb: np.ndarray, expected_px, found_px, radius_px: float) -> np.ndarray:
    """BGR image showing the prediction, the search disc and what was found."""
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ex, ey = int(expected_px[0]), int(expected_px[1])
    cv2.circle(img, (ex, ey), int(radius_px), (0, 180, 255), 1)
    cv2.drawMarker(img, (ex, ey), (0, 180, 255), cv2.MARKER_CROSS, 16, 2)
    if found_px is not None:
        fx, fy = int(found_px[0]), int(found_px[1])
        cv2.drawMarker(img, (fx, fy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.arrowedLine(img, (ex, ey), (fx, fy), (0, 0, 255), 2, tipLength=0.25)
    return img


# ── Measuring the piece under the jaws ─────────────────────────────────────────
#
# The overhead camera looks almost straight down, which is the one view a
# piece's height cannot be read from. This camera looks at the board at a
# slant from a few centimetres away — ~1.5 px of image per mm of height — so
# at the hover it measures what the overhead scan cannot: height and the
# width at every height, which is what the grasp height and the grip are
# chosen from (see mrcf/policies/parallel_jaw/planning.py, plan_from_shape). Outline by segment.py, maths
# by plane_geometry.
#
# Where the camera is, two ways. With the backend's camera model (calibrated
# by calibrate_gripper_pose.py: the camera's pose from the joint angles), on
# any surface, and the foot comes out in the arm's frame — where to aim the
# jaws. Without it, from the board's own pattern: sizes, but no position. The
# board, when it is in view, also tells a piece from a square.

_MEASURE_SEARCH_PX = 140      # prompts cover this far either side of the expected pixel
_MEASURE_PROMPT_STEP_PX = 20
_JAW_MARGIN_PX = 4
_MIN_HEIGHT_MM, _MAX_HEIGHT_MM = 15.0, 120.0
_MAX_FOOT_RADIUS_MM = 18.0    # wider than any piece's foot on a 40 mm board
# Where on a piece to aim the prompts when its foot's position is known: well
# above the foot, inside even a pawn's body.
_PROMPT_LIFT_MM = 20.0


def jaw_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Where the gripper's own jaw is in the frame: dark, colourless, and
    attached to the frame's bottom edge, which the camera mount guarantees.

    By place, not by colour alone. The colour rule (_JAW_L, _JAW_CHROMA) also
    matches an ebonised black piece, and used on its own it would have erased
    every black piece from view.
    """
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dark = ((lab[..., 0] < _JAW_L) & (np.hypot(lab[..., 1] - 128, lab[..., 2] - 128) < _JAW_CHROMA))
    n, labels, _, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    bottom = set(np.unique(labels[-1])) - {0}
    jaw = np.isin(labels, list(bottom))
    return cv2.dilate(jaw.astype(np.uint8), np.ones((2 * _JAW_MARGIN_PX + 1,) * 2, np.uint8)).astype(bool)


def gripper_board_view(frame_rgb: np.ndarray):
    """piece_geometry.BoardView for a gripper-camera frame, or None without a board pose."""
    from . import piece_geometry as geo

    if cfg.GRIPPER_CAM_K is None or cfg.GRIPPER_CAM_DIST is None:
        return None
    pose = _board_pose(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))
    if pose is None:
        return None
    pitch = cfg.CHESSBOARD_SQUARE_SIZE_M * 1000.0
    rot, t, _ = pose
    return geo.BoardView(np.asarray(cfg.GRIPPER_CAM_K, float), np.asarray(cfg.GRIPPER_CAM_DIST, float),
                         rot, t * pitch, pitch_mm=pitch)      # _board_pose works in squares


def measure_piece(frame_rgb: np.ndarray, expected_px=None, model: "dict | None" = None,
                  expected_xy_m=None, known=None, target: "str | None" = None, need_height: bool = True) -> dict:
    """
    Measure the piece nearest where it is expected in a gripper-camera frame:
    the framework's plane_geometry.measure_near, plus what this rig's camera
    and a chessboard add — the jaw, which is in every frame, is not a piece;
    nor is a patch of board, told by its edges running along the grid; and
    without a located camera, the board in view gives the pose.

    model is the backend's camera model for this frame (RobotEnv.camera_model,
    read with the arm still), or None to use the board's pattern. The piece is
    expected at expected_xy_m (its foot, arm frame) when the model can place
    that in the image, else at expected_px, else mid-frame.

    With a model, known, target and need_height are measure_near's: the
    other pieces the scan found near this one, so a neighbour's outline is not
    taken for it (on a full board from over b1, the c1 bishop's was).

    Returns {"ok", "why", and when ok: "measurement"
    (plane_geometry.ObjectMeasurement: height_mm, widths_mm, and base_world_m
    when a model was given), "mask", "score", "occluded"}. "occluded" means the
    outline runs into the jaw or off the frame's edge, so the top may be hidden
    and the height is then only a lower bound.
    """
    from . import piece_geometry as geo
    from mrcf.perception import plane_geometry as pg
    from mrcf.perception import segment

    if not segment.available():
        return {"ok": False, "why": segment.why_unavailable()}
    board = gripper_board_view(frame_rgb)
    located = pg.PlaneView.from_camera_model(model) if model is not None else None
    view = located or board
    if view is None:
        return {"ok": False, "why": "no camera pose — GRIPPER_CAM_POSE not calibrated and no board in view"}
    jaw = jaw_mask(frame_rgb)

    def reject(m):
        if (m & jaw).sum() > 0.5 * m.sum():
            return True                                # the jaw itself
        return board is not None and geo.grid_edge_fraction(m, board) > geo.GRID_EDGE_FRACTION

    among = {"known": known, "target": target, "need_height": need_height} if located is not None else {}
    r = pg.measure_near(frame_rgb, view, expected_px, expected_xy_m, reject_mask=reject,
                        foot_radius_mm=(4.0, _MAX_FOOT_RADIUS_MM), height_mm=(_MIN_HEIGHT_MM, _MAX_HEIGHT_MM),
                        search_px=_MEASURE_SEARCH_PX, step_px=_MEASURE_PROMPT_STEP_PX,
                        prompt_lift_mm=_PROMPT_LIFT_MM, **among)
    if r["ok"]:
        r["occluded"] = bool(r.get("occluded")) or bool((r["mask"] & jaw).any())
    return r
