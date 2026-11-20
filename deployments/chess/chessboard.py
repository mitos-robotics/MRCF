"""
chessboard.py — chess-specific perception for the chess deployment.

Replaces open-ended VLM object discovery with something the domain lets us do
far better.

The generic pipeline asks a 7B vision model to invent object names and regress
pixel coordinates for them. Both are things it is weak at: it names what stands
out most, the robot and the furniture over the pieces, and a pixel estimate a few
percent off is a centimetre of grasp error. Neither is necessary here, because
a chessboard is a known, rigid, highly structured object:

  - WHERE the squares are is pure geometry. OpenCV finds the 49 internal
    corners to sub-pixel accuracy, and the calibrated homography turns those
    into robot-frame coordinates. Every one of the 64 square centres then
    follows by arithmetic — no model involved, no estimation error.

  - WHAT is on each square is a discrete classification over a fixed, known
    vocabulary ("which of these squares has a piece, and which piece"), which
    is enormously easier for a small model than free-form localisation. We
    also draw the square labels onto the image we send it, so it reads a label
    off the picture instead of computing coordinates.

A piece's position is therefore the computed centre of the square it stands
on, and its accuracy is the board detection's, not the language model's.

Squares are named as a player sitting where the arm's base is would name them
(config.ROBOT_PLAYS): playing white, rank 1 is the row nearest the arm and the
a-file is on its left. annotate() writes a labelled image so this is checkable
by eye rather than assumed, and every scan checks the board's own colours agree
(a1 is a dark square).

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

import time
from pathlib import Path as _P

import numpy as np

import cv2

from mrcf.perception import outlines, tones
from mrcf.perception.outlines import keep_outlines

from . import config as cfg
from mrcf import privacy

FILES = "abcdefgh"
N_SQUARES = 8


# ── Board detection ────────────────────────────────────────────────────────────

def _preprocess_variants(frame_rgb: np.ndarray) -> list:
    """
    Several grayscale renderings of one frame, tried in order. Each helps with
    a different failure mode: uneven lighting and glare, low contrast between
    the two square colours (wooden boards are brown/cream, not black/white),
    and one colour channel separating the two better than luminance does.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    variants = [gray]
    # The CLAHE clip limit matters as much as which channel: 3.0 rescues a
    # washed-out or unevenly lit board, 1.0 a board whose squares already
    # separate cleanly and that 3.0 over-amplifies into noise. One real frame
    # from this rig is found ONLY at clip 1.0 on the blue channel, another
    # only at 3.0 on green, so both passes earn their keep.
    for clip in (3.0, 1.0):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        variants.append(clahe.apply(gray))
        # Individual colour channels often separate two specific board colours
        # (brown/cream) far better than the luminance-weighted gray does.
        for i in range(3):
            variants.append(clahe.apply(frame_rgb[:, :, i]))
    # Saturation ignores brightness entirely, so it survives the glare and the
    # shadow gradient that a window throws across the board.
    variants.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)[:, :, 1])
    return variants


def detect_corners(frame_rgb: np.ndarray) -> "np.ndarray | None":
    """(49, 2) sub-pixel internal-corner pixel coords, or None if not found."""
    pattern = (cfg.CHESSBOARD_INTERNAL_COLS, cfg.CHESSBOARD_INTERNAL_ROWS)
    for variant in _preprocess_variants(frame_rgb):
        found, corners = cv2.findChessboardCornersSB(
            variant, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )
        if found:
            return corners.reshape(-1, 2)
        found, corners = cv2.findChessboardCorners(
            variant, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        )
        if found:
            cv2.cornerSubPix(
                variant, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
            )
            return corners.reshape(-1, 2)
    return None


def grid_index(flat_i: int) -> "tuple[int, int]":
    """(row, col) for a flat index into the corner array OpenCV returns."""
    return flat_i // cfg.CHESSBOARD_INTERNAL_COLS, flat_i % cfg.CHESSBOARD_INTERNAL_COLS


def fit_similarity(src: np.ndarray, dst: np.ndarray):
    """
    Least-squares similarity (uniform scale + rotation + translation, mirror
    allowed) mapping grid coordinates to world coordinates — Umeyama.

    Constrained to similarity rather than a general affine on purpose: a
    chessboard's cells are square and its axes perpendicular, so a fit that
    could skew would be fitting something no physical board can be. Mirrors ARE
    allowed, because corner indices come from the image and a camera looking
    down at a z-up table has image axes that are left-handed relative to the
    world.

    Returns (scale, R, t, rms_residual_m).
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = (d_c.T @ s_c) / len(src)
    U, sv, Vt = np.linalg.svd(cov)
    var_s = (s_c ** 2).sum() / len(src)
    best = None
    for flip in (1.0, -1.0):
        d = np.array([1.0, flip])
        R = U @ np.diag(d) @ Vt
        scale = float((sv * d).sum() / var_s) if var_s > 0 else 0.0
        if scale <= 0:
            continue
        t = mu_d - scale * (R @ mu_s)
        pred = (scale * (R @ src.T)).T + t
        rms = float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))
        if best is None or rms < best[3]:
            best = (scale, R, t, rms)
    return best


# ── Square geometry ────────────────────────────────────────────────────────────

def _corner_world_positions(corners_px: np.ndarray, homography: np.ndarray,
                            capture_wh, render_wh) -> np.ndarray:
    """Back-project detected corners into the robot frame via the homography."""
    scale = np.array([render_wh[0] / capture_wh[0], render_wh[1] / capture_wh[1]])
    pts = corners_px * scale
    h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (np.asarray(homography, float) @ h.T).T
    return proj[:, :2] / proj[:, 2:3]


def capture_spot(squares: dict, memory=None) -> "np.ndarray | None":
    """
    Where the next captured piece goes (config.CAPTURE_FILE, CAPTURE_BORDER_M,
    CAPTURE_SLOTS): on the border past that file, beside ranks 1-2 for the
    first, a square further along for each the memory has put there since. None
    when the slots are full or the board's squares are not known.
    """
    f, other = cfg.CAPTURE_FILE, "b" if cfg.CAPTURE_FILE == "a" else "g"
    if any(f"{c}{r}" not in squares for c in (f, other) for r in (1, 2)):
        return None
    taken = 0
    if memory is not None:
        taken = sum(1 for state in ("present", "missing") for t in memory.in_state(state)
                    if t.place == cfg.CAPTURE_TARGET)
    if taken >= cfg.CAPTURE_SLOTS:
        return None
    at = lambda sq: np.asarray(squares[sq]["world"], float)[:2]           # noqa: E731
    out = at(f"{f}1") - at(f"{other}1")
    out /= np.linalg.norm(out) or 1.0
    along = at(f"{f}2") - at(f"{f}1")
    return (at(f"{f}1") + at(f"{f}2")) / 2 + along * taken + out * cfg.CAPTURE_BORDER_M


def square_map(corners_px: np.ndarray, homography: np.ndarray, capture_wh, render_wh) -> dict:
    """
    Compute every square's centre, in both the robot frame and image pixels.

    Internal corners are the board's inner lattice points: internal corner
    (i, j) is lattice point (i+1, j+1), and square (R, C) is centred at lattice
    (R+0.5, C+0.5) — i.e. internal-corner coordinates (R-0.5, C-0.5). So the
    same two fitted maps extrapolate cleanly to the outer squares the internal
    corners never touch.

    Returns (squares, pitch_m, rms_m): squares is
    {square_name: {"world": (x, y), "pixel": (u, v), "rc": (R, C)}}, pitch_m the
    square size in the robot frame, rms_m the lattice fit's residual.
    """
    world_corners = _corner_world_positions(corners_px, homography, capture_wh, render_wh)
    grid = np.array([[c, r] for r, c in (grid_index(i) for i in range(len(corners_px)))], float)

    scale, rot, trans, rms = fit_similarity(grid, world_corners)

    # grid -> pixel keeps full perspective, so drawn labels land correctly even
    # on an oblique view; grid -> world is the similarity above.
    h_grid_to_px, _ = cv2.findHomography(grid, corners_px, method=0)

    squares = {}
    for R in range(N_SQUARES):
        for C in range(N_SQUARES):
            g = np.array([C - 0.5, R - 0.5])
            world = scale * (rot @ g) + trans
            p = h_grid_to_px @ np.array([g[0], g[1], 1.0])
            squares[(R, C)] = {
                "world": np.array([world[0], world[1]]),
                "pixel": (float(p[0] / p[2]), float(p[1] / p[2])),
            }
    return _name_squares(squares), scale, rms


def _name_squares(by_rc: dict) -> dict:
    """
    Assign a1-h8 to computed squares as a player in the arm's seat would.

    The arm's base frame has +x pointing out over the board and +y to the
    arm's left (z up), so "nearest row" is the smallest x and "left" is +y.
    For config.ROBOT_PLAYS = "white": rank 1 is the nearest row and the a-file
    is at the left; for "black": rank 8 nearest, a-file at the right.

    What this cannot fix: a report that the piece is on the square NEXT TO
    where this labels it — not mirrored, just off by one. Check against the
    LABELLED image (annotate(), board_overlay.png) rather than counting on the
    live board by eye — an oblique view of an unmarked board is an easy
    one-square miscount, and it looks identical to a labelling bug until
    compared against the photo this function produced.
    """
    rcs = list(by_rc)

    # The two grid axes in world terms, from the corners of the (R, C) lattice.
    def world(rc):
        return by_rc[rc]["world"]
    axis_r = world((N_SQUARES - 1, 0)) - world((0, 0))
    axis_c = world((0, N_SQUARES - 1)) - world((0, 0))

    # Whichever lattice axis points more along +x runs away from the arm.
    rank_along_r = abs(axis_r[0]) >= abs(axis_c[0])
    away = np.sign((axis_r if rank_along_r else axis_c)[0]) or 1.0
    leftward = np.sign((axis_c if rank_along_r else axis_r)[1]) or 1.0
    white = str(getattr(cfg, "ROBOT_PLAYS", "white")).lower() != "black"

    named = {}
    for rc in rcs:
        r, c = rc
        out_i = r if rank_along_r else c         # 0 = nearest the arm when away > 0
        side_i = c if rank_along_r else r        # 0 = rightmost when leftward > 0
        if away < 0:
            out_i = N_SQUARES - 1 - out_i
        if leftward < 0:
            side_i = N_SQUARES - 1 - side_i
        # Playing white: rank counts away from the arm, files count from its
        # left. Playing black: both the other way round.
        rank_i = out_i if white else N_SQUARES - 1 - out_i
        file_i = N_SQUARES - 1 - side_i if white else side_i
        named[f"{FILES[file_i]}{rank_i + 1}"] = dict(by_rc[rc], rc=rc)
    return named


def check_orientation(frame_rgb: np.ndarray, squares: dict, occupied=()) -> "str | None":
    """
    A warning if the board is not set up the way a player in the arm's seat
    would have it, else None.

    Every chessboard has a dark square at each player's near-left corner, so
    a1 must be dark. If the squares a1's colour are the light ones, the board
    is turned a quarter-turn: the names are then right about rows and files
    but every square's colour is wrong, which a person reading the board will
    notice. Judged on all empty squares of each colour, not a1 alone, since a1
    may have a piece on it or a glare across it.
    """
    lab_l = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)[..., 0]
    a1_parity = sum(squares["a1"]["rc"]) % 2
    same, other = [], []
    for name, sq in squares.items():
        if name in occupied:
            continue
        u, v = (int(x) for x in sq["pixel"])
        patch = lab_l[max(v - 6, 0):v + 7, max(u - 6, 0):u + 7]
        if patch.size:
            (same if sum(sq["rc"]) % 2 == a1_parity else other).append(float(np.median(patch)))
    if not same or not other or np.median(same) < np.median(other):
        return None
    return ("a1 came out a light square, and should be dark (light on the right). "
            "Either the corner grid was found one square off — compare the labelled "
            "image; delete board_corners.npy and rescan with the board clear — or "
            "the board is turned a quarter-turn from how a player in the arm's seat "
            "would set it up.")


def lattice_from_tags(frame_rgb: np.ndarray) -> "np.ndarray | None":
    """
    The board's 49 inner corners, as detect_corners gives them, from its tags
    and whichever corners they can find (board_tags.measure_board) — for
    when a piece hides corners and there is no lattice cached. None without a
    calibrated overhead camera, enough tags in view, or a1 coming out dark.
    """
    model = cfg.OVERHEAD_CAM_MODEL                  # the arm's, served live
    if model is None or tuple(model["size"]) != (frame_rgb.shape[1], frame_rgb.shape[0]):
        return None
    from . import board_tags
    board = board_tags.measure_board(frame_rgb, np.asarray(model["K"], float), np.asarray(model["D"], float))
    if board is None or not board["a1_dark"]:
        return None
    corners = np.asarray(board["lattice_all_px"], float)
    return corners if _lattice_fits(frame_rgb, corners) else None


def _lattice_fits(frame_rgb: np.ndarray, corners: np.ndarray) -> bool:
    """Whether a detected corner grid puts a dark square on a1 in this frame (check_orientation)."""
    squares, _, _ = square_map(corners, cfg.OVERHEAD_HOMOGRAPHY, (CAPTURE_W, CAPTURE_H),
                               (cfg.RENDER_W, cfg.RENDER_H))
    return check_orientation(frame_rgb, squares) is None


def annotate(frame_rgb: np.ndarray, squares: dict) -> np.ndarray:
    """Draw each square's name at its computed centre (BGR image, for saving)."""
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for name, sq in squares.items():
        u, v = int(sq["pixel"][0]), int(sq["pixel"][1])
        cv2.circle(img, (u, v), 2, (0, 255, 0), -1)
        cv2.putText(img, name, (u - 12, v + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return img


# ── Perception ─────────────────────────────────────────────────────────────────

# Detection resolution for this module's own board scan. Higher than
# RENDER_W/H because corner detection on an oblique, low-contrast board needs
# the pixels; square coordinates are computed in world units afterwards, so
# this does not have to match what anything else uses.
CAPTURE_W, CAPTURE_H = 1280, 720

# Last successfully detected internal corners, at CAPTURE_W x CAPTURE_H.
_CORNER_CACHE = "board_corners.npy"

def _estimate_pitch_px(squares: dict) -> float:
    """Centre-to-centre pixel distance between adjacent squares (nearest-neighbour)."""
    pts = np.array([s["pixel"] for s in squares.values()])
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))


# ── What each piece is: the vision model, shown every piece at once ──────────
#
# Whether a piece is there, and where, is the camera's outlines' to say
# (measure_pieces): a vision model shown the board puts pieces on empty
# squares. What a piece IS — its colour and type — is the model's, shown the
# pieces it is asked about cut from the frame, each standing upright, numbered
# on one sheet (perception/tiles.py), in one call. Its answer for each is held
# to LABEL_KINDS; "not a piece" stands only if a second frame says so too.
# Only the pieces the memory cannot already name are asked about (discover).

NOT_A_PIECE = "not_a_piece"
_NOT_A_PIECE_WORDS = "not a piece"
LABEL_KINDS = [f"{c} {t}" for c in cfg.PIECE_COLOURS for t in cfg.PIECE_TYPES] + [_NOT_A_PIECE_WORDS]
_LABEL_SHEET_MAX = 40          # pieces a sheet: 5 rows of 8 tiles

_LABEL_SYSTEM = "You identify chess pieces in close-up photos. Reply with JSON only."

_LABEL_PROMPT = (
    "This image is a sheet of numbered tiles cut from one photo of a wooden chessboard. Each tile shows "
    "one object standing on the board, turned so that it stands upright, its foot near the tile's bottom; "
    "every tile is tall enough for the tallest piece.\n"
    "For every tile number, say what the object is: the colour (white for the pale-wood pieces, black for "
    f"the dark-wood ones) and which chess piece, or '{_NOT_A_PIECE_WORDS}' if it is not one (a shadow, a "
    "hand, a cable, part of the board).")


def _next_frame(env, frame: np.ndarray, tries: int = 20) -> np.ndarray:
    """
    A frame from the overhead camera that is not `frame`: the model's second
    look. A camera read straight after another can hand back the same image,
    and two looks at one image agree whatever the model makes of it.
    """
    again = frame
    for _ in range(tries):
        again = env.render_image(cfg.OBSERVE_CAMERA, width=CAPTURE_W, height=CAPTURE_H)
        if not np.array_equal(again, frame):
            break
        time.sleep(0.03)
    return again


def _ask_sheet(frame: np.ndarray, measured: dict, view, squares: list) -> dict:
    """{square: (colour, type)} for these squares' pieces, from sheets of them cut from frame."""
    from mrcf import vlm as _vlm
    from mrcf.perception import tiles

    out: dict = {}
    for i in range(0, len(squares), _LABEL_SHEET_MAX):
        some = squares[i:i + _LABEL_SHEET_MAX]
        cut = [tiles.upright(frame, view, measured[sq]["measurement"].base_mm,
                             measured[sq]["measurement"].base_radius_mm, cfg.TALLEST_PIECE_M * 1000.0)
               for sq in some]
        said = _vlm.label_numbered(tiles.png_b64(tiles.sheet(cut)), len(some), LABEL_KINDS, _LABEL_PROMPT,
                                   system=_LABEL_SYSTEM, model=cfg.LABEL_MODEL)
        for sq, kind in zip(some, said):
            if kind == _NOT_A_PIECE_WORDS:
                out[sq] = (None, NOT_A_PIECE)
            elif kind:
                colour, ptype = kind.split(" ", 1)
                out[sq] = (colour, ptype)
            else:
                out[sq] = (None, None)
    return out


def label_pieces(frame: np.ndarray, measured: dict, view, again=None) -> dict:
    """
    {square: (colour, type)} for each piece in measured (measure_pieces'
    pieces, with view its "_view"): what the vision model calls it, every
    piece on one sheet cut from frame. type is NOT_A_PIECE where it says the
    outline is not a piece and a second look — at again(), another frame from
    the same camera — says so too; where the second look disagrees, the type
    is None. Either is None where no answer came.
    """
    squares = sorted(sq for sq in measured if not sq.startswith("_"))
    if not squares:
        return {}
    out = _ask_sheet(frame, measured, view, squares)
    vetoed = [sq for sq in squares if out[sq][1] == NOT_A_PIECE]
    if vetoed and again is not None:
        second = _ask_sheet(again(), measured, view, vetoed)
        for sq in vetoed:
            if second[sq][1] != NOT_A_PIECE:
                out[sq] = (second[sq][0], None)
    return out


def label_unknown(env, frame: np.ndarray, measured: dict, squares: list, tracks: list, changes: dict,
                  memory) -> "tuple[dict, set]":
    """
    What the vision model calls the pieces on these squares (tracks[i] the
    memory's for squares[i], from observe; changes its report) that the
    memory cannot already name: new to it, moved by hand, back after going
    missing, or never named. Each name goes to the memory as the model's
    guess. Returns ({square: (colour, type)}, the squares asked about); the
    rest keep what the memory has. Says what it did in a few lines.
    """
    view = measured.get("_view", {}).get("view")
    fresh = {t.id for kind in ("new", "moved", "returned", "replaced") for t, _ in changes.get(kind, [])}
    ask = [sq for sq, t in zip(squares, tracks) if not t.label or t.id in fresh] if view is not None else []
    started = time.time()
    labels = label_pieces(frame, {sq: measured[sq] for sq in ask}, view,
                          again=lambda: _next_frame(env, frame)) if ask else {}
    for sq, t in zip(squares, tracks):
        colour, ptype = labels.get(sq, (None, None))
        if ptype and ptype != NOT_A_PIECE:
            memory.set_label(t.id, ptype, "vlm",
                             attributes={"colour": colour} if colour and not measured[sq]["colour"] else None)
    print(label_summary(labels, len(squares) - len(ask), time.time() - started))
    return labels, set(ask)


def label_summary(labels: dict, known: int, seconds: float) -> str:
    """The scan's labelling in a few lines: what the model called which squares, grouped."""
    if not labels:
        return f"  [Label] All {known} pieces as the memory has them — nothing to ask the vision model."
    head = (f"  [Label] Asked the vision model about {len(labels)} piece{'s' if len(labels) != 1 else ''} "
            f"in one look ({seconds:.0f} s)" + (f"; {known} known from before." if known else "."))
    lines = [head]
    for colour in cfg.PIECE_COLOURS:
        kinds = [f"{t} {' '.join(sq for sq, (c, ty) in sorted(labels.items()) if c == colour and ty == t)}"
                 for t in cfg.PIECE_TYPES
                 if any(c == colour and ty == t for c, ty in labels.values())]
        if kinds:
            lines.append(f"    {colour:6} {' · '.join(kinds)}")
    vetoed = [sq for sq, (_, t) in sorted(labels.items()) if t == NOT_A_PIECE]
    unknown = [sq for sq, (_, t) in sorted(labels.items()) if t is None]
    if vetoed:
        lines.append(f"    not a piece (both looks): {' '.join(vetoed)}")
    if unknown:
        lines.append(f"    could not tell: {' '.join(unknown)}")
    return "\n".join(lines)


# ── Measured geometry: where each piece actually stands, and how big it is ─────
#
# The tile classifier says WHICH square a piece is on, not where on it: a
# hand-placed piece is rarely centred, and the classifier says nothing about
# its size. This measures every piece on the board from the same overhead
# frame: its foot's true position and width, and its colour from its own
# pixels — see piece_geometry for how an outline becomes millimetres. Height is NOT measured here: this camera
# looks almost straight down, which is the one view a height cannot be read
# from (see piece_geometry.MIN_HEIGHT_PX_PER_MM).
#
# Runs only when mrcf/perception/segment.py can (it needs transformers); otherwise each
# piece is placed at its square's centre.

# A prompt this far up a square's centre lands on the body of whatever stands
# there, rather than on the board beside a small piece's foot.
_PROMPT_HEIGHT_MM = 12.0
_MIN_BASE_RADIUS_MM = 4.0
_MAX_BASE_RADIUS = 0.45     # fraction of a square
# An outline along the grid is still a piece when this much of its middle has
# changed from the empty-board photo (piece_geometry.on_the_grid's `differs`):
# the cream pawn 0.79; the patch of light square beside a queen 0.41.
_CORE_CHANGED = 0.6
# Telling this set's black pieces from its white ones, against the board's own
# two squares in the same frame — under the same light, through the same
# exposure and white balance. This set is the board's two woods: cream like the
# light squares, red-brown like the dark. So a piece is placed on the line
# between the two squares' colours (Lab a, b), 0 at the light squares, 1 at the
# dark:
#     darkness = (piece - light) . (dark - light) / |dark - light|^2
# below _WHITE_BELOW white, above _BLACK_ABOVE black, between not said (the
# memory's majority of recent scans keeps what it had; a classifier's colour
# stands in). Measured on this rig's saved frames and a live scan: cream pawn
# -0.2 to 0.42, red-brown queen, rook and others 0.77 to 1.57.
#
# Not lightness: a brown piece's lit top reads lighter than the dark squares,
# as light as the cream pawn (L 167 and 163, the dark squares 125). Nor a fixed
# hue: auto white balance moves the cream's hue with the camera's place (30-43
# deg from one, 0-15 from another; the brown 13-26), and washed out to near
# white it has no hue at all. Lightness decides only on a board whose squares
# do not differ in colour (_MIN_SQUARE_COLOUR_GAP), as their L would.
_WHITE_BELOW, _BLACK_ABOVE = 0.5, 0.65
_MIN_SQUARE_COLOUR_GAP = 8.0       # Lab a, b units between the two squares' colours


def piece_darkness(piece_lab, dark_lab, light_lab) -> "float | None":
    """
    Where a piece's colour falls between the board's squares': 0 at the light
    squares', 1 at the dark (perception.tones.between). None when the squares
    are not told apart.
    """
    return tones.between(piece_lab, dark_lab, light_lab, min_colour_gap=_MIN_SQUARE_COLOUR_GAP)


def colour_of(darkness: "float | None") -> "str | None":
    """'white', 'black' or None (too close to call) for piece_darkness's value."""
    if darkness is None:
        return None
    if darkness < _WHITE_BELOW:
        return cfg.PIECE_COLOURS[0]
    if darkness > _BLACK_ABOVE:
        return cfg.PIECE_COLOURS[1]
    return None


# ── The light on the board ─────────────────────────────────────────────────────
# The overhead camera's auto exposure cannot be turned off (it ignores its
# controls), so how washed out the board is depends on the room: from 8% of the
# light squares' pixels at the camera's maximum to over a third, with a dozen
# light squares more than 30% washed out at the window end and a cream piece
# among them outlined as a scrap. Every scan says so, and where, before
# anything is picked.
_WASHED_SQUARE = 0.30          # a light square this much at the maximum: washed out
_OVEREXPOSED = 0.15            # the light squares as a whole, this much: the board is too bright
_DARK_LIGHT_SQUARES_L = 140    # light squares darker than this (Lab L, 0-255): too dark
_DARK_DARK_SQUARES_L = 45      # or dark squares: a dark piece on them stops standing out


def lighting_report(frame_rgb: np.ndarray, squares: dict, occupied=()) -> dict:
    """
    The light on the board from its bare squares (not those in `occupied`):
    {"light_clipped": share of the light squares' pixels at the camera's maximum,
     "washed_out": [light squares more than _WASHED_SQUARE so], "light_L",
     "dark_L": the two kinds' median lightness}. {} when both kinds are not seen.
    """
    from mrcf.perception import lighting
    half = max(4, int(0.3 * _estimate_pitch_px(squares)))       # inside the square, off its edges
    stats = lighting.region_stats(frame_rgb, {n: sq["pixel"] for n, sq in squares.items() if n not in occupied}, half)
    by_parity = {0: [], 1: []}
    for n, (lum, at_max) in stats.items():
        by_parity[sum(squares[n]["rc"]) % 2].append((n, lum, at_max))
    if not by_parity[0] or not by_parity[1]:
        return {}
    light, dark = sorted(by_parity.values(), key=lambda kind: -float(np.median([l for _, l, _ in kind])))
    return {"light_clipped": round(float(np.mean([c for _, _, c in light])), 3),
            "washed_out": sorted(n for n, _, c in light if c > _WASHED_SQUARE),
            "light_L": round(float(np.median([l for _, l, _ in light])), 1),
            "dark_L": round(float(np.median([l for _, l, _ in dark])), 1)}


def lighting_warning(report: dict, pieces_on=()) -> "str | None":
    """What is wrong with the light on the board, in words, or None. pieces_on: squares pieces stand on."""
    if not report:
        return None
    washed = report["washed_out"]
    if washed or report["light_clipped"] > _OVEREXPOSED:
        ranks = sorted({int(n[1]) for n in washed})
        where = (f", ranks {ranks[0]}-{ranks[-1]}" if ranks and ranks[-1] - ranks[0] <= 3 else "")
        near = sorted(sq for sq in pieces_on if any(abs(ord(sq[0]) - ord(w[0])) <= 1 and abs(int(sq[1]) - int(w[1])) <= 1
                                                    for w in washed))
        msg = (f"Too bright: {report['light_clipped']:.0%} of the light squares are at the camera's maximum"
               + (f" — {len(washed)} washed out ({' '.join(washed)}{where})" if washed else "") + ". "
               "A light piece there can be missed, mis-measured or called the wrong colour")
        msg += f" — pieces on {', '.join(near)} stand among them." if near else "."
        return msg + " Draw the curtain, or turn the light away from that part of the board."
    if report["light_L"] < _DARK_LIGHT_SQUARES_L or report["dark_L"] < _DARK_DARK_SQUARES_L:
        return (f"Too dark: the light squares read {report['light_L']:.0f}, the dark {report['dark_L']:.0f} "
                "(of 255). A dark piece on a dark square can be missed. More light on the board.")
    return None


def _square_centre_mm(rc, pitch_mm: float) -> np.ndarray:
    r, c = rc
    return np.array([(c - 0.5) * pitch_mm, (r - 0.5) * pitch_mm])


def _pixel_to_world(px, capture_wh) -> np.ndarray:
    """Overhead pixel (capture resolution) -> robot-frame (x, y) on the table."""
    return _corner_world_positions(np.asarray(px, float).reshape(1, 2), cfg.OVERHEAD_HOMOGRAPHY,
                                   capture_wh, (cfg.RENDER_W, cfg.RENDER_H))[0]


def realign_stale_reference(corners: np.ndarray, capture_wh) -> "np.ndarray | None":
    """
    The empty-board photo the camera's self-check set aside when it found the
    camera moved (skills.BoardTarget.camera_moved), lined up with where the
    board is in the frame now, saved back as the photo in use — or None.

    The photo is of the EMPTY board, so its 49 corners are all there to find,
    and the board is a plane: a homography from those to the lattice now puts
    every square of the photo where it is now. Without the photo a scan can
    take a piece's shadow for a piece.
    """
    path = _P(cfg.SURFACE_REFERENCE_PATH)
    stale = path.with_name(path.stem + "_stale" + path.suffix)
    from mrcf.perception import surface_scan
    old = surface_scan.load_reference(stale, capture_wh)
    if old is None:
        return None
    then = detect_corners(old)
    got = None if then is None else surface_scan.realign_reference(old, then, corners)
    if got is None:
        return None
    lined_up, shift = got
    surface_scan.save_reference(lined_up, path)
    print(f"  [Chess] The empty-board photo lined up with where the camera is now ({shift:.1f} px) — in use again.")
    return lined_up


def measure_pieces(frame_rgb: np.ndarray, corners: np.ndarray, squares: dict) -> dict:
    """
    Every piece standing on the board, measured from one overhead frame.

    Returns {square: piece} with each piece a dict:
        base_world   (x, y) of the foot's centre, robot frame, metres
        offset_mm    foot centre minus the square's centre, board frame
        measurement  piece_geometry.PieceMeasurement
        colour       "white"/"black" from the piece's own pixels
        lightness    their median Lab L, and hue_deg their Lab hue angle — what
                     that call was made from
        score        the segmenter's confidence in the outline
        mask         the outline (frame-sized bool array)
    plus a "_view" entry with the fitted camera, its fit error and the two
    square colours' lightness. Empty dict if the segmenter is unavailable.
    """
    from . import piece_geometry as geo
    from mrcf.perception import segment

    if not segment.available():
        return {}
    pitch = cfg.CHESSBOARD_SQUARE_SIZE_M * 1000.0
    capture_wh = (frame_rgb.shape[1], frame_rgb.shape[0])
    view, rms = geo.fixed_camera_view(corners, capture_wh, pitch, cols=cfg.CHESSBOARD_INTERNAL_COLS)

    names = sorted(squares)
    centres = {n: _square_centre_mm(squares[n]["rc"], pitch) for n in names}
    # Prompt a grid over the whole board, a third of a square apart, rather
    # than each square's centre: a hand-placed piece can stand well off its
    # centre, and a prompt that lands on the board beside it outlines the
    # square instead. A piece's body is wider than the spacing, so every piece
    # is hit by at least one prompt.
    n_sq = cfg.CHESSBOARD_INTERNAL_COLS + 1
    ticks = np.arange(-pitch + pitch / 6, (n_sq - 1) * pitch, pitch / 3)
    grid = np.array([view.lifted((x, y), _PROMPT_HEIGHT_MM) for y in ticks for x in ticks])
    found = []
    lab_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    # What the empty board looks like, where there is a photo of it: an outline
    # that has not changed from it is the board, wherever it is and whatever
    # its shape — no rules about squares. Pieces stand anywhere; the squares'
    # names are only a convenience. Without the photo, the grid-line rule
    # stands in.
    from mrcf.perception import surface_scan
    reference = surface_scan.load_reference(cfg.SURFACE_REFERENCE_PATH, capture_wh)
    if reference is None:
        reference = realign_stale_reference(corners, capture_wh)
    changed = surface_scan.change_map(frame_rgb, reference) if reference is not None else None
    for score, mask, _ in segment.candidates(frame_rgb, view.project(grid)):
        m = geo.clean_mask(mask)
        if geo.touches_frame(m):
            continue
        if changed is not None and changed[m].mean() < surface_scan.MIN_CHANGED:
            continue
        differs = changed is not None and bool(changed[outlines.core(m, min_px=0)].mean() >= _CORE_CHANGED)
        # A patch of board the photo does not match any more — brighter light
        # than the photo's, a piece's shadow, the photo a pixel off along the
        # grid lines — still runs along the grid.
        if geo.on_the_grid(m, view, lab_frame, differs=differs):
            continue
        meas = geo.measure(m, view)
        if meas is None or not (_MIN_BASE_RADIUS_MM <= meas.base_radius_mm <= _MAX_BASE_RADIUS * pitch):
            continue
        found.append((score, m, meas))

    kept = keep_outlines(found)

    # The board's own two colours, logged beside each piece's so a scan in odd
    # light can be checked after the fact.
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    occupied_px = np.zeros(frame_rgb.shape[:2], bool)
    for _, m, _ in kept:
        occupied_px |= m
    by_parity = {0: [], 1: []}
    for n in names:
        u, v = (int(x) for x in squares[n]["pixel"])     # square_map's pixels are capture-sized
        patch = (slice(max(v - 6, 0), v + 7), slice(max(u - 6, 0), u + 7))
        if not occupied_px[patch].any():
            by_parity[sum(squares[n]["rc"]) % 2].append(np.median(lab[patch].reshape(-1, 3), 0))
    # The two squares' colours, darker first.
    square_lab = sorted((np.median(v, 0) for v in by_parity.values() if v), key=lambda c: c[0])
    square_l = [float(c[0]) for c in square_lab]

    by_rc = {tuple(squares[n]["rc"]): n for n in names}
    out = {}
    for score, m, meas in kept:
        # Square (R, C) spans [(C - 1), C) pitches along x and [(R - 1), R) along y.
        r = int(np.floor(meas.base_mm[1] / pitch)) + 1
        c = int(np.floor(meas.base_mm[0] / pitch)) + 1
        name = by_rc.get((r, c))
        if name is None:
            continue           # stands off the board's squares
        # Colour from the middle of the outline, well away from its edge, where
        # any board the outline took in sits.
        core = outlines.core(m)
        lum, a, b = (float(np.median(lab[..., i][core])) for i in range(3))
        hue = float(np.degrees(np.arctan2(b - 128, a - 128)))
        darkness = (piece_darkness((lum, a, b), square_lab[0], square_lab[1])
                    if len(square_lab) == 2 else None)
        # One outline per square: two there are a piece and a patch of
        # something else (board, shadow) — keep the one that differs most from
        # the empty board, else the surer one. Never the later one by chance.
        share = float(changed[m].mean()) if changed is not None else None
        if name in out:
            other = out[name]
            mine, theirs = ((share, score), (other["changed"], other["score"])) if share is not None \
                else ((score,), (other["score"],))
            print(f"  [Chess] Two outlines on {name} — keeping the one "
                  f"{'that differs most from the empty board' if share is not None else 'the segmenter is surest of'}.")
            if mine <= theirs:
                continue
        out[name] = {
            "base_world": _pixel_to_world(meas.base_px, capture_wh),
            "offset_mm": meas.base_mm - centres[name],
            "measurement": meas,
            "colour": colour_of(darkness),
            "lightness": lum,
            "darkness": None if darkness is None else round(darkness, 2),
            "hue_deg": hue,
            "score": score,
            "changed": share,
            "mask": m,
        }
    out["_view"] = {"view": view, "fit_rms_px": rms, "square_lightness": square_l,
                    "square_lab": [[round(float(v), 1) for v in c] for c in square_lab]}
    return out


def annotate_pieces(img_bgr: np.ndarray, measured: dict, names: dict) -> np.ndarray:
    """Copy of img_bgr with each measured piece outlined, its foot marked and labelled."""
    img = img_bgr.copy()
    for sq, p in measured.items():
        if sq.startswith("_"):
            continue
        contours, _ = cv2.findContours(p["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 0, 255), 2)
        bx, by = (int(v) for v in p["measurement"].base_px)
        cv2.drawMarker(img, (bx, by), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        label = f"{names.get(sq, sq)} {2 * p['measurement'].base_radius_mm:.0f}mm"
        for colour, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, label, (bx + 8, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, width, cv2.LINE_AA)
    return img


_GEOMETRY_NOTE_SHOWN = False


def _measure_safely(frame, corners, squares) -> dict:
    """measure_pieces, but a failure only costs the measurements, never the scan."""
    global _GEOMETRY_NOTE_SHOWN
    try:
        from mrcf.perception import segment
        if not segment.available():
            if not _GEOMETRY_NOTE_SHOWN:
                print(f"  [Chess] Piece measurement off: {segment.why_unavailable()}")
                _GEOMETRY_NOTE_SHOWN = True
            return {}
        return measure_pieces(frame, corners, squares)
    except Exception as exc:
        print(f"  [Chess] Piece measurement failed ({exc}) — using square centres.")
        return {}


def merge_sources(labels: dict, measured: dict, classified: bool = True, asked: "set | None" = None) -> dict:
    """
    One record per piece the camera's outlines found: where its foot is and how
    big (measured), and what the vision model called it (labels, from
    label_pieces). {square: {colour, type, sources, geometry, note, confirmed,
    vetoed}}. asked: the squares the model was asked about, when not all of
    them were (the rest the memory names).

    Only the outlines put a piece on the board: a label without an outline is
    not a piece. Its colour is the camera's, from the piece's own pixels, where
    they tell; otherwise the model's. An outline the model says is not a piece
    is kept, vetoed: something to keep clear of, not a piece to pick.
    """
    merged = {}
    for sq, piece in sorted(measured.items()):
        if sq.startswith("_"):
            continue
        colour, ptype = labels.get(sq, (None, None))
        vetoed = ptype == NOT_A_PIECE
        if vetoed:
            note = "the vision model says it is not a piece"
        elif ptype is None and classified and (asked is None or sq in asked):
            note = "the vision model could not tell what it is"
        else:
            note = ""
        merged[sq] = {"colour": piece["colour"] or colour, "type": None if vetoed else ptype,
                      "sources": ["camera", "model"] if sq in labels else ["camera"],
                      "geometry": piece, "note": note, "confirmed": True, "vetoed": vetoed}
    return merged


def kind_text(track, colour: "str | None" = None) -> str:
    """'white knight', or 'white piece (vision model guesses pawn)' — for descriptions."""
    colour = colour or track.attributes.get("colour")
    if colour is None:
        what = track.label if track.label else "unidentified object"
        foot = f", foot {track.foot_mm:.0f} mm" if track.foot_mm else ""
        return f"{what}{foot}"
    if track.label and track.label_source in ("shape", "taught", "user"):
        return f"{colour} {track.label}"
    if track.label:
        return f"{colour} piece (vision model guesses {track.label})"
    return f"{colour} unidentified piece"


def _print_changes(memory, changes: dict) -> None:
    """One line per piece whose story changed in this scan."""
    new = changes["new"]
    if len(new) > 3:
        print(f"  [Memory] {len(new)} pieces not seen before")
    else:
        for t, d in new:
            print(f"  [Memory] {t.id} new on {d['at']}")
    for t, d in changes["moved"]:
        guess = " (or an identical piece)" if d["uncertain"] else ""
        print(f"  [Memory] {t.id} {memory.name(t)}{guess} moved {d['from']} -> {d['to']} by hand")
    for t, d in changes["returned"]:
        print(f"  [Memory] {t.id} {memory.name(t)} is back, on {d['to']}")
    for t, d in changes["missing"]:
        print(f"  [Memory] {t.id} {memory.name(t)} not seen (last on {d['last_at']})")


# The last scan's squares: {name: {"world": (x, y), "pixel": (u, v)}}, for what
# needs a square's place between scans (skills._vacated).
LAST_SQUARES: dict = {}

# A lattice whose squares fit a regular grid worse than this is not the board's:
# a right one fits to 1.0-1.5 mm; on 2026-11-04 the corner finder put one
# together from corners in the wrong order, 41.5 mm squares at 10.2 mm.
_LATTICE_MAX_RMS_M = 0.004


def _lattice_ok(frame_rgb: np.ndarray, corners: "np.ndarray | None") -> bool:
    """Whether a corner grid is the board's here: a dark a1, and squares on a regular grid."""
    if corners is None:
        return False
    squares, _, rms = square_map(corners, cfg.OVERHEAD_HOMOGRAPHY, (CAPTURE_W, CAPTURE_H),
                                 (cfg.RENDER_W, cfg.RENDER_H))
    return rms <= _LATTICE_MAX_RMS_M and check_orientation(frame_rgb, squares) is None


def board_lattice(frame_rgb: np.ndarray) -> "np.ndarray | None":
    """
    The board's 49 inner corners in this frame, from the first that holds
    together (_lattice_ok): its tags (lattice_from_tags), the corner finder,
    the last good lattice (board_corners.npy). The one used is kept as the
    last good one. None if none does.

    Tags first: on a full board the corner finder fails on most frames (a tall
    piece hides a corner) and once put a lattice together from corners in the
    wrong order, which then stood in the cache for every scan after; the tags
    found the lattice on every frame of 2026-11-03/04, within 0.1-0.5 mm of
    the corner finder's when that was right, in 0.1 s against its 1-2 s.
    """
    cache = _P(__file__).parent / _CORNER_CACHE
    tags = lattice_from_tags(frame_rgb)
    if _lattice_ok(frame_rgb, tags):
        corners, said = tags, None
    else:
        corners, said = detect_corners(frame_rgb), None
        if corners is not None and not _lattice_ok(frame_rgb, corners):
            print("  [Chess] The corner finder's lattice does not hold together (a1 light, or not a regular "
                  "grid) — not using it.")
            corners = None
        if corners is None and cache.exists():
            cached = np.load(cache)
            if _lattice_ok(frame_rgb, cached):
                corners, said = cached, ("  [Chess] Some board corners are hidden and the board's tags are not "
                                         "all in view — reusing the last good lattice. Valid only if neither the "
                                         "board nor the camera has moved since.")
    if said:
        print(said)
    if corners is not None and said is None:
        try:
            np.save(cache, corners)
        except OSError:
            pass
    return corners


def discover(env, scene_state, save_to=None, classify: bool = True) -> bool:
    """
    Populate scene_state from the board: pieces at computed square centres,
    reachable empty squares as placement targets.

    What is on the board, and where, comes from the camera's outlines
    (measure_pieces): never from the model, which shown the board puts pieces
    on empty squares. What each outlined piece is comes from the vision model,
    shown every piece the memory cannot already name on one sheet
    (label_unknown); an outline it says is not a piece stays as something to
    keep clear of, not to pick.

    classify=False skips the vision model and reports only what the camera's
    outlines find: positions, sizes and colours, no types. Faster, and needs no
    model server — for checking the measurements on their own.

    Returns True if the board was found and parsed.
    """
    from mrcf.vlm import SceneObject

    if cfg.OVERHEAD_HOMOGRAPHY is None:
        print("  [Chess] OVERHEAD_HOMOGRAPHY is None — run tools/calibrate_overhead_camera.py.")
        return False

    frame = env.render_image(cfg.OBSERVE_CAMERA, width=CAPTURE_W, height=CAPTURE_H)
    corners = board_lattice(frame)
    if corners is None:
        # Save what the camera actually saw. Without this the only evidence of
        # a failed scan is this one line, and "not detected" covers occlusion,
        # a board out of frame, a camera opened at too low a resolution to
        # resolve the lattice, and a genuinely absent board — all of which are
        # obvious from the frame and indistinguishable from the message.
        debug = (_P(save_to).with_name("board_debug.png") if save_to is not None
                 else _P(__file__).parent / "board_debug.png")
        try:
            privacy.imwrite(str(debug), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            where = f" Saved what the camera saw to {debug}."
        except OSError as exc:
            where = f" (could not write {debug}: {exc})"
        print(f"  [Chess] Board not detected ({frame.shape[1]}x{frame.shape[0]} frame). "
              f"Check it's fully in frame, unoccluded and in focus.{where}")
        return False

    global LAST_SQUARES
    squares, square_m, rms = square_map(
        corners, cfg.OVERHEAD_HOMOGRAPHY, (CAPTURE_W, CAPTURE_H), (cfg.RENDER_W, cfg.RENDER_H)
    )
    LAST_SQUARES = squares
    print(f"  [Chess] Board found — 64 squares, {square_m * 1000:.1f} mm each "
          f"(grid residual {rms * 1000:.1f} mm)")

    warning = check_orientation(frame, squares)
    if warning:
        print(f"  [Chess] WARNING: {warning}")
    labelled = annotate(frame, squares)
    if save_to is not None:
        privacy.imwrite(str(save_to), labelled)
        # Persist the square->robot-frame map beside the overlay so offline
        # tools can work in real board coordinates without a camera or a live
        # arm — visualise_ik.py's --square takes its targets from here.
        import json as _json
        try:
            _P(save_to).with_name("board_squares.json").write_text(_json.dumps(
                {name: {"world": [float(sq["world"][0]), float(sq["world"][1])],
                        "pixel": [float(sq["pixel"][0]), float(sq["pixel"][1])]}
                 for name, sq in squares.items()}, indent=2))
        except OSError as exc:
            print(f"  [Chess] Could not write board_squares.json ({exc}) — "
                  "visualise_ik.py --square will fall back to a nominal grid.")

    bounds = cfg.WORKSPACE_BOUNDS

    def reachable(w):
        return bounds.x_min <= w[0] <= bounds.x_max and bounds.y_min <= w[1] <= bounds.y_max

    from . import board_log
    from mrcf.object_memory import Detection
    measured = _measure_safely(frame, corners, squares)

    # Who each piece is. The scan is matched against the memory, so a piece
    # keeps what was learned about it — its measured height, what the user
    # called it, what the vision model called it — through rescans, moves by
    # hand and restarts, and a move the camera saw is recorded as one instead
    # of as a piece vanishing and another appearing.
    memory = scene_state.memory if scene_state.memory is not None else board_log.memory()
    scene_state.memory = memory
    seen = merge_sources({}, measured, classified=False)
    detections = []
    for sq, rec in sorted(seen.items()):
        geom = rec["geometry"]
        w = geom["base_world"] if geom is not None else squares[sq]["world"]
        detections.append(Detection(
            position=(float(w[0]), float(w[1])), place=sq,
            foot_mm=2 * geom["measurement"].base_radius_mm if geom is not None else None,
            attributes={"colour": rec["colour"]} if rec["colour"] in cfg.PIECE_COLOURS else {},
            extra={"lightness": geom["lightness"], "hue_deg": geom["hue_deg"]} if geom is not None else {},
            confirmed=rec["confirmed"]))
    if measured:
        tracks, changes = memory.observe(detections)
    else:
        # The outlines did not run (no segmenter), and only they say what is on
        # the board: an empty result here is not an empty board, so it must
        # not mark every piece gone.
        tracks, changes = [], {k: [] for k in ("new", "moved", "returned", "missing", "replaced")}

    # What each piece is: the vision model is asked only about those the
    # memory cannot already name, so a restart on the board as it was asks
    # it nothing.
    labels: dict = {}
    asked: set = set()
    if classify and tracks:
        labels, asked = label_unknown(env, frame, measured, sorted(seen), tracks, changes, memory)
    merged = merge_sources(labels, measured, classified=classify, asked=asked)

    # A piece in the jaws is not on the board to be seen; it stays in the scene
    # as held (position None), as the SceneObject it was picked as.
    held_before = {o.track_id: o for o in scene_state.objects if o.position is None and o.track_id}
    scene_state.objects = []
    scene_state.targets = []
    log_pieces = []
    for (sq, rec), track in zip(sorted(merged.items()), tracks):
        geom = rec["geometry"]
        colour = track.attributes.get("colour") or rec["colour"] or "unknown"
        w = geom["base_world"] if geom is not None else squares[sq]["world"]
        name = memory.name(track)
        obj = SceneObject(name, f"{kind_text(track, colour)} on {sq}" + (f" ({rec['note']})" if rec["note"] else ""),
                          interactable=not rec.get("vetoed"), position=np.array([w[0], w[1], cfg.TABLE_Z]),
                          pixel_xy=list(squares[sq]["pixel"]))
        obj.track_id = track.id
        obj.reachable = reachable(w)
        obj.square = sq
        obj.geometry = None
        # Through the track: the outlines missing a piece they have seen before
        # does not make it a guess.
        obj.confirmed = track.confirmed
        # A shape the gripper camera measured, while the piece still has the
        # foot it was measured with (see object_memory.Track.shape_current).
        obj.measured_shape = track.shape() if track.shape_current() else None
        entry = {"square": sq, "id": track.id, "name": name, "type": track.label,
                 "type_source": track.label_source, "colour": colour, "sources": rec["sources"],
                 "base_world_m": [float(w[0]), float(w[1])], "reachable": bool(obj.reachable),
                 "height_mm": track.height_mm if obj.measured_shape else None, "note": rec["note"],
                 "confirmed": track.confirmed}
        if geom is not None:
            meas = geom["measurement"]
            obj.geometry = {"base_diameter_m": 2 * meas.base_radius_mm / 1000.0,
                            "offset_mm": [float(v) for v in geom["offset_mm"]]}
            entry.update({"offset_mm": geom["offset_mm"],
                          "base_diameter_mm": round(2 * meas.base_radius_mm, 1),
                          "colour_measured": geom["colour"], "lightness": geom["lightness"],
                          "darkness": geom.get("darkness"),
                          "hue_deg": geom["hue_deg"],
                          "outline_score": geom["score"]})
        scene_state.objects.append(obj)
        log_pieces.append(entry)
    for track in memory.in_state("held"):
        obj = held_before.get(track.id)
        if obj is None:
            colour = track.attributes.get("colour") or "unknown"
            obj = SceneObject(memory.name(track), f"{kind_text(track, colour)}, held in the gripper",
                              interactable=True, position=None)
            obj.track_id, obj.square, obj.geometry = track.id, track.place, None
            obj.measured_shape = track.shape()
        scene_state.objects.append(obj)
    # A held piece is named for the square it was picked from, where another
    # piece may stand now: names must stay unique for plans to mean one thing.
    seen = set()
    for o in scene_state.objects:
        if o.name in seen:
            o.name = f"{o.name}_{o.track_id.replace('-', '')}"
        seen.add(o.name)

    # Empty squares are placement destinations. Only the reachable ones are
    # listed: all 64 would swamp the planner's prompt, and the unreachable ones
    # are not legal destinations for this arm anyway.
    for sq in sorted(squares):
        if sq in merged:
            continue
        w = squares[sq]["world"]
        if not reachable(w):
            continue
        t = SceneObject(sq, f"empty square {sq}", interactable=False,
                        position=np.array([w[0], w[1], cfg.TABLE_Z]),
                        pixel_xy=list(squares[sq]["pixel"]))
        scene_state.targets.append(t)
    spot = capture_spot(squares, memory)
    if spot is not None and reachable(spot):
        scene_state.targets.append(SceneObject(
            cfg.CAPTURE_TARGET, "off the board: where a captured piece goes", interactable=False,
            position=np.array([spot[0], spot[1], cfg.TABLE_Z])))

    view_info = measured.get("_view", {})
    light = lighting_report(frame, squares, occupied=set(merged))
    if save_to is not None and measured:
        names = {sq: e["name"] for sq, e in zip(sorted(merged), log_pieces)}
        privacy.imwrite(str(save_to), annotate_pieces(labelled, measured, names))
    board_log.record_scan(
        log_pieces, [sq for sq in squares if sq not in merged],
        {"square_mm_robot_frame": round(square_m * 1000, 2),
         "camera_fit_rms_px": view_info.get("fit_rms_px"),
         "square_lightness": view_info.get("square_lightness"),
         "lighting": light,
         "measured": bool(measured),
         "memory": {k: [{"id": t.id, **d} for t, d in v] for k, v in changes.items() if v}},
        frame_rgb=frame)
    _print_changes(memory, changes)

    n_reach = sum(1 for o in scene_state.objects if o.reachable)
    print(f"  [Chess] pieces={len(scene_state.objects)} ({n_reach} reachable)  "
          f"empty reachable squares={len(scene_state.targets)}")
    for o, e in zip(scene_state.objects, log_pieces):
        tag = "" if o.reachable else "  [OUT OF REACH]"
        size = f"  foot {e['base_diameter_mm']:.0f} mm" if "base_diameter_mm" in e else ""
        if e.get("height_mm"):
            size += f", {e['height_mm']:.0f} mm tall"
        note = f"  — {e['note']}" if e["note"] else ""
        print(f"           {o.name} @ ({o.position[0]:+.3f}, {o.position[1]:+.3f}){size}{tag}{note}")
    # Last, so it is what is on screen when the scan ends.
    warning = lighting_warning(light, pieces_on=[sq for sq, rec in merged.items() if rec.get("confirmed", True)])
    if warning:
        print(f"  [Light] WARNING — {warning}")
    elif light:
        print(f"  [Light] ok ({light['light_clipped']:.0%} of the light squares at the camera's maximum).")
    return True
