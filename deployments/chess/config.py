"""
deployments/chess/config.py — the chess deployment: the SO-101 moving wooden
chess pieces on its board, by plain English, openings and games against
Stockfish.

A deployment is one arm, one layout and one task (DEPLOYMENT_SPEC.md): chess
is the SO-101 (ARM_NAME), its wooden board and pieces, and the game. The ROBOT
is configured in its own backend (mrcf/backends/so101/robot_config.py): port,
cameras, measured poses, workspace bounds. Those are re-exported below so this
stays the single import for the deployment, but they are edited there — one
recalibration should not have to be copied into every deployment on the arm.
How chess drives the arm — its rules for it — is its profile (profile.py).

What belongs here: the piece vocabulary and sizes, the board's geometry,
which models to ask, and what to tell the vision model it is looking at. How
the arm picks things up is not chess's business: that is the arm's policy
library (its make_policies(), mrcf/policies/parallel_jaw/).

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

import os as _os

from pathlib import Path as _Path

from mrcf.safety import SafetyThresholds

# ── Robot setup ───────────────────────────────────────────────────────────────
# The robot settings deployment code reads as `cfg.X` or `from .config import
# X`, served LIVE from the arm's robot_config: config.X is robot_config.X at the
# moment it is read, never a copy taken at import. The arm's calibration
# changes while the controller runs (overhead_calibration re-poses a moved
# camera), and a copy here would keep the old homography placing every square
# until a restart. Change them in robot_config, not here.
#
# Chess itself reads only the rig settings every arm has (backends.RIG_SETTINGS);
# any other setting of the arm's (FOLLOWER_PORT, HOME_DEGREES on the SO-101) is
# served too, for the arm's own tools that ask a deployment for it.
from mrcf import backends as _backends

ARM_NAME = "so101"
ARM = _backends.arm(ARM_NAME)
_rc = ARM.config

_ROBOT_SETTINGS = tuple(_backends.RIG_SETTINGS)

# Chess's own rules for its arm (profile.py, mrcf/profiles.py): laid over the
# arm's facts wherever the arm's env and policies are made for it, and served
# first here.
from mrcf import profiles as _profiles

PROFILE = _profiles.load(_Path(__file__).parent)
_OWN = _profiles.settings(PROFILE)


def __getattr__(name: str):
    if name in _OWN:
        return _OWN[name]
    if name in _ROBOT_SETTINGS or (name.isupper() and hasattr(_rc, name)):
        return getattr(_rc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {n for n in vars(_rc) if n.isupper()} | set(_OWN))


_HERE = _Path(__file__).parent

# calibrate_overhead_camera.py detects a chessboard's internal corners
# automatically (sub-pixel accurate, dozens of points from one photo) and
# only needs 3 of them touched with the gripper to anchor the whole grid
# into the robot frame. COLS/ROWS = internal corners per side = squares - 1
# (7x7 for a standard 8x8 board). SQUARE_SIZE_M is MEASURE with a ruler —
# used only as a sanity check against what the 3 touches imply, not as an
# input to the fit, so it doesn't need to be exact.
CHESSBOARD_INTERNAL_COLS = 7
CHESSBOARD_INTERNAL_ROWS = 7

# Squares are named the way the arm would name them if it were a player sitting
# where its base is, facing the board:
#   "white"  rank 1 is the row nearest the arm, the a-file on the arm's left
#   "black"  rank 8 is nearest, the a-file on the arm's right
#
# A board set up for a player there has a dark square at each player's near-left
# corner (a1 for white, h8 for black — "light on the right"). Every scan checks
# that and warns if the board is turned a quarter-turn; the naming cannot fix a
# turned board, only turning the board can. Check names against the labelled
# deployments/chess/board_overlay.png after any scan, not the live board by eye.
ROBOT_PLAYS = "white"

# Fixed vocabulary. The VLM classifies into THIS list rather than inventing
# names, so the planner and perception share one set of terms.
PIECE_TYPES = ("pawn", "rook", "knight", "bishop", "queen", "king")

# Same idea for colour. The vision model's reply is held to these lists
# (chessboard.label_pieces); where it gives none it is unknown, never a default.
PIECE_COLOURS = ("white", "black")

CHESSBOARD_SQUARE_SIZE_M = 0.040   # measured: 4 cm squares

# AprilTags (tag36h11) on the board's wooden border, so the camera finds the
# board in every frame without the lattice (pieces and the arm hide corners of
# it). Printed by tools/make_tag_sheets.py. Where each sits is
# measured from a frame, not assumed: they went on by hand, half a turn from the
# printed diagram (BOARD_TAG_ROUGH_XY_M says which is where).
BOARD_TAG_IDS = (0, 1, 2, 3, 4, 5)
BOARD_TAG_SIZE_M = 0.026
# Roughly where each tag's centre is, by eye — only a starting point for finding
# the lattice, which then measures them. Board frame: origin at the board's
# centre, x toward the h-file, y toward rank 8 (m). Centred in the 36 mm border.
BOARD_TAG_ROUGH_XY_M = {
    0: (0.178, 0.178),     # h8 corner
    1: (-0.178, 0.178),    # a8 corner
    2: (-0.178, -0.178),   # a1 corner
    3: (0.178, -0.178),    # h1 corner
    4: (0.178, 0.0),       # h-file side
    5: (-0.178, 0.0),      # a-file side
}

OBSERVE_CAMERA = _rc.OVERHEAD_CAMERA_NAME
# Overhead frames for measuring are taken at the camera's full resolution.
CAPTURE_W_OVERHEAD, CAPTURE_H_OVERHEAD = _rc.OVERHEAD_CAMERA_W, _rc.OVERHEAD_CAMERA_H

# ── Pieces ────────────────────────────────────────────────────────────────────
# What chess knows about its objects before the gripper camera has measured
# one: (height, widest width) in metres, for a 40 mm-square Staunton set.
# PROVISIONAL estimates, except the pawn's, which was measured; replace with
# ruler measurements of your set. How to GRASP a piece of a given size is the
# arm's business (mrcf/policies/parallel_jaw/, tuned in robot_config) — this only
# says how big pieces are.
PIECE_SIZES_M: dict[str, tuple[float, float]] = {
    "pawn":   (0.045, 0.024),
    "rook":   (0.048, 0.026),
    "knight": (0.055, 0.027),
    "bishop": (0.062, 0.027),
    "queen":  (0.075, 0.030),
    "king":   (0.082, 0.030),
}

# A piece the gripper camera has not measured is approached as if it were the
# TALLEST piece, whatever the classifier called it: the classifier can call a
# king a pawn, and a pawn's hover puts the jaws at a king's shoulders. The
# pick measures it from that hover anyway. A piece the user has named
# ("identify") is planned at its own type's height instead.
TALLEST_PIECE_M = max(h for h, _ in PIECE_SIZES_M.values())
# And grasped as high as the SHORTEST piece still has the pads on it: every
# piece is 45-82 mm, and higher up all of them are slimmer.
SHORTEST_PIECE_M = min(h for h, _ in PIECE_SIZES_M.values())

# How slim a piece not yet measured can be counted, above its foot: the widest
# it could be at each height (mm), as a share of the foot the overhead camera
# measures. From the 138 profiles the wrist camera measured on this rig up to
# 2026-11-03, the widest each piece is from that height up, over its foot, at
# the 90th percentile (median 0.77 from 24 mm up). A neighbour is counted so
# too, and the jaws open by it, until the pick's camera measures it: by the
# whole foot all the way up, no back-rank piece could be picked between two
# others.
UNMEASURED_SHAPE_SHARE: dict[int, float] = {0: 1.0, 16: 1.0, 20: 0.89, 24: 0.87}


# Where captured pieces go: off the squares, on the board's wooden border past
# CAPTURE_FILE, CAPTURE_BORDER_M out from that file's middle (the border is
# level with the squares; the table round it is lower, and a piece let go at
# the board's height there could topple). The first beside ranks 1-2, each
# next one a square further along, up to CAPTURE_SLOTS (chessboard.capture_spot).
CAPTURE_FILE = "a"
CAPTURE_BORDER_M = 0.033
CAPTURE_SLOTS = 6
CAPTURE_TARGET = "captured"


def piece_type(name: str) -> "str | None":
    """The piece type in an object name such as 'white_king_e1', or None."""
    for part in str(name).lower().split("_"):
        if part in PIECE_SIZES_M:
            return part
    return None


# ── Approach rules ───────────────────────────────────────────────────────────────
# Heuristics for how far the gripper leans from straight down to pick from, or
# place on, a square. By rank, counted from the arm's side (rank 1 is its row,
# as it plays white), and optionally by file; checked in order, the first that
# includes the square winning; no rule: the least lean that reaches.
#
# A rule is where the arm's search for an orientation STARTS, not a promise:
# where it cannot hold the lean asked for, it takes the nearest it can, so a
# wrong number costs margin, not the pick.
#
# Empty: the arm's approaches (mrcf/policies/approaches.py, Approach;
# robot_config NEAR_BASE_M / FAR_REACH_M) choose how to come at a square by its
# distance from the base — near_base, top_down, reach_out — and keep joint
# margin themselves. Add a rule, such as
#   {"ranks": (1, 2), "files": "bcdefg", "tilt_deg": 20},
# for a square that needs something the approaches do not give it — a rule is
# where the arm's search for an orientation STARTS, not a promise.
# Ranks 7-8, at the arm's edge with the board 2 cm out (2026-11-04): picks on e7
# planned at the least lean, -45, stalled 7-17 cm short every time (four
# picks); at -55 they reached. Starting there saves the stalled attempt.
APPROACH_RULES: list = [
    {"ranks": (7, 8), "tilt_deg": 55},
]


def approach_tilt(square: "str | None") -> "float | None":
    """APPROACH_RULES' lean for a square such as 'e6', or None (no rule, or not a square)."""
    s = str(square or "").lower()
    if len(s) != 2 or s[0] not in "abcdefgh" or s[1] not in "12345678":
        return None
    file, rank = s[0], int(s[1])
    if str(ROBOT_PLAYS).lower() == "black":
        rank = 9 - rank                  # the rules count from the arm's side
    for rule in APPROACH_RULES:
        lo, hi = rule["ranks"]
        if lo <= rank <= hi and file in rule.get("files", "abcdefgh"):
            return float(rule["tilt_deg"])
    return None


# ── Learned skills ───────────────────────────────────────────────────────────────
# A trained model (lerobot-train's checkpoint folder; mrcf/policies/learned.py) as a
# verb the planner can ask for, on one named piece: the arm hovers
# start_height_m over it — where the demonstrations began — then the model runs
# on the arm's stream for at most max_s, and a fresh scan says where it left
# things. It is trained elsewhere (adding one, step by step: POLICIES.md), on
# this arm's stream (env.stream_spec; python tools/show_stream.py prints it):
# one trained on another arm, or another camera size, is refused before
# anything moves. Running a LeRobot model needs LeRobot here
# (pip install lerobot): without it the skill is left out at start-up, saying so.
# Instead of "checkpoint", "model" takes any policies.learned.Model.
LEARNED_SKILLS: list = [
    # A policy package (mrcf/policies/packages.py), by reference — a folder, "hf:owner/name@rev", or a
    # plugin source's scheme; its manifest gives the description, task, start height and time
    # limit, and anything here overrides them:
    # {"name": "upright", "policy": "hf:me/upright-so101@v1"},
    # Or a LeRobot checkpoint directly:
    # {"name": "upright", "description": "Stand a fallen piece back up on its base.",
    #  "checkpoint": "outputs/train/upright/checkpoints/last/pretrained_model",
    #  "task": "stand the pawn up", "start_height_m": 0.06, "max_s": 20},
]

# ── Models ───────────────────────────────────────────────────────────────────────
# Where the models run (mrcf/model_server.py): "ollama", this machine's or one
# on the network at MODEL_URL ("http://gpu-box:11434"), or "openai", any
# OpenAI-compatible API at MODEL_URL (its base, ".../v1"): a hosted one, or
# vLLM, LM Studio, llama.cpp's server. An API key is read from
# MRCF_MODEL_API_KEY, never from here. A server off this machine is sent the
# camera's pictures and the scene. MODEL, PLANNING_MODEL and LABEL_MODEL are
# then that server's names for its models.
MODEL_SERVER = _os.environ.get("MRCF_MODEL_SERVER", "ollama")
MODEL_URL = _os.environ.get("MRCF_MODEL_URL", "") or None   # None: the server's own (Ollama's: OLLAMA_URL)

# One model for every call: on two overhead frames (2026-10-30) qwen2.5vl:7b
# described three walnut pawns as two light pieces and "possibly a king or
# queen"; gemma4:31b, thinking off, had all three right, ~22-26 s a look
# against 4-7 s. One model loaded, not three.
MODEL          = _os.environ.get("ROBOTICS_VLM_MODEL", "gemma4:31b")

PLANNING_MODEL = _os.environ.get("ROBOTICS_PLANNING_MODEL", "gemma4:31b")  # plan()/decompose_task()

# What each piece the scan outlines is (chessboard.label_pieces): chosen with
# deployments/chess/tests/eval_piece_labels.py, one crop at a time. On the
# rig's crops gemma4:31b named every overhead pawn and empty square right,
# ~4.5 s a look, where qwen2.5vl:7b called every overhead pawn "not a piece".
# Shown all 32 pieces on one sheet (2026-11-03): 16-27 s for all of them.
# About 20 GB in memory.
LABEL_MODEL    = _os.environ.get("ROBOTICS_LABEL_MODEL", "gemma4:31b")

TASK_PLANNER_ENABLED = True

# ── Chess engine (engine.py, game.py) ──────────────────────────────────────────
# Who chooses the arm's moves in a game against a person: any UCI engine.
# None: MRCF_CHESS_ENGINE, else `stockfish` on the PATH (brew install stockfish).
ENGINE_PATH: "str | None" = None
# How well it plays, as an Elo rating: Stockfish limits itself to 1320-3190
# (UCI_Elo); a club player is ~1500. Chosen per game with new_game's strength.
ENGINE_ELO = 1500
# How long it thinks per move. A second is plenty at a limited strength; the
# arm takes ~30 s to make the move anyway.
ENGINE_MOVETIME_MS = 1000

# ── Scene discovery ──────────────────────────────────────────────────────────────
# The VLM is the ONLY source of object identity here (no simulator body
# registry), so it will happily name whatever is most visually salient in the
# frame — the arm itself, the furniture behind the table — over the pieces.
# Anything whose name or description contains one of these is dropped at
# discovery.
ARM_KEYWORDS: set[str] = {
    "robot", "arm", "gripper", "claw", "servo", "motor", "wire", "cable",
    "chair", "curtain", "floor", "wall", "carpet", "rug", "person", "hand",
    "leg", "backpack", "bag", "sofa", "couch", "table", "desk", "socket",
}

# Injected into discover_scene's prompt. Steers the VLM toward task objects —
# without it the prompt is domain-agnostic and the model reports the room.
DISCOVER_GUIDANCE = (
    "This is a chessboard on a table, viewed from above.\n"
    "Report ONLY the individual chess pieces standing on the board, one entry "
    "per piece. Name them by colour and type where you can tell "
    "(e.g. white_pawn, black_knight); if you cannot tell the type, use a "
    "simple name like 'pawn'.\n"
    "pixel_xy must be the centre of the piece's BASE where it meets the board, "
    "since that is the point the robot will reach for.\n"
    "Do NOT report the robot arm, the board itself, the table, the floor, "
    "furniture, cables, or anything in the background — only pieces on the board.\n"
    "If there are no chess pieces on the board, return empty arrays."
)

# On an arm with no force/torque sensor (the SO-101) the contact-force checks
# in ExecutionMonitor never trigger (get_contact_force() defaults to 0.0) —
# this just keeps the EE-descent-floor check active with sane defaults. The
# force limits are left to the arm (env.contact_force_limits), whose sensor
# decides what they can be.
SAFETY_THRESHOLDS = SafetyThresholds()

# What runs alongside the arm for the whole session, each on its own thread
# (watchers.py): functions that take the env and return a started Watcher.
#
# The motion guard (the sentry) freezes the arm the moment anything other
# than the arm moves over the board. The arm runs it itself, on every
# connection (its robot_config's MOTION_GUARD; RigEnv._start_sentry,
# from SO101Env.connect) — so the tours, checks and rest.py are watched too,
# not just this controller. Its zone is the arm's (MOTION_GUARD_ZONE, one of
# the rig settings served above). Anything else that should run alongside the
# arm for a whole session goes in WATCHERS.

WATCHERS: list = []

# The deployment's full configuration surface: this task's own settings, and
# the robot's, served live (_ROBOT_SETTINGS).
__all__ = [
    "_HERE",
    "CHESSBOARD_INTERNAL_COLS",
    "CHESSBOARD_INTERNAL_ROWS",
    "ROBOT_PLAYS",
    "PIECE_TYPES",
    "PIECE_COLOURS",
    "CHESSBOARD_SQUARE_SIZE_M",
    "OBSERVE_CAMERA",
    "PIECE_SIZES_M",
    "TALLEST_PIECE_M",
    "piece_type",
    "APPROACH_RULES",
    "approach_tilt",
    "LEARNED_SKILLS",
    "MODEL_SERVER",
    "MODEL_URL",
    "MODEL",
    "PLANNING_MODEL",
    "LABEL_MODEL",
    "TASK_PLANNER_ENABLED",
    "ENGINE_PATH",
    "ENGINE_ELO",
    "ENGINE_MOVETIME_MS",
    "ARM_KEYWORDS",
    "DISCOVER_GUIDANCE",
    "SAFETY_THRESHOLDS",
    "WATCHERS",
] + list(_ROBOT_SETTINGS)
