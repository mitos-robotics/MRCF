"""
settings.py — the table task's defaults, and how a deployment's config is
served (serve()).

A deployment that does the table task (mrcf/tabletop/__init__.py) sets in its
config what is its own — its arm (ARM_NAME, ARM), its rules for the arm
(PROFILE: profile.py), what its objects are and where its people stand — and
ends with serve(globals()). Then any other setting it is asked for is, in
order: its profile's; the task's default below; its arm's
(mrcf/backends/<arm>/robot_config.py), read live, as the arm's calibration
changes while the controller runs. Its kept state — the surface it touched,
marked places, object memory, containers — is named after it in its arm's
rig folder (rig/<arm>/<name>_surface.json and so on), and the surface it
touched is what every height is measured from.

The runs behind these values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from mrcf import profiles
from mrcf.safety import SafetyThresholds

# ── The calibration target ────────────────────────────────────────────────────
# tools/calibrate_overhead_camera.py finds the inner corners of a checkerboard
# laid on the table, and the arm touches three of them. Any checkerboard will
# do: the SO-101 rig's wooden chessboard (8x8 squares of 40 mm: 7x7 inner
# corners), or one printed at 100%. Set these to the one used, then take it
# away: on the table during a session, its squares are objects to the scan.
CHESSBOARD_INTERNAL_COLS = 7
CHESSBOARD_INTERNAL_ROWS = 7
CHESSBOARD_SQUARE_SIZE_M = 0.040

# ── Objects ───────────────────────────────────────────────────────────────────
# Objects are found and measured by the fixed overhead camera (the arm's
# OVERHEAD_CAMERA_NAME). Mounted looking down at an angle (above the far side
# of the table, looking back at the arm), it measures each object's height as
# well as its foot; straight down, it cannot, and a pick then plans for the
# tallest object it could be (below).
# The tallest object an unmeasured one could be: a pick hovers clear of it.
TALLEST_OBJECT_M = 0.15
# Sizes known by what an object is: a word of the vision model's name for it
# -> (height, width across its flat sides) in mm: boxes, gripped across their
# sides and found by their top face (skills.boxed), recorded each scan for
# every one that has no size of its own (skills.known_sizes) — on an arm that
# cannot find out for itself (GripperSpec.height_from_views off). None here:
# a deployment whose objects are all one kind of box says so (the UR5's
# letter blocks, deployments/tabletop/config.py).
KNOWN_SIZES_MM: dict = {}
# What the vision model may call an object's colour (naming.py holds it to these).
COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "grey",
           "silver", "gold", "clear", "multicoloured")
# The model that names objects, from a crop of each (naming.py). gemma4:31b
# named the chess rig's crops right where smaller models did not.
LABEL_MODEL = os.environ.get("ROBOTICS_LABEL_MODEL", "gemma4:31b")
# The model that finds a thing by name, as a box round it (perception/grounding.py):
# a container the scan does not list (a tub). 2026-11-17, the SO-101's overhead
# camera: qwen2.5vl:7b boxed the clear plastic tub to within a few pixels in
# 0.5-3.5 s; whether a thing is a container at all is MODEL's to say, on a crop.
GROUNDING_MODEL = os.environ.get("ROBOTICS_GROUNDING_MODEL", "qwen2.5vl:7b")

# ── Looking ───────────────────────────────────────────────────────────────────
# Looking at the table past the arm where it is, not parking it out of the
# overhead camera's view first, when it hides no more than ARM_HIDES_AT_MOST of
# the work surface there (skills.goto_observe_pose; scene_scan.observe's
# past_arm: what is on the arm is left out, what it hides kept as it was). A
# park and back cost 4-8 s a look; after every command and every missed grip
# (user, 2026-11-17: "why does the camera only work when the arm is out of the
# way ... It makes it slow and inefficient").
SCAN_PAST_ARM = True
ARM_HIDES_AT_MOST = 0.08
# Each scan checks the overhead camera against the empty-table photo
# (SURFACE_REFERENCE_PATH) and stops when its picture has slid more than this
# (px; on the UR5 rig about a millimetre each): its calibration no longer holds.
# 2026-11-13: the mount crept 12 px in 15 minutes and a pick landed on its
# block; standing still the check reads 0.3-0.6 px. None: no check.
CAMERA_DRIFT_PX = 4.0

# ── Places ────────────────────────────────────────────────────────────────────
# The object memory starts empty at each start (the first scan says what is
# on the table) unless True; named places (PLACES_PATH) are kept either way.
# HISTORY.md, 2026-11-13: blocks remembered from the last run blocked picks
# and places as things last seen there, and runs began with "forget everything".
REMEMBER_ACROSS_RESTARTS = False
# The kinds of object this deployment works with, learned by the arm and kept
# across restarts (mrcf/catalogue.py, CATALOGUE_PATH): each sighting matched by
# how it looks from above, not by the vision model's name for it (so named the
# same each time, and not asked about); a pick of one picked by the grip that
# held it before, with no looks for its height; what each pick showed learned.
# False: every object is a first meeting.
CATALOGUE = False
# A close that found nothing: why, asked of the vision model on the wrist
# camera's pictures (perception/grasp_diagnosis.py) — too high, aimed off,
# lying on its side, pushed away — and the next attempt changes that (gripped
# lower, found again; one lying not picked as it stands). False: re-aimed, as
# before.
DIAGNOSE_MISSES = False
# "Put these in the tub": a plan of pick-and-put units runs each in turn; one
# whose pick fails is set aside and tried once more at the end after a fresh
# look, and what was not done is said — not planned again from scratch
# (controller.Deployment.defer_failed). False: the first failed step stops the
# plan and the goal is planned again.
DEFER_FAILED = False
# A gripper that reads empty after shutting on something, and that the wrist
# camera's own check does not see carrying it: the vision model asked, yes or
# no, on the wrist camera's picture (perception/grasp_diagnosis.held). False:
# taken as empty.
HELD_BY_MODEL = False
# A command planned before, word for word, its objects all on the table and
# none in a container: that plan again, not planned afresh (PLANS_PATH).
# False: every command planned by the model.
PLAN_MEMORY = False
# A failed skill described by the vision model from a picture, for the log
# (skill_executor.DESCRIBE_FAILURES). False: not asked (the flight recorder
# keeps the pictures).
DESCRIBE_FAILURES = True
# The motion guard stops a move, rather than pausing it, when what it sees
# moving starts again beside the gripper each time the arm moves on: the arm
# pushing something (motion_guard.STOP_PUSHING). False: it pauses and carries on.
GUARD_STOPS_PUSHING = False
# A thing lying along the table — as long as LONG_MIN_ASPECT times its widest
# across, as the scan saw it (plane_geometry.long_extent_mm) — told to the
# planner by its length and how narrow it is across, where the jaws close on it
# (skills.describe, lying_along); and "wider than the gripper opens" by that,
# not by the round foot an upright object is measured by. 2026-11-20, the
# UR5's tools: a screwdriver "73 mm across" by its foot was called too wide for
# the 52 mm jaws, and the planner picked nothing. False: by the foot.
ACROSS_FOR_LONG = False
LONG_MIN_ASPECT = 2.0
# A task's words (TASKS, and "put ... into ..." laid out by the quick plan)
# find things by their kind's words, any colour — "the hammer" is the
# grey_hammer — and a plural is every one of them, "the screwdrivers" each
# screwdriver on the table; and a container not yet found is looked for by its
# words. 2026-11-20, the UR5's tools were named red, then orange screwdriver,
# grey, then silver hammer, run to run, and its box had never been found.
# False: by the names as written, the container one kept already.
TASK_WORDS_BY_KIND = False
# The tallest a container's walls may be where they are not measured (no wrist
# camera to look at them): what is held is carried over them that high, and let
# go there. None: TALLEST_OBJECT_M. 2026-11-20, the UR5's cardboard box stood
# about twice the table task's 150 mm, flaps up.
CONTAINER_TALLEST_M = None
# A container's opening, where it is an open box, found as the part of its
# outline in its own shade (containers.open_top), not the whole outline with
# its flaps folded out: 2026-11-20, the UR5's cardboard box, a screwdriver let
# go over a flap. False: the whole outline (a clear tub has no such shade).
CONTAINER_OPENING_BY_SHADE = False
# An object the scan put inside the one being picked (its centre within that
# one's foot) is part of it, not kept clear of (manipulation.PARTS_INSIDE_ARE_ITS_OWN).
PARTS_INSIDE_ARE_ITS_OWN = False
# A plain pick outlines its object again by what it is first, as a pick by a
# wall does (manipulation.OUTLINE_BEFORE_PICK): about 4 s, the vision model's
# box and the segmenter in it.
OUTLINE_BEFORE_PICK = False
# ... and, with no catalogue to judge by, only an outline that looks like a
# piece of the thing (skills._outline_partial: under 40 mm long, or touching
# another), not every one. False: every one, before its pick.
OUTLINE_ONLY_PARTIAL = False
# "Put <things> into <container>" planned from the words themselves (skills.
# quick_plan): each thing found by what the user calls it — by name, by its
# kind in the catalogue, else looked for by those words — then a pick and a
# put for each; the planner only for anything else. False: always planned.
QUICK_PUT_INTO = False
# A pick by the catalogue's recipe opens the jaws past what the object's kind
# is across, resting so (its outline's narrow side at the most it has been
# seen), not only past the scan's foot (skills._recipe, manipulation). False:
# by the scan's foot.
RECIPE_OPENS_TO_KIND = False
# This deployment's own tasks, planned in advance (skills.task_plan,
# controller.Deployment.task_plan): {the words that run it: the task}. A task
# is a sentence the quick plan lays out ("put the red cap and the blue scoop
# into the plastic tub"), or its steps written out, the planner's way, the
# objects named as anyone would call them: [{"action": "pick", "object_name":
# "the red cap"}, {"action": "place_in", "container": "the plastic tub"}, ...]
# — each found on the table when the task runs; "pick" picked the way its
# kind is. A deployment with one task also runs it on "go", "run", "start".
TASKS: dict = {}
# False: the planner is never asked (controller.Deployment.planner): a command
# that is none of TASKS and no "put ... into ..." the quick plan lays out is
# answered with the tasks there are; a failed step is said, not planned again;
# a thing a task names that is not found is left out, and the rest done.
PLANNER = True
# Home at the end of each command that moved the arm (controller.Deployment.
# home_after_command). False: it stays where the command left it.
HOME_AFTER_COMMAND = False
# A pick set aside whose object the next look tracks afresh (pushed): the one
# of its name nearest where it was, within this, taken for it
# (controller.Deployment.refind_near_m). None: by its track only.
REFIND_NEAR_M = None
# Naming: the room round a long object's tile at most this many pixels of
# the camera's frame (naming.crop_tile); None: a share of its length, however
# long it is.
NAME_TILE_PAD_MAX_PX = None
# Naming: an outline the sheet calls "nothing" asked about again by itself
# before it is left out as not an object, at most NAME_NOTHING_TWICE_MAX a
# look (naming._second_look). False: the sheet's word stands.
NAME_NOTHING_TWICE = False
NAME_NOTHING_TWICE_MAX = 3
# place_beside: the gap left between the object put down and its neighbours.
BESIDE_GAP_M = 0.025
# Whose left and right "left of the box" means: "arm" (as seen from the arm's
# base, looking out over the table) or "facing" (as seen by a person standing
# across the table, facing the arm).
LEFT_RIGHT_SEEN_FROM = "facing"

# ── Models ────────────────────────────────────────────────────────────────────
# Where the models run (mrcf/model_server.py): "ollama", this machine's or one
# on the network at MODEL_URL ("http://gpu-box:11434"), or "openai", any
# OpenAI-compatible API at MODEL_URL (its base, ".../v1"): a hosted one, or
# vLLM, LM Studio, llama.cpp's server. An API key is read from
# MRCF_MODEL_API_KEY, never from here. A server off this machine is sent the
# camera's pictures and the scene. MODEL, PLANNING_MODEL and LABEL_MODEL are
# then that server's names for its models.
MODEL_SERVER = os.environ.get("MRCF_MODEL_SERVER", "ollama")
MODEL_URL = os.environ.get("MRCF_MODEL_URL", "") or None   # None: the server's own (Ollama's: OLLAMA_URL)
MODEL = os.environ.get("ROBOTICS_VLM_MODEL", "gemma4:31b")
PLANNING_MODEL = os.environ.get("ROBOTICS_PLANNING_MODEL", "gemma4:31b")
TASK_PLANNER_ENABLED = True

# What the vision model must not report as an object (the rig, the room).
ARM_KEYWORDS: set[str] = {
    "robot", "arm", "gripper", "claw", "servo", "motor", "wire", "cable",
    "chair", "curtain", "floor", "wall", "carpet", "rug", "person", "hand",
    "leg", "backpack", "bag", "sofa", "couch", "table", "desk", "socket",
}
DISCOVER_GUIDANCE = (
    "Report only the small, movable objects standing on the table in front of the robot (bottles, cups, "
    "boxes, toys, tools and the like), not the robot, the table or the room."
)

# ── Safety ────────────────────────────────────────────────────────────────────
# The force limits are the arm's own (its env.contact_force_limits).
SAFETY_THRESHOLDS = SafetyThresholds()

# Run alongside the arm for the whole session (watchers.py). The motion guard
# is the arm's own, started on every connection (its MOTION_GUARD).
WATCHERS: list = []


# ── Serving a deployment's config ─────────────────────────────────────────────

_KEPT = ("surface", "places", "memory", "containers", "catalogue", "plans")   # its kept state: rig/<arm>/<name>_<kind>.json


def _defaults() -> dict:
    """The task's defaults above: {NAME: value}."""
    return {name: value for name, value in globals().items() if name.isupper() and not name.startswith("_")}


def touched_surface(path: Path, rc) -> "float | None":
    """The bare table: as touched for this deployment (path), else as the rig has it (TABLE_TOP_Z), else None."""
    try:
        return float(json.loads(Path(path).read_text())["table_z"])
    except (OSError, ValueError, KeyError, TypeError):
        top = getattr(rc, "TABLE_TOP_Z", None)
        return None if top is None else float(top)


def serve(ns: dict) -> None:
    """
    Complete a deployment's config (its module's globals(), ns) for the table
    task: what it has not set itself is set, and any other setting it is asked
    for is served (the module's __getattr__): its profile's (ns["PROFILE"]),
    then the task's default (above), then its arm's (ns["ARM"].config, live).

    Set, unless the deployment set it:
      OBSERVE_CAMERA, CAPTURE_W_OVERHEAD, CAPTURE_H_OVERHEAD
                    the arm's overhead camera, at its capture size
      SURFACE_PATH, PLACES_PATH, MEMORY_PATH, CONTAINERS_PATH, CATALOGUE_PATH, PLANS_PATH
                    its kept state, named after it in the arm's rig folder
      SURFACE_Z     the surface it touched (skills.measure_surface), else the
                    rig's TABLE_TOP_Z, else None
      TABLE_Z, WORKSPACE_BOUNDS, ENV_OVERRIDES, POLICY_OVERRIDES
                    every height from that surface. The arm's TABLE_Z is the
                    surface IT was measured on — the SO-101's is its
                    chessboard's top, and its table work runs on the bare
                    table, 20-30 mm lower (2026-11-17: every grasp closed
                    round the objects' heads).
    """
    arm = ns["ARM"]
    rc = arm.config
    name = ns["__name__"].rsplit(".", 2)[-2] if ns["__name__"].count(".") >= 1 else ns["__name__"]
    ns.setdefault("OBSERVE_CAMERA", rc.OVERHEAD_CAMERA_NAME)
    ns.setdefault("CAPTURE_W_OVERHEAD", rc.OVERHEAD_CAMERA_W)
    ns.setdefault("CAPTURE_H_OVERHEAD", rc.OVERHEAD_CAMERA_H)
    for kind in _KEPT:
        ns.setdefault(f"{kind.upper()}_PATH", Path(rc.RIG_DIR) / f"{name}_{kind}.json")
    ns.setdefault("SURFACE_Z", touched_surface(ns["SURFACE_PATH"], rc))
    env_overrides: dict = {}
    policy_overrides: dict = {}
    z = ns["SURFACE_Z"]
    if z is not None:
        ns.setdefault("TABLE_Z", z)
        ns.setdefault("WORKSPACE_BOUNDS", dataclasses.replace(rc.WORKSPACE_BOUNDS, z_min=z))
        env_overrides = {"work_plane_z": z, "floor_z": z}          # the table the gripper must not go into
        if getattr(rc, "GRIPPER_CAM_PLANE_Z_M", None) is not None:
            # The wrist camera's plane is the same surface as its own calibration saw it, moved as much.
            env_overrides["gripper_cam_plane_z_m"] = rc.GRIPPER_CAM_PLANE_Z_M + (z - rc.TABLE_Z)
        policy_overrides["table_z"] = z
    ns.setdefault("ENV_OVERRIDES", env_overrides)
    ns.setdefault("POLICY_OVERRIDES", policy_overrides)

    own = profiles.settings(ns.get("PROFILE"))
    task = _defaults()
    module = ns["__name__"]

    def __getattr__(attr: str):
        if attr in own:
            return own[attr]
        if attr in task:
            return task[attr]
        if attr.isupper() and not attr.startswith("_") and hasattr(rc, attr):
            return getattr(rc, attr)
        raise AttributeError(f"module {module!r} has no attribute {attr!r}")

    def __dir__():
        return sorted(set(ns) | set(own) | set(task) | {n for n in vars(rc) if n.isupper() and not n.startswith("_")})

    ns["__getattr__"] = __getattr__
    ns["__dir__"] = __dir__
