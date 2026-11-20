"""
robot_config.py (mrcf/backends/<arm>/, copied from mrcf/backends/_template) — the arm's
settings: how to reach it, its cameras, its gripper, and the poses and limits
measured on its rig.

Two kinds of setting live here:

  - the RIG SETTINGS (backends.RIG_SETTINGS), which deployments and the
    framework read without knowing which arm this is. Every one must be
    defined — None where the rig has no such thing — and
    tests/test_template.py checks this file has them all;
  - the arm's own, which only its env.py, its make_policies() and its tools read.

A value marked YOUR ARM is a placeholder until it is measured on the rig: the
order to measure them in is COMMANDS.md, "First-time setup on another arm".
Values a tool prints are pasted here; the overhead camera's live calibration
is kept apart, in overhead_calibration.json (bottom of this file).
"""

from pathlib import Path as _Path

from mrcf.backends.rig import Camera, Sensor  # noqa: E402,F401
from mrcf.safety import WorkspaceBounds  # noqa: E402

_HERE = _Path(__file__).parent

# This rig's own files — the overhead camera's calibration, the empty surface's
# photo, what the tools save — in rig/<this folder's name>/ (rig_state.py), not
# in git and not beside this file.
from mrcf import rig_state as _rig_state  # noqa: E402
RIG_DIR = _rig_state.rig_dir(_HERE.name, code_dir=_HERE)

# ── Connection (the arm's own) ────────────────────────────────────────────────
# YOUR ARM: what env.make_sdk() reaches the controller at — an IP address, a
# serial port. None: the in-memory stand-in (env.InMemoryArm), which is what
# tests/test_template.py runs the backend contract on.
HOST = None
# The task the tools take their task values from when none is named
# (tools/_common.task_config; --deployment and MRCF_DEPLOYMENT win). None: ask.
DEPLOYMENT = None
CONTROL_HZ = 125.0          # how often move_ee_to checks the interlock and the controller
SPEED_M_S = 0.10            # move_ee_to's speed; the policies' gentle steps slow it from there

# ── Cameras (rig settings) ────────────────────────────────────────────────────
# Every camera on the rig, by the name render_image() and the deployments use:
# where it is ("fixed" off the arm, or on the "gripper"), the OpenCV index or
# stream URL it opens from (None: not opened), its capture size. Any number of
# either kind; take out the ones the rig does not have (HARDWARE.md says what
# each is for, and what is lost without it). YOUR ARM: the sources.
CAMERAS = {
    "wrist":    Camera(mount="gripper", source=None, size=(640, 480)),
    "overhead": Camera(mount="fixed", source=None, size=(1280, 720)),
    # Another, a side view say: its calibration goes in its entry (model=,
    # homography=, printed by tools/calibrate_overhead_camera.py --camera side),
    # and the motion guard can watch through it too (MOTION_GUARD_CAMERAS).
    # "side": Camera(mount="fixed", source=2, size=(1280, 720)),
}

# The two cameras with jobs, by name (None: the rig has no such camera).
CAMERA_NAME = "wrist"                   # on the gripper: what the policies measure and align with
OVERHEAD_CAMERA_NAME = "overhead"       # fixed over the surface: scans, the gripper tag, the motion guard
CAMERA_W, CAMERA_H = CAMERAS[CAMERA_NAME].size if CAMERA_NAME else (640, 480)
OVERHEAD_CAMERA_W, OVERHEAD_CAMERA_H = CAMERAS[OVERHEAD_CAMERA_NAME].size if OVERHEAD_CAMERA_NAME else (1280, 720)
GRIPPER_CAM_K = None                    # the wrist camera's lens, at RENDER_W x RENDER_H (None: not
GRIPPER_CAM_DIST = None                 #   calibrated; the SO-101's calibrate_gripper_lens.py shows how)
# The overhead camera's pixels -> table (x, y), and its lens and pose: printed
# by tools/calibrate_overhead_camera.py (three touches on the board), refined
# by tools/tag_tour.py --calibrate, which saves to overhead_calibration.json.
OVERHEAD_HOMOGRAPHY = None
OVERHEAD_CAM_MODEL = None

RENDER_W, RENDER_H = 640, 480           # what the pipeline reasons in; render_image resizes to it

# ── Sensors (rig setting) ─────────────────────────────────────────────────────
# Every sensor beyond the joints and the gripper ({}: none). Each is read by
# read_sensor(), kept by the flight recorder and carried in a learned policy's
# stream; by its kind, a "force" one (a force, or a wrench whose first three
# are the force) feeds get_contact_force() and so the safety monitor's force
# limits, and a "gripper_contact" one gripper_loaded(). `read` names the env
# method that reads it (env.py), or is a function of the env.
SENSORS: dict = {
    # "wrench": Sensor(kind="force", unit="N, N·m", shape=(6,), read="tcp_wrench"),   # a UR: getActualTCPForce()
}

# ── The stream: what a learned policy is given and sends ──────────────────────
# A tick at a time (TemplateEnv.stream_spec; POLICIES.md): the grasp point and
# the jaws' opening, in and out, at STREAM_FPS. A model for this arm is trained
# on exactly this, elsewhere. YOUR ARM: the rate its demonstrations are
# recorded at, and its own name for ROBOT_TYPE — models are matched to an arm
# by it (StreamSpec.fingerprint), so two arms that differ must not share one.
STREAM_FPS = 30.0
ROBOT_TYPE = "my_arm"

# ── The work surface (rig settings) ───────────────────────────────────────────
# Its height in the arm's frame: where the closed jaws' TIPS rest on it — the
# grasp point's height there (get_ee_pos()[2]) less JAW_TIP_BELOW_GRASP_M.
# tools/calibrate_overhead_camera.py prints it from its touches. YOUR ARM.
TABLE_Z = 0.0
# Where the grasp point may go: the safety layer refuses a plan outside it
# and stops a move that leaves it. YOUR ARM: from a reach check on the rig;
# match the controller's own safety planes to it (a UR's, on the pendant).
WORKSPACE_BOUNDS = WorkspaceBounds(x_min=0.20, x_max=0.60, y_min=-0.30, y_max=0.30,
                                   z_min=TABLE_Z - 0.005, z_max=TABLE_Z + 0.40)
# The empty surface, for telling objects from its pattern (tools/capture_surface_reference.py).
SURFACE_REFERENCE_PATH = RIG_DIR / "surface_reference.png"

# ── The sentry (rig setting) ──────────────────────────────────────────────────
# The motion guard on the overhead camera, from every connect() to stop(): the
# arm freezes when anything other than the arm moves over this polygon (x, y in
# the arm's frame), and carries on 2 s after it has gone. It needs the overhead
# camera calibrated and arm_shape(), and an arm it cannot watch does not
# connect: so set it False on a new rig until tools/calibrate_overhead_camera.py
# has calibrated that camera (COMMANDS.md, "First-time setup on another arm"),
# then True for good. It ships True, so nothing moves unwatched by default.
# YOUR ARM: the zone is the work surface and a margin, not the floor beside it
# (tools/check_motion_guard.py tunes it).
MOTION_GUARD = True
MOTION_GUARD_ZONE = [(0.15, -0.35), (0.65, -0.35), (0.65, 0.35), (0.15, 0.35)]
# The fixed cameras it watches through, a guard each: more than one sees round
# what one cannot (under the arm, behind it). Each must be calibrated.
MOTION_GUARD_CAMERAS = [OVERHEAD_CAMERA_NAME] if OVERHEAD_CAMERA_NAME else []

# ── The gripper ───────────────────────────────────────────────────────────────
# A tag on the gripper body the overhead camera reads at every pick pose (rig
# setting; None: none). The sizes are what tools/make_tag_sheets.py prints;
# where it sits on the gripper, and the size it reads at, tools/tag_tour.py
# --calibrate measures.
GRIPPER_TAG_ID = None
GRIPPER_TAG_SIZES_M = {10: 0.032, 11: 0.028, 12: 0.024}
GRIPPER_TAG_POSE = None
GRIPPER_TAG_READS_AS_M = None

# The jaws, in the flange's frame (z toward the table when the jaws point
# down). YOUR ARM: from the gripper's drawing, then checked by touching.
GRIPPER_MAX_OPEN_M = 0.085              # the widest they open (a Robotiq 2F-85's)
JAW_CLOSED_POINT_M = (0.0, 0.0, 0.15)   # where they meet when closed
JAW_CENTRE_SLIDE = 0.0                  # 0: both jaws move; 0.5: one does, and the grasp point slides with it
JAW_SLIDE_AXIS = (0.0, 1.0, 0.0)        # which way it slides: toward the moving jaw — a guess until
                                        # tools/check_jaw_direction.py checks it on the rig (picks wait for that)
JAW_TIP_BELOW_GRASP_M = 0.010           # the jaw tips below the pads' centre
JAW_FREE_DEPTH_M = 0.040                # how far an object can stand above the grasp point before it meets the gripper
JAW_FINGER_THICKNESS_M = 0.008
JAW_FINGER_HALF_WIDTH_M = 0.011
# Which way the jaws open on a straight-down move (RigEnv.down_rotation): across
# the line from the base ("tangential") or along it ("radial"). The env turns
# them so, and the policies plan the room either side of an object for it: the
# one setting feeds both (backends/wiring.py). YOUR ARM: whichever turns the
# wrist least over the work surface, and suits what it picks.
JAW_AXIS = "radial"

# ── How it picks (the arm's own; backends/wiring.py SPEC_SETTING_FOR) ─────────
# Any GripperSpec field (mrcf/policies/parallel_jaw/spec.py) by its setting's
# name here; one left out keeps the policy's default (tuned on the SO-101).
# No wrist camera calibrated yet: no measuring at the hover, no visual
# servoing, and picks line up with the overhead camera alone (the wrist camera
# is used once GRIPPER_CAM_K is set).
MEASURE_AT_HOVER = False
VISUAL_SERVO_ENABLED = False
# Six joints reach straight down over the whole surface: no leaning retries.
DESCENT_LEAN_RETRIES = 0

# ── Poses (the arm's own) ─────────────────────────────────────────────────────
# Where go_home() and go_park() take the grasp point, straight down; park is
# clear of the overhead camera's view of the surface, for a scan. YOUR ARM:
# measured joint poses (a UR's moveJ) are safer to reach from anywhere.
HOME_XYZ = (0.35, 0.0, 0.25)
PARK_XYZ = (0.15, -0.35, 0.30)
BASE_XY = (0.0, 0.0)                    # the axis the arm swings about, in its own frame

# ── The overhead camera's live calibration (rig setting) ──────────────────────
# Last: what tools/tag_tour.py --calibrate saved, over the values above.
OVERHEAD_CALIBRATION_PATH = RIG_DIR / "overhead_calibration.json"

from mrcf import overhead_calibration as _overhead_calibration  # noqa: E402

_overhead_calibration.PATH = OVERHEAD_CALIBRATION_PATH
_overhead_calibration.apply_to(globals())
