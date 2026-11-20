"""
robot_config.py (mrcf/backends/ur5) — the UR5's settings: how to reach its
controller, its cameras, its gripper, and the poses and limits measured on
its rig.

Two kinds of setting live here:

  - the RIG SETTINGS (backends.RIG_SETTINGS), which deployments and the
    framework read without knowing which arm this is — every one defined,
    None where the rig has no such thing;
  - the arm's own, read only by env.py, arm.py, tools/ur5/, and make_arm()
    and make_policies() in __init__.py.

A value marked MEASURE is a placeholder until it is measured on the rig, in
the order of COMMANDS.md, "First-time setup on another arm" ("On the UR5").
Values a tool prints are pasted here;
the overhead camera's live calibration is kept apart, in
overhead_calibration.json (bottom of this file).
"""

import os as _os
from pathlib import Path as _Path

from mrcf.backends.rig import Camera, Sensor, zone_around  # noqa: E402,F401
from mrcf.safety import WorkspaceBounds  # noqa: E402

_HERE = _Path(__file__).parent

# This rig's own files in rig/ur5/ (rig_state.py), not in git.
from mrcf import rig_state as _rig_state  # noqa: E402
RIG_DIR = _rig_state.rig_dir(_HERE.name, code_dir=_HERE)

# ── Connection (the arm's own) ────────────────────────────────────────────────
# The controller's address is the rig's, not the code's, so it is not in git:
# one line in rig/ur5/host (HOST_FILE; "#" starts a comment), or MRCF_UR5_HOST,
# which wins. MRCF_UR5_HOST=offline, or no address in either, drives
# offline.py's UR5 in memory instead (HOST None; HOST_SOURCE None says nothing
# was set). The arm is cabled straight to the laptop (a USB-C Ethernet
# adapter), on a range no Wi-Fi the laptop joins uses: with the same range on
# both, macOS sends the arm's traffic out over Wi-Fi. The arm's static address
# is on the pendant (Setup Robot > Setup Network, mask 255.255.255.0); the
# laptop's adapter is set by hand on the same range, mask 255.255.255.0, no
# router.
HOST_FILE = RIG_DIR / "host"


def _host_setting() -> "tuple[str, str | None]":
    """(the address or "offline", where it came from: "MRCF_UR5_HOST", HOST_FILE, or None for nowhere)."""
    env = _os.environ.get("MRCF_UR5_HOST", "").strip()
    if env:
        return env, "MRCF_UR5_HOST"
    try:
        for line in HOST_FILE.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                return line, str(HOST_FILE)
    except OSError:
        pass
    return "offline", None


_HOST, HOST_SOURCE = _host_setting()
HOST: "str | None" = None if _HOST.lower() == "offline" else _HOST
NO_HOST_HINT = f"no controller address set: write it in {HOST_FILE}, or set MRCF_UR5_HOST"

# The task the tools take their task values from when none is named
# (tools/_common.task_config: --deployment and MRCF_DEPLOYMENT win): picking up
# and putting down everyday objects (deployments/tabletop).
DEPLOYMENT = "tabletop"

CONTROL_HZ = 125.0          # how often a move checks the interlock and the controller (a CB3's RTDE rate)
# Slow while the rig is new; the pendant's speed slider slows it further.
SPEED_M_S = 0.05            # move_ee_to's straight lines; the policies' gentle steps slow it from there
JOINT_SPEED_RAD_S = 0.4     # go_home/go_park's joint moves, the fastest joint
TRANSIT_HEIGHT_M = 0.10     # above TABLE_Z: go_home/go_park rise straight up to this before a joint move

# Passed to arm.URArm: accelerations, how hard a hold() stops, the stream's servoL.
ARM_TUNING = dict(
    accel_m_s2=0.5,
    joint_accel_rad_s2=1.0,
    hold_decel_m_s2=2.0,       # 0.05 m/s stops in under 1 mm
    hold_decel_rad_s2=3.0,
    servo_dt_s=1.0 / 30,
    servo_lookahead_s=0.1,
    servo_gain=300.0,
)

# ── Cameras (rig settings) ────────────────────────────────────────────────────
# Every camera on the rig, by name: where it is ("fixed" off the arm, or on the
# "gripper"), the OpenCV index or URL it opens from (None: not opened), its
# capture size. No wrist camera yet.
# The overhead is the USB "WebCamera" (32E6:9221), high above the table at a
# slant: cv2.VideoCapture(0) on this Mac (2026-11-12; 1 was the MacBook's own
# camera, 2 an iPhone's Continuity Camera). The index is OpenCV's, not ffmpeg's
# (the two count differently), and cameras coming or going can renumber it:
# probe with a raw cv2.VideoCapture(i).read() and look at the frame.
# The RealSense at the table's side (below): True to open it and check each
# place from the side too. Off since 2026-11-13 13:25: every run that went
# wrong that afternoon had it streaming (the overhead webcam's odd frames on a
# shared USB 2 hub, phantom movement in front of it); the one at 11:32 that
# built the tower in one command did not. Its calibration is kept for when it is on.
USE_SIDE_CAMERA = False
CAMERAS = {
    "overhead": Camera(mount="fixed", source=0, size=(1280, 720)),
}
# An Intel RealSense D435, low at the table's edge, looking along it toward
# the windows (2026-11-13). OpenCV opens its infrared view: greyscale, with
# its projector's dots on everything; its depth needs Intel's library,
# which has no build for this Mac. Not in MOTION_GUARD_CAMERAS: that low,
# it sees the room beyond the table. Index 1 on 2026-11-13; plugging a
# camera in can renumber them (tools/find_cameras.py). Moved twice off the
# windows' glare (its tag would not read); calibrated 2026-11-13 12:41 by
# tools/calibrate_camera_by_hand.py --camera side --fov-deg 87 (tag 3 on
# link 6, 20 poses, 0.78 px): 42 cm above the table beside the base, looking
# along it 20 deg down. It and the overhead camera put the blocks within
# 10-17 mm of each other. Moved again: calibrate it again.
if USE_SIDE_CAMERA:
    CAMERAS["side"] = Camera(
        mount="fixed", source=1, size=(1280, 720),
        model={"size": [1280, 720],
               "K": [[647.412, 0.0, 640.0], [0.0, 647.412, 360.0], [0.0, 0.0, 1.0]],
               "D": [0.02204, 0.0, 0.0, 0.0, 0.0],
               "T_world_cam": [[0.19686, -0.35014, 0.91578, -0.12263], [-0.98012, -0.0467, 0.19283, 0.37962],
                               [-0.02475, -0.93553, -0.35238, 0.41573], [0.0, 0.0, 0.0, 1.0]]})

# The two cameras with jobs, by name (None: the rig has no such camera).
CAMERA_NAME = None                      # on the gripper: none yet
OVERHEAD_CAMERA_NAME = "overhead"       # fixed over the surface: scans, the gripper tag, the motion guard
CAMERA_W, CAMERA_H = CAMERAS[CAMERA_NAME].size if CAMERA_NAME else (640, 480)
OVERHEAD_CAMERA_W, OVERHEAD_CAMERA_H = CAMERAS[OVERHEAD_CAMERA_NAME].size if OVERHEAD_CAMERA_NAME else (1280, 720)
GRIPPER_CAM_K = None
GRIPPER_CAM_DIST = None
# The overhead camera's pixels -> table (x, y), and its lens and pose: printed
# by tools/calibrate_overhead_camera.py, refined by tools/tag_tour.py
# --calibrate, which saves to overhead_calibration.json.
OVERHEAD_HOMOGRAPHY = None
# The lens: borrowed from the SO-101 rig's calibration of the same webcam and
# refined by tools/calibrate_camera_by_hand.py (2026-11-12, the wrist tag on
# link 5, 20 poses, 0.56 px); the pose as it found it. overhead_calibration.json
# (below) holds the pose in use, the homography and the lens as last refined
# (2026-11-20), and wins over these.
OVERHEAD_CAM_MODEL = {
    "size": (1280, 720),
    "K": [[875.135, 0.0, 640.0], [0.0, 875.135, 360.0], [0.0, 0.0, 1.0]],
    "D": [-0.33889, 0.0, 0.0, 0.0, 0.0],
    "T_world_cam": [
        [-0.88806, -0.406947, 0.213878, 0.15747],
        [-0.459675, 0.793089, -0.399636, 0.66891],
        [-0.006994, -0.453215, -0.891374, 1.089805],
        [0.0, 0.0, 0.0, 1.0],
    ],
}

RENDER_W, RENDER_H = 640, 480           # what the pipeline reasons in; render_image resizes to it

# ── Sensors (rig setting) ─────────────────────────────────────────────────────
# A CB3 estimates the force at the tool from its joint currents: off by several
# newtons at rest, by an amount that moves with the pose and the payload. So
# the sensor is its change since the move began (env.contact_wrench): what a
# contact adds. It feeds get_contact_force(), and so the safety monitor.
SENSORS: dict = {
    "wrench": Sensor(kind="force", unit="N, N·m", shape=(6,), read="contact_wrench"),
}
# The safety monitor's force limits for this arm's reading (safety.ExecutionMonitor
# takes them, for any task, unless the task sets its own): a skill stops at the
# first, and the second is logged. The framework's 5 N is for a real force sensor; this estimate moves
# by more than that in a slow move. Measured with tools/ur5/check_motion.py
# touching nothing, the slider at 25-33%, the payload set: the change swung
# 35-42 N on 3 cm lines (10:30, no payload), 43-74 N on lines and 45-67 N on
# home/park (13:50, 1.1 kg). So the stop sits above all of that: a collision
# detector, not a touch sensor; at a faster slider, measure again. What catches
# a collision first is the gripper's box and the slow speeds; the arm's own
# protective stop (150 N by default, Installation > Safety) is the backstop.
# A gentler sense is the joints' current against the controller's model
# (UR5Env.current_error): 0.003-0.014 A at rest, within ~0.6 A in those moves.
# The final approach is quieter: in the 6.7 s of a pick's last descent touching
# nothing the change read 1-10 N; a finger landing on the block then went 19 ->
# 70 -> 140 N in 0.2 s (2026-11-12 15:26, the stop at 109 N). So there it stops
# at 45 N (safety.approach_contact_limit; a pick's and a place's last descent).
# It was 30 N until 2026-11-13: descents touching nothing then read 24-27 N
# after a wrist turn, even counted from a settled reading (FORCE_SETTLE_S), and
# one stopped at 31.9 N. A finger on a block passes 45 N within ~0.05 s of
# passing 30 at that rise; the user agreed the change.
CONTACT_FORCE_LIMITS = {"max_contact_force_n": 100.0, "near_miss_force_n": 80.0,
                        "approach_contact_force_n": 45.0}
# A move's force is counted from a reading taken this long after the last
# move ended (UR5Env._settled_wrench): the estimate goes on changing after the
# arm stops, most after a wrist turn — 44 N over 0.25 s on 2026-11-13, the jaws
# turned 34° over a block, and a descent zeroed mid-change read 25-30 N
# touching nothing and stopped. Each move then waits out what is left of it.
FORCE_SETTLE_S = 0.35

# ── The stream: what a learned policy is given and sends ──────────────────────
STREAM_FPS = 30.0
ROBOT_TYPE = "ur5"

# ── The work surface (rig settings) ───────────────────────────────────────────
# Its height in the arm's base frame: where the closed jaw TIPS rest on it (the
# grasp point, get_ee_pos(), is JAW_TIP_BELOW_GRASP_M above that). The base
# frame's origin is the base's mounting face, so on the same table as the arm
# this is about 0. Measured 2026-11-12: tips jogged down onto the table at
# (+0.116, +0.304), flange at 0.1575 m pointing down: tips at +0.7 mm, inside
# the ruler's 1 mm on the 157 mm flange-to-tips.
TABLE_Z = 0.0
# ── Where the gripper may go (rig setting) ────────────────────────────────────
# The box the whole gripper stays inside, flange to jaw tips (arm frame, m:
# x, y, z ranges); the arm itself may pass outside it. Every move is checked
# along its way before it is sent (mrcf/envelope.py), and one that would take
# the gripper out is not sent. The user chose it on 2026-11-12 from the
# overhead camera's picture (tools/show_workspace.py): from the base out over
# the blocks; the teach pendant on the table and all behind the base outside.
# The pendant's safety planes should be the same box once its password is had
# (Installation > Safety > Boundaries), with the TCP at the jaw tips.
# x out to 0.90 m since 2026-11-20 (was 0.75; the user: "just expand the safety
# box - it can definitely reach it"): the toolbench's cardboard box stands at
# x 0.60-0.83 m, its far half outside 0.75; pointing straight down the arm
# reaches 0.89 m out with the grasp point 5 cm up, 0.79 m at 30 cm, 0.67 m at
# 45 cm (kinematics.ik), so out there it reaches low, not high.
GRIPPER_BOX = ((0.0, 0.90), (0.0, 0.75), (TABLE_Z, TABLE_Z + 0.80))
GRIPPER_BODY_RADIUS_M = 0.05     # the Hand-E round its flange-to-tips line, generously
# The gripper's body (that radius round it) this far at least from the base's
# axis: the box reaches in to the base (x, y = 0), the gripper must not — the
# housing is 75 mm round, and reaching in closer folds the forearm over the
# upper arm. Home and park keep 0.53 and 0.33 m to the gripper's line.
GRIPPER_BASE_CLEAR_M = 0.15
# Where the grasp point may go, for the plan-time checks and the safety
# monitor: the gripper's box.
WORKSPACE_BOUNDS = WorkspaceBounds(x_min=GRIPPER_BOX[0][0], x_max=GRIPPER_BOX[0][1],
                                   y_min=GRIPPER_BOX[1][0], y_max=GRIPPER_BOX[1][1],
                                   z_min=TABLE_Z - 0.005, z_max=GRIPPER_BOX[2][1])
SURFACE_REFERENCE_PATH = RIG_DIR / "surface_reference.png"

# ── The sentry (rig setting) ──────────────────────────────────────────────────
# The motion guard on the overhead camera, from every connect() to stop(). It
# needs the overhead camera calibrated, and an arm it cannot watch does not
# connect — so it is False until the camera is calibrated
# (tools/calibrate_camera_by_hand.py). Until then the arm moves unwatched: slowly, with the
# pendant's speed slider low and a hand on its emergency stop. Then True for good.
# On since 2026-11-12, the camera calibrated (tools/calibrate_camera_by_hand.py).
# It watches the table under the gripper's box and a margin round it: the
# teach pendant and the tape on the table (x < 0) are outside it.
MOTION_GUARD = True
MOTION_GUARD_MARGIN_M = 0.06
MOTION_GUARD_ZONE = zone_around(GRIPPER_BOX[0], GRIPPER_BOX[1], MOTION_GUARD_MARGIN_M)
MOTION_GUARD_CAMERAS = [OVERHEAD_CAMERA_NAME] if OVERHEAD_CAMERA_NAME else []

# ── The gripper ───────────────────────────────────────────────────────────────
# "robotiq": a Robotiq gripper through its URCap's socket (gripper.RobotiqSocket);
# None: no gripper — the arm runs, the jaws read closed, and pick and place are
# refused (mrcf/policies/arm_only.py). The rig's is a Robotiq Hand-E (photo,
# 2026-11-02): two parallel fingers, 0-50 mm, 20-185 N, 20-150 mm/s, 1.07 kg
# with its coupling (set that payload on the pendant: Installation > TCP).
# On a CB3 its cable runs to the control box, and the URCap serves the socket.
GRIPPER = "robotiq"
# What hangs on the flange, set by every connect (URArm: setPayload), so the
# controller's model of the arm, and through it the force it estimates, does
# not hang on what the pendant's installation holds: the Hand-E and coupling,
# 1.1 kg (the pendant, 2026-11-12). Its centre of gravity ESTIMATED: on the
# tool axis, half the body's 112 mm out; MEASURE from Robotiq's Hand-E manual
# ("Center of mass and tool center point").
PAYLOAD_KG = 1.1
PAYLOAD_COG_M = (0.0, 0.0, 0.055)
GRIPPER_MAX_OPEN_M = 0.052
# Room either side of the object as the open jaws come down round it. No wrist
# camera: the aim is the overhead camera's alone, millimetres off, and a block's
# sides read to a few degrees. At the framework's 4 mm a 30 mm block got 38 mm
# of opening and a finger landed on it (2026-11-12 15:26); at 11 they open fully.
# The room the neighbours leave still narrows it (ParallelJaw.fit).
JAW_SLACK_M = 0.011     # between the fingertip pads, fully open: measured 2026-11-12 (datasheet 50)
# Its position at fully open and fully closed (0-255, 0 open), read with
# tools/ur5/check_connection.py after opening, then closing (nothing between the
# fingers), from the pendant: 3 and 248 on 2026-11-12, just after activation.
ROBOTIQ_POS_OPEN = 3
ROBOTIQ_POS_CLOSED = 248
ROBOTIQ_SPEED = 64          # 0-255 over 20-150 mm/s: ~50 mm/s
ROBOTIQ_FORCE = 10          # 0-255 over 20-185 N: ~26 N, gentle; holds ~1 kg with its aluminium fingers

# A tag on the gripper body the overhead camera reads (None: none yet).
GRIPPER_TAG_ID = None
GRIPPER_TAG_SIZES_M = {10: 0.032, 11: 0.028, 12: 0.024}
GRIPPER_TAG_POSE = None
GRIPPER_TAG_READS_AS_M = None

# The jaws in the flange's frame (z out of the flange, toward the table when
# the jaws point down). The Hand-E with its UR coupling is 112 mm without
# fingertips and 157 mm to their ends (Universal Robots' listing; Robotiq's
# manual has 100.5 mm without the coupling): fingers ~45 mm long, the pads'
# centre ~11 mm up from their ends. MEASURE by touching the table; an error
# here does not reach the table, whose height is measured through this.
JAW_CLOSED_POINT_M = (0.0, 0.0, 0.146)
JAW_CENTRE_SLIDE = 0.0                  # both fingers move: the grasp point stays put
# The fingers open along the flange's x, as mounted. Until 2026-11-20 this was
# (0, 1, 0), never measured: the square letter blocks look the same either way,
# and the first tool pick turned the jaws along the hammer's handle, not
# across it — the fingers came down on it, 50 N (user: "obviously it should
# grab it 90 degrees from what its trying to do").
JAW_SLIDE_AXIS = (1.0, 0.0, 0.0)
# Which way the jaws open by default (env.down_rotation): across the line
# from the base ("tangential"), or along it ("radial").
JAW_AXIS = "tangential"
JAW_TIP_BELOW_GRASP_M = 0.011
JAW_FREE_DEPTH_M = 0.034                # from the grasp point up to the gripper's body (0.146 - 0.112)
JAW_FINGER_THICKNESS_M = 0.010          # MEASURE: the fingertips, with a ruler
JAW_FINGER_HALF_WIDTH_M = 0.010
# A pick's descent at full speed until the jaw tips are this far above the
# object's top (its height known), then gently (GripperSpec.fast_descent_above_top_m).
# 2026-11-13: 15 cm at 1.8 cm/s was 8.6 s of every pick.
FAST_DESCENT_ABOVE_TOP_M = 0.015

# ── How it picks (the arm's own; backends/wiring.py SPEC_SETTING_FOR) ─────────
# Any GripperSpec field (mrcf/policies/parallel_jaw/spec.py) by its setting's
# name here; one left out keeps the policy's default (tuned on the SO-101).
# No wrist camera: no measuring at the hover, no visual servoing; picks line
# up with the overhead camera alone.
MEASURE_AT_HOVER = False
VISUAL_SERVO_ENABLED = False
# Six joints reach straight down over the whole table: no leaning retries.
DESCENT_LEAN_RETRIES = 0
# Closed this wide, nothing is held: just short of fully open, since the jaws
# hold anything up to their stroke (the SO-101's 30 mm is for chess pieces)
# and the gripper itself says whether it stopped on something (gripper_loaded).
GRASP_MISS_WIDTH_M = GRIPPER_MAX_OPEN_M - 0.002
# What the jaws open to round an object whose width is not known.
UNKNOWN_OPEN_M = GRIPPER_MAX_OPEN_M - 0.002

# ── Poses (the arm's own) ─────────────────────────────────────────────────────
# go_home() and go_park() as joint angles, degrees, base to wrist 3. MEASURE:
# jog the arm there on the pendant, then tools/ur5/check_connection.py prints
# the line to paste. None: that pose is not measured, and going to it refuses.
# (HOME_XYZ/PARK_XYZ: a grasp point to go to instead, wrist down.)
# Home, 2026-11-12: the jaws straight down over the middle of the gripper's box,
# flange (+0.349, +0.399, +0.336) m, the tips 18 cm above the table.
HOME_JOINTS_DEG = (60.99, -96.10, -97.18, -79.06, -271.60, 257.30)
# Park, 2026-11-12: folded up beside the base, flange (+0.116, +0.304, +0.557) m,
# so the overhead camera sees every block; home <-> park keeps the gripper in its box.
PARK_JOINTS_DEG = (88.74, -70.07, -86.67, -116.08, -270.32, 285.01)
HOME_XYZ = None
PARK_XYZ = None
BASE_XY = (0.0, 0.0)                    # the axis the arm swings about, in its own frame

# ── The overhead camera's live calibration (rig setting) ──────────────────────
OVERHEAD_CALIBRATION_PATH = RIG_DIR / "overhead_calibration.json"

from mrcf import overhead_calibration as _overhead_calibration  # noqa: E402

_overhead_calibration.PATH = OVERHEAD_CALIBRATION_PATH
_overhead_calibration.apply_to(globals())
