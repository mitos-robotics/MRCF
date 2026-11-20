"""
mrcf/backends/so101/robot_config.py — one SO-101 arm, as it is physically set up.

Everything here describes the ROBOT, not the task: which port it is on, which
cameras are attached and where they point, the poses measured on this rig, and
the limits of its reach. A deployment imports these rather than restating them,
so a recalibration lands in one place instead of drifting between every task
that drives this arm.

Several values cannot be guessed from software — they depend on how your arm is
mounted and wired. Measure them with the scripts in tools/so101/
before the first run; each one prints what to paste back here.

Task-specific values — object vocabulary, grasp heights, perception prompts —
belong in the deployment's own config.py, not here.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

import os as _os
from pathlib import Path as _Path

from mrcf.backends.rig import Camera, Sensor, zone_around  # noqa: E402,F401
from mrcf.safety import WorkspaceBounds  # noqa: E402

_HERE = _Path(__file__).parent

# This rig's own files — the overhead camera's calibration, the empty surface's
# photo, the URDF, what the tools save — in rig/so101/ (rig_state.py), not in
# git and not beside this file.
from mrcf import rig_state as _rig_state  # noqa: E402
RIG_DIR = _rig_state.rig_dir(_HERE.name, code_dir=_HERE)

# The task the tools take their task values from when none is named
# (tools/_common.task_config: --deployment and MRCF_DEPLOYMENT win).
DEPLOYMENT = "chess"

# ── Hardware connection ─────────────────────────────────────────────────────────
FOLLOWER_PORT = _os.environ.get("SO101_FOLLOWER_PORT", "/dev/tty.usbmodem5B610326071")

FOLLOWER_ID   = _os.environ.get("SO101_FOLLOWER_ID", "my_follower")

CAMERA_NAME  = "front"   # gripper-mounted camera

# CAMERA_DEVICE_NAME-based resolution (resolve_camera_index() in
# mrcf/backends/so101/cameras.py, via ffmpeg's AVFoundation device listing) is OFF —
# empirically disproven on this machine: cv2.VideoCapture's own index space is
# NOT the same as ffmpeg's AVFoundation listing here (cv2 sees only 2 devices
# at indices 0/1; ffmpeg's listing interleaves 3+ including the built-in
# camera and virtual Continuity Camera entries at different positions). Using
# ffmpeg to resolve an index and feeding it to cv2.VideoCapture silently opens
# the WRONG camera. CAMERA_INDEX below was verified directly against cv2 (not
# ffmpeg) — probe with a raw per-index cv2.VideoCapture().read() + save, not
# the ffmpeg device listing, if this needs re-verifying.
CAMERA_DEVICE_NAME = None

# 2026-11-11: 1 since the wrist camera moved off the USB-C dongle's hub (it kept
# dropping off USB there) onto its own Mac port, which swapped cv2's order.
# Plugged back into the dongle, it is 0 again.
CAMERA_INDEX = 1   # verified directly via cv2.VideoCapture(1) — arm/gripper-mounted camera

CAMERA_W, CAMERA_H, CAMERA_FPS = 640, 480, 30

# ── External overhead camera (fixed, NOT gripper-mounted) ──────────────────────
# A stationary camera that sees the whole table — on this rig ~510 mm above the
# board's far side, looking back at the arm, so it sees the board's tags and the
# gripper tag. Unlike the gripper camera (which moves with the arm, so
# pixel_to_world() for it only works at one fixed joint pose — see
# OBSERVE_DEGREES below), a stationary camera can have ONE real calibrated
# mapping that works regardless of arm pose: OVERHEAD_HOMOGRAPHY and
# OVERHEAD_CAM_MODEL, both from tools/tag_tour.py --calibrate. Until they are
# measured they are None and using this camera for perception raises a clear
# error instead of silently back-projecting garbage.
# Set OVERHEAD_CAMERA_NAME = None to disable the overhead camera entirely
# (e.g. before you've physically connected one) — every script falls back to
# gripper-camera-only behavior.
OVERHEAD_CAMERA_NAME = "overhead"

# See CAMERA_DEVICE_NAME's comment above — name-based resolution is OFF here too.
OVERHEAD_CAMERA_DEVICE_NAME = None

OVERHEAD_CAMERA_INDEX = 0   # verified directly via cv2.VideoCapture(0) — sees the whole table (2026-11-11; was 1, see CAMERA_INDEX)

# Verified directly against cv2.VideoCapture(1). Both 640x480 and 1280x720 are
# self-consistent modes here: requested == get() == actual frame shape.
#
# 1280x720 even though render_image() resizes down to RENDER_W/RENDER_H for
# the VLM anyway: chessboard corner detection reads the RAW frame, and a
# natively-captured 640x480 of this board is too soft to find the 7x7 lattice
# in — calibrate_overhead_camera.py forces 1280x720 for exactly that reason.
# Upscaling 640x480 does not recover it; the detail has to be captured.
OVERHEAD_CAMERA_W, OVERHEAD_CAMERA_H, OVERHEAD_CAMERA_FPS = 1280, 720, 30

# 3x3 homography mapping overhead-camera pixel (u, v, 1) -> robot base-frame
# (x, y) at the table plane — i.e. the same frame get_ee_pos()/TABLE_Z use.
#
# Fitted with no touches, by tools/tag_tour.py --calibrate: the board's lattice
# in a clean frame gives camera -> board, the gripper tag over the hovers gives
# board -> arm (the arm's kinematics agree with it to ~1 mm). This fit (run
# 20261022_095819, 23 hovers): 0.90 mm mean, 3.00 max over the corners — a
# homography cannot model the lens's barrel distortion; OVERHEAD_CAM_MODEL
# below can. Re-run the tour if the camera or the arm's base moves.
OVERHEAD_HOMOGRAPHY = [
    [-0.0002199583991571237, 0.0011167180369747274, 0.07139259847847404],
    [0.001314963753458118, 0.00014382339088402278, -0.46759357658258305],
    [-0.00010625414075036726, 0.00039650754014540235, 1.0],
]

# RENDER_W/H is the resolution discover_scene()/perceive() actually see and the
# resolution pixel coordinates from the VLM are assumed to be in. render_image()
# resizes whatever the camera actually captures to this, in software, regardless
# of OVERHEAD_CAMERA_W/H above — so this is independent of the overhead camera's
# native/actual capture resolution.
RENDER_W, RENDER_H = 640, 480

# ── Gripper-camera visual servoing (optional) ────────────────────────────────
# Closing the control loop on the gripper camera instead of on forward
# kinematics. move_ee_to() is closed-loop on the ENCODERS: it minimises
# |FK(measured q) - goal|, a model estimate against a model target, so a wrong
# TCP offset, a wrong URDF or a wrong homography are all invisible to it — it
# converges confidently to the wrong place. Servoing on what the gripper
# camera actually sees removes every one of those from the error budget,
# because the feedback is the real relative geometry of jaws and piece.
#
# All three are MEASURED together by tools/so101/calibrate_gripper_servo.py. Leave
# them None and visual servoing stays off; nothing else changes.
#
# Where a piece STANDING ON THE BOARD appears in the gripper camera's frame
# when the open jaws are centred over it at servo (hover) height, as (u, v) at
# RENDER_W x RENDER_H. Servoing drives the piece's image onto this pixel.
#
# Measured by eye, not by the arm: a piece centred between the OPEN jaws by
# hand, the arm rising straight up with the jaws still open, the piece then
# located by piece_vision. Two shortcuts give a wrong pixel that looks right —
# servoing converges perfectly onto it: a piece held up in the closed jaws (the
# camera looks at an angle, so a piece 6 cm lower appears elsewhere), and the
# arm's gap centre read after closing on the piece (get_gripper_width under a
# squeeze is not the gap; the "centre" came out ~35 mm off).
#
# It depends on everything that moves the jaws or the view relative to each
# other: re-measure when JAW_AXIS, PREGRASP_OPEN_M, JOINT_OFFSETS_DEG or the
# servo height (calibrate_gripper_servo.py --height) changes — each moved it
# 20-60 px before.
#
# Measured with calibrate_gripper_servo.py --height 0.07: the cream pawn
# (39-41 mm) centred by hand between the open jaws (35 mm) at (0.24, 0), 71 mm
# hover, -5.1 deg held down and up. To be replaced by servoing on the jaw
# marks, which the wrist camera sees in the same frame as the piece.
GRIPPER_CAM_GRASP_PX: "tuple[float, float] | None" = (353.2, 259.7)

# 2x2 metres-per-pixel mapping from image offsets to GRIPPER-frame (x, y), at
# GRIPPER_CAM_REF_HEIGHT_M above the work surface. Gripper frame, not world:
# the camera is bolted to the gripper, so this is fixed, and the runtime
# rotates it into the world using the gripper's measured orientation. Scales
# linearly with height (pinhole), so one height calibrates all of them.
# Re-measure whenever JAW_AXIS changes (a quarter turn turned it ~20 deg) or
# JOINT_OFFSETS_DEG does (they turned it ~13).
#
# Measured with the grasp pixel, from calibrate_gripper_servo.py's four probes
# at the 71 mm hover: the cream pawn tracked cleanly on all four (0.34 mm/px;
# x: 20.8 mm -> 62 px, y: 23.0 mm -> 68 px).
GRIPPER_CAM_JACOBIAN: "list[list[float]] | None" = [
    [1.02013668e-04, -3.16699405e-04],
    [3.17075490e-04, 1.23232920e-04],
]

# Height above the work surface the Jacobian above was measured at — and so the
# hover the pick servos from (GripperSpec.servo_height_m).
GRIPPER_CAM_REF_HEIGHT_M: "float | None" = 0.0709

# The tilt the gripper had when GRIPPER_CAM_GRASP_PX was taken (tilted_rotation's
# degrees; calibrate_gripper_servo.py prints it). The target pixel is one sight
# line fixed in the gripper, straight down only at THIS tilt: a pick's held tilt
# runs from ~5 deg on ranks 3-4 to 15 on rank 6 and 35-50 on rank 8, and
# aligning the same pixel there puts the jaws 8 and 25-41 mm off the piece (the
# slack is ~7 mm a side). The servo steps for the difference
# (SO101Env._servo_tilt_offset). None: aims as calibrated at every tilt.
#
# Measured with the grasp pixel.
GRIPPER_CAM_REF_TILT_DEG: "float | None" = -5.1

# How high up the calibration's object the detector's point was: it reports
# the centroid of the object's outline, taken as SERVO_POINT_HEIGHT_FRACTION of
# its height — the cream pawn (40 mm) the calibration used. A pick passes the
# same fraction of its own object's height when it knows it.
SERVO_POINT_HEIGHT_FRACTION = 0.4
GRIPPER_CAM_REF_POINT_HEIGHT_M = SERVO_POINT_HEIGHT_FRACTION * 0.040

# The wrist camera's centre in gripper_frame_link, metres: from the hand-eye
# fit rejected for GRIPPER_CAM_POSE below (76 mm along -x, as an overhead
# photo had it).
# Approximate is enough for the tilt step above — 20 mm off in any axis moves
# it under 1.5 mm up to rank 7 and 2-4 mm on rank 8.
GRIPPER_CAM_POSITION_M: "tuple[float, float, float] | None" = (-0.076, 0.033, -0.053)

# Gripper camera intrinsics at RENDER_W x RENDER_H: OpenCV camera matrix and
# distortion (k1, k2, p1, p2, k3). From tools/so101/calibrate_gripper_lens.py: 28
# views of the chessboard, 481 corners after rejecting the ones pieces and the
# jaw corrupted, 0.53 px RMS. Lets the piece detector model the board plane
# across the whole frame instead of only near the corners it found.
# Re-measure if the lens is refocused: focal length moves with focus.
GRIPPER_CAM_K: "list[list[float]] | None" = [
    [343.09, 0.0, 341.30],
    [0.0, 342.23, 263.10],
    [0.0, 0.0, 1.0],
]
GRIPPER_CAM_DIST: "list[float] | None" = [0.08525, -0.12433, -0.00052, -0.00130, 0.03745]

# ── Camera models: measuring objects on ANY surface ──────────────────────────
# Where each camera is, so plane_geometry can turn an object's outline into
# its foot, height and widths without a calibration target in view (served by
# SO101Env.camera_model). Positions come out in the arm's own frame — where to
# send it — and sizes in real mm via ARM_FRAME_SCALE.
#
# Real mm per arm-frame mm. With JOINT_OFFSETS_DEG correcting the joint
# readings the 40 mm board measures 40.0 mm in the arm frame; without them the
# arm's frame reads ~9% short.
ARM_FRAME_SCALE = 1.0001

# The overhead camera, at OVERHEAD_CAMERA_W x OVERHEAD_CAMERA_H: lens (K, D) and
# pose in the arm frame. More exact than OVERHEAD_HOMOGRAPHY, which cannot model
# the lens's barrel distortion (k1 = -0.34).
#   Lens: from one photo of the board (calibrate_arm_offsets.py, 0.17 px RMS).
#   It holds when the camera moves: from its present place it reprojects the
#   board's lattice to 0.22 px, and freeing the focal length in a joint fit
#   with the gripper tag moves it under 1%. Re-fit if the camera is refocused.
#   Pose: tools/tag_tour.py --calibrate, as OVERHEAD_HOMOGRAPHY.
OVERHEAD_CAM_MODEL: "dict | None" = {
    "size": (1280, 720),
    "K": [[858.78, 0.0, 640.0], [0.0, 858.78, 360.0], [0.0, 0.0, 1.0]],
    "D": [-0.3449, 0.0, 0.0, 0.0, 0.0],
    "T_world_cam": [
        [-0.144315, 0.971165, -0.189765, 0.353706],
        [0.989518, 0.140613, -0.032907, 0.006002],
        [-0.005275, -0.192525, -0.981278, 0.534828],
        [0.0, 0.0, 0.0, 1.0],
    ],
}

# The gripper camera's pose on the gripper: gripper_frame_link -> camera, 4x4,
# metres (camera at T_world_gripper @ GRIPPER_CAM_POSE). With it every gripper
# frame comes with the camera's pose from the joint angles, so an object can be
# measured — and located in the arm frame, for aiming — with no board in view.
# GRIPPER_CAM_PLANE_Z_M is the work surface's height as this calibration sees
# it (TABLE_Z if None). MEASURED by tools/so101/calibrate_gripper_pose.py; None
# until then, and the gripper camera measures off the board's own pattern.
#
# GRIPPER_CAM_JOINT_DEG goes with it: corrections to the joint readings, on top
# of JOINT_OFFSETS_DEG, for the wrist camera's pose only (SO101Env.
# _gripper_frames), fitted together with the mount. Under load this arm
# reads its joints a few degrees off, the same way at the same pose, and the
# mount alone came out centimetres wrong (backends/so101/HISTORY.md).
#
# GRIPPER_CAM_SCAN_SHIFT_M: where a position the overhead scan measured
# stands in the frame the wrist camera is placed in (scan + this, x y z,
# metres), from the same calibration: both cameras saw the same board. It
# holds while the overhead camera keeps the pose it had then
# (GRIPPER_CAM_SCAN_SHIFT_AT, its calibration's time; a warning below when it
# has moved on). It is under 10 mm, but a pick takes an outline for a piece
# only within 6 mm of where the scan has it: a stale one costs who is who.
# Re-fitted 2026-11-03 17:10 after the overhead camera and the board were moved
# (the old one, (-8.3, +2.9, -2.5) mm, left the d1 queen, c1 and e1 all
# unmatched, 6-13 mm off): 64 feet in the wrist frames of tag tour
# reports/tag_tour/20261103_170807 against the 17:11 scan, median spread 3.4 mm.
#
# Re-fitted 2026-11-18 17:08 on the bare table, the overhead camera re-posed
# (49 mm, 7 deg) on 11-16 since: tools/calibrate_wrist_on_objects.py, the
# objects on the table as the ruler — 20 feet of a figure and the plug in the
# wrist frames of rig/so101/wrist_on_objects/20261118_170740 against the scan,
# 3.0 mm median left over (the figure's own 0.4-1.3 mm, but at two steep
# leans). It had been (-6.5, -3.2, -2.5) mm: 4.8 mm off across.
#
# Measured 2026-11-03 (rig/so101/gripper_pose_frames/20261103_104117/
# result.json, two captures, 148 views from 8 to 17 cm over the whole board:
# 3.1 px on views the fit did not see). A pick's look over a full board then
# finds each piece's foot within 0.1-3.5 mm of the scan's (look tour
# 20261103_110243). The camera says who and where; the jaws are aimed by the
# overhead camera's sighting moved by that (aligners.FootAligner): this mount
# and the jaw model put the centre between the open jaws ~11 mm from where a
# piece centred there by hand appears (GRIPPER_CAM_GRASP_PX).
GRIPPER_CAM_POSE: "list[list[float]] | None" = [
    [-0.323671, 0.77706, 0.539828, -0.058912],
    [-0.945968, -0.253995, -0.201571, 0.017139],
    [-0.019519, -0.575903, 0.817285, -0.044197],
    [0.0, 0.0, 0.0, 1.0],
]
GRIPPER_CAM_JOINT_DEG: "dict[str, float] | None" = {
    "shoulder_pan": 1.19, "shoulder_lift": 0.97, "elbow_flex": 3.409, "wrist_flex": 0.114,
}
GRIPPER_CAM_PLANE_Z_M: "float | None" = 0.0142
GRIPPER_CAM_SCAN_SHIFT_M: "tuple[float, float, float] | None" = (-0.0087, -0.0132, -0.0025)
GRIPPER_CAM_SCAN_SHIFT_AT: "str | None" = '2026-11-16T10:27:02'

# A photo of the EMPTY work surface from the overhead camera, at
# OVERHEAD_CAMERA_W x OVERHEAD_CAMERA_H — how surface_scan tells an object from
# a patterned surface (tiles, a printed mat, a board) without knowing the
# pattern. Taken by tools/capture_surface_reference.py; retake when the surface
# or the camera changes. A plain table works without one.
SURFACE_REFERENCE_PATH = RIG_DIR / "surface_reference.png"

# Per-cycle joint-degree safety clamp — no single send_action() call may move
# a joint more than this many degrees (see lerobot's max_relative_target).
# Keeps IK convergence, a bad VLM pixel estimate, or a software bug from
# producing a sudden large motion. Lower if the arm ever jerks.
MAX_RELATIVE_TARGET_DEG = 8.0

# ── Fiducials (tools/make_tag_sheets.py prints them) ──────────────────────────
# An AprilTag (tag36h11) on the back of the wrist camera's mount, which is
# screwed to the gripper body's moving-jaw end: no servo between it and the
# fixed jaw, so its offset to the jaws is a constant, and it faces outward and
# up — at the overhead camera across the board. Three sizes were printed, one id
# each, so the id alone says which is stuck on. Black-square edge, metres.
GRIPPER_TAG_SIZES_M = {10: 0.032, 11: 0.028, 12: 0.024}
# The one on the arm: the 24 mm, the biggest that fitted.
GRIPPER_TAG_ID = 12

# Where that tag sits on the gripper: gripper_frame_link -> tag, 4x4, metres —
# ~70 mm out, 20 sideways and 53 back from the closed-jaw grasp point, where the
# wrist camera sits. And the size it reads at through the overhead camera
# (GRIPPER_TAG_READS_AS_M), under the 24.0 mm it is printed (ruler) — the
# camera's image sharpening puts the detected corners a fraction of a pixel
# inside the black square (it reads the same with clear tape over the tag and
# without). The pose is fitted at that size, so the two go together. Measured
# by tools/tag_tour.py with the overhead camera's pose (run 20261022_095819, 23
# hovers, 1.68 px). Re-run the tour if it is re-stuck.
GRIPPER_TAG_POSE: "list[list[float]] | None" = [
    [0.255452, 0.809341, -0.528878, -0.069847],
    [0.964624, -0.176489, 0.195839, 0.014439],
    [0.065159, -0.560196, -0.825794, -0.052600],
    [0.0, 0.0, 0.0, 1.0],
]
GRIPPER_TAG_READS_AS_M = 0.02322

# ── Kinematics (Cartesian control) ───────────────────────────────────────────
# Fetched by fetch_urdf.py. Without this file, go_home()/gripper/camera still
# work; move_ee_to()-based skills (pick/place/move_above) will raise
# NotImplementedError until you fetch it and `pip install placo`.
URDF_PATH = RIG_DIR / "urdf" / "so101_new_calib.urdf"

# ── Joint-space home pose (degrees) — MEASURE on your physical arm ─────────────
# Run find_poses.py, disable torque, move the arm BY HAND to a safe pose that
# is raised clear of the table and any obstacles, and paste the printed
# degrees here. This pose is used directly (no IK), so it must be safe
# regardless of whether kinematics/placo are installed.
HOME_DEGREES = {
    "shoulder_pan":  -2.2,
    "shoulder_lift": -87.0,
    "elbow_flex":     40.5,
    "wrist_flex":     68.0,
    "wrist_roll":    -14.1,
}

# ── Joint-space overhead-camera parking pose (degrees) — MEASURE on your arm ──
# Run find_poses.py, move the arm BY HAND somewhere that keeps it fully clear
# of the overhead camera's view of the board, label it "calib", and paste the
# printed degrees here.
#
# Used before EVERY overhead scan, not just calibration: by
# calibrate_overhead_camera.py before it captures the board photo, and by
# skills.py's goto_observe_pose() before each discover_scene(). The fixed
# overhead camera's homography works from any arm pose, so there's no pose to
# "reach" for it — the only thing that matters is that the arm isn't occluding
# the objects being looked for. Kept separate from HOME_DEGREES (used
# everywhere else: between commands, on shutdown, etc.) since the best parking
# spot for "out of this specific camera's frame" isn't necessarily the same as
# the best general resting pose. Falls back to HOME_DEGREES if left as None.
#
# Folded back below the board's near edge, placed by hand. Clear with the
# camera on the board's far side: scans from there find all 64 squares.
CALIBRATION_PARK_DEGREES: "dict[str, float] | None" = {
    "shoulder_pan":  -2.9,
    # Measured at -99.8, stored at the calibrated limit: the joint was nudged
    # slightly past its swept range by hand, and anything beyond -96.0 just
    # gets clamped (printing a warning) on every single scan.
    "shoulder_lift": -96.0,
    "elbow_flex":     39.2,
    "wrist_flex":    -27.6,
    "wrist_roll":    -13.8,
}

# ── Joint-space observe pose (degrees) — MEASURE on your physical arm ──────────
# Run find_poses.py, move the arm BY HAND so the gripper-mounted camera looks
# down at the whole table, and paste the printed degrees here.
#
# This is deliberately a fixed joint pose, not a Cartesian move_ee_to() target:
# an unconstrained IK solution can ask for an angle outside this arm's
# calibrated range (e.g. shoulder_lift beyond ±57°) and the servo will just
# stall against its own limit instead of reaching a pose that can see the
# table. A directly-measured joint pose can't have that problem.
OBSERVE_DEGREES = {
    "shoulder_pan":   0.0,
    "shoulder_lift":  0.0,
    "elbow_flex":     0.0,
    "wrist_flex":     0.0,
    "wrist_roll":     0.0,
}

# ── Table geometry (metres, in the robot base frame get_ee_pos() returns) ──────
# TABLE_Z: the surface the arm's calibrations were made on — the chessboard's
# top, where calibrate_arm_offsets.py's touches fitted the joints and the wrist
# camera's pose was measured. The tools that work on that board, and a task
# that names no surface of its own, go by it. It is NOT the table the board
# stands on (TABLE_TOP_Z below): a task on the bare table works from its own
# surface (the tabletop measures it: rig/so101/tabletop_surface.json), not this.
#
# MEASURE on your setup: with find_poses.py (torque disabled), move the gripper
# fingertips down until they just touch the surface and read off the printed FK
# z value. From calibrate_arm_offsets.py's fit: the board's surface, level in
# the corrected arm frame.
TABLE_Z = 0.0167
# The table itself, under everything the arm works on (the board stands 2-3 cm
# on it). Touched with the jaw tips, the arm limp, read through its own
# kinematics: +0.9 and +0.3 mm at (0.304, 0.016) and (0.237, 0.117), 2026-11-17.
TABLE_TOP_Z = 0.0007

# Added to the motors' joint angles before the kinematic model sees them
# (SO101Env, OffsetKinematics). Fitted by tools/so101/calibrate_arm_offsets.py from
# 9 closed-jaw touches on the board's corners: 12.3 mm RMS off a flat, square
# 40 mm grid with the readings as they were, 3.4 mm corrected.
# shoulder_lift is well pinned down (+/- 0.6 deg) — it is what made the far
# edge read 23 mm low; elbow and wrist less so (+/- 2-3 deg), held near zero
# where the touches cannot tell them apart. Re-fit after lerobot-calibrate.
JOINT_OFFSETS_DEG: "dict[str, float] | None" = {
    "shoulder_lift": -5.96,
    "elbow_flex": -4.02,
    "wrist_flex": -2.72,
}

# Joints the IK may take over their whole calibrated range, not just the
# URDF's. wrist_flex: the URDF stops it at +-95 deg, the servo's calibration
# at +-101. Close to the base that is what decides reach: over d1 (91 mm from
# the pan axis) the highest point within +-95 is 46 mm, under the 56 a 41 mm
# pawn's hover needs; within +-101 a pick and a place there both reach.
# Nothing on the gripper comes near the forearm at full bend.
# () puts every joint back to the tighter of the two.
IK_CALIBRATED_RANGE_JOINTS: tuple = ("wrist_flex",)

# Used only by the GRIPPER camera's pinhole back-projection fallback (e.g.
# look-around poses) — NOT as a motion target (see OBSERVE_DEGREES above),
# and NOT used by the overhead camera, which uses OVERHEAD_HOMOGRAPHY above
# instead. OBSERVE_CENTER_XY should be roughly the table-frame (x, y) the
# gripper camera is centred on at OBSERVE_DEGREES; OBSERVE_HEIGHT_M should be
# its height above TABLE_Z there (find_poses.py prints FK z at OBSERVE_DEGREES
# — use that minus TABLE_Z).
OBSERVE_CENTER_XY = (0.20, 0.0)

OBSERVE_HEIGHT_M  = 0.25

# pixel_to_world() in mrcf/backends/so101/cameras.py approximates the gripper camera as
# looking straight down from OBSERVE_HEIGHT_M, centred on OBSERVE_CENTER_XY —
# only valid while the arm is actually at the OBSERVE_DEGREES pose. This is a
# rough pinhole estimate (a few cm of error). The overhead camera doesn't need
# this approximation — it's stationary, so OVERHEAD_HOMOGRAPHY is a real
# calibrated mapping instead.

# ── Look-around poses (optional) ────────────────────────────────────────────────
# If a referenced object isn't in the primary observe pose's view, skills.py's
# look_around() sweeps through these alternate vantage points before giving
# up. Each needs the same three pieces as the primary observe pose — a
# directly-measured joint pose (never Cartesian IK, for the same reason
# OBSERVE_DEGREES is joint-space) plus its own pixel_to_world calibration,
# since a different camera position/angle needs a different one.
#
# Capture with find_poses.py — it accepts any label, not just home/table/
# observe, and prints a {"degrees": ..., "center_xy": ..., "height_m": ...}
# block for each one you use. Empty list = feature is off; objects out of the
# primary view's frame just report "not found" the same as before.
LOOK_AROUND_POSES: list[dict] = [
    # {"degrees": {"shoulder_pan": ..., "shoulder_lift": ..., "elbow_flex": ...,
    #              "wrist_flex": ..., "wrist_roll": ...},
    #  "center_xy": (x, y), "height_m": h},
]

# ── Gripper ──────────────────────────────────────────────────────────────────
# The widest the pick and place policies ever open the jaws (open_gripper, and
# the cap on any planned opening). Without GRIPPER_WIDTH_TABLE it is also the
# opening assumed at 100% for a straight-line map — badly wrong for this jaw
# (see the table). The jaws open to 100 mm (69.5%); 80 leaves the fingers short
# of their stop. A task among close neighbours opens less (chess: 45 mm among
# 40 mm squares, its profile) — each pick is narrowed to the room its
# neighbours leave anyway.
GRIPPER_MAX_OPEN_M  = 0.08


# The real jaw opening at each gripper motor position: [(motor %, metres)],
# measured with calibrate_gripper_width.py (gap at the jaw ENDS, by ruler).
# get_gripper_width / set_gripper interpolate it.
#
# A straight line puts 100% at 45 mm. In fact 69.5% is 100 mm, so "30 mm" on
# the line is ~96 mm — the jaws nearly wide open. Every width in this file is
# real millimetres.
#
# Under a squeeze the reading is not the gap: the motor runs on past contact
# (a held knight read 1.6%, which is 3 mm here). gripper_loaded() — the
# servo's load — is what says something is held.
GRIPPER_WIDTH_TABLE: "list | None" = [
    (  0.0, 0.0000),
    (  5.2, 0.0100),
    (  9.9, 0.0200),
    ( 15.1, 0.0250),
    ( 24.7, 0.0420),
    ( 40.2, 0.0650),
    ( 69.5, 0.1000),
]

# How far the grasp point the kinematics aim moves per millimetre of opening
# (JAW_CENTRE_SLIDE), and which way, in gripper_frame_link (JAW_SLIDE_AXIS:
# toward the moving jaw). One jaw moves, the other is fixed, so the centre of
# the gap moves half as far as the opening: 0.5. Closed, neither matters.
#
# The axis is the gripper frame's -y, not the -x the URDF's meshes suggested.
# The wrist camera sees the two jaw tips side by side, the moving one on the
# right, and its servo Jacobian (measured) says image-right is the gripper's
# -y (0.34 mm of y to 0.08 of x per pixel). Along x the true slide is zero: a
# model slid along x sees the servo move the jaws back by exactly the slide.
# The wrist servo hides an error here (its target pixel was set with the jaws
# open); descents aimed by the overhead camera, and places, do not.
#
# JAW_CLOSED_POINT_M is where the jaws close. Across the jaws (y) it is the
# mean of where places land in the overhead scans (confirmed by eye in the
# wrist camera's grasp frames), since the held object's centre and the open
# gap's centre both hang off this point; along the jaws places land within
# ~3 mm. The kinematics' calibrations do not use this point (the gripper tag
# is fitted to the gripper frame; the servo descends where it aligned), so
# changing it moves only where a descent aimed by the overhead camera, and a
# place, put the jaws.
JAW_CENTRE_SLIDE = 0.5
JAW_SLIDE_AXIS = (0.0, -1.0, 0.0)
# Checked by use (mrcf/arm_facts.py): every chess and table pick since
# 2026-10 turned the jaws by it, the wrist camera watching them close.
JAW_SLIDE_AXIS_CHECKED = True
JAW_CLOSED_POINT_M = (0.00035, -0.002, 0.0)    # gripper_frame_link, the jaws shut

# Jaw geometry along the approach axis, relative to the grasp point (the pad
# centre get_ee_pos reports). Measured off the URDF's own meshes, carried into
# gripper_frame_link:
#   - both jaw tips end ~7 mm PAST the grasp point, so at a hover the tips are
#     that much lower than the reported height;
#   - the gripper servo's housing starts ~62 mm BEHIND it, directly over the gap.
# So a piece can stand at most ~62 mm above the grasp point before its top meets
# the servo. 55 mm keeps a margin. A deployment uses these to work out how high
# to hover over, and how high to grasp, an object of a given height.
JAW_TIP_BELOW_GRASP_M = 0.007
JAW_FREE_DEPTH_M      = 0.055

# Each finger's footprint near its tip, from the same meshes: 7-9 mm thick
# along the opening direction over the bottom ~12 mm, ~+/-7 mm wide across it.
# It thickens to ~12 mm further up, but there a neighbouring object is
# narrower than at its foot, which is what the bottom of the finger must miss.
# A deployment uses these to check open jaws will not land on whatever stands
# next to the object.
JAW_FINGER_THICKNESS_M  = 0.009
JAW_FINGER_HALF_WIDTH_M = 0.008

# Which way the jaws open, relative to the arm:
#   "radial"      along the arm, toward and away from the base
#   "tangential"  across it
# The wrist roll that gives it follows from JAW_SLIDE_AXIS, the gripper axis
# the jaws open along (SO101Env._x_along_arm).
#
# It decides what a tilted approach does to the object between the jaws, and
# which neighbours a finger can land on. This arm can only tilt the gripper in
# its own vertical plane (three pitch joints, no yaw), and it must tilt
# wherever straight down is out of reach — ranks 5-8, and every hover high
# enough to clear a tall piece. With radial jaws that tilt is in the plane the
# jaws open in, so a standing piece's top swings toward one jaw by (height
# above the grasp point) x tan(tilt). With tangential jaws the tilt runs along
# the pad faces and the piece stays parallel to them. The fingers come down
# beside the object along the same axis: radial, on the squares in front of
# and behind it; tangential, on the files either side.
#
# Tangential: the wrist roll near home (~0 deg on every square), the wrist
# camera hanging along the arm. Its half turn (+-180) is outside the roll's IK
# range, so the camera always hangs that way.
# Radial needs the roll at about -92 deg everywhere, which hangs the camera
# 75 mm out SIDEWAYS at jaw height, over the neighbouring files, and takes the
# wrist roll mid-pick to -96 deg, where it overloads and drops off the bus.
#
# Offline with deployments/chess/tests/reach_map.py, on estimated piece sizes:
# a king's top meets a jaw on 43 of 64 squares with the jaws along the arm, on
# none across it.
JAW_AXIS = "tangential"

# The wrist camera sticks out of the gripper, and it is what hits neighbouring
# pieces: (x, y, radius) in metres, gripper frame, from the grasp point — its
# board ~75 mm out along the gripper's -x (square to the line the jaws open
# along), at about jaw height, ~35 mm across. Read off an overhead photo of the
# gripper with the gripper's axes projected onto it, so approximate: measure
# with a ruler to refine. With the jaws radial it hangs out sideways, over the
# neighbouring files; a pick or place then turns the wrist half a turn when
# that keeps it clearer (mrcf/policies/parallel_jaw/reach.py, hold_orientation).
# With the setting it hangs along the arm, and the half turn is out of the
# wrist roll's range.
GRIPPER_CAM_BODY_M: "tuple[float, float, float] | None" = (-0.075, 0.0, 0.020)

# ── Grasping ─────────────────────────────────────────────────────────────────
# What is true of THIS gripper whatever it grips (mrcf/policies/parallel_jaw/;
# make_policies() turns these into its GripperSpec): what an empty close reads,
# how fast the jaws may close on something small, the grasp its calibrations
# were measured with. How a DEPLOYMENT drives it — how hard to hold, how wide
# to open among its neighbours, what a look that disagrees means — is that
# deployment's own: deployments/<name>/profile.py (mrcf/profiles.py; chess,
# tidy). 2026-11-17: ~30 rules tuned on chess pieces lived here, and the
# SO-101's table work inherited them all.

# Gripper wider than this after closing = nothing grasped.
GRASP_MISS_WIDTH_M  = 0.030

# Gripper narrower than this after closing = fully closed on nothing.
#
# Read after squeeze_gripper() (gripper.py), which closes the jaws the rest of
# the way: set_gripper's own close stops early, wherever its stall detector
# fires. Measured on the rig after the squeeze, on the old straight-line map:
#   empty   0.04 mm  (2/2 trials)
#   knight  1.7 mm
#   pawn    2.8 mm
# 0.8 mm sits between empty and the narrowest piece measured; on
# GRIPPER_WIDTH_TABLE the same motor position (1.8%) is 3.4 mm.
#
# It is half of the grip signal, with the servo's load: with grip pads on, the
# pads meet at 0.9-1.0 % (1.7-1.9 mm) and press each other at the torque cap —
# an empty close reads load 160-200 of 200, a grip by load alone. Jaws stopped
# narrower than this hold nothing, whatever the load. A held pawn reads ~16 mm.
# Re-measure (an empty close in the air) if the pads change.
MIN_HOLD_WIDTH_M    = 0.0034


# MEASURED against the arm's actual straight-down reach, not chosen for clearance.
# Holding the gripper vertical costs wrist_flex range, and that range runs out
# with height: sweeping z over the board, a vertical grasp solves with joint
# margin to spare up to ~0.07 m and saturates wrist_flex at its +95 deg limit
# above that. Past that ceiling (at 0.08 or 0.10) every hover and lift stalls
# ~1.9 cm short with wrist_flex pinned — and tilting does not buy anything back
# (it saturates at every tilt from 0 to 60 deg at those heights). Re-measure
# with check_reach.py after any lerobot-calibrate.
HOVER_HEIGHT      = 0.06   # m above TABLE_Z: the least pre-grasp / pre-place hover
LIFT_HEIGHT       = 0.07   # m above TABLE_Z: the least carry height after a pick

# The grasp the rig was first tuned on: a ~45 mm wooden pawn, gripped at 20 mm
# with the jaws open 30 mm. A pick sizes its own grasp from the object (see
# mrcf/policies/parallel_jaw/); the tools use these as a typical small object.
# GRASP_HEIGHT_M also sets the carry clearance: a lift goes LIFT_HEIGHT -
# GRASP_HEIGHT_M above whatever height an object was grasped at.
GRASP_HEIGHT_M    = 0.02
# Also the opening the wrist-camera servo is calibrated at (the centre of the
# gap moves with it) — re-run calibrate_gripper_servo.py when changing it.
# 35 mm clears any piece's foot here with slack.
PREGRASP_OPEN_M   = 0.035


# Per-iteration joint step for the final descent onto an object, in degrees.
# Deliberately below MAX_RELATIVE_TARGET_DEG (8.0, the arm-wide ceiling): the
# jaws straddle a ~2 cm piece over the last few centimetres, and arriving fast
# enough to nudge it loses the grasp however well-aimed the target was.
GRASP_STEP_DEG    = 3.0

# Per-cycle jaw travel when closing onto an object, in percent of full span.
# The arm-wide ceiling (MAX_RELATIVE_TARGET_DEG = 8.0) applies to the gripper
# motor too, and 8 pct/cycle at control_hz=20 is ~160 pct/s — near closed,
# ~300 mm/s of jaw travel. A jaw moving that fast reaches a 2 cm wooden pawn
# before the opposite jaw does and knocks it off the board instead of closing
# on it. 1.5 is ~40-60 mm/s near closed (GRIPPER_WIDTH_TABLE: ~2 mm per %), and
# grips fine. Only the CLOSE needs this; opening happens clear of the object.
GRASP_CLOSE_STEP_PCT = 1.5


APPROACH_TOL      = 0.015  # m — a hover counts as reached within twice this
MAX_SKILL_RETRIES = 3      # attempts per pick before reporting failure

# ── Approaches: how a pick comes at an object, by where it stands ────────────
# mrcf/policies/approaches.py's Approach: top_down with the wrist servo wherever
# it reaches short of the far zone, near_base close to the base where it does
# not, reach_out far out; the others are the fallbacks. Distances in the arm's
# frame from the shoulder_pan axis (SO101Env.base_xy).
#
# From an offline reach study (the real kinematics with JOINT_OFFSETS_DEG,
# wrist_flex held to the URDF's +-95, a 45 mm pawn):
#   * nearer than ~0.15 m (rank 1 b-g, rank 2 c-f) the grasp is reachable
#     straight down, but the pick's climb to the servo's 70 mm and its 70 mm
#     lift are not everywhere: c1 needed 25 deg of lean, d1/e1 were out of
#     reach. With the hover just clearing the object and the lift as high as
#     reaches, c1 and f1 take 5-10 deg, rank 2 none. d1/e1 stay out of reach
#     unless the wrist gets its calibrated +-101 (not used: see
#     effective_ik_limits).
#   * beyond ~0.29 m (rank 6 on) the grasp alone needs 15-50 deg, whatever the
#     path to it (coming down the gripper's axis instead of straight down
#     saved 0-5 deg) — so reach_out changes what happens at the grasp, not the
#     path: the fixed camera aligns (the servo's pixel holds near its own
#     -5 deg), and a little joint margin is kept. (It also grasps higher up
#     the object so tipped jaws clear its top — for radial jaws only; these
#     open across the arm, see JAW_AXIS.)
#
# Near the base top_down goes first where it reaches: near_base is aligned by
# the overhead camera alone, and misses when the scan's foot is off along the
# arm, where top_down's wrist servo corrects it.
NEAR_BASE_M = 0.15
FAR_REACH_M = 0.29
# The arm's own base, as an obstacle no finger or wrist camera may enter:
# radius about the pan axis. From the URDF's collision meshes: the shoulder,
# which turns to face the object, reaches 50-54 mm toward it at board height
# (the base plate reaches 74 mm, but behind and beside the axis).
BASE_KEEP_OUT_M = 0.060


# After a close on nothing that left the piece where it was, the next attempts
# aim this much further out along the arm (GripperSpec.miss_steps_along_m).
# This arm's misses close short of the piece, toward the base: 2026-10-30/31
# (pawns beyond the fixed jaw's tip, +10-11 mm along at d3), 2026-11-04 (e4,
# seen from beside the board; the wrist camera's jaw model put the jaws 7 mm
# short). A finger pad is 16 mm wide, so 5 mm steps keep the piece on a pad.
MISS_STEPS_ALONG_M = (0.005, 0.010)

# The table under everything the arm works on: no command is sent whose path
# would take the jaw tips or the wrist below it (and lower than they are) —
# SO101Env's floor guard (follower._below_floor). The table itself, whatever
# stands on it (TABLE_TOP_Z). 2026-11-17 14:27: a height look's move between
# two out-of-reach poses swung the gripper into the table; nothing checked.
FLOOR_Z = TABLE_TOP_Z


# ── Servo tuning ──────────────────────────────────────────────────────────────
# Registers re-written after lerobot's own configure() (see follower.py, _tune_servos).
# Set any of these to None to keep lerobot's value.
#
# lerobot tunes these for teleoperation — recording demonstrations by hand —
# where a responsive, springy arm is exactly what you want. Autonomous
# Cartesian servoing wants the opposite, and both defaults cost us something
# measurable.

# lerobot sets 254, the MAXIMUM, explicitly "to speedup acceleration and
# deceleration of the motors". move_ee_to streams a position goal every control
# tick, so at full acceleration the arm snaps at each one rather than flowing
# between them — the visible jerkiness. Units are 100 steps/s^2 and a step is
# 360/4096 deg, so 50 is roughly 440 deg/s^2: brisk, but a ramp rather than a
# bang. Raise it if moves feel sluggish, lower it if the arm still snaps.
SERVO_ACCELERATION  = 50

# lerobot sets P=16 (down from the firmware's 32, "to avoid shakiness"), I=0,
# D=32. The zero I is the expensive one: with no integral term, stiction or a
# constant load leaves a PERMANENT position error, which is precisely what
# check_hold.py measured — elbow_flex a flat 1.6-1.8 deg off commanded at every
# pose, unchanged after 8 s of settling. That is ~6.5 mm of Cartesian error by
# itself and is why move_ee_to stalls ~1 cm out while the solver sits at a fixed
# point commanding correct angles.
#
# A small I lets the servo wind that out on its own. Start low: too much I makes
# the arm hunt around its target instead of holding. Verify with check_hold.py,
# which reports the commanded-vs-held offset directly — if 4 does not close it,
# try 8 then 16; if the arm oscillates, go back to 0.
SERVO_P_COEFFICIENT = None   # leave lerobot's 16
SERVO_I_COEFFICIENT = 4
SERVO_D_COEFFICIENT = None   # leave lerobot's 32

# Gripper output cap, 0-1000 (Torque_Limit, a RAM register written at every
# connect). lerobot leaves it at 500, and its overload protection trips when
# output stays above 25% (Overload_Torque=25) for a moment. Closing the jaws on
# nothing drives them into their own stop at that cap, and the servo then
# latches off and drops off the bus until power-cycled. 200 stays under the
# trip point with margin and still grips a chess piece firmly. None leaves
# lerobot's.
GRIPPER_TORQUE_LIMIT = 200

# ── Safety ───────────────────────────────────────────────────────────────────────
# MEASURED with check_reach.py, not guessed, and TILT-AWARE: move_ee_to picks
# a gripper orientation per target (straight down where possible, leaning in
# where that is the only way), so the reachable set is much larger than a
# vertical-wrist-only analysis suggests.
#
# Re-run check_reach.py and update this after any lerobot-calibrate — the
# reachable set moves with the calibration.
# x_max widened from 0.40 on 2026-11-03, NOT re-measured: the board was moved
# 2 cm out (d1 and e1 out of reach before), which put f7 at 0.402 m; the
# arm's model reaches it leaning 30-40 deg. Re-run check_reach.py to settle it.
# x_max widened again to 0.48 on 2026-11-04, at the user's word that the arm
# reaches the whole board leaning almost level: rank 8 is 0.43-0.46 m out,
# reached at 60-80 deg in the arm's model (orientation._APPROACH_TILTS_DEG).
# z_max raised from TABLE_Z + 0.40 on 2026-11-04: the park pose
# (CALIBRATION_PARK_DEGREES) holds the gripper at 0.406 m, 11 mm under that,
# and swinging there from a far reach (a pick's re-find after a miss) the
# elbow and wrist arrive before the shoulder and the gripper passed 0.419 m —
# the safety monitor stopped the arm mid-swing. The overhead camera is ~0.57 m up.
WORKSPACE_BOUNDS = WorkspaceBounds(
    x_min=0.10, x_max=0.48,
    y_min=-0.25, y_max=0.25,
    z_min=TABLE_Z, z_max=TABLE_Z + 0.44,
)

# Goals streamed a tick at a time — a learned policy's action
# (SO101Env.stream_joints) — have no IK or reach check behind them, so
# each is checked before it goes: the grasp point inside WORKSPACE_BOUNDS (x, y
# and the ceiling), and the jaw tips no lower than this far below TABLE_Z. A
# little below, not at: pressing a piece up on its edge needs the tips on the
# board, and TABLE_Z is only known to a few mm.
STREAM_TIP_BELOW_TABLE_M = 0.004
# And joints a stream may not take as far as their calibrated range. The wrist
# roll meets something near -95 deg (the wrist camera's cable or body): pushed
# there it overloads and drops off the bus until power-cycled. The +80 side is
# a mirror of it, not measured: widen it once it has been.
STREAM_JOINT_LIMITS_DEG = {"wrist_roll": (-80.0, 80.0)}

# ── The sentry: nothing moves while a hand is in the way ─────────────────────
# motion_guard.MotionGuard on the overhead camera, started by SO101Env.connect
# for every program that drives the arm — the controller, the tours and checks,
# rest.py — and stopped by stop(). It freezes the arm the moment anything other
# than the arm moves over where it works, and the move carries on by itself
# once nothing has moved for 2 s. With MOTION_GUARD on, an env that
# cannot start it (no overhead camera, no camera model) refuses to connect:
# any movement of the arm runs the sentry.
#
# Where it watches: the work surface the arm reaches (WORKSPACE_BOUNDS, and back
# over its base) widened by MOTION_GUARD_MARGIN_M — the table, not the floor
# beside it, where a disc round the base takes in people's feet.
# tools/check_motion_guard.py tries it.
MOTION_GUARD = True
MOTION_GUARD_MARGIN_M = 0.06
MOTION_GUARD_ZONE = zone_around((min(WORKSPACE_BOUNDS.x_min, 0.0), WORKSPACE_BOUNDS.x_max),
                                (WORKSPACE_BOUNDS.y_min, WORKSPACE_BOUNDS.y_max), MOTION_GUARD_MARGIN_M)
# The cameras it watches through, a guard each: the overhead one.
MOTION_GUARD_CAMERAS = [OVERHEAD_CAMERA_NAME]

# ── Every camera and sensor on the rig (backends.rig.Camera, Sensor) ──────────
# The two above with jobs, and room for more: another fixed camera (a side
# view, its calibration in its entry — tools/calibrate_overhead_camera.py
# --camera NAME prints it) is opened beside them, recorded, and given its
# geometry; a sensor (a Sensor with a way to read it) is read, recorded and
# streamed, and a "force" one feeds the safety monitor's force limits.
CAMERAS = {
    name: cam for name, cam in (
        (CAMERA_NAME, Camera(mount="gripper", source=CAMERA_INDEX, size=(CAMERA_W, CAMERA_H), fps=CAMERA_FPS)),
        (OVERHEAD_CAMERA_NAME, Camera(mount="fixed", source=OVERHEAD_CAMERA_INDEX,
                                      size=(OVERHEAD_CAMERA_W, OVERHEAD_CAMERA_H), fps=OVERHEAD_CAMERA_FPS)),
    ) if name is not None
}
SENSORS: dict = {}

# ── The overhead camera's calibration, as it was last measured ───────────────
# The framework's overhead_calibration.py keeps it in this file and rewrites it
# when a scan finds the camera (or the board) has moved, or a tag tour re-fits
# it. Applied over the values above, so every tool and deployment starts from
# the latest measurement; those above are the fallback when there is no file.
OVERHEAD_CALIBRATION_PATH = RIG_DIR / "overhead_calibration.json"

# Where the jaws really close against the jaw model, by lean and reach:
# measured by tools/calibrate_grasp.py, applied by RigEnv._grasp_correction.
# No file: as modelled.
GRASP_CORRECTION_PATH = RIG_DIR / "grasp_correction.json"

from mrcf import overhead_calibration as _overhead_calibration  # noqa: E402
_overhead_calibration.PATH = OVERHEAD_CALIBRATION_PATH
_overhead_calibration.apply_to(globals())

# The wrist camera's scan shift is measured against one pose of the overhead
# camera; its self-check re-poses it when the camera or the board moves. A
# re-pose within what counts as moved (MOVED_MM, TURNED_DEG) is the same pose
# measured again: 1.5 mm and 0.18 deg, 2026-11-03.
if GRIPPER_CAM_SCAN_SHIFT_M is not None and GRIPPER_CAM_SCAN_SHIFT_AT is not None:
    _cal = _overhead_calibration.load() or {}
    if _cal.get("time") not in (None, GRIPPER_CAM_SCAN_SHIFT_AT):
        _then = next((h.get("T_arm_cam") for h in _cal.get("history", [])
                      if h.get("time") == GRIPPER_CAM_SCAN_SHIFT_AT), None)
        _mm, _deg = (_overhead_calibration.difference(_cal["T_arm_cam"], _then)
                     if _then is not None and _cal.get("T_arm_cam") is not None else (float("inf"),) * 2)
        if _mm > _overhead_calibration.MOVED_MM or _deg > _overhead_calibration.TURNED_DEG:
            _by = f" by {_mm:.1f} mm and {_deg:.2f} deg" if _then is not None else ""
            print(f"  [Calibration] The overhead camera has been re-posed{_by} ({_cal['time']}) since the "
                  f"wrist camera's calibration ({GRIPPER_CAM_SCAN_SHIFT_AT}): its picks aim a few mm less "
                  "well until it is fitted again: tools/calibrate_wrist_on_objects.py (a few objects on the table, "
                  "apart), or tools/so101/calibrate_gripper_pose.py --wide (the board clear).")

