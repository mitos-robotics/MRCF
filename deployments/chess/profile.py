"""
Chess's own rules for its arm, the SO-101 — how chess drives it, laid over the
arm's facts (mrcf/backends/so101/robot_config.py) wherever chess makes the
arm's env and policies (mrcf/profiles.py). What is true of the arm whatever
it does stays in robot_config; what chess learned on its pieces and its board
is here, with why.

Moved out of robot_config on 2026-11-17, values unchanged: chess builds the
same env and policies it did (tests/baselines/so101/deployments.json). Only
the tabletop's rules of 2026-11-16/17, which had reached chess through the
arm's config untried, are no longer applied to it.

A new deployment does not inherit these. Copy one into its profile only where
its reason holds there too (tools/new_deployment.py lists them).
"""

# ── Among the pieces ─────────────────────────────────────────────────────────
# The widest the jaws open here: among 40 mm squares, a neighbour a square away.
# The arm opens to 80 (robot_config GRIPPER_MAX_OPEN_M).
GRIPPER_MAX_OPEN_M = 0.045

# ── Gripping a wooden piece ──────────────────────────────────────────────────
# How hard the jaws hold a piece once squeeze_gripper() has found it: the
# goal stays this many % of travel past where they stopped on it, and the
# servo presses in proportion, up to GRIPPER_TORQUE_LIMIT. 10 is the squeeze's
# own pressure, at that cap all through the carry (the cap is under the
# servo's overload trip); 3 eased it to a fraction of it.
GRIPPER_HOLD_PCT    = 10.0

# Clearance above the table for an object's FOOT when releasing it.
PLACE_RELEASE_M   = 0.005

# Sizing a grasp from a measured shape: grasp at this fraction of the height,
# never below MIN_GRASP_HEIGHT_M (the jaw tips, JAW_TIP_BELOW_GRASP_M lower,
# stay off the table and clear of a base collar).
#
# At 0.45 a 41 mm pawn is grasped at 18 mm, the jaw tips down at 11 mm on its
# flared base, and the close squeezes it out toward the arm's base. At 0.6
# (~25 mm) the tips are above the flare and the pads on the straight neck.
#
# A PREFERENCE, not the rule: what matters is that the pads close on a straight
# part, like the pawn's neck, not its flare. The policy grasps where the
# measured profile is straight over the pads' height — its width changing no
# more than GRASP_FLAT_MM there (ParallelJaw.grasp_height_for) — and of those
# heights takes the one nearest this fraction. That puts a pawn within 1 mm of
# where 0.6 alone would, and moves a king's grasp off the collar ridge at 0.6
# of its height down onto its neck.
GRASP_HEIGHT_FRACTION = 0.6

GRASP_FLAT_MM         = 2.5

MIN_GRASP_HEIGHT_M    = 0.016

# Room either side of an object's width when opening for it, and the least
# clearance to leave between a finger and a neighbouring object.
JAW_SLACK_M = 0.004

NEIGHBOUR_CLEARANCE_M = 0.003

# Opening for an object the gripper camera has not measured yet: its foot as the
# scan saw it plus slack, but at least this (a tall chess piece's).
UNKNOWN_OPEN_M = 0.038

# A neighbour whose foot nobody measured is assumed this wide (radius): wider
# than any piece on a 40 mm board, so the room check errs toward refusing.
UNMEASURED_OBSTACLE_RADIUS_M = 0.016

# Correct the hover position from the GRIPPER camera before descending.
# move_ee_to is closed-loop on forward kinematics only, so it cannot see a TCP,
# URDF or homography error — it converges confidently to the wrong place.
# Servoing measures object-vs-jaws in one image, so those cancel. Needs
# GRIPPER_CAM_GRASP_PX / GRIPPER_CAM_JACOBIAN (calibrate_gripper_servo.py) and a
# detector from the deployment. Harmless when uncalibrated: the pick falls
# back to kinematics and says so once.
VISUAL_SERVO_ENABLED = True

# How close (gripper-camera pixels) the object must come to GRIPPER_CAM_GRASP_PX
# for the servo to call it aligned. visual_servo_xy's own default, 5 px, is
# ~1.5 mm at hover: tighter than the detector's reading holds still (its blob
# shifts several px as a cream piece crosses light and dark squares) and far
# tighter than the grasp needs — 30 mm of open jaw round a ~20 mm piece leaves
# ~5 mm either way. Not much looser: aligned to 13.5 px the jaws can close on
# nothing (they stop within 2 mm of an aim that is off), where at 4.5 px they
# grip. A servo that does not converge retries the approach instead of
# descending uncorrected.
SERVO_TOL_PX = 8.0

# Measure each object with the gripper camera once over it, and take the grasp
# height and opening from that. Needs mrcf/perception/segment.py (transformers); without it, or
# if the measurement fails, a pick uses what the scan gave.
MEASURE_AT_HOVER = True

# ... and measure it there even when its shape is known. A known shape is the
# object memory's, and the memory can pair a piece with another one's track
# (one that last stood on the same square) — then the grip is planned for the
# other piece's height, too high on a short one, and the close tips it over.
# The look costs about 1.5 s a pick.
REMEASURE_KNOWN = True

# Once the gripper camera is located (GRIPPER_CAM_POSE), a pick looks at the
# object from the hover by WHERE it should be — the scan's position projected
# into the frame — rather than servoing a fixed "object appears here" pixel,
# which locks onto whatever is nearest that pixel and is only right at one
# hover height. It confirms the object is the one sent for — its foot within
# MAX_AIM_SHIFT_M of the scan's position, its height matching what was known —
# and measures it. Then the descent aims at the scan's position, or with
# AIM_AT_MEASURED_FOOT at the foot the gripper camera found — only worth it
# where the located camera is the more accurate of the two. (Unused while
# GRIPPER_CAM_POSE is None; the servo has MAX_SERVO_SHIFT_M below.)
# On: the located camera's foot moves the overhead camera's aim by where the
# piece stands against the scan (aligners.FootAligner).
AIM_AT_MEASURED_FOOT = True

# How close the descent onto an object must get to its aim. 8 mm lets an
# aligned pick stop 4 mm short — most of the slack either side of a piece.
GRASP_TOL_M = 0.004

# How far the fixed-pixel servo may move the jaws from the scan's estimate
# before the pick suspects it locked onto something else — it is also refused
# whenever that puts the jaws nearer another object's estimate. Not
# MAX_AIM_SHIFT_M: the scan can sit 20-26 mm short of the arm near the board's
# middle, and 20 mm refuses every aligned pick there.
MAX_SERVO_SHIFT_M = 0.035

MAX_AIM_SHIFT_M = 0.02

# The first attempt at a pick aligns with the wrist camera as well, not only
# by the overhead camera that has just put the jaws over the object: the
# overhead camera checks the jaws against where the object was remembered, and
# cannot see it under the gripper. True: the wrist camera looks only after a
# miss (GripperSpec.overhead_first).
OVERHEAD_FIRST = False

# With no camera to confirm the jaws over a piece, come down at the scan's
# position anyway instead of retrying the same look (GripperSpec.
# descend_unconfirmed): a pick that goes, at the risk of a knocked piece.
# 2026-11-16, for a demo: the d1 queen beside the base was never picked, its
# foot hidden from the tilted wrist camera and the gripper's tag off the top of
# the overhead camera's frame. False: retry, and give up after MAX_SKILL_RETRIES.
DESCEND_UNCONFIRMED = True

# The wrist servo within this of the scan's estimate: descend at the scan's.
# Above the servo's own bias (up to 8 mm near the base), below how far a
# placed piece can stand from where it was put (9-15 mm).
SERVO_AGREE_M = 0.008
