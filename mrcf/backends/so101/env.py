"""
mrcf/backends/so101/env.py — RobotEnv for a real LeRobot SO-101 follower arm.

Wraps lerobot's SOFollower (serial/Feetech motor bus + camera) and, optionally,
lerobot's placo-based RobotKinematics for Cartesian control.

Two independent capability tiers:
  - Joint-space control (go_home, gripper, camera) always works — it only
    needs `pip install lerobot` and your existing LeRobot calibration file.
  - Cartesian control (get_ee_pos, move_ee_to — what pick/place skills use)
    additionally needs `pip install placo` and the SO-101 URDF + meshes.
    Run `python tools/so101/fetch_urdf.py` once to fetch them.
    Without these, get_ee_pos/move_ee_to raise NotImplementedError with a
    clear message instead of silently guessing a wrong kinematic model.

Frame convention: all positions are in the URDF's base_link frame (metres),
the same frame lerobot's RobotKinematics.forward_kinematics returns. There is
no simulator ground truth and no "world frame" independent of the robot base
— TABLE_Z and friends (mrcf/backends/so101/robot_config.py) are measured in
this frame via find_poses.py.

Cameras: the primary one (camera_name) is rigidly mounted on the gripper; an
optional second (overhead_camera_name) is fixed above the table. What follows
from where they are — camera_model(), the gripper tag's sighting, the wrist
servo, the jaw model behind get_ee_pos(), the sentry — is the shared rig layer
(mrcf/backends/rig.py, RigEnv), the same for every arm built on it; this class gives
it the gripper's pose (gripper_frame()) and drives the motors. camera_model()
gives either camera's lens and pose for measuring objects on the work surface
(plane_geometry): the overhead camera's from its calibration
(OVERHEAD_CAM_MODEL), the gripper camera's from the joint angles
(GRIPPER_CAM_POSE, measured by tools/so101/calibrate_gripper_pose.py).

pixel_to_world() is the older, coarser path the vision model's discovery
uses: a calibrated homography for the overhead camera, valid from any arm
pose; for the gripper camera a pinhole approximation valid only at the fixed
joint-space "observe pose" reached via go_observe().

Safety: every joint-space command sent through this class — from go_home(),
go_observe(), or the per-iteration output of move_ee_to()'s IK loop — is
clamped to that joint's calibrated safe range (read from the same LeRobot
calibration file find_poses.py's calibration step produced). lerobot's own
DEGREES normalisation does NOT clamp to this range on write, so an
unconstrained IK solution can otherwise ask for an angle the servo will
physically refuse, silently stalling instead of reaching the target.

The class is assembled from parts, one file each:

    follower.py     the follower: its bus, servos, joints, limits, outline
    kinematics.py   the kinematic model, apart from the arm (functions, data)
    gripper.py      opening, closing, holding
    orientation.py  which way the gripper points to reach a target
    motion.py       move_ee_to, joint-space moves, settling, named poses
    cameras.py      finding, listing and reading cameras; pixel_to_world
    stream.py       streamed goals, for learned policies (demos.py: the stream)
"""

from __future__ import annotations

from mrcf.backends.so101.cameras import CamerasMixin
from mrcf.backends.so101.gripper import GripperMixin
from mrcf.backends.so101.motion import MotionMixin
from mrcf.backends.so101.stream import StreamMixin


class SO101Env(StreamMixin, CamerasMixin, MotionMixin, GripperMixin):
    """Real-hardware RobotEnv for one SO-101 follower arm with a gripper-mounted camera."""
