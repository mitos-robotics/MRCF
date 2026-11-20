"""
mrcf/backends/so101 — the SO-101 follower arm: its RobotEnv backend, its physical
configuration, and the tools for measuring and checking it.

Nothing here knows about any particular task. A deployment imports the backend
and the robot config; what the arm is being asked to DO lives in
deployments/<name>/.

It is one of mrcf/backends/'s arms (mrcf/backends/__init__.py), with the interface
every arm exports:

    NAME              "SO-101"
    config            robot_config: this rig's measured values and tuning
    make_env()        the arm (SO101Env), wired from robot_config
    make_policies()   how it picks and places: the framework's two-finger
                      policies (mrcf/policies/parallel_jaw/) with this gripper's
                      measurements and tuning
    describe()        which follower, on which port, and where its overhead calibration came from
    present()         whether the follower's USB serial port is there (run.py's choice of arm)
    tour_and_calibrate(env)
                      re-measure the overhead camera with a tag tour
"""

from mrcf.backends.so101 import robot_config as rc  # noqa: F401
from mrcf.backends.so101.cameras import resolve_camera_index  # noqa: F401
from mrcf.backends.so101.env import SO101Env  # noqa: F401
from mrcf.backends.so101.kinematics import (  # noqa: F401
    JOINT_NAMES,
    apply_solver_limits,
    calibrated_degree_limits,
    effective_ik_limits,
    straight_down_rotation,
    urdf_degree_limits,
)

NAME = "SO-101"
config = rc

__all__ = [
    "NAME", "config", "describe", "present", "tour_and_calibrate",
    "SO101Env", "make_env", "make_policies", "JOINT_NAMES", "resolve_camera_index",
    "apply_solver_limits", "calibrated_degree_limits", "urdf_degree_limits",
    "effective_ik_limits", "straight_down_rotation",
]


def describe() -> str:
    """What the arm connects to, for the start-up message — and where its overhead calibration came from."""
    from mrcf import overhead_calibration
    path = rc.OVERHEAD_CALIBRATION_PATH
    calibration = (f"overhead calibration from {path}" if overhead_calibration.load(path) is not None else
                   f"no saved overhead calibration ({path} is not there): robot_config's own values")
    return f"SO-101 follower '{rc.FOLLOWER_ID}' on {rc.FOLLOWER_PORT} — {calibration}"


def present() -> "tuple[bool, str]":
    """
    Whether the follower looks plugged in: its serial port's device is there
    (the USB cable in, the board powered from it). The port is not opened —
    a check, not a connection, and nothing on the bus hears it.
    """
    import os
    port = rc.FOLLOWER_PORT
    if os.path.exists(port):
        return True, f"follower on {port}"
    return False, f"no follower on {port} (its USB cable out? SO101_FOLLOWER_PORT names another port)"


def tour_and_calibrate(env) -> dict:
    """
    overhead_calibration.check_camera's `tour`: re-measure where the board is
    on the arm, and the overhead camera, with a tag tour (tools/tag_tour.py).
    """
    from tools import tag_tour
    return tag_tour.tour_and_calibrate(env)


def make_env(profile=None, **overrides) -> SO101Env:
    """
    An SO101Env wired from robot_config: the settings every arm shares
    (wiring.rig_kwargs: cameras' jobs and calibrations, the gripper tag, the
    sentry, the jaw model), and the SO-101's own below. Keyword overrides win.

    profile: a deployment's rules for this arm (mrcf/profiles.py), laid over
    robot_config. Pass overrides for the few tools that need to differ — e.g. a camera check
    that wants the overhead camera disabled, or find_poses wanting no URDF.
    """
    from mrcf import profiles
    from mrcf.backends import wiring

    cfg = profiles.layered(rc, profile)

    kwargs: dict = dict(
        wiring.rig_kwargs(cfg),
        port=cfg.FOLLOWER_PORT,
        robot_id=cfg.FOLLOWER_ID,
        # It opens its cameras itself (cameras.py), by index or device name.
        camera_index=cfg.CAMERA_INDEX,
        camera_device_name=cfg.CAMERA_DEVICE_NAME,
        camera_width=cfg.CAMERA_W,
        camera_height=cfg.CAMERA_H,
        camera_fps=cfg.CAMERA_FPS,
        overhead_camera_index=cfg.OVERHEAD_CAMERA_INDEX,
        overhead_camera_device_name=cfg.OVERHEAD_CAMERA_DEVICE_NAME,
        overhead_camera_width=cfg.OVERHEAD_CAMERA_W,
        overhead_camera_height=cfg.OVERHEAD_CAMERA_H,
        overhead_camera_fps=cfg.OVERHEAD_CAMERA_FPS,
        urdf_path=cfg.URDF_PATH,
        home_degrees=cfg.HOME_DEGREES,
        observe_degrees=cfg.OBSERVE_DEGREES,
        park_degrees=getattr(cfg, "CALIBRATION_PARK_DEGREES", None),
        max_relative_target_deg=cfg.MAX_RELATIVE_TARGET_DEG,
        gripper_width_table=getattr(cfg, "GRIPPER_WIDTH_TABLE", None),
        joint_offsets_deg=getattr(cfg, "JOINT_OFFSETS_DEG", None),
        ik_calibrated_range_joints=tuple(getattr(cfg, "IK_CALIBRATED_RANGE_JOINTS", ())),
        servo_acceleration=getattr(cfg, "SERVO_ACCELERATION", None),
        servo_p_coefficient=getattr(cfg, "SERVO_P_COEFFICIENT", None),
        servo_i_coefficient=getattr(cfg, "SERVO_I_COEFFICIENT", None),
        servo_d_coefficient=getattr(cfg, "SERVO_D_COEFFICIENT", None),
        gripper_torque_limit=getattr(cfg, "GRIPPER_TORQUE_LIMIT", None),
        gripper_min_hold_m=getattr(cfg, "MIN_HOLD_WIDTH_M", 0.0034),
        gripper_hold_pct=getattr(cfg, "GRIPPER_HOLD_PCT", 3.0),
        stream_bounds=cfg.WORKSPACE_BOUNDS,
        stream_tip_floor_z=cfg.TABLE_Z - getattr(cfg, "STREAM_TIP_BELOW_TABLE_M", 0.004),
        floor_z=getattr(cfg, "FLOOR_Z", None),
        descent_line_above_m=getattr(cfg, "DESCENT_LINE_ABOVE_M", None),
        floor_caps_push=bool(getattr(cfg, "FLOOR_CAPS_PUSH", False)),
        stream_joint_limits=getattr(cfg, "STREAM_JOINT_LIMITS_DEG", None),
        jaw_tip_below_m=cfg.JAW_TIP_BELOW_GRASP_M,
        jaw_finger_half_width_m=cfg.JAW_FINGER_HALF_WIDTH_M,
        jaw_axis=getattr(cfg, "JAW_AXIS", "radial"),
        observe_center_xy=cfg.OBSERVE_CENTER_XY,
        observe_height_m=cfg.OBSERVE_HEIGHT_M,
        gripper_cam_joint_deg=getattr(cfg, "GRIPPER_CAM_JOINT_DEG", None),
    )
    kwargs.update(overrides)
    return SO101Env(**kwargs)


def make_policies(profile=None, **overrides):
    """
    The SO-101's policy library: mrcf/policies/parallel_jaw/'s ParallelJaw with
    this gripper's GripperSpec from robot_config (wiring.gripper_spec), with a
    deployment's profile laid over it (profile: mrcf/profiles.py). Keyword
    overrides win.
    """
    from mrcf import profiles
    from mrcf.backends import wiring
    from mrcf.policies.parallel_jaw import ParallelJaw

    cfg = profiles.layered(rc, profile)

    # The wrist camera's learned view of the jaws (ParallelJaw.learn_jaw_view) is kept with the rig.
    jaws = rc.RIG_DIR / JAW_VIEW_FILE if getattr(rc, "CAMERA_NAME", None) else None
    return ParallelJaw(wiring.gripper_spec(cfg, **overrides), jaw_view_path=jaws)


# The wrist camera's learned view of the jaws, in the rig folder (not in git).
JAW_VIEW_FILE = "wrist_jaw_view.npz"
