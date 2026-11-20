"""
mrcf/backends/ur5 — a Universal Robots UR5 (CB3, PolyScope 3.15) driven over
Ethernet with ur_rtde, from mrcf/backends/_template.

    export MRCF_ARM=ur5            # run.py and tools/ drive it
    echo 192.168.x.y > rig/ur5/host   # the controller's address, once per rig (or MRCF_UR5_HOST)
    export MRCF_UR5_HOST=offline   # ... or a UR5 in memory (offline.py), no controller

    NAME              "UR5"
    config            robot_config.py: the controller's address, cameras, gripper, poses, limits
    make_env()        env.UR5Env over arm.URArm, wired from config, not yet connected
    make_policies()   ParallelJaw with the gripper's GripperSpec
    describe()        what it connects to, for the start-up line
    present()         whether its controller answers on the network (run.py's choice of arm)
    prepare_gripper(ask)
                      the Robotiq activated, asked first (tools/check_motion.py)

No tour_and_calibrate: the tag tour (tools/tag_tour.py) tours a chessboard's
squares and needs a gripper tag, and this arm runs neither. Its overhead
camera is calibrated with tools/calibrate_camera_by_hand.py (a tag on the arm,
moved in freedrive).

Setting it up on a rig, in order: COMMANDS.md, "First-time setup on another
arm" ("On the UR5"); what the rig has shown: mrcf/backends/ur5/NOTES.md.
"""

from . import robot_config as config

NAME = "UR5"

__all__ = ["NAME", "config", "make_env", "make_policies", "describe", "present", "prepare_gripper"]

_UNSET = object()


def describe() -> str:
    """What the arm connects to, for the start-up message."""
    if config.HOST is not None:
        return f"{NAME} at {config.HOST}"
    return f"{NAME} (offline, in memory" + ("" if config.HOST_SOURCE else f"; {config.NO_HOST_HINT}") + ")"


RTDE_PORT = 30004       # where ur_rtde's interfaces connect


def present(port: int = RTDE_PORT, timeout_s: float = 0.5) -> "tuple[bool, str]":
    """
    Whether the controller answers: a TCP connection to its RTDE port, opened
    and closed at once. Nothing is sent, so the controller does nothing —
    the arm is not connected to, only knocked on. No address set (or
    MRCF_UR5_HOST=offline) is not there: the arm would run in memory.

    From a process macOS keeps off the local network (VS Code's own
    terminal, sometimes) it is not there either, and nor could the arm be
    driven from it.
    """
    import socket
    if config.HOST is None:
        return False, describe()
    try:
        with socket.create_connection((config.HOST, port), timeout=timeout_s):
            pass
    except OSError as exc:
        return False, f"no answer from {config.HOST} ({exc.strerror or type(exc).__name__}: powered off, cable out?)"
    return True, f"controller answering at {config.HOST}"


def prepare_gripper(ask) -> bool:
    """
    Whether to use the gripper (tools/check_motion.py): the configured Robotiq
    activated, offering to activate it if it is not — activating runs its
    fingers fully closed, then fully open. ask(prompt) is '' to go. False: run
    without it (none configured, not reachable, or not to be activated).
    """
    if config.GRIPPER is None or config.HOST is None:
        return config.GRIPPER is not None
    import socket

    from .gripper import GripperError, RobotiqSocket

    grip = RobotiqSocket(config.HOST, max_open_m=config.GRIPPER_MAX_OPEN_M)
    try:
        grip._sock = socket.create_connection((config.HOST, RobotiqSocket.PORT), timeout=2.0)
        if grip.get("STA") == 3 and grip.get("ACT") == 1:
            return True
        print("\nThe gripper is not activated. Activating runs its fingers fully closed, then fully open:")
        print("keep your fingers and anything else out of them.")
        if ask("Activate the gripper?") != "":
            return False
        grip.activate()
        print("Activated.")
        return True
    except (OSError, GripperError) as exc:
        print(f"\nNo gripper: {exc}. The gripper step is skipped.")
        return False
    finally:
        grip.disconnect()


def _gripper_kind(gripper):
    kind = config.GRIPPER if gripper is _UNSET else gripper
    if kind not in ("robotiq", None):
        raise ValueError(f"the UR5's gripper is 'robotiq' or None, not {kind!r} (robot_config.GRIPPER)")
    return kind


def make_arm(host=_UNSET, gripper=_UNSET, rc=None):
    """
    arm.URArm for `host` (config.HOST unless given) with `gripper`
    (config.GRIPPER unless given: "robotiq", or None for none); host None:
    offline.OfflineController and OfflineGripper, already connected (an
    offline arm has nothing to open). rc: the settings to build it by — a
    deployment's rules over the arm's facts (make_env's); None: the facts alone.
    """
    from .arm import URArm
    from .gripper import RobotiqSocket

    rc = config if rc is None else rc
    host = rc.HOST if host is _UNSET else host
    kind = _gripper_kind(gripper)
    # What hangs on the flange, set by every connect (robot_config PAYLOAD_*).
    payload = dict(payload_kg=getattr(rc, "PAYLOAD_KG", None), payload_cog_m=getattr(rc, "PAYLOAD_COG_M", None))
    if host is None:
        from .offline import OfflineController, OfflineGripper, connect_offline
        hand = OfflineGripper(rc.GRIPPER_MAX_OPEN_M) if kind is not None else None
        arm = URArm(None, gripper=hand, connect_with=connect_offline(OfflineController()), **payload,
                    **rc.ARM_TUNING)
        arm.connect()
        return arm
    hand = None
    if kind == "robotiq":
        hand = RobotiqSocket(host, max_open_m=rc.GRIPPER_MAX_OPEN_M, pos_open=rc.ROBOTIQ_POS_OPEN,
                             pos_closed=rc.ROBOTIQ_POS_CLOSED, speed=rc.ROBOTIQ_SPEED, force=rc.ROBOTIQ_FORCE)
    return URArm(host, gripper=hand, **payload, **rc.ARM_TUNING)


def make_env(profile=None, **overrides):
    """
    The arm's env, wired from config, not yet connected to the controller (its
    cameras open as it is made: RealRobotEnv). profile: a deployment's rules for
    this arm (mrcf/profiles.py), laid over config. Keyword overrides win; host=
    picks another controller (None: offline, which opens no cameras unless
    sources= is given); gripper= another gripper (None: none);
    overhead_camera_width/_height set the overhead camera's capture size
    (tools/calibrate_overhead_camera.py asks for 1280x720).
    """
    from mrcf.backends import wiring

    from mrcf import profiles

    from .env import UR5Env

    rc = profiles.layered(config, profile)      # a deployment's rules for this arm over its facts (mrcf/profiles.py)
    host = overrides.pop("host", _UNSET)
    gripper = overrides.pop("gripper", _UNSET)
    kwargs = dict(
        wiring.camera_kwargs(rc, overrides),
        **wiring.rig_kwargs(rc),            # the cameras' jobs, the tag, the sentry, the jaws, the gripper's box
        control_hz=rc.CONTROL_HZ,
        speed_m_s=rc.SPEED_M_S,
        joint_speed_rad_s=rc.JOINT_SPEED_RAD_S,
        home_joints_deg=rc.HOME_JOINTS_DEG,
        park_joints_deg=rc.PARK_JOINTS_DEG,
        home_xyz=rc.HOME_XYZ,
        park_xyz=rc.PARK_XYZ,
        base_xy=rc.BASE_XY,
        jaw_axis=rc.JAW_AXIS,
        transit_height_m=rc.TRANSIT_HEIGHT_M,
        stream_fps=rc.STREAM_FPS,
        stream_bounds=rc.WORKSPACE_BOUNDS,
        robot_type=rc.ROBOT_TYPE,
        contact_force_limits=rc.CONTACT_FORCE_LIMITS,
        force_settle_s=rc.FORCE_SETTLE_S,
    )
    if (rc.HOST if host is _UNSET else host) is None and "sdk" not in overrides:
        kwargs["sources"] = {}              # the in-memory UR5 opens no hardware: no cameras unless sources= says
        kwargs["motion_guard_zone"] = None  # and so nothing for the sentry to watch with
        kwargs["force_settle_s"] = 0.0      # and its force reading does not go on changing after a move
    kwargs.update(overrides)
    sdk = kwargs.pop("sdk", None) or make_arm(host, gripper, rc=rc)
    if getattr(sdk, "host", None) is None and "unchecked_facts" not in overrides:
        kwargs["unchecked_facts"] = ()      # an arm in memory: nothing physical to be wrong about (mrcf/arm_facts.py)
    return UR5Env(sdk, **kwargs)


def make_policies(profile=None, **overrides):
    """
    ParallelJaw with this gripper's GripperSpec; with no gripper (config.GRIPPER,
    or gripper=None), policies/arm_only.ArmOnly, which moves and refuses what
    needs jaws. profile: a deployment's rules for this arm (mrcf/profiles.py),
    laid over config. Keyword overrides win.
    """
    from mrcf.backends import wiring
    from mrcf.policies.arm_only import ArmOnly
    from mrcf.policies.parallel_jaw import ParallelJaw

    kind = _gripper_kind(overrides.pop("gripper", _UNSET))
    from mrcf import profiles

    spec = wiring.gripper_spec(profiles.layered(config, profile), **overrides)
    return (ParallelJaw if kind is not None else ArmOnly)(spec)

