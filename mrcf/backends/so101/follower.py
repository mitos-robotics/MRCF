"""
mrcf/backends/so101/follower.py — the SO-101 follower itself: building it from
robot_config's values, connecting to its motor bus and tuning its servos,
sending and reading joint angles (every command clamped to the joint's
calibrated range), holding still, and the arm's outline for the motion guard.

Follower is the base the other parts of SO101Env (env.py) build on; use
SO101Env.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from mrcf import flight_recorder
from mrcf.backends.rig import RigEnv
from mrcf.backends.so101.kinematics import (
    _TCP_OFFSET,
    JOINT_NAMES,
    OffsetKinematics,
    apply_solver_limits,
    calibrated_degree_limits,
    effective_ik_limits,
    straight_down_rotation,
    urdf_degree_limits,
    within_limits,
)

# The arm's body for arm_shape(), base to tip: URDF frames, each with a radius
# that takes in what is around it — motor bodies, brackets, cables, the wrist
# camera. Generous on purpose: what it leaves out of the arm, a motion guard
# (motion_guard.py) takes for a hand.
_ARM_SHAPE = (
    ("base_link", 0.05),
    ("shoulder_link", 0.05),
    ("upper_arm_link", 0.04),
    ("lower_arm_link", 0.04),
    ("wrist_link", 0.045),
    ("gripper_link", 0.05),
    ("gripper_frame_link", 0.04),
)
# Then the moving jaw's tip, at the jaws' opening: the jaw, and whatever is
# held between them.
_JAW_TIP_RADIUS_M = 0.035
# A trip holds the arm at once (RobotEnv.may_be_moving) only within this long
# of its last command: by then any move has finished.
_MOVING_AFTER_COMMAND_S = 1.0


# The floor guard lets the model's gripper this far into the table: its corners
# leaning, the kinematics a millimetre or two out — a grip on a scoop's wall, its
# jaws 16 mm up, was refused for 1 mm (2026-11-17). What it is for goes in by
# centimetres.
_FLOOR_SLACK_M = 0.003


class FloorRefused(RuntimeError):
    """A command not sent: it would take the gripper down into the table (Follower.floor_z)."""


class Follower(RigEnv):
    """The follower's bus, joints and state: what every other part of SO101Env reads."""

    SERVO_CALIBRATION_TOOL = "tools/so101/calibrate_gripper_servo.py"

    # goto_joint_degrees (motion.py) lets go of a joint still more than
    # STALL_FAR_DEG off that has moved under STALL_MOVE_DEG in the last STALL_S
    # (see there); a stream (stream.py) holds such a joint where it stopped.
    STALL_S = 1.0
    STALL_FAR_DEG = 10.0
    STALL_MOVE_DEG = 1.0

    def __init__(
        self,
        port: str,
        robot_id: str,
        # The wrist camera (None: none — the policies then align with the
        # overhead camera alone).
        camera_name: "str | None" = "front",
        camera_index: "int | str" = 0,
        # If set, resolved to the device's CURRENT index via resolve_camera_index()
        # at construction time instead of trusting camera_index directly — macOS
        # AVFoundation indices aren't stable across process launches. camera_index
        # is still used as the fallback if resolution fails.
        camera_device_name: "str | None" = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        urdf_path: "str | Path | None" = None,
        home_degrees: "dict[str, float] | None" = None,
        observe_degrees: "dict[str, float] | None" = None,
        # Where go_park() takes the arm: out of the overhead camera's view
        # (robot_config CALIBRATION_PARK_DEGREES). None: home.
        park_degrees: "dict[str, float] | None" = None,
        max_relative_target_deg: float = 8.0,
        control_hz: float = 20.0,
        gripper_max_open_m: float = 0.045,
        # [(motor %, opening m), ...] measured by tools/so101/calibrate_gripper_width.py;
        # None: a straight line from 0 to gripper_max_open_m.
        gripper_width_table: "list | None" = None,
        # {joint: degrees} added to the motors' angles before the kinematic
        # model sees them (tools/so101/calibrate_arm_offsets.py). None: as reported.
        joint_offsets_deg: "dict | None" = None,
        # Joints IK may take over their whole calibrated range rather than
        # stopping at the URDF's (robot_config.IK_CALIBRATED_RANGE_JOINTS).
        ik_calibrated_range_joints: tuple = (),
        # The sentry (motion_guard.MotionGuard on the overhead camera): where it
        # watches, an (x, y) polygon on the work surface. Not None: connect()
        # starts it, and refuses to connect when it cannot (robot_config
        # MOTION_GUARD). None: no sentry (a tool that runs its own).
        motion_guard_zone=None,
        # How far the jaws' centre moves per metre of opening (JAW_CENTRE_SLIDE),
        # and which way, in gripper_frame_link (JAW_SLIDE_AXIS: toward the moving
        # jaw). 0.5 is the midpoint of one moving jaw.
        jaw_centre_slide: float = 0.5,
        jaw_slide_axis: "tuple | np.ndarray" = (-1.0, 0.0, 0.0),
        # The grasp facts not yet checked on the rig (mrcf/arm_facts.py; the
        # SO-101's jaws are, JAW_SLIDE_AXIS_CHECKED): picks that need one refused.
        unchecked_facts: tuple = (),
        # Servo registers re-tuned after lerobot's configure() — see
        # _tune_servos. None leaves lerobot's value in place.
        servo_acceleration: "int | None" = None,
        servo_p_coefficient: "int | None" = None,
        servo_i_coefficient: "int | None" = None,
        servo_d_coefficient: "int | None" = None,
        gripper_torque_limit: "int | None" = None,
        # The narrowest the jaws stop on a piece; narrower is shut on nothing,
        # whatever the load says (see _min_hold_m). robot_config.MIN_HOLD_WIDTH_M.
        gripper_min_hold_m: float = 0.0034,
        # squeeze_gripper's hold: the goal this many % past a held piece.
        # robot_config.GRIPPER_HOLD_PCT.
        gripper_hold_pct: float = 3.0,
        # What stream_joints checks each goal against: the grasp point inside
        # these bounds (x, y, ceiling) and the jaw tips above this height. None:
        # not checked. The jaw tips' reach past the grasp point along the
        # approach axis and each finger's half-width across the opening, for
        # finding the tips (robot_config JAW_TIP_BELOW_GRASP_M, JAW_FINGER_HALF_WIDTH_M).
        stream_bounds=None,
        stream_tip_floor_z: "float | None" = None,
        # Every command's path: the jaw tips and the wrist kept above this height
        # (the table under everything the arm works on). A command that would
        # take them lower, and lower than they are, is not sent (FloorRefused).
        # None: not checked. robot_config FLOOR_Z.
        floor_z: "float | None" = None,
        # A held-orientation descent's upper part one line in joint space, to
        # the solver's own answer this far over its goal; the rest the
        # Cartesian loop's (motion._descend_by_joint_line). None: all of it
        # the loop's. robot_config / a deployment's DESCENT_LINE_ABOVE_M.
        descent_line_above_m: "float | None" = None,
        # A command pushed past the solver's step (a stuck joint, a settle)
        # that the floor guard would refuse sent as the step itself
        # (motion._floor_kept). False: pushed, and refused. FLOOR_CAPS_PUSH.
        floor_caps_push: bool = False,
        # {joint: (lo, hi)} degrees a stream may not take a joint past, inside
        # its calibrated range (robot_config STREAM_JOINT_LIMITS_DEG).
        stream_joint_limits: "dict | None" = None,
        jaw_tip_below_m: float = 0.007,
        jaw_finger_half_width_m: float = 0.008,
        # Grasp point with the jaws CLOSED, in gripper_frame_link coords; the
        # open-jaw grasp point is derived from it per opening by _tcp(). See
        # _TCP_OFFSET. Override if you fit a different gripper.
        tcp_offset: "np.ndarray | None" = None,
        # Where the jaws really close against that model, by lean and reach
        # (grasp_correction.GraspCorrection; robot_config.GRASP_CORRECTION_PATH).
        grasp_correction=None,
        # Which way the jaws open relative to the arm: "radial" (along it) or
        # "tangential" (across it). The wrist's roll that gives it comes from
        # the gripper axis they open along (jaw_slide_axis). See
        # robot_config.JAW_AXIS.
        jaw_axis: str = "radial",
        observe_center_xy: "tuple[float, float]" = (0.20, 0.0),
        observe_height_m: float = 0.25,
        disable_torque_on_disconnect: bool = False,
        overhead_camera_name: "str | None" = None,
        overhead_camera_index: "int | str | None" = None,
        overhead_camera_device_name: "str | None" = None,
        # Required (non-None) whenever overhead_camera_name is set — lerobot's
        # RobotConfig rejects None here outright for any camera attached to a
        # robot. Must be a resolution/FPS this specific camera's driver actually
        # honors; cheap USB webcams often silently snap an unsupported request to
        # a different mode (set()/get() can even both falsely report success —
        # see mrcf/backends/so101/robot_config.py's OVERHEAD_CAMERA_W comment
        # for how to probe a camera's real supported modes directly via cv2).
        # render_image() resizes in software regardless, so this doesn't need to
        # match the resolution the rest of the pipeline actually sees.
        overhead_camera_width: "int | None" = None,
        overhead_camera_height: "int | None" = None,
        overhead_camera_fps: "int | None" = None,
        overhead_homography: "list | np.ndarray | None" = None,
        # Resolution the pipeline reasons in; render_image() resizes to it.
        render_w: int = 640,
        render_h: int = 480,
        # Height of the work surface in the base frame, used to turn the
        # gripper's height above it into a pixel scale for visual servoing.
        work_plane_z: float = 0.0,
        # Gripper-camera visual servoing — all three measured together by
        # tools/so101/calibrate_gripper_servo.py. None disables servoing.
        gripper_cam_grasp_px: "tuple[float, float] | None" = None,
        gripper_cam_jacobian: "list | np.ndarray | None" = None,
        gripper_cam_ref_height_m: "float | None" = None,
        # Where the servo's calibration was taken, so it can aim right at any
        # other tilt (see _servo_tilt_offset): the gripper's tilt then
        # (tilted_rotation's degrees), the height of the point the detector
        # reports on the object it used, and the camera's centre in
        # gripper_frame_link (approximate is enough). Tilt None: aims as
        # calibrated whatever the tilt.
        gripper_cam_ref_tilt_deg: "float | None" = None,
        gripper_cam_ref_point_height_m: float = 0.0,
        gripper_cam_position_m: "tuple | np.ndarray | None" = None,
        # Camera models for measuring on the work surface — see camera_model()
        # and robot_config's "Camera models" section. None leaves a camera's
        # model unknown.
        overhead_cam_model: "dict | None" = None,
        gripper_cam_k: "list | np.ndarray | None" = None,
        gripper_cam_dist: "list | np.ndarray | None" = None,
        gripper_cam_pose: "list | np.ndarray | None" = None,
        gripper_cam_plane_z_m: "float | None" = None,
        # Corrections to the joint readings, degrees by joint name, fitted with
        # gripper_cam_pose (GRIPPER_CAM_JOINT_DEG): for the wrist camera's pose only.
        gripper_cam_joint_deg: "dict | None" = None,
        gripper_cam_scan_shift_m=None,
        arm_frame_scale: float = 1.0,
        # A tag on the gripper the overhead camera can see (overhead_gripper_offset):
        # its id, its pose on the gripper (gripper_frame_link -> tag, 4x4 m) and
        # the size it reads at. None: no such tag.
        gripper_tag_id: "int | None" = None,
        gripper_tag_pose: "list | np.ndarray | None" = None,
        gripper_tag_size_m: "float | None" = None,
        # Every camera and sensor on the rig (robot_config CAMERAS, SENSORS:
        # backends.rig.Camera, Sensor). Cameras other than the two above are
        # opened by LeRobot beside them, from their Camera's source and size.
        cameras: "dict | None" = None,
        sensors: "dict | None" = None,
        # The cameras the sentry watches through (None: the overhead camera).
        motion_guard_cameras: "list | None" = None,
    ):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        from mrcf.backends.so101.cameras import resolve_camera_index   # cameras.py builds on this module

        if camera_device_name is not None:
            camera_index = resolve_camera_index(camera_device_name, fallback=camera_index)
        if overhead_camera_name is not None and overhead_camera_device_name is not None:
            overhead_camera_index = resolve_camera_index(overhead_camera_device_name, fallback=overhead_camera_index)

        # The cameras and the gripper's geometry (mrcf/backends/rig.py). The fixed
        # external camera (not gripper-mounted) is optional; unlike the gripper
        # camera, its pixel->world mapping doesn't depend on arm pose, so it
        # uses a real calibrated homography instead of the pinhole
        # approximation (see pixel_to_world below and calibrate_overhead_camera.py).
        self._init_rig(
            camera_name=camera_name, render_w=render_w, render_h=render_h, work_plane_z=work_plane_z,
            overhead_camera_name=overhead_camera_name, overhead_homography=overhead_homography,
            overhead_cam_model=overhead_cam_model, gripper_cam_k=gripper_cam_k,
            gripper_cam_dist=gripper_cam_dist, gripper_cam_pose=gripper_cam_pose,
            gripper_cam_plane_z_m=gripper_cam_plane_z_m, gripper_cam_scan_shift_m=gripper_cam_scan_shift_m,
            arm_frame_scale=arm_frame_scale,
            gripper_cam_grasp_px=gripper_cam_grasp_px, gripper_cam_jacobian=gripper_cam_jacobian,
            gripper_cam_ref_height_m=gripper_cam_ref_height_m, gripper_cam_ref_tilt_deg=gripper_cam_ref_tilt_deg,
            gripper_cam_ref_point_height_m=gripper_cam_ref_point_height_m,
            gripper_cam_position_m=gripper_cam_position_m, gripper_tag_id=gripper_tag_id,
            gripper_tag_pose=gripper_tag_pose, gripper_tag_size_m=gripper_tag_size_m,
            motion_guard_zone=motion_guard_zone, motion_guard_cameras=motion_guard_cameras,
            cameras=cameras, sensors=sensors,
            tcp_offset=tcp_offset if tcp_offset is not None else _TCP_OFFSET.copy(),
            jaw_centre_slide=jaw_centre_slide, jaw_slide_axis=jaw_slide_axis,
            gripper_max_open_m=gripper_max_open_m, grasp_correction=grasp_correction,
            unchecked_facts=unchecked_facts)
        self._gripper_cam_joint_deg = (None if not gripper_cam_joint_deg else
                                       np.array([float(gripper_cam_joint_deg.get(n, 0.0)) for n in JOINT_NAMES]))
        self._control_hz  = control_hz
        self._max_relative_target_deg = max_relative_target_deg
        self._running      = False
        # One thread drives the arm, but perception and safety watchers read
        # joints (a wrist camera's pose) from others; the serial bus takes one
        # transaction at a time. Re-entrant, so a caller already holding it can
        # still read the joints.
        self._bus_lock = threading.RLock()
        self._gripper_width_table = sorted(gripper_width_table) if gripper_width_table else None
        self._servo_acceleration  = servo_acceleration
        self._servo_p_coefficient = servo_p_coefficient
        self._servo_i_coefficient = servo_i_coefficient
        self._servo_d_coefficient = servo_d_coefficient
        self._gripper_torque_limit = gripper_torque_limit
        self._gripper_min_hold_m = float(gripper_min_hold_m)
        self._gripper_hold_pct = float(gripper_hold_pct)
        self._stream_bounds = stream_bounds
        self._stream_tip_floor_z = stream_tip_floor_z
        self._floor_z = None if floor_z is None else float(floor_z)
        self._descent_line_above_m = None if descent_line_above_m is None else float(descent_line_above_m)
        self._floor_caps_push = bool(floor_caps_push)
        self._stream_joint_limits = dict(stream_joint_limits or {})
        self._jaw_tip_below_m = float(jaw_tip_below_m)
        self._jaw_finger_half_width_m = float(jaw_finger_half_width_m)
        if jaw_axis not in ("radial", "tangential"):
            raise ValueError(f"jaw_axis must be 'radial' or 'tangential', not {jaw_axis!r}")
        self._jaw_axis = jaw_axis
        # "Active" calibration used by pixel_to_world() — swapped out by
        # go_to_look_around_pose() and restored by go_observe(). The
        # "_primary_*" copies are the reset target and never change.
        self._observe_center_xy = self._primary_observe_center_xy = observe_center_xy
        self._observe_height_m  = self._primary_observe_height_m  = observe_height_m
        self._home_degrees = dict(home_degrees) if home_degrees else {
            "shoulder_pan": 0.0, "shoulder_lift": -90.0, "elbow_flex": 90.0,
            "wrist_flex": 0.0, "wrist_roll": 0.0,
        }
        self._observe_degrees = dict(observe_degrees) if observe_degrees else None
        self._park_degrees = dict(park_degrees) if park_degrees else None
        # The wrist camera's frames as a stream gives them (stream_spec): at its capture size.
        self._camera_capture_wh = (camera_width, camera_height)

        opened = {}
        if camera_name is not None:
            opened[camera_name] = OpenCVCameraConfig(
                index_or_path=camera_index,
                width=camera_width, height=camera_height, fps=camera_fps,
            )
        if overhead_camera_name is not None:
            if overhead_camera_index is None:
                raise ValueError("overhead_camera_index is required when overhead_camera_name is set")
            opened[overhead_camera_name] = OpenCVCameraConfig(
                index_or_path=overhead_camera_index,
                width=overhead_camera_width, height=overhead_camera_height, fps=overhead_camera_fps,
            )
        # Any other camera on the rig, as its Camera entry says.
        for name, cam in (cameras or {}).items():
            if name in opened or cam.source is None:
                continue
            opened[name] = OpenCVCameraConfig(index_or_path=cam.source, width=cam.size[0], height=cam.size[1],
                                              fps=cam.fps)

        config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            max_relative_target=max_relative_target_deg,
            # lerobot's default (True) cuts servo holding torque immediately on
            # disconnect. SO-101 isn't self-locking, so that drops the arm from
            # wherever it is — including straight after go_home() in our own
            # shutdown path. Default to holding position instead; call
            # disable_torque() explicitly (as find_poses.py does) when you want
            # to move the arm by hand.
            disable_torque_on_disconnect=disable_torque_on_disconnect,
            cameras=opened,
        )
        self._robot = SO101Follower(config)

        # Hard per-joint degree limits, derived from the SAME calibration file
        # lerobot loaded (self._robot.calibration) — these are the physical
        # Min/Max_Position_Limit values the arm was calibrated with, which
        # write_calibration() already pushed into the servos' own firmware.
        # Nothing in lerobot's DEGREES normalisation clamps to these on write
        # (unlike its 0-100%/±100% modes), so an unconstrained IK solution can
        # ask for an angle the servo will physically refuse — it just sits
        # there stalled against its own limit. Clamp before ever sending.
        self._joint_deg_limits = calibrated_degree_limits(self._robot.calibration)
        self._setup_kinematics(urdf_path, joint_offsets_deg, ik_calibrated_range_joints)

    def _setup_kinematics(self, urdf_path, joint_offsets_deg: "dict | None", beyond_urdf=()) -> None:
        """
        The kinematic model and the limits its solver works to, from the URDF,
        the joint corrections (JOINT_OFFSETS_DEG) and self._joint_deg_limits.
        Its own method so an offline arm (tools/so101/visualise_ik.offline_arm, the
        reach maps) is built by the same code as the live one, joint
        corrections and all: without them it maps reach for a different arm.
        """
        # Kinematics are optional — only needed for Cartesian control.
        self._kinematics = None
        self._r_down: "np.ndarray | None" = None
        # For arm_shape(), which builds a model of its own from these.
        self._urdf_path = urdf_path
        self._joint_offsets_deg = None
        self._pan_centre = np.zeros(2)
        # Reachability-check limits (see effective_ik_limits). Falls back to the
        # calibration alone when there is no URDF to intersect with. Inside the
        # stream limits too: those are things a joint meets before its
        # calibrated stop (the wrist roll, the camera's cable near -95 deg),
        # whatever moves it. 2026-11-03: a stand-up planned the roll at -88
        # deg, the wrist stopped 8 deg short against the cable, and the arm 4
        # cm short of its hover.
        self._ik_limits = within_limits(dict(self._joint_deg_limits), self.__dict__.get("_stream_joint_limits"))
        urdf_path = Path(urdf_path) if urdf_path else None
        if urdf_path is not None and urdf_path.is_file():
            try:
                from lerobot.model.kinematics import RobotKinematics
                raw = RobotKinematics(str(urdf_path), target_frame_name="gripper_frame_link")
                # The uncorrected model, for calibrate_arm_offsets.py (which fits the corrections).
                self._kinematics_raw = raw
                offsets = np.zeros(len(raw.joint_names))
                for name, deg in (joint_offsets_deg or {}).items():
                    offsets[JOINT_NAMES.index(name)] = float(deg)
                self._kinematics = OffsetKinematics(raw, offsets) if offsets.any() else raw
                self._joint_offsets_deg = offsets
                # A property of the model's own angles, so of the uncorrected
                # model: taken through the offsets it is 12.7 deg from vertical.
                self._r_down = straight_down_rotation(raw)
                self._pan_centre = self._find_pan_centre()
                self._ik_limits = within_limits(effective_ik_limits(
                    self._joint_deg_limits, urdf_degree_limits(urdf_path), beyond_urdf), self.__dict__.get("_stream_joint_limits"))
                # The solver works in corrected angles, so its limits move with them.
                apply_solver_limits(raw, {n: (lo + offsets[JOINT_NAMES.index(n)], hi + offsets[JOINT_NAMES.index(n)])
                                          if n in JOINT_NAMES else (lo, hi)
                                          for n, (lo, hi) in self._ik_limits.items()})
            except Exception as exc:
                # Clear both: failures above can leave _kinematics usable but
                # _r_down wrong (the approach-axis guard is exactly that case),
                # and a half-initialised solver would silently go on driving
                # move_ee_to with a bad orientation while this prints that
                # Cartesian control is off.
                self._kinematics = None
                self._r_down = None
                print(f"  [SO101Env] Kinematics unavailable ({exc}).")
                print("  [SO101Env] go_home()/gripper/camera still work; "
                      "move_ee_to()-based skills (pick/place) will raise NotImplementedError.")
        elif urdf_path is not None:
            print(f"  [SO101Env] URDF not found at {urdf_path} — run fetch_urdf.py for Cartesian control.")

    # ── Connection lifecycle ───────────────────────────────────────────────────
    # RigEnv starts the sentry as connect() returns (and disconnects again if
    # it cannot), and stops it before stop() lets go of the bus.

    # A camera that sends no frame within lerobot's warmup at connect is closed
    # and opened again, this many times more, before connect gives up.
    CONNECT_CAMERA_RETRIES = 2

    def connect(self) -> None:
        self._refuse_personal_cameras()
        try:
            self._robot.connect(calibrate=False)  # use the existing LeRobot calibration file as-is
        except (TimeoutError, ConnectionError) as exc:
            # The bus is open; a camera opened but sent nothing in time
            # (2026-11-03 and 11-04, the wrist camera, at start-up; a fresh
            # process read it at once). Open the cameras that are not
            # reading again, then finish what lerobot's connect does after them.
            if not self._robot.bus.is_connected:
                raise
            self._connect_cameras_again(exc)
            self._robot.configure()
        self._tune_servos()
        self._running = True

    def _refuse_personal_cameras(self) -> None:
        """No camera opened that is, or may now be, this computer's own (mrcf/devices.check_sources)."""
        from mrcf import devices

        cameras = getattr(getattr(self._robot, "config", None), "cameras", None) or {}
        devices.check_sources({name: getattr(cam, "index_or_path", None) for name, cam in cameras.items()})

    def _connect_cameras_again(self, first: Exception) -> None:
        """Every camera not yet reading, closed and opened again, up to CONNECT_CAMERA_RETRIES times."""
        last: Exception = first
        for attempt in range(1, self.CONNECT_CAMERA_RETRIES + 1):
            pending = [(n, c) for n, c in self._robot.cameras.items()
                       if not c.is_connected or getattr(c, "latest_frame", True) is None]
            if not pending:
                return
            for name, cam in pending:
                state = "sent nothing at connect" if cam.is_connected else "not open yet (an earlier camera failed)"
                print(f"  [camera] '{name}' {state} — opening it "
                      f"({attempt}/{self.CONNECT_CAMERA_RETRIES}; the failure was: {last}).")
                try:
                    cam.disconnect()
                except Exception:          # noqa: BLE001 — half-open: connect() below is what matters
                    pass
                time.sleep(0.5)
                try:
                    cam.connect()
                except (TimeoutError, ConnectionError) as exc:
                    last = exc
        pending = [n for n, c in self._robot.cameras.items() if not c.is_connected or getattr(c, "latest_frame", True) is None]
        if not pending:
            return
        # The arm can run without a camera nothing needs to see by: the wrist camera
        # (the pick then aims by the overhead camera alone, ParallelJaw.wrist_sending).
        # 2026-11-14 it hung after a USB drop and every restart failed here until it was
        # replugged. Not the overhead camera or one the sentry watches through.
        needed = {self._overhead_camera_name, *self._sentry_cameras()} - {None}
        if needed & set(pending):
            raise last
        closed = self.__dict__.setdefault("_closed_cameras", {})
        for name in pending:
            closed[name] = self._robot.cameras.pop(name)
            print(f"  [camera] WARNING — '{name}' sends nothing ({last}); carrying on without it. It is "
                  "opened again whenever it is asked for: unplug it and plug it back in.")

    def _tune_servos(self) -> None:
        """
        Re-tune servo registers that lerobot sets for teleoperation.

        lerobot's defaults are deliberate and good for recording demonstrations
        by hand, but both of them hurt autonomous Cartesian servoing:

        `Acceleration = 254` — the maximum. `FeetechMotorsBus.configure_motors()`
        sets it "to speedup acceleration and deceleration of the motors", which
        makes the arm feel responsive to a human but means every position goal
        is chased bang-bang. move_ee_to streams a goal per control tick, so at
        full acceleration the arm snaps at each one instead of flowing between
        them. Lower is gentler. The register is in units of 100 steps/s^2 and a
        step is 360/4096 deg, so 50 is roughly 440 deg/s^2.

        `I_Coefficient = 0` — no integral term (`SOFollower.configure()` sets
        P=16, I=0, D=32). With zero I, any constant load or stiction leaves a
        PERMANENT position error, which is exactly the symptom check_hold.py
        measured: elbow_flex sitting a flat 1.6-1.8 deg off commanded at every
        pose, unchanged after 8 s of settling. That is ~6.5 mm of Cartesian
        error on its own and is why move_ee_to can stall ~1 cm out while the
        solver is at a fixed point commanding the right angles. A small I lets
        the servo wind that out itself.

        Each value is None-able in robot_config to leave lerobot's alone.
        Verify a change with check_hold.py, which measures the offset directly.
        """
        regs = (("Acceleration", self._servo_acceleration),
                ("P_Coefficient", self._servo_p_coefficient),
                ("I_Coefficient", self._servo_i_coefficient),
                ("D_Coefficient", self._servo_d_coefficient))
        todo = [(r, v) for r, v in regs if v is not None]
        if not todo:
            self._limit_gripper_torque()
            return
        bus = self._robot.bus
        # Torque off to write, exactly as lerobot's own configure() does: the
        # PID coefficients live at addresses 21-23, in the servo's EEPROM
        # region, which will not accept a write while the motor is driven. This
        # is safe here for the same reason it is safe there — SOFollower.connect
        # documents that the arm is assumed to be at rest at connection time,
        # and configure() has just finished its own torque_disabled block.
        try:
            with self._bus_lock, bus.torque_disabled():
                for reg, value in todo:
                    for motor in bus.motors:
                        bus.write(reg, motor, int(value))
        except Exception as exc:
            # Not fatal: a servo that rejects a tuning register still moves on
            # lerobot's defaults. Silence would be worse — it would look like
            # the tuning took effect.
            print(f"  [SO101Env] Servo tuning failed ({exc}) — "
                  f"running on lerobot's defaults "
                  f"(Acceleration=254, P=16, I=0, D=32).")
            return
        print("  [SO101Env] Servo tuning: "
              + ", ".join(f"{r}={v}" for r, v in todo))
        self._limit_gripper_torque()

    def _limit_gripper_torque(self) -> None:
        """Cap gripper output below its overload trip point (see GRIPPER_TORQUE_LIMIT)."""
        if self._gripper_torque_limit is None:
            return
        try:
            # RAM register: writable with torque on, and reset at power-up.
            with self._bus_lock:
                self._robot.bus.write("Torque_Limit", "gripper", int(self._gripper_torque_limit))
            print(f"  [SO101Env] Gripper torque limited to {self._gripper_torque_limit}/1000")
        except Exception as exc:
            print(f"  [SO101Env] Could not limit gripper torque ({exc}) — "
                  "closing on nothing may trip its overload protection.")

    def stop(self) -> None:
        if self._running:
            with self._bus_lock:
                try:
                    self._robot.disconnect()
                except Exception as exc:
                    # lerobot refuses while any camera of its is closed (is_connected):
                    # the bus is closed all the same, its torque as configured.
                    print(f"  [SO101Env] Disconnecting piece by piece ({exc}).")
                    bus = self._robot.bus
                    if bus.is_connected:
                        bus.disconnect(getattr(getattr(self._robot, "config", None),
                                               "disable_torque_on_disconnect", False))
                    for cam in self._robot.cameras.values():
                        try:
                            cam.disconnect()
                        except Exception:
                            pass
            # A camera closed for a reopen that never came back (cameras._reopen).
            for cam in self.__dict__.pop("_closed_cameras", {}).values():
                try:
                    cam.disconnect()
                except Exception:
                    pass
        self._running = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_exc):
        self.stop()

    def is_viewer_running(self) -> bool:
        return self._running

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _send(self, action: dict) -> dict:
        """
        Every motor command goes through here: guard() first, so a tripped
        interlock stops the arm before the command is sent (RobotEnv: Stopping),
        then the bus, under its lock.
        """
        why = self._below_floor(action)
        if why is not None:
            raise FloorRefused(why)
        while True:
            # Not under the bus lock: a pause waits in guard(), and the sentry
            # that ends it reads the joints (arm_shape) through that lock — held
            # here, it could never see the scene go quiet, and the pause never ended.
            self.guard()
            # Checked again under the lock: a trip from another thread holds the
            # arm under it as well (RobotEnv._freeze), so its hold cannot be
            # overtaken by a command that got past guard() just before the trip.
            with self._bus_lock:
                if not self.interlock.tripped():
                    sent = self._robot.send_action(action)
                    self._last_command_t = time.monotonic()
                    self.__dict__.pop("_pos_cache", None)
                    flight_recorder.sample("sent", {k.removesuffix(".pos"): v for k, v in action.items()})
                    return sent

    def _lowest_z(self, q_arm_deg, gap_m: float) -> float:
        """The gripper's lowest point (either jaw's tip, at either edge of its finger, and the wrist end), world z, at q_arm_deg."""
        assert self._kinematics is not None
        t = self._kinematics.forward_kinematics(np.append(np.asarray(q_arm_deg, float), 0.0))
        rot, origin = t[:3, :3], t[:3, 3]
        slide = self._jaw_slide_axis
        across = np.cross([0.0, 0.0, 1.0], slide) * self._jaw_finger_half_width_m
        reach = np.array([0.0, 0.0, self._jaw_tip_below_m])
        tips = min(float((origin + rot @ (pad + reach + side * across))[2])
                   for pad in (self._tcp_offset, self._tcp_offset + slide * gap_m) for side in (-1.0, 1.0))
        return min(tips, float(origin[2]))

    def below_grasp_m(self, rot, gap_m: float) -> float:
        """capabilities.GripperBelowGrasp: _lowest_z's points (each jaw's tip, either edge, the wrist end) under the grasp point."""
        rot = np.asarray(rot, float)
        slide = self._jaw_slide_axis
        across = np.cross([0.0, 0.0, 1.0], slide) * self._jaw_finger_half_width_m
        reach = np.array([0.0, 0.0, self._jaw_tip_below_m])
        gap = float(np.clip(gap_m, 0.0, self._gripper_max_open_m))
        grasp = self._tcp(gap)
        points = [pad + reach + side * across for pad in (self._tcp_offset, self._tcp_offset + slide * gap)
                  for side in (-1.0, 1.0)] + [np.zeros(3)]
        return max(0.0, max(float((rot @ (grasp - p))[2]) for p in points))

    def _below_floor(self, action: dict) -> "str | None":
        """
        Why `action` must not be sent: along the joints' straight path from
        where they are to it, the gripper would go below floor_z — and lower
        than it is now (a move up out of a low spot is let through). None: it
        may be sent, or there is no floor to keep to (floor_z None), or the
        action does not move the arm.
        """
        floor = getattr(self, "_floor_z", None)
        if floor is None or getattr(self, "_kinematics", None) is None or not all(f"{n}.pos" in action for n in JOINT_NAMES):
            return None
        q1 = np.array([float(action[f"{n}.pos"]) for n in JOINT_NAMES])
        pos = self._motor_pos()
        q0 = np.array([float(pos[n]) for n in JOINT_NAMES])
        pct = float(action.get("gripper.pos", pos["gripper"]))
        gap = float(np.clip(self._pct_to_width(pct), 0.0, self._gripper_max_open_m))
        steps = int(np.clip(np.max(np.abs(q1 - q0)) / 2.0, 1, 90))
        now = self._lowest_z(q0, gap)
        low = min([now] + [self._lowest_z(q0 + f * (q1 - q0), gap) for f in np.linspace(0.0, 1.0, steps + 1)[1:]])
        if low >= floor - _FLOOR_SLACK_M or low >= now - 0.001:
            return None
        return (f"the gripper would go {(floor - low) * 1000:.0f} mm below the table "
                f"(z {low * 1000:+.0f} mm, the table {floor * 1000:+.0f}) on the way to "
                f"{np.round(q1, 1).tolist()}")

    def may_be_moving(self) -> bool:
        """RobotEnv.may_be_moving: commanded within the last second."""
        last = self.__dict__.get("_last_command_t")
        return last is not None and time.monotonic() - last < _MOVING_AFTER_COMMAND_S

    # A reading this recent is used again rather than read again: in one
    # control tick the loop reads the joints for its step, and the safety
    # monitor's per-cycle check reads them twice more (the arm, and the jaw
    # opening its grasp point needs) a few ms later, before the command goes
    # out — 4 bus round trips a tick for 2 readings. The sentry's own reads
    # (15 Hz) share them too. Sending a command makes the next read a fresh one.
    POS_REUSE_S = 0.010

    def _motor_pos(self) -> dict:
        """All motor positions, normalised. Bus only — no camera traffic."""
        with self._bus_lock:
            cached = self.__dict__.get("_pos_cache")
            if cached is not None and time.monotonic() - cached[0] < self.POS_REUSE_S:
                return dict(cached[1])
            pos = self._robot.bus.sync_read("Present_Position")
            self.__dict__["_pos_cache"] = (time.monotonic(), dict(pos))
            flight_recorder.sample("read", pos)
            return pos

    def _bus_read(self, register: str, motor: str) -> float:
        with self._bus_lock:
            return float(self._robot.bus.read(register, motor, normalize=False))

    def hold(self) -> None:
        """
        RobotEnv.hold: command the arm's joints where they are now. A servo
        otherwise carries on to the last goal it was sent, up to a step
        (MAX_RELATIVE_TARGET_DEG) further — centimetres at the jaws.

        The gripper's goal is left alone: re-commanding it where it is would
        take the squeeze off a held piece and drop it.
        """
        try:
            with self._bus_lock:
                pos = self._robot.bus.sync_read("Present_Position")
                self._robot.send_action({f"{n}.pos": float(pos[n]) for n in JOINT_NAMES})
                self.__dict__.pop("_pos_cache", None)
        except Exception as exc:
            print(f"  [SO101Env] Could not command a hold ({exc}) — the arm finishes its last step.")

    def arm_shape(self) -> "list[tuple[np.ndarray, float]] | None":
        """
        RobotEnv.arm_shape: _ARM_SHAPE's frames from the joint angles read now,
        then the moving jaw's tip at the jaws' opening.

        Called from a watcher's thread while the control loop runs IK, so it
        uses a kinematic model of its own: placo keeps the joint angles of the
        last call in the model, and a forward kinematics call from here in the
        middle of an IK solve would change the angles it is solving from.
        """
        if self._kinematics is None:
            return None
        pos = self._motor_pos()
        q = np.array([pos[n] for n in JOINT_NAMES] + [0.0], float)
        with self.__dict__.setdefault("_shape_lock", threading.Lock()):
            robot = self.__dict__.get("_shape_robot")
            if robot is None:
                import placo
                robot = self.__dict__["_shape_robot"] = placo.RobotWrapper(str(self._urdf_path))
            names = self._kinematics.joint_names
            q = np.deg2rad(q[:len(names)] + (0.0 if self._joint_offsets_deg is None
                                             else self._joint_offsets_deg[:len(names)]))
            for name, v in zip(names, q):
                robot.set_joint(name, float(v))
            robot.update_kinematics()
            shape = [(np.array(robot.get_T_world_frame(f)[:3, 3], float), r) for f, r in _ARM_SHAPE]
            tip = np.array(robot.get_T_world_frame("gripper_frame_link"), float)
        gap = float(np.clip(self._pct_to_width(float(pos["gripper"])), 0.0, self._gripper_max_open_m))
        axis = getattr(self, "_jaw_slide_axis", np.array([-1.0, 0.0, 0.0]))
        jaw = tip[:3, 3] + tip[:3, :3] @ (self._tcp_offset + axis * gap)
        return shape + [(jaw, _JAW_TIP_RADIUS_M)]

    @property
    def base_xy(self) -> np.ndarray:
        """RobotEnv.base_xy: where the shoulder_pan axis meets the table (see _find_pan_centre)."""
        return np.asarray(self._pan_centre, float).copy()

    def _q_arm_deg(self, obs: "dict | None" = None) -> np.ndarray:
        """Current 5 arm-joint positions in degrees, from the motor bus."""
        if obs is not None:
            return np.array([obs[f"{n}.pos"] for n in JOINT_NAMES], dtype=float)
        pos = self._motor_pos()
        return np.array([pos[n] for n in JOINT_NAMES], dtype=float)

    def gripper_frame(self) -> np.ndarray:
        """RigEnv.gripper_frame: gripper_frame_link in the base frame, from the joints read now."""
        return self._kinematics.forward_kinematics(np.append(self._q_arm_deg(), 0.0))  # type: ignore[union-attr]  # rig.py asks with kinematics (see report)

    def _gripper_frames(self) -> "tuple[np.ndarray, np.ndarray]":
        """
        RigEnv._gripper_frames: gripper_frame(), and the same reading with the
        joints corrected as GRIPPER_CAM_POSE was fitted (GRIPPER_CAM_JOINT_DEG, on
        top of JOINT_OFFSETS_DEG). Under load this arm reads its joints a few
        degrees off, the same way at the same pose: without them the mount is
        centimetres out.
        """
        q = self._q_arm_deg()
        arm = self._kinematics.forward_kinematics(np.append(q, 0.0))  # type: ignore[union-attr]
        extra = getattr(self, "_gripper_cam_joint_deg", None)
        if extra is None:
            return arm, arm
        return arm, self._kinematics.forward_kinematics(np.append(q + extra, 0.0))  # type: ignore[union-attr]

    def _has_kinematics(self) -> bool:
        return self._kinematics is not None

    def _require_kinematics(self) -> None:
        if self._kinematics is None:
            raise NotImplementedError(
                "Cartesian control unavailable: install placo (`pip install placo`) and fetch "
                "the URDF (`python tools/so101/fetch_urdf.py`), then pass "
                "urdf_path to SO101Env."
            )

    def _clamp_joint_degrees(self, degrees: dict[str, float]) -> dict[str, float]:
        """Clip to each joint's calibrated safe range (see __init__) before sending."""
        clamped = {}
        for name, value in degrees.items():
            lo_hi = self._joint_deg_limits.get(name)
            if lo_hi is None:
                clamped[name] = value
                continue
            lo, hi = lo_hi
            v = float(np.clip(value, lo, hi))
            if abs(v - value) > 0.5:
                print(f"  [SO101Env] '{name}' target {value:.1f}deg exceeds calibrated range "
                      f"[{lo:.1f}, {hi:.1f}] — clamped to {v:.1f}deg")
            clamped[name] = v
        return clamped

    # ── RobotEnv: physics / time ───────────────────────────────────────────────

    def step(self, n: int = 1) -> None:
        time.sleep(n / self._control_hz)

    # ── RobotEnv: sensing ──────────────────────────────────────────────────────

    def get_joint_angles(self) -> dict[str, float]:
        deg = self._q_arm_deg()
        return {name: float(np.deg2rad(v)) for name, v in zip(JOINT_NAMES, deg)}

    def get_body_pos(self, name: str) -> "np.ndarray | None":
        return None  # no ground-truth object poses on real hardware — see vlm.discover_scene

    def _pct_to_width(self, pct: float) -> float:
        """
        Jaw opening for a gripper motor position: GRIPPER_WIDTH_TABLE when it
        has been measured, else a straight line from 0 to gripper_max_open_m.

        The line is a guess, and a poor one for a hinged jaw: a ~20 mm piece
        reads 2-6 mm on it, so every opening it is asked for is far wider in
        fact. tools/so101/calibrate_gripper_width.py measures the table.
        """
        pct = max(0.0, float(pct))
        if self._gripper_width_table:
            p, w = zip(*self._gripper_width_table)
            return float(np.interp(pct, p, w))
        return pct / 100.0 * self._gripper_max_open_m

    def _width_to_pct(self, width: float) -> float:
        """The motor position for a jaw opening — _pct_to_width's inverse."""
        if self._gripper_width_table:
            p, w = zip(*self._gripper_width_table)
            return float(np.interp(width, w, p))
        return float(np.clip(width / self._gripper_max_open_m * 100.0, 0.0, 100.0))

    # ── RobotEnv: scene introspection ──────────────────────────────────────────

    def get_movable_body_names(self) -> list[str]:
        return []  # VLM discovers objects purely from the camera image

    def get_static_body_names(self, exclude_keywords: "set[str] | None" = None) -> list[str]:
        return []

    # ── Extra helpers (not part of RobotEnv) — used by find_poses.py ───────────

    def get_joint_degrees(self) -> dict[str, float]:
        return {name: float(v) for name, v in zip(JOINT_NAMES, self._q_arm_deg())}

    def disable_torque(self) -> None:
        """Let the arm be moved by hand — used by find_poses.py for calibration."""
        with self._bus_lock:
            self._robot.bus.disable_torque()

    def enable_torque(self) -> None:
        with self._bus_lock:
            self._robot.bus.enable_torque()

    def _find_pan_centre(self) -> np.ndarray:
        """
        Where the shoulder_pan axis meets the table plane, in the base frame.

        Not the base origin: the URDF puts the pan joint ~39 mm in front of it,
        and the arm's plane has to be measured from the axis it swings about.
        Found from FK — the gripper's position at three pan angles lies on a
        circle round the axis — so it follows the URDF rather than a constant.
        """
        assert self._kinematics is not None     # run just after it is made
        q = np.zeros(len(self._kinematics.joint_names))
        q[JOINT_NAMES.index("shoulder_lift")] = 45.0
        pts = []
        for pan in (-60.0, 0.0, 60.0):
            q[JOINT_NAMES.index("shoulder_pan")] = pan
            pts.append(self._kinematics.forward_kinematics(q)[:2, 3])
        (x1, y1), (x2, y2), (x3, y3) = pts
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
        uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
        return np.array([ux, uy])
