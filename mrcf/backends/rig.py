"""
mrcf/backends/rig.py — the part of an arm's backend that is about its cameras and
its gripper rather than its motors, shared by every arm built on it.

An arm on a rig: a two-finger gripper, a camera fixed above the work surface
and/or one on the wrist, and, optionally, a tag on the gripper the fixed camera
can see. Given where the gripper is (gripper_frame(), from the arm's own
kinematics), RigEnv works out the rest the same way for every arm:

  camera_model()            the fixed camera's calibrated lens and pose; the
                            wrist camera's from the gripper's pose now
  pixel_to_world()          the fixed camera's pixels -> the table (homography)
  get_ee_pos()              the grasp point: the gripper frame, plus where the
                            jaws close at their opening (_tcp — the centre slides
                            as they open when only one jaw moves)
  hold_width()              the held object's width, for that grasp point
  overhead_gripper_offset() where the fixed camera sees the gripper's tag, against
                            the kinematics (capabilities.OverheadSighting)
  visual_servo_xy()         centre the jaws over an object from the wrist camera
                            (capabilities.WristServo)
  wait_still()              hold, and wait until the gripper stops (Settling)
  set_overhead_calibration  re-pose the fixed camera on a running arm
                            (capabilities.OverheadCalibration; overhead_calibration.py)
  the sentry                motion_guard.MotionGuard on the fixed camera, for as
                            long as the arm is connected: it starts as the
                            arm's own connect() returns and stops as its stop()
                            begins, whatever those do (__init_subclass__)

What an arm built on it provides (mrcf/backends/so101/env.py is the example):

  gripper_frame()           4x4: the gripper's frame in the world NOW, from the
                            joints read now — the frame GRIPPER_CAM_POSE,
                            GRIPPER_TAG_POSE and the jaw model are measured in
  get_gripper_width()       the jaws' opening, metres
  set_gripper(width, n_steps=, max_step_pct=)
                            and, as the jaws let go, self._held_width_m = None
                            (hold_width holds until then)
  move_ee_to(target, tol, max_step_deg=, orientation=)
                            the grasp point to target, holding orientation
  render_image()            frames, safe from any thread
  connect(), stop()         open and close its hardware (RobotEnv's); connect()
                            must not move the arm, and neither starts the sentry
  tilted_rotation(deg)      the gripper's orientation at a lean (only for a wrist
                            servo calibrated at a known lean; see _servo_tilt_offset)
  gripper_tilt_deg()        likewise
  _has_kinematics()         whether gripper_frame() can answer (default True)

and calls _init_rig(...) from its constructor with its measured values.
Everything here is pure geometry over those, so it is the same on any arm.

Cameras and sensors: a rig has any number of each, described in its
robot_config (CAMERAS, SENSORS: Camera and Sensor below). Two cameras have
jobs the framework knows by name, and either may be absent: the one on the
gripper the policies measure and align with (camera_name, CAMERA_NAME), and
the fixed one that scans the surface, sights the gripper tag and checks its
own calibration (overhead_camera_name, OVERHEAD_CAMERA_NAME); the motion guard
watches every camera in motion_guard_cameras. Any other camera gets the same
geometry from its own Camera entry — camera_model() and pixel_to_world() for a
calibrated fixed one, its pose from the joints for one on the gripper — and
any sensor is read through read_sensor(), a force sensor feeding
get_contact_force().

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, cast

import numpy as np

from mrcf.robot_env import RobotEnv

if TYPE_CHECKING:
    from mrcf.motion_guard import MotionGuard


@dataclass
class Camera:
    """
    One camera on the rig, as its robot_config's CAMERAS describes it. The
    calibration fields are for a camera with no job of its own: the gripper
    camera's and the overhead camera's live in their own settings
    (GRIPPER_CAM_*, OVERHEAD_CAM_MODEL and OVERHEAD_HOMOGRAPHY), which their
    tools write and a running arm re-poses.
    """
    mount: str = "fixed"                       # "fixed": still, off the arm; "gripper": moves with the jaws
    source: "int | str | None" = None          # OpenCV index or URL, for an env that opens it itself
    size: "tuple[int, int]" = (640, 480)       # capture width, height
    fps: int = 30
    # A fixed camera: its lens and pose in the arm's frame, {size, K, D, T_world_cam},
    # and its pixels (at RENDER_W x RENDER_H) -> table (x, y), 3x3 — both printed
    # by tools/calibrate_overhead_camera.py --camera NAME.
    model: "dict | None" = None
    homography: "list | None" = None
    # A camera on the gripper: its lens (K at RENDER_W x RENDER_H, distortion),
    # where it sits on the gripper (gripper frame -> camera, 4x4, metres), and the
    # plane it measures on (None: the work surface).
    K: "list | None" = None
    dist: "list | None" = None
    pose_on_gripper: "list | None" = None
    plane_z_m: "float | None" = None


@dataclass
class Sensor:
    """
    One sensor beyond the joints and the gripper, as its robot_config's SENSORS
    describes it. `kind` says what the framework may use it for:

      "force"            the contact force (N), or a wrench whose first three are
                         the force: get_contact_force(), so the safety monitor's
                         force limits work
      "gripper_contact"  whether the jaws are pressing on something (an arm's
                         gripper_loaded() can read it)
      anything else      the arm's own: read_sensor(), the flight recorder, a
                         learned policy's stream
    """
    kind: str
    unit: str = ""
    shape: tuple = ()                          # () a number; (6,) a wrench ...
    # How the env reads it: the name of one of its methods, or a function of the env.
    read: "str | Callable | None" = None


def zone_around(x: "tuple[float, float]", y: "tuple[float, float]", margin_m: float) -> list:
    """
    A robot_config's MOTION_GUARD_ZONE: the work surface x (min, max) by y
    (min, max) in the arm's frame, metres, widened by margin_m all round, as
    the four corners of the patch of table the sentry watches.
    """
    (x0, x1), (y0, y1), m = x, y, float(margin_m)
    return [(x0 - m, y0 - m), (x1 + m, y0 - m), (x1 + m, y1 + m), (x0 - m, y1 + m)]


def read_sensor_value(env, name: str, sensor: Sensor):
    """sensor's reading on env: its method by name, or its function of the env."""
    if callable(sensor.read):
        return sensor.read(env)
    if isinstance(sensor.read, str):
        return getattr(env, sensor.read)()
    raise KeyError(f"Sensor '{name}' says nothing about how to read it (Sensor.read).")


def _watched(connect):
    """
    connect(), then the sentry — once the outermost connect() has returned, so
    an arm whose connect() calls super().connect() first is not watched before
    its own hardware is open. An arm the sentry cannot watch is closed again.
    """
    if getattr(connect, "_starts_sentry", False):
        return connect

    @functools.wraps(connect)
    def connect_then_watch(self, *args, **kwargs):
        depth = self.__dict__.get("_connect_depth", 0)
        self._connect_depth = depth + 1
        try:
            connect(self, *args, **kwargs)
        finally:
            self._connect_depth = depth
        if depth == 0:
            try:
                self._start_sentry()
            except BaseException:
                self.stop()
                raise

    connect_then_watch._starts_sentry = True
    return connect_then_watch


def _unwatched(stop):
    """The sentry stopped, then stop(): it reads the arm, so it goes before the arm's connection does."""
    if getattr(stop, "_starts_sentry", False):
        return stop

    @functools.wraps(stop)
    def unwatch_then_stop(self, *args, **kwargs):
        self._stop_sentry()
        stop(self, *args, **kwargs)

    unwatch_then_stop._starts_sentry = True
    return unwatch_then_stop


class RigEnv(RobotEnv):
    """A RobotEnv whose cameras and gripper geometry follow from its kinematics. See the module docstring."""

    # Named in "not calibrated" messages: where this arm measures its wrist servo.
    SERVO_CALIBRATION_TOOL = "the arm's wrist-servo calibration"

    # What an env made without _init_rig has (a test's, an offline tool's): no
    # other cameras or sensors, and the sentry on the overhead camera. Never
    # changed in place: _init_rig gives each env its own.
    _cameras: dict = {}
    _sensors: dict = {}
    _motion_guard_cameras = None
    _gripper_box = None
    # Moves the gripper's box has refused, and why the last one was: a policy
    # whose move came to nothing reads them (RobotEnv.move_refusals) and says
    # why, instead of "did not converge" — which a planner takes as worth
    # another go (2026-11-13: the same refused pick planned again and again).
    move_refusals = 0
    last_refusal: "str | None" = None

    # Whether this arm's moves keep the gripper inside a box (envelope.py): each
    # checks its path with _refused_by_box before it is sent (the UR5's, the
    # template's). An arm that does not is refused a box, rather than given one
    # it would ignore.
    CHECKS_GRIPPER_BOX = False

    def __init_subclass__(cls, **kwargs):
        # Every arm on a rig is watched while it is connected, whether or not
        # its own connect() remembers to (see _start_sentry). So each arm's
        # connect() and stop() are wrapped here.
        super().__init_subclass__(**kwargs)
        if "connect" in vars(cls):
            cls.connect = _watched(vars(cls)["connect"])
        if "stop" in vars(cls):
            cls.stop = _unwatched(vars(cls)["stop"])

    def _init_rig(
        self,
        *,
        camera_name: "str | None" = None,
        render_w: int = 640,
        render_h: int = 480,
        # Height of the work surface in the arm's frame, used to turn the
        # gripper's height above it into a pixel scale for visual servoing.
        work_plane_z: float = 0.0,
        overhead_camera_name: "str | None" = None,
        overhead_homography: "list | np.ndarray | None" = None,
        # Camera models for measuring on the work surface — see camera_model().
        # None leaves a camera's model unknown.
        overhead_cam_model: "dict | None" = None,
        gripper_cam_k: "list | np.ndarray | None" = None,
        gripper_cam_dist: "list | np.ndarray | None" = None,
        gripper_cam_pose: "list | np.ndarray | None" = None,
        gripper_cam_plane_z_m: "float | None" = None,
        # Where what the fixed camera's scan measured stands in the frame the
        # wrist camera is placed in: scan position + this (x, y, z), metres.
        # Measured with GRIPPER_CAM_POSE (both cameras saw the same board).
        gripper_cam_scan_shift_m=None,
        arm_frame_scale: float = 1.0,
        # Gripper-camera visual servoing — measured together (on the SO-101 by
        # tools/so101/calibrate_gripper_servo.py). None disables servoing.
        gripper_cam_grasp_px: "tuple[float, float] | None" = None,
        gripper_cam_jacobian: "list | np.ndarray | None" = None,
        gripper_cam_ref_height_m: "float | None" = None,
        # Where the servo's calibration was taken, so it can aim right at any
        # other tilt (see _servo_tilt_offset): the gripper's tilt then
        # (tilted_rotation's degrees), the height of the point the detector
        # reports on the object it used, and the camera's centre in the
        # gripper frame (approximate is enough). Tilt None: aims as calibrated
        # whatever the tilt.
        gripper_cam_ref_tilt_deg: "float | None" = None,
        gripper_cam_ref_point_height_m: float = 0.0,
        gripper_cam_position_m: "tuple | np.ndarray | None" = None,
        # A tag on the gripper the overhead camera can see (overhead_gripper_offset):
        # its id, its pose on the gripper (gripper frame -> tag, 4x4 m) and the
        # size it reads at. None: no such tag.
        gripper_tag_id: "int | None" = None,
        gripper_tag_pose: "list | np.ndarray | None" = None,
        gripper_tag_size_m: "float | None" = None,
        # The sentry (motion_guard.MotionGuard on the overhead camera): where it
        # watches, an (x, y) polygon on the work surface. Not None: connecting
        # starts it (_start_sentry), and refuses when it cannot. None: no sentry.
        motion_guard_zone=None,
        # The fixed cameras it watches through, each a guard of its own (None:
        # the overhead camera).
        motion_guard_cameras: "list | None" = None,
        # Every camera on the rig, {name: Camera}, the two with jobs among them,
        # and every sensor beyond the joints and the gripper, {name: Sensor}.
        cameras: "dict | None" = None,
        sensors: "dict | None" = None,
        # The jaw model (_tcp): the grasp point with the jaws CLOSED, in the
        # gripper frame; how far it moves per metre of opening
        # (jaw_centre_slide: 0.5 for one moving jaw, 0.0 for two that move
        # together) and which way (jaw_slide_axis, toward the moving jaw); the
        # widest they open.
        tcp_offset: "np.ndarray | None" = None,
        jaw_centre_slide: float = 0.5,
        jaw_slide_axis: "tuple | np.ndarray" = (-1.0, 0.0, 0.0),
        gripper_max_open_m: float = 0.045,
        # Where the jaws really close against that model, by lean and reach
        # (grasp_correction.GraspCorrection, tools/calibrate_grasp.py). None: as modelled.
        grasp_correction=None,
        # The box the whole gripper stays inside, flange to jaw tips
        # (envelope.GripperBox, from config GRIPPER_BOX: wiring.gripper_box).
        # None: no box. Only for an arm whose moves check it (CHECKS_GRIPPER_BOX).
        gripper_box=None,
        # The arm's facts a pick depends on that have not been checked on the rig
        # (mrcf/arm_facts.py, from wiring.rig_kwargs): a pick that needs one is
        # refused. () for an arm with nothing physical to be wrong (offline).
        unchecked_facts=(),
    ) -> None:
        """Set the rig's measured values (each described at its parameter)."""
        self._camera_name = camera_name
        self._render_w = render_w
        self._render_h = render_h
        self._work_plane_z = work_plane_z
        self._overhead_camera_name = overhead_camera_name
        self._overhead_homography = (
            np.asarray(overhead_homography, dtype=float) if overhead_homography is not None else None
        )
        self._overhead_cam_model = overhead_cam_model
        self._gripper_cam_k = None if gripper_cam_k is None else np.asarray(gripper_cam_k, float)
        self._gripper_cam_dist = None if gripper_cam_dist is None else np.asarray(gripper_cam_dist, float)
        self._gripper_cam_pose = None if gripper_cam_pose is None else np.asarray(gripper_cam_pose, float)
        self._gripper_cam_plane_z = gripper_cam_plane_z_m
        self._gripper_cam_scan_shift = (None if gripper_cam_scan_shift_m is None
                                        else np.asarray(gripper_cam_scan_shift_m, float))
        self._arm_frame_scale = float(arm_frame_scale)
        self._gripper_cam_grasp_px = gripper_cam_grasp_px
        self._gripper_cam_jacobian = gripper_cam_jacobian
        self._gripper_cam_ref_height_m = gripper_cam_ref_height_m
        self._gripper_cam_ref_tilt_deg = gripper_cam_ref_tilt_deg
        self._gripper_cam_ref_point_height_m = float(gripper_cam_ref_point_height_m)
        self._gripper_cam_position_m = (np.zeros(3) if gripper_cam_position_m is None
                                        else np.asarray(gripper_cam_position_m, float))
        self._gripper_tag_id = gripper_tag_id
        self._gripper_tag_pose = None if gripper_tag_pose is None else np.asarray(gripper_tag_pose, float)
        self._gripper_tag_size_m = gripper_tag_size_m
        self.last_tag_fit_px: "float | None" = None
        self._motion_guard_zone = motion_guard_zone
        self._motion_guard_cameras = None if motion_guard_cameras is None else list(motion_guard_cameras)
        self._guard: "MotionGuard | None" = None
        self._cameras = dict(cameras or {})
        self._sensors = dict(sensors or {})
        self._gripper_max_open_m = gripper_max_open_m
        self._tcp_offset = np.zeros(3) if tcp_offset is None else np.asarray(tcp_offset, dtype=float)
        self._jaw_centre_slide = float(jaw_centre_slide)
        axis = np.asarray(jaw_slide_axis, float)
        self._jaw_slide_axis = axis / (np.linalg.norm(axis) or 1.0)
        self._grasp_table = grasp_correction
        # Width of the object held between the jaws, when known (see hold_width).
        self._held_width_m: "float | None" = None
        if gripper_box is not None and not self.CHECKS_GRIPPER_BOX:
            raise ValueError(f"{type(self).__name__}'s moves do not check a gripper box: GRIPPER_BOX would "
                             "do nothing. Leave it out, or check each move's path with _refused_by_box "
                             "and set CHECKS_GRIPPER_BOX = True.")
        self._gripper_box = gripper_box
        self._unchecked_facts = tuple(unchecked_facts)

    # ── What the arm provides ──────────────────────────────────────────────────

    def gripper_frame(self) -> np.ndarray:
        """
        The gripper's frame in the world now, 4x4 (metres), from the joints
        read now: the frame the wrist camera's pose (GRIPPER_CAM_POSE), the
        gripper tag's (GRIPPER_TAG_POSE) and the jaw model (_tcp) are measured in.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement gripper_frame().")

    def _flange_for(self, target: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """The gripper-frame pose that puts the grasp point (at the jaws' opening now: _tcp) on target, turned to rotation."""
        goal = np.eye(4)
        goal[:3, :3], goal[:3, 3] = rotation, np.asarray(target, float) - np.asarray(rotation, float) @ self._tcp()
        return goal

    def unchecked_facts(self) -> tuple:
        """RobotEnv.unchecked_facts: as the rig's settings had them (wiring.rig_kwargs, mrcf/arm_facts.py)."""
        return self._unchecked_facts

    def jaw_direction(self, rotation) -> "np.ndarray | None":
        """capabilities.JawDirection: rotation's jaw_slide_axis on the surface."""
        d = (np.asarray(rotation, float) @ np.asarray(self._jaw_slide_axis, float))[:2]
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-6 else None

    def down_rotation(self, at, flip: bool = False, turn_deg: float = 0.0) -> np.ndarray:
        """
        The gripper's rotation straight down (its z at the table) with the jaws
        opening along the arm's jaw_axis at `at` — "tangential": across the
        line from the base (base_xy); "radial": along it — turned turn_deg
        counter-clockwise seen from above, half a turn more when flip. The
        jaws open along jaw_slide_axis in the gripper frame (config
        JAW_SLIDE_AXIS), as mounted. The arm's policies plan the room either
        side of an object for jaws opening this way (GripperSpec.jaw_axis,
        from the same JAW_AXIS), so an arm's default orientation for a
        straight-down move is this one.
        """
        u = np.asarray(at, float)[:2] - np.asarray(getattr(self, "base_xy", (0.0, 0.0)), float)[:2]
        u = u / (np.linalg.norm(u) or 1.0)
        jaw = np.array([-u[1], u[0]]) if getattr(self, "_jaw_axis", "radial") == "tangential" else u
        a = np.radians(turn_deg) + (np.pi if flip else 0.0)
        jx, jy = np.cos(a) * jaw[0] - np.sin(a) * jaw[1], np.sin(a) * jaw[0] + np.cos(a) * jaw[1]
        # Gripper x and y in the table's plane, z straight down; turned so the
        # slide axis (sx, sy) lands on the jaw direction (jx, jy).
        sx, sy = self._jaw_slide_axis[:2] / (np.linalg.norm(self._jaw_slide_axis[:2]) or 1.0)
        c, s_ = sx * jx - sy * jy, sy * jx + sx * jy
        x_axis, z_axis = np.array([c, s_, 0.0]), np.array([0.0, 0.0, -1.0])
        return np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])

    # ── The gripper's box (envelope.py) ────────────────────────────────────────
    # For an arm that sets CHECKS_GRIPPER_BOX: each move's path, as gripper-frame
    # poses, is checked before the move is sent; one that would take any part of
    # the gripper out of its box (or, from outside, further out) is not sent.

    def _refused_by_box(self, path) -> bool:
        """True (and why, said) when a move along path (gripper-frame poses) would take the gripper out of its box."""
        why = self._gripper_box.refuses(path) if self._gripper_box is not None else None
        if why:
            print(f"  [Box] Not moving: {why} (robot_config GRIPPER_BOX).")
            self.move_refusals += 1
            self.last_refusal = f"the gripper would leave its box: {why}" if "box" not in why else why
        return bool(why)

    def gripper_fits(self, at) -> bool:
        """
        Whether the gripper straight down with its grasp point at `at` (x, y, z,
        the jaws turned as down_rotation turns them there) is inside its box:
        where a pick or a place could reach without the box refusing it. True
        with no box. An object outside it can be seen but not picked (the
        planner is told it is out of reach).
        """
        if self._gripper_box is None:
            return True
        at = np.asarray(at, float)
        return self._gripper_box.outside(self._flange_for(at, self.down_rotation(at)))[0] <= 0.0

    def _line_refused_by_box(self, goal: np.ndarray, rotation: np.ndarray) -> bool:
        """_refused_by_box for a straight line to the gripper-frame pose goal, from here with the jaws turned to rotation."""
        if self._gripper_box is None:
            return False
        from mrcf import envelope
        return self._refused_by_box(envelope.linear_path(self._flange_for(self.get_ee_pos(), rotation), goal))

    def _step_refused_by_box(self, goal_xyz) -> "str | None":
        """Why one streamed step of the grasp point to goal_xyz, orientation held, would leave the box (None: it would not)."""
        if self._gripper_box is None:
            return None
        now = self.gripper_frame()
        return self._gripper_box.refuses([now, self._flange_for(np.asarray(goal_xyz, float), now[:3, :3])])

    def _has_kinematics(self) -> bool:
        """Whether gripper_frame() can answer."""
        return True

    def _gripper_frames(self) -> "tuple[np.ndarray, np.ndarray]":
        """
        (gripper_frame(), the gripper's frame as the wrist camera's pose
        (GRIPPER_CAM_POSE) was fitted against), from one reading of the joints:
        here the same. An arm whose calibration corrects its joint readings for
        the camera (the SO-101's GRIPPER_CAM_JOINT_DEG) overrides it.
        """
        g = self.gripper_frame()
        return g, g

    def _require_kinematics(self) -> None:
        if not self._has_kinematics():
            raise NotImplementedError(f"{type(self).__name__}: no kinematic model, so no Cartesian control.")

    # ── The sentry ─────────────────────────────────────────────────────────────

    @_watched
    def connect(self) -> None:
        super().connect()

    @_unwatched
    def stop(self) -> None:
        super().stop()

    def _start_sentry(self) -> None:
        """
        The motion guard over the work surface, for as long as this connection
        lasts (motion_guard_zone; robot_config MOTION_GUARD). Every program that
        drives the arm connects through here, so every movement is watched and
        stops if there is a hand in the way. One that cannot watch does not
        connect, rather than move unwatched.
        """
        if self._motion_guard_zone is None or self._guard is not None:
            return
        cameras = self._sentry_cameras()
        wrist = self.__dict__.get("_camera_name")
        on_gripper = [c for c in cameras if c == wrist or getattr(self._cameras.get(c), "mount", "fixed") != "fixed"]
        if on_gripper:
            raise RuntimeError(f"MOTION_GUARD_CAMERAS lists {', '.join(on_gripper)}, on the gripper: the guard "
                               "watches through fixed cameras — not moving unwatched.")
        if not cameras:
            raise RuntimeError("The sentry needs the overhead camera, and this arm has none configured "
                               "(OVERHEAD_CAMERA_NAME) — not moving unwatched. MOTION_GUARD = False turns it off.")
        from mrcf.motion_guard import MotionGuard
        guards = []
        try:
            for i, camera in enumerate(cameras):
                # An interlock source each ("motion", the guard's own, for the
                # first): guards sharing one would clear each other's trips.
                named: dict = {} if i == 0 else {"name": f"motion:{camera}"}
                guards.append(MotionGuard(self, camera, zone=self._motion_guard_zone, **named).start())
        except BaseException as exc:
            for guard in guards:                 # the ones that did start: none left watching a closed arm
                guard.stop()
            if not isinstance(exc, RuntimeError):
                raise
            # A new rig's first calibration of that camera has to connect
            # before there is anything for the sentry to watch with.
            raise RuntimeError(f"{exc} Not moving unwatched. On a new rig, MOTION_GUARD = False in the arm's "
                               "robot_config until that camera is calibrated (COMMANDS.md, \"First-time setup "
                               "on another arm\").") from exc
        self._guard, self._more_guards = guards[0], guards[1:]
        print(f"  [{type(self).__name__}] Sentry on: the arm freezes if anything else moves over the work surface"
              + (f" (watched by {', '.join(cameras)})." if len(cameras) > 1 else "."))

    def motion_seen_since(self, t: float) -> "list | None":
        """
        Where the sentry saw something other than the arm move since t
        (time.monotonic), on the work surface: [(x, y, radius_m)]. None with no
        sentry watching: nothing can say what did not move.
        """
        guards = self._guards()
        if not guards:
            return None
        seen = [g.moved_since(t) for g in guards]
        # One that cannot say (t before it watched, or forgotten): it may have
        # been anything, anywhere — a thing moved across the whole table.
        return [p for s in seen for p in (s if s is not None else [(0.0, 0.0, float("inf"))])]

    def _guards(self) -> list:
        return [g for g in [self.__dict__.get("_guard"), *self.__dict__.get("_more_guards", [])] if g is not None]

    def sentry_sources(self) -> set:
        """The interlock sources the sentry's guards trip by when they see movement (not their failures)."""
        return {g.name for g in self._guards()}

    def _sentry_cameras(self) -> list:
        """The cameras the sentry watches through: motion_guard_cameras, else the overhead camera."""
        if self._motion_guard_cameras is not None:
            return [c for c in self._motion_guard_cameras if c is not None]
        return [] if self._overhead_camera_name is None else [self._overhead_camera_name]

    def _stop_sentry(self) -> None:
        guards = [self.__dict__.get("_guard"), *self.__dict__.get("_more_guards", ())]
        self._guard, self._more_guards = None, []
        for guard in guards:
            if guard is not None:
                guard.stop()

    # ── Where the arm is ───────────────────────────────────────────────────────

    def wait_still(self, tol_mm: float = 0.2, timeout_s: float = 4.0) -> float:
        """
        Hold the arm where it is and wait for it to stop: the gripper's last
        0.2 s step, mm. A move that stalls short leaves the servos creeping on
        toward their last goal for seconds — up to 10 mm on the SO-101.
        Anything measured of the arm's position belongs after this.
        """
        self.hold()
        start, prev, step = self.motion_clock(), self.gripper_frame()[:3, 3], float("inf")
        while self.motion_clock() - start < timeout_s:
            time.sleep(0.2)
            cur = self.gripper_frame()[:3, 3]
            step = float(np.linalg.norm(cur - prev) * 1000)
            if step < tol_mm:
                break
            prev = cur
        return step

    @property
    def frame_scale(self) -> float:
        """RobotEnv.frame_scale: ARM_FRAME_SCALE (the SO-101's frame reads ~9% short)."""
        return self._arm_frame_scale

    def get_ee_pos(self) -> np.ndarray:
        """
        Centre of the gap between the jaws AT THEIR CURRENT OPENING — not the
        gripper frame's origin, and not a fixed point (see _tcp: with one jaw
        moving, the midpoint slides as the gripper opens).

        Everything that compares against an object position wants this: skill
        impls' approach checks, the safety floor check, and the overhead
        camera's calibration by corner touches — which is why that asks you to
        CLOSE the jaws first, so the point it records is the pads' own contact
        point rather than the centre of an arbitrary gap.
        """
        self._require_kinematics()
        t = self.gripper_frame()
        here = t[:3, 3] + t[:3, :3] @ self._tcp()
        return here + self._grasp_correction(t[:3, :3], here)

    def grasp_lean_deg(self, rot, at) -> float:
        """
        The lean the grasp correction is measured and looked up by: the
        backend's own (capabilities.GripperTilt) where it has one; else how far
        the approach axis leans from straight down in the vertical plane
        through the base and `at`, positive away from the base.
        """
        if hasattr(self, "gripper_tilt_deg"):
            return float(self.gripper_tilt_deg(rot))
        u = np.asarray(at, float)[:2] - np.asarray(getattr(self, "base_xy", (0.0, 0.0)), float)[:2]
        u = u / (np.linalg.norm(u) or 1.0)
        approach = np.asarray(rot, float)[:, 2]
        return float(np.degrees(np.arctan2(approach[:2] @ u, -approach[2])))

    def _grasp_correction(self, rot, at) -> np.ndarray:
        """
        Where the jaws really close minus where the model puts them (world,
        horizontal), for the gripper turned to `rot` with its grasp point at
        `at`: the measured table by lean and reach (grasp_correction.py).
        Zeros without one. Added wherever the grasp point comes from the
        joints (get_ee_pos, and a backend's own IK); taken off a target before
        solving for it, so the jaws, not the model, arrive there.
        """
        table = getattr(self, "_grasp_table", None)
        if not table:
            return np.zeros(3)
        return table.offset(self.grasp_lean_deg(rot, at), at, getattr(self, "base_xy", (0.0, 0.0)))

    def get_ee_rotation(self) -> np.ndarray:
        """The gripper's orientation now (its frame, in the world): columns are its x, y, z."""
        self._require_kinematics()
        return self.gripper_frame()[:3, :3]

    # ── The jaws ───────────────────────────────────────────────────────────────

    def hold_width(self, width_m: float) -> None:
        """
        Tell the grasp-point model how wide the held object really is.

        get_gripper_width may not: on the SO-101 it is a map of the motor's
        0-100 % onto a hinged jaw, and with a ~16 mm pawn body between the pads
        it reads ~2-3 mm. _tcp puts the grasp point half the opening from the
        fixed pad, so on that reading it sits ~1 mm from the pad while the
        piece's centre is ~8 mm from it. Every move made while holding — the
        lift, the transit, the release — then aims the wrong point, and a placed
        piece lands that far off the square's centre along the jaw axis.

        Cleared by the next set_gripper call, i.e. when the jaws let go.
        """
        self._held_width_m = float(width_m)

    def _tcp(self, gap_m: "float | None" = None) -> np.ndarray:
        """
        Grasp-point offset in gripper-frame coords for a given jaw opening:
        the closed-jaw point (tcp_offset), slid jaw_centre_slide of the opening
        along jaw_slide_axis.

        Defaults to the jaws' MEASURED opening, which is what callers want: the
        point to drive onto a piece during a descent is the centre of the gap
        the jaws are actually holding open, not where they would pinch once
        closed. Driving the closed-jaw point onto the piece instead puts the
        fixed pad through it — see the SO-101's _TCP_OFFSET for the
        measurement and for what that cost.

        While an object is held (hold_width), its own width stands in for the
        reading, which is badly wrong for anything between the pads.
        """
        if gap_m is None and self._jaw_centre_slide == 0.0:
            # Both fingers move (a UR's Hand-E): the point does not move with the
            # opening, so the gripper is not asked for it — on every get_ee_pos,
            # the safety monitor's and the sentry's, a reply a stalled gripper
            # socket would hold up for seconds.
            return self._tcp_offset.copy()
        if gap_m is None:
            gap_m = self._grasp_gap()
        off = self._tcp_offset.copy()
        axis = getattr(self, "_jaw_slide_axis", np.array([-1.0, 0.0, 0.0]))
        return off + axis * self._jaw_centre_slide * float(np.clip(gap_m, 0.0, self._gripper_max_open_m))

    def _grasp_gap(self) -> float:
        """The jaw gap the grasp point is modelled at: the held object's width, else the reading."""
        return self._held_width_m if self._held_width_m is not None else self.get_gripper_width()

    # ── The fixed camera ───────────────────────────────────────────────────────

    @property
    def overhead_camera_pose(self) -> "np.ndarray | None":
        """capabilities.OverheadCalibration: the fixed camera's pose (camera -> world), or None without a model."""
        m = getattr(self, "_overhead_cam_model", None)
        return None if m is None else np.asarray(m["T_world_cam"], float)

    def set_overhead_calibration(self, T_world_cam=None, homography=None, gripper_tag_pose=None,
                                 gripper_tag_size_m: "float | None" = None) -> None:
        """capabilities.OverheadCalibration: re-pose the fixed camera (and the gripper tag) on a running arm."""
        if T_world_cam is not None and self._overhead_cam_model is not None:
            self._overhead_cam_model = {**self._overhead_cam_model, "T_world_cam": np.asarray(T_world_cam, float)}
        if homography is not None:
            self._overhead_homography = np.asarray(homography, float)
        if gripper_tag_pose is not None:
            self._gripper_tag_pose = np.asarray(gripper_tag_pose, float)
        if gripper_tag_size_m:
            self._gripper_tag_size_m = float(gripper_tag_size_m)

    # A gripper-tag position fitting its corners worse than this is not used.
    MAX_TAG_FIT_PX = 3.0

    def overhead_gripper_offset(self, frames: int = 3) -> "np.ndarray | None":
        """
        Where the overhead camera sees the gripper minus where the kinematics
        put it (arm frame, m) — the correction a move needs; None when the
        gripper tag is not in view (or there is no tag or camera model). Take
        it with the arm still (wait_still).

        The gripper's rotation comes from the kinematics (good to a degree or
        two), only its position from the tag: over 24 hovers on the SO-101 that
        agreed with the arm to 0.6 mm median, where the small tag's own tilt as
        well gave 2.2 — its tilt noise, times the ~90 mm from the tag to the
        jaws. The target it is compared with should come through the same
        camera (a scan's object, a board's square), so that camera's own pose
        errors cancel.
        """
        if (self._gripper_tag_id is None or self._gripper_tag_pose is None or not self._gripper_tag_size_m
                or self._overhead_cam_model is None or self._overhead_camera_name is None
                or not self._has_kinematics()):
            return None
        from mrcf.perception import fiducials
        m = self._overhead_cam_model
        K, D = np.asarray(m["K"], float), np.asarray(m["D"], float)
        T_arm_cam = np.asarray(m["T_world_cam"], float)
        w, h = m["size"]
        seen = []
        for _ in range(frames):
            found = fiducials.detect_tags(self.render_image(self._overhead_camera_name, width=w, height=h))
            if self._gripper_tag_id in found:
                seen.append(found[self._gripper_tag_id])
            time.sleep(0.04)
        if not seen:
            return None
        T_arm_gf = np.asarray(self.gripper_frame(), float)
        T_cam_arm = np.linalg.inv(T_arm_cam)
        T_cam_tag = T_cam_arm @ T_arm_gf @ self._gripper_tag_pose      # where the kinematics say it is
        t, err = fiducials.position_with_known_rotation(np.median(seen, axis=0), self._gripper_tag_size_m,
                                                        T_cam_tag[:3, :3], K, D, T_cam_tag[:3, 3])
        self.last_tag_fit_px = err
        if err > self.MAX_TAG_FIT_PX:
            # Said here: "cannot see the tag" was all a pick said of both (2026-11-19).
            print(f"  [Tag] The gripper's tag is in view, but fits where the arm's angles put it only to "
                  f"{err:.1f} px (over {self.MAX_TAG_FIT_PX:.1f}) — not used.")
            return None
        return T_arm_cam[:3, :3] @ (t - T_cam_tag[:3, 3])

    # ── Cameras ────────────────────────────────────────────────────────────────

    def fixed_views(self) -> dict:
        """
        The other views of the work surface from off the arm: every fixed
        camera in CAMERAS with its calibration (model=), the overhead camera
        aside, {name: capture size}. A scan keeps a frame from each
        (scene_scan.observe), for a check after a place to compare with.
        """
        return {name: tuple(cam.size) for name, cam in self._cameras.items()
                if cam.mount == "fixed" and cam.model is not None and name != self._overhead_camera_name}

    def camera_model(self, camera: str, width: int, height: int) -> "dict | None":
        """
        RobotEnv.camera_model: the overhead camera's calibrated pose, or the
        gripper camera's from the joint angles read NOW (gripper_frame() x
        GRIPPER_CAM_POSE), or any other camera's from its Camera entry, the
        same two ways — so take a gripper camera's frame and call this together, with the
        arm still. K is rescaled from the resolution it was measured at to
        width x height; render_image resizes frames the same way, axis by axis.
        """
        def scaled(k, size):
            k = np.asarray(k, float).copy()
            k[0] *= width / size[0]
            k[1] *= height / size[1]
            return k

        if camera == self._overhead_camera_name and self._overhead_cam_model is not None:
            m = self._overhead_cam_model
            return {"K": scaled(m["K"], m["size"]), "D": np.asarray(m["D"], float),
                    "T_world_cam": np.asarray(m["T_world_cam"], float),
                    "plane_z_m": self._work_plane_z, "scale": self._arm_frame_scale}
        if (camera == self._camera_name and self._gripper_cam_pose is not None
                and self._gripper_cam_k is not None and self._has_kinematics()):
            arm, seen = self._gripper_frames()
            plane = self._gripper_cam_plane_z if self._gripper_cam_plane_z is not None else self._work_plane_z
            model = {"K": scaled(self._gripper_cam_k, (self._render_w, self._render_h)),
                     "D": self._gripper_cam_dist if self._gripper_cam_dist is not None else np.zeros(5),
                     "T_world_cam": seen @ self._gripper_cam_pose, "plane_z_m": plane, "scale": self._arm_frame_scale}
            if self._gripper_cam_scan_shift is not None:
                model["scan_to_camera"] = self._gripper_cam_scan_shift.copy()
            if not np.allclose(arm, seen):
                # The camera is placed through the joints as its calibration
                # corrected them; the arm moves by its own model. A point the
                # camera finds is turned into that model's frame by this, so
                # it keeps its place relative to the jaws (both are bolted to
                # the gripper): to_arm @ [x, y, z, 1].
                model["to_arm"] = arm @ np.linalg.inv(seen)
            return model
        other = self._cameras.get(camera)
        if other is not None and other.mount == "fixed" and other.model is not None:
            m = other.model
            return {"K": scaled(m["K"], m["size"]), "D": np.asarray(m["D"], float),
                    "T_world_cam": np.asarray(m["T_world_cam"], float),
                    "plane_z_m": self._work_plane_z, "scale": self._arm_frame_scale}
        if (other is not None and other.mount == "gripper" and other.K is not None
                and other.pose_on_gripper is not None and self._has_kinematics()):
            T = self.gripper_frame() @ np.asarray(other.pose_on_gripper, float)
            return {"K": scaled(other.K, (self._render_w, self._render_h)),
                    "D": np.asarray(other.dist, float) if other.dist is not None else np.zeros(5),
                    "T_world_cam": T, "plane_z_m": other.plane_z_m if other.plane_z_m is not None else self._work_plane_z,
                    "scale": self._arm_frame_scale}
        # Any other camera is the next class's (RealRobotEnv's calibration files;
        # RobotEnv's None).
        return super().camera_model(camera, width, height)

    def camera_model_at(self, camera: str, width: int, height: int, at, rotation,
                        gap_m: "float | None" = None) -> "dict | None":
        """
        capabilities.CameraAtPose: camera_model with the grasp point at `at` and
        the gripper turned to `rotation`. A gripper camera is placed through the
        arm's own model of the joints (_flange_for), not the corrected joints its
        calibration was fitted against (_gripper_frames): a few mm and a degree
        or so off where it will be — for choosing where to look from, not for
        measuring (the look itself reads camera_model there).
        """
        def scaled(k, size):
            k = np.asarray(k, float).copy()
            k[0] *= width / size[0]
            k[1] *= height / size[1]
            return k

        if not self._has_kinematics():
            return None
        rotation = np.asarray(rotation, float)
        flange = np.eye(4)
        flange[:3, :3], flange[:3, 3] = rotation, np.asarray(at, float) - rotation @ self._tcp(gap_m)
        if camera == self._camera_name and self._gripper_cam_pose is not None and self._gripper_cam_k is not None:
            plane = self._gripper_cam_plane_z if self._gripper_cam_plane_z is not None else self._work_plane_z
            model = {"K": scaled(self._gripper_cam_k, (self._render_w, self._render_h)),
                     "D": self._gripper_cam_dist if self._gripper_cam_dist is not None else np.zeros(5),
                     "T_world_cam": flange @ self._gripper_cam_pose, "plane_z_m": plane, "scale": self._arm_frame_scale}
            if self._gripper_cam_scan_shift is not None:
                model["scan_to_camera"] = self._gripper_cam_scan_shift.copy()
            return model
        other = self._cameras.get(camera)
        if other is not None and other.mount == "gripper" and other.K is not None and other.pose_on_gripper is not None:
            return {"K": scaled(other.K, (self._render_w, self._render_h)),
                    "D": np.asarray(other.dist, float) if other.dist is not None else np.zeros(5),
                    "T_world_cam": flange @ np.asarray(other.pose_on_gripper, float),
                    "plane_z_m": other.plane_z_m if other.plane_z_m is not None else self._work_plane_z,
                    "scale": self._arm_frame_scale}
        if other is not None and other.mount == "gripper":
            return None
        return self.camera_model(camera, width, height)       # a fixed camera: wherever the arm is

    def pixel_to_world(self, u: float, v: float, z_world: float, camera: str,
                       render_w: int = 640, render_h: int = 480) -> np.ndarray:
        """
        The overhead camera: its calibrated homography (OVERHEAD_HOMOGRAPHY) —
        valid from ANY arm pose, since the camera is stationary. Other cameras
        are the next class's (RealRobotEnv's), or the arm's own to override.
        """
        if camera == self._overhead_camera_name:
            if self._overhead_homography is None:
                raise RuntimeError(
                    "Overhead camera homography not calibrated (OVERHEAD_HOMOGRAPHY) — calibrate the "
                    "overhead camera first (COMMANDS.md, first-time setup)."
                )
            pt = self._overhead_homography @ np.array([u, v, 1.0])
            x, y = pt[0] / pt[2], pt[1] / pt[2]
            return np.array([x, y, z_world])
        other = self._cameras.get(camera)
        if other is not None and other.mount == "fixed" and other.homography is not None:
            pt = np.asarray(other.homography, float) @ np.array([u, v, 1.0])
            return np.array([pt[0] / pt[2], pt[1] / pt[2], z_world])
        if other is not None and other.mount == "fixed" and other.model is not None:
            # Its ray to the plane z_world: the lens undone to convergence, and
            # no answer for a pixel past a wide lens's fold or above the horizon.
            from mrcf.motion_guard import surface_xy
            m = {**cast(dict, self.camera_model(camera, render_w, render_h)), "plane_z_m": z_world}  # it has a model
            xy = surface_xy(m, np.array([[u, v]], float))[0]
            if not np.isfinite(xy).all():
                raise ValueError(f"Pixel ({u:g}, {v:g}) of camera '{camera}' does not see the plane z = {z_world} m.")
            return np.array([xy[0], xy[1], z_world])
        nxt = super().pixel_to_world  # type: ignore[safe-super]  # RobotEnv's abstract one is caught just below
        if getattr(nxt, "__isabstractmethod__", False):         # RobotEnv's: there is no next one
            raise NotImplementedError(f"{type(self).__name__}: no pixel_to_world for camera '{camera}'.")
        return nxt(u, v, z_world, camera, render_w, render_h)

    # ── Sensors ────────────────────────────────────────────────────────────────

    def list_sensors(self) -> dict[str, str]:
        """RobotEnv.list_sensors: the rig's sensors (SENSORS), and the next class's."""
        found = dict(super().list_sensors())
        for name, s in self._sensors.items():
            found[name] = f"{s.kind}" + (f" ({s.unit})" if s.unit else "") + (f", shape {tuple(s.shape)}" if s.shape else "")
        return found

    def read_sensor(self, name: str):
        """RobotEnv.read_sensor: a sensor of the rig's, read the way its Sensor says; else the next class's."""
        if name in self._sensors:
            return read_sensor_value(self, name, self._sensors[name])
        return super().read_sensor(name)

    def sensor_of_kind(self, kind: str) -> "str | None":
        """The name of the rig's first sensor of this kind ("force", "gripper_contact", ...), or None."""
        return next((name for name, s in self._sensors.items() if s.kind == kind), None)

    def get_contact_force(self) -> float:
        """RobotEnv.get_contact_force: a "force" sensor's magnitude (a wrench's first three), else the next class's."""
        name = self.sensor_of_kind("force")
        if name is None:
            return super().get_contact_force()
        value = np.atleast_1d(np.asarray(self.read_sensor(name), float))
        return float(np.linalg.norm(value[:3]) if value.size > 1 else abs(value[0]))

    # ── The wrist servo ────────────────────────────────────────────────────────

    def _servo_tilt_offset(self, point_height_m: "float | None" = None) -> "np.ndarray | None":
        """
        Where the object is relative to the jaws, horizontally (metres), once
        the servo has it on GRIPPER_CAM_GRASP_PX with the gripper as it is now
        — (0, 0) at the calibration's own tilt and height. None when the
        calibration's tilt is not known.

        The SO-101's target pixel was set with the gripper leaning ~5 deg, over
        the middle of the board. Elsewhere the held tilt is different — 15 deg
        on rank 6, 35-50 on rank 8, up to 25 near the base — and aligning that
        same pixel there put the jaws 8, 25-41, and 17 mm off the piece
        (plane_geometry.aligned_offset). point_height_m: how high up the
        object the detector's point is (None: as on the calibration's object).
        """
        if self._gripper_cam_ref_tilt_deg is None or self._gripper_cam_ref_height_m is None:
            return None
        from mrcf.perception import plane_geometry
        ref_point = self._gripper_cam_ref_point_height_m
        point = ref_point if point_height_m is None else float(point_height_m)
        down_ref = self.tilted_rotation(self._gripper_cam_ref_tilt_deg).T @ np.array([0.0, 0.0, -1.0])  # type: ignore[attr-defined]  # the arm's own (module docstring)
        height = float(self.get_ee_pos()[2]) - self._work_plane_z
        try:
            return plane_geometry.aligned_offset(
                self.get_ee_rotation(), down_ref, self._gripper_cam_position_m - self._tcp(),
                self._gripper_cam_ref_height_m - ref_point, height - point)
        except ValueError:
            return None

    @property
    def wrist_servo_target_px(self) -> "tuple[float, float] | None":
        """
        capabilities.WristServo: GRIPPER_CAM_GRASP_PX — where the object shows in
        the wrist camera with the jaws over it — or None until the servo's
        calibration has measured it and its Jacobian.
        """
        px = getattr(self, "_gripper_cam_grasp_px", None)
        if px is None or getattr(self, "_gripper_cam_jacobian", None) is None:
            return None
        return px

    def wrist_to_surface(self, px, height_m: float = 0.0) -> "np.ndarray | None":
        """
        capabilities.WristSurface, from the servo's calibration: each pixel is
        where the servo would move the jaws to bring it onto
        GRIPPER_CAM_GRASP_PX (the Jacobian, scaled for how far the camera is
        from a plane height_m up against its calibration's), plus the step for
        the tilt (_servo_tilt_offset). One linear step of a perspective
        camera: a few percent off at the servo's own height a few cm from the
        grasp pixel, more further out or higher up. None uncalibrated.
        """
        if self.wrist_servo_target_px is None or self._gripper_cam_ref_height_m is None:
            return None
        self._require_kinematics()
        px = np.atleast_2d(np.asarray(px, dtype=float))
        t = self.gripper_frame()
        rot = t[:3, :3]
        here = t[:3, 3] + rot @ self._tcp()
        # Pinhole: metres per pixel go with the camera's distance from the
        # plane, which is the grasp point's height plus how far above it the
        # camera sits.
        above = float((rot @ (self._gripper_cam_position_m - self._tcp()))[2])
        ref = self._gripper_cam_ref_height_m - self._gripper_cam_ref_point_height_m + above
        scale = (float(here[2]) - self._work_plane_z - float(height_m) + above) / ref
        jac = np.asarray(self._gripper_cam_jacobian, dtype=float)
        moved = (jac @ (np.asarray(self._gripper_cam_grasp_px, dtype=float) - px).T).T * scale
        world = (rot[:, :2] @ moved.T).T[:, :2]
        lean = self._servo_tilt_offset(float(height_m))
        return here[:2] + world + (np.zeros(2) if lean is None else np.asarray(lean, float)[:2])

    def visual_servo_xy(
        self,
        detect,
        max_iters: int = 7,
        tol_px: float = 5.0,
        max_correction_m: float = 0.05,
        gain: float = 0.8,
        acquire_radius_px: float = 190.0,
        track_radius_px: float = 60.0,
        agree_px: float = 12.0,
        settle_s: float = 0.35,
        orientation: "np.ndarray | None" = None,
        max_step_m: float = 0.015,
        point_height_m: "float | None" = None,
    ) -> "tuple[bool, str]":
        """
        Correct the gripper's (x, y) using the gripper camera, holding z.

        `detect(frame_rgb, expected_px, radius_px) -> (u, v) | None` locates the
        target in the gripper camera's frame, searching within radius_px of
        expected_px.

        Why this is worth having at all: move_ee_to() closes its loop on
        forward kinematics, so it cannot see a TCP, URDF or homography error.
        Here the error is measured between the piece and the jaws in the same
        image, so all of those drop out. Only the RELATIVE correction has to be
        accurate, and locally it is, even when the absolute frame is skewed.

        Built to be wrong as rarely as possible, because a wrong detection
        moves the arm:

          acquire, then track  The first look searches wide (acquire_radius_px,
                               ~55 mm at hover): the scan's error is what this
                               corrects, and it has been a full square. After
                               each move the piece's new pixel is PREDICTED
                               from the arm's measured displacement, and later
                               looks search only near that (track_radius_px).
                               One fixed radius would leave the first look one
                               bad scan from failing.
          two frames agree     Every look detects on two frames and needs them
                               within agree_px; a glint or one frame of motion
                               blur does not survive that.
          consistency          A piece that turns up far from its predicted
                               pixel aborts the servo instead of being chased.

        Returns (aligned, message). Never raises on a detection failure: it
        reports and leaves the arm where it is, so the caller can proceed on
        kinematics alone.

        `orientation` is held through every correction (see move_ee_to). A
        correction is a translation; letting each one choose its own tilt would
        rotate the camera as well as move it, and the next image would then
        disagree with the prediction made from the arm's displacement.

        Any tilt (see _servo_tilt_offset). The target pixel is right only at
        the tilt it was calibrated at; at another, the arm first steps to where
        aligning it will leave the jaws — so the servo starts close and tracks
        a short way, rather than chasing a pixel 100 px off across the board —
        and once aligned steps back over the object. Both steps are plain
        moves; neither depends on the Jacobian. point_height_m: how high up the
        object the detector's point is, when known.
        """
        self._require_kinematics()
        if (self._gripper_cam_grasp_px is None
                or self._gripper_cam_jacobian is None
                or self._gripper_cam_ref_height_m is None):
            return False, f"gripper camera not calibrated — run {self.SERVO_CALIBRATION_TOOL}"

        def step_by(offset) -> bool:
            here = self.get_ee_pos()
            return self.move_ee_to(np.array([here[0] + offset[0], here[1] + offset[1], here[2]]),
                                   tol=0.004, max_step_deg=3.0, orientation=orientation)

        # Where the jaws should start from, given the scan put them over it.
        # Short of there (a reach limit) the servo just has further to go.
        lean = self._servo_tilt_offset(point_height_m)
        if lean is not None and float(np.linalg.norm(lean)) > 0.002:
            step_by(-lean)

        grasp_px = np.asarray(self._gripper_cam_grasp_px, dtype=float)
        jac = np.asarray(self._gripper_cam_jacobian, dtype=float)   # m per px @ ref height
        start = None
        err_px = np.zeros(2)
        expected, radius = grasp_px.copy(), acquire_radius_px

        def look(expected, radius):
            # Two sightings that agree, from up to four frames. The detector
            # misses the odd frame outright — from a 60 mm hover, one frame in
            # three — and quitting on the first miss would turn that into
            # "target lost" on every other look.
            hits = []
            for _ in range(4):
                frame = self.render_image(self._camera_name,
                                          width=self._render_w, height=self._render_h)
                found = detect(frame, tuple(expected), radius)
                if found is not None:
                    hits.append(np.asarray(found, dtype=float))
                    if len(hits) == 2:
                        break
            if len(hits) < 2:
                return None, "not found"
            if float(np.linalg.norm(hits[0] - hits[1])) > agree_px:
                return None, (f"two frames disagree by "
                              f"{np.linalg.norm(hits[0] - hits[1]):.0f} px")
            return (hits[0] + hits[1]) / 2.0, ""

        for i in range(max_iters):
            time.sleep(settle_s)          # let the arm stop before judging blur
            t = self.gripper_frame()
            here = t[:3, 3] + t[:3, :3] @ self._tcp()
            if start is None:
                start = here.copy()
            # Pinhole: metres per pixel scales with distance to the surface.
            height = max(float(here[2]) - self._work_plane_z, 1e-3)
            scale = height / self._gripper_cam_ref_height_m

            found, why = look(expected, radius)
            if found is None:
                stage = "acquiring" if i == 0 else "tracking"
                return False, f"target lost while {stage} (iteration {i + 1}: {why})"

            err_px = found - grasp_px
            moved_mm = float(np.linalg.norm(here[:2] - start[:2])) * 1000
            if float(np.linalg.norm(err_px)) <= tol_px:
                msg = (f"aligned to {np.linalg.norm(err_px):.1f} px "
                       f"after {i} correction(s), moved {moved_mm:.1f} mm")
                lean = self._servo_tilt_offset(point_height_m)
                if lean is None or float(np.linalg.norm(lean)) <= 0.001:
                    return True, msg
                lean_mm = float(np.linalg.norm(lean)) * 1000
                tilt = (f"tilt {self.gripper_tilt_deg():+.0f} deg against the calibration's "  # type: ignore[attr-defined]  # the arm's own
                        f"{self._gripper_cam_ref_tilt_deg:+.0f}")
                if not step_by(lean):
                    return False, f"{msg}, but could not step the {lean_mm:.0f} mm over it for the {tilt}"
                return True, f"{msg}; {tilt}: stepped {lean_mm:.0f} mm to be over it"

            # Image offset -> gripper-frame metres -> world metres. The camera
            # is fixed to the gripper, so the Jacobian is constant in the
            # gripper frame and the arm's own orientation does the rest.
            #
            # Sign: the Jacobian maps CAMERA motion to IMAGE motion (that is
            # how it was measured — step the arm, watch the piece shift). We
            # want the piece to appear at grasp_px instead of where it is, so
            # the image must move by (grasp_px - found) = -err_px, and the arm
            # must move by jac @ -err_px. Getting this backwards drives the
            # arm away from the piece until the guard below stops it.
            delta_gripper = jac @ (-err_px) * scale * gain
            # At most max_step_m per correction, looking again in between: a
            # long correction asked for in one move can stall the solver
            # partway (36 mm does), and a short move also keeps the predicted
            # pixel inside the tracking window.
            step = float(np.linalg.norm(delta_gripper))
            if step > max_step_m:
                delta_gripper = delta_gripper * (max_step_m / step)
            delta_world = t[:3, :3] @ np.array([delta_gripper[0], delta_gripper[1], 0.0])
            goal = np.array([here[0] + delta_world[0], here[1] + delta_world[1], here[2]])
            if np.linalg.norm(goal[:2] - start[:2]) > max_correction_m:
                return False, (f"correction would exceed {max_correction_m * 1000:.0f} mm — "
                               "refusing; the detection or the calibration is probably wrong")
            if not self.move_ee_to(goal, tol=0.004, max_step_deg=3.0, orientation=orientation):
                return False, f"could not reach the corrected pose (iteration {i + 1})"

            # Predict where the piece should now appear from where the arm
            # ACTUALLY went (move_ee_to stops within a tolerance, not exactly),
            # mapped back through the inverse Jacobian.
            t2 = self.gripper_frame()
            now = t2[:3, 3] + t2[:3, :3] @ self._tcp()
            moved_gripper = (t[:3, :3].T @ (now - here))[:2]
            expected = found + np.linalg.solve(jac * scale, moved_gripper)
            radius = track_radius_px

        return False, f"still {np.linalg.norm(err_px):.1f} px off after {max_iters} corrections"
