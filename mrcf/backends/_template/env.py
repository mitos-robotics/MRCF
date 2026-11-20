"""
env.py (mrcf/backends/<arm>/, copied from mrcf/backends/_template) — the arm's RobotEnv,
over its SDK.

TemplateEnv is a backends.rig.RigEnv — give it the gripper's pose
(gripper_frame) and drive the motors, and the cameras' geometry, the grasp
point, the gripper tag, the wrist servo and the sentry come with it — and a
RealRobotEnv, which opens the cameras (OpenCV, each on a reader thread that
keeps its newest frame).

Everything that talks to the arm goes through self._sdk, whose calls are
InMemoryArm's below. YOUR ARM: make_sdk() returns your SDK's client in its
place — or a thin adapter giving it these few methods — and the env above it
stays as it is. What the env keeps, each because it has bitten this codebase
(DEPLOYMENT_SPEC.md, "What you must implement"):

  - guard() before every command: a tripped interlock then stops the arm
    within one poll, mid-move, and a pause holds it and carries on after;
  - hold() stops the arm where it is, from any thread, without guard();
  - move_ee_to() says whether it arrived, and drives the grasp point
    (get_ee_pos, RigEnv's jaw model), not the flange;
  - the SDK under one lock, never held while guard() waits out a pause.

It also takes a stream (capabilities.ActionStreaming), so a learned policy
runs on it: the grasp point and the jaws' opening in, the same out as goals,
each checked before it goes (POLICIES.md).

Left to the arm when it has them (capabilities.py): approach_orientation_for
(OrientationSearch) when straight down does not reach everywhere,
get_contact_force from a force sensor, the links in arm_shape().
"""

from __future__ import annotations

import threading
import time

import numpy as np

from mrcf.backends.real_robot_env import RealRobotEnv
from mrcf.backends.rig import RigEnv

DOWN = np.diag([1.0, -1.0, -1.0])    # the flange's z (the jaws' approach) pointing at the table


def make_sdk(host):
    """
    The arm's SDK client, not yet connected (TemplateEnv.connect opens it).
    YOUR ARM: return yours for `host`. None: the in-memory stand-in.
    """
    if host is None:
        return InMemoryArm()
    raise NotImplementedError(f"No SDK for {host!r} yet: mrcf/backends/<arm>/env.py, make_sdk()")


class InMemoryArm:
    """
    A stand-in for the arm's SDK: a Cartesian arm in memory whose controller
    moves the flange in a straight line at the speed it was sent, in real
    time. What each call is on a UR through ur_rtde is beside it
    (DEPLOYMENT_SPEC.md, "Worked example: a UR5").
    """

    REACH_M = 0.85           # from the base

    def __init__(self):
        self._lock = threading.Lock()
        self._flange = np.eye(4)
        self._flange[:3, :3] = DOWN
        self._flange[:3, 3] = [0.35, 0.0, 0.35]
        self._goal: "np.ndarray | None" = None
        self._t0, self._duration_s = 0.0, 0.0
        self._width = 0.0

    def _now(self) -> np.ndarray:
        """Where the flange is (under the lock): along the line from where the move began."""
        if self._goal is None:
            return self._flange
        done = 1.0 if self._duration_s <= 0 else min(1.0, (time.monotonic() - self._t0) / self._duration_s)
        pose = self._goal.copy()
        pose[:3, 3] = self._flange[:3, 3] + done * (self._goal[:3, 3] - self._flange[:3, 3])
        if done >= 1.0:
            self._flange, self._goal = pose, None
        return pose

    def connect(self) -> None:
        """Open the connection.  UR: RTDEControlInterface(host), RTDEReceiveInterface(host), setTcp([0]*6)."""

    def disconnect(self) -> None:
        """Close it.  UR: stopScript(), disconnect() on both."""

    def flange_pose(self) -> np.ndarray:
        """The flange in the base frame, 4x4.  UR: getActualTCPPose() [x, y, z, rx, ry, rz], cv2.Rodrigues."""
        with self._lock:
            return self._now().copy()

    def joints(self) -> list:
        """Joint angles, radians.  UR: getActualQ()."""
        return []

    def reachable(self, pose: np.ndarray) -> bool:
        """Whether the flange can get to pose.  UR: getInverseKinematics(pose) solves, isPoseWithinSafetyLimits."""
        return float(np.linalg.norm(pose[:2, 3])) <= self.REACH_M

    def move_linear(self, pose: np.ndarray, speed_m_s: float) -> None:
        """Start a straight-line move of the flange to pose; returns at once.  UR: moveL(..., asynchronous=True)."""
        with self._lock:
            self._flange = self._now().copy()
            self._goal = np.asarray(pose, float).copy()
            self._t0 = time.monotonic()
            self._duration_s = float(np.linalg.norm(self._goal[:3, 3] - self._flange[:3, 3])) / speed_m_s

    def servo(self, pose: np.ndarray) -> None:
        """Go to pose now: one goal of a stream, a control tick away.  UR: servoL(pose, 0, 0, dt, lookahead, gain)."""
        with self._lock:
            self._flange, self._goal = np.asarray(pose, float).copy(), None

    def moving(self) -> bool:
        """Whether a move is under way.  UR: getAsyncOperationProgress() >= 0, or not isSteady()."""
        with self._lock:
            self._now()
            return self._goal is not None

    def stop_linear(self) -> None:
        """Stop where it is.  UR: stopL(acceleration), which decelerates by itself."""
        with self._lock:
            self._flange, self._goal = self._now().copy(), None

    def set_gripper(self, width_m: float) -> None:
        """Open or close the jaws to width_m.  A Robotiq on a UR: its URCap socket (port 63352), POS then GTO."""
        with self._lock:
            self._width = float(width_m)

    def gripper_width(self) -> float:
        """The jaws' opening, metres.  Robotiq: POS, through its own position-to-width table."""
        with self._lock:
            return self._width

    def gripper_holding(self) -> bool:
        """Whether the jaws stopped on an object.  Robotiq: OBJ is 1 or 2."""
        return False

    def tcp_wrench(self) -> np.ndarray:
        """Force and torque at the tool, (fx, fy, fz, tx, ty, tz).  UR: getActualTCPForce()."""
        return np.zeros(6)


class TemplateEnv(RigEnv, RealRobotEnv):
    """An arm on a rig, over its SDK. See the module docstring."""

    # The speed move_ee_to's max_step_deg is a fraction of: the step a full-speed
    # SO-101 move takes per cycle, which is what the policies' gentle steps are
    # measured against (GripperSpec.grasp_step_deg).
    FULL_STEP_DEG = 8.0
    # A controller may not yet say it is moving when asked just after a move is
    # sent (a UR's async moveL): a move counts as ended once it has been seen
    # moving and has stopped, or has not started this long after it was sent.
    START_S = 0.5
    # Every move is checked along its way against the gripper's box (config
    # GRIPPER_BOX, RigEnv's gripper_box), and one that would take it out is not
    # sent. YOUR ARM: a joint move (moveJ) checks envelope.joint_path through
    # the arm's forward kinematics, as the UR5's move_joints_to does.
    CHECKS_GRIPPER_BOX = True

    def __init__(self, sdk, *, sources=None, capture_sizes=None, control_hz: float = 125.0, speed_m_s: float = 0.10,
                 home_xyz=(0.35, 0.0, 0.25), park_xyz=(0.15, -0.35, 0.30), base_xy=(0.0, 0.0),
                 jaw_axis: str = "radial", stream_fps: float = 30.0, stream_bounds=None, robot_type: str = "my_arm",
                 **rig):
        # sources: the cameras RealRobotEnv opens, {name: OpenCV index or URL};
        # rig: RigEnv's (_init_rig) — the cameras' jobs and calibrations, every
        # Camera and Sensor (cameras=, sensors=), the gripper's geometry.
        self._init_rig(**rig)               # first: a wrong keyword fails before a camera is opened
        RealRobotEnv.__init__(self, cameras=sources, capture_sizes=capture_sizes, control_hz=control_hz,
                              work_plane_z=rig.get("work_plane_z", 0.0))
        self._sdk = sdk
        # The SDK one call at a time: hold() comes from whichever thread
        # tripped the interlock, and watchers read the arm mid-move.
        self._sdk_lock = threading.RLock()
        self._speed_m_s = float(speed_m_s)
        self._home_xyz = np.asarray(home_xyz, float)
        self._park_xyz = np.asarray(park_xyz, float)
        self._base_xy = np.asarray(base_xy, float)
        # Which way the jaws open on a straight-down move (RigEnv.down_rotation):
        # config JAW_AXIS, which the policies plan the room round an object for.
        if jaw_axis not in ("tangential", "radial"):
            raise ValueError(f"jaw_axis is 'tangential' or 'radial', not {jaw_axis!r}")
        self._jaw_axis = jaw_axis
        self._stream_fps, self._robot_type = float(stream_fps), str(robot_type)
        self._stream_bounds = stream_bounds          # safety.WorkspaceBounds a streamed goal stays inside

    # ── Connection ─────────────────────────────────────────────────────────────
    # It opens the arm and moves nothing; RigEnv starts the sentry as connect()
    # returns, and stops it before stop() lets go.

    def connect(self) -> None:
        with self._sdk_lock:
            self._sdk.connect()

    def stop(self) -> None:
        with self._sdk_lock:
            self._sdk.disconnect()
        super().stop()                       # and the cameras (RealRobotEnv)

    # ── Where the arm is ───────────────────────────────────────────────────────

    def gripper_frame(self) -> np.ndarray:
        with self._sdk_lock:
            return np.asarray(self._sdk.flange_pose(), float)

    def get_joint_angles(self) -> dict[str, float]:
        with self._sdk_lock:
            q = self._sdk.joints()
        return {f"joint_{i + 1}": float(v) for i, v in enumerate(q)}     # YOUR ARM: its joints' names

    def may_be_moving(self) -> bool:
        with self._sdk_lock:
            return bool(self._sdk.moving())

    @property
    def base_xy(self) -> np.ndarray:
        return self._base_xy

    def arm_shape(self):
        """
        Where the arm's body is, for the motion guard (motion_guard.py): points
        from base to tip, each with a radius. YOUR ARM: the links' frames from
        its joints (a UR's from its DH parameters); here only base, flange, jaws.
        """
        f = self.gripper_frame()
        base = np.r_[self._base_xy, self._work_plane_z]
        return [(base, 0.07), (f[:3, 3], 0.05), (self.get_ee_pos(), 0.03)]

    # ── Moving ─────────────────────────────────────────────────────────────────

    def move_ee_to(self, target_pos, tol: float = 0.01, max_iters: int = 300, step_size: float = 0.0,
                   max_step_deg: "float | None" = None, orientation=None) -> bool:
        """
        The grasp point to target_pos, with the jaws at `orientation` (straight
        down, turned as down_rotation turns them there, when None): a straight
        line the controller plans. True when it got
        there. max_step_deg slows it, as a fraction of FULL_STEP_DEG; max_iters
        and step_size are the SO-101's IK loop's and mean nothing here.
        """
        target = np.asarray(target_pos, float)
        goal = self._flange_for(target, self.down_rotation(target) if orientation is None
                                else np.asarray(orientation, float))
        with self._sdk_lock:
            if not self._sdk.reachable(goal):
                return False
        if self._line_refused_by_box(goal, goal[:3, :3]):
            return False
        speed = self._speed_m_s * (1.0 if max_step_deg is None else min(1.0, max_step_deg / self.FULL_STEP_DEG))
        dist = float(np.linalg.norm(target - self.get_ee_pos()))
        deadline = self.motion_clock() + 2.0 + 3.0 * dist / speed
        sent_at, sent_t, seen_moving = None, 0.0, False    # paused_s and the clock when last sent
        while self.motion_clock() < deadline:
            self.guard()                     # stopped: hold() has run, and this raises; paused: waits, then on
            with self._sdk_lock:
                if self._sdk.moving():
                    seen_moving = True
                elif sent_at != self.paused_s:   # not sent yet, or a pause held it since: (re)send
                    self._sdk.move_linear(goal, speed)
                    sent_at, sent_t, seen_moving = self.paused_s, self.motion_clock(), False
                elif seen_moving or self.motion_clock() - sent_t > self.START_S:
                    break                    # it ended by itself: there, or as near as the controller got
            time.sleep(1.0 / self._control_hz)
        return bool(np.linalg.norm(target - self.get_ee_pos()) < tol)

    def hold(self) -> None:
        with self._sdk_lock:
            self._sdk.stop_linear()          # a UR decelerates by itself: nothing to re-command

    def go_home(self) -> bool:
        return self.move_ee_to(self._home_xyz)     # YOUR ARM: a measured joint pose (UR: moveJ) is safer

    def go_park(self) -> bool:
        return self.move_ee_to(self._park_xyz)     # out of the overhead camera's view of the work surface

    # ── The gripper ────────────────────────────────────────────────────────────

    def set_gripper(self, width: float, n_steps: int = 200, max_step_pct: "float | None" = None) -> None:
        self.guard()
        self._held_width_m = None            # the jaws let go: RigEnv's grasp point stops using the held width
        with self._sdk_lock:
            self._sdk.set_gripper(float(np.clip(width, 0.0, self._gripper_max_open_m)))

    def get_gripper_width(self) -> float:
        with self._sdk_lock:
            return float(self._sdk.gripper_width())

    def gripper_loaded(self) -> bool:
        """
        capabilities.GripSensing: a "gripper_contact" sensor's reading if the rig
        has one (SENSORS), else the gripper's own sense of having stopped on something.
        """
        contact = self.sensor_of_kind("gripper_contact")
        if contact is not None:
            return bool(self.read_sensor(contact))
        with self._sdk_lock:
            return bool(self._sdk.gripper_holding())

    # ── capabilities.ActionStreaming: the stream a learned policy runs on ─────
    # The grasp point (x, y, z, metres) and the jaws' opening (metres) in, the
    # same out as goals, STREAM_FPS a second, and the wrist camera when it is
    # open. A model for this arm is trained on exactly this (POLICIES.md). YOUR
    # ARM: what its demonstrations record — a UR's joints (getActualQ, servoJ)
    # or its TCP (servoL) — with every goal checked here before it goes.

    STREAM_NAMES = ("x", "y", "z", "gripper")

    @property
    def stream_spec(self):
        """streaming.StreamSpec: what the stream carries, and so what a model for this arm is trained on."""
        from mrcf.streaming import StreamSpec, sensor_specs
        cameras = {}
        if self._camera_name is not None and self._camera_name in self._caps:
            cameras["wrist"] = (self._camera_name, *self._cameras[self._camera_name].size)
        return StreamSpec(fps=self._stream_fps, state=self.STREAM_NAMES, action=self.STREAM_NAMES, cameras=cameras,
                          robot_type=self._robot_type, sensors=sensor_specs(self))

    def stream_observation(self) -> dict:
        """The arm now, as stream_spec says: the grasp point and the opening, the wrist camera, the sensors."""
        from mrcf.streaming import IMAGE_PREFIX, STATE, sensor_observation
        spec = self.stream_spec
        out = {STATE: np.r_[self.get_ee_pos(), self.get_gripper_width()].astype(np.float32)}
        for key, (camera, width, height) in spec.cameras.items():
            out[IMAGE_PREFIX + key] = self.render_image(camera, width, height)
        out.update(sensor_observation(self, spec))
        return out

    def stream_action(self, action) -> "tuple[np.ndarray, str | None]":
        """
        One goal, sent at once (InMemoryArm.servo; a UR's servoL): the grasp
        point to (x, y, z), the jaws to the opening. A goal the arm cannot
        reach, or further outside stream_bounds than the grasp point is now, is
        held back — the arm stays, the jaws still follow — and why is said.
        (Further outside, not merely outside: an arm parked outside the bounds
        can still be streamed back in.) The interlock stops it as it stops any
        command. Returns the action as sent, and why any of it was held back.
        """
        goal = np.asarray(action, float).ravel()
        if goal.shape != (len(self.STREAM_NAMES),):
            raise ValueError(f"an action is {len(self.STREAM_NAMES)} values ({', '.join(self.STREAM_NAMES)}), "
                             f"not {goal.shape}")
        self.guard()
        here = self.get_ee_pos()
        why = None
        if self._outside(goal[:3]) > self._outside(here) + 1e-9:
            why = f"({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}) m is outside the workspace"
        elif box_why := self._step_refused_by_box(goal[:3]):
            why = f"({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}) m: {box_why}"
        width = float(np.clip(goal[3], 0.0, self._gripper_max_open_m))
        with self._sdk_lock:
            if why is None:
                pose = self._flange_for(goal[:3], self.gripper_frame()[:3, :3])
                if self._sdk.reachable(pose):
                    self._sdk.servo(pose)
                else:
                    why = f"({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}) m is out of reach"
            self._sdk.set_gripper(width)
        self._held_width_m = None            # as set_gripper: the jaws are the stream's now
        sent = np.r_[here if why else goal[:3], width]
        return sent.astype(np.float32), why

    def _outside(self, xyz) -> float:
        """How far the point is outside stream_bounds, metres summed over the axes (0: inside, or no bounds)."""
        b = self._stream_bounds
        if b is None:
            return 0.0
        return float(sum(max(0.0, lo - v) + max(0.0, v - hi)
                         for v, lo, hi in zip(xyz, (b.x_min, b.y_min, b.z_min), (b.x_max, b.y_max, b.z_max))))

    # ── Sensors ────────────────────────────────────────────────────────────────
    # What robot_config's SENSORS name to read them (Sensor.read). RigEnv does
    # the rest: read_sensor(), and a "force" one as get_contact_force().

    def tcp_wrench(self) -> np.ndarray:
        """The force and torque at the tool, (fx, fy, fz, tx, ty, tz), N and N·m."""
        with self._sdk_lock:
            return np.asarray(self._sdk.tcp_wrench(), float)
