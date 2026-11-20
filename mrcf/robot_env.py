"""
robot_env.py — Abstract interface for a robot simulation or hardware backend.

Implement RobotEnv to swap in any physics simulator (Isaac Sim, PyBullet,
Gazebo) or real robot without changing any framework code (skill_executor,
vlm, planner).

Methods are grouped into two tiers:

  Universal (abstract) — every backend must implement these:
    step, get_ee_pos, get_joint_angles, get_body_pos,
    get_movable_body_names, get_static_body_names,
    render_image, pixel_to_world

  Connection — connect() before the first command, stop() at the end. Both
    do nothing by default; a backend with hardware to open overrides them.

  Manipulation (optional) — only needed if your robot has a controllable
    end-effector and grasping mechanism:
    move_ee_to, set_gripper, get_gripper_width, go_home

    These raise NotImplementedError by default with a clear message.
    Mobile robots, humanoids, or other embodiments that don't use these
    primitives simply leave them unimplemented and register different skills.

Stopping (every backend that moves)
-----------------------------------
Each env has an `interlock` (interlock.py) that anything — a safety watcher on
another thread, a per-cycle check, the user — can trip. A backend calls
`self.guard()` before EVERY command it sends to the motors: when the interlock
is tripped, guard() calls hold() and raises MotionStopped, so the arm stops
within one control cycle wherever it is in a move, and stays stopped until the
interlock clears. Implement hold() to stop the arm where it is. A trip from a
pausing source (Interlock.pause_only — the motion guard's) instead holds the
arm until it clears and lets the move carry on; loops that time themselves use
motion_clock(), which leaves the pause out.

A trip also holds the arm at once, from the thread that tripped it, when the
arm may be moving (may_be_moving) — so a watcher's stop does not wait for the
control loop's next command. hold() runs once per trip either way, so it must
be safe to call from any thread.

Threads
-------
Policies drive the arm from one thread; perception and safety watchers may run
on others at the same time. So render_image() and camera_model() must be safe
to call from any thread while a motion runs, and must not disturb it (on a
serial motor bus: share the bus under a lock, and read cameras without going
through it). Motion commands themselves come from one thread at a time.

Example: new simulator backend
-------------------------------
    from mrcf.robot_env import RobotEnv

    class IsaacEnv(RobotEnv):
        def step(self, n=1): ...
        def get_ee_pos(self): ...
        # ... implement all abstract methods ...
        # Only implement move_ee_to / set_gripper if your robot uses them.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

import base64
import io
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType

import numpy as np

from mrcf import flight_recorder

# The keys RobotEnv.contact_force_limits may give, as safety reads them: the
# hard stop, the near miss logged, and the tighter stop on a final approach.
FORCE_LIMIT_KEYS = ("max_contact_force_n", "near_miss_force_n", "approach_contact_force_n")
from mrcf.interlock import Interlock, MotionStopped

# Two threads asking for a new env's interlock at once must get the same one.
_INTERLOCK_LOCK = threading.Lock()


class RobotEnv(ABC):
    """
    Simulator-agnostic interface for a robot environment.

    The framework (skill_executor, vlm) uses only this interface — no
    MuJoCo, ROS, or hardware-specific calls appear in framework code.
    """

    # ── Physics ────────────────────────────────────────────────────────────────

    @abstractmethod
    def step(self, n: int = 1) -> None:
        """Advance the simulation by n timesteps (or yield n control cycles)."""

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Open the arm: its motors, its cameras, whatever its SDK needs before the
        first command. Every program calls it once after make_env() and before
        anything moves. It must not move the arm: on an arm on a rig the sentry
        starts as it returns (mrcf/backends/rig.py), and nothing moves unwatched.
        The default has nothing to open.
        """

    def stop(self) -> None:
        """
        Close the arm: let go of the motors and the cameras. Called once as a
        program ends, and after an error too, so it must work on an arm that
        connected only partly. The default has nothing to close.
        """

    # ── Stopping ───────────────────────────────────────────────────────────────

    @property
    def interlock(self) -> Interlock:
        """This robot's stop flag (interlock.py), created on first use."""
        il = self.__dict__.get("_interlock")
        if il is None:
            with _INTERLOCK_LOCK:
                il = self.__dict__.get("_interlock")
                if il is None:
                    self.__dict__["_hold_lock"] = threading.Lock()
                    self.__dict__["_held_for_trip"] = 0
                    il = Interlock()
                    il.on_trip(self._freeze)
                    il.on_trip(lambda reason: flight_recorder.event("interlock", "trip", reason=reason))
                    self.__dict__["_interlock"] = il
        return il

    def may_be_moving(self) -> bool:
        """
        Whether the arm may be moving now, under a command it was sent. A trip
        from another thread holds the arm at once only then: holding an arm at
        rest re-commands it where gravity has sagged it to, so it would creep a
        little lower with every trip. The default, True, holds on every trip.
        """
        return True

    def _freeze(self, _reason: str = "") -> None:
        """The interlock has just tripped: hold the arm now, not at its next command."""
        if self.may_be_moving():
            self._hold_once()

    def _hold_once(self) -> None:
        """
        hold(), once per trip: the trip itself and every command it refuses
        after all want it. Counted held only once hold() has returned: one
        that raised (the tripping thread says so and goes on) is tried again
        by the next guard(), and while a pause lasts.
        """
        il = self.interlock
        trip = il.trip_count
        with self.__dict__["_hold_lock"]:
            if self.__dict__["_held_for_trip"] == trip or self.__dict__.get("_holding_trip") == trip:
                return
            self.__dict__["_holding_trip"] = trip
        try:
            self.hold()
            with self.__dict__["_hold_lock"]:
                self.__dict__["_held_for_trip"] = trip
        finally:
            with self.__dict__["_hold_lock"]:
                self.__dict__["_holding_trip"] = None

    def hold(self) -> None:
        """
        Stop moving and hold the current pose. Called once per trip of the
        interlock: as it trips, on the tripping thread — not necessarily the
        one driving the arm — or else by guard(). A real arm overrides this to
        command where it is now
        (a servo otherwise carries on to the last goal it was sent). The
        default does nothing, which suits a simulator that only moves when
        stepped. Must not call guard().
        """

    # A pause (Interlock.pause_only) that has not cleared after this long stops
    # the arm instead: the command is abandoned, as for any other trip. None:
    # it waits as long as the cause lasts — a person working beside the arm can
    # take minutes, and the move should still carry on once they are done.
    PAUSE_MAX_S: "float | None" = None

    def guard(self) -> None:
        """
        Call before every motion command. Returns when the arm may move;
        otherwise holds it (hold()) and raises MotionStopped — unless only
        pausing sources hold the interlock (Interlock.pause_only): then it holds
        the arm, waits for them to clear, and returns, so the move carries on
        where it was — however long that takes, unless PAUSE_MAX_S is set.
        Tripped meanwhile by a source that stops, it raises as any other trip does.
        """
        announce = self.__dict__.pop("_next_move", None)
        if announce is not None:
            try:
                announce()
            except Exception:                  # noqa: BLE001 — a message, never in the way of a move
                pass
        while True:
            try:
                self.interlock.check()
                return
            except MotionStopped as stop:
                self._hold_once()
                if not stop.pausing:
                    raise
                self._wait_out_pause(stop.reason)

    def on_next_move(self, announce: "Callable[[], None] | None") -> None:
        """
        announce() once, as the next motion command is checked (guard()): the
        moment the arm is about to move, whatever the action — for saying so
        (controller: "the arm moves now"). None forgets one set and not yet run.
        """
        if announce is None:
            self.__dict__.pop("_next_move", None)
        else:
            self.__dict__["_next_move"] = announce

    def _wait_out_pause(self, reason: str) -> None:
        il, start = self.interlock, time.monotonic()
        while not il.wait_clear(0.1):
            # stopping(), not "not only_pausing()": the pause clearing just
            # after the wait timed out is not a stop.
            if il.stopping():
                raise MotionStopped(il.reason or reason)
            if self.__dict__.get("_held_for_trip") != il.trip_count and self.may_be_moving():
                self._hold_once()            # the hold as it tripped failed: the arm may still be going
            if self.PAUSE_MAX_S is not None and time.monotonic() - start > self.PAUSE_MAX_S:
                raise MotionStopped(f"{il.reason or reason} — still there after {self.PAUSE_MAX_S:.0f} s")
        self.__dict__["_paused_s"] = self.paused_s + (time.monotonic() - start)

    @property
    def paused_s(self) -> float:
        """
        Seconds this arm has spent paused (guard()), in all. A motion loop that
        times itself measures time.monotonic() - paused_s, so a pause neither
        uses up its time limit nor looks like a joint that has stopped moving.
        """
        return float(self.__dict__.get("_paused_s", 0.0))

    def motion_clock(self) -> float:
        """time.monotonic() less the time spent paused: what motion loops time themselves by."""
        return time.monotonic() - self.paused_s

    @property
    def frame_scale(self) -> float:
        """
        Real metres per world-frame metre: 1.0 where the frame is exact. An
        arm whose kinematic frame reads short or long says so here, so a
        distance between two positions can be turned into a real clearance.
        """
        return 1.0

    @property
    def base_xy(self) -> np.ndarray:
        """
        Where the arm stands on the work surface, world frame (x, y), metres:
        the axis it swings about. Policies measure "near the base" and "far
        out" from here, and keep the gripper off it. The world origin by default.
        """
        return np.zeros(2)

    def zero_velocities(self) -> None:
        """
        Zero all joint velocities.

        Called before closing the end-effector so the controller can settle
        cleanly after a motion command.  No-op by default — real-robot backends
        typically don't have a global velocity state to reset.
        """

    def clamp_ctrl(self) -> None:
        """
        Clip all actuator commands to their declared control ranges.

        No-op by default — simulators that enforce limits internally don't
        need this.
        """

    # ── Sensing ────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_ee_pos(self) -> np.ndarray:
        """
        Controlled end-point XYZ in world frame (metres), shape (3,).

        For an arm this is the end-effector.  For a mobile robot this could
        be the tool mount, camera, or base position — whatever the skills
        treat as the controllable reference point.
        """

    @abstractmethod
    def get_joint_angles(self) -> dict[str, float]:
        """Joint positions as {joint_name: angle_rad} for all relevant joints."""

    def get_contact_force(self) -> float:
        """
        Contact force at the active end-point, in Newtons.

        Returns 0.0 by default.  Override if your robot has a tactile sensor,
        FSR, or force-torque sensor.
        """
        return 0.0

    @abstractmethod
    def get_body_pos(self, name: str) -> np.ndarray | None:
        """
        World-frame XYZ of a named body/object, shape (3,).

        Returns None if the name is not found.  On real hardware substitute
        object pose from your perception pipeline.
        """

    # ── Manipulation (optional) ────────────────────────────────────────────────
    # These methods are only required for robots with a controllable
    # end-effector and grasping mechanism.  They raise NotImplementedError
    # by default so that non-manipulation backends fail fast and clearly
    # rather than silently returning wrong values.

    def get_gripper_width(self) -> float:
        """
        End-effector opening in metres (0 = fully closed).

        Only required for robots with a grasping mechanism.  Override this
        if your robot has a gripper, hand, or similar actuated end-effector.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_gripper_width(). "
            "Override this method if your robot has a grasping mechanism."
        )

    def set_gripper(self, width: float, n_steps: int = 200, max_step_pct: "float | None" = None) -> None:
        """
        Command the end-effector to target opening (metres) and settle n_steps cycles.
        max_step_pct: close gently — the most the jaws travel per cycle, in
        percent of their range (the policies pass it when closing onto an object).

        Only required for robots with a grasping mechanism.  Override this
        if your robot has a gripper, hand, or similar actuated end-effector.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement set_gripper(). "
            "Override this method if your robot has a grasping mechanism."
        )

    def squeeze_gripper(self) -> "tuple[float, bool]":
        """
        Firm up a grip after set_gripper() has closed. Returns (opening_m,
        holding). Real grippers override this with a sensed answer; the default
        infers holding from the opening, which is exact in simulation.
        """
        w = self.get_gripper_width()
        return w, w > 0.001

    def move_ee_to(
        self,
        target_pos: np.ndarray,
        tol: float = 2e-3,
        max_iters: int = 300,
        step_size: float = 0.15,
        max_step_deg: "float | None" = None,
        orientation: "np.ndarray | None" = None,
    ) -> bool:
        """
        Move the controlled end-point to target_pos.

        Blocks until the end-point is within tol metres of target or max_iters
        iterations are exhausted.  Returns True if the target was reached
        within tolerance, False otherwise.

        The policies also pass orientation — the gripper's 3x3 orientation to
        hold through the move (from capabilities.OrientationSearch), None to
        choose one — and max_step_deg: go gently, the most any joint moves a
        control cycle (an arm that commands speeds reads it as a slower move).
        mrcf/backends/contract.py checks a backend takes them.

        Only required for robots with Cartesian end-point control (e.g. an arm
        with IK).  Mobile robots or other embodiments should implement their own
        motion primitive and register it as a skill instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement move_ee_to(). "
            "Override this method if your robot uses Cartesian end-point control."
        )

    def go_home(self) -> "bool | None":
        """
        Move the robot to a canonical safe starting configuration.

        What 'home' means is robot-specific: for an arm it might be gripper-up
        and clear of the workspace; for a mobile robot it might be a neutral
        stance or a known waypoint.

        No-op by default.  Override to define a home pose for your robot.
        """

    def go_park(self) -> "bool | None":
        """
        Move out of the fixed camera's view of the work surface, so a scan
        sees all of it; whether the arm got there. Wherever this arm keeps
        clear of that camera — go_home() by default.
        """
        return self.go_home()

    def pan_only_to(self, target_pan: float) -> None:
        """
        Rotate the primary rotation joint to target_pan (radians) while holding
        all other joints at their current setpoints.

        Used during pick-place transits to avoid IK failures on large horizontal
        arcs.  No-op by default — override for robots where this optimisation
        applies (e.g. a single-arm manipulator with a shoulder-pan joint).
        """

    # ── Named actuator access ──────────────────────────────────────────────────

    def set_ctrl(self, actuator_name: str, value: float) -> None:
        """
        Set a named actuator command.

        No-op by default — implement if your backend supports named actuators.
        """

    def get_ctrl(self, actuator_name: str) -> float:
        """Read a named actuator's current command.  Returns 0.0 by default."""
        return 0.0

    # ── Scene introspection ────────────────────────────────────────────────────

    @abstractmethod
    def get_movable_body_names(self) -> list[str]:
        """
        Names of all bodies that can be freely moved.

        Used by discover_scene() to tell the VLM what objects exist.
        On real hardware, return the list of objects from your detector.
        """

    @abstractmethod
    def get_static_body_names(self, exclude_keywords: "set[str] | None" = None) -> list[str]:
        """
        Names of non-movable bodies — potential interaction targets.

        `exclude_keywords`: body names containing any of these substrings are
        omitted (e.g. robot links: "shoulder", "elbow").
        """

    # ── Camera ────────────────────────────────────────────────────────────────

    @abstractmethod
    def render_image(self, camera: str, width: int, height: int) -> np.ndarray:
        """Render a named camera to an (H, W, 3) uint8 RGB array."""

    @abstractmethod
    def pixel_to_world(
        self,
        u: float,
        v: float,
        z_world: float,
        camera: str,
        render_w: int = 640,
        render_h: int = 480,
    ) -> np.ndarray:
        """
        Back-project pixel (u, v) to a world point on the plane z = z_world.

        Returns shape (3,) world point [x, y, z_world].

        Real-robot implementation: use OpenCV camera calibration (intrinsics K
        + extrinsics R, t) to compute the ray–plane intersection.
        """

    def unchecked_facts(self) -> tuple:
        """
        The arm's facts a pick depends on that have not been checked on the
        rig (mrcf/arm_facts.py): () for an arm with nothing physical to be
        wrong about (in memory, in a test). RigEnv says its own.
        """
        return ()

    def camera_model(self, camera: str, width: int, height: int) -> "dict | None":
        """
        Where a camera is and how it sees, so objects can be measured on the
        work surface from its images (plane_geometry.PlaneView.from_camera_model).
        None when this backend does not know — the default.

            K            3x3 intrinsics for a width x height image
            D            distortion coefficients (OpenCV order)
            T_world_cam  4x4 camera -> world transform, metres, NOW: a camera on
                         the arm moves, so call this with the frame it describes
            plane_z_m    the work surface's height in the world frame
            scale        real mm per world-frame mm (1.0 where the frame is exact)
        """
        return None

    # The arm's own contact limits, where its force reading needs its own
    # (safety.ExecutionMonitor; keys: FORCE_LIMIT_KEYS, newtons): a UR's
    # estimate moves 35-74 N in free air, where an SO-101 stops at 25. Empty:
    # the framework's defaults. Replaced, never changed in place.
    contact_force_limits: "Mapping[str, float]" = MappingProxyType({})

    def zero_contact_force(self) -> None:
        """
        get_contact_force() counted from now: for an arm whose reading is its
        change since a move began (a UR's), now is the start. Nothing for an
        arm whose reading is absolute — the default.
        """

    def link_frames(self) -> "list[np.ndarray] | None":
        """
        Every link's frame NOW, base first, in the world frame (4x4, metres):
        whatever is stuck on any of them — a tag a fixed camera finds the arm
        by (perception/camera_from_arm.py) — has a pose from the joints. None
        when this backend does not know — the default.
        """
        return None

    def arm_shape(self) -> "list[tuple[np.ndarray, float]] | None":
        """
        Where the arm's body is NOW, so a camera can tell it from anything else
        that moves (motion_guard.py): points from its base to its tip, in the
        world frame (metres), each with the radius of the body around it —
        consecutive points are joined. Generous rather than tight: what it
        misses of the arm, a guard takes for a hand. None when this backend
        does not know — the default.
        """
        return None

    # ── Camera helpers ────────────────────────────────────────────────────────

    def list_cameras(self) -> list[str]:
        """
        Return the names of all available cameras.

        Default returns an empty list.  Override to advertise the cameras your
        backend supports (e.g. ["overhead", "wrist", "side"]).
        vlm.discover_scene() and vlm.perceive() use this to decide how many
        images to send to the VLM.
        """
        return []

    def render_to_base64(self, camera: str, width: int = 640, height: int = 480) -> str:
        """
        Render a camera view and return a base64-encoded PNG string.

        Default implementation calls render_image() and encodes the result.
        Override if your backend can produce encoded images more efficiently.
        """
        from PIL import Image  # soft import — only needed when actually called
        pixels = self.render_image(camera, width, height)
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Generic sensor access ─────────────────────────────────────────────────

    def list_sensors(self) -> dict[str, str]:
        """
        Return metadata for all named non-image sensors.

        Returns a dict of {sensor_name: type_description}, e.g.::

            {
                "left_finger_force":  "float (N)",
                "right_finger_force": "float (N)",
                "wrist_torque":       "array(3,) (Nm)",
                "imu_accel":          "array(3,) (m/s²)",
            }

        Used by framework code to include live sensor readings in planning
        prompts and episode logs.  Default returns an empty dict — override
        to advertise your robot's sensors.
        """
        return {}

    def read_sensor(self, name: str) -> "float | np.ndarray":
        """
        Read a named non-image sensor by name.

        Returns a scalar float or numpy array depending on the sensor.
        Raises KeyError for unknown sensor names so callers can handle
        missing sensors cleanly.

        Override in your backend to expose force/torque, pressure, IMU,
        depth, or any other non-image sensor data.
        """
        raise KeyError(
            f"Sensor '{name}' not found.  "
            f"Available sensors: {list(self.list_sensors())}"
        )

    # ── Viewer ────────────────────────────────────────────────────────────────

    def is_viewer_running(self) -> bool:
        """
        True while an interactive viewer is open.

        Returns True by default (headless backends always continue).
        """
        return True
