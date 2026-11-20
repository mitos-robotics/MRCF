"""
env.py (mrcf/backends/ur5) — the UR5's RobotEnv, over its controller (arm.URArm).

UR5Env is a backends.rig.RigEnv — the cameras' geometry, the grasp point, the
gripper tag and the sentry come with it, given the flange's pose — and a
RealRobotEnv, which opens the cameras. It started as a copy of
mrcf/backends/_template/env.py and keeps its rules (DEPLOYMENT_SPEC.md, "What
you must implement"):

  - guard() before every command, and every cycle of a move: a tripped
    interlock stops the arm within one cycle, and a pause holds it and
    carries on after;
  - hold() stops the arm where it is, from any thread, without guard();
  - move_ee_to() says whether it arrived, and drives the grasp point
    (get_ee_pos, RigEnv's jaw model), not the flange;
  - the controller under one lock, never held while guard() waits out a pause.

What a UR adds:

  - the controller plans every move, so there is no IK loop here: a straight
    line (moveL) for move_ee_to, the wrist turned in place first by a joint
    move when the jaws must turn, and a joint move (moveJ) for home and park,
    each sent once and watched;
  - the arm's own safety system: a protective stop, an emergency stop or the
    control script stopping ends a move with MotionStopped and its reason;
  - the pendant's speed slider slows every move, and the time a move is given
    grows with it;
  - six joints reach straight down over the whole table, so
    approach_orientation_for (OrientationSearch) is straight down, the jaws
    turned as asked, and no lean — and none with the elbow nearly straight
    (ELBOW_STRAIGHT_MARGIN_DEG);
  - freedrive (disable_torque/enable_torque), for the overhead camera's
    calibration by touching;
  - the links' positions from the joints (arm_shape), for the motion guard.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from mrcf import envelope
from mrcf.backends.real_robot_env import RealRobotEnv
from mrcf.backends.rig import RigEnv
from mrcf.interlock import MotionStopped

from . import kinematics

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def _angle(r0, r1) -> float:
    """The angle between two orientations, radians."""
    return float(np.arccos(np.clip((np.trace(np.asarray(r0).T @ np.asarray(r1)) - 1.0) / 2.0, -1.0, 1.0)))


# No pose with the elbow nearer straight than this (deg): out there the arm is at
# the edge of its reach and the CB3's force estimate swings wildly. 2026-11-20
# 16:35, putting a screwdriver in the box 0.77 m out, 0.34 m up: the elbow went
# -31 to -9 deg, the "contact" force +60, -71, +95 N in 0.2 s touching nothing,
# and the move was halted at 120 N.
ELBOW_STRAIGHT_MARGIN_DEG = 25.0


class UR5Env(RigEnv, RealRobotEnv):
    """A UR5 on a rig. See the module docstring."""

    # The speed move_ee_to's max_step_deg is a fraction of: the step a full-speed
    # SO-101 move takes per cycle, which the policies' gentle steps are measured
    # against (GripperSpec.grasp_step_deg).
    FULL_STEP_DEG = 8.0
    # A move has ended when the controller says so (URArm.move_ended). Should it
    # never say, one that has not been seen moving this long after it was sent
    # counts as ended (the controller reports motion ~30 ms after a moveL).
    START_S = 2.0
    # How long the jaws get to finish opening or closing.
    GRIPPER_S = 4.0
    # After the jaws stop, how long they still count as moving for the sentry
    # (may_be_moving): the last frames it compares still show them moving.
    JAWS_TAIL_S = 0.5
    # A move that turns the jaws further than this turns them in place first.
    TURN_IN_PLACE_DEG = 5.0
    # A force zero is the median of this long a run of readings (force_settle_s
    # after the last move ended): one reading can land mid-change.
    WRENCH_ZERO_S = 0.1
    # back_off: slowly, since it starts pressing on something; stopped by a
    # force this much above the press it starts from (something else in the way).
    BACK_OFF_SPEED_M_S = 0.02
    BACK_OFF_FORCE_N = 25.0
    # Every move is checked along its way against the gripper's box (config
    # GRIPPER_BOX, RigEnv's gripper_box), and one that would take it out is not sent.
    CHECKS_GRIPPER_BOX = True

    def __init__(self, sdk, *, sources=None, capture_sizes=None, control_hz: float = 125.0,
                 speed_m_s: float = 0.05, joint_speed_rad_s: float = 0.4,
                 home_joints_deg=None, park_joints_deg=None, home_xyz=None, park_xyz=None,
                 base_xy=(0.0, 0.0), jaw_axis: str = "tangential", transit_height_m: float = 0.10,
                 stream_fps: float = 30.0, stream_bounds=None, robot_type: str = "ur5",
                 contact_force_limits: "dict | None" = None, force_settle_s: float = 0.35, **rig):
        # sources: the cameras RealRobotEnv opens, {name: OpenCV index or URL};
        # rig: RigEnv's (_init_rig), the gripper's box among them.
        self._init_rig(**rig)               # first: a wrong keyword fails before a camera is opened
        RealRobotEnv.__init__(self, cameras=sources, capture_sizes=capture_sizes, control_hz=control_hz,
                              work_plane_z=rig.get("work_plane_z", 0.0))
        self._sdk = sdk
        self._sdk_lock = threading.RLock()
        self._carried: "tuple[np.ndarray, float] | None" = None   # what the jaws hold, for arm_shape (capabilities.Carries)
        self._speed_m_s = float(speed_m_s)
        self._joint_speed = float(joint_speed_rad_s)
        self._home_q = None if home_joints_deg is None else np.radians(np.asarray(home_joints_deg, float))
        self._park_q = None if park_joints_deg is None else np.radians(np.asarray(park_joints_deg, float))
        self._home_xyz = None if home_xyz is None else np.asarray(home_xyz, float)
        self._park_xyz = None if park_xyz is None else np.asarray(park_xyz, float)
        self._base_xy = np.asarray(base_xy, float)
        if jaw_axis not in ("tangential", "radial"):
            raise ValueError(f"jaw_axis is 'tangential' or 'radial', not {jaw_axis!r}")
        self._jaw_axis = jaw_axis
        self._transit_z = float(rig.get("work_plane_z", 0.0)) + float(transit_height_m)
        self._stream_fps, self._robot_type = float(stream_fps), str(robot_type)
        self._stream_bounds = stream_bounds
        self._wrench_zero: np.ndarray = np.zeros(6)   # tcp_wrench() as the last move began (contact_wrench)
        # How long after a move the force reading takes to settle before a zero
        # is read from it (robot_config FORCE_SETTLE_S; _settled_wrench), and
        # when (time.monotonic()) the last move ended.
        self._force_settle_s = max(0.0, float(force_settle_s))
        self._moved_at = 0.0
        self._jaws_until = 0.0              # the jaws count as moving until then, for the sentry (set_gripper)
        self._jaws_going = False            # set_gripper is waiting on them: a hold stops them too
        # What the safety monitor stops a skill at, for this arm's force reading
        # (safety.ExecutionMonitor takes them unless a deployment sets its own).
        self.contact_force_limits = dict(contact_force_limits or {})

    # ── Connection ─────────────────────────────────────────────────────────────
    # It opens the controller and moves nothing; RigEnv starts the sentry as
    # connect() returns, and stops it before stop() lets go.

    def connect(self) -> None:
        with self._sdk_lock:
            self._sdk.connect()

    def stop(self) -> None:
        with self._sdk_lock:
            if self._sdk.connected:
                self._sdk.disconnect()
        super().stop()                       # and the cameras (RealRobotEnv)

    # ── Where the arm is ───────────────────────────────────────────────────────

    def gripper_frame(self) -> np.ndarray:
        with self._sdk_lock:
            return np.asarray(self._sdk.flange_pose(), float)

    def link_frames(self) -> list:
        """The base and each joint's frame after it (kinematics.frames), from the joints now: 7, the flange last."""
        with self._sdk_lock:
            q = self._sdk.joints()
        return kinematics.frames(q)

    def get_joint_angles(self) -> dict[str, float]:
        with self._sdk_lock:
            q = self._sdk.joints()
        return {name: float(v) for name, v in zip(JOINT_NAMES, q)}

    def may_be_moving(self) -> bool:
        """
        Moving under a command: the arm, or its jaws (set_gripper waiting on
        them) — the fingers' own movement by an object is not something else
        moving. In freedrive a hand moves it: a trip must not hold it against that hand.
        """
        if time.monotonic() < self._jaws_until:
            return True
        with self._sdk_lock:
            return bool(self._sdk.moving()) and not getattr(self._sdk, "teaching", False)

    @property
    def base_xy(self) -> np.ndarray:
        return self._base_xy

    def arm_shape(self):
        """The body from base to jaws (kinematics.body_points, from the joints now), for the motion guard."""
        with self._sdk_lock:
            q = self._sdk.joints()
        shape = kinematics.body_points(q)
        ee = self.get_ee_pos()
        shape.append((ee, 0.045))          # the gripper, out to the jaws
        if self._carried is not None:
            # And what it carries, where its outline is now (capabilities.Carries): not something moving.
            jaw = self.jaw_direction(self.get_ee_rotation())
            if jaw is not None:
                square = np.array([-jaw[1], jaw[0]])
                pts = self._carried[0]
                for u, v in pts[::max(1, len(pts) // 24)]:
                    shape.append((np.r_[ee[:2] + u * jaw + v * square, ee[2]], 0.03))
                shape.append((ee, 0.045))
        return shape

    def carrying(self, outline_uv, grip_width_m: float) -> None:
        """capabilities.Carries: held from now until the jaws open wider than grip_width_m."""
        self._carried = None if outline_uv is None else (np.asarray(outline_uv, float).reshape(-1, 2),
                                                          float(grip_width_m))

    # ── Orientation ────────────────────────────────────────────────────────────

    def approach_orientation_for(self, targets: list, gap_m: "float | None" = None, tols: "list | None" = None,
                                 flip: bool = False, turn_deg: float = 0.0) -> "tuple | None":
        """
        capabilities.OrientationSearch: (rotation, 0.0) — straight down, the
        jaws as down_rotation (RigEnv's) turns them at the first target — when the arm
        reaches every target with it, near the joints it has now, the elbow no
        nearer straight than ELBOW_STRAIGHT_MARGIN_DEG; else None.
        """
        if not targets:
            return None
        rot = self.down_rotation(targets[0], flip=flip, turn_deg=turn_deg)
        tcp = self._tcp(gap_m)
        with self._sdk_lock:
            q = self._sdk.joints()
        for target in targets:
            goal = np.eye(4)
            goal[:3, :3], goal[:3, 3] = rot, np.asarray(target, float) - rot @ tcp
            q = kinematics.nearest(goal, q)
            if q is None:
                return None
            if abs(float(np.degrees(q[2]))) < ELBOW_STRAIGHT_MARGIN_DEG:
                return None              # the elbow nearly straight: out at the edge of its reach
        return rot, 0.0

    # ── Moving ─────────────────────────────────────────────────────────────────

    def move_ee_to(self, target_pos, tol: float = 0.01, max_iters: int = 300, step_size: float = 0.0,
                   max_step_deg: "float | None" = None, orientation=None) -> bool:
        """
        The grasp point to target_pos, the jaws at `orientation` (straight down,
        down_rotation, when None), in a straight line the controller plans.
        True when it got there. max_step_deg slows it, as a fraction of
        FULL_STEP_DEG; max_iters and step_size are the SO-101's IK loop's and
        mean nothing here.
        """
        target = np.asarray(target_pos, float)
        rot = self.down_rotation(target) if orientation is None else np.asarray(orientation, float)
        goal = self._flange_for(target, rot)
        with self._sdk_lock:
            if not self._sdk.reachable(goal):
                return False
        if max_step_deg is None and (q := self._transit_joints(target, goal)) is not None:
            self.move_joints_to(q)
            return bool(np.linalg.norm(target - self.get_ee_pos()) < tol)
        if self._line_refused_by_box(goal, rot):      # the line as it will run: the jaws already turned to rot
            return False
        # A line that turns the wrist is as slow as the turn, which a UR runs at
        # about 4x the tool speed in rad/s (5 s for 60 deg at 0.05 m/s, URSim):
        # the wrist turns where it is first (a joint move, wrist 3 alone when
        # the jaws point down, twice as fast), and the line holds the orientation.
        if _angle(self.get_ee_rotation(), rot) > np.radians(self.TURN_IN_PLACE_DEG):
            if not self._turn_in_place(rot):
                return False
        speed = self._speed_m_s * (1.0 if max_step_deg is None else min(1.0, max_step_deg / self.FULL_STEP_DEG))
        dist = float(np.linalg.norm(target - self.get_ee_pos()))
        turn = _angle(self.get_ee_rotation(), rot)
        self._run(lambda: self._sdk.move_linear(goal, speed), 2.0 + 3.0 * max(dist, turn) / speed)
        return bool(np.linalg.norm(target - self.get_ee_pos()) < tol)

    # A move across — TRANSIT_MIN_M or more sideways, both ends TRANSIT_ABOVE_M
    # or more above the work plane, at full speed — goes as one joint move, the
    # wrist turning on the way. A UR's lines run at speed_m_s (5 cm/s on this
    # rig) and a line that turns the wrist first turns it in place; its joints
    # run at joint_speed_rad_s. 2026-11-13: the user asked why the twist does
    # not happen on the way. A joint move curves, so only where its path keeps
    # the jaws pointing down (within TRANSIT_TILT_DEG), the gripper in its box,
    # and the grasp point no lower than TRANSIT_DIP_M below the lower end; else
    # the line. Descents and gentle steps are lines, as before.
    TRANSIT_MIN_M = 0.05
    TRANSIT_ABOVE_M = 0.05
    TRANSIT_DIP_M = 0.01
    TRANSIT_TILT_DEG = 10.0

    def _transit_joints(self, target: np.ndarray, goal: np.ndarray) -> "np.ndarray | None":
        """The joints a move across to target (the flange at goal) goes to as one joint move; None: as a line."""
        here = self.get_ee_pos()
        if float(np.hypot(*(target[:2] - here[:2]))) < self.TRANSIT_MIN_M:
            return None
        if min(here[2], target[2]) < self._work_plane_z + self.TRANSIT_ABOVE_M:
            return None
        with self._sdk_lock:
            now = np.asarray(self._sdk.joints(), float)
        q = kinematics.nearest(goal, now)
        if q is None:
            return None
        with self._sdk_lock:
            if not self._sdk.joints_reachable(q):
                return None
        from mrcf import envelope
        path = envelope.joint_path(now, q, kinematics.fk)
        tcp, lowest = self._tcp(), min(here[2], target[2]) - self.TRANSIT_DIP_M
        down = np.cos(np.radians(self.TRANSIT_TILT_DEG))
        for T in path:
            if float((T[:3, :3] @ tcp + T[:3, 3])[2]) < lowest or float(-T[2, 2]) < down:
                return None
        if self._gripper_box is not None and self._gripper_box.refuses(path):
            return None                              # said by the line, if it is refused too
        return q

    def back_off(self, distance_m: float) -> bool:
        """
        capabilities.BackOff: the grasp point straight up by distance_m, at
        BACK_OFF_SPEED_M_S, the jaws as they are. 2026-11-13: a pick stopped at
        47.5 N with a finger on its block, and the arm held there, pressing,
        until the power was cut. A move's force is counted from its start,
        here mid-press, so letting the press go reads as a force as large as
        the press: the limit is that and BACK_OFF_FORCE_N more.
        """
        press = float(np.linalg.norm(self.contact_wrench()[:3]))      # against the stopped move's zero
        rot = self.get_ee_rotation()
        target = self.get_ee_pos() + np.array([0.0, 0.0, float(distance_m)])
        goal = self._flange_for(target, rot)
        with self._sdk_lock:
            if not self._sdk.reachable(goal):
                return False
        if self._line_refused_by_box(goal, rot):
            return False
        self._run(lambda: self._sdk.move_linear(goal, self.BACK_OFF_SPEED_M_S),
                  2.0 + 3.0 * float(distance_m) / self.BACK_OFF_SPEED_M_S, limit_n=press + self.BACK_OFF_FORCE_N)
        return bool(np.linalg.norm(target - self.get_ee_pos()) < 0.003)

    def _turn_in_place(self, rot) -> bool:
        """Turn the jaws to rot about the grasp point where it is now, by a joint move."""
        goal = self._flange_for(self.get_ee_pos(), rot)
        with self._sdk_lock:
            q = kinematics.nearest(goal, self._sdk.joints())
        return q is not None and self.move_joints_to(q)

    def move_joints_to(self, q) -> bool:
        """A joint move to q (radians, the joints in JOINT_NAMES' order); True when there."""
        q = np.asarray(q, float)
        with self._sdk_lock:
            if not self._sdk.joints_reachable(q):
                print("UR5: the controller will not take that joint move (outside its joint or safety limits).")
                return False
            now = np.asarray(self._sdk.joints(), float)
            far = float(np.max(np.abs(q - now)))
        if self._gripper_box is not None and self._refused_by_box(envelope.joint_path(now, q, kinematics.fk)):
            return False
        self._run(lambda: self._sdk.move_joints(q, self._joint_speed), 2.0 + 3.0 * far / self._joint_speed)
        with self._sdk_lock:
            return bool(np.max(np.abs(q - np.asarray(self._sdk.joints()))) < 0.005)

    def _run(self, send, budget_s: float, limit_n: "float | None" = None) -> None:
        """
        Send a move and watch it to its end: guard() and the arm's faults every
        cycle, sent again after a pause held it. Given budget_s at full speed,
        stretched by the speed slider; past that, held and left. limit_n: the
        force it stops at, in place of the arm's max_contact_force_n.
        """
        sent_at, sent_t, seen_moving = None, 0.0, False    # paused_s and the clock when last sent
        limit = limit_n if limit_n is not None else self.contact_force_limits.get("max_contact_force_n")
        # The move's own start, for its force, before anything is sent (guard()
        # runs first, at the loop's top): once the reading has settled after
        # the last move. Not taken again on a resend after a pause: a press
        # already on still counts.
        self._wrench_zero = self._settled_wrench()
        start = self.motion_clock()
        try:
            while True:
                self.guard()                     # stopped: hold() has run, and this raises; paused: waits, then on
                pressed = None
                with self._sdk_lock:
                    why = self._sdk.fault()
                    if why is not None:
                        raise MotionStopped(why)
                    if limit and sent_at is not None:
                        # Every move's own, not only a pick's or a place's (the safety
                        # monitor runs inside those): home, park, a look's trip.
                        force = float(np.linalg.norm((np.asarray(self._sdk.tcp_wrench(), float)
                                                      - self._wrench_zero)[:3]))
                        if force > limit:
                            self._sdk.stop()
                            pressed = f"contact force {force:.0f} N on a move (limit {limit:.0f} N)"
                if pressed is not None:
                    # Said on the interlock — a stop, so the next command looks again
                    # — and cleared at once: it is this move that is over.
                    self.interlock.trip(pressed, source="ur5:force")
                    self.interlock.clear("ur5:force")
                    raise MotionStopped(pressed)
                with self._sdk_lock:
                    scaling = max(self._sdk.speed_scaling(), 0.02)
                    if self.motion_clock() - start > budget_s / scaling:
                        self._sdk.stop()
                        return
                    if sent_at != self.paused_s:     # not sent yet, or a pause held (stopped) it since: (re)send
                        if not send():
                            return               # the controller refused it
                        sent_at, sent_t, seen_moving = self.paused_s, self.motion_clock(), False
                    elif self._sdk.move_ended():
                        return                   # there, or as near as the controller got
                    elif self._sdk.moving():
                        seen_moving = True
                    elif not seen_moving and self.motion_clock() - sent_t > self.START_S:
                        return                   # never seen to start, nor said to end
                time.sleep(1.0 / self._control_hz)
        except BaseException:
            # Left with the move still under way — Ctrl-C, the connection lost,
            # any error but a stop (whose hold() has run): the controller is
            # running it by itself, so it is stopped here, not left to finish
            # unwatched.
            try:
                with self._sdk_lock:
                    self._sdk.stop()
            except Exception:                 # noqa: BLE001 — the first error is the one to raise
                pass
            raise
        finally:
            self._moved_at = time.monotonic()

    def hold(self) -> None:
        with self._sdk_lock:
            if self._sdk.connected:
                self._sdk.stop()             # a UR decelerates by itself: nothing to re-command
                if self._jaws_going:
                    # The fingers too, while set_gripper waits on them: they close at
                    # their own force whatever the arm does, and the trip may be a
                    # hand at the jaws. It sends them on again once the pause clears.
                    # Not once they have stopped (on an object): a halt is no grip.
                    try:
                        self._sdk.gripper_halt()
                    except Exception as exc:          # noqa: BLE001 — the arm is held; say it, go on
                        print(f"  [UR5Env] Could not stop the gripper: {exc}")

    def _pose(self, q, xyz, name: str) -> bool:
        """go_home/go_park: up to the transit height first if low, then a joint move."""
        if q is None and xyz is None:
            print(f"UR5: no {name} pose measured yet (robot_config {name.upper()}_JOINTS_DEG): not moving.")
            return False
        here = self.get_ee_pos()
        if here[2] < self._transit_z - 0.005:
            rot = self.get_ee_rotation()
            if not self.move_ee_to(np.r_[here[:2], self._transit_z], orientation=rot):
                # Not straight up that far here — out near the edge of its reach — so as far up as it goes, then
                # the joint move. 2026-11-20: after stops 0.77 m out, "quit" sent it nowhere, and said nothing.
                for dz in (0.10, 0.05, 0.02):
                    if self.move_ee_to(np.r_[here[:2], here[2] + dz], orientation=rot):
                        break
                print(f"UR5: not straight up to {self._transit_z:.2f} m here — to {name} by a joint move from "
                      f"{float(self.get_ee_pos()[2]):.2f} m up.")
        if q is None:
            goal = self._flange_for(xyz, self.down_rotation(xyz))
            with self._sdk_lock:
                q = kinematics.nearest(goal, self._sdk.joints())
            if q is None:
                return False
        return self.move_joints_to(q)

    def go_home(self) -> bool:
        return self._pose(self._home_q, self._home_xyz, "home")

    def go_park(self) -> bool:
        return self._pose(self._park_q, self._park_xyz, "park")

    # ── Freedrive (capabilities.Freedrive) ─────────────────────────────────────
    # tools/calibrate_camera_by_hand.py and tabletop's freedrive skill; and, as
    # disable_torque/enable_torque, tools/calibrate_overhead_camera.py, which
    # moves either arm by hand between touches.

    def freedrive(self, on: bool) -> None:
        """on: the arm can be pushed by hand (the pendant's freedrive, from here) and holds where let go; off: holds."""
        with self._sdk_lock:
            self._sdk.teach(bool(on))

    def disable_torque(self) -> None:
        self.freedrive(True)

    def enable_torque(self) -> None:
        self.freedrive(False)

    # ── The gripper ────────────────────────────────────────────────────────────

    def set_gripper(self, width: float, n_steps: int = 200, max_step_pct: "float | None" = None) -> None:
        """Open or close the jaws to width and wait for them to stop (on the width, or on an object)."""
        self.guard()
        self._held_width_m = None            # the jaws let go: RigEnv's grasp point stops using the held width
        if self._carried is not None and width > self._carried[1] + 0.005:
            self._carried = None             # let go: no longer part of the arm's shape
        goal = float(np.clip(width, 0.0, self._gripper_max_open_m))
        self._jaws_until = time.monotonic() + self.GRIPPER_S + 1.0
        self._jaws_going = True
        try:
            with self._sdk_lock:
                self._sdk.set_gripper(goal)
            sent_at = self.paused_s
            deadline = time.monotonic() + self.GRIPPER_S
            while time.monotonic() < deadline:
                self.guard()
                with self._sdk_lock:
                    if sent_at != self.paused_s:     # a pause halted the fingers (hold): on again
                        self._sdk.set_gripper(goal)
                        sent_at, deadline = self.paused_s, time.monotonic() + self.GRIPPER_S
                        self._jaws_until = deadline + 1.0
                    # Not "not moving": until the gripper has taken the new position it
                    # still says the last move is done (arm.URArm.gripper_settled).
                    elif self._sdk.gripper_settled():
                        return
                time.sleep(1.0 / self._control_hz)
        finally:
            self._jaws_going = False
            self._jaws_until = time.monotonic() + self.JAWS_TAIL_S

    def get_gripper_width(self) -> float:
        with self._sdk_lock:
            return float(self._sdk.gripper_width())

    def gripper_loaded(self) -> bool:
        """capabilities.GripSensing: a "gripper_contact" sensor if the rig has one, else the gripper's own OBJ."""
        contact = self.sensor_of_kind("gripper_contact")
        if contact is not None:
            return bool(self.read_sensor(contact))
        with self._sdk_lock:
            return bool(self._sdk.gripper_holding())

    def squeeze_gripper(self) -> "tuple[float, bool]":
        """The opening, and whether the fingers stopped on something (the gripper's own sense, not the width)."""
        return self.get_gripper_width(), self.gripper_loaded()

    # ── capabilities.ActionStreaming: the stream a learned policy runs on ─────
    # The grasp point (x, y, z, metres) and the jaws' opening (metres) in, the
    # same out as goals (servoL), STREAM_FPS a second, and the wrist camera when
    # it is open. A model for this arm is trained on exactly this (POLICIES.md).

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
        One goal, sent at once (servoL): the grasp point to (x, y, z), the jaws
        to the opening. A goal the arm cannot reach, or further outside
        stream_bounds than the grasp point is now, is held back — the arm
        stays, the jaws still follow — and why is said. Returns the action as
        sent, and why any of it was held back.
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
            fault = self._sdk.fault()
            if fault is not None:
                raise MotionStopped(fault)
            if why is None:
                pose = self._flange_for(goal[:3], self.gripper_frame()[:3, :3])
                if self._sdk.reachable(pose):
                    self._sdk.servo(pose)
                else:
                    why = f"({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}) m is out of reach"
            if self._sdk.has_gripper:
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

    def tcp_wrench(self) -> np.ndarray:
        """The force and torque at the tool, (fx, fy, fz, tx, ty, tz), N and N·m (a CB3's from its joint currents)."""
        with self._sdk_lock:
            return np.asarray(self._sdk.tcp_wrench(), float)

    def zero_contact_force(self) -> None:
        """The contact wrench counted from now (RobotEnv.zero_contact_force), from a settled reading."""
        self._wrench_zero = self._settled_wrench()

    def _settled_wrench(self) -> np.ndarray:
        """
        tcp_wrench() to count a contact from: the median of WRENCH_ZERO_S of
        readings, once force_settle_s has passed since the last move ended. A
        CB3's estimate goes on changing after a move: turning the jaws 34° over
        a block shifted it 44 N over 0.25 s, and a zero read mid-change put 25-30
        N on a descent touching nothing (2026-11-13: stopped at 31.9 N, HISTORY.md).
        With no settle time (an arm in memory): the reading now.
        """
        if self._force_settle_s <= 0.0:
            return self.tcp_wrench()
        wait = self._moved_at + self._force_settle_s - time.monotonic()
        if wait > 0.0:
            time.sleep(wait)
        readings, end = [], time.monotonic() + self.WRENCH_ZERO_S
        while True:
            readings.append(self.tcp_wrench())
            if time.monotonic() >= end:
                break
            time.sleep(1.0 / self._control_hz)
        return np.median(np.asarray(readings), axis=0)

    def contact_wrench(self) -> np.ndarray:
        """
        tcp_wrench() less what it read as the last move began: what touching
        something has added. A CB3's estimate is off by several newtons at
        rest, by an amount that changes with the pose and the payload, so the
        reading itself says little and its change says what a contact does.
        """
        return self.tcp_wrench() - self._wrench_zero

    def current_error(self) -> np.ndarray:
        """Each joint's current less the controller's model's (A): arm.URArm.current_error."""
        with self._sdk_lock:
            return np.asarray(self._sdk.current_error(), float)

    def speed_scaling(self) -> float:
        """The pendant's speed slider, as a fraction."""
        with self._sdk_lock:
            return float(self._sdk.speed_scaling())
