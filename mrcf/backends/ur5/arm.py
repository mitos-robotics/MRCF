"""
arm.py (mrcf/backends/ur5) — the UR5's controller, over ur_rtde, as the few calls
env.UR5Env makes (the template's SDK interface, mrcf/backends/_template/env.py,
plus a UR's joint moves, freedrive and faults).

    URArm(host)                      ur_rtde's interfaces on the controller at host
    URArm(host, connect_with=...)    anything else that answers the same calls
                                     (offline.OfflineController, URSim is the real one)

Only this file names ur_rtde's calls, so a newer ur_rtde, or another way to
reach the controller, is a change here. The env holds a lock round every
call: ur_rtde's control interface is not safe from two threads.

What the controller is asked, and why:

  - poses are [x, y, z, rx, ry, rz] (a rotation vector) in the base frame,
    with the TCP set to the flange (setTcp zeros), so the jaw model stays in
    one place (RigEnv's tcp_offset);
  - moves are asynchronous (moveL/moveJ with asynchronous=True) so the env
    can call guard() every cycle and stop them (stopL/stopJ) mid-way; one has
    ended when getAsyncOperationProgress() turns negative and differs from
    before it was sent (it alternates -1 and -2 as each ends, a stopped one
    and one to where the arm already is included: URSim 3.15.8);
  - reachable() solves the joints here (kinematics.py, which matches the
    controller's model to 0.0 mm on URSim) and asks the controller whether
    the pose and those joints are inside the safety limits set on the
    pendant, so a move it would refuse is never sent. (Not its own IK query:
    getInverseKinematicsHasSolution with a qnear and ur_rtde's default
    tolerances said no to the pose the arm stood in, on its first call in 5
    of 8 sessions on URSim 3.15.8.)
  - fault() reads the safety state each cycle: a protective stop, an
    emergency stop or a stopped control script ends a move with its reason.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from . import kinematics
from .gripper import NoGripper

ROBOT_MODE_RUNNING = 7
# getSafetyMode(): what each means for a move. 1 normal, 2 reduced (still moves, slower).
SAFETY_MODES = {
    3: "a protective stop: the arm met a force or a limit — clear it on the pendant",
    4: "safety recovery: move the arm back inside its limits on the pendant",
    5: "a safeguard stop: a safeguard input is open",
    6: "a system emergency stop",
    7: "the emergency stop is pressed",
    8: "a safety violation: restart the safety system on the pendant",
    9: "a safety fault: restart the safety system on the pendant",
    10: "a validate-joint-ids fault",
    11: "the arm is undefined",
    12: "an automatic-mode safeguard stop",
    13: "a system three-position-enabling stop",
}
ROBOT_MODES = {-1: "no controller", 0: "connecting", 1: "confirming safety", 2: "booting", 3: "powered off",
               4: "powering on (the joints not up yet)", 5: "idle (powered, brakes on)", 6: "backdrive",
               7: "running", 8: "updating firmware"}
# Each joint's own state (RTDE joint_mode). All six stuck at "booting" after ON was this arm with a
# fault in its control box (2026-11: NOTES.md).
JOINT_MODES = {236: "shutting down", 237: "part D calibration", 238: "backdrive", 239: "powered off",
               245: "not responding", 246: "motor initialisation", 247: "booting", 248: "part D calibration error",
               249: "bootloader", 250: "calibration", 252: "fault", 253: "running", 255: "idle"}


def pose_to_matrix(pose) -> np.ndarray:
    """[x, y, z, rx, ry, rz] (rotation vector) -> 4x4."""
    pose = np.asarray(pose, float)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotvec_to_matrix(pose[3:6]), pose[:3]
    return T


def matrix_to_pose(T) -> list:
    """4x4 -> [x, y, z, rx, ry, rz], as ur_rtde takes it."""
    T = np.asarray(T, float)
    return [float(v) for v in np.r_[T[:3, 3], matrix_to_rotvec(T[:3, :3])]]


def rotvec_to_matrix(r) -> np.ndarray:
    r = np.asarray(r, float)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.eye(3)
    k = r / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def matrix_to_rotvec(R) -> np.ndarray:
    R = np.asarray(R, float)
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-4:            # near half a turn: the axis from R + I's largest column
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        axis = M[:, i] / np.sqrt(max(M[i, i], 1e-12))
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def _ur_rtde(host: str):
    """ur_rtde's control and receive interfaces on host (the control one uploads its script)."""
    try:
        import rtde_control
        import rtde_receive
    except ImportError as exc:
        raise ImportError("ur_rtde is not installed: pip install ur_rtde (on macOS it builds from source: "
                          "mrcf/backends/ur5/NOTES.md, 'Installing ur_rtde on macOS').") from exc
    receive = rtde_receive.RTDEReceiveInterface(host)
    _refuse_unless_running(receive)
    try:
        control = rtde_control.RTDEControlInterface(host)
    except RuntimeError as exc:
        receive.disconnect()
        if "synchronization" not in str(exc):
            raise
        # The controller sends a message in place of its version until its safety
        # setup is confirmed, and ur_rtde then sets up the wrong registers.
        raise ConnectionError(f"{exc}. On the pendant: confirm the safety configuration if it asks (on the "
                              "initialization screen), and check no fieldbus (EtherNet/IP, PROFINET) adapter "
                              "holds the RTDE registers (Installation > Fieldbus).") from exc
    return control, receive


def _refuse_unless_running(receive) -> None:
    """Before the control script goes up: the arm powered, its brakes off, nothing stopping it."""
    mode = int(receive.getRobotMode())
    if mode != ROBOT_MODE_RUNNING:
        raise ConnectionError(f"The UR5 is {ROBOT_MODES.get(mode, f'in robot mode {mode}')}: on the pendant, "
                              "go to the initialization screen and press ON, then START.")
    safety = int(receive.getSafetyMode())
    if safety in SAFETY_MODES:
        raise ConnectionError(f"The UR5 is in {SAFETY_MODES[safety]}.")


class URArm:
    """The UR5's controller and its gripper, as UR5Env calls them. See the module docstring."""

    def __init__(self, host: "str | None", *, gripper=None, connect_with=None,
                 accel_m_s2: float = 0.5, joint_accel_rad_s2: float = 1.0,
                 hold_decel_m_s2: float = 2.0, hold_decel_rad_s2: float = 3.0,
                 servo_dt_s: float = 1.0 / 30, servo_lookahead_s: float = 0.1, servo_gain: float = 300.0,
                 moving_rad_s: float = 0.002, payload_kg: "float | None" = None, payload_cog_m=None):
        self.host = host
        # What hangs on the flange, set as the control script starts: its own
        # setTcp() would otherwise leave an unset centre of gravity at the flange.
        self._payload = None if payload_kg is None else (float(payload_kg), [float(v) for v in (payload_cog_m or (0, 0, 0))])
        self.gripper = gripper if gripper is not None else NoGripper()
        self._connect_with: Callable[[Any], tuple] = connect_with or _ur_rtde   # host -> (control, receive)
        self._control: Any = None
        self._receive: Any = None
        self._accel, self._joint_accel = float(accel_m_s2), float(joint_accel_rad_s2)
        self._hold_decel, self._hold_decel_rad = float(hold_decel_m_s2), float(hold_decel_rad_s2)
        self._servo = (float(servo_dt_s), float(servo_lookahead_s), float(servo_gain))
        self._moving_rad_s = float(moving_rad_s)
        self._mode: "str | None" = None        # the last command: "linear", "joint", "servo", "teach"
        self._ended_before: "int | None" = None   # getAsyncOperationProgress() as the last move was sent

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._control is not None:
            self.disconnect()                    # one control script at a time: a second connect replaces the first
        self._control, self._receive = self._connect_with(self.host)
        try:
            self._control.setTcp([0.0] * 6)      # poses are the flange's: the jaws are RigEnv's tcp_offset
            if self._payload is not None:
                self._control.setPayload(*self._payload)
            self._wait_ready()
            self.gripper.connect()
        except BaseException:
            self.disconnect()                    # nothing left running on the controller
            raise

    def _wait_ready(self, timeout_s: float = 3.0) -> None:
        """
        Until the control script answers what must be yes: that the pose and
        joints the arm is in are inside its safety limits. A no that lasts
        means the arm stands outside them (the pendant shows which), or the
        script is not answering.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            q, pose = self._receive.getActualQ(), self._receive.getActualTCPPose()
            if (self._control.isProgramRunning() and self._control.isPoseWithinSafetyLimits(pose)
                    and self._control.isJointsWithinSafetyLimits(q)):
                return
            if time.monotonic() > deadline:
                raise ConnectionError(f"The UR5's control script did not say, within {timeout_s:.0f} s, that the "
                                      "arm is inside its safety limits: check the pendant's safety settings.")
            time.sleep(0.02)

    def disconnect(self) -> None:
        try:
            if self._control is not None:
                if self._mode == "teach":
                    self._control.endTeachMode()
                self._control.stopScript()
                self._control.disconnect()
            if self._receive is not None:
                self._receive.disconnect()
        finally:
            self._control = self._receive = None
            self.gripper.disconnect()

    @property
    def connected(self) -> bool:
        return self._control is not None

    @property
    def has_gripper(self) -> bool:
        return not isinstance(self.gripper, NoGripper)

    def _need(self):
        if self._control is None:
            raise ConnectionError("The UR5 is not connected: call env.connect() first.")
        return self._control, self._receive

    # ── Where it is ────────────────────────────────────────────────────────────

    def flange_pose(self) -> np.ndarray:
        return pose_to_matrix(self._need()[1].getActualTCPPose())

    def joints(self) -> list:
        return [float(v) for v in self._need()[1].getActualQ()]

    def joint_speeds(self) -> list:
        return [float(v) for v in self._need()[1].getActualQd()]

    def tcp_wrench(self) -> np.ndarray:
        return np.asarray(self._need()[1].getActualTCPForce(), float)

    def current_error(self) -> np.ndarray:
        """
        Each joint's motor current less what the controller's model expected
        (A): what nothing in its model explains — a contact, or the model wrong.
        """
        receive = self._need()[1]
        return np.asarray(receive.getActualCurrent(), float) - np.asarray(receive.getTargetCurrent(), float)

    def speed_scaling(self) -> float:
        """
        The pendant's speed slider as a fraction: moves take 1/this as long.
        (Not getSpeedScaling(), the trajectory's actual scaling, which reads 0
        while nothing runs.)
        """
        return float(self._need()[1].getTargetSpeedFraction())

    def moving(self) -> bool:
        control, receive = self._need()
        if control.getAsyncOperationProgress() >= 0:
            return True
        return bool(np.max(np.abs(receive.getActualQd())) > self._moving_rad_s)

    def move_ended(self) -> bool:
        """Whether the last move sent has ended: arrived, stopped, or refused once under way."""
        if self._ended_before is None:
            return False
        progress = int(self._need()[0].getAsyncOperationProgress())
        return progress < 0 and progress != self._ended_before

    def fault(self) -> "str | None":
        """Why the arm cannot move now, or None: a stop on the arm itself, or its control script gone."""
        control, receive = self._need()
        if receive.isEmergencyStopped():
            return "the UR5's emergency stop is pressed"
        if receive.isProtectiveStopped():
            return "the UR5 is in " + SAFETY_MODES[3]
        safety = int(receive.getSafetyMode())
        if safety in SAFETY_MODES:
            return "the UR5 is in " + SAFETY_MODES[safety]
        mode = int(receive.getRobotMode())
        if mode != ROBOT_MODE_RUNNING:
            return f"the UR5 is {ROBOT_MODES.get(mode, f'in robot mode {mode}')}"
        if not control.isProgramRunning():
            return ("the UR5's control script stopped (a program was started or stopped on the pendant, or "
                    "a stop was cleared): reconnect")
        return None

    # ── Reach ──────────────────────────────────────────────────────────────────

    def reachable(self, pose: np.ndarray) -> bool:
        """Whether the flange can get to pose, near where its joints are, inside the controller's safety limits."""
        control, _ = self._need()
        q = kinematics.nearest(pose, self.joints())
        if q is None:
            return False
        return (bool(control.isPoseWithinSafetyLimits(matrix_to_pose(pose)))
                and bool(control.isJointsWithinSafetyLimits([float(v) for v in q])))

    def joints_reachable(self, q) -> bool:
        return bool(self._need()[0].isJointsWithinSafetyLimits([float(v) for v in q]))

    # ── Moving ─────────────────────────────────────────────────────────────────

    def _leave_servo(self, control) -> None:
        if self._mode == "servo":
            control.servoStop(self._hold_decel)
        elif self._mode == "teach":
            control.endTeachMode()

    def move_linear(self, pose: np.ndarray, speed_m_s: float) -> bool:
        """A straight line of the flange to pose, started; returns at once. False: the controller refused it."""
        control, _ = self._need()
        self._leave_servo(control)
        self._mode = "linear"
        self._ended_before = int(control.getAsyncOperationProgress())
        return bool(control.moveL(matrix_to_pose(pose), float(speed_m_s), self._accel, True))

    def move_joints(self, q, speed_rad_s: float) -> bool:
        """A move in joint space to q (radians), started; returns at once. False: the controller refused it."""
        control, _ = self._need()
        self._leave_servo(control)
        self._mode = "joint"
        self._ended_before = int(control.getAsyncOperationProgress())
        return bool(control.moveJ([float(v) for v in q], float(speed_rad_s), self._joint_accel, True))

    def servo(self, pose: np.ndarray) -> None:
        """One goal of a stream (servoL), a control tick away."""
        control, _ = self._need()
        if self._mode == "teach":
            control.endTeachMode()
        self._mode = "servo"
        dt, lookahead, gain = self._servo
        control.servoL(matrix_to_pose(pose), 0.0, 0.0, dt, lookahead, gain)

    def stop(self) -> None:
        """Decelerate to a stop where it is, whatever it was doing (from any thread)."""
        control, _ = self._need()
        if self._mode == "servo":
            control.servoStop(self._hold_decel)
        elif self._mode == "joint":
            control.stopJ(self._hold_decel_rad)
        elif self._mode != "teach":
            control.stopL(self._hold_decel)

    @property
    def teaching(self) -> bool:
        """In freedrive: moved by hand, not by a command."""
        return self._mode == "teach"

    def teach(self, on: bool) -> None:
        """Freedrive: the arm can be pushed by hand (teachMode), or holds again (endTeachMode)."""
        control, _ = self._need()
        if on:
            self.stop()
            control.teachMode()
            self._mode = "teach"
        elif self._mode == "teach":
            control.endTeachMode()
            self._mode = None

    # ── The gripper ────────────────────────────────────────────────────────────

    def set_gripper(self, width_m: float) -> None:
        self.gripper.set_width(float(width_m))

    def gripper_halt(self) -> None:
        """The fingers stopped where they are (a gripper without a halt: nothing to do)."""
        halt = getattr(self.gripper, "halt", None)
        if halt is not None:
            halt()

    def gripper_width(self) -> float:
        return float(self.gripper.width_m())

    def gripper_holding(self) -> bool:
        return bool(self.gripper.holding())

    def gripper_moving(self) -> bool:
        return bool(self.gripper.moving())

    def gripper_settled(self) -> bool:
        """Done with the last set_gripper (the gripper's settled(); not moving, for one that cannot say more)."""
        settled = getattr(self.gripper, "settled", None)
        return bool(settled()) if settled is not None else not self.gripper.moving()
