"""
offline.py (mrcf/backends/ur5) — a UR5 that exists only in memory, answering
the ur_rtde calls arm.URArm makes, so the backend runs end to end with no
controller: the contract tests (tests/test_ur5.py), and the tools with
MRCF_UR5_HOST=offline.

    OfflineController   ur_rtde's RTDEControlInterface and RTDEReceiveInterface
                        in one, on kinematics.py's UR5: moveL in a straight
                        line, moveJ joint by joint, in real time at the speed
                        sent (and the pendant's speed slider, speed_scaling)
    OfflineGripper      a Robotiq through its socket, as gripper.RobotiqSocket
                        is called: the fingers move over time and stop on an
                        object of width object_m
    OfflineDashboard    ur_rtde's DashboardClient, the calls the tools make

What it is not: the controller's own planning and safety. Its safety limits
are a floor (safety_z_min) and nothing else, and it never protective-stops by
itself (protective_stop() does it, for a test). URSim, Universal Robots'
simulator, is the real controller without the arm.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import kinematics
from .arm import matrix_to_pose, matrix_to_rotvec, pose_to_matrix, rotvec_to_matrix

# Where it starts: the UR's usual home, wrist down, the flange about 0.49 m out
# and 0.43 m up (kinematics.fk).
START_Q_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)


def connect_offline(controller: "OfflineController"):
    """URArm's connect_with for this controller: the same object answers both interfaces."""
    def connect(host):
        controller.reconnect()
        return controller, controller
    return connect


class OfflineController:
    """ur_rtde's control and receive interfaces on a UR5 in memory. See the module docstring."""

    def __init__(self, q_deg=START_Q_DEG, safety_z_min: "float | None" = None):
        self._lock = threading.RLock()
        self._q = np.radians(np.asarray(q_deg, float))
        self._plan: "tuple | None" = None       # (kind, start, goal, t0, duration)
        self._progress = -1                     # as a UR's: 0 while a move runs, then -1 and -2 by turns
        self._ended = -1
        self.speed_scaling = 1.0                # the pendant's speed slider
        self.safety_z_min = safety_z_min
        self.connected = False
        self.program_running = False
        self.protective_stopped = False
        self.emergency_stopped = False
        self.teaching = False
        self.tcp = [0.0] * 6
        self.wrench = np.zeros(6)
        self.current_error = np.zeros(6)          # actual less target joint current, A (tests set it)
        self.payload: "tuple | None" = None
        self.calls: list = []                   # (name, args) of each command, for tests

    # ── The simulation ─────────────────────────────────────────────────────────

    def _q_at(self, t: float) -> np.ndarray:
        assert self._plan is not None
        kind, start, goal, t0, duration = self._plan
        done = 1.0 if duration <= 0 else min(1.0, (t - t0) * self.speed_scaling / duration)
        if kind == "joint":
            return start + done * (goal - start)
        T0, T1, q0 = start
        T = np.eye(4)
        T[:3, 3] = T0[:3, 3] + done * (T1[:3, 3] - T0[:3, 3])
        rel = matrix_to_rotvec(T0[:3, :3].T @ T1[:3, :3])
        T[:3, :3] = T0[:3, :3] @ rotvec_to_matrix(done * rel)
        q = kinematics.nearest(T, self._q)
        return self._q if q is None else q

    def _update(self) -> None:
        if self._plan is None:
            return
        now = time.monotonic()
        self._q = self._q_at(now)
        kind, start, goal, t0, duration = self._plan
        if duration <= 0 or (now - t0) * self.speed_scaling >= duration:
            self._end()

    def _end(self) -> None:
        self._plan, self._progress = None, (-2 if self._ended == -1 else -1)
        self._ended = self._progress

    def _start(self, kind, start, goal, duration: float) -> None:
        self._plan = (kind, start, goal, time.monotonic(), max(duration, 1e-3))
        self._progress = 0

    def _freeze(self) -> None:
        self._update()
        if self._plan is not None:
            self._end()

    def protective_stop(self) -> None:
        """What the arm does when it hits something: stops, and the control script with it."""
        with self._lock:
            self._freeze()
            self.protective_stopped, self.program_running = True, False

    def _can_move(self) -> bool:
        return self.connected and self.program_running and not (self.protective_stopped or self.emergency_stopped)

    # ── RTDEControlInterface ───────────────────────────────────────────────────

    def reconnect(self) -> bool:
        with self._lock:
            self.connected = self.program_running = True
        return True

    def disconnect(self) -> None:
        with self._lock:
            self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def stopScript(self) -> None:
        with self._lock:
            self._freeze()
            self.program_running = False

    def isProgramRunning(self) -> bool:
        return self.program_running

    def setTcp(self, tcp) -> bool:
        self.tcp = list(tcp)
        return True

    def setPayload(self, mass, cog) -> bool:
        self.payload = (float(mass), list(cog))
        return True

    def moveL(self, pose, speed=0.25, acceleration=1.2, asynchronous=False) -> bool:
        with self._lock:
            self.calls.append(("moveL", (list(pose), speed, acceleration, asynchronous)))
            T1 = pose_to_matrix(pose)
            if not self._can_move() or kinematics.nearest(T1, self._q) is None or not self.isPoseWithinSafetyLimits(pose):
                return False
            self._freeze()
            T0 = kinematics.fk(self._q)
            dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
            turn = float(np.linalg.norm(matrix_to_rotvec(T0[:3, :3].T @ T1[:3, :3])))
            # As URSim 3.15 timed it: the turn goes at about 4x the tool speed, in
            # rad/s, and the line takes as long as the slower of the two.
            self._start("linear", (T0, T1, self._q.copy()), None, max(dist / speed, turn / (4.0 * speed)))
        if not asynchronous:
            self._wait()
        return True

    def moveJ(self, q, speed=1.05, acceleration=1.4, asynchronous=False) -> bool:
        with self._lock:
            self.calls.append(("moveJ", (list(q), speed, acceleration, asynchronous)))
            goal = np.asarray(q, float)
            if not self._can_move() or np.any(np.abs(goal) > kinematics.JOINT_LIMIT_RAD):
                return False
            self._freeze()
            self._start("joint", self._q.copy(), goal, float(np.max(np.abs(goal - self._q))) / speed)
        if not asynchronous:
            self._wait()
        return True

    def _wait(self) -> None:
        while True:
            with self._lock:
                self._update()
                if self._plan is None:
                    return
            time.sleep(0.002)

    def servoL(self, pose, speed, acceleration, time_s, lookahead_time, gain) -> bool:
        with self._lock:
            self.calls.append(("servoL", (list(pose),)))
            if not self._can_move():
                return False
            q = kinematics.nearest(pose_to_matrix(pose), self._q)
            if q is not None:
                self._plan = None
                self._q = q
        return True

    def stopL(self, a=10.0, asynchronous=False) -> None:
        with self._lock:
            self.calls.append(("stopL", (a,)))
            self._freeze()

    def stopJ(self, a=2.0, asynchronous=False) -> None:
        with self._lock:
            self.calls.append(("stopJ", (a,)))
            self._freeze()

    def servoStop(self, a=10.0) -> bool:
        with self._lock:
            self.calls.append(("servoStop", (a,)))
            self._freeze()
        return True

    def teachMode(self) -> bool:
        self.calls.append(("teachMode", ()))
        self.teaching = True
        return True

    def endTeachMode(self) -> bool:
        self.calls.append(("endTeachMode", ()))
        self.teaching = False
        return True

    def getAsyncOperationProgress(self) -> int:
        with self._lock:
            self._update()
            return self._progress

    def getInverseKinematicsHasSolution(self, pose, qnear=(), max_position_error=1e-10,
                                        max_orientation_error=1e-10) -> bool:
        return bool(kinematics.ik(pose_to_matrix(pose)))

    def getInverseKinematics(self, pose, qnear=(), max_position_error=1e-10, max_orientation_error=1e-10) -> list:
        q = kinematics.nearest(pose_to_matrix(pose), qnear if len(qnear) else self._q)
        if q is None:
            raise RuntimeError("getInverseKinematics: no solution (on a real controller, the script stops)")
        return list(q)

    def isPoseWithinSafetyLimits(self, pose) -> bool:
        return self.safety_z_min is None or float(pose[2]) >= self.safety_z_min

    def isJointsWithinSafetyLimits(self, q) -> bool:
        return bool(np.all(np.abs(np.asarray(q, float)) <= kinematics.JOINT_LIMIT_RAD))

    # ── RTDEReceiveInterface ───────────────────────────────────────────────────

    def getActualQ(self) -> list:
        with self._lock:
            self._update()
            return list(self._q)

    def getActualQd(self) -> list:
        with self._lock:
            if self._plan is None:
                return [0.0] * 6
            now = time.monotonic()
            return list((self._q_at(now + 0.004) - self._q_at(now)) / 0.004)

    def getActualTCPPose(self) -> list:
        with self._lock:
            self._update()
            return matrix_to_pose(kinematics.fk(self._q))

    def getActualTCPForce(self) -> list:
        return list(self.wrench)

    def getActualCurrent(self) -> list:
        return [float(v) for v in self.current_error]       # the target is 0: the error is the reading

    def getTargetCurrent(self) -> list:
        return [0.0] * 6

    def getTargetSpeedFraction(self) -> float:
        return float(self.speed_scaling)

    def getRobotMode(self) -> int:
        return 7

    def getSafetyMode(self) -> int:
        return 7 if self.emergency_stopped else 3 if self.protective_stopped else 1

    def isProtectiveStopped(self) -> bool:
        return self.protective_stopped

    def isEmergencyStopped(self) -> bool:
        return self.emergency_stopped


class OfflineGripper:
    """A Robotiq gripper in memory, called as gripper.RobotiqSocket is. See the module docstring."""

    def __init__(self, max_open_m: float = 0.050, speed_m_s: float = 0.15):
        self.max_open_m, self._speed = float(max_open_m), float(speed_m_s)
        self._lock = threading.Lock()
        self._from, self._to, self._t0 = max_open_m, max_open_m, 0.0
        self.object_m: "float | None" = None        # something between the jaws this wide, or nothing

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def _stop_at(self) -> float:
        if self.object_m is not None and self._to < self.object_m <= self._from:
            return self.object_m
        return self._to

    def width_m(self) -> float:
        with self._lock:
            end = self._stop_at()
            travelled = (time.monotonic() - self._t0) * self._speed
            if abs(end - self._from) <= travelled:
                return end
            return self._from + np.sign(end - self._from) * travelled

    def set_width(self, width_m: float) -> None:
        now = self.width_m()
        with self._lock:
            self._from, self._to, self._t0 = now, min(max(float(width_m), 0.0), self.max_open_m), time.monotonic()

    def halt(self) -> None:
        now = self.width_m()
        with self._lock:
            self._from = self._to = now
        self.halts = getattr(self, "halts", 0) + 1

    def moving(self) -> bool:
        return abs(self.width_m() - self._stop_at()) > 1e-6

    def settled(self) -> bool:
        return not self.moving()

    def holding(self) -> bool:
        return not self.moving() and self._stop_at() != self._to


class OfflineDashboard:
    """ur_rtde's DashboardClient, the calls tools/ur5/check_connection.py makes."""

    def __init__(self, host: str = "offline"):
        self.host = host

    def connect(self, timeout_ms: int = 2000) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def isConnected(self) -> bool:
        return True

    def polyscopeVersion(self) -> str:
        return "URSoftware 3.15.8.106339 (offline)"

    def getRobotModel(self) -> str:
        return "UR5"

    def getSerialNumber(self) -> str:
        return "offline"

    def robotmode(self) -> str:
        return "Robotmode: RUNNING"

    def safetymode(self) -> str:
        return "Safetymode: NORMAL"

    def programState(self) -> str:
        return "STOPPED <unnamed>"

    def getLoadedProgram(self) -> str:
        return "No program loaded"
