"""
mrcf/backends/ur5 offline: the backend contract on a UR5 in memory
(offline.OfflineController answering arm.URArm's ur_rtde calls), its
kinematics, and what a UR adds — the controller's faults, the speed slider,
joint moves, freedrive, the Robotiq's socket.
"""

import socket
import threading
import time

import numpy as np
import pytest

from mrcf import backends
from mrcf import overhead_calibration
from mrcf.backends.contract import check_backend
from mrcf.interlock import MotionStopped

REACHABLE = [(-0.45, -0.10, 0.10), (-0.35, 0.20, 0.15)]
UNREACHABLE = [(1.2, 0.0, 0.1)]
FAST = dict(host=None, speed_m_s=4.0, joint_speed_rad_s=6.0, control_hz=1000.0,
            gripper_box=None)     # the arm's own behaviour; the rig's box has its own tests


@pytest.fixture
def ur5(monkeypatch):
    # Its config points overhead_calibration at its own file as it is imported; put that back after.
    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    return backends.arm("ur5")


def _env(ur5, **kw):
    return ur5.make_env(**{**FAST, **kw})


# ── The arm's package ─────────────────────────────────────────────────────────

def test_the_ur5_is_an_arm_with_every_rig_setting(ur5):
    assert "ur5" in backends.available()
    assert backends.missing_rig_settings(ur5.config) == []
    assert ur5.NAME == "UR5"


@pytest.mark.parametrize("host_file, env, want", [
    (None, None, "None|None"),                                        # nowhere: in memory, and says why
    ("# the arm\n10.0.0.7  # static\n", None, "10.0.0.7|file"),        # the rig's own file
    ("10.0.0.7\n", "10.9.9.9", "10.9.9.9|MRCF_UR5_HOST"),            # the variable wins
    ("10.0.0.7\n", "offline", "None|MRCF_UR5_HOST"),                 # in memory, asked for
])
def test_the_controllers_address_is_the_rigs_not_the_codes(tmp_path, host_file, env, want):
    # In a fresh interpreter: robot_config reads its address once, as it is imported.
    import os
    import subprocess
    import sys
    if host_file is not None:
        (tmp_path / "rig" / "ur5").mkdir(parents=True)
        (tmp_path / "rig" / "ur5" / "host").write_text(host_file)
    run_env = {k: v for k, v in os.environ.items() if k != "MRCF_UR5_HOST"}
    run_env["MRCF_HOME"] = str(tmp_path)
    if env is not None:
        run_env["MRCF_UR5_HOST"] = env
    code = ("from mrcf.backends import ur5; c = ur5.config; "
            "print(c.HOST, 'file' if c.HOST_SOURCE == str(c.HOST_FILE) else c.HOST_SOURCE, ur5.describe(), sep='|')")
    out = subprocess.run([sys.executable, "-c", code], env=run_env, capture_output=True, text=True, check=True)
    host, source, described = out.stdout.strip().split("|")
    assert f"{host}|{source}" == want
    assert (os.path.join("rig", "ur5", "host") in described) == (want == "None|None")   # "\" on Windows


def test_making_the_real_arms_env_opens_nothing(ur5):
    # TEST-NET: nothing there, and nothing is opened (the rig's cameras are another matter: none here)
    env = ur5.make_env(host="192.0.2.1", sources={})
    assert env._sdk.host == "192.0.2.1" and not env._sdk.connected
    env.hold()                                   # an unconnected arm holds without error


def test_the_offline_ur5_opens_no_camera_unless_asked(ur5, monkeypatch):
    from mrcf.backends.real_robot_env import RealRobotEnv
    from mrcf.backends.rig import Camera
    opened = []
    monkeypatch.setattr(RealRobotEnv, "_open_camera", lambda self, name, src: opened.append((name, src)))
    monkeypatch.setattr(ur5.config, "CAMERAS", {"overhead": Camera(mount="fixed", source=0, size=(1280, 720))})
    _env(ur5)                                    # host None: in memory, no hardware
    assert opened == []
    _env(ur5, sources={"overhead": 0})           # unless the caller asks for it
    assert opened == [("overhead", 0)]


def test_the_ur5_keeps_the_contract(ur5):
    found = check_backend(lambda: _env(ur5), reachable=REACHABLE, unreachable=UNREACHABLE)
    assert {"OrientationSearch", "GripSensing", "HeldWidth", "OverheadCalibration", "ActionStreaming",
            "Freedrive"} <= set(found)


# ── Kinematics ────────────────────────────────────────────────────────────────

def test_the_dh_model_puts_the_zero_pose_where_universal_robots_does():
    from mrcf.backends.ur5 import kinematics

    assert np.allclose(kinematics.fk(np.zeros(6))[:3, 3], [-0.81725, -0.19145, -0.005491], atol=1e-5)


def test_every_ik_solution_puts_the_flange_back_and_one_is_the_joints_it_came_from():
    from mrcf.backends.ur5 import kinematics

    rng = np.random.default_rng(1)
    for _ in range(300):
        q = rng.uniform(-np.pi, np.pi, 6)
        T = kinematics.fk(q)
        sols = kinematics.ik(T)
        assert sols and all(np.allclose(kinematics.fk(s), T, atol=1e-9) for s in sols)
        assert any(np.allclose(np.cos(s - q), 1.0, atol=1e-9) for s in sols)
        # The nearest is q itself, or (near a singularity) another as close: never a joint the long way round.
        near = kinematics.nearest(T, q + 0.05)
        assert np.allclose(kinematics.fk(near), T, atol=1e-9) and np.max(np.abs(near - (q + 0.05))) < 0.2


def test_out_of_reach_has_no_solution():
    from mrcf.backends.ur5 import kinematics

    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.diag([1.0, -1.0, -1.0]), [1.2, 0.0, 0.1]
    assert kinematics.ik(T) == [] and kinematics.nearest(T, np.zeros(6)) is None


def test_nearest_turns_each_joint_the_short_way_round():
    from mrcf.backends.ur5 import kinematics

    q = np.radians([170.0, -80.0, 70.0, -90.0, -90.0, 175.0])
    q_now = q.copy()
    q_now[0] -= 2 * np.pi - 0.1                 # the base at -190 deg: the same pose a turn away
    got = kinematics.nearest(kinematics.fk(q), q_now)
    assert got is not None and abs(got[0] - q_now[0]) < 0.2


def test_rotation_vectors_round_trip_even_at_half_a_turn():
    from mrcf.backends.ur5.arm import matrix_to_rotvec, rotvec_to_matrix

    rng = np.random.default_rng(2)
    for r in [*rng.normal(size=(50, 3)), [np.pi, 0, 0], [0, np.pi - 1e-6, 0], [0, 0, 0]]:
        R = rotvec_to_matrix(r)
        assert np.allclose(rotvec_to_matrix(matrix_to_rotvec(R)), R, atol=1e-6)


def test_the_arms_shape_runs_from_the_base_to_the_jaws(ur5):
    env = _env(ur5)
    shape = env.arm_shape()
    assert len(shape) == 10 and np.allclose(shape[0][0], 0.0)
    assert np.allclose(shape[-2][0], env.gripper_frame()[:3, 3]) and np.allclose(shape[-1][0], env.get_ee_pos())


# ── Moving ────────────────────────────────────────────────────────────────────

def test_a_move_is_one_asynchronous_moveL_of_the_flange(ur5):
    env = _env(ur5)
    calls = env._sdk._control.calls
    assert env.move_ee_to(np.array(REACHABLE[0]), tol=0.002)
    moves = [args for name, args in calls if name == "moveL"]
    assert len(moves) == 1 and moves[0][3] is True and moves[0][1] == pytest.approx(4.0)
    assert env._sdk._control.tcp == [0.0] * 6     # poses are the flange's; the jaws are RigEnv's
    flange = env.gripper_frame()
    assert np.allclose(flange[:3, 2], [0, 0, -1], atol=1e-6)      # straight down
    assert flange[2, 3] == pytest.approx(REACHABLE[0][2] + ur5.config.JAW_CLOSED_POINT_M[2], abs=0.002)


def test_a_move_across_at_height_is_one_joint_move_turning_on_the_way(ur5):
    # 2026-11-13, the user: "surely the arm twist can happen while moving?"
    env = _env(ur5)
    calls = env._sdk._control.calls
    a, b = np.array([-0.45, -0.10, 0.15]), np.array([-0.35, 0.20, 0.15])
    assert env.move_ee_to(a, tol=0.002)
    turned = env.down_rotation(b, turn_deg=40.0)
    before = len(calls)
    assert env.move_ee_to(b, tol=0.002, orientation=turned)
    sent = [name for name, _ in calls[before:] if name in ("moveL", "moveJ")]
    assert sent == ["moveJ"]                                          # the turn and the move, together
    assert np.allclose(env.get_ee_rotation(), turned, atol=1e-6)
    before = len(calls)
    assert env.move_ee_to(b - [0.0, 0.0, 0.10], tol=0.002, orientation=turned)   # straight down: a line
    assert [name for name, _ in calls[before:] if name in ("moveL", "moveJ")] == ["moveL"]
    before = len(calls)
    assert env.move_ee_to(b - [0.0, 0.0, 0.10] + [0.08, 0.0, 0.0], tol=0.002, orientation=turned)
    assert [name for name, _ in calls[before:] if name in ("moveL", "moveJ")] == ["moveL"]   # low: a line


def test_a_gentle_step_is_a_slower_line(ur5):
    env = _env(ur5)
    env.move_ee_to(np.array(REACHABLE[0]), max_step_deg=2.0)
    speed = [args for name, args in env._sdk._control.calls if name == "moveL"][-1][1]
    assert speed == pytest.approx(4.0 * 2.0 / env.FULL_STEP_DEG)


def test_an_unreachable_target_is_never_sent(ur5):
    env = _env(ur5)
    assert env.move_ee_to(np.array(UNREACHABLE[0])) is False
    assert not [c for c in env._sdk._control.calls if c[0] == "moveL"]


def test_a_target_below_the_controllers_safety_floor_is_never_sent(ur5):
    env = _env(ur5)
    env._sdk._control.safety_z_min = 0.20      # the pendant's safety plane, as the controller reports it
    assert env.move_ee_to(np.array([-0.45, -0.10, 0.02])) is False
    assert not [c for c in env._sdk._control.calls if c[0] == "moveL"]


def test_a_protective_stop_ends_the_move_with_its_reason(ur5):
    env = _env(ur5, speed_m_s=0.2)
    threading.Timer(0.15, env._sdk._control.protective_stop).start()
    with pytest.raises(MotionStopped, match="protective stop"):
        env.move_ee_to(np.array(REACHABLE[1]))
    with pytest.raises(MotionStopped, match="protective stop"):
        env.move_ee_to(np.array(REACHABLE[0]))       # and the next one, until it is cleared


def test_the_speed_slider_gives_a_move_longer_to_finish(ur5):
    env = _env(ur5)
    env.move_ee_to(np.array([-0.45, -0.10, 0.30]))
    env._sdk._control.speed_scaling = 0.02      # the slider at 2%: 0.2 m at 4 m/s takes 2.5 s, past the 2.15 s budget
    start = time.monotonic()
    assert env.move_ee_to(np.array([-0.45, -0.10, 0.10]), tol=0.002)
    assert time.monotonic() - start > 2.15


def test_a_move_that_turns_the_jaws_turns_them_in_place_first(ur5):
    env = _env(ur5)
    env.move_ee_to(np.array(REACHABLE[0]))
    here, calls = env.get_ee_pos(), env._sdk._control.calls
    start = len(calls)
    q0 = np.asarray(env._sdk.joints())
    turned = env.down_rotation(here, turn_deg=60.0)
    assert env.move_ee_to(here + [0.0, 0.0, 0.05], orientation=turned)
    assert [c[0] for c in calls[start:]] == ["moveJ", "moveL"]
    turn_q = np.asarray(calls[start][1][0])
    assert np.allclose(turn_q[:5], q0[:5], atol=1e-6) and abs(turn_q[5] - q0[5]) == pytest.approx(np.radians(60), abs=1e-6)
    assert np.allclose(env.get_ee_rotation(), turned, atol=1e-6)


def test_a_move_to_where_the_arm_is_ends_at_once(ur5):
    # It never shows as moving: the controller's end-of-move flip says it is done.
    env = _env(ur5)
    env.move_ee_to(np.array(REACHABLE[0]))
    start = time.monotonic()
    assert env.move_ee_to(np.array(REACHABLE[0]))
    assert time.monotonic() - start < 0.3


def test_a_controller_slow_to_report_a_move_is_not_taken_for_one_that_ended(ur5):
    env = _env(ur5)
    control, sent = env._sdk._control, {"t": float("-inf"), "before": None}
    move, progress, speeds = control.moveL, control.getAsyncOperationProgress, control.getActualQd

    def late_move(*args, **kw):
        before = progress()
        sent.update(t=time.monotonic(), before=before)
        return move(*args, **kw)

    def late(fn, quiet):
        return lambda: quiet if time.monotonic() - sent["t"] < 0.8 else fn()

    control.moveL = late_move
    control.getAsyncOperationProgress = lambda: (sent["before"] if time.monotonic() - sent["t"] < 0.8 else progress())
    control.getActualQd = late(speeds, [0.0] * 6)
    env.move_ee_to(np.array([-0.45, -0.10, 0.20]))
    assert env.move_ee_to(np.array([-0.45, -0.10, 0.10]), tol=0.002)


def test_hold_stops_the_kind_of_move_that_is_running(ur5):
    env = _env(ur5, joint_speed_rad_s=0.5)
    control = env._sdk._control
    q = np.asarray(env._sdk.joints()) + np.radians([20, 0, 0, 0, 0, 0])
    def move():
        try:
            env.move_joints_to(q)
        except MotionStopped:
            pass

    mover = threading.Thread(target=move, daemon=True)
    mover.start()
    time.sleep(0.1)
    env.interlock.trip("test", source="test")
    mover.join(2.0)
    assert control.calls[-1][0] == "stopJ" and not mover.is_alive()
    env.interlock.clear("test")
    env.stream_action(np.r_[env.get_ee_pos() + [0.001, 0, 0], 0.02])
    env.hold()
    assert control.calls[-1][0] == "servoStop"
    env.move_ee_to(np.array(REACHABLE[0]))
    names = [c[0] for c in control.calls]
    assert names[-2:] == ["servoStop", "moveL"] or "servoStop" in names[names.index("servoL"):]


def test_home_and_park_refuse_until_measured_then_rise_and_join(ur5, capsys):
    env = _env(ur5, home_joints_deg=None, park_joints_deg=None)    # a rig that has not measured them
    assert env.go_home() is False and "no home pose measured" in capsys.readouterr().out
    home = (10.0, -90.0, 90.0, -90.0, -90.0, 0.0)
    env = _env(ur5, home_joints_deg=home)
    env.move_ee_to(np.array([-0.45, -0.10, 0.03]))          # low over the table
    assert env.go_home() is True
    names = [c[0] for c in env._sdk._control.calls]
    assert names[-2:] == ["moveL", "moveJ"]                  # straight up first, then the joint move
    assert np.allclose(np.degrees(env._sdk.joints()), home, atol=0.3)


def test_freedrive_is_teach_mode_and_a_move_ends_it(ur5):
    env = _env(ur5)
    env.disable_torque()
    assert env._sdk._control.teaching
    env.enable_torque()
    assert not env._sdk._control.teaching
    env.disable_torque()
    env.move_ee_to(np.array(REACHABLE[0]))
    assert not env._sdk._control.teaching


def test_connecting_to_an_arm_that_is_not_running_says_what_to_press():
    from mrcf.backends.ur5.arm import _refuse_unless_running

    class Receive:
        def __init__(self, mode, safety=1):
            self.mode, self.safety = mode, safety

        def getRobotMode(self):
            return self.mode

        def getSafetyMode(self):
            return self.safety

    with pytest.raises(ConnectionError, match="press ON, then START"):
        _refuse_unless_running(Receive(4))
    with pytest.raises(ConnectionError, match="protective stop"):
        _refuse_unless_running(Receive(7, safety=3))
    _refuse_unless_running(Receive(7))


# ── Orientation ───────────────────────────────────────────────────────────────

def test_the_approach_is_straight_down_with_the_jaws_as_asked(ur5):
    env = _env(ur5)
    target = np.array([-0.40, 0.0, 0.05])
    slide = np.asarray(ur5.config.JAW_SLIDE_AXIS, float)     # which way the fingers open, as mounted
    rot, tilt = env.approach_orientation_for([target, target + [0, 0, 0.05]])
    assert tilt == 0.0 and np.allclose(rot[:, 2], [0, 0, -1])
    assert abs((rot @ slide) @ [1.0, 0.0, 0.0]) < 1e-9      # tangential: the jaws open across the line from the base
    turned, _ = env.approach_orientation_for([target], turn_deg=90.0)
    assert abs((turned @ slide) @ [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    flipped, _ = env.approach_orientation_for([target], flip=True)
    assert np.allclose(flipped @ slide, -(rot @ slide))
    assert env.approach_orientation_for([np.array(UNREACHABLE[0])]) is None


# ── Force ─────────────────────────────────────────────────────────────────────

def test_the_force_is_what_changed_since_the_move_began(ur5):
    # A CB3's estimate sits several newtons off at rest; a contact is the change.
    from mrcf import safety

    env = _env(ur5)
    env._sdk._control.wrench = np.array([3.0, -4.0, 9.0, 0.1, 0.0, 0.0])     # an offset, at rest
    env.move_ee_to(np.array(REACHABLE[0]))
    assert env.get_contact_force() == pytest.approx(0.0)
    press = ur5.config.CONTACT_FORCE_LIMITS["max_contact_force_n"] + 10.0       # just past this arm's stop
    env._sdk._control.wrench = np.array([3.0, -4.0, 9.0 + press, 0.1, 0.0, 0.0])   # pressing down
    assert env.get_contact_force() == pytest.approx(press)
    monitor = safety.ExecutionMonitor(env, "test", thresholds=safety.SafetyThresholds())   # any task's defaults
    assert monitor.max_contact_force_n == ur5.config.CONTACT_FORCE_LIMITS["max_contact_force_n"]
    assert monitor.check() is True and "contact force" in monitor.halt_reason
    env._sdk._control.wrench = np.array([3.0, -4.0, 14.0, 0.1, 0.0, 0.0])    # 5 N: a stop on most arms, not this one
    assert safety.ExecutionMonitor(env, "test").check() is False


def test_after_a_press_it_backs_off_straight_up_slowly(ur5, monkeypatch):
    # 2026-11-13: a pick stopped at 47.5 N with a finger on its block, and the
    # arm held there, pressing, until the power was cut.
    env = _env(ur5)
    monkeypatch.setattr(env, "BACK_OFF_SPEED_M_S", 0.5)                     # the test's time, not the arm's
    env._sdk._control.wrench = np.array([3.0, -4.0, 9.0, 0.1, 0.0, 0.0])    # at rest, before the descent
    assert env.move_ee_to(np.array(REACHABLE[0]), tol=0.002)
    env._sdk._control.wrench = np.array([3.0, -4.0, 9.0 + 47.5, 0.1, 0.0, 0.0])    # stopped on the block
    before, turned = env.get_ee_pos(), env.get_ee_rotation()
    limits = []
    run = env._run
    monkeypatch.setattr(env, "_run", lambda send, budget, limit_n=None: limits.append(limit_n) or run(send, budget, limit_n))
    assert env.back_off(0.02)
    assert np.allclose(env.get_ee_pos() - before, [0.0, 0.0, 0.02], atol=0.001)
    assert np.allclose(env.get_ee_rotation(), turned, atol=1e-6)                 # the jaws as they were
    assert [args for name, args in env._sdk._control.calls if name == "moveL"][-1][1] == pytest.approx(0.5)
    # The press letting go reads as a force as large as the press: stopped only past it.
    assert limits == [pytest.approx(47.5 + env.BACK_OFF_FORCE_N)]
    assert not env.back_off(5.0)                                             # out of reach: not sent


# ── The gripper ───────────────────────────────────────────────────────────────

def test_set_gripper_waits_for_the_fingers_and_they_stop_on_an_object(ur5):
    env = _env(ur5)
    most = ur5.config.GRIPPER_MAX_OPEN_M
    env._sdk.gripper.object_m = 0.6 * most
    env.set_gripper(0.0)
    assert env.get_gripper_width() == pytest.approx(0.6 * most) and env.squeeze_gripper() == (pytest.approx(0.6 * most),
                                                                                            True)
    env.set_gripper(0.9 * most)
    assert env.get_gripper_width() == pytest.approx(0.9 * most) and not env.gripper_loaded()
    env.set_gripper(2 * most)                                 # asked past fully open: fully open
    assert env.get_gripper_width() == pytest.approx(most)


class _FakeRobotiq:
    """
    The URCap's socket on localhost: GET/SET lines, answered as the gripper on
    the rig does — a GET with a line, a SET with a bare "ack" and no newline.
    """

    def __init__(self, activated=True, takes_gets=0, moves_gets=0, object_pos=None):
        # takes_gets: GETs after a SET POS before it takes it (PRE still the old, OBJ still 3), as
        # the rig's gripper is slow to; moves_gets: GETs it then reports moving (OBJ 0); object_pos:
        # it stops there on something (OBJ 2) when closing past it.
        self.vars = {"ACT": int(activated), "STA": 3 if activated else 0, "GTO": 0, "POS": 3, "OBJ": 3,
                     "SPE": 0, "FOR": 0, "PRE": 3}
        self.takes_gets, self.moves_gets, self.object_pos = takes_gets, moves_gets, object_pos
        self._pending: "list | None" = None
        self.lines: list = []
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.server.accept()
        buf = b""
        while True:
            data = conn.recv(1024)
            if not data:
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                words = line.decode().split()
                self.lines.append(" ".join(words))
                if words[0] == "GET":
                    self._tick()
                    conn.sendall(f"{words[1]} {self.vars[words[1]]}\n".encode())
                else:
                    if words[1] == "POS":
                        self._pending = [int(words[2]), self.takes_gets, self.moves_gets]
                        if not (self.takes_gets or self.moves_gets):
                            self._tick()                  # no delay asked for: at once
                    else:
                        self.vars[words[1]] = int(words[2])
                    conn.sendall(b"ack")

    def _tick(self):
        """One GET's worth of the gripper getting on with the last SET POS."""
        if self._pending is None:
            return
        goal, take, move = self._pending
        if take > 0:
            self._pending[1] -= 1
            return
        self.vars["PRE"], self.vars["OBJ"] = goal, 0
        if move > 0:
            self._pending[2] -= 1
            return
        stops = self.object_pos is not None and goal > self.object_pos >= self.vars["POS"]
        self.vars["POS"], self.vars["OBJ"] = (self.object_pos, 2) if stops else (goal, 3)
        self._pending = None


def test_the_robotiq_socket_speaks_the_urcaps_protocol():
    from mrcf.backends.ur5.gripper import RobotiqSocket

    fake = _FakeRobotiq()
    grip = RobotiqSocket("127.0.0.1", port=fake.port, max_open_m=0.085, pos_open=3, pos_closed=227, force=40)
    grip.connect()
    assert fake.vars["FOR"] == 40
    grip.set_width(0.085)
    assert fake.vars["POS"] == 3 and fake.lines[-1] == "SET GTO 1"
    grip.set_width(0.0)
    assert fake.vars["POS"] == 227
    fake.vars["POS"] = 115
    assert grip.width_m() == pytest.approx(0.0425)
    fake.vars["OBJ"] = 2
    assert grip.holding() and not grip.moving()
    grip.disconnect()


def test_a_robotiq_that_is_not_activated_is_refused_not_activated():
    from mrcf.backends.ur5.gripper import GripperError, RobotiqSocket

    fake = _FakeRobotiq(activated=False)
    with pytest.raises(GripperError, match="not activated"):
        RobotiqSocket("127.0.0.1", port=fake.port, max_open_m=0.085).connect()
    assert not any(line.startswith("SET ACT") for line in fake.lines)


# ── Policies on it ────────────────────────────────────────────────────────────

def test_a_pick_on_the_ur5_runs_to_an_outcome(ur5):
    from mrcf.policies.base import Outcome, Target

    env = _env(ur5, home_joints_deg=(0.0, -90.0, 90.0, -90.0, -90.0, 0.0))
    got = ur5.make_policies().pick(env, Target(np.array([-0.45, -0.05, ur5.config.TABLE_Z]), foot_m=0.02,
                                               max_height_m=0.05))
    assert isinstance(got, Outcome) and not got.ok              # nothing there to hold: a miss, not an error


def test_a_learned_policy_runs_on_the_ur5s_stream(ur5):
    from mrcf.policies.learned import CallableModel, LearnedPolicy

    env = _env(ur5)
    spec = env.stream_spec
    assert spec.state == spec.action == ("x", "y", "z", "gripper") and spec.robot_type == "ur5"
    goal = env.get_ee_pos() + np.array([0.03, 0.02, -0.05])

    def toward(observation):                 # 5 mm a tick toward goal, jaws 2 cm open
        here = np.asarray(observation["observation.state"][:3], float)
        gap = goal - here
        return np.r_[here + gap * min(1.0, 0.005 / max(np.linalg.norm(gap), 1e-9)), 0.02]

    done = lambda env, observation: np.linalg.norm(observation["observation.state"][:3] - goal) < 0.001  # noqa: E731
    got = LearnedPolicy(CallableModel(toward), name="reach", until=done, max_s=10).run(env)
    assert got.ok, got.message
    assert np.linalg.norm(env.get_ee_pos() - goal) < 0.001


# ── With no gripper ───────────────────────────────────────────────────────────

def test_with_no_gripper_the_ur5_keeps_the_contract_and_refuses_grasps_before_moving(ur5):
    from mrcf.policies.base import Target

    make = lambda: _env(ur5, gripper=None, home_joints_deg=None)  # noqa: E731  — home not measured
    found = check_backend(make, reachable=REACHABLE, unreachable=UNREACHABLE, gripper_widths=(0.0,))
    assert "ActionStreaming" in found
    env, policies = make(), ur5.make_policies(gripper=None)
    env.set_gripper(0.05)                                     # nothing to open: the jaws read closed
    assert env.get_gripper_width() == 0.0 and not env.gripper_loaded()
    target = Target(np.array([-0.45, -0.05, ur5.config.TABLE_Z]), foot_m=0.02, max_height_m=0.05)
    start = len(env._sdk._control.calls)
    for got in (policies.pick(env, target), policies.place(env, (-0.45, -0.05)), policies.close_gripper(env)):
        assert got.refused and not got.ok and "no gripper" in got.message
    assert len(env._sdk._control.calls) == start              # refused before moving
    assert policies.move_above(env, target).ok and policies.home(env).ok is False   # home: not measured yet


# ── Against URSim (opt in) ────────────────────────────────────────────────────

def test_the_ur5_keeps_the_contract_on_ursim(ur5):
    """
    The contract through the real ur_rtde, against Universal Robots' simulator
    of the controller (no gripper: URSim has none). Skipped unless
    MRCF_URSIM_HOST names one, started like this:

        docker run -d --rm --platform linux/amd64 --name ursim -p 6080:6080 \\
            -p 29999:29999 -p 30001-30004:30001-30004 universalrobots/ursim_cb3:3.15.8

    then, once: http://localhost:6080/vnc.html, "Go to initialization
    screen", "Confirm Safety Configuration" (until then RTDE refuses the
    control interface), and power on with the brakes released (the
    dashboard's "power on" then "brake release" on port 29999, or ON, START).
    It moves the simulated arm for ~2 minutes.
    """
    import os

    host = os.environ.get("MRCF_URSIM_HOST")
    if not host:
        pytest.skip("MRCF_URSIM_HOST is not set (see this test's docstring)")
    pytest.importorskip("rtde_control")
    live = []

    def make():
        for env in live:
            env.stop()
        live.clear()
        env = ur5.make_env(host=host, gripper=None, sources={}, speed_m_s=0.25, joint_speed_rad_s=1.0,
                           home_joints_deg=(0.0, -90.0, 90.0, -90.0, -90.0, 0.0))
        env.connect()
        live.append(env)
        return env

    try:
        make().go_home()
        found = check_backend(make, reachable=REACHABLE, unreachable=UNREACHABLE, gripper_widths=(0.0,))
    finally:
        for env in live:
            env.stop()
    assert {"OrientationSearch", "ActionStreaming"} <= set(found)


def test_connect_sets_what_hangs_on_the_flange_and_the_joints_say_what_the_model_cannot(ur5):
    env = _env(ur5)
    env.connect()
    control = env._sdk._control
    assert control.payload == (ur5.config.PAYLOAD_KG, list(ur5.config.PAYLOAD_COG_M))
    control.current_error = np.array([0.0, 0.4, -0.2, 0.0, 0.0, 0.1])
    assert np.allclose(env.current_error(), [0.0, 0.4, -0.2, 0.0, 0.0, 0.1])
    env.stop()


def test_an_arm_pushed_by_hand_is_not_one_the_sentry_holds(ur5, monkeypatch):
    env = _env(ur5)
    env.connect()
    monkeypatch.setattr(env._sdk, "moving", lambda: True)    # the joints turning
    assert env.may_be_moving()                                # under a command: a trip holds it
    env.disable_torque()                                      # freedrive: a hand turns them
    assert not env.may_be_moving()
    env.enable_torque()
    assert env.may_be_moving()
    env.stop()


def test_the_gripper_is_done_only_once_it_has_taken_the_new_position():
    # The rig's gripper, asked to close, still said the last move was done for a moment: a check
    # then read the old opening, took a held block for a miss, and the retry dropped it.
    from mrcf.backends.ur5.gripper import RobotiqSocket

    fake = _FakeRobotiq(takes_gets=4, moves_gets=3, object_pos=180)
    grip = RobotiqSocket("127.0.0.1", port=fake.port, max_open_m=0.052, pos_open=3, pos_closed=248)
    grip.connect()
    grip.set_width(0.0)
    assert not grip.moving()                  # it has not started: "not moving" says nothing yet
    polls = 0
    while not grip.settled():
        polls += 1
        assert polls < 50
    assert polls >= 3                         # it waited for the gripper to take it and move
    assert grip.holding() and grip.width_m() == pytest.approx(0.052 * (248 - 180) / 245, abs=1e-6)
    grip.disconnect()


def test_after_a_miss_with_nothing_to_look_again_the_pick_does_not_come_down_again(ur5):
    # 2026-11-12: a close taken for a miss had held the block; the retry opened, dropped it, and came
    # down where it had stood, onto it. With no wrist camera nothing can find it again: stop instead.
    from mrcf.policies.base import Target

    env = _env(ur5, home_joints_deg=(0.0, -90.0, 90.0, -90.0, -90.0, 0.0))
    target = Target(np.array([-0.45, -0.05, ur5.config.TABLE_Z]), foot_m=0.04, height_m=0.04,
                    widths_mm={h: 40.0 for h in range(4, 40, 4)})
    lows = []
    move = env.move_ee_to

    def logged(goal, *a, **k):
        lows.append(float(np.asarray(goal, float)[2]))
        return move(goal, *a, **k)

    env.move_ee_to = logged
    got = ur5.make_policies().pick(env, target)                  # nothing between the jaws: closes on nothing
    descents = sum(1 for z0, z1 in zip(lows, lows[1:]) if z1 < 0.05 <= z0)
    assert not got.ok and "Scan again" in got.message
    assert descents == 1                                         # came down once, not again where it was


def test_the_final_approach_counts_force_from_its_own_start(ur5):
    # 2026-11-12: over a long move the reading drifted 30 N; the tight approach limit, applied
    # before the descent's own start was taken, stopped a gripper touching nothing.
    from mrcf import safety

    env = _env(ur5)
    env.connect()
    env.move_ee_to(np.array(REACHABLE[0]))
    env._sdk._control.wrench = np.array([0.0, 0.0, 35.0, 0.0, 0.0, 0.0])    # drifted since that move began
    env.contact_force_limits = {**env.contact_force_limits, "approach_contact_force_n": 30.0}
    assert env.get_contact_force() == pytest.approx(35.0)
    monitor = safety.ExecutionMonitor(env, "test")
    with safety.approach_contact_limit(env):
        assert env.get_contact_force() == pytest.approx(0.0)
        assert monitor.check() is False
        env._sdk._control.wrench = np.array([0.0, 0.0, 70.0, 0.0, 0.0, 0.0])  # a real press: +35 N
        assert monitor.check() is True
    env.stop()


# 2026-11-13 11:22: the tool force (fx, fy, fz, N) as the jaws finished turning 34° over a block,
# 0.1 s apart (runs/flight incident): mid-change, then where it settled as the descent began.
_TURN_ENDING = [(-2.01, -26.18, -28.82), (-1.86, -3.17, -0.61), (10.63, 22.12, 4.85)]
_SETTLED = [(9.79, 21.47, 7.39), (8.63, 19.80, 9.36), (10.44, 23.63, 8.19)]


def _play(env, readings, every_s):
    """The controller's force reading through readings, every_s apart, from now (a thread)."""
    import threading

    def run():
        for f in readings:
            env._sdk._control.wrench = np.array([*f, 0.0, 0.0, 0.0])
            time.sleep(every_s)

    threading.Thread(target=run, daemon=True).start()


def test_a_descent_is_counted_from_the_force_once_it_has_settled_after_the_turn(ur5):
    """Zeroed mid-change, the descent read 25-30 N touching nothing and stopped at the 30 N limit."""
    from mrcf import safety

    for settle_s, settled in ((0.0, False), (0.35, True)):
        env = _env(ur5, force_settle_s=settle_s)
        env.connect()
        env.move_ee_to(np.array(REACHABLE[0]))                  # the turn: a move that has just ended
        _play(env, _TURN_ENDING + _SETTLED, 0.1)
        with safety.approach_contact_limit(env):                 # the descent's own start
            env._sdk._control.wrench = np.array([*_SETTLED[-1], 0.0, 0.0, 0.0])
            time.sleep(0.65)                                     # on down, touching nothing
            force = env.get_contact_force()
        assert (force < 8.0) is settled, (settle_s, force)       # unsettled: ~28 N, the stop's near-misses
        env.stop()


def test_one_stray_reading_does_not_set_the_zero(ur5):
    env = _env(ur5, force_settle_s=0.05)
    env.connect()
    reads = iter([(0, 0, 5.0)] * 3 + [(0, 0, 60.0)] + [(0, 0, 5.0)] * 200)
    env._sdk.tcp_wrench = lambda: [*next(reads), 0.0, 0.0, 0.0]
    env.zero_contact_force()
    assert env.get_contact_force() == pytest.approx(0.0)
    env.stop()


def test_a_move_sent_again_after_a_pause_counts_force_from_its_first_start(ur5):
    """Zeroed again on the resend, a press already on when the sentry paused it would read as nothing."""
    env = _env(ur5)
    env.connect()
    env._sdk._control.wrench = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    looks, guard = {"ended": 0, "paused": False}, env.guard

    def move_ended():
        looks["ended"] += 1
        return looks["ended"] > 4

    def paused_once():
        if looks["ended"] == 2 and not looks["paused"]:      # held by a pause, pressing 40 N by then
            looks["paused"] = True
            env.__dict__["_paused_s"] = env.paused_s + 1.0
            env._sdk._control.wrench = np.array([0.0, 0.0, 45.0, 0.0, 0.0, 0.0])
        guard()

    env._sdk.move_ended, env.guard = move_ended, paused_once
    sends: list = []
    env._run(lambda: sends.append(1) or True, 5.0)
    assert len(sends) == 2                                   # sent again after the pause
    assert env.get_contact_force() == pytest.approx(40.0)
    env.stop()


def test_a_move_left_by_ctrl_c_or_an_error_is_stopped_not_left_running(ur5):
    """The controller runs a move by itself: leaving the watch without a stop leaves it going, unwatched."""
    env = _env(ur5)
    env.connect()
    stops, ended = [], {"n": 0}
    real_stop = env._sdk.stop

    def stop():
        stops.append(1)
        real_stop()

    def move_ended():
        ended["n"] += 1
        if ended["n"] == 2:
            raise KeyboardInterrupt
        return False

    env._sdk.stop, env._sdk.move_ended = stop, move_ended
    with pytest.raises(KeyboardInterrupt):
        env._run(lambda: True, 5.0)
    assert stops == [1]
    env.stop()


def test_a_hold_during_a_close_stops_the_fingers_and_they_go_on_after_the_pause(ur5):
    """A hand at the jaws trips the sentry: the arm and the fingers stop, then the close carries on."""
    import threading

    env = _env(ur5)
    env.connect()
    grip = env._sdk.gripper
    env.interlock.pause_only("motion")
    assert not env.may_be_moving()
    threading.Timer(0.05, lambda: (env.interlock.trip("a hand", source="motion"), env.hold())).start()
    threading.Timer(0.40, lambda: env.interlock.clear("motion")).start()
    env.set_gripper(0.0)
    assert getattr(grip, "halts", 0) >= 1                         # stopped while the hand was there
    assert env.get_gripper_width() == pytest.approx(0.0, abs=1e-6)  # and closed after
    env.stop()


def test_the_jaws_moving_count_as_the_arm_moving_for_the_sentry(ur5):
    env = _env(ur5)
    env.connect()
    seen = []
    real = env._sdk.gripper_settled

    def settled():
        seen.append(env.may_be_moving())
        return real()

    env._sdk.gripper_settled = settled
    env.set_gripper(0.02)
    assert seen and all(seen)
    env.stop()


def test_any_move_stops_on_a_press_not_only_a_pick_or_a_place(ur5):
    """go_home and go_park ran with no software force limit: only the controller's factory 150 N."""
    from mrcf.interlock import MotionStopped

    env = _env(ur5)
    env.connect()
    looks = {"n": 0}

    def move_ended():
        looks["n"] += 1
        if looks["n"] == 2:
            env._sdk._control.wrench = np.array([0.0, 0.0, 130.0, 0.0, 0.0, 0.0])    # into something
        return False

    env._sdk.move_ended = move_ended
    stops = env.interlock.stop_count
    with pytest.raises(MotionStopped, match="contact force 130 N on a move"):
        env._run(lambda: True, 5.0)
    assert env.interlock.stop_count == stops + 1 and not env.interlock.tripped()    # said, and over
    env.stop()


def test_a_turn_past_a_joints_limit_the_short_way_goes_the_other_way_round_not_through_another_arm_pose():
    """2026-11-13: from park (wrist 3 at 285 deg) a 145 deg turn of the jaws chose a base swing of 139 deg instead."""
    from mrcf.backends.ur5 import kinematics

    park = np.radians([88.74, -70.07, -86.67, -116.08, -270.32, 285.01])
    turned = park.copy()
    turned[5] = np.radians(285.01 + 145.0 - 360.0)                 # the jaws turned 145 deg: wrist 3 at 70
    got = kinematics.nearest(kinematics.fk(turned), park)
    assert got is not None
    assert np.allclose(got[:5], park[:5], atol=1e-6)               # only the wrist turns
    assert np.degrees(got[5]) == pytest.approx(70.01, abs=1e-3)    # the long way round, inside +-360


def test_a_pick_the_box_will_not_allow_is_refused_once_with_why_not_tried_three_times(ur5):
    """2026-11-13: refused moves read "did not converge", and the planner planned the same pick again and again."""
    from mrcf.policies.base import Target

    env = _env(ur5, gripper_box=None)
    env.connect()
    env.move_joints_to(np.radians(ur5.config.PARK_JOINTS_DEG))
    env._gripper_box = ur5.make_env(host=None).__dict__["_gripper_box"]   # the rig's box (offline: no camera opened)
    assert not env.gripper_fits((0.003, 0.207, 0.03)) and env.gripper_fits((0.40, 0.40, 0.03))
    got = ur5.make_policies().pick(env, Target(np.array([0.003, 0.207, 0.0]), foot_m=0.034, height_m=0.03))
    assert not got.ok and got.refused and "box" in got.message
    assert env.move_refusals >= 1
    env.stop()
