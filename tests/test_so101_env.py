"""SO101Env's stop and threading contract, against a stand-in for LeRobot's follower: no hardware."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from mrcf.backends.so101.kinematics import JOINT_NAMES
from mrcf.backends.so101.env import SO101Env
from mrcf.backends.so101.orientation import _tilts_from
from mrcf.interlock import MotionStopped


class _Bus:
    def __init__(self):
        self.positions = {**{n: 10.0 for n in JOINT_NAMES}, "gripper": 5.0}
        self.reads = 0

    def sync_read(self, _register):
        self.reads += 1
        return dict(self.positions)


class _Camera:
    def read_latest(self):
        return np.full((480, 640, 3), 7, np.uint8)


class _Follower:
    def __init__(self):
        self.bus = _Bus()
        self.cameras = {"front": _Camera()}
        self.sent: list = []
        self.connected = False

    def connect(self, calibrate=True):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def send_action(self, action):
        self.sent.append(dict(action))
        return action


def offline_so101() -> SO101Env:
    env = SO101Env.__new__(SO101Env)
    env._robot = _Follower()
    env._bus_lock = threading.RLock()
    env._camera_name = "front"
    env._overhead_camera_name = None
    env._control_hz = 1e9
    env._joint_deg_limits = {}
    # Tests move the stand-in's joints by hand between reads, in no time at
    # all: every read is a fresh one (see test_one_reading_serves_a_whole_control_tick).
    env.POS_REUSE_S = 0.0
    return env


def test_a_tripped_interlock_holds_the_joints_and_sends_nothing_else():
    env = offline_so101()
    env.interlock.trip("hand near the arm", source="hands")
    with pytest.raises(MotionStopped):
        env.goto_joint_degrees({n: 20.0 for n in JOINT_NAMES})
    # Only the hold went out: the arm's joints where they are. Not the gripper,
    # whose goal is what keeps a held piece squeezed.
    assert env._robot.sent == [{f"{n}.pos": 10.0 for n in JOINT_NAMES}]


def test_moves_run_once_it_clears():
    env = offline_so101()
    env.interlock.trip("hand", source="hands")
    env.interlock.clear("hands")
    env.goto_joint_degrees({n: 10.0 for n in JOINT_NAMES}, timeout_s=0.5)
    assert env._robot.sent and env._robot.sent[0] == {f"{n}.pos": 10.0 for n in JOINT_NAMES}


def test_frames_come_from_the_camera_not_the_motor_bus():
    env = offline_so101()
    frame = env.render_image("front", 640, 480)
    assert env._robot.bus.reads == 0
    frame[:] = 0                                   # a caller drawing on its copy...
    assert env.render_image("front", 640, 480).max() == 7   # ...does not touch anyone else's


class _Reader:
    def __init__(self, alive):
        self.alive = alive

    def is_alive(self):
        return self.alive


class _DyingCamera(_Camera):
    """lerobot's camera after its reader gave up: frames raise until it is opened again."""

    def __init__(self, env=None, opens=True):
        self.thread = _Reader(False)
        self.env, self.opens = env, opens
        self.connects = 0
        self.paused_while_closed: list = []

    def disconnect(self):
        self.thread = None

    def connect(self):
        self.connects += 1
        if self.env is not None:
            self.paused_while_closed.append(self.env.interlock.only_pausing())
        if not self.opens:
            raise ConnectionError("Failed to open OpenCVCamera(0).")
        self.thread = _Reader(True)

    def read_latest(self):
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError("OpenCVCamera(0) read thread is not running.")
        return super().read_latest()


def test_a_camera_whose_reader_died_is_opened_again_and_gives_frames():
    env = offline_so101()
    env._running = True
    env._robot.cameras = {"front": (cam := _DyingCamera(env))}
    assert env.render_image("front", 640, 480).max() == 7
    assert cam.connects == 1
    assert cam.paused_while_closed == [False]       # not one the sentry watches through: no pause
    env.render_image("front", 640, 480)
    assert cam.connects == 1                        # reading again: left alone


def test_the_arm_pauses_while_a_camera_the_sentry_watches_through_is_reopened():
    env = offline_so101()
    env._running = True
    env._overhead_camera_name = "overhead"
    env._robot.cameras = {"front": _Camera(), "overhead": (cam := _DyingCamera(env))}
    env.render_image("overhead", 640, 480)
    assert cam.paused_while_closed == [True]        # paused, not stopped
    assert not env.interlock.tripped()              # and free again once it is back


def test_a_camera_that_will_not_open_again_raises_and_waits_before_another_try():
    env = offline_so101()
    env._running = True
    env._overhead_camera_name = "overhead"
    env._robot.cameras = {"overhead": (cam := _DyingCamera(env, opens=False))}
    for _ in range(3):
        with pytest.raises(RuntimeError, match="not running"):
            env.render_image("overhead", 640, 480)
    assert cam.connects == 1                        # not hammered on every frame
    assert not env.interlock.tripped()              # the sentry's own fail-safe holds the arm, not this
    env._reopened_at["overhead"] -= 10.0
    with pytest.raises(RuntimeError):
        env.render_image("overhead", 640, 480)
    assert cam.connects == 2


def test_a_camera_is_reopened_once_however_many_threads_ask():
    env = offline_so101()
    env._running = True
    env._robot.cameras = {"front": (cam := _DyingCamera())}
    slow = cam.connect

    def connect():
        time.sleep(0.05)
        slow()
    cam.connect = connect
    threads = [threading.Thread(target=env.render_image, args=("front", 640, 480)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cam.connects == 1


def test_a_stopped_arms_cameras_are_not_opened_again():
    env = offline_so101()                           # never connected
    env._robot.cameras = {"front": (cam := _DyingCamera())}
    with pytest.raises(RuntimeError):
        env.render_image("front", 640, 480)
    assert cam.connects == 0


class _NotConnected(Exception):
    pass


class _CheckedFollower(_Follower):
    """lerobot's follower as it is: connected only while the bus and EVERY camera it holds are (is_connected)."""

    @property
    def is_connected(self):
        return all(getattr(c, "thread", True) is not None for c in self.cameras.values())

    def send_action(self, action):
        if not self.is_connected:
            raise _NotConnected("SOFollower is not connected. Run `.connect()` first.")
        return super().send_action(action)

    def disconnect(self):
        if not self.is_connected:
            raise _NotConnected("SOFollower is not connected. Run `.connect()` first.")
        super().disconnect()


def test_the_arm_keeps_moving_while_a_camera_is_closed_to_be_reopened():
    # 2026-11-14: the wrist camera dropped off USB mid-move; closed for its
    # reopen, it made lerobot refuse the next joint command and the controller died.
    env = offline_so101()
    env._running = True
    env._robot = _CheckedFollower()
    env._robot.cameras = {"front": (cam := _DyingCamera())}
    sent_while_closed = []

    def connect():
        env.goto_joint_degrees({n: 10.0 for n in JOINT_NAMES}, timeout_s=0.5)   # the move under way
        sent_while_closed.append(len(env._robot.sent))
        _DyingCamera.connect(cam)
    cam.connect = connect
    env.render_image("front", 640, 480)
    assert sent_while_closed and sent_while_closed[0] >= 1
    assert env._robot.cameras["front"] is cam       # back with the follower once it reads


def test_a_camera_that_will_not_open_again_leaves_the_arm_working_and_is_tried_again():
    env = offline_so101()
    env._running = True
    env._robot = _CheckedFollower()
    env._robot.cameras = {"front": (cam := _DyingCamera(opens=False))}
    with pytest.raises(RuntimeError, match="not running"):
        env.render_image("front", 640, 480)
    assert "front" not in env._robot.cameras
    assert "front" in env.list_cameras()
    env.goto_joint_degrees({n: 10.0 for n in JOINT_NAMES}, timeout_s=0.5)
    assert env._robot.sent
    cam.opens = True
    env._reopened_at["front"] -= 10.0
    assert env.render_image("front", 640, 480).max() == 7
    assert env._robot.cameras["front"] is cam


def test_stop_disconnects_even_with_a_camera_closed():
    env = offline_so101()
    env._running = True
    env._robot = _CheckedFollower()
    env._robot.bus.is_connected = True
    closed = []
    env._robot.bus.disconnect = lambda disable_torque: closed.append(disable_torque)
    env._robot.cameras = {"front": _DyingCamera()}
    env._robot.cameras["front"].thread = None       # closed, and lerobot refuses to disconnect
    env.stop()
    assert closed == [False]                        # the bus closed, the arm's torque left on
    assert not env._running


def test_a_blocked_joint_is_let_go_of_instead_of_pushed_until_it_overloads():
    env = offline_so101()               # the stand-in's joints never move
    env.STALL_S = 0.05
    target = {n: 40.0 for n in JOINT_NAMES}
    assert env.goto_joint_degrees(target, timeout_s=5.0) is False
    # The last command holds every joint where it is, not where it was going.
    assert env._robot.sent[-1] == {f"{n}.pos": 10.0 for n in JOINT_NAMES}


def test_a_slow_joint_that_is_moving_is_not_a_stall():
    env = offline_so101()
    env.STALL_S = 0.05
    follower = env._robot

    def creep(action):                  # 0.5 deg per command, towards the goal
        follower.sent.append(dict(action))
        for key, goal in action.items():
            name = key.removesuffix(".pos")
            here = follower.bus.positions[name]
            follower.bus.positions[name] = here + float(np.clip(goal - here, -0.5, 0.5))
        return action

    follower.send_action = creep
    assert env.goto_joint_degrees({n: 40.0 for n in JOINT_NAMES}, timeout_s=5.0) is True


class _LinearKinematics:
    """Stands in for RobotKinematics: the 'pose' is just the joint angles."""
    joint_names = JOINT_NAMES + ["gripper"]

    def forward_kinematics(self, q):
        return np.asarray(q, float).copy()

    def inverse_kinematics(self, current, desired, **_):
        return np.asarray(desired, float).copy()


def test_joint_offsets_are_added_going_in_and_taken_off_coming_out():
    from mrcf.backends.so101.kinematics import OffsetKinematics
    off = np.array([0.0, -6.0, -4.0, -2.7, 0.0, 0.0])
    kin = OffsetKinematics(_LinearKinematics(), off)
    raw = np.array([1.0, -40.0, 60.0, 60.0, 0.0, 0.0])
    model_angles = kin.forward_kinematics(raw)
    assert np.allclose(model_angles, raw + off)
    # Asked for the pose those model angles give, it answers in the motors' own angles.
    assert np.allclose(kin.inverse_kinematics(raw, model_angles), raw)
    assert kin.joint_names == _LinearKinematics.joint_names       # everything else passes through


def test_jaw_width_comes_from_the_measured_table_both_ways():
    env = offline_so101()
    env._gripper_max_open_m = 0.045
    env._gripper_width_table = [(0.0, 0.0), (9.9, 0.020), (24.7, 0.042), (69.5, 0.100)]
    assert env._pct_to_width(9.9) == pytest.approx(0.020)
    assert env._pct_to_width(17.3) == pytest.approx(0.031)          # between points: interpolated
    for width in (0.010, 0.035, 0.080):
        assert env._pct_to_width(env._width_to_pct(width)) == pytest.approx(width)
    # Without a table, the straight line it replaces: 66.7% read as 30 mm (really ~97).
    env._gripper_width_table = None
    assert env._pct_to_width(66.7) == pytest.approx(0.030, abs=1e-4)


# ── The servo at another tilt ──────────────────────────────────────────────────

def _posed_so101(tilt_deg, at=(0.30, 0.0, 0.0855)):
    """
    An SO101Env whose arm is just a pose: a grasp point and an orientation,
    which move_ee_to sets. Servo calibrated as on the rig: at -5 deg, 68.8 mm
    up, on a point 20 mm up a piece.
    """
    env = offline_so101()
    env._r_down = np.diag([1.0, -1.0, -1.0])
    env._pan_centre = np.zeros(2)
    env._jaw_axis, env._jaw_slide_axis = "tangential", np.array([0.0, -1.0, 0.0])     # the rig's
    env._tcp_offset, env._jaw_centre_slide, env._gripper_max_open_m = np.zeros(3), 0.5, 0.045
    env._grasp_gap = lambda: 0.035
    env._work_plane_z = 0.0167
    env._render_w, env._render_h = 640, 480
    env._gripper_cam_grasp_px = (376.0, 209.0)
    env._gripper_cam_jacobian = [[7.8e-5, -2.8e-4], [3.4e-4, 1.4e-4]]
    env._gripper_cam_ref_height_m = 0.0688
    env._gripper_cam_ref_tilt_deg = -5.0
    env._gripper_cam_ref_point_height_m = 0.020
    env._gripper_cam_position_m = np.array([-0.076, 0.033, -0.053])
    pose = {"tcp": np.array(at, float), "rot": env.tilted_rotation(tilt_deg, np.array(at))}

    class Kinematics:
        def forward_kinematics(self, _q):
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = pose["rot"], pose["tcp"] - pose["rot"] @ env._tcp()
            return t

    env._kinematics = Kinematics()
    env._q_arm_deg = lambda obs=None: np.zeros(len(JOINT_NAMES))
    moves = []

    def move_ee_to(target, tol=0.01, max_step_deg=None, orientation=None):
        moves.append(np.array(target, float))
        pose["tcp"] = np.array(target, float)
        return True

    env.move_ee_to = move_ee_to
    env.render_image = lambda *a, **k: np.zeros((480, 640, 3), np.uint8)
    return env, pose, moves


def _on_target(env):
    """A detector that sees the object on the servo's target pixel."""
    return lambda frame, expected_px, radius_px: tuple(env._gripper_cam_grasp_px)


def test_the_gripper_tilt_reads_back_whatever_the_pan_or_half_turn():
    env, _, _ = _posed_so101(0.0)
    for slide in ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)):
        for jaws in ("radial", "tangential"):
            env._jaw_slide_axis, env._jaw_axis = np.array(slide), jaws
            for deg in (-50.0, -5.0, 0.0, 12.0):
                for at in ((0.30, 0.0), (0.20, 0.15), (0.25, -0.10)):
                    for flip in (False, True):
                        rot = env.tilted_rotation(deg, np.array([*at, 0.05]), flip=flip)
                        assert env.gripper_tilt_deg(rot) == pytest.approx(deg)


def test_the_jaws_open_the_way_jaw_axis_says_whichever_gripper_axis_they_slide_along():
    # The policy checks fingers and the lean along JAW_AXIS; the wrist has to
    # put the jaws there. This gripper's open along its y (JAW_SLIDE_AXIS), and
    # "radial" used to mean its x along the arm: the jaws were across it.
    env, _, _ = _posed_so101(0.0)
    for slide in ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)):
        env._jaw_slide_axis = np.array(slide)
        for at in ((0.30, 0.0), (0.20, 0.15), (0.25, -0.10)):
            out = np.array(at) / np.linalg.norm(at)
            for jaws in ("radial", "tangential"):
                env._jaw_axis = jaws
                for deg in (0.0, -30.0):
                    opens = (env.tilted_rotation(deg, np.array([*at, 0.05])) @ np.array(slide))[:2]
                    along = abs(float(opens @ out)) / np.linalg.norm(opens)
                    assert along == pytest.approx(1.0 if jaws == "radial" else 0.0, abs=1e-9)


def test_renaming_the_jaw_axis_left_the_rigs_wrist_where_it_was():
    # 2026-10-21: "radial" with jaws along x-assumed became "tangential" with
    # the measured slide along y. The same rotation, so no calibration moved.
    env, _, _ = _posed_so101(0.0)
    at = np.array([0.25, 0.08, 0.05])
    env._jaw_axis, env._jaw_slide_axis = "radial", np.array([-1.0, 0.0, 0.0])
    before = env.tilted_rotation(-20.0, at)
    env._jaw_axis, env._jaw_slide_axis = "tangential", np.array([0.0, -1.0, 0.0])
    assert np.allclose(env.tilted_rotation(-20.0, at), before)


def test_aligned_at_another_tilt_the_servo_steps_the_jaws_over_the_object():
    # Rank 8 holds -35 deg: the target pixel's sight line is ~25 mm off the vertical there.
    env, pose, moves = _posed_so101(-35.0)
    start = pose["tcp"].copy()
    lean = env._servo_tilt_offset()
    assert 0.02 < np.linalg.norm(lean) < 0.03
    ok, msg = env.visual_servo_xy(_on_target(env), settle_s=0.0)
    assert ok and "stepped 2" in msg, msg
    # First to where aligning leaves the jaws, then — aligned — back over the object.
    assert np.allclose(moves[0][:2], start[:2] - lean) and np.allclose(moves[-1][:2], start[:2])
    assert all(m[2] == pytest.approx(start[2]) for m in moves)      # sideways only


def test_at_the_calibrations_own_tilt_and_height_the_servo_aims_as_calibrated():
    env, _, moves = _posed_so101(-5.0, at=(0.24, 0.0, 0.0167 + 0.0688))
    ok, msg = env.visual_servo_xy(_on_target(env), settle_s=0.0)
    assert ok and moves == [] and "stepped" not in msg


def test_without_a_calibrated_tilt_the_servo_aims_as_it_always_did():
    env, _, moves = _posed_so101(-35.0)
    env._gripper_cam_ref_tilt_deg = None
    ok, _ = env.visual_servo_xy(_on_target(env), settle_s=0.0)
    assert ok and moves == []



def test_the_orientation_search_starts_at_straight_down_or_at_the_lean_a_rule_asks_for():
    assert _tilts_from()[:3] == [0, 5, 10]
    assert _tilts_from(30)[:5] == [30, 25, 35, 20, 40]
    assert _tilts_from(-30)[0] == 30                       # a lean either way round
    assert sorted(_tilts_from(30)) == sorted(_tilts_from())   # nothing dropped, only reordered


class _PointKinematics:
    """
    The first three joints, in degrees, are the grasp point's x, y, z in mm.
    IK takes one damped step from where the arm measures, as placo's does.
    """
    joint_names = JOINT_NAMES + ["gripper"]

    def forward_kinematics(self, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q, float)[:3] * 1e-3
        return T

    def inverse_kinematics(self, current, desired, **_):
        q = np.asarray(current, float).copy()
        q[:3] += 0.5 * (np.asarray(desired, float)[:3, 3] * 1e3 - q[:3])
        return q


def _lagging_servo(follower, accel=0.6, vmax=6.0):
    """Each joint chases its latest goal one tick late, accelerating gently (Acceleration=50)."""
    vel = {n: 0.0 for n in JOINT_NAMES}
    pending = {}

    def send(action):
        follower.sent.append(dict(action))
        for name, goal in pending.items():
            here, v = follower.bus.positions[name], vel[name]
            gap = goal - here
            want = np.sign(gap) * min(vmax, np.sqrt(2 * accel * abs(gap)))
            v += float(np.clip(want - v, -accel, accel))
            vel[name] = v
            follower.bus.positions[name] = here + v
        pending.clear()
        pending.update({k.removesuffix(".pos"): float(g) for k, g in action.items()})
        return action
    return send


def _deadband_servo(follower, band=2.0):
    """Under load: a joint stops `band` short of its goal, and does not move for less."""
    def send(action):
        follower.sent.append(dict(action))
        for key, goal in action.items():
            name = key.removesuffix(".pos")
            here = follower.bus.positions[name]
            gap = float(goal) - here
            if abs(gap) > band:
                follower.bus.positions[name] = here + gap - np.sign(gap) * band
        return action
    return send


def _moving_so101(servo):
    env = offline_so101()
    env._kinematics = _PointKinematics()
    env._max_relative_target_deg = 8.0
    env._grasp_gap = lambda: 0.0
    env._tcp = lambda gap_m=None: np.zeros(3)
    env._half_turn_roll = lambda rot, q: None
    env._ik_limits = {}
    for n in JOINT_NAMES:
        env._robot.bus.positions[n] = 0.0
    env._robot.send_action = servo(env._robot)
    return env



def _perfect_servo(follower):
    """Each joint goes where it is sent."""
    def send(action):
        follower.sent.append(dict(action))
        for key, goal in action.items():
            follower.bus.positions[key.removesuffix(".pos")] = float(goal)
        return action
    return send


def test_joints_step_together_so_the_jaws_keep_to_the_line():
    # Clipped joint by joint, the joints with little to do finished first and
    # the jaws left the line: from a hover to a grasp the shoulder ran ahead of
    # the elbow and lifted the jaws 5 cm, and the stall check stopped the
    # descent 8-10 cm from its target (2026-10-20).
    env = _moving_so101(_perfect_servo)
    goal = np.array([0.012, 0.040, 0.0])
    path = []
    step = env.step
    env.step = lambda n=1: (step(n), path.append(env.get_ee_pos().copy()))
    assert env.move_ee_to(goal, tol=0.001, orientation=np.eye(3), max_step_deg=3.0) is True
    assert max(np.linalg.norm(np.cross(p, goal)) / np.linalg.norm(goal) for p in path) < 0.0005

def test_a_long_move_on_lagging_servos_arrives_instead_of_winding_up():
    # On the rig: travel moves "stalled" 2-3 cm short because the command, run
    # on ahead of the measured joints, wound up against the servos' lag — pan
    # swung 15 -> 24 -> 15 deg on a 0.7 s cycle until the stall check gave up.
    env = _moving_so101(_lagging_servo)
    assert env.move_ee_to(np.array([0.12, 0.08, 0.06]), tol=0.003, orientation=np.eye(3)) is True


def test_a_long_move_keeps_its_joints_together_when_one_is_slow():
    # From the park pose to a hover over c5 the elbow trailed its command by
    # ~6 deg while the shoulder ran on, and the jaws came down 3-5 cm below the
    # hover, 3-4 cm short of it (2026-10-28). Along one joint line, with the
    # slowest joint setting the pace, they keep to the line.
    def slow_y(follower):
        def send(action):
            follower.sent.append(dict(action))
            for key, goal in action.items():
                name = key.removesuffix(".pos")
                here = follower.bus.positions[name]
                most = 1.5 if name == "shoulder_lift" else 8.0
                follower.bus.positions[name] = here + float(np.clip(float(goal) - here, -most, most))
            return action
        return send
    env = _moving_so101(slow_y)
    goal = np.array([0.090, 0.060, -0.030])
    path = []
    step = env.step
    env.step = lambda n=1: (step(n), path.append(env.get_ee_pos().copy()))
    assert env.move_ee_to(goal, tol=0.002, orientation=np.eye(3)) is True

    def off(p, a, b):
        t = float(np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * (b - a))))
    above = goal + np.array([0.0, 0.0, 0.015])           # the joint line ends here, then straight down
    assert max(min(off(p, np.zeros(3), above), off(p, above, goal)) for p in path) < 0.009   # within a step
    assert min(p[2] for p in path) > goal[2] - 0.002     # never below it on the way


def test_a_long_move_whose_joint_line_would_cross_the_ceiling_starts_in_cartesian_space():
    # Out of the park pose, folded back 10 mm under the workspace's ceiling,
    # the joint line to a hover rose 7-14 mm first: the safety monitor halted
    # the pick (2026-10-28). The Cartesian loop starts it; the line takes over nearer.
    from types import SimpleNamespace
    env = _moving_so101(_perfect_servo)
    env._stream_bounds = SimpleNamespace(z_max=0.003)
    lines = []
    joint_space = env._joint_space_to
    env._joint_space_to = lambda q, step: (lines.append(env.get_ee_pos().copy()), joint_space(q, step))
    path = []
    step = env.step
    env.step = lambda n=1: (step(n), path.append(env.get_ee_pos().copy()))
    goal = np.array([0.150, 0.0, -0.060])
    assert env.move_ee_to(goal, tol=0.002, orientation=np.eye(3)) is True
    assert len(lines) == 1 and 0.0 < np.linalg.norm((goal - lines[0])[:2]) <= 0.10   # handed over, nearer
    assert max(p[2] for p in path) <= 0.0005


def test_a_nudge_inside_the_deadband_arrives_without_swinging_past():
    # At a grasp leaning 15-20 deg, 2-4 mm corrections streamed against the
    # deadband stalled with the jaws swung 7.6 mm round the piece (2026-10-28).
    def lagging_with_deadband(follower):
        lag = _lagging_servo(follower)

        def send(action):
            before = dict(follower.bus.positions)
            out = lag(action)
            for key, goal in action.items():          # a command within 1.5 deg does not move it
                name = key.removesuffix(".pos")
                if abs(float(goal) - before[name]) < 1.5:
                    follower.bus.positions[name] = before[name]
            return out
        return send
    env = _moving_so101(lagging_with_deadband)
    goal = np.array([0.0, 0.003, 0.0])
    path = []
    step = env.step
    env.step = lambda n=1: (step(n), path.append(env.get_ee_pos().copy()))
    assert env.move_ee_to(goal, tol=0.001, orientation=np.eye(3), max_step_deg=3.0) is True
    assert max(np.linalg.norm(p) for p in path) <= np.linalg.norm(goal) + 0.0005   # never swung past it


def test_a_small_move_against_a_deadband_still_arrives():
    # What running the command on ahead is for: a joint the servo holds 2 deg
    # short is asked further until it moves (2026-10-19, "it doesn't bend the shoulder").
    env = _moving_so101(_deadband_servo)
    assert env.move_ee_to(np.array([0.006, 0.0, 0.004]), tol=0.0015, orientation=np.eye(3)) is True


def test_a_gentle_descent_against_a_deadband_as_wide_as_its_steps_arrives():
    # A descent's 3 deg steps against a 3 deg deadband: a plain command never
    # gets past it, pushing only after a still tick moves it every other tick —
    # on the rig a place's descent at d5, the arm reaching out, stalled 5-6 cm short.
    env = _moving_so101(lambda f: _deadband_servo(f, band=3.0))
    assert env.move_ee_to(np.array([0.0, 0.0, -0.05]), tol=0.004, orientation=np.eye(3),
                          max_step_deg=3.0) is True


def _loaded_servo(follower, sag=1.2, creep=0.02):
    """Under load: a joint parks `sag` deg short of its goal; the integral creeps that in `creep` deg a tick."""
    wound = {n: 0.0 for n in JOINT_NAMES}

    def send(action):
        follower.sent.append(dict(action))
        for key, goal in action.items():
            name = key.removesuffix(".pos")
            wound[name] = min(sag, wound[name] + creep)
            follower.bus.positions[name] = float(goal) - (sag - wound[name])
        return action
    return send


def test_a_settle_closes_a_joints_load_offset_instead_of_waiting_on_the_integral():
    # On the rig each move's settle took 0.6-1.5 s, the shoulder or elbow
    # parked ~1 deg short while the integral crept it in (2026-10-28).
    env = _moving_so101(_loaded_servo)
    goal = {n: 10.0 for n in JOINT_NAMES}
    env._send({f"{n}.pos": v for n, v in goal.items()})          # parked 1.2 deg short
    env._robot.sent.clear()
    env._settle_to(goal, timeout_s=60.0)
    assert len(env._robot.sent) <= 5                              # not the ~50 ticks of creep
    assert env._robot.sent[-1] == {f"{n}.pos": 10.0 for n in JOINT_NAMES}   # left holding the goal itself
    assert all(max(s.values()) <= 10.0 + 2.0 for s in env._robot.sent)       # never asked far past it


def test_a_move_that_stalls_leaves_the_arm_held_where_it_stopped():
    # At a grasp on d5 (2026-10-28) two 2-3 mm corrections stalled and left
    # their last command standing, run on ahead of the joints: the jaws carried
    # on 7.7 mm round the queen after the moves had given up.
    def blocked_pan(follower):
        send = _deadband_servo(follower, band=0.0)

        def blocked(action):
            pan = follower.bus.positions["shoulder_pan"]
            out = send(action)
            follower.bus.positions["shoulder_pan"] = pan         # against something
            return out
        return blocked
    env = _moving_so101(blocked_pan)
    assert env.move_ee_to(np.array([0.03, 0.0, 0.0]), tol=0.002, orientation=np.eye(3)) is False
    here = env._robot.bus.positions
    assert env._robot.sent[-1] == {f"{n}.pos": here[n] for n in JOINT_NAMES}


def test_a_move_need_not_wait_to_settle_but_is_left_heading_for_its_goal():
    env = _moving_so101(_lagging_servo)
    settles = []
    env._settle_to = lambda commanded, **_: settles.append(commanded)
    assert env.move_ee_to(np.array([0.03, 0.0, 0.02]), tol=0.003, orientation=np.eye(3), settle=False)
    assert settles == []
    last = env._robot.sent[-1]
    assert np.allclose([last[f"{n}.pos"] for n in ("shoulder_pan", "shoulder_lift", "elbow_flex")],
                       [30.0, 0.0, 20.0], atol=3.0)               # where the solver says, not run on ahead


def test_the_jaws_centre_slides_half_the_opening_toward_the_moving_jaw_along_its_axis():
    env = offline_so101()
    env._tcp_offset, env._jaw_centre_slide, env._gripper_max_open_m = np.zeros(3), 0.5, 0.045
    env._jaw_slide_axis = np.array([0.0, -1.0, 0.0])        # this rig: the gripper frame's -y
    assert np.allclose(env._tcp(0.040), [0.0, -0.020, 0.0])
    assert np.allclose(env._tcp(0.0), 0.0)                   # closed, it makes no difference
    del env._jaw_slide_axis                                  # an env built before the axis: -x, as it was
    assert np.allclose(env._tcp(0.040), [-0.020, 0.0, 0.0])


def test_a_joint_can_be_given_its_whole_calibrated_range_over_the_urdfs():
    from mrcf.backends.so101.kinematics import effective_ik_limits
    calibrated = {"wrist_flex": (-101.2, 101.2), "elbow_flex": (-97.0, 97.0)}
    urdf = {"wrist_flex": (-95.0, 95.0), "elbow_flex": (-90.0, 90.0)}
    assert effective_ik_limits(calibrated, urdf) == {"wrist_flex": (-95.0, 95.0), "elbow_flex": (-90.0, 90.0)}
    # 2026-10-21: d1 is out of reach at the URDF's 95, in reach at the servo's 101.
    assert effective_ik_limits(calibrated, urdf, ("wrist_flex",)) == {"wrist_flex": (-101.2, 101.2),
                                                                     "elbow_flex": (-90.0, 90.0)}


def test_a_trip_from_another_thread_mid_move_holds_the_arm_at_once():
    env = offline_so101()
    env._send({f"{n}.pos": 12.0 for n in JOINT_NAMES})     # a move is under way
    assert env.may_be_moving()
    t = threading.Thread(target=env.interlock.trip, args=("hand near the arm", "hands"))
    t.start()
    t.join()
    # The hold went out from the tripping thread, before any next command...
    assert env._robot.sent[-1] == {f"{n}.pos": 10.0 for n in JOINT_NAMES}
    # ...and the command that then finds it tripped sends nothing more.
    with pytest.raises(MotionStopped):
        env._send({f"{n}.pos": 14.0 for n in JOINT_NAMES})
    assert len(env._robot.sent) == 2


def test_the_sentry_can_read_the_joints_while_it_pauses_a_move():
    # The sentry reads where the arm is (arm_shape) on every frame, and only it
    # can end its own pause: a move waiting out the pause must not hold the bus.
    env = offline_so101()
    env.interlock.pause_only("motion")
    env.interlock.trip("hand", source="motion")
    done = []
    move = threading.Thread(target=lambda: done.append(
        env.goto_joint_degrees({n: 10.0 for n in JOINT_NAMES}, timeout_s=1.0)), daemon=True)
    move.start()
    move.join(0.2)
    assert move.is_alive()                         # paused, not abandoned
    read = []
    sentry = threading.Thread(target=lambda: read.append(env._motor_pos()), daemon=True)
    sentry.start()
    sentry.join(1.0)
    assert read, "the paused move held the motor bus"
    env.interlock.clear("motion")
    move.join(2.0)
    assert done == [True]


def test_a_trip_while_the_arm_is_at_rest_sends_nothing():
    # Re-commanding a resting arm where it has sagged to would creep it lower
    # with every hand that reaches in between commands.
    env = offline_so101()
    env.interlock.trip("hand near the arm", source="hands")
    assert not env.may_be_moving() and env._robot.sent == []


def test_arm_shape_follows_the_joints_and_the_jaw():
    pytest.importorskip("placo")
    from mrcf.backends.so101 import robot_config as rc
    if not Path(rc.URDF_PATH).is_file():
        pytest.skip("no URDF — run fetch_urdf.py")
    env = offline_so101()
    env._setup_kinematics(rc.URDF_PATH, None)
    env._gripper_width_table, env._gripper_max_open_m = None, 0.04
    env._tcp_offset, env._jaw_slide_axis = np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    shape = env.arm_shape()
    assert np.allclose(shape[0][0], 0.0)                       # the base first
    tip = shape[-2][0]
    env._robot.bus.positions["shoulder_pan"] = 40.0
    turned = env.arm_shape()
    # The pan turns the arm about the shoulder's vertical axis: same reach, same height.
    axis = shape[1][0][:2]
    assert np.isclose(np.linalg.norm(turned[-2][0][:2] - axis), np.linalg.norm(tip[:2] - axis), atol=1e-3)
    assert np.isclose(turned[-2][0][2], tip[2], atol=1e-6) and not np.allclose(turned[-2][0], tip)
    env._robot.bus.positions["gripper"] = 100.0                # open: the moving jaw's tip moves
    opened = env._pct_to_width(100.0) - env._pct_to_width(5.0)
    assert np.isclose(np.linalg.norm(env.arm_shape()[-1][0] - turned[-1][0]), opened, atol=1e-6)


def test_padded_jaws_pressing_each_other_are_not_a_grip(monkeypatch):
    # 2026-10-22, grip pads fitted: three empty closes stopped at 0.9 % and read
    # load 160 of 200 — a grip by load alone. A held pawn stops near 8 %.
    import mrcf.backends.so101.gripper as so101_gripper
    monkeypatch.setattr(so101_gripper.time, "sleep", lambda s: None)
    env = offline_so101()
    env._gripper_max_open_m, env._gripper_torque_limit, env._gripper_min_hold_m = 0.045, 200, 0.0034
    env._gripper_width_table = [(0.0, 0.0), (5.2, 0.010), (9.9, 0.020), (15.1, 0.025)]
    pos, sent = {"gripper": 0.9}, []
    env._motor_pos = lambda obs=None: dict(pos)
    env._send = lambda action: sent.append(action)
    env.step = lambda n=1: None
    env._bus_read = lambda register, motor: 160
    width, loaded = env.squeeze_gripper()
    assert width < 0.0034 and not loaded and not env.gripper_loaded()
    assert sent[-1] == {"gripper.pos": 0.9}            # stops pushing pad into pad
    pos["gripper"] = 8.0
    width, loaded = env.squeeze_gripper()
    assert loaded and env.gripper_loaded() and width == pytest.approx(0.016, abs=0.001)
    # Carried as hard as it was squeezed: the goal stays GRIPPER_HOLD_PCT past
    # the jaws. 2026-10-30: eased to 3 %, a pawn slipped out at the carry's start.
    assert sent[-1] == {"gripper.pos": 8.0 - 3.0}                  # offline_so101's default
    from mrcf import profiles
    hold = profiles.settings(profiles.load(Path(__file__).resolve().parents[1] / "deployments" / "chess")
                             )["GRIPPER_HOLD_PCT"]          # chess's, for its pieces
    env._gripper_hold_pct = hold
    env.squeeze_gripper()
    assert sent[-1] == {"gripper.pos": max(0.0, 8.0 - hold)}


def test_a_thin_wall_is_held_by_the_load_when_the_squeeze_is_told_how_thin(monkeypatch):
    # 2026-11-17: a scoop's wall stopped the jaws at 1.1 mm, load 113 of 200; an empty close
    # read 0.14 mm, load 36. Held by the load — and let go as "nothing" under the 3.4 mm a piece needs.
    import mrcf.backends.so101.gripper as so101_gripper
    monkeypatch.setattr(so101_gripper.time, "sleep", lambda s: None)
    env = offline_so101()
    env._gripper_max_open_m, env._gripper_torque_limit, env._gripper_min_hold_m = 0.045, 200, 0.0034
    env._gripper_width_table = [(0.0, 0.0), (5.2, 0.010), (9.9, 0.020), (15.1, 0.025)]
    pos, load = {"gripper": 0.57}, {"now": 113}                   # 1.1 mm
    env._motor_pos = lambda obs=None: dict(pos)
    env._send = lambda action: None
    env.step = lambda n=1: None
    env._bus_read = lambda register, motor: load["now"]
    assert env.squeeze_gripper()[1] is False                      # a piece's rule: nothing
    width, loaded = env.squeeze_gripper(min_hold_m=0.0008)
    assert loaded and width == pytest.approx(0.0011, abs=0.0002)
    assert env.gripper_loaded()                                   # and still held on the way
    pos["gripper"], load["now"] = 0.07, 36                        # an empty close: 0.14 mm, no load
    assert env.squeeze_gripper(min_hold_m=0.0008)[1] is False and not env.gripper_loaded()


def test_the_so101_has_every_capability_the_policies_can_use():
    from mrcf import capabilities
    env = offline_so101()
    missing = [name for name in capabilities.__all__ if not isinstance(env, getattr(capabilities, name))]
    assert missing == ["Freedrive", "BackOff"]  # its torque off, it falls: no freedrive; back-off: the UR5 only so far


def test_jaws_that_have_arrived_are_not_waited_on():
    # The stand-in's jaws move straight to their goal; once there, nothing moves.
    env = offline_so101()
    env._gripper_max_open_m, env._gripper_width_table, env._held_width_m = 0.045, None, None
    bus = env._robot.bus

    def to_goal(action):
        bus.positions["gripper"] = action.get("gripper.pos", bus.positions["gripper"])
        return action

    env._robot.send_action = to_goal
    ticks = []
    env.step = lambda n=1: ticks.append(n)
    env.set_gripper(0.030, n_steps=15)
    assert bus.positions["gripper"] == pytest.approx(0.030 / 0.045 * 100)
    # Two cycles: the move itself, and one reading that shows it has stopped —
    # not the fifteen of settling it used to wait out regardless.
    assert sum(ticks) <= 3


def test_one_reading_serves_a_whole_control_tick():
    # The loop's reading, then the safety monitor's two (arm and jaw opening)
    # a few ms later: one bus round trip, not three. A command sent, and the
    # next reading is a new one.
    env = offline_so101()
    env.POS_REUSE_S = 0.010
    env._gripper_max_open_m, env._gripper_width_table = 0.045, None
    env._motor_pos(), env._q_arm_deg(), env.get_gripper_width()
    assert env._robot.bus.reads == 1
    env._send({f"{n}.pos": 12.0 for n in JOINT_NAMES})
    env._motor_pos()
    assert env._robot.bus.reads == 2


# ── Streamed goals (stream_joints): a learned policy ─────────────────────────

class _DownKinematics:
    """Stands in for RobotKinematics: the gripper straight down at x=0.25, shoulder_lift's degrees its height in mm."""
    joint_names = JOINT_NAMES + ["gripper"]

    def forward_kinematics(self, q):
        t = np.eye(4)
        t[:3, :3] = np.diag([1.0, -1.0, -1.0])         # the approach axis (+z) points down
        t[:3, 3] = [0.25, 0.0, q[1] / 1000.0]
        return t


def streaming_so101(floor=0.0, bounds=None) -> SO101Env:
    env = offline_so101()
    env._kinematics = _DownKinematics()
    env._stream_tip_floor_z, env._stream_bounds = floor, bounds
    env._jaw_tip_below_m, env._jaw_finger_half_width_m = 0.007, 0.008
    env._tcp_offset, env._jaw_slide_axis, env._jaw_centre_slide = np.zeros(3), np.array([0.0, -1.0, 0.0]), 0.5
    env._gripper_width_table, env._gripper_max_open_m = None, 0.045
    return env


def _goal(**joints):
    return {**{n: 20.0 for n in JOINT_NAMES}, "gripper": 30.0, **joints}


def test_a_streamed_goal_clear_of_the_board_goes_out_whole():
    env = streaming_so101()
    sent, why = env.stream_joints(_goal())                 # jaw tips 13 mm up
    assert why is None and sent == _goal()
    assert env._robot.sent[-1] == {f"{n}.pos": v for n, v in _goal().items()}


def test_a_streamed_goal_into_the_board_keeps_the_arm_and_the_gripper_still_follows():
    env = streaming_so101()
    env.stream_joints(_goal())
    sent, why = env.stream_joints(_goal(shoulder_lift=5.0, gripper=40.0))   # tips 2 mm under
    assert "below the floor" in why
    assert sent == _goal(gripper=40.0)                     # the arm where the last goal that passed had it


def test_a_stream_that_went_quiet_starts_again_from_where_the_arm_is():
    # Kept from a stream that ended long ago, the last goal would send the arm
    # back to wherever that stream left off.
    env = streaming_so101()
    env.stream_joints(_goal())
    env.__dict__["_stream_t"] -= env.STREAM_GAP_S + 0.1
    sent, why = env.stream_joints(_goal(shoulder_lift=5.0))
    assert why and all(sent[n] == 10.0 for n in JOINT_NAMES)   # the arm as it reads now


def test_a_streamed_goal_out_of_the_workspace_keeps_the_arm_and_the_gripper_still_follows():
    from mrcf.safety import WorkspaceBounds
    env = streaming_so101(floor=None, bounds=WorkspaceBounds(z_max=0.040))   # jaws up to 20 mm
    sent, why = env.stream_joints(_goal(shoulder_lift=25.0))                # 25 mm: out
    assert "z=+0.025" in why
    assert sent == {**{n: 10.0 for n in JOINT_NAMES}, "gripper": 30.0}       # the arm as it is


def test_streamed_goals_are_clipped_to_the_calibrated_range_without_a_word(capsys):
    env = streaming_so101(floor=None)
    env._joint_deg_limits = {"elbow_flex": (-90.0, 90.0)}
    sent, _ = env.stream_joints(_goal(elbow_flex=120.0))
    assert sent["elbow_flex"] == 90.0 and capsys.readouterr().out == ""


def test_a_tripped_interlock_stops_a_stream():
    env = streaming_so101()
    env.interlock.trip("hand", source="hands")
    with pytest.raises(MotionStopped):
        env.stream_joints(_goal())
    assert all(cmd != {f"{n}.pos": v for n, v in _goal().items()} for cmd in env._robot.sent)


def test_a_stream_keeps_a_joint_inside_its_stream_limits():
    env = streaming_so101(floor=None)
    env._stream_joint_limits = {"wrist_roll": (-80.0, 80.0)}
    sent, why = env.stream_joints(_goal(wrist_roll=-119.0))
    assert sent["wrist_roll"] == -80.0 and why is None


def test_a_stream_stays_short_of_the_ceiling_the_safety_monitor_halts_at():
    from mrcf.safety import WorkspaceBounds
    env = streaming_so101(floor=None, bounds=WorkspaceBounds(z_max=0.030))
    _, why = env.stream_joints(_goal(shoulder_lift=12.0))       # 12 mm: past z_max less the margin
    assert why and "z=" in why
    env.__dict__["_stream_t"] -= env.STREAM_GAP_S + 0.1
    _, why = env.stream_joints(_goal(shoulder_lift=9.0))
    assert why is None


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s


def test_a_joint_pushed_against_something_that_does_not_give_is_held_where_it_stopped(monkeypatch):
    import mrcf.backends.so101.stream as env_module
    clock = _Clock()
    monkeypatch.setattr(env_module, "time", clock)
    env = streaming_so101(floor=None)                         # the stand-in's joints never move: 10 deg
    for _ in range(40):                                       # 1.3 s of the roll asked 30 deg further
        sent, why = env.stream_joints(_goal(wrist_roll=-20.0))
        clock.now += 1 / 30
    assert sent["wrist_roll"] == 10.0 and "wrist_roll blocked at 10 deg" in why
    assert sent["shoulder_lift"] == 20.0                      # the others go where they are sent
    # Not pushed that way again, however long the goal stays there...
    sent, _ = env.stream_joints(_goal(wrist_roll=-5.0))
    assert sent["wrist_roll"] == 10.0
    # ...but free to come back off it.
    sent, why = env.stream_joints(_goal(wrist_roll=14.0))
    assert sent["wrist_roll"] == 14.0 and why is None


def test_a_squeezing_gripper_is_not_a_blocked_joint(monkeypatch):
    import mrcf.backends.so101.stream as env_module
    clock = _Clock()
    monkeypatch.setattr(env_module, "time", clock)
    env = streaming_so101(floor=None)
    for _ in range(40):
        sent, why = env.stream_joints({**{n: 10.0 for n in JOINT_NAMES}, "gripper": 40.0})
        clock.now += 1 / 30
    assert sent["gripper"] == 40.0 and why is None


def test_a_stream_that_starts_outside_the_workspace_can_come_back_in_but_not_go_further_out():
    # The overhead parking pose is behind the workspace: every goal from there
    # was refused, the way back in included, and the arm never left it.
    from mrcf.safety import WorkspaceBounds
    env = streaming_so101(floor=None, bounds=WorkspaceBounds(z_max=0.030))
    env._robot.bus.positions["shoulder_lift"] = 20.0          # the jaws at 20 mm: over the 10 mm allowed
    sent, why = env.stream_joints(_goal(shoulder_lift=25.0))   # further out: refused
    assert "z=" in why and sent["shoulder_lift"] == 20.0
    sent, why = env.stream_joints(_goal(shoulder_lift=15.0))   # on the way back in: goes
    assert why is None and sent["shoulder_lift"] == 15.0
    sent, why = env.stream_joints(_goal(shoulder_lift=17.0))   # out again past where it got to: refused
    assert why and sent["shoulder_lift"] == 15.0


def test_a_streams_joint_ranges_are_the_calibrated_ones_inside_the_stream_limits():
    env = streaming_so101()
    env._joint_deg_limits = {"wrist_roll": (-180.0, 180.0), "elbow_flex": (-92.0, 92.0)}
    env._stream_joint_limits = {"wrist_roll": (-80.0, 80.0)}
    assert env.stream_joint_limits() == {"wrist_roll": (-80.0, 80.0), "elbow_flex": (-92.0, 92.0)}


def test_a_joint_that_stood_still_and_is_then_asked_to_move_is_not_taken_for_blocked(monkeypatch):
    # 2026-10-26: the shoulder rested at its limit, the leader was lifted
    # briskly, and the goal 15 deg away was set against the last second's
    # stillness: "blocked", and held down for the rest of the run.
    import mrcf.backends.so101.stream as env_module
    clock = _Clock()
    monkeypatch.setattr(env_module, "time", clock)
    env = streaming_so101(floor=None)
    for _ in range(60):                                       # 2 s still, asked to stay
        env.stream_joints(_goal(shoulder_lift=10.0))
        clock.now += 1 / 30
    sent, why = env.stream_joints(_goal(shoulder_lift=25.0))  # then lifted
    assert why is None and sent["shoulder_lift"] == 25.0


def test_a_goal_the_floor_refuses_is_no_push(monkeypatch):
    import mrcf.backends.so101.stream as env_module
    clock = _Clock()
    monkeypatch.setattr(env_module, "time", clock)
    env = streaming_so101()
    for _ in range(60):                                       # 2 s of goals into the board, all refused
        sent, why = env.stream_joints(_goal(shoulder_lift=-20.0))
        clock.now += 1 / 30
    assert "floor" in why and "blocked" not in why


def test_an_so101_lists_every_camera_it_opened_and_none_it_did_not():
    env = offline_so101()
    env._overhead_camera_name = "overhead"
    env._robot.cameras = {"front": _Camera(), "overhead": _Camera(), "side": _Camera()}
    assert env.list_cameras() == ["front", "overhead", "side"]
    env._camera_name = None                                    # a rig with no wrist camera
    env._robot.cameras = {"overhead": _Camera()}
    assert env.list_cameras() == ["overhead"]


def test_an_so101_opens_its_rigs_other_cameras_beside_its_own(tmp_path, monkeypatch):
    # Made, not connected: LeRobot builds the cameras from their configs and opens nothing.
    pytest.importorskip("lerobot")
    from mrcf.backends.rig import Camera

    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    env = SO101Env(port="/dev/nonexistent", robot_id="test", camera_name=None, overhead_camera_name="overhead",
                   overhead_camera_index=1, overhead_camera_width=1280, overhead_camera_height=720, overhead_camera_fps=30,
                   cameras={"side": Camera(mount="fixed", source=2, size=(1280, 720))})
    assert env.list_cameras() == ["overhead", "side"]
    assert env._robot.config.cameras["side"].width == 1280 and env._robot.config.cameras["side"].index_or_path == 2


def test_an_so101s_other_camera_is_placed_by_its_own_calibration():
    from mrcf.backends.rig import Camera

    env = offline_so101()
    env._cameras = {"side": Camera(mount="fixed", homography=np.diag([0.001, 0.001, 1.0]))}
    assert np.allclose(env.pixel_to_world(100, 200, 0.0, "side"), [0.1, 0.2, 0.0])


def test_the_wrist_cameras_pose_takes_its_own_joint_corrections_and_nothing_else_does():
    # GRIPPER_CAM_JOINT_DEG was fitted with GRIPPER_CAM_POSE: the camera's pose
    # reads the joints through it; the gripper's own frame (where the jaws go) does not.
    class _Kin:
        def forward_kinematics(self, q):
            T = np.eye(4)
            T[:3, 3] = np.asarray(q, float)[:3] / 1000.0     # mm of position per degree, to see what was asked
            return T

    env = offline_so101()
    env._init_rig(camera_name="front", gripper_cam_k=np.diag([500.0, 500.0, 1.0]), gripper_cam_pose=np.eye(4))
    env._kinematics = _Kin()
    env._q_arm_deg = lambda obs=None: np.array([10.0, 20.0, 30.0, 0.0, 0.0])
    assert np.allclose(env.camera_model("front", 640, 480)["T_world_cam"][:3, 3], [0.010, 0.020, 0.030])
    assert "to_arm" not in env.camera_model("front", 640, 480)
    env._gripper_cam_joint_deg = np.array([1.0, 1.0, 4.0, 0.0, 0.0])
    model = env.camera_model("front", 640, 480)
    assert np.allclose(model["T_world_cam"][:3, 3], [0.011, 0.021, 0.034])
    assert np.allclose(env.gripper_frame()[:3, 3], [0.010, 0.020, 0.030])
    # What the camera finds, in the frame the arm moves by: where it stands from the jaws kept.
    assert np.allclose((model["to_arm"] @ [0.011, 0.021, 0.034, 1.0])[:3], [0.010, 0.020, 0.030])


def test_a_joint_is_planned_no_further_than_what_it_meets_before_its_calibrated_stop():
    # 2026-11-03: a stand-up planned the wrist roll at -88 deg; it stopped 8 deg
    # short against the wrist camera's cable, and the arm 4 cm short of its hover.
    from mrcf.backends.so101.kinematics import within_limits
    limits = {"wrist_roll": (-150.0, 150.0), "elbow_flex": (-90.0, 90.0)}
    capped = within_limits(limits, {"wrist_roll": (-80.0, 80.0), "not_a_joint": (0.0, 1.0)})
    assert capped == {"wrist_roll": (-80.0, 80.0), "elbow_flex": (-90.0, 90.0)}
    assert within_limits(limits, None) == limits


class _SlowToWake:
    """A camera that opens but sends nothing in time on its first `silent` connects."""

    def __init__(self, silent=1):
        self.silent, self.connects, self.is_connected, self.latest_frame = silent, 0, False, None

    def connect(self):
        self.connects += 1
        self.is_connected = True
        if self.connects <= self.silent:
            raise TimeoutError("Timed out waiting for frame from camera OpenCVCamera(0) after 1000 ms.")
        self.latest_frame = np.zeros((2, 2, 3), np.uint8)

    def disconnect(self):
        self.is_connected = False


def test_a_camera_that_sends_nothing_at_connect_is_opened_again(monkeypatch):
    # 2026-11-03 and 11-04: the wrist camera timed out at start-up, and the
    # controller exited; a fresh process read it at once.
    import mrcf.backends.so101.follower as follower
    monkeypatch.setattr(follower.time, "sleep", lambda s: None)
    env = offline_so101()
    env._tune_servos = lambda: None
    env._start_sentry = lambda: None             # the rig's watch on the cameras: not under test here
    wrist = _SlowToWake(silent=1)
    env._robot.cameras = {"front": wrist}
    configured = []

    def connect(calibrate=True):
        env._robot.bus.is_connected = True
        wrist.connect()

    env._robot.connect = connect
    env._robot.configure = lambda: configured.append(True)
    env.connect()
    assert wrist.connects == 2 and wrist.latest_frame is not None and configured

    # One that never sends, after its retries: the overhead camera fails the
    # connect; the wrist camera is set aside (2026-11-14), and the arm runs without it.
    for name, starts in (("overhead", False), ("front", True)):
        env = offline_so101()
        env._overhead_camera_name = "overhead"
        env._tune_servos = lambda: None
        env._start_sentry = lambda: None
        env.stop = lambda: None
        dead = _SlowToWake(silent=99)
        env._robot.cameras = {name: dead}
        env._robot.connect = lambda calibrate=True: (setattr(env._robot.bus, "is_connected", True), dead.connect())
        env._robot.configure = lambda: None
        if starts:
            env.connect()
            assert env._running and name not in env._robot.cameras and env._closed_cameras[name] is dead
        else:
            with pytest.raises(TimeoutError):
                env.connect()
        assert dead.connects == 1 + env.CONNECT_CAMERA_RETRIES


def test_a_gentle_move_held_to_a_millimetre_on_lagging_servos_does_not_swing_back_and_forth():
    # 2026-11-04, the settle at a grasp on e5 (10 mm, held to 1 mm, 3 deg steps):
    # shoulder_pan swung -6.5 -> -2.6 -> -6.5 deg every 0.6 s, its command
    # -2 -> -7 with it — the jaws ~2 cm side to side over the piece. The command,
    # pushed on past where the joint lagged, ran past the solver's own goal.
    env = _moving_so101(_lagging_servo)
    goal = np.array([0.0, 0.010, 0.0])
    path = []
    step = env.step
    env.step = lambda n=1: (step(n), path.append(env.get_ee_pos().copy()))
    env.move_ee_to(goal, tol=0.001, orientation=np.eye(3), max_step_deg=3.0)
    y = np.array([p[1] for p in path])
    assert y.max() <= goal[1] + 0.0015                        # never swung past it by more than a step's settle
    v = np.diff(y)
    moving = v[np.abs(v) > 1e-5]
    assert int(np.sum(np.sign(moving[1:]) != np.sign(moving[:-1]))) <= 2   # no back and forth
    assert abs(y[-1] - goal[1]) <= 0.0015


# ── A pick's descent at the edge of reach (descent_line_above_m) ──────────────

def _real_so101_on_loaded_servos(monkeypatch):
    """The SO-101's own kinematics, its servos as the flight recorder has them (tools/so101/replay_descents.py)."""
    import arm_baselines as ab
    from mrcf import backends
    from tools.so101 import replay_descents

    monkeypatch.setattr(time, "sleep", lambda s: None)
    from mrcf.grasp_correction import GraspCorrection

    try:
        env = ab.offline_env(backends.arm("so101"))
    except ab.Skipped as exc:
        pytest.skip(str(exc))
    # The rig's measured jaw table (rig/so101/grasp_correction.json, 2026-10-31) as the runs replayed here had it,
    # not whatever this computer's rig folder holds: CI has none, and without it the old descent reached.
    env._grasp_table = GraspCorrection.load(Path(__file__).parent / "fixtures" / "so101_grasp_correction.json")
    return replay_descents.loaded(env)


def test_a_leaning_descent_at_the_edge_of_reach_goes_most_of_the_way_by_one_joint_line(monkeypatch):
    """2026-11-19, run 39's cap 32 cm out at -20 deg (and run 37's bobbing): the loop went up and out."""
    env = _real_so101_on_loaded_servos(monkeypatch)
    tz, gap = env._floor_z + 0.016, 0.036
    goal, hover = np.array([0.289, 0.147, tz + 0.019]), np.array([0.289, 0.147, tz + 0.077])
    rot = np.asarray(env.approach_orientation_for([hover, goal], prefer_tilt_deg=-20.0)[0])
    start = env._probe(rot, hover, env._tcp(gap), gap, np.r_[env._q_arm_deg(), 0.0], 0.002, None)

    def descend(line_above_m):
        for n, v in zip(JOINT_NAMES, start[:5]):
            env._robot.bus.positions[n] = float(v)
        env._descent_line_above_m = line_above_m
        ok = env.move_ee_to(goal, tol=0.004, max_step_deg=3.0, orientation=rot)
        q = np.array([env._robot.bus.positions[n] for n in JOINT_NAMES])
        return ok, float(np.linalg.norm(env._grasp_point(np.append(q, 0.0), gap) - goal))

    ok, off = descend(None)
    assert not ok and off > 0.03                                   # as it was: off its line, stopped
    ok, off = descend(0.025)
    assert ok and off < 0.004, off
    # A travel move, or a short one, is not taken this way.
    assert env._descend_by_joint_line(np.r_[start[:5], 0.0], rot, hover + [0.05, 0.0, 0.0], gap, 3.0) is False
    assert env._descend_by_joint_line(np.r_[start[:5], 0.0], rot, hover - [0.0, 0.0, 0.02], gap, 3.0) is False


def test_a_push_the_floor_guard_would_refuse_is_sent_as_the_step_itself():
    """2026-11-19, run 40: a reseat's stuck joint pushed 3 deg past its goal, refused, the jaws left 15 mm up."""
    env = offline_so101()
    env._ik_limits = {}
    env._below_floor = lambda action: "into the table" if action["elbow_flex.pos"] > 10.0 else None
    pushed, plain = {n: 0.0 for n in JOINT_NAMES}, {n: 0.0 for n in JOINT_NAMES}
    pushed["elbow_flex"], plain["elbow_flex"] = 12.0, 8.0
    env._floor_caps_push = False
    assert env._floor_kept(pushed, plain) == pushed                   # as it was: pushed, and the guard refuses
    env._floor_caps_push = True
    assert env._floor_kept(pushed, plain)["elbow_flex"] == 8.0         # the step itself
    plain["elbow_flex"] = 11.0
    assert env._floor_kept(pushed, plain) == pushed                   # the step itself goes in: the guard's to stop


def test_a_reseat_down_onto_a_low_grip_is_not_refused_for_its_push(monkeypatch):
    """Run 40's cap reseat, 15 mm down to a grip 16 mm over the table, leaning 25 deg, on loaded servos."""
    env = _real_so101_on_loaded_servos(monkeypatch)
    env._floor_z, gap = 0.0007, 0.036
    start, goal = np.array([0.28118, 0.14263, 0.0317]), np.array([0.28118, 0.14263, 0.0167])
    rot = np.asarray(env.approach_orientation_for([start, goal], prefer_tilt_deg=-25.0)[0])
    q = env._probe(rot, start, env._tcp(gap), gap, np.r_[env._q_arm_deg(), 0.0], 0.002, None)

    def reseat(caps):
        for n, v in zip(JOINT_NAMES, q[:5]):
            env._robot.bus.positions[n] = float(v)
        env._floor_caps_push, env.move_refusals = caps, 0
        env.move_ee_to(goal, tol=0.004, max_step_deg=3.0, orientation=rot)
        return env.move_refusals
    assert reseat(False) == 1                                           # as it was: refused
    assert reseat(True) == 0
