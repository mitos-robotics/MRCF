"""
mrcf/backends/rig.py on an arm of its own: what a new arm gets from giving RigEnv
its gripper's pose — and that such an arm keeps the backend contract.

_CartesianArm is an arm whose controller moves the grasp point itself (as a
UR's does: moveL to a pose), with the jaws opening symmetrically — the
smallest RigEnv there is, and the shape a UR5 backend starts from.
"""

import sys
import threading
import types

import numpy as np
import pytest

from mrcf import capabilities
from mrcf.backends.contract import check_backend, check_lifecycle
from mrcf.backends.rig import RigEnv

DOWN = np.diag([1.0, -1.0, -1.0])         # gripper z (its approach axis) pointing at the table
TCP_M = np.array([0.0, 0.0, 0.15])        # flange to where the jaws close, along the approach axis


class _CartesianArm(RigEnv):
    STEP_M = 0.01

    def __init__(self, **rig):
        self._init_rig(**{"camera_name": "wrist", "tcp_offset": TCP_M, "jaw_centre_slide": 0.0,
                          "gripper_max_open_m": 0.085, **rig})
        self._flange = np.eye(4)
        self._flange[:3, :3] = DOWN
        self._flange[:3, 3] = [0.30, 0.0, 0.40]
        self._width, self._lock = 0.0, threading.Lock()

    # ── what an arm on a rig provides ─────────────────────────────────────────
    def gripper_frame(self):
        with self._lock:
            return self._flange.copy()

    def move_ee_to(self, target, tol=0.01, max_iters=200, step_size=0.0, max_step_deg=None, orientation=None):
        target = np.asarray(target, float)
        if np.linalg.norm(target[:2]) > 0.85:                 # a UR5's reach
            return False
        for _ in range(max_iters):
            gap = target - self.get_ee_pos()
            if np.linalg.norm(gap) < 1e-6:
                break
            self.guard()
            with self._lock:
                self._flange[:3, 3] += gap * min(1.0, self.STEP_M / np.linalg.norm(gap))
        return bool(np.linalg.norm(target - self.get_ee_pos()) < tol)

    def set_gripper(self, width, n_steps=15, max_step_pct=None):
        self.guard()
        self._held_width_m = None                             # the jaws let go
        self._width = float(np.clip(width, 0.0, self._gripper_max_open_m))

    def get_gripper_width(self):
        return self._width

    def gripper_loaded(self):
        return False

    def render_image(self, camera, width=640, height=480):
        return np.zeros((height, width, 3), np.uint8)

    def list_cameras(self):
        return ["wrist"]

    # ── the rest of RobotEnv ───────────────────────────────────────────────────
    def step(self, n=1):
        pass

    def get_joint_angles(self):
        return {}

    def get_body_pos(self, name):
        return None

    def get_movable_body_names(self):
        return []

    def get_static_body_names(self, exclude_keywords=None):
        return []


def test_an_arm_on_the_rig_keeps_the_contract_and_has_the_rigs_capabilities():
    found = check_backend(_CartesianArm, reachable=[(0.40, 0.10, 0.05), (0.25, -0.20, 0.12)],
                          unreachable=[(1.2, 0.0, 0.1)])
    assert {"WristServo", "OverheadSighting", "Settling", "OverheadCalibration", "HeldWidth",
            "GripperRotation"} <= set(found)


# ── The sentry: on from when the arm's own connect() returns until its stop() ──

ZONE = [(0.0, -0.3), (0.5, -0.3), (0.5, 0.3), (0.0, 0.3)]


@pytest.fixture
def sentry_log(monkeypatch):
    log = []

    class Guard:
        def __init__(self, env, camera, zone=None):
            log.append(f"sentry on {camera}")

        def start(self):
            return self

        def stop(self):
            log.append("sentry off")

    monkeypatch.setitem(sys.modules, "mrcf.motion_guard", types.SimpleNamespace(MotionGuard=Guard))
    return log


def test_an_arm_is_watched_from_when_its_own_connect_returns_until_its_stop(sentry_log):
    class Arm(_CartesianArm):
        def connect(self):
            super().connect()                 # RigEnv's: the sentry still waits for this one to finish
            sentry_log.append("arm open")

        def stop(self):
            sentry_log.append("arm closed")
            super().stop()

    arm = Arm(overhead_camera_name="overhead", motion_guard_zone=ZONE)
    arm.connect()
    arm.stop()
    assert sentry_log == ["arm open", "sentry on overhead", "sentry off", "arm closed"]


def test_an_arm_with_no_connect_of_its_own_is_watched_too(sentry_log):
    arm = _CartesianArm(overhead_camera_name="overhead", motion_guard_zone=ZONE)
    arm.connect()
    arm.stop()
    assert sentry_log == ["sentry on overhead", "sentry off"]


def test_an_arm_the_sentry_cannot_watch_is_closed_again(sentry_log):
    class Arm(_CartesianArm):
        def stop(self):
            sentry_log.append("arm closed")

    with pytest.raises(RuntimeError, match="not moving unwatched"):
        Arm(motion_guard_zone=ZONE).connect()             # no overhead camera to watch from
    assert sentry_log == ["arm closed"]


def test_the_contract_catches_an_arm_whose_connect_goes_round_the_sentry():
    class Arm(_CartesianArm):
        pass

    Arm.connect = lambda self: None                       # put there after the class was made: not wrapped
    with pytest.raises(AssertionError, match="did not start the sentry"):
        check_lifecycle(Arm)


def test_the_grasp_point_is_the_flange_plus_the_jaws_and_a_symmetric_one_does_not_slide():
    arm = _CartesianArm()
    assert np.allclose(arm.get_ee_pos(), [0.30, 0.0, 0.40 - 0.15])
    arm.set_gripper(0.06)
    assert np.allclose(arm.get_ee_pos(), [0.30, 0.0, 0.25])          # both jaws move: no slide
    one_jaw = _CartesianArm(jaw_centre_slide=0.5, jaw_slide_axis=(0.0, 1.0, 0.0))
    one_jaw.set_gripper(0.06)
    assert np.allclose(one_jaw.get_ee_pos(), [0.30, -0.03, 0.25])    # half the opening, gripper y is world -y
    one_jaw.hold_width(0.02)
    assert np.allclose(one_jaw.get_ee_pos(), [0.30, -0.01, 0.25])    # at the held object's centre


def test_the_wrist_camera_rides_on_the_gripper():
    cam = np.eye(4)
    cam[:3, 3] = [0.0, 0.05, 0.02]
    arm = _CartesianArm(gripper_cam_k=np.diag([500.0, 500.0, 1.0]) + [[0, 0, 320], [0, 0, 240], [0, 0, 0]],
                        gripper_cam_pose=cam)
    before = arm.camera_model("wrist", 640, 480)["T_world_cam"]
    assert arm.move_ee_to([0.40, 0.10, 0.05])
    after = arm.camera_model("wrist", 640, 480)["T_world_cam"]
    assert np.allclose(after[:3, 3] - before[:3, 3], np.array([0.40, 0.10, 0.05]) - [0.30, 0.0, 0.25])
    assert arm.camera_model("side", 640, 480) is None


def test_a_rig_arm_with_the_frameworks_own_cameras_keeps_both_classes_cameras():
    # RealRobotEnv owns capture and any calibration-file camera; RigEnv the
    # overhead camera and the wrist one. Rig first, so its cameras win.
    from mrcf.backends.real_robot_env import CameraCalibration, RealRobotEnv

    class _Combo(_CartesianArm, RealRobotEnv):
        def __init__(self):
            RealRobotEnv.__init__(self, calibrations={"side": CameraCalibration(
                np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]]), np.zeros(5),
                np.diag([1.0, -1.0, -1.0]), [0.3, 0.0, 0.8], (640, 480))})
            _CartesianArm.__init__(self, overhead_camera_name="overhead",
                                   overhead_homography=np.diag([0.001, 0.001, 1.0]),
                                   overhead_cam_model={"size": (640, 480), "K": np.eye(3), "D": np.zeros(5),
                                                       "T_world_cam": np.eye(4)})

    arm = _Combo()
    assert arm.camera_model("overhead", 640, 480)["T_world_cam"].shape == (4, 4)      # the rig's
    assert np.allclose(arm.camera_model("side", 640, 480)["T_world_cam"][:3, 3], [0.3, 0.0, 0.8])
    assert np.allclose(arm.pixel_to_world(100, 200, 0.0, "overhead"), [0.1, 0.2, 0.0])
    assert np.allclose(arm.pixel_to_world(320, 240, 0.0, "side")[:2], [0.3, 0.0])     # RealRobotEnv's


def test_a_rig_arm_without_a_mapping_for_a_camera_says_so():
    with pytest.raises(NotImplementedError, match="no pixel_to_world"):
        _CartesianArm().pixel_to_world(1, 2, 0.0, "wrist")


def test_capabilities_are_what_the_rig_gives_not_what_the_arm_lacks():
    arm = _CartesianArm()
    assert isinstance(arm, capabilities.WristServo) and arm.wrist_servo_target_px is None   # not calibrated
    assert not isinstance(arm, capabilities.OrientationSearch)                              # its own to add


# ── Any cameras and sensors: a rig's CAMERAS and SENSORS ──────────────────────

def _side_model():
    """A fixed camera 0.8 m over (0.3, 0), looking straight down, at 640x480."""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.3, 0.0, 0.8]
    return {"size": (640, 480), "K": [[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], "D": [0.0] * 5, "T_world_cam": T}


def test_another_fixed_camera_has_its_geometry_from_its_entry():
    from mrcf.backends.rig import Camera

    homography = np.diag([0.001, 0.001, 1.0])
    arm = _CartesianArm(cameras={"side": Camera(mount="fixed", model=_side_model(), homography=homography),
                                 "top": Camera(mount="fixed", model=_side_model())})
    m = arm.camera_model("side", 320, 240)
    assert np.allclose(m["K"][0, 0], 250.0) and np.allclose(m["T_world_cam"][:3, 3], [0.3, 0.0, 0.8])
    assert np.allclose(arm.pixel_to_world(100, 200, 0.0, "side"), [0.1, 0.2, 0.0])       # its homography
    pytest.importorskip("cv2")
    # No homography: its pixel's ray to the plane. The centre pixel is straight below the camera.
    assert np.allclose(arm.pixel_to_world(320, 240, 0.0, "top"), [0.3, 0.0, 0.0], atol=1e-9)
    assert arm.camera_model("uncalibrated", 640, 480) is None


def test_another_camera_on_the_gripper_rides_on_it():
    from mrcf.backends.rig import Camera

    pose = np.eye(4)
    pose[:3, 3] = [0.0, -0.04, 0.03]
    arm = _CartesianArm(cameras={"palm": Camera(mount="gripper", K=np.diag([400.0, 400.0, 1.0]) + [[0, 0, 320],
                                                                                                    [0, 0, 240],
                                                                                                    [0, 0, 0]],
                                                  pose_on_gripper=pose)})
    before = arm.camera_model("palm", 640, 480)["T_world_cam"]
    assert arm.move_ee_to([0.40, 0.10, 0.05])
    after = arm.camera_model("palm", 640, 480)["T_world_cam"]
    assert np.allclose(after[:3, 3] - before[:3, 3], np.array([0.40, 0.10, 0.05]) - [0.30, 0.0, 0.25])


def test_the_sentry_watches_through_every_guard_camera_each_its_own_source(monkeypatch):
    started, stopped = [], []

    class Guard:
        def __init__(self, env, camera, zone=None, name="motion"):
            started.append((camera, name))
            self.camera = camera

        def start(self):
            return self

        def stop(self):
            stopped.append(self.camera)

    monkeypatch.setitem(sys.modules, "mrcf.motion_guard", types.SimpleNamespace(MotionGuard=Guard))
    arm = _CartesianArm(overhead_camera_name="overhead", motion_guard_zone=ZONE,
                        motion_guard_cameras=["overhead", "side"])
    arm.connect()
    assert started == [("overhead", "motion"), ("side", "motion:side")]
    arm.stop()
    assert stopped == ["overhead", "side"]


def test_a_rig_with_no_camera_to_watch_through_does_not_connect(sentry_log):
    arm = _CartesianArm(overhead_camera_name="overhead", motion_guard_zone=ZONE, motion_guard_cameras=[])
    with pytest.raises(RuntimeError, match="not moving unwatched"):
        arm.connect()


def test_a_rigs_sensors_are_read_and_a_force_one_is_the_contact_force():
    from mrcf.backends.rig import Sensor

    class Arm(_CartesianArm):
        def wrench(self):
            return np.array([3.0, 0.0, 4.0, 0.1, 0.2, 0.3])

    arm = Arm(sensors={"wrench": Sensor(kind="force", unit="N", shape=(6,), read="wrench"),
                       "pad": Sensor(kind="pressure", read=lambda env: 0.25)})
    assert set(arm.list_sensors()) == {"wrench", "pad"}
    assert arm.read_sensor("pad") == 0.25
    assert arm.get_contact_force() == pytest.approx(5.0)               # the force part only
    assert arm.sensor_of_kind("force") == "wrench" and arm.sensor_of_kind("gripper_contact") is None
    with pytest.raises(KeyError):
        arm.read_sensor("nothing")
    assert _CartesianArm().get_contact_force() == 0.0                    # no force sensor: RobotEnv's


def test_a_force_sensor_trips_the_safety_monitor():
    from mrcf import safety
    from mrcf.backends.rig import Sensor

    reading = {"N": 2.0}
    arm = _CartesianArm(sensors={"f": Sensor(kind="force", read=lambda env: reading["N"])})
    monitor = safety.ExecutionMonitor(arm, "test", thresholds=safety.SafetyThresholds(max_contact_force_n=5.0))
    assert monitor.check() is False
    reading["N"] = 6.0
    assert monitor.check() is True and "6.0" in monitor.halt_reason


def test_the_contract_checks_the_sensors():
    from mrcf.backends.contract import check_sensors
    from mrcf.backends.rig import Sensor

    check_sensors(_CartesianArm(sensors={"f": Sensor(kind="force", shape=(3,), read=lambda env: np.zeros(3))}))
    with pytest.raises(AssertionError, match="shape"):
        check_sensors(_CartesianArm(sensors={"f": Sensor(kind="force", shape=(6,), read=lambda env: np.zeros(3))}))


def test_the_sentry_will_not_watch_through_a_camera_on_the_gripper(sentry_log):
    from mrcf.backends.rig import Camera

    arm = _CartesianArm(overhead_camera_name="overhead", motion_guard_zone=ZONE, motion_guard_cameras=["palm"],
                        cameras={"palm": Camera(mount="gripper")})
    with pytest.raises(RuntimeError, match="on the gripper"):
        arm.connect()


def test_a_pixel_that_does_not_see_the_surface_says_so():
    pytest.importorskip("cv2")
    from mrcf.backends.rig import Camera

    level = _side_model()
    level["T_world_cam"] = np.array([[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.3],
                                     [0.0, 0.0, 0.0, 1.0]])          # looking along +x, level: the horizon mid-frame
    arm = _CartesianArm(cameras={"level": Camera(mount="fixed", model=level)})
    with pytest.raises(ValueError, match="does not see the plane"):
        arm.pixel_to_world(320, 100, 0.0, "level")                   # above the horizon
    assert arm.pixel_to_world(320, 400, 0.0, "level")[0] > 0         # below it: on the table ahead


def test_the_sentry_will_not_watch_through_the_wrist_camera_even_unlisted(sentry_log):
    arm = _CartesianArm(overhead_camera_name="overhead", motion_guard_zone=ZONE, motion_guard_cameras=["wrist"])
    with pytest.raises(RuntimeError, match="on the gripper"):
        arm.connect()


def test_a_guard_that_fails_to_start_leaves_none_of_the_others_running(monkeypatch):
    stopped = []

    class Guard:
        def __init__(self, env, camera, zone=None, name="motion"):
            if camera == "side":
                raise KeyError("size")
            self.camera = camera

        def start(self):
            return self

        def stop(self):
            stopped.append(self.camera)

    monkeypatch.setitem(sys.modules, "mrcf.motion_guard", types.SimpleNamespace(MotionGuard=Guard))
    arm = _CartesianArm(overhead_camera_name="overhead", motion_guard_zone=ZONE, motion_guard_cameras=["overhead", "side"])
    with pytest.raises(KeyError):
        arm.connect()
    assert stopped == ["overhead"] and arm._guard is None
