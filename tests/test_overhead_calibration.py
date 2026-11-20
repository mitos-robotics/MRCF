"""The overhead camera's calibration as live state, and the arm's sentry starting with every connection."""

import threading

import numpy as np
import pytest

from mrcf import overhead_calibration as oc
from mrcf.backends.rig import RigEnv


def _pose(rz_deg=0.0, t=(0.0, 0.0, 0.0)):
    a = np.radians(rz_deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


# A camera 0.53 m over a board whose centre is 0.27 m out from the arm, looking down.
T_ARM_BOARD = _pose(1.5, (0.27, 0.014, 0.019))
DOWN = np.diag([1.0, -1.0, -1.0, 1.0])
T_ARM_CAM = _pose(0.0, (0.35, 0.0, 0.53)) @ DOWN
K = np.array([[858.78, 0.0, 640.0], [0.0, 858.78, 360.0], [0.0, 0.0, 1.0]])
D = np.zeros(5)


def _board_points():
    p = 0.04
    return np.array([[(c - 3) * p, (r - 3) * p, 0.0] for r in range(7) for c in range(7)])


def _seen(T_arm_cam):
    """What one still frame gives: the board's pose in the camera, and its corners' pixels."""
    import cv2
    T_cam_board = oc._inv(T_arm_cam) @ T_ARM_BOARD
    rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0]
    px = cv2.projectPoints(_board_points(), rvec, T_cam_board[:3, 3], K, D)[0].reshape(-1, 2)
    return {"T_cam_target": T_cam_board, "points_px": px, "points_target": _board_points()}


class _Env:
    """The fixed camera's part of an arm: RigEnv's own calibration methods."""
    overhead_camera_pose = RigEnv.overhead_camera_pose
    set_overhead_calibration = RigEnv.set_overhead_calibration

    def __init__(self, T_arm_cam):
        self._overhead_cam_model = {"size": (1280, 720), "K": K, "D": D, "T_world_cam": np.asarray(T_arm_cam)}
        self._overhead_homography = None


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(oc, "PATH", tmp_path / "overhead_calibration.json")
    oc.save({"T_arm_target": T_ARM_BOARD, "T_arm_cam": T_ARM_CAM, "homography": np.eye(3),
             "source": "tag tour", "time": "2026-10-22T10:00:00"})
    return tmp_path


def test_a_camera_still_where_it_was_is_left_alone(saved):
    env = _Env(T_ARM_CAM)
    got = oc.recalibrate_camera(env, _seen(T_ARM_CAM), (1280, 720), (640, 480))
    assert got["checked"] and not got["moved"] and got["shift_mm"] < 0.01
    assert env._overhead_homography is None


def test_a_camera_that_moved_is_re_posed_from_one_frame_and_saved(saved):
    # 2026-10-22: moved overnight; every square then read 25-35 mm off.
    moved = _pose(2.0, (0.02, -0.015, 0.0)) @ T_ARM_CAM
    env, rc_ns, deployment_ns = _Env(T_ARM_CAM), {"OVERHEAD_CAM_MODEL": {"K": K, "D": D}}, \
        {"OVERHEAD_CAM_MODEL": {"K": K, "D": D}}
    got = oc.recalibrate_camera(env, _seen(moved), (1280, 720), (640, 480), namespaces=(rc_ns, deployment_ns))
    assert got["checked"] and got["moved"] and got["shift_mm"] > 15
    assert oc.difference(env._overhead_cam_model["T_world_cam"], moved) == pytest.approx((0, 0), abs=1e-6)
    # The homography now takes the corners' pixels to where they are on the arm.
    px = _seen(moved)["points_px"] * np.array([640 / 1280, 480 / 720])     # in render pixels
    h = (env._overhead_homography @ np.c_[px, np.ones(len(px))].T).T
    arm = (T_ARM_BOARD[:3, :3] @ _board_points().T).T + T_ARM_BOARD[:3, 3]
    assert np.abs(h[:, :2] / h[:, 2:] - arm[:, :2]).max() < 1e-4
    for ns in (rc_ns, deployment_ns):
        assert np.allclose(ns["OVERHEAD_CAM_MODEL"]["T_world_cam"], moved)
        assert np.allclose(ns["OVERHEAD_HOMOGRAPHY"], env._overhead_homography)
    data = oc.load()
    assert data["source"] == "scan" and len(data["history"]) == 1
    assert np.allclose(data["history"][0]["T_arm_cam"], T_ARM_CAM)


def test_without_a_board_pose_on_the_arm_nothing_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(oc, "PATH", tmp_path / "none.json")
    got = oc.recalibrate_camera(_Env(T_ARM_CAM), _seen(T_ARM_CAM), (1280, 720), (640, 480))
    assert not got["checked"] and "tag tour" in got["why"]


def test_robot_config_starts_from_the_saved_calibration(saved):
    ns = {"OVERHEAD_CAM_MODEL": {"size": (1280, 720), "K": K, "D": D, "T_world_cam": np.eye(4)},
          "OVERHEAD_HOMOGRAPHY": None}
    assert oc.apply_to(ns)
    assert np.allclose(ns["OVERHEAD_CAM_MODEL"]["T_world_cam"], T_ARM_CAM) and ns["OVERHEAD_CAM_MODEL"]["K"] is K
    assert np.allclose(ns["OVERHEAD_HOMOGRAPHY"], np.eye(3)) and ns["OVERHEAD_CALIBRATION"]["source"] == "tag tour"


def test_a_deployments_copy_of_the_homography_follows_the_calibration_too(saved):
    # The chess config re-exports OVERHEAD_HOMOGRAPHY but not the camera model;
    # its squares are placed by that copy, so a re-pose has to reach it.
    ns = {"OVERHEAD_HOMOGRAPHY": None}
    assert oc.apply_to(ns)
    assert np.allclose(ns["OVERHEAD_HOMOGRAPHY"], np.eye(3)) and "OVERHEAD_CAM_MODEL" not in ns


def test_the_arm_and_camera_disagreeing_is_measured_with_one_hover():
    class Arm:
        def __init__(self):
            self.ee, self.offset = np.zeros(3), np.array([0.012, 0.004, 0.0])

        def approach_orientation_for(self, targets, tols=None):
            return np.eye(3), 0.0

        def move_ee_to(self, target, tol=0.01, orientation=None):
            self.ee = np.asarray(target, float)
            return True

        def overhead_gripper_offset(self):
            return self.offset

    assert oc.arm_disagreement_mm(Arm(), (0.27, 0.01), 0.12) == pytest.approx(np.hypot(12, 4))


# ── The sentry: started with every connection ──────────────────────────────────

def _bare_env():
    from mrcf.backends.so101.env import SO101Env
    env = SO101Env.__new__(SO101Env)
    env._bus_lock = threading.RLock()
    env._overhead_camera_name = "overhead"
    env._motion_guard_zone = [(0.0, -0.3), (0.5, -0.3), (0.5, 0.3), (0.0, 0.3)]
    env._guard = None
    env._running = False
    return env


def test_the_arm_starts_its_sentry_and_stops_it_with_the_connection(monkeypatch):
    from mrcf import motion_guard
    started = []

    class Guard:
        def __init__(self, env, camera, zone=None):
            started.append((camera, zone))
            self.stopped = False

        def start(self):
            return self

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(motion_guard, "MotionGuard", Guard)
    env = _bare_env()
    env._start_sentry()
    assert started == [("overhead", env._motion_guard_zone)]
    guard = env._guard
    env.stop()
    assert guard.stopped and env._guard is None


def test_an_arm_that_cannot_watch_does_not_move_unwatched():
    env = _bare_env()
    env._overhead_camera_name = None
    with pytest.raises(RuntimeError, match="not moving unwatched"):
        env._start_sentry()
    env._motion_guard_zone = None            # a tool running its own guard
    env._start_sentry()
    assert env._guard is None


# ── The pick says how far apart the camera and the arm were ────────────────────

def test_each_attempt_records_the_cameras_disagreement_before_correcting():
    from fake_arm import Cylinder
    from mrcf.policies.base import Target
    from test_approaches import SPEC, ReachArm, _finds_it, on_table
    from mrcf.policies.parallel_jaw import ParallelJaw

    class Off(ReachArm):
        looks = 0

        def overhead_gripper_offset(self):
            self.looks += 1                                 # 20 mm off at the first look, then corrected
            return np.array([0.02, 0.0, 0.0]) if self.looks == 1 else np.zeros(3)

    pawn = Cylinder((0.22, 0.0), 0.022, 0.045)
    got = ParallelJaw(SPEC).pick(Off([pawn]), Target(on_table(pawn.xy), foot_m=0.022, height_m=0.045),
                                 vision=_finds_it())
    assert got.attempts[0]["overhead_err_mm"] == pytest.approx(20.0)


class _Board:
    """A CalibrationTarget: the board as a camera at `cam` sees it."""
    name = "the board"

    def __init__(self, cam):
        self.cam, self.moved = cam, 0

    def measure(self, frame, K_, D_):
        return _seen(self.cam)

    def camera_moved(self):
        self.moved += 1


class _CheckEnv(_Env):
    """An arm whose kinematics are off by `off` (m) from where the fixed camera sees its gripper."""

    def __init__(self, T_arm_cam, off=(0.0, 0.0, 0.0)):
        super().__init__(T_arm_cam)
        self.off, self.hovers = np.asarray(off, float), []

    def camera_model(self, camera, w, h):
        m = self._overhead_cam_model
        return {"K": m["K"], "D": m["D"], "T_world_cam": m["T_world_cam"]}

    def render_image(self, camera, width, height):
        return np.zeros((height, width, 3), np.uint8)

    def approach_orientation_for(self, targets, tols=None):
        return np.eye(3), 0.0

    def move_ee_to(self, target, tol=0.01, orientation=None):
        self.hovers.append(np.asarray(target, float))
        return True

    def overhead_gripper_offset(self):
        return self.off


def test_a_camera_still_where_it_was_costs_no_movement(saved):
    env, board = _CheckEnv(T_ARM_CAM), _Board(T_ARM_CAM)
    got = oc.check_camera(env, board, "overhead", (1280, 720), (640, 480), hover_z=0.12, say=lambda *_: None)
    assert got == {"moved": False, "hovered": False, "apart_mm": None, "toured": False}
    assert env.hovers == [] and board.moved == 0


def test_a_moved_camera_is_re_posed_and_the_arm_checked_with_one_hover(saved):
    moved = _pose(2.0, (0.02, -0.015, 0.0)) @ T_ARM_CAM
    env, board, tours = _CheckEnv(T_ARM_CAM, off=(0.001, 0.0, 0.0)), _Board(moved), []
    got = oc.check_camera(env, board, "overhead", (1280, 720), (640, 480), hover_z=0.12,
                          tour=tours.append, say=lambda *_: None)
    assert got["moved"] and got["hovered"] and got["apart_mm"] == pytest.approx(1.0)
    assert board.moved == 1 and tours == []                       # they agree: no tour
    assert np.allclose(env.hovers[0][:2], T_ARM_BOARD[:2, 3]) and env.hovers[0][2] == 0.12
    assert oc.difference(env._overhead_cam_model["T_world_cam"], moved) == pytest.approx((0, 0), abs=1e-6)


def test_a_frame_that_puts_the_camera_implausibly_far_changes_nothing_and_the_arm_decides(saved):
    # 2026-10-31: one frame put the camera 360 mm and 38 deg away; re-posed and
    # saved, the arm hovered by it into a joint limit and the height ceiling.
    far = _pose(38.0, (0.25, 0.26, -0.19)) @ T_ARM_CAM
    env, board, tours, said = _CheckEnv(T_ARM_CAM, off=(0.001, 0.0, 0.0)), _Board(far), [], []
    got = oc.check_camera(env, board, "overhead", (1280, 720), (640, 480), hover_z=0.12,
                          tour=tours.append, say=said.append)
    assert not got["moved"] and got["implausible"] and got["hovered"] and tours == []
    assert board.moved == 0 and env._overhead_homography is None
    assert oc.difference(env._overhead_cam_model["T_world_cam"], T_ARM_CAM) == pytest.approx((0, 0), abs=1e-9)
    data = oc.load()
    assert data["source"] == "tag tour" and not data.get("history")     # the saved calibration untouched
    assert any("not re-posing it from one frame" in s for s in said)
    assert len(list(saved.glob("overhead_implausible_*.jpg"))) == 1      # the frame, to see what misled it


def test_a_camera_really_knocked_far_is_left_to_the_hover_and_a_tag_tour(saved):
    # Moved that far for real, the arm and the old calibration disagree: a tour re-measures.
    far = _pose(38.0, (0.25, 0.26, -0.19)) @ T_ARM_CAM
    env, tours = _CheckEnv(T_ARM_CAM, off=(0.09, 0.02, 0.0)), []
    got = oc.check_camera(env, _Board(far), "overhead", (1280, 720), (640, 480), hover_z=0.12,
                          tour=tours.append, say=lambda *_: None)
    assert got["implausible"] and got["toured"] and tours == [env]


def test_an_arm_that_disagrees_after_the_re_pose_gets_a_tag_tour(saved):
    # The board (or the arm's base) moved too: one frame cannot tell, the hover can.
    moved = _pose(2.0, (0.02, -0.015, 0.0)) @ T_ARM_CAM
    env, tours = _CheckEnv(T_ARM_CAM, off=(0.012, 0.0, 0.0)), []
    got = oc.check_camera(env, _Board(moved), "overhead", (1280, 720), (640, 480), hover_z=0.12,
                          tour=tours.append, say=lambda *_: None)
    assert got["toured"] and tours == [env]
