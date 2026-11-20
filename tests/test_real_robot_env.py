"""RealRobotEnv's cameras: the newest frame, never a queued one; back-projection and camera_model agree."""

import collections
import threading
import time

import numpy as np
import pytest

try:
    import cv2  # skip below if absent
except ImportError:
    pytest.skip("needs OpenCV", allow_module_level=True)

from mrcf.perception import plane_geometry as pg
from mrcf.backends.real_robot_env import CameraCalibration, RealRobotEnv


class _Arm(RealRobotEnv):
    def get_ee_pos(self):
        return np.zeros(3)

    def get_joint_angles(self):
        return {}


def overhead_calibration(size=(1280, 720)) -> CameraCalibration:
    """A camera 0.5 m up, looking down at the table, image size `size`."""
    K = np.array([[860.0, 0, size[0] / 2], [0, 860.0, size[1] / 2], [0, 0, 1]])
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])     # camera -> world
    return CameraCalibration(K, np.array([-0.1, 0.0, 0.0, 0.0, 0.0]), R, [0.25, 0.0, 0.5], size)


def test_pixel_to_world_uses_the_size_pixels_come_in():
    env = _Arm(calibrations={"overhead": overhead_calibration()})
    model = env.camera_model("overhead", 640, 480)
    view = pg.PlaneView.from_camera_model(model)
    target = np.array([0.31, -0.07])
    u, v = view.project(np.array([[target[0] * 1000, target[1] * 1000, 0.0]]))[0]
    got = env.pixel_to_world(u, v, 0.0, camera="overhead", render_w=640, render_h=480)
    assert np.linalg.norm(got[:2] - target) < 1e-4


def test_calibration_round_trips_with_its_size(tmp_path):
    cal = overhead_calibration((1280, 720))
    cal.save(tmp_path / "cal.npz")
    loaded = CameraCalibration.load(tmp_path / "cal.npz")
    assert loaded.size == (1280, 720)
    assert np.allclose(loaded.K_for(640, 360), cal.K * [[0.5], [0.5], [1.0]])


def test_uncalibrated_camera_has_no_model():
    assert _Arm().camera_model("overhead", 640, 480) is None


class _Driver:
    """A camera driver as OpenCV's: a frame every 5 ms into a queue it keeps 5 of; read() hands out the oldest."""

    def __init__(self, stops_after: "int | None" = None, black: bool = False):
        self.n, self.queue, self.lock = 0, collections.deque(maxlen=5), threading.Lock()
        self.t0, self.stops_after, self.black = time.monotonic(), stops_after, black

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        while True:
            with self.lock:
                due = int((time.monotonic() - self.t0) / 0.005)
                if self.stops_after is not None:
                    due = min(due, self.stops_after)
                while self.n < due:
                    self.n += 1
                    self.queue.append(self.n)
                if self.queue:
                    k = self.queue.popleft()
                    frame = np.zeros((48, 64, 3), np.uint8)
                    if not self.black:
                        frame[1:] = 120                              # a picture: not black
                    frame[0, 0, 0] = frame[0, 0, 2] = k // 256      # the frame's number, in its first pixel
                    frame[0, 0, 1] = k % 256
                    return True, frame
                if self.stops_after is not None and self.n >= self.stops_after:
                    return False, None
            time.sleep(0.001)

    def release(self):
        pass


def _camera(monkeypatch, driver):
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: driver)
    return _Arm(cameras={"wrist": 0})


def test_a_frame_is_the_newest_the_camera_took_not_the_oldest_it_queued(monkeypatch):
    # Read on demand, the driver's oldest queued frame comes out — five back,
    # 25 ms old here; at 30 fps with a deeper queue, before the last move, and
    # a wrist servo acts on where the piece was then.
    driver = _Driver()
    env = _camera(monkeypatch, driver)
    try:
        time.sleep(0.1)
        frame = env.render_image("wrist", 64, 48)
        number = int(frame[0, 0, 1]) + 256 * int(frame[0, 0, 0])
        assert driver.n - number <= 2 and env.frame_age_s("wrist") < 0.05
    finally:
        env.stop()


def test_a_camera_that_stops_sending_is_not_read_as_now(monkeypatch):
    env = _camera(monkeypatch, _Driver(stops_after=3))
    try:
        time.sleep(RealRobotEnv.MAX_FRAME_AGE_S + 0.2)
        with pytest.raises(RuntimeError, match="has sent nothing"):
            env.render_image("wrist", 64, 48)
    finally:
        env.stop()


def test_a_camera_that_shows_only_black_is_refused(monkeypatch):
    # 2026-11-12, the UR5: its webcam off USB, number 0 opened the MacBook's own camera, lid shut.
    monkeypatch.setattr(RealRobotEnv, "FIRST_LIGHT_S", 0.3)
    with pytest.raises(RuntimeError, match="black picture"):
        _camera(monkeypatch, _Driver(black=True))


def test_a_camera_whose_first_frames_are_dark_is_taken(monkeypatch):
    class _WarmingUp(_Driver):
        def read(self):
            ok, frame = super().read()
            if ok and self.n < 20:
                frame[1:] = 0                          # the exposure still settling
            return ok, frame

    env = _camera(monkeypatch, _WarmingUp())
    try:
        assert env.render_image("wrist", 64, 48).mean() > 50
    finally:
        env.stop()


def test_each_camera_captures_at_its_own_size(monkeypatch):
    # An overhead camera at 1280x720 beside a wrist one at 640x480: asked for
    # one size between them, one is stretched and its lens fit comes out wrong.
    asked = {}

    class _Sized(_Driver):
        def __init__(self, src):
            super().__init__()
            self.src = src

        def set(self, prop, value):
            asked.setdefault(self.src, {})[prop] = value
            return True

    monkeypatch.setattr(cv2, "VideoCapture", _Sized)
    env = _Arm(cameras={"wrist": 0, "overhead": 1}, capture_sizes={"overhead": (1280, 720)})
    try:
        sizes = {src: (got[cv2.CAP_PROP_FRAME_WIDTH], got[cv2.CAP_PROP_FRAME_HEIGHT]) for src, got in asked.items()}
        assert sizes == {0: (640, 480), 1: (1280, 720)}
    finally:
        env.stop()
