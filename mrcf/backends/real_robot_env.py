"""
mrcf/backends/real_robot_env.py — Base class for real-hardware RobotEnv backends.

WHEN TO USE THIS
----------------
Use it when you drive the arm with your own SDK and want the framework to own
the cameras: OpenCV capture, calibration files, pixel->world back-projection,
serial port discovery, clean shutdown.

Do NOT use it when your robot library already owns the cameras and the motor
bus. mrcf/backends/so101/env.py is that case — LeRobot opens the cameras and runs
their reader threads, so SO101Env is a backends.rig.RigEnv but not one of these,
and reimplements render_image rather than fighting this class for control of
the camera handles. It is a deliberate fork, not an oversight; see
DEPLOYMENT_SPEC.md ("Which base class").

An arm with a gripper and cameras on its rig combines the two —
`class MyArmEnv(RigEnv, RealRobotEnv)`: this class captures, RigEnv does the
geometry (mrcf/backends/_template/env.py is that shape). The UR5 runs on it
(mrcf/backends/ur5/env.py, since 2026-11-12): read it, and so101/env.py, for
the contracts that matter either way (get_ee_pos returns the grasp point;
move_ee_to returns whether it arrived; set_gripper returns once the jaws are
there, not once the command is sent).

Provides the cross-cutting plumbing that every real-robot backend needs:
  - Multiple named OpenCV cameras (overhead, wrist, side, …)
  - Named non-image sensors (force, pressure, IMU, depth, …)
  - Camera calibration loading / pixel-to-world back-projection
  - Graceful shutdown on KeyboardInterrupt / window close
  - Cross-platform serial port discovery helper

Subclass this and implement the arm-specific methods:
  get_ee_pos, get_joint_angles, move_ee_to, set_gripper,
  get_gripper_width, go_home, get_movable_body_names, get_static_body_names

and the stop (robot_env.py, "Stopping"): call self.guard() before every command
you send your SDK — step() already does, so a loop of command + step() is
covered — and override hold() to stop the arm where it is.

Each camera is read on its own thread, which keeps only its newest frame, so
perception threads can take frames while the arm moves, and no frame is one
the driver queued before the last move.

Usage — single overhead camera::

    from mrcf.backends.real_robot_env import RealRobotEnv

    class MyArmEnv(RealRobotEnv):
        def __init__(self):
            super().__init__(
                cameras={"overhead": 0},
                calibrations={"overhead": "overhead_calib.npz"},
            )
            self._sdk = MyArmSDK.connect("COM3")

        def get_ee_pos(self):     ...
        def move_ee_to(self, ...) ...

Usage — multiple cameras and pressure sensors::

    class MyArmEnv(RealRobotEnv):
        def __init__(self):
            super().__init__(
                cameras={
                    "overhead": 0,
                    "wrist":    1,
                    "side":     "rtsp://192.168.1.10/stream",
                },
                calibrations={
                    "overhead": "calib_overhead.npz",
                    "wrist":    "calib_wrist.npz",
                },
                sensors={
                    "left_finger_force":  ("float (N)",  self._read_left_force),
                    "right_finger_force": ("float (N)",  self._read_right_force),
                    "wrist_torque":       ("array(3,) (Nm)", self._read_torque),
                },
            )

        def _read_left_force(self):
            return self._sdk.get_force(finger="left")

Camera calibration
------------------
Run cv2.calibrateCamera() with a checkerboard to get K and dist, and locate
the camera in the world (solvePnP on points whose world positions you know).
Save with::

    np.savez("calib.npz", K=K, dist=dist, R=R, t=t, size=(width, height))

  K    : (3, 3) intrinsic matrix, for images of `size`
  dist : (1, 5) distortion coefficients
  R    : (3, 3) rotation matrix (camera → world)
  t    : (3,)   translation vector (camera origin in world frame), metres
  size : (width, height) the calibration images had; optional, but without it
         pixels are assumed to come from images of that same size

A calibrated camera also gets camera_model(), so plane_geometry and
surface_scan can measure objects from it.

Cross-platform serial port discovery
--------------------------------------
    from mrcf.backends.real_robot_env import list_serial_ports
    print(list_serial_ports())
    # Linux:   ['/dev/ttyUSB0', '/dev/ttyACM0']
    # macOS:   ['/dev/cu.usbserial-0001']
    # Windows: ['COM3', 'COM7']
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from mrcf.robot_env import RobotEnv


# ── Serial port discovery (cross-platform) ────────────────────────────────────

def list_serial_ports() -> list[str]:
    """
    Return available serial port names on Mac, Linux, and Windows.

    Requires pyserial (``pip install pyserial``) or falls back to a
    platform glob if pyserial is not installed.
    """
    try:
        import serial.tools.list_ports
        return [p.device for p in serial.tools.list_ports.comports()]
    except ImportError:
        pass

    import sys, glob
    if sys.platform.startswith("win"):
        return [f"COM{i}" for i in range(1, 257)]
    elif sys.platform == "darwin":
        return glob.glob("/dev/cu.*") + glob.glob("/dev/tty.*")
    else:
        return glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")


# ── Calibration data container ────────────────────────────────────────────────

class CameraCalibration:
    """
    Intrinsics + extrinsics for one camera.

    Attributes
    ----------
    K    : (3, 3) intrinsic matrix
    dist : distortion coefficients (any OpenCV-compatible shape)
    R    : (3, 3) rotation — maps camera frame → world frame
    t    : (3,) camera origin in world frame
    size : (width, height) of the images K was calibrated on, or None
    """

    def __init__(
        self,
        K: np.ndarray,
        dist: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        size: "tuple[int, int] | None" = None,
    ):
        self.K    = np.asarray(K,    dtype=float)
        self.dist = np.asarray(dist, dtype=float)
        self.R    = np.asarray(R,    dtype=float)
        self.t    = np.asarray(t,    dtype=float).ravel()
        self.size = None if size is None else (int(size[0]), int(size[1]))

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        """Load from a .npz file with keys K, dist, R, t (and optionally size)."""
        data = np.load(str(path))
        size = tuple(data["size"]) if "size" in data.files else None
        return cls(data["K"], data["dist"], data["R"], data["t"], size)

    def save(self, path: str | Path) -> None:
        """Save to a .npz file."""
        extra: dict = {} if self.size is None else {"size": np.asarray(self.size)}
        np.savez(str(path), K=self.K, dist=self.dist, R=self.R, t=self.t, **extra)

    def K_for(self, width: int, height: int) -> np.ndarray:
        """K for images resized to width x height (as render_image resizes them)."""
        if self.size is None or (width, height) == self.size:
            return self.K
        k = self.K.copy()
        k[0] *= width / self.size[0]
        k[1] *= height / self.size[1]
        return k


# ── RealRobotEnv ──────────────────────────────────────────────────────────────

class _LatestFrame:
    """
    One camera read on a thread of its own, keeping only its newest frame.

    Reading the driver on demand hands back the OLDEST frame it has queued
    (OpenCV keeps several), taken before the arm's last move — and a wrist
    servo or a motion guard acts on what it is given as if it were now. The
    SO-101 has the same from lerobot's camera reader threads (read_latest).
    """

    def __init__(self, cap, name: str):
        self._cap, self._name = cap, name
        self._lock = threading.Lock()
        self._frame: "np.ndarray | None"
        self._t: "float | None"
        self._frame, self._t = None, None
        self._first = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{name}")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._t = frame, time.monotonic()
            self._first.set()

    def latest(self, wait_s: float) -> "tuple[np.ndarray, float]":
        """The newest frame (BGR, the driver's own) and how old it is, seconds."""
        if not self._first.wait(wait_s):
            raise RuntimeError(f"Camera '{self._name}' has sent no frame — check its connection.")
        with self._lock:
            assert self._frame is not None and self._t is not None     # set before _first
            return self._frame.copy(), time.monotonic() - self._t

    def age_s(self) -> float:
        with self._lock:
            return float("inf") if self._t is None else time.monotonic() - self._t

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(1.0)
        self._cap.release()


class RealRobotEnv(RobotEnv):
    """
    Abstract base for real-hardware backends.

    Handles multi-camera capture, calibration, and generic sensor access.
    Subclasses provide arm control.

    Parameters
    ----------
    cameras : dict[str, int | str] | None
        Named camera sources.  Keys are the camera names used everywhere in the
        framework (e.g. "overhead", "wrist").  Values are OpenCV VideoCapture
        indices (0, 1, …), device paths ("/dev/video0"), or stream URLs.
        Pass None (or omit) to disable camera capture.

        Single-camera shorthand: pass camera_index=0 instead.

    calibrations : dict[str, str | Path | CameraCalibration] | None
        Per-camera calibration.  Keys must match entries in `cameras`.
        Values are either a path to a .npz file or a CameraCalibration object.
        Cameras without a calibration fall back to the simple pinhole estimator.

    sensors : dict[str, tuple[str, Callable]] | None
        Named non-image sensors.  Each entry is::

            "sensor_name": ("type description", callable)

        The callable takes no arguments and returns float | np.ndarray.
        Example::

            sensors={
                "left_finger_force":  ("float (N)",     lambda: self._sdk.force_left()),
                "wrist_torque":       ("array(3,) (Nm)", lambda: self._sdk.torque()),
            }

    capture_width / capture_height : int
        Resolution to request from every camera driver.
    control_hz : float
        Target control loop rate used by step() to sleep between cycles.
    work_plane_z : float
        Height of the work surface in the world frame (camera_model's plane).

    camera_index : int | str | None
        Convenience parameter for a single camera.  Equivalent to
        cameras={"overhead": camera_index}.  Ignored when `cameras` is given.
    """

    def __init__(
        self,
        cameras: "dict[str, int | str] | None"                  = None,
        calibrations: "dict[str, str | Path | CameraCalibration] | None" = None,
        sensors: "dict[str, tuple[str, Callable]] | None"        = None,
        capture_width:  int   = 640,
        capture_height: int   = 480,
        # A camera's own capture size, where it differs: {name: (width, height)}
        capture_sizes: "dict[str, tuple[int, int]] | None" = None,
        control_hz:     float = 20.0,
        work_plane_z:   float = 0.0,
        # Single-camera convenience parameter (ignored when cameras= is given)
        camera_index: "int | str | None" = None,
    ):
        self._capture_width  = capture_width
        self._capture_height = capture_height
        self._capture_sizes  = dict(capture_sizes or {})
        self._control_hz     = control_hz
        self._work_plane_z   = work_plane_z
        self._running        = True

        # ── Camera registry ───────────────────────────────────────────────────
        # _caps:   {name: _LatestFrame} — each camera read on its own thread
        # _calibs: {name: CameraCalibration | None}
        self._caps:   dict[str, _LatestFrame] = {}
        self._calibs: dict[str, CameraCalibration | None] = {}

        # Resolve cameras argument
        cam_map: dict[str, int | str] = {}
        if cameras is not None:
            cam_map = dict(cameras)
        elif camera_index is not None:
            cam_map = {"overhead": camera_index}

        from mrcf import devices

        devices.check_sources(dict(cam_map))           # all of them counted: never this computer's own camera
        for name, src in cam_map.items():
            self._open_camera(name, src)

        # Resolve calibrations
        if calibrations:
            for name, cal in calibrations.items():
                if isinstance(cal, CameraCalibration):
                    self._calibs[name] = cal
                else:
                    try:
                        self._calibs[name] = CameraCalibration.load(cal)
                        print(f"  [Camera:{name}] Calibration loaded from {cal}")
                    except Exception as exc:
                        print(f"  [Camera:{name}] Calibration load failed: {exc}")
                        self._calibs[name] = None

        # ── Sensor registry ───────────────────────────────────────────────────
        # _sensor_meta: {name: type_str}
        # _sensor_fns:  {name: Callable -> float | np.ndarray}
        self._sensor_meta: dict[str, str]      = {}
        self._sensor_fns:  dict[str, Callable] = {}
        if sensors:
            for name, (type_str, fn) in sensors.items():
                self._sensor_meta[name] = type_str
                self._sensor_fns[name]  = fn

    # ── Camera setup ──────────────────────────────────────────────────────────

    def _open_camera(self, name: str, src: "int | str") -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for real-robot camera capture.\n"
                "  pip install opencv-python"
            ) from exc
        from mrcf import devices

        devices.check_sources({name: src})            # never this computer's own (add_camera's too)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera '{name}' at source {src!r}.\n"
                "Check that the camera is connected and not in use by another process."
            )
        w, h = self._capture_sizes.get(name, (self._capture_width, self._capture_height))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        old = self._caps.pop(name, None)
        if old is not None:
            old.stop()
        self._caps[name] = _LatestFrame(cap, name)
        self._first_light(name, src)
        print(f"  [Camera:{name}] Opened at source {src!r}")

    # How long a camera has to give a picture (lighting.no_picture: not all
    # but black) — a webcam's first frames can be dark.
    FIRST_LIGHT_S = 3.0

    def _first_light(self, name: str, src) -> None:
        """
        The camera gives a picture, not only black, within FIRST_LIGHT_S; else
        it is closed and refused. A camera that comes off USB makes macOS
        number the rest again, and its number may then open another (a
        laptop's own, its lid shut): 2026-11-12, on the UR5, the overhead
        webcam had dropped, number 0 opened the MacBook's camera, a scan saw an
        empty table, and the motion guard watched a black picture.
        """
        from mrcf.perception.lighting import no_picture

        cam = self._caps[name]
        deadline, mean = time.monotonic() + self.FIRST_LIGHT_S, None
        while time.monotonic() < deadline:
            try:
                frame, _ = cam.latest(0.2)
            except RuntimeError:
                continue                              # nothing yet: wait on
            mean = float(frame.mean())
            if not no_picture(frame):
                return
            time.sleep(0.05)
        cam.stop()
        del self._caps[name]
        seen = "sent no frame" if mean is None else f"gives only a black picture (mean {mean:.1f})"
        raise RuntimeError(
            f"Camera '{name}' at source {src!r} {seen} in {self.FIRST_LIGHT_S:.0f} s. It may have come "
            "unplugged, and its number now opens another camera (macOS numbers them again when one comes "
            "or goes; a laptop's own is black with its lid shut). Check its cable, then which number it is "
            "(tools/check_camera.py).")

    def add_camera(
        self,
        name: str,
        src: "int | str",
        calibration: "str | Path | CameraCalibration | None" = None,
    ) -> None:
        """Add or replace a named camera at runtime."""
        self._open_camera(name, src)
        if calibration is not None:
            self._calibs[name] = (
                calibration if isinstance(calibration, CameraCalibration)
                else CameraCalibration.load(calibration)
            )

    def add_sensor(
        self,
        name: str,
        type_description: str,
        fn: Callable,
    ) -> None:
        """Register or replace a named non-image sensor at runtime."""
        self._sensor_meta[name] = type_description
        self._sensor_fns[name]  = fn

    # ── RobotEnv: physics ─────────────────────────────────────────────────────

    def step(self, n: int = 1) -> None:
        """Sleep for n control cycles (real time, not simulation steps); guarded."""
        self.guard()
        time.sleep(n / self._control_hz)

    # ── RobotEnv: camera enumeration ──────────────────────────────────────────

    def list_cameras(self) -> list[str]:
        """Return names of all registered cameras."""
        return list(self._caps)

    # ── RobotEnv: camera rendering ────────────────────────────────────────────

    # A newest frame older than this means the camera has stopped sending:
    # render_image raises rather than hand it out as now (lerobot's
    # read_latest does the same for the SO-101's cameras).
    MAX_FRAME_AGE_S = 0.5

    def frame_age_s(self, camera: str) -> float:
        """How old the camera's newest frame is, seconds; inf if it has none."""
        reader = self._caps.get(camera)
        return float("inf") if reader is None else reader.age_s()

    def render_image(self, camera: str, width: int = 640, height: int = 480) -> np.ndarray:
        """
        The named camera's newest frame (from its reader thread, so safe from
        any thread, and never one queued from before the last move), resized
        to (height, width, 3) RGB.

        Raises RuntimeError if the camera name is not registered, or if it has
        sent nothing for MAX_FRAME_AGE_S.
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required: pip install opencv-python") from exc

        cap = self._caps.get(camera)
        if cap is None:
            available = list(self._caps)
            raise RuntimeError(
                f"Camera '{camera}' not registered.  "
                f"Available cameras: {available} — a camera is opened only when the arm's "
                f"robot_config gives it an OpenCV index or URL (CAMERA_INDEX, OVERHEAD_CAMERA_INDEX)."
            )

        frame, age = cap.latest(wait_s=max(1.0, 2 * self.MAX_FRAME_AGE_S))
        if age > self.MAX_FRAME_AGE_S:
            raise RuntimeError(f"Camera '{camera}' has sent nothing for {age:.1f} s — check its connection.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[:2] != (height, width):
            frame_rgb = cv2.resize(frame_rgb, (width, height))
        return frame_rgb

    # ── RobotEnv: pixel → world back-projection ───────────────────────────────

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
        Back-project pixel (u, v) → world point on the plane z = z_world.

        Uses the calibration for `camera` if available, else falls back to
        a simple pinhole estimate centred above the workspace.
        """
        calib = self._calibs.get(camera)
        if calib is not None:
            return self._back_project_calibrated(u, v, z_world, calib, render_w, render_h)
        return self._back_project_pinhole(u, v, z_world, render_w, render_h)

    def camera_model(self, camera: str, width: int, height: int) -> "dict | None":
        """RobotEnv.camera_model, from the camera's calibration file; None without one."""
        calib = self._calibs.get(camera)
        if calib is None:
            return None
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = calib.R, calib.t
        return {"K": calib.K_for(width, height), "D": calib.dist.ravel(), "T_world_cam": T,
                "plane_z_m": self._work_plane_z, "scale": 1.0}

    def _back_project_calibrated(
        self, u: float, v: float, z_world: float, calib: CameraCalibration,
        render_w: int, render_h: int,
    ) -> np.ndarray:
        """Full calibrated back-projection with distortion removal."""
        import cv2
        # (u, v) is in the render_w x render_h image; K is for the calibration's.
        K = calib.K_for(render_w, render_h)
        pts = np.array([[[u, v]]], dtype=np.float32)
        pts_ud = cv2.undistortPoints(pts, K, calib.dist, P=K)
        u2, v2 = float(pts_ud[0, 0, 0]), float(pts_ud[0, 0, 1])

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ray_cam   = np.array([(u2 - cx) / fx, (v2 - cy) / fy, 1.0])
        ray_world = calib.R @ ray_cam

        if abs(ray_world[2]) < 1e-8:
            raise ValueError("Camera ray nearly parallel to ground plane.")
        t_param = (z_world - calib.t[2]) / ray_world[2]
        return calib.t + t_param * ray_world

    def _back_project_pinhole(
        self, u: float, v: float, z_world: float, render_w: int, render_h: int
    ) -> np.ndarray:
        """
        Simple pinhole fallback — no calibration required.

        Assumes the camera is directly overhead, looking straight down.
        Accuracy: ±3–5 cm at typical heights.  Load a calibration file for
        production pick-and-place.
        """
        cam_height = 0.60
        f = max(render_w, render_h) * 0.9
        scale = cam_height - z_world
        x_world =  (u - render_w / 2.0) / f * scale
        y_world = -(v - render_h / 2.0) / f * scale
        return np.array([x_world, y_world, z_world])

    # ── RobotEnv: generic sensor access ───────────────────────────────────────

    def list_sensors(self) -> dict[str, str]:
        """Return {sensor_name: type_description} for all registered sensors."""
        return dict(self._sensor_meta)

    def read_sensor(self, name: str) -> "float | np.ndarray":
        """Read a named sensor.  Raises KeyError for unknown names."""
        fn = self._sensor_fns.get(name)
        if fn is None:
            raise KeyError(
                f"Sensor '{name}' not registered.  "
                f"Available sensors: {list(self._sensor_meta)}"
            )
        return fn()

    # ── RobotEnv: scene introspection stubs ───────────────────────────────────

    def get_body_pos(self, name: str) -> np.ndarray | None:
        """Returns None — real hardware has no ground-truth object poses."""
        return None

    def get_movable_body_names(self) -> list[str]:
        """
        Override to return object names your detector can see.
        Called by discover_scene() to seed the VLM prompt.
        """
        return []

    def get_static_body_names(self, exclude_keywords: "set[str] | None" = None) -> list[str]:
        """
        Override to return placement target names.
        Called by discover_scene() to seed the VLM prompt.
        """
        return []

    # ── Viewer ────────────────────────────────────────────────────────────────

    def is_viewer_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """Graceful shutdown: release all cameras and signal control loop to exit."""
        self._running = False
        for reader in list(self._caps.values()):
            reader.stop()
        self._caps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
