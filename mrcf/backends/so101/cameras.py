"""
mrcf/backends/so101/cameras.py — the SO-101 rig's cameras: finding one by name
(macOS renumbers them), listing and reading them, and pixel_to_world.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import subprocess
import threading
import time

import numpy as np

from mrcf.backends.so101.follower import Follower

# A camera whose reader thread has died is opened again: lerobot's reader gives
# up for good after eleven failed reads in a row, which a hiccup of a moment
# uses up. One that would not open is tried again no sooner than this.
REOPEN_EVERY_S = 2.0


def _list_avfoundation_video_devices() -> dict[str, int]:
    """{device name: current index} for AVFoundation video devices, via ffmpeg (mrcf/devices.py). macOS only."""
    from mrcf import devices

    try:
        listed = devices._avfoundation()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not run ffmpeg to list cameras ({exc})") from exc
    return {name: index for index, name in listed.items()}


def resolve_camera_index(name: str, fallback: "int | str | None" = None) -> "int | str":
    """
    Resolve a camera's CURRENT numeric index by (case-insensitive substring) device
    name via ffmpeg's AVFoundation device listing.

    macOS AVFoundation indices aren't stable across process launches — they can
    swap even with nothing unplugged, e.g. when a Continuity Camera device (an
    iPhone) appears/disappears from the list and shifts every index after it. A device's own name (from its USB product string) doesn't
    change with enumeration order, so it's a far more durable identifier.

    Falls back to `fallback` (printing a warning) if ffmpeg isn't available, times
    out, or no/multiple devices match — never silently returns a wrong index.
    """
    try:
        devices = _list_avfoundation_video_devices()
    except Exception as exc:
        if fallback is None:
            raise
        print(f"  [camera] Could not resolve '{name}' by device name ({exc}) — using fallback index {fallback}.")
        return fallback

    matches = [(dev_name, idx) for dev_name, idx in devices.items() if name.lower() in dev_name.lower()]
    if len(matches) == 1:
        return matches[0][1]

    if len(matches) > 1:
        msg = f"multiple AVFoundation devices match '{name}': {matches}"
    else:
        msg = f"no AVFoundation device matches '{name}'. Available: {list(devices)}"
    if fallback is None:
        raise RuntimeError(msg)
    print(f"  [camera] {msg} — using fallback index {fallback}.")
    return fallback


def _reader_died(cam) -> bool:
    """A lerobot camera's reader thread gone or stopped; False for a camera with none."""
    if not hasattr(cam, "thread"):
        return False
    return cam.thread is None or not cam.thread.is_alive()


class CamerasMixin(Follower):
    """SO101Env's cameras (see env.py)."""

    # ── RobotEnv: camera ───────────────────────────────────────────────────────

    def list_cameras(self) -> list[str]:
        names = [] if self._camera_name is None else [self._camera_name]
        if self._overhead_camera_name is not None:
            names.append(self._overhead_camera_name)
        opened = list(getattr(self.__dict__.get("_robot"), "cameras", None) or {})
        opened += list(self.__dict__.get("_closed_cameras", {}))
        return names + [n for n in dict.fromkeys(opened) if n not in names]

    def _camera(self, name: "str | None"):
        """The camera called name: one of the follower's, or one closed for a reopen (_reopen); None if neither."""
        cam = self._robot.cameras.get(name)
        return cam if cam is not None else self.__dict__.get("_closed_cameras", {}).get(name)

    def frame_age_s(self, camera: str) -> float:
        """
        How old the camera's newest frame is, seconds; inf if it has none. The
        wrist camera can stop sending for seconds at a time and then come back
        by itself: render_image then raises after 0.5 s, and anything recording
        it wants to know before it does.
        """
        cam = self._camera(camera or self._camera_name)
        stamp = getattr(cam, "latest_timestamp", None)
        return float("inf") if stamp is None else time.perf_counter() - stamp

    def render_image(self, camera: str, width: int = 640, height: int = 480) -> np.ndarray:
        """
        The camera's newest frame, from its own reader thread — not through
        get_observation(), which also reads the motor bus: this is called from
        perception threads while a move streams commands on it. A copy, since
        callers on different threads may draw on what they get.
        """
        name = camera or self._camera_name
        cam = self._camera(name)
        if cam is None:
            raise RuntimeError(f"Camera '{camera}' not available. Known: {self.list_cameras()}")
        if name is not None and self.__dict__.get("_running", False) and _reader_died(cam):
            self._reopen(name, cam)
        frame = cam.read_latest().copy()
        if frame.shape[1] != width or frame.shape[0] != height:
            import cv2
            frame = cv2.resize(frame, (width, height))
        return frame

    def _reopen(self, name: str, cam) -> None:
        """
        Open a camera whose reader has died again. One thread reopens it; any
        other asking meanwhile waits, then finds it reading. While a camera the
        sentry watches through is closed, the sentry is blind: the arm pauses
        (a pause, not a stop — the move under way carries on) until it is back.
        One that would not open is left for read_latest to raise.

        While it is closed it is out of the follower's cameras (kept in
        _closed_cameras instead): lerobot's follower counts itself connected
        only while every camera it holds is, and refuses motor commands
        otherwise (send_action, disconnect). 2026-11-14 the wrist camera
        dropped off USB during a move, this closed it to reopen it, and the
        next joint command raised "SOFollower is not connected" and ended the
        controller, the arm stopped where it was. A camera that will not open
        stays out, so the arm runs on without it, and is tried again.
        """
        locks = self.__dict__.setdefault("_reopen_locks", {})
        with locks.setdefault(name, threading.Lock()):
            if not _reader_died(cam):
                return
            tried = self.__dict__.setdefault("_reopened_at", {})
            if time.monotonic() - tried.get(name, -np.inf) < REOPEN_EVERY_S:
                return
            tried[name] = time.monotonic()
            source = f"camera:{name}"
            if name in self._sentry_cameras():
                self.interlock.pause_only(source)
                self.interlock.trip(f"camera '{name}' stopped sending; reopening it", source)
            print(f"  [camera] '{name}' stopped sending — reopening it.")
            closed = self.__dict__.setdefault("_closed_cameras", {})
            cameras = self._robot.cameras
            # Under the bus lock: lerobot reads the dict on every command (is_connected).
            with self._bus_lock:
                if cameras.get(name) is cam:
                    del cameras[name]
                closed[name] = cam
            try:
                try:
                    cam.disconnect()
                except Exception:
                    pass              # half-closed already: connect() is what matters
                from mrcf import devices

                # Unplugged, its index may now be this computer's own camera: never that one.
                devices.check_sources({n: getattr(getattr(c, "config", None), "index_or_path", None)
                                       for n, c in {**self._robot.cameras, **closed}.items()})
                cam.connect()
                with self._bus_lock:
                    cameras[name] = closed.pop(name)
                print(f"  [camera] '{name}' is sending again.")
            except Exception as exc:
                print(f"  [camera] '{name}' would not open again: {exc}")
            finally:
                self.interlock.clear(source)

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
        Overhead camera: real calibrated homography (see calibrate_overhead_camera.py)
        — valid from ANY arm pose, since this camera is stationary.

        Gripper camera (or overhead before it's calibrated... see below):
        pinhole back-projection, valid ONLY while the arm is at the fixed
        "observe pose" (see deployments/chess/skills.py:goto_observe_pose
        and robot_config.py's OBSERVE_CENTER_XY / OBSERVE_HEIGHT_M). The camera is
        gripper-mounted, so there is no single fixed camera-to-world transform
        like a stationary overhead camera would have — this approximates the
        observe pose as "looking straight down from OBSERVE_HEIGHT_M above
        z_world", centred on OBSERVE_CENTER_XY.

        The sign of the u/v -> x/y mapping below is a placeholder — it depends
        on how the camera is physically rotated relative to the arm's base
        frame, which is only knowable by testing on the real rig. Verify by
        pointing the camera at a known object (e.g. one placed toward +x from
        OBSERVE_CENTER_XY) and checking that the back-projected position comes
        out on the correct side; flip the signs or swap u/v below if not.
        """
        if camera == self._overhead_camera_name or (camera != self._camera_name and camera in self._cameras):
            # The overhead camera's homography; another camera's own geometry (RigEnv, its Camera entry).
            return super().pixel_to_world(u, v, z_world, camera, render_w, render_h)

        f = max(render_w, render_h) * 0.9
        cx, cy = self._observe_center_xy
        x = cx - (v - render_h / 2.0) / f * self._observe_height_m
        y = cy - (u - render_w / 2.0) / f * self._observe_height_m
        return np.array([x, y, z_world])
