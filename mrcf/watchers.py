"""
watchers.py — perception that runs alongside the arm.

A Watcher calls fn(frame, model) with a camera's newest frame, on its own
thread, up to `hz` times a second, while everything else carries on — a pick, a
plan, the prompt. `model` is the camera's RobotEnv.camera_model at that moment
(None if the backend does not know where the camera is).

The use it exists for — stop the arm when a hand comes near (motion_guard.py
is one, ready-made; this is the shape of your own):

    def hands(frame, model):
        if hand_near_arm(frame, model):           # your detector
            env.interlock.trip("hand near the arm", source="hands")
        else:
            env.interlock.clear("hands")

    watcher = Watcher(env, "overhead", hands, hz=10, fail_safe=True).start()
    ...
    watcher.stop()

The arm stops at the trip (interlock.py: the backend holds it from this
thread). The time to notice a hand is this watcher's period plus fn's own run
time, so a safety watcher's fn wants to be fast — far cheaper than a segmenter.

fail_safe=True is for watchers the arm's safety depends on: if fn raises, or
the camera stops delivering frames, the watcher trips the interlock itself
(source "<name>:failed") and clears it once fn runs cleanly again. A safety
watcher that cannot see is not watching.

What fn may do, from its thread: read cameras (render_image, camera_model),
read where the arm is (arm_shape), trip and clear the interlock, update the
object memory (it is thread-safe). What it must not do: send motion commands —
one thread drives the arm; the hold a trip sends is the backend's.

A background scan feeding the object memory works the same way, with one
caveat: while the arm is over the surface it hides what is under it and is in
view itself, so such a watcher should skip frames the arm is in, or mask it out.
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class Watcher:
    """Run fn(frame, model) on a camera's newest frame, on a thread of its own."""

    def __init__(self, env, camera: str, fn: Callable, hz: float = 5.0,
                 width: int = 640, height: int = 480, name: "str | None" = None,
                 fail_safe: bool = False):
        self.env = env
        self.camera = camera
        self.fn = fn
        self.period_s = 1.0 / float(hz)
        self.width, self.height = int(width), int(height)
        self.name = name or getattr(fn, "__name__", "watcher")
        self.fail_safe = fail_safe
        self.runs = 0          # calls of fn that returned
        self.errors = 0        # calls that raised (fn, or reading the camera)
        self.last_error: "str | None" = None
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None
        self._failing = False

    @property
    def failed_source(self) -> str:
        return f"{self.name}:failed"

    def start(self) -> "Watcher":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        # Daemon: a watcher must never keep a crashed controller alive.
        self._thread = threading.Thread(target=self._run, name=f"watcher-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the thread. A deliberate stop is not a failure: its fail-safe trip is cleared."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
        if self.fail_safe:
            self.env.interlock.clear(self.failed_source)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                frame = self.env.render_image(self.camera, self.width, self.height)
                model = self.env.camera_model(self.camera, self.width, self.height)
                self.fn(frame, model)
                self.runs += 1
                if self._failing:
                    print(f"  [Watch:{self.name}] Running again.")
                    self._failing = False
                    if self.fail_safe:
                        self.env.interlock.clear(self.failed_source)
            except Exception as exc:
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                if not self._failing:
                    # Once per run of failures, not every period.
                    print(f"  [Watch:{self.name}] Failed: {self.last_error}"
                          + (" — stopping the arm until it recovers." if self.fail_safe else ""))
                    self._failing = True
                if self.fail_safe:
                    self.env.interlock.trip(f"{self.name} cannot watch ({self.last_error})",
                                            source=self.failed_source)
            self._stop.wait(max(0.0, self.period_s - (time.monotonic() - started)))
