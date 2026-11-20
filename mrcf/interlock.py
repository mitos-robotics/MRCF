"""
interlock.py — one way to stop the arm, from anywhere, mid-move.

Anything that should stop the arm trips the interlock: a safety watcher on
another thread (a hand seen near the arm), a check the control loop runs every
cycle (the workspace floor), a button. The backend checks it before every
command it sends to the motors, so a stop lands within one control cycle — in
the middle of a move, not between a policy's phases — and no policy has to
remember to look.

    env.interlock.trip("hand near the arm", source="hands")    # from any thread
    env.interlock.clear("hands")                                # when it has gone

A trip is latched: the interlock stays tripped until every source that tripped
it has cleared. While it is tripped, every motion command raises MotionStopped,
after the backend has held the arm where it is (RobotEnv.guard) — except when
only sources registered with pause_only hold it: then the move pauses, and
carries on by itself when they clear.

The moment it trips, it calls whatever is registered with on_trip, on the
tripping thread — the backend registers its hold there (RobotEnv.interlock), so
a hand seen by a watcher freezes the arm at once, even while the control loop
is between commands.

MotionStopped derives from BaseException, as KeyboardInterrupt does, so an
`except Exception` in a policy or a skill cannot swallow it: it unwinds the
policy and the plan, and whoever started the plan decides what happens next.

Checks polled every cycle (add_check) run on the control loop's own thread,
just before a command is sent, so they may read the robot's state; a check
that fires trips the interlock under its own name, and removing the check
clears that trip.

Nothing here knows about any robot. It imports only the standard library.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import threading
from typing import Callable


class MotionStopped(BaseException):
    """Raised by a motion command while the interlock is tripped."""

    def __init__(self, reason: str, pausing: bool = False):
        super().__init__(reason)
        self.reason = reason
        # Tripped only by pausing sources (Interlock.pause_only) when raised.
        self.pausing = pausing


class Interlock:
    """Thread-safe stop flag with named sources and per-cycle checks."""

    def __init__(self):
        self._lock = threading.Lock()
        self._trips: dict[str, str] = {}
        self._checks: dict[str, Callable[[], "str | None"]] = {}
        self._on_trip: list[Callable[[str], None]] = []
        self._clear = threading.Event()
        self._clear.set()
        self._trip_count = 0
        self._stop_count = 0
        self._by_source: dict[str, int] = {}     # trips, by the source that made them
        self._pausing: set = set()

    # ── Pausing sources ────────────────────────────────────────────────────────

    def pause_only(self, source: str) -> None:
        """
        Make `source`'s trips pause the arm rather than stop it: the move under
        way holds, and carries on by itself once `source` clears
        (RobotEnv.guard), instead of raising MotionStopped. A trip from any
        other source still stops it. For a watcher whose cause goes away by
        itself — a hand over the work surface.
        """
        with self._lock:
            self._pausing.add(source)

    def only_pausing(self) -> bool:
        """Tripped, and only by sources that pause (pause_only)."""
        with self._lock:
            return bool(self._trips) and all(s in self._pausing for s in self._trips)

    def stopping(self) -> bool:
        """Tripped by at least one source that stops rather than pauses."""
        with self._lock:
            return any(s not in self._pausing for s in self._trips)

    # ── Tripping and clearing (any thread) ─────────────────────────────────────

    def trip(self, reason: str, source: str = "manual") -> None:
        """Stop the arm. Stays tripped until `source` clears it."""
        with self._lock:
            tripping = not self._trips
            if source not in self._trips:
                self._by_source[source] = self._by_source.get(source, 0) + 1
                if source not in self._pausing:
                    self._stop_count += 1
            self._trips[source] = str(reason)
            self._clear.clear()
            if tripping:
                self._trip_count += 1
            listeners = list(self._on_trip) if tripping else []
        # Outside the lock: a listener holds the arm, which takes the motor
        # bus, and the control loop may be holding the bus while it checks here.
        for fn in listeners:
            try:
                fn(str(reason))
            except Exception as exc:
                print(f"  [Interlock] on_trip {getattr(fn, '__name__', fn)} failed: {exc}")

    def on_trip(self, fn: Callable[[str], None]) -> None:
        """
        Call fn(reason) each time the interlock trips — goes from clear to
        tripped — on the tripping thread. Not again for more sources while it
        stays tripped: whatever fn did for the first still stands.
        """
        with self._lock:
            self._on_trip.append(fn)

    @property
    def trip_count(self) -> int:
        """How many times it has gone from clear to tripped."""
        with self._lock:
            return self._trip_count

    @property
    def trips_by_source(self) -> dict:
        """How many times each source has tripped it (a copy)."""
        with self._lock:
            return dict(self._by_source)

    @property
    def stop_count(self) -> int:
        """How many trips came from a source that stops rather than pauses (pause_only): a stop, a fault, a press."""
        with self._lock:
            return self._stop_count

    def clear(self, source: "str | None" = None) -> None:
        """Clear the trip from `source`, or every trip when None."""
        with self._lock:
            if source is None:
                self._trips.clear()
            else:
                self._trips.pop(source, None)
            if not self._trips:
                self._clear.set()

    @property
    def reason(self) -> "str | None":
        """Why the arm is stopped — every source's reason — or None when it may move."""
        with self._lock:
            return "; ".join(self._trips.values()) or None

    def tripped(self) -> bool:
        return not self._clear.is_set()

    def wait_clear(self, timeout: "float | None" = None) -> bool:
        """Block until nothing holds the interlock tripped. False on timeout."""
        return self._clear.wait(timeout)

    # ── Per-cycle checks (registered from anywhere, run on the control loop) ──

    def add_check(self, name: str, check: Callable[[], "str | None"]) -> None:
        """
        Poll `check` before every motion command. It returns a reason to stop,
        or None; a reason trips the interlock with `name` as its source.
        """
        with self._lock:
            self._checks[name] = check

    def remove_check(self, name: str) -> None:
        """Stop polling `name`, and clear any trip it caused."""
        with self._lock:
            self._checks.pop(name, None)
        self.clear(name)

    def check(self) -> None:
        """
        For the backend's control loop, before each command: run the per-cycle
        checks, then raise MotionStopped if anything holds the interlock tripped.
        """
        with self._lock:
            checks = list(self._checks.items())
        for name, fn in checks:
            why = fn()
            if why:
                self.trip(why, source=name)
        # Reason and kind read together: a pausing source clearing between the
        # two would otherwise turn a pause into a stop.
        with self._lock:
            reason = "; ".join(self._trips.values()) or None
            pausing = bool(self._trips) and all(s in self._pausing for s in self._trips)
        if reason:
            raise MotionStopped(reason, pausing=pausing)
