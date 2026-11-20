"""
gripper.py (mrcf/backends/ur5) — the grippers a UR5 can carry, each behind the
same few calls the env makes:

    connect(), disconnect()
    set_width(width_m)      open or close to an opening, returning at once
    width_m()               the opening now
    holding()               whether the fingers stopped on something
    moving()                whether they are still going

RobotiqSocket drives a Robotiq Hand-E, 2F-85 or 2F-140 through its URCap's
socket on the controller (port 63352: the URCap must be installed and the
gripper activated on the pendant). Its position runs 0 (open) to 255 (closed); the
opening for a position comes from two measured ends (POS_OPEN, POS_CLOSED in
robot_config), straight between them.

NoGripper is a rig without one: the arm runs as it does with one, the jaws
read closed (0 m) whatever they are asked, and the policies refuse what needs
them (mrcf/policies/arm_only.py).
"""

from __future__ import annotations

import socket
import threading
import time


class GripperError(RuntimeError):
    pass


class NoGripper:
    """No gripper on the flange: nothing to open or close, so nothing happens."""

    max_open_m = 0.0

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def set_width(self, width_m: float) -> None:
        pass

    def halt(self) -> None:
        pass

    def width_m(self) -> float:
        return 0.0

    def holding(self) -> bool:
        return False

    def moving(self) -> bool:
        return False

    def settled(self) -> bool:
        return True


class RobotiqSocket:
    """
    A Robotiq gripper (Hand-E, 2F-85, 2F-140) through its URCap's socket (port 63352). The protocol
    is lines of text: "SET POS 120" answered "ack", "GET POS" answered
    "POS 120". Variables used: ACT (activated), GTO (go to), POS (position,
    0-255, 0 open), SPE (speed), FOR (force), OBJ (0 moving, 1 or 2 stopped on
    something, 3 at the position asked for), STA (3: activated).

    Activation moves the fingers through their whole range, so it is never
    done here unasked: connect() refuses a gripper that is not activated
    (activate() does it, from a tool, with the user told first).
    """

    PORT = 63352

    def __init__(self, host: str, *, max_open_m: float, pos_open: int = 0, pos_closed: int = 227,
                 speed: int = 128, force: int = 64, timeout_s: float = 2.0, port: "int | None" = None):
        self._host, self._port = host, port or self.PORT
        self.max_open_m = float(max_open_m)
        self._pos_open, self._pos_closed = int(pos_open), int(pos_closed)
        self._speed, self._force = int(speed), int(force)
        self._timeout_s = float(timeout_s)
        self._sock: "socket.socket | None" = None
        self._lock = threading.Lock()
        self._goal: "int | None" = None          # the position last asked for (settled)

    # ── The socket ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
        if self.get("STA") != 3 or self.get("ACT") != 1:
            self.disconnect()
            raise GripperError("The Robotiq gripper is not activated: activate it on the pendant "
                               "(Installation > Gripper), or with tools/ur5/check_motion.py's gripper step.")
        self._send(f"SET SPE {self._speed}")
        self._send(f"SET FOR {self._force}")

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _ask(self, line: str) -> str:
        """
        One line sent, its answer read. A GET is answered with a line ("POS 3\n");
        a SET with a bare "ack", no newline after it (the URCap on this rig,
        2026-11-12: waiting for the newline timed out on the first SET).
        """
        if self._sock is None:
            raise GripperError("The gripper is not connected.")
        with self._lock:
            self._sock.sendall((line + "\n").encode())
            data = b""
            while not (data.endswith(b"\n") or data.strip() == b"ack"):
                try:
                    chunk = self._sock.recv(1024)
                except socket.timeout:
                    raise GripperError(f"The gripper did not answer {line!r} within {self._timeout_s:.0f} s "
                                       f"(got {data!r}).") from None
                if not chunk:
                    raise GripperError(f"The gripper's socket closed after {line!r}.")
                data += chunk
        return data.decode().strip()

    def _send(self, line: str) -> None:
        got = self._ask(line)
        if got != "ack":
            raise GripperError(f"The gripper answered {got!r} to {line!r}.")

    def get(self, var: str) -> int:
        got = self._ask(f"GET {var}").split()
        if len(got) != 2 or got[0] != var:
            raise GripperError(f"The gripper answered {' '.join(got)!r} to GET {var}.")
        return int(got[1])

    def activate(self, timeout_s: float = 5.0) -> None:
        """Activate it: the fingers close and open fully. Only on the user's say-so."""
        self._send("SET ACT 1")
        self._send("SET GTO 1")
        deadline = time.monotonic() + timeout_s
        while self.get("STA") != 3:
            if time.monotonic() > deadline:
                raise GripperError("The gripper did not finish activating.")
            time.sleep(0.05)

    # ── Openings ───────────────────────────────────────────────────────────────

    def _pos_for(self, width_m: float) -> int:
        frac = min(max(width_m / self.max_open_m, 0.0), 1.0) if self.max_open_m > 0 else 0.0
        return int(round(self._pos_closed + frac * (self._pos_open - self._pos_closed)))

    def _width_for(self, pos: int) -> float:
        span = self._pos_open - self._pos_closed
        frac = (pos - self._pos_closed) / span if span else 0.0
        return float(min(max(frac, 0.0), 1.0) * self.max_open_m) + 0.0     # + 0.0: no "-0.0 mm" closed

    def set_width(self, width_m: float) -> None:
        self._goal = self._pos_for(width_m)
        self._send(f"SET POS {self._goal}")
        self._send("SET GTO 1")

    def halt(self) -> None:
        """Stop the fingers where they are (GTO 0): a stop, or a pause for a hand at the jaws."""
        self._send("SET GTO 0")

    def settled(self) -> bool:
        """
        Done with the last set_width: the gripper has taken the position asked
        for (PRE echoes it) and stopped — there, or on something. Before it has
        taken it, it still says the last move is done (OBJ 3): on the rig
        (2026-11-12) a check 20 ms after asking read the old opening, took a
        held block for a miss, and the retry opened and dropped it.
        """
        if self._goal is not None and self.get("PRE") != self._goal:
            return False
        return self.get("OBJ") != 0

    def width_m(self) -> float:
        return self._width_for(self.get("POS"))

    def holding(self) -> bool:
        return self.get("OBJ") in (1, 2)

    def moving(self) -> bool:
        return self.get("OBJ") == 0
