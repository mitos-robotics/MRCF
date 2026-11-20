"""
mrcf/backends/so101/gripper.py — the SO-101's gripper: opening the jaws to a
width, closing them on an object, and whether it holds one, judged from the
gripper servo's load (the SO-101 has no force sensor).

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import time

import numpy as np

from mrcf import flight_recorder
from mrcf.backends.so101.follower import Follower

# squeeze_gripper: holding if the gripper's load is at least this fraction of
# its torque cap. Empty closes read ~30 of 200; a held piece reads the cap.
_GRIP_LOAD_FRACTION = 0.5
_HOLD_LOAD_FRACTION = 0.25   # gripper_loaded(): eased hold, so lower


class GripperMixin(Follower):
    """SO101Env's gripper (see env.py)."""

    # The narrowest the jaws hold at while shut on a thin wall (squeeze_gripper's min_hold_m); None: a piece's.
    _thin_hold_m: "float | None" = None

    # ── RobotEnv: manipulation ────────────────────────────────────────────────

    def get_gripper_width(self) -> float:
        """The opening in metres, from the motor's 0-100% (see _pct_to_width)."""
        return self._pct_to_width(float(self._motor_pos()["gripper"]))

    def set_gripper(self, width: float, n_steps: int = 15, timeout_s: float = 3.0,  # type: ignore[override]  # see report
                    max_step_pct: "float | None" = None) -> None:
        """
        Drive the jaws to `width` metres, re-commanding until the motor actually
        gets there (or stops moving).

        This MUST be a loop, not one send_action. lerobot applies
        max_relative_target to every motor including the gripper, so a single
        call moves the jaws at most that many percentage points — with the 8.0
        this deployment uses, asking for 100% from 8% gets 16%, jaws that have
        barely opened.

        Returns once the commanded width is reached OR the jaws stop moving —
        the latter being the normal outcome of closing onto an object. The last
        goal position stays below the physical stop in that case, so the servo
        keeps applying holding torque after this returns.

        `max_step_pct` caps how far the jaws are asked to travel per cycle.
        Left unset, the goal is simply re-sent every cycle and lerobot's
        max_relative_target governs: 8 percentage points at control_hz=20 is
        ~160 %/s, and across a 45 mm span that is ~70 mm/s of jaw travel. One
        jaw reaches a light piece before the other and flicks it off the board
        rather than gripping it. Closing onto something small wants a much
        lower number — see GRASP_CLOSE_STEP_PCT.
        """
        if float(width) > float(self.get_gripper_width()) + 0.001:
            self._thin_hold_m = None            # opened: whatever a thin grip held is let go
        # Whatever was held is being let go or re-gripped; hold_width says again
        # once a grasp is confirmed.
        self._held_width_m = None
        self.drive_gripper_pct(self._width_to_pct(width), n_steps, timeout_s, max_step_pct)
        flight_recorder.event("gripper", "set", asked_mm=width * 1000, width_mm=self.get_gripper_width() * 1000)

    def drive_gripper_pct(self, pct: float, n_steps: int = 15, timeout_s: float = 3.0,
                          max_step_pct: "float | None" = None) -> None:
        """set_gripper by motor position (0-100%) — what calibrating the width needs."""
        pct = float(np.clip(pct, 0.0, 100.0))
        reached_pct = 2.0   # close enough to the commanded opening
        moving_pct  = 0.5   # per-cycle travel below this counts as not moving
        step_pct = float(max_step_pct) if max_step_pct is not None else None
        last_pos = float(self._motor_pos()["gripper"])
        if step_pct is not None:
            # A gentler step needs proportionally longer to cross the same
            # span; keeping the caller's timeout would turn a slow close into
            # a timeout that reads as a failed grasp.
            needed = (abs(pct - last_pos) / max(step_pct, 0.1)) / self._control_hz
            timeout_s = max(timeout_s, needed * 2.5)
        stalled_cycles = 0
        previous = None
        t0 = self.motion_clock()
        while self.motion_clock() - t0 < timeout_s:
            # Ramp the GOAL from where the jaws actually are, rather than
            # re-sending the destination and letting max_relative_target set
            # the speed. last_pos is the reading taken at the end of the
            # previous cycle, so this costs no extra bus traffic.
            goal = (pct if step_pct is None
                    else float(np.clip(pct, last_pos - step_pct, last_pos + step_pct)))
            self._send({"gripper.pos": goal})
            self.step(1)
            # Position-only read: normalize=True, the default, puts this in
            # the same 0-100 space send_action's goal_pos uses.
            current = float(self._motor_pos()["gripper"])
            last_pos = current
            if abs(current - pct) <= reached_pct:
                break
            if previous is not None and abs(current - previous) < moving_pct:
                stalled_cycles += 1
                if stalled_cycles >= 3:
                    break
            else:
                stalled_cycles = 0
            previous = current
        if step_pct is not None:
            # Ramping stops a step short of `pct`, which on a closed grasp is
            # almost no holding torque. A FULL un-ramped jump to `pct` here
            # would bypass step_pct entirely and fall back to lerobot's
            # arm-wide max_relative_target (8.0, more than 5x
            # GRASP_CLOSE_STEP_PCT's 1.5), i.e. one large, sudden squeeze
            # applied exactly when the jaws have just made contact — plausibly
            # what pops a smooth, round piece sideways out of the grip: a round
            # profile has no flat to key into, so it has somewhere to go under
            # a sudden overshoot of force, where it has nowhere to go under the
            # gradual ramp. A few more steps at the SAME gentle rate ratchets
            # up holding torque instead of slamming it on.
            for _ in range(5):
                goal = float(np.clip(pct, last_pos - step_pct, last_pos + step_pct))
                self._send({"gripper.pos": goal})
                self.step(1)
                last_pos = float(self._motor_pos()["gripper"])
                if abs(last_pos - pct) <= reached_pct:
                    break
        # Let them come to rest: up to n_steps cycles, while they still move.
        # Not all n_steps whatever the jaws do: by now they have arrived or
        # stopped on the object (the loop above ends on one or the other).
        for _ in range(n_steps):
            self.step(1)
            now = float(self._motor_pos()["gripper"])
            if abs(now - last_pos) < moving_pct:
                break
            last_pos = now

    def gripper_loaded(self) -> bool:
        """
        Is the gripper pushing against something right now? Live, no motion.

        After squeeze_gripper the goal sits hold_pct past a held piece, so the
        servo keeps pressing; on an empty close the goal is where the jaws are
        and the load is ~0. Threshold is lower than the squeeze's own, because
        holding pressure is less than the squeeze.
        """
        if self.get_gripper_width() < (getattr(self, "_thin_hold_m", None) or self._min_hold_m()):
            return False                       # pads pressed on each other, not on a piece
        try:
            load = abs(self._bus_read("Present_Load", "gripper"))
        except Exception:
            return True
        return load >= _HOLD_LOAD_FRACTION * float(self._gripper_torque_limit or 500)

    def _min_hold_m(self) -> float:
        """
        The narrowest the jaws can stop on a piece (gripper_min_hold_m); narrower is shut on nothing.

        Load alone cannot tell them apart with grip pads fitted: the pads meet
        at 0.9-1.0 % of travel (1.7-1.9 mm on the width table) and press each
        other at the torque cap — load 160-200 of 200 on an empty close, where
        the bare jaws go to 0.1 % at ~30.
        """
        return float(getattr(self, "_gripper_min_hold_m", 0.0034))

    def squeeze_gripper(self, overshoot_pct: float = 10.0, hold_pct: "float | None" = None,
                        settle_s: float = 0.4, min_hold_m: "float | None" = None) -> "tuple[float, bool]":
        """
        After a closing set_gripper(), drive the jaws firmly shut and report
        where they stop. That reading is the only reliable "is anything held"
        signal this gripper has.

        set_gripper's ramped close cannot provide it. Its stall detector fires
        whenever the jaws creep under 0.5% per cycle, which friction alone does
        before they meet, so an EMPTY close stops anywhere from 0.6% to 7.8% of
        span, overlapping the 4-12% a held pawn or knight stops at. The ramp
        also leaves the goal only a step past the jaws, so the servo's holding
        torque is luck: a held piece can read the same load (~25) as an empty
        close.

        Overshooting the goal removes both: with nothing between them the jaws
        go all the way (0.1% on every empty trial), and a piece blocks them at
        3.7-6.2% with the servo at its torque cap. Then the goal is set
        hold_pct past the jaws — what it carries the piece with; None: the
        arm's GRIPPER_HOLD_PCT — or to exactly where they are, if they shut on
        nothing. GRIPPER_TORQUE_LIMIT keeps the cap itself under the overload
        trip point — pressing at lerobot's 500 trips the servo.

        Returns (opening_m, loaded). `loaded` is the grip signal to trust:
        whether the servo is pushing against something at the end of the
        squeeze. With the torque capped the opening alone does not separate the
        two — an empty close reads up to 0.48 mm and a held knight 0.74 mm —
        while load is ~30 empty against the cap (200-500) on a piece, a 6x gap
        or more.

        min_hold_m: narrower than this is shut on nothing, for this grip and
        while the jaws stay shut on it (gripper_loaded) — for a thin wall (a
        scoop's, a cup's) the jaws close to a millimetre on, under the
        narrowest a piece stops them at (_min_hold_m), and only the load tells.
        2026-11-17: a scoop's wall stopped them at 1.1 mm, load 113, where an
        empty close read 0.14 mm, load 36 — and was let go as nothing.
        """
        start = float(self._motor_pos()["gripper"])
        goal = max(0.0, start - overshoot_pct)
        # Several sends: max_relative_target caps how far one call moves the goal.
        for _ in range(6):
            self._send({"gripper.pos": goal})
            self.step(1)
        time.sleep(settle_s)
        held = float(self._motor_pos()["gripper"])
        try:
            load = abs(self._bus_read("Present_Load", "gripper"))
        except Exception:
            load = float("nan")
        cap = float(self._gripper_torque_limit or 500)
        # Shut on nothing: the jaws (or their pads) met. Then the load is the
        # pads pressing each other, not a grip (see _min_hold_m).
        floor = self._min_hold_m() if min_hold_m is None else float(min_hold_m)
        empty = self._pct_to_width(held) < floor
        # Fall back to the opening alone if the load could not be read.
        loaded = not empty and (load >= _GRIP_LOAD_FRACTION * cap if load == load else min_hold_m is None)
        self._thin_hold_m = floor if loaded and min_hold_m is not None else None
        # Shut on nothing: stop pushing into the stop. A piece is held as hard
        # as hold_pct asks; GRIPPER_TORQUE_LIMIT bounds it either way.
        if hold_pct is None:
            hold_pct = float(getattr(self, "_gripper_hold_pct", 3.0))
        hold_goal = held if empty else max(0.0, held - hold_pct)
        self._send({"gripper.pos": hold_goal})
        self.step(1)
        flight_recorder.event("gripper", "squeeze", width_mm=self._pct_to_width(held) * 1000, load=load,
                              loaded=bool(loaded))
        return self._pct_to_width(held), bool(loaded)
