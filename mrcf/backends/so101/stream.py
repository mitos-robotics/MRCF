"""
mrcf/backends/so101/stream.py — streamed goals (capabilities.ActionStreaming):
joint goals sent as they come, as a learned policy sends them, held back at the
workspace's bounds and at a joint that has met something. What the stream is —
its motors, camera and rate — is demos.py.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import time

import numpy as np

from mrcf.backends.so101.follower import Follower
from mrcf.backends.so101.kinematics import JOINT_NAMES


class StreamMixin(Follower):
    """SO101Env's streamed goals (see env.py)."""

    # ── Streamed goals: a learned policy ──────────────────────────────────────

    # A stream that has gone quiet this long is over: the next goal starts a
    # new one, from where the arm is, not from the last goal of the old one.
    STREAM_GAP_S = 0.5

    def stream_joints(self, goal: dict[str, float]) -> "tuple[dict[str, float], str | None]":
        """
        One goal of a stream sent a control tick at a time — a learned
        policy's action: {joint: degrees} for every arm joint, and "gripper"
        (%) if it moves too. Sent through _send, so the interlock and the
        sentry stop it as they stop any other move.

        A stream has no IK or reach check behind it, so each goal is checked
        here before it goes. Clipped to the calibrated ranges and
        stream_joint_limits, quietly — a stream at a limit would otherwise
        print every tick. A joint commanded STALL_FAR_DEG past where it is for
        STALL_S without moving is blocked (_stream_watch_stall): it is held
        where it stopped, and not pushed that way again until the goal backs
        off UNBLOCK_DEG — a joint pushed into whatever stops it overloads and
        drops off the bus (the wrist roll meets something near -95 deg). Not
        the gripper: squeezing is a push it is meant to hold, under
        its torque limit. Then, if the jaw tips would go below stream_tip_floor_z
        or the grasp point leave stream_bounds — further than the last goal
        sent had them — the arm's part is refused and the arm kept at the last
        goal that passed, the gripper still following when that passes too: a
        goal into the table is one the servos push against until they overload.

        Returns the goal as sent (after lerobot's per-step clamp) and why the
        arm's part was held back, or None.
        """
        goal = {name: float(v) for name, v in goal.items()}
        for name, v in goal.items():
            for lo_hi in (self._joint_deg_limits.get(name), self.__dict__.get("_stream_joint_limits", {}).get(name)):
                if lo_hi is not None:
                    v = float(np.clip(v, *lo_hi))
            goal[name] = v
        now = time.monotonic()
        pos = self._motor_pos()
        kept = self.__dict__.get("_stream_kept")
        if kept is None or now - self.__dict__.get("_stream_t", -np.inf) > self.STREAM_GAP_S:
            kept = {name: float(pos[name]) for name in (*JOINT_NAMES, "gripper")}
            self.__dict__["_stream_pushing"], self.__dict__["_stream_held"] = {}, {}
        self.__dict__["_stream_t"] = now
        blocked = self._stream_blocked(goal)
        refused = self._stream_refusal(goal, kept)
        out = goal
        if refused is not None:
            out = {**goal, **{name: kept[name] for name in JOINT_NAMES}}
            if self._stream_refusal(out, kept) is not None:
                out = {name: kept[name] for name in goal}
        stalled = self._stream_watch_stall(out, pos, now)
        self.__dict__["_stream_kept"] = {**kept, **out}
        sent = self._send({f"{name}.pos": v for name, v in out.items()})
        why = "; ".join(filter(None, (blocked, stalled, refused))) or None
        return {key.removesuffix(".pos"): float(v) for key, v in sent.items()}, why

    # ── capabilities.ActionStreaming: the stream a learned policy runs on ─────
    # What it is — six motors in, the same six out, the wrist camera — is
    # mrcf/backends/so101/demos.py, the demonstrations a policy for it was trained
    # on: one definition, so a policy runs on what it was shown.

    @property
    def stream_spec(self):
        """streaming.StreamSpec: the SO-101's (demos.spec), with this arm's wrist camera at its capture size, and its sensors."""
        from mrcf.backends.so101 import demos
        from mrcf.streaming import sensor_specs
        width, height = self.__dict__.get("_camera_capture_wh") or (getattr(self, "_render_w", 640),
                                                                     getattr(self, "_render_h", 480))
        return demos.spec(self._camera_name, width, height, sensors=sensor_specs(self))

    def stream_observation(self) -> dict:
        """The follower now, as a demonstration recorded it (demos.observe)."""
        from mrcf.backends.so101 import demos
        spec = self.stream_spec
        camera, width, height = spec.cameras.get(demos.CAMERA_KEY, (None, 0, 0))
        return demos.observe(self, camera, width, height, spec)

    def stream_action(self, action) -> "tuple[np.ndarray, str | None]":
        """One action vector (demos.NAMES' order) through stream_joints: (as sent, why any was held back)."""
        from mrcf.backends.so101 import demos
        sent, why = self.stream_joints(demos.goal(action))
        return demos.action(sent), why

    def stream_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Each joint's range a stream may send it over: calibrated, and inside stream_joint_limits."""
        out = {}
        for name, (lo, hi) in self._joint_deg_limits.items():
            s_lo, s_hi = self.__dict__.get("_stream_joint_limits", {}).get(name, (lo, hi))
            out[name] = (max(lo, s_lo), min(hi, s_hi))
        return out

    # How far past the grasp point's ceiling (stream_bounds.z_max) a stream
    # stays: the safety monitor halts on the MEASURED grasp point above it, and
    # an arm sent right up to it overshoots.
    STREAM_CEILING_MARGIN_M = 0.02

    # A blocked joint is free again once the goal has come back this far from
    # where it stopped.
    UNBLOCK_DEG = 2.0

    def _stream_blocked(self, goal: dict[str, float]) -> "str | None":
        """
        Holds back, in `goal`, any arm joint found blocked (_stream_watch_stall)
        from going further the way it was blocked; frees it once the goal has
        backed off UNBLOCK_DEG. Says which are held, or None.
        """
        blocked = self.__dict__.setdefault("_stream_held", {})
        held = []
        for name, (at, side) in list(blocked.items()):
            if (goal[name] - at) * side > 0:
                goal[name] = at
                held.append(f"{name} blocked at {at:.0f} deg")
            elif (at - goal[name]) * side > self.UNBLOCK_DEG:
                del blocked[name]
        return "; ".join(held) or None

    def _stream_watch_stall(self, out: dict[str, float], pos: dict, now: float) -> "str | None":
        """
        A joint COMMANDED (`out`, after every refusal) more than STALL_FAR_DEG
        from where it is, for STALL_S without a break, that has moved under
        STALL_MOVE_DEG in that time, is pushing against something: found
        blocked, and stopped at once (in `out`). Says which, or None.

        Only what was commanded counts, and only for the whole window. A goal
        that has just jumped is not a push yet: compared with the last second's
        stillness, a shoulder lifted briskly off its rest would look blocked.
        Nor is a goal a refusal kept from being sent.
        """
        pushing = self.__dict__.setdefault("_stream_pushing", {})
        blocked = self.__dict__.setdefault("_stream_held", {})
        found = []
        for name in JOINT_NAMES:
            push = out[name] - pos[name]
            if name in blocked or abs(push) <= self.STALL_FAR_DEG:
                pushing.pop(name, None)
                continue
            since, start = pushing.setdefault(name, (now, float(pos[name])))
            if now - since < self.STALL_S:
                continue
            del pushing[name]                       # a fresh window from here either way
            if abs(pos[name] - start) < self.STALL_MOVE_DEG:
                blocked[name] = (float(pos[name]), float(np.sign(push)))
                out[name] = float(pos[name])
                found.append(f"{name} blocked at {pos[name]:.0f} deg")
        return "; ".join(found) or None

    def _stream_refusal(self, goal: dict[str, float], kept: dict[str, float]) -> "str | None":
        """
        Why stream_joints may not send `goal`, or None: it would take the jaw
        tips under the floor or the grasp point out of bounds — further than
        `kept`, the goal last sent, already had them. A goal on the way back
        in, or no further out, goes: an arm that starts outside (the overhead
        parking pose is behind the workspace) must be able to come back, and
        refusing every goal there would leave it stuck where it is.
        """
        out, why = self._stream_violation(goal)
        if why is None or out <= self._stream_violation(kept)[0] + 1e-4:
            return None
        return why

    def _stream_violation(self, goal: dict[str, float]) -> "tuple[float, str | None]":
        """
        How far `goal` takes the jaws outside what a stream may reach, metres
        (jaw tips under the floor, grasp point out of bounds, added up), and
        why; (0.0, None) inside.
        """
        if self._stream_tip_floor_z is None and self._stream_bounds is None:
            return 0.0, None
        self._require_kinematics()
        out, why = 0.0, None
        pct = goal["gripper"] if "gripper" in goal else float(self._motor_pos()["gripper"])
        gap = float(np.clip(self._pct_to_width(pct), 0.0, self._gripper_max_open_m))
        assert self._kinematics is not None     # _require_kinematics() above
        t = self._kinematics.forward_kinematics(np.append([goal[name] for name in JOINT_NAMES], 0.0))
        rot, origin = t[:3, :3], t[:3, 3]
        if self._stream_tip_floor_z is not None:
            # Each jaw's tip, at both edges of its finger: whichever way the
            # gripper leans or rolls, one of these is its lowest point.
            slide = self._jaw_slide_axis
            across = np.cross([0.0, 0.0, 1.0], slide) * self._jaw_finger_half_width_m
            reach = np.array([0.0, 0.0, self._jaw_tip_below_m])
            low = min(float((origin + rot @ (pad + reach + side * across))[2])
                      for pad in (self._tcp_offset, self._tcp_offset + slide * gap)
                      for side in (-1.0, 1.0))
            if low < self._stream_tip_floor_z:
                out += self._stream_tip_floor_z - low
                why = f"jaw tips {(self._stream_tip_floor_z - low) * 1000:.0f} mm below the floor"
        b = self._stream_bounds
        if b is not None:
            # Not z_min: the tips' floor above is the stricter check on height.
            g = origin + rot @ self._tcp(gap)
            for axis, v, lo, hi in (("x", g[0], b.x_min, b.x_max), ("y", g[1], b.y_min, b.y_max),
                                    ("z", g[2], -np.inf, b.z_max - self.STREAM_CEILING_MARGIN_M)):
                if not lo <= v <= hi:
                    out += max(lo - v, v - hi)
                    why = why or f"grasp point {axis}={v:+.3f} m outside the workspace [{lo:+.3f}, {hi:+.3f}]"
        return out, why
