"""
motion_guard.py — stop the arm when something moves near it.

A fixed camera that sees the work area watches for movement that is not the
arm's own — a hand reaching in, a sleeve, something knocked over — and trips
the interlock the moment it sees any. The arm freezes where it is, mid-move
(interlock.py; RobotEnv holds it from the guard's own thread). Once nothing
has moved for quiet_s the guard clears its trip and the move carries on where
it was (resume=True, the default: Interlock.pause_only). With resume=False the
command is abandoned instead, as for any other stop, and the arm moves again
on the next one.

    guard = MotionGuard(env, "overhead").start()     # a fail-safe Watcher
    ...
    guard.stop()

Seeing movement
---------------
Each frame is compared with the one compare_s before it, in Lab colour, after
evening out the whole frame's shift in each channel (auto exposure and white
balance follow the arm in and out of view). A pixel that changed by more than diff_lab has moved. Comparing
across compare_s, not with the very last frame, sees a hand that creeps in
slowly as well as one that darts.

Telling a hand from the arm
---------------------------
The arm moving is movement too. The backend says where the arm's body is
(RobotEnv.arm_shape, from its joint angles) and where the camera is
(RobotEnv.camera_model), so the guard draws the arm into the image — every pose
read in the last arm_window_s and the sweep between them, widened by margin_m
for the arm's shadow and the cables along it — and ignores what changed inside
that. It does so while the arm is being driven (RobotEnv.may_be_moving) and
until its joints have been still for arm_window_s: the picture runs behind the
joint readings (~70 ms on the SO-101's overhead camera), and a move ends in
small settling motions. Once the arm is idle nothing is ignored: a hand on the
gripper is seen.

"Still" means the joint readings have not changed at all (still_m). An arm
holding a pose under load is not always still: the SO-101's servos twitch it
1-2 mm every few seconds, and its gripper tag, motors and wires turn that into
a thousand changed pixels — with still_m at millimetres, enough to stop the
arm. On its readings, which do not move by a hundredth of a millimetre while it
is still, every one of 47 such frames in a minute followed a joint step of
0.3 mm or more within the second before. They put the outline back for about
40% of the time it holds a pose. (On the SO-101's tag tour frames, 5 cm of margin
left nothing big enough to stop the arm in 50 moves; 3 cm left the wrist cable
swinging out of the outline, 47 cm² of it.)

What is left is split into connected regions, and the largest, measured on the
work surface, trips the guard when it is at least min_area_cm2 — about two
fingers. Smaller ones are noise.

Where it watches
----------------
Everything in view, unless `zone` is set: then only the pixels that look down
onto that part of the work surface — a polygon of world (x, y) points, or a
distance from the arm's base (the first point of its shape). Set it to where
the arm works plus the distance a hand covers in a moment, and keep it on the
surface: the floor beside a table is far below it, so it reads as surface just
past the table's edge, and feet shuffling there would stop the arm. Pixels the
lens model cannot map — a strong barrel distortion folds back on itself
towards a wide lens's corners — are outside any zone.

Limits
------
- It sees movement, not hands: anything moving in view stops the arm, and a
  hand held perfectly still for quiet_s lets it move again.
- While the arm moves, movement inside its widened outline is not seen until it
  comes out of it, or the arm stops.
- What hangs off the arm and swings — a loose cable — moves outside its
  outline: a false stop. Tie cables along the arm. A low lamp throwing the
  arm's shadow wide does the same; widen margin_m, or light from above.
- One camera: it cannot guard what it cannot see.
It is a guard, not a certified safety system; keep the power switch in reach.

start() refuses to run without a camera model or the arm's shape: a guard that
cannot tell the arm from a hand would stop every move, or none.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import collections
import math
import time
from typing import Sequence, cast

import cv2
import numpy as np

from mrcf import flight_recorder
from mrcf.perception import lighting
from mrcf.perception import plane_geometry as pg
from mrcf.watchers import Watcher


# Movement beside the gripper that starts again each time the arm moves on is
# the arm pushing something, not a hand: the move is stopped, not paused
# (MotionGuard._trip). A deployment turns it on (mrcf/tabletop: GUARD_STOPS_
# PUSHING). 2026-11-19, the SO-101: a plug carried into a tub caught its wall;
# the guard paused four times in 9 s, each about a second after carrying on,
# and the arm pushed the tub toward the table's edge (user: "why can't the
# camera see something is wrong when this happens and stop it?").
STOP_PUSHING = False


class MotionGuard(Watcher):
    """Trip env.interlock when something other than the arm moves in a fixed camera's view."""

    PUSH_AFTER_RESUME_S = 2.5     # a trip this soon after carrying on ...
    PUSH_NEAR_M = 0.25            # ... this near the gripper (what it pushes moves all over) ...
    PUSH_STRIKES = 2              # ... this many times running: it is pushing something

    def __init__(self, env, camera: str, *,
                 hz: float = 15.0,
                 width: int = 320,
                 height: int = 180,
                 name: str = "motion",
                 diff_lab: float = 15.0,        # Lab change that counts as movement
                 min_area_cm2: float = 20.0,    # the least region that stops the arm
                 quiet_s: float = 2.0,          # nothing moving this long clears the stop
                 compare_s: float = 0.2,        # each frame is compared with the one this much older
                 margin_m: float = 0.05,        # added around the arm's shape
                 zone: "float | Sequence | np.ndarray | None" = None,  # where to watch: (x, y) polygon, or metres from the base
                 arm_window_s: float = 1.0,     # arm poses kept: compare_s plus the camera's lag, and more
                 still_m: float = 0.0002,       # the arm moving less than this is standing still
                 warmup_s: float = 5.0,         # start() waits this long for the camera's first fresh frame
                 resume: bool = True,           # pause the move and carry on when it clears (else abandon it)
                 verbose: bool = True):
        super().__init__(env, camera, self._look, hz=hz, width=width, height=height,
                         name=name, fail_safe=True)
        self.diff_lab = float(diff_lab)
        self.min_area_cm2 = float(min_area_cm2)
        self.quiet_s = float(quiet_s)
        self.compare_s = float(compare_s)
        self.margin_m = float(margin_m)
        self.zone = (None if zone is None else float(zone) if np.isscalar(zone)
                     else np.asarray(zone, float).reshape(-1, 2))
        self.arm_window_s = float(arm_window_s)
        self.still_m = float(still_m)
        self.warmup_s = float(warmup_s)
        # A stop for a hand pauses the move, and it carries on by itself once
        # nothing has moved for quiet_s (Interlock.pause_only, RobotEnv.guard).
        # Only on movement: a hand that comes in and then holds still counts as
        # gone once it has been still for quiet_s. There is no check for "still
        # there" (the scene against the frame before the trip): the zone takes
        # in the floor beside the table, and a foot moved and rested there
        # would hold the arm for minutes.
        self.resume = bool(resume)
        self.verbose = verbose
        self.trips = 0                 # times it has stopped the arm
        self.last: "dict | None" = None    # what the latest frame showed (for check tools)
        self._now = time.monotonic
        # Where it has seen something move, as (t, x, y, radius_m) on the work
        # surface: every region of every frame that counted as moving, not
        # only where the trip began (a hand trips the guard at the zone's edge,
        # and goes on to the object it moves). moved_since() reads them. Once
        # the trip clears, those where the scene is as it was before it are
        # forgotten (_forget_what_came_and_went): a hand came and went.
        self._moved_at: collections.deque = collections.deque()
        self._episode: "dict | None" = None   # this trip's: the frame before it, the arm then, where it saw
        self._unplaced_at: "float | None" = None   # when it last saw a moving region off the surface's plane
        self._watching_since: "float | None" = None     # its first look, on its own clock
        self._reset()

    def _reset(self) -> None:
        self._frames: collections.deque = collections.deque()   # (t, Lab image)
        self._held: "tuple | None" = None     # (t, Lab image, mean L): a frame held back (GLITCH_L)
        self.glitches = 0                     # frames dropped as the camera's, not the scene's
        self._shapes: collections.deque = collections.deque()   # (t, points (N, 3), radii (N,))
        self._tripped = False
        self._episode = None
        self._announced = False
        self._last_motion = 0.0
        self._resumed_at: "float | None" = None   # when it last carried on after a pause
        self._push_strikes = 0
        self._zone: "tuple | None" = None   # (key, mask) for the camera and base it was made for

    # ── Starting and stopping ─────────────────────────────────────────────────

    def start(self) -> "MotionGuard":
        if self.env.camera_model(self.camera, self.width, self.height) is None:
            raise RuntimeError(
                f"MotionGuard: the backend has no camera model for '{self.camera}', so it "
                "cannot tell the arm moving from a hand moving. Calibrate that camera first.")
        if self.env.arm_shape() is None:
            raise RuntimeError(
                "MotionGuard: the backend cannot say where the arm is (RobotEnv.arm_shape), so "
                "it cannot tell the arm moving from a hand moving.")
        # Loading the arm's model above can hold up the camera's reader thread
        # for a second; the first look would find a stale frame and stop the arm.
        deadline = time.monotonic() + self.warmup_s
        while True:
            try:
                self.env.render_image(self.camera, self.width, self.height)
                break
            except Exception:
                if time.monotonic() > deadline:
                    break          # watch anyway: the fail-safe holds the arm until it sees
                time.sleep(0.05)
        self._reset()
        if self.resume:
            self.env.interlock.pause_only(self.name)
        super().start()
        return self

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop watching. A deliberate stop takes the guard's own trip with it, as Watcher's fail-safe one."""
        super().stop(timeout_s)
        self.env.interlock.clear(self.name)
        self._reset()

    @property
    def tripped(self) -> bool:
        return self._tripped

    # How long it remembers where it saw movement (moved_since).
    KEEP_MOVED_S = 600.0
    # A frame whose whole picture is this much brighter or darker (mean Lab L,
    # 0-255) than the last one taken is held back, and taken only if the next
    # one agrees: one frame like that is the camera, not the scene. 2026-11-13,
    # the UR5's webcam sharing USB 2 with a RealSense: single frames at 75 and
    # 202 between frames at 136, each read as 1800-3600 cm² moving, and picks
    # refused to come down beside them. A hand moves the mean by a few levels;
    # a change that lasts (the lights, something large) trips one frame later.
    GLITCH_L = 20.0

    def moved_since(self, t: float) -> "list | None":
        """
        Where it saw something move since t (its clock, time.monotonic):
        [(x, y, radius_m)], world frame. None when it cannot say: t is from
        before it watched, or longer ago than it remembers.
        """
        if self._watching_since is None or t < max(self._watching_since, self._now() - self.KEEP_MOVED_S):
            return None
        if self._unplaced_at is not None and self._unplaced_at >= t:
            return None                  # it saw something move that it could not place on the surface
        return [(x, y, r) for when, x, y, r in list(self._moved_at) if when >= t]

    # ── Each frame ────────────────────────────────────────────────────────────

    def _look(self, frame: np.ndarray, model: "dict | None") -> None:
        now = self._now()
        if self._watching_since is None:
            self._watching_since = now
        if model is None:
            raise RuntimeError(f"no camera model for '{self.camera}'")
        if lighting.no_picture(frame):
            # Black shows no change, so nothing would ever stop the arm: a
            # camera covered, or its number now another's (HISTORY.md).
            raise RuntimeError(f"the picture from '{self.camera}' is black: covered, or another camera")
        shape = self.env.arm_shape()
        if shape is None:
            raise RuntimeError("the backend no longer says where the arm is")
        self._shapes.append((now, np.array([p for p, _ in shape], float),
                             np.array([r for _, r in shape], float)))
        _keep_since(self._shapes, now - self.arm_window_s)

        lab = _lab(frame)
        level = float(lab[..., 0].mean())
        last = float(self._frames[-1][1][..., 0].mean()) if self._frames else level
        if abs(level - last) > self.GLITCH_L:
            if self._held is None:
                self._held = (now, lab, level)         # one like it next, and it is the scene
                return
            if abs(level - self._held[2]) > self.GLITCH_L:
                self._held = (now, lab, level)         # unlike the last held one too: hold this instead
                return
            self._frames.append(self._held[:2])        # two alike: a change that lasts, compared as any
        elif self._held is not None:
            self.glitches += 1                         # back to before: the held frame was the camera's
            print(f"  [Sentry] Dropped a frame from '{self.camera}' {self._held[2] - last:+.0f} levels off "
                  "the ones round it (the camera, not the scene).")
        self._held = None
        self._frames.append((now, lab))
        _keep_since(self._frames, now - self.compare_s)
        if len(self._frames) < 2:
            return
        changed = changed_pixels(lab, self._frames[0][1], self.diff_lab)
        if self.zone is not None:
            changed &= self._zone_mask(model, shape[0][0], changed.shape)

        arm = None
        if self.env.may_be_moving() or arm_moved(self._shapes, self.still_m):
            arm = arm_mask(self._shapes, model, changed.shape, self.margin_m)
            changed &= ~arm
        changed = _clean(changed)
        found = regions(changed, model)
        moving = bool(found) and found[0]["area_cm2"] >= self.min_area_cm2
        self.last = {"t": now, "frame": frame, "changed": changed, "arm": arm,
                     "regions": found, "moving": moving, "tripped": self._tripped or moving}

        if moving:
            self._last_motion = now
            if not self._tripped:
                # What it looked like just before (the frame compared with), the arm then.
                self._episode = {"before": self._frames[0][1], "seen": [],
                                 "arm": arm_mask(self._shapes, model, changed.shape, self.margin_m)}
            for r in found:
                # As large as a trip, not less: beside the driven arm its mask's
                # edge leaks up to 14 cm² (the UR5's sentry check, 2026-11-12), and
                # recorded, a leak by the gripper reads as something moving by
                # the very object it is over.
                if r["xy_m"] is None and r["area_cm2"] >= self.min_area_cm2:
                    self._unplaced_at = now
                if r["xy_m"] is not None and r["area_cm2"] >= self.min_area_cm2:
                    entry = (now, r["xy_m"][0], r["xy_m"][1], math.sqrt(r["area_cm2"] / math.pi) / 100)
                    self._moved_at.append(entry)
                    if self._episode is not None:
                        self._episode["seen"].append((entry, r["box"], r["xy_m"]))
            while self._moved_at and self._moved_at[0][0] < now - self.KEEP_MOVED_S:
                self._moved_at.popleft()
            if not self._tripped:
                self._trip(found[0])
        elif self._tripped and now - self._last_motion >= self.quiet_s:
            # Before the clear: a move waiting on it (a place's moved_near) reads what is left.
            self._forget_what_came_and_went(lab, model)
            self._tripped = False
            self.env.interlock.clear(self.name)
            self.env.interlock.clear(self._pushing_source)
            self._resumed_at = now
            flight_recorder.event("sentry", "clear", quiet_s=self.quiet_s)
            if self.verbose and self._announced:
                print(f"\n  [Guard] Nothing has moved for {self.quiet_s:.0f} s — "
                      + ("carrying on." if self.resume else "the arm may move again on the next command."))
            self._announced = False

    # Once a trip clears, where it saw something move (each region's box,
    # LASTING_PAD wider each way) is compared with the frame before the trip:
    # what moved is then kept (moved_since) only where the scene still differs,
    # by LASTING_CM2 or more, at where it differs — what was moved or put down
    # there, or someone still standing in view. A hand that came and went
    # leaves nothing. UR5 tabletop, 2026-11-13: a hand passing within 1 cm of
    # the block a place was going to, touching nothing, sent the arm back still
    # holding its block, and the look after found nothing moved; on its frames
    # the block's spot was unchanged after it, and the one change that lasted
    # was the user, still at the table's edge 15 cm away. A 30 mm block's top
    # is 9 cm²; moved, it changes both where it was and where it went.
    LASTING_CM2 = 4.0
    LASTING_PAD = 0.3
    # A region more than this share behind the arm (the frame before the trip,
    # or now) is not checked: what moved there is kept.
    UNSEEN_SHARE = 0.25

    def _forget_what_came_and_went(self, lab: np.ndarray, model: dict) -> None:
        episode, self._episode = self._episode, None
        if episode is None or not episode["seen"]:
            return
        arm = episode["arm"] | arm_mask(self._shapes, model, lab.shape[:2], self.margin_m)
        lasting = changed_pixels(lab, episode["before"], self.diff_lab) & ~arm
        if self.zone is not None and self._zone is not None:
            lasting &= self._zone[1]
        h, w = lasting.shape
        looked = np.zeros_like(lasting)
        for _, (x, y, bw, bh), _ in episode["seen"]:
            px, py = int(bw * self.LASTING_PAD) + 1, int(bh * self.LASTING_PAD) + 1
            looked[max(0, y - py):min(h, y + bh + py), max(0, x - px):min(w, x + bw + px)] = True
        stays = [r for r in regions(_clean(lasting & looked), model, top=10)
                 if r["xy_m"] is not None and r["area_cm2"] >= self.LASTING_CM2]
        # Where the arm hid it, before or now, the two frames cannot say it is as
        # it was: what moved there is kept as seen. 2026-11-13, the UR5: the user
        # took the block just stacked while the arm was still over it; with its
        # spot left out as the arm's, the trip "came and went", and the next
        # block went on where it had been.
        unseen = {entry for entry, (x, y, bw, bh), _ in episode["seen"]
                  if float(arm[y:y + bh, x:x + bw].mean()) > self.UNSEEN_SHARE}
        began = min(entry[0] for entry, _, _ in episode["seen"])
        seen = {entry for entry, _, _ in episode["seen"]}
        self._moved_at = collections.deque(
            [e for e in self._moved_at if e not in seen or e in unseen]
            + [(began, r["xy_m"][0], r["xy_m"][1], math.sqrt(r["area_cm2"] / math.pi) / 100) for r in stays])
        flight_recorder.event("sentry", "came_and_went" if not stays and not unseen else "left_a_change",
                              regions=len(seen), behind_the_arm=len(unseen),
                              lasting=[{"xy_m": r["xy_m"], "area_cm2": r["area_cm2"]} for r in stays])

    def _zone_mask(self, model: dict, base, size) -> np.ndarray:
        """zone_mask, made again only when the camera or the base has moved."""
        key = (np.round(np.asarray(model["T_world_cam"], float), 4).tobytes(),
               np.round(np.asarray(base, float)[:2], 3).tobytes(), tuple(size))
        if self._zone is None or self._zone[0] != key:
            self._zone = (key, zone_mask(model, cast("float | np.ndarray", self.zone), size, base=base))  # set: _look checks
        return self._zone[1]

    def _trip(self, region: dict) -> None:
        x, y = region["xy_m"] if region["xy_m"] is not None else (float("nan"), float("nan"))
        reason = (f"something moving in view of the {self.camera} camera "
                  f"({region['area_cm2']:.0f} cm² near x={x:+.2f}, y={y:+.2f} m)")
        was_moving = self.env.may_be_moving()
        self._tripped = True
        self.trips += 1
        self.env.interlock.trip(reason, source=self.name)
        # On the timeline with the frame that showed it: a piece rolling off
        # the board looks like this too (flight_recorder's docstring).
        flight_recorder.event("sentry", "pause" if self.resume else "stop", image=(self.last or {}).get("frame"),
                              area_cm2=region["area_cm2"], xy_m=[x, y], arm_moving=was_moving)
        # A trip while the arm is at rest (the user moving a piece between
        # commands) stops nothing, and would only interrupt their typing.
        if self.verbose and was_moving:
            print(f"\n  [Guard] {'Paused' if self.resume else 'Stopped'} the arm: {reason}."
                  + (f" It carries on once nothing has moved for {self.quiet_s:.0f} s." if self.resume else ""))
            self._announced = True
        if self._pushing(region, was_moving):
            why = (f"the arm is pushing something: movement beside the gripper each time it moves on "
                   f"({self._push_strikes} times) — {reason}")
            self.env.interlock.trip(why, source=self._pushing_source)
            flight_recorder.event("sentry", "pushing", xy_m=[x, y], strikes=self._push_strikes)
            if self.verbose:
                print("  [Guard] It moves each time the arm does, right beside the gripper: the arm is pushing it. "
                      "Stopping this move, not carrying on.")

    @property
    def _pushing_source(self) -> str:
        return f"{self.name}:pushing"

    def _pushing(self, region: dict, was_moving: bool) -> bool:
        """Whether this trip, with those before it, is the arm pushing something (STOP_PUSHING)."""
        if not STOP_PUSHING or self.resume is False:
            return False
        soon = (was_moving and region.get("xy_m") is not None and self._resumed_at is not None
                and self._now() - self._resumed_at <= self.PUSH_AFTER_RESUME_S)
        near = False
        if soon:
            try:
                ee = np.asarray(self.env.get_ee_pos(), float)
                near = math.hypot(region["xy_m"][0] - ee[0], region["xy_m"][1] - ee[1]) <= self.PUSH_NEAR_M
            except Exception:                                   # noqa: BLE001 — no position: not a push
                near = False
        self._push_strikes = self._push_strikes + 1 if near else 0
        return self._push_strikes >= self.PUSH_STRIKES


# ── The steps, each usable on its own ─────────────────────────────────────────

def _keep_since(q: collections.deque, since: float) -> None:
    """Drop entries older than `since`, keeping the newest one before it (the window's start)."""
    while len(q) > 1 and q[1][0] <= since:
        q.popleft()


def _lab(frame_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    return cv2.GaussianBlur(lab, (5, 5), 0)


def changed_pixels(lab: np.ndarray, before: np.ndarray, diff_lab: float) -> np.ndarray:
    """Pixels whose colour changed by more than diff_lab, the whole frame's shift in each channel aside."""
    d = lab - before
    d -= np.median(d.reshape(-1, 3), axis=0)
    return np.einsum("ijk,ijk->ij", d, d) > diff_lab * diff_lab


def _clean(mask: np.ndarray) -> np.ndarray:
    """Specks out, then the pieces of one moving thing joined up."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return m.astype(bool)


def zone_mask(model: dict, zone: "float | Sequence | np.ndarray", size, base=(0.0, 0.0)) -> np.ndarray:
    """
    Pixels whose ray meets the work surface inside `zone`, bool (h, w): a
    polygon of world (x, y) points, or a distance in metres from `base`.
    Pixels the lens model cannot map are outside.
    """
    h, w = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    xy = surface_xy(model, np.c_[u.ravel(), v.ravel()])
    ok = np.isfinite(xy).all(axis=1)
    inside = np.zeros(len(xy), bool)
    if np.isscalar(zone):
        inside[ok] = np.linalg.norm(xy[ok] - np.asarray(base, float)[:2], axis=1) <= float(zone)
    else:
        inside[ok] = _in_polygon(xy[ok], np.asarray(zone, float).reshape(-1, 2))
    return inside.reshape(h, w)


def surface_xy(model: dict, px: np.ndarray) -> np.ndarray:
    """
    World (x, y), metres, where each pixel's ray meets the work surface; NaN
    where it does not, or where the lens model cannot say which ray it is.

    cv2.undistortPoints' few default iterations go wrong toward a wide lens's
    edge, and past the fold of a strong barrel distortion there is no answer
    to find. So: iterate to convergence, and keep only pixels the model
    projects back to where they were.
    """
    K = np.asarray(model["K"], float)
    D = np.asarray(model["D"], float).ravel()
    px = np.asarray(px, np.float64).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
    if hasattr(cv2, "undistortPointsIter"):
        n = cv2.undistortPointsIter(px, K, D, None, None, crit)  # type: ignore[call-overload]  # takes R, P None; its stubs say not
    else:                                   # OpenCV 5: undistortPoints takes the criteria itself
        n = cv2.undistortPoints(px, K, D, R=None, P=None, criteria=crit)  # type: ignore[call-overload]  # OpenCV 5's; the stubs are 4's
    n = n.reshape(-1, 2)
    back, _ = cv2.projectPoints(np.c_[n, np.ones(len(n))], np.zeros(3), np.zeros(3), K, D)
    ok = (np.linalg.norm(back.reshape(-1, 2) - px.reshape(-1, 2), axis=1) < 0.5) \
        & (np.linalg.norm(n, axis=1) < _fold_radius(D))
    T = np.asarray(model["T_world_cam"], float)
    with np.errstate(all="ignore"):
        rays = np.c_[n, np.ones(len(n))] @ T[:3, :3].T
        s = (float(model.get("plane_z_m", 0.0)) - T[2, 3]) / rays[:, 2]
        xy = T[:2, 3] + rays[:, :2] * s[:, None]
    xy[~ok | ~(s > 0)] = np.nan
    return xy


def _fold_radius(D) -> float:
    """
    How far from the optical axis (normalised, undistorted) the lens model
    still maps rays to distinct pixels. Beyond it a barrel distortion folds
    back: rays far outside the picture land inside it.
    """
    D = np.asarray(D, float).ravel()
    k1, k2, k3 = (D[0] if len(D) > 0 else 0.0), (D[1] if len(D) > 1 else 0.0), (D[4] if len(D) > 4 else 0.0)
    r = np.linspace(0.0, 4.0, 4001)
    rd = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    folds = np.nonzero(np.diff(rd) <= 0)[0]
    return float(r[folds[0]]) if len(folds) else float("inf")


def _in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Whether each (x, y) is inside the polygon (even-odd rule)."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    with np.errstate(all="ignore"):
        crosses = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
    return (crosses.sum(axis=1) % 2) == 1


def arm_moved(shapes, still_m: float) -> bool:
    """Whether any point of the arm moved more than still_m across these poses."""
    if len(shapes) < 2:
        return False
    n = len(shapes[-1][1])
    same = [p[:n] for _, p, _ in shapes if len(p) >= n]          # its body, as many points as now: what it
    if len(same) < 2:                                            # carries (capabilities.Carries) comes and goes
        return True
    pts = np.stack(same)
    return float((pts.max(axis=0) - pts.min(axis=0)).max()) > still_m


def arm_mask(shapes, model: dict, size, margin_m: float, step_m: float = 0.015) -> np.ndarray:
    """
    Where the arm is in the image, bool (h, w): each pose in `shapes` — (t,
    points, radii) — and the poses between consecutive ones, as round-ended
    tubes of radius + margin_m joining its points.
    """
    h, w = size
    pts: list | np.ndarray
    rad: list | np.ndarray
    pts, rad = [], []
    poses = [(p, r) for _, p, r in shapes]
    for i in range(len(poses)):
        p1, r1 = poses[i]
        p0 = poses[i - 1][0] if i else p1
        if len(p0) != len(p1):
            p0 = p1               # what it carries was taken up or let go between them: no sweep across that
        sweep = float(np.linalg.norm(p1 - p0, axis=1).max())
        n = max(1, math.ceil(sweep / step_m))
        for k in (range(1, n + 1) if i else (0,)):
            chain = p0 + (p1 - p0) * (k / n)
            for a, b, ra, rb in zip(chain[:-1], chain[1:], r1[:-1], r1[1:]):
                m = max(1, math.ceil(np.linalg.norm(b - a) / step_m))
                s = np.linspace(0.0, 1.0, m + 1)[:, None]
                pts.append(a + (b - a) * s)
                rad.append(ra + (rb - ra) * s[:, 0])
    mask = np.zeros((h, w), np.uint8)
    if not pts:
        return mask.astype(bool)
    pts, rad = np.concatenate(pts), np.concatenate(rad) + margin_m

    K = np.asarray(model["K"], float)
    D = np.asarray(model["D"], float).ravel()
    T_cw = np.linalg.inv(np.asarray(model["T_world_cam"], float))
    with np.errstate(all="ignore"):     # numpy on macOS's Accelerate warns on clean float64 matmuls
        pc = pts @ T_cw[:3, :3].T + T_cw[:3, 3]
    # Behind the camera, or past the fold of its lens model (a strong barrel
    # distortion maps such points back INTO the image), a point is not in the
    # picture.
    with np.errstate(all="ignore"):
        ok = (pc[:, 2] > 0.02) & (np.linalg.norm(pc[:, :2] / pc[:, 2:3], axis=1) < _fold_radius(D))
    if not ok.any():
        return mask.astype(bool)
    px, _ = cv2.projectPoints(pts[ok], cv2.Rodrigues(T_cw[:3, :3])[0], T_cw[:3, 3], K, D)
    r_px = np.ceil(rad[ok] * max(K[0, 0], K[1, 1]) / pc[ok, 2])
    circles = np.unique(np.c_[np.round(px.reshape(-1, 2)), r_px].astype(np.int64), axis=0)
    for u, v, r in circles:
        if -r <= u < w + r and -r <= v < h + r:
            cv2.circle(mask, (int(u), int(v)), int(r), 1, -1)
    return mask.astype(bool)


def regions(mask: np.ndarray, model: dict, top: int = 3) -> list:
    """
    The largest connected regions of `mask`, largest first: area_cm2 on the
    work surface, xy_m (where on the surface, world frame; None if its ray
    misses the surface) and box (x, y, w, h) in pixels.
    """
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    view = pg.PlaneView.from_camera_model(model)
    K = np.asarray(model["K"], float)
    order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:top]
    where = surface_xy(model, np.vstack([centroids[order], K[:2, 2]]))
    centre_scale = _px_per_cm2(view, where[-1]) if np.isfinite(where[-1]).all() else None
    found = []
    for i, xy in zip(order, where[:-1]):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        # Where the lens cannot say, the scale at the middle of the picture.
        scale = _px_per_cm2(view, xy) if np.isfinite(xy).all() else centre_scale
        found.append({"area_cm2": area / scale if scale else float("inf"),
                      "xy_m": tuple(float(v) for v in xy) if np.isfinite(xy).all() else None,
                      "box": (x, y, bw, bh)})
    return sorted(found, key=lambda f: -f["area_cm2"])


def _px_per_cm2(view: "pg.PlaneView", xy_m) -> float:
    """Pixels per cm² of surface at world xy_m: the image of a 1 cm square there."""
    p = (np.asarray(xy_m, float) - view.origin_m) * 1000.0
    a, b, d = view.project(np.array([[p[0], p[1], 0.0], [p[0] + 10, p[1], 0.0], [p[0], p[1] + 10, 0.0]]))
    (ux, uy), (vx, vy) = b - a, d - a
    return abs(ux * vy - uy * vx) / view.scale ** 2
