"""
check_motion_guard.py — does the motion guard stop the arm for a hand, and only
for a hand?

The guard (motion_guard.py) watches the overhead camera and freezes the arm the
moment something other than the arm moves near it; every program that connects
to the arm runs it (robot_config MOTION_GUARD_ZONE, RigEnv's connect()). This runs the
same guard on its own, instrumented, and says what it sees.

    python tools/check_motion_guard.py           # the arm stays put
    python tools/check_motion_guard.py --move    # the arm sweeps the board
    python tools/check_motion_guard.py --camera side   # through another of MOTION_GUARD_CAMERAS (default: the first)

Arm still: wave a hand over the board. STOP should print as it comes into
view, CLEAR 2 s after it has gone. With nothing moving it stays quiet, and
every few seconds prints the largest moving region it saw — the noise floor,
which wants to be well under --min-area.

--move: the arm hovers 15 cm over four points around the board, three times,
the guard running. Stand clear the first time: any STOP then is a false one —
the arm's own cable or shadow — and its picture says where. Then run it again
and put a hand in view mid-sweep: the arm should freeze at once, and carry on
2 s after the hand has gone. At the end it says how far the camera's picture
runs behind the joint readings, from how much the picture keeps
changing on the arm against how far the joints moved — what the guard's
arm_window_s has to cover — and leaves every frame's numbers in ticks.csv.

Each STOP leaves two pictures under reports/motion_guard/<time>/: the frame
that stopped it (stop_NN) and the one where most moved before it cleared
(stop_NN_most) — what moved (red), the arm's outline it ignored (blue), the
watch zone's edge (yellow). A near miss — half a stop's area or more, but
short of it — leaves one too (near_NN), so a margin that is thin shows where.

Options (the guard's defaults; the zone from the deployment's MOTION_GUARD_ZONE,
else the arm's, robot_config MOTION_GUARD_ZONE):
    --deployment chess  --cycles 3  --zone 0.5 (m from the base, instead)
    --margin 0.05  --min-area 20  --diff 15
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import (FRAMEWORK_ROOT, connecting_to, fixed_camera, gripper_box, make_env,  # noqa: E402
                           orientation_for, rc, resolve_camera, task_config)
from mrcf.interlock import MotionStopped  # noqa: E402
from mrcf.motion_guard import MotionGuard, arm_mask, changed_pixels  # noqa: E402
from mrcf import privacy  # noqa: E402

HOVER_ABOVE_TABLE_M = 0.15
SWEEP_XY = [(0.16, 0.10), (0.30, 0.10), (0.30, -0.10), (0.16, -0.10)]
REPORT_EVERY_S = 3.0


class CheckedGuard(MotionGuard):
    """The guard, keeping the largest region it has seen and a picture of each stop."""

    def __init__(self, *args, out_dir: Path, **kwargs):
        super().__init__(*args, verbose=False, **kwargs)
        self.out_dir = out_dir
        self.peak_cm2 = 0.0
        self.pictures: list = []
        self._was_tripped = False
        self._most: "dict | None" = None     # the frame where most moved, during this stop
        self._near: "dict | None" = None     # the frame where most moved short of a stop, since save_near
        self._nears = 0
        self.ticks: list = []                # one row per frame: ticks.csv, and camera_lag()
        self._prev_pts: "np.ndarray | None" = None

    def _look(self, frame, model):
        trips = self.trips
        super()._look(frame, model)
        last = self.last
        if last is None:
            return
        biggest = last["regions"][0]["area_cm2"] if last["regions"] else 0.0
        self.peak_cm2 = max(self.peak_cm2, biggest)
        if self.trips > trips:
            r = last["regions"][0]
            where = f" near ({r['xy_m'][0]:+.2f}, {r['xy_m'][1]:+.2f}) m" if r["xy_m"] else ""
            print(f"  STOP   {r['area_cm2']:4.0f} cm² moving{where}")
            self._save(last, f"stop_{self.trips:02d}")
            self._most = None
            self._near = None          # that was the stop coming, not a near miss
        if self.tripped and (self._most is None or biggest > self._most["regions"][0]["area_cm2"]):
            self._most = last if last["regions"] else self._most
        if not self.tripped and last["regions"] and (
                self._near is None or biggest > self._near["regions"][0]["area_cm2"]):
            self._near = last
        self._log(model, biggest)
        if self._was_tripped and not self.tripped:
            most = self._most["regions"][0]["area_cm2"] if self._most else 0.0
            print(f"  CLEAR  nothing has moved for {self.quiet_s:.0f} s (most at once: {most:.0f} cm²)")
            if self._most is not None:
                self._save(self._most, f"stop_{self.trips:02d}_most")
            self.peak_cm2 = 0.0
        self._was_tripped = self.tripped

    def _log(self, model, biggest: float) -> None:
        """This frame's numbers: how far the joints moved, how much changed on the arm's body."""
        if len(self._frames) < 2:
            return
        pts = self._shapes[-1][1]
        step = 0.0 if self._prev_pts is None else float(np.linalg.norm(pts - self._prev_pts, axis=1).max())
        self._prev_pts = pts
        raw = changed_pixels(self._frames[-1][1], self._frames[0][1], self.diff_lab)
        body = arm_mask(list(self._shapes), model, raw.shape, 0.0)
        self.ticks.append({"t": round(self.last["t"], 4), "joint_step_mm": round(step * 1000, 2),
                           "tip_x": round(float(pts[-2][0]), 4), "tip_y": round(float(pts[-2][1]), 4),
                           "tip_z": round(float(pts[-2][2]), 4),
                           "driven": self.env.may_be_moving(), "masked": self.last["arm"] is not None,
                           "on_arm_px": int((raw & body).sum()), "biggest_cm2": round(biggest, 1),
                           "tripped": self.tripped})

    def save_ticks(self) -> "Path | None":
        if not self.ticks:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "ticks.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.ticks[0]))
            w.writeheader()
            w.writerows(self.ticks)
        return path

    def save_near(self) -> None:
        """A picture of the biggest near miss since the last call, if it came within half of a stop."""
        near, self._near = self._near, None
        if near is not None and near["regions"][0]["area_cm2"] >= self.min_area_cm2 / 2:
            self._nears += 1
            print(f"         near miss: {near['regions'][0]['area_cm2']:.0f} cm²")
            self._save(near, f"near_{self._nears:02d}")

    def _save(self, last: dict, name: str) -> None:
        img = last["frame"].copy()
        if last["arm"] is not None:
            img[last["arm"]] = (img[last["arm"]] * 0.5 + np.array([0, 0, 110])).astype(np.uint8)
        if self._zone is not None:
            z = self._zone[1].astype(np.uint8)
            img[(z - cv2.erode(z, np.ones((3, 3), np.uint8))).astype(bool)] = (255, 255, 0)
        img[last["changed"]] = (255, 0, 0)
        img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
        for r in last["regions"]:
            x, y, w, h = (3 * v for v in r["box"])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.putText(img, f"{r['area_cm2']:.0f} cm2", (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{name}.png"
        privacy.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.pictures.append(path)
        print(f"         {path.relative_to(FRAMEWORK_ROOT)}")


def _describe(zone) -> str:
    if zone is None:
        return "everything in view"
    if np.isscalar(zone):
        return f"within {zone:.2f} m of the base"
    xy = np.asarray(zone, float).reshape(-1, 2)
    return (f"x {xy[:, 0].min():+.2f}..{xy[:, 0].max():+.2f} m, "
            f"y {xy[:, 1].min():+.2f}..{xy[:, 1].max():+.2f} m")


def watch(env, guard: CheckedGuard) -> None:
    print("\nWatching, the arm still. Wave a hand over the board; Ctrl-C to stop.\n")
    while True:
        time.sleep(REPORT_EVERY_S)
        if env.interlock.tripped() and not guard.tripped:
            print(f"  STOPPED by something else: {env.interlock.reason}")
        elif not guard.tripped:
            print(f"  quiet  largest moving region in the last {REPORT_EVERY_S:.0f} s: "
                  f"{guard.peak_cm2:4.1f} cm² (a stop takes {guard.min_area_cm2:.0f})")
            guard.save_near()
        guard.peak_cm2 = 0.0


def camera_lag(ticks: list, compare_s: float) -> "tuple[float, float] | None":
    """
    How far the picture runs behind the joint readings, in seconds, and how
    well that fits (a correlation, 0-1): the delay that best lines up how far
    the joints moved with how much the picture changed on the arm. Each frame
    is compared with one compare_s older, so the joints' movement is summed
    over that long first. (Timing each start and stop instead misreads a slow
    start as the picture leading.)
    """
    if len(ticks) < 20:
        return None
    t = np.array([r["t"] for r in ticks], float)
    dt = float(np.median(np.diff(t)))
    step = np.array([r["joint_step_mm"] for r in ticks], float)
    seen = np.array([r["on_arm_px"] for r in ticks], float)
    # The guard compares with the newest frame at least compare_s old: up to a tick older still.
    moved = np.convolve(step, np.ones(round(compare_s / dt) + 1))[:len(step)]
    best = None
    for lag in range(int(1.0 / dt) + 1):
        a, b = moved[:len(moved) - lag], seen[lag:]
        if a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if best is None or r > best[1]:
            best = (lag * dt, r)
    return best


def _until_clear(env) -> None:
    while not env.interlock.wait_clear(timeout=0.5):
        pass


def sweep_xy() -> list:
    """Where to hover: inside the arm's gripper box (GRIPPER_BOX), when it has one, else SWEEP_XY."""
    box = gripper_box()
    if box is None:
        return SWEEP_XY
    (x0, x1), (y0, y1) = box[0], box[1]
    xs, ys = (x0 + 0.33 * (x1 - x0), x0 + 0.67 * (x1 - x0)), (y0 + 0.4 * (y1 - y0), y0 + 0.73 * (y1 - y0))
    return [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]


def sweep(env, guard: CheckedGuard, cycles: int) -> None:
    z = rc.TABLE_Z + HOVER_ABOVE_TABLE_M
    points = []
    for x, y in sweep_xy():
        target = np.array([x, y, z])
        chosen = orientation_for(env, target)
        if chosen is None:
            print(f"  ({x:+.2f}, {y:+.2f}) is out of reach — left out.")
            continue
        points.append((target, chosen[0]))
    if not points:
        print("None of the sweep's points is in reach.")
        return
    print(f"\nThe arm will hover {HOVER_ABOVE_TABLE_M * 100:.0f} cm over the board at {len(points)} "
          f"points, {cycles} times, then go home.\n"
          "Nothing on the board taller than 10 cm. Stand clear the first time: a STOP\n"
          "then is the guard seeing the arm's own cable or shadow.")
    input("Press Enter to start (Ctrl-C to cancel)... ")
    print("Sweeping. Keep your hands out of the camera's view until it is home.")
    frozen = 0
    for cycle in range(cycles):
        guard.peak_cm2 = 0.0
        for target, rot in points:
            while True:
                try:
                    if not env.move_ee_to(target, tol=0.01, orientation=rot):
                        print(f"  Did not reach ({target[0]:+.2f}, {target[1]:+.2f}).")
                    break
                except MotionStopped as stop:
                    frozen += 1
                    print(f"  Frozen mid-move: {stop.reason}. Carrying on once it clears...")
                    _until_clear(env)
        print(f"  Sweep {cycle + 1}/{cycles} done. Largest moving region outside the arm: "
              f"{guard.peak_cm2:.1f} cm² (a stop takes {guard.min_area_cm2:.0f}).")
        guard.save_near()
    while True:
        try:
            env.go_home()
            break
        except MotionStopped:
            _until_clear(env)
    print(f"\n{frozen} stop(s) mid-move in {cycles * len(points)} moves.")
    lag = camera_lag(guard.ticks, guard.compare_s)
    if lag is not None and lag[1] >= 0.5:
        print(f"The picture runs about {1000 * lag[0]:.0f} ms behind the joint readings (fit {lag[1]:.2f}); "
              f"the guard keeps the arm's outline {guard.arm_window_s:.1f} s after its joints stop.")
    else:
        print("Too little movement to say how far the picture runs behind the joints.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deployment", help="whose MOTION_GUARD_ZONE to use (default: the only one)")
    ap.add_argument("--move", action="store_true", help="sweep the arm over the board while watching")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--camera", help="the camera to watch through (default: the first of MOTION_GUARD_CAMERAS)")
    ap.add_argument("--zone", type=float, help="watch this far from the arm's base, m, instead of the deployment's zone")
    ap.add_argument("--margin", type=float, default=0.05, help="added around the arm's outline, m")
    ap.add_argument("--min-area", type=float, default=20.0, help="the least region that stops the arm, cm²")
    ap.add_argument("--diff", type=float, default=15.0, help="the Lab change that counts as movement")
    args = ap.parse_args()
    # lerobot warns at every step a long move is limited to MAX_RELATIVE_TARGET_DEG — normal, and it
    # buries the guard's own lines.
    logging.getLogger().setLevel(logging.ERROR)

    # As the arm reads it (RigEnv._sentry_cameras): None is the overhead camera, [] is none.
    watched = getattr(rc, "MOTION_GUARD_CAMERAS", None)
    watched = [c for c in ([rc.OVERHEAD_CAMERA_NAME] if watched is None else watched) if c is not None]
    camera = fixed_camera(resolve_camera(args.camera)) if args.camera else (watched[0] if watched else None)
    if camera is None:
        raise SystemExit("No camera to watch with: MOTION_GUARD_CAMERAS is empty (or OVERHEAD_CAMERA_NAME None). "
                         "--camera NAME tries one.")
    zone = args.zone
    if zone is None:
        try:
            zone = getattr(task_config(args.deployment), "MOTION_GUARD_ZONE", None)
        except SystemExit:
            zone = None
        if zone is None:
            zone = getattr(rc, "MOTION_GUARD_ZONE", None)

    env = make_env(motion_guard_zone=None)     # this tool runs its own, instrumented guard
    print(f"Connecting to {connecting_to()}...")
    env.connect()
    out = FRAMEWORK_ROOT / "reports" / "motion_guard" / datetime.now().strftime("%Y%m%d_%H%M%S")
    guard = CheckedGuard(env, camera, out_dir=out, zone=zone,
                         margin_m=args.margin, min_area_cm2=args.min_area, diff_lab=args.diff)
    try:
        guard.start()
        print(f"Guard on '{camera}': watching {_describe(zone)}, "
              f"margin {args.margin * 100:.0f} cm, stops at {args.min_area:.0f} cm², "
              f"clears after {guard.quiet_s:.0f} s quiet.")
        if args.move:
            sweep(env, guard, args.cycles)
        else:
            watch(env, guard)
    except KeyboardInterrupt:
        print()
    finally:
        guard.stop()
        env.stop()
        ticks = guard.save_ticks()
        if ticks is not None:
            print(f"Every frame's numbers: {ticks.relative_to(FRAMEWORK_ROOT)}")
    print(f"{guard.trips} stop(s)" + (f"; pictures in {out.relative_to(FRAMEWORK_ROOT)}" if guard.pictures else "")
          + (f"; the guard failed {guard.errors} time(s), last: {guard.last_error}" if guard.errors else ""))


if __name__ == "__main__":
    main()
