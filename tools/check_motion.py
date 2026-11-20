"""
check_motion.py — an arm's first moves from this computer, on any arm
(MRCF_ARM): small, slow, and each one asked for before it runs. THE ARM MOVES.
The first thing to run on a new arm once it connects, before any task.

Before you start:
  - the arm 15 cm or more clear of the table and everything else, the jaws
    pointing roughly down;
  - its speed turned down, if it has a slider (a UR's pendant: 10-25%);
  - a hand on its emergency stop (or its power) for the whole run.

    MRCF_ARM=ur5 python tools/check_motion.py               # every step
    MRCF_ARM=ur5 python tools/check_motion.py --steps 1,4   # only these

The steps (at each: Enter runs it, s skips it, q stops here):

  1. Up 3 cm and back down: a straight line, the orientation held.
  2. 3 cm along the base's x and back.
  3. The jaws turned 20 deg about their own axis, and back.
  4. A stop: a 6 cm move up, stopped by the interlock part-way; how far the
     arm ran on after the stop, then back down.
  5. A pause: a 4 cm move paused for 2 s by the interlock, then carried on.
  6. The gripper: open, half, closed, half (if it has one, and it is ready:
     an arm whose gripper needs readying asks first, prepare_gripper below).
  7. Freedrive: 15 s in which the arm can be pushed by hand, then it holds
     (an arm with capabilities.Freedrive; skipped on one without).
  8. Home (go_home).
  9. Park (go_park).
  An arm that keeps its gripper in a box (config GRIPPER_BOX) checks every
  move against it: one that would take it out is not sent, and says so.

Each step's result is printed and saved in rig/<arm>/check_motion/<time>.json,
with the largest force the arm read at the tool in each step: what its
reading does touching nothing, which CONTACT_FORCE_LIMITS must sit well
above (0 on an arm with no force sensing); and, on an arm that reports it
(current_error: a UR), how far each joint's current strayed from the
controller's model, the other way to feel a contact.

Nothing here needs a camera, and the motion guard is not running (it needs
the overhead camera calibrated): you are the guard.

What the arm's package may add (mrcf/backends/__init__.py): prepare_gripper(ask)
gets its gripper ready (the UR5's Robotiq: activated) and says whether to use
it; without one, the gripper is used as it comes.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mrcf.capabilities import GripSensing  # noqa: E402
from tools._common import backends, connecting_to  # noqa: E402

STEPS = {1: "up 3 cm and back", 2: "3 cm along x and back", 3: "the jaws turned 20 deg and back",
         4: "a move stopped part-way by the interlock", 5: "a move paused for 2 s", 6: "the gripper",
         7: "freedrive for 15 s", 8: "go home", 9: "go to the park pose"}
FREEDRIVE_S = 15


def ask(prompt: str) -> str:
    """'' (go), 's' (skip) or 'q' (stop)."""
    while True:
        got = input(f"{prompt} [Enter: go, s: skip, q: stop] ").strip().lower()
        if got in ("", "g", "go", "y", "yes"):
            return ""
        if got in ("s", "skip", "n", "no"):
            return "s"
        if got in ("q", "quit", "stop"):
            return "q"
        print(f"  ({got!r} is not an answer: press Enter to go, s to skip, q to stop)")


def mm(v) -> str:
    return f"{float(v) * 1000:.1f} mm"


def _turned(rotation: np.ndarray, deg: float) -> np.ndarray:
    """rotation turned deg about its own z (the jaws' axis)."""
    a = np.radians(deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(rotation, float) @ rz


class Check:
    def __init__(self, env, max_open_m: float):
        self.env = env
        self.max_open_m = max_open_m
        self.force_max = 0.0                 # the largest get_contact_force() since the step began
        self.current_max: "np.ndarray | None" = None   # each joint's largest |current - the model's|, A
        self._sampling = threading.Event()
        threading.Thread(target=self._sample_force, daemon=True).start()

    def _sample_force(self) -> None:
        current_error = getattr(self.env, "current_error", None)
        while not self._sampling.is_set():
            try:
                self.force_max = max(self.force_max, float(self.env.get_contact_force()))
                if current_error is not None:
                    now = np.abs(np.asarray(current_error(), float))
                    self.current_max = now if self.current_max is None else np.maximum(self.current_max, now)
            except Exception:                    # noqa: BLE001 — disconnected at the end: stop sampling
                return
            time.sleep(0.02)

    def close(self) -> None:
        self._sampling.set()

    def here(self) -> np.ndarray:
        return np.asarray(self.env.get_ee_pos(), float)

    def line(self, target) -> "tuple[bool, float]":
        """A straight line to target, the orientation held; (arrived, how far off)."""
        arrived = self.env.move_ee_to(np.asarray(target, float), tol=0.002, orientation=self.env.get_ee_rotation())
        return arrived, float(np.linalg.norm(self.here() - target))

    def there_and_back(self, offset) -> dict:
        start = self.here()
        out, off_out = self.line(start + offset)
        back, off_back = self.line(start)
        print(f"   out: {'arrived' if out else 'DID NOT ARRIVE'}, {mm(off_out)} off; "
              f"back: {'arrived' if back else 'DID NOT ARRIVE'}, {mm(off_back)} off")
        return {"out": out, "out_off_m": off_out, "back": back, "back_off_m": off_back}

    def turn(self) -> dict:
        env, start, rot = self.env, self.here(), self.env.get_ee_rotation()
        turned = bool(env.move_ee_to(start, tol=0.003, orientation=_turned(rot, 20.0)))
        by = float(np.degrees(np.arccos(np.clip((np.trace(rot.T @ env.get_ee_rotation()) - 1) / 2, -1, 1))))
        back = bool(env.move_ee_to(start, tol=0.003, orientation=rot))
        print(f"   turned: {'yes' if turned else 'NO'}, by {by:.1f} deg; back: {'yes' if back else 'NO'}, "
              f"the grasp point {mm(np.linalg.norm(self.here() - start))} from where it began")
        return {"turned": turned, "turned_deg": by, "back": back}

    def _moving(self, target, during) -> dict:
        """move to target in a thread; when it has gone 1/3 of the way, run during(); join."""
        env, start = self.env, self.here()
        result: dict = {}

        def move():
            try:
                result["arrived"] = env.move_ee_to(np.asarray(target, float), tol=0.002,
                                                   orientation=env.get_ee_rotation())
            except BaseException as exc:            # noqa: BLE001 — MotionStopped is the expected end of step 4
                result["raised"] = f"{type(exc).__name__}: {exc}"

        mover = threading.Thread(target=move, daemon=True)
        mover.start()
        third = np.linalg.norm(np.asarray(target) - start) / 3
        deadline = time.monotonic() + 60
        while np.linalg.norm(self.here() - start) < third and mover.is_alive() and time.monotonic() < deadline:
            time.sleep(0.005)
        during(result)
        mover.join(120)
        return result

    def stop(self) -> dict:
        env, start = self.env, self.here()
        info: dict = {}

        def trip(result):
            info["at_trip"] = self.here()
            t0 = time.monotonic()
            env.interlock.trip("check_motion: a stop", source="check_motion")     # holds the arm as it trips
            while env.may_be_moving() and time.monotonic() - t0 < 5:
                time.sleep(0.01)
            info["stopped_after_s"] = time.monotonic() - t0

        result = self._moving(start + [0, 0, 0.06], trip)
        ran_on = float(np.linalg.norm(self.here() - info["at_trip"]))
        env.interlock.clear("check_motion")
        print(f"   stopped: {result.get('raised', 'NOT STOPPED: ' + str(result))}")
        print(f"   ran on {mm(ran_on)} after the trip, still after {info['stopped_after_s']:.2f} s")
        back, off = self.line(start)
        print(f"   back down: {back}, {mm(off)} off")
        return {"raised": result.get("raised"), "ran_on_m": ran_on, "still_after_s": info["stopped_after_s"],
                "back": back}

    def pause(self) -> dict:
        env, start = self.env, self.here()
        info: dict = {}

        def pause(result):
            env.interlock.pause_only("check_motion_pause")
            env.interlock.trip("check_motion: a pause", source="check_motion_pause")
            time.sleep(0.5)                           # it decelerates, then holds
            held = self.here()
            time.sleep(1.5)
            info["crept"] = float(np.linalg.norm(self.here() - held))
            env.interlock.clear("check_motion_pause")

        result = self._moving(start + [0, 0, 0.04], pause)
        print(f"   crept {mm(info['crept'])} while paused; carried on and "
              f"{'arrived' if result.get('arrived') else 'DID NOT ARRIVE: ' + str(result)}")
        back, off = self.line(start)
        print(f"   back down: {back}, {mm(off)} off")
        return {"crept_m": info["crept"], "arrived": result.get("arrived"), "back": back}

    def gripper(self) -> dict:
        env, got, full = self.env, {}, self.max_open_m
        for name, width in (("open", full), ("half", full / 2), ("closed", 0.0), ("half again", full / 2)):
            env.set_gripper(width)
            got[name] = env.get_gripper_width()
            loaded = env.gripper_loaded() if isinstance(env, GripSensing) else None
            print(f"   {name}: asked {mm(width)}, reads {mm(got[name])}"
                  + ("" if loaded is None else f", holding: {loaded}"))
        return got

    def go(self, name: str) -> dict:
        """go_home() or go_park(): where it ended, and whether it got there."""
        env, start = self.env, self.here()
        arrived = env.go_home() if name == "home" else env.go_park()
        end = self.here()
        print(f"   {name}: {'arrived' if arrived else 'NOT reached (why: above)'}; the grasp point "
              f"{np.round(start, 3).tolist()} -> {np.round(end, 3).tolist()} m")
        return {"arrived": bool(arrived), "from": start.tolist(), "to": end.tolist()}

    def freedrive(self) -> dict:
        env, start = self.env, self.here()
        env.freedrive(True)
        try:
            for left in range(FREEDRIVE_S, 0, -1):
                print(f"\r   freedrive: push the arm by hand ... {left:2d} s ", end="", flush=True)
                time.sleep(1.0)
        finally:
            env.freedrive(False)
        moved = float(np.linalg.norm(self.here() - start))
        print(f"\r   holding again; it was moved {mm(moved)}.                ")
        return {"moved_m": moved}


def make_env(with_gripper: bool):
    """
    The arm's env for this check: no sentry, no cameras opened, and no gripper
    unless with_gripper — each as far as the arm's make_env takes it (an arm
    that opens its own cameras takes no sources=; one with one gripper, no gripper=).
    """
    make = backends.arm().make_env
    kwargs: dict = {"motion_guard_zone": None, "sources": {}}
    if not with_gripper:
        kwargs["gripper"] = None
    while True:
        try:
            return make(**kwargs)
        except TypeError as exc:
            dropped = next((k for k in ("sources", "gripper") if k in kwargs and repr(k) in str(exc)), None)
            if dropped is None:
                raise
            kwargs.pop(dropped)


def run(env, steps, *, with_gripper: bool, out_dir: Path) -> dict:
    """Run the steps on the connected env, asking before each; save and return what each measured."""
    from mrcf import capabilities

    config = backends.arm().config
    done: dict = {}                              # step -> what it measured
    check = Check(env, float(config.GRIPPER_MAX_OPEN_M))
    results: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": backends.arm_name(), "steps": done}
    try:
        p, rot = check.here(), env.get_ee_rotation()
        line = (f"Connected. The grasp point is at {np.round(p, 3).tolist()} m, the jaws pointing "
                f"{np.round(rot[:, 2], 2).tolist()}")
        scaling = env.speed_scaling() if hasattr(env, "speed_scaling") else None
        print(line + ("." if scaling is None else f"; speed slider {scaling * 100:.0f}%."))
        if scaling is not None and scaling > 0.3 and ask(
                f"The speed slider is at {scaling * 100:.0f}%: lower it to 10-25% first. Carry on anyway?") != "":
            return results
        for n in steps:
            if n == 6 and not with_gripper:
                print("\n6. The gripper: skipped (none, or not ready).")
                continue
            if n == 7 and not isinstance(env, capabilities.Freedrive):
                print("\n7. Freedrive: skipped (this arm has none).")
                continue
            got = ask(f"\n{n}. {STEPS[n].capitalize()}.")
            if got == "q":
                break
            if got == "s":
                continue
            step = {1: lambda: check.there_and_back([0, 0, 0.03]), 2: lambda: check.there_and_back([0.03, 0, 0]),
                    3: check.turn, 4: check.stop, 5: check.pause, 6: check.gripper, 7: check.freedrive,
                    8: lambda: check.go("home"), 9: lambda: check.go("park")}[n]
            try:
                check.force_max, check.current_max = 0.0, None
                done[n] = step()
                if n != 7:                            # freedrive: your hands are the force
                    print(f"   force at the tool, the most it read: {check.force_max:.1f} N")
                    done[n]["force_max_n"] = check.force_max
                    if check.current_max is not None:
                        print("   joint currents, the most each strayed from the controller's model (A): "
                              + "  ".join(f"{a:.2f}" for a in check.current_max))
                        done[n]["current_error_max_a"] = check.current_max.tolist()
            except BaseException as exc:            # noqa: BLE001 — say what stopped it, then stop
                env.hold()
                print(f"   STOPPED: {type(exc).__name__}: {exc}")
                done[n] = {"error": f"{type(exc).__name__}: {exc}"}
                if isinstance(exc, KeyboardInterrupt):
                    break
                if ask("Carry on with the next step?") != "":
                    break
    finally:
        check.close()
        out = out_dir / time.strftime("%Y%m%d-%H%M%S.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=lambda v: np.asarray(v).tolist()))
        print(f"\nResults: {out}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--steps", default=",".join(map(str, STEPS)), help="which steps, e.g. 1,4 (default: all)")
    args = parser.parse_args()
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    arm = backends.arm()

    print(f"check_motion on the {arm.NAME}.")
    print(__doc__.split("Before you start:")[1].split("    MRCF_ARM")[0].rstrip())
    if ask("\nIs all of that so?") != "":
        return
    prepare = getattr(arm, "prepare_gripper", None)
    with_gripper = bool(prepare(ask)) if prepare is not None else True
    print(f"\nConnecting to {connecting_to()} ...")
    env = make_env(with_gripper)
    env.connect()
    try:
        run(env, steps, with_gripper=with_gripper,
            out_dir=Path(getattr(arm.config, "RIG_DIR", Path("rig") / backends.arm_name())) / "check_motion")
    finally:
        env.stop()
        print("Disconnected.")


if __name__ == "__main__":
    main()
