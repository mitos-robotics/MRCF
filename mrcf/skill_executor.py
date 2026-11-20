"""
skill_executor.py — run a skill or a policy with the cross-cutting concerns.

    run_policy(fn, name, env, **kwargs) -> Outcome
    run_skill(fn, name, env, **kwargs)  -> bool       (the same, for (ok, msg) skills)

Both handle what is universal across robots:
  - The safety monitor (safety.ExecutionMonitor): registered as one of the
    interlock's per-cycle checks for the duration, so a hard limit stops the
    arm mid-move (interlock.py), and set as safety's module-level monitor for
    skills that poll it themselves.
  - A visual failure description from the VLM, on failure only — not when
    the policy refused before moving, and not when the arm was stopped.

Neither implements motion. `fn` is a policy method (mrcf/policies/, returning an
Outcome) or a skill function returning (success, message):

    from mrcf.skill_executor import run_policy
    outcome = run_policy(policies.pick, "pick", env, target=target, obstacles=obstacles)

A stop (MotionStopped, from the interlock) is not a failure to describe: it is
reported and passed on to whoever started the plan.
"""

from __future__ import annotations

from typing import Callable

from mrcf import safety as _safety
from mrcf.interlock import MotionStopped
from mrcf.policies.base import Outcome
from mrcf.robot_env import RobotEnv


# What the last skill to say came to — {"skill", "ok", "refused", "message"} —
# so the controller can tell the planner why a step failed, not only that it
# did. run_policy says it for every policy; a skill that gives up before it
# runs one (no room, an unknown name) says its own with report().
_last_report: "dict | None" = None


def report(skill_name: str, ok: bool, message: str, refused: bool = False) -> None:
    """Say what a skill came to, for the controller's next plan (last_report)."""
    global _last_report
    _last_report = {"skill": skill_name, "ok": bool(ok), "refused": bool(refused), "message": str(message)}


def last_report() -> "dict | None":
    """What the last skill to say came to, since clear_report(); None if none said."""
    return _last_report


def clear_report() -> None:
    global _last_report
    _last_report = None


# After a press stops a skill, the arm backs off this far straight up
# (capabilities.BackOff) before the stop goes on. Held where it stopped, it
# went on pressing: UR5, 2026-11-13, a finger on a block at 47.5 N until the
# power was cut.
BACK_OFF_M = 0.02


def back_off_after_press(env: RobotEnv, stop: MotionStopped) -> bool:
    """
    The arm backed off (capabilities.BackOff) when stop was a press (a contact
    force) and nothing else holds the interlock: a hand in view or an e-stop
    keeps it where it is. True when it backed off all the way.
    """
    from mrcf import capabilities, flight_recorder

    if (("contact force" not in str(stop.reason) and "pushing something" not in str(stop.reason))
            or not isinstance(env, capabilities.BackOff)):
        return False
    interlock = env.interlock
    if interlock.tripped():
        print(f"  [Safety] Not backing off from the press: {interlock.reason}. Release it from the pendant.")
        return False
    print(f"  [Safety] Pressed on something: backing off {BACK_OFF_M * 1000:.0f} mm straight up.")
    try:
        ok = bool(env.back_off(BACK_OFF_M))
    except MotionStopped as again:
        print(f"  [Safety] The back-off was stopped: {again.reason}. Release it from the pendant.")
        ok = False
    except Exception as exc:                       # noqa: BLE001 — said, and the stop goes on
        print(f"  [Safety] The back-off failed: {exc}. Release it from the pendant.")
        ok = False
    flight_recorder.event("safety", "back_off", ok=ok, distance_m=BACK_OFF_M, after=str(stop.reason))
    print("  [Safety] Backed off; holding there." if ok else
          "  [Safety] Did not back all the way off: check the arm before going on.")
    return ok


def run_policy(
    fn:          Callable,
    skill_name:  str,
    env:         RobotEnv,
    object_name: str | None = None,
    target_name: str | None = None,
    **kwargs,
) -> Outcome:
    """
    Run fn(env, **kwargs) under the safety monitor and return what it did, as
    an Outcome (a (success, message) return is converted). object_name and
    target_name are forwarded to fn when given, for skills that take them.
    """
    call_kwargs: dict = {}
    if object_name is not None:
        call_kwargs["object_name"] = object_name
    if target_name is not None:
        call_kwargs["target_name"] = target_name
    call_kwargs.update(kwargs)

    # ── Safety monitor lifecycle ───────────────────────────────────────────────
    # try/finally so a raising fn — or a stop — still clears the monitor;
    # otherwise later checks would poll a stale instance.
    monitor = _safety.ExecutionMonitor(env, skill_name)
    _safety.set_monitor(monitor)
    interlock = env.interlock
    check_name = f"safety-monitor:{skill_name}"
    interlock.add_check(check_name, monitor.poll)
    stopped: "MotionStopped | None" = None
    try:
        result = fn(env, **call_kwargs)
    except MotionStopped as stop:
        print(f"  [STOPPED] {skill_name}: {stop.reason}")
        stopped = stop
    except Exception as exc:
        result = Outcome(False, f"skill raised exception: {exc}")
    finally:
        _safety.clear_monitor()
        interlock.remove_check(check_name)
    if stopped is not None:
        back_off_after_press(env, stopped)          # the monitor's trip cleared above: it may move, once
        raise stopped
    if monitor.near_misses:
        print(monitor.summary())

    outcome = result if isinstance(result, Outcome) else Outcome(bool(result[0]), str(result[1]))

    # ── Visual failure description (optional VLM call) ─────────────────────────
    if not outcome.ok and not outcome.refused and DESCRIBE_FAILURES:
        try:
            from mrcf.vlm import render_to_base64, call_vlm
            described = call_vlm(
                prompt=(
                    "The robot just failed a skill. "
                    "Describe what you see in one short paragraph: "
                    "robot state, where each visible object is, and anything that looks wrong."
                ),
                image_b64=render_to_base64(env),
                json_mode=False,
            )
            if isinstance(described, str) and described.strip():
                print(f"  [Vision] {described.strip()}")
        except Exception as exc:
            print(f"  [Vision] Scene description unavailable: {exc}")

    label = "OK" if outcome.ok else ("REFUSED" if outcome.refused else "FAILED")
    print(f"  [{label}] {skill_name}: {outcome.message}")
    report(skill_name, outcome.ok, outcome.message, outcome.refused)
    return outcome


# A failed skill described by the vision model from a camera picture, said and
# not otherwise used. A deployment may turn it off (mrcf/tabletop: DESCRIBE_
# FAILURES): 5-16 s a failed pick on the SO-101's table (2026-11-19), the
# flight recorder keeping the pictures anyway.
DESCRIBE_FAILURES = True


def run_skill(
    skill_fn:    Callable,
    skill_name:  str,
    env:         RobotEnv,
    object_name: str | None = None,
    target_name: str | None = None,
    **kwargs,
) -> bool:
    """
    run_policy for a skill function returning (success, message): True on
    success. Kept for skills written that way.
    """
    return run_policy(skill_fn, skill_name, env, object_name=object_name,
                      target_name=target_name, **kwargs).ok
