"""
controller.py — the natural-language controller: a person's command in, the
arm doing it, back to the prompt.

    command ─► decompose into sub-goals ─► plan actions ─► safety ─► dispatch, step by step
                                              ▲                            │ a step failed
                                              └── look again, re-plan ◄────┘

The loop is the same for every deployment. What differs comes in as a
Deployment — how the scene is looked at, where the arm waits while it is, what
to do when a named object is not in view, where the arm goes home to:

    from mrcf import controller
    controller.run(env, scene_state, controller.Deployment(
        name="SO-101 chess",
        scan=scan_scene,                  # (env, scene_state) -> bool
        observe_pose=goto_observe_pose,   # (env) -> bool: the arm out of the camera's way
        home=POLICIES.home,               # (env): at the end, or after each command (home_after_command)
        examples=["pick up the pawn", "move the knight to f6"],
    ))

Between commands the arm stays where the last one left it, and the scene
stands as the arm's own actions left it. It is looked at again before the next
command only if the interlock has tripped since the last look — the motion
guard seeing a hand over the work surface, a stop — as only then can
something other than the arm have moved (look_again_if_disturbed).

A deployment's own entry point connects the arm, configures the models and the
safety layer, registers its actions, and calls run(); see
deployments/chess/llm_controller.py.

A stop (MotionStopped — the interlock tripped by a watcher or the safety
monitor) abandons the command where it is; nothing moves again until the cause
has cleared and the next command is given. A pause (the motion guard's) is not
a stop: the move waits, and carries on.

Debugging commands go straight to one action, with no model in the loop:

    /actions                 every registered action and its parameters
    /objects                 the world model, with coordinates
    /pick white_pawn_d4      that action on that object (positional or key=value)
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from mrcf import flight_recorder
from mrcf import safety
from mrcf import skill_executor
from mrcf import vlm
from mrcf.action_registry import ACTIONS, dispatch
from mrcf.capabilities import Sentry
from mrcf.interlock import MotionStopped
from mrcf.scene_state import SceneState


@dataclass
class Deployment:
    """What a deployment gives the controller loop. Every hook is optional."""
    name: str = "robot"
    # Update the world model from the cameras: (env, scene_state) -> bool.
    # Default: the vision model's generic discovery (vlm.discover_scene).
    scan: "Callable | None" = None
    # Put the arm where scan needs it (out of a fixed camera's view, or at a
    # wrist camera's calibrated pose): (env) -> bool.
    observe_pose: "Callable | None" = None
    # A named object still missing after a fresh scan: look for it elsewhere,
    # leaving the arm back where scan needs it: (env, scene_state, name) -> bool.
    look_around: "Callable | None" = None
    # Send the arm home: (env). Default: env.go_home().
    home: "Callable | None" = None
    # Home after every command that moved the arm, as well as at the end. Off,
    # it waits where the command left it: going home and coming back again
    # only costs time.
    home_after_command: bool = False
    examples: Sequence[str] = ()          # commands shown at the prompt
    task_planner: bool = True             # decompose a command into sub-goals first
    # Started for the whole session, before the arm first moves: each
    # (env) -> a started watchers.Watcher, stopped at the end.
    watchers: Sequence[Callable] = field(default_factory=tuple)
    # After the session's first look, before the prompt: the deployment's own
    # start from what that look saw (chess: carry on the saved game, or begin
    # one on a board set up for it): (env, scene_state). It moves nothing.
    on_start: "Callable | None" = None
    # A plan that is pick-and-put units ("put these in the tub"): a unit whose
    # pick fails is set aside and tried once more at the end, after a fresh
    # look, while the rest go on; then what was not done is said, and nothing
    # is planned again (execute_plan, _run_units). False: the first failed
    # step stops the plan and the goal is planned again from a fresh look.
    defer_failed: bool = False
    # A command planned before: (words, scene_state) -> the plan made for them
    # then, when it still fits what is there now; else None, and it is planned
    # as usual. plan_made (words, actions) is told each first plan for a
    # command, to keep. 2026-11-19 (user: "its the same objects in the same
    # context - so it shouldn't be wasting time"): "put the four in the tub",
    # the same four by the same names, planned afresh in 54-65 s every time.
    remembered_plan: "Callable | None" = None
    plan_made: "Callable | None" = None
    # A plan the deployment can make from the words themselves, without the
    # planner: (words, env, scene_state) -> actions, or None (planned as
    # usual). The planner stays for everything else. 2026-11-19, the SO-101's
    # table: "put these four in the tub" planned in 54-240 s, timing out twice
    # and once coming back cut off — and the plug looked for by the scan's name
    # for its pins, "silver cylinder", not the user's "white plug".
    quick_plan: "Callable | None" = None
    # The deployment's own tasks, planned in advance: (words, env, scene_state)
    # -> the actions for the task those words name, or None (not one of them).
    # Tried before anything else. task_names: what they are called, said at the
    # prompt. User, 2026-11-19: "i want to have the option now for any new
    # deployment that we can skip the planner - so it just does the task we pre
    # plan for it".
    task_plan: "Callable | None" = None
    # After a task's units: (env, scene_state, done names) -> its plan again for what is left on the table
    # (tabletop skills.task_more), run once more. None: not looked for.
    task_more: "Callable | None" = None
    task_names: Sequence[str] = ()
    # False: the planner is never asked. A command the deployment's own plans
    # (task_plan, remembered_plan, quick_plan) do not cover is answered with what
    # it does, nothing planned; a plan that fails is said, not planned again.
    # 2026-11-19 18:13, the tidy table: "the red ball" not named by the scan, the
    # planner asked instead — 61 s, a plan that stopped at its first miss, and a
    # 71 s re-plan that left the ball out.
    planner: bool = True
    # A unit set aside whose object the fresh look no longer tracks: taken to
    # be the one of its name (less any _objN) nearest where it was, within
    # this (None: not looked for). 2026-11-19 16:23, the SO-101's table: the
    # plug, pushed 4 cm by a failed pick, was named white_plug again by the
    # look but under a new track, and was never tried again.
    refind_near_m: "float | None" = None
    max_auto_steps: int = 8               # follow-up planning cycles per sub-goal
    max_history: int = 6                  # past commands the planner sees

    def look(self, env, scene_state: SceneState) -> bool:
        """The observe pose, then a scan."""
        if self.observe_pose is not None:
            self.observe_pose(env)
        return self.scan_here(env, scene_state)

    def scan_here(self, env, scene_state: SceneState) -> bool:
        """A scan from wherever the arm is."""
        trips, stops, started, sources = _trip_count(env), _stop_count(env), time.monotonic(), _by_source(env)
        if self.scan is not None:
            ok = bool(self.scan(env, scene_state))
        else:
            vlm.discover_scene(env, scene_state)
            ok = True
        # Counted from before the scan: a trip during it may have moved
        # something it had already looked at.
        _stamp_look(scene_state, trips, stops, started, sources)
        return ok

    def go_home(self, env) -> None:
        if self.home is not None:
            self.home(env)
        else:
            env.go_home()


# ── Running a plan ─────────────────────────────────────────────────────────────

def referenced_names(action_dict: dict) -> list[str]:
    """object_name/target_name from an action dict (flattened as dispatch() flattens it)."""
    params = {k: v for k, v in action_dict.items() if k != "action"}
    if isinstance(params.get("parameters"), dict):
        params.update(params["parameters"])
    return [v for k, v in params.items() if k in ("object_name", "target_name") and isinstance(v, str)]


def ensure_visible(actions: list, scene_state: SceneState, env, deployment: Deployment) -> None:
    """
    Make sure every object/target a plan references is actually in scene_state
    before the plan is validated or dispatched.

    This has to happen BEFORE dispatch — check_preconditions() (called inside
    dispatch()) blocks pick/place outright if the named object isn't in
    scene_state, so trying to recover from inside those skill functions would
    never run. safety.validate_plan() likewise blocks on unknown names.

    Two escalating steps, because they fix different failures (a name still
    unmatched is said, with what was seen, and left to fail its step):
      1. Re-scan from the primary view. The belief state may simply be stale —
         nothing has looked since the object was put there, or since the last
         scan found nothing. This is the common case and needs no extra setup.
      2. Only if that fails, the deployment's look_around, if it has one.
    """
    held = scene_state.held_object()
    for action_dict in actions:
        for name in referenced_names(action_dict):
            obj = scene_state.find(name)
            if obj is not None and (obj.position is not None or obj is held):
                continue                  # there, or in the gripper: not something to look for

            print(f"  [Recover] '{name}' not in the current scene — re-scanning...")
            deployment.look(env, scene_state)

            obj = scene_state.find(name)
            if obj is not None and obj.position is not None:
                print(f"  [Recover] Found '{name}'.")
                continue

            if deployment.look_around is not None:
                deployment.look_around(env, scene_state, name)

            obj = scene_state.find(name)
            if obj is not None and obj.position is not None:
                continue

            # Still unmatched. The planner and the vision model are separate
            # models that never agreed on a vocabulary: the planner echoes the
            # user's word ("pawn") while discover_scene invents its own label
            # ("chess_piece"), and no amount of string matching bridges two
            # words that share no characters. Say what WAS seen — otherwise the
            # only symptom is "not found in scene", which looks identical to
            # the camera having seen nothing at all.
            seen = [o for o in scene_state.all_objects() if o.position is not None]
            if not seen:
                print(f"  [Recover] Still nothing visible to match '{name}'.")
                continue

            print(f"  [Recover] '{name}' matched nothing. Scene currently holds: "
                  + ", ".join(f"'{o.name}'" for o in seen))
            # Not taken for the one object that is there: the plan names only
            # what the scene had (its schema), so a name gone is a scene that
            # changed, and the one left is another object.


def _trip_count(env) -> int:
    """How many times env's interlock has tripped."""
    return env.interlock.trip_count


def _stop_count(env) -> int:
    """How many of env's interlock trips were stops, not the sentry's pauses."""
    return env.interlock.stop_count


def _stamp_look(scene_state: SceneState, trips, stops, started: float, by_source=None) -> None:
    """The scene was looked at as of `started`, with the interlock's counts then."""
    scene_state.looked_at_trip, scene_state.looked_at_stops, scene_state.looked_at_time = trips, stops, started
    scene_state.looked_at_sources = by_source


def _by_source(env) -> dict:
    return env.interlock.trips_by_source


def moved_only_far_off(env, scene_state: SceneState) -> "float | None":
    """
    Since the last look, the arm was only paused by the sentry, and all it
    saw move was further than safety.MOVED_NEAR_M from every object and marked spot:
    how near it came, metres; inf when nothing it saw left a change (a hand
    came and went: MotionGuard forgets those once the pause clears). None
    when a look is needed — something stopped the arm (a press, a fault, the
    sentry gone blind), nothing says where it moved, or it came near something.
    """
    stops = _stop_count(env)
    if stops is None or scene_state.looked_at_stops is None or stops != scene_state.looked_at_stops:
        return None
    # Tripped by anything but the sentry seeing movement — a camera that went
    # and came back (a pause, but a blind one), a fault — and nothing can say
    # what moved meanwhile.
    now, was = _by_source(env), getattr(scene_state, "looked_at_sources", None)
    if not isinstance(env, Sentry):
        return None                         # nothing watches: nothing can say what moved
    if was is not None:
        sentry = env.sentry_sources()
        if any(n > was.get(src, 0) for src, n in now.items() if src not in sentry):
            return None
    moved = env.motion_seen_since(scene_state.looked_at_time) if scene_state.looked_at_time is not None else None
    if moved is None:
        return None
    if not moved:
        return float("inf")
    spots = [np.asarray(o.position[:2], float) for o in scene_state.objects + scene_state.targets
             if o.position is not None]
    if not spots:
        return None
    nearest = min(float(np.hypot(x - s[0], y - s[1])) - r for x, y, r in moved for s in spots)
    return nearest if nearest > safety.MOVED_NEAR_M else None


def moved_near_scene(env, scene_state: SceneState, since: float, names=None) -> "tuple[str, float] | None":
    """
    The object or marked spot nearest which the sentry saw something move
    since `since`, and how near (safety.moved_near), when anything moved near
    one — of those called `names`, when given; None otherwise, or with no
    sentry to say.
    """
    things = scene_state.objects + scene_state.targets
    if names is not None:
        things = [o for n in names if (o := scene_state.find(n)) is not None]
    hits = [(o.name, d) for o in things if o.position is not None
            and (d := safety.moved_near(env, o.position[:2], since)) is not None]
    return min(hits, key=lambda h: h[1]) if hits else None


def _moved_near_the_rest(env, scene_state: SceneState, rest: list, since: float) -> "tuple[str, float] | None":
    """
    Something moved, since `since`, near what the rest of a plan works on —
    the objects and spots it names. Holding something, only where it goes
    next counts: what moved elsewhere waits for the jaws to be empty (looking
    now would carry it off to the observe pose and back), and the place looks
    at its own spot again before coming down.
    """
    if scene_state.held_object() is not None:
        return moved_near_scene(env, scene_state, since, referenced_names(rest[0])) if rest else None
    return moved_near_scene(env, scene_state, since, sorted({n for a in rest for n in referenced_names(a)}))


def look_again_if_disturbed(env, scene_state: SceneState, deployment: Deployment) -> bool:
    """
    Look at the scene again (Deployment.look) if the interlock has tripped
    since it was last looked at: the motion guard saw something other than the
    arm move over the work surface — a hand, a piece knocked over — or the arm
    was stopped. Otherwise what the scene holds still stands: nothing but the
    arm has moved anything, and its own actions keep the scene up to date.
    True if it looked.

    Not with an object in the gripper: looking takes the arm out of the
    camera's view, and it would carry the object there and back. 2026-11-03,
    the full chess board: a pause during the f1 bishop's pick had the arm
    lift the bishop away to the park pose and back before putting it on c4.
    It looks once the gripper is empty again.
    """
    trips, looked = _trip_count(env), scene_state.looked_at_trip
    if trips is None or looked is None or trips == looked:
        return False
    held = scene_state.held_object()
    if held is not None:
        print(f"  [Scene] Something paused the arm since the last look — looking again once {held.name} is "
              "put down, not carrying it off to look.")
        return False
    far = moved_only_far_off(env, scene_state)
    if far is not None:
        # A look costs the arm's trip to its observe pose and back, and a scan:
        # half a minute between two steps of one plan (2026-11-12, the UR5).
        print("  [Scene] The sentry paused the arm, but whatever moved came and went — not looking again."
              if far == float("inf") else
              f"  [Scene] The sentry paused the arm, but nothing moved within {far * 100:.0f} cm of anything on "
              "the table — not looking again.")
        # As good as a look, as of now: what is there is what was.
        _stamp_look(scene_state, trips, _stop_count(env), time.monotonic(), _by_source(env))
        return False
    print("  [Scene] Something paused or stopped the arm since the last look — looking again.")
    deployment.look(env, scene_state)
    return True


# The step execute_plan last stopped at (its action dict), for handle_command.
LAST_FAILED_STEP: "dict | None" = None
# The last plan was pick-and-put units run to their end (_run_units): what failed has had its
# second go, and is said rather than planned again.
LAST_PLAN_SETTLED = False
# The last plan was not run at all: the safety check blocked it (nothing moved).
LAST_PLAN_BLOCKED: bool = False


def _headed_for(actions: list, step: int) -> "str | None":
    """For a pick at actions[step - 1] (1-based step): the target the plan's next place names, before any other pick."""
    if actions[step - 1].get("action") != "pick":
        return None
    for later in actions[step:]:
        if later.get("action") in ("place", "place_on"):
            nested = later.get("parameters") if isinstance(later.get("parameters"), dict) else {}
            key = "target_name" if later.get("action") == "place" else "object_name"
            return later.get(key) or nested.get(key) or None
        if later.get("action") == "pick":
            return None
    return None


# ── Saying which phase it is in ────────────────────────────────────────────────
# A command spends a minute or more thinking (the vision model, then the
# planner) before the arm moves, and from the console there was no telling
# when it would (2026-11-13, recording demos). So: THINKING while it plans,
# a bell and a banner as the arm is first commanded (RobotEnv.on_next_move:
# the moment guard() is called, whatever the action), each move as it
# starts, and DONE when the command is over.

BELL = "\a"


def _what(action: dict) -> str:
    """A step in a few words: "pick green_block_7", "place on red_block_u"."""
    params = {k: v for k, v in action.items() if k != "action"}
    if isinstance(params.get("parameters"), dict):
        params.update(params.pop("parameters"))
    name = str(action.get("action", "?"))
    obj, tgt = params.get("object_name"), params.get("target_name")
    if obj and tgt:
        return f"{name} {obj} on {tgt}"
    return " ".join(str(v) for v in (name, obj or tgt) if v)


def _when_it_moves(env, text: str) -> None:
    """Say text, with the bell, as the arm is next commanded (once)."""
    env.on_next_move(lambda: print(f"{BELL}  {text}", flush=True))


def _not_moving(env) -> None:
    env.on_next_move(None)


def _banner(text: str) -> str:
    line = "═" * min(78, max(40, len(text) + 4))
    return f"{line}\n  {text}\n  {line}"


def execute_plan(actions: list, env, scene_state: SceneState, deployment: "Deployment | None" = None) -> bool:
    """
    Validate and run a plan, in order. True if a step failed.

    Stops at the first step that fails: the rest of the plan was written
    assuming it worked — a place after a failed pick would carry nothing to the
    square, and the step after that would start from the wrong state. The
    caller re-plans from a fresh look instead.

    A stop (MotionStopped) is not a failed step: it passes straight through,
    to the REPL, which abandons the whole command.
    """
    global LAST_FAILED_STEP, LAST_PLAN_BLOCKED, LAST_PLAN_SETTLED
    LAST_PLAN_BLOCKED = LAST_PLAN_SETTLED = False
    print(f"  [VLM] -> {json.dumps(actions)}")

    _when_it_moves(env, "▶ The arm moves to look at the table again.")
    look_again_if_disturbed(env, scene_state, deployment or Deployment())
    ensure_visible(actions, scene_state, env, deployment or Deployment())
    _not_moving(env)

    filtered, violations = safety.validate_plan(actions, scene_state)
    if len(filtered) < len(actions):
        # Not a plan that ran: the rest was written assuming the removed steps
        # happen, so none of it runs, and the command counts as failed. Why, for
        # the next plan: without it the planner writes the same one again.
        print(f"  [Safety] {len(actions) - len(filtered)} action(s) blocked — not running this plan.")
        why = "; ".join(v.reason for v in violations if getattr(v, "severity", "block") == "block")
        skill_executor.report("plan", False, f"the plan was not run: {why}" if why else "the plan was not run")
        LAST_FAILED_STEP, LAST_PLAN_BLOCKED = None, True
        return True

    units = _units(actions) if deployment is not None and deployment.defer_failed else None
    if units is not None:
        return _run_units(*units, env, scene_state, deployment or Deployment())

    since = time.monotonic()                      # movement from here on is looked at between steps
    moves = ", ".join(_what(a) for a in actions)
    _when_it_moves(env, _banner(f"▶ MOVING NOW — {len(actions)} step(s): {moves}"))
    for i, action_dict in enumerate(actions, 1):
        if i > 1 and (near := _moved_near_the_rest(env, scene_state, actions[i - 1:], since)) is not None:
            # A hand, or something knocked over (a stack the arm built): the rest
            # was written for the table as it was. Stopped, the caller looks again
            # and plans the goal afresh from what is there now.
            why = (f"something moved {near[1] * 100:.0f} cm from {near[0]} while the arm worked — stopped "
                   "to look again before the rest")
            print(f"  [Plan] {why[0].upper()}{why[1:]} ({len(actions) - i + 1} step(s) left).")
            skill_executor.report("plan", False, why)
            LAST_FAILED_STEP = None
            return True
        if i > 1 and scene_state.held_object() is None:
            since = time.monotonic()                  # all of it looked at, the jaws empty: from here on
        print(f"  ▶ Step {i}/{len(actions)}: {_what(action_dict)}")
        if not env.is_viewer_running():
            print("  Connection closed — stopping sequence.")
            return True
        trips, stops, started, sources = _trip_count(env), _stop_count(env), time.monotonic(), _by_source(env)
        scene_state.headed_for = _headed_for(actions, i)
        try:
            ok = dispatch(action_dict, env, scene_state)
        except Exception as exc:
            print(f"  [Error] {exc}")
            ok = False
        finally:
            scene_state.headed_for = None
        if action_dict.get("action") == "observe" and ok is not False:
            _stamp_look(scene_state, trips, stops, started, sources)    # a scan the plan asked for
        if ok is False:
            LAST_FAILED_STEP = action_dict
            if i < len(actions):
                print(f"  [Plan] Step {i} failed — not running the remaining "
                      f"{len(actions) - i} step(s).")
            return True
    _not_moving(env)                               # a plan that never moved: no banner left for later
    return False


_PICKS = ("pick", "pick_by_rim")
_PUTS = ("place", "place_in", "place_on", "place_beside")
_LEAD = ("home", "observe")


def _units(actions: list) -> "tuple[list, list] | None":
    """
    (lead, units) when the plan is pick-and-put units — each a pick then a put,
    after any home/observe first — at least two of them; else None.
    """
    lead = []
    rest = list(actions)
    while rest and rest[0].get("action") in _LEAD:
        lead.append(rest.pop(0))
    if len(rest) < 4 or len(rest) % 2:
        return None
    pairs = [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]
    if not all(a.get("action") in _PICKS and b.get("action") in _PUTS for a, b in pairs):
        return None
    return lead, pairs


def _object_of(action: dict) -> "str | None":
    return action.get("object_name") or (action.get("parameters") or {}).get("object_name")


def _with_object(action: dict, name: str) -> dict:
    """action, its object_name changed to name (where it is kept: at the top, or in its parameters)."""
    if "parameters" in action and "object_name" in (action["parameters"] or {}):
        return {**action, "parameters": {**action["parameters"], "object_name": name}}
    return {**action, "object_name": name}


def _run_units(lead: list, units: list, env, scene_state: SceneState, deployment: Deployment) -> bool:
    """
    execute_plan for pick-and-put units (Deployment.defer_failed): each unit in
    turn; one whose pick fails set aside, the rest going on; then a fresh look,
    and each set aside tried once more, found again by who it is (its track),
    not its name — the vision model's names change from scan to scan; no
    longer tracked (pushed, and found afresh), the nearest of its name to
    where it was, within Deployment.refind_near_m. True when
    any was not done; what and why are said, and reported for the planner.
    2026-11-18: "put the four things in the tub" stopped at the first failed
    pick and was planned again from scratch, 45-60 s a time, and once the new
    plan picked a figure already in the tub.
    """
    global LAST_FAILED_STEP, LAST_PLAN_SETTLED
    moves = ", ".join(_what(a) for a in lead) + (", " if lead else "") + \
        f"{len(units)} x pick and put ({', '.join(str(_object_of(p)) for p, _ in units)})"
    _when_it_moves(env, _banner(f"▶ MOVING NOW — {moves}"))
    for action in lead:
        print(f"  ▶ {_what(action)}")
        if dispatch(action, env, scene_state) is False:
            LAST_FAILED_STEP = action
            return True
    stop = object()

    def run(pick: dict, put: dict, tag: str):
        """None: done. stop: holding what it could not put (nothing more runs). Else why it was not done."""
        global LAST_FAILED_STEP
        print(f"  ▶ {tag}: {_what(pick)}, then {_what(put)}")
        scene_state.headed_for = _headed_for([pick, put], 1)
        try:
            ok = dispatch(pick, env, scene_state)
        except MotionStopped:
            raise
        except Exception as exc:                         # noqa: BLE001 — one unit's error is that unit's
            print(f"  [Error] {exc}")
            ok = False
        finally:
            scene_state.headed_for = None
        if ok is False and scene_state.held_object() is None:
            said = skill_executor.last_report()
            return (said or {}).get("message") or "the pick failed"
        if dispatch(put, env, scene_state) is False:
            LAST_FAILED_STEP = put
            return stop
        return None

    def track_of(name):
        obj = scene_state.find(name, among="objects") if name else None
        return getattr(obj, "track_id", None)

    def where_is(name):
        obj = scene_state.find(name, among="objects") if name else None
        pos = getattr(obj, "position", None)
        return None if pos is None else np.asarray(pos, float)[:2].copy()

    done, aside = [], []
    for k, (pick, put) in enumerate(units, 1):
        name = _object_of(pick)
        track, where = track_of(name), where_is(name)
        why = run(pick, put, f"{k}/{len(units)}")
        if why is stop:
            return True
        if why:
            print(f"  [Plan] {name}: {why} — set aside, tried again at the end.")
            after = where_is(name)
            aside.append((pick, put, track, name, after if after is not None else where))
        else:
            done.append(str(name))
    left = []
    if aside:
        print(f"  [Plan] {len(aside)} set aside — looking at the table again first.")
        try:
            if deployment.observe_pose is not None:
                deployment.observe_pose(env)
            deployment.scan_here(env, scene_state)
        except MotionStopped:
            raise
        except Exception as exc:                          # noqa: BLE001
            print(f"  [Plan] The look failed: {exc}")
        for pick, put, track, name, where in aside:
            now = next((o for o in scene_state.objects if track is not None and o.track_id == track
                        and o.position is not None), None)
            if now is None and deployment.refind_near_m is not None and where is not None:
                now = _same_name_near(scene_state, name, where, deployment.refind_near_m, done)
                if now is not None and track is not None and scene_state.memory is not None:
                    # Its old track is not something else to keep clear of beside it: 2026-11-20 09:15, the
                    # plug found again 5 cm off, and refused for "white_plug (missed by the last scan; last
                    # seen there)" — itself, where it had been.
                    scene_state.memory.superseded(track, by=str(now.track_id),
                                                  why=f"found again by its name near where {name} was")
            if now is None:
                left.append((name, "not on the table where the scan can find it now"))
                continue
            if now.name != name:
                print(f"  [Plan] {name} is called {now.name} now.")
            why = run(_with_object(pick, now.name), put, "again")
            if why is stop:
                return True
            if why:
                left.append((str(now.name), why))
            else:
                done.append(str(now.name))
    if deployment.task_more is not None and done:
        # More of what the task names, now that the rest is away: looked for once more, and done.
        try:
            more = deployment.task_more(env, scene_state, list(done)) or []
        except MotionStopped:
            raise
        except Exception as exc:                          # noqa: BLE001 — what was done stands
            print(f"  [Plan] Could not look for more: {exc}")
            more = []
        pairs = [(more[i], more[i + 1]) for i in range(0, len(more) - 1, 2)]
        if pairs:
            print(f"  [Plan] {len(pairs)} more on the table: " + ", ".join(str(_object_of(p)) for p, _ in pairs) + ".")
        for pick, put in pairs:
            name = _object_of(pick)
            why = run(pick, put, "more")
            if why is stop:
                return True
            if why:
                left.append((str(name), why))
            else:
                done.append(str(name))
    LAST_PLAN_SETTLED = True
    _not_moving(env)
    said = (f"done: {', '.join(done)}" if done else "none done") + \
        (f"; not done: {'; '.join(f'{n} ({w})' for n, w in left)}" if left else "")
    print(f"  [Plan] {said[0].upper()}{said[1:]}.")
    if left:
        skill_executor.report("plan", False, said)
        LAST_FAILED_STEP = None
    return bool(left)


def _base_name(name) -> str:
    """A scene name less the _objN a scan adds to tell same-named objects apart."""
    return re.sub(r"_obj\d+$", "", str(name or ""))


def _same_name_near(scene_state: SceneState, name, where, within_m: float, done=()):
    """
    The object of name's own name (less any _objN) nearest `where` (x, y),
    within within_m, and not one of those done; said. None: none there.
    """
    base, finished = _base_name(name), {str(n) for n in done}
    near = []
    for o in scene_state.objects:
        if o.position is None or _base_name(o.name) != base or str(o.name) in finished:
            continue
        d = float(np.linalg.norm(np.asarray(o.position, float)[:2] - np.asarray(where, float)[:2]))
        if d <= within_m:
            near.append((d, o))
    if not near:
        return None
    d, found = min(near, key=lambda t: t[0])
    print(f"  [Plan] {name}: no longer tracked, but {found.name} is {d * 100:.0f} cm from where it was — taken for it.")
    return found


def direct_command(line: str, env, scene_state: SceneState, deployment: "Deployment | None" = None) -> bool:
    """
    Run one action directly, bypassing perceive / decompose / plan.

    Why this exists: every other route to a skill goes through three model
    calls, so when a pick misbehaves you cannot tell a motion bug from the
    planner naming the wrong square from the scan mislabelling it. This
    collapses the pipeline to just the part under test — same dispatch(), same
    preconditions, same safety validation, no VLM.

        /pick white_pawn_d4
        /move_above white_pawn_d4 0.08
        /move_above object_name=white_pawn_d4 height=0.08
        /actions                 list every registered action and its parameters
        /objects                 what the scan believes, with world coordinates

    Positional arguments fill the action's declared parameters in declaration
    order; key=value sets one by name. Returns True if the line was handled.
    """
    if not line.startswith("/"):
        return False
    parts = line[1:].split()
    if not parts:
        print("  Usage: /<action> [args]   —   /actions lists them, /objects "
              "shows what the scan found.")
        return True
    name, args = parts[0], parts[1:]

    if name in ("actions", "help"):
        for known, info in sorted(ACTIONS.items()):
            params = ", ".join(info["parameters"]) or "—"
            print(f"  /{known:14} {params}")
        return True

    if name == "objects":
        if not scene_state.objects and not scene_state.targets:
            print("  Nothing in the world model. Run /observe.")
            return True
        print(f"  {'name':24} {'world (m)':34} {'camera px':14} reach")
        for o in list(scene_state.objects) + list(scene_state.targets):
            pos = ("held / position unknown".ljust(34) if o.position is None else
                   f"x={o.position[0]:+.4f} y={o.position[1]:+.4f} z={o.position[2]:+.4f}")
            # pixel_xy is what the scan actually saw; the world column is that
            # pixel through the camera's calibration. Printing both is what
            # separates "the scan looked at the wrong object" from "the
            # calibration is off".
            px = ("—".ljust(14) if not o.pixel_xy
                  else f"({o.pixel_xy[0]:.0f}, {o.pixel_xy[1]:.0f})".ljust(14))
            print(f"  {o.name:24} {pos} {px} {'yes' if o.reachable else 'NO'}")
        return True

    if name not in ACTIONS:
        print(f"  Unknown action '{name}'. Try /actions.")
        return True

    # Typed like the planner's JSON would be, not like shell words: skills take
    # real numbers (move_above's `height` goes straight into a coordinate), so a
    # string "0.08" would reach the arm as a TypeError deep inside the skill.
    def typed(value: str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    action: dict = {"action": name}
    declared = list(ACTIONS[name]["parameters"])
    positional = []
    for arg in args:
        if "=" in arg:
            key, _, value = arg.partition("=")
            action[key] = typed(value)
        else:
            positional.append(arg)
    for key, value in zip(declared, positional):
        action[key] = typed(value)
    if len(positional) > len(declared):
        print(f"  '{name}' takes {len(declared)} parameter(s) "
              f"({', '.join(declared) or 'none'}) — ignoring the extras.")

    execute_plan([action], env, scene_state, deployment)
    return True


def handle_command(user_input: str, env, scene_state: SceneState, history: list,
                   deployment: Deployment) -> None:
    """One command, from the user's words to the arm going home afterwards."""
    global LAST_FAILED_STEP
    # Direct dispatch, no VLM. Checked before the planner so a debugging
    # command never costs three model calls.
    if direct_command(user_input, env, scene_state, deployment):
        return

    # Note: this is a rough, free-text description from wherever the arm
    # currently is — it does NOT move to the observe pose first. Moving
    # the arm for every single command (including pure utility queries
    # like "what can you do?") would be both slow and a bad idea on real
    # hardware. Only a scan — by the "observe" action, at startup, and in
    # failure recovery below — needs the observe pose.
    # Before the plan, so it names what is there now.
    thinking = time.monotonic()
    print("  ⏳ THINKING — the arm stays still until it says ▶ MOVING NOW (usually under 2 minutes).", flush=True)
    _when_it_moves(env, "▶ The arm moves out of the camera's way to look at the table.")
    look_again_if_disturbed(env, scene_state, deployment)
    _not_moving(env)
    reused = None
    from_memory = False
    if deployment.task_plan is not None:
        try:
            reused = deployment.task_plan(user_input, env, scene_state)
        except Exception as exc:                            # noqa: BLE001 — then as any other command
            print(f"  [Plan] Could not lay out the task: {exc}")
            reused = None
        if reused:
            print(f"  [Plan] Its own task, as planned for it: {len(reused)} step(s), no planning.")
    if not reused and deployment.remembered_plan is not None:
        try:
            reused = deployment.remembered_plan(user_input, scene_state)
        except Exception as exc:                            # noqa: BLE001 — planned as usual
            print(f"  [Plan] Could not use a plan from before: {exc}")
            reused = None
        from_memory = bool(reused)
    if not reused and deployment.quick_plan is not None:
        try:
            reused = deployment.quick_plan(user_input, env, scene_state)
        except Exception as exc:                            # noqa: BLE001 — planned as usual
            print(f"  [Plan] Could not plan it from the words: {exc}")
            reused = None
        if reused:
            print(f"  [Plan] Planned from the words themselves: {len(reused)} step(s), no planning.")
    elif from_memory and reused:
        print(f"  [Plan] Planned for these words before, and it fits what is on the table: "
              f"{len(reused)} step(s), no planning.")
    if not reused and not deployment.planner:
        tasks = ", ".join(f'"{t}"' for t in deployment.task_names)
        print(f"  [Plan] This deployment plans nothing itself: \"{user_input}\" is none of its tasks"
              + (f" ({tasks})." if tasks else ".") + " Nothing moved.")
        _not_moving(env)
        return
    if reused:
        sub_goals, initial_scene_desc = [user_input], scene_state.summary()
    elif deployment.task_planner:
        print("  [VLM] Perceiving + decomposing...")
        pre_frames = vlm.render_all_cameras(env)
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_sub  = ex.submit(vlm.decompose_task, user_input,
                               [o.name for o in scene_state.objects if o.name])
            f_desc = ex.submit(vlm.perceive_from_frames, pre_frames)
        sub_goals          = f_sub.result()
        initial_scene_desc = f_desc.result() or scene_state.summary()
    else:
        print("  [VLM] Perceiving...")
        sub_goals          = [user_input]
        initial_scene_desc = vlm.perceive(env) or scene_state.summary()

    if len(sub_goals) > 1:
        print(f"  [Planner] Decomposed into {len(sub_goals)} sub-goals:")
        for idx, sg in enumerate(sub_goals, 1):
            print(f"    {idx}. {sg}")

    max_steps = deployment.max_auto_steps
    total_dispatched = 0
    any_error = False

    for goal_i, sub_goal in enumerate(sub_goals):
        current_command: str = sub_goal
        auto_step = 0
        cached_scene_desc = initial_scene_desc
        failure_context: str | None = None
        last_failed: "list | None" = None      # the plan that failed last, to tell a repeat
        refused: "tuple | None" = None         # the step refused last (_step_key), and why
        retried: "set[tuple]" = set()          # refused steps that have had their one more go

        stuck = False                          # given up on: the command's later sub-goals do not run
        while current_command and auto_step < max_steps:
            if not env.is_viewer_running():
                break
            if auto_step > 0:
                print(f"\n  [Auto {auto_step}/{max_steps}] Continuing: \"{current_command}\"")
                print("  ⏳ THINKING again — the arm stays still until it says ▶ MOVING NOW.", flush=True)
                thinking = time.monotonic()

            continue_with: "str | None"
            if reused and auto_step == 0 and goal_i == 0:
                actions, continue_with, reused = list(reused), "", None
            else:
                print("  [VLM] Planning...")
                actions, continue_with = vlm.plan(
                    current_command, cached_scene_desc, scene_state,
                    history=history, failure_context=failure_context,
                )
                if (actions and auto_step == 0 and failure_context is None and len(sub_goals) == 1
                        and deployment.plan_made is not None):
                    try:
                        deployment.plan_made(user_input, actions)
                    except Exception as exc:                # noqa: BLE001 — not kept, no harm
                        print(f"  [Plan] Could not keep the plan: {exc}")
            failure_context = None

            if not actions:
                if auto_step == 0:
                    print("  [VLM] No actions planned.")
                elif last_failed is not None:
                    print("  [Recovery] The planner has nothing more to try — returning to manual control.")
                    stuck = True
                break
            blocked = _asks_again(actions, refused, retried)
            if blocked:
                print(f"  [Recovery] {blocked} — returning to manual control.")
                any_error = stuck = True
                break
            if refused is not None and any(_step_key(a) == refused[0] for a in actions):
                retried.add(refused[0])

            if len(sub_goals) > 1:
                continue_with = ""
            if continue_with:
                print(f"  [VLM] Will follow up with: \"{continue_with}\"")

            total_dispatched += len(actions)
            skill_executor.clear_report()
            LAST_FAILED_STEP = None
            print(f"  ✓ Planned in {time.monotonic() - thinking:.0f} s.", flush=True)
            flight_recorder.RECORDER.timed("planning", time.monotonic() - thinking)
            flight_recorder.event("controller", "planned", seconds=round(time.monotonic() - thinking, 2),
                                  steps=[a.get("action") for a in actions])
            had_error = execute_plan(actions, env, scene_state, deployment)
            any_error = any_error or had_error
            said = skill_executor.last_report()
            refused = ((_step_key(LAST_FAILED_STEP), said["message"])
                       if had_error and LAST_FAILED_STEP is not None and said and said["refused"] else None)
            if had_error and actions == last_failed:
                # Planned again from the same failure, the same plan: a third
                # time would only fail the same way.
                print("  [Recovery] The same plan failed again — returning to manual control.")
                stuck = True
                break
            last_failed = actions if had_error else None
            if had_error and LAST_PLAN_SETTLED:
                # Every unit tried, those that failed twice: what is left was said, not planned again.
                stuck = True
                break

            if had_error and not deployment.planner:
                # Nothing to plan it again with: what failed is said (above), and the arm goes home.
                print("  [Plan] A step failed, and this deployment plans nothing itself: not planned again.")
                stuck = True
                break
            if had_error:
                # A step that failed: the planner's follow-up was written for it
                # having worked, so the goal is planned again from what is there.
                # A plan the safety check would not run moved nothing: no look.
                continue_with = ""
                fresh_desc = None
                try:
                    if "observe" not in [a.get("action") for a in actions] and not LAST_PLAN_BLOCKED:
                        print("  [Recovery] Skill failed — perceiving current state...")
                        if deployment.observe_pose is not None:
                            deployment.observe_pose(env)
                        fresh_desc = vlm.perceive(env)
                        deployment.scan_here(env, scene_state)
                        cached_scene_desc = fresh_desc or scene_state.summary()
                except Exception as exc:
                    print(f"  [Recovery] Perception failed: {exc}")
                if auto_step + 1 < max_steps:
                    spent = refused is not None and refused[0] in retried
                    failure_context = (_why_it_failed()
                                       + ("It has had its one more go and will not run again "
                                          "in this command.\n" if spent else "")
                                       + (f"Camera currently shows: {fresh_desc}" if fresh_desc else ""))
                    continue_with = sub_goal
                    print(f"  [Recovery] Scheduling recovery re-plan for: \"{sub_goal}\"")

            auto_step += 1
            current_command = continue_with or ""

        if current_command and auto_step >= max_steps:
            print(f"  [Auto] Reached {max_steps}-step limit — returning to manual control.")
            any_error = stuck = True
        if stuck and goal_i < len(sub_goals) - 1:
            print(f"  [Plan] Not going on to the rest of the command ({len(sub_goals) - goal_i - 1} sub-goal(s) left).")
            break

    if total_dispatched > 0 and deployment.home_after_command:
        _when_it_moves(env, "▶ The arm goes home.")
        deployment.go_home(env)
        _not_moving(env)

    outcome = (
        "succeeded" if (total_dispatched > 0 and not any_error)
        else "failed" if any_error
        else "no action taken"
    )
    history.append({"command": user_input, "outcome": outcome})
    if len(history) > deployment.max_history:
        history.pop(0)
    if total_dispatched > 0:
        print("  " + _banner("■ DONE — the arm has stopped." if outcome == "succeeded"
                             else "■ FINISHED, not all of it done — the arm has stopped (above: why)."), flush=True)


def _step_key(action: dict) -> tuple:
    """An action as (name, its parameters), however the planner nested them."""
    params = {k: v for k, v in action.items() if k != "action"}
    if isinstance(params.get("parameters"), dict):
        params.update(params.pop("parameters"))
    return action.get("action"), tuple(sorted((k, str(v)) for k, v in params.items()))


def _asks_again(actions: list, refused: "tuple | None", retried: set) -> str:
    """
    Why a plan may not run because it asks again for the step just refused
    ("" if it may).

    Refused before it moved, the step would be refused the same way again,
    unless the plan first does something that can change the answer: home
    (the SO-101's reach is searched from where the arm starts) or moving what
    was in the way. Then it gets one more go — one, so a spot that is simply
    taken costs a single wasted home, not eight. Refused for room (another
    object leaves the jaws too little), home or a look first changes nothing:
    the UR5 tabletop, 2026-11-13, went home and asked for the same place on a
    block's shadow, outlined as a block.
    """
    if refused is None:
        return ""
    key, why = refused
    for i, action in enumerate(actions):
        if _step_key(action) != key:
            continue
        if key in retried:
            return f"The plan asks again for what was refused twice ({why})"
        if i == 0:
            return f"The plan asks again for what was refused ({why}), with nothing done first"
        before = [a.get("action") for a in actions[:i]]
        if _FOR_ROOM.search(why) and all(a in _NO_ROOM_MADE for a in before):
            return (f"The plan asks again for what was refused ({why}), after only {', '.join(before)}: "
                    "that does not make room")
        return ""
    return ""


# A refusal for room: parallel_jaw's reach and pick say "... but <object> leaves N mm".
_FOR_ROOM = re.compile(r"\bleaves -?\d+ mm\b")
_NO_ROOM_MADE = {"home", "observe"}


def _why_it_failed() -> str:
    """The failed step's own reason (skill_executor.report), for the planner's next try; "" if none was given."""
    said = skill_executor.last_report()
    if not said or said["ok"]:
        return ""
    how = "was refused without touching anything" if said["refused"] else "failed"
    return f"The step '{said['skill']}' {how}: {said['message']}.\n"


# ── The session ────────────────────────────────────────────────────────────────

def _drop_typed_ahead() -> None:
    """
    What was typed while a command ran, dropped before the next prompt: not
    taken as the next command. An Enter pressed while the arm moved came back
    as an empty line, and the prompt again on the same line ("You: You:",
    2026-11-13); a whole line typed then would run unseen.
    """
    import sys

    try:
        if not sys.stdin.isatty():
            return
    except (AttributeError, ValueError):
        return
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except ImportError:                      # Windows
        import msvcrt
        while msvcrt.kbhit():               # type: ignore[attr-defined]
            msvcrt.getwch()                  # type: ignore[attr-defined]
    except (OSError, ValueError):
        pass


def repl(env, scene_state: SceneState, deployment: Deployment) -> None:
    """Read commands until the user quits or the connection closes."""
    print()
    print("=" * 60)
    print(f"  {deployment.name} — natural-language controller")
    print("=" * 60)
    if not deployment.planner:
        print("  This deployment runs its own tasks, planned in advance — say one:")
        for task in deployment.task_names or ("(none set: its config's TASKS)",):
            print(f"    {task}")
        print()
    elif deployment.examples:
        print("  Talk to the robot in plain English, for example:")
        for example in deployment.examples:
            print(f"    {example}")
        print()
    print("  Useful queries:")
    print("    what can you do?         — list all available actions")
    print("    what objects are there?  — list detected objects and positions")
    print("    look at the scene        — re-scan with camera")
    print()
    print("  Debugging — runs one action directly, no VLM in the loop:")
    print("    /actions                 — every registered action + parameters")
    print("    /objects                 — world model, with coordinates")
    print("    /<action> <args>         — e.g. /pick <object name>")
    print()
    print("  To exit: type  quit / exit / q  or press Ctrl-C")
    print("=" * 60)
    print()

    history: list[dict] = []
    while True:
        if not env.is_viewer_running():
            print("Connection closed — exiting.")
            break
        try:
            _drop_typed_ahead()
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        flight_recorder.event("user", "command", text=user_input)
        if env.interlock.tripped():
            print(f"  [Stop] Still stopped: {env.interlock.reason}. Nothing will move until it clears.")
        began, outcome = time.monotonic(), "done"
        try:
            handle_command(user_input, env, scene_state, history, deployment)
            if LAST_FAILED_STEP is not None:
                outcome = "failed"
        except MotionStopped as stop:
            outcome = "stopped"
            flight_recorder.event("controller", "stopped", command=user_input, reason=str(stop.reason))
            print(f"\n  [Stop] {stop.reason}")
            print("  The arm is holding where it stopped; the command is abandoned. Nothing moves "
                  "again until the cause clears and you give the next command.")
            _remember(history, user_input, f"stopped: {stop.reason}", deployment)
        except KeyboardInterrupt:
            # Ctrl-C in the middle of a command: that command is over, not the
            # session — the arm held where it is, rather than the session's end
            # sending it home with whatever it holds. At the prompt, Ctrl-C quits.
            outcome = "interrupted"
            _hold_quietly(env)
            print("\n  [Stop] Ctrl-C — the command is abandoned and the arm is holding where it is. "
                  "Ctrl-C again at the prompt, or quit, to leave.")
            flight_recorder.event("user", "interrupt", text=user_input)
            _remember(history, user_input, "stopped: Ctrl-C", deployment)
        except Exception as exc:              # noqa: BLE001 — one command's error is not the session's end
            outcome = "error"
            _hold_quietly(env)
            print(f"\n  [Error] The command stopped on an error — {type(exc).__name__}: {exc}. "
                  "The arm is holding where it is; the next command can carry on.")
            flight_recorder.RECORDER.failure(f"the command '{user_input}' stopped on an error — "
                                             f"{type(exc).__name__}: {exc}", exc=exc, command=user_input)
            _remember(history, user_input, f"error: {exc}", deployment)
        finally:
            seconds = time.monotonic() - began
            flight_recorder.RECORDER.timed("command", seconds, failed=outcome != "done")
            flight_recorder.event("user", "command_end", text=user_input, outcome=outcome, seconds=round(seconds, 2))


def _hold_quietly(env) -> None:
    """The arm held where it is, if it can be; a hold that fails is said, not raised."""
    try:
        env.hold()
    except Exception as exc:                  # noqa: BLE001 — after an error already
        print(f"  [Stop] Could not hold the arm: {exc}")


def _remember(history: list, command: str, outcome: str, deployment: Deployment) -> None:
    history.append({"command": command, "outcome": outcome})
    if len(history) > deployment.max_history:
        history.pop(0)


def run(env, scene_state: SceneState, deployment: Deployment) -> None:
    """
    The session on a connected arm: the deployment's watchers started, home,
    a first look at the scene, the prompt; home again at the end, watchers
    stopped. Connecting and disconnecting the arm is the caller's. All of it
    on the flight recorder (flight_recorder.py): what happened, and when
    something goes wrong, everything that led up to it.
    """
    from mrcf import privacy

    cameras = env.list_cameras()
    flight_recorder.RECORDER.start(session={"deployment_name": deployment.name, "cameras_open": cameras,
                                            "faces_covered": privacy.has_face_model()},
                                   console=True, crashes=True)
    # Kept frames: what a calibrated camera sees of the work volume, faces covered (privacy.py).
    flight_recorder.RECORDER.watch_cameras(env, cameras, keep=privacy.frame_guard(env, safety.bounds()))
    flight_recorder.RECORDER.watch_sensors(env)
    watchers = [start(env) for start in deployment.watchers]
    try:
        try:
            print("Homing...")
            env.go_home()
            print("Looking at the scene...")
            looked = deployment.look(env, scene_state)
            print(scene_state.summary())
        except MotionStopped as stop:
            looked = False
            flight_recorder.event("controller", "stopped", during="start", reason=str(stop.reason))
            print(f"\n  [Stop] {stop.reason} — not scanning; say 'observe' once it has cleared.")
        if looked and deployment.on_start is not None:
            try:
                deployment.on_start(env, scene_state)
            except Exception as exc:          # noqa: BLE001 — a deployment's start is not the session's end
                print(f"  [Start] {deployment.name}'s start failed — {type(exc).__name__}: {exc}")
                flight_recorder.RECORDER.failure(f"{deployment.name}'s start failed — {type(exc).__name__}: {exc}",
                                                 exc=exc)
        repl(env, scene_state, deployment)
    finally:
        print("Returning home...")
        try:
            if not env.go_home():
                print("  [Shutdown] Could not go home from here (above: why) — jog it home on the pendant.")
                flight_recorder.event("controller", "not_home", during="shutdown")
        except MotionStopped as stop:
            print(f"  [Shutdown] Not going home: {stop.reason}.")
            flight_recorder.event("controller", "stopped", during="shutdown", reason=str(stop.reason))
        except Exception as exc:
            print(f"  [Shutdown] go_home failed: {exc}")
            flight_recorder.RECORDER.failure(f"going home at the end failed — {type(exc).__name__}: {exc}", exc=exc)
        for w in watchers:
            w.stop()
        flight_recorder.RECORDER.stop()


__all__ = ["Deployment", "run", "repl", "handle_command", "execute_plan", "direct_command", "ensure_visible",
           "referenced_names"]
