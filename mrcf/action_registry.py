"""
action_registry.py — the actions a planner may use, the conditions that guard
them, and dispatch: one planned step, checked against the world model, run.

    @register_action("nudge", description="Push an object a short distance.",
                     parameters={"object_name": "Exact name of the object."},
                     conditions={"pre": ["object_exists", "nothing_carried"]})
    def skill_nudge(env, scene_state, object_name, **_): ...

    dispatch({"action": "nudge", "object_name": "red_cube"}, env, scene_state)

One registry per process (ACTIONS, CONDITIONS): what the planner is offered,
and what dispatch runs. Robot-agnostic; manipulation.py registers the standard
manipulation actions, a deployment its own.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

import inspect
import time

from mrcf.robot_env import RobotEnv
from mrcf.scene_state import SceneState


# ── Action registry ────────────────────────────────────────────────────────────

ACTIONS: dict = {}   # name -> {description, parameters, conditions, always, fn}

# Condition name -> predicate(params, scene_state) -> (ok, reason_when_not_ok).
# Deployments extend this with register_condition() for guards the framework
# cannot know about ("board_visible", "gripper_calibrated", …).
CONDITIONS: dict = {}


def register_condition(name: str):
    """
    Register a condition usable in @register_action's conditions={"pre"/"post"}.

    The framework ships the handful of conditions it can evaluate from a
    SceneState alone. Anything task-specific belongs to the deployment, so this
    is the hook::

        @register_condition("board_visible")
        def _board_visible(params, scene_state):
            return bool(scene_state.targets), "no board squares in view"

    The predicate returns (ok, reason). It must not move the robot — conditions
    are evaluated before dispatch and after it, purely against the world model.
    """
    def decorator(fn):
        CONDITIONS[name] = fn
        return fn
    return decorator


@register_condition("object_exists")
def _cond_object_exists(params, scene_state):
    obj_name = params.get("object_name")
    if obj_name and scene_state.find(obj_name) is None:
        return False, f"'{obj_name}' not in scene — try 'observe' first"
    return True, ""


@register_condition("object_not_carried")
def _cond_object_not_carried(params, scene_state):
    obj_name = params.get("object_name")
    if obj_name:
        obj = scene_state.find(obj_name)
        if obj is not None and obj.position is None:
            return False, f"'{obj_name}' is already being carried"
    return True, ""


@register_condition("something_carried")
def _cond_something_carried(_params, scene_state):
    if scene_state.held_object() is None:
        return False, "not carrying any object"
    return True, ""


@register_condition("nothing_carried")
def _cond_nothing_carried(_params, scene_state):
    held = scene_state.held_object()
    if held is not None:
        return False, f"already carrying {held.name}"
    return True, ""


@register_condition("target_exists")
def _cond_target_exists(params, scene_state):
    tgt_name = params.get("target_name")
    if tgt_name and scene_state.find(tgt_name) is None:
        return False, f"target '{tgt_name}' not in scene — try 'observe' first"
    return True, ""


def register_action(
    name: str,
    description: str,
    parameters: dict | None = None,
    conditions: dict | None = None,
    always: bool = False,
):
    """
    Decorator that registers a robot skill into ACTIONS.

    The decorated function receives at minimum:
        env, scene_state
    plus whatever named parameters the action declares.
    Use **_ in skill signatures to swallow any extra kwargs from dispatch.

    always     — if True, this action is included in the planner schema even
                 when the scene has no detected objects.  Use for utility and
                 meta actions (observe, calculate, answer, …) that are always
                 relevant.  Motion skills that only apply when there are objects
                 to act on should leave always=False (the default).

    conditions — optional dict with "pre" and/or "post" lists of condition names.
        Pre-conditions are checked before dispatch and block the action when
        they fail; post-conditions are checked after it and only warn (the
        skill has already reported its own outcome). Built in:
            "object_exists"      — params["object_name"] is in the scene
            "object_not_carried" — params["object_name"] is not being carried
            "something_carried"  — the robot is carrying an object
            "nothing_carried"    — the robot is carrying nothing
            "target_exists"      — params["target_name"] is in the scene
        Register more with register_condition().

    The skill should return True or False for success, so dispatch() — and the
    controller's failure handling and command history — can tell a failed
    action from one that worked. None (no return) counts as "not reported".
    """
    def decorator(fn):
        ACTIONS[name] = {
            "description": description,
            "parameters":  parameters or {},
            "conditions":  conditions or {},
            "always":      always,
            "fn":          fn,
        }
        return fn
    return decorator


def action_schema() -> dict:
    """Return ACTIONS without internal keys — safe to serialise for the VLM."""
    return {
        n: {k: v for k, v in info.items() if k not in ("fn", "conditions", "always")}
        for n, info in ACTIONS.items()
    }


def context_action_schema(scene_state: "SceneState") -> dict:
    """Return a scene-filtered subset of the action schema for the planner.

    Sending all actions on every call wastes tokens and increases the chance
    of a small model picking an irrelevant tool.  Rules:
      - Actions registered with always=True are always included (utility, meta, observe, …).
      - All other actions are included only when there are interactable objects in the scene.
    This is fully driven by the always flag set in each @register_action call —
    no hardcoded action names appear here.
    """
    has_objects = bool(scene_state.objects)
    result = {}
    for name, info in ACTIONS.items():
        if info.get("always", False) or has_objects:
            result[name] = {k: v for k, v in info.items() if k not in ("fn", "conditions", "always")}
    return result


def scene_names(scene_state: "SceneState") -> list[str]:
    """
    The names a planner may give an "..._name" parameter: the objects it may act
    on and the places it may go, as the scene has them now. An object the scene
    keeps only as an obstacle (interactable=False) is not one of them.
    """
    names = [o.name for o in scene_state.objects if o.interactable] + [t.name for t in scene_state.targets]
    return list(dict.fromkeys(names))


def plan_schema(scene_state: "SceneState") -> dict:
    """
    The JSON schema a planner's reply is held to (vlm.plan passes it to the
    model as its output format): {"actions": [...], "continue_with": "..."},
    each action one of context_action_schema's, by name, and every parameter
    whose name ends in "_name" one of scene_names(). The model cannot then name
    an action that is not registered, or an object or place that is not in the
    scene; it still chooses which. An action needing a name when the scene has
    none is left out.
    """
    names = scene_names(scene_state)
    choices = []
    for name, info in context_action_schema(scene_state).items():
        params = info.get("parameters") or {}
        named = [p for p in params if p.endswith("_name")]
        if named and not names:
            continue
        properties = {p: ({"enum": names} if p in named else {}) for p in params}
        choices.append({
            "type": "object",
            "properties": {"action": {"const": name},
                           "parameters": {"type": "object", "properties": properties, "required": named}},
            "required": ["action", "parameters"],
        })
    return {
        "type": "object",
        "properties": {"actions": {"type": "array", "items": {"anyOf": choices}},
                       "continue_with": {"type": "string"}},
        "required": ["actions", "continue_with"],
    }


def _check_conditions(action_name: str, params: dict, scene_state: SceneState,
                      phase: str) -> tuple[bool, str]:
    """Evaluate the "pre" or "post" condition list of an action."""
    info = ACTIONS.get(action_name, {})
    for condition in info.get("conditions", {}).get(phase, []):
        predicate = CONDITIONS.get(condition)
        if predicate is None:
            # A typo, or a condition the deployment forgot to register, must
            # not look like a guard that passed. Say so instead of quietly
            # allowing it.
            print(f"  [Condition] '{action_name}' declares unknown {phase}-condition "
                  f"'{condition}' — not checked. Register it with "
                  f"vlm.register_condition(). Known: {', '.join(sorted(CONDITIONS))}")
            continue
        ok, reason = predicate(params, scene_state)
        if not ok:
            return False, reason
    return True, ""


def check_preconditions(
    action_name: str,
    params: dict,
    scene_state: SceneState,
) -> tuple[bool, str]:
    """Validate pre-conditions for an action against the current world model.

    Returns (ok, reason) where reason is a human-readable message on failure.
    """
    return _check_conditions(action_name, params, scene_state, "pre")


def check_postconditions(
    action_name: str,
    params: dict,
    scene_state: SceneState,
) -> tuple[bool, str]:
    """
    Validate post-conditions after an action ran.

    Advisory, not a failure path: the skill has already reported its own
    success or failure, and the world model may simply be stale. What this
    catches is the disagreement — a pick that returned success while nothing
    is marked carried — which is otherwise invisible until a later skill
    fails for an unrelated-looking reason.
    """
    return _check_conditions(action_name, params, scene_state, "post")


# ── Action dispatcher ──────────────────────────────────────────────────────────

def _step_on_record(name: str, params: dict, began: float, result, why: "str | None", refused: bool = False) -> None:
    """A step's end on the flight recorder: what it came to, why not, and how long it took."""
    from mrcf import flight_recorder

    seconds = time.monotonic() - began
    flight_recorder.RECORDER.timed(f"step:{name}", seconds, failed=result is False)
    extra: "dict[str, object]" = {}
    if why:
        extra["why"] = why
    if refused:
        extra["refused"] = True
    flight_recorder.event("step", "end", None, action=name, params=params, ok=result, seconds=round(seconds, 3),
                          **extra)


def dispatch(action_dict: dict, env: RobotEnv, scene_state: SceneState) -> "bool | None":
    """
    Look up the registered skill by action name, resolve missing parameters
    interactively, then call the skill.

    Returns what the skill returned — True or False for an outcome it
    reported, None if it reported none — or False when the action could not
    run at all (unknown, blocked by a pre-condition, cancelled, raised).
    """
    from mrcf import skill_executor        # here: it runs the policies, which this sits below

    name = action_dict.get("action", "")
    # What this step came to, and nothing older: the controller's next plan is
    # told why it failed (skill_executor.last_report), and a step that failed
    # before saying anything must not leave the last step's "ok" standing.
    skill_executor.clear_report()
    if name not in ACTIONS:
        print(f"  [Error] Unknown action '{name}'. Known: {list(ACTIONS)}")
        skill_executor.report(name, False, f"there is no action called '{name}'", refused=True)
        return False

    info = ACTIONS[name]
    fn   = info["fn"]

    # Flatten nested {"parameters": {...}} that the VLM sometimes emits
    params = {k: v for k, v in action_dict.items() if k != "action"}
    if "parameters" in params and isinstance(params.get("parameters"), dict):
        nested = params.pop("parameters")
        params.update(nested)

    # Validate pre-conditions against the current world model before motion.
    ok, reason = check_preconditions(name, params, scene_state)
    if not ok:
        print(f"  [PreCondition] {name}: {reason}")
        skill_executor.report(name, False, reason, refused=True)
        return False

    # Prompt interactively for parameters that are missing AND have no function default.
    # Parameters with defaults (e.g. height=None) are left to the function
    # itself so an automated VLM call that omits them never blocks on input().
    sig = inspect.signature(fn)
    for key, description in info["parameters"].items():
        if key in params:
            continue
        param_obj = sig.parameters.get(key)
        if param_obj is not None and param_obj.default is not inspect.Parameter.empty:
            continue  # function has a default — no need to prompt
        hint = ""
        if "object" in key and scene_state.objects:
            hint = f"  Options: {[o.name for o in scene_state.objects]}\n"
        elif "target" in key and scene_state.targets:
            hint = f"  Options: {[t.name for t in scene_state.targets]}\n"
        print(f"  '{name}' needs '{key}': {description}")
        if hint:
            print(hint, end="")
        try:
            value = input(f"  {key}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False
        if not value:
            print(f"  Skipping '{name}' — no value provided.")
            return False
        params[key] = value

    from mrcf import flight_recorder

    # Every step on the record, timed (a session's timings.json: where the time
    # goes), what it came to, and — when it raised — an incident with the traceback.
    began = time.monotonic()
    try:
        result = fn(env=env, scene_state=scene_state, **params)
    except ValueError as exc:
        print(f"  [Error] {exc}")
        skill_executor.report(name, False, str(exc))
        _step_on_record(name, params, began, False, f"{type(exc).__name__}: {exc}")
        return False
    except Exception as exc:
        print(f"  [Error] Action '{name}' raised: {exc}")
        skill_executor.report(name, False, f"it raised {type(exc).__name__}: {exc}")
        _step_on_record(name, params, began, False, f"{type(exc).__name__}: {exc}")
        flight_recorder.RECORDER.failure(f"action '{name}' raised {type(exc).__name__}: {exc}", exc=exc,
                                         step=name, params=params)
        return False
    if result is False and skill_executor.last_report() is None:
        # Said only on the console ("not in scene", "is on top of it"): the
        # planner hears that it failed, at least. Not as refused: it may have
        # moved first, and a failure gets its one more go.
        skill_executor.report(name, False, "it failed (what it said is on the console)")
    said = skill_executor.last_report()
    _step_on_record(name, params, began, result, said["message"] if said and not said["ok"] else None,
                    refused=bool(said and said["refused"]))

    # The declared post-conditions — advisory only, since the skill has
    # already reported its own outcome. Not after a step that failed or was
    # refused: it said why, and it never claimed to be done.
    ok, reason = (True, "") if result is False else check_postconditions(name, params, scene_state)
    if not ok:
        print(f"  [PostCondition] {name} reported done but {reason} — "
              "the world model and the robot disagree; try 'observe'.")
    return result if isinstance(result, bool) else None


__all__ = ["ACTIONS", "CONDITIONS", "register_action", "register_condition", "action_schema",
           "context_action_schema", "check_preconditions", "check_postconditions", "dispatch"]
