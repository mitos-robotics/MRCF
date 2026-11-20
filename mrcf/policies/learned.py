"""
mrcf/policies/learned.py — a policy whose motion comes from a trained model rather
than from geometry. Trained outside this repository, by imitation (ACT,
diffusion, SmolVLA, pi0 — from demonstrations) or by reinforcement learning
(SAC — HIL-SERL), it runs the same way: a loop, a tick at a time, at the rate
it was trained at,

    observation = env.stream_observation()      the arm's own (capabilities.ActionStreaming)
    action      = model.act(observation)        the model, through an adapter (Model)
    env.stream_action(action)                   the arm checks it and sends it — or holds it back

until the segment is done (`until`), its time is up, or the arm has held back
everything the model asked for a while. Then an Outcome, as every policy
returns. A planned policy (parallel_jaw) is the other kind: a Target in, moves
out; both live in the arm's policy library's world and never see the world
model or the memory.

What an observation and an action ARE is the arm's (env.stream_spec, a
streaming.StreamSpec — mrcf/backends/so101/demos.py on the SO-101): a model trained
on one arm's stream runs only on an arm with the same one, and LearnedPolicy
checks that before it moves (Model.check). A model that needs a library this
machine does not have (LeRobot, Stable-Baselines3) says so (unavailable()).
LearnedPolicy refuses both — and chain() before its first step moves the arm.
Nothing else here knows any arm.

A learned policy is usually one segment of a skill: a planned move to where
the demonstrations started, the learned part, then a look to see that it
worked. chain() runs policies in order on one target, and
manipulation.object_action makes the result something the planner can ask for:

    from mrcf.policies.learned import LearnedPolicy, chain, hover_over
    from mrcf.policies.lerobot_model import LeRobotModel

    upright = LearnedPolicy(LeRobotModel(".../checkpoints/last/pretrained_model", task="stand the pawn up"),
                            name="upright", max_s=20)
    manipulation.object_action("upright", chain(hover_over(0.06), upright),
                               description="Stand a fallen piece up.", after=look_again)
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, replace
from typing import Callable, Protocol, cast, runtime_checkable

import numpy as np

from mrcf.capabilities import ActionStreaming
from mrcf.policies.base import Outcome
from mrcf.streaming import ENVIRONMENT, StreamSpec, differences


@runtime_checkable
class Model(Protocol):
    """
    A trained model, as a learned policy runs it. Adapters: CallableModel,
    lerobot_model.LeRobotModel, rl.SB3Model. Optional, beside these two:
    check(spec) -> [why it cannot run on this stream], and unavailable() -> why
    it cannot run on this machine at all (a library it needs: needs()), or None.
    """

    def reset(self) -> None:
        """A new episode: forget anything queued or remembered from the last (an action chunk, a history)."""

    def act(self, observation: dict):
        """The next action for this observation: a vector in the stream's action order."""


def _installed(module: str) -> bool:
    """Whether a Python package is installed: looked up, not imported (torch alone takes seconds to import)."""
    if sys.modules.get(module) is not None:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):     # "a.b" with no "a"; a module whose spec was never set
        return False


def needs(*modules: str, install: str) -> "str | None":
    """
    A model's unavailable(): which of the Python packages it imports are not
    installed here, and how to add them (install); None when all are.
    """
    gone = [m for m in modules if not _installed(m)]
    return f"needs {' and '.join(gone)}, not installed here ({install})" if gone else None


def _unavailable(step) -> "str | None":
    """Why a model, a LearnedPolicy or a chain() cannot run on this machine; None when it can, or cannot say."""
    why = getattr(step, "unavailable", None)
    return why() if callable(why) else None


class CallableModel:
    """
    Any function observation -> action as a Model: a policy exported some other
    way (ONNX, a plain network), a scripted stand-in, a test's. `check(spec) ->
    [problems]` and `reset()` are optional.
    """

    def __init__(self, act: Callable, reset: "Callable | None" = None, check: "Callable | None" = None):
        self._act, self._reset, self._check = act, reset, check

    def reset(self) -> None:
        if self._reset is not None:
            self._reset()

    def act(self, observation: dict):
        return self._act(observation)

    def check(self, spec) -> list:
        return list(self._check(spec)) if self._check is not None else []


@dataclass
class LearnedPolicy:
    """
    A trained model run on the arm's stream for one segment of a skill.

    model: a Model, or a function returning one (loaded on first use, once —
    loading a network takes seconds, and the planner may never ask for it).
    """
    model: "Model | Callable[[], Model]"
    name: str = "learned"
    max_s: float = 30.0                   # longest it runs; pauses (the sentry) do not count
    until: "Callable | None" = None       # until(env, observation) -> True when the segment is done;
                                          # None: it runs max_s and that is success
    max_held_back_s: float = 1.0          # the arm held back what the model asked for this long,
                                          # without a break: stop — it is pushing where it may not go
    goal: "Callable | None" = None        # goal(target) -> vector: what a goal-conditioned model is told
                                          # each tick as observation.environment_state (where to reach);
                                          # None: the model is told nothing of the target
    trained_on: "StreamSpec | None" = None   # the stream the model was trained on, when known
                                             # (a package's manifest): held against the arm's, names and
                                             # robot type too (streaming.differences), before anything moves

    def __post_init__(self):
        self._loaded: "Model | None" = self.model if isinstance(self.model, Model) else None

    def unavailable(self) -> "str | None":
        """
        Why its model cannot run on this machine (Model.unavailable: a library it
        needs is not installed), without loading it; None when it can — or when
        it cannot say, a model given as a function.
        """
        return _unavailable(self._loaded if self._loaded is not None else self.model)

    def loaded(self) -> Model:
        """The model, loaded."""
        if self._loaded is None:
            self._loaded = cast("Callable[[], Model]", self.model)()    # not a Model, so the function making one
        return self._loaded

    def __call__(self, pol, env, target=None, obstacles=(), **_) -> Outcome:
        """As a step of chain(), or an object_action policy: (policies, env, target, obstacles) -> Outcome."""
        return self.run(env, target, obstacles)

    def refusal(self, env, target=None) -> "Outcome | None":
        """
        Why it would not run on env for target — its model's library missing,
        no stream, no target for its goal, a stream other than the one it was
        trained on — as the refused Outcome; None when it would. Nothing moves
        (the model is loaded, to check it). chain() asks every step this before
        any of them moves the arm.
        """
        return self._ready(env, target)[1]

    def _ready(self, env, target):
        """((the stream's spec, the goal, the model), None) to run with; or (None, the refusal)."""
        missing = self.unavailable()
        if missing:
            return None, self._outcome(False, "unavailable", f"cannot run on this machine — {missing}", refused=True)
        if not isinstance(env, ActionStreaming):
            return None, self._outcome(False, "no_stream", f"{type(env).__name__} takes no stream of actions "
                                       "(capabilities.ActionStreaming)", refused=True)
        spec = env.stream_spec
        goal = None
        if self.goal is not None:
            if target is None:
                return None, self._outcome(False, "no_target", "it is told where to go, and was given no target",
                                           refused=True)
            goal = np.asarray(self.goal(target), np.float32).ravel()
            spec = replace(spec, environment=tuple(f"goal.{i}" for i in range(len(goal))))
        problems = differences(self.trained_on, spec) if self.trained_on is not None else []
        if problems:
            return None, self._outcome(False, "mismatch", "the model was trained on another stream than this "
                                       "arm's: " + "; ".join(problems), refused=True)
        model = self.loaded()
        missing = _unavailable(model)        # a model given as a function says only once it is made
        if missing:
            return None, self._outcome(False, "unavailable", f"cannot run on this machine — {missing}", refused=True)
        problems = model.check(spec) if hasattr(model, "check") else []
        if problems:
            return None, self._outcome(False, "mismatch", "the model was trained on another stream than this "
                                       "arm's: " + "; ".join(problems), refused=True)
        return (spec, goal, model), None

    def run(self, env, target=None, obstacles=()) -> Outcome:
        """Run the model on env's stream until done, out of time, or held back; refused without moving if it cannot."""
        ready, refused = self._ready(env, target)
        if refused is not None:
            return refused
        spec, goal, model = ready
        tag = self.name
        model.reset()
        period = 1.0 / float(spec.fps)
        start = env.motion_clock()
        due, paused = start, env.paused_s
        ticks, behind, resets, held_since, why, result = 0, 0, 0, None, None, None
        act_s: list = []
        try:
            while True:
                now = env.motion_clock()
                if now - start >= self.max_s:
                    result = "not_done" if self.until is not None else "ran"
                    break
                observation = env.stream_observation()
                if goal is not None:
                    observation[ENVIRONMENT] = goal
                if self.until is not None and self.until(env, observation):
                    result = "done"
                    break
                t0 = time.perf_counter()
                action = np.asarray(model.act(observation), float).ravel()
                act_s.append(time.perf_counter() - t0)
                if action.shape != (len(spec.action),):
                    result, why = "bad_action", f"the model gave {action.shape} values, the arm takes {len(spec.action)}"
                    break
                _sent, held = env.stream_action(action)
                ticks += 1
                if env.paused_s > paused:
                    # Paused mid-segment (the sentry): what it had queued was for a
                    # scene that has since changed. It starts again from what it sees.
                    paused = env.paused_s
                    model.reset()
                    resets += 1
                now = env.motion_clock()
                if held:
                    why = held
                    held_since = now if held_since is None else held_since
                    if now - held_since >= self.max_held_back_s:
                        result = "held_back"
                        break
                else:
                    held_since = None
                due += period
                wait = due - env.motion_clock()
                if wait > 0:
                    time.sleep(wait)
                elif wait < -period:
                    behind += 1                   # a slow tick: carry on from now, do not catch up
                    due = env.motion_clock()
        except Exception:
            env.hold()
            raise
        env.hold()                                # where the last action left it, not chasing it on
        elapsed = env.motion_clock() - start
        timing = (f"{ticks} ticks in {elapsed:.1f} s at {spec.fps:g} Hz"
                  + (f"; the model took {1000 * float(np.mean(act_s)):.0f} ms a tick "
                     f"(at most {1000 * max(act_s):.0f})" if act_s else "")
                  + (f"; {behind} slow" if behind else "")
                  + (f"; started again {resets}x after a pause" if resets else ""))
        said = {"done": "done", "ran": f"ran its {self.max_s:g} s",
                "not_done": f"not done after {self.max_s:g} s",
                "held_back": f"stopped: the arm held back what it asked for {self.max_held_back_s:g} s ({why})",
                "bad_action": f"stopped: {why}"}[result]
        return self._outcome(result in ("done", "ran"), result, f"{said} — {timing}", notes=[f"[{tag}] {timing}"])

    def _outcome(self, ok: bool, code: str, message: str, refused: bool = False, notes=()) -> Outcome:
        return Outcome(ok, f"{self.name}: {message}", refused=refused, approach=self.name, notes=list(notes),
                       attempts=[{"attempt": 1, "approach": self.name, "tilt_deg": None, "result": "ok" if ok else code}])


def hover_over(height_m: "float | None" = None) -> Callable:
    """A chain() step: the policy library's move_above, height_m over the target (None: its own hover)."""
    def step(pol, env, target, obstacles=(), **_) -> Outcome:
        return pol.move_above(env, target, height_m)
    return step


def chain(*steps: Callable) -> Callable:
    """
    One policy from several, run in order on the same target — each
    step(policies, env, target, obstacles, **params) -> Outcome, as
    manipulation.object_action takes. Stops at the first that fails. The
    Outcome is the last one run, with every step's message, notes and attempts.
    A step that would refuse (LearnedPolicy.refusal: a missing library, no
    stream, a stream other than its model's) refuses the whole chain before its
    first step moves the arm; the chain's own unavailable() says beforehand
    which library is missing, with no arm to ask.
    """
    if not steps:
        raise ValueError("chain() needs at least one step")

    def unavailable() -> "str | None":
        return next(filter(None, map(_unavailable, steps)), None)

    def refusal(env, target=None) -> "Outcome | None":
        for step in steps:                   # every step, before any of them moves the arm
            missing = _unavailable(step)
            if missing:
                name = getattr(step, "name", "a step")
                return Outcome(False, f"{name}: cannot run on this machine — {missing}", refused=True,
                               approach=name, attempts=[{"attempt": 1, "approach": name, "tilt_deg": None,
                                                         "result": "unavailable"}])
            ask = getattr(step, "refusal", None)
            refused = ask(env, target) if callable(ask) else None
            if refused is not None:
                return refused
        return None

    def run(pol, env, target, obstacles=(), **params) -> Outcome:
        refused = refusal(env, target)       # no step is run, so nothing moves
        if refused is not None:
            return refused
        ran = []
        for step in steps:
            ran.append(step(pol, env, target, obstacles, **params))
            if not ran[-1].ok:
                break
        last = ran[-1]
        return replace(last, message="; ".join(o.message for o in ran),
                       notes=[n for o in ran for n in o.notes], attempts=[a for o in ran for a in o.attempts],
                       refused=last.refused and len(ran) == 1)
    run.unavailable, run.refusal = unavailable, refusal  # type: ignore[attr-defined]  # read back with getattr
    return run
