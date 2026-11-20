"""
mrcf/policies/rl.py — a policy trained by reinforcement learning, elsewhere, as a
learned.Model, run like any other learned policy (mrcf/policies/learned.py).

Such a policy is a small network that reads a vector — the arm's state, and
the task's goal (observation.environment_state) — and answers with a step in
[-1, 1] for each action. VectorCodec is how: normalising the observation, and
turning each step into the absolute goal the arm's stream takes (where that
joint is now, plus the step times its size). It is saved beside the network,
and whatever trained the network must have read and acted through the same
codec, so the network at run time reads and acts exactly as it did in
training. Its check() refuses a stream with other names: a policy trained on
one arm's stream runs where the stream is the same, and nowhere else.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mrcf.policies.learned import needs
from mrcf.streaming import ENVIRONMENT, STATE


@dataclass(frozen=True)
class VectorCodec:
    """How a vector policy reads the arm's stream and drives it. See the module docstring."""
    state: tuple                 # the stream's state names, in order
    action: tuple                # its action names, in order — each a goal for one of the state's
    step: tuple                  # per action: how far a step of 1 moves its goal from where it is, a tick
    environment: int = 0         # observation.environment_state's length (a goal's); 0: none
    mid: tuple = ()              # the observation (state, then environment) normalised to (x - mid) / half
    half: tuple = ()

    def __post_init__(self):
        missing = [a for a in self.action if a not in self.state]
        if missing:
            raise ValueError(f"each action must be a goal for one of the state's; {missing} are not")

    def observation(self, observation: dict) -> np.ndarray:
        """The network's input: the state (and the goal), normalised."""
        parts = [np.asarray(observation[STATE], np.float32).ravel()]
        if self.environment:
            parts.append(np.asarray(observation[ENVIRONMENT], np.float32).ravel())
        x = np.concatenate(parts)
        if self.mid:
            x = (x - np.asarray(self.mid, np.float32)) / np.asarray(self.half, np.float32)
        return x.astype(np.float32)

    def action_for(self, observation: dict, step) -> np.ndarray:
        """The stream's action for a step in [-1, 1]: each goal where its joint is now, moved step x its size."""
        state = np.asarray(observation[STATE], float).ravel()
        here = np.array([state[self.state.index(a)] for a in self.action])
        return (here + np.clip(np.asarray(step, float).ravel(), -1.0, 1.0) * np.asarray(self.step)).astype(np.float32)

    def check(self, spec) -> list:
        """Why this codec cannot run on a stream (spec); [] when it can."""
        problems = []
        if tuple(spec.state) != tuple(self.state):
            problems.append(f"trained on the state {list(self.state)}, the arm streams {list(spec.state)}")
        if tuple(spec.action) != tuple(self.action):
            problems.append(f"trained to send {list(self.action)}, the arm takes {list(spec.action)}")
        if len(spec.environment) != self.environment:
            problems.append(f"trained with a goal of {self.environment} values, given {len(spec.environment)}"
                            + (" (give LearnedPolicy a goal)" if self.environment and not spec.environment else ""))
        return problems

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> "VectorCodec":
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(Path(path).read_text()).items()})  # type: ignore[arg-type]  # fields read from JSON


class SB3Model:
    """
    A trained Stable-Baselines3 policy — its folder: policy.zip (model.save),
    codec.json (VectorCodec.save) and meta.json ({"algo": "sac"}, or "td3",
    "ppo", ...) — as a learned.Model. Needs stable-baselines3; nothing imports
    it until the model is loaded.
    """

    def __init__(self, run_dir: "str | Path", deterministic: bool = True):
        self.run_dir, self.deterministic = Path(run_dir), deterministic
        self._policy: "Any | None" = None   # Stable-Baselines3's policy, once load() has run

    def unavailable(self) -> "str | None":
        """learned.Model.unavailable: Stable-Baselines3 not installed here."""
        return needs("stable_baselines3", install='pip install -e ".[sb3]"')

    def load(self) -> "SB3Model":
        if self._policy is None:
            import stable_baselines3 as sb3
            meta = json.loads((self.run_dir / "meta.json").read_text())
            self.codec = VectorCodec.load(self.run_dir / "codec.json")
            self._policy = getattr(sb3, meta["algo"].upper()).load(self.run_dir / "policy.zip", device="cpu")
        return self

    def check(self, spec) -> list:
        """learned.Model.check: the codec's — the stream it was trained on."""
        return self.load().codec.check(spec)

    def reset(self) -> None:
        self.load()

    def act(self, observation: dict) -> np.ndarray:
        self.load()
        assert self._policy is not None
        step, _ = self._policy.predict(self.codec.observation(observation), deterministic=self.deterministic)
        return self.codec.action_for(observation, step)
