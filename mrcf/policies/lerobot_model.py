"""
mrcf/policies/lerobot_model.py — a trained LeRobot policy as a learned.Model: ACT,
diffusion, SmolVLA, pi0, SAC (HIL-SERL) — anything `lerobot-train` saves.

    model = LeRobotModel("outputs/train/upright/checkpoints/last/pretrained_model", task="stand the pawn up")
    LearnedPolicy(model, name="upright").run(env)

It loads the checkpoint's config, its weights and the pre/post-processors
saved beside them (the normalisation it was trained with), on the first use
or load(), and runs each tick through LeRobot's own predict_action — the same
path lerobot-record takes with a policy. Before anything moves, check() holds
the model's inputs and outputs against the arm's stream (streaming.StreamSpec):
a model trained on another arm, camera size or joint set is refused.

Needs `lerobot` (and torch); nothing imports them until a model is loaded,
and without them unavailable() says so, so the model is refused before
anything moves.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mrcf.policies.learned import needs


def _device(want: "str | None") -> str:
    """The device asked for, else the best there is: CUDA, then Apple's MPS, then the CPU."""
    import torch
    if want:
        return str(want)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LeRobotModel:
    """
    A LeRobot checkpoint (the folder with config.json and model.safetensors).
    task: the instruction a language-conditioned policy (SmolVLA, pi0) is given
    each tick, as in its dataset; others ignore it. device: None for the best
    one there is — not the one it was trained on, which is usually a GPU this
    machine does not have.
    """

    def __init__(self, path: "str | Path", task: "str | None" = None, device: "str | None" = None):
        self.path, self.task, self._want_device = Path(path), task, device
        self._policy: "Any | None" = None   # LeRobot's policy, once load() has run
        self._robot_type = None

    def unavailable(self) -> "str | None":
        """learned.Model.unavailable: LeRobot (and torch) not installed here."""
        return needs("lerobot", "torch", install="pip install lerobot")

    def load(self) -> "LeRobotModel":
        if self._policy is not None:
            return self
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"No LeRobot checkpoint at {self.path} (no config.json) — point at the "
                                    "pretrained_model folder lerobot-train saved.")
        cfg = PreTrainedConfig.from_pretrained(self.path)
        cfg.device = _device(self._want_device)
        self._policy = get_policy_class(cfg.type).from_pretrained(self.path, config=cfg)
        self._policy.to(cfg.device)
        self._pre, self._post = make_pre_post_processors(
            cfg, pretrained_path=str(self.path), preprocessor_overrides={"device_processor": {"device": cfg.device}})
        self.config = cfg
        # A network's first pass costs seconds (torch setting itself up: 1.6 s
        # for an ACT on the CPU) — paid here, before a segment's clock starts,
        # not out of its first tick. Then forgotten.
        self.act({k: (np.zeros((f.shape[1], f.shape[2], f.shape[0]), np.uint8) if len(f.shape) == 3
                      else np.zeros(f.shape, np.float32)) for k, f in cfg.input_features.items()})
        self.reset()
        return self

    def features(self) -> "tuple[dict, dict]":
        """({input key: shape}, {output key: shape}), as the model was trained."""
        self.load()
        return ({k: tuple(f.shape) for k, f in self.config.input_features.items()},
                {k: tuple(f.shape) for k, f in self.config.output_features.items()})

    def check(self, spec) -> list:
        """learned.Model.check: why this model cannot run on the arm's stream (spec.mismatches); [] when it can."""
        inputs, outputs = self.features()
        self._robot_type = spec.robot_type or None
        return spec.mismatches(inputs, outputs)

    def reset(self) -> None:
        self.load()
        assert self._policy is not None
        self._policy.reset()
        self._pre.reset()
        self._post.reset()

    def act(self, observation: dict) -> np.ndarray:
        import torch
        from lerobot.utils.control_utils import predict_action

        self.load()
        action = predict_action(dict(observation), self._policy, torch.device(self.config.device), self._pre,
                                self._post, use_amp=bool(getattr(self.config, "use_amp", False)), task=self.task,
                                robot_type=self._robot_type)
        return action.detach().cpu().numpy().reshape(-1)
