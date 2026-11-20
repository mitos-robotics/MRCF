"""
profiles.py — a deployment's own rules for its arm, laid over the arm's facts.

A deployment is one arm, one layout and one task (DEPLOYMENT_SPEC.md): the
chess board on the SO-101, the letter blocks on the UR5's table, things put
away into a tub on the SO-101's bare table. The same task on another arm, or
on another layout, is another deployment. Each names its arm in its config
(ARM_NAME) and has one profile:

    deployments/<name>/profile.py

An arm's config (mrcf/backends/<arm>/robot_config.py) says what is true of
the arm whatever it does: its joints, its jaws, its cameras and their
calibrations, how far it reaches. How it should go about this deployment's
work — how wide to open among chess pieces, how hard to hold a wooden piece,
what a height that disagrees means on a crowded board — is the deployment's,
and lives in its profile: a module of UPPER_CASE settings by the names the
arm's config uses (GRIPPER_MAX_OPEN_M, GRIPPER_HOLD_PCT, PLACE_IN, ...), each
with why. When the deployment builds its arm's env and policies
(make_env(profile=...), make_policies(profile=...)) its profile is laid over
the arm's config: a setting the profile names wins, the rest are the arm's,
and what neither names is the framework's default.

A new deployment starts from the framework's defaults and its arm's facts,
not from another deployment's tuning. What another learned is copied into the
new profile only where it is relevant to the new work, with its reason
(tools/new_deployment.py lists another deployment's rules for the choosing).
2026-11-17: the SO-101's table work inherited ~30 rules tuned on chess pieces
from the arm's config — a 45 mm jaw opening, "held" meaning 3.4 mm or wider, a
height that disagrees meaning the neighbouring piece — and every one of them
surfaced as a failure on a scoop, a plug and a sweet. The same day its rules,
filed as deployments/tabletop/profiles/so101.py inside the UR5's task, were
read by the SO-101 alone: the tabletop on the UR5 had none of them. So a
profile is a deployment's, and a deployment has one arm.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

PROFILE_FILE = "profile.py"


def path(deployment_dir) -> Path:
    """Where a deployment's profile is: deployments/<name>/profile.py."""
    return Path(deployment_dir) / PROFILE_FILE


def load(deployment_dir) -> "ModuleType | None":
    """The deployment's profile, imported; None when it has none (its arm's config alone)."""
    where = path(deployment_dir)
    if not where.is_file():
        return None
    name = f"mrcf_profile_{Path(deployment_dir).name}"
    spec = importlib.util.spec_from_file_location(name, where)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(profile: "ModuleType | None") -> dict:
    """The settings a profile names: {NAME: value}, its UPPER_CASE names only."""
    if profile is None:
        return {}
    return {name: value for name, value in vars(profile).items()
            if name.isupper() and not name.startswith("_")}


class Layered:
    """
    An arm's config with a profile over it: an attribute the profile names
    is the profile's, any other the config's (AttributeError when neither has
    it, as a module's). What wiring and the backends read a config through.
    """

    def __init__(self, config, profile: "ModuleType | None"):
        self._config = config
        self._own = settings(profile)

    def __getattr__(self, name: str):
        own = self.__dict__.get("_own", {})
        if name in own:
            return own[name]
        return getattr(self.__dict__["_config"], name)

    def __dir__(self):
        return sorted(set(dir(self._config)) | set(self._own))


def layered(config, profile: "ModuleType | None"):
    """config with profile laid over it (Layered); config itself when there is no profile."""
    return config if not settings(profile) else Layered(config, profile)
