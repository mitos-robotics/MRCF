"""
skills.py — the toolbench deployment's skills: the framework's table task
(mrcf/tabletop/skills.py), bound to this deployment's config (config.py): its
arm, its rules for it (profile.py), its kept state. The skills register as
this is imported; anything else of the table task's is read through here as
this deployment's (skills.POLICIES, skills.memory()).
"""

from mrcf import tabletop as _tabletop

from . import config

_tabletop.use(config)

from mrcf.tabletop import skills as _skills  # noqa: E402 — after use(): registers the table task's skills


def __getattr__(name: str):
    return getattr(_skills, name)
