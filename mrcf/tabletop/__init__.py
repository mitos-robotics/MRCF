"""
mrcf/tabletop — the table task: picking up and putting down whatever stands
on a table, by plain English ("put the red bottle on the tray", "put the cup
next to the box", "put the figure in the tub").

What the objects are comes from the vision model, named in its own words from
a crop of each outline the overhead camera's surface scan finds (naming.py),
and kept across scans by the object memory. Where things go: the spots marked
once and named (places.py), the free table beside any object (place_beside),
a container (place_in), and where something was picked up from.

It is not a deployment. A deployment is one arm, one layout and one task
(DEPLOYMENT_SPEC.md); each that does this task has a small package of its own
— its arm (ARM_NAME), its layout, its rules for the arm (profile.py) — and
hands its config to this one:

    deployments/<name>/config.py       ARM_NAME, its own settings, then
                                       tabletop.serve(globals()): the rest
    deployments/<name>/skills.py       tabletop.use(config); the skills registered
    deployments/<name>/llm_controller.py   main: tabletop.controller.main

The UR5's letter blocks (deployments/tabletop), the SO-101's things put
away into a tub (deployments/tidy) and the UR5's hand tools put into a box
(deployments/toolbench) are three such. The code they share is here,
once: 2026-11-17, it moved here from deployments/tabletop when the SO-101's
table work became a deployment of its own.

One deployment is bound per process in a run, as the action registry holds
one set of skills: use() before importing skills or controller. Bound again
to another deployment (the tests bind several), skills and controller are
made again for it on their next import.
"""

from __future__ import annotations

import sys
from types import ModuleType

from .settings import serve

__all__ = ["cfg", "serve", "use", "bound"]

_BOUND: "list[ModuleType]" = []


def use(config: ModuleType) -> None:
    """
    Bind the table task to a deployment's config (deployments/<name>/config.py,
    served by serve()). Bound to another deployment's before: what was made for
    that one as skills and controller were imported (its policies, its memory)
    goes, made again for this one when they are imported next.
    """
    if not hasattr(config, "ARM_NAME"):
        raise TypeError(f"{config.__name__} names no arm (ARM_NAME): a deployment is one arm, one layout, one task")
    if _BOUND and _BOUND[0] is not config:
        for made in ("controller", "skills"):
            sys.modules.pop(f"{__name__}.{made}", None)
            globals().pop(made, None)            # or `from mrcf.tabletop import skills` finds the old one
    _BOUND[:] = [config]


def bound() -> ModuleType:
    """The deployment config the table task works for (use())."""
    if not _BOUND:
        raise RuntimeError("mrcf.tabletop is not bound to a deployment: a deployment's skills.py calls "
                           "tabletop.use(config) before importing mrcf.tabletop.skills")
    return _BOUND[0]


class _Config:
    """The bound deployment's config, read at the moment each setting is read (its arm's calibration changes live)."""

    def __getattr__(self, name: str):
        return getattr(bound(), name)

    def __dir__(self):
        return dir(bound())


cfg = _Config()
