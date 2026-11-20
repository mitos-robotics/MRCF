"""
config.py — the toolbench deployment: the UR5 picking up hand tools on its
table and putting them into a box, by plain English ("put the hammer in the
box", "put the screwdrivers in the box", "put the tape measure next to the
box"). Tools with a handle (screwdrivers, a hammer) and compact boxy ones (a
tape measure); the box a cardboard one standing on the table.

A deployment is one arm, one layout and one task (DEPLOYMENT_SPEC.md):

  arm      the UR5 (mrcf/backends/ur5) with its Robotiq Hand-E (50 mm
           stroke), no wrist camera: its facts in its robot_config
  layout   its table, as for deployments/tabletop: the overhead camera over
           the table (moved and calibrated again 2026-11-20), the box on the
           table; kept state in rig/ur5/ (toolbench_memory.json,
           toolbench_places.json, toolbench_containers.json, ...)
  task     the framework's table task (mrcf/tabletop): find, name, pick,
           place, put into a container, put beside, mark a spot
  rules    how it drives the UR5 for tools: profile.py, each rule with why

Its task is planned in advance (TASKS), the planner off (PLANNER): "put the
tools away" (or "go") puts the hammer, every screwdriver, the tape measure
and the tape into the box, each found on the table by what it is when the
task runs. The user,
2026-11-20, after two runs in which the planner moved nothing: "lets maybe
consider also allow a non planner method - so we pre plan that the tools go
into the box". PLANNER = True puts plain English back for anything else.

Set here: what is this deployment's own. Any other setting is served by
tabletop.serve() at the end: its profile's, else the table task's default
(mrcf/tabletop/settings.py), else the UR5's (robot_config, read live).

Started 2026-11-20 from deployments/tabletop (tools/new_deployment.py); the
blocks' sizes it set (KNOWN_SIZES_MM) are not this one's: no sizes are
written in by hand — what a tool measures comes from the cameras, or from
the user saying so (identify). The runs behind its values: HISTORY.md, and
the table task's (mrcf/tabletop/HISTORY.md).
"""

from pathlib import Path as _Path

from mrcf import backends as _backends
from mrcf import profiles as _profiles
from mrcf import tabletop as _tabletop

# ── The arm, and this deployment's rules for it ───────────────────────────────
ARM_NAME = "ur5"
ARM = _backends.arm(ARM_NAME)
PROFILE = _profiles.load(_Path(__file__).parent)      # profile.py

# ── Its task ──────────────────────────────────────────────────────────────────
# Laid out by the quick plan when it runs (mrcf/tabletop/skills.task_plan): the
# hammer and the screwdrivers — every one on the table (TASK_WORDS_BY_KIND) —
# each into the box, and the tape measure (the user, 2026-11-20: "the tape
# measure is not too wide"; the 58 mm said of it was the scan's round foot),
# and the tape.
TASKS = {"put the tools away": "put the hammer, the screwdrivers, the tape measure and the tape into the box"}
PLANNER = False

# Everything else: its profile's, the table task's, the UR5's.
_tabletop.serve(globals())
