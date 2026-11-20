"""
config.py — the tidy deployment: the SO-101 putting everyday things away into
a tub on the bare table. Wooden figures, a plug, a measuring scoop, sweets,
into a clear plastic tub, by plain English ("put the red figure in the tub",
"put everything into the tub"); the chessboard taken away.

A deployment is one arm, one layout and one task (DEPLOYMENT_SPEC.md):

  arm      the SO-101 (mrcf/backends/so101): its facts in its robot_config
  layout   the bare table, 20-30 mm below the chessboard top the arm was
           measured on; the arm's overhead camera (chess's); a tub; kept
           state in rig/so101/ (tidy_surface.json: the table as touched,
           tidy_memory.json, tidy_containers.json, ...)
  task     the framework's table task (mrcf/tabletop): find, name, pick,
           place, put into a container, put beside, mark a spot
  rules    how it drives the SO-101: profile.py, each rule with why

Set here: what is this deployment's own. Any other setting is served by
tabletop.serve() at the end: its profile's, else the table task's default
(mrcf/tabletop/settings.py), else the SO-101's (robot_config, read live).

2026-11-17: this was the SO-101 running deployments/tabletop, the UR5's, with
its rules filed there as profiles/so101.py; it became a deployment of its own
the same evening (HISTORY.md).
"""

from pathlib import Path as _Path

from mrcf import backends as _backends
from mrcf import profiles as _profiles
from mrcf import tabletop as _tabletop

# ── The arm, and this deployment's rules for it ───────────────────────────────
ARM_NAME = "so101"
ARM = _backends.arm(ARM_NAME)
PROFILE = _profiles.load(_Path(__file__).parent)      # profile.py

# ── Its task, planned in advance ─────────────────────────────────────────────
# Run by its name, or "go" (mrcf/tabletop/settings.py TASKS); "put ... into
# ..." is laid out from its words as before (QUICK_PUT_INTO); nothing else is
# planned, and a failed step is not planned again (PLANNER). User, 2026-11-19:
# "i want to have the option now for any new deployment that we can skip the
# planner - so it just does the task we pre plan for it". 18:13 that day, the
# planner asked for "the red ball" the scan had not named: 61 s, a plan that
# stopped at its first miss, and a 71 s re-plan that left the ball out.
TASKS = {"tidy up": "put the red cap, the red ball, the white plug and the blue scoop into the plastic tub"}
PLANNER = False

# Everything else: its profile's, the table task's, the SO-101's.
_tabletop.serve(globals())
