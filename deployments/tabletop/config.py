"""
config.py — the tabletop deployment: the UR5 picking up, stacking and putting
down letter blocks on its table, by plain English ("put the Z block on the P
block", "stack the blocks", "put the cup next to the box").

A deployment is one arm, one layout and one task (DEPLOYMENT_SPEC.md):

  arm      the UR5 (mrcf/backends/ur5): its facts in its robot_config
  layout   its table, the overhead camera above the far side looking back at
           the arm, a person across the table; kept state in rig/ur5/
           (tabletop_memory.json, tabletop_places.json, ...)
  task     the framework's table task (mrcf/tabletop): find, name, pick,
           place, stack, put beside, mark a spot
  rules    how it drives the UR5: profile.py — none yet, the arm's facts alone

Set here: what is this deployment's own. Any other setting is served by
tabletop.serve() at the end: its profile's, else the table task's default
(mrcf/tabletop/settings.py), else the UR5's (robot_config, read live).

The runs behind its values: HISTORY.md, and the table task's
(mrcf/tabletop/HISTORY.md).
"""

from pathlib import Path as _Path

from mrcf import backends as _backends
from mrcf import profiles as _profiles
from mrcf import tabletop as _tabletop

# ── The arm, and this deployment's rules for it ───────────────────────────────
ARM_NAME = "ur5"
ARM = _backends.arm(ARM_NAME)
PROFILE = _profiles.load(_Path(__file__).parent)      # profile.py; None: the arm's facts alone

# ── Its objects ───────────────────────────────────────────────────────────────
# The letter blocks, by what they are called: (height, width across its flat
# sides) in mm, recorded each scan for every block with no size of its own
# (skills.known_sizes). The overhead camera looks down at them at 14-29 deg
# from straight down (2026-11-12), 0.2-0.34 px a mm of height: below what it
# measures from (MIN_HEIGHT_PX_PER_MM), and too few pixels to stack by anyway.
# The user's letter blocks: said "about 4 cm", but the Hand-E closed to 30 mm
# across one (2026-11-12 15:09, blue P), and a place onto the yellow block,
# taking it for 40 mm tall, let go ~1.5 cm above it. MEASURE with a ruler.
KNOWN_SIZES_MM = {"block": (30.0, 30.0)}

# Everything else: its profile's, the table task's, the UR5's.
_tabletop.serve(globals())
