"""
check_motion.py — the UR5's first moves from this computer: tools/check_motion.py
on the UR5 (its docstring says what each step does). THE ARM MOVES.

Before you start: the arm 15 cm or more clear of the table, the jaws pointing
roughly down (jog it there on the pendant); the pendant's speed slider at
10-25%; a hand on the pendant's emergency stop for the whole run.

    python tools/ur5/check_motion.py               # every step
    python tools/ur5/check_motion.py --steps 1,4   # only these

It offers to activate the Robotiq first if it is not (prepare_gripper in
mrcf/backends/ur5/__init__.py). Results: rig/ur5/check_motion/<time>.json.
MRCF_UR5_HOST=offline runs it on offline.py's UR5.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ["MRCF_ARM"] = "ur5"

from tools.check_motion import main  # noqa: E402

if __name__ == "__main__":
    main()
