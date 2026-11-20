"""
Shared plumbing for the UR5's own tool scripts. The tools that work on any arm
are in the top-level tools/ (run them with MRCF_ARM=ur5); their task and image
helpers are shared from there.

MRCF_UR5_HOST picks the controller (robot_config.HOST); MRCF_UR5_HOST=offline
runs these tools on offline.py's UR5 in memory.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mrcf.backends.ur5 import make_env, robot_config as rc  # noqa: E402,F401
from tools._common import deployment_config, show_image, task_config  # noqa: E402,F401

# What these tools save about this rig goes in rig/ur5/ (rig_state.py), not beside the scripts.
rc.RIG_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["rc", "make_env", "deployment_config", "task_config", "show_image", "FRAMEWORK_ROOT"]

FRAMEWORK_ROOT = _ROOT
