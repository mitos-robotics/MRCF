"""
Shared plumbing for the SO-101's own tool scripts — its poses, motors and
joint calibration. The tools that work on any arm are in the top-level tools/,
and their task and image helpers (deployment_config, task_config, show_image)
are shared from there.

Re-exports make_env (which lives in mrcf/backends/so101, since deployments need it
too).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mrcf.backends.so101 import SO101Env, make_env    # noqa: E402,F401
from mrcf.backends.so101 import robot_config as rc   # noqa: E402

# What these tools save about this rig — frames, touches, fits, the URDF — goes in
# its own folder (rc.RIG_DIR: rig/so101/, rig_state.py), not beside the scripts.
rc.RIG_DIR.mkdir(parents=True, exist_ok=True)
from tools._common import deployment_config, show_image, task_config  # noqa: E402,F401

__all__ = ["rc", "SO101Env", "make_env", "deployment_config", "task_config",
           "FRAMEWORK_ROOT", "calibration_path", "show_image"]

FRAMEWORK_ROOT = _ROOT


def calibration_path() -> Path:
    """
    Where lerobot keeps this arm's joint calibration (lerobot-calibrate writes
    it), for tools that read it without connecting. Asked of lerobot rather
    than assumed, since HF_LEROBOT_CALIBRATION can move it; the default
    location if lerobot cannot be imported.
    """
    try:
        from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
        base = Path(HF_LEROBOT_CALIBRATION) / ROBOTS
    except ImportError:
        base = Path.home() / ".cache/huggingface/lerobot/calibration/robots"
    return base / "so_follower" / f"{rc.FOLLOWER_ID}.json"
