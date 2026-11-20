"""
backends — the arms. One package per arm, mrcf/backends/<name>/, and every one
exports the same few things, so that a deployment or a tool never names the
arm it drives:

    NAME              what to call it in messages ("SO-101")
    config            its settings module (robot_config.py): ports, cameras,
                      measured poses, limits, gripper tuning — and the
                      RIG_SETTINGS below
    make_env()        its RobotEnv, wired from config, not yet connected
    make_policies()   its policy library (policies.base.Policies)

and, optionally,

    describe()        what it is connecting to, for the start-up message
    tour_and_calibrate(env)
                      how it re-measures its fixed camera's calibration by
                      itself (overhead_calibration.check_camera's `tour`)
    prepare_gripper(ask)
                      gets its gripper ready before tools/check_motion.py
                      moves it (a Robotiq: activated, asked first), and says
                      whether to use it
    present()         whether it looks plugged in to this computer, checked
                      without connecting to it: (True/False, what was found),
                      for run.py's choice of arm. Nothing is opened or sent.

make_env() and make_policies() take what every arm shares from config by
the settings' names, through wiring.py, and pass only what is the arm's own.

Which arm runs: the MRCF_ARM environment variable (MRCF_ARM=so101), else
DEFAULT. One arm per process. `python run.py` asks which (or takes --arm NAME)
and sets MRCF_ARM before anything reads it. An arm is a folder here, or an installed Python
package that names itself in the "mrcf.arms" entry point group
([project.entry-points."mrcf.arms"] my_arm = "my_package.my_arm"), so an arm
can be distributed and installed without this repository changing.

    from mrcf import backends
    arm = backends.arm()
    env, policies = arm.make_env(), arm.make_policies()
    arm.config.TABLE_Z

Adding an arm: DEPLOYMENT_SPEC.md, "Adding an arm".
"""

from __future__ import annotations

import importlib
import os
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType

PLUGIN_GROUP = "mrcf.arms"
_plugin_arms: "dict[str, ModuleType]" = {}      # installed plugins' arms, once loaded

ENV_VAR = "MRCF_ARM"
DEFAULT = "so101"

# What every arm's config defines, because deployments and the framework read
# them without knowing which arm it is. A camera or a calibration the rig does
# not have is None. tests/test_backends.py checks every arm has them all.
RIG_SETTINGS = {
    "CAMERAS":                   "every camera on the rig, {name: backends.rig.Camera}: where it is, its source and size",
    "SENSORS":                   "every sensor beyond the joints and the gripper, {name: backends.rig.Sensor} ({}: none)",
    "CAMERA_NAME":               "the wrist camera's name: the one the policies measure and align with (None: none)",
    "CAMERA_W":                  "its capture width, pixels",
    "CAMERA_H":                  "its capture height, pixels",
    "GRIPPER_CAM_K":             "its 3x3 intrinsics at RENDER_W x RENDER_H (None: not calibrated)",
    "GRIPPER_CAM_DIST":          "its distortion coefficients, OpenCV order",
    "OVERHEAD_CAMERA_NAME":      "the fixed camera that scans the surface and sights the gripper tag (None: none)",
    "OVERHEAD_CAMERA_W":         "its capture width, pixels",
    "OVERHEAD_CAMERA_H":         "its capture height, pixels",
    "OVERHEAD_CAM_MODEL":        "its lens and pose, {size, K, D, T_world_cam} (None: not calibrated)",
    "OVERHEAD_HOMOGRAPHY":       "its pixels (RENDER_W x RENDER_H) -> table (x, y), 3x3 (None: not calibrated)",
    "OVERHEAD_CALIBRATION_PATH": "where overhead_calibration.py keeps its live calibration",
    "RENDER_W":                  "the width the pipeline reasons in; render_image resizes to it",
    "RENDER_H":                  "the height likewise",
    "TABLE_Z":                   "the work surface's height in the arm's frame, metres",
    "WORKSPACE_BOUNDS":          "safety.WorkspaceBounds the grasp point stays inside",
    "SURFACE_REFERENCE_PATH":    "a photo of the empty work surface from the fixed camera",
    "MOTION_GUARD_ZONE":         "the (x, y) polygon the motion guard watches (None: no guard)",
    "MOTION_GUARD_CAMERAS":      "the fixed cameras it watches through, a guard each (usually [OVERHEAD_CAMERA_NAME])",
    "GRIPPER_TAG_ID":            "the tag on the gripper the fixed camera sees (None: none)",
}

_ROOT = Path(__file__).resolve().parent


def available() -> list[str]:
    """The arms there are: each folder here with an __init__.py, and each installed plugin's."""
    here = {p.name for p in _ROOT.iterdir() if (p / "__init__.py").is_file() and not p.name.startswith(("_", "."))}
    return sorted(here | {ep.name for ep in entry_points(group=PLUGIN_GROUP)})


def arm_name() -> str:
    """The arm this process drives: MRCF_ARM, else DEFAULT."""
    return os.environ.get(ENV_VAR, "").strip() or DEFAULT


def arm(name: "str | None" = None) -> ModuleType:
    """The arm's package (backends.<name>): MRCF_ARM's, unless `name` says which."""
    name = name or arm_name()
    if name in _plugin_arms:
        return _plugin_arms[name]
    try:
        return importlib.import_module(f"mrcf.backends.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"mrcf.backends.{name}":
            raise                       # the arm is there; something it imports is not
        for ep in entry_points(group=PLUGIN_GROUP):
            if ep.name == name:
                _plugin_arms[name] = ep.load()       # an installed package's arm
                return _plugin_arms[name]
        raise ModuleNotFoundError(f"No arm '{name}' ({ENV_VAR}={name}). Available: "
                                  f"{', '.join(available()) or '(none)'}", name=exc.name) from None


def missing_rig_settings(config: ModuleType) -> list[str]:
    """The RIG_SETTINGS an arm's config does not define."""
    return [name for name in RIG_SETTINGS if not hasattr(config, name)]
