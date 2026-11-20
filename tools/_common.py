"""
Shared plumbing for the framework's tools: the ones that work on whichever arm
runs (MRCF_ARM, backends.arm()) — its env, its settings, its policies — rather
than on one arm. An arm's own tools (tools/<arm>/: its poses, its motors, its
joint calibration) have a _common of their own.

A few tools need task values (the board, its tags, grasp heights). They take
`--deployment NAME`, and with only one deployment present they use it.
"""

import os
import subprocess
import sys
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

import numpy as np  # noqa: E402

from mrcf import backends  # noqa: E402
from mrcf.capabilities import OrientationSearch  # noqa: E402

# And, from __getattr__ below: ARM, rc, make_env, make_policies.
__all__ = ["connecting_to", "chosen_camera", "fixed_camera", "camera_entry", "camera_source", "camera_size", "camera_model_of",
           "orientation_for", "deployment_config", "task_config", "show_image", "arg", "make_env_without_gripper",
           "gripper_box", "FRAMEWORK_ROOT"]


def __getattr__(name: str):
    """
    ARM (the arm's package), rc (its robot_config), make_env, make_policies —
    looked up when a tool first asks, so an arm's own tools can share the task
    helpers below without choosing an arm.
    """
    arm = backends.arm()
    found = {"ARM": arm, "rc": arm.config, "make_env": arm.make_env, "make_policies": arm.make_policies}
    if name in found:
        return found[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def arg(flag: str, default=None):
    """The value given after flag on the command line, else default."""
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv[:-1] else default


def make_env_without_gripper():
    """
    The arm's env with no gripper and no sentry, for a tool that neither opens
    nor closes the jaws and commands no move while hands are meant to be in the
    camera's view: an arm whose gripper is not ready (a Robotiq not yet
    activated) must not refuse to connect for it. An arm whose make_env()
    takes no gripper= is made without that.
    """
    make_env = backends.arm().make_env
    try:
        return make_env(gripper=None, motion_guard_zone=None)
    except TypeError as exc:
        if "gripper" not in str(exc):
            raise
        return make_env(motion_guard_zone=None)


def gripper_box() -> "tuple | None":
    """The box the arm keeps its whole gripper inside (config GRIPPER_BOX, envelope.py), ((x0, x1), (y0, y1), (z0, z1)) m; None: it has none."""
    box = getattr(backends.arm().config, "GRIPPER_BOX", None)
    if box is None:
        return None
    return tuple((float(lo), float(hi)) for lo, hi in box)


def connecting_to() -> str:
    """What the arm connects to, for "Connecting to ..." (the arm's describe())."""
    arm = backends.arm()
    return arm.describe() if hasattr(arm, "describe") else f"the {arm.NAME}"


# ── Cameras: any of the arm's CAMERAS, by name ────────────────────────────────
# "overhead" and "wrist" stand for the two with jobs (OVERHEAD_CAMERA_NAME,
# CAMERA_NAME), whatever the rig calls them.

def chosen_camera(default: str = "overhead") -> str:
    """The camera a tool works with: `--camera NAME` from the command line, else `default`."""
    name = sys.argv[sys.argv.index("--camera") + 1] if "--camera" in sys.argv[:-1] else default
    return resolve_camera(name)


def resolve_camera(name: str) -> str:
    """name as the arm's CAMERAS has it: "overhead" and "wrist" are the cameras with those jobs."""
    rc = backends.arm().config
    role = {"overhead": rc.OVERHEAD_CAMERA_NAME, "wrist": rc.CAMERA_NAME}
    known = getattr(rc, "CAMERAS", None) or {}
    if name in known:
        return name
    if name in role:
        if role[name] is None:
            raise SystemExit(f"The {backends.arm().NAME} has no {name} camera "
                             f"({'OVERHEAD_CAMERA_NAME' if name == 'overhead' else 'CAMERA_NAME'} is None).")
        return role[name]
    raise SystemExit(f"No camera '{name}' on the {backends.arm().NAME}: its CAMERAS are {', '.join(known) or 'none'}.")


def fixed_camera(name: str) -> str:
    """name, if it is a fixed camera; stops the tool if it is on the gripper."""
    cam = camera_entry(name)
    if cam is not None and cam.mount != "fixed":
        raise SystemExit(f"'{name}' is on the gripper: this works with a fixed camera.")
    return name


def camera_entry(name: str):
    """The camera's backends.rig.Camera from the arm's CAMERAS (None: not listed)."""
    return (getattr(backends.arm().config, "CAMERAS", None) or {}).get(name)


def camera_source(name: str):
    """
    The camera's OpenCV index or URL, for tools that open it themselves without
    connecting the arm — checked first: never an index that is, or may now be,
    this computer's own camera (mrcf/devices.check_sources; SystemExit, why).
    """
    cam = camera_entry(name)
    if cam is None or cam.source is None:
        raise SystemExit(f"This tool opens the '{name}' camera itself, by its OpenCV index or URL: give it a "
                         f"source in the {backends.arm().NAME}'s robot_config CAMERAS.")
    refuse_personal_cameras()
    return cam.source


def refuse_personal_cameras() -> None:
    """SystemExit when any of the arm's cameras might open this computer's own (mrcf/devices.check_sources)."""
    from mrcf import devices

    cams = getattr(backends.arm().config, "CAMERAS", None) or {}
    try:
        devices.check_sources({n: c.source for n, c in cams.items() if c.source is not None})
    except devices.PersonalCamera as exc:
        raise SystemExit(f"Not opening a camera: {exc}") from None


def camera_size(name: str) -> "tuple[int, int]":
    """The camera's capture size: the overhead and wrist cameras' settings, else its CAMERAS entry."""
    rc = backends.arm().config
    if name == rc.OVERHEAD_CAMERA_NAME:
        return rc.OVERHEAD_CAMERA_W, rc.OVERHEAD_CAMERA_H
    if name == rc.CAMERA_NAME:
        return rc.CAMERA_W, rc.CAMERA_H
    cam = camera_entry(name)
    return tuple(cam.size) if cam is not None else (rc.RENDER_W, rc.RENDER_H)


def camera_model_of(name: str) -> "dict | None":
    """A fixed camera's calibrated lens and pose {size, K, D, T_world_cam}: the overhead one's
    OVERHEAD_CAM_MODEL, another's from its CAMERAS entry; None when it has none."""
    rc = backends.arm().config
    if name == rc.OVERHEAD_CAMERA_NAME:
        return rc.OVERHEAD_CAM_MODEL
    cam = camera_entry(name)
    return None if cam is None else cam.model


def orientation_for(env, target, tol: float = 0.01):
    """
    (rotation, lean in degrees) the arm reaches target with, or None when it
    cannot. The arm's own search when it has one — one target at a time
    (approach_orientation, the SO-101's) or several (approach_orientation_for,
    capabilities.OrientationSearch); otherwise (None, 0.0): the arm's own
    default, straight down.
    """
    target = np.asarray(target, float)
    if hasattr(env, "approach_orientation"):
        return env.approach_orientation(target, tol=tol)
    if isinstance(env, OrientationSearch):
        return env.approach_orientation_for([target], tols=[tol])
    return None, 0.0


def deployment_config(name: str):
    """
    Import a deployment's config module by folder name, e.g. "chess": the
    module deployments.<name>.config. Its package (`cfg.__package__`) gives
    the deployment's other modules, e.g.
    importlib.import_module(f"{cfg.__package__}.piece_vision").

    A few tools need task values a robot cannot supply — the grasp heights a
    plan preview walks through, the calibration target's dimensions. Rather
    than hardcoding one task's numbers into a robot-level tool, they ask for
    the deployment by name.
    """
    import importlib

    path = FRAMEWORK_ROOT / "deployments" / name / "config.py"
    if not path.exists():
        available = sorted(
            p.name for p in (FRAMEWORK_ROOT / "deployments").iterdir()
            if (p / "config.py").exists()
        )
        raise SystemExit(
            f"No deployment '{name}' (looked for {path}).\n"
            f"Available: {', '.join(available) or '(none)'}")
    cfg = importlib.import_module(f"deployments.{name}.config")
    # A deployment is one arm, one layout and one task: another arm's values
    # (its surface, its rules for its arm) are not this arm's.
    its_arm, running = getattr(cfg, "ARM_NAME", None), backends.arm_name()
    if its_arm is not None and its_arm != running:
        raise SystemExit(f"Deployment '{name}' is the {its_arm}'s, and this tool is measuring the {running} "
                         f"({backends.ENV_VAR}). Run it with {backends.ENV_VAR}={its_arm}, or name one of "
                         f"the {running}'s deployments.")
    return cfg


def _last_run() -> "str | None":
    """The deployment run.py last launched on this arm (rig/<arm>/last_deployment.txt), if it still exists."""
    try:
        path = Path(getattr(backends.arm().config, "RIG_DIR")) / "last_deployment.txt"
        name = path.read_text().strip() if path.exists() else ""
    except Exception:                                         # noqa: BLE001 — none known
        return None
    return name if name and (FRAMEWORK_ROOT / "deployments" / name / "config.py").exists() else None


def task_config(name: "str | None" = None):
    """
    The deployment config a robot-level tool needs task values from.

    Resolution order: an explicit `name`, then a `--deployment X` flag in
    argv, then MRCF_DEPLOYMENT, then the task the arm's config names as its own
    (DEPLOYMENT: chess on the SO-101), then — if there is exactly one
    deployment — that one. With several deployments and nothing to choose by,
    it stops and lists them rather than guessing, because picking the wrong
    task silently gives the tool the wrong grasp heights or the wrong
    calibration target.
    """
    if name is None and "--deployment" in sys.argv:
        i = sys.argv.index("--deployment")
        if i + 1 < len(sys.argv):
            name = sys.argv[i + 1]
    if name is None:
        name = os.environ.get("MRCF_DEPLOYMENT", "").strip() or _last_run() or \
            getattr(backends.arm().config, "DEPLOYMENT", None)
    if name is None:
        found = sorted(p.name for p in (FRAMEWORK_ROOT / "deployments").iterdir()
                       if (p / "config.py").exists())
        if len(found) == 1:
            name = found[0]
        else:
            raise SystemExit(
                "This tool needs task values (grasp heights, board size) from a "
                "deployment.\n"
                f"Pass --deployment NAME. Available: {', '.join(found) or '(none)'}")
    return deployment_config(name)


def show_image(path, enabled: bool = True) -> None:
    """
    Open a saved frame in the system viewer.

    These tools ask you to look at a frame and then confirm or correct what was
    detected, and a path printed in a terminal means digging through a folder
    mid-run with the arm sitting powered at a prompt. Opening it is the whole
    point of having saved it.

    Never fatal: on a headless box, over SSH, or with no viewer configured
    there is simply nothing to open, and that must not take the calibration
    down with it. Set MRCF_NO_IMAGE_OPEN=1 to suppress it.
    """
    if not enabled or os.environ.get("MRCF_NO_IMAGE_OPEN"):
        return
    path = Path(path)
    if not path.is_file():
        return
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))   # type: ignore[attr-defined]
    except Exception:
        pass    # a viewer that will not open is not a reason to stop measuring
