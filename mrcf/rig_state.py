"""
rig_state.py — where this machine keeps what it measures and downloads for a
robot: the fixed camera's calibration, the photo of its empty work surface,
its URDF, what its calibration tools save to fit again. One folder per
backend, rig/<backend>/, beside the code rather than in it: none of it is in
git (another rig's would silently override the values its owner measured),
and the code can move without leaving it behind.

    rig_state.rig_dir("so101")       <MRCF_HOME>/rig/so101

MRCF_HOME names the folder that holds rig/. By default it is the repository
mrcf/ is in, when it runs from one (a pyproject.toml beside it), else — mrcf
installed into an environment — ~/.mrcf. A backend's robot_config asks for its
folder by its own name, so a copy of backends/_template gets one of its own.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "MRCF_HOME"

# What a backend reads as it starts. Left in its code folder by an older
# layout, it would be passed over for robot_config's older values without a
# word, and every square would read off — so rig_dir stops.
STARTUP_STATE = ("overhead_calibration.json", "surface_reference.png", "urdf")


def home() -> Path:
    """The folder that holds rig/: MRCF_HOME; else the repository mrcf/ is in; else, installed, ~/.mrcf."""
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR])
    repo = Path(__file__).resolve().parent.parent
    return repo if (repo / "pyproject.toml").is_file() else Path.home() / ".mrcf"


def rig_dir(backend: str, code_dir: "Path | None" = None) -> Path:
    """
    rig/<backend>/ under home() — not made here: whatever writes into it makes
    it. code_dir: the backend's own folder. If that — or where a backend's
    folder was before mrcf/, home()/backends/<backend>/, which git leaves
    behind holding what it never tracked — still holds what it reads as it
    starts, stop, rather than start from older values:
    tools/migrate_rig_state.py moves it.
    """
    path = home() / "rig" / backend
    for folder in ([Path(code_dir)] if code_dir is not None else []) + [home() / "backends" / backend]:
        left = [name for name in STARTUP_STATE if (folder / name).exists()]
        if left:
            raise RuntimeError(
                f"{folder} still holds this rig's {', '.join(left)}, which is read from {path} now. "
                "Move it there: python tools/migrate_rig_state.py --apply (without --apply it lists what it "
                "would move).")
    return path
