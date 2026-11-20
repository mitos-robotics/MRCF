"""
new_arm.py — start a new arm: its backend package, copied from
mrcf/backends/_template.

As copied it is an arm that exists only in memory (its env.InMemoryArm) and
keeps the backend contract, so run.py can already take it, show its rig and
make it a deployment; every place to change for the real arm is marked YOUR
ARM. From a factory's side, the order: this arm (its SDK in env.py, its
measured values in robot_config.py), then its sensors (cameras in CAMERAS,
calibrated; anything else in SENSORS), then a deployment for the job
(tools/new_deployment.py, or run.py's "new"). DEPLOYMENT_SPEC.md, "Adding an
arm"; COMMANDS.md, "First-time setup on another arm".

    python tools/new_arm.py NAME [--label "Its Name"]

Writes files only; connects to nothing, moves nothing.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ROOT / "mrcf" / "backends"


def create(name: str, label: "str | None" = None, backends_dir: Path = BACKENDS) -> Path:
    """mrcf/backends/<name>/ from _template, NAME set to label; its folder. SystemExit: why not."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise SystemExit(f"'{name}': name an arm as a Python module (lower-case letters, digits, underscores)")
    dst = backends_dir / name
    if dst.exists():
        raise SystemExit(f"{dst} exists: not overwriting it")
    shutil.copytree(backends_dir / "_template", dst, ignore=shutil.ignore_patterns("__pycache__"))
    init = dst / "__init__.py"
    text = init.read_text()
    shown = label or name.replace("_", " ").title()
    text = re.sub(r'^NAME = "[^"]*"', f'NAME = "{shown}"', text, count=1, flags=re.M)
    text = text.replace("mrcf/backends/_template — a new arm's package, to copy:",
                        f"mrcf/backends/{name} — {shown}, started from mrcf/backends/_template:", 1)
    init.write_text(text)
    return dst


def next_steps(name: str) -> str:
    """What to do after create(), in order."""
    return (f"Made mrcf/backends/{name}/ — an arm in memory until its SDK is in:\n"
            f"  1. the arm      env.py's make_sdk() (YOUR ARM), robot_config.py's measured values;\n"
            f"                  python tools/check_motion.py (MRCF_ARM={name}) for its first moves\n"
            f"  2. its sensors  CAMERAS and SENSORS in robot_config.py; python tools/find_cameras.py;\n"
            f"                  the overhead camera calibrated (tools/calibrate_camera_by_hand.py)\n"
            f"  3. the job      a deployment: python run.py --arm {name}, then 'n' (or tools/new_deployment.py)\n"
            "COMMANDS.md, 'First-time setup on another arm'.")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    label = None
    if "--label" in argv:
        i = argv.index("--label")
        label = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    if not argv:
        raise SystemExit('python tools/new_arm.py NAME [--label "Its Name"]')
    create(argv[0], label)
    print(next_steps(argv[0]))


if __name__ == "__main__":
    main()
