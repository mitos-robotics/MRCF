"""
fetch_urdf.py — one-time download of the SO-101 URDF and its mesh assets.

The URDF is needed for Cartesian control (get_ee_pos / move_ee_to, used by
pick/place/move_above skills), via lerobot's placo-based RobotKinematics. NOT
needed for go_home(), the gripper, or the camera — those work without this.
Files already there are not fetched again.

Source: TheRobotStudio/SO-ARM100 (Apache-2.0), the official SO-101 design repo.
Downloads ~19 MB into this rig's folder, rig/so101/urdf/ (rig_state.py; not
in git — a third-party asset, not something to commit).

Usage:
    pip install placo
    python tools/so101/fetch_urdf.py
"""

import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.so101._common import rc  # noqa: E402

# Pinned: the commit whose files the SO-101 rig was calibrated against (byte for
# byte, 2026-10-28), so a later change upstream cannot move a new arm's kinematics.
_COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"
_REPO = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{_COMMIT}/Simulation/SO101"
_DEST = Path(rc.URDF_PATH).parent             # rig/so101/urdf, where robot_config looks

_MODEL_FILES = [
    "so101_new_calib.urdf",     # the kinematics the real arm runs on (placo)
]
_ASSET_FILES = [
    "base_motor_holder_so101_v1.stl",
    "base_so101_v2.stl",
    "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl",
    "moving_jaw_so101_v1.stl",
    "rotation_pitch_so101_v1.stl",
    "sts3215_03a_no_horn_v1.stl",
    "sts3215_03a_v1.stl",
    "under_arm_so101_v1.stl",
    "upper_arm_so101_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl",
    "wrist_roll_follower_so101_v1.stl",
    "wrist_roll_pitch_so101_v2.stl",
]


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {dest.relative_to(_DEST.parent)}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 — fixed, known-good GitHub raw URL


def main() -> None:
    print(f"Downloading SO-101 URDF + meshes (~19 MB) from TheRobotStudio/SO-ARM100 into {_DEST}/")
    jobs = [(f"{_REPO}/{name}", _DEST / name) for name in _MODEL_FILES]
    jobs += [(f"{_REPO}/assets/{name}", _DEST / "assets" / name) for name in _ASSET_FILES]
    todo = [job for job in jobs if not job[1].exists()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda job: _fetch(*job), todo))
    print(f"Done. {len(todo)} files fetched, {len(jobs) - len(todo)} already there. "
          "robot_config's URDF_PATH already points here.")


if __name__ == "__main__":
    main()
