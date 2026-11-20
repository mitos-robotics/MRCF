"""
check_camera.py — standalone camera sanity check for the arm that runs (MRCF_ARM): every camera it opens.

Connects just long enough to grab a few frames from each configured camera
(gripper + overhead, if set up) and saves the last frame of each to disk,
with shape/brightness stats printed. No VLM, no planner, no arm motion — this
isolates "is the camera actually working" from "is mrcf/perception/planning
working", since a plan that hinges on seeing the table can fail for either
reason.

Usage:
    python tools/check_camera.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import connecting_to, make_env, rc as cfg  # noqa: E402
from mrcf.robot_env import RobotEnv  # noqa: E402
from mrcf import privacy  # noqa: E402

OUT_DIR = Path(__file__).parent


def _check_one(env: RobotEnv, camera_name: str, width: int, height: int) -> None:
    print(f"-- camera '{camera_name}' --")
    frame = None
    for i in range(5):
        frame = env.render_image(camera_name, width=width, height=height)
        print(f"  frame {i + 1}: shape={frame.shape} dtype={frame.dtype} "
              f"mean={frame.mean():.1f} min={frame.min()} max={frame.max()}")
        env.step(1)

    out_path = OUT_DIR / f"check_camera_frame_{camera_name}.png"
    privacy.save_rgb(out_path, frame)
    print(f"  Saved last frame to {out_path} — open it and check it actually shows what you expect.")

    if frame.mean() < 5:
        print("  WARNING: frame is almost entirely black — check the camera index, cable, or lens cap.")
    elif frame.mean() > 250:
        print("  WARNING: frame is almost entirely white — check exposure/lighting or a stuck camera.")
    print()


def main() -> None:
    have_overhead = cfg.OVERHEAD_CAMERA_NAME is not None
    env = make_env()
    print(f"Connecting to {connecting_to()}...")
    env.connect()
    print("Connected. Grabbing frames (arm will not move)...\n")

    try:
        cameras = env.list_cameras()
        if cfg.CAMERA_NAME is None:
            print("(CAMERA_NAME is None: no wrist camera)")
        if not have_overhead:
            print("(OVERHEAD_CAMERA_NAME is None: no overhead camera)")
        if not cameras:
            print("No cameras open: give each one a source in the arm's robot_config CAMERAS.")
        for name in cameras:
            # The wrist camera at its capture size; every other at RENDER_W/H —
            # render_image() resizes to whatever concrete size is asked for,
            # whatever the camera's native one.
            wh = (cfg.CAMERA_W, cfg.CAMERA_H) if name == cfg.CAMERA_NAME else (cfg.RENDER_W, cfg.RENDER_H)
            _check_one(env, name, *wh)
    finally:
        env.stop()


if __name__ == "__main__":
    main()
