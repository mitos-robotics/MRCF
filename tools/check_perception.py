"""
check_perception.py — standalone perception check for the arm that runs (MRCF_ARM).

Runs ONLY the perception half of the pipeline — no planner, no safety layer,
no motion beyond parking the arm clear of the camera — and prints exactly what
the VLM sees, what it names things, and where each name back-projects to in
the robot's own frame.

This isolates the three things that look identical from the REPL when a pick
never happens:
  1. The VLM doesn't see the object at all.
  2. It sees it but reports a pixel that isn't actually on it (VLMs are weak
     at precise pixel localisation) — so the world position is wrong.
  3. It sees it and localises it fine, but the position lands outside
     WORKSPACE_BOUNDS, so the safety layer blocks every action on it.

It writes an annotated image with each detected object circled at the pixel
the VLM claimed, so (2) is visible at a glance rather than inferred.

Usage:
    python tools/check_perception.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from tools._common import connecting_to, make_env, task_config  # noqa: E402

cfg = task_config()   # chess-style tasks supply ARM_KEYWORDS, models, OBSERVE_CAMERA
from mrcf import vlm as _vlm  # noqa: E402
from mrcf.vlm import SceneState, discover_scene, perceive  # noqa: E402
from mrcf import privacy  # noqa: E402

# The deployment's skills (deployments.<name>.skills): its observe pose; importing them also registers its actions.
goto_observe_pose = importlib.import_module(f"{cfg.__package__}.skills").goto_observe_pose

OUT_PATH = Path(__file__).parent / "check_perception_overlay.png"


def _in_workspace(pos) -> bool:
    b = cfg.WORKSPACE_BOUNDS
    return (b.x_min <= pos[0] <= b.x_max
            and b.y_min <= pos[1] <= b.y_max
            and b.z_min - 0.01 <= pos[2] <= b.z_max + 0.01)


def main() -> None:
    env = make_env()

    _vlm.configure(
        model_server=cfg.MODEL_SERVER,
        model_url=cfg.MODEL_URL,
        model=cfg.MODEL,
        planning_model=cfg.PLANNING_MODEL,
        observe_cameras=[cfg.OBSERVE_CAMERA],
        render_w=cfg.RENDER_W,
        render_h=cfg.RENDER_H,
        work_plane_z=cfg.TABLE_Z,
    )

    print(f"Connecting to {connecting_to()}...")
    env.connect()
    print(f"Connected. Primary perception camera: '{cfg.OBSERVE_CAMERA}' "
          f"at {cfg.RENDER_W}x{cfg.RENDER_H}\n")

    try:
        print("Parking the arm clear of the camera...")
        goto_observe_pose(env)

        print("\n--- perceive() free-text description ---")
        print(perceive(env) or "(empty)")

        print("\n--- discover_scene() structured output ---")
        scene_state = SceneState()
        discover_scene(env, scene_state, exclude_keywords=cfg.ARM_KEYWORDS)

        found = scene_state.all_objects()
        if not found:
            print("\nNOTHING DETECTED. The VLM returned no objects for this frame — so the "
                  "planner has nothing to pick, and every pick/place will be blocked.")
        for o in found:
            pos_str = (f"[{o.position[0]:.3f}, {o.position[1]:.3f}, {o.position[2]:.3f}]"
                       if o.position is not None else "None")
            flag = ""
            if o.position is not None and not _in_workspace(o.position):
                flag = "  <-- OUTSIDE WORKSPACE_BOUNDS: every action on this is blocked"
            print(f"  {o.name:22s} pixel={str(o.pixel_xy):14s} world={pos_str}{flag}")
            print(f"  {'':22s} desc: {o.description}")

        # Annotated overlay: did the VLM's pixel actually land on the object?
        frame = env.render_image(cfg.OBSERVE_CAMERA, width=cfg.RENDER_W, height=cfg.RENDER_H)
        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for o in found:
            if not o.pixel_xy or len(o.pixel_xy) != 2:
                continue
            u, v = int(o.pixel_xy[0]), int(o.pixel_xy[1])
            cv2.circle(canvas, (u, v), 12, (0, 0, 255), 2)
            cv2.putText(canvas, o.name, (u + 15, v), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
        privacy.imwrite(str(OUT_PATH), canvas)
        print(f"\nSaved annotated frame to {OUT_PATH}")
        print("Open it: each red circle is where the VLM claimed that object is. If the "
              "circles don't sit on the real objects, the pixel estimates are the problem, "
              "not the homography.")
    finally:
        env.stop()


if __name__ == "__main__":
    main()
