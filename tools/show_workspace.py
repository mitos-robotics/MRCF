"""
show_workspace.py — draw the arm's box on what a fixed camera sees: the box
the gripper must stay inside (GRIPPER_BOX), a 10 cm grid on the table in the arm's
own frame (labelled, so a box edge can be named by where it should be), and
the arm's base axes. Moves nothing.

    python tools/show_workspace.py                         # the latest tools/calibrate_camera_by_hand.py solution
    python tools/show_workspace.py --box -0.5 0.5 -0.2 0.8 0 0.9   # try a box: x0 x1 y0 y1 z0 z1 (m, arm frame)
    python tools/show_workspace.py --arm                   # and the arm's body, where its joints put it
                                                           # (connects; moves nothing): a check on the calibration
    python tools/show_workspace.py --solution reports/camera_by_hand/<run>/solution.json
    python tools/show_workspace.py --camera side

The box: --box, else the arm's GRIPPER_BOX, else its WORKSPACE_BOUNDS. The
camera: --solution, else its calibration in the arm's config, else the latest
tools/calibrate_camera_by_hand.py run's. Saves reports/workspace/<time>.png and opens it.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mrcf.perception import camera_from_arm as cfa  # noqa: E402
from tools._common import (FRAMEWORK_ROOT, camera_model_of, camera_size, chosen_camera, fixed_camera,  # noqa: E402
                           gripper_box, make_env_without_gripper, rc, show_image)
from mrcf import privacy  # noqa: E402

OUT = FRAMEWORK_ROOT / "reports" / "workspace"
BOX_COLOUR = (255, 140, 0)          # RGB
GRID_COLOUR = (255, 255, 255)
AXES = ((255, 60, 60), (60, 220, 60), (80, 140, 255))



def chosen_box() -> "tuple[tuple[float, float], ...]":
    if "--box" in sys.argv:
        i = sys.argv.index("--box")
        v = [float(x) for x in sys.argv[i + 1:i + 7]]
        if len(v) != 6:
            raise SystemExit("--box takes six numbers: x0 x1 y0 y1 z0 z1 (metres, the arm's frame)")
        return (v[0], v[1]), (v[2], v[3]), (v[4], v[5])
    box = gripper_box()
    if box is not None:
        return box
    b = rc.WORKSPACE_BOUNDS
    return (b.x_min, b.x_max), (b.y_min, b.y_max), (b.z_min, b.z_max)


def chosen_model(camera: str) -> "tuple[dict, str]":
    """The camera's lens and pose: --solution's, else the arm's config (the calibration in use), else the latest run's."""
    given = sys.argv[sys.argv.index("--solution") + 1] if "--solution" in sys.argv[:-1] else None
    if given is not None:
        return json.loads(Path(given).read_text())["model"], given
    model = camera_model_of(camera)
    if model and model.get("T_world_cam") is not None:
        return model, f"the {camera} camera's calibration in the arm's config"
    runs = sorted((FRAMEWORK_ROOT / "reports" / "camera_by_hand").glob("*/solution.json"))
    runs = [r for r in runs if json.loads(r.read_text()).get("camera") == camera]
    if not runs:
        raise SystemExit(f"No calibration for '{camera}': run tools/calibrate_camera_by_hand.py first.")
    return json.loads(runs[-1].read_text())["model"], str(runs[-1])


def _segment(a, b, n: int = 40) -> np.ndarray:
    """A straight edge as points (the lens bends it, so it is drawn as a polyline)."""
    return np.linspace(np.asarray(a, float), np.asarray(b, float), n)


def draw_polyline(img, world_pts, model, colour, thickness=2) -> None:
    px, front = cfa.project(world_pts, model["T_world_cam"], model["K"], model["D"])
    h, w = img.shape[:2]
    inside = front & (px[:, 0] > -w) & (px[:, 0] < 2 * w) & (px[:, 1] > -h) & (px[:, 1] < 2 * h)
    for i in range(len(px) - 1):
        if inside[i] and inside[i + 1]:
            cv2.line(img, tuple(np.round(px[i]).astype(int)), tuple(np.round(px[i + 1]).astype(int)), colour,
                     thickness, cv2.LINE_AA)


def label(img, world_pt, text, model, colour=(255, 255, 255), scale=0.45) -> None:
    px, front = cfa.project([world_pt], model["T_world_cam"], model["K"], model["D"])
    if front[0]:
        p = tuple(np.round(px[0]).astype(int) + [3, -3])
        cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def render(img, model, box, table_z: float, body=None) -> np.ndarray:
    out = img.copy()
    (x0, x1), (y0, y1), (z0, z1) = box
    # The table grid, 10 cm, a little past the box; labelled every 20 cm.
    gx0, gx1 = np.floor(min(x0, -0.2) * 10) / 10 - 0.1, np.ceil(max(x1, 0.2) * 10) / 10 + 0.1
    gy0, gy1 = np.floor(min(y0, -0.2) * 10) / 10 - 0.1, np.ceil(max(y1, 0.2) * 10) / 10 + 0.1
    grid = out.copy()
    for x in np.arange(gx0, gx1 + 1e-9, 0.1):
        draw_polyline(grid, _segment((x, gy0, table_z), (x, gy1, table_z)), model, GRID_COLOUR, 1)
    for y in np.arange(gy0, gy1 + 1e-9, 0.1):
        draw_polyline(grid, _segment((gx0, y, table_z), (gx1, y, table_z)), model, GRID_COLOUR, 1)
    out = cv2.addWeighted(grid, 0.45, out, 0.55, 0)
    # Every 20 cm crossing labelled where it is (x, y in metres), so a spot
    # can be named by what is drawn beside it.
    h, w = out.shape[:2]
    for x in np.arange(np.ceil(gx0 / 0.2) * 0.2, gx1 + 1e-9, 0.2):
        for y in np.arange(np.ceil(gy0 / 0.2) * 0.2, gy1 + 1e-9, 0.2):
            px, ok = cfa.project([(x, y, table_z)], model["T_world_cam"], model["K"], model["D"])
            if ok[0] and 0 <= px[0, 0] < w - 60 and 10 <= px[0, 1] < h:
                cv2.circle(out, tuple(np.round(px[0]).astype(int)), 3, GRID_COLOUR, -1, cv2.LINE_AA)
                label(out, (x, y, table_z), f"{x + 0.0:+.1f},{y + 0.0:+.1f}", model, scale=0.38)
    # The box: its floor tinted, its twelve edges.
    floor = out.copy()
    pts, front = cfa.project([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], model["T_world_cam"],
                             model["K"], model["D"])
    if front.all():
        cv2.fillPoly(floor, [np.round(pts).astype(np.int32)], BOX_COLOUR)
        out = cv2.addWeighted(floor, 0.18, out, 0.82, 0)
    corners = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    for i, a in enumerate(corners):
        for j, b in enumerate(corners):
            if j > i and sum(u != v for u, v in zip(a, b)) == 1:
                draw_polyline(out, _segment(a, b), model, BOX_COLOUR, 3 if a[2] == z0 and b[2] == z0 else 2)
    label(out, (x1, y1, z1), f"top z {z1:.2f} m", model, BOX_COLOUR, 0.5)
    # The arm's base axes, 20 cm.
    for k, colour in enumerate(AXES):
        end = np.zeros(3)
        end[k] = 0.2
        draw_polyline(out, _segment((0, 0, table_z), end + [0, 0, table_z]), model, colour, 3)
        label(out, end + [0, 0, table_z], "xyz"[k], model, colour, 0.6)
    if body is not None:                                    # where the joints say the arm is
        for (a, _ra), (b, _rb) in zip(body[:-1], body[1:]):
            draw_polyline(out, _segment(a, b, 10), model, (0, 255, 255), 2)
        for p, _r in body:
            px, front = cfa.project([p], model["T_world_cam"], model["K"], model["D"])
            if front[0]:
                cv2.circle(out, tuple(np.round(px[0]).astype(int)), 4, (0, 255, 255), -1, cv2.LINE_AA)
    return out


def main() -> None:
    camera = fixed_camera(chosen_camera())
    model, source = chosen_model(camera)
    size = tuple(model.get("size") or camera_size(camera))
    box = chosen_box()
    with_arm = "--arm" in sys.argv
    env = make_env_without_gripper()
    try:
        body = None
        if with_arm:
            env.connect()                                   # to read the joints; nothing moves
            body = env.arm_shape()
        frame = env.render_image(camera, width=size[0], height=size[1])
    finally:
        env.stop()
    out = render(frame, model, box, float(rc.TABLE_Z), body)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}.png"
    privacy.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    (x0, x1), (y0, y1), (z0, z1) = box
    print(f"Box x {x0:+.2f}..{x1:+.2f}, y {y0:+.2f}..{y1:+.2f}, z {z0:+.2f}..{z1:+.2f} m (the arm's frame), "
          f"camera from {source}.\nSaved {path}")
    show_image(path)


if __name__ == "__main__":
    main()
