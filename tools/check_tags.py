"""
check_tags.py — can the cameras read the tags and jaw marks? No arm motion.

Opens the arm's cameras directly (the arm does not need to be on), grabs a
few frames from each, and reports, for every camera in its CAMERAS:

  a fixed one     which board tags are found, and how reliably; any that are
                  not (out of frame, hidden, glare); the gripper tag if the arm
                  is holding it in view; and, once the camera's lens is
                  calibrated, where it is relative to the board, worked out
                  from the board tags themselves — so it can be run while the
                  camera is being re-aimed.
  one on the      the two jaw marks (left = fixed jaw, right = moving jaw), how
  gripper         steady they are frame to frame, and the gap between them.

Annotated frames are saved next to this script (check_tags_<camera>.png).

Usage:
    python tools/check_tags.py
    python tools/check_tags.py --frames 30
    python tools/check_tags.py --camera overhead     # just that one (overhead, wrist, or a name)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mrcf.perception import fiducials  # noqa: E402
from tools._common import (camera_entry, camera_model_of, camera_size, camera_source, chosen_camera, rc,  # noqa: E402
                           task_config)
from mrcf import privacy  # noqa: E402

cfg = task_config()
HERE = Path(__file__).parent
WARMUP_FRAMES = 30          # let auto exposure settle before measuring


def grab(index: int, width: int, height: int, n: int) -> list:
    """n frames (BGR) straight from the camera, after a warm-up."""
    from tools._common import refuse_personal_cameras

    refuse_personal_cameras()                  # never this computer's own camera
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frames = []
    try:
        for i in range(WARMUP_FRAMES + n):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"Camera {index} gave no frame — is something else using it?")
            if i >= WARMUP_FRAMES:
                frames.append(frame)
    finally:
        cap.release()
    if frames[0].shape[1] != width or frames[0].shape[0] != height:
        raise SystemExit(f"Camera {index} gives {frames[0].shape[1]}x{frames[0].shape[0]}, "
                         f"config expects {width}x{height}.")
    return frames


def board_plane(poses: dict) -> "tuple[np.ndarray, np.ndarray, float] | None":
    """
    The board's plane in the camera frame from the board tags' poses: (point,
    unit normal toward the camera, worst tag off it in m). Each tag has two
    candidate poses; the one whose face agrees with the others' is used.
    """
    if len(poses) < 2:
        return None
    normals = np.array([c[0].R[:, 2] for c in poses.values()])
    median = np.median(normals, axis=0)
    chosen = {i: fiducials.pick_pose(c, R_expected=None) if len(c) == 1 else
              max(c, key=lambda p: p.R[:, 2] @ median) for i, c in poses.items()}
    centres = np.array([p.t for p in chosen.values()])
    n = np.mean([p.R[:, 2] for p in chosen.values()], axis=0)
    if len(centres) >= 3:
        _, _, vt = np.linalg.svd(centres - centres.mean(0))
        n = vt[2] if vt[2] @ n > 0 else -vt[2]
    n /= np.linalg.norm(n)
    if n @ centres.mean(0) > 0:          # face the camera (origin)
        n = -n
    off = np.abs((centres - centres.mean(0)) @ n).max()
    return centres.mean(0), n, float(off)


def check_fixed(camera: str, n: int) -> None:
    model = camera_model_of(camera)
    K, D = (None, None) if model is None else (np.array(model["K"]), np.array(model["D"]))
    size = {i: cfg.BOARD_TAG_SIZE_M for i in cfg.BOARD_TAG_IDS}
    size.update(getattr(rc, "GRIPPER_TAG_SIZES_M", None) or {})
    frames = grab(camera_source(camera), *camera_size(camera), n)
    seen: dict[int, list] = {}
    for f in frames:
        for i, q in fiducials.detect_tags(f).items():
            seen.setdefault(i, []).append(q)
    print(f"{camera.upper()} camera, fixed (source {camera_source(camera)!r}), {n} frames"
          + ("" if model is not None else " — lens not calibrated: tags counted, not placed"))
    board_poses = {}
    for i in cfg.BOARD_TAG_IDS:
        qs = seen.get(i, [])
        if not qs:
            print(f"  board tag {i}: NOT FOUND — out of frame, hidden, or washed out")
            continue
        q = np.median(qs, axis=0)
        jitter = np.median([np.abs(x - q).max() for x in qs])
        side = np.linalg.norm(q[1] - q[0])
        poses = fiducials.tag_poses(q, size[i], K, D) if K is not None else []
        if poses:
            board_poses[i] = poses
        print(f"  board tag {i}: {len(qs)}/{n} frames, {side:4.0f} px across, at ({q[:, 0].mean():4.0f},"
              f"{q[:, 1].mean():4.0f}), corner jitter {jitter:.2f} px")
    found = len(board_poses) if K is not None else len([i for i in cfg.BOARD_TAG_IDS if seen.get(i)])
    print(f"  -> {found}/{len(cfg.BOARD_TAG_IDS)} board tags in view"
          + ("" if found == len(cfg.BOARD_TAG_IDS) else "; aim the camera so all of them are, then run this again"))
    plane = board_plane(board_poses)
    if plane is not None:
        p0, nrm, off = plane
        height = float(-nrm @ p0)
        dist = float(np.linalg.norm(p0))
        axis_angle = float(np.degrees(np.arcsin(min(1.0, abs(nrm[2])))))
        print(f"  camera: {height * 1000:.0f} mm above the board, {dist * 1000:.0f} mm from the middle of the "
              f"tags in view, looking at them {np.degrees(np.arcsin(height / dist)):.0f} deg down "
              f"(optical axis {axis_angle:.0f} deg down); tags agree on one plane within {off * 1000:.1f} mm")
    g = rc.GRIPPER_TAG_ID
    qs = seen.get(g, []) if g is not None else []
    if g is None:
        pass
    elif qs:
        q = np.median(qs, axis=0)
        poses = fiducials.tag_poses(q, size[g], K, D) if K is not None and g in size else []
        where = ""
        if poses:
            pose = poses[0]
            facing = np.degrees(np.arccos(abs(pose.R[:, 2] @ (-pose.t / np.linalg.norm(pose.t)))))
            where = f", {np.linalg.norm(pose.t) * 1000:.0f} mm away, {facing:.0f} deg from facing the camera"
        print(f"  gripper tag {g}: {len(qs)}/{n} frames, {np.linalg.norm(q[1] - q[0]):.0f} px across{where}")
    else:
        print(f"  gripper tag {g}: not in view (fine unless the arm is holding it toward the camera)")
    strays = sorted(set(seen) - set(cfg.BOARD_TAG_IDS) - {g})
    if strays:
        print(f"  other tag ids seen: {strays} — a spare left lying in view, or a misread")
    vis = frames[-1].copy()
    for i, qs in seen.items():
        q = np.median(qs, axis=0)
        colour = (0, 200, 255) if i == g else (0, 255, 0)
        cv2.polylines(vis, [q.astype(np.int32)], True, colour, 2)
        cv2.putText(vis, str(i), tuple(int(v) for v in q.mean(0) + [-8, -22]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colour, 2)
    out = HERE / f"check_tags_{camera}.png"
    privacy.imwrite(str(out), vis)
    print(f"  saved {out}")


def check_on_gripper(camera: str, n: int) -> None:
    w, h = camera_size(camera)
    frames = grab(camera_source(camera), w, h, n)
    per_frame = [fiducials.find_bullseyes(f) for f in frames]
    print(f"\n{camera.upper()} camera, on the gripper (source {camera_source(camera)!r}), {n} frames")
    left = [m[0].centre for m in per_frame if m and m[0].centre[0] < w / 2]
    right = [m[-1].centre for m in per_frame if m and m[-1].centre[0] >= w / 2]
    for name, pts in (("fixed jaw (left)", left), ("moving jaw (right)", right)):
        if not pts:
            print(f"  {name}: mark NOT FOUND")
            continue
        p = np.median(pts, axis=0)
        spread = np.median(np.linalg.norm(np.array(pts) - p, axis=1))
        print(f"  {name}: {len(pts)}/{n} frames at ({p[0]:5.1f},{p[1]:5.1f}) px, frame-to-frame {spread:.2f} px")
    if left and right:
        a, b = np.median(left, axis=0), np.median(right, axis=0)
        print(f"  marks {np.linalg.norm(b - a):.0f} px apart; midpoint ({(a[0] + b[0]) / 2:.1f},{(a[1] + b[1]) / 2:.1f}) px")
    vis = frames[-1].copy()
    for m in fiducials.find_bullseyes(frames[-1]):
        cv2.ellipse(vis, m.ellipse, (0, 255, 0), 2)
        cv2.drawMarker(vis, tuple(int(v) for v in m.centre), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    out = HERE / f"check_tags_{camera}.png"
    privacy.imwrite(str(out), vis)
    print(f"  saved {out}")


def main() -> None:
    n = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 10
    t0 = time.time()
    # Every camera with a source, fixed ones first; or the one --camera names.
    names = [chosen_camera()] if "--camera" in sys.argv else [
        name for name, cam in sorted((getattr(rc, "CAMERAS", None) or {}).items(), key=lambda nc: nc[1].mount != "fixed")
        if cam.source is not None]
    if not names:
        raise SystemExit("No camera with a source in the arm's CAMERAS to open.")
    for name in names:
        (check_fixed if camera_entry(name) is None or camera_entry(name).mount == "fixed" else check_on_gripper)(name, n)
    print(f"\n({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
