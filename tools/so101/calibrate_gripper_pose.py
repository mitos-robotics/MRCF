"""
calibrate_gripper_pose.py — where the gripper camera sits on the gripper.

Measures GRIPPER_CAM_POSE (gripper_frame_link -> camera) and
GRIPPER_CAM_PLANE_Z_M for robot_config.py. With them, every gripper-camera
frame comes with the camera's pose from the joint angles, so an object under
the jaws can be measured — height, widths, and where its foot is in the arm's
frame — on any surface, with no board in view (plane_geometry, hand_eye).

The chessboard is only the calibration target here: a flat grid whose corners
the overhead camera locates in the arm frame (OVERHEAD_CAM_MODEL). The arm
visits poses over it — straight down, then the wrist leant and rolled — and at
each records its joint angles and the grid corners the gripper camera sees.
hand_eye then finds the one camera mount that explains them all.

Before running:
  - the board flat in the overhead camera's view, where it was when
    OVERHEAD_CAM_MODEL was fitted, and CLEAR of pieces (the arm passes 13 cm
    above it, and every hidden corner is one fewer to fit)
  - OVERHEAD_CAM_MODEL set (or OVERHEAD_HOMOGRAPHY, as a fallback)

About 70 poses, 3-4 minutes. Frames and readings are saved, so the fit can be
re-run without the arm:

    python tools/so101/calibrate_gripper_pose.py
    python tools/so101/calibrate_gripper_pose.py --wide
    python tools/so101/calibrate_gripper_pose.py --fit rig/so101/gripper_pose_frames/<stamp> [<stamp> ...]

--wide visits where picks look from instead: lower (8 and 12 cm) and over the
whole board, near the base to rank 7, about 90 poses, 5 minutes. The joints
sag differently there, and a mount fitted only over the board's middle has to
hold at a pick's hover.

--target tags: no board — a deployment that works on a bare table has none.
The target is the table target sheet instead (tools/make_tag_sheets.py:
tag_sheets/table_target_<paper>.pdf, 12 AprilTags, printed at 100% and laid
flat, uncut, in the overhead camera's view where the arm reaches). The
overhead camera finds each tag's corners on the surface the sheet lies on
(--surface-z, else the deployment's measured surface with --deployment, else
the rig's table, TABLE_TOP_Z); the wrist camera sees the tags from the same
poses (round the sheet's middle); and the fit starts from the last good
calibration's mount and joint corrections (a tag sheet has no lattice to start
one from), so it needs one. The result is said as the board's would be: the
surface where the camera's frame has it, moved to TABLE_Z.

    python tools/so101/calibrate_gripper_pose.py --wide --target tags --deployment tidy
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2            # noqa: E402
import numpy as np    # noqa: E402

from tools.so101._common import make_env, rc  # noqa: E402
from mrcf import privacy  # noqa: E402

OUT = rc.RIG_DIR / "gripper_pose_frames"
CAPTURE_W, CAPTURE_H = 1280, 720            # overhead, as tools/calibrate_overhead_camera
MIN_CORNERS = 9
HEIGHTS_M = (0.13, 0.17)                    # grasp point above the table
OFFSETS_M = [(0, 0), (0.06, 0.06), (0.06, -0.06), (-0.06, 0.06), (-0.06, -0.06)]
# (wrist_flex, wrist_roll) added to the straight-down pose, degrees. Both
# kinds are needed: leaning alone leaves the camera's rotation about one axis
# unobservable (hand_eye.initial_from_plane).
WRIST_VARIATIONS = [(0, 0), (15, 0), (-15, 0), (0, 30), (0, -30), (12, 25), (-12, -25)]

# --wide: a pick's hover heights, from near the base to the far ranks.
WIDE_HEIGHTS_M = (0.08, 0.12)
WIDE_OFFSETS_M = [(dx, dy) for dx in (-0.10, 0.0, 0.10) for dy in (-0.10, 0.0, 0.10)]
WIDE_WRIST_VARIATIONS = [(0, 0), (12, 0), (-12, 0), (0, 25), (0, -25)]

_CB_FLAGS = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE


def detect_patch(frame_rgb: np.ndarray):
    """
    The largest patch of chessboard corners the camera sees: (corners (N, 2),
    lattice indices (N, 2) within the patch) or None. Several detector seeds
    and a contrast-equalised copy, because the detector is randomised and
    returns different patches of a partly visible board from call to call.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    best = None
    for img in (gray, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)):
        for seed in (0, 1, 2, 3):
            cv2.setRNGSeed(seed)
            ok, corners, meta = cv2.findChessboardCornersSBWithMeta(img, (3, 3), _CB_FLAGS)
            if ok and (best is None or len(corners) > len(best[0])):
                cols = meta.shape[1]
                c = corners.reshape(-1, 2).astype(float)
                ij = np.array([[i % cols, i // cols] for i in range(len(c))], float)
                best = (c, ij)
    return best


def grid_in_world(env) -> "tuple[np.ndarray, float] | None":
    """The board's internal corners in the arm frame, (N, 3) m, and their spacing, m."""
    from tools.calibrate_overhead_camera import _detect_chessboard
    from mrcf.perception.plane_geometry import PlaneView

    frame = env.render_image(rc.OVERHEAD_CAMERA_NAME, width=CAPTURE_W, height=CAPTURE_H)
    corners = _detect_chessboard(frame)
    if corners is None:
        return None
    model = env.camera_model(rc.OVERHEAD_CAMERA_NAME, CAPTURE_W, CAPTURE_H)
    if model is not None:
        view = PlaneView.from_camera_model(model)
        xy = view.to_plane(corners) / 1000.0
        z = model["plane_z_m"]
    else:
        print("  OVERHEAD_CAM_MODEL not set — locating the board with OVERHEAD_HOMOGRAPHY instead.")
        sx, sy = rc.RENDER_W / CAPTURE_W, rc.RENDER_H / CAPTURE_H
        xy = np.array([env.pixel_to_world(u * sx, v * sy, rc.TABLE_Z, rc.OVERHEAD_CAMERA_NAME,
                                          rc.RENDER_W, rc.RENDER_H)[:2] for u, v in corners])
        z = rc.TABLE_Z
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.c_[xy, np.full(len(xy), z)], float(np.median(d.min(1)))


def tags_in_world(env, surface_z: float) -> "dict[int, np.ndarray] | None":
    """
    The table target's tags the overhead camera sees, {id: (4, 3) corners in
    the arm frame, m} on the surface at surface_z; None without the camera's
    model. A tag whose corners do not come out its printed size apart is left
    out, said: the surface height or the camera's model is not right there.
    """
    from mrcf.perception import fiducials
    from mrcf.perception.plane_geometry import PlaneView

    frame = env.render_image(rc.OVERHEAD_CAMERA_NAME, width=CAPTURE_W, height=CAPTURE_H)
    model = env.camera_model(rc.OVERHEAD_CAMERA_NAME, CAPTURE_W, CAPTURE_H)
    if model is None:
        return None
    view = PlaneView.from_camera_model(dict(model, plane_z_m=surface_z))
    origin = np.zeros(2) if view.origin_m is None else np.asarray(view.origin_m, float)[:2]
    want = fiducials.TABLE_TARGET_TAG_M
    out = {}
    for tid, corners in fiducials.detect_tags(frame).items():
        if tid not in fiducials.TABLE_TARGET_IDS:
            continue
        xy = origin + view.to_plane(np.asarray(corners, float).reshape(4, 2)) / 1000.0
        side = float(np.mean(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)))
        if abs(side - want) > 0.004:
            print(f"  tag {tid}: its corners come out {side * 1000:.1f} mm apart, not {want * 1000:.0f} — left out")
            continue
        out[tid] = np.c_[xy, np.full(4, surface_z)]
    return out


def detect_target_tags(frame_rgb: np.ndarray, tags_world: dict) -> "tuple[np.ndarray, np.ndarray] | None":
    """The table target's corners the wrist camera sees and where each is: ((N, 2) px, (N, 3) m), or None."""
    from mrcf.perception import fiducials

    px, world = [], []
    for tid, corners in fiducials.detect_tags(frame_rgb).items():
        if tid in tags_world:
            px.append(np.asarray(corners, float).reshape(4, 2))
            world.append(tags_world[tid])
    if not px:
        return None
    return np.vstack(px), np.vstack(world)


def capture(env, grid_world: np.ndarray, folder: Path, heights=HEIGHTS_M, offsets=OFFSETS_M,
            variations=WRIST_VARIATIONS, tags: "dict | None" = None, surface_z: "float | None" = None) -> list:
    """
    Visit the poses, save what the gripper camera sees at each. The views'
    records. tags: the table target's corners in the world (tags_in_world) —
    each view then keeps the tags it sees and where they are, not a patch of
    the board's lattice. Heights are above surface_z (TABLE_Z if None).
    """
    base_z = rc.TABLE_Z if surface_z is None else float(surface_z)
    min_corners = 4 if tags is not None else MIN_CORNERS
    centre = grid_world[:, :2].mean(0)
    records = []
    n = len(heights) * len(offsets) * len(variations)
    k = 0
    for h in heights:
        for dx, dy in offsets:
            target = np.array([centre[0] + dx, centre[1] + dy, base_z + h])
            found = env.approach_orientation(target)
            if found is None or not env.move_ee_to(target, orientation=found[0]):
                print(f"  ({target[0]:.2f}, {target[1]:.2f}, h {h:.2f}) unreachable — skipped")
                k += len(variations)
                continue
            base = env.get_joint_degrees()
            for dflex, droll in variations:
                k += 1
                goal = dict(base, wrist_flex=base["wrist_flex"] + dflex,
                            wrist_roll=base["wrist_roll"] + droll)
                env.goto_joint_degrees(goal)
                time.sleep(0.4)                  # let it stop before the frame
                frame = env.render_image(rc.CAMERA_NAME, width=rc.RENDER_W, height=rc.RENDER_H)
                q = env._q_arm_deg()
                T = env._kinematics.forward_kinematics(np.append(q, 0.0))
                patch = detect_patch(frame) if tags is None else detect_target_tags(frame, tags)
                n_c = 0 if patch is None else len(patch[0])
                print(f"  [{k}/{n}] h {h:.2f} offset ({dx:+.2f}, {dy:+.2f}) wrist ({dflex:+d}, {droll:+d}): "
                      f"{n_c} corners" + ("" if n_c >= min_corners else " — not used"))
                if patch is None or n_c < min_corners:
                    continue
                name = f"view{len(records):03d}.png"
                privacy.imwrite(str(folder / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                record = {"frame": name, "q_deg": q.tolist(), "T_world_gripper": T.tolist(),
                          "corners_px": patch[0].tolist()}
                if tags is None:
                    record["grid_ij"] = patch[1].tolist()
                else:
                    record["world"] = patch[1].tolist()
                records.append(record)
    return records


def last_good_result() -> "tuple[Path, dict] | None":
    """The newest saved calibration whose joint-corrected fit held on unseen views (under 4 px), or None."""
    for path in sorted(OUT.glob("*/result.json"), reverse=True):
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        held = (r.get("stage3") or {}).get("held_out_mean_px")
        if held is not None and held < 4.0 and "GRIPPER_CAM_JOINT_DEG" in r:
            return path, r
    return None


def fit(folders: "list[Path]") -> dict:
    """
    Fit the camera mount from saved captures; print and return the result,
    which is written beside the last capture. Several captures fit together,
    each against the board where its own run found it.
    """
    from mrcf.perception import hand_eye as he

    datas = [json.loads((f / "views.json").read_text()) for f in folders]
    if any(d.get("target") == "tags" for d in datas):
        if not all(d.get("target") == "tags" for d in datas):
            raise SystemExit("Board and tag-sheet captures cannot be fitted together.")
        return fit_tags(folders, datas)
    K, D = np.array(datas[0]["K"]), np.array(datas[0]["D"])
    if any(not np.allclose(d["K"], K) or not np.allclose(d["D"], D) for d in datas):
        raise SystemExit("These captures were taken with different lens calibrations — fit them apart.")
    # Stages 1-2 against the last capture's board; stage 3 against each view's own.
    grid_world, pitch = np.array(datas[-1]["grid_world"]), float(datas[-1]["pitch_m"])
    views, q_deg, grids = [], [], []
    for d in datas:
        for v in d["views"]:
            views.append(he.GridView(np.array(v["T_world_gripper"]), np.array(v["corners_px"]),
                                     np.array(v["grid_ij"])))
            q_deg.append(v["q_deg"])
            grids.append(np.array(d["grid_world"]))
    print(f"\n{len(views)} views, {sum(len(v.corners_px) for v in views)} corners"
          f"{f' from {len(folders)} captures' if len(folders) > 1 else ''}; "
          f"grid of {len(grid_world)} corners {pitch * 1000:.1f} mm apart in the arm frame.")
    X0, plane_z, d0 = he.initial_from_plane(views, K, D, pitch)
    print(f"  from the surface alone: {d0['views_used']} views, normals agree to "
          f"{d0['normal_err_deg']:.2f} deg, heights to {d0['height_rms_mm']:.1f} mm; "
          f"surface at z = {plane_z * 1000:+.1f} mm (TABLE_Z {rc.TABLE_Z * 1000:+.1f})")
    X, d1 = he.refine(views, X0, grid_world, K, D)
    moved = np.linalg.norm(X[:3, 3] - X0[:3, 3]) * 1000
    turned = np.degrees(np.linalg.norm(cv2.Rodrigues(X[:3, :3] @ X0[:3, :3].T)[0]))
    print(f"  against the grid: {d1['corners']} corners in {d1['views']} views, reprojection "
          f"{d1['reproj_mean_px']:.2f} px mean, {d1['reproj_p95_px']:.2f} px 95th percentile "
          f"(refinement moved the camera {moved:.1f} mm, {turned:.2f} deg)")
    # Where each view, with the fitted mount, puts the surface: the spread is
    # how consistent the arm's FK heights are, pose to pose.
    zs = []
    for v in views:
        pp = he.patch_pose(v, K, D, pitch)
        if pp is not None:
            zs.append(float((v.T_world_gripper @ X)[2, 3] - pp[1]))
    surface_z = float(np.mean(grid_world[:, 2]))
    if zs:
        print(f"  each view's surface height with this mount: {np.mean(zs) * 1000:+.1f} mm, "
              f"spread {np.std(zs) * 1000:.1f} mm (the grid is at {surface_z * 1000:+.1f})")
    # Stage 3: the joint readings corrected too. On this arm one mount alone
    # leaves 8-9 px and is centimetres out: the joints read a few degrees off
    # under load, the same way at the same pose (hand_eye.refine_with_joints).
    from tools.so101.visualise_ik import offline_arm
    from mrcf.backends.so101.kinematics import JOINT_NAMES

    kin = offline_arm()._kinematics            # the live arm's model, JOINT_OFFSETS_DEG and all
    # Not wrist_roll: it only turns the camera about the axis it sits on, which
    # the mount's own rotation does too — the fit cannot tell them apart.
    free = [n != "wrist_roll" for n in JOINT_NAMES]

    def stage3(X_start, dq_start=None, shift_start=None):
        return he.refine_with_joints(views, np.array(q_deg), lambda q: kin.forward_kinematics(np.append(q, 0.0)),
                                     X_start, grids, K, D, free=free, dq0=dq_start, shift0=shift_start)

    # From this fit's own mount, and from the last good calibration's answer:
    # views close to the surface can leave the first too far out to pair the
    # corners right (2026-11-03 at 8-12 cm: 18 px from its own start, 3.5 from
    # the morning's). The one that does better on views it did not see wins.
    tries = [("this capture's mount", lambda: stage3(X))]
    previous = last_good_result()
    if previous is not None:
        path, prev = previous
        tries.append((f"the calibration in {path.parent.name}", lambda: stage3(
            np.array(prev["GRIPPER_CAM_POSE"]), [prev["GRIPPER_CAM_JOINT_DEG"].get(n, 0.0) for n in JOINT_NAMES],
            np.array(prev.get("grid_shift_mm", [0.0, 0.0, 0.0])) / 1000.0)))
    fits = []
    for name, run in tries:
        try:
            fits.append((name, *run()))
        except ValueError as exc:                      # too few corners paired from that start
            print(f"  from {name}: {exc}")
    if not fits:
        raise SystemExit("No start paired enough corners — check the frames.")
    start, X3, dq, shift, d3 = min(fits, key=lambda f: f[4]["held_out_mean_px"])
    if len(tries) > 1:
        print(f"  (started from {start}: " + ", ".join(
            f"{f[0]} {f[4]['held_out_mean_px']:.1f} px" for f in fits) + " on views the fit did not see)")
    print(f"  with the joint readings corrected: {d3['corners']} corners in {d3['views']} views, reprojection "
          f"{d3['reproj_mean_px']:.2f} px mean, {d3['reproj_p95_px']:.2f} px 95th percentile; on views the fit "
          f"did not see {d3['held_out_mean_px']:.2f} px mean, {d3['held_out_p95_px']:.2f} px 95th percentile")
    print("  joint corrections on top of JOINT_OFFSETS_DEG: "
          + ", ".join(f"{n} {d:+.2f} deg" for n, d in zip(JOINT_NAMES, dq) if n != "wrist_roll")
          + f"; the overhead camera's grid {np.round(shift * 1000, 1).tolist()} mm from the arm's")
    if d3["held_out_mean_px"] is not None and d3["held_out_mean_px"] > 4.0:
        print("  WARNING: over 4 px on views the fit did not see — no one mount and set of joint "
              "corrections explains them. Check the frames, and that nothing on the gripper moved.")
    return _finish(folders, X3, dq, shift, d3, X, surface_z, JOINT_NAMES, {"stage1": d0, "stage2": d1, "stage3": d3,
                                                                     "view_surface_z_mm": [round(z * 1000, 2) for z in zs]})


def _finish(folders, X3, dq, shift, d3, X_mount, surface_z: float, joint_names, extra: dict) -> dict:
    """Save the fit beside the last capture (result.json) and print it as robot_config's lines."""
    folder = folders[-1]
    from mrcf import overhead_calibration
    result = {"GRIPPER_CAM_POSE": [[round(float(v), 6) for v in r] for r in X3],
              "GRIPPER_CAM_JOINT_DEG": {n: round(float(d), 3) for n, d in zip(joint_names, dq) if n != "wrist_roll"},
              # The surface where the camera's frame has it: the overhead's grid, moved by the shift.
              "GRIPPER_CAM_PLANE_Z_M": round(surface_z + float(shift[2]), 4),
              "GRIPPER_CAM_SCAN_SHIFT_M": [round(float(v), 4) for v in shift],
              "GRIPPER_CAM_SCAN_SHIFT_AT": (overhead_calibration.load() or {}).get("time"),
              "grid_shift_mm": np.round(shift * 1000, 2).tolist(),
              "mount_only": [[round(float(v), 6) for v in r] for r in X_mount],
              "captures": [str(f) for f in folders], **extra}
    (folder / "result.json").write_text(json.dumps(result, indent=1))
    print(f"\nSaved to {folder / 'result.json'}. Paste into mrcf/backends/so101/robot_config.py, all together "
          "(a mount without its joint corrections is centimetres out on this arm):\n")
    print("GRIPPER_CAM_POSE: \"list[list[float]] | None\" = [")
    for r in result["GRIPPER_CAM_POSE"]:
        print(f"    {r},")
    print("]")
    print(f"GRIPPER_CAM_JOINT_DEG: \"dict[str, float] | None\" = {result['GRIPPER_CAM_JOINT_DEG']}")
    print(f"GRIPPER_CAM_PLANE_Z_M: \"float | None\" = {result['GRIPPER_CAM_PLANE_Z_M']}")
    print(f"GRIPPER_CAM_SCAN_SHIFT_M: \"tuple[float, float, float] | None\" = {tuple(result['GRIPPER_CAM_SCAN_SHIFT_M'])}")
    print(f"GRIPPER_CAM_SCAN_SHIFT_AT: \"str | None\" = {result['GRIPPER_CAM_SCAN_SHIFT_AT']!r}")
    return result


def fit_tags(folders: "list[Path]", datas: list) -> dict:
    """
    Fit the camera mount from table-target captures: the joint-corrected fit
    (hand_eye.refine_with_joints) against each view's own tags, from the last
    good calibration's mount, joint corrections and shift, and from the arm's
    config as it is; the one that does better on views it did not see wins.
    """
    from mrcf.perception import hand_eye as he
    from mrcf.backends.so101.kinematics import JOINT_NAMES
    from tools.so101.visualise_ik import offline_arm

    K, D = np.array(datas[0]["K"]), np.array(datas[0]["D"])
    views, q_deg, grids = [], [], []
    for d in datas:
        for v in d["views"]:
            px = np.array(v["corners_px"], float)
            views.append(he.GridView(np.array(v["T_world_gripper"]), px, np.zeros((len(px), 2))))
            q_deg.append(v["q_deg"])
            grids.append(np.array(v["world"], float))
    surface_z = float(datas[-1]["surface_z_m"])
    print(f"\n{len(views)} views, {sum(len(v.corners_px) for v in views)} tag corners; the sheet on the surface "
          f"at z = {surface_z * 1000:+.1f} mm (TABLE_Z {rc.TABLE_Z * 1000:+.1f}).")
    kin = offline_arm()._kinematics
    free = [n != "wrist_roll" for n in JOINT_NAMES]
    starts = []
    previous = last_good_result()
    if previous is not None:
        path, prev = previous
        starts.append((f"the calibration in {path.parent.name}", np.array(prev["GRIPPER_CAM_POSE"]),
                       [prev["GRIPPER_CAM_JOINT_DEG"].get(n, 0.0) for n in JOINT_NAMES],
                       np.array(prev.get("grid_shift_mm", [0.0, 0.0, 0.0])) / 1000.0))
    if getattr(rc, "GRIPPER_CAM_POSE", None) is not None:
        joint = getattr(rc, "GRIPPER_CAM_JOINT_DEG", None) or {}
        starts.append(("robot_config's", np.array(rc.GRIPPER_CAM_POSE),
                       [joint.get(n, 0.0) for n in JOINT_NAMES],
                       np.array(getattr(rc, "GRIPPER_CAM_SCAN_SHIFT_M", None) or (0.0, 0.0, 0.0), float)))
    if not starts:
        raise SystemExit("A tag sheet has no lattice to start a fit from: calibrate on the board once first.")
    fits = []
    for name, X0, dq0, shift0 in starts:
        try:
            fits.append((name, X0, *he.refine_with_joints(views, np.array(q_deg),
                                                          lambda q: kin.forward_kinematics(np.append(q, 0.0)),
                                                          X0, grids, K, D, free=free, dq0=dq0, shift0=shift0)))
        except ValueError as exc:
            print(f"  from {name}: {exc}")
    if not fits:
        raise SystemExit("No start paired enough tag corners — check the frames.")
    start, X0, X3, dq, shift, d3 = min(fits, key=lambda f: f[5]["held_out_mean_px"] if f[5]["held_out_mean_px"]
                                       is not None else f[5]["reproj_mean_px"])
    print(f"  (started from {start}" + (": " + ", ".join(f"{f[0]} {f[5]['held_out_mean_px']:.1f} px" for f in fits
                                                         if f[5]["held_out_mean_px"] is not None)
                                       if len(fits) > 1 else "") + ")")
    print(f"  {d3['corners']} corners in {d3['views']} views, reprojection {d3['reproj_mean_px']:.2f} px mean, "
          f"{d3['reproj_p95_px']:.2f} px 95th percentile; on views the fit did not see "
          + (f"{d3['held_out_mean_px']:.2f} px mean" if d3["held_out_mean_px"] is not None else "(too few views)"))
    print("  joint corrections on top of JOINT_OFFSETS_DEG: "
          + ", ".join(f"{n} {d:+.2f} deg" for n, d in zip(JOINT_NAMES, dq) if n != "wrist_roll")
          + f"; the overhead camera's tags {np.round(shift * 1000, 1).tolist()} mm from the arm's")
    if d3["held_out_mean_px"] is not None and d3["held_out_mean_px"] > 4.0:
        print("  WARNING: over 4 px on views the fit did not see — check the frames, and that nothing on the "
              "gripper moved.")
    # Said as a board calibration says it: the surface in the camera's frame, at TABLE_Z (deployments move it
    # by their own surface's height above TABLE_Z).
    return _finish(folders, X3, dq, shift, d3, X0, rc.TABLE_Z, JOINT_NAMES,
                   {"target": "tags", "surface_z_m": surface_z, "stage3": d3})


def surface_z(args) -> float:
    """The surface the table target lies on: --surface-z, else the deployment's measured one, else TABLE_TOP_Z."""
    if args.surface_z is not None:
        return float(args.surface_z)
    if args.deployment is not None:
        from tools._common import task_config

        z = getattr(task_config(args.deployment), "SURFACE_Z", None)
        if z is not None:
            return float(z)
    top = getattr(rc, "TABLE_TOP_Z", None)
    if top is None:
        raise SystemExit("No surface height for the sheet: give --surface-z, or a --deployment that measured one.")
    return float(top)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fit", type=Path, nargs="+", help="re-fit saved capture folders, together; no arm needed")
    ap.add_argument("--wide", action="store_true",
                    help="poses at a pick's hover heights over the whole board, not the middle from higher up")
    ap.add_argument("--target", choices=("board", "tags"), default="board",
                    help="what the cameras calibrate against: the chessboard, or the table target sheet of tags "
                         "(tools/make_tag_sheets.py) for a surface with no board")
    ap.add_argument("--surface-z", type=float, default=None,
                    help="tags: the height of the surface the sheet lies on, m (else the deployment's measured "
                         "surface with --deployment, else the rig's TABLE_TOP_Z)")
    ap.add_argument("--deployment", default=None, help="tags: the deployment whose measured surface the sheet is on")
    args = ap.parse_args()
    if args.fit:
        fit(args.fit)
        return
    if rc.GRIPPER_CAM_K is None:
        raise SystemExit("GRIPPER_CAM_K is None — run calibrate_gripper_lens.py first.")

    env = make_env(overhead_camera_width=CAPTURE_W, overhead_camera_height=CAPTURE_H)
    env.connect()
    try:
        env.get_ee_pos()          # fail fast without placo/URDF
        print("Parking the arm out of the overhead camera's view...")
        env.goto_joint_degrees(rc.CALIBRATION_PARK_DEGREES or rc.HOME_DEGREES)
        poses = ((WIDE_HEIGHTS_M, WIDE_OFFSETS_M, WIDE_WRIST_VARIATIONS) if args.wide
                 else (HEIGHTS_M, OFFSETS_M, WRIST_VARIATIONS))
        folder = OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.target == "tags":
            z = surface_z(args)
            tags = tags_in_world(env, z)
            if not tags or len(tags) < 4:
                raise SystemExit(f"The overhead camera sees {len(tags or {})} of the table target's tags — lay the "
                                 "sheet flat in its view, clear of the arm, and retry (at least 4).")
            print(f"Table target found: {len(tags)} tags on the surface at z = {z * 1000:+.1f} mm.\n")
            folder.mkdir(parents=True)
            env.go_home()
            records = capture(env, np.vstack(list(tags.values())), folder, *poses, tags=tags, surface_z=z)
            (folder / "views.json").write_text(json.dumps({
                "target": "tags", "K": rc.GRIPPER_CAM_K, "D": rc.GRIPPER_CAM_DIST, "surface_z_m": z,
                "tags_world": {str(t): c.tolist() for t, c in tags.items()}, "views": records}, indent=1))
        else:
            found = grid_in_world(env)
            if found is None:
                raise SystemExit("The overhead camera cannot see the whole board — clear it and retry.")
            grid_world, pitch = found
            print(f"Board found: {len(grid_world)} corners, {pitch * 1000:.1f} mm apart in the arm frame.\n")
            folder.mkdir(parents=True)
            env.go_home()
            records = capture(env, grid_world, folder, *poses)
            (folder / "views.json").write_text(json.dumps({
                "K": rc.GRIPPER_CAM_K, "D": rc.GRIPPER_CAM_DIST, "grid_world": grid_world.tolist(),
                "pitch_m": pitch, "views": records}, indent=1))
        print(f"\nSaved {len(records)} views to {folder}")
    finally:
        try:
            env.go_home()
        finally:
            env.stop()
    fit([folder])


if __name__ == "__main__":
    main()
