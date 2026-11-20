"""
overhead_calibration.py — a fixed camera's calibration as live state, not
numbers pasted into an arm's robot_config by hand. Any arm whose env has the
capabilities.OverheadCalibration extras (mrcf/backends/rig.py gives them).

Three transforms tie what the overhead camera sees to where the arm is:

  T_arm_target   where the work surface's reference target sits in the arm's
                 frame (for chess, the board: its lattice and its tags). Only a
                 view of the arm itself can measure it — the gripper tag over
                 many hovers (tools/tag_tour.py). It changes only when the
                 target or the arm's base moves.
  T_arm_cam      where the camera is: the target's pose in one still frame,
                 followed by T_arm_target. It changes whenever the camera is
                 bumped — and then one frame is enough to measure it again.
  T_gf_tag       where the gripper tag sits on the gripper (and the size it
                 reads at), which the tour fits with T_arm_target.

plus the homography from the camera's pixels to the table, fitted to the
target's points. All saved in the arm's file (PATH: its robot_config's
OVERHEAD_CALIBRATION_PATH, which that config sets here as it is imported, and
then applies the file over its own values), so every tool and deployment starts
from the latest measurement; recalibrate_camera() updates a running arm in place.

Why it exists: a camera that has moved puts everything it sees off in the
arm's frame, and picks miss with nothing to say why. So a scan checks the
camera against the board in every clean frame it takes.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from mrcf.capabilities import OrientationSearch, OverheadCalibration, OverheadSighting, Settling
from mrcf import privacy

if TYPE_CHECKING:
    from mrcf.robot_env import RobotEnv

# The arm's calibration file — set by its robot_config (OVERHEAD_CALIBRATION_PATH)
# as it is imported; one arm per process.
PATH: "Path | None" = None

# The camera counts as moved past this. Between two still frames of a tour
# the board's pose repeats to 0.05 mm and 0.006 deg; a bump that matters moves
# it millimetres.
MOVED_MM = 2.0
TURNED_DEG = 0.3
# The arm and the camera agree on where the gripper is to ~1.5 mm median after
# a tour; past this, the target or the arm's base has moved and only a tour can
# say where to.
ARM_AGREE_MM = 5.0
# A fixed camera knocked moves centimetres; one frame that puts it further than
# this is more likely the frame's measurement than the camera, and is not
# re-posed from: the arm's hover decides, and a tag tour re-measures if they
# disagree.
REPOSE_MAX_MM = 80.0
REPOSE_MAX_DEG = 10.0
_HISTORY = 20


def _inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def difference(T_a, T_b) -> "tuple[float, float]":
    """How far apart two poses are: (mm between their origins, deg between their rotations)."""
    T_a, T_b = np.asarray(T_a, float), np.asarray(T_b, float)
    mm = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000)
    c = (np.trace(T_a[:3, :3].T @ T_b[:3, :3]) - 1) / 2
    return mm, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def load(path: "Path | None" = None) -> "dict | None":
    """The saved calibration (PATH, looked up when called), or None when there is none (or it cannot be read)."""
    path = path or PATH
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def save(data: dict, path: "Path | None" = None) -> None:
    """Write it (PATH, looked up when called), keeping the camera poses it replaces in "history" (newest last)."""
    if (path or PATH) is None:
        raise RuntimeError("No file to keep the overhead camera's calibration in: the arm's robot_config "
                           "sets overhead_calibration.PATH (OVERHEAD_CALIBRATION_PATH) as it is imported.")
    path = Path(cast(Path, path or PATH))      # not None: checked above
    old = load(path)
    history = list((old or {}).get("history", []))
    if old and old.get("T_arm_cam") is not None and old.get("T_arm_cam") != _plain(data.get("T_arm_cam")):
        history.append({k: old.get(k) for k in ("time", "source", "T_arm_cam")})
    out = {**data, "history": history[-_HISTORY:]}
    tmp = Path(path).with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)       # a new rig's folder (rig_state.py)
    tmp.write_text(json.dumps(_plain(out), indent=1))
    tmp.replace(path)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def apply_to(namespace: dict, data: "dict | None" = None) -> bool:
    """
    The saved calibration over a robot_config's values (its globals(), or a
    module's vars()): OVERHEAD_HOMOGRAPHY, OVERHEAD_CAM_MODEL's pose and, when
    the record has one ("lens": K, D, size — a fit that refined it), its lens
    (where the namespace has a model — a deployment's config may re-export the
    homography alone), GRIPPER_TAG_POSE, GRIPPER_TAG_READS_AS_M; and
    OVERHEAD_CALIBRATION, the whole record. False (nothing changed) without a
    saved one.
    """
    data = load() if data is None else data
    if not data or data.get("T_arm_cam") is None:
        return False
    model = namespace.get("OVERHEAD_CAM_MODEL")
    if model:
        lens = data.get("lens") or {}
        namespace["OVERHEAD_CAM_MODEL"] = {
            **model, "T_world_cam": [list(map(float, r)) for r in data["T_arm_cam"]],
            **({"K": [list(map(float, r)) for r in lens["K"]], "D": [float(v) for v in lens["D"]]}
               if lens.get("K") is not None and lens.get("D") is not None else {}),
            **({"size": tuple(int(v) for v in lens["size"])} if lens.get("size") else {})}
    if data.get("homography") is not None:
        namespace["OVERHEAD_HOMOGRAPHY"] = [list(map(float, r)) for r in data["homography"]]
    if data.get("T_gf_tag") is not None:
        namespace["GRIPPER_TAG_POSE"] = [list(map(float, r)) for r in data["T_gf_tag"]]
    if data.get("tag_size_m"):
        namespace["GRIPPER_TAG_READS_AS_M"] = float(data["tag_size_m"])
    namespace["OVERHEAD_CALIBRATION"] = data
    return True


def apply_to_env(env, data: dict) -> None:
    """The same, onto a running arm (capabilities.OverheadCalibration): what its overhead camera and gripper tag use."""
    env.set_overhead_calibration(
        T_world_cam=None if data.get("T_arm_cam") is None else np.asarray(data["T_arm_cam"], float),
        homography=None if data.get("homography") is None else np.asarray(data["homography"], float),
        gripper_tag_pose=None if data.get("T_gf_tag") is None else np.asarray(data["T_gf_tag"], float),
        gripper_tag_size_m=float(data["tag_size_m"]) if data.get("tag_size_m") else None)


def homography(points_px, points_xy_m, capture_wh, render_wh) -> "tuple[np.ndarray, float, float]":
    """
    Pixels (at capture_wh) of points on the table -> their arm-frame (x, y),
    as a homography in render_wh pixels: (H, mean mm, max mm over the points).
    A homography cannot model the lens's barrel distortion: ~1 mm mean on the
    board's 49 corners.
    """
    import cv2
    s = np.array([render_wh[0] / capture_wh[0], render_wh[1] / capture_wh[1]])
    px = np.asarray(points_px, float) * s
    xy = np.asarray(points_xy_m, float)
    H, _ = cv2.findHomography(px, xy, 0)
    h = (H @ np.c_[px, np.ones(len(px))].T).T
    res = np.linalg.norm(h[:, :2] / h[:, 2:] - xy, axis=1) * 1000
    return H, float(res.mean()), float(res.max())


def recalibrate_camera(env, measured: dict, capture_wh, render_wh, namespaces=(),
                       source: str = "scan") -> dict:
    """
    Where the overhead camera is now, from one still frame of the target —
    and, if that is not where its calibration has it, the calibration made to
    agree: saved, put on env, and on each of `namespaces` (robot_config's, and
    a deployment's copies of its values).

    measured: the target in the frame — {"T_cam_target": 4x4, "points_px":
    (N, 2) at capture_wh, "points_target": (N, 3) in the target's frame}.

    Assumes the target has not moved on the arm: that is what one frame cannot
    tell from the camera moving. arm_disagreement_mm() checks it afterwards.
    A frame that puts the camera more than REPOSE_MAX_MM / REPOSE_MAX_DEG away
    changes nothing ("implausible": True).
    Returns {"checked", "moved", "shift_mm", "turn_deg", ...}; checked False,
    with "why", when there is nothing to check against.
    """
    data = load()
    if not data or data.get("T_arm_target") is None:
        return {"checked": False, "moved": False,
                "why": "no saved pose of the board on the arm — run a tag tour (tools/tag_tour.py, then --calibrate)"}
    current = env.overhead_camera_pose if isinstance(env, OverheadCalibration) else None
    if current is None:
        return {"checked": False, "moved": False, "why": "the arm has no overhead camera model"}
    T_arm_target = np.asarray(data["T_arm_target"], float)
    T_arm_cam = T_arm_target @ _inv(measured["T_cam_target"])
    mm, deg = difference(T_arm_cam, current)
    report: dict = {"checked": True, "moved": mm > MOVED_MM or deg > TURNED_DEG, "shift_mm": mm, "turn_deg": deg}
    if mm > REPOSE_MAX_MM or deg > REPOSE_MAX_DEG:
        report.update(moved=False, implausible=True)
    if not report["moved"]:
        return report
    pts = (T_arm_target[:3, :3] @ np.asarray(measured["points_target"], float).T).T + T_arm_target[:3, 3]
    H, mean_mm, max_mm = homography(measured["points_px"], pts[:, :2], capture_wh, render_wh)
    data = {**data, "T_arm_cam": T_arm_cam, "homography": H, "source": source,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save(data)
    data = load()
    apply_to_env(env, data)  # type: ignore[arg-type]  # see report: None if the file just saved cannot be read back
    for ns in namespaces:
        apply_to(ns, data)
    report.update(homography_mm=(mean_mm, max_mm), T_arm_cam=T_arm_cam)
    return report


class CalibrationTarget(Protocol):
    """
    Something fixed on the work surface that the overhead camera can measure
    its own pose against — supplied by the deployment (chess: the board, by its
    lattice and tags). T_arm_target in the saved calibration is this target's
    pose on the arm, measured by a tag tour.
    """
    name: str

    def measure(self, frame_rgb: np.ndarray, K: np.ndarray, D: np.ndarray) -> "dict | None":
        """
        The target in this frame: {"T_cam_target": 4x4, "points_px": (N, 2),
        "points_target": (N, 3) in the target's frame}, or None when it cannot
        be measured (hidden, out of view).
        """

    def camera_moved(self) -> None:
        """The camera has just been re-posed: drop whatever was kept in its old pixels."""


def check_camera(env: "RobotEnv", target: CalibrationTarget, camera: str, capture_wh, render_wh, *,
                 hover_z: float, suspect: bool = False, tour=None, namespaces=(), say=print) -> dict:
    """
    Is the overhead camera still where its calibration says? Run with the arm
    parked out of its view; makes it so when it is not. A camera that moves
    with nothing noticing puts every square off and the picks fail, so a run
    finds the discrepancy itself and updates what depends on it.

      1. The target's pose in this frame against the calibration: a camera
         that moved is re-posed from it on the spot (recalibrate_camera), and
         the target drops what it kept in the old pixels (camera_moved).
      2. Then — or when the caller has reason to doubt it (`suspect`: a pick's
         hovers were far off) — one hover over the target at hover_z, to see
         whether the gripper's tag agrees with the arm. A target or arm base
         that moved does not show in step 1; it shows here, and `tour(env)`
         (a tag tour) re-measures it, when given.

    Every move is the arm's own, under its sentry; the arm is left where the
    hover put it. Returns {"moved", "hovered", "apart_mm", "toured"}.
    """
    report: dict = {"moved": False, "hovered": False, "apart_mm": None, "toured": False}
    if not isinstance(env, OverheadSighting) or not isinstance(env, OverheadCalibration) \
            or env.overhead_camera_pose is None:
        return report
    model = env.camera_model(camera, *capture_wh)
    if model is None:
        return report                    # no lens model to measure the target with: not checked
    frame = env.render_image(camera, width=capture_wh[0], height=capture_wh[1])
    seen = target.measure(frame, np.asarray(model["K"], float), np.asarray(model["D"], float))
    if seen is None:
        say(f"  [Calibration] {target.name[0].upper() + target.name[1:]} is not measurable in this frame — "
            "the camera's calibration is not checked this scan.")
    else:
        got = recalibrate_camera(env, seen, capture_wh, render_wh, namespaces=namespaces)
        if not got["checked"]:
            say(f"  [Calibration] Not checked: {got['why']}.")
        elif got.get("implausible"):
            # 2026-10-31: one frame put it 360 mm and 38 deg away; re-posed and
            # saved, the arm hovered by it into a joint limit and the height
            # ceiling, and the motion guard took the arm for a hand.
            report["implausible"] = True
            kept = _keep_frame(frame, "overhead_implausible")
            say(f"  [Calibration] This frame puts the overhead camera {got['shift_mm']:.0f} mm and "
                f"{got['turn_deg']:.0f} deg from its calibration — further than a knocked camera moves; not "
                f"re-posing it from one frame." + (f" Frame kept: {kept}" if kept else ""))
        elif not got["moved"]:
            say(f"  [Calibration] Overhead camera where its calibration has it ({got['shift_mm']:.1f} mm, "
                f"{got['turn_deg']:.2f} deg).")
        else:
            report["moved"] = True
            say(f"  [Calibration] The overhead camera has moved — {got['shift_mm']:.1f} mm and "
                f"{got['turn_deg']:.2f} deg from its calibration. Re-posed it from {target.name} in this frame "
                f"(homography {got['homography_mm'][0]:.1f} mm mean over its points).")
            target.camera_moved()
    if not (report["moved"] or report.get("implausible") or suspect):
        return report
    saved = load()
    if not saved or saved.get("T_arm_target") is None:
        return report
    centre = np.asarray(saved["T_arm_target"], float)[:2, 3]
    why = ("the camera moved" if report["moved"] else "that frame disagrees" if report.get("implausible")
           else "the last pick's hovers were far off")
    say(f"  [Calibration] {why[0].upper() + why[1:]}: one hover over {target.name}'s centre, to check the arm "
        "and the camera agree...")
    apart = report["apart_mm"] = arm_disagreement_mm(env, centre, hover_z)
    report["hovered"] = True
    if apart is not None and apart > ARM_AGREE_MM:
        if tour is None:
            say(f"  [Calibration] They are {apart:.1f} mm apart — {target.name} or the arm's base has moved too. "
                "Re-measure with a tag tour (tools/tag_tour.py, then --calibrate).")
        else:
            say(f"  [Calibration] They are {apart:.1f} mm apart — {target.name} or the arm's base has moved too. "
                f"Re-measuring with a tag tour (a few minutes; the arm hovers over {target.name}).")
            tour(env)
            report["toured"] = True
    elif apart is not None:
        say(f"  [Calibration] They agree to {apart:.1f} mm.")
    else:
        say("  [Calibration] Could not check (hover not reached, or the gripper's tag not seen).")
    return report


def _keep_frame(frame_rgb, stem: str) -> "Path | None":
    """A frame saved beside the calibration (the rig's own folder), to see later what misled a check; None if it cannot be."""
    if PATH is None:
        return None
    try:
        import cv2
        out = Path(cast(Path, PATH)).with_name(f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        privacy.imwrite(str(out), cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR))
        return out
    except Exception:          # noqa: BLE001 — evidence, not a reason to stop the scan
        return None


def arm_disagreement_mm(env, xy, z: float, tol: float = 0.008) -> "float | None":
    """
    Hover the gripper over (xy, z) and compare where the overhead camera sees
    its tag with where the arm's own kinematics put it: mm apart (xy), or None
    when the hover is not reached or the tag not seen. The one check that tells
    a camera that moved (fixed by recalibrate_camera) from a target or arm base
    that moved (only a tag tour fixes that). Moves the arm — under its sentry.
    """
    target = np.array([float(xy[0]), float(xy[1]), float(z)])
    chosen = env.approach_orientation_for([target], tols=[tol]) if isinstance(env, OrientationSearch) else None
    rot = chosen[0] if chosen else None
    if not env.move_ee_to(target, tol=tol, orientation=rot):
        return None
    if isinstance(env, Settling):
        env.wait_still()
    off = env.overhead_gripper_offset()
    return None if off is None else float(np.linalg.norm(np.asarray(off, float)[:2]) * 1000)
