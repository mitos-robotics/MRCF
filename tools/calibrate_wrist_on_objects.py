"""
calibrate_wrist_on_objects.py — the wrist camera put right against the
overhead camera with the objects on the table as the ruler: no board, no
printed sheet.

What it measures: where a position the overhead scan measured stands for the
wrist camera (the arm's GRIPPER_CAM_SCAN_SHIFT_M, model["scan_to_camera"]).
A pick takes the wrist camera's outline of its object only within a few mm
of where the scan has it, and moves its aim by how far the wrist camera puts
the foot from there (parallel_jaw/sighting.py): with that shift stale, every
outline is "a part of it, not all of it" and the jaws come down on the
overhead camera's aim alone. It goes stale whenever the overhead camera is
moved (the SO-101's was re-posed 49 mm and 7 deg on 2026-11-16, after its
11-03 fit).

How: the overhead camera scans the table; the small objects standing apart
are the ruler (their feet as the scan has them, to ~2 mm); the arm hovers
over each from two heights and five wrist leans, as a pick looks; the wrist
camera's frame and model are kept at each; each foot is found in each frame
by the same measure a pick uses (plane_geometry.measure_in_frame); the shift
is the median of where the wrist camera puts each foot less where the scan
does, and the spread round it says whether one shift explains them all (a
spread that grows with the lean or the height is the mount, not the shift).

    python tools/calibrate_wrist_on_objects.py --deployment tidy           # the arm hovers over the objects
    python tools/calibrate_wrist_on_objects.py --fit rig/so101/wrist_on_objects/<stamp>   # again, no arm
    python tools/calibrate_wrist_on_objects.py --from-runs [--since 2026-11-18]           # from the picks
    --apply   write the fitted shift into the arm's robot_config.py (else printed, to paste)

--from-runs: no arm and no capture — every pick whose wrist camera found its
object's foot logged where it put it against the scan (flight recorder,
"calibration" / "wrist_foot"); the shift is moved by their median. Those are
only the feet a pick took (within a few mm of the scan's): enough to follow a
drift, and say it, not to find one far off — for that, the capture above.

The objects: anything the scan finds 15-60 mm across, within reach, with no
other foot within 40 mm and not in a container. Leave a few on the table,
apart; take nothing away while it runs. About 20 views, 2-3 minutes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

HEIGHTS_M = (0.10, 0.14)                 # the grasp point over the surface, as a pick's hover looks
WRIST_VARIATIONS = [(0, 0), (10, 0), (-10, 0), (0, 20), (0, -20)]   # (wrist_flex, wrist_roll) added, degrees
FOOT_MM = (15.0, 60.0)                   # the objects that make the ruler: this wide across
APART_M = 0.04                           # ... with no other foot nearer than this
FOOT_TOL_MM = 30.0                       # how far from the scan's foot an outline may put it, here
MIN_FEET = 6
SPREAD_OK_MM = 3.0


def ruler(feet: list, bounds, containers=()) -> list:
    """
    The scan's feet that make the ruler: [(key, (x, y), diameter mm)], each
    FOOT_MM across, inside bounds, APART_M from every other foot, and not in
    any of containers (callables xy -> bool).
    """
    out = []
    for i, (xy, d) in enumerate(feet):
        xy = np.asarray(xy, float)[:2]
        if not FOOT_MM[0] <= d <= FOOT_MM[1]:
            continue
        if not (bounds.x_min <= xy[0] <= bounds.x_max and bounds.y_min <= xy[1] <= bounds.y_max):
            continue
        if any(j != i and float(np.linalg.norm(np.asarray(o, float)[:2] - xy)) < APART_M for j, (o, _) in enumerate(feet)):
            continue
        if any(inside(xy) for inside in containers):
            continue
        out.append((f"obj{i}", xy, float(d)))
    return out


OBJECT_APART_MM = 6.0                    # an object whose feet sit this far from the others' is left out


def fit_shift(pairs: list) -> dict:
    """
    pairs: [(where the wrist camera put a foot, where the scan has it), (x, y) m,
    with "object", "height_m", "lean" for the report]. The shift (median, m),
    the spread round it (mm), and the mean left over by object, height and lean.
    An object whose own median sits OBJECT_APART_MM from the median of the
    others' is left out ("left_out"): 2026-11-18, a figure inside the clear tub,
    its foot seen through the wall, 10 mm from the two on the table.
    """
    def offset(p):
        return np.asarray(p["wrist"], float)[:2] - np.asarray(p["scan"], float)[:2]
    keys = sorted({p["object"] for p in pairs})
    left_out = []
    if len(keys) >= 3:
        own = {k: np.median([offset(p) for p in pairs if p["object"] == k], axis=0) for k in keys}
        for k in keys:
            rest = np.median([own[j] for j in keys if j != k], axis=0)
            if float(np.linalg.norm(own[k] - rest)) * 1000.0 > OBJECT_APART_MM:
                left_out.append(k)
        pairs = [p for p in pairs if p["object"] not in left_out]
    d = np.array([offset(p) for p in pairs])
    shift = np.median(d, axis=0)
    left = np.linalg.norm(d - shift, axis=1) * 1000.0

    def by(key):
        groups: dict = {}
        for p, r in zip(pairs, left):
            groups.setdefault(str(p[key]), []).append(r)
        return {k: round(float(np.mean(v)), 1) for k, v in sorted(groups.items())}
    return {"shift_m": [round(float(v), 4) for v in shift], "feet": len(pairs), "left_out": left_out,
            "spread_mm": round(float(np.median(left)), 1), "p90_mm": round(float(np.percentile(left, 90)), 1),
            "by_object_mm": by("object"), "by_height_mm": by("height_m"), "by_lean_mm": by("lean")}


def _model_from(record: dict) -> dict:
    m = {k: np.asarray(v, float) for k, v in record["model"].items() if k not in ("plane_z_m", "scale")}
    m["plane_z_m"], m["scale"] = float(record["model"]["plane_z_m"]), float(record["model"].get("scale", 1.0))
    return m


def measure(folder: Path) -> list:
    """Each saved view's foot of its object as the wrist camera finds it (a pick's measure): the pairs to fit."""
    import cv2

    from mrcf.perception import plane_geometry
    from mrcf.perception.expected_view import KnownObject

    data = json.loads((folder / "views.json").read_text())
    feet = {k: np.asarray(v["xy"], float) for k, v in data["ruler"].items()}
    others = [np.asarray(xy, float) for xy in data["all_feet"]]
    pairs = []
    for r in data["views"]:
        frame = cv2.cvtColor(cv2.imread(str(folder / r["frame"])), cv2.COLOR_BGR2RGB)
        model = _model_from(r)
        shift = np.asarray(model.get("scan_to_camera", np.zeros(3)), float)[:2]
        key = r["object"]
        known = [KnownObject(key, feet[key] + shift, data["ruler"][key]["d_mm"] / 2000.0, 0.06)]
        known += [KnownObject(f"n{i}", xy + shift, 0.012, 0.06) for i, xy in enumerate(others)
                  if float(np.linalg.norm(xy - feet[key])) > 0.005]
        got = plane_geometry.measure_in_frame(frame, expected_px=(frame.shape[1] / 2, frame.shape[0] / 2),
                                              model=model, expected_xy_m=feet[key] + shift, known=known,
                                              target=key, need_height=False, foot_tol_mm=FOOT_TOL_MM)
        m = got.get("measurement") if got.get("ok") else None
        if m is None or m.base_world_m is None:
            print(f"  {r['frame']} ({key}, {r['height_m'] * 100:.0f} cm, lean {r['lean']}): not found — {got.get('why')}")
            continue
        pairs.append({"wrist": np.asarray(m.base_world_m, float)[:2].tolist(), "scan": feet[key].tolist(),
                      "object": key, "height_m": r["height_m"], "lean": r["lean"], "frame": r["frame"]})
    return pairs


def report(folder: Path, old_shift) -> dict:
    pairs = measure(folder)
    if len(pairs) < MIN_FEET:
        raise SystemExit(f"Only {len(pairs)} feet found in the wrist frames (at least {MIN_FEET}): more light, or "
                         "objects further apart, and again.")
    got = fit_shift(pairs)
    old = np.asarray(old_shift if old_shift is not None else (0.0, 0.0, 0.0), float)
    print(f"\n{got['feet']} feet in the wrist frames. The scan's positions stand for the wrist camera "
          f"{np.round(np.asarray(got['shift_m']) * 1000, 1).tolist()} mm off (was "
          f"{np.round(old[:2] * 1000, 1).tolist()} mm); left over round that: {got['spread_mm']} mm median, "
          f"{got['p90_mm']} mm at 90%.")
    print(f"  by object {got['by_object_mm']}; by height {got['by_height_mm']}; by lean {got['by_lean_mm']}")
    if got["left_out"]:
        print(f"  left out: {', '.join(got['left_out'])} — its feet sit over {OBJECT_APART_MM:.0f} mm from the others' "
              "(in a container, seen through its wall, or not standing upright)")
    if got["spread_mm"] > SPREAD_OK_MM:
        print(f"  WARNING: over {SPREAD_OK_MM:.0f} mm left over — one shift does not explain them all. Where it grows "
              "with the lean or the height, the camera's mount is off (calibrate_gripper_pose.py).")
    got["shift_m"] = got["shift_m"] + [round(float(old[2]), 4)]        # the height: feet on the surface do not say it
    got["pairs"] = pairs
    (folder / "result.json").write_text(json.dumps(got, indent=1))
    return got


def apply(config_path: Path, shift_m, at: "str | None") -> None:
    """Write GRIPPER_CAM_SCAN_SHIFT_M (and _AT) into an arm's robot_config.py, the lines as they are."""
    text = config_path.read_text()
    value = "(" + ", ".join(f"{v:.4f}" for v in shift_m) + ")"
    text, n = re.subn(r"^(GRIPPER_CAM_SCAN_SHIFT_M:[^=]*=\s*).*$", lambda m: m.group(1) + value, text, flags=re.M)
    if n != 1:
        raise SystemExit(f"No GRIPPER_CAM_SCAN_SHIFT_M line in {config_path}: paste it by hand.")
    if at is not None:
        text = re.sub(r"^(GRIPPER_CAM_SCAN_SHIFT_AT:[^=]*=\s*).*$", lambda m: m.group(1) + repr(at), text, flags=re.M)
    config_path.write_text(text)
    print(f"  Written to {config_path}: GRIPPER_CAM_SCAN_SHIFT_M = {value}" + (f", _AT = {at!r}" if at else ""))


def capture(env, cfg, folder: Path) -> None:
    """Scan, choose the ruler, hover over each of it as a pick looks, keep each wrist frame and its camera's model."""
    import cv2

    from mrcf import privacy
    from mrcf.perception import surface_scan

    rc = cfg.ARM.config
    size = (cfg.CAPTURE_W_OVERHEAD, cfg.CAPTURE_H_OVERHEAD)
    b = cfg.WORKSPACE_BOUNDS
    got = surface_scan.scan(env, cfg.OBSERVE_CAMERA, *size, (b.x_min, b.x_max), (b.y_min, b.y_max),
                            reference=surface_scan.load_reference(cfg.SURFACE_REFERENCE_PATH, size))
    if got is None:
        raise SystemExit("The overhead camera could not scan the table (not calibrated, or no segmenter).")
    feet = [(np.asarray(f["measurement"].base_world_m, float)[:2], 2 * float(f["measurement"].base_radius_mm))
            for f in got[1]]
    chosen = ruler(feet, b)
    print(f"The scan found {len(feet)} objects; the ruler: " + (", ".join(
        f"{k} at ({xy[0]:+.3f}, {xy[1]:+.3f}) {d:.0f} mm" for k, xy, d in chosen) or "none") + ".")
    if not chosen:
        raise SystemExit(f"No object {FOOT_MM[0]:.0f}-{FOOT_MM[1]:.0f} mm across stands {APART_M * 100:.0f} cm "
                         "from the others within reach: spread a few out, and again.")
    camera, w, h = rc.CAMERA_NAME, rc.RENDER_W, rc.RENDER_H
    z0 = float(cfg.TABLE_Z)
    records = []
    for key, xy, _ in chosen:
        for hgt in HEIGHTS_M:
            target = np.array([xy[0], xy[1], z0 + hgt])
            found = env.approach_orientation(target)
            if found is None or not env.move_ee_to(target, orientation=found[0]):
                print(f"  {key}: over it at {hgt * 100:.0f} cm cannot be reached — skipped")
                continue
            base = env.get_joint_degrees()
            for dflex, droll in WRIST_VARIATIONS:
                env.goto_joint_degrees(dict(base, wrist_flex=base["wrist_flex"] + dflex,
                                            wrist_roll=base["wrist_roll"] + droll))
                time.sleep(0.4)                                       # still, before the frame
                frame = env.render_image(camera, width=w, height=h)
                model = env.camera_model(camera, w, h)              # read with the frame, the arm still
                if frame is None or model is None:
                    continue
                name = f"view{len(records):03d}.png"
                privacy.imwrite(str(folder / name), cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
                records.append({"frame": name, "object": key, "height_m": hgt, "lean": f"{dflex:+d}/{droll:+d}",
                                "model": {k: np.asarray(v).tolist() if not isinstance(v, (int, float)) else v
                                          for k, v in model.items()}})
                print(f"  [{len(records)}] {key} at {hgt * 100:.0f} cm, wrist {dflex:+d}/{droll:+d}")
            env.goto_joint_degrees(base)
    (folder / "views.json").write_text(json.dumps({
        "ruler": {k: {"xy": xy.tolist(), "d_mm": d} for k, xy, d in chosen},
        "all_feet": [xy.tolist() for xy, _ in feet], "views": records}, indent=1))
    print(f"Saved {len(records)} views to {folder}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--deployment", default=None, help="the deployment whose table and surface the objects are on")
    ap.add_argument("--fit", type=Path, default=None, help="a saved capture folder: fit again, no arm")
    ap.add_argument("--apply", action="store_true", help="write the shift into the arm's robot_config.py")
    ap.add_argument("--from-runs", action="store_true", help="fit from the picks' logged wrist feet, no arm")
    ap.add_argument("--since", default=None, help="--from-runs: sessions from this date (YYYY-MM-DD)")
    args = ap.parse_args(argv)
    from tools._common import task_config

    cfg = task_config(args.deployment)
    rc = cfg.ARM.config
    if not hasattr(rc, "GRIPPER_CAM_SCAN_SHIFT_M"):
        raise SystemExit(f"The {cfg.ARM_NAME}'s config has no GRIPPER_CAM_SCAN_SHIFT_M: nothing here to calibrate.")
    out = Path(rc.RIG_DIR) / "wrist_on_objects"
    if args.from_runs:
        from mrcf import flight_recorder

        from mrcf import wrist_shift

        offs = wrist_shift.from_runs(flight_recorder.ROOT, args.since)
        got = wrist_shift.drift(offs, MIN_FEET)
        if got is None:
            raise SystemExit(f"Only {len(offs)} picks logged a wrist-camera foot (at least {MIN_FEET}).")
        med, spread = got
        old = np.asarray(getattr(rc, "GRIPPER_CAM_SCAN_SHIFT_M", None) or (0.0, 0.0, 0.0), float)
        shift = [round(float(old[0] + med[0]), 4), round(float(old[1] + med[1]), 4), round(float(old[2]), 4)]
        print(f"{len(offs)} picks' wrist-camera feet sit {np.round(med * 1000, 1).tolist()} mm off the scan's "
              f"(spread {spread:.1f} mm): the shift {np.round(old[:2] * 1000, 1).tolist()} -> "
              f"{np.round(np.asarray(shift[:2]) * 1000, 1).tolist()} mm.")
        got = {"shift_m": shift}
    elif args.fit is not None:
        folder = args.fit
    else:
        folder = out / datetime.now().strftime("%Y%m%d_%H%M%S")
        folder.mkdir(parents=True)
        env = cfg.ARM.make_env(profile=cfg.PROFILE, **cfg.ENV_OVERRIDES)
        env.connect()
        try:
            env.go_home()
            capture(env, cfg, folder)
        finally:
            try:
                env.go_home()
            finally:
                env.stop()
    if not args.from_runs:
        got = report(folder, getattr(rc, "GRIPPER_CAM_SCAN_SHIFT_M", None))
    from mrcf import overhead_calibration

    at = (overhead_calibration.load() or {}).get("time")
    if args.apply:
        import importlib

        apply(Path(importlib.import_module(rc.__name__).__file__), got["shift_m"], at)
    else:
        print(f"\nGRIPPER_CAM_SCAN_SHIFT_M: \"tuple[float, float, float] | None\" = {tuple(got['shift_m'])}")
        print(f"GRIPPER_CAM_SCAN_SHIFT_AT: \"str | None\" = {at!r}")


if __name__ == "__main__":
    main()
