"""
look_tour.py — what the wrist camera sees from where a pick looks, over the
objects on the surface, kept for working on perception without the arm.
Nothing is gripped.

For each object it visits, the arm comes over it as a pick's first attempt
does (ParallelJaw.go_over: the approach a pick would choose, the jaws open, the
fixed camera's correction), then down to the height the wrist servo aligns
from (look_from). At each of the two it keeps a few wrist-camera frames, the
joint readings, the gripper's frame from them, and a frame from the overhead
camera, which sees the gripper tag: a check on where the wrist camera was that
does not go through the joints. Before and after, with the arm parked, the
deployment's scan (which objects, where, how big) and the overhead camera's
view: a piece the arm nudged shows.

A piece a pick would refuse (no room for the jaws between its neighbours) is
looked at from the hover only: that clears the tallest object, where the
servo's height would bring the open jaws down beside its neighbours' heads.

Why: on a full chessboard (2026-11-03) the wrist camera saw the whole first
rank from over e2, the e1 king in front of the pawn, and the servo locked onto
the king. Telling the pieces apart in that view (each by what the overhead
scan knows of it) is built and checked against these tours.

Setup: the objects where they are used (for chess, the board set up), and the
surface otherwise clear. The arm moves on its own; the sentry stops it for a
hand; Ctrl-C stops it at its next step and leaves it clear.

    python tools/look_tour.py [--deployment chess]          # 10 objects spread over the surface
    python tools/look_tour.py --objects e2,d2,b1,g1          # these (names, or squares)
    python tools/look_tour.py --count 6

Saved to rig/<arm>/look_tours/<stamp>/: tour.json (the scans, and per look the
readings and the frames' file names) and the frames, as PNG.
"""

import argparse
import inspect
import json
import signal
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mrcf.interlock import MotionStopped  # noqa: E402
from tools import _common  # noqa: E402  — the arm (rc, make_env, ...) only once main() asks
from mrcf import privacy  # noqa: E402

COUNT = 10
WRIST_FRAMES = 3            # per look: a frame can come back the same as the one before


def _deployment():
    """The deployment's controller module (its DEPLOYMENT, and its skills installed into manipulation)."""
    import importlib
    cfg = _common.task_config()
    module = importlib.import_module(f"{cfg.__package__}.llm_controller")
    if getattr(module, "DEPLOYMENT", None) is None:
        raise SystemExit(f"{cfg.__package__} has no DEPLOYMENT to scan with.")
    return module


def scan(env, deployment) -> "tuple[object, list]":
    """(SceneState, [object record]) from the deployment's own scan, the arm parked. Positions, not names:
    a scan that can skip its vision model's naming (chess: classify=False) does."""
    from mrcf.scene_state import SceneState
    state = SceneState()
    look = deployment.scan
    if look is None:
        raise SystemExit("The deployment has no scan of its own.")
    if "classify" in inspect.signature(look).parameters:
        look = partial(look, classify=False)
    if deployment.observe_pose is not None:
        deployment.observe_pose(env)
    if not look(env, state):
        raise SystemExit("The deployment's scan found nothing to look at.")
    records = []
    for o in state.objects:
        if o.position is None:
            continue
        geometry = getattr(o, "geometry", None) or {}
        shape = getattr(o, "measured_shape", None) or {}
        records.append({"name": o.name, "track_id": getattr(o, "track_id", None),
                        "place": getattr(o, "square", None), "description": o.description,
                        "xy_m": [float(v) for v in o.position[:2]],
                        "foot_m": geometry.get("base_diameter_m"), "height_mm": shape.get("height_mm"),
                        "confirmed": bool(getattr(o, "confirmed", True))})
    return state, records


def spread(objects: list, n: int, start_xy) -> list:
    """n of objects spread over the surface: the one nearest start_xy, then each the furthest from those chosen."""
    left = [o for o in objects if o.position is not None]
    if not left:
        return []
    xy = lambda o: np.asarray(o.position[:2], float)            # noqa: E731
    chosen = [min(left, key=lambda o: float(np.linalg.norm(xy(o) - start_xy)))]
    left.remove(chosen[0])
    while left and len(chosen) < n:
        far = max(left, key=lambda o: min(float(np.linalg.norm(xy(o) - xy(c))) for c in chosen))
        chosen.append(far)
        left.remove(far)
    return chosen


def pick_objects(state, names: "str | None", n: int, base_xy) -> list:
    if not names:
        return spread(state.objects, n, base_xy)
    out = []
    for want in names.split(","):
        found = [o for o in state.objects if o.position is not None
                 and want in (o.name, getattr(o, "square", None))]
        if not found:
            print(f"  [Look] Nothing called or standing on '{want}' in the scan — skipped.")
        out += found[:1]
    return out


def keep(frame, path: Path) -> str:
    import cv2
    privacy.imwrite(str(path), cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    return path.name


def look(env, pol, ctx, folder: Path, stem: str) -> dict:
    """What the arm reads and both cameras see, here: a record, the frames saved beside it."""
    rc, s = _common.rc, pol.spec
    record = {"ee_m": [float(v) for v in env.get_ee_pos()],
              "rotation": None if ctx.rot is None else np.asarray(ctx.rot, float).tolist(),
              "tilt_deg": None if ctx.tilt is None else float(ctx.tilt),
              "gripper_width_m": float(env.get_gripper_width()),
              "gripper_frame": np.asarray(env.gripper_frame(), float).tolist(),
              "joints_deg": dict(env.get_joint_degrees()) if hasattr(env, "get_joint_degrees") else None,
              "wrist_frames": [], "overhead_frame": None}
    if s.camera is not None:
        for k in range(WRIST_FRAMES):
            if k:
                time.sleep(0.15)
            frame = env.render_image(s.camera, width=s.render_w, height=s.render_h)
            record["wrist_frames"].append(keep(frame, folder / f"{stem}_wrist{k}.png"))
    if rc.OVERHEAD_CAMERA_NAME is not None:
        frame = env.render_image(rc.OVERHEAD_CAMERA_NAME, width=rc.OVERHEAD_CAMERA_W, height=rc.OVERHEAD_CAMERA_H)
        record["overhead_frame"] = keep(frame, folder / f"{stem}_overhead.png")
    return record


def visit(env, pol, state, obj, folder: Path, index: int) -> dict:
    """Over obj as a pick's first attempt comes, then down to the servo's height: what each look saw."""
    from mrcf import manipulation
    from mrcf.policies.base import Outcome

    name = getattr(obj, "square", None) or obj.name
    entry = {"name": obj.name, "track_id": getattr(obj, "track_id", None), "place": getattr(obj, "square", None),
             "scan_xy_m": [float(v) for v in obj.position[:2]], "looks": [], "refused": None, "why": None}
    target = manipulation.target_of(obj)
    obstacles = manipulation.obstacles_around(state, obj)
    _, crowded, _ = pol.fit(env, pol.plan(target), target.position[:2], obstacles)
    env.go_home()
    if crowded:
        # No room for the jaws among its neighbours: from the hover, which
        # clears the tallest object, and no lower.
        entry["refused"] = crowded
        print(f"  [Look] {name}: a pick would refuse it ({crowded}) — looking from the hover only.")
        obstacles = ()
    only_high = bool(crowded)
    ctx = pol.go_over(env, target, obstacles, manipulation.vision())
    if isinstance(ctx, Outcome):
        entry["why"] = ctx.message
        print(f"  [Look] {name}: cannot come over it — {ctx.message}")
        return entry
    entry.update({"approach": ctx.approach.name, "corrected_xy_m": [float(v) for v in ctx.object_xy],
                  "planned_height_m": float(ctx.gp.height_m)})
    stem = f"{index:02d}_{name}"
    entry["looks"].append({"at": "hover", "height_m": float(ctx.hover[2] - pol.spec.table_z),
                           **look(env, pol, ctx, folder, f"{stem}_hover")})
    low = pol.spec.servo_height_m
    if not only_high and low is not None and ctx.hover[2] - pol.spec.table_z > low + 0.005:
        if pol.look_from(env, ctx, low):
            entry["looks"].append({"at": "servo", "height_m": float(low), **look(env, pol, ctx, folder, f"{stem}_servo")})
        else:
            print(f"  [Look] {name}: could not come down to the servo's {low * 1000:.0f} mm.")
        env.move_ee_to(np.r_[ctx.object_xy, ctx.hover[2] + 0.03], tol=0.01, orientation=ctx.rot)
    print(f"  [Look] {name}: {len(entry['looks'])} look(s), by {entry['approach']}.")
    return entry


def parked_view(env, folder: Path, stem: str) -> "str | None":
    rc = _common.rc
    if rc.OVERHEAD_CAMERA_NAME is None:
        return None
    env.go_park()
    time.sleep(0.5)
    frame = env.render_image(rc.OVERHEAD_CAMERA_NAME, width=rc.OVERHEAD_CAMERA_W, height=rc.OVERHEAD_CAMERA_H)
    return keep(frame, folder / f"{stem}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Look at objects from where a pick looks, without picking, and keep "
                                             "what both cameras see (rig/<arm>/look_tours/).")
    ap.add_argument("--objects", help="names or squares to visit, comma-separated (default: --count spread out)")
    ap.add_argument("--count", type=int, default=COUNT, help=f"how many to visit when none are named ({COUNT})")
    ap.add_argument("--deployment", help="the task whose scan finds the objects (default: the arm's own)")
    args = ap.parse_args()
    module = _deployment()
    deployment = module.DEPLOYMENT
    from mrcf import manipulation
    pol = manipulation.policies()
    if not hasattr(pol, "go_over"):
        raise SystemExit("This arm's pick policy cannot come over an object without picking it (go_over).")
    if hasattr(module, "configure_models_and_safety"):
        module.configure_models_and_safety()          # the safety layer's limits, as the controller runs
    folder = _common.rc.RIG_DIR / "look_tours" / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True)

    env = _common.make_env()
    print(f"Connecting to {_common.connecting_to()}...")
    env.connect()
    signal.signal(signal.SIGINT, lambda *_: env.interlock.trip("stopped from the keyboard", source="keyboard"))
    tour = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arm": _common.ARM.NAME, "visits": []}
    try:
        env.go_home()
        state, tour["scan_before"] = scan(env, deployment)
        tour["overhead_before"] = parked_view(env, folder, "overhead_before")
        chosen = pick_objects(state, args.objects, args.count, np.asarray(env.base_xy, float))
        print(f"  [Look] {len(chosen)} to look at: " + ", ".join(getattr(o, "square", None) or o.name for o in chosen))
        for i, obj in enumerate(chosen):
            tour["visits"].append(visit(env, pol, state, obj, folder, i))
            (folder / "tour.json").write_text(json.dumps(tour, indent=1))     # kept as it goes
        env.go_home()
        _, tour["scan_after"] = scan(env, deployment)
        tour["overhead_after"] = parked_view(env, folder, "overhead_after")
    except MotionStopped as stop:
        print(f"\n  Stopped: {stop.reason}.")
    finally:
        try:
            env.interlock.clear("keyboard")
            env.go_home()
        finally:
            env.stop()
        (folder / "tour.json").write_text(json.dumps(tour, indent=1))
    moved = moved_objects(tour)
    looks = sum(len(v["looks"]) for v in tour["visits"])
    print(f"\nSaved {looks} looks at {len(tour['visits'])} objects to {folder}")
    if moved:
        print("  Moved during the tour (the scans before and after): " + ", ".join(moved))


def moved_objects(tour: dict, more_than_m: float = 0.004) -> list:
    """Objects whose scanned position changed by more than more_than_m between the tour's scans."""
    after = {o["track_id"]: o for o in tour.get("scan_after") or [] if o.get("track_id")}
    out = []
    for o in tour.get("scan_before") or []:
        a = after.get(o.get("track_id"))
        if a is not None and np.linalg.norm(np.subtract(a["xy_m"], o["xy_m"])) > more_than_m:
            out.append(f"{o['place'] or o['name']} {np.linalg.norm(np.subtract(a['xy_m'], o['xy_m'])) * 1000:.0f} mm")
    return out


if __name__ == "__main__":
    main()
