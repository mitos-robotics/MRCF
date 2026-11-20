"""
calibrate_camera_by_hand.py — where a fixed camera is, from a tag stuck on the
arm: you push the arm around by hand, it takes a picture each time you leave it
still, and works out the camera's pose in the arm's frame (and which link the
tag is on, and where) from all of them: perception/camera_from_arm.py. Nothing
goes on the table.

THE ARM IS IN FREEDRIVE for the recording: it moves only by your hand and holds
where you leave it (capabilities.Freedrive: a UR's teach mode; not an arm that
goes limp). Before you start: the arm's payload set (a UR's, on the pendant —
else it sags when let go), the space around it clear, a hand free for the e-stop.

    python tools/calibrate_camera_by_hand.py                    # record, then solve
    python tools/calibrate_camera_by_hand.py --lens-from so101  # the lens of the same camera on another arm
    python tools/calibrate_camera_by_hand.py --solve [RUN]      # solve a recording again (default: the latest)
    python tools/calibrate_camera_by_hand.py --camera side      # another fixed camera
    python tools/calibrate_camera_by_hand.py --camera side --fov-deg 87   # one with no lens model yet

Options: --poses N (default 20: stops by itself after N poses with the tag
seen; Enter stops sooner), --tag ID (default: the one seen most often),
--tag-mm SIZE (the black square's edge as printed; fitted anyway; default: the
arm's GRIPPER_TAG_SIZES_M for that tag, else 40).

The lens: the camera's own model in the arm's config if it has one; else
--lens-from ARM borrows that arm's (a camera moved from one rig to another
keeps its lens), or --fov-deg DEG starts from its horizontal field of view
(the maker's figure: a RealSense D435's infrared view, 87). The focal length
and first distortion term are refined.

Each run is kept in reports/camera_by_hand/<time>/: the frames, poses.json and
solution.json, which tools/show_workspace.py draws from. For the overhead
camera, a good fit (2 px or better) is then offered as the arm's overhead
calibration (--save: without asking): overhead_calibration.json in its rig
folder, which every tool and task loads at start. Another camera's is its
CAMERAS entry's model, printed to go there.
"""

import json
import select
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mrcf import backends, capabilities, overhead_calibration  # noqa: E402
from mrcf.perception import camera_from_arm as cfa  # noqa: E402
from mrcf.perception import fiducials, lighting  # noqa: E402
from tools._common import (FRAMEWORK_ROOT, arg, camera_model_of, camera_size, chosen_camera, connecting_to,  # noqa: E402
                           fixed_camera, gripper_box, make_env_without_gripper, rc)
from mrcf import privacy  # noqa: E402

RUNS = FRAMEWORK_ROOT / "reports" / "camera_by_hand"
STILL_S = 0.6            # the arm counts as left still after this long ...
STILL_MM = 1.0           # ... with every link's origin inside this
NEW_MM = 40.0            # a pose counts as new this far from every one taken ...
NEW_DEG = 8.0            # ... or the last link turned this much from it
TIMEOUT_S = 300.0


def lens_for(camera: str, size) -> "tuple[np.ndarray, np.ndarray, str]":
    """K (scaled to size), D, and where they came from."""
    model, source = camera_model_of(camera), f"the {camera} camera's model"
    other, fov = arg("--lens-from"), arg("--fov-deg")
    if other is not None:
        rc = backends.arm(other).config
        model, source = rc.OVERHEAD_CAM_MODEL, f"{other}'s overhead camera model"
    elif fov is not None and (not model or "K" not in model):
        f = (size[0] / 2) / np.tan(np.radians(float(fov)) / 2)     # square pixels, centred, no distortion
        model = {"size": list(size), "K": [[f, 0.0, size[0] / 2], [0.0, f, size[1] / 2], [0.0, 0.0, 1.0]],
                 "D": [0.0] * 5}
        source = f"a {float(fov):.0f} deg field of view"
    if not model or "K" not in model:
        raise SystemExit(f"No lens for '{camera}': it has no model yet. Pass --fov-deg DEG (its horizontal field "
                         "of view, from its maker: refined by the fit), --lens-from ARM (an arm whose config has "
                         "this camera's lens: the same camera, moved), or calibrate the lens first "
                         "(tools/calibrate_overhead_camera.py --camera NAME).")
    K = np.asarray(model["K"], float).copy()
    msize = model.get("size", size)
    K[0] *= size[0] / msize[0]
    K[1] *= size[1] / msize[1]
    return K, np.asarray(model["D"], float), source


def _pose_moved(a: list, b: list) -> "tuple[float, float]":
    """How far apart two poses are: the largest link-origin move (mm), and the last link's turn (deg)."""
    mm = max(float(np.linalg.norm(np.asarray(x)[:3, 3] - np.asarray(y)[:3, 3])) for x, y in zip(a, b)) * 1000
    return mm, fiducials.rotation_angle_deg(np.asarray(a[-1])[:3, :3], np.asarray(b[-1])[:3, :3])


def _enter_pressed() -> bool:
    """Enter typed at the terminal since last asked (never true without one: a closed stdin reads as ready)."""
    try:
        if not sys.stdin.isatty():
            return False
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return False
    return bool(ready) and sys.stdin.readline() != ""



def check_picture(env, camera: str, size) -> None:
    """
    Before the arm goes into freedrive: the camera gives a real picture, and
    which tags it sees now. A camera that has come off USB makes macOS number
    the others again, and its number may then open another camera (a laptop's
    own, black with the lid shut): every pose would read "no tag".
    """
    image = env.render_image(camera, width=size[0], height=size[1])
    if lighting.no_picture(image):
        raise SystemExit(f"The '{camera}' camera gives a black picture: it may have come unplugged, or its number "
                         "now opens another camera (macOS renumbers them when one comes or goes). Check its cable, "
                         "then look at what it sees (tools/check_camera.py) before running this again.")
    tags = fiducials.detect_tags(image)
    print(f"The camera sees tag(s) {sorted(tags)} now." if tags else
          "The camera sees no tag at this pose: push the arm so the tag faces it for the first picture.")


def record(env, camera: str, size, want: int, run: Path) -> list:
    """Freedrive; a capture each time the arm is left still at a new pose. Returns the poses."""
    poses: list = []
    history: list = []                                        # (t, frames)
    seen = 0
    t0 = time.monotonic()
    env.freedrive(True)
    print("\nFreedrive is ON. Push the arm to a pose, let go, and hold off for a second: it takes a picture.\n"
          "Then the next. Keep the tag facing the camera; turn the base, raise and lower the arm, tilt the\n"
          f"wrist — the more the tag turns, the better. It stops after {want} poses with the tag seen; Enter "
          "stops sooner.\n")
    try:
        while time.monotonic() - t0 < TIMEOUT_S and seen < want:
            if _enter_pressed():
                break
            now = time.monotonic()
            frames = [np.asarray(f, float) for f in env.link_frames()]
            # One reading from before the window kept, so the readings always span
            # STILL_S, however they fall. Kept only within it, they never reached
            # 0.9 x STILL_S on a clock as exact as Linux's (2026-11-13, CI: the
            # test's 0.12 s at 0.05 s apart spanned 0.10 s, and no picture was taken).
            older = [(t, f) for t, f in history if now - t > STILL_S]
            history = older[-1:] + [(t, f) for t, f in history if now - t <= STILL_S] + [(now, frames)]
            if now - history[0][0] < STILL_S:
                time.sleep(0.05)
                continue
            still = max(_pose_moved(f, frames)[0] for _, f in history) < STILL_MM
            new = all(_pose_moved(p["frames"], frames)[0] > NEW_MM or _pose_moved(p["frames"], frames)[1] > NEW_DEG
                      for p in poses)
            if still and new:
                time.sleep(0.15)                              # a frame from after it stopped
                before = [np.asarray(f, float) for f in env.link_frames()]
                image = env.render_image(camera, width=size[0], height=size[1])
                frames = [np.asarray(f, float) for f in env.link_frames()]
                if _pose_moved(before, frames)[0] > STILL_MM:  # moved while the picture was taken: not this one
                    history = []
                    continue
                if lighting.no_picture(image):
                    print("\a  the camera gave a BLACK picture — is it still plugged in? (not counted)")
                    history = []
                    time.sleep(1.0)
                    continue
                tags = {int(k): np.asarray(v, float).reshape(4, 2) for k, v in fiducials.detect_tags(image).items()}
                n = len(poses)
                privacy.imwrite(str(run / f"pose_{n:02d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                poses.append({"frames": [f.tolist() for f in frames], "tags": {k: v.tolist() for k, v in tags.items()},
                              "t": round(now - t0, 2)})
                seen += bool(tags)
                what = ", ".join(f"tag {k} ({np.linalg.norm(v[0] - v[2]) / np.sqrt(2):.0f} px)" for k, v in tags.items())
                print(f"\a  pose {n + 1}: {what or 'NO TAG SEEN — turn the tag toward the camera'}   "
                      f"[{seen}/{want} with a tag]")
                history = []
            time.sleep(0.05)
    finally:
        env.freedrive(False)
        print("\nFreedrive OFF: the arm holds where it is.")
    (run / "poses.json").write_text(json.dumps({"camera": camera, "size": list(size), "poses": poses}))
    return poses


def solve_run(run: Path) -> None:
    data = json.loads((run / "poses.json").read_text())
    camera, size, poses = data["camera"], tuple(data["size"]), data["poses"]
    counts: dict = {}
    for p in poses:
        for k in p["tags"]:
            counts[int(k)] = counts.get(int(k), 0) + 1
    if not counts:
        raise SystemExit(f"No tag in any of the {len(poses)} pictures ({run}).")
    tag = int(arg("--tag", max(counts, key=lambda k: counts[k])))
    index = [i for i, p in enumerate(poses) if str(tag) in p["tags"] or tag in p["tags"]]
    use = [poses[i] for i in index]
    printed = getattr(rc, "GRIPPER_TAG_SIZES_M", {}).get(tag)    # the arm's own tag sizes, if this is one
    size_m = float(arg("--tag-mm", 1000 * printed if printed else 40.0)) / 1000
    K, D, source = lens_for(camera, size)
    print(f"\n{run.name}: tag {tag} in {len(use)} of {len(poses)} pictures (tags seen: {counts}); lens from {source}.")
    frames = [p["frames"] for p in use]
    corners = [p["tags"].get(str(tag), p["tags"].get(tag)) for p in use]
    if len(use) >= 2:
        sightings = [cfa.Sighting(np.asarray(f[-1]), np.asarray(c)) for f, c in zip(frames, corners)]
        a, b = cfa.rotation_spread_deg(sightings)
        print(f"The arm's end turned {a:.0f} and {b:.0f} deg about its two strongest axes across them "
              f"(the solve wants both well above {cfa.MIN_SPREAD_DEG:.0f}).")
    got = cfa.best_link(frames, corners, K, D, size_m)
    if got is None:
        raise SystemExit(f"No solution: too few pictures with the tag ({len(use)}; at least {cfa.MIN_SIGHTINGS}), "
                         "or the poses turned it too little. Record again with more varied poses.")
    link, sol, scores = got
    print("Which link the tag is on, by how well each explains every picture (mean corner error):")
    for k in sorted(scores):
        print(f"  link {k}: {scores[k]:.2f} px{'   <- this one' if k == link else ''}")
    rivals = [k for k in scores if k != link and scores[k] < 1.5 * scores[link]]
    if rivals:
        print(f"  (links {', '.join(map(str, rivals))} fit nearly as well: the poses barely turned the joints between "
              "them. The camera's pose is still good; record again turning every wrist joint to tell them apart.)")
    on = sol.T_link_tag[:3, 3] * 1000
    cam = sol.T_world_cam[:3, 3]
    looks = sol.T_world_cam[:3, 2]
    print(f"\nThe tag sits ({on[0]:+.0f}, {on[1]:+.0f}, {on[2]:+.0f}) mm from link {link}'s origin; it reads "
          f"{sol.tag_size_m * 1000:.1f} mm across (measure it).")
    print(f"The camera is at ({cam[0]:+.3f}, {cam[1]:+.3f}, {cam[2]:+.3f}) m in the arm's frame, looking along "
          f"({looks[0]:+.2f}, {looks[1]:+.2f}, {looks[2]:+.2f}); focal length {sol.K[0, 0]:.0f} px "
          f"(was {K[0, 0]:.0f}), k1 {sol.D[0]:+.3f} (was {D[0]:+.3f}).")
    per = np.asarray(sol.per_sighting_px)
    print(f"Fit: {sol.rms_px:.2f} px mean over {sol.n} pictures; worst {per.max():.2f} px "
          f"(pose_{index[int(per.argmax())]:02d}.jpg).")
    if sol.rms_px > 2.0:
        print("  That is a poor fit: a picture blurred, the tag moved on its link, or the wrong tag size/lens.")
    out = {"camera": camera, "model": sol.camera_model(size), "link": link, "T_link_tag": sol.T_link_tag.tolist(),
           "tag_id": tag, "tag_size_m": sol.tag_size_m, "rms_px": sol.rms_px, "n": sol.n, "lens_from": source,
           "arm": backends.arm().NAME, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (run / "solution.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {run / 'solution.json'}. tools/show_workspace.py draws from it.")
    if camera != rc.OVERHEAD_CAMERA_NAME:
        # Not the overhead camera's: the live calibration file is that one's alone.
        if sol.rms_px <= 2.0:
            print(f"\nIts calibration goes in robot_config CAMERAS['{camera}'] as model=:\n"
                  + json.dumps(_rounded(out["model"])))
        return
    if sol.rms_px <= 2.0 and ("--save" in sys.argv or _yes(
            f"Make it the {backends.arm().NAME}'s overhead calibration (writes {overhead_calibration.PATH})? [y/N] ")):
        overhead_calibration.save(calibration_record(out, run))
        print(f"Written: {overhead_calibration.PATH}. The next start of any tool or task uses it.")


def _rounded(value, places: int = 6):
    """value with every number rounded, for pasting."""
    if isinstance(value, dict):
        return {k: _rounded(v, places) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v, places) for v in value]
    return round(float(value), places) if isinstance(value, float) else value


def _yes(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("y", "yes")
    except (EOFError, OSError):                 # no one to answer (a pipe, a test): no
        return False


def calibration_record(solution: dict, run: Path, n_grid: int = 7) -> dict:
    """
    solution.json as the arm's live calibration (overhead_calibration.save):
    the camera's pose, its lens as fitted, and the table's pixels -> arm (x, y)
    homography from them — over the gripper's box (else the workspace) at
    TABLE_Z, in the pipeline's render size. What else the saved one holds (a
    gripper tag's pose, a board's) is the arm's, not the camera's, and is kept.
    """
    model = solution["model"]
    K, D = np.asarray(model["K"], float), np.asarray(model["D"], float)
    T = np.asarray(model["T_world_cam"], float)
    size = tuple(int(v) for v in model["size"])
    box = gripper_box()
    if box is None:
        b = rc.WORKSPACE_BOUNDS
        box = ((b.x_min, b.x_max), (b.y_min, b.y_max))
    xs, ys = np.linspace(*box[0], n_grid), np.linspace(*box[1], n_grid)
    xy = np.array([(x, y) for x in xs for y in ys])
    px, ok = cfa.project(np.c_[xy, np.full(len(xy), rc.TABLE_Z)], T, K, D)
    ok &= (px[:, 0] >= 0) & (px[:, 0] < size[0]) & (px[:, 1] >= 0) & (px[:, 1] < size[1])
    if ok.sum() < 4:
        raise SystemExit("Too little of the table is in view to give its homography: is the camera pointed at it?")
    H, mean_mm, max_mm = overhead_calibration.homography(px[ok], xy[ok], size, (rc.RENDER_W, rc.RENDER_H))
    print(f"The table's homography from it: {mean_mm:.1f} mm mean, {max_mm:.1f} mm worst over {int(ok.sum())} points "
          "(a homography cannot follow the lens's bend; the scan goes by the lens model).")
    return {**(overhead_calibration.load() or {}),
            "T_arm_cam": T.tolist(), "homography": H.tolist(),
            "lens": {"K": K.tolist(), "D": D.tolist(), "size": list(size)},
            "source": (f"tools/calibrate_camera_by_hand.py {run.relative_to(FRAMEWORK_ROOT) if run.is_relative_to(FRAMEWORK_ROOT) else run} "
                       f"(tag {solution['tag_id']} on link {solution['link']}, {solution['n']} poses, "
                       f"{solution['rms_px']:.2f} px)"),
            "time": solution["time"]}


def latest_run() -> Path:
    runs = sorted(p for p in RUNS.glob("*") if (p / "poses.json").exists())
    if not runs:
        raise SystemExit(f"No recordings in {RUNS}.")
    return runs[-1]


def main() -> None:
    if "--solve" in sys.argv:
        given = sys.argv[sys.argv.index("--solve") + 1] if sys.argv.index("--solve") + 1 < len(sys.argv) else None
        run = Path(given) if given and not given.startswith("--") else latest_run()
        solve_run(run)
        return
    camera = fixed_camera(chosen_camera())
    size = camera_size(camera)
    lens_for(camera, size)                                    # no lens: say so before anything connects
    want = int(arg("--poses", 20))
    env = make_env_without_gripper()
    if not isinstance(env, capabilities.Freedrive):
        raise SystemExit("This arm has no freedrive (capabilities.Freedrive: pushed by hand, it holds where it is "
                         "left): it cannot hold the poses this needs.")
    run = RUNS / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to {connecting_to()} (camera '{camera}', {size[0]}x{size[1]}) ...")
    env.connect()
    try:
        if env.link_frames() is None:
            raise SystemExit("This arm does not say where its links are (RobotEnv.link_frames): nothing to solve from.")
        check_picture(env, camera, size)
        input("\nThe arm goes into freedrive next: it will move only by hand. Ready? [Enter] ")
        poses = record(env, camera, size, want, run)
    finally:
        env.stop()
    print(f"{len(poses)} poses, {sum(bool(p['tags']) for p in poses)} with a tag: {run}")
    solve_run(run)


if __name__ == "__main__":
    main()
