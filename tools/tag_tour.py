"""
tag_tour.py — where is the gripper really, square by square? Measurement only.

The arm hovers over a spread of squares. At each one the overhead camera reads
the board tags and the gripper tag, the wrist camera reads the jaw marks, and
the joint angles are logged. Nothing measured is fed back into a move.

Before and after the tour the arm parks out of the board's view, and a clean
frame gives: the chessboard lattice (49 corners at a known 40 mm pitch), which
checks the overhead lens model and pins where each board tag sits on the board
— the tags are stuck on by hand, so that is measured, not assumed — and, after,
whether anything moved during the run.

--analyse then works out, from all the hovers together, the two unknowns that
tie the camera's view to the arm's own sense of where it is: where the arm's
base sits on the board, and where the tag sits on the gripper (OpenCV's
robot-world/hand-eye solve). What is left over at each square is the error map
of the arm's kinematics against the camera.

Usage:
    python tools/tag_tour.py                 # run it (MOVES THE ARM)
    python tools/tag_tour.py --analyse       # the latest run
    python tools/tag_tour.py --analyse reports/tag_tour/<run>
    python tools/tag_tour.py --calibrate [run]   # overhead camera + chess cache from a tour
    python tools/tag_tour.py --camera side       # through another fixed camera, its lens calibrated
                                                 # (tools/calibrate_overhead_camera.py --camera side);
                                                 # --calibrate then prints its CAMERAS entry's pose

A run keeps which camera it was (camera.txt): --analyse and --calibrate use it.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mrcf.capabilities import OrientationSearch  # noqa: E402
from mrcf.perception import fiducials  # noqa: E402
from tools._common import (FRAMEWORK_ROOT, camera_model_of, camera_size, chosen_camera,  # noqa: E402
                           connecting_to, fixed_camera, make_env, rc, task_config)
from mrcf import privacy  # noqa: E402

cfg = task_config()
RUNS = FRAMEWORK_ROOT / "reports" / "tag_tour"
HOVER_M = 0.10            # jaw tips this far above the board: clears every piece
FRAMES_PER_POSE = 5
SETTLE_S = 0.6
STILL_MM = 0.5            # a hover counts if the gripper moved less than this while its frames were taken
MAX_MISSES = 3            # gripper tag unseen this many hovers running: stop the tour
# A snake over the board, every other square: ranks 1-2 are mostly out of reach.
TOUR = ["b2", "g2", "h3", "f3", "d3", "b3", "a4", "c4", "e4", "g4", "h5", "f5", "d5", "b5",
        "a6", "c6", "e6", "g6", "h7", "f7", "d7", "b7", "a8", "c8", "e8", "g8"]
# The lean each hover starts its search from, by rank: the chess approach
# rules the calibrated tours (2026-10-20) were run with, kept here when those
# rules gave way to the arm's approaches — the tour's poses are its recipe.
TOUR_LEAN_BY_RANK = {1: 20, 2: 20, 6: 20, 7: 30, 8: 40}


def tour_lean(square: str) -> "float | None":
    """TOUR_LEAN_BY_RANK's lean for a square such as 'e6' (rank 1-2 only on files b-g, as the tours had)."""
    file, rank = square[0], int(square[1])
    if rank <= 2 and file not in "bcdefg":
        return None
    lean = TOUR_LEAN_BY_RANK.get(rank)
    return None if lean is None else float(lean)


# ── Board: lattice + where the tags are on it (chess's board_tags) ───────────
from deployments.chess.board_tags import lattice_board_points, measure_board  # noqa: E402,F401


def print_board(b: dict, label: str) -> None:
    C = -b["T_cam_board"][:3, :3].T @ b["T_cam_board"][:3, 3]
    mean, mx = b["lattice_err_px"]
    print(f"  {label}: lattice {b['lattice_used']}/49 corners; the lens model reprojects them to {mean:.2f} px mean, "
          f"{mx:.2f} px max; camera {C[2] * 1000:.0f} mm above the board; a1 "
          f"{'dark, as it should be' if b['a1_dark'] else 'LIGHT — the tag placement table is turned'}")
    for i in sorted(b["tags"]):
        P = b["tags"][i]
        side = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).mean()
        kind = "board" if i in cfg.BOARD_TAG_IDS else ("gripper" if i == rc.GRIPPER_TAG_ID else "?")
        c = P.mean(0) * 1000
        note = f", printed {side * 1000:.2f} mm (want {cfg.BOARD_TAG_SIZE_M * 1000:.0f})" if kind == "board" else ""
        print(f"    tag {i:2d} ({kind}) at ({c[0]:+6.1f},{c[1]:+6.1f}) mm on the board{note}")


# ── The tour ──────────────────────────────────────────────────────────────────

def grab(env, n: int, camera: "str | None" = None) -> list:
    camera = camera or rc.OVERHEAD_CAMERA_NAME
    w, h = camera_size(camera)
    frames = []
    for _ in range(n):
        frames.append(env.render_image(camera, width=w, height=h))
        time.sleep(0.08)
    return frames


def lens(camera: str):
    """The camera's K and D (camera_model_of); a tour needs its lens calibrated first."""
    model = camera_model_of(camera)
    if model is None:
        raise SystemExit(f"The '{camera}' camera's lens is not calibrated: tools/calibrate_overhead_camera.py"
                         + ("" if camera == rc.OVERHEAD_CAMERA_NAME else f" --camera {camera}") + " first.")
    return np.array(model["K"]), np.array(model["D"])


def run_camera(run: Path) -> str:
    """The camera a run was made through (camera.txt; the overhead one for runs from before it was kept)."""
    kept = run / "camera.txt"
    return kept.read_text(encoding="utf-8").strip() if kept.exists() else rc.OVERHEAD_CAMERA_NAME


def clean_board(env, run: Path, label: str, K, D, camera: "str | None" = None):
    env.go_park()
    time.sleep(SETTLE_S)
    frames = grab(env, 15, camera)
    median = np.median(np.array(frames), axis=0).astype(np.uint8)
    privacy.imwrite(str(run / f"board_{label}.png"), cv2.cvtColor(median, cv2.COLOR_RGB2BGR))
    b = measure_board(median, K, D)
    if b is None:
        print(f"  {label}: board not found in the parked frame (fewer than 4 board tags, or the lattice)")
    else:
        print_board(b, label)
    return b


def fk(env) -> np.ndarray:
    """gripper_frame_link in the arm frame, from the joint angles read now (RigEnv.gripper_frame)."""
    return np.asarray(env.gripper_frame(), float)


def settle(env, timeout_s: float = 4.0, tol_mm: float = 0.2) -> float:
    """
    Hold the arm where it is, then wait for it to stop: the last 0.2 s step of
    the gripper, mm. Needed after every move, and above all after one that
    stalled short: on the first tour the servos crept on toward their last goal
    for seconds, up to 10 mm during the frames, and those hovers were useless.
    """
    env.hold()
    t0, prev, step = time.time(), fk(env)[:3, 3], float("inf")
    while time.time() - t0 < timeout_s:
        time.sleep(0.2)
        cur = fk(env)[:3, 3]
        step = float(np.linalg.norm(cur - prev) * 1000)
        if step < tol_mm:
            break
        prev = cur
    return step


def _tour_squares() -> dict:
    """
    Where to hover, {square: {"world": [x, y]}}: from the board's pose on the
    arm as last measured (overhead_calibration), which a moved camera does not
    change; else the chess deployment's square map, which it does.
    """
    from mrcf import overhead_calibration
    saved = overhead_calibration.load()
    if saved and saved.get("T_arm_target") is not None:
        T = np.asarray(saved["T_arm_target"], float)
        p = cfg.CHESSBOARD_SQUARE_SIZE_M
        return {f"{f}{r}": {"world": (T @ np.array([("abcdefgh".index(f) - 3.5) * p, (r - 4.5) * p, 0, 1]))[:2].tolist()}
                for f in "abcdefgh" for r in range(1, 9)}
    return json.loads((FRAMEWORK_ROOT / "deployments/chess/board_squares.json").read_text())


def run_tour(env=None, camera: "str | None" = None) -> Path:
    """
    The tour: hovers, frames, joint angles, logged to a new run folder (returned).
    env: an arm already connected (a running deployment's, under its sentry);
    None connects one here, and disconnects it after. camera: the fixed camera
    it watches through (None: the overhead one).
    """
    camera = fixed_camera(camera or rc.OVERHEAD_CAMERA_NAME)
    K, D = lens(camera)
    squares = _tour_squares()
    run = RUNS / time.strftime("%Y%m%d_%H%M%S")
    (run / "frames").mkdir(parents=True)
    (run / "camera.txt").write_text(camera, encoding="utf-8")
    own = env is None
    if own:
        env = make_env()
        print(f"Connecting to {connecting_to()}...")
        env.connect()
    log = open(run / "poses.jsonl", "w", encoding="utf-8")
    try:
        env.go_home()
        before = clean_board(env, run, "before", K, D, camera)
        env.go_home()
        misses = 0
        for i, sq in enumerate(TOUR):
            if sq not in squares:
                continue
            x, y = squares[sq]["world"]
            target = np.array([x, y, rc.TABLE_Z + HOVER_M])
            tilt = tour_lean(sq)
            chosen = (env.approach_orientation_for([target], tols=[0.01], prefer_tilt_deg=tilt)
                      if isinstance(env, OrientationSearch) else (None, 0.0))   # else straight down
            if chosen is None:
                print(f"  {sq}: no orientation reaches it — skipped")
                continue
            rot, used_tilt = chosen
            ok = env.move_ee_to(target, tol=0.008, orientation=rot)
            for _ in range(2):                       # capture; again if it moved meanwhile
                settle(env)
                T = fk(env)
                frames = grab(env, FRAMES_PER_POSE, camera)
                wrist = (env.render_image(rc.CAMERA_NAME, width=rc.CAMERA_W, height=rc.CAMERA_H)
                         if rc.CAMERA_NAME is not None else None)
                if np.linalg.norm(fk(env)[:3, 3] - T[:3, 3]) * 1000 < STILL_MM:
                    break
            dets = [fiducials.detect_tags(f) for f in frames]
            tags = {}
            for i in set().union(*dets):
                qs = [d[i] for d in dets if i in d]
                tags[str(i)] = {"median": np.median(qs, axis=0).tolist(), "n": len(qs),
                                "spread_px": float(np.median([np.abs(q - np.median(qs, axis=0)).max() for q in qs]))}
            marks = [m.centre for m in fiducials.find_bullseyes(wrist)] if wrist is not None else []
            T_after = fk(env)
            rec = {"square": sq, "target": target.tolist(), "reached": bool(ok), "tilt_deg": float(used_tilt),
                   "joints_deg": {k: float(np.degrees(v)) for k, v in env.get_joint_angles().items()}, "T_arm_gripper_frame": T.tolist(),
                   "moved_mm_during_capture": float(np.linalg.norm(T_after[:3, 3] - T[:3, 3]) * 1000),
                   "tags": tags, "jaw_marks_px": [[float(a), float(b)] for a, b in marks],
                   "ee_pos": [float(v) for v in env.get_ee_pos()]}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            privacy.imwrite(str(run / "frames" / f"{sq}.jpg"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if wrist is not None:
                privacy.imwrite(str(run / "frames" / f"{sq}_wrist.jpg"), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
            g = tags.get(str(rc.GRIPPER_TAG_ID))
            nb = sum(1 for i in cfg.BOARD_TAG_IDS if str(i) in tags)
            print(f"  {sq}: {'reached' if ok else 'NOT reached'}, lean {abs(used_tilt):.0f} deg | board tags {nb}/"
                  f"{len(cfg.BOARD_TAG_IDS)} | gripper tag {('%d/%d' % (g['n'], FRAMES_PER_POSE)) if g else 'NOT SEEN'}"
                  f" | jaw marks {len(marks)}", flush=True)
            # A tag missing from the start, or gone for several hovers in a row
            # (one fell off mid-run), makes the rest of the tour pointless.
            misses = 0 if g else misses + 1
            if (i == 0 and not g) or misses >= MAX_MISSES:
                print(f"  Gripper tag {rc.GRIPPER_TAG_ID} not seen {'at the first hover' if i == 0 else f'{misses} hovers running'}"
                      " — is it still on? Stopping the tour.")
                break
        env.go_home()
        after = clean_board(env, run, "after", K, D, camera)
        for label, b in (("before", before), ("after", after)):
            if b is not None:
                np.savez(run / f"board_{label}.npz", T_cam_board=b["T_cam_board"],
                         ids=np.array(sorted(b["tags"])), tags=np.array([b["tags"][i] for i in sorted(b["tags"])]),
                         lattice_px=b["lattice_px"])
    finally:
        log.close()
        try:
            env.go_home()
        finally:
            if own:
                env.stop()
    print(f"\nLogged to {run}")
    return run


# ── Analysis ──────────────────────────────────────────────────────────────────

def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _from_params(p):
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = cv2.Rodrigues(p[:3])[0], p[3:6]
    Y = np.eye(4)
    Y[:3, :3], Y[:3, 3] = cv2.Rodrigues(p[6:9])[0], p[9:12]
    return X, Y


def refine(rows: list, T_board_arm: np.ndarray, T_gf_tag: np.ndarray, K, D, iters: int = 40):
    """
    The closed form is a start, not the answer: it fits poses, and a small tag's
    pose is weakest exactly where the tour's rotations give least leverage (on
    synthetic data it left the tag's offset along the jaw axis and the board's
    height 6-7 mm off, together). This fits what the camera measured — the
    gripper tag's corner pixels, every hover — by Gauss-Newton on both
    transforms and the tag's printed size, which a printer's scaling changes
    (the first tour read a 24 mm tag as ~23.3 mm). Returns (T_board_arm,
    T_gf_tag, tag size m, mean reprojection px).
    """
    size0 = rc.GRIPPER_TAG_SIZES_M[rc.GRIPPER_TAG_ID]

    def residuals(p):
        X, Y = _from_params(p)
        obj_h = np.c_[fiducials.tag_object_points(size0 * (1 + p[12])), np.ones(4)].T
        out = []
        for row in rows:
            Pc = (row["T_cam_board"] @ X @ row["T_arm_gf"] @ Y @ obj_h)[:3].T
            px, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, D)
            out.append((px.reshape(-1, 2) - row["corners"]).ravel())
        return np.concatenate(out)

    p = np.concatenate([cv2.Rodrigues(T_board_arm[:3, :3])[0].ravel(), T_board_arm[:3, 3],
                        cv2.Rodrigues(T_gf_tag[:3, :3])[0].ravel(), T_gf_tag[:3, 3], [0.0]])
    lam = 1e-3
    r = residuals(p)
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for j in range(p.size):
            dp = np.zeros(p.size)
            dp[j] = 1e-6
            J[:, j] = (residuals(p + dp) - r) / 1e-6
        if not (np.isfinite(J).all() and np.isfinite(r).all()):
            break
        # numpy on Apple's Accelerate warns "divide by zero in matmul" on finite
        # input (it does for random data); the check above is the real guard.
        with np.errstate(all="ignore"):
            A, g = J.T @ J, J.T @ r
        step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
        r_new = residuals(p + step)
        if r_new @ r_new < r @ r:
            p, r, lam = p + step, r_new, lam * 0.3
            if np.abs(step).max() < 1e-9:
                break
        else:
            lam *= 10
    X, Y = _from_params(p)
    return X, Y, size0 * (1 + p[12]), float(np.sqrt((r.reshape(-1, 2) ** 2).sum(1)).mean())


def analyse(run: Path) -> None:
    K, D = lens(run_camera(run))
    # The board's pose from the clean parked frame's lattice: far more exact than
    # any one hover's board tags. Used for every hover when nothing moved between
    # the frames before and after the tour (each hover's tags are then a check).
    boards = {}
    for label in ("before", "after"):
        png = run / f"board_{label}.png"
        if png.exists():
            b = measure_board(cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB), K, D)
            if b is not None:
                boards[label] = b
    if not boards:
        raise SystemExit("The board was not found in either clean frame — cannot place it.")
    board = boards.get("before") or boards["after"]
    print_board(board, "board")
    if len(boards) == 2:
        moved = np.linalg.norm(boards["before"]["T_cam_board"][:3, 3] - boards["after"]["T_cam_board"][:3, 3]) * 1000
        turned = fiducials.rotation_angle_deg(boards["before"]["T_cam_board"][:3, :3],
                                              boards["after"]["T_cam_board"][:3, :3])
        print(f"  board vs camera, before vs after the tour: moved {moved:.2f} mm, turned {turned:.3f} deg")
    T_cam_board = board["T_cam_board"]
    layout = {i: P for i, P in board["tags"].items() if i in cfg.BOARD_TAG_IDS}
    recs = [json.loads(line) for line in (run / "poses.jsonl").read_text().splitlines() if line.strip()]
    g, size = rc.GRIPPER_TAG_ID, rc.GRIPPER_TAG_SIZES_M[rc.GRIPPER_TAG_ID]
    rows = []
    moving = [r["square"] for r in recs if r.get("moved_mm_during_capture", 0.0) >= STILL_MM]
    if moving:
        print(f"Left out, the arm moving while their frames were taken: {', '.join(moving)}")
    for r in recs:
        found = {int(i): np.array(v["median"]) for i, v in r["tags"].items()}
        if g not in found or r["square"] in moving:
            continue
        sol = fiducials.layout_pose({i: q for i, q in found.items() if i in layout}, layout, K, D)
        n_board = sol[1] if sol else 0
        # how far this hover's board tags would put the board from the lattice's pose
        board_err = (float(np.linalg.norm((sol[0] @ _inv(T_cam_board))[:3, 3]) * 1000) if sol else float("nan"))
        cands = [p.matrix() for p in fiducials.tag_poses(found[g], size, K, D)]
        if not cands:
            continue
        rows.append({"square": r["square"], "T_arm_gf": np.array(r["T_arm_gripper_frame"]),
                     "T_board_tag": [_inv(T_cam_board) @ c for c in cands], "n_board": n_board,
                     "board_err": board_err, "tilt": r["tilt_deg"], "T_cam_board": T_cam_board,
                     "corners": found[g]})
    print(f"{len(rows)} of {len(recs)} hovers usable (still, gripper tag seen); each hover's board tags agree with the "
          f"lattice's board to {np.nanmedian([r['board_err'] for r in rows]):.1f} mm median")
    if len(rows) < 4:
        raise SystemExit("Too few hovers to solve.")
    pick = [0] * len(rows)
    for _ in range(3):
        A = [_inv(row["T_board_tag"][k]) for row, k in zip(rows, pick)]       # board -> tag
        B = [_inv(row["T_arm_gf"]) for row in rows]                           # arm base -> gripper
        R_bw, t_bw, R_gc, t_gc = cv2.calibrateRobotWorldHandEye(
            [a[:3, :3] for a in A], [a[:3, 3] for a in A], [x[:3, :3] for x in B], [x[:3, 3] for x in B])
        T_board_arm = np.vstack([np.c_[R_bw, t_bw.ravel()], [0, 0, 0, 1]])  # arm base -> board
        T_tag_gf = np.vstack([np.c_[R_gc, t_gc.ravel()], [0, 0, 0, 1]])     # gripper frame -> tag
        T_gf_tag = _inv(T_tag_gf)
        new = []
        for row in rows:
            pred = T_board_arm @ row["T_arm_gf"] @ T_gf_tag
            new.append(int(np.argmin([fiducials.rotation_angle_deg(c[:3, :3], pred[:3, :3])
                                      for c in row["T_board_tag"]])))
        if new == pick:
            break
        pick = new
    T_board_arm, T_gf_tag, size, rms = refine(rows, T_board_arm, T_gf_tag, K, D)
    T_tag_gf = _inv(T_gf_tag)
    for row in rows:                        # the tag's own poses again, at its fitted size
        cands = [p.matrix() for p in fiducials.tag_poses(row["corners"], size, K, D)]
        if cands:
            row["T_board_tag"] = [_inv(row["T_cam_board"]) @ c for c in cands]
    for i, row in enumerate(rows):          # the pose solution nearest the fitted model, for the report
        pred = T_board_arm @ row["T_arm_gf"] @ T_gf_tag
        pick[i] = int(np.argmin([fiducials.rotation_angle_deg(c[:3, :3], pred[:3, :3]) for c in row["T_board_tag"]]))
    print(f"Fitted to the gripper tag's corners in every hover: {rms:.2f} px mean reprojection; the tag reads "
          f"{size * 1000:.2f} mm across (config: {rc.GRIPPER_TAG_SIZES_M[rc.GRIPPER_TAG_ID] * 1000:.1f}) — "
          f"measure the printed one with a ruler")
    print("\nThe two unknowns:")
    tb = T_gf_tag[:3, 3] * 1000
    print(f"  tag on the gripper: ({tb[0]:+.0f}, {tb[1]:+.0f}, {tb[2]:+.0f}) mm from the closed-jaw grasp point"
          f" (gripper frame), face {np.degrees(np.arccos(np.clip(T_gf_tag[2, 2], -1, 1))):.0f} deg from the jaw axis")
    base = _inv(T_board_arm)[:3, 3]
    print(f"  arm base on the board: the board centre is at ({base[0] * 1000:+.0f}, {base[1] * 1000:+.0f}, "
          f"{base[2] * 1000:+.0f}) mm in the arm frame")
    print("\nPer square: where the camera puts the closed-jaw grasp point minus where the arm's kinematics put it,"
          " in the arm frame (mm):")
    res = []
    for row, k in zip(rows, pick):
        meas = _inv(T_board_arm) @ row["T_board_tag"][k] @ T_tag_gf        # gripper frame, as the tag says
        pred = row["T_arm_gf"]
        d = (meas[:3, 3] - pred[:3, 3]) * 1000
        ang = fiducials.rotation_angle_deg(meas[:3, :3], pred[:3, :3])
        res.append(d)
        alt = "" if k == 0 else "  (tag pose: the 2nd solution, by agreement)"
        print(f"  {row['square']}: dx {d[0]:+6.1f} dy {d[1]:+6.1f} dz {d[2]:+6.1f}  |xy| {np.hypot(d[0], d[1]):5.1f}"
              f"  turn {ang:4.1f} deg  lean {abs(row['tilt']):2.0f}  board tags {row['n_board']}{alt}")
    res = np.array(res)
    xy = np.hypot(res[:, 0], res[:, 1])
    print(f"\n|xy| median {np.median(xy):.1f} mm, 90th pct {np.percentile(xy, 90):.1f}, max {xy.max():.1f}; "
          f"|z| median {np.median(np.abs(res[:, 2])):.1f} mm")
    # Where the arm's last calibration put the squares (board_squares.json) vs
    # where the camera and the tags put them now: board frame x toward h, y toward rank 8.
    sq = json.loads((FRAMEWORK_ROOT / "deployments/chess/board_squares.json").read_text())
    pitch = cfg.CHESSBOARD_SQUARE_SIZE_M
    offs = []
    for name, v in sq.items():
        f, r = "abcdefgh".index(name[0]), int(name[1]) - 1
        now = (_inv(T_board_arm) @ np.array([(f - 3.5) * pitch, (r - 3.5) * pitch, 0.0, 1.0]))[:3]
        offs.append((now[:2] - np.array(v["world"])) * 1000)
    offs = np.array(offs)
    mean = offs.mean(0)
    print(f"Squares now vs board_squares.json (the arm's last calibration): shifted ({mean[0]:+.1f}, {mean[1]:+.1f}) mm "
          f"on average (arm frame x, y), spread {np.linalg.norm(offs - mean, axis=1).max():.1f} mm max around that")
    np.savez(run / "fit.npz", T_board_arm=T_board_arm, T_gf_tag=T_gf_tag, tag_size_m=size, residuals_mm=res,
             squares=np.array([row["square"] for row in rows]))
    print(f"Saved {run / 'fit.npz'}")


def calibrate(run: Path) -> dict:
    """
    The overhead camera's calibration, from a tour instead of corner touches:
    its pose in the arm frame is the lattice's (camera -> board) followed by the
    tour's fit (board -> arm). Saves it (overhead_calibration.json, which
    robot_config applies over its own values) and returns it; rewrites the chess
    deployment's lattice cache and square map for the camera where it is now;
    prints the values too, for robot_config's fallbacks.
    """
    from mrcf import overhead_calibration
    import shutil
    from deployments.chess import chessboard
    camera = run_camera(run)
    K, D = lens(camera)
    cap_w, cap_h = camera_size(camera)
    fit = np.load(run / "fit.npz")
    T_board_arm, T_gf_tag, size = fit["T_board_arm"], fit["T_gf_tag"], float(fit["tag_size_m"])
    rgb = cv2.cvtColor(cv2.imread(str(run / "board_before.png")), cv2.COLOR_BGR2RGB)
    board = measure_board(rgb, K, D)
    if board is None:
        raise SystemExit("Board not found in the run's clean frame.")
    T_arm_board = _inv(T_board_arm)
    T_arm_cam = T_arm_board @ _inv(board["T_cam_board"])
    corners = board["lattice_all_px"]
    arm = (T_arm_board[:3, :3] @ lattice_board_points().T).T + T_arm_board[:3, 3]
    s = np.array([rc.RENDER_W / cap_w, rc.RENDER_H / cap_h])
    H, _ = cv2.findHomography(corners * s, arm[:, :2], 0)
    h = (H @ np.c_[corners * s, np.ones(len(corners))].T).T
    res = np.linalg.norm(h[:, :2] / h[:, 2:] - arm[:, :2], axis=1) * 1000
    squares, pitch, rms = chessboard.square_map(corners, H, (cap_w, cap_h), (rc.RENDER_W, rc.RENDER_H))
    warning = chessboard.check_orientation(rgb, squares)
    if warning:
        raise SystemExit(f"The new lattice fails the a1-dark check: {warning}")
    fmt = lambda M: "[\n" + "".join("        [" + ", ".join(f"{v:.6f}" for v in row) + "],\n" for row in M) + "    ]"
    if camera != rc.OVERHEAD_CAMERA_NAME:
        # Another fixed camera: its pose and homography for its CAMERAS entry.
        # The overhead camera's live calibration and the chess caches are its own.
        print(f"# robot_config.py, CAMERAS[{camera!r}]")
        print("homography=[\n" + "".join("    [" + ", ".join(repr(float(v)) for v in row) + "],\n" for row in H) + "]")
        print("model T_world_cam: " + fmt(T_arm_cam))
        return {"camera": camera, "T_arm_cam": T_arm_cam, "homography": H}
    chess = FRAMEWORK_ROOT / "deployments" / "chess"
    for name in ("board_corners.npy", "board_squares.json"):
        if (chess / name).exists():
            shutil.copy(chess / name, run / f"stale_{name}")
    np.save(chess / "board_corners.npy", corners)
    (chess / "board_squares.json").write_text(json.dumps(
        {n: {"world": [float(q["world"][0]), float(q["world"][1])], "pixel": [float(q["pixel"][0]), float(q["pixel"][1])]}
         for n, q in squares.items()}, indent=2))
    print(f"Lattice cache and square map rewritten ({len(squares)} squares, {pitch * 1000:.1f} mm, a1 dark); "
          f"the old ones are in {run}.")
    print(f"Homography residual over the 49 corners: {res.mean():.2f} mm mean, {res.max():.2f} mm max "
          f"(it cannot model the lens's barrel distortion)\n")
    print("# robot_config.py")
    print("OVERHEAD_HOMOGRAPHY = [\n" + "".join("    [" + ", ".join(repr(float(v)) for v in row) + "],\n" for row in H) + "]")
    print("OVERHEAD_CAM_MODEL T_world_cam: " + fmt(T_arm_cam))
    print(f"GRIPPER_TAG_POSE: {fmt(T_gf_tag)}")
    print(f"GRIPPER_TAG_READS_AS_M = {size:.5f}")
    data = {"T_arm_target": T_arm_board, "T_arm_cam": T_arm_cam, "homography": H, "T_gf_tag": T_gf_tag,
            "tag_size_m": size, "source": f"tag tour {run.name}", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target": "chess board: origin at its centre, x toward the h-file, y toward rank 8"}
    overhead_calibration.save(data)
    print(f"\nSaved to {overhead_calibration.PATH} — what robot_config, and so every tool, now starts from.")
    return overhead_calibration.load()


def tour_and_calibrate(env) -> dict:
    """
    The whole re-measurement, for a running arm (under its sentry): tour,
    fit, save, and put the result on env. What a deployment calls when the
    arm and the overhead camera disagree about where the gripper is — the
    board or the arm's base has moved, which one still frame cannot measure.
    """
    from mrcf import overhead_calibration
    run = run_tour(env)
    analyse(run)
    data = calibrate(run)
    overhead_calibration.apply_to_env(env, data)
    overhead_calibration.apply_to(vars(rc), data)
    return data


def main() -> None:
    if "--calibrate" in sys.argv:
        i = sys.argv.index("--calibrate")
        calibrate(Path(sys.argv[i + 1]) if i + 1 < len(sys.argv) else max(RUNS.iterdir()))
        return
    if "--analyse" in sys.argv:
        i = sys.argv.index("--analyse")
        run = Path(sys.argv[i + 1]) if i + 1 < len(sys.argv) else max(RUNS.iterdir())
        analyse(run)
        return
    run = run_tour(camera=chosen_camera() if "--camera" in sys.argv else None)
    analyse(run)


if __name__ == "__main__":
    main()
