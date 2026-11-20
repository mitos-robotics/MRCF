"""
calibrate_overhead_camera.py — calibrate the fixed overhead camera's
pixel -> robot-base-frame (x, y) mapping, using a chessboard as a dense,
automatically-detected calibration target.

Why a chessboard instead of touching every point with the arm: OpenCV's
chessboard-corner detector finds dozens of corners per photo with sub-pixel
accuracy automatically — far more precise and far less manual work than
clicking each point by hand. But knowing WHERE those corners are in the
robot's own coordinate frame (the frame everything else in this deployment —
TABLE_Z, HOME_DEGREES, etc. — is already measured in) still has to come from
somewhere, and a ruler measurement from the robot's base to the board is
typically less accurate than the robot's own encoders. So this script asks
you to touch just 3 of the board's corners with the gripper fingertip, which
pins the whole detected grid into the robot's frame. The script chooses which
3 and shows you each one circled, so there is nothing to identify or type.

grid -> world is fitted as a SIMILARITY (scale + rotation + translation, with
a mirror allowed), never a general affine: a chessboard's cells are square, so
constraining the fit to shapes a real board can have turns a mis-touched
corner into a visible residual instead of a silent skew.

Procedure:
  1. Place the chessboard flat on the table, fully inside the overhead
     camera's view, and leave it there until step 3 is done.
  2. The script detects all internal corners automatically.
  3. Three times: an image opens with ONE corner circled in red. Put the
     fingertip on that corner and press Enter. Nothing to read or type.
  4. It fits a homography from all detected corners and prints
     OVERHEAD_HOMOGRAPHY ready to paste into config.py — and, from the same
     photo, OVERHEAD_CAM_MODEL and ARM_FRAME_SCALE (the camera itself, for
     measuring objects on any surface; see mrcf/perception/plane_geometry.py).

Needs the arm's kinematics (get_ee_pos; on the SO-101, placo + the fetched
URDF), and a way to move it by hand: the env's disable_torque() and
enable_torque() when it has them (the SO-101's torque off), otherwise the arm's
own hand-guiding (a UR's freedrive), switched on by you. No GUI window is
used; target images are written to disk and opened in the OS viewer.

Usage:
    python tools/calibrate_overhead_camera.py
    python tools/calibrate_overhead_camera.py --camera side   # another fixed camera, for its CAMERAS entry
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from tools._common import (ARM, camera_size, chosen_camera, connecting_to, fixed_camera, make_env,  # noqa: E402
                           task_config)
from mrcf import privacy  # noqa: E402

cfg = task_config()   # the calibration target's dimensions are a task setting

N_ANCHORS = 3
# Capture resolution used ONLY for this script's own chessboard detection —
# independent of config.py's OVERHEAD_CAMERA_W/H (which stays small for fast
# VLM perception at runtime). More pixels per square meaningfully helps corner
# detection on an oblique view, especially with a low-contrast board or glare.
# 1280x720 is a verified-clean mode on this camera (requested == get() ==
# actual frame shape — see conversation history for the probe that checked
# this; re-probe if using a different camera). Detected corners get scaled
# down to RENDER_W/RENDER_H before fitting, so the saved OVERHEAD_HOMOGRAPHY
# stays correct for whatever resolution discover_scene() actually uses.
CAPTURE_W, CAPTURE_H, CAPTURE_FPS = 1280, 720, 30


def _preprocess_variants(frame_rgb: np.ndarray) -> list[np.ndarray]:
    """
    A few different grayscale versions of the same frame to try detection on,
    in order — cheap to attempt, and each helps with a different failure mode
    (uneven lighting/glare, low black/white contrast, one bad color channel).
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    variants = [gray]
    # The CLAHE clip limit matters as much as which channel: 3.0 rescues a
    # washed-out or unevenly lit board, 1.0 a board whose squares already
    # separate cleanly and that 3.0 over-amplifies into noise. One real frame
    # from this rig is found ONLY at clip 1.0 on the blue channel, another
    # only at 3.0 on green, so both passes earn their keep.
    for clip in (3.0, 1.0):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        variants.append(clahe.apply(gray))
        # Individual colour channels often separate two specific board colours
        # (brown/cream) far better than the luminance-weighted gray does.
        for i in range(3):
            variants.append(clahe.apply(frame_rgb[:, :, i]))
    # Saturation ignores brightness entirely, so it survives the glare and the
    # shadow gradient that a window throws across the board.
    variants.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)[:, :, 1])
    return variants


def _detect_chessboard(frame_rgb: np.ndarray) -> "np.ndarray | None":
    """(N, 2) sub-pixel corner pixel coords in OpenCV's own row-major order, or None if not found."""
    pattern_size = (cfg.CHESSBOARD_INTERNAL_COLS, cfg.CHESSBOARD_INTERNAL_ROWS)

    for variant in _preprocess_variants(frame_rgb):
        # findChessboardCornersSB is newer and considerably more tolerant of
        # uneven lighting, blur, and low contrast than the classic detector —
        # try it first on each preprocessing variant.
        found, corners = cv2.findChessboardCornersSB(
            variant, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )
        if found:
            return corners.reshape(-1, 2)

        # Classic detector's flags default to 0 (no adaptive thresholding/gamma
        # normalization) if not passed explicitly — pass the flags that make it
        # actually robust to real lighting, as a fallback.
        found, corners = cv2.findChessboardCorners(
            variant, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        )
        if found:
            cv2.cornerSubPix(
                variant, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
            )
            return corners.reshape(-1, 2)

    return None


def _grid_index(flat_i: int) -> "tuple[int, int]":
    """(row, col) for a flat index into the (rows*cols) corner array OpenCV returns."""
    cols = cfg.CHESSBOARD_INTERNAL_COLS
    return flat_i // cols, flat_i % cols


def _fit_similarity(src: np.ndarray, dst: np.ndarray):
    """
    Least-squares similarity (uniform scale + rotation + translation) mapping
    grid coordinates to world coordinates — the Umeyama solution.

    Deliberately NOT a general affine fit. A chessboard's cells are square and
    its axes perpendicular, so the grid->world map has only 4 degrees of freedom
    (scale, rotation, 2x translation). A general affine has 6, and those extra
    two let a single mis-identified anchor silently skew the result into a
    shape no physical board can have — which produces a homography that fits
    its own bad data beautifully while being completely wrong.

    Constraining the fit means squareness can't be violated, and the leftover
    residual becomes a real error signal: with 3 anchors there are 6 equations
    for 4 unknowns, so a wrong anchor has nowhere to hide.

    Reflections are ALLOWED here, unlike the textbook Umeyama fit. The corner
    indices come from the image, and a camera looking down at a z-up table has
    image axes that are left-handed with respect to the world, so grid->world
    legitimately includes a mirror. Forbidding it makes a perfectly good set of
    anchors unfittable. Allowing it adds one discrete choice, not continuous
    freedom, so squareness is still enforced and the residual is still a valid
    error signal.

    Returns (scale, R, t, rms_residual_m).
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = (d_c.T @ s_c) / len(src)
    U, sv, Vt = np.linalg.svd(cov)
    var_s = (s_c ** 2).sum() / len(src)

    best = None
    for flip in (1.0, -1.0):
        d = np.array([1.0, flip])
        R = U @ np.diag(d) @ Vt
        scale = float((sv * d).sum() / var_s) if var_s > 0 else 0.0
        if scale <= 0:
            continue
        t = mu_d - scale * (R @ mu_s)
        pred = (scale * (R @ src.T)).T + t
        rms = float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))
        if best is None or rms < best[3]:
            best = (scale, R, t, rms)
    return best



def _open_image(path: Path) -> None:
    """Open an image in the OS viewer so the user doesn't have to go find it."""
    import subprocess
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)
        elif sys.platform == "win32":
            os.startfile(str(path))  # noqa: F821 — windows-only
    except Exception as exc:
        print(f"  (couldn't open it automatically: {exc} — open it manually)")


def _anchor_consistency_error(new_anchor, anchors) -> "str | None":
    """
    Check a just-entered anchor against the ones already taken.

    Any two corners a known number of squares apart must be that many squares
    apart in the world too. Checking immediately means a mistyped number is
    caught while it can still be re-done in isolation, instead of surfacing as
    an unattributable whole-calibration failure three touches later.
    """
    r1, c1, x1, y1 = new_anchor
    for (r2, c2, x2, y2) in anchors:
        grid_d = float(np.hypot(r1 - r2, c1 - c2))
        if grid_d == 0:
            continue
        expected = grid_d * cfg.CHESSBOARD_SQUARE_SIZE_M
        measured = float(np.hypot(x1 - x2, y1 - y2))
        if abs(measured - expected) > max(0.02, 0.3 * expected):
            return (f"this corner is {measured * 1000:.0f} mm from a previous anchor, but their "
                    f"grid positions are {grid_d:.1f} squares apart, which should be "
                    f"{expected * 1000:.0f} mm")
    return None


def _save_single_target(frame_rgb: np.ndarray, corners: np.ndarray, idx: int, path: Path) -> None:
    """
    Write the frame with exactly ONE corner highlighted.

    The script already knows which corner it wants, so asking the user to read
    a number off a map of 49 of them and type it back only creates a chance to
    get it wrong — which is precisely the error that kept wrecking the fit.
    Showing one unmistakable target removes the identification step entirely.
    """
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    # every other corner faint, so the board is still legible for orientation
    for i, (cx, cy) in enumerate(corners):
        if i != idx:
            cv2.circle(img, (int(cx), int(cy)), 3, (120, 120, 120), -1)
    x, y = int(corners[idx][0]), int(corners[idx][1])
    h, w = img.shape[:2]
    cv2.line(img, (0, y), (w, y), (0, 0, 255), 1)
    cv2.line(img, (x, 0), (x, h), (0, 0, 255), 1)
    cv2.circle(img, (x, y), 26, (0, 0, 255), 3)
    cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
    cv2.putText(img, "TOUCH THIS CORNER", (max(8, x - 150), max(32, y - 36)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    privacy.imwrite(str(path), img)


def _pick_anchor_corners() -> list:
    """
    Three grid corners for the script to ask for: two ends of one edge plus one
    end of the adjacent edge. Maximally separated (best-conditioned fit) and
    guaranteed non-collinear.
    """
    rows, cols = cfg.CHESSBOARD_INTERNAL_ROWS, cfg.CHESSBOARD_INTERNAL_COLS
    return [0, cols - 1, (rows - 1) * cols]


def main() -> None:
    # The overhead camera, or another fixed one: --camera NAME (any of the arm's
    # CAMERAS). Another one's calibration is printed for its Camera entry.
    camera = fixed_camera(chosen_camera())
    overhead = camera == cfg.OVERHEAD_CAMERA_NAME
    # The overhead camera is asked for CAPTURE_W x CAPTURE_H (above); another at its own size.
    cap_w, cap_h = (CAPTURE_W, CAPTURE_H) if overhead else camera_size(camera)

    try:
        env = make_env(overhead_camera_width=CAPTURE_W, overhead_camera_height=CAPTURE_H,
                       overhead_camera_fps=CAPTURE_FPS)
    except TypeError as exc:
        if "overhead_camera_" not in str(exc):
            raise                     # a fault in the arm's own construction, not the keyword
        # An arm whose make_env() does not take a capture size: its own, resized
        # to CAPTURE_W x CAPTURE_H, which detects fewer corners on a small one.
        env = make_env()
    print(f"Connecting to {connecting_to()} (camera '{camera}')...")
    env.connect()

    try:
        env.get_ee_pos()  # fail fast if placo/URDF aren't set up, before torque is touched
    except NotImplementedError as exc:
        print(f"\n{exc}")
        env.stop()
        return

    # The arm's park pose, out of the camera's view (on the SO-101,
    # CALIBRATION_PARK_DEGREES from find_poses.py's 'calib', else home).
    print("Parking the arm so it isn't occluding the board...")
    env.go_park()

    print("Looking for the chessboard in the overhead camera view...")
    frame = env.render_image(camera, width=cap_w, height=cap_h)
    corners = _detect_chessboard(frame)
    if corners is None:
        debug_path = Path(__file__).parent / "calibrate_overhead_debug.png"
        privacy.save_rgb(debug_path, frame)
        print(f"Could not find a {cfg.CHESSBOARD_INTERNAL_COLS}x{cfg.CHESSBOARD_INTERNAL_ROWS}-corner "
              f"chessboard in the image. Saved exactly what the camera captured to {debug_path} — "
              "check it's fully in frame, flat, well-lit, in focus, and that CHESSBOARD_INTERNAL_COLS/"
              "ROWS in config.py match your board (squares - 1 per side).")
        env.stop()
        return
    print(f"Found {len(corners)} corners.\n")

    target_path = Path(__file__).parent / "calibrate_overhead_target.png"

    hand_guided = hasattr(env, "disable_torque")
    if hand_guided:
        print("Disabling torque — the arm is free to move by hand.")
    else:
        print(f"Switch on the {ARM.NAME}'s own hand-guiding (freedrive) to move it by hand.")
    print("For each of 3 corners: an image opens with ONE corner circled in red.")
    print("CLOSE the jaws first, then touch that corner with the point where the two")
    print("fingertips MEET, and press Enter. Nothing to type.")
    print("  (Not 'a fingertip': get_ee_pos() reports the point midway between the")
    print("   jaws, so touching with one finger bakes a half-jaw-width offset into the")
    print("   homography — and at whatever wrist angle you happened to hold, which")
    print("   is not the wrist angle a grasp uses.)\n")
    if hand_guided:
        env.disable_torque()

    anchors: list[tuple[int, int, float, float]] = []  # (row, col, x, y)
    # The touches are on the board's surface, so their height is TABLE_Z: worth
    # checking whenever the setup has moved.
    touched_z: list[float] = []

    try:
        for n, idx in enumerate(_pick_anchor_corners(), 1):
            r, c = _grid_index(idx)
            while True:
                _save_single_target(frame, corners, idx, target_path)
                _open_image(target_path)
                input(f"Anchor {n}/{N_ANCHORS} — touch the CIRCLED corner (see the image that "
                      f"just opened), then press Enter> ")
                ee = env.get_ee_pos()

                candidate = (r, c, float(ee[0]), float(ee[1]))
                problem = _anchor_consistency_error(candidate, anchors)
                if problem is not None:
                    print(f"  REJECTED — {problem}.")
                    print("  Most likely the closed jaws' contact point wasn't on the circled "
                          "corner. Retrying.\n")
                    continue

                anchors.append(candidate)
                touched_z.append(float(ee[2]))
                print(f"  Anchor {n}: grid ({r},{c}) <-> world ({ee[0]:.3f}, {ee[1]:.3f})"
                      f"  z {ee[2] * 1000:+.1f} mm\n")
                break
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        return
    finally:
        if hand_guided:
            env.enable_torque()
        env.stop()

    # Fit grid(col, row) -> world(x, y) as a SIMILARITY, so the result is
    # guaranteed to describe a square, unskewed grid (see _fit_similarity).
    grid_pts = np.array([[c, r] for (r, c, _x, _y) in anchors], dtype=float)
    world_anchors = np.array([[x, y] for (_r, _c, x, y) in anchors], dtype=float)
    if np.linalg.matrix_rank(grid_pts - grid_pts.mean(0)) < 2:
        print("\nThe touched anchors are collinear in the grid — they can't pin down a rotation. "
              "Re-run and pick corners that aren't in a straight line (e.g. two along one edge "
              "and one along an adjacent edge).")
        return

    square_m, R, t, rms = _fit_similarity(grid_pts, world_anchors)

    print(f"\nDerived square size: {square_m * 1000:.1f} mm "
          f"(you measured CHESSBOARD_SQUARE_SIZE_M = {cfg.CHESSBOARD_SQUARE_SIZE_M * 1000:.1f} mm)")
    print(f"Anchor fit residual: {rms * 1000:.1f} mm")

    # Hard failures, not warnings. A bad fit here produces a homography that
    # looks excellent by its own reprojection error while being entirely wrong,
    # and that costs far more to debug later than redoing three touches now.
    bad = []
    if abs(square_m - cfg.CHESSBOARD_SQUARE_SIZE_M) > cfg.CHESSBOARD_SQUARE_SIZE_M * 0.20:
        bad.append(f"derived square size {square_m * 1000:.1f} mm is >20% from the "
                   f"{cfg.CHESSBOARD_SQUARE_SIZE_M * 1000:.0f} mm you measured")
    if rms > 0.01:
        bad.append(f"anchor residual {rms * 1000:.1f} mm is too large for a rigid square grid "
                   "— at least one touched corner doesn't match the number entered for it")
    if bad:
        print("\n" + "=" * 60)
        print("CALIBRATION REJECTED — not writing a homography:")
        for b in bad:
            print(f"  - {b}")
        print("=" * 60)
        print("\nMost likely cause: a corner number was mistyped, or a different corner was")
        print("touched than the one entered. Re-run, and after each touch check the number")
        print("against calibrate_overhead_corners.png before pressing Enter.")
        return

    world_pts = np.array([
        (square_m * (R @ np.array([float(c), float(r)]))) + t
        for flat_i in range(len(corners))
        for r, c in [_grid_index(flat_i)]
    ])

    # Corners were detected at CAPTURE_W/H (for detection accuracy) but
    # discover_scene() at runtime sees frames at RENDER_W/H — scale pixel
    # coordinates down so the fitted homography is valid for THAT pixel space,
    # matching exactly what render_image()'s own cv2.resize() does.
    scale = np.array([cfg.RENDER_W / cap_w, cfg.RENDER_H / cap_h])
    corners_scaled = corners * scale

    H, _mask = cv2.findHomography(corners_scaled, world_pts, method=cv2.RANSAC)
    if H is None:
        print("\nfindHomography failed.")
        return

    ones = np.ones((len(corners_scaled), 1))
    src_h = np.hstack([corners_scaled, ones])
    proj = (H @ src_h.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    errors = np.linalg.norm(proj_xy - world_pts, axis=1)

    print("\n" + "=" * 60)
    print(f"Homography fit from {len(corners)} chessboard corners:")
    print(f"  reprojection error — mean: {errors.mean() * 1000:.1f}mm  max: {errors.max() * 1000:.1f}mm")
    if errors.max() > 0.02:
        print("  (max error > 2cm — consider re-running with better lighting/focus, or a flatter board)")
    print("=" * 60)
    print("\nOVERHEAD_HOMOGRAPHY = [" if overhead else f"\nCAMERAS[{camera!r}]: homography=[")
    for row in H:
        # float(...) before repr — numpy 2.x's own repr of a float64 scalar is
        # "np.float64(x)", which isn't valid to paste into config.py (no numpy
        # import there). Plain Python floats repr as just the number.
        print(f"    [{float(row[0])!r}, {float(row[1])!r}, {float(row[2])!r}],")
    print("]")

    # The same photo also gives the camera itself — lens and pose in the arm
    # frame — which is what lets surface_scan and the board scan measure
    # objects (OVERHEAD_CAM_MODEL), and how far the arm's frame reads short of
    # real millimetres (ARM_FRAME_SCALE).
    model, report = overhead_camera_model(corners, world_pts, (cap_w, cap_h),
                                          cfg.CHESSBOARD_SQUARE_SIZE_M, square_m)
    print(f"\nCamera model: lens fit {report['lens_rms_px']:.2f} px RMS, pose fit "
          f"{report['pose_mean_px']:.2f} px mean / {report['pose_max_px']:.2f} px max, camera "
          f"{report['height_mm']:.0f} mm above the board (arm frame).")
    print("\nOVERHEAD_CAM_MODEL: \"dict | None\" = {" if overhead else f"\nCAMERAS[{camera!r}]: model={{")
    print(f"    \"size\": {model['size']},")
    print(f"    \"K\": {model['K']},")
    print(f"    \"D\": {model['D']},")
    print("    \"T_world_cam\": [")
    for row in model["T_world_cam"]:
        print(f"        {row},")
    print("    ],")
    print("}")
    print(f"ARM_FRAME_SCALE = {cfg.CHESSBOARD_SQUARE_SIZE_M / square_m:.4f}")
    if touched_z:
        # The touches are the grasp point's height (get_ee_pos); the surface is
        # where the closed jaws' tips rest, that far below it — TABLE_Z as the
        # policies read it (the tips' clearance above it).
        tips_below = float(getattr(ARM.config, "JAW_TIP_BELOW_GRASP_M", 0.0))
        print(f"\nThe touches were at z {', '.join(f'{z * 1000:+.1f}' for z in touched_z)} mm (the grasp point): "
              f"TABLE_Z = {float(np.mean(touched_z)) - tips_below:.4f} (now {cfg.TABLE_Z:.4f}) if the jaw tips "
              f"were resting on the board, {tips_below * 1000:.0f} mm below it.")


def overhead_camera_model(corners_px: np.ndarray, world_pts: np.ndarray, image_wh,
                          real_pitch_m: float, arm_pitch_m: float) -> "tuple[dict, dict]":
    """
    The overhead camera as OVERHEAD_CAM_MODEL: its lens from the board's
    corners at their real size (one slanted view of a flat grid pins down the
    focal length and barrel distortion), then its pose from the same corners
    placed where the arm frame has them (world_pts, the similarity fit). Using
    the fit's lattice rather than the homography's own output matters: the
    homography cannot model the lens's distortion and is ~0.7 mm off the
    lattice; the camera model reprojects it to ~0.3 px.
    """
    w, h = image_wh
    cols = cfg.CHESSBOARD_INTERNAL_COLS
    obj = np.array([[(i % cols) * real_pitch_m * 1000, (i // cols) * real_pitch_m * 1000, 0.0]
                    for i in range(len(corners_px))], np.float32)
    k0 = np.array([[0.7 * w, 0, w / 2], [0, 0.7 * w, h / 2], [0, 0, 1]], float)
    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_USE_INTRINSIC_GUESS)
    rms, k, d, _, _ = cv2.calibrateCamera([obj], [np.asarray(corners_px, np.float32)], (int(w), int(h)),
                                          k0, np.zeros(5), flags=flags)
    arm = np.c_[np.asarray(world_pts, float) * 1000, np.full(len(world_pts), cfg.TABLE_Z * 1000)]
    _, rvec, tvec = cv2.solvePnP(arm, np.asarray(corners_px, np.float64), k, d, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(arm, rvec, tvec, k, d)
    err = np.linalg.norm(proj.reshape(-1, 2) - corners_px, axis=1)
    R = cv2.Rodrigues(rvec)[0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.T, -R.T @ tvec.ravel() / 1000
    model = {"size": (int(w), int(h)),
             "K": [[round(float(v), 2) for v in row] for row in k],
             "D": [round(float(v), 5) for v in d.ravel()[:5]],
             "T_world_cam": [[round(float(v), 6) for v in row] for row in T]}
    return model, {"lens_rms_px": float(rms), "pose_mean_px": float(err.mean()),
                   "pose_max_px": float(err.max()),
                   "height_mm": float(T[2, 3] * 1000 - cfg.TABLE_Z * 1000)}


if __name__ == "__main__":
    main()
