"""
calibrate_gripper_servo.py — measure what gripper-camera visual servoing needs.

Produces the values SO101Env.visual_servo_xy depends on:

  GRIPPER_CAM_GRASP_PX       where a piece on the board appears, from servo height,
                             when the open jaws are centred over it (by eye)
  GRIPPER_CAM_JACOBIAN       2x2 metres-per-pixel, in the GRIPPER frame
  GRIPPER_CAM_REF_HEIGHT_M   the height that Jacobian was measured at
  GRIPPER_CAM_REF_TILT_DEG   the gripper's tilt when the grasp pixel was taken: the
                             pixel is right only at that tilt, and the servo
                             steps for the difference at any other

Why bother, given the arm already has IK
----------------------------------------
move_ee_to() closes its loop on forward kinematics: it minimises
|FK(measured q) - goal|, a model estimate against a model target. Nothing in
that loop observes the world, so a wrong TCP offset, a skewed URDF or a
mis-fitted homography are all invisible — it converges confidently to the
wrong place, which is exactly the failure that keeps closing the jaws beside
a piece instead of around it.

Servoing on the gripper camera measures the error between the piece and the
jaws IN THE SAME IMAGE, so all of those cancel. Only the RELATIVE correction
has to be right, and it is, even when the absolute frame is skewed.

How the Jacobian is measured
----------------------------
Not from a camera model — from the arm itself. With a piece in view, the arm
steps a known distance in world x, then in world y, and the piece's apparent
displacement is recorded. That gives image-displacement per world-displacement
directly, including whatever rotation and mirroring the camera's mounting
introduces, which is the part nobody can work out reliably on paper.

Central differences (-d then +d) rather than one-sided, because this arm's
backlash and gravity sag are a bigger error source than the detector's
centroid noise.

The result is stored in the GRIPPER frame, not the world: the camera is bolted
to the gripper, so expressed that way it is constant, and the runtime rotates
it into the world using the arm's measured orientation. That is what makes one
calibration valid at every square rather than only the one used here.

Usage:
    python tools/so101/calibrate_gripper_servo.py [--deployment chess] [--probe X,Y]

--probe moves the working point (default PROBE_XY) to a clear spot: step 1
lowers the open jaws to grasp height there, onto whatever stands in the way.
--grasp-px U,V skips step 1 with a grasp pixel already measured, when only the
Jacobian is left to do (step 2 needs the piece in view, not centred).
--height M servos from M metres above the surface instead of HOVER_HEIGHT. The
runtime servos from whatever height this measured at (GRIPPER_CAM_REF_HEIGHT_M).
"""

import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2            # noqa: E402
import numpy as np    # noqa: E402

from tools.so101._common import make_env, rc, show_image, task_config  # noqa: E402
from mrcf import privacy  # noqa: E402

cfg = task_config()

HERE = rc.RIG_DIR             # this rig's folder (rig_state.py)

# Where to do the Jacobian probes. "Comfortably inside the straight-down band"
# has to be verified by DRIVING there, not by asking whether IK has a solution.
#
# This was 0.27, which puts the x+ probe at 0.282 — right at the edge of the
# straight-down workspace at hover height (0.290 already needs a 5 deg tilt).
# approach_orientation approves 0.282 with an exact solution, and it is still
# wrong: the real arm stalled 2.5 cm short of it, twice. Pure kinematics has no
# sag, no friction and no torque limit, so it is optimistic near the edge and
# cannot be used to choose a working point. Same mistake cost the first
# check_targeting.py probe set its two far squares.
#
# Driving move_ee_to through the full probe sequence offline instead:
#   0.270 -> worst residual 7.9 mm   (tolerance is 8.0 — no margin at all)
#   0.250 -> 8.0 mm
#   0.240 -> 6.6 mm                  <- chosen
#   0.230 -> 6.3 mm
PROBE_XY = (0.24, 0.0)
# From a 60 mm hover, with the gripper truly vertical, the wrist camera sees too
# small a patch of board for piece_vision's grid finder (it needs 4 x 4 squares
# clear of the piece and the jaws): it found no grid in a frame in three, and at
# some squares in none. 70 mm found it in every frame.
SERVO_HEIGHT_M = cfg.HOVER_HEIGHT
if "--height" in sys.argv:
    SERVO_HEIGHT_M = float(sys.argv[sys.argv.index("--height") + 1])
if "--probe" in sys.argv:
    PROBE_XY = tuple(float(v) for v in sys.argv[sys.argv.index("--probe") + 1].split(","))

# Probe distance per axis, each way. Big enough that the image displacement is
# many pixels, small enough that the piece stays in frame and the arm stays in
# the same IK branch.
PROBE_M = 0.012

# Searching for the held piece during the grasp-pixel step: generous, because
# there is no prior yet — this is the one place the detector runs blind.
GRASP_SEARCH_RADIUS_PX = 240


def _detector():
    """The deployment's piece detector (task-specific; this tool is not)."""
    try:
        return importlib.import_module(f"{cfg.__package__}.piece_vision")   # deployments.<name>.piece_vision
    except ImportError as exc:
        raise SystemExit(
            f"Could not import piece_vision from the deployment ({exc}).\n"
            "Visual servoing needs a task-specific detector; see "
            "deployments/chess/piece_vision.py for the shape of one.") from exc


def _capture(env) -> np.ndarray:
    return env.render_image(rc.CAMERA_NAME, width=rc.RENDER_W, height=rc.RENDER_H)


def _detect_at(env, pv, expected, radius, save_as: str, rank: str = "nearest"):
    """
    Find the piece, with a human confirming or correcting every measurement.

    Why this asks every time rather than trusting find_piece: tried against a
    real frame from this rig, under glare that washed the piece out to near
    white, several different automatic strategies (widened colour range,
    local Otsu, local brightness variance) each LOOKED like they worked —
    landed a centroid within ~10 px of the real piece — but turned out to be
    merging the piece with grid-line edges or neighbouring light squares and
    landing close by coincidence, not from a clean isolation of the piece.
    That is worse than an honest failure: it would have fed silently wrong
    numbers into the Jacobian. A human glancing at the saved frame catches
    that in a second; no classical-CV rule tried here did reliably.

    Blank input accepts whatever was automatically found (or aborts this
    measurement if nothing was). Typing "x,y" overrides it.
    """
    frame = _capture(env)
    found = pv.find_piece(frame, expected, search_radius_px=radius, rank=rank)
    path = HERE / save_as
    vis = pv.annotate(frame, expected, found, radius)
    privacy.imwrite(str(path), vis)
    # Also keep a timestamped copy. These overwrite themselves every run, and a
    # frame that silently changed underneath an analysis once cost an afternoon
    # of chasing a regression that was not there. The archive is also the only
    # way this rig accumulates a real test set for the detector.
    archive = HERE / "detector_frames"
    archive.mkdir(exist_ok=True)
    privacy.imwrite(str(archive / f"{datetime.now():%Y%m%d_%H%M%S}_{save_as}"), vis)
    show_image(path)

    # If nothing was in range, say what WAS seen. Out-of-range candidates mean
    # "move the piece", not "the detector is blind", and the two look identical
    # from a bare failure message.
    if found is None:
        seen = pv.candidates(frame)
        if seen:
            print(f"  Nothing within {radius:.0f} px of "
                  f"({expected[0]:.0f}, {expected[1]:.0f}), but {len(seen)} candidate(s) "
                  f"elsewhere in frame:")
            for cx, cy, cr, area in seen[:4]:
                d = float(np.hypot(cx - expected[0], cy - expected[1]))
                print(f"    ({cx:.0f}, {cy:.0f})  r={cr:.0f}  area={area:.0f}  "
                      f"{d:.0f} px away{'  <- move the piece this way' if d < radius * 2.5 else ''}")

    if found is not None:
        prompt = (f"  Found the piece at ({found[0]:.0f}, {found[1]:.0f}) — "
                  f"the frame is open. Press Enter if that's right, or type "
                  f"the correct pixel as 'x,y': ")
    else:
        prompt = ("  Nothing automatically detected — the frame is open. "
                  "Type the piece's pixel as 'x,y', or leave blank to skip: ")

    reply = input(prompt).strip()
    if not reply:
        return found
    try:
        x_str, y_str = reply.split(",")
        return (float(x_str), float(y_str))
    except ValueError:
        print("  Could not parse that as 'x,y' — skipping this measurement.")
        return None


def measure_grasp_pixel(env, pv):
    """
    Where a piece standing on the board appears from servo (hover) height when
    the open jaws are centred over it — the exact situation the servo is in.

    You centre the piece by eye between the OPEN jaws at grasp height; the arm
    rises straight up with the jaws still open, and the piece is located.
    Nothing here depends on the arm's model of the jaws.

    Two cleverer versions were badly wrong (see GRIPPER_CAM_GRASP_PX in
    robot_config.py): holding the piece up in the closed jaws, and closing the
    jaws to centre it then reopening. The second leaned on get_gripper_width,
    which reads ~0.7 mm for a held 20 mm piece, and was off by ~35 mm.
    """
    print("\n── Step 1 of 2: the grasp pixel " + "─" * 44)
    print("This records where a piece on the board appears, from the height the")
    print("servo works at, when the open jaws are centred over it.\n")

    print("  Before you start: ONE piece on the board, nothing else in view, and")
    print("  no direct sun on it.\n")
    x, y = PROBE_XY
    env.set_gripper(cfg.PREGRASP_OPEN_M, n_steps=15)
    hover = np.array([x, y, cfg.TABLE_Z + SERVO_HEIGHT_M])
    low = np.array([x, y, cfg.TABLE_Z + cfg.GRASP_HEIGHT_M])
    # One wrist orientation down, and back up: the servo uses the pixel at the
    # hover's tilt (GRIPPER_CAM_REF_TILT_DEG), and a rise that re-chose its own
    # tilt part-way swung the camera between where the piece was centred and
    # where its pixel was read.
    chosen = env.approach_orientation_for([hover, low], tols=[0.01, 0.008])
    rot = None if chosen is None else chosen[0]
    env.move_ee_to(hover, orientation=rot)
    env.move_ee_to(low, tol=0.008, max_step_deg=cfg.GRASP_STEP_DEG, orientation=rot)
    input("  Stand a piece on the board EXACTLY centred between the open jaw pads\n"
          "  (by eye — take your time, this placement is the reference), take your\n"
          "  hand away, then press Enter...")

    # The view with the jaws round a centred piece: what a grasp should look
    # like from the wrist camera, for checking a grasp before the close.
    time.sleep(0.5)
    view = _capture(env)
    archive = HERE / "detector_frames"
    archive.mkdir(exist_ok=True)
    privacy.imwrite(str(archive / f"{datetime.now():%Y%m%d_%H%M%S}_calibrate_gripper_grasp_view.png"),
                cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
    near = pv.find_piece(view, (rc.RENDER_W / 2.0, rc.RENDER_H / 2.0), 400, rank="strongest")
    print(f"  At grasp height the centred piece shows at "
          f"{'nothing found' if near is None else f'({near[0]:.0f}, {near[1]:.0f})'} "
          f"(jaws {env.get_gripper_width() * 1000:.0f} mm, gripper tilted {env.gripper_tilt_deg():+.1f} deg).")

    here = env.get_ee_pos()
    # Straight up, same jaw opening, so the gap stays over the piece.
    env.move_ee_to(np.array([here[0], here[1], cfg.TABLE_Z + SERVO_HEIGHT_M]),
                   tol=0.006, max_step_deg=3.0, orientation=rot)
    time.sleep(1.0)

    # rank="strongest": no prior yet for where the piece appears from here.
    frame_centre = (rc.RENDER_W / 2.0, rc.RENDER_H / 2.0)
    found = _detect_at(env, pv, frame_centre, GRASP_SEARCH_RADIUS_PX,
                       "calibrate_gripper_grasp_px.png", rank="strongest")
    if found is None:
        print("\n  No grasp pixel recorded — nothing to calibrate against.")
        return None
    print("  Leave the piece where it is: step 2 uses it.")
    return found


def measure_jacobian(env, pv, grasp_px):
    """Image displacement per unit of world motion, expressed in the gripper frame."""
    print("\n── Step 2 of 2: the image/motion Jacobian " + "─" * 30)
    print("The arm will step a known distance each way along x and y and watch")
    print("how far the piece appears to move. Keep clear of the arm.\n")

    z = cfg.TABLE_Z + SERVO_HEIGHT_M
    home_xy = np.array([PROBE_XY[0], PROBE_XY[1], z])
    # One wrist orientation for the centre and every probe. Left to choose its
    # own per probe, the wrist tilted differently at each (5.0-6.4 deg offline,
    # more on the arm), and the "image motion per mm" measured the camera
    # turning as well as moving: two runs on 2026-10-17 disagreed with each
    # other by 70 px on the same probe.
    probes = [home_xy + np.array([dx, dy, 0.0]) for dx, dy in
              ((-PROBE_M, 0), (PROBE_M, 0), (0, -PROBE_M), (0, PROBE_M))]
    chosen = env.approach_orientation_for([home_xy] + probes, tols=[0.008] * 5)
    if chosen is None:
        print("  No single wrist orientation reaches the centre and all four probes — "
              "move --probe closer to the base.")
        return None, None
    rot, tilt = chosen
    print(f"  Holding one orientation for every probe: tilted {tilt:+.0f} deg.")
    if not env.move_ee_to(home_xy, orientation=rot):
        print(f"  Could not reach the probe pose {home_xy.round(3).tolist()}.")
        return None, None

    # Dry-run every probe extreme BEFORE asking for a piece. A reach problem
    # that only shows up at the third of four probes wastes the whole run and
    # the stage-1 measurement with it, which is exactly how this failed twice.
    print("  Checking the probe poses are reachable...")
    for axis, label in ((0, "x"), (1, "y")):
        for sign in (-1.0, +1.0):
            t = home_xy.copy()
            t[axis] += sign * PROBE_M
            # From the centre, as the probes themselves go (see below): straight
            # from x- to x+ is twice the move from a different seed, and in the
            # radial orientation that sent the arm 15 mm down and stalled it
            # 25 mm short of a pose the real probe sequence reaches.
            env.move_ee_to(home_xy, tol=0.008, max_step_deg=3.0, orientation=rot)
            ok = env.move_ee_to(t, tol=0.008, max_step_deg=3.0, orientation=rot)
            here = env.get_ee_pos()
            print(f"    preflight {label}{sign:+.0f} -> [{here[0]:.4f} {here[1]:+.4f} {here[2]:.4f}]"
                  f"  short {np.linalg.norm(here - t) * 1000:5.1f} mm  "
                  f"{'ok' if ok else 'FAILED'}")
            if not ok:
                print(f"  Cannot reach the {label}{sign:+.0f} probe pose "
                      f"{t.round(3).tolist()} — stopped {np.linalg.norm(here - t) * 1000:.0f} mm short.")
                print(f"  Move PROBE_XY closer to the base than {PROBE_XY} and re-run;")
                print("  this is a reach limit, not a detection problem.")
                return None, None
    env.move_ee_to(home_xy, orientation=rot)
    print("  All four reachable.\n")

    print("  The piece from step 1 should still be standing under the jaws. It does")
    print("  NOT have to be exactly under them: this step measures how the image")
    print("  moves when the ARM moves, so all that matters is that the same piece")
    print("  stays in view across four small steps.")
    input("  Press Enter to start...")

    # The prior is grasp_px only for this FIRST look, and only because nothing
    # better exists yet. It is a poor prior: grasp_px was measured with a piece
    # at the JAWS' height, while this piece is on the board ~60 mm lower, and
    # the camera is neither at the grasp point nor looking straight down — so
    # the same (x, y) projects to a different pixel at a different height. A
    # piece genuinely under the jaws can therefore sit well outside this circle.
    # Hence the generous radius here, and the switch to a real prior below.
    found = _detect_at(env, pv, grasp_px, GRASP_SEARCH_RADIUS_PX,
                       "calibrate_gripper_probe0.png")
    if found is None:
        print("  Cannot see the piece from the probe pose — reposition it under the jaws")
        print("  so it falls inside the search circle drawn on the saved frame.")
        print(f"  See {HERE / 'calibrate_gripper_probe0.png'}.")
        return None, None

    last_seen = tuple(found)
    z_at: list[float] = [float(env.get_ee_pos()[2])]
    px_cols, world_cols = [], []
    for axis, label in ((0, "x"), (1, "y")):
        px_at, pos_at = {}, {}
        for sign in (-1.0, +1.0):
            target = home_xy.copy()
            target[axis] += sign * PROBE_M
            # Deliberately loose. The probe does NOT need the arm to land on the
            # commanded point — it needs to know where the arm actually went,
            # which get_ee_pos reports. Demanding 4 mm here bought nothing and
            # cost a whole run: move_ee_to reported a stall at a pose whose IK
            # solution is exact.
            # Always start a probe from the SAME place. placo is a local solver
            # seeded from the arm's measured pose, so where a move starts
            # selects which branch it converges to. Going x- -> x+ directly
            # worked on one run and on the next sent the arm 16 mm DOWNWARD
            # while x barely moved, from a start just 3 mm different. Returning
            # to the centre first makes every probe the same 12 mm move from the
            # same seed, instead of a 24 mm move from wherever the last one
            # happened to stop.
            env.move_ee_to(home_xy, tol=0.008, max_step_deg=3.0, orientation=rot)
            before = env.get_ee_pos()
            gap_before = env.get_gripper_width()
            ok = env.move_ee_to(target, tol=0.008, max_step_deg=3.0, orientation=rot)
            after = env.get_ee_pos()
            # Printed every time, not just on failure: the pre-flight drives
            # these exact poses successfully moments earlier, so when a probe
            # then fails the interesting question is what differed — where the
            # arm actually started, how far it got, and whether the jaws moved
            # (the grasp point shifts with the opening, so a gripper that has
            # drifted since the pre-flight moves the target itself).
            print(f"    {label}{sign:+.0f}: from [{before[0]:.4f} {before[1]:+.4f} {before[2]:.4f}]"
                  f" -> [{after[0]:.4f} {after[1]:+.4f} {after[2]:.4f}]"
                  f"  want [{target[0]:.4f} {target[1]:+.4f} {target[2]:.4f}]"
                  f"  moved {np.linalg.norm(after - before) * 1000:5.1f} mm"
                  f"  short {np.linalg.norm(after - target) * 1000:5.1f} mm"
                  f"  dz {(after[2] - before[2]) * 1000:+5.1f} mm"
                  f"  jaws {gap_before * 1000:.1f}mm")
            if not ok:
                print(f"  Could not reach the {label}{sign:+.0f} probe pose.")
                return None, None
            time.sleep(0.3)
            # Track from where the piece was LAST seen, not from grasp_px. The
            # probe only needs to follow one piece across four small moves, so
            # the previous sighting is the right prior — it is a few tens of
            # pixels away, where grasp_px can be hundreds. Using grasp_px also
            # imported its height error (see above) into every probe.
            seen = _detect_at(env, pv, last_seen, pv.DEFAULT_SEARCH_RADIUS_PX,
                              f"calibrate_gripper_probe_{label}{'m' if sign < 0 else 'p'}.png")
            if seen is None:
                print(f"  Lost the piece at the {label}{sign:+.0f} probe pose.")
                return None, None
            px_at[sign] = np.asarray(seen, dtype=float)
            pos_at[sign] = env.get_ee_pos()[:2].copy()
            z_at.append(float(after[2]))
            last_seen = tuple(px_at[sign])

        d_px = px_at[+1.0] - px_at[-1.0]
        d_world = pos_at[+1.0] - pos_at[-1.0]
        print(f"  world {label}: arm moved ({d_world[0] * 1000:+5.1f}, "
              f"{d_world[1] * 1000:+5.1f}) mm  ->  image moved "
              f"({d_px[0]:+6.1f}, {d_px[1]:+6.1f}) px")
        if float(np.linalg.norm(d_world)) < 0.004:
            print(f"  The arm barely moved ({np.linalg.norm(d_world) * 1000:.1f} mm) — "
                  f"nothing to measure against.")
            return None, None
        if float(np.linalg.norm(d_px)) < 5.0:
            print("  That is too small to invert reliably. Either the camera "
                  "barely sees this motion, or the detector is locking onto "
                  "something static (check the saved probe frames).")
            return None, None
        px_cols.append(d_px)
        world_cols.append(d_world)

    env.move_ee_to(home_xy, orientation=rot)

    # Height actually held during the probes, not the height that was asked for.
    # move_ee_to converges to a 3D tolerance, so some of that budget is spent in
    # z — observed drifting 5-16 mm below the commanded plane. The Jacobian is
    # metres-per-pixel at a particular distance from the surface and
    # visual_servo_xy rescales it by height at runtime, so recording the
    # commanded height when the arm was somewhere else bakes in a scale error of
    # exactly that ratio.
    z_mean = float(np.mean(z_at))
    z_spread = float(np.max(z_at) - np.min(z_at))
    print(f"\n  probe height: commanded {z * 1000:.0f} mm, actually held "
          f"{z_mean * 1000:.0f} mm (spread {z_spread * 1000:.1f} mm)")
    if z_spread > 0.012:
        print(f"  WARNING: {z_spread * 1000:.0f} mm of height variation across the probes.")
        print("  Apparent size changes with distance, so part of the image motion")
        print("  measured here was the piece getting nearer, not sideways travel.")
        print(f"  The Jacobian will be scaled wrong by roughly "
              f"{z_spread / max(z_mean, 1e-6) * 100:.0f}%.")

    # px_delta = M_world @ world_delta, solved from the MEASURED displacements
    # rather than the commanded ones. Using the commanded +/-PROBE_M assumed the
    # arm landed exactly on both probe points; this makes any shortfall
    # irrelevant, and it also picks up cross-coupling (an x probe that drifts in
    # y) that the old diagonal assumption silently folded into the scale.
    d_world_mat = np.column_stack(world_cols)
    if abs(float(np.linalg.det(d_world_mat))) < 1e-9:
        print("  The two probe moves were not independent in the world — "
              "cannot separate the axes.")
        return None, None
    m_world = np.column_stack(px_cols) @ np.linalg.inv(d_world_mat)
    if abs(float(np.linalg.det(m_world))) < 1e-6:
        print("  The two probe directions are not independent in the image — "
              "the camera may be looking along one of them. Cannot invert.")
        return None, None

    # Invert to metres-of-world per pixel, then rotate into the GRIPPER frame
    # so one calibration holds at every board square rather than only here.
    j_world = np.linalg.inv(m_world)
    q6 = np.append(env._q_arm_deg(), 0.0)
    rot = env._kinematics.forward_kinematics(q6)[:3, :3]
    j_gripper = rot[:2, :2].T @ j_world
    return j_gripper, z_mean - cfg.TABLE_Z


def main() -> None:
    pv = _detector()
    env = make_env()
    print(f"Connecting to {rc.FOLLOWER_ID} on {rc.FOLLOWER_PORT}...")
    env.connect()
    print("Connected.")
    try:
        env.go_home()
        # The tilt the grasp pixel is taken at: it is right only at that tilt, and
        # the servo steps for any other (SO101Env._servo_tilt_offset).
        tilt = None
        if "--grasp-px" in sys.argv:
            grasp_px = tuple(float(v) for v in sys.argv[sys.argv.index("--grasp-px") + 1].split(","))
            print(f"Grasp pixel given: {grasp_px} — skipping step 1.")
        else:
            grasp_px = measure_grasp_pixel(env, pv)
            tilt = env.gripper_tilt_deg()     # still where it was taken
        if grasp_px is None:
            print("\nStopped: no grasp pixel, so there is nothing to aim at.")
            return
        if tilt is not None:
            print(f"  Taken with the gripper tilted {tilt:+.1f} deg.")

        jac, ref_h = measure_jacobian(env, pv, grasp_px)
        if jac is None:
            print("\nStopped: the Jacobian could not be measured.")
            print("The grasp pixel above is still valid — re-run to finish.")
            if tilt is not None:
                print(f"It was taken at GRIPPER_CAM_REF_TILT_DEG = {tilt:.1f}")
            return
        if tilt is None:
            # Back at the probes' centre, over the same spot at the same height:
            # close to, but not exactly, how the given pixel was taken.
            tilt = env.gripper_tilt_deg()
            print(f"\n  The given grasp pixel's own tilt is unknown; at the probes' centre the "
                  f"gripper is at {tilt:+.1f} deg — used below.")

        print("\n" + "=" * 70)
        print("Paste into mrcf/backends/so101/robot_config.py:\n")
        print(f"GRIPPER_CAM_GRASP_PX = ({grasp_px[0]:.1f}, {grasp_px[1]:.1f})")
        print("GRIPPER_CAM_JACOBIAN = [")
        for row in jac:
            print(f"    [{row[0]:.8e}, {row[1]:.8e}],")
        print("]")
        print(f"GRIPPER_CAM_REF_HEIGHT_M = {ref_h:.4f}")
        print(f"GRIPPER_CAM_REF_TILT_DEG = {tilt:.1f}")
        print("GRIPPER_CAM_REF_POINT_HEIGHT_M = SERVO_POINT_HEIGHT_FRACTION * <this piece's height, m>")
        print("=" * 70)
        mm_per_px = float(np.linalg.norm(jac, axis=0).mean()) * 1000
        print(f"\nScale check: ~{mm_per_px:.2f} mm per pixel at "
              f"{ref_h * 1000:.0f} mm above the surface.")
        print("If that is wildly off what the view suggests, the detector probably")
        print("tracked something other than the piece — check the probe frames.")
    finally:
        env.go_home()
        env.stop()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
