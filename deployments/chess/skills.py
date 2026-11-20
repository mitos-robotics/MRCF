"""
skills.py — the chess deployment's actions.

Picking and placing are not here. They are the framework's standard actions
(manipulation.py), driving the arm's own policy library (its make_policies():
on the SO-101, its two-finger pick and place); install_manipulation() below plugs chess
into them — how it looks at a piece from the hover (piece_vision), how tall an
unmeasured piece could be, how it describes and logs a piece. What is left here
is chess: scanning the board, naming and identifying pieces, and the utility
actions.

Skills are registered as a side-effect of `from . import skills` in llm_controller.py.
Each returns True or False for whether it worked, which is how the controller
knows to recover and what it tells the planner happened.

Object and target positions come from scene_state, which scan_scene fills
from the board scan (chessboard.discover), or from the framework's surface
scan when there is no board — never from env.get_body_pos(), which always
returns None on real hardware.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

import ast
import dataclasses
import itertools
import math
import operator
from pathlib import Path as _Path

import numpy as np

from mrcf import manipulation
from mrcf.policies.base import Vision
from mrcf.robot_env import RobotEnv
from mrcf.vlm import ACTIONS, call_vlm, discover_scene, register_action, render_to_base64

from . import board_log
from .config import PROFILE as _PROFILE
from .config import (ARM, LEARNED_SKILLS, OBSERVE_CAMERA, PIECE_COLOURS, PIECE_SIZES_M, PIECE_TYPES,
                     SHORTEST_PIECE_M, TALLEST_PIECE_M, UNMEASURED_SHAPE_SHARE, approach_tilt, piece_type)


class BoardTarget:
    """
    The chess board as the overhead camera's calibration target
    (overhead_calibration.CalibrationTarget): its pose from the lattice and
    tags (board_tags.measure_board), origin at its centre, x toward the
    h-file, y toward rank 8.
    """
    name = "the board"

    def measure(self, frame_rgb, K, D) -> "dict | None":
        from . import board_tags
        board = board_tags.measure_board(frame_rgb, K, D)
        if board is None:
            return None
        return {"T_cam_target": board["T_cam_board"], "points_px": board["lattice_all_px"],
                "points_target": board_tags.lattice_board_points()}

    def camera_moved(self) -> None:
        from . import config as cfg
        # The corner cache is in the old pixels.
        cache = _Path(__file__).parent / "board_corners.npy"
        if cache.exists():
            cache.unlink()
        # The empty-board photo is in the old pixels too: set aside, and the
        # next scan lines it up again from the board's corners
        # (chessboard.realign_stale_reference).
        ref = _Path(cfg.SURFACE_REFERENCE_PATH)
        if ref.exists():
            ref.replace(ref.with_name(ref.stem + "_stale" + ref.suffix))
            print("  [Calibration] The empty-board photo no longer lines up — the next scan lines it up again.")


def check_overhead_calibration(env: RobotEnv) -> None:
    """
    At every scan, the arm parked out of view: is the overhead camera still
    where its calibration says, and if it has moved, re-pose it — and, if the
    board or the arm's base moved too, re-measure with the arm's own tour (the
    SO-101's tag tour), by itself (overhead_calibration.check_camera, against
    the board).
    """
    global _calibration_suspect
    from mrcf import overhead_calibration as oc
    from . import config as cfg
    report = oc.check_camera(
        env, BoardTarget(), cfg.OBSERVE_CAMERA, (cfg.CAPTURE_W_OVERHEAD, cfg.CAPTURE_H_OVERHEAD),
        (cfg.RENDER_W, cfg.RENDER_H), hover_z=cfg.TABLE_Z + 0.10, suspect=_calibration_suspect,
        # the SO-101's rewrites the corner cache and square map too
        tour=getattr(ARM, "tour_and_calibrate", None),
        namespaces=(vars(ARM.config),))                # config serves the arm's values live
    if report["hovered"]:
        _calibration_suspect = False
        goto_observe_pose(env)


def scan_scene(env: RobotEnv, scene_state, classify: bool = True) -> bool:
    """
    The one perception entry point for this deployment. classify=False: the
    board's pieces where they stand, with no vision model asked what each is
    (chessboard.discover) — the object memory keeps what they were called.

    Prefers the chess-specific path: square positions come from board geometry
    plus the calibrated homography (exact), and the VLM only classifies which
    piece stands on which labelled square. That removes the generic pipeline's
    two weakest steps — having a small model invent object names, and having it
    regress pixel coordinates — both of which failed on this rig.

    Without a board, the framework's surface scan finds and measures whatever
    stands on the table; without that (no camera model, no segmenter), the
    vision model's generic discovery. Returns False only if none could run.
    """
    from . import chessboard
    from mrcf import scene_scan
    from mrcf.perception import surface_scan
    from .config import (ARM_KEYWORDS, CAPTURE_W_OVERHEAD, CAPTURE_H_OVERHEAD,
                         SURFACE_REFERENCE_PATH, TABLE_Z, WORKSPACE_BOUNDS)

    check_overhead_calibration(env)
    overlay = _Path(__file__).parent / "board_overlay.png"
    if chessboard.discover(env, scene_state, save_to=overlay, classify=classify):
        return True
    b = WORKSPACE_BOUNDS
    reference = surface_scan.load_reference(SURFACE_REFERENCE_PATH, (CAPTURE_W_OVERHEAD, CAPTURE_H_OVERHEAD))
    if reference is None:
        # The surface here is a board of squares: without a photo of it empty,
        # a surface scan takes the squares for objects, each remembered, and
        # every pick is refused for want of room between them. Better no scan
        # than that one.
        print("  [Scan] Board not usable, and no empty-board photo to tell pieces from its squares — "
              "not scanning. Take one with tools/capture_surface_reference.py (board clear).")
        return False
    print("  [Scan] Board not usable — scanning the surface for objects instead.")
    if scene_scan.observe(env, scene_state, board_log.memory(), OBSERVE_CAMERA,
                            CAPTURE_W_OVERHEAD, CAPTURE_H_OVERHEAD,
                            (b.x_min, b.x_max), (b.y_min, b.y_max), reference=reference,
                            describe=chessboard.kind_text, table_z=TABLE_Z,
                            reachable=lambda xy: b.x_min <= xy[0] <= b.x_max and b.y_min <= xy[1] <= b.y_max):
        return True
    print("  [Scan] Surface scan unavailable — falling back to generic VLM discovery.")
    discover_scene(env, scene_state, exclude_keywords=ARM_KEYWORDS)
    return True


def goto_observe_pose(env: RobotEnv) -> bool:
    """
    Put the arm wherever the PRIMARY perception camera needs it before a scan.

    Overhead (fixed) camera: it's stationary, so its pixel_to_world homography
    is valid from any arm pose — there is no pose to "reach". What matters
    instead is the opposite: the arm must be OUT of its view, or it occludes
    the very objects discover_scene is trying to find. So park it clear.

    Gripper-mounted camera: its pixel_to_world is only valid at one fixed,
    directly-measured pose (on the SO-101 OBSERVE_DEGREES, env.go_observe) —
    see mrcf/backends/so101/env.py's pixel_to_world and go_observe docstrings for
    why this has to be a measured pose rather than a Cartesian IK target.

    Returns True if the pose was reached, False on timeout (still proceeds
    with whatever the camera sees — the caller prints what it found either
    way, which is itself useful signal if this keeps returning False).

    Never with a piece in the jaws: the arm would swing it off to the park
    pose and back (2026-11-03, the f1 bishop on its way to c4). It stays where
    it is, and the scan sees what it can past the arm.
    """
    from .config import OBSERVE_CAMERA, OVERHEAD_CAMERA_NAME

    try:
        holding = POLICIES.holding(env)
    except AttributeError:                           # an env with no gripper to read
        holding = False
    if holding:
        print("  [Observe] A piece is in the jaws — not carrying it off to the park pose to look.")
        return False
    if OVERHEAD_CAMERA_NAME is not None and OBSERVE_CAMERA == OVERHEAD_CAMERA_NAME:
        reached = env.go_park()                      # the arm's pose out of the camera's view
        if not reached:
            print("  [Observe] Did not reach the park pose within tolerance — "
                  "the arm may be occluding the overhead view.")
        return reached

    reached = env.go_observe()
    if not reached:
        print("  [Observe] Did not reach the observe pose within tolerance — "
              "scan may be from an off-target view.")
    return reached


def look_around(env: RobotEnv, scene_state, name: str) -> bool:
    """
    If `name` isn't visible from the primary observe pose, sweep through
    config.py's LOOK_AROUND_POSES (if any are configured) re-running
    discover_scene at each until it's found or every pose has been tried.

    Always leaves the arm back at the primary observe pose/calibration
    before returning, whether or not it found anything, so callers don't
    need to worry about restoring state themselves. An arm without
    LOOK_AROUND_POSES (or go_to_look_around_pose) has nowhere else to look.
    """
    from . import config as cfg

    LOOK_AROUND_POSES = getattr(cfg, "LOOK_AROUND_POSES", None)
    if not LOOK_AROUND_POSES or not hasattr(env, "go_to_look_around_pose"):
        return False

    print(f"  [LookAround] '{name}' not visible from the primary view — "
          f"checking {len(LOOK_AROUND_POSES)} alternate view(s)...")
    found = False
    for i, pose in enumerate(LOOK_AROUND_POSES, 1):
        print(f"  [LookAround] Alternate view {i}/{len(LOOK_AROUND_POSES)}...")
        if not env.go_to_look_around_pose(pose):
            print(f"  [LookAround] Could not reach alternate view {i} — skipping")
            continue
        scan_scene(env, scene_state)
        if scene_state.find(name) is not None:
            print(f"  [LookAround] Found '{name}'.")
            found = True
            break

    goto_observe_pose(env)  # restore the primary pose/calibration either way
    if not found:
        print(f"  [LookAround] '{name}' still not found after checking all alternate views.")
    return found


# ── Manipulation: the arm's policies, with chess's hooks ───────────────────────

# The arm's policy library. The controller uses it directly too, to go home
# and to ask whether the gripper is holding something.
POLICIES = ARM.make_policies(profile=_PROFILE)


def _measure_piece(frame_rgb, expected_px=None, model=None, expected_xy_m=None, **among):
    """
    Vision.measure: piece_vision's, which also knows the jaw and the board's
    pattern — and, with among (known=, target=, need_height=), the pieces
    around the one sent for, so a neighbour's outline is not taken for it.
    """
    from . import piece_vision
    return piece_vision.measure_piece(frame_rgb, expected_px, model=model, expected_xy_m=expected_xy_m, **among)


def _find_piece(frame_rgb, expected_px, radius_px):
    """Vision.find, for visual servoing before the gripper camera is located."""
    from . import piece_vision
    found = piece_vision.find_piece(frame_rgb, expected_px, radius_px)
    if found is None:
        # Kept raw, with where it looked in the name: a servo that loses the
        # piece says only "not found", and the frame is the only way to tell a
        # detector miss from a piece that really was out of view.
        board_log.save_image("servo_misses", f"at{expected_px[0]:.0f}_{expected_px[1]:.0f}"
                             f"_r{radius_px:.0f}_{next(_SERVO_MISSES)}", frame_rgb)
    return found


_SERVO_MISSES = itertools.count()     # tells apart misses saved in the same second

# Vision.relocate: a piece found again after a miss is the scan's piece nearest
# where it was, no further than this. Misses pushed a pawn 2-27 mm
# (2026-10-30); the next square's piece stands 40 mm off.
RELOCATE_RADIUS_M = 0.030


def _relocate_piece(env, near_xy):
    """
    Vision.relocate: the piece nearest near_xy as a fresh board scan finds it
    (the arm out of the camera's view first), the arm then home; None when no
    piece is within RELOCATE_RADIUS_M. Home, not where the scan parks it: a
    pick plans its next approach from where the arm is, and from the park
    pose leans that reach from home came out out of reach.
    """
    from mrcf.scene_state import SceneState
    near_xy = np.asarray(near_xy, float)[:2]
    state = SceneState()
    goto_observe_pose(env)
    # Where, not what: the vision model's two looks at every piece took 4.5 min
    # of a full board's re-find (2026-11-03), and the memory keeps the names.
    scan_scene(env, state, classify=False)
    env.go_home()
    feet = [np.asarray(o.position[:2], float) for o in state.objects if o.position is not None]
    near = min(feet, key=lambda p: float(np.linalg.norm(p - near_xy)), default=None)
    if near is None or float(np.linalg.norm(near - near_xy)) > RELOCATE_RADIUS_M:
        return None
    return near


# Set when a pick found the overhead camera and the arm far apart before
# correcting (Outcome.attempts' "overhead_err_mm"): the next scan checks the
# calibration with the gripper tag, and re-measures it if they still disagree.
# A calibrated camera's hovers need 1-3 mm; a moved one's, 22-26 mm.
_OVERHEAD_SUSPECT_MM = 8.0
_calibration_suspect = False


def _log_pick(obj, start_xy, outcome) -> None:
    """Scenario.picked: what was measured and what happened, in board_log's logs."""
    global _calibration_suspect
    errs = [a["overhead_err_mm"] for a in (outcome.attempts or []) if a.get("overhead_err_mm") is not None]
    if errs and float(np.median(errs)) > _OVERHEAD_SUSPECT_MM:
        _calibration_suspect = True
        print(f"  [Calibration] The overhead camera and the arm were {np.median(errs):.0f} mm apart at this "
              "pick's hovers — checking the camera's calibration at the next scan.")
    memory = board_log.memory()
    track = memory.get(obj.track_id)
    ptype = track.label if track is not None else piece_type(obj.name)
    colour = track.attributes.get("colour") if track is not None else None
    square = getattr(obj, "square", None)
    # The wrist camera at each grasp, before the close: what was between the jaws.
    # And where a pick looked from beside the piece, what it saw from there.
    views = [board_log.save_image("grasp_frames", label, frame) for label, frame in outcome.views]
    if outcome.refused:
        board_log.record_pick("pick", obj.name, square, False, refused=outcome.message, notes=outcome.notes,
                              at_m=[float(v) for v in start_xy], grasp_frames=views)
        return
    measured = None
    m = outcome.measured
    if m is not None:
        # Kept, so any measurement — a wrong one above all — can be looked at.
        frames = [board_log.save_image("hover_frames", f"look{k}", f) for k, f in enumerate(m.frames)]
        measured = {**m.as_dict(), "frames": frames}
        geom = obj.geometry
        board_log.record_shape(ptype, colour, square, m.height_mm, m.widths_mm, m.foot_mm,
                               overhead_foot_mm=geom["base_diameter_m"] * 1000 if geom else None,
                               slant_deg=m.slant_deg, frames=frames, during="pick", id=obj.track_id)
    board_log.record_pick("pick", obj.name, square, outcome.ok, at_m=[float(v) for v in start_xy],
                          profile=dataclasses.asdict(outcome.grasp) if outcome.grasp is not None else None,
                          measured=measured, geometry=obj.geometry, id=obj.track_id,
                          # Why each attempt was retried, and the timing: from
                          # pass/fail alone a failure cannot be traced.
                          message=outcome.message, notes=outcome.notes, grasp_frames=views,
                          # How it came at the piece, and each attempt's result as a
                          # code, so a map of what works where needs no text-matching.
                          approach=outcome.approach, attempts=outcome.attempts)


def _vacated(scene_state, obj, start_xy) -> None:
    """
    Scenario.vacated: the square a pick emptied is a place target again, where
    the last scan put it. Without it, putting the piece back on that square
    makes the controller rescan first — to park and back, the piece in the
    jaws — and the swing can shake it out.
    """
    from . import chessboard
    from . import config as cfg
    from mrcf.scene_state import SceneObject
    square = getattr(obj, "square", None)
    if square is None or any(t.name == square for t in scene_state.targets):
        return
    entry = chessboard.LAST_SQUARES.get(square)
    x, y = entry["world"] if entry is not None else start_xy[:2]
    scene_state.targets.append(SceneObject(square, f"empty square {square}", interactable=False,
                                           position=np.array([x, y, cfg.TABLE_Z]),
                                           pixel_xy=list(entry["pixel"]) if entry is not None else None))


def _log_place(held, target, outcome) -> None:
    """Scenario.placed: log it, and the piece now stands on the target square."""
    if outcome.refused:
        board_log.record_pick("place", held.name, target.name, False, refused=outcome.message)
        return
    board_log.record_pick("place", held.name, target.name, outcome.ok,
                          at_m=[float(v) for v in target.position[:2]],
                          profile=dataclasses.asdict(outcome.grasp) if outcome.grasp is not None else None,
                          id=held.track_id, message=outcome.message, approach=outcome.approach)
    if outcome.ok or outcome.released:
        held.square = target.name


def _max_height(obj) -> float:
    """
    The tallest a piece the gripper camera has not measured could be: what the
    user said it is, else the tallest piece, whatever the classifier called it
    (see config.TALLEST_PIECE_M). Planned as a king, a pawn's pick re-aligns
    from a hover so high the wrist camera loses it.
    """
    track = board_log.memory().get(getattr(obj, "track_id", None))
    if track is not None and track.label_source in ("user", "taught") and track.label in PIECE_SIZES_M:
        return PIECE_SIZES_M[track.label][0]
    return TALLEST_PIECE_M


def _min_height(obj) -> float:
    """The shortest a piece not yet measured could be: what the user said it is, else the shortest piece."""
    track = board_log.memory().get(getattr(obj, "track_id", None))
    if track is not None and track.label_source in ("user", "taught") and track.label in PIECE_SIZES_M:
        return PIECE_SIZES_M[track.label][0]
    return SHORTEST_PIECE_M


def _assumed_shape(obj) -> "dict | None":
    """
    The widest a piece not yet measured could be up its height, {height_mm:
    width_mm}: its foot as the overhead camera measured it, times
    config.UNMEASURED_SHAPE_SHARE, up to the tallest it could be.
    """
    geom = getattr(obj, "geometry", None)
    if not geom or not geom.get("base_diameter_m"):
        return None
    foot = geom["base_diameter_m"] * 1000
    levels = sorted(UNMEASURED_SHAPE_SHARE.items())
    top = int(round(_max_height(obj) * 1000))
    shape = {h: share * foot for h, share in levels if h < top}
    shape[top] = levels[-1][1] * foot
    return {h: round(w, 1) for h, w in shape.items()}


def install_manipulation() -> None:
    """Register pick, place, move_above, home, open/close_gripper (manipulation.py) for chess."""
    from . import chessboard
    manipulation.install(
        POLICIES,
        vision=Vision(measure=_measure_piece, find=_find_piece, relocate=_relocate_piece),
        scenario=manipulation.Scenario(
            max_height=_max_height,
            min_height=_min_height,
            assumed_shape=_assumed_shape,
            describe=chessboard.kind_text,
            picked=_log_pick,
            vacated=_vacated,
            placed=_log_place,
            # A piece by the square it stands on; an empty square by its name.
            approach_tilt=lambda obj: approach_tilt(getattr(obj, "square", None) or obj.name),
            aside_spots=_aside_spots,
            home_of=_home_of,
        ),
    )


# How far from a piece an empty square may be to set it down on for a moment,
# and how near the piece being picked it may not be (in squares).
ASIDE_WITHIN_SQUARES = 2.5
ASIDE_CLEAR_OF_SQUARES = 1.5


def _aside_spots(scene_state, obj, for_obj, headed_for=None) -> list:
    """
    Scenario.aside_spots: empty squares in reach to set obj down on while
    for_obj is picked — within ASIDE_WITHIN_SQUARES of obj, not beside for_obj
    or the square it is going to (ASIDE_CLEAR_OF_SQUARES), the fewest pieces
    round it first, then the nearest. 1. e4 e5 2. Bd3 with the f1 bishop
    crowded by the king: e3 is the king's best free square, and the king set
    down there would crowd the bishop off d3.
    """
    from . import play, rules

    pitch = play.pitch_m()
    here = np.asarray(obj.position[:2], float)
    keep_clear = [np.asarray(for_obj.position[:2], float)]
    going = play.square_xy(headed_for) if headed_for in rules.SQUARES else None
    if going is not None:
        keep_clear.append(np.asarray(going, float)[:2])
    taken = play.occupied(scene_state)
    out = []
    for t in scene_state.targets:
        if t.name not in rules.SQUARES or t.name in taken or t.name == headed_for or not play.reachable(t.name):
            continue
        xy = play.square_xy(t.name)
        if xy is None:
            continue
        d = float(np.linalg.norm(xy - here))
        if d > ASIDE_WITHIN_SQUARES * pitch or any(float(np.linalg.norm(xy - c)) < ASIDE_CLEAR_OF_SQUARES * pitch
                                                   for c in keep_clear):
            continue
        crowd = len((play.around({t.name}) - {t.name}) & taken)
        out.append((crowd, d, t))
    return [t for _, _, t in sorted(out, key=lambda c: (c[0], c[1]))]


def _home_of(scene_state, obj, xy) -> "tuple[str, np.ndarray] | None":
    """Scenario.home_of: the square whose middle is nearest xy, within 0.6 of a square, and that middle."""
    from . import chessboard, play

    xy = np.asarray(xy, float)[:2]
    best, best_d = None, None
    for sq, entry in chessboard.LAST_SQUARES.items():
        d = float(np.linalg.norm(np.asarray(entry["world"], float)[:2] - xy))
        if best_d is None or d < best_d:
            best, best_d = sq, d
    if best is None or best_d > 0.6 * play.pitch_m():
        return None
    return best, np.asarray(chessboard.LAST_SQUARES[best]["world"], float)[:2]


install_manipulation()


# ── Learned skills: a trained model as a verb (config.LEARNED_SKILLS) ─────────

def _look_again(env, scene_state, obj, outcome) -> None:
    """object_action's after: a learned skill can leave the piece anywhere — a fresh scan says where."""
    print(f"  {outcome.approach or 'The skill'} may have moved {obj.name} — looking again.")
    skill_observe(env, scene_state)


def learned_skill(entry: dict):
    """One LEARNED_SKILLS entry as an object_action policy: hover where its demonstrations began, then the model."""
    from mrcf.policies import packages
    from mrcf.policies.learned import LearnedPolicy, chain, hover_over
    entry = packages.skill_entry(entry)
    model = entry.get("model")
    if model is None:
        from mrcf.policies.lerobot_model import LeRobotModel
        model = LeRobotModel(entry["checkpoint"], task=entry.get("task"))     # its weights load when first run
    learned = LearnedPolicy(model, name=entry["name"], max_s=float(entry.get("max_s", 30.0)),
                            trained_on=entry.get("trained_on"))
    start = entry.get("start_height_m")
    return chain(*([hover_over(float(start))] if start is not None else []), learned)


def install_learned_skills(entries) -> list:
    """
    Register each entry as a planner verb on one named piece; their names. A
    policy package that cannot be had (offline, a wrong reference), or a model
    this machine lacks a library for (LeRobot, Stable-Baselines3), is left out,
    saying why, rather than stopping the deployment from starting: the planner
    is never offered a verb that cannot run.
    """
    from mrcf.policies import packages
    installed = []
    for entry in entries:
        try:
            entry = packages.skill_entry(entry)
            policy = learned_skill(entry)
            missing = policy.unavailable()
        except Exception as exc:          # noqa: BLE001 — whatever the source or entry raised: it is that skill's
            print(f"  [Skills] Learned skill {entry.get('name', entry.get('policy'))!r} left out: {exc!r}")
            continue
        if missing:
            print(f"  [Skills] Learned skill {entry['name']!r} left out: its model {missing}.")
            continue
        manipulation.object_action(entry["name"], policy, description=entry["description"],
                                   after=_look_again)
        installed.append(entry["name"])
    return installed


# The planner is told these are motion skills too (llm_controller).
LEARNED = install_learned_skills(LEARNED_SKILLS)


# ── Standing a fallen piece back up (mrcf/policies/stand_up.py) ────────────────────

def _stand_up(pol, env, target, obstacles, **_):
    """
    object_action's policy: how the piece lies, from the overhead camera with
    the arm parked out of its view (mrcf/perception/lying.py) — then the policy
    stands it up from that. Refused, having only parked, when the camera
    cannot see it lying: not outlined whole (a cream head on a maple square),
    or it may be standing.
    """
    from .config import CAPTURE_H_OVERHEAD, CAPTURE_W_OVERHEAD, SURFACE_REFERENCE_PATH
    from mrcf.perception import lying, surface_scan
    from mrcf.policies.base import Lying, Outcome

    goto_observe_pose(env)
    reference = surface_scan.load_reference(SURFACE_REFERENCE_PATH, (CAPTURE_W_OVERHEAD, CAPTURE_H_OVERHEAD))
    tallest = target.max_height_m or TALLEST_PIECE_M
    seen, why = lying.find_lying(env, OBSERVE_CAMERA, CAPTURE_W_OVERHEAD, CAPTURE_H_OVERHEAD, target.position[:2],
                                 reference=reference, max_length_mm=1150.0 * tallest)
    if seen is None:
        return Outcome(False, f"cannot see it lying from above: {why}", refused=True)
    print(f"  [Stand up] Seen lying: {seen.length_mm:.0f} mm long, its top {seen.top_width_mm:.0f} mm across at "
          f"({seen.top_m[0]:+.3f}, {seen.top_m[1]:+.3f}), its foot at "
          f"({seen.foot_end_m[0]:+.3f}, {seen.foot_end_m[1]:+.3f}).")
    target.lying = Lying(foot_xy=seen.foot_end_m, top_xy=seen.top_m, length_m=seen.length_mm / 1000.0,
                         top_width_m=seen.top_width_mm / 1000.0,
                         widths=tuple((at / 1000.0, w / 1000.0) for at, w in seen.widths_mm))
    # Out of the park pose (shoulder at its stop, wrist folded back) the arm's
    # solver finds no straight-down approach anywhere on the board, and a lean
    # lets the head squeeze out of the jaws; from home it comes straight down.
    env.go_home()
    return pol.stand_up(env, target, obstacles)


manipulation.object_action(
    "stand_up", _stand_up,
    description=("Stand a piece that has fallen over onto its side back up, where it lies: grips its top "
                 "and lifts it upright. Only for a piece lying on its side."),
    moves_object=True, after=_look_again, finds_missing=True)


@register_action(
    "identify",
    description=(
        "Record what a piece is when the user says so — 'that's the queen on d5', "
        "'the piece on b4 is the black king'. Remembered from then on, through moves "
        "and restarts, and trusted over what the camera guesses."
    ),
    parameters={
        "object_name": "Exact name of the piece being identified.",
        "what":        "What the user says it is: a type, optionally with its colour.",
    },
    conditions={"pre": ["object_exists"]},
)
def skill_identify(env: RobotEnv, scene_state, object_name, what, **_):
    obj = scene_state.find(object_name)
    memory = scene_state.memory
    if obj is None or memory is None or memory.get(obj.track_id) is None:
        print(f"  [Error] '{object_name}' is not a remembered piece. Try 'observe' first.")
        return False
    words = str(what).lower().replace("_", " ").split()
    colours = [w for w in words if w in PIECE_COLOURS]
    types = [w for w in words if w in PIECE_TYPES]
    # A chess word if there is one; otherwise what the user said, as said —
    # the memory takes any label.
    label = types[0] if types else (" ".join(w for w in words if w not in PIECE_COLOURS) or None)
    track = memory.get(obj.track_id)
    memory.set_label(track.id, label, "user", attributes={"colour": colours[0]} if colours else None)
    from . import chessboard
    old = obj.name
    obj.name = memory.name(track)
    where = getattr(obj, "square", None) if obj.position is not None else None
    obj.description = chessboard.kind_text(track) + (f" on {where}" if where else ", held in the gripper")
    print(f"  Noted: {old} is a {chessboard.kind_text(track)} — now called {obj.name}.")
    return True


@register_action(
    "forget",
    description=(
        "Forget everything remembered about the pieces, then look again. Use only when "
        "the user says they have reset, cleared or rearranged the whole board."
    ),
    always=True,
)
def skill_forget(env: RobotEnv, scene_state, **_):
    if scene_state.memory is not None:
        scene_state.memory.clear()
    print("  Forgot every remembered piece.")
    return skill_observe(env, scene_state)


# ── Scene / status skills ──────────────────────────────────────────────────────

@register_action(
    "observe",
    description="Re-scan the scene with the camera to update object positions.",
    always=True,
)
def skill_observe(env: RobotEnv, scene_state, **_):
    print("  Moving to observe pose...")
    goto_observe_pose(env)
    print("  Re-observing scene...")
    ok = scan_scene(env, scene_state)
    print(scene_state.summary())
    return ok


@register_action(
    "list_objects",
    description="Print all currently known objects and their last observed positions.",
    always=True,
)
def skill_list_objects(scene_state, **_):
    print()
    print(scene_state.summary())
    print()
    return True


@register_action("list_actions", description="Print all available robot actions.", always=True)
def skill_list_actions(**_):
    print("\nAvailable actions:")
    for name, info in ACTIONS.items():
        params    = list(info["parameters"].keys())
        param_str = f"({', '.join(params)})" if params else "()"
        print(f"  {name}{param_str}")
        print(f"    {info['description']}")
    print()
    return True


@register_action(
    "status",
    description="Print the current robot sensor readings: joint angles, end-effector position, and gripper state.",
    always=True,
)
def skill_status(env: RobotEnv, **_):
    width = env.get_gripper_width()
    joints = env.get_joint_angles()
    print("\nRobot status (live sensor readings):")
    try:
        ee = env.get_ee_pos()
        print(f"  End-effector (FK)  : [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] m")
    except NotImplementedError as exc:
        print(f"  End-effector (FK)  : unavailable ({exc})")
    print(f"  Gripper opening    : {width * 1000:.1f} mm  ({'open' if width > 0.01 else 'closed'})")
    print("  Joint angles (rad) :")
    for name, angle in joints.items():
        print(f"    {name:<16}: {angle:+.4f}")
    print()
    return True


# ── Vision / Q&A skills ────────────────────────────────────────────────────────

@register_action(
    "answer",
    description=(
        "Answer a descriptive question about the scene by looking at the camera — "
        "use this for any question about appearance, shape, colour, position, or "
        "what is visible."
    ),
    parameters={"question": "The question to answer based on the current camera view."},
    always=True,
)
def skill_answer(env: RobotEnv, question, **_):
    print("  Looking at scene...")
    img_b64 = None
    try:
        img_b64 = render_to_base64(env)
    except Exception as exc:
        print(f"  [Answer] Render failed: {exc}")

    prompt = (
        # render_to_base64() returns the PRIMARY observation camera, which on
        # this rig is the fixed overhead one — not the gripper camera. Telling
        # the model it is looking down the wrist makes it misread the geometry.
        f"You are observing a robot tabletop manipulation scene from the "
        f"'{OBSERVE_CAMERA}' camera, looking down at the table from above.\n"
        "Answer this question concisely based on what you can see:\n\n"
        f"Question: {question}"
    )
    response = call_vlm(prompt, img_b64, json_mode=False)
    print(f"\n  {response}\n")
    return bool(response)


# ── General-purpose tools (no robot hardware required) ─────────────────────────

_SAFE_OPS: dict = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}

_SAFE_NAMES: dict = {
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "degrees": math.degrees, "radians": math.radians,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "min": min, "max": max,
    "factorial": math.factorial, "gcd": math.gcd,
    "inf": math.inf,
}


_MAX_FACTORIAL = 1000


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only safe math operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported literal: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            val = _SAFE_NAMES[node.id]
            if callable(val):
                raise ValueError(f"'{node.id}' is a function — call it with arguments")
            return float(val)  # type: ignore[arg-type]
        raise ValueError(f"Unknown name: '{node.id}'")
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named functions are allowed")
        fn = _SAFE_NAMES.get(node.func.id)
        if not callable(fn):
            raise ValueError(f"Unknown function: '{node.func.id}'")
        args = [_safe_eval(a) for a in node.args]
        # The planner passes on whatever the user typed, and factorial of a
        # large number runs for minutes; nothing else here is unbounded.
        if node.func.id == "factorial" and args and abs(args[0]) > _MAX_FACTORIAL:
            raise ValueError(f"factorial is limited to {_MAX_FACTORIAL}")
        try:
            return float(fn(*args))  # type: ignore[operator]
        except TypeError:
            coerced = [int(a) if isinstance(a, float) and a == int(a) else a for a in args]
            return float(fn(*coerced))  # type: ignore[operator]
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


@register_action(
    "calculate",
    description=(
        "Evaluate a mathematical expression and print the result. "
        "Supports +, -, *, /, **, %, //, and math functions: "
        "sqrt, sin, cos, tan, asin, acos, atan, atan2, log, log10, degrees, radians, "
        "abs, round, floor, ceil, factorial, gcd, min, max. Constants: pi, e, tau, inf."
    ),
    parameters={"expression": "A mathematical expression to evaluate (only math, no Python code)."},
    always=True,
)
def skill_calculate(expression: str, **_):
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        display = int(result) if result == int(result) and abs(result) < 1e15 else result
        print(f"\n  {expression} = {display}\n")
        return True
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        print(f"\n  [Calculate] Error: {exc}\n")
    except Exception as exc:
        print(f"\n  [Calculate] Unexpected error: {exc}\n")
    return False
