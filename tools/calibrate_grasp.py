"""
calibrate_grasp.py — where the jaws really close at each lean the arm comes
down at, against where its model puts them: the table RigEnv corrects every
grasp point by (mrcf/grasp_correction.py).

Why: the overhead camera corrects where the gripper is, not how it is turned,
and the jaws are ~90 mm below the tag it watches. On the SO-101 (2026-10-30)
grasps at a lean of 0 to -5 deg gripped a pawn 6 times in 7, at -15 to -20 deg
2 in 8: the wrist camera showed the pawn beyond the fingertips while both
cameras had the jaws over it.

How, for each object and each lean the arm reaches it at:
  1. Over it at that lean, a little above its top, the overhead camera puts the
     jaws over it, as a pick's does.
  2. Down to the grasp height at offsets along the arm, closing the jaws as a
     pick does, and up again: out from the middle (0, +step, -step, +2 step...)
     to the first catch, then on each side of that to the first miss.
  3. The offsets where they caught it are a span along the arm; its middle is
     where the jaws really close, and the correction is that, the other way.
It measures the model as it is: any saved table is left out while it does.

Setup: objects standing free, as they do in use — one near the base, one in
the middle, one far out covers the leans the whole surface needs. A miss
pushes a free-standing object along the arm (2026-10-30: 3-14 mm a lean, all
toward the base), so after each one the object is found again and the next
offsets are from where it stands: every offset is from where the object stood
when the jaws closed. A catch slides it across the jaws, which leaves where it
is along the arm alone. The arm turns to each lean as high above it as that
lean reaches from, so the jaws do not swing through it. The arm moves on its
own; the sentry stops it for a hand.

    python tools/calibrate_grasp.py --squares e3,e5,e7 [--deployment chess]
    python tools/calibrate_grasp.py --at 0.20,0.0 --at 0.30,0.0
    python tools/calibrate_grasp.py --show
Options: --height-mm 43 (the objects' height), --leans 0,-5,-10,-15,-20,-25,-30,
--sweep-mm 40 (the furthest either side it looks), --step-mm 4, --dry (measure,
do not save).
"""

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mrcf.capabilities import OverheadSighting, Settling  # noqa: E402
from mrcf.grasp_correction import REACH_BAND_M, GraspCorrection, Point  # noqa: E402
from mrcf.interlock import MotionStopped  # noqa: E402
from tools import _common  # noqa: E402  — the arm (rc, make_env, ...) only once main() asks
from mrcf import privacy  # noqa: E402

LEANS = (0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0)
CLEAR_M = 0.02              # the probes' hover: this far above the object's top
# ... and the gripper turns to each lean at the highest of these above that
# hover the lean reaches from: on the SO-101 (2026-10-30) at 6 cm the arm
# reached e2 at -25 and -30 only, e4 at -20 to -30, e6 at nothing.
TURN_ABOVE_M = (0.06, 0.04, 0.03, 0.02, 0.0)
LEAN_TOL_DEG = 0.5          # a lean the arm only reaches near: not that lean
MAX_SWEEP_M = 0.04          # the furthest either side of the object the probes go


def overhead_correct(env, at, rot, tol_m: float = 0.001, tries: int = 4) -> "np.ndarray | None":
    """
    Move the jaws over `at` as the overhead camera sees them (the gripper tag),
    as a pick's _overhead_correct does; the (x, y) shift it took. Zeros for an
    arm without the camera; None when it has one that cannot see the tag here
    — what was measured then would hold an error a pick corrects.
    """
    shift = np.zeros(2)
    if not isinstance(env, OverheadSighting):
        return shift
    for i in range(tries):
        if isinstance(env, Settling):
            env.wait_still()
        off = env.overhead_gripper_offset()
        if off is None:
            return None if i == 0 else shift
        err = (np.asarray(at, float) - (env.get_ee_pos() + off))[:2]
        if np.linalg.norm(err) < tol_m:
            break
        shift += err
        env.move_ee_to(np.r_[np.asarray(at, float)[:2] + shift, at[2]], tol=0.002, orientation=rot)
    return shift


class JawsShut(RuntimeError):
    """The jaws would not open off an object: nothing may rise with it between them."""


def open_jaws(env, opening_m: float) -> None:
    """
    Open the jaws to opening_m and make sure they did, before anything rises:
    rising with them still shut on a held-down object pulls it off the surface
    (2026-10-30: a pawn ended up a square further out). JawsShut if they will not.
    """
    for _ in range(2):
        env.set_gripper(opening_m)
        if env.get_gripper_width() >= 0.7 * opening_m:
            return
        time.sleep(0.3)
    raise JawsShut(f"the jaws did not open ({env.get_gripper_width() * 1000:.0f} mm of {opening_m * 1000:.0f})")


def _wrist_frame(env, pol):
    """What the wrist camera sees now (jaws open at the grasp), or None without one."""
    s = pol.spec
    if s.camera is None:
        return None
    try:
        return env.render_image(s.camera, width=s.render_w, height=s.render_h)
    except Exception:                                           # noqa: BLE001 — a frame is a nicety here
        return None


def _keep(frame_rgb, path: Path) -> None:
    """Save an RGB frame as a JPEG (a missing cv2 or a full disk costs the frame, not the run)."""
    try:
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        privacy.imwrite(str(path), cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR))
    except Exception:                                           # noqa: BLE001
        pass


class _LeanEnded(Exception):
    """The object went, or out of reach at this lean, part-way through its probes."""


def _over(env, pol, xy, height_m: float, lean: float, say=print) -> "dict | None":
    """
    The gripper turned to `lean` high over xy, then down to the probes' hover
    and put over xy by the overhead camera: {"rot", "high", "hover", "grasp",
    "u" (along the arm), "shift", "opening"}. None when the arm does not reach
    xy at that lean or the camera cannot see the gripper tag.
    """
    s = pol.spec
    xy = np.asarray(xy, float)
    plan = pol.plan(_target(xy, s.table_z, height_m))
    hover = np.r_[xy, s.table_z + height_m + CLEAR_M]
    grasp = np.r_[xy, s.table_z + plan.grasp_height_m]
    for above in TURN_ABOVE_M:
        high = hover + np.array([0.0, 0.0, above])
        chosen = env.approach_orientation_for([high, hover, grasp], tols=[0.01, 0.01, 0.004], prefer_tilt_deg=lean)
        if chosen is not None and abs(chosen[1] - lean) <= LEAN_TOL_DEG:
            break
    else:
        say(f"    lean {lean:+.0f}: out of reach here")
        return None
    rot = chosen[0]
    u = xy - np.asarray(env.base_xy, float)[:2]
    env.set_gripper(plan.pregrasp_open_m)
    top = hover + np.array([0.0, 0.0, TURN_ABOVE_M[0]])
    if above < TURN_ABOVE_M[0]:
        # Over it high up first, however the arm reaches there, then down:
        # straight in from home to 2 cm over a pawn's top on e2, the arm
        # knocked it over before the jaws closed (2026-10-30).
        way = env.approach_orientation_for([top], prefer_tilt_deg=lean)
        if way is not None:
            env.move_ee_to(top, tol=0.01, orientation=way[0])
    # Turned to this lean up here, then straight down: turning the wrist at the
    # probes' hover swings the jaws through the object (2026-10-30: a pawn
    # held down on e2 knocked off twice).
    env.move_ee_to(high, tol=0.01, orientation=rot)
    if not env.move_ee_to(hover, tol=0.01, orientation=rot):
        say(f"    lean {lean:+.0f}: did not reach the hover — skipped")
        return None
    shift = overhead_correct(env, hover, rot)
    if shift is None:
        say(f"    lean {lean:+.0f}: the overhead camera cannot see the gripper tag here — skipped")
        return None
    return {"rot": rot, "high": high, "hover": hover, "grasp": grasp, "u": u / np.linalg.norm(u),
            "shift": shift, "opening": plan.pregrasp_open_m}


def probe_lean(env, pol, xy, height_m: float, lean: float, step_m: float = 0.004, say=print,
               frames: "Path | None" = None, name: str = "", recheck=None,
               max_m: float = MAX_SWEEP_M) -> "dict | None":
    """
    One object, one lean: the offsets along the arm, each from where the
    object stood when the jaws closed, that the jaws caught it at — out from
    the middle to the first catch, then each side of it to the first miss
    (see the module docstring). recheck(xy) -> where the object last at xy
    stands now (None: gone), asked after every miss; without it the object is
    taken not to move. None where the arm does not reach it at that lean.
    Returns {"lean_deg" (as the arm read it at the grasps), "caught": [offset_m...],
    "tried", "widths_mm", "moved_mm" (along the arm, at each finding again),
    "xy" (where it stands now; None: gone)}.
    """
    s = pol.spec
    at = _over(env, pol, xy, height_m, lean, say)
    if at is None:
        return None
    here: dict = {"xy": np.asarray(xy, float), "at": at}
    caught, widths, leans, tried, moved = [], [], [], [], []

    def close_at(d: float) -> bool:
        """Down at offset d from the object, close, open, up: whether the jaws caught it."""
        if here["at"] is None:
            here["at"] = _over(env, pol, here["xy"], height_m, lean, say)
            if here["at"] is None:
                raise _LeanEnded
        a = here["at"]
        rot = a["rot"]
        above = a["hover"] + np.r_[a["shift"] + d * a["u"], 0.0]
        down = a["grasp"] + np.r_[a["shift"] + d * a["u"], 0.0]
        env.move_ee_to(above, tol=0.004, orientation=rot)
        if not env.move_ee_to(down, tol=0.004, max_step_deg=s.grasp_step_deg, orientation=rot):
            say(f"    lean {lean:+.0f}, {d * 1000:+.0f} mm: stalled short — not counted")
            env.move_ee_to(above, tol=0.01, orientation=rot)
            return False
        tried.append(float(d))
        leans.append(env.grasp_lean_deg(env.get_ee_rotation(), env.get_ee_pos()))
        seen = _wrist_frame(env, pol) if frames is not None else None
        # Closed as a pick closes, gently: at the arm-wide rate (~70 mm/s) the
        # jaws threw a pawn off the board on a near miss (2026-10-30).
        env.set_gripper(0.0, n_steps=20, max_step_pct=s.close_step_pct)
        width, loaded = env.squeeze_gripper()
        if seen is not None and frames is not None:
            _keep(seen, frames / f"{name}_lean{lean:+03.0f}_{d * 1000:+05.1f}mm_{'caught' if loaded else 'missed'}"
                                 f"_{len(tried)}.jpg")
        widths.append(round(width * 1000, 1))
        if loaded:
            caught.append(float(d))
        open_jaws(env, a["opening"])
        env.move_ee_to(above, tol=0.01, orientation=rot)
        say(f"    lean {lean:+.0f}, {d * 1000:+5.1f} mm along: {'caught' if loaded else 'missed'} "
            f"({width * 1000:.1f} mm)")
        if not loaded and recheck is not None:
            # A miss may have pushed it: found again, and the next offsets from there.
            env.move_ee_to(a["high"], tol=0.01, orientation=rot)
            now = recheck(here["xy"])
            if now is None:
                here["xy"] = None
                raise _LeanEnded
            along = float((np.asarray(now, float) - here["xy"]) @ a["u"])
            moved.append(round(along * 1000, 1))
            if abs(along) >= 0.002:
                say(f"    {name} moved {along * 1000:+.0f} mm along the arm — the next offsets from where it stands")
            here["xy"], here["at"] = np.asarray(now, float), None
        return loaded

    n = int(round(max_m / step_m))
    try:
        first = None
        for k in range(n + 1):
            for d in ((0.0,) if k == 0 else (round(k * step_m, 4), round(-k * step_m, 4))):
                if close_at(d):
                    first = d
                    break
            if first is not None:
                break
        for sign in (1.0, -1.0):
            edge = first
            while edge is not None and abs(edge + sign * step_m) <= max_m + 1e-9:
                edge = round(edge + sign * step_m, 4)
                if edge in tried or not close_at(edge):
                    break
    except _LeanEnded:
        pass
    if here["at"] is not None:
        env.move_ee_to(here["at"]["high"], tol=0.01, orientation=here["at"]["rot"])
    return {"lean_deg": float(np.median(leans)) if leans else lean, "caught": sorted(caught),
            "tried": sorted(tried), "widths_mm": widths, "moved_mm": moved, "xy": here["xy"]}


def point_from(probe: dict, reach_m: float, note: str = "") -> "Point | None":
    """
    The correction one lean's probes give: minus the middle of the run of
    offsets that caught the object (the longest unbroken one). None when none
    did, or when the run touches an end of the sweep — the span may go on
    past it, so its middle is not known.
    """
    tried, caught = probe["tried"], set(probe["caught"])
    runs, run = [], []
    for d in tried:
        if d in caught:
            run.append(d)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    if not runs:
        return None
    best = max(runs, key=len)
    if best[0] == tried[0] or best[-1] == tried[-1]:
        return None
    centre = (best[0] + best[-1]) / 2.0
    return Point(lean_deg=round(probe["lean_deg"], 1), reach_m=round(reach_m, 4), along_m=round(-centre, 4),
                 span_m=[round(best[0], 4), round(best[-1], 4)], note=note)


def calibrate(env, pol, objects: dict, height_m: float, leans=LEANS, sweep_m: float = MAX_SWEEP_M,
              step_m: float = 0.004, say=print, recheck=None, into: "list | None" = None,
              frames: "Path | None" = None) -> list:
    """
    Every object ({name: (x, y)}), at every lean in `leans` it can be reached
    at: the Points measured (see probe_lean, point_from). Leaves the jaws open
    and the arm over the last object. recheck(xy) -> where the object last
    at xy stands now (None: gone), asked after every miss (see probe_lean);
    each lean starts from where the object was last found. sweep_m: the
    furthest either side of it the probes go. `into`: the list to add them
    to as they come, so a stop part-way keeps those already measured.
    `frames`: a folder to keep the wrist camera's view at each probe in, just
    before the close, named by object, lean, offset and whether it caught.
    """
    points = into if into is not None else []
    base = np.asarray(env.base_xy, float)[:2]
    for name, xy in objects.items():
        xy = np.asarray(xy, float)
        say(f"  {name} at ({xy[0]:+.3f}, {xy[1]:+.3f}), {float(np.linalg.norm(xy - base)) * 1000:.0f} mm out:")
        for lean in leans:
            reach = float(np.linalg.norm(xy - base))
            probe = probe_lean(env, pol, xy, height_m, lean, step_m, say=say, frames=frames, name=name,
                               recheck=recheck, max_m=sweep_m)
            if probe is None:
                continue
            p = point_from(probe, reach, note=name)
            if p is None:
                say(f"    lean {lean:+.0f}: no clear span (caught at "
                    f"{[round(d * 1000) for d in probe['caught']]} mm) — not used")
            else:
                lo, hi = p.span_m or (0.0, 0.0)
                say(f"    lean {p.lean_deg:+.1f}: caught {lo * 1000:+.0f} to {hi * 1000:+.0f} mm — "
                    f"the jaws close {abs(p.along_m) * 1000:.1f} mm "
                    f"{'further out' if p.along_m > 0 else 'nearer the base'} than the model has them")
                points.append(p)
            if probe["xy"] is None:
                say(f"  {name} is gone — no more leans on it.")
                break
            xy = np.asarray(probe["xy"], float)      # the next lean starts from where it stands now
    return points


def merged(old: GraspCorrection, new: list) -> GraspCorrection:
    """old's points, less any at a lean and reach measured again in `new`, plus `new`."""
    keep = [p for p in old.points
            if not any(abs(p.lean_deg - n.lean_deg) < 2.5 and abs(p.reach_m - n.reach_m) <= REACH_BAND_M for n in new)]
    return GraspCorrection(points=keep + list(new))


def _target(xy, table_z: float, height_m: float):
    from mrcf.policies.base import Target
    return Target(position=np.r_[xy, table_z], height_m=height_m)


def find_objects(env, expected: dict, height_m: float) -> dict:
    """
    Each object's foot, where a pick will aim at it: through the deployment's
    own scan when it has one (its llm_controller's DEPLOYMENT — on a board,
    the board's own measuring, which the picks aim by), else measured by the
    overhead camera near where expected. Stops if any is not found.
    """
    scanned = _deployment_scan(env)
    found, missing = {}, []
    for name, xy in expected.items():
        xy = np.asarray(xy, float)
        if scanned is not None:
            near = min(scanned, key=lambda p: float(np.linalg.norm(p - xy)), default=None)
            foot = near if near is not None and np.linalg.norm(near - xy) <= 0.025 else None
            why = "the scan found nothing within 25 mm"
        else:
            foot, why = _measure_near(env, xy, height_m)
        if foot is None:
            missing.append(f"{name} ({why})")
            continue
        print(f"  {name}: foot at ({foot[0]:+.4f}, {foot[1]:+.4f}), "
              f"{np.linalg.norm(foot - xy) * 1000:.1f} mm from where expected")
        found[name] = foot
    if missing:
        raise SystemExit("Not found: " + "; ".join(missing) + " — nothing probed.")
    return found


def locate(env, xy, height_m: float) -> "np.ndarray | None":
    """
    Where the object nearest xy stands now (as find_objects finds it); None
    when nothing is within 25 mm. Leaves the arm home: the next lean is
    planned from where the arm is, and from where a scan parks the SO-101,
    e2 at 0 and -5 deg came out out of reach (2026-10-30; from home both reach).
    """
    scanned = _deployment_scan(env)
    if scanned is None:
        found = _measure_near(env, np.asarray(xy, float), height_m)[0]
    else:
        near = min(scanned, key=lambda p: float(np.linalg.norm(p - xy)), default=None)
        found = near if near is not None and np.linalg.norm(near - xy) <= 0.025 else None
    env.go_home()
    return found


def leave_clear(env, opening_m: float) -> None:
    """
    The jaws opened, lifted straight up 4 cm, then home. If they will not
    open, the arm stays where it is: lifting would take the object with it.
    """
    try:
        open_jaws(env, opening_m)
    except JawsShut as shut:
        print(f"  {shut}: the arm stays where it is, holding on. Take the object out of the jaws by hand "
              "(python tools/so101/rest.py lets the arm go limp first, if you need it to).")
        return
    here = env.get_ee_pos()
    env.move_ee_to(here + np.array([0.0, 0.0, 0.04]), tol=0.01, orientation=env.get_ee_rotation())
    env.go_home()


def _deployment_scan(env) -> "list | None":
    """The feet of everything the deployment's own scan finds (the arm moved out of view first); None without one."""
    import importlib

    from mrcf.scene_state import SceneState
    try:
        cfg = _common.task_config()
        deployment = getattr(importlib.import_module(f"{cfg.__package__}.llm_controller"), "DEPLOYMENT", None)
    except (SystemExit, ImportError):
        return None
    if deployment is None or deployment.scan is None:
        return None
    state = SceneState()
    deployment.look(env, state)
    return [np.asarray(o.position[:2], float) for o in state.objects if o.position is not None]


def _measure_near(env, xy, height_m: float) -> tuple:
    """(foot, why): the object nearest xy, as the overhead camera measures it with the arm parked."""
    from mrcf.perception import plane_geometry
    rc = _common.rc
    camera = rc.OVERHEAD_CAMERA_NAME
    w, h = rc.OVERHEAD_CAMERA_W, rc.OVERHEAD_CAMERA_H
    env.go_park()
    time.sleep(0.5)
    frame = env.render_image(camera, width=w, height=h)
    view = plane_geometry.PlaneView.from_camera_model(env.camera_model(camera, w, h))
    r = plane_geometry.measure_near(frame, view, expected_xy_m=xy, search_px=60.0,
                                    height_mm=(500 * height_m, 1500 * height_m))
    if not r["ok"]:
        return None, r["why"]
    return np.asarray(r["measurement"].base_world_m, float), ""


def _squares(names) -> dict:
    """
    {square: (x, y)} for the board's squares: from where the board was last
    measured (overhead_calibration, as tag_tour does), else the squares the
    deployment's last scan saved beside its config (board_squares.json).
    """
    import json

    from mrcf import overhead_calibration
    saved = overhead_calibration.load()
    cfg = _common.task_config()
    if saved and saved.get("T_arm_target") is not None:
        T, p = np.asarray(saved["T_arm_target"], float), cfg.CHESSBOARD_SQUARE_SIZE_M
        return {n: (T @ np.array([("abcdefgh".index(n[0]) - 3.5) * p, (int(n[1]) - 4.5) * p, 0, 1]))[:2]
                for n in names}
    scanned = Path(cfg.__file__).with_name("board_squares.json")
    if not scanned.is_file():
        raise SystemExit("Neither the board's pose (overhead_calibration) nor a scan's board_squares.json "
                         "is there — give --at x,y instead.")
    squares = json.loads(scanned.read_text(encoding="utf-8"))
    return {n: np.asarray(squares[n]["world"], float) for n in names}


def show(table: GraspCorrection) -> None:
    if not table:
        print("No grasp correction saved: grasp points are as the model has them.")
        return
    print(f"Grasp correction, measured {table.measured}:")
    print(f"  {'reach':>7} {'lean':>6} {'along':>8}  caught (along the arm)    note")
    for p in sorted(table.points, key=lambda p: (p.reach_m, -p.lean_deg)):
        span = f"{p.span_m[0] * 1000:+.0f} to {p.span_m[1] * 1000:+.0f} mm" if p.span_m else ""
        print(f"  {p.reach_m * 1000:5.0f}mm {p.lean_deg:+5.1f}° {p.along_m * 1000:+6.1f}mm  {span:24} {p.note}")


def main() -> None:
    path = getattr(_common.rc, "GRASP_CORRECTION_PATH", None)
    if path is None:
        raise SystemExit("This arm's config has no GRASP_CORRECTION_PATH.")
    saved = GraspCorrection.load(path)
    if "--show" in sys.argv:
        show(saved)
        return
    expected = {}
    if _common.arg("--squares"):
        expected.update(_squares(_common.arg("--squares").split(",")))
    for i, a in enumerate(sys.argv[:-1]):
        if a == "--at":
            x, y = (float(v) for v in sys.argv[i + 1].split(","))
            expected[f"({x:+.3f}, {y:+.3f})"] = (x, y)
    if not expected:
        raise SystemExit(__doc__)
    height = float(_common.arg("--height-mm", 43)) / 1000
    leans = [float(v) for v in _common.arg("--leans", ",".join(f"{v:g}" for v in LEANS)).split(",")]
    sweep, step = float(_common.arg("--sweep-mm", MAX_SWEEP_M * 1000)) / 1000, float(_common.arg("--step-mm", 4)) / 1000

    env, pol = _common.make_env(), _common.make_policies()
    env._grasp_table = None              # measure the model as it is
    print(f"Connecting to {_common.connecting_to()}...")
    env.connect()
    # Ctrl-C stops the arm through its interlock, at its next step, whatever
    # else in the process handles the signal; then it is left clear (below).
    signal.signal(signal.SIGINT, lambda *_: env.interlock.trip("stopped from the keyboard", source="keyboard"))
    points: list = []
    try:
        env.go_home()
        objects = find_objects(env, expected, height)
        env.go_home()
        frames = _common.FRAMEWORK_ROOT / "reports" / "calibrate_grasp" / time.strftime("%Y%m%d_%H%M%S")
        print(f"  The wrist camera's view at each probe goes to {frames}")
        calibrate(env, pol, objects, height, leans, sweep, step,
                  recheck=lambda xy: locate(env, xy, height), into=points, frames=frames)
    except MotionStopped as stop:
        print(f"\n  Stopped: {stop.reason}.")
    except JawsShut as shut:
        print(f"\n  Stopped: {shut}.")
    finally:
        try:
            env.interlock.clear("keyboard")
            leave_clear(env, 0.8 * pol.spec.max_open_m)
        finally:
            env.stop()
    table = merged(saved, points)
    show(GraspCorrection(points=points, measured=time.strftime("%Y-%m-%d %H:%M")))
    if "--dry" in sys.argv:
        print("--dry: not saved.")
        return
    if not points:
        print("Nothing measured: the saved table is left as it was.")
        return
    table.measured = time.strftime("%Y-%m-%d %H:%M")
    table.save(path)
    print(f"Saved to {path} ({len(table.points)} points in all). The arm applies it from its next connect.")


if __name__ == "__main__":
    main()
