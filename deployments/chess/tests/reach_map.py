"""
reach_map.py — on which squares can the arm pick up and put down each piece?

Every board square, every phase of a pick and of a place, per piece type —
because the answer depends on the piece. A king needs a hover high enough for
the jaw tips to clear its crown, and the higher the hover, the less of the
board the arm reaches with the gripper straight down. Each piece is planned by
the arm's own policy (make_policies().plan) from its size in config.PIECE_SIZES_M.

Two modes, the SAME motion: the approach the policy's pick would choose on the
square (top_down, near_base or reach_out — mrcf/policies/approaches.py, Approach),
with its waypoints, tolerances, step limits and held orientation, and the
place's likewise, starting from home as every command does. A map made any
other way approves squares the skills then fail on.

  predicted (default)  Offline. Drives the real move_ee_to against a simulated
                       servo (visualise_ik.offline_arm). No hardware. Optimistic:
                       no sag, backlash or torque limit, so a square it fails is
                       certainly out, one it passes is only probably in.

  --live               The real arm over an EMPTY board, jaws open. Drives every
                       phase at every square and records what the arm actually
                       reached and held. At the hover it also grabs a gripper-
                       camera frame and counts the board corners the servo would
                       work from — the part kinematics cannot predict: near the
                       board's edge the camera sees border and table instead of
                       grid, and with no board pose there is no servo.

Per square, the verdict is the worst of:
  fail      no approach reaches it, or a phase did not converge
  clash     the held tilt leans the piece into a jaw, even at the grasp height
            reach_out raises it to (ParallelJaw.top_overlap). Only possible with
            JAW_AXIS = "radial", where the tilt is in the plane the jaws open in.
            Tangential jaws tilt along the pads
  blind     (live) fewer than MIN_SERVO_INLIERS board corners at the hover the
            wrist servo works from
  tilted    the wrist servo aligns, at a lean more than SERVO_LEAN_WARN_DEG from
            the one it was calibrated at (its step for the difference is a model)
  marginal  some joint within MARGIN_WARN_DEG of its limit
  ok
Each square also records which approach was chosen, and why the ones before it
in that square's order could not be done.

Usage (from the repo root):
    python deployments/chess/tests/reach_map.py                     # predicted, every piece
    python deployments/chess/tests/reach_map.py --piece king
    python deployments/chess/tests/reach_map.py --piece unmeasured  # any piece before it is measured
    python deployments/chess/tests/reach_map.py --live --piece pawn
    python deployments/chess/tests/reach_map.py --live --piece queen --squares a1,h1,e8
    python deployments/chess/tests/reach_map.py --report            # HTML of what is recorded

Results go to reports/reach_<predicted|live>_<piece>.json, rewritten after
every square so an interrupted live run keeps what it measured. --report
gathers them into reports/reach_map.html.
"""

import argparse
import contextlib
import dataclasses
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np    # noqa: E402

from deployments.chess import config as cfg  # noqa: E402
from mrcf.backends.so101 import make_policies  # noqa: E402
from mrcf.backends.so101 import robot_config as rc  # noqa: E402
from mrcf.policies.base import Target  # noqa: E402
from mrcf import privacy

POLICIES = make_policies()


# Not a piece type: any piece before the wrist camera has measured it. The pick
# plans it as the tallest piece (skills._max_height) and refuses where no
# approach reaches that plan — so this map is where an object nobody has
# measured or named can be picked at all.
UNMEASURED = "unmeasured"
PIECES = list(cfg.PIECE_SIZES_M) + [UNMEASURED]


def piece_plan(piece: str):
    """The grasp the arm's policy would plan for a piece of this type, measured (or not: UNMEASURED)."""
    if piece == UNMEASURED:
        widest = max(w for _, w in cfg.PIECE_SIZES_M.values())
        return POLICIES.plan(Target(np.zeros(3), foot_m=widest, max_height_m=cfg.TALLEST_PIECE_M))
    height, width = cfg.PIECE_SIZES_M[piece]
    return POLICIES.plan(Target(np.zeros(3), foot_m=width, height_m=height))

REPORTS = ROOT / "reports"
SQUARES_PATH = HERE.parent / "board_squares.json"
FILES, RANKS = "abcdefgh", "12345678"

MARGIN_WARN_DEG = 2.0
SERVO_LEAN_WARN_DEG = 10.0
# The pick's wrist servo runs where it is calibrated (the chess deployment gives
# it a detector): the map assumes so, as the pick on this rig would.
SERVO_READY = POLICIES.spec.servo and bool(getattr(rc, "GRIPPER_CAM_GRASP_PX", None)) \
    and bool(getattr(rc, "GRIPPER_CAM_JACOBIAN", None))
# piece_vision's own floor (_MIN_INLIERS): below it there is no board pose, so
# no servo. Mid-board hovers measured 7-21 on the rig, so the count is recorded
# per square too — a square that scrapes past with 6 over an empty board may
# not survive a piece hiding some of them.
MIN_SERVO_INLIERS = 6
SETTLE_S = 0.5     # live: hold a low pose this long, then re-read it, to catch sag

ORDER = ("fail", "clash", "blind", "tilted", "marginal", "ok")


# ── Running one square ──────────────────────────────────────────────────────────

def load_squares() -> dict:
    if not SQUARES_PATH.is_file():
        raise SystemExit(f"No {SQUARES_PATH} — let the controller scan the board once first.")
    return {k: np.asarray(v["world"], float) for k, v in json.loads(SQUARES_PATH.read_text()).items()}


def _quiet(fn, *a, **kw):
    """fn with its console output swallowed: move_ee_to prints on every stall."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _tilt_deg(rot) -> "float | None":
    """How far a held rotation leans from straight down; None if none was held."""
    if rot is None:
        return None
    return float(np.degrees(np.arccos(np.clip(-np.asarray(rot)[2, 2], -1.0, 1.0))))


def _tightest(env) -> "tuple[float, str]":
    """Smallest joint margin at the current pose, against the IK limits."""
    from mrcf.backends.so101 import JOINT_NAMES
    q = env._q_arm_deg()
    return min((min(q[i] - lo, hi - q[i]), n)
               for i, n in enumerate(JOINT_NAMES)
               for lo, hi in [env._ik_limits.get(n, (-1e9, 1e9))])


def _move(env, name, target, tol, step, rot, live) -> dict:
    t0 = time.time()
    ok = _quiet(env.move_ee_to, target, tol=tol, max_step_deg=step, orientation=rot)
    rec = {"phase": name, "ok": bool(ok), "z": round(float(target[2]), 4),
           "residual_mm": round(float(np.linalg.norm(env.get_ee_pos() - target)) * 1000, 1),
           "seconds": round(time.time() - t0, 1)}
    if live and ok and name in ("grasp", "release"):
        # The low poses are where sag costs a grasp: does the arm stay put?
        time.sleep(SETTLE_S)
        rec["held_mm"] = round(float(np.linalg.norm(env.get_ee_pos() - target)) * 1000, 1)
    margin, joint = _tightest(env)
    rec["margin_deg"], rec["tightest"] = round(float(margin), 1), joint
    return rec


def _open(env, profile, live) -> None:
    if live:
        env.set_gripper(profile.pregrasp_open_m, n_steps=5)   # also clears any held width
    else:
        env._held_width_m = None
        env._robot.bus.positions["gripper"] = env._width_to_pct(profile.pregrasp_open_m)


def survey_square(env, xy, profile, live: bool, camera_check=None) -> dict:
    """
    One pick then one place at xy, as the policy's pick and place would drive
    them — the approach it would choose there, its held orientation and its
    heights — minus the servo and the jaws closing. The arm must start where a
    command starts (home); it is left above the square with the jaws open.
    """
    out = {"xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)]}
    _open(env, profile, live)
    obstacles = POLICIES._with_base(env, [])

    fit, why_not = _quiet(POLICIES.choose_approach, env, xy, profile, obstacles,
                          servo_ready=SERVO_READY, measured=True)
    out["why_not"] = why_not
    out["pick"] = []
    if fit is None:
        out["approach"] = None
        return out
    plan = fit.plan
    out["approach"], out["servo"] = fit.approach.name, fit.servo
    out["grasp_m"], out["lift_m"] = round(plan.grasp_height_m, 4), round(fit.lift_height_m, 4)
    out["pick_tilt"] = None if fit.tilt_deg is None else abs(float(fit.tilt_deg))
    out["open_m"] = round(plan.pregrasp_open_m, 4)
    out["overlap_mm"] = round(POLICIES.top_overlap(plan, None, fit.tilt_deg) * 1000, 1)
    (_, hover, hover_tol, _), (_, grasp, grasp_tol, grasp_step), (_, _, lift_tol, _) = \
        POLICIES.pick_waypoints(xy, plan)
    phases = [("hover", hover, hover_tol, None)]
    low = POLICIES._servo_height_for(hover) if fit.servo else None
    if low is not None:
        phases.append(("servo hover", np.array([xy[0], xy[1], rc.TABLE_Z + low]), hover_tol, None))
    phases += [("grasp", grasp, grasp_tol, grasp_step),
               ("lift", np.array([xy[0], xy[1], rc.TABLE_Z + fit.lift_height_m]), lift_tol, None)]
    for name, target, tol, step in phases:
        if name == "lift":
            # As the pick: the jaws now hold the piece, and the lift goes
            # straight up from where the grasp point is.
            env.hold_width(plan.grip_width_m)
            here = env.get_ee_pos()
            target = np.array([here[0], here[1], target[2]])
        rec = _move(env, name, target, tol, step, fit.rot, live)
        out["pick"].append(rec)
        if rec["ok"] and camera_check is not None and name == phases[-3][0]:
            out["board_inliers"] = camera_check()      # where the wrist servo would look from
        if not rec["ok"]:
            break

    if len(out["pick"]) == len(phases) and all(r["ok"] for r in out["pick"]):
        # A place leans as the piece was picked, where it can (ParallelJaw.place).
        lean = env.gripper_tilt_deg(fit.rot) if fit.rot is not None else None
        approach, rot, carry, _ = _quiet(POLICIES._place_hold, env, xy, plan, obstacles, lean)
        out["place"] = []
        if approach is None:
            out["place_approach"] = None
        else:
            out["place_approach"], out["carry_m"] = approach.name, round(carry, 4)
            out["place_tilt"] = _tilt_deg(rot)
            for name, target, tol, step in POLICIES.place_waypoints(
                    xy, dataclasses.replace(plan, lift_height_m=carry)):
                if name == "retreat":
                    _open(env, plan, live)
                rec = _move(env, name, target, tol, step, rot, live)
                out["place"].append(rec)
                if not rec["ok"]:
                    break
    _open(env, profile, live)
    return out


def verdict(sq: dict, profile, live: bool) -> "tuple[str, str]":
    """(status, why) for one square's record — see the module docstring."""
    if sq.get("approach") is None:
        why = "; ".join(sq.get("why_not", []))
        return "fail", (f"no approach reaches it — {why}" if why else "did not run")
    pick, place = sq.get("pick", []), sq.get("place", [])
    phases = pick + place
    bad = [r for r in phases if not r["ok"]]
    if sq.get("place_approach", "") is None:
        return "fail", "picked, but no approach reaches it to put a piece down"
    if bad or not place:
        r = bad[0] if bad else None
        return "fail", (f"{r['phase']} stopped {r['residual_mm']:.0f} mm short"
                        if r else "did not run")
    tilt = sq.get("pick_tilt")
    if sq.get("overlap_mm", 0.0) > 0:
        return "clash", (f"{tilt:.0f} deg puts the piece {sq['overlap_mm']:.0f} mm past a jaw, grasped at "
                         f"{sq['grasp_m'] * 1000:.0f} mm")
    if live and sq.get("servo") and sq.get("board_inliers", 0) < MIN_SERVO_INLIERS:
        return "blind", f"{sq.get('board_inliers', 0)} board corners at the servo's hover"
    servo_tilt = POLICIES.spec.servo_tilt_deg
    if sq.get("servo") and tilt is not None and servo_tilt is not None and abs(tilt - abs(servo_tilt)) > SERVO_LEAN_WARN_DEG:
        return "tilted", f"the wrist servo aligns at {tilt:.0f} deg; calibrated at {abs(servo_tilt):.0f}"
    worst = min(phases, key=lambda r: r["margin_deg"])
    if worst["margin_deg"] < MARGIN_WARN_DEG:
        return "marginal", f"{worst['tightest']} {worst['margin_deg']:.1f} deg from its limit at {worst['phase']}"
    return "ok", (f"{sq['approach']}, " + ("straight down" if not tilt or tilt < 0.5 else f"held at {tilt:.0f} deg"))


def _jaw_axis() -> str:
    from mrcf.backends.so101 import robot_config as rc
    return getattr(rc, "JAW_AXIS", "radial")


# ── Predicted (offline) ─────────────────────────────────────────────────────────

def _start_pose(name: str) -> dict:
    return cfg.CALIBRATION_PARK_DEGREES if name == "park" else cfg.HOME_DEGREES


def _predict_one(args) -> "tuple[str, dict]":
    """Worker: a fresh simulated arm at the start pose for one square."""
    sq_name, xy, piece, start = args
    import logging
    from tools.so101 import visualise_ik as vi
    logging.getLogger().setLevel(logging.ERROR)
    env = _quiet(vi.offline_arm, _start_pose(start))
    return sq_name, survey_square(env, xy, piece_plan(piece), live=False)


def predict(piece: str, squares: dict, names: list, start: str) -> dict:
    jobs = [(n, squares[n], piece, start) for n in names]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        return dict(pool.map(_predict_one, jobs))


# ── Live ────────────────────────────────────────────────────────────────────────

def live(piece: str, squares: dict, names: list, start: str, path: Path, header: dict) -> dict:
    import cv2
    from deployments.chess import piece_vision
    from tools.so101._common import make_env
    from mrcf.backends.so101 import robot_config as rc

    profile = piece_plan(piece)
    print(f"\nLive survey: {piece}, {len(names)} squares, starting each from {start}.")
    print(f"  hover {profile.hover_height_m:.3f} m, grasp {profile.grasp_height_m:.3f} m, "
          f"lift {profile.lift_height_m:.3f} m, jaws {profile.pregrasp_open_m * 1000:.0f} mm")
    print("  The jaws descend to grasp height on every square. CLEAR THE BOARD of pieces.")
    input("  Press Enter when the board is empty and the arm is clear...")

    frames = REPORTS / f"reach_frames_{piece}"
    frames.mkdir(parents=True, exist_ok=True)
    env = make_env()
    env.connect()
    results: dict = {}
    try:
        for i, name in enumerate(names, 1):
            if start == "park" and cfg.CALIBRATION_PARK_DEGREES:
                env.goto_joint_degrees(cfg.CALIBRATION_PARK_DEGREES)
            else:
                env.go_home()

            def camera_check(name=name):
                time.sleep(0.4)
                env.render_image(rc.CAMERA_NAME)          # drop a possibly stale frame
                frame = env.render_image(rc.CAMERA_NAME, width=rc.RENDER_W, height=rc.RENDER_H)
                privacy.imwrite(str(frames / f"{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                return piece_vision.board_inliers(frame)

            rec = survey_square(env, squares[name], profile, live=True, camera_check=camera_check)
            rec["status"], rec["why"] = verdict(rec, profile, live=True)
            results[name] = rec
            print(f"  [{i:2d}/{len(names)}] {name}: {rec['status']:<8} {rec['why']}")
            _save(path, header, results)
    except KeyboardInterrupt:
        print("\n  Interrupted — keeping what was measured.")
    finally:
        try:
            env.go_home()
        finally:
            env.stop()
    return results


# ── Output ──────────────────────────────────────────────────────────────────────

def _save(path: Path, header: dict, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**header, "squares": results}, indent=1))


_MARK = {"ok": " ", "marginal": "m", "tilted": "t", "blind": "B", "clash": "C"}
_LETTER = {"top_down": "T", "near_base": "N", "reach_out": "R"}


def _cell(sq: "dict | None") -> str:
    """Approach letter, held tilt and a status mark, e.g. 'T10 ', 'R45C', ' XX  '."""
    if sq is None:
        return "  ·  "
    if sq["status"] == "fail":
        return " XX  "
    t = sq.get("pick_tilt")
    return f"{_LETTER.get(sq.get('approach'), '?')}{(t or 0):<3.0f}{_MARK[sq['status']]}"


def print_grid(piece: str, results: dict) -> None:
    print(f"\n{piece}" + "".join(f"   {f}  " for f in FILES))
    for r in reversed(RANKS):
        print(f"  {r}  " + " ".join(_cell(results.get(f + r)) for f in FILES))
    counts = {s: sum(1 for v in results.values() if v["status"] == s) for s in ORDER}
    print("     " + "   ".join(f"{s} {n}" for s, n in counts.items() if n))
    used = {a: sum(1 for v in results.values() if v.get("approach") == a and v["status"] != "fail")
            for a in _LETTER}
    print("     " + "   ".join(f"{a} {n}" for a, n in used.items() if n))


def write_html(path: Path) -> None:
    runs = sorted(REPORTS.glob("reach_*.json"))
    if not runs:
        raise SystemExit(f"Nothing recorded in {REPORTS} yet.")
    order = {p: i for i, p in enumerate(PIECES)}
    data = [json.loads(p.read_text()) for p in runs]
    data.sort(key=lambda d: (order.get(d["piece"], 99), d["mode"]))

    def cell(name, sq):
        if sq is None:
            return f'<div class="c none"><b>{name}</b></div>'
        t = sq.get("pick_tilt")
        # A failed square's tilt is only what was planned, not what was reached.
        bits = [f"{_LETTER.get(sq.get('approach'), '')}{t:.0f}&deg;"
                if t is not None and sq["status"] != "fail" else ""]
        if "board_inliers" in sq:
            bits.append(f"{sq['board_inliers']}c")
        phases = "&#10;".join(
            f"{r['phase']}: {'ok' if r['ok'] else 'STALLED'} {r['residual_mm']:.0f} mm, "
            f"{r['tightest']} {r['margin_deg']:+.1f}&deg;"
            for r in sq.get("pick", []) + sq.get("place", []))
        title = (f"{name}: {sq['status']} — {sq['why']}&#10;approach: {sq.get('approach')}"
                 f"{' (wrist servo)' if sq.get('servo') else ''}, place: {sq.get('place_approach')}&#10;{phases}")
        return (f'<div class="c {sq["status"]}" title="{title}"><b>{name}</b>'
                f'<span>{" ".join(b for b in bits if b)}</span></div>')

    boards = []
    for d in data:
        sqs = d["squares"]
        grid = "".join(cell(f + r, sqs.get(f + r)) for r in reversed(RANKS) for f in FILES)
        counts = " · ".join(f"{sum(1 for v in sqs.values() if v['status'] == s)} {s}"
                            for s in ORDER if any(v["status"] == s for v in sqs.values()))
        p = d["profile"]
        boards.append(
            f'<section><h2>{d["piece"]} <small>{d["mode"]}</small></h2>'
            f'<p class="sub">hover {p["hover_height_m"] * 100:.1f} cm · grasp '
            f'{p["grasp_height_m"] * 100:.1f} cm · lift {p["lift_height_m"] * 100:.1f} cm · '
            f'jaws {p["pregrasp_open_m"] * 1000:.0f} mm {d.get("jaw_axis", "")} · from {d["start"]} · {d["time"]}</p>'
            f'<div class="board">{grid}</div><p class="sub">{counts}</p></section>')

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Board Reach Map</title><style>
:root{{--bg:#fbfbfa;--fg:#1c1c1a;--muted:#6b6b66;--line:#e3e3df;--card:#fff;
--ok:#3f8f5e;--marginal:#8aa84a;--tilted:#c08a2e;--blind:#7a6fb0;--clash:#c0632e;--fail:#b4493f;--none:#d9d9d4}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16161a;--fg:#ececea;
--muted:#9a9a94;--line:#2c2c32;--card:#1d1d22;--ok:#5fb87f;--marginal:#a3c264;--tilted:#d9a94e;
--blind:#9d92d6;--clash:#e08350;--fail:#d4695e;--none:#3a3a40}}}}
:root[data-theme="dark"]{{--bg:#16161a;--fg:#ececea;--muted:#9a9a94;--line:#2c2c32;--card:#1d1d22;
--ok:#5fb87f;--marginal:#a3c264;--tilted:#d9a94e;--blind:#9d92d6;--clash:#e08350;--fail:#d4695e;--none:#3a3a40}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;padding:28px 16px 64px}}
.wrap{{max-width:1100px;margin:0 auto}}h1{{font-size:1.5rem;margin:0 0 4px}}
h2{{font-size:1.05rem;margin:0 0 2px;text-transform:capitalize}}h2 small{{color:var(--muted);font-weight:500;text-transform:none}}
.sub{{color:var(--muted);margin:2px 0 8px;font-size:13px}}
.grid{{display:grid;gap:22px;grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}
.board{{display:grid;grid-template-columns:repeat(8,1fr);gap:2px}}
.c{{aspect-ratio:1;border-radius:4px;display:flex;flex-direction:column;justify-content:center;
align-items:center;font-size:11px;color:#fff;line-height:1.2}}.c b{{font-weight:600}}.c span{{opacity:.9;font-size:10px}}
.c.none{{background:var(--none);color:var(--muted)}}
.ok{{background:var(--ok)}}.marginal{{background:var(--marginal)}}.tilted{{background:var(--tilted)}}
.blind{{background:var(--blind)}}.clash{{background:var(--clash)}}.fail{{background:var(--fail)}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0 22px;font-size:13px;color:var(--muted)}}
.legend i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
</style></head><body><div class="wrap">
<h1>Board reach map</h1>
<p class="sub">Every square, a full pick and place, per piece, as the pick would come at it:
T top_down, N near_base, R reach_out. Rank 1 (bottom row) is nearest the robot. Hover a square
for each phase's residual and tightest joint. Cells show the approach and held tilt and, for live
runs, the board corners the gripper camera found where the servo looks from.</p>
<div class="legend"><span><i class="ok"></i>ok</span><span><i class="marginal"></i>joint near a limit</span>
<span><i class="tilted"></i>servo aligns far from its calibrated lean</span><span><i class="blind"></i>camera can't see enough board</span>
<span><i class="clash"></i>tilt leans piece into a jaw</span><span><i class="fail"></i>does not reach</span></div>
<div class="grid">{''.join(boards)}</div></div></body></html>"""
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path}")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--piece", choices=PIECES + ["all"], default="all")
    ap.add_argument("--squares", help="comma-separated, e.g. a1,h1,e8 (default: all 64)")
    ap.add_argument("--from", dest="start", choices=("home", "park"), default="home",
                    help="pose each square starts from (default home, as a command does)")
    ap.add_argument("--live", action="store_true", help="drive the real arm (empty board!)")
    ap.add_argument("--report", action="store_true", help="only write reports/reach_map.html")
    args = ap.parse_args()

    if args.report:
        write_html(REPORTS / "reach_map.html")
        return

    squares = load_squares()
    names = ([s.strip().lower() for s in args.squares.split(",") if s.strip()]
             if args.squares else [f + r for r in RANKS for f in FILES])
    unknown = [n for n in names if n not in squares]
    if unknown:
        raise SystemExit(f"Not in the board map: {', '.join(unknown)}")
    pieces = PIECES if args.piece == "all" else [args.piece]
    if args.live and len(pieces) > 1:
        raise SystemExit("--live surveys one piece at a time: pass --piece.")

    mode = "live" if args.live else "predicted"
    for piece in pieces:
        profile = piece_plan(piece)
        for problem in POLICIES.problems(profile):
            print(f"  [{piece}] profile problem: {problem}")
        header = {"mode": mode, "piece": piece, "start": args.start, "jaw_axis": _jaw_axis(),
                  "time": f"{datetime.now():%Y-%m-%d %H:%M}",
                  "profile": dict(profile.__dict__)}
        path = REPORTS / f"reach_{mode}_{piece}.json"
        if args.live:
            results = live(piece, squares, names, args.start, path, header)
        else:
            t0 = time.time()
            results = predict(piece, squares, names, args.start)
            for rec in results.values():
                rec["status"], rec["why"] = verdict(rec, profile, live=False)
            print(f"  [{piece}] predicted {len(results)} squares in {time.time() - t0:.0f} s")
            _save(path, header, results)
        print_grid(piece, results)
    print("\n  T/N/R = top_down / near_base / reach_out, then the held tilt; marks: m = joint near a "
          "limit · t = servo far from its calibrated lean · C = tilt leans the piece into a jaw · "
          "B = camera blind · XX = unreachable")
    write_html(REPORTS / "reach_map.html")


if __name__ == "__main__":
    main()
