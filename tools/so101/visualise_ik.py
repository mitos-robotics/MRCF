"""
visualise_ik.py — inspect the IK layer and preview a plan, without moving the arm.

Everything here runs offline: it loads the URDF and your LeRobot calibration
file and drives the REAL SO101Env methods (approach_orientation, move_ee_to)
against a simulated servo that reproduces lerobot's per-cycle
max_relative_target clamp. No serial port, no cameras, nothing commanded.

Why drive the real methods rather than re-derive the maths: a preview that
reimplements the solver eventually disagrees with the solver. That exact bug
cost a long debugging session here — approach_orientation seeded its
reachability probe from the zero pose while move_ee_to's loop seeded from the
arm's measured pose, so the probe happily approved orientations the loop then
drove into a joint stop. A preview is only worth trusting if it is the same
code path.

Three views:

  limits     Per joint: the URDF limit, the calibrated limit, and which of the
             two actually binds. These are not the same, and the tighter one
             wins silently — wrist_flex is calibrated to +/-101 deg but the URDF
             caps it at +/-95, and placo enforces the URDF, so IK saturates at
             95 while the servo still has 6 deg in hand.

  workspace  Reachability across the table at each phase height, with the
             smallest gripper tilt that works. The straight-down reachable set
             shrinks as height increases — the ceiling that made every hover
             and lift stall is visible here at a glance.

  plan       A full pick (or pick-and-place) broken down per waypoint: the
             orientation chosen, the IK solution, the tightest joint margin,
             whether the motion loop converges, and the joint trajectory it
             would follow.

Usage:
    python tools/so101/visualise_ik.py
    python tools/so101/visualise_ik.py --workspace
    python tools/so101/visualise_ik.py --plan --square e5
    python tools/so101/visualise_ik.py --plan --point 0.261 0.029
    python tools/so101/visualise_ik.py --plan --square e5 --to d4
    python tools/so101/visualise_ik.py --all --html ik_report.html

--square reads board_squares.json, written beside board_overlay.png by the last
board scan. Without one, use --point with coordinates (the [Chess] scan line
prints them per piece).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from tools.so101._common import calibration_path, rc, task_config  # noqa: E402

# Hardware values come through the deployment config, which re-exports them;
# the heights a plan walks through are the deployment's own.
cfg = task_config()
from mrcf.backends.so101.kinematics import _TCP_OFFSET  # noqa: E402
from mrcf.backends.so101 import (  # noqa: E402
    JOINT_NAMES,
    SO101Env,
    calibrated_degree_limits,
    effective_ik_limits,
    urdf_degree_limits,
)

_OUT = rc.RIG_DIR             # this rig's folder (rig_state.py): the report goes there
CAL_PATH = calibration_path()
# Written beside the deployment's board overlay by its last scan.
SQUARES_PATH = Path(cfg.__file__).parent / "board_squares.json"


# ── Offline arm ────────────────────────────────────────────────────────────────

class _FakeBus:
    """Stands in for lerobot's motor bus: holds positions, no serial."""

    def __init__(self, positions: dict):
        self.positions = dict(positions)

    def sync_read(self, _data_name: str) -> dict:
        return dict(self.positions)


class _FakeFollower:
    """
    A perfect servo plus lerobot's own per-call max_relative_target clamp.

    Imports ensure_safe_goal_position from lerobot rather than reimplementing
    the clamp, so the simulated step size is exactly the real one.
    """

    def __init__(self, degrees: dict, max_relative_target: float):
        self.bus = _FakeBus({**degrees, "gripper": 50.0})
        self.max_relative_target = max_relative_target
        self.commanded: list[dict] = []   # every goal that was sent, in order
        self.measured: list[dict] = []    # resulting position after each send

    def send_action(self, action: dict) -> dict:
        from lerobot.robots.utils import ensure_safe_goal_position

        goal = {k.removesuffix(".pos"): v for k, v in action.items() if k.endswith(".pos")}
        present = self.bus.sync_read("Present_Position")
        pairs = {k: (v, present[k]) for k, v in goal.items() if k in present}
        safe = ensure_safe_goal_position(pairs, self.max_relative_target)
        self.commanded.append(dict(goal))
        self.bus.positions.update(safe)
        self.measured.append({n: self.bus.positions[n] for n in JOINT_NAMES})
        return safe


def offline_arm(start_degrees: dict | None = None) -> SO101Env:
    """
    A real SO101Env with the hardware replaced by _FakeFollower.

    Built via __new__ so none of the constructor's serial/camera work runs,
    then given exactly the attributes the kinematic methods use — derived
    through the same shared helpers the live constructor calls, so the preview
    cannot drift from the arm.
    """
    import logging

    # lerobot warns on every clamped goal. Offline that is the normal case —
    # the whole point is to watch the clamp work — and it buries the report.
    logging.getLogger().setLevel(logging.ERROR)

    if not CAL_PATH.exists():
        raise SystemExit(f"No calibration file at {CAL_PATH} — run lerobot-calibrate first.")
    if not Path(cfg.URDF_PATH).is_file():
        raise SystemExit(f"No URDF at {cfg.URDF_PATH} — run fetch_urdf.py first.")

    env = SO101Env.__new__(SO101Env)
    from mrcf.backends.so101 import robot_config as rc
    env._jaw_axis = getattr(rc, "JAW_AXIS", "radial")
    env._joint_deg_limits = calibrated_degree_limits(json.loads(CAL_PATH.read_text()))
    # The live constructor's own kinematics setup — joint corrections included.
    env._setup_kinematics(cfg.URDF_PATH, getattr(rc, "JOINT_OFFSETS_DEG", None),
                          tuple(getattr(rc, "IK_CALIBRATED_RANGE_JOINTS", ())))
    if env._kinematics is None:
        raise SystemExit("Kinematics unavailable — see the message above.")
    closed = getattr(rc, "JAW_CLOSED_POINT_M", None)
    env._tcp_offset = _TCP_OFFSET.copy() if closed is None else np.asarray(closed, float)
    env._jaw_centre_slide = float(getattr(rc, "JAW_CENTRE_SLIDE", 0.5))
    axis = np.asarray(getattr(rc, "JAW_SLIDE_AXIS", (-1.0, 0.0, 0.0)), float)
    env._jaw_slide_axis = axis / np.linalg.norm(axis)
    env._held_width_m = None
    env._bus_lock = threading.RLock()
    env._gripper_max_open_m = cfg.GRIPPER_MAX_OPEN_M
    table = getattr(rc, "GRIPPER_WIDTH_TABLE", None)
    env._gripper_width_table = sorted(table) if table else None
    env._work_plane_z = getattr(rc, "TABLE_Z", 0.0)
    env._control_hz = 1e9          # no real waiting offline
    env.POS_REUSE_S = 0.0          # and no time between readings: each one fresh
    env._max_relative_target_deg = cfg.MAX_RELATIVE_TARGET_DEG
    env._robot = _FakeFollower(start_degrees or cfg.HOME_DEGREES, cfg.MAX_RELATIVE_TARGET_DEG)
    # The grasp point depends on how far the jaws are open (see env._TCP_OFFSET),
    # so preview the opening a pick descends with rather than the fake bus's
    # arbitrary 50 %. Through the width table: the motor's % is not the opening.
    pregrasp = getattr(cfg, "PREGRASP_OPEN_M", None)
    if pregrasp is not None:
        env._robot.bus.positions["gripper"] = env._width_to_pct(pregrasp)
    return env


def limit_table(env: SO101Env) -> dict[str, dict]:
    """Per joint: urdf, calibrated, effective (the tighter), and which binds."""
    urdf = urdf_degree_limits(cfg.URDF_PATH)
    from mrcf.backends.so101 import robot_config as rc
    eff = effective_ik_limits(env._joint_deg_limits, urdf, tuple(getattr(rc, "IK_CALIBRATED_RANGE_JOINTS", ())))
    rows = {}
    for name in JOINT_NAMES:
        u, c = urdf.get(name), env._joint_deg_limits.get(name)
        if u is None or c is None:
            continue
        binds = "equal" if tuple(u) == tuple(c) else ("URDF" if eff[name] == tuple(u) else "calibration")
        rows[name] = {"urdf": u, "calibrated": c, "effective": eff[name], "binds": binds}
    return rows


# ── Probing ────────────────────────────────────────────────────────────────────

def probe_point(env: SO101Env, target: np.ndarray) -> dict:
    """
    What the arm would do at one target, using the real approach_orientation.

    Returns the chosen tilt, the IK solution, and the tightest joint margin —
    or reachable=False when no tried orientation works.
    """
    chosen = env.approach_orientation(target)
    if chosen is None:
        return {"target": target, "reachable": False}

    rot, tilt = chosen
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = target - rot @ env._tcp()
    q = np.zeros(len(env._kinematics.joint_names))
    q[: len(JOINT_NAMES)] = env._q_arm_deg()
    for _ in range(150):
        q = env._kinematics.inverse_kinematics(q, t)

    margins = {
        n: min(q[i] - env._ik_limits[n][0], env._ik_limits[n][1] - q[i])
        for i, n in enumerate(JOINT_NAMES) if n in env._ik_limits
    }
    tightest = min(margins, key=margins.get)
    return {
        "target": target,
        "reachable": True,
        "tilt": tilt,
        "q": {n: float(q[i]) for i, n in enumerate(JOINT_NAMES)},
        "margins": margins,
        "tightest": tightest,
        "tightest_margin": float(margins[tightest]),
        "residual_m": float(np.linalg.norm(env._grasp_point(q) - target)),
    }


def workspace_grid(env: SO101Env, z: float, step: float = 0.02) -> dict:
    """Probe a grid over WORKSPACE_BOUNDS at height z. Returns {(x,y): probe}."""
    b = cfg.WORKSPACE_BOUNDS
    xs = np.arange(b.x_min, b.x_max + 1e-9, step)
    ys = np.arange(b.y_min, b.y_max + 1e-9, step)
    out = {}
    for x in xs:
        for y in ys:
            out[(round(float(x), 4), round(float(y), 4))] = probe_point(
                env, np.array([x, y, cfg.TABLE_Z + z]))
    return out


def _policies():
    """The arm's own pick and place (mrcf/backends/so101 make_policies)."""
    from mrcf.backends.so101 import make_policies
    return make_policies()


def _plan_for(piece: "str | None"):
    """
    A grasp plan for `piece` — its size from the deployment's PIECE_SIZES_M
    when it has one (chess does) — else for the typical object the arm was
    tuned on (GRASP_HEIGHT_M, PREGRASP_OPEN_M).
    """
    from mrcf.policies.base import Target
    sizes = getattr(cfg, "PIECE_SIZES_M", {})
    if piece in sizes:
        height, width = sizes[piece]
        return _policies().plan(Target(np.zeros(3), foot_m=width, height_m=height))
    pol = _policies()
    # Sized so the policy plans exactly the tuned grasp: GRASP_HEIGHT_M up an
    # object of the height that makes that its usual fraction, PREGRASP_OPEN_M open.
    return pol.plan(Target(np.zeros(3), foot_m=cfg.PREGRASP_OPEN_M - 2 * pol.spec.jaw_slack_m,
                           height_m=cfg.GRASP_HEIGHT_M / pol.spec.grasp_height_fraction))


def plan_waypoints(xy: tuple[float, float], kind: str = "pick",
                   piece: str | None = None) -> list[tuple[str, np.ndarray, float, float | None]]:
    """
    The waypoints the arm's pick / place actually drive, in order, each with
    the SAME move_ee_to tolerance and step limit the policy passes.

    The tolerance is part of the waypoint, not a detail: it decides whether a
    given residual counts as converged, so a preview using a looser one would
    report success exactly where the pick reports failure.
    """
    pol, plan = _policies(), _plan_for(piece)
    return pol.pick_waypoints(xy, plan) if kind == "pick" else pol.place_waypoints(xy, plan)


def simulate_plan(xy: tuple[float, float], kind: str = "pick",
                  piece: str | None = None) -> list[dict]:
    """
    Run the real move_ee_to loop for each waypoint against the simulated servo,
    holding one orientation through the sequence as the skills do.

    Each step reports what the motion loop did, not what a separate model
    predicts it would do.
    """
    env = offline_arm()
    waypoints = plan_waypoints(xy, kind, piece)
    rot = _policies().hold_orientation(env, [(n, t, tol) for n, t, tol, _ in waypoints], "[plan]")
    steps = []
    for name, target, tol, step in waypoints:
        before = len(env._robot.measured)
        probe = probe_point(env, target)
        converged = env.move_ee_to(target, tol=tol, max_step_deg=step, orientation=rot)
        traj = env._robot.measured[before:]
        steps.append({
            "phase": name,
            "target": target,
            "tol": tol,
            "probe": probe,
            "converged": bool(converged),
            "residual_m": float(np.linalg.norm(env.get_ee_pos() - target)),
            "iterations": len(traj),
            "trajectory": traj,
            "final_q": {n: float(v) for n, v in zip(JOINT_NAMES, env._q_arm_deg())},
        })
    return steps


def load_squares() -> dict:
    if not SQUARES_PATH.exists():
        return {}
    try:
        return json.loads(SQUARES_PATH.read_text())
    except (OSError, ValueError):
        return {}


def resolve_target(args) -> tuple[tuple[float, float], str]:
    if args.point:
        return (float(args.point[0]), float(args.point[1])), f"({args.point[0]}, {args.point[1]})"
    squares = load_squares()
    if not squares:
        raise SystemExit(
            f"No {SQUARES_PATH.name} yet — it is written by the next board scan.\n"
            "Until then pass coordinates directly, e.g.  --point 0.261 0.029")
    sq = args.square.lower()
    if sq not in squares:
        raise SystemExit(f"Unknown square '{sq}'. Known: {', '.join(sorted(squares))}")
    w = squares[sq]["world"]
    return (float(w[0]), float(w[1])), sq


# ── Terminal views ─────────────────────────────────────────────────────────────

def print_limits(env: SO101Env) -> None:
    print("\nJoint limits — URDF vs calibration\n")
    print(f"  {'joint':<15}{'URDF':>18}{'calibrated':>18}{'effective':>18}   binds")
    print("  " + "-" * 76)
    for name, r in limit_table(env).items():
        f = lambda t: f"[{t[0]:+6.1f},{t[1]:+6.1f}]"  # noqa: E731
        print(f"  {name:<15}{f(r['urdf']):>18}{f(r['calibrated']):>18}"
              f"{f(r['effective']):>18}   {r['binds']}")
    print("\n  The tighter of the two wins, silently. placo enforces the URDF during IK,")
    print("  while SO101Env clamps to the calibration before sending — so a joint can")
    print("  saturate in the solver while the servo still has travel left, or vice versa.")


def print_workspace(env: SO101Env, heights: list[tuple[str, float]], step: float = 0.02) -> None:
    b = cfg.WORKSPACE_BOUNDS
    ys = np.arange(b.y_min, b.y_max + 1e-9, step)
    for label, z in heights:
        grid = workspace_grid(env, z, step)
        print(f"\n{label} — z = TABLE_Z + {z:.3f} m")
        header = "x\\y"
        print(f"  {header:>6} " + "".join(f"{y:+5.2f}" for y in ys))
        for x in np.arange(b.x_min, b.x_max + 1e-9, step):
            cells = []
            for y in ys:
                p = grid[(round(float(x), 4), round(float(y), 4))]
                if not p["reachable"]:
                    cells.append("    ·")
                elif abs(p["tilt"]) < 0.5:
                    cells.append("    #")
                else:
                    cells.append(f"{abs(p['tilt']):5.0f}")
            print(f"  {x:6.2f} " + "".join(cells))
    print("\n  # = straight down   N = smallest tilt (deg from vertical) that reaches")
    print("  · = no tried orientation reaches it inside the effective limits")


def print_plan(xy: tuple[float, float], label: str, kind: str, piece: str | None = None) -> None:
    steps = simulate_plan(xy, kind, piece)
    print(f"\n{kind} at {label} — target (x={xy[0]:.3f}, y={xy[1]:.3f})\n")
    print(f"  {'phase':<9}{'z':>7}{'tilt':>7}{'conv':>7}{'resid':>8}{'tol':>7}{'iters':>7}"
          f"   {'tightest joint margin':<28}")
    print("  " + "-" * 89)
    for s in steps:
        p = s["probe"]
        tilt = "n/a" if not p["reachable"] else ("down" if abs(p["tilt"]) < 0.5 else f"{p['tilt']:+.0f}")
        tight = ("unreachable" if not p["reachable"]
                 else f"{p['tightest']} {p['tightest_margin']:+.1f} deg")
        warn = "" if (s["converged"] and p["reachable"] and p["tightest_margin"] > 2) else "  <-"
        print(f"  {s['phase']:<9}{s['target'][2]:7.3f}{tilt:>7}"
              f"{'yes' if s['converged'] else 'NO':>7}"
              f"{s['residual_m'] * 100:6.1f}cm{s['tol'] * 100:6.1f}cm"
              f"{s['iterations']:7d}   {tight:<28}{warn}")
    print("\n  Final joint angles per phase (deg):")
    print(f"    {'phase':<9}" + "".join(f"{n:>15}" for n in JOINT_NAMES))
    for s in steps:
        print(f"    {s['phase']:<9}" + "".join(f"{s['final_q'][n]:15.1f}" for n in JOINT_NAMES))
    bad = [s["phase"] for s in steps if not s["converged"]]
    if bad:
        print(f"\n  WOULD NOT CONVERGE: {', '.join(bad)} — the arm stops short here.")
    else:
        print("\n  All waypoints converge.")


# ── SVG / HTML report ──────────────────────────────────────────────────────────

def _svg_workspace(env: SO101Env, label: str, z: float, squares: dict, step: float = 0.02) -> str:
    """Top-down reachability map: +x right, +y up, base at the origin marker."""
    b = cfg.WORKSPACE_BOUNDS
    grid = workspace_grid(env, z, step)
    pad, scale = 46, 760.0 / max(b.x_max - b.x_min, b.y_max - b.y_min)
    W = (b.x_max - b.x_min) * scale + pad * 2
    H = (b.y_max - b.y_min) * scale + pad * 2
    sx = lambda x: pad + (x - b.x_min) * scale          # noqa: E731
    sy = lambda y: H - pad - (y - b.y_min) * scale      # noqa: E731

    parts = [f'<svg viewBox="0 0 {W:.0f} {H:.0f}" class="map" role="img" '
             f'aria-label="Reachability at {label}">']
    cell = step * scale
    for (x, y), p in grid.items():
        if not p["reachable"]:
            fill, op = "var(--bad)", 0.5
        else:
            t = min(abs(p["tilt"]), 60.0) / 60.0
            fill, op = ("var(--good)" if t == 0 else "var(--warn)"), 0.30 + 0.5 * t
        parts.append(f'<rect x="{sx(x) - cell/2:.1f}" y="{sy(y) - cell/2:.1f}" '
                     f'width="{cell:.1f}" height="{cell:.1f}" fill="{fill}" '
                     f'opacity="{op:.2f}"/>')
    for name, sq in squares.items():
        wx, wy = sq["world"]
        if b.x_min <= wx <= b.x_max and b.y_min <= wy <= b.y_max:
            parts.append(f'<circle cx="{sx(wx):.1f}" cy="{sy(wy):.1f}" r="3" class="sq"/>')
            parts.append(f'<text x="{sx(wx):.1f}" y="{sy(wy) - 6:.1f}" class="sqlab">{name}</text>')
    parts.append(f'<circle cx="{sx(0.0):.1f}" cy="{sy(0.0):.1f}" r="6" class="base"/>')
    parts.append(f'<text x="{sx(0.0) + 10:.1f}" y="{sy(0.0) + 4:.1f}" class="axis">base</text>')
    parts.append(f'<text x="{W/2:.0f}" y="{H - 12:.0f}" class="axis" '
                 f'text-anchor="middle">+x toward the table (m)</text>')
    parts.append(f'<text x="14" y="{H/2:.0f}" class="axis" '
                 f'transform="rotate(-90 14 {H/2:.0f})" text-anchor="middle">+y (m)</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_trajectory(steps: list[dict]) -> str:
    """Joint angle vs command cycle, across the whole plan, with limit bands."""
    traj, marks, n = [], [], 0
    for s in steps:
        traj.extend(s["trajectory"])
        n += len(s["trajectory"])
        marks.append((n, s["phase"]))
    if not traj:
        return "<p class='muted'>No motion recorded.</p>"
    W, H, pad = 860, 300, 46
    lo = min(min(p.values()) for p in traj)
    hi = max(max(p.values()) for p in traj)
    lo, hi = min(lo, -10) - 5, max(hi, 10) + 5
    px = lambda i: pad + i * (W - 2 * pad) / max(len(traj) - 1, 1)    # noqa: E731
    py = lambda v: H - pad - (v - lo) * (H - 2 * pad) / (hi - lo)     # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" class="map" role="img" aria-label="Joint trajectory">']
    out.append(f'<line x1="{pad}" y1="{py(0):.1f}" x2="{W - pad}" y2="{py(0):.1f}" class="grid"/>')
    for at, _name in marks[:-1]:
        out.append(f'<line x1="{px(at):.1f}" y1="{pad}" x2="{px(at):.1f}" y2="{H - pad}" class="grid"/>')
    for at, name in marks:
        out.append(f'<text x="{px(at - 1):.1f}" y="{pad - 8}" class="axis" '
                   f'text-anchor="end">{name}</text>')
    for k, jname in enumerate(JOINT_NAMES):
        pts = " ".join(f"{px(i):.1f},{py(p[jname]):.1f}" for i, p in enumerate(traj))
        out.append(f'<polyline points="{pts}" class="j j{k}" fill="none"/>')
    out.append(f'<text x="{pad - 8}" y="{py(hi) + 10:.1f}" class="axis" '
               f'text-anchor="end">{hi:.0f}&#176;</text>')
    out.append(f'<text x="{pad - 8}" y="{py(lo):.1f}" class="axis" '
               f'text-anchor="end">{lo:.0f}&#176;</text>')
    out.append("</svg>")
    out.append('<p class="legend">' + "".join(
        f'<span class="key j{k}">{n}</span>' for k, n in enumerate(JOINT_NAMES)) + "</p>")
    return "".join(out)


_CSS = """
:root{--bg:#fbfbfa;--fg:#1c1c1a;--muted:#6b6b66;--line:#e3e3df;--card:#fff;
--good:#3f8f5e;--warn:#c08a2e;--bad:#b4493f;--accent:#2f6f9f;
--j0:#2f6f9f;--j1:#b4493f;--j2:#3f8f5e;--j3:#8a5bb8;--j4:#c08a2e;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#16161a;--fg:#ececea;--muted:#9a9a94;--line:#2c2c32;--card:#1d1d22;
--good:#5fb87f;--warn:#d9a94e;--bad:#d4695e;--accent:#6aa9d8;
--j0:#6aa9d8;--j1:#d4695e;--j2:#5fb87f;--j3:#a981d6;--j4:#d9a94e;}}
:root[data-theme="dark"]{--bg:#16161a;--fg:#ececea;--muted:#9a9a94;--line:#2c2c32;
--card:#1d1d22;--good:#5fb87f;--warn:#d9a94e;--bad:#d4695e;--accent:#6aa9d8;
--j0:#6aa9d8;--j1:#d4695e;--j2:#5fb87f;--j3:#a981d6;--j4:#d9a94e;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,-apple-system,
"Segoe UI",Roboto,sans-serif;padding:32px 16px 72px}
.wrap{max-width:920px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:1.1rem;margin:36px 0 10px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{text-align:right;padding:7px 9px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.map{width:100%;height:auto;display:block;background:transparent}
.grid{stroke:var(--line);stroke-width:1}
.axis{fill:var(--muted);font-size:12px;font-family:ui-sans-serif,sans-serif}
.sq{fill:none;stroke:var(--accent);stroke-width:1.2}
.sqlab{fill:var(--muted);font-size:9px;text-anchor:middle}
.base{fill:var(--fg)}
.j{stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.j0{stroke:var(--j0)}.j1{stroke:var(--j1)}.j2{stroke:var(--j2)}
.j3{stroke:var(--j3)}.j4{stroke:var(--j4)}
.legend{margin:10px 0 0;display:flex;gap:14px;flex-wrap:wrap}
.key{font-size:12px;color:var(--muted);display:inline-flex;align-items:center;gap:6px}
.key::before{content:"";width:14px;height:3px;border-radius:2px;background:currentColor}
.key.j0{color:var(--j0)}.key.j1{color:var(--j1)}.key.j2{color:var(--j2)}
.key.j3{color:var(--j3)}.key.j4{color:var(--j4)}
.ok{color:var(--good)}.warnc{color:var(--warn)}.badc{color:var(--bad)}
.muted{color:var(--muted)}
.swatches{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px}
.sw{font-size:12px;color:var(--muted);display:inline-flex;align-items:center;gap:6px}
.sw i{width:12px;height:12px;border-radius:3px;display:inline-block}
.grids{display:grid;gap:18px;grid-template-columns:1fr}
@media(min-width:760px){.grids{grid-template-columns:repeat(2,1fr)}}
"""


def write_html(path: Path, env: SO101Env, heights: list[tuple[str, float]],
               plan: tuple[list[dict], str, str] | None) -> None:
    squares = load_squares()
    lim_rows = "".join(
        f"<tr><td>{n}</td><td class='mono'>{r['urdf'][0]:+.1f} / {r['urdf'][1]:+.1f}</td>"
        f"<td class='mono'>{r['calibrated'][0]:+.1f} / {r['calibrated'][1]:+.1f}</td>"
        f"<td class='mono'>{r['effective'][0]:+.1f} / {r['effective'][1]:+.1f}</td>"
        f"<td>{r['binds']}</td></tr>"
        for n, r in limit_table(env).items())

    maps = "".join(
        f"<div><h3 class='sub mono'>{label} — z = {z:.3f} m</h3>"
        f"{_svg_workspace(env, label, z, squares)}</div>"
        for label, z in heights)

    plan_html = ""
    if plan is not None:
        steps, label, kind = plan
        rows = ""
        for s in steps:
            p = s["probe"]
            tilt = "—" if not p["reachable"] else ("down" if abs(p["tilt"]) < 0.5 else f"{p['tilt']:+.0f}&#176;")
            if not p["reachable"]:
                tight, cls = "unreachable", "badc"
            else:
                m = p["tightest_margin"]
                cls = "ok" if m > 2 else ("warnc" if m > 0 else "badc")
                tight = f"{p['tightest']} {m:+.1f}&#176;"
            conv = ("<span class='ok'>yes</span>" if s["converged"]
                    else "<span class='badc'>NO</span>")
            rows += (f"<tr><td>{s['phase']}</td><td class='mono'>{s['target'][2]:.3f}</td>"
                     f"<td class='mono'>{tilt}</td><td>{conv}</td>"
                     f"<td class='mono'>{s['residual_m']*100:.1f} / {s['tol']*100:.1f} cm</td>"
                     f"<td class='mono'>{s['iterations']}</td>"
                     f"<td class='{cls} mono'>{tight}</td></tr>")
        plan_html = f"""
<h2>Plan — {kind} at {label}</h2>
<div class="card"><table>
<thead><tr><th>phase</th><th>z (m)</th><th>tilt</th><th>converges</th>
<th>residual / tol</th><th>cycles</th><th>tightest joint margin</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h2>Joint trajectory</h2>
<p class="sub">What the motion loop actually commands, cycle by cycle, with lerobot's
per-cycle clamp applied.</p>
<div class="card">{_svg_trajectory(steps)}</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 IK Inspector</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>SO-101 IK Inspector</h1>
<p class="sub">Offline preview — URDF + calibration, driving the real
<code>approach_orientation</code> and <code>move_ee_to</code>. Nothing was commanded.</p>

<h2>Joint limits</h2>
<p class="sub">The tighter of URDF and calibration wins, silently. placo enforces the
URDF during IK; SO101Env clamps to the calibration before sending.</p>
<div class="card"><table>
<thead><tr><th>joint</th><th>URDF</th><th>calibrated</th><th>effective</th><th>binds</th></tr></thead>
<tbody>{lim_rows}</tbody></table></div>

<h2>Reachability</h2>
<p class="sub">Smallest gripper tilt from vertical that reaches each point, per phase height.
Circles mark board squares from the last scan.</p>
<div class="swatches">
<span class="sw"><i style="background:var(--good)"></i>straight down</span>
<span class="sw"><i style="background:var(--warn)"></i>needs a tilt (darker = more)</span>
<span class="sw"><i style="background:var(--bad)"></i>unreachable</span></div>
<div class="grids">{maps}</div>
{plan_html}
</div></body></html>"""
    path.write_text(html, encoding="utf-8")
    print(f"\nWrote {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limits", action="store_true", help="joint limits: URDF vs calibration")
    ap.add_argument("--workspace", action="store_true", help="reachability map per phase height")
    ap.add_argument("--plan", action="store_true", help="simulate a pick at --square/--point")
    ap.add_argument("--all", action="store_true", help="every view")
    ap.add_argument("--square", metavar="NAME", help="board square, e.g. e5 (needs board_squares.json)")
    ap.add_argument("--point", nargs=2, metavar=("X", "Y"), help="target in metres, base frame")
    ap.add_argument("--to", metavar="NAME", help="also simulate a place at this square")
    ap.add_argument("--piece", metavar="TYPE",
                    help="piece whose heights to preview, e.g. king, from the deployment's "
                         "PIECE_SIZES_M (default: the small object the arm was tuned on)")
    ap.add_argument("--step", type=float, default=0.02, help="workspace grid step (m), default 0.02")
    ap.add_argument("--html", nargs="?", const="ik_report.html", metavar="PATH",
                    help="also write an HTML report (default ik_report.html, in rig/so101/)")
    args = ap.parse_args()

    if not any([args.limits, args.workspace, args.plan, args.all]):
        args.limits = args.workspace = True   # default view

    env = offline_arm()
    heights = [("hover", cfg.HOVER_HEIGHT),
               ("grasp", cfg.GRASP_HEIGHT_M),
               ("lift",  cfg.LIFT_HEIGHT)]

    if args.limits or args.all:
        print_limits(env)
    if args.workspace or args.all:
        print_workspace(env, heights, args.step)

    plan = None
    if args.plan or args.all:
        if not (args.square or args.point):
            if args.all:
                print("\n(skipping plan view — pass --square or --point to simulate one)")
            else:
                raise SystemExit("--plan needs --square NAME or --point X Y")
        else:
            xy, label = resolve_target(args)
            print_plan(xy, label, "pick", args.piece)
            plan = (simulate_plan(xy, "pick", args.piece), label, "pick")
            if args.to:
                squares = load_squares()
                if args.to.lower() not in squares:
                    raise SystemExit(f"Unknown square '{args.to}' for --to")
                w = squares[args.to.lower()]["world"]
                print_plan((float(w[0]), float(w[1])), args.to.lower(), "place", args.piece)

    if args.html:
        write_html(Path(args.html) if Path(args.html).is_absolute() else _OUT / args.html,
                   env, heights, plan)


if __name__ == "__main__":
    main()
