"""
replay_descents.py — a pick's descents replayed on the SO-101's own
kinematics and its floor guard, its servos as the flight recorder has them,
before a change to how the arm moves goes onto the arm. Moves nothing,
connects to nothing.

Why: at the edge of its reach the arm's last few centimetres onto a grasp
depend on how its joints fall short of their commands under load — 2-3 deg,
one way more than the other — and on the day's runs a descent's outcome
flipped with 3 mm of grip height. One live run cannot tell a fix from luck;
a hundred replayed descents can. 2026-11-19 (runs 35-41, the tidy table): fitted
from the recorder's samples, the replay put the old descent at 72 of 110
arriving, the joint line plus the floor-kept push (DESCENT_LINE_ABOVE_M,
FLOOR_CAPS_PUSH) at 102, and a joint line all the way down at 110 — but
swinging the jaws 18 mm sideways near the object, as the real arm once did.

    python tools/so101/replay_descents.py fit [--since 2026-11-19]   # the servos, from the flight recorder
    python tools/so101/replay_descents.py cases [--deployment tidy]  # the moves logged in runs 37-40
    python tools/so101/replay_descents.py grid [--deployment tidy]   # 110 descents: the deployment's settings
                                                                     # against the arm's own

What it is not: a prediction of any one move — the bottom of a descent is
chaotic, and the replay matched about half of the logged moves one by one.
Compare rates over the grid, never single cases. And the servos are a fit to
the runs it was fitted from: all of 2026-11-19's put the elbow 1.8 deg short
going + and the wrist 1.0 going -, its afternoon's (runs 35-40, SERVO_BANDS)
2.8 and 2.8 — `fit --since` again before relying on a small difference.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.so101._common import rc  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# How far each joint stops short of its command (deg), moving + and moving -:
# the 75th percentile, far out and low, from the flight recorder's samples of
# runs 35-40 on 2026-11-19 (`fit` prints these again from newer runs).
SERVO_BANDS = {"shoulder_pan": (0.52, 0.44), "shoulder_lift": (1.55, 1.61), "elbow_flex": (2.77, 0.86),
               "wrist_flex": (0.40, 2.77), "wrist_roll": (0.41, 0.45)}
# ... and how far one moves in a 20 Hz tick, at most: its 90th-percentile speed, ~30 deg/s.
SERVO_STEP_DEG = 1.5
GAP_M = 0.036                                  # the jaws' opening the replayed descents carry


# ── The arm, offline ──────────────────────────────────────────────────────────

class _Bus:
    """The follower's bus: joints where the servo model puts them; nothing written."""

    def __init__(self, positions: dict):
        self.positions = dict(positions)
        self.is_connected = False

    def sync_read(self, _register, *a, **k) -> dict:
        return dict(self.positions)

    def write(self, *a, **k) -> None:
        pass

    def read(self, *a, **k):
        raise RuntimeError("the replay reads no servo registers")

    enable_torque = disable_torque = write


class _Follower:
    """lerobot's SO-101 follower, offline: each command moves the joints as SERVO_BANDS says the real ones do."""

    def __init__(self, degrees: dict, gripper_pct: float, bands=None, step_deg: float = SERVO_STEP_DEG):
        self.bus = _Bus({**degrees, "gripper": gripper_pct})
        self.cameras: dict = {}
        self.calibration: dict = {}
        self.bands, self.step_deg = dict(bands or SERVO_BANDS), float(step_deg)

    def send_action(self, action: dict) -> dict:
        for key, goal in action.items():
            name = key.removesuffix(".pos")
            if name not in self.bands:
                continue
            here = self.bus.positions[name]
            d = float(goal) - here
            short = self.bands[name][0] if d > 0 else self.bands[name][1]
            if abs(d) > short:
                self.bus.positions[name] = here + float(np.clip(d - np.sign(d) * short, -self.step_deg, self.step_deg))
        return action

    def connect(self, *a, **k):
        raise RuntimeError("the replay connects nothing")

    configure = disconnect = connect


def loaded(env, bands=None, step_deg: float = SERVO_STEP_DEG):
    """env (an SO101Env) with its follower replaced by the servo model: nothing reaches a real arm."""
    pct = env._width_to_pct(rc.PREGRASP_OPEN_M) if getattr(rc, "PREGRASP_OPEN_M", None) else 50.0
    env._robot = _Follower(rc.HOME_DEGREES, pct, bands, step_deg)
    env.hold = lambda *a, **k: None
    env._grasp_gap = lambda: GAP_M
    env.POS_REUSE_S = 0.0
    return env


def offline(deployment: "str | None" = None, bands=None):
    """
    The SO-101's env as a deployment runs it (its profile, its table: ENV_OVERRIDES), no camera opened, its joint
    ranges from the arm's own calibration, on the servo model. deployment None: the arm's own (robot_config DEPLOYMENT).
    """
    import logging

    from mrcf import backends, profiles
    from mrcf.backends.so101.kinematics import calibrated_degree_limits
    from tools.so101.visualise_ik import CAL_PATH

    logging.getLogger().setLevel(logging.ERROR)                 # lerobot warns on every clamped goal
    arm = backends.arm("so101")
    name = deployment or getattr(arm.config, "DEPLOYMENT", None)
    profile = profiles.load(ROOT / "deployments" / name) if name else None
    overrides: dict = {}
    if name:
        from tools._common import deployment_config
        overrides = dict(getattr(deployment_config(name), "ENV_OVERRIDES", None) or {})
    env = arm.make_env(profile=profile, camera_device_name=None, overhead_camera_device_name=None,
                       **{**overrides, "cameras": {}, "camera_name": None, "overhead_camera_name": None})
    if CAL_PATH.exists():
        env._joint_deg_limits = calibrated_degree_limits(json.loads(CAL_PATH.read_text()))
    env._setup_kinematics(rc.URDF_PATH, getattr(rc, "JOINT_OFFSETS_DEG", None),
                          tuple(getattr(rc, "IK_CALIBRATED_RANGE_JOINTS", ())))
    if env._kinematics is None:
        raise SystemExit("The SO-101's kinematics would not load (placo, its URDF: tools/so101/fetch_urdf.py).")
    return loaded(env, bands)


# ── One descent ───────────────────────────────────────────────────────────────

def descend(env, start, goal, lean_deg: float) -> "dict | None":
    """
    From over `start` (held at the lean nearest lean_deg that reaches both), down to `goal` as a pick's descent
    goes (3 deg steps, held orientation, 4 mm): {"arrived", "refused", "off_mm", "sway_mm", "lean"}. sway_mm: the
    furthest the jaws strayed sideways in the last 2 cm. None: no orientation reaches it, or no pose over start.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    got = env.approach_orientation_for([start, goal], prefer_tilt_deg=float(lean_deg))
    if got is None:
        return None
    rot = np.asarray(got[0], float)
    q = env._probe(rot, start, env._tcp(GAP_M), GAP_M, np.r_[env._q_arm_deg(), 0.0], 0.002, None)
    if q is None:
        return None
    for n, v in zip(JOINTS, q[:5]):
        env._robot.bus.positions[n] = float(v)
    path = []
    step = env.step

    def tick(n=1):
        step(n)
        here = np.array([env._robot.bus.positions[j] for j in JOINTS])
        path.append(env._grasp_point(np.append(here, 0.0), GAP_M))
    env.step = tick
    env.move_refusals = 0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ok = env.move_ee_to(goal, tol=0.004, max_step_deg=3.0, orientation=rot)
    finally:
        env.step = step
    here = np.array([env._robot.bus.positions[j] for j in JOINTS])
    near = [p for p in path if p[2] < goal[2] + 0.02]
    return {"arrived": bool(ok), "refused": env.move_refusals > 0, "lean": float(got[1]),
            "off_mm": float(np.linalg.norm(env._grasp_point(np.append(here, 0.0), GAP_M) - goal)) * 1000,
            "sway_mm": max((float(np.linalg.norm(p[:2] - goal[:2])) for p in near), default=0.0) * 1000}


# ── The moves logged in runs 37-40 (2026-11-19, the tidy table) ───────────────

S = 0.0007                                     # the bare table, as touched (rig/so101/tidy_surface.json)
CASES = [
    ("r39 cap -20 descent", (0.289, 0.147, S + 0.094), (0.289, 0.147, S + 0.019), -20, "floor refused"),
    ("r39 cap -15 descent", (0.289, 0.147, S + 0.094), (0.289, 0.147, S + 0.019), -15, "floor refused"),
    ("r40 cap -15 descent", (0.2665, 0.1456, 0.0903), (0.2665, 0.1456, 0.0167), -15, "floor refused"),
    ("r40 cap -25 descent", (0.258, 0.1424, 0.0947), (0.258, 0.1424, 0.0167), -25, "arrived"),
    ("r40 plug descent", (0.23626, 0.04231, 0.0947), (0.23626, 0.04231, 0.0184), -5, "arrived"),
    ("r40 ball descent", (0.30466, 0.06223, 0.0947), (0.30466, 0.06223, 0.0224), -20, "arrived"),
    ("r38 lying cap -20", (0.290, 0.142, S + 0.094), (0.290, 0.142, S + 0.016), -20, "floor refused"),
    ("r37 cap -15 descent", (0.30137, 0.10284, 0.0708), (0.30137, 0.10284, 0.02447), -15, "off its line"),
    ("r40 plug reseat 1", (0.24214, 0.04221, 0.0334), (0.24214, 0.04221, 0.0184), -5, "floor refused"),
    ("r40 plug reseat 2", (0.24955, 0.04341, 0.0334), (0.24955, 0.04341, 0.0184), -5, "floor refused"),
    ("r40 plug reseat 3", (0.2238, 0.05687, 0.0334), (0.2238, 0.05687, 0.0184), -5, "arrived"),
    ("r40 cap reseat", (0.28118, 0.14263, 0.0317), (0.28118, 0.14263, 0.0167), -25, "arrived"),
    ("r40 handle descent", (0.19647, 0.17074, 0.0947), (0.19647, 0.17074, 0.0277), -10, "arrived"),
]

# The grid: spots far and mid out, leans, grips over the table.
GRID_SPOTS = ((0.289, 0.147), (0.30466, 0.06223), (0.24214, 0.04221), (0.26, 0.12), (0.31, 0.0), (0.22, 0.10))
GRID_LEANS = (-10, -20, -25, -30)
GRID_GRIPS_M = (0.016, 0.019, 0.022, 0.025, 0.028)


def _said(r: "dict | None") -> str:
    if r is None:
        return "not reachable"
    what = "ARRIVED" if r["arrived"] else ("REFUSED" if r["refused"] else "stopped")
    return f"{what:8s} lean {r['lean']:+3.0f}  off {r['off_mm']:5.1f} mm  sway {r['sway_mm']:4.1f} mm"


def run_cases(deployment: "str | None") -> None:
    env = offline(deployment)
    print(f"The moves logged in runs 37-40, replayed as {deployment or 'the arm'} has it set now"
          " (\"then\": what the arm did, as that run had it set):")
    for label, start, goal, lean, live in CASES:
        print(f"  {label:22s} then: {live:14s} replay: {_said(descend(env, start, goal, lean))}")


def grid(env) -> dict:
    """Every grid descent: {"n", "arrived", "refused", "off_mm", "sway_mm", "missed": [(spot, lean, grip, said)]}."""
    out: dict = {"n": 0, "arrived": 0, "refused": 0, "off_mm": [], "sway_mm": [], "missed": []}
    for xy in GRID_SPOTS:
        for lean in GRID_LEANS:
            for grip in GRID_GRIPS_M:
                r = descend(env, (xy[0], xy[1], S + 0.094), (xy[0], xy[1], S + grip), lean)
                if r is None:
                    continue
                out["n"] += 1
                out["arrived"] += r["arrived"]
                out["refused"] += r["refused"]
                out["off_mm"].append(r["off_mm"])
                out["sway_mm"].append(r["sway_mm"])
                if not r["arrived"]:
                    out["missed"].append((xy, lean, grip, _said(r)))
    return out


def run_grid(deployment: "str | None") -> None:
    runs = ([(f"as {deployment} has it set", deployment)] if deployment else []) + [
        ("as the arm's own deployment has it", None)]
    for label, name in runs:
        print(f"\n{label}:", flush=True)
        g = grid(offline(name))
        off, sway = np.array(g["off_mm"] or [0.0]), np.array(g["sway_mm"] or [0.0])
        print(f"  {g['arrived']}/{g['n']} arrived, {g['refused']} refused by the floor guard, "
              f"{g['n'] - g['arrived'] - g['refused']} stopped short; off median {np.median(off):.1f} mm, "
              f"worst {off.max():.0f}; sideways near the object median {np.median(sway):.1f}, max {sway.max():.1f} mm")
        for xy, lean, grip, said in g["missed"]:
            print(f"    missed: ({xy[0]:.3f}, {xy[1]:.3f}) lean {lean:+d}, grip {grip * 1000:.0f} mm: {said}")


# ── The servos, from the flight recorder ──────────────────────────────────────

def fit(paths, kinematic_env=None, far_m: float = 0.28, low_m: float = 0.07) -> dict:
    """
    From flight-recorder samples ("sent" commands, "read" joints), how far each joint stops short of its command
    while it is held there (moving under 2 deg/s, 0.2-6 deg short), moving + and moving -: {joint: (p75 +, p75 -)}
    — far out and low only when kinematic_env (an SO101Env with kinematics) can say where the gripper was.
    """
    gaps: dict = {j: {+1: [], -1: []} for j in JOINTS}
    for path in paths:
        rows = sorted((json.loads(line) for line in open(path)), key=lambda r: r["t"])
        sent: dict = {}
        prev = None
        for r in rows:
            if r.get("stream") == "sent":
                sent.update({j: r[j] for j in JOINTS if j in r})
                continue
            if r.get("stream") != "read" or not all(j in r for j in JOINTS):
                continue
            if prev is not None and sent and 0.02 < r["t"] - prev["t"] < 0.1:
                if kinematic_env is not None:
                    g = kinematic_env._grasp_point(np.array([r[j] for j in JOINTS] + [0.0]), GAP_M)
                    if float(np.hypot(g[0], g[1])) <= far_m or g[2] >= low_m:
                        prev = r
                        continue
                dt = r["t"] - prev["t"]
                for j in JOINTS:
                    if j in sent:
                        gap = sent[j] - r[j]
                        if abs((r[j] - prev[j]) / dt) < 2.0 and 0.2 < abs(gap) < 6.0:
                            gaps[j][int(np.sign(gap))].append(abs(gap))
            prev = r
    return {j: tuple(round(float(np.percentile(gaps[j][d], 75)), 2) if gaps[j][d] else float("nan") for d in (+1, -1))
            for j in JOINTS}


def run_fit(since: "str | None") -> None:
    from mrcf.flight_recorder import ROOT as FLIGHTS

    stamp = (since or "").replace("-", "")
    paths = [p for p in sorted(glob.glob(str(FLIGHTS / "*" / "incidents" / "*" / "samples.jsonl")))
             if Path(p).parts[-4] >= stamp]
    print(f"{len(paths)} sample windows" + (f" since {since}" if since else ""))
    bands = fit(paths, offline())
    print("SERVO_BANDS = {" + ", ".join(f'"{j}": {b}' for j, b in bands.items()) + "}")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    time.sleep = lambda s: None                                 # type: ignore[assignment]  # nothing real to wait for

    def take(flag):
        if flag in argv:
            i = argv.index(flag)
            argv.pop(i)
            return argv.pop(i) if i < len(argv) else None
        return None
    deployment, since = take("--deployment"), take("--since")
    what = argv[0] if argv else ""
    if what == "fit":
        run_fit(since)
    elif what == "cases":
        run_cases(deployment)
    elif what == "grid":
        run_grid(deployment)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
