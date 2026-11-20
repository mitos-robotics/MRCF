"""
arm_baselines.py — what each arm comes to as the shared code builds it, so
that a change made for one arm cannot change another without a test saying so
(tests/test_arm_baselines.py compares against tests/baselines/<arm>/).

The arms share most of the framework: the policies, RigEnv, wiring.py, the
GripperSpec's defaults. The shared code's own tests run on the test arm
(fake_arm.py), which is neither real arm, so a change tuned on the UR5 could
move the SO-101 with every test passing. What this records, per arm:

  settings   its GripperSpec as its make_policies() builds it, and the keywords
             wiring.py hands its make_env() (rig_kwargs, camera_kwargs)
  env        its env, made offline as its make_env() makes it: the capabilities
             it has (capabilities.py), its class's constants, the state it
             starts in
  plans      standard scenes laid out from its own workspace — a lone object
             near, mid-reach and far from the base, unmeasured, tall, a block,
             one standing on another, one too wide, one out of reach, a
             neighbour beside it, neighbours all round, a nudge — each planned
             by its own policy with nothing moved: the grasp, the room, the
             approaches it weighs and why the others cannot be done, the
             waypoints

Nothing connects and nothing moves. The UR5 and the template are made in
memory, as their make_env() makes them with no controller; the SO-101 by its
make_env() with its follower replaced by one that refuses every command
(_StillFollower), which needs lerobot, placo and its URDF — without them its
env and plans are skipped. The SO-101's joint limits are the ones below
rather than the calibration file's, which only its rig has; its cameras
send (a frame always just taken) but hand out no picture, so it plans as
the rig does with its wrist camera working, and a plan that looks fails.
The outline segmenter is taken as there (perception.segment.available), as
on the rigs: the real check loads its model, and downloads it the first time.

Every arm is recorded in a fresh process, MRCF_HOME an empty folder (bar the
downloads, DOWNLOADS) and no lerobot calibration: no rig's saved
calibration (rig/<arm>/) reaches what is recorded, so every machine records
the same.

    python tests/arm_baselines.py ur5            # print what the UR5 records
    python tests/arm_baselines.py ur5 --diff     # ... only how it differs from its baseline
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))           # run as a script, from anywhere
BASELINES = Path(__file__).resolve().parent / "baselines"
SECTIONS = ("settings", "env", "plans", "deployments")

# What a rig folder holds that is downloaded, not measured: the same on every
# machine, so linked into the empty MRCF_HOME (the SO-101's URDF: fetch_urdf.py).
DOWNLOADS = ("urdf",)

# The SO-101's joint limits, degrees, as its calibration file gave them on the
# rig (2026-11-14). Fixed here so its plans are the same on every machine; a
# rig's own file decides on the rig.
SO101_JOINT_LIMITS_DEG = {
    "shoulder_pan": (-72.1, 72.1),
    "shoulder_lift": (-96.0, 96.0),
    "elbow_flex": (-92.1, 92.1),
    "wrist_flex": (-101.2, 101.2),
    "wrist_roll": (-180.0, 180.0),
}

# Environment variables that point an arm at real hardware or a rig's files.
_CLEARED = ("MRCF_ARM", "MRCF_UR5_HOST", "SO101_FOLLOWER_PORT", "SO101_FOLLOWER_ID", "MRCF_URSIM_HOST")


class Skipped(Exception):
    """A section this machine cannot record (a library or a download missing)."""


# ── Recording, from the test or the command line ──────────────────────────────

def record(arm: str, timeout_s: float = 300.0) -> dict:
    """
    What `arm` records, {section: data}, made in a fresh process with no rig's
    files and no hardware. A section that cannot be made here is
    {"skipped": why}.
    """
    from mrcf import rig_state

    with tempfile.TemporaryDirectory(prefix="mrcf-baseline-") as tmp:
        home = Path(tmp) / "home"
        for name in DOWNLOADS:
            have = rig_state.home() / "rig" / arm / name
            if have.exists():
                dest = home / "rig" / arm / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dest.symlink_to(have, target_is_directory=have.is_dir())
                except OSError:                 # Windows without the right to link
                    (shutil.copytree if have.is_dir() else shutil.copy)(have, dest)
        env = {k: v for k, v in os.environ.items() if k not in _CLEARED}
        env.update(MRCF_HOME=str(home), HF_LEROBOT_CALIBRATION=str(Path(tmp) / "calibration"),
                   PYTHONPATH=os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
                   PYTHONHASHSEED="0")
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), arm, "--inside"],
                             env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout_s)
        if run.returncode != 0:
            raise RuntimeError(f"Recording the {arm} failed:\n{run.stderr[-4000:]}")
        data = json.loads(run.stdout)
        return _replace_paths(data, {str(home): "<MRCF_HOME>", str(ROOT): "<repo>", tmp: "<tmp>"})


def path(arm: str, section: str) -> Path:
    return BASELINES / arm / f"{section}.json"


def load(arm: str, section: str) -> "dict | None":
    p = path(arm, section)
    return json.loads(p.read_text()) if p.is_file() else None


def save(arm: str, section: str, data) -> None:
    p = path(arm, section)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def _replace_paths(value, prefixes: dict):
    if isinstance(value, str):
        for prefix, name in sorted(prefixes.items(), key=lambda kv: -len(kv[0])):
            value = value.replace(prefix, name)
        return value.replace("\\", "/") if "<" in value else value
    if isinstance(value, dict):
        return {k: _replace_paths(v, prefixes) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_paths(v, prefixes) for v in value]
    return value


# ── Comparing ─────────────────────────────────────────────────────────────────

# How far a number may drift and still be the same: the settings and the env
# are copied values (only a float's last bits); the plans come out of IK
# solvers (placo, the UR5's numerical solve) whose last digits differ by
# platform. Plans are recorded in mm and degrees.
TOLERANCE = {"settings": 1e-9, "env": 1e-9, "plans": 0.05, "deployments": 1e-9}
# A number inside a message ("the jaws have 23 mm here") is rounded where the
# message is written: one either side of a rounding edge moves by 1.
MESSAGE_TOLERANCE = 1.0

_NUMBER = r"[-+]?\d+(?:\.\d+)?"


def differences(want, got, tol: float, where: str = "") -> list:
    """Every place got differs from want, as 'where: want -> got' lines."""
    import re

    out: list = []
    if isinstance(want, dict) and isinstance(got, dict):
        for k in sorted(set(want) | set(got)):
            here = f"{where}.{k}" if where else str(k)
            if k not in got:
                out.append(f"{here}: {_short(want[k])} -> (gone)")
            elif k not in want:
                out.append(f"{here}: (new) -> {_short(got[k])}")
            else:
                out += differences(want[k], got[k], tol, here)
        return out
    if isinstance(want, list) and isinstance(got, list):
        if want != got and want and got and all(isinstance(x, str) for x in want + got):
            # Names (capabilities, approaches): which came and which went.
            added, gone = [x for x in got if x not in want], [x for x in want if x not in got]
            if added or gone:
                return [f"{where}: " + "; ".join(([f"added {', '.join(added)}"] if added else [])
                                                 + ([f"gone {', '.join(gone)}"] if gone else []))]
            return [f"{where}: reordered {_short(want)} -> {_short(got)}"]
        if len(want) != len(got):
            return [f"{where}: {_short(want)} -> {_short(got)}"]
        for i, (w, g) in enumerate(zip(want, got)):
            out += differences(w, g, tol, f"{where}[{i}]")
        return out
    if isinstance(want, bool) or isinstance(got, bool):            # True == 1 to Python, not here
        return [] if type(want) is type(got) and want == got else [f"{where}: {want} -> {got}"]
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        scale = max(1.0, abs(want), abs(got)) if tol < 1e-6 else 1.0
        return [] if abs(want - got) <= tol * scale else [f"{where}: {want} -> {got}"]
    if isinstance(want, str) and isinstance(got, str) and want != got:
        split = re.compile(f"({_NUMBER})")
        w, g = split.split(want), split.split(got)
        same = len(w) == len(g) and all(
            (a == b) if i % 2 == 0 else abs(float(a) - float(b)) <= max(tol, MESSAGE_TOLERANCE)
            for i, (a, b) in enumerate(zip(w, g)))
        return [] if same else [f"{where}: {want!r} -> {got!r}"]
    return [] if want == got else [f"{where}: {_short(want)} -> {_short(got)}"]


def _short(value, n: int = 160) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= n else text[:n - 3] + "..."


# ── Plain data ────────────────────────────────────────────────────────────────

def plain(value, depth: int = 0):
    """value as JSON: numbers to 9 significant digits, arrays as lists, dataclasses as dicts, other objects by type."""
    if depth > 12:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return str(v) if not math.isfinite(v) else float(f"{v:.9g}")
    if isinstance(value, np.ndarray):
        return plain(value.tolist(), depth + 1)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"__type__": type(value).__name__,
                **{f.name: plain(getattr(value, f.name), depth + 1) for f in dataclasses.fields(value)}}
    if isinstance(value, dict):
        return {str(k): plain(v, depth + 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [plain(v, depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((plain(v, depth + 1) for v in value), key=str)
    if callable(value):
        return f"<function {getattr(value, '__qualname__', type(value).__name__)}>"
    return f"<{type(value).__name__}>"


# ── settings ──────────────────────────────────────────────────────────────────

def own_profile(arm):
    """
    The profile of the arm's own deployment (its config's DEPLOYMENT: chess on
    the SO-101) — settings, env and plans record the arm as that deployment
    runs it, every deployment's own rules in deployments. None: it names none,
    or that deployment has no profile.
    """
    from mrcf import profiles

    task = getattr(arm.config, "DEPLOYMENT", None)
    return profiles.load(ROOT / "deployments" / task) if task else None


def arm_deployments(arm) -> list:
    """The deployments on this arm: the folders whose config.py names it (ARM_NAME)."""
    import re

    name = arm.__name__.rsplit(".", 1)[-1]
    found = []
    for d in sorted(p for p in (ROOT / "deployments").iterdir() if (p / "config.py").is_file()):
        named = re.search(r'^ARM_NAME\s*=\s*["\']([^"\']+)["\']', (d / "config.py").read_text(), re.M)
        if named and named.group(1) == name:
            found.append(d)
    return found


def settings(arm) -> dict:
    from mrcf import backends, profiles
    from mrcf.backends import wiring

    own = own_profile(arm)
    policies = arm.make_policies(profile=own)
    spec = getattr(policies, "spec", None)
    config = profiles.layered(arm.config, own)
    return {
        "name": arm.NAME,
        "missing_rig_settings": backends.missing_rig_settings(arm.config),
        "policies": type(policies).__name__,
        "spec": plain(spec),
        "rig_kwargs": plain(wiring.rig_kwargs(config)),
        "camera_kwargs": plain(wiring.camera_kwargs(config, {})),
    }


def deployments(arm) -> dict:
    """
    The rules of each deployment on this arm (its config's ARM_NAME; its
    profile.py, mrcf/profiles.py): what its profile sets, and the policies'
    spec it builds — {deployment: {"profile", "spec"}}, for those that have
    one. A deployment's rules changing, or one's reaching another, shows here.
    """
    from mrcf import profiles

    out = {}
    for d in arm_deployments(arm):
        profile = profiles.load(d)
        if profile is None:
            continue
        policies = arm.make_policies(profile=profile)
        out[d.name] = {"profile": plain(profiles.settings(profile)), "spec": plain(getattr(policies, "spec", None))}
    return out


# ── env ───────────────────────────────────────────────────────────────────────

class _StillBus:
    """The follower's motor bus, offline: joints where they were put, and nothing written."""

    def __init__(self, positions: dict):
        self.positions = dict(positions)
        self.is_connected = False

    def sync_read(self, _register, *a, **k) -> dict:
        return dict(self.positions)

    def read(self, *a, **k):
        raise RuntimeError("the baseline recorder reads no servo registers")

    def write(self, *a, **k):
        raise RuntimeError("the baseline recorder writes nothing to the servos")

    enable_torque = disable_torque = write


class _StillCamera:
    """A lerobot camera, offline: sending (its newest frame always just taken), but it hands out none."""

    @property
    def latest_timestamp(self) -> float:
        return time.perf_counter()

    def read_latest(self, *a, **k):
        raise RuntimeError("the baseline recorder takes no pictures: planning looked")

    def connect(self, *a, **k):
        raise RuntimeError("the baseline recorder opens no cameras")

    disconnect = read = async_read = connect


class _StillFollower:
    """lerobot's SO-101 follower, offline: refuses every command, so a plan that would move the arm fails."""

    def __init__(self, degrees: dict, gripper_pct: float, cameras=()):
        self.bus = _StillBus({**degrees, "gripper": gripper_pct})
        self.cameras: dict = {name: _StillCamera() for name in cameras if name is not None}
        self.calibration: dict = {}

    def send_action(self, action):
        raise RuntimeError(f"the baseline recorder moves nothing, but was sent {action}")

    def connect(self, *a, **k):
        raise RuntimeError("the baseline recorder connects nothing")

    configure = disconnect = connect


def offline_env(arm):
    """arm's env as its make_env() makes it for its own task (own_profile), with nothing connected and nothing able to move."""
    if arm.__name__.rsplit(".", 1)[-1] != "so101":
        return arm.make_env(profile=own_profile(arm))
    import importlib.util

    for lib in ("lerobot", "placo"):
        if importlib.util.find_spec(lib) is None:
            raise Skipped(f"needs {lib} (pip install -e '.[so101]')")
    rc = arm.config
    if not Path(rc.URDF_PATH).is_file():
        raise Skipped("needs the SO-101's URDF: python tools/so101/fetch_urdf.py")
    env = arm.make_env(profile=own_profile(arm), camera_device_name=None, overhead_camera_device_name=None)
    pct = env._width_to_pct(rc.PREGRASP_OPEN_M) if getattr(rc, "PREGRASP_OPEN_M", None) else 50.0
    env._robot = _StillFollower(rc.HOME_DEGREES, pct, cameras=(env._camera_name, env._overhead_camera_name))
    env._joint_deg_limits = dict(SO101_JOINT_LIMITS_DEG)
    env._setup_kinematics(rc.URDF_PATH, getattr(rc, "JOINT_OFFSETS_DEG", None),
                          tuple(getattr(rc, "IK_CALIBRATED_RANGE_JOINTS", ())))
    if env._kinematics is None:
        raise Skipped("the SO-101's kinematics would not load")
    env.POS_REUSE_S = 0.0
    return env


def _constants(env) -> dict:
    """The UPPER_CASE class attributes the env's class and its bases set: the shared tuning each arm runs with."""
    out = {}
    for cls in type(env).__mro__:
        for name, value in vars(cls).items():
            bare = name.lstrip("_")
            if bare and bare.upper() == bare and bare[0].isalpha() and name not in out \
                    and not callable(value) and not isinstance(value, (staticmethod, classmethod, property)):
                out[name] = plain(value)
    return dict(sorted(out.items()))


def env_record(arm) -> dict:
    from mrcf import capabilities

    first, second = offline_env(arm), offline_env(arm)
    a, b = plain(vars(first)), plain(vars(second))
    # What differs between two envs made alike (a clock read as it was made) is not the arm's.
    state = {k: v for k, v in a.items() if b.get(k) == v}
    return {
        "class": [c.__name__ for c in type(first).__mro__ if c is not object],
        "capabilities": [n for n in capabilities.__all__ if isinstance(first, getattr(capabilities, n))],
        "constants": _constants(first),
        "state": state,
        "base_xy": plain(first.base_xy),
        "frame_scale": plain(first.frame_scale),
        "contact_force_limits": plain(first.contact_force_limits),
    }


# ── plans ─────────────────────────────────────────────────────────────────────

def _ray_inside(bounds, base, u) -> "tuple[float, float]":
    """How far along u from base the ray is inside the bounds' x-y rectangle: (enter, leave), metres."""
    lo, hi = 0.0, float("inf")
    for axis, (mn, mx) in enumerate(((bounds.x_min, bounds.x_max), (bounds.y_min, bounds.y_max))):
        if abs(u[axis]) < 1e-12:
            continue
        t1, t2 = (mn - base[axis]) / u[axis], (mx - base[axis]) / u[axis]
        lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
    return lo, hi


def scenes(arm, env, spec) -> dict:
    """
    {name: (Target, [Obstacle])} for this arm, laid out along the line from its
    base through the middle of its WORKSPACE_BOUNDS: near the base, mid-reach
    and far out by its own spec.near_base_m and far_reach_m where it sets
    them, otherwise by where that line crosses its bounds.
    """
    from mrcf.policies.base import Obstacle, Target

    cfg = arm.config
    bounds = cfg.WORKSPACE_BOUNDS
    base = np.asarray(env.base_xy, float)[:2]
    centre = np.array([(bounds.x_min + bounds.x_max) / 2, (bounds.y_min + bounds.y_max) / 2])
    u = (centre - base) / np.linalg.norm(centre - base)
    v = np.array([-u[1], u[0]])
    t_in, t_out = _ray_inside(bounds, base, u)
    t_in, t_out = t_in + 0.02, t_out - 0.02
    near = spec.near_base_m - 0.015 if spec.near_base_m else t_in + 0.2 * (t_out - t_in)
    far = spec.far_reach_m + 0.015 if spec.far_reach_m else t_in + 0.9 * (t_out - t_in)
    mid = (near + far) / 2 if spec.near_base_m and spec.far_reach_m else t_in + 0.55 * (t_out - t_in)
    near, mid, far = (float(np.clip(r, t_in, t_out)) for r in (near, mid, far))

    def at(r, side=0.0):
        xy = base + r * u + side * v
        return np.array([xy[0], xy[1], cfg.TABLE_Z])

    foot = round(min(0.030, 0.55 * spec.max_open_m), 3)
    tall = round(max(0.09, spec.default_max_height_m + 0.01), 3)
    h = 0.045
    gap = foot + 0.006                                  # centre to centre: 6 mm between their feet

    def thing(r, side=0.0, **kw):
        return Target(at(r, side), foot_m=kw.pop("foot_m", foot), **({"height_m": h} | kw))

    def neighbour(p, label):
        return Obstacle(np.asarray(p, float), radius_m=foot / 2, label=label, height_m=h)

    m = at(mid)
    return {
        "near_base": (thing(near), []),
        "mid": (thing(mid), []),
        "far": (thing(far), []),
        "out_of_reach": (thing(t_out + 0.30), []),
        "unmeasured": (Target(m, foot_m=foot, max_height_m=tall), []),
        "tall": (thing(mid, height_m=tall), []),
        "too_wide": (thing(mid, foot_m=spec.max_open_m + 0.010), []),
        "block_faces_30deg": (thing(mid, faces_deg=30.0, widths_mm={0: foot * 1000, h * 1000: foot * 1000}), []),
        "on_a_block": (Target(m + [0, 0, 0.03], foot_m=foot, height_m=h, surface_z_m=cfg.TABLE_Z + 0.03), []),
        "neighbour_toward_base": (thing(mid), [neighbour(m[:2] - gap * u, "toward base")]),
        "neighbour_beside": (thing(mid), [neighbour(m[:2] + gap * v, "beside")]),
        "neighbours_all_round": (thing(mid), [neighbour(m[:2] + gap * d, n) for d, n in
                                              ((u, "out"), (-u, "in"), (v, "left"), (-v, "right"))]),
    }


def _outcome(o) -> dict:
    return {"ok": o.ok, "refused": o.refused, "message": o.message, "approach": o.approach,
            "blocked_by": list(o.blocked_by), "grasp_mm": _mm(o.grasp)}


def _mm(value):
    """A GraspPlan or waypoint list with its lengths in mm."""
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return {f.name: (plain(getattr(value, f.name) * 1000) if isinstance(getattr(value, f.name), float)
                         else plain(getattr(value, f.name))) for f in dataclasses.fields(value)}
    return plain(value)


def _fit(fit) -> dict:
    return {"approach": fit.approach.name, "tilt_deg": plain(fit.tilt_deg), "lift_height_mm": plain(fit.lift_height_m * 1000),
            "servo": fit.servo, "checked": fit.checked, "servo_skipped": fit.servo_skipped,
            "grasp_over_mm": plain(None if fit.grasp_over_m is None else fit.grasp_over_m * 1000),
            "plan_mm": _mm(fit.plan),
            "rot": plain(None if fit.rot is None else np.round(np.asarray(fit.rot), 4))}


def _waypoints(points) -> list:
    return [[phase, plain(np.round(np.asarray(goal) * 1000, 2)), plain(tol * 1000), plain(step)]
            for phase, goal, tol, step in points]


def _where(env):
    try:
        return np.asarray(env.get_ee_pos(), float).copy()
    except Exception:
        return None


def plan_one(policies, env, target, obstacles) -> dict:
    from mrcf.policies.base import Vision
    from mrcf.policies.parallel_jaw.pick import PickContext

    xy = target.position[:2]
    rec = {"target_mm": plain(np.round(np.asarray(target.position) * 1000, 2)),
           "plan_mm": _mm(policies.plan(target)),
           "zone": policies.zone(env, xy),
           "approach_order": [a.name for a in policies.approach_order(env, xy)],
           "room_check": _outcome(policies.pick_check(env, target, obstacles, quick=True)),
           "pick_check": _outcome(policies.pick_check(env, target, obstacles))}
    # Every approach the pick would weigh here, in its order: the ones it can do and why not the rest.
    policies._quiet = True
    try:
        ctx = policies._begin_pick(env, target, obstacles, Vision())
    finally:
        policies._quiet = False
    if isinstance(ctx, PickContext):
        k = 0
        while ctx.choices.feasible(k) is not None:
            k += 1
        rec["room_mm"] = plain(ctx.room * 1000)
        rec["turn_deg"] = plain(ctx.turn_deg)
        rec["room_unsettled"] = ctx.room_unsettled
        rec["servo_ready"] = ctx.servo_ready
        rec["fits"] = [_fit(f) for f in ctx.choices.fits]
        rec["cannot"] = list(ctx.choices.why_not)
        first = ctx.choices.feasible(0)
        rec["pick_waypoints"] = _waypoints(policies.pick_waypoints(ctx.object_xy, first.plan))
        rec["place_waypoints"] = _waypoints(policies.place_waypoints(ctx.object_xy, first.plan))
    return rec


def plans(arm, env) -> dict:
    policies = arm.make_policies(profile=own_profile(arm))
    spec = getattr(policies, "spec", None)
    if spec is None or not hasattr(policies, "_begin_pick"):
        raise Skipped(f"{type(policies).__name__} plans no grasps")
    start = _where(env)
    out = {}
    for name, (target, obstacles) in scenes(arm, env, spec).items():
        out[name] = plan_one(policies, env, target, obstacles)
        now = _where(env)
        if start is not None and now is not None and np.linalg.norm(now - start) > 1e-6:
            raise RuntimeError(f"planning '{name}' moved the arm from {start} to {now}")
    m = scenes(arm, env, spec)["mid"][0]
    toward = np.asarray(env.base_xy, float)[:2] - m.position[:2]
    to_xy = m.position[:2] + 0.03 * toward / np.linalg.norm(toward)
    out["nudge_toward_base"] = {"nudge": _outcome(policies.nudge(env, m, to_xy, [], check_only=True))}
    # What the pick can see with: a change that leaves an arm's camera unused shows here.
    from mrcf.policies.base import Vision

    detector = Vision(find=lambda *a, **k: None)
    out["wrist_camera"] = {"camera": spec.camera,
                           "sending": bool(policies.wrist_sending(env)),
                           "located": bool(policies.camera_located(env)),
                           "servo_ready_with_a_detector": bool(policies._servo_ready(env, detector))}
    return out


# ── Inside the fresh process ──────────────────────────────────────────────────

# The extras' libraries (pyproject.toml): a section that needs one this
# machine lacks is skipped, as conftest.py skips a test that needs OpenCV.
# Any other missing module is a failure.
OPTIONAL_LIBRARIES = ("cv2", "lerobot", "placo", "torch", "torchvision", "transformers", "serial",
                      "scservo_sdk", "rtde_control", "rtde_receive", "rtde_io", "stable_baselines3")


def _section(make, *args) -> dict:
    try:
        return make(*args)
    except Skipped as why:
        return {"skipped": str(why)}
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in OPTIONAL_LIBRARIES:
            return {"skipped": f"needs {exc.name}"}
        raise


def _inside(name: str) -> dict:
    from mrcf import backends
    from mrcf.perception import segment

    segment.available = lambda: True        # as on the rigs (the module docstring)
    with contextlib.redirect_stdout(io.StringIO()):     # arms and policies say what they do as they start
        arm = backends.arm(name)
        out = {"settings": _section(settings, arm), "env": _section(env_record, arm),
               "deployments": _section(deployments, arm)}
        out["plans"] = (dict(out["env"]) if "skipped" in out["env"]
                        else _section(lambda: plans(arm, offline_env(arm))))
    return out


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("arm")
    p.add_argument("--diff", action="store_true", help="only how it differs from tests/baselines/<arm>/")
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.inside:
        print(json.dumps(_inside(a.arm), sort_keys=True))
        return
    got = record(a.arm)
    if not a.diff:
        print(json.dumps(got, indent=1, sort_keys=True))
        return
    for section in SECTIONS:
        want = load(a.arm, section)
        lines = (["(no baseline)"] if want is None else
                 differences(want, got[section], TOLERANCE[section]))
        print(f"── {a.arm} {section}: " + ("same" if not lines else f"{len(lines)} differences"))
        for line in lines:
            print("  " + line)


if __name__ == "__main__":
    main()
