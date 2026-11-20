"""
rest.py — park the arm somewhere safe and cut torque, for long idle periods.

Why not just disable torque where it stands
-------------------------------------------
Every script here leaves the servos holding position when it exits (see
disable_torque_on_disconnect in robot_config). Holding a pose against gravity
draws current continuously and heats the motors, and a hot STS3215 trips its
overload protection and drops off the bus entirely — which is exactly what
happened to this arm's gripper after an afternoon of calibration runs.

But cutting torque is only safe from the right pose. At HOME the gripper is
186 mm above the table and at the overhead parking pose 233 mm; releasing
torque there drops the whole arm that distance onto the board.

So this drives to a folded, low pose FIRST — gripper a couple of centimetres
above the table, arm bent back over its own base, so there is almost no height
to fall from and almost no stored energy — verifies it actually arrived, and
only then releases.

Where it folds down depends on what stands on the table, so there are several
rest poses (REST_POSES: the same fold, the base turned). Before moving, each
one's path is worked out as the arm will drive it and checked against what
stands on the table now (table_now): the overhead camera looks first, and the
objects the deployment remembers are kept clear of where it still sees them,
along with whatever it sees that they are not, and the kept containers' walls;
without a camera look, the memory as the last scan left it. The first pose
that keeps the jaw tips clear of all of it is used, and why the others were
not is said.

The jaws are left open rather than closed: a gripper resting against a hard
stop is still developing torque, and the gripper is the motor that overheats.

Usage:
    python tools/so101/rest.py                 # choose a rest pose, park and release
    python tools/so101/rest.py --pose right    # that pose (still checked; --force to skip)
    python tools/so101/rest.py --check         # the poses, checked from home against the last scan; connects to nothing
    python tools/so101/rest.py --wake          # re-enable torque, unfold, drive home
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from tools.so101._common import make_env, rc, task_config   # noqa: E402

# Folded and low, chosen against the arm's own FK and its calibrated limits:
# at least 11 deg of margin on every joint whichever way the base is turned
# (to +-60 deg). Height is not the only thing that matters: an earlier
# candidate sat 8 mm up but left elbow_flex 2.1 deg from its stop, and a servo
# parked hard against its limit is still developing torque, which is the
# opposite of resting.
_FOLD = {"shoulder_lift": -80.0, "elbow_flex": 80.0, "wrist_flex": 90.0, "wrist_roll": 0.0}

# The rest poses, in the order they are tried: the fold with the base turned
# (+ toward the arm's right, the h-file side of the chess board). With the
# board 2 cm further out (2026-11-03) straight ahead came down on its near edge
# beside the e1 king below its top; turned right it rested on the bare table
# beside h1 (grasp point about [0.065, -0.076, 0.024]).
REST_POSES = {name: {"shoulder_pan": pan, **_FOLD}
              for name, pan in (("right", 60.0), ("left", -60.0), ("ahead", 0.0),
                                ("half-right", 30.0), ("half-left", -30.0))}

# How close the arm must actually be to the rest pose before torque is cut.
# The whole safety argument rests on the arm being folded; if the move was
# obstructed and it stopped halfway, releasing would drop it from wherever it
# stopped — so this refuses rather than guessing.
ARRIVE_TOL_DEG = 6.0

# The path check. The jaw tips keep CLEAR_ABOVE_M above any object whose
# outline the gripper (BODY_M round its grasp point: the jaws and the wrist
# camera) comes within NEAR_M of, and every joint MIN_MARGIN_DEG from its limit
# at rest. An object never measured is as tall as the deployment's tallest.
CLEAR_ABOVE_M = 0.03
BODY_M = 0.025
NEAR_M = 0.02
MIN_MARGIN_DEG = 5.0
_GAP_M = 0.027          # the jaws' opening the grasp point is modelled at, on the way


def _turned_home(pan: float) -> dict:
    return {**rc.HOME_DEGREES, "shoulder_pan": pan}


def _as_array(deg: dict) -> np.ndarray:
    from mrcf.backends.so101.kinematics import JOINT_NAMES
    return np.array([float(deg[n]) for n in JOINT_NAMES])


def _equal_rate(a: np.ndarray, b: np.ndarray) -> list:
    """
    The joints from a to b as goto_joint_degrees drives them: each toward its
    goal at most MAX_RELATIVE_TARGET_DEG a tick, so the one with least to go
    gets there first — not a straight line in joint space.
    """
    step = float(getattr(rc, "MAX_RELATIVE_TARGET_DEG", 8.0)) / 2     # sampled finer than the arm moves
    q, out = a.copy(), [a.copy()]
    while np.abs(b - q).max() > 0.25:
        q = q + np.clip(b - q, -step, step)
        out.append(q.copy())
    return out


def rest_path(start_deg: dict, pose: dict) -> list:
    """The joints from start to pose: the base turned at home's height first, then folded down."""
    turned = _as_array(_turned_home(pose["shoulder_pan"]))
    return _equal_rate(_as_array(start_deg), turned) + _equal_rate(turned, _as_array(pose))


def wake_path(start_deg: dict) -> list:
    """The joints from a rest pose to home: unfolded where it rests first, then the base turned."""
    unfolded = _as_array(_turned_home(start_deg["shoulder_pan"]))
    return _equal_rate(_as_array(start_deg), unfolded) + _equal_rate(unfolded, _as_array(rc.HOME_DEGREES))


def closest_call(env, path: list, objects: list) -> "tuple[float, str] | None":
    """
    Over path, the least height of the jaw tips above the top of an object the
    gripper passes beside or over: (metres, its label), or None when it passes
    none. objects: [(xy, foot radius, height, label)].
    """
    worst = None
    for q in path:
        g = env._grasp_point(np.append(q, 0.0), _GAP_M)
        tips = g[2] - rc.JAW_TIP_BELOW_GRASP_M - rc.TABLE_Z
        for xy, radius, height, label in objects:
            if float(np.linalg.norm(g[:2] - xy)) - radius - BODY_M < NEAR_M:
                clear = tips - height
                if worst is None or clear < worst[0]:
                    worst = (clear, label)
    return worst


def joint_margin(env, pose: dict) -> float:
    """The least distance of any of pose's joints from its limit (the arm's reach limits), degrees."""
    limits = getattr(env, "_ik_limits", {})
    return min((min(v - limits[n][0], limits[n][1] - v) for n, v in pose.items() if n in limits), default=180.0)


def judge(env, start_deg: dict, name: str, objects: list) -> "tuple[bool, str]":
    """Whether rest pose name can be reached from start_deg clear of objects, and what was found."""
    pose = REST_POSES[name]
    rest_at = env._grasp_point(np.append(_as_array(pose), 0.0), _GAP_M)
    where = f"rests at ({rest_at[0]:+.3f}, {rest_at[1]:+.3f})"
    margin = joint_margin(env, pose)
    if margin < MIN_MARGIN_DEG:
        return False, f"{where}; a joint {margin:.1f} deg from its limit"
    call = closest_call(env, rest_path(start_deg, pose), objects)
    if call is None:
        return True, f"{where}; passes beside nothing"
    clear, label = call
    if clear < CLEAR_ABOVE_M:
        return False, f"{where}; the jaw tips pass {clear * 1000:+.0f} mm above {label}'s top"
    return True, f"{where}; at the closest {clear * 1000:.0f} mm above {label}'s top"


def choose(env, start_deg: dict, objects: list, only: "str | None" = None) -> "str | None":
    """The first rest pose (only: that one) that judge passes, each said; None when none does."""
    chosen = None
    for name in ([only] if only else REST_POSES):
        ok, why = judge(env, start_deg, name, objects)
        print(f"  {name:<10} {'ok' if ok else 'no'}: {why}")
        if ok and chosen is None:
            chosen = name
    return chosen


def _deployment():
    """The deployment's config this tool works for (tools/_common.task_config), or None, said."""
    try:
        return task_config()
    except SystemExit as exc:
        print(f"  (No deployment to ask what stands on the table: {str(exc).splitlines()[0]})")
        return None


def _tallest(cfg) -> float:
    """The deployment's tallest object (TALLEST_PIECE_M or TALLEST_OBJECT_M; 0.10 m if it says neither)."""
    return float(getattr(cfg, "TALLEST_PIECE_M", None) or getattr(cfg, "TALLEST_OBJECT_M", None) or 0.10)


def _memory_objects(cfg) -> list:
    """What the deployment's object memory has standing on the table, [(xy, foot radius, height, label)]."""
    # Where the task keeps it (its MEMORY_PATH: the tabletop's is in the rig folder), else beside its config.
    path = Path(getattr(cfg, "MEMORY_PATH", None) or Path(cfg.__file__).parent / "object_memory.json")
    if not path.exists():
        print(f"  (No object memory at {path}: nothing known to stand on the table.)")
        return []
    from mrcf.object_memory import ObjectMemory
    memory = ObjectMemory(path)
    tallest = _tallest(cfg)
    out = []
    for t in memory.in_state("present"):
        if t.position is None:
            continue
        height = max(t.heights_mm) / 1000 if getattr(t, "heights_mm", None) else tallest
        radius = (t.foot_mm or 30.0) / 2000
        out.append((np.asarray(t.position[:2], float), radius, height, memory.name(t)))
    print(f"  {len(out)} objects on the table, from {path}.")
    return out


def remembered_objects() -> list:
    """
    What the deployment's object memory has standing on the table, and its
    kept containers' walls, as [(xy, foot radius, height, label)]: measured
    heights where it has them, else the deployment's tallest. [] when there is
    no deployment. As the table was at the last scan: table_now looks.
    """
    cfg = _deployment()
    if cfg is None:
        return []
    return _memory_objects(cfg) + _container_walls(cfg)


# A remembered object is still there when something is seen this near where it was (beyond its own foot).
SEEN_NEAR_M = 0.03


def seen_now(env, cfg) -> "tuple[list, object] | None":
    """
    What stands on the table now, as the deployment's overhead camera sees it
    from where the arm is: ([(xy, foot radius)], hidden), each outlined against
    the photo of the empty table (surface_scan: no names, no vision model),
    those on the arm itself left out; hidden(xy) whether the arm hides that
    point of the table from the camera. None when it cannot look (no camera
    model, no segmenter, no picture).
    """
    from mrcf import scene_scan
    from mrcf.perception import plane_geometry as pg
    from mrcf.perception import surface_scan

    camera = getattr(cfg, "OBSERVE_CAMERA", None) or getattr(rc, "OVERHEAD_CAMERA_NAME", None)
    if camera is None:
        return None
    w = int(getattr(cfg, "CAPTURE_W_OVERHEAD", rc.OVERHEAD_CAMERA_W))
    h = int(getattr(cfg, "CAPTURE_H_OVERHEAD", rc.OVERHEAD_CAMERA_H))
    path = getattr(cfg, "SURFACE_REFERENCE_PATH", None)
    reference = surface_scan.load_reference(path, (w, h)) if path else None
    b = getattr(cfg, "WORKSPACE_BOUNDS", rc.WORKSPACE_BOUNDS)
    try:
        got = surface_scan.scan(env, camera, w, h, (b.x_min, b.x_max), (b.y_min, b.y_max), reference=reference)
    except Exception as exc:                                     # noqa: BLE001 — then the memory, as before
        print(f"  (Could not look at the table: {type(exc).__name__}: {exc})")
        return None
    if got is None:
        return None
    found = got[1]
    arm = scene_scan.arm_in_view(env, camera, w, h)
    model = env.camera_model(camera, w, h)

    def hidden(xy) -> bool:
        return False
    if arm is not None and model is not None:
        found = [f for f in found if float((np.asarray(f["mask"], bool) & arm).sum())
                 < scene_scan.ARM_OVERLAP * max(1.0, float(np.asarray(f["mask"], bool).sum()))]
        hidden = scene_scan.hidden_by(arm, pg.PlaneView.from_camera_model(model),
                                       float(getattr(cfg, "TABLE_Z", rc.TABLE_Z)))
    feet = [(np.asarray(f["measurement"].base_world_m, float)[:2], float(f["measurement"].base_radius_mm) / 1000.0)
            for f in found]
    return feet, hidden


def on_the_table(remembered: list, seen, tallest: float) -> list:
    """
    The remembered objects as the table is now (seen: seen_now's): each kept
    where something is still seen within SEEN_NEAR_M of it — at where it is
    seen — or where the arm hides the table; the rest have gone and are not
    kept clear of. Whatever is seen that nothing remembered is near is kept
    clear of, as tall as the tallest there may be. seen None: the memory as it is.
    """
    if seen is None:
        return list(remembered)
    feet, hidden = seen
    out, matched = [], set()
    for xy, radius, height, label in remembered:
        xy = np.asarray(xy, float)
        near = sorted((float(np.linalg.norm(p - xy)), i) for i, (p, r) in enumerate(feet)
                      if float(np.linalg.norm(p - xy)) <= SEEN_NEAR_M + max(radius, r))
        if near:
            i = near[0][1]
            matched.add(i)
            out.append((feet[i][0], max(radius, feet[i][1]), height, label))
        elif hidden(xy):
            out.append((xy, radius, height, label))
        else:
            print(f"  {label}: not there now — not kept clear of.")
    for i, (p, r) in enumerate(feet):
        if i not in matched:
            out.append((p, r, tallest, f"something at ({p[0]:+.3f}, {p[1]:+.3f})"))
    return out


def table_now(env) -> list:
    """
    What a rest's or a wake's path is checked against: the table as the
    overhead camera sees it now (seen_now, on_the_table), with the kept
    containers' walls; as the last scan left it when the camera cannot look.
    2026-11-19 18:20: the user moved the scoop and a figure off the left rest's
    path, and every rest was refused for them still, from the last scan.
    """
    cfg = _deployment()
    if cfg is None:
        return []
    remembered = _memory_objects(cfg)
    seen = seen_now(env, cfg)
    if seen is None:
        print("  (The overhead camera could not look: checked against the last scan.)")
    else:
        print(f"  Looked at the table: {len(seen[0])} thing(s) standing on it.")
    return on_the_table(remembered, seen, _tallest(cfg)) + _container_walls(cfg)


WALL_STEP_M = 0.015      # a container's walls kept clear of as points this far apart along its outline


def _container_walls(cfg) -> list:
    """
    The deployment's kept containers' walls (its CONTAINERS_PATH: each one's
    outline at the rim where it was last seen, and its rim's height), as
    points along the outline, each kept clear of like an object that tall.
    2026-11-19: the rest folded the arm into the tub, which this tool knew
    nothing of — and checked its path against the chess board's pieces.
    """
    import json

    path = getattr(cfg, "CONTAINERS_PATH", None)
    if path is None or not Path(path).exists():
        return []
    try:
        kept = dict(json.loads(Path(path).read_text()))
    except (OSError, ValueError, TypeError):
        return []
    out = []
    for name, c in kept.items():
        poly = np.asarray((c or {}).get("outline_xy") or [], float).reshape(-1, 2)
        rim = float((c or {}).get("rim_m") or 0.0)
        if len(poly) < 3 or rim <= 0:
            continue
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            n = max(1, int(np.linalg.norm(b - a) / WALL_STEP_M))
            for f in np.linspace(0.0, 1.0, n, endpoint=False):
                out.append((a + f * (b - a), 0.008, rim + 0.01, f"{name}'s wall"))
        print(f"  {name}: its walls, {rim * 1000:.0f} mm, where it was last seen ({c.get('seen', '?')}).")
    return out


def rest(env, only: "str | None" = None, force: bool = False) -> bool:
    """
    Park the connected arm in a rest pose clear of what stands on the table,
    and release its torque, refusing to release unless it got there. only: that
    rest pose; force: whatever the check says. True once it is resting.
    """
    print("\nChoosing a rest pose, its path checked against what stands on the table:")
    start = env._motor_pos()
    name = choose(env, start, table_now(env), only)
    if name is None:
        if not (force and only):
            print("\nREFUSING to move: no rest pose keeps the jaws clear of what stands on the table.")
            print("Clear a way, or name one with --pose NAME --force.")
            return False
        name = only
        print(f"  --force: resting at {name} anyway.")
    pose = REST_POSES[name]
    print(f"\nParking at '{name}' before release. Keep clear of the arm.")
    print(f"  target: {', '.join(f'{k}={v:+.0f}' for k, v in pose.items())}")
    print("  turning the base at home's height first...")
    if not env.goto_joint_degrees(_turned_home(pose["shoulder_pan"]), tol_deg=ARRIVE_TOL_DEG, timeout_s=8.0):
        print("\nREFUSING to fold: the base did not turn to the rest side at height.")
        return False
    reached = env.goto_joint_degrees(pose, tol_deg=ARRIVE_TOL_DEG, timeout_s=8.0)

    pos = env._motor_pos()
    worst = max(abs(pos[n] - v) for n, v in pose.items())
    here = env.get_ee_pos()
    print(f"  gripper now {(here[2] - 0.0) * 1000:.0f} mm above the table "
          f"(worst joint {worst:.1f} deg from target)")

    if not reached or worst > ARRIVE_TOL_DEG:
        print("\nREFUSING to release torque: the arm did not reach the rest pose.")
        print("Releasing from a part-folded pose would drop it from height.")
        print("Clear whatever is obstructing it and re-run, or use unlock.py")
        print("if you want to take the weight by hand.")
        return False

    # Jaws open: no contact, no holding current in the motor that overheats.
    env.set_gripper(getattr(rc, "GRIPPER_MAX_OPEN_M", 0.045) * 0.6, n_steps=10)

    print("\nReleasing torque in 3 seconds — the arm will go limp.")
    for n in (3, 2, 1):
        print(f"  {n}...", end="", flush=True)
        time.sleep(1.0)
    print()
    env.disable_torque()
    print("""
Torque off. The arm is drawing no holding current and will not heat.
Safe to leave for hours. You can move it by hand.

To bring it back:
    python tools/so101/rest.py --wake""")
    return True


def wake(env, force: bool = False) -> bool:
    """Torque back on, unfolded where it rests, then home — once that path is checked clear."""
    start = env._motor_pos()
    call = closest_call(env, wake_path(start), table_now(env))
    if call is not None and call[0] < CLEAR_ABOVE_M and not force:
        print(f"\nREFUSING to wake: on the way home the jaw tips would pass {call[0] * 1000:+.0f} mm above "
              f"{call[1]}'s top. Move it, or re-run with --force.")
        return False
    print("Re-enabling torque...")
    env.enable_torque()
    time.sleep(0.3)
    print("Unfolding where it rests, then turning to HOME_DEGREES — keep clear.")
    env.goto_joint_degrees(_turned_home(start["shoulder_pan"]), tol_deg=ARRIVE_TOL_DEG, timeout_s=8.0)
    if env.go_home():
        print("At home. The arm is holding position again.")
        return True
    print("Did not reach home. Check for an obstruction, then re-run.")
    return False


def connected(overrides: dict):
    """
    The arm's env, connected: with the deployment's table (overrides: for the
    camera's plane and the floor — mrcf/tabletop's bare table, not the board)
    and its cameras, for table_now's look; without its cameras when they are
    not there (unplugged, or the wrist camera dropped off the bus), said — a
    rest must not depend on a camera, and its path is then checked against the
    last scan.
    """
    from mrcf import devices

    print(f"Connecting to {rc.FOLLOWER_ID} on {rc.FOLLOWER_PORT}...")
    env = make_env(**overrides)
    try:
        env.connect()
        return env
    except devices.PersonalCamera as exc:
        print(f"  (The rig's cameras are not there — {str(exc).split(';')[0]}: connecting without them, "
              "and checking against the last scan.)")
    env = make_env(**{**overrides, "camera_name": None, "overhead_camera_name": None, "cameras": {}})
    env.connect()
    return env


def _arg(flag: str) -> "str | None":
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv) else None


def main() -> None:
    only, force = _arg("--pose"), "--force" in sys.argv
    if only is not None and only not in REST_POSES:
        raise SystemExit(f"No rest pose '{only}'. Known: {', '.join(REST_POSES)}")
    if "--check" in sys.argv:
        from tools.so101.visualise_ik import offline_arm      # the arm's model, no hardware
        print("The rest poses, each checked from home (nothing connected, nothing moves):")
        choose(offline_arm(), dict(rc.HOME_DEGREES), remembered_objects(), only)
        return
    cfg = _deployment()
    env = connected(dict(getattr(cfg, "ENV_OVERRIDES", None) or {}))
    try:
        if "--wake" in sys.argv:
            wake(env, force)
            return
        rest(env, only, force)
    finally:
        # Deliberately NOT go_home() here: that is the opposite of what this
        # tool is for, and would re-energise everything we just released.
        env.stop()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
