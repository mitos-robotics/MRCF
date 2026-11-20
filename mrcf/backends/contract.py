"""
mrcf/backends/contract.py — what every arm's RobotEnv must do, as checks a new
backend runs on an offline instance of itself (its motors and cameras faked,
no hardware), before it ever drives the real arm.

    from mrcf.backends.contract import check_backend

    def test_my_arm_keeps_the_contract():
        check_backend(make_offline_env, reachable=[(0.25, 0.0, 0.08)], unreachable=[(2.0, 0.0, 0.1)])

`make_env()` returns a fresh env each call; an env that takes a stream of
actions (capabilities.ActionStreaming) has that checked too. Every check drives it only as the
framework does — RobotEnv's methods, the extras in capabilities.py, the
keyword arguments the policies pass — so an env that passes is one the
policies, the controller and the watchers can drive. Each check_* runs on its
own too; check_backend returns which capabilities the env has.

What it cannot check is the arm itself: that get_ee_pos() is really where the
jaws close, that hold() really stops the motors. Those are for the arm's own
tools (on the SO-101: check_hold.py, check_targeting.py), on the rig.
"""

from __future__ import annotations

import inspect
import threading
import time

import numpy as np

from mrcf import capabilities
from mrcf.interlock import MotionStopped

# Keyword arguments the policies pass, by method: an env that does not take
# them fails with a TypeError mid-pick.
POLICY_KWARGS = {
    "move_ee_to": ("tol", "orientation", "max_step_deg"),
    "set_gripper": ("n_steps", "max_step_pct"),
}
# Protocol keyword arguments a backend may leave out (capabilities.OrientationSearch:
# "callers retry without them").
MAY_OMIT = {"approach_orientation_for": {"gap_m", "flip", "prefer_tilt_deg", "margin_deg"}}


def _accepts(fn, names) -> list:
    """The names fn cannot be called with as keywords."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []
    return [n for n in names if n not in params]


def check_sensing(env) -> None:
    """Where the grasp point is (3 finite metres), the joints, the opening, whether it may be moving."""
    ee = np.asarray(env.get_ee_pos(), float)
    assert ee.shape == (3,) and np.isfinite(ee).all(), f"get_ee_pos() gave {ee!r}, not three finite metres"
    joints = env.get_joint_angles()
    assert isinstance(joints, dict) and all(np.isfinite(float(v)) for v in joints.values()), \
        f"get_joint_angles() gave {joints!r}, not {{name: radians}}"
    width = float(env.get_gripper_width())
    assert width >= -1e-4, f"get_gripper_width() gave {width} m"
    assert isinstance(env.may_be_moving(), bool), "may_be_moving() must say True or False"
    xy = np.asarray(env.base_xy, float)
    assert xy.shape == (2,), f"base_xy gave {xy!r}, not (x, y)"
    assert float(env.frame_scale) > 0


def check_signatures(env) -> None:
    """The keyword arguments the policies pass, and each capability's own, are accepted."""
    for method, names in POLICY_KWARGS.items():
        missing = _accepts(getattr(env, method), names)
        assert not missing, f"{method}() does not take {missing}, which the policies pass"
    for name in capabilities.__all__:
        proto = getattr(capabilities, name)
        if not isinstance(env, proto):
            continue
        for attr, member in vars(proto).items():
            if attr.startswith("_") or not callable(member):
                continue
            wanted = [p for p in list(inspect.signature(member).parameters)[1:]]
            missing = [n for n in _accepts(getattr(env, attr), wanted) if n not in MAY_OMIT.get(attr, ())]
            assert not missing, f"{attr}() does not take {missing} (capabilities.{name})"


def check_lifecycle(make_env) -> None:
    """
    connect() and stop() run with no arguments, as every program calls them;
    on an arm on a rig (mrcf/backends/rig.py) connect() starts the sentry and stop()
    stops it, so nothing the arm does while connected is unwatched. The sentry
    itself is not started: whether it sees is motion_guard's to check.
    """
    from mrcf.backends.rig import RigEnv

    env = make_env()
    for name in ("connect", "stop"):
        params = inspect.signature(getattr(env, name)).parameters.values()
        needed = [p.name for p in params if p.default is p.empty
                  and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
        assert not needed, f"{name}() needs {needed}, and every program calls it with nothing"
    sentry = []
    if isinstance(env, RigEnv):
        env._start_sentry = lambda: sentry.append("start")  # type: ignore[method-assign]  # stand-ins, to count the calls
        env._stop_sentry = lambda: sentry.append("stop")  # type: ignore[method-assign]
    env.connect()
    if isinstance(env, RigEnv):
        assert sentry == ["start"], ("connect() did not start the sentry once: an arm on a rig defines connect() "
                                     "in its class, where RigEnv starts the sentry after it")
    env.stop()
    if isinstance(env, RigEnv):
        assert sentry[1:2] == ["stop"], "stop() did not stop the sentry"


def check_moves(env, reachable, unreachable=(), tol: float = 0.01) -> None:
    """
    move_ee_to says whether it arrived: True, and the grasp point within tol,
    for each reachable point; False — not an exception — for each unreachable one.
    """
    for p in reachable:
        p = np.asarray(p, float)
        assert env.move_ee_to(p, tol=tol) is True, f"move_ee_to({p.tolist()}) did not arrive"
        off = float(np.linalg.norm(np.asarray(env.get_ee_pos(), float) - p))
        assert off <= tol * 1.5, f"move_ee_to({p.tolist()}) said it arrived, {off * 1000:.1f} mm away"
    for p in unreachable:
        arrived = env.move_ee_to(np.asarray(p, float), tol=tol)
        assert arrived is False, f"move_ee_to({list(p)}) should report False, not {arrived!r}"


def check_gripper(env, widths=(0.02, 0.0), tol: float = 0.003) -> None:
    """set_gripper reaches each opening (nothing between the jaws); squeeze_gripper says (opening, holding)."""
    for w in widths:
        env.set_gripper(w, n_steps=5)
        got = float(env.get_gripper_width())
        assert abs(got - w) <= tol, f"set_gripper({w}) left the jaws at {got:.4f} m"
    opening, holding = env.squeeze_gripper()
    assert isinstance(float(opening), float) and isinstance(holding, (bool, np.bool_)), \
        f"squeeze_gripper() gave {(opening, holding)!r}, not (metres, bool)"
    assert not holding, "squeeze_gripper() says it holds something with nothing between the jaws"


def check_stops(make_env, goal) -> None:
    """
    A tripped interlock: hold() runs, every motion command raises MotionStopped
    and nothing moves, until it clears — then the arm moves again.
    """
    env = make_env()
    holds = []
    hold = env.hold
    env.hold = lambda: (holds.append(1), hold())[1]  # type: ignore[func-returns-value]  # the tuple only orders the calls
    start = np.asarray(env.get_ee_pos(), float)
    env.interlock.trip("contract check", source="contract")
    for name, call in (("move_ee_to", lambda: env.move_ee_to(np.asarray(goal, float))),
                       ("set_gripper", lambda: env.set_gripper(0.01, n_steps=5))):
        try:
            call()
        except MotionStopped:
            pass
        else:
            raise AssertionError(f"{name}() ran with the interlock tripped: call guard() before every command")
    assert holds, "the interlock tripped and hold() was never called"
    moved = float(np.linalg.norm(np.asarray(env.get_ee_pos(), float) - start))
    assert moved < 1e-4, f"the arm moved {moved * 1000:.1f} mm with the interlock tripped"
    env.interlock.clear("contract")
    assert env.move_ee_to(np.asarray(goal, float)) is True, "the arm did not move again once the interlock cleared"


def check_pauses(make_env, goal, wait_s: float = 0.3) -> None:
    """A pausing source (Interlock.pause_only) holds a move where it is, and the move carries on once it clears."""
    env = make_env()
    start = np.asarray(env.get_ee_pos(), float)
    env.interlock.pause_only("contract")
    env.interlock.trip("contract pause", source="contract")
    result = {}

    def move():
        try:
            result["arrived"] = env.move_ee_to(np.asarray(goal, float))
        except BaseException as exc:          # noqa: BLE001 — MotionStopped included: that is the failure
            result["raised"] = exc

    mover = threading.Thread(target=move, daemon=True)
    mover.start()
    mover.join(wait_s)
    assert mover.is_alive(), ("a paused move did not wait: it "
                              + (f"raised {result['raised']!r}" if "raised" in result else
                                 f"returned {result.get('arrived')!r}") + " while the interlock held it")
    moved = float(np.linalg.norm(np.asarray(env.get_ee_pos(), float) - start))
    assert moved < 1e-4, f"the arm moved {moved * 1000:.1f} mm while paused"
    env.interlock.clear("contract")
    mover.join(30.0)
    assert not mover.is_alive() and result.get("arrived") is True, "a paused move did not carry on once it cleared"


def check_cameras(env, size=(320, 240)) -> None:
    """Every listed camera gives frames at the size asked for, and a camera model is a usable one or None."""
    w, h = size
    for cam in env.list_cameras():
        frame = env.render_image(cam, w, h)
        assert isinstance(frame, np.ndarray) and frame.shape == (h, w, 3) and frame.dtype == np.uint8, \
            f"render_image({cam!r}, {w}, {h}) gave {getattr(frame, 'shape', frame)!r} {getattr(frame, 'dtype', '')}"
        model = env.camera_model(cam, w, h)
        if model is None:
            continue
        K, T = np.asarray(model["K"], float), np.asarray(model["T_world_cam"], float)
        assert K.shape == (3, 3) and T.shape == (4, 4), f"camera_model({cam!r}): K {K.shape}, T_world_cam {T.shape}"
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), f"camera_model({cam!r}): T_world_cam is not a rigid pose"
        assert np.isfinite(float(model["plane_z_m"])) and float(model["scale"]) > 0


def check_sensors(env) -> None:
    """
    Every sensor list_sensors() names reads: finite, and the shape its Sensor
    declares (backends.rig.Sensor.shape); get_contact_force() is a finite,
    non-negative force (a "force" sensor's, or 0 without one).
    """
    declared = getattr(env, "_sensors", None) or {}
    for name in env.list_sensors():
        value = np.asarray(env.read_sensor(name), float)
        assert np.isfinite(value).all(), f"read_sensor({name!r}) gave {value!r}, not finite"
        shape = tuple(getattr(declared.get(name), "shape", ()) or ())
        if shape:
            assert value.shape == shape, f"read_sensor({name!r}) gave shape {value.shape}, its Sensor says {shape}"
    force = float(env.get_contact_force())
    assert np.isfinite(force) and force >= 0, f"get_contact_force() gave {force}"


def check_threads(make_env, goal, size=(320, 240)) -> None:
    """Frames, camera models and the joints can be read from another thread while a move runs (the watchers do)."""
    env = make_env()
    errors, stop = [], threading.Event()

    def watch():
        while not stop.is_set():
            try:
                for cam in env.list_cameras():
                    env.render_image(cam, *size)
                    env.camera_model(cam, *size)
                env.arm_shape()
                env.get_ee_pos()
            except Exception as exc:          # noqa: BLE001 — any of them is the failure
                errors.append(exc)
                return
            time.sleep(0.001)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        assert env.move_ee_to(np.asarray(goal, float)) is True, "the move did not arrive with a watcher reading"
    finally:
        stop.set()
        watcher.join(5.0)
    assert not errors, f"reading the arm from another thread mid-move failed: {errors[0]!r}"


def check_streaming(make_env) -> None:
    """
    An arm that takes a stream (capabilities.ActionStreaming — a learned
    policy): an observation is what its spec says, every key at its
    shape; an action of the spec's length goes out and comes back as sent (sent
    where the arm is, when its state and its action are the same thing); and
    the interlock stops a stream as it stops any command.
    """
    env = make_env()
    spec = env.stream_spec
    observation = env.stream_observation()
    for key, feature in spec.features().items():
        if key == "action":
            continue
        assert key in observation, f"stream_observation() has no {key!r} (stream_spec has it)"
        got = np.asarray(observation[key])
        assert tuple(got.shape) == tuple(feature["shape"]), \
            f"stream_observation()[{key!r}] is {got.shape}, stream_spec says {tuple(feature['shape'])}"
        if feature["dtype"] == "video":
            assert got.dtype == np.uint8, f"stream_observation()[{key!r}] is {got.dtype}, not an RGB uint8 frame"
    if tuple(spec.state) != tuple(spec.action):
        return
    here = np.asarray(observation["observation.state"], float)
    sent, why = env.stream_action(here)
    assert np.asarray(sent).shape == (len(spec.action),), f"stream_action() gave back {np.asarray(sent).shape}"
    assert why is None or isinstance(why, str), f"stream_action()'s reason is {why!r}, not text or None"
    env.interlock.trip("contract check", source="contract")
    try:
        env.stream_action(here)
    except MotionStopped:
        pass
    else:
        raise AssertionError("stream_action() ran with the interlock tripped: call guard() before every command")
    finally:
        env.interlock.clear("contract")


def check_arm_shape(env) -> None:
    """arm_shape() is None, or points from base to tip, each with a radius (motion_guard.py)."""
    shape = env.arm_shape()
    if shape is None:
        return
    assert len(shape) >= 2, "arm_shape() needs at least the base and the tip"
    for point, radius in shape:
        assert np.asarray(point, float).shape == (3,) and float(radius) > 0, \
            f"arm_shape() entry {(point, radius)!r} is not ((x, y, z), radius)"


def check_force(env) -> None:
    """The arm's contact limits are ones safety knows, and positive; zero_contact_force() leaves a finite reading."""
    from mrcf.robot_env import FORCE_LIMIT_KEYS

    limits = dict(env.contact_force_limits)
    unknown = set(limits) - set(FORCE_LIMIT_KEYS)
    assert not unknown, f"contact_force_limits has {sorted(unknown)}, which safety does not read ({FORCE_LIMIT_KEYS})"
    assert all(float(v) > 0 for v in limits.values()), f"contact_force_limits {limits} must all be above 0 N"
    env.zero_contact_force()
    force = float(env.get_contact_force())
    assert np.isfinite(force) and force >= 0.0, f"get_contact_force() gave {force} after zero_contact_force()"


def check_frames(env) -> None:
    """link_frames(): None, or each link's 4x4 frame (a rotation, finite metres), base first."""
    frames = env.link_frames()
    if frames is None:
        return
    assert len(frames) >= 2, "link_frames() needs at least the base and one link"
    for i, T in enumerate(frames):
        T = np.asarray(T, float)
        assert T.shape == (4, 4) and np.isfinite(T).all(), f"link_frames()[{i}] is not a finite 4x4"
        assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6) and np.allclose(T[3], [0, 0, 0, 1]), \
            f"link_frames()[{i}] is not a rigid transform"


def check_poses(env) -> None:
    """go_home() and go_park() run and say whether they got there (or nothing, for an arm with no such pose)."""
    for name in ("go_home", "go_park"):
        got = getattr(env, name)()
        assert got is None or isinstance(got, (bool, np.bool_)), f"{name}() gave {got!r}"


def check_backend(make_env, reachable, unreachable=(), gripper_widths=(0.02, 0.0), gripper_tol: float = 0.003,
                  move_tol: float = 0.01) -> list:
    """
    All of the above, each on a fresh env from make_env(). reachable: points the
    grasp point can reach, the first also used for the stop, pause and thread
    checks; unreachable: points it cannot. Returns the env's capabilities.
    """
    goal = reachable[0]
    check_lifecycle(make_env)
    check_sensing(make_env())
    check_signatures(make_env())
    check_moves(make_env(), reachable, unreachable, tol=move_tol)
    check_gripper(make_env(), gripper_widths, tol=gripper_tol)
    check_stops(make_env, goal)
    check_pauses(make_env, goal)
    check_cameras(make_env())
    check_sensors(make_env())
    check_threads(make_env, goal)
    if isinstance(make_env(), capabilities.ActionStreaming):
        check_streaming(make_env)
    check_arm_shape(make_env())
    check_force(make_env())
    check_frames(make_env())
    check_poses(make_env())
    env = make_env()
    return [name for name in capabilities.__all__ if isinstance(env, getattr(capabilities, name))]
