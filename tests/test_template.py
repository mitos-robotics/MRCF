"""
mrcf/backends/_template, the arm a new arm is copied from: it keeps the backend
contract, has every rig setting, and copies as it is — so a new arm starts
from one that works.
"""

import shutil
import time
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from mrcf import backends  # noqa: E402
from mrcf import overhead_calibration  # noqa: E402
from mrcf.backends.contract import check_backend  # noqa: E402


@pytest.fixture
def template(monkeypatch):
    # Its config points overhead_calibration at its own file as it is imported,
    # as every arm's does; put that back for the tests after.
    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    return backends.arm("_template")


def test_the_template_defines_every_rig_setting(template):
    assert backends.missing_rig_settings(template.config) == []


def test_the_template_is_not_an_arm_until_it_is_copied():
    assert "_template" not in backends.available()


def test_the_template_keeps_the_contract(template):
    found = check_backend(lambda: template.make_env(speed_m_s=4.0, control_hz=1000.0),
                          reachable=[(0.40, 0.10, 0.05), (0.25, -0.20, 0.12)], unreachable=[(1.2, 0.0, 0.1)])
    assert {"OverheadSighting", "WristServo", "GripSensing", "HeldWidth", "OverheadCalibration",
            "ActionStreaming"} <= set(found)


def test_a_learned_policy_runs_on_the_templates_stream(template):
    from mrcf.policies.learned import CallableModel, LearnedPolicy

    env = template.make_env(speed_m_s=4.0, control_hz=1000.0)
    spec = env.stream_spec
    assert spec.state == spec.action == ("x", "y", "z", "gripper") and spec.robot_type == template.config.ROBOT_TYPE
    goal = np.array([0.40, 0.05, template.config.TABLE_Z + 0.10])

    def toward(observation):                 # a stand-in model: 5 mm a tick toward goal, jaws 2 cm open
        here = np.asarray(observation["observation.state"][:3], float)
        gap = goal - here
        return np.r_[here + gap * min(1.0, 0.005 / max(np.linalg.norm(gap), 1e-9)), 0.02]

    done = lambda env, observation: np.linalg.norm(observation["observation.state"][:3] - goal) < 0.001  # noqa: E731
    got = LearnedPolicy(CallableModel(toward), name="reach", until=done, max_s=10).run(env)
    assert got.ok, got.message
    assert np.linalg.norm(env.get_ee_pos() - goal) < 0.001 and env.get_gripper_width() == pytest.approx(0.02)


def test_the_templates_stream_holds_back_a_goal_into_the_table(template):
    from mrcf.policies.learned import CallableModel, LearnedPolicy

    env = template.make_env(speed_m_s=4.0, control_hz=1000.0)
    start = env.get_ee_pos()
    below = np.r_[start[:2], template.config.TABLE_Z - 0.05, 0.0]
    sent, why = env.stream_action(below)
    assert "outside the workspace" in why and np.allclose(sent[:3], start) and np.allclose(env.get_ee_pos(), start)
    got = LearnedPolicy(CallableModel(lambda observation: below), name="dive", max_held_back_s=0.2, max_s=5).run(env)
    assert not got.ok and "held back" in got.message and np.allclose(env.get_ee_pos(), start)
    # Parked outside the workspace, the arm can still be streamed back towards it.
    env.move_ee_to(np.array([0.15, -0.35, 0.30]))
    inward = np.r_[env.get_ee_pos() + [0.01, 0.0, 0.0], 0.0]
    assert env.stream_action(inward)[1] is None and env.get_ee_pos()[0] == pytest.approx(inward[0], abs=1e-6)


def test_a_pick_on_the_template_runs_to_an_outcome(template):
    from mrcf.policies.base import Outcome, Target

    env = template.make_env(speed_m_s=4.0, control_hz=1000.0)
    got = template.make_policies().pick(env, Target(np.array([0.40, 0.05, template.config.TABLE_Z]), foot_m=0.02,
                                                    max_height_m=0.05))
    assert isinstance(got, Outcome) and not got.ok              # nothing there to hold: a miss, not an error


def test_a_copy_of_the_template_needs_no_edits(tmp_path, monkeypatch):
    # Relative imports only, so `cp -r mrcf/backends/_template mrcf/backends/my_arm` runs.
    for f in (ROOT / "mrcf" / "backends" / "_template").glob("*.py"):
        assert "backends._template" not in f.read_text(encoding="utf-8"), f"{f.name} names the template's own package"
    pkg = tmp_path / "mrcf"                    # the whole package, copied, the copy beside the template
    shutil.copytree(ROOT / "mrcf", pkg, ignore=shutil.ignore_patterns("so101*", "__pycache__", "urdf"))
    shutil.copytree(ROOT / "mrcf" / "backends" / "_template", pkg / "backends" / "my_arm",
                    ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    for name in [m for m in sys.modules if m == "mrcf" or m.startswith("mrcf.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    from mrcf import backends as copied

    assert "my_arm" in copied.available()
    arm = copied.arm("my_arm")
    assert arm.NAME == "My Arm" and copied.missing_rig_settings(arm.config) == []
    assert arm.make_env().move_ee_to(np.array([0.30, 0.0, 0.10]))


def test_the_template_takes_the_overhead_capture_size_the_calibration_asks_for(template):
    # tools/calibrate_overhead_camera.py asks every arm's make_env for 1280x720,
    # as the SO-101's takes it; each camera keeps a size of its own.
    env = template.make_env(overhead_camera_width=1280, overhead_camera_height=720, overhead_camera_fps=30)
    rc = template.config
    assert env._capture_sizes == {rc.CAMERA_NAME: (rc.CAMERA_W, rc.CAMERA_H), rc.OVERHEAD_CAMERA_NAME: (1280, 720)}


# ── The template in three rigs: no wrist camera, a third camera, a force sensor ─

def _side():
    from mrcf.backends.rig import Camera

    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.diag([1.0, -1.0, -1.0]), [0.4, 0.0, 0.9]
    return Camera(mount="fixed", size=(1280, 720), homography=np.diag([0.001, 0.001, 1.0]),
                  model={"size": (1280, 720), "K": [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]], "D": [0.0] * 5,
                         "T_world_cam": T})


def test_a_rig_with_no_wrist_camera_keeps_the_contract_and_picks(template):
    from mrcf.policies.base import Outcome, Target

    overhead = {"overhead": template.config.CAMERAS["overhead"]}
    make = lambda: template.make_env(camera_name=None, cameras=overhead, speed_m_s=4.0, control_hz=1000.0)  # noqa: E731
    found = check_backend(make, reachable=[(0.40, 0.10, 0.05)], unreachable=[(1.2, 0.0, 0.1)])
    assert "OverheadSighting" in found
    assert make().camera_model("wrist", 640, 480) is None
    got = template.make_policies(camera=None).pick(make(), Target(np.array([0.40, 0.05, template.config.TABLE_Z]),
                                                                  foot_m=0.02, max_height_m=0.05))
    assert isinstance(got, Outcome) and not got.ok


def test_a_rig_with_a_third_camera_has_its_geometry_and_is_watched_through(template, monkeypatch):
    import types

    started = []

    class Guard:
        def __init__(self, env, camera, zone=None, name="motion"):
            started.append((camera, name))

        def start(self):
            return self

        def stop(self):
            pass

    monkeypatch.setitem(sys.modules, "mrcf.motion_guard", types.SimpleNamespace(MotionGuard=Guard))
    cameras = {**template.config.CAMERAS, "side": _side()}
    env = template.make_env(cameras=cameras, motion_guard_cameras=["overhead", "side"],
                            motion_guard_zone=template.config.MOTION_GUARD_ZONE, speed_m_s=4.0, control_hz=1000.0)
    assert np.allclose(env.camera_model("side", 640, 360)["T_world_cam"][:3, 3], [0.4, 0.0, 0.9])
    assert np.allclose(env.pixel_to_world(100, 200, 0.0, "side"), [0.1, 0.2, 0.0])
    env.connect()
    env.stop()
    assert started == [("overhead", "motion"), ("side", "motion:side")]


def test_a_rig_with_a_force_sensor_reads_it_and_the_monitor_stops_on_it(template, monkeypatch):
    from mrcf import safety
    from mrcf.backends.rig import Sensor
    from mrcf.backends.contract import check_sensors

    sensors = {"wrench": Sensor(kind="force", unit="N, N·m", shape=(6,), read="tcp_wrench")}
    make = lambda: template.make_env(sensors=sensors, speed_m_s=4.0, control_hz=1000.0)  # noqa: E731
    check_backend(make, reachable=[(0.40, 0.10, 0.05)], unreachable=[(1.2, 0.0, 0.1)])
    env = make()
    assert env.list_sensors() == {"wrench": "force (N, N·m), shape (6,)"} and env.get_contact_force() == 0.0
    monkeypatch.setattr(env._sdk, "tcp_wrench", lambda: np.array([0.0, 0.0, -12.0, 0.0, 0.0, 0.0]))
    check_sensors(env)
    assert env.get_contact_force() == pytest.approx(12.0)
    monitor = safety.ExecutionMonitor(env, "test", thresholds=safety.SafetyThresholds(max_contact_force_n=5.0))
    assert monitor.check() is True


def test_a_gripper_contact_sensor_is_what_the_template_says_about_its_grip(template):
    from mrcf.backends.rig import Sensor

    touching = {"now": False}
    env = template.make_env(sensors={"pads": Sensor(kind="gripper_contact", read=lambda env: touching["now"])})
    assert env.gripper_loaded() is False
    touching["now"] = True
    assert env.gripper_loaded() is True


def test_a_controller_slow_to_say_it_is_moving_is_waited_for(template):
    # A UR's asynchronous moveL returns before the controller reports motion.
    env = template.make_env(speed_m_s=4.0, control_hz=1000.0)
    sdk, sent = env._sdk, {"t": float("-inf")}
    move, moving = sdk.move_linear, sdk.moving

    def late_move(pose, speed):
        sent["t"] = time.monotonic()
        move(pose, speed)

    sdk.move_linear = late_move
    sdk.moving = lambda: moving() if time.monotonic() - sent["t"] > 0.05 else False
    assert env.move_ee_to(np.array([0.40, 0.10, 0.05])) is True


def test_the_template_keeps_its_gripper_in_a_box_set_in_its_config(template, capsys):
    from mrcf.backends import wiring

    cfg = type("Config", (), {k: getattr(template.config, k) for k in dir(template.config) if k.isupper()})
    z = template.config.TABLE_Z
    cfg.GRIPPER_BOX = ((0.20, 0.50), (-0.30, 0.30), (z, z + 0.50))       # inside WORKSPACE_BOUNDS' x (0.20-0.60)
    env = template.make_env(speed_m_s=4.0, control_hz=1000.0, gripper_box=wiring.gripper_box(cfg),
                            motion_guard_zone=None)
    env.connect()
    try:
        assert env.move_ee_to((0.40, 0.10, 0.20)) is True
        assert env.move_ee_to((0.56, 0.10, 0.20)) is False                  # past the box's far side
        assert "[Box] Not moving" in capsys.readouterr().out
        assert np.linalg.norm(env.get_ee_pos() - (0.40, 0.10, 0.20)) < 0.01   # not sent: it stayed
        sent, why = env.stream_action([0.56, 0.10, 0.20, 0.02])              # a streamed step: held back too
        assert why is not None and "outside the workspace" not in why
        assert np.allclose(sent[:3], env.get_ee_pos(), atol=1e-6)
    finally:
        env.stop()


def test_the_templates_straight_down_moves_open_the_jaws_the_way_its_policies_plan_for(template):
    env = template.make_env(speed_m_s=4.0, control_hz=1000.0, jaw_axis="tangential", motion_guard_zone=None)
    env.connect()
    try:
        for at in ((0.40, 0.0, 0.10), (0.30, 0.25, 0.10), (0.30, -0.25, 0.10)):
            assert env.move_ee_to(at) is True
            jaw = env.gripper_frame()[:3, :3] @ env._jaw_slide_axis
            out = np.asarray(at[:2]) / np.linalg.norm(at[:2])
            assert abs(float(jaw[:2] @ out)) < 1e-6                              # across the line from the base
    finally:
        env.stop()
