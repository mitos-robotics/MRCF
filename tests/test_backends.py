"""
The arms: which one runs, what every arm's package exports, and the RobotEnv
contract (mrcf/backends/contract.py) on the test arm and on an offline SO-101.
"""

import numpy as np
import pytest

from mrcf import backends
from mrcf.backends.contract import check_backend
from fake_arm import FakeArm


def test_the_arm_is_the_so101_unless_mrcf_arm_says_otherwise(monkeypatch):
    monkeypatch.delenv(backends.ENV_VAR, raising=False)
    assert backends.arm_name() == "so101" and backends.arm().NAME == "SO-101"
    monkeypatch.setenv(backends.ENV_VAR, "so101")
    assert backends.arm() is backends.arm("so101")


def test_an_arm_that_is_not_there_says_which_are(monkeypatch):
    monkeypatch.setenv(backends.ENV_VAR, "ur5_not_yet")
    with pytest.raises(ModuleNotFoundError, match="No arm 'ur5_not_yet'.*so101"):
        backends.arm()


@pytest.mark.parametrize("name", backends.available())
def test_every_arm_exports_the_same_interface(name):
    arm = backends.arm(name)
    assert isinstance(arm.NAME, str) and callable(arm.make_env) and callable(arm.make_policies)
    assert backends.missing_rig_settings(arm.config) == []


@pytest.mark.parametrize("name", backends.available())
def test_the_policy_plans_for_the_jaws_the_arm_turns(name):
    """2026-11-12: the UR5 turned its jaws "tangential" and its policy planned room for "radial"."""
    arm = backends.arm(name)
    spec = getattr(arm.make_policies(), "spec", None)
    if spec is not None:
        assert spec.jaw_axis == getattr(arm.config, "JAW_AXIS", "radial")


class _Unguarded(FakeArm):
    def guard(self):
        pass                                             # moves without asking the interlock


class _Silent(FakeArm):
    def move_ee_to(self, target, tol=0.01, max_iters=60, step_size=0.0, max_step_deg=None, orientation=None):
        super().move_ee_to(target, tol, max_iters, step_size, max_step_deg, orientation)   # arrived? says nothing


class _NoGentleClose(FakeArm):
    def set_gripper(self, width: float, n_steps: int = 15) -> None:
        super().set_gripper(width, n_steps)


class _WrongSize(FakeArm):
    def list_cameras(self):
        return ["overhead"]

    def render_image(self, camera, width=640, height=480):
        return np.zeros((480, 640, 3), np.uint8)         # whatever size was asked for


class _Impatient(FakeArm):
    def guard(self):
        self.interlock.check()                           # a pause raises, as a stop does


@pytest.mark.parametrize("arm, says", [
    (_Unguarded, "guard"), (_Silent, "did not arrive"), (_NoGentleClose, "max_step_pct"),
    (_WrongSize, "render_image"), (_Impatient, "paused move"),
])
def test_the_contract_catches_an_arm_that_breaks_it(arm, says):
    with pytest.raises(AssertionError, match=says):
        check_backend(lambda: arm(cycle_s=0.0005), reachable=[(0.25, 0.0, 0.08)], unreachable=[(2.0, 0.0, 0.1)])


def test_the_test_arm_keeps_the_contract():
    found = check_backend(lambda: FakeArm(cycle_s=0.0005), reachable=[(0.25, 0.0, 0.08), (0.18, 0.05, 0.03)],
                          unreachable=[(2.0, 0.0, 0.1)])
    assert "GripSensing" in found


def _offline_so101():
    """The real SO101Env — move_ee_to, the gripper loop, guard and hold — on a perfect servo and point kinematics."""
    from mrcf.backends.so101.kinematics import JOINT_NAMES
    from test_so101_env import _perfect_servo, _PointKinematics, offline_so101

    env = offline_so101()
    env._init_rig(camera_name="front", gripper_max_open_m=0.045, tcp_offset=np.zeros(3))
    env._kinematics = _PointKinematics()                 # joints 1-3: the grasp point's x, y, z in mm
    env._robot.send_action = _perfect_servo(env._robot)
    for n in JOINT_NAMES:
        env._robot.bus.positions[n] = 0.0
    env._joint_deg_limits = {n: (-500.0, 500.0) for n in JOINT_NAMES}    # half a metre each way
    env._ik_limits = dict(env._joint_deg_limits)
    env._r_down, env._pan_centre, env._jaw_axis = np.eye(3), np.zeros(2), "radial"
    env._max_relative_target_deg = 8.0
    env._home_degrees = {n: 0.0 for n in JOINT_NAMES}
    env._gripper_width_table, env._gripper_min_hold_m, env._gripper_torque_limit = None, 0.0034, None
    env._servo_acceleration = env._servo_p_coefficient = env._servo_i_coefficient = env._servo_d_coefficient = None
    env._stream_tip_floor_z, env._stream_bounds = None, None           # its stream: no floor or bounds here
    env.arm_shape = lambda: None                         # the real one needs the URDF (test_so101_env has it)
    return env


def test_the_so101_keeps_the_contract():
    found = check_backend(_offline_so101, reachable=[(0.25, 0.0, 0.08), (0.18, 0.05, 0.03)],
                          unreachable=[(2.0, 0.0, 0.1)])
    from mrcf import capabilities
    assert found == [c for c in capabilities.__all__ if c not in ("Freedrive", "BackOff")]  # torque off, it falls; back-off: UR5 so far


# ── backends/wiring.py: an arm's settings, by their shared names, reach its env and policies ──

def _arms():
    return backends.available() + ["_template"]


def _concrete(cls, **attrs):
    """cls with its abstract methods stubbed out, to construct for what it does itself."""
    stubs = {name: (lambda self, *a, **k: None) for name in getattr(cls, "__abstractmethods__", ())}
    return type(f"Concrete{cls.__name__}", (cls,), {**stubs, **attrs})


@pytest.mark.parametrize("name", _arms())
def test_every_gripper_setting_an_arm_defines_reaches_its_policies(name, monkeypatch):
    """The UR5's make_policies once copied its settings by hand and missed JAW_AXIS (2026-11-12)."""
    from mrcf import overhead_calibration
    from mrcf.backends import wiring

    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    arm = backends.arm(name)
    spec = getattr(arm.make_policies(), "spec", None)
    assert spec is not None
    for field, setting in wiring.SPEC_SETTING_FOR.items():
        if hasattr(arm.config, setting):
            want = getattr(arm.config, setting)
            assert getattr(spec, field) == (tuple(want) if isinstance(want, list) else want), (field, setting)
    calibrated = getattr(arm.config, "GRIPPER_CAM_K", None) is not None
    assert spec.camera == (arm.config.CAMERA_NAME if calibrated else None)


def test_the_rig_keywords_are_what_the_config_defines():
    from types import SimpleNamespace

    from mrcf.backends import wiring

    cfg = SimpleNamespace(CAMERA_NAME="wrist", TABLE_Z=0.02, JAW_SLIDE_AXIS=(0, 1, 0),
                          MOTION_GUARD=False, MOTION_GUARD_ZONE=[(0, 0), (1, 0), (1, 1)])
    got = wiring.rig_kwargs(cfg)
    assert got == {"camera_name": "wrist", "work_plane_z": 0.02, "jaw_slide_axis": (0, 1, 0),
                   "motion_guard_zone": None,                  # the sentry off: no zone; nothing else made up
                   "unchecked_facts": ("JAW_SLIDE_AXIS",)}     # a guess until checked on the rig (arm_facts)
    cfg.JAW_SLIDE_AXIS_CHECKED = True
    assert wiring.rig_kwargs(cfg)["unchecked_facts"] == ()
    cfg.MOTION_GUARD = True
    assert wiring.rig_kwargs(cfg)["motion_guard_zone"] == cfg.MOTION_GUARD_ZONE
    cfg.GRIPPER_BOX = ((0.0, 0.5), (0.0, 0.5), (0.0, 0.6))
    cfg.JAW_CLOSED_POINT_M, cfg.JAW_TIP_BELOW_GRASP_M = (0, 0, 0.15), 0.01
    assert wiring.rig_kwargs(cfg)["gripper_box"].tip_m == pytest.approx(0.16)


def test_an_arm_whose_moves_do_not_check_a_box_is_not_given_one():
    from mrcf import envelope
    from mrcf.backends.rig import RigEnv

    Unchecked = _concrete(RigEnv)
    box = envelope.GripperBox.from_config(((0.0, 0.5), (0.0, 0.5), (0.0, 0.6)), 0.16)
    with pytest.raises(ValueError, match="do not check a gripper box"):
        Unchecked()._init_rig(gripper_box=box)
    Unchecked()._init_rig()                                   # no box: as before


def test_straight_down_the_jaws_open_along_the_arms_jaw_axis_whatever_way_they_are_mounted():
    from mrcf.backends.rig import RigEnv

    Arm = _concrete(RigEnv, base_xy=np.zeros(2))
    for slide in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)):
        for axis in ("tangential", "radial"):
            env = Arm()
            env._init_rig(jaw_slide_axis=slide)
            env._jaw_axis = axis
            for at in ((0.4, 0.0), (0.3, 0.3), (-0.1, 0.5)):
                r = env.down_rotation(at)
                assert np.allclose(r.T @ r, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)
                assert np.allclose(r[:, 2], (0, 0, -1))                      # straight down
                jaw, out = r @ env._jaw_slide_axis, np.asarray(at) / np.linalg.norm(at)
                along = abs(float(jaw[:2] @ out))
                assert along == pytest.approx(0.0 if axis == "tangential" else 1.0, abs=1e-9), (slide, axis, at)


def test_the_other_calibrated_fixed_cameras_are_the_views_a_scan_keeps():
    from mrcf.backends.rig import Camera, RigEnv

    env = _concrete(RigEnv)()
    model = {"size": [1280, 720], "K": np.eye(3).tolist(), "D": [0.0] * 5, "T_world_cam": np.eye(4).tolist()}
    env._init_rig(overhead_camera_name="overhead", cameras={
        "overhead": Camera(mount="fixed", source=0, size=(1280, 720), model=model),   # its own calibration elsewhere
        "side": Camera(mount="fixed", source=1, size=(1280, 720), model=model),
        "door": Camera(mount="fixed", source=2, size=(640, 480)),                      # not calibrated
        "wrist": Camera(mount="gripper", source=3, size=(640, 480)),                   # moves with the jaws
    })
    assert env.fixed_views() == {"side": (1280, 720)}
