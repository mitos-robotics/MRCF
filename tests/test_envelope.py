"""
mrcf/envelope.py: the box the whole gripper stays inside, checked along a move
before it is sent; and the UR5 in memory refusing what would take it out.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")

from mrcf import backends  # noqa: E402
from mrcf.envelope import GripperBox, joint_path, linear_path  # noqa: E402

BOX = GripperBox.from_config(((0.0, 0.75), (0.0, 0.75), (0.0, 0.80)), tip_m=0.157, radius_m=0.05)
DOWN = np.diag([1.0, -1.0, -1.0])            # the flange's z straight down


def flange(xyz, R=DOWN):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, xyz
    return T


def test_inside_is_inside_and_each_wall_says_which():
    assert BOX.outside(flange((0.35, 0.35, 0.30)))[0] == 0.0
    out, why = BOX.outside(flange((0.80, 0.35, 0.30)))
    assert out > 0 and "x > +0.75" in why
    out, why = BOX.outside(flange((0.35, 0.02, 0.30)))           # its body, 5 cm round it, through the wall
    assert out == pytest.approx(0.03 * 2, abs=1e-6) and "y < +0.00" in why
    out, why = BOX.outside(flange((0.35, 0.35, 0.78)))
    assert "ceiling" in why


def test_the_tips_may_come_down_to_the_table_and_no_further():
    assert BOX.outside(flange((0.35, 0.35, 0.157 + 0.004)))[0] == 0.0     # tips 4 mm up
    out, why = BOX.outside(flange((0.35, 0.35, 0.157 - 0.002)))           # tips 2 mm into it
    assert out == pytest.approx(0.005, abs=1e-6) and "too low" in why


def test_leaning_brings_the_body_lower_than_the_tips():
    lean = np.array([[1, 0, 0], [0, -np.cos(0.6), np.sin(0.6)], [0, -np.sin(0.6), -np.cos(0.6)]])
    # Tips 10 mm above the table both ways; leaning, the body's edge is lower.
    upright = BOX.outside(flange((0.35, 0.35, 0.157 + 0.010)))[0]
    leaning = BOX.outside(flange((0.35, 0.35, 0.157 * np.cos(0.6) + 0.010), lean))[0]
    assert upright == 0.0 and leaning > 0.0


def test_a_move_out_is_refused_and_one_back_in_is_not():
    inside, outside = flange((0.35, 0.35, 0.30)), flange((0.35, -0.20, 0.30))
    assert BOX.refuses(linear_path(inside, outside)) is not None
    assert BOX.refuses(linear_path(outside, inside)) is None            # put out there by hand: back in is fine
    assert BOX.refuses(linear_path(outside, flange((0.35, -0.30, 0.30)))) is not None   # not further out
    assert BOX.refuses(linear_path(outside, flange((0.55, -0.20, 0.30)))) is not None   # nor along it, outside
    assert BOX.refuses(linear_path(inside, flange((0.50, 0.50, 0.25)))) is None


def test_paths_run_from_start_to_end():
    a, b = flange((0.1, 0.2, 0.3)), flange((0.4, 0.5, 0.2))
    path = linear_path(a, b, n=10)
    assert np.allclose(path[0], a) and np.allclose(path[-1], b) and len(path) == 10
    from mrcf.backends.ur5.kinematics import fk
    q0, q1 = np.zeros(6), np.radians([30, -20, 10, 0, 0, 0])
    path = joint_path(q0, q1, fk)
    assert np.allclose(path[0], fk(q0)) and np.allclose(path[-1], fk(q1)) and len(path) >= 3


# ── On the UR5 in memory ──────────────────────────────────────────────────────

IN_BOX_Q_DEG = (212.3, -90.5, 87.8, -87.3, -90.0, -57.7)      # pointing down over (0.35, 0.35, 0.45)


@pytest.fixture
def env(monkeypatch):
    from mrcf import overhead_calibration
    from mrcf.backends.ur5.arm import URArm
    from mrcf.backends.ur5.offline import OfflineController, OfflineGripper, connect_offline
    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    ur5 = backends.arm("ur5")
    rc = ur5.config
    arm = URArm(None, gripper=OfflineGripper(rc.GRIPPER_MAX_OPEN_M),
                connect_with=connect_offline(OfflineController(q_deg=IN_BOX_Q_DEG)), **rc.ARM_TUNING)
    arm.connect()
    e = ur5.make_env(sdk=arm, sources={}, motion_guard_zone=None,       # no camera: no sentry to watch with
                     speed_m_s=4.0, joint_speed_rad_s=6.0, control_hz=1000.0,
                     gripper_box=GripperBox.from_config(((0.0, 0.75), (0.0, 0.75), (0.0, 0.80)),
                                                        rc.JAW_CLOSED_POINT_M[2] + rc.JAW_TIP_BELOW_GRASP_M))
    e.connect()
    yield e
    e.stop()


def test_the_rig_config_makes_the_box(monkeypatch):
    from mrcf import overhead_calibration
    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    ur5 = backends.arm("ur5")
    e = ur5.make_env(host=None)
    box = e._gripper_box
    assert box is not None and box.x == tuple(ur5.config.GRIPPER_BOX[0])
    assert box.tip_m == pytest.approx(ur5.config.JAW_CLOSED_POINT_M[2] + ur5.config.JAW_TIP_BELOW_GRASP_M)


def test_the_ur5_will_not_take_the_gripper_out_of_its_box(env, capsys):
    start = env.get_ee_pos().copy()
    assert env.move_ee_to(np.array([0.40, 0.40, 0.20])) is True               # inside: goes
    assert env.move_ee_to(np.array([0.40, -0.10, 0.20])) is False             # through the y = 0 wall: not sent
    assert np.linalg.norm(env.get_ee_pos() - np.array([0.40, 0.40, 0.20])) < 0.01
    assert "[Box] Not moving" in capsys.readouterr().out
    assert env.move_ee_to(np.array([0.40, 0.40, 0.157 - 0.146 - 0.005])) is False   # tips into the table
    assert np.linalg.norm(start - env.get_ee_pos()) > 0.05                    # it did move, the first time


def test_a_joint_move_that_swings_the_gripper_out_is_refused(env):
    q = np.radians(IN_BOX_Q_DEG)
    swung = q.copy()
    swung[0] -= np.radians(120)                       # the base turned a third of the way round
    before = np.asarray(env._sdk.joints()).copy()
    assert env.move_joints_to(swung) is False
    assert np.allclose(np.asarray(env._sdk.joints()), before)


def test_a_streamed_step_out_is_held_back(env):
    here = env.get_ee_pos()
    # The grasp point 3 cm from the y = 0 wall, inside the workspace; the gripper's body through it.
    sent, why = env.stream_action(np.r_[here[0], 0.03, here[2], 0.02])
    assert why is not None and "box" in why and np.allclose(sent[:3], here, atol=1e-6)


def test_the_gripper_keeps_clear_of_the_arms_base():
    """The box reaches in to the base; the gripper's body must not (its housing, the arm folding onto itself)."""
    import numpy as np

    from mrcf.envelope import GripperBox

    box = GripperBox((0.0, 0.75), (0.0, 0.75), (0.0, 0.8), tip_m=0.157, radius_m=0.05, base_clear_m=0.15)

    def down_at(x, y, z=0.3):
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])               # pointing straight down
        T[:3, 3] = (x, y, z)
        return T

    assert box.outside(down_at(0.35, 0.40))[0] == 0.0          # home's
    out, why = box.outside(down_at(0.10, 0.10))
    assert out > 0 and "base" in why
    assert box.refuses([down_at(0.35, 0.40), down_at(0.10, 0.10)]) is not None
