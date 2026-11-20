"""tools/so101/rest.py: the rest pose it folds down to is chosen by what stands on the table, its path checked first."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("placo")


@pytest.fixture(scope="module")
def arm():
    from mrcf.backends.so101 import robot_config as rc
    if not Path(rc.URDF_PATH).is_file():
        pytest.skip("no URDF — run fetch_urdf.py")
    from tools.so101.visualise_ik import CAL_PATH, offline_arm
    if not CAL_PATH.exists():
        # The rest poses are checked against this arm's own joint ranges:
        # there is none off the rig (CI).
        pytest.skip(f"no arm calibration at {CAL_PATH} — run lerobot-calibrate")
    return offline_arm()


def _home():
    from mrcf.backends.so101 import robot_config as rc
    return dict(rc.HOME_DEGREES)


def test_with_nothing_on_the_table_it_rests_in_the_first_pose(arm, capsys):
    from tools.so101 import rest
    assert rest.choose(arm, _home(), []) == next(iter(rest.REST_POSES))
    assert "passes beside nothing" in capsys.readouterr().out


def test_an_object_beside_a_poses_path_sends_it_to_the_next_that_clears_it(arm):
    # 2026-11-03: straight ahead came down beside the e1 king below its top.
    from tools.so101 import rest
    first, second = list(rest.REST_POSES)[:2]
    where = arm._grasp_point(np.append(rest._as_array(rest.REST_POSES[first]), 0.0), rest._GAP_M)[:2]
    king = (where + [0.02, 0.0], 0.014, 0.082, "the king")
    assert not rest.judge(arm, _home(), first, [king])[0]
    assert rest.choose(arm, _home(), [king]) == second


def test_the_base_turns_at_home_height_before_the_arm_folds(arm):
    # All at once, the joints at one rate, the fold finishes before the turn and
    # the gripper sweeps low past what stands between the start and the rest.
    from tools.so101 import rest
    pose = rest.REST_POSES["right"]
    path = rest.rest_path(_home(), pose)
    pans = np.array([q[0] for q in path])
    turned = int(np.argmax(np.isclose(pans, pose["shoulder_pan"], atol=0.3)))
    heights = [arm._grasp_point(np.append(q, 0.0), rest._GAP_M)[2] for q in path[:turned + 1]]
    assert min(heights) > 0.15                                   # turned while still high
    wake = rest.wake_path(dict(pose))
    lifted = [arm._grasp_point(np.append(q, 0.0), rest._GAP_M)[2] for q in wake]
    pan_moves = [i for i in range(1, len(wake)) if abs(wake[i][0] - wake[i - 1][0]) > 1e-6]
    assert pan_moves and lifted[pan_moves[0]] > 0.15            # unfolded before it turns back


def test_every_rest_pose_leaves_its_joints_clear_of_their_limits(arm):
    # A servo parked hard against its stop is still developing torque.
    from tools.so101 import rest
    for name, pose in rest.REST_POSES.items():
        assert set(pose) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}, name
        assert rest.joint_margin(arm, pose) >= rest.MIN_MARGIN_DEG, name



def test_a_kept_containers_walls_are_kept_clear_of_like_objects_as_tall(tmp_path, capsys):
    # 2026-11-19: the rest folded the arm into the tub, which the tool knew nothing of.
    import json
    from types import SimpleNamespace

    from tools.so101 import rest
    (tmp_path / "rims.json").write_text(json.dumps({
        "plastic_tub": {"rim_m": 0.074, "outline_xy": [[0.20, 0.10], [0.30, 0.10], [0.30, 0.20], [0.20, 0.20]],
                        "seen": "2026-11-19 16:40"},
        "tray": {"rim_m": 0.03}}))                                          # never seen: no outline to keep clear of
    walls = rest._container_walls(SimpleNamespace(CONTAINERS_PATH=tmp_path / "rims.json"))
    assert len(walls) >= 4 * int(0.10 / rest.WALL_STEP_M)
    assert all(label == "plastic_tub's wall" and h == pytest.approx(0.084) for _, _, h, label in walls)
    for xy, _, _, _ in walls:
        on_edge = min(abs(xy[0] - 0.20), abs(xy[0] - 0.30), abs(xy[1] - 0.10), abs(xy[1] - 0.20))
        assert on_edge < 1e-9
    assert "plastic_tub: its walls, 74 mm" in capsys.readouterr().out
    assert rest._container_walls(SimpleNamespace()) == []                   # a deployment with no containers
    assert rest._container_walls(SimpleNamespace(CONTAINERS_PATH=tmp_path / "none.json")) == []


def test_the_tubs_wall_beside_a_poses_path_sends_it_to_another(arm, tmp_path):
    import json
    from types import SimpleNamespace

    from tools.so101 import rest
    first, second = list(rest.REST_POSES)[:2]
    where = arm._grasp_point(np.append(rest._as_array(rest.REST_POSES[first]), 0.0), rest._GAP_M)[:2]
    box = [[where[0] - 0.06, where[1] - 0.06], [where[0] + 0.06, where[1] - 0.06],
           [where[0] + 0.06, where[1] + 0.06], [where[0] - 0.06, where[1] + 0.06]]
    (tmp_path / "rims.json").write_text(json.dumps({"plastic_tub": {"rim_m": 0.074, "outline_xy": box}}))
    walls = rest._container_walls(SimpleNamespace(CONTAINERS_PATH=tmp_path / "rims.json"))
    assert not rest.judge(arm, _home(), first, walls)[0]
    assert rest.choose(arm, _home(), walls) != first



def test_a_rest_keeps_clear_of_the_table_as_it_is_now_not_as_last_scanned(capsys):
    """2026-11-19 18:20: the scoop and a figure moved off the left rest's path, and every rest refused for them."""
    from tools.so101 import rest
    scoop = (np.array([0.162, 0.138]), 0.028, 0.05, "blue_scoop")
    figure = (np.array([0.181, 0.061]), 0.011, 0.045, "red_figure")
    plug = (np.array([0.284, 0.040]), 0.019, 0.06, "white_plug")
    moved_scoop = np.array([0.33, 0.15])
    seen = ([(np.array([0.286, 0.041]), 0.019), (moved_scoop, 0.028)], lambda xy: False)
    now = rest.on_the_table([scoop, figure, plug], seen, tallest=0.10)
    labels = [o[3] for o in now]
    assert "blue_scoop" not in labels and "red_figure" not in labels           # gone from where they were
    plug_now = next(o for o in now if o[3] == "white_plug")
    assert np.allclose(plug_now[0], (0.286, 0.041)) and plug_now[2] == 0.06    # where it is seen, its own height
    new = next(o for o in now if o[3].startswith("something at"))
    assert np.allclose(new[0], moved_scoop) and new[2] == 0.10                 # seen, not known: as tall as any
    out = capsys.readouterr().out
    assert "blue_scoop: not there now" in out and "red_figure: not there now" in out


def test_what_the_arm_hides_from_the_camera_is_kept_as_remembered():
    from tools.so101 import rest
    figure = (np.array([0.181, 0.061]), 0.011, 0.045, "red_figure")
    under_arm = ([], lambda xy: bool(np.linalg.norm(np.asarray(xy) - (0.18, 0.06)) < 0.03))
    assert [o[3] for o in rest.on_the_table([figure], under_arm, tallest=0.10)] == ["red_figure"]
    assert rest.on_the_table([figure], None, tallest=0.10) == [figure]          # no look: the memory as it is


def test_with_the_table_cleared_the_left_rest_is_chosen_again(arm):
    from tools.so101 import rest
    left = rest.REST_POSES["left"]
    where = arm._grasp_point(np.append(rest._as_array(left), 0.0), rest._GAP_M)[:2]
    on_its_path = (where + [0.03, 0.0], 0.02, 0.05, "blue_scoop")
    assert not rest.judge(arm, _home(), "left", [on_its_path])[0]
    cleared = rest.on_the_table([on_its_path], ([], lambda xy: False), tallest=0.10)
    assert rest.judge(arm, _home(), "left", cleared)[0]


def test_a_rest_without_the_rigs_cameras_connects_without_them(monkeypatch, capsys):
    """A rest must not depend on a camera: the wrist camera drops off the bus, and cameras get unplugged."""
    from mrcf import devices
    from tools.so101 import rest
    made = []

    class Env:
        def __init__(self, kw):
            self.kw = kw

        def connect(self):
            if self.kw.get("cameras") != {}:
                raise devices.PersonalCamera("2 rig camera(s) to open (front, overhead) but only 0 plugged in; ...")

    monkeypatch.setattr(rest, "make_env", lambda **kw: made.append(kw) or Env(kw))
    env = rest.connected({"floor_z": 0.0007})
    assert env.kw["cameras"] == {} and env.kw["camera_name"] is None and env.kw["overhead_camera_name"] is None
    assert env.kw["floor_z"] == 0.0007 and len(made) == 2
    assert "connecting without them, and checking against the last scan" in capsys.readouterr().out
