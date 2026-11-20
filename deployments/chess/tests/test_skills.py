"""Chess skills, offline: grasp sizing through the arm's policies, the calculator, the tile prompt, plans."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import cv2  # noqa: F401 — the deployment's perception needs it
except ImportError:
    pytest.skip("needs OpenCV", allow_module_level=True)

from mrcf import manipulation
from mrcf.streaming import StreamSpec
from mrcf.vlm import SceneObject, SceneState

from deployments.chess import config as cfg
from deployments.chess import llm_controller
from deployments.chess import skills


class _Env:
    base_xy = np.zeros(2)
    frame_scale = 1.0

    def is_viewer_running(self):
        return True


class _StreamingEnv(_Env):
    """Six values in, six out, at once; on a clock of its own."""
    stream_spec = StreamSpec(fps=100, state=tuple("abcdef"), action=tuple("abcdef"))
    paused_s = 0.0

    def __init__(self):
        self.now = 0.0

    def motion_clock(self):
        self.now += 0.005
        return self.now

    def stream_observation(self):
        return {"observation.state": np.zeros(6, np.float32)}

    def stream_action(self, action):
        return np.asarray(action, np.float32), None

    def hold(self):
        pass


def piece(name, xy, foot_m=None):
    obj = SceneObject(name, name, position=np.array([xy[0], xy[1], cfg.TABLE_Z]))
    obj.geometry = {"base_diameter_m": foot_m} if foot_m else None
    return obj


def test_an_unmeasured_piece_is_gripped_by_its_measured_foot_and_how_slim_pieces_are_above_it():
    # A 21 mm pawn with another one square away along the jaw axis, leaving
    # 35 mm: room for a 21 mm foot plus slack (29 mm), not for the tallest
    # piece's 30 mm width plus slack (38 mm), which is what it was once sized by.
    # Gripped as high as the shortest piece allows, it is opened by the share
    # of its foot pieces are from there up (config.UNMEASURED_SHAPE_SHARE).
    pol = skills.POLICIES
    target = piece("white_piece_e4", (0.20, 0.0), foot_m=0.021)
    axis = pol.jaw_axis(_Env(), target.position)
    neighbour = piece("white_piece_f4", target.position[:2] + axis * 0.040, foot_m=0.021)
    state = SceneState(objects=[target, neighbour])
    plan, problem, room = pol.fit(_Env(), pol.plan(manipulation.target_of(target)), target.position[:2],
                                  manipulation.obstacles_around(state, target))
    assert problem is None, problem
    assert plan.grasp_height_m == pytest.approx(cfg.SHORTEST_PIECE_M - 0.010)
    assert plan.width_m == pytest.approx(0.87 * 0.021, abs=1e-4)
    assert plan.pregrasp_open_m <= room
    # Not measured, so hovered over as the tallest piece would need.
    assert plan.height_m == cfg.TALLEST_PIECE_M


def test_the_d1_queen_between_her_bishop_and_king_fits_before_anything_moves():
    # 2026-11-03, live: refused before moving ("c1 leaves 27 mm", 35 needed),
    # gripped as low as a king allows and every piece counted as its foot.
    # The scan's positions and feet; c1 as the wrist camera measured it.
    pol = skills.POLICIES
    queen = piece("white_piece_d1", (0.1544, 0.0291), foot_m=0.0266)
    bishop = piece("white_piece_c1", (0.1510, 0.0681), foot_m=0.0250)
    bishop.measured_shape = {"height_mm": 60.2, "widths_mm": {4: 25.9, 12: 24.4, 20: 21.5, 24: 20.8, 28: 20.5,
                                                               32: 19.7, 40: 18.8, 48: 15.6, 56: 9.9}}
    king = piece("white_piece_e1", (0.1580, -0.0130), foot_m=0.0270)
    state = SceneState(objects=[queen, bishop, king])
    env = _Env()
    plan, problem, room = pol.fit(env, pol.plan(manipulation.target_of(queen)), queen.position[:2],
                                  pol._with_base(env, manipulation.obstacles_around(state, queen)))
    assert problem is None, problem
    assert plan.width_m + 2 * pol.spec.jaw_slack_m <= room



def test_a_piece_the_user_named_is_planned_at_its_own_height(monkeypatch):
    # The classifier's guess is not trusted with the hover (it once called a
    # king a pawn), but the user's word is: an unmeasured pawn they named is
    # planned 45 mm tall, not as the king that sends the servo up to lose it.
    tracks = {"t1": SimpleNamespace(label="pawn", label_source="user"),
              "t2": SimpleNamespace(label="pawn", label_source="vlm")}
    monkeypatch.setattr(skills.board_log, "memory", lambda: SimpleNamespace(get=tracks.get))
    named, guessed, untracked = (piece("white_pawn_e4", (0.20, 0.0)), piece("white_piece_d4", (0.20, 0.04)),
                                 piece("white_piece_c4", (0.20, 0.08)))
    named.track_id, guessed.track_id = "t1", "t2"
    assert skills._max_height(named) == cfg.PIECE_SIZES_M["pawn"][0]
    assert skills._max_height(guessed) == cfg.TALLEST_PIECE_M
    assert skills._max_height(untracked) == cfg.TALLEST_PIECE_M

def test_calculate_refuses_runaway_factorials():
    assert skills.skill_calculate("factorial(5) + sqrt(16)") is True
    assert skills.skill_calculate("factorial(1000000)") is False
    assert skills.skill_calculate("__import__('os')") is False


def test_a_failed_retreat_updates_the_chess_square_after_release(monkeypatch):
    from mrcf.policies.base import Outcome
    records = []
    monkeypatch.setattr(skills.board_log, "record_pick", lambda *args, **kw: records.append((args, kw)))
    held = piece("white_piece_e4", (0.2, 0))
    held.square = "e4"
    target = piece("e5", (0.24, 0))
    skills._log_place(held, target, Outcome(False, "retreat failed", released=True))
    assert held.square == "e5"
    assert records[0][0][3] is False  # the action still failed, despite placing the piece


def test_the_chess_controller_runs_the_framework_loop_with_the_board_scan():
    d = llm_controller.DEPLOYMENT
    assert d.scan is skills.scan_scene and d.observe_pose is skills.goto_observe_pose
    assert d.look_around is skills.look_around and d.home == skills.POLICIES.home


def test_the_config_serves_the_arms_settings_live_never_a_copy(monkeypatch):
    # A scan that re-poses a moved camera updates robot_config; squares are
    # placed by cfg.OVERHEAD_HOMOGRAPHY, which must be that, not an import-time copy.
    from mrcf.backends.so101 import robot_config
    monkeypatch.setattr(robot_config, "OVERHEAD_HOMOGRAPHY", [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
    assert cfg.OVERHEAD_HOMOGRAPHY == [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]]
    assert "OVERHEAD_HOMOGRAPHY" not in vars(cfg)


def test_chess_runs_on_the_arm_it_names():
    # A deployment is one arm, one layout and one task: chess is the SO-101's
    # (ARM_NAME). It reads the rig settings every arm has, and the arm's own
    # settings reach the arm's tools through it too.
    from mrcf import backends
    assert cfg.ARM_NAME == "so101" and cfg.ARM is backends.arm("so101") and skills.POLICIES is not None
    for name in backends.RIG_SETTINGS:
        assert getattr(cfg, name) is getattr(cfg.ARM.config, name)
    assert cfg.FOLLOWER_PORT == cfg.ARM.config.FOLLOWER_PORT           # an SO-101 setting, served through
    assert "rig_settings_typo" not in dir(cfg)


def test_a_square_a_pick_empties_is_a_place_target_again_where_the_scan_put_it(monkeypatch):
    from deployments.chess import chessboard
    monkeypatch.setattr(chessboard, "LAST_SQUARES", {"d4": {"world": (0.252, 0.033), "pixel": (705.0, 366.0)}})
    state = SceneState(targets=[SceneObject("e4", "empty square e4", position=np.zeros(3))])
    piece = SceneObject("white_piece_d4", "a white piece on d4")
    piece.square = "d4"
    skills._vacated(state, piece, np.array([0.2518, 0.0327]))
    d4 = next(t for t in state.targets if t.name == "d4")
    assert np.allclose(d4.position, [0.252, 0.033, cfg.TABLE_Z]) and d4.pixel_xy == [705.0, 366.0]
    skills._vacated(state, piece, np.array([0.2518, 0.0327]))
    assert [t.name for t in state.targets] == ["e4", "d4"]                    # once
    # Putting it back is now a plan the controller runs as it is: no scan first.
    looked = []
    deployment = SimpleNamespace(look=lambda env, s: looked.append(1), look_around=None)
    llm_controller.controller.ensure_visible([{"action": "place", "target_name": "d4"}], state, _Env(), deployment)
    assert looked == []
    monkeypatch.setattr(chessboard, "LAST_SQUARES", {})                       # no scan's map: where the piece stood
    piece.square = "c4"
    skills._vacated(state, piece, np.array([0.247, 0.071]))
    assert np.allclose(state.find("c4", among="targets").position[:2], [0.247, 0.071])


def test_a_scan_parks_the_arm_where_the_arm_parks():
    parked = []
    env = SimpleNamespace(go_park=lambda: parked.append(1) or True)
    assert skills.goto_observe_pose(env) is True and parked == [1]


def test_finding_a_piece_again_after_a_miss_does_not_ask_the_vision_model(monkeypatch):
    from deployments.chess import chessboard
    asked = []

    def discover(env, scene_state, save_to=None, classify=True):
        asked.append(classify)
        scene_state.objects = [SceneObject("white_pawn_e2", "", position=np.array([0.178, -0.016, 0.0]))]
        return True

    monkeypatch.setattr(chessboard, "discover", discover)
    monkeypatch.setattr(skills, "check_overhead_calibration", lambda env: None)
    env = SimpleNamespace(go_park=lambda: True, go_home=lambda: True)
    assert np.allclose(skills._relocate_piece(env, (0.177, -0.017)), [0.178, -0.016])
    assert asked == [False]


@pytest.fixture
def lerobot_installed(monkeypatch):
    """A LeRobot checkpoint's skill registers only where LeRobot is installed: here, it is, whether or not it is."""
    from mrcf.policies import learned
    monkeypatch.setattr(learned, "_installed", lambda module: True)


def test_a_learned_skill_in_the_config_is_a_verb_that_hovers_then_runs_the_model(lerobot_installed):
    from mrcf.policies.base import Outcome, Target
    from mrcf.policies.learned import CallableModel
    from mrcf.vlm import ACTIONS
    names = skills.install_learned_skills([
        {"name": "upright_test", "description": "Stand a fallen piece up.", "start_height_m": 0.06,
         "model": CallableModel(lambda obs: np.zeros(6))},
        # A checkpoint is not loaded until the verb is first used: a path that is not there yet registers.
        {"name": "later_test", "description": "Not trained yet.", "checkpoint": "/nowhere/pretrained_model"},
    ])
    assert names == ["upright_test", "later_test"] and "object_name" in ACTIONS["upright_test"]["parameters"]
    hovers = []
    library = SimpleNamespace(move_above=lambda env, target, h: hovers.append(h) or Outcome(True, "hovering"))
    skill = skills.learned_skill({"name": "upright_test", "start_height_m": 0.06, "max_s": 0.05,
                                  "model": CallableModel(lambda obs: np.zeros(6))})
    got = skill(library, _StreamingEnv(), Target(np.zeros(3)))
    assert hovers == [0.06] and got.ok and got.message.startswith("hovering; upright_test: ran its 0.05 s")
    hovers.clear()
    got = skill(library, _Env(), Target(np.zeros(3)))                  # _Env streams nothing:
    assert hovers == [] and got.refused and "takes no stream" in got.message     # refused before the hover


def test_a_learned_skill_can_name_a_policy_package(tmp_path, lerobot_installed):
    from mrcf.policies import packages
    from mrcf.streaming import StreamSpec
    from mrcf.vlm import ACTIONS

    spec = StreamSpec(fps=30, state=("a",), action=("a",))
    packages.pack(Path(__file__), tmp_path / "pkg", packages.Manifest(
        name="nudge_pkg", format="lerobot", stream=spec, description="Nudge a piece a little."))
    names = skills.install_learned_skills([{"name": "nudge_pkg_test", "policy": str(tmp_path / "pkg")}])
    assert names == ["nudge_pkg_test"] and ACTIONS["nudge_pkg_test"]["description"].startswith("Nudge a piece")


def test_a_policy_package_that_cannot_be_had_leaves_out_its_skill_not_the_deployment(capsys, lerobot_installed):
    names = skills.install_learned_skills([{"name": "gone_test", "policy": "/nowhere/at/all"},
                                           {"name": "here_test", "description": "Here.", "checkpoint": "/later"}])
    assert names == ["here_test"] and "gone_test" in capsys.readouterr().out


def test_a_policy_package_is_fetched_once_per_skill(tmp_path, monkeypatch, lerobot_installed):
    from mrcf.policies import packages
    from mrcf.streaming import StreamSpec

    packages.pack(Path(__file__), tmp_path / "pkg", packages.Manifest(
        name="once_pkg", format="lerobot", stream=StreamSpec(fps=30, state=("a",), action=("a",)), description="Once."))
    fetched = []
    monkeypatch.setattr(packages, "SOURCES", {**packages.SOURCES, "count": lambda rest: fetched.append(rest) or tmp_path / "pkg"})
    assert skills.install_learned_skills([{"name": "once_test", "policy": "count:pkg"}]) == ["once_test"]
    assert fetched == ["pkg"]


def test_a_malformed_learned_skill_is_left_out_not_the_deployment(capsys):
    from mrcf.policies.learned import CallableModel
    names = skills.install_learned_skills([
        {"name": "no_model_test", "description": "Neither a checkpoint nor a model."},
        {"name": "fine_test", "description": "Fine.", "model": CallableModel(lambda obs: np.zeros(6))},
    ])
    assert names == ["fine_test"] and "no_model_test" in capsys.readouterr().out


def test_a_learned_skill_this_machine_cannot_run_is_never_offered_to_the_planner(capsys, monkeypatch):
    from mrcf.policies import learned
    from mrcf.policies.learned import CallableModel
    from mrcf.vlm import ACTIONS
    monkeypatch.setattr(learned, "_installed", lambda module: module != "lerobot")
    ACTIONS.pop("no_lerobot_test", None)
    names = skills.install_learned_skills([
        {"name": "no_lerobot_test", "description": "Needs LeRobot.", "checkpoint": "/somewhere/pretrained_model"},
        {"name": "scripted_test", "description": "Needs nothing.", "model": CallableModel(lambda obs: np.zeros(6))},
    ])
    out = capsys.readouterr().out
    assert names == ["scripted_test"] and "no_lerobot_test" not in ACTIONS
    assert "'no_lerobot_test' left out: its model needs lerobot, not installed here (pip install lerobot)" in out


def test_the_picks_measure_is_given_the_pieces_around_the_one_sent_for(monkeypatch):
    # The pick passes known=, target= and need_height= only to a measure that
    # takes them: chess's must, or the located camera measures blind among a crowd.
    from mrcf.policies.parallel_jaw.sighting import _takes_known
    from deployments.chess import piece_vision
    assert _takes_known(skills.manipulation.vision().measure)
    seen = {}
    monkeypatch.setattr(piece_vision, "measure_piece", lambda frame, px=None, **kw: seen.update(kw) or {"ok": False, "why": ""})
    skills._measure_piece(None, model={}, known=["k"], target="target", need_height=False)
    assert seen["known"] == ["k"] and seen["target"] == "target" and seen["need_height"] is False


def test_captured_pieces_go_on_the_border_past_the_a_file_each_a_square_further_along():
    from types import SimpleNamespace as NS
    from deployments.chess import chessboard
    sq = lambda x, y: {"world": [x, y]}                                     # noqa: E731
    squares = {"a1": sq(0.140, 0.150), "b1": sq(0.140, 0.110), "a2": sq(0.180, 0.150), "b2": sq(0.180, 0.110)}

    class Memory:
        def __init__(self, places):
            self.places = places

        def in_state(self, state):
            return [NS(place=p) for p in self.places] if state == "present" else []

    first = chessboard.capture_spot(squares, Memory([]))
    assert np.allclose(first, (0.160, 0.150 + cfg.CAPTURE_BORDER_M))          # beside a1-a2, on the border
    second = chessboard.capture_spot(squares, Memory(["e4", cfg.CAPTURE_TARGET]))
    assert np.allclose(second, first + (0.040, 0.0))                           # a square further along
    assert chessboard.capture_spot(squares, Memory([cfg.CAPTURE_TARGET] * cfg.CAPTURE_SLOTS)) is None
