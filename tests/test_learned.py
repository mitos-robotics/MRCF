"""
A learned policy (mrcf/policies/learned.py) on an arm's stream — here an arm that
is not the SO-101: a Cartesian one (test_rig's) streaming its grasp point and
opening, as a UR's would. Then a real, if untrained, LeRobot ACT checkpoint on
it, and a learned verb reaching the planner.
"""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from mrcf import manipulation
from fake_arm import FakeArm
from mrcf.interlock import MotionStopped
from mrcf.policies import learned
from mrcf.policies.base import Outcome
from mrcf.policies.learned import CallableModel, LearnedPolicy, chain, hover_over
from mrcf.streaming import StreamSpec
from test_rig import _CartesianArm
from mrcf.vlm import SceneObject, SceneState, dispatch

GOAL = np.array([0.35, 0.05, 0.10])


class _StreamingArm(_CartesianArm):
    """The grasp point (x, y, z) and the opening in; the same out — where it goes, at once."""
    stream_spec = StreamSpec(fps=100, state=("x", "y", "z", "gripper"), action=("x", "y", "z", "gripper"),
                             cameras={"wrist": ("wrist", 64, 48)}, robot_type="cartesian")
    REACH_M = 0.85

    def __init__(self, **rig):
        super().__init__(**rig)
        self.sent, self.holds = [], 0

    def hold(self):
        self.holds += 1

    def stream_observation(self):
        x, y, z = self.get_ee_pos()
        return {"observation.state": np.array([x, y, z, self._width], np.float32),
                "observation.images.wrist": self.render_image("wrist", 64, 48)}

    def stream_action(self, action):
        self.guard()
        a = np.asarray(action, float)
        if np.linalg.norm(a[:2]) > self.REACH_M:
            return self.stream_observation()["observation.state"], "out of reach"
        shift = a[:3] - self.get_ee_pos()
        with self._lock:
            self._flange[:3, 3] += shift
        self._width = float(a[3])
        self.sent.append(a.copy())
        return a, None


def toward(goal, step_m=0.005):
    """A stand-in model: a step toward goal each tick."""
    def act(observation):
        here = np.asarray(observation["observation.state"][:3], float)
        gap = goal - here
        return np.r_[here + gap * min(1.0, step_m / max(np.linalg.norm(gap), 1e-9)), 0.02]
    return act


def near(goal, tol=0.001):
    return lambda env, observation: np.linalg.norm(observation["observation.state"][:3] - goal) < tol


def test_a_learned_segment_runs_until_it_is_done():
    arm = _StreamingArm()
    got = LearnedPolicy(CallableModel(toward(GOAL)), name="reach", until=near(GOAL), max_s=5).run(arm)
    assert got.ok and got.message.startswith("reach: done") and got.attempts[0]["result"] == "ok"
    assert np.linalg.norm(arm.get_ee_pos() - GOAL) < 0.001 and arm.holds == 1   # and left still
    assert "ticks" in got.notes[0] and "100 Hz" in got.notes[0]


class _ClockedArm(_StreamingArm):
    """
    On a clock of its own, each tick's work taking 1 ms: what is tested is the
    loop's pacing, not the machine's sleep — a CI Mac's overshoots 10 ms twice over.
    """

    def __init__(self, **rig):
        super().__init__(**rig)
        self.now, self.slept = 0.0, []

    def motion_clock(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s

    def stream_action(self, action):
        self.now += 0.001
        return super().stream_action(action)


def test_without_a_done_check_it_runs_its_time_at_its_rate(monkeypatch):
    arm = _ClockedArm()
    monkeypatch.setattr(learned.time, "sleep", arm.sleep)
    got = LearnedPolicy(CallableModel(toward(GOAL)), max_s=0.2).run(arm)
    assert got.ok and "ran its 0.2 s" in got.message and "20 ticks" in got.message
    assert len(arm.sent) == 20                           # 100 Hz, not as fast as it can
    assert arm.slept == pytest.approx([0.009] * 20)      # the tick's own work taken off its wait


def test_a_segment_not_done_in_its_time_has_failed():
    got = LearnedPolicy(CallableModel(toward(GOAL, step_m=1e-5)), until=near(GOAL), max_s=0.1).run(_StreamingArm())
    assert not got.ok and got.attempts[0]["result"] == "not_done" and not got.refused


def test_an_arm_that_holds_back_everything_the_model_asks_stops_it():
    arm = _StreamingArm()
    got = LearnedPolicy(CallableModel(lambda obs: np.array([2.0, 0.0, 0.1, 0.0])), max_s=5,
                        max_held_back_s=0.05).run(arm)
    assert not got.ok and got.attempts[0]["result"] == "held_back" and "out of reach" in got.message
    assert arm.sent == []


def test_a_model_trained_on_another_stream_is_refused_before_anything_moves():
    arm = _StreamingArm()
    model = CallableModel(toward(GOAL), check=lambda spec: [f"wants 6 joints, {spec.robot_type} streams 4"])
    got = LearnedPolicy(model).run(arm)
    assert got.refused and not got.ok and "another stream" in got.message and arm.sent == []


def test_an_arm_that_takes_no_stream_is_refused():
    got = LearnedPolicy(CallableModel(toward(GOAL))).run(FakeArm())
    assert got.refused and "ActionStreaming" in got.message


def test_a_wrong_sized_action_stops_it():
    got = LearnedPolicy(CallableModel(lambda obs: np.zeros(3))).run(_StreamingArm())
    assert not got.ok and got.attempts[0]["result"] == "bad_action" and "takes 4" in got.message


def test_a_stop_ends_it_where_it_is_and_a_crash_leaves_the_arm_held():
    arm = _StreamingArm()
    calls = {"n": 0}

    def act(observation):
        calls["n"] += 1
        if calls["n"] == 3:
            arm.interlock.trip("a hand", source="test")
        return toward(GOAL)(observation)
    with pytest.raises(MotionStopped):
        LearnedPolicy(CallableModel(act), max_s=5).run(arm)
    assert len(arm.sent) == 2

    arm = _StreamingArm()
    with pytest.raises(RuntimeError, match="out of memory"):
        LearnedPolicy(CallableModel(lambda obs: (_ for _ in ()).throw(RuntimeError("out of memory")))).run(arm)
    assert arm.holds == 1


def test_after_a_pause_the_model_starts_again_from_what_it_sees():
    # Paused by the sentry mid-segment: what an action-chunking model had
    # queued was for the scene before the hand came in.
    arm = _StreamingArm()
    arm.interlock.pause_only("motion")
    resets, calls = [], {"n": 0}

    def act(observation):
        calls["n"] += 1
        if calls["n"] == 4:
            arm.interlock.trip("a hand", source="motion")
            threading.Timer(0.1, arm.interlock.clear, args=("motion",)).start()
        return toward(GOAL)(observation)
    got = LearnedPolicy(CallableModel(act, reset=lambda: resets.append(1)), until=near(GOAL), max_s=5).run(arm)
    assert got.ok and len(resets) == 2 and "started again 1x after a pause" in got.notes[0]


def test_a_model_is_loaded_once_and_only_when_first_needed():
    loads = []
    policy = LearnedPolicy(lambda: loads.append(1) or CallableModel(toward(GOAL)), max_s=0.03)
    assert loads == []
    policy.run(_StreamingArm())
    policy.run(_StreamingArm())
    assert loads == [1]


# ── Skills: a planned approach, the learned segment ──────────────────────────

class _Library:
    """A policy library's move_above, as chain()'s steps see it."""

    def __init__(self, ok=True):
        self.ok, self.hovers = ok, []

    def move_above(self, env, target, height_m=None):
        self.hovers.append(height_m)
        if not self.ok:
            return Outcome(False, "out of reach", refused=True)
        assert env.move_ee_to(np.r_[target.position[:2], height_m])
        return Outcome(True, f"hovering at {height_m:.3f} m")


def test_a_skill_runs_its_steps_in_order_and_stops_at_the_first_that_fails():
    from mrcf.policies.base import Target
    arm, lib = _StreamingArm(), _Library()
    target = Target(np.array([0.35, 0.05, 0.0]))
    skill = chain(hover_over(0.06), LearnedPolicy(CallableModel(toward(GOAL)), name="reach", until=near(GOAL)))
    got = skill(lib, arm, target)
    assert got.ok and lib.hovers == [0.06] and got.message.startswith("hovering at 0.060 m; reach: done")
    assert [a["approach"] for a in got.attempts] == ["reach"]

    arm, lib = _StreamingArm(), _Library(ok=False)
    got = skill(lib, arm, target)
    assert not got.ok and got.refused and arm.sent == []          # refused: nothing moved


class _NeedsALibrary(CallableModel):
    """A model whose library is not installed on this machine."""

    def unavailable(self):
        return learned.needs("no_such_library", install="pip install no_such_library")


def test_a_model_whose_library_is_not_installed_is_refused_before_anything_moves():
    from mrcf.policies.base import Target
    model = _NeedsALibrary(toward(GOAL))
    assert model.unavailable() == "needs no_such_library, not installed here (pip install no_such_library)"
    arm = _StreamingArm()
    policy = LearnedPolicy(model, name="reach")
    got = policy.run(arm)
    assert got.refused and not got.ok and "cannot run on this machine" in got.message and "pip install" in got.message
    assert arm.sent == [] and got.attempts[0]["result"] == "unavailable"
    # In a skill, the steps before it do not move the arm either: the whole chain is refused at once.
    arm, lib = _StreamingArm(), _Library()
    skill = chain(hover_over(0.06), policy)
    assert "no_such_library" in skill.unavailable()
    got = skill(lib, arm, Target(np.array([0.35, 0.05, 0.0])))
    assert got.refused and lib.hovers == [] and arm.sent == []
    assert chain(hover_over(0.06), LearnedPolicy(CallableModel(toward(GOAL)))).unavailable() is None


def test_a_chain_refuses_a_step_that_cannot_run_without_calling_it():
    from mrcf.policies.base import Target
    called = []

    def step(pol, env, target, obstacles=(), **_):        # says it cannot run, yet would move if called
        called.append(1)
        return Outcome(True, "moved anyway")
    step.name, step.unavailable = "odd_step", lambda: "needs something"
    got = chain(hover_over(0.06), step)(_Library(), _StreamingArm(), Target(np.array([0.35, 0.05, 0.0])))
    assert got.refused and called == [] and got.message == "odd_step: cannot run on this machine — needs something"


def test_a_chain_refuses_a_model_for_another_stream_before_its_first_step_moves():
    from dataclasses import replace
    from mrcf.policies.base import Target
    other = replace(_StreamingArm.stream_spec, state=("a", "b", "c", "d"), action=("a", "b", "c", "d"))
    arm, lib = _StreamingArm(), _Library()
    skill = chain(hover_over(0.06), LearnedPolicy(CallableModel(toward(GOAL)), name="reach", trained_on=other))
    got = skill(lib, arm, Target(np.array([0.35, 0.05, 0.0])))
    assert got.refused and "another stream" in got.message and lib.hovers == [] and arm.sent == []
    assert skill.refusal(arm).refused                              # asked beforehand, the same answer
    # The model's own check counts too: one that says it does not fit is refused before the hover as well.
    picky = CallableModel(toward(GOAL), check=lambda spec: ["it reads 7 values"])
    got = chain(hover_over(0.06), LearnedPolicy(picky, name="reach"))(lib, arm, Target(np.zeros(3)))
    assert got.refused and "7 values" in got.message and lib.hovers == []


def test_a_model_made_by_a_function_is_checked_once_it_is_made():
    arm = _StreamingArm()
    got = LearnedPolicy(lambda: _NeedsALibrary(toward(GOAL)), name="later").run(arm)
    assert got.refused and "no_such_library" in got.message and arm.sent == []


def test_a_package_that_is_blocked_or_inside_a_missing_one_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "blocked_library", None)
    assert not learned._installed("blocked_library") and not learned._installed("no_such_library.sub")


def test_the_lerobot_and_stable_baselines_models_say_when_their_library_is_missing(monkeypatch):
    from mrcf.policies.lerobot_model import LeRobotModel
    from mrcf.policies.rl import SB3Model
    monkeypatch.setattr(learned, "_installed", lambda module: True)
    assert LeRobotModel("/later").unavailable() is None and SB3Model("/later").unavailable() is None
    monkeypatch.setattr(learned, "_installed", lambda module: module not in ("lerobot", "torch", "stable_baselines3"))
    assert LeRobotModel("/later").unavailable() == "needs lerobot and torch, not installed here (pip install lerobot)"
    assert "stable_baselines3" in SB3Model("/later").unavailable() and ".[sb3]" in SB3Model("/later").unavailable()
    got = LearnedPolicy(LeRobotModel("/later"), name="upright").run(_StreamingArm())
    assert got.refused and "pip install lerobot" in got.message              # and nothing was loaded: there is no checkpoint


def test_a_model_trained_on_an_arm_with_other_joints_is_refused_before_it_is_loaded():
    from dataclasses import replace
    spec = _StreamingArm.stream_spec
    other = replace(spec, state=("a", "b", "c", "d"), action=("a", "b", "c", "d"), robot_type="another")
    loads = []
    arm = _StreamingArm()
    got = LearnedPolicy(lambda: loads.append(1) or CallableModel(toward(GOAL)), trained_on=other).run(arm)
    assert got.refused and "'another'" in got.message and "['a', 'b', 'c', 'd']" in got.message
    assert arm.sent == [] and loads == []
    assert LearnedPolicy(CallableModel(toward(GOAL)), trained_on=spec, max_s=0.03).run(_StreamingArm()).ok


def test_whether_a_library_is_installed_is_looked_up_not_imported():
    assert learned._installed("numpy") and not learned._installed("no_such_library")
    learned._installed("this_package_does_not_exist_either")
    assert "this_package_does_not_exist_either" not in sys.modules


def test_a_learned_verb_reaches_the_planner_and_the_deployment_looks_again():
    manipulation.install(_Library())
    looked = []
    manipulation.object_action(
        "reach_it", chain(hover_over(0.06), LearnedPolicy(CallableModel(toward(GOAL)), name="reach_it",
                                                          until=near(GOAL))),
        description="Reach the object.", after=lambda env, state, obj, outcome: looked.append((obj.name, outcome.ok)))
    state = SceneState(objects=[SceneObject("cup", "a cup", position=np.array([0.35, 0.05, 0.0]))])
    arm = _StreamingArm()
    assert dispatch({"action": "reach_it", "object_name": "cup"}, arm, state) is True
    assert looked == [("cup", True)] and np.linalg.norm(arm.get_ee_pos() - GOAL) < 0.001


# ── A real LeRobot checkpoint ─────────────────────────────────────────────────

def _act_checkpoint(path: Path, state: int, action: int, image=(3, 48, 64)) -> Path:
    """An untrained ACT, saved as lerobot-train saves one: config, weights, its processors."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = ACTConfig(input_features={"observation.state": PolicyFeature(FeatureType.STATE, (state,)),
                                    "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, image)},
                    output_features={"action": PolicyFeature(FeatureType.ACTION, (action,))},
                    device="cpu", pretrained_backbone_weights=None, dim_model=32, n_heads=2, dim_feedforward=64,
                    n_encoder_layers=1, n_decoder_layers=1, use_vae=False, chunk_size=5, n_action_steps=5)
    near_here = {"mean": torch.tensor([0.30, 0.0, 0.25, 0.02][:action] + [0.0] * max(0, action - 4)),
                 "std": torch.full((action,), 0.001)}
    stats = {"observation.state": {"mean": torch.zeros(state), "std": torch.ones(state)}, "action": near_here,
             "observation.images.wrist": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)}}
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    for part in (ACTPolicy(cfg), pre, post):
        part.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    pytest.importorskip("lerobot")
    root = tmp_path_factory.mktemp("checkpoints")
    return {"this_arm": _act_checkpoint(root / "cartesian", 4, 4), "so101": _act_checkpoint(root / "so101", 6, 6)}


def test_a_trained_lerobot_policy_runs_on_the_arm_whose_stream_it_was_trained_on(checkpoints):
    from mrcf.policies.lerobot_model import LeRobotModel
    arm = _StreamingArm()
    got = LearnedPolicy(LeRobotModel(checkpoints["this_arm"], device="cpu"), name="act", max_s=0.15).run(arm)
    assert got.ok, got.message
    assert arm.sent and all(a.shape == (4,) for a in arm.sent) and "the model took" in got.notes[0]


def test_a_lerobot_policy_trained_on_another_arm_is_refused(checkpoints):
    from mrcf.policies.lerobot_model import LeRobotModel
    arm = _StreamingArm()
    got = LearnedPolicy(LeRobotModel(checkpoints["so101"], device="cpu"), name="act").run(arm)
    assert got.refused and "observation.state" in got.message and arm.sent == []


def test_a_missing_checkpoint_says_where_it_looked(tmp_path):
    pytest.importorskip("lerobot")
    from mrcf.policies.lerobot_model import LeRobotModel
    with pytest.raises(FileNotFoundError, match="pretrained_model"):
        LeRobotModel(tmp_path / "nothing").load()
