"""
A policy trained by reinforcement learning, elsewhere: the codec it reads and
drives the stream through (mrcf/policies/rl.py), and a Stable-Baselines3 policy run
through LearnedPolicy, as an arm runs it.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from mrcf.policies import learned
from mrcf.policies.base import Target
from mrcf.policies.learned import LearnedPolicy
from mrcf.policies.rl import SB3Model, VectorCodec
from mrcf.streaming import ENVIRONMENT, StreamSpec
from test_learned import _ClockedArm


def codec():
    return VectorCodec(state=("a.pos", "b.pos", "g.pos"), action=("a.pos", "b.pos", "g.pos"), step=(2.0, 2.0, 0.0),
                       environment=2, mid=(0.0, 10.0, 50.0, 0.2, 0.0), half=(90.0, 90.0, 50.0, 0.1, 0.1))


def test_a_codec_reads_the_state_and_goal_and_turns_a_step_into_a_goal_from_here():
    c = codec()
    obs = {"observation.state": np.array([45.0, 10.0, 50.0]), ENVIRONMENT: np.array([0.3, -0.1])}
    assert np.allclose(c.observation(obs), [0.5, 0.0, 0.0, 1.0, -1.0])
    assert np.allclose(c.action_for(obs, [1.0, -0.5, 1.0]), [47.0, 9.0, 50.0])      # the gripper's step is 0
    assert np.allclose(c.action_for(obs, [3.0, 0.0, 0.0]), [47.0, 10.0, 50.0])      # clipped to a step of 1


def test_a_codec_runs_only_on_the_stream_it_was_trained_on(tmp_path):
    c = codec()
    spec = StreamSpec(fps=30, state=c.state, action=c.action, environment=("x", "y"))
    assert c.check(spec) == []
    assert "streams" in c.check(replace(spec, state=("a.pos", "z.pos", "g.pos")))[0]
    assert "give LearnedPolicy a goal" in c.check(replace(spec, environment=()))[0]
    c.save(tmp_path / "codec.json")
    assert VectorCodec.load(tmp_path / "codec.json") == c
    with pytest.raises(ValueError, match="goal for one of the state's"):
        VectorCodec(state=("a.pos",), action=("b.pos",), step=(1.0,))


def _saved_sb3(folder, c):
    """An untrained SAC network over c's vectors, saved as a training run leaves one."""
    sb3 = pytest.importorskip("stable_baselines3")
    import gymnasium

    class _Spaces(gymnasium.Env):
        observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (len(c.state) + c.environment,), np.float32)
        action_space = gymnasium.spaces.Box(-1.0, 1.0, (len(c.action),), np.float32)

    sb3.SAC("MlpPolicy", _Spaces(), seed=0, device="cpu", policy_kwargs=dict(net_arch=[8])).save(folder / "policy.zip")
    c.save(folder / "codec.json")
    (folder / "meta.json").write_text(json.dumps({"algo": "sac"}))
    return folder


def test_a_policy_trained_with_stable_baselines_runs_through_learned_policy(tmp_path, monkeypatch):
    names = ("x", "y", "z", "gripper")
    vectors = VectorCodec(state=names, action=names, step=(0.005, 0.005, 0.005, 0.0), environment=3)
    folder = _saved_sb3(tmp_path, vectors)
    arm = _ClockedArm()
    monkeypatch.setattr(learned.time, "sleep", arm.sleep)
    start = arm.get_ee_pos().copy()
    policy = LearnedPolicy(SB3Model(folder), name="reach", max_s=0.2, goal=lambda target: target.position)
    got = policy.run(arm, Target(position=np.array([0.35, 0.05, 0.10])))
    assert got.ok and "ran its 0.2 s" in got.message and "20 ticks" in got.message      # 100 Hz, the arm's clock
    assert len(arm.sent) == 20 and all(a[3] == arm.sent[0][3] for a in arm.sent)          # a gripper step of 0
    assert np.all(np.abs(arm.get_ee_pos() - start) <= 20 * 0.005 + 1e-6)                  # a step a tick at most
    assert policy.run(arm).refused                                                          # no target, no goal
    got = LearnedPolicy(SB3Model(folder), name="reach").run(arm, Target(position=start))
    assert got.refused and "goal of 3 values" in got.message
