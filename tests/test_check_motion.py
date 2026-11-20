"""
tools/check_motion.py, the first moves on any arm: every step runs on the
template's in-memory arm, each asked for first; its env is made with what the
arm's make_env takes.
"""

import builtins
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

from mrcf import backends, overhead_calibration  # noqa: E402
from tools import check_motion  # noqa: E402


@pytest.fixture
def template(monkeypatch):
    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    monkeypatch.setenv(backends.ENV_VAR, "_template")
    monkeypatch.setattr(check_motion, "FREEDRIVE_S", 0)
    return backends.arm("_template")


def test_every_step_runs_on_the_template_and_is_saved(template, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": asked.append(prompt) or "")
    env = check_motion.make_env(with_gripper=True)
    env.connect()
    try:
        out = check_motion.run(env, list(check_motion.STEPS), with_gripper=True, out_dir=tmp_path)
    finally:
        env.stop()
    steps = out["steps"]
    assert not [n for n, r in steps.items() if "error" in r]
    assert set(steps) == {1, 2, 3, 4, 5, 6, 8, 9}                       # 7: the template has no freedrive
    assert steps[1]["out"] and steps[1]["back"] and steps[3]["turned_deg"] == pytest.approx(20.0, abs=0.5)
    assert steps[4]["raised"].startswith("MotionStopped") and steps[5]["arrived"] and steps[9]["arrived"]
    assert len(asked) == 8                                                 # each step asked for first
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["arm"] == "_template" and set(saved["steps"]) == {str(n) for n in steps}


def test_q_stops_the_run_and_s_skips_a_step(template, monkeypatch, tmp_path):
    answers = iter(["s", "", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    env = check_motion.make_env(with_gripper=True)
    env.connect()
    try:
        out = check_motion.run(env, [1, 2, 3, 4], with_gripper=True, out_dir=tmp_path)
    finally:
        env.stop()
    assert set(out["steps"]) == {2}


def test_the_env_is_made_with_what_the_arms_make_env_takes(monkeypatch):
    got = {}

    def so101_like(**kwargs):                 # opens its own cameras: no sources=; one gripper: no gripper=
        for k in ("sources", "gripper"):
            if k in kwargs:
                raise TypeError(f"SO101Env.__init__() got an unexpected keyword argument '{k}'")
        got.update(kwargs)
        return "env"

    monkeypatch.setattr(backends, "arm", lambda name=None: SimpleNamespace(make_env=so101_like))
    assert check_motion.make_env(with_gripper=False) == "env" and got == {"motion_guard_zone": None}

    def broken(**kwargs):
        raise TypeError("something else entirely")

    monkeypatch.setattr(backends, "arm", lambda name=None: SimpleNamespace(make_env=broken))
    with pytest.raises(TypeError, match="something else"):
        check_motion.make_env(with_gripper=True)
