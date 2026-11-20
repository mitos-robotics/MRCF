"""
tools/new_deployment.py: a new deployment starts from an existing one's code,
on the arm it names, and its rules come over as candidates only — none in
force until taken over.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import new_deployment as nd  # noqa: E402
from mrcf import profiles  # noqa: E402


def _copy(tmp_path, name):
    shutil.copytree(ROOT / "deployments" / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))


def test_a_deployments_rules_are_read_with_their_reasons():
    found = nd.rules(ROOT / "deployments" / "chess")
    names = [n for n, _, _ in found]
    assert set(names) == set(profiles.settings(profiles.load(ROOT / "deployments" / "chess")))
    hold = next(c for n, _, c in found if n == "GRIPPER_HOLD_PCT")
    assert any("piece" in line for line in hold)


def test_every_deployment_names_one_arm():
    for d in sorted(p for p in (ROOT / "deployments").iterdir() if (p / "config.py").is_file()):
        assert nd.arm_of(d) in {"so101", "ur5"}, d.name


def test_a_new_deployment_takes_the_code_and_none_of_the_rules(tmp_path):
    _copy(tmp_path, "tidy")
    made = nd.create("tidy_desk", "tidy", tmp_path)
    for f in ("__init__.py", "config.py", "skills.py", "llm_controller.py", "NOTES.md", "profile.py"):
        assert (made / f).is_file(), f
    assert not (made / "tests").exists() and not (made / "HISTORY.md").exists()
    assert nd.arm_of(made) == "so101"                                           # the source's arm
    profile = profiles.load(made)
    assert profile is not None and profiles.settings(profile) == {}            # nothing in force
    text = profiles.path(made).read_text()
    assert "# PLACE_IN = True" in text and "without it: the framework's: False" in text


def test_the_same_task_on_another_arm_is_another_deployment(tmp_path):
    _copy(tmp_path, "tidy")
    made = nd.create("tidy_ur5", "tidy", tmp_path, "ur5")
    assert nd.arm_of(made) == "ur5"
    text = profiles.path(made).read_text()
    assert "its rules on the so101 are below as" in text.replace("\n", " ")
    assert "the ur5 does without it" in text.replace("\n", " ")
    assert profiles.settings(profiles.load(made)) == {}


def test_kept_state_named_in_the_copy_is_renamed(tmp_path):
    _copy(tmp_path, "chess")
    (tmp_path / "chess" / "config.py").write_text(
        (tmp_path / "chess" / "config.py").read_text() + '\nKEPT = RIG / "chess_memory.json"\n')
    made = nd.create("draughts", "chess", tmp_path)
    text = (made / "config.py").read_text()
    assert '"draughts_memory.json"' in text and '"chess_memory.json"' not in text


def test_it_will_not_overwrite_a_deployment(tmp_path):
    _copy(tmp_path, "tidy")
    nd.create("again", "tidy", tmp_path)
    try:
        nd.create("again", "tidy", tmp_path)
    except SystemExit as exc:
        assert "not overwriting" in str(exc)
    else:
        raise AssertionError("overwrote a deployment")



def test_a_copy_starts_with_tasks_of_its_own_the_sources_kept_for_reference(tmp_path):
    import ast
    import shutil

    from tools import new_deployment
    root = tmp_path / "deployments"
    shutil.copytree(ROOT / "deployments" / "tidy", root / "tidy", ignore=shutil.ignore_patterns("__pycache__"))
    dst = new_deployment.create("pantry", "tidy", root)
    text = (dst / "config.py").read_text()
    ast.parse(text)
    assert "# TASKS = {\"tidy up\"" in text and "# PLANNER = False" in text
    assert "\nTASKS: dict = {}\nPLANNER = True\n" in text
