"""
mrcf/policies/packages.py: a learned policy as a package — its model, and a manifest
saying which stream it runs on — fetched by reference, checked against an arm,
and run; and the plugin seams a policy store would use (a source, a format, an arm).
"""

import types
from pathlib import Path

import numpy as np
import pytest

from mrcf import backends
from mrcf.policies import packages
from mrcf.policies.learned import CallableModel, LearnedPolicy
from mrcf.streaming import StreamSpec
from test_learned import GOAL, _StreamingArm, near, toward

SPEC = _StreamingArm.stream_spec


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    # Formats and sources a test registers go when it ends.
    monkeypatch.setattr(packages, "FORMATS", dict(packages.FORMATS))
    monkeypatch.setattr(packages, "SOURCES", dict(packages.SOURCES))


def _package(tmp_path, **manifest) -> Path:
    """A package of the "toward" stand-in model (format "test"), trained on SPEC."""
    packages.register_format("test", lambda folder, m: CallableModel(toward(np.array(m.extra["goal"]))))
    (tmp_path / "weights.bin").write_bytes(b"\0")
    fields = dict(name="reach", format="test", stream=SPEC, version="1.2.0", description="Reach the goal.",
                  max_s=5.0, extra={"goal": GOAL.tolist()})
    fields.update(manifest)
    return packages.pack(tmp_path / "weights.bin", tmp_path / "reach", packages.Manifest(**fields))


# ── The stream an arm runs: its portable identity ─────────────────────────────

def test_a_stream_survives_a_file_and_its_fingerprint_ignores_device_names():
    spec = StreamSpec(fps=30, state=("a", "b"), action=("a", "b"), cameras={"wrist": ("front", 640, 480)},
                      robot_type="so101_follower", sensors={"wrench": ("tcp_wrench", (6,))})
    assert StreamSpec.from_dict(spec.to_dict()) == spec
    renamed = StreamSpec(fps=30, state=("a", "b"), action=("a", "b"), cameras={"wrist": ("/dev/video4", 640, 480)},
                         robot_type="so101_follower", sensors={"wrench": ("read_ft", (6,))})
    assert renamed.fingerprint() == spec.fingerprint()


@pytest.mark.parametrize("change", [
    {"fps": 20}, {"state": ("a", "c")}, {"cameras": {"wrist": ("front", 320, 240)}}, {"robot_type": "ur5"},
    {"sensors": {}},
])
def test_the_fingerprint_changes_with_what_a_model_depends_on(change):
    base = dict(fps=30, state=("a", "b"), action=("a", "b"), cameras={"wrist": ("front", 640, 480)},
                robot_type="so101_follower", sensors={"wrench": ("tcp_wrench", (6,))})
    assert StreamSpec(**{**base, **change}).fingerprint() != StreamSpec(**base).fingerprint()


# ── A package ─────────────────────────────────────────────────────────────────

def test_a_package_round_trips_and_says_which_arms_can_run_it(tmp_path):
    folder = _package(tmp_path)
    manifest, where = packages.open_package(str(folder))
    assert where == folder and manifest.name == "reach" and manifest.version == "1.2.0"
    assert manifest.stream == SPEC and manifest.embodiment == SPEC.fingerprint()
    assert (folder / "model").is_file()
    assert packages.fits(manifest, SPEC) == []
    smaller = StreamSpec(fps=100, state=SPEC.state, action=SPEC.action, cameras={"wrist": ("w", 32, 24)})
    assert any("frames" in p for p in packages.fits(manifest, smaller))
    slower = StreamSpec(fps=30, state=SPEC.state, action=SPEC.action, cameras=SPEC.cameras)
    assert any("Hz" in p for p in packages.fits(manifest, slower))


def test_a_packaged_policy_runs_on_an_arm_from_a_deployments_entry(tmp_path):
    folder = _package(tmp_path)
    entry = packages.skill_entry({"name": "reach_it", "policy": f"file:{folder}", "max_s": 4.0})
    assert entry["description"] == "Reach the goal." and entry["max_s"] == 4.0      # the entry's own wins
    assert entry["embodiment"] == SPEC.fingerprint() and entry["version"] == "1.2.0"
    arm = _StreamingArm()
    got = LearnedPolicy(entry["model"], name=entry["name"], until=near(GOAL), max_s=entry["max_s"]).run(arm)
    assert got.ok and np.linalg.norm(arm.get_ee_pos() - GOAL) < 0.001
    assert packages.skill_entry({"name": "x", "checkpoint": "/a"}) == {"name": "x", "checkpoint": "/a"}
    assert entry["trained_on"] == SPEC                                  # a skill holds the arm's stream against it


def test_a_goal_the_model_was_trained_with_is_the_tasks_not_held_against_the_arm():
    reach = packages.Manifest(name="reach", format="sb3", stream=StreamSpec(
        fps=SPEC.fps, state=SPEC.state, action=SPEC.action, cameras=SPEC.cameras, robot_type=SPEC.robot_type,
        environment=("goal.0", "goal.1", "goal.2")))
    assert packages.fits(reach, SPEC) == []


def test_a_package_that_is_not_there_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no such folder"):
        packages.open_package("packages/nothing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="not a policy package"):
        packages.open_package(str(tmp_path / "empty"))


def test_a_package_whose_format_needs_a_missing_library_says_so_before_it_is_run(tmp_path, monkeypatch):
    from mrcf.policies import learned
    packages.pack(Path(__file__), tmp_path / "pkg", packages.Manifest(name="upright", format="lerobot", stream=SPEC))
    monkeypatch.setattr(learned, "_installed", lambda module: module != "lerobot")
    entry = packages.skill_entry({"name": "upright", "policy": str(tmp_path / "pkg")})
    assert "pip install lerobot" in LearnedPolicy(entry["model"]).unavailable()


def test_a_reference_is_a_source_or_a_path_on_any_os(tmp_path):
    assert packages.fetch(str(tmp_path)) == tmp_path
    assert packages.fetch(f"file:{tmp_path}") == tmp_path
    assert packages.fetch("C:\\models\\reach") == Path("C:\\models\\reach")            # a drive, not a scheme
    with pytest.raises(KeyError, match="No policy source 'store'.*file, hf"):
        packages.fetch("store:reach@1")


# ── What a store would plug in ────────────────────────────────────────────────

def test_a_plugin_adds_a_source_a_format_and_an_arm_without_changing_the_repo(tmp_path, monkeypatch):
    folder = _package(tmp_path, format="vendor-net")
    del packages.FORMATS["test"]
    asked = []

    def store(rest):                         # a store's sign-in, licence check and download would be here
        asked.append(rest)
        return folder

    arm_module = types.ModuleType("vendor_arm")
    plugins = {"mrcf.policy_sources": [types.SimpleNamespace(name="store", load=lambda: store)],
               "mrcf.model_formats": [types.SimpleNamespace(
                   name="vendor-net", load=lambda: lambda f, m: CallableModel(toward(np.array(m.extra["goal"]))))],
               "mrcf.arms": [types.SimpleNamespace(name="vendor_arm", load=lambda: arm_module)]}
    found = lambda group: plugins.get(group, [])       # noqa: E731
    monkeypatch.setattr(packages, "entry_points", found)
    monkeypatch.setattr(backends, "entry_points", found)
    monkeypatch.setattr(backends, "_plugin_arms", {})

    manifest, where = packages.open_package("store:reach@1.2.0")
    assert asked == ["reach@1.2.0"] and where == folder
    model = packages.load_model(where, manifest)
    assert np.asarray(model.act({"observation.state": np.zeros(4, np.float32)})).shape == (4,)
    assert backends.arm("vendor_arm") is arm_module and "vendor_arm" in backends.available()
