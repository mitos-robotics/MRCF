"""tools/show_stream.py: what a model for the arm must be trained on, and whether a package fits — on the template arm."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from mrcf import backends
from mrcf import overhead_calibration
from fake_arm import FakeArm
from mrcf.policies import learned, packages
from tools import show_stream


@pytest.fixture
def template(monkeypatch):
    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    arm = backends.arm("_template")
    monkeypatch.setattr(show_stream, "ARM", arm)
    monkeypatch.setattr(show_stream, "make_env", lambda: arm.make_env(speed_m_s=4.0, control_hz=1000.0))
    monkeypatch.setattr(learned, "_installed", lambda module: True)       # LeRobot, here or not
    return arm


def run(monkeypatch, *args) -> int:
    monkeypatch.setattr(sys, "argv", ["show_stream.py", *args])
    return show_stream.main()


def test_it_prints_the_stream_a_model_must_be_trained_on(template, monkeypatch, capsys):
    assert run(monkeypatch) == 0
    out = capsys.readouterr().out
    spec = template.make_env().stream_spec
    assert "30 Hz" in out and spec.fingerprint() in out and "x, y, z, gripper" in out


def test_it_says_whether_a_package_fits_and_why_not(template, monkeypatch, capsys, tmp_path):
    from mrcf.policies.learned import CallableModel
    spec = template.make_env().stream_spec
    monkeypatch.setitem(packages.FORMATS, "stand_in", lambda folder, manifest: CallableModel(lambda o: None))
    packages.pack(Path(__file__), tmp_path / "fits", packages.Manifest(name="reach", format="stand_in", stream=spec))
    assert run(monkeypatch, "--package", str(tmp_path / "fits")) == 0
    out = capsys.readouterr().out
    assert "It fits this arm's stream." in out and "It can run on this machine." in out
    packages.pack(Path(__file__), tmp_path / "act", packages.Manifest(name="act", format="lerobot", stream=spec))
    assert run(monkeypatch, "--package", str(tmp_path / "act")) == 1            # LeRobot here: but no checkpoint
    assert "Its model cannot be loaded" in capsys.readouterr().out

    other = replace(spec, state=("a", "b", "c", "d"), action=("a", "b", "c", "d"), robot_type="another")
    packages.pack(Path(__file__), tmp_path / "other", packages.Manifest(name="x", format="lerobot", stream=other))
    assert run(monkeypatch, "--package", str(tmp_path / "other")) == 1
    out = capsys.readouterr().out
    assert "does NOT fit" in out and "'another'" in out

    monkeypatch.setattr(learned, "_installed", lambda module: module != "lerobot")
    assert run(monkeypatch, "--package", str(tmp_path / "act")) == 1
    assert "pip install lerobot" in capsys.readouterr().out


def test_a_package_is_checked_by_its_model_too_and_a_bad_one_said_not_shown(template, monkeypatch, capsys,
                                                                            tmp_path):
    from mrcf.policies.learned import CallableModel
    spec = template.make_env().stream_spec
    monkeypatch.setitem(packages.FORMATS, "picky", lambda folder, manifest: CallableModel(
        lambda observation: None, check=lambda spec: ["it reads ('a', 'b'), the arm's state is ('x', ...)"]))
    packages.pack(Path(__file__), tmp_path / "picky", packages.Manifest(name="p", format="picky", stream=spec))
    assert run(monkeypatch, "--package", str(tmp_path / "picky")) == 1          # the manifest fits; the model not
    assert "does NOT fit" in capsys.readouterr().out
    packages.pack(Path(__file__), tmp_path / "odd", packages.Manifest(name="o", format="no_such_format", stream=spec))
    assert run(monkeypatch, "--package", str(tmp_path / "odd")) == 1
    assert "cannot be used: \"No model format 'no_such_format'" in capsys.readouterr().out
    assert run(monkeypatch, "--package", str(tmp_path / "missing")) == 1
    assert "no such folder" in capsys.readouterr().out


def test_an_arm_with_no_stream_is_told_where_to_start(template, monkeypatch, capsys):
    monkeypatch.setattr(show_stream, "make_env", FakeArm)
    assert run(monkeypatch) == 1
    assert "takes no stream" in capsys.readouterr().out
