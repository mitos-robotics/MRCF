"""
What an arm brings to a job, said before anything runs (mrcf/rig_report.py):
nothing connected to, nothing opened — and new arms and deployments started
from the launcher (tools/new_arm.py, run.py's "n").
"""

import importlib
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mrcf import rig_report
from mrcf.backends.rig import Camera, Sensor

ROOT = Path(__file__).resolve().parents[1]


def _arm(found=True, cameras=None, sensors=None, overhead_model=True):
    arm = ModuleType("mrcf.backends.testarm")
    arm.NAME = "Test Arm"
    arm.present = lambda: (found, "answering" if found else "no answer")
    arm.config = SimpleNamespace(
        CAMERAS=cameras if cameras is not None else {"overhead": Camera(mount="fixed", source=0, size=(1280, 720)),
                                                     "front": Camera(mount="gripper", source=1)},
        SENSORS=sensors or {}, OVERHEAD_CAMERA_NAME="overhead", CAMERA_NAME="front",
        OVERHEAD_CAM_MODEL={"K": 1} if overhead_model else None, GRIPPER_CAM_K=[[1]], GRIPPER_CAM_POSE=None,
        GRIPPER_MAX_OPEN_M=0.08)
    return arm


def test_the_rig_is_said_without_opening_anything(monkeypatch):
    # A Mac's listing, so a Mac's rule (devices.check_sources): on Linux index 0 would be /dev/video0, the
    # MacBook's camera here, and rightly refused — CI's Linux runners failed this test until it said which.
    from mrcf import devices
    monkeypatch.setattr(devices, "platform", SimpleNamespace(system=lambda: "Darwin"))
    listed = {0: "MacBook Pro Camera", 1: "USB Camera", 2: "HD USB Camera"}
    rep = rig_report.report(_arm(sensors={"wrench": Sensor(kind="force", unit="N")}), listed=listed)
    text = "\n".join(rig_report.lines(rep))
    assert "Test Arm (testarm): plugged in" in text and "opening 80 mm" in text
    assert "overhead" in text and "pose and lens" in text and "lens only" in text      # the wrist: no place yet
    assert "plugged in: HD USB Camera, USB Camera" in text and "never opened: MacBook Pro Camera" in text
    assert "wrench" in text and "force" in text
    assert "faces NOT covered" in text                                                  # conftest: no model


def test_a_rig_camera_unplugged_is_said_before_the_run(monkeypatch):
    from mrcf import devices

    monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
    rep = rig_report.report(_arm(found=False), listed={0: "MacBook Pro Camera", 1: "USB Camera"})
    text = "\n".join(rig_report.lines(rep))
    assert "NOT found" in text and "NOT ALL PLUGGED IN" in text
    assert rep["camera_check"]["plugged_in"] is False


def test_a_deployment_is_said_in_a_line():
    tidy = importlib.import_module("deployments.tidy.config")
    dep = rig_report.deployment(tidy)
    assert dep["name"] == "tidy" and dep["arm"] == "so101" and "PLACE_IN" in dep["rules"]
    line = rig_report.deployment_line(dep)
    assert line.startswith("the SO-101 putting everyday things away") and "rule(s) of its own" in line


def test_a_new_arm_is_the_template_in_memory_under_its_own_name(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import new_arm

    backends = tmp_path / "backends"
    shutil.copytree(ROOT / "mrcf" / "backends" / "_template", backends / "_template",
                    ignore=shutil.ignore_patterns("__pycache__"))
    made = new_arm.create("cell_two", "Cell Two", backends)
    assert (made / "env.py").is_file() and (made / "robot_config.py").is_file()
    assert 'NAME = "Cell Two"' in (made / "__init__.py").read_text()
    with pytest.raises(SystemExit, match="not overwriting"):
        new_arm.create("cell_two", None, backends)
    with pytest.raises(SystemExit, match="Python module"):
        new_arm.create("Cell-2", None, backends)
    assert "the job" in new_arm.next_steps("cell_two")
