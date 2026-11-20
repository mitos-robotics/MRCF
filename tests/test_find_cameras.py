"""
tools/find_cameras.py: the cameras by name, none opened; the computer's own
never opened; one index's picture only when asked, and only while every rig
camera the arm names is plugged in.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, ".")

from mrcf import devices  # noqa: E402
from mrcf.backends.rig import Camera  # noqa: E402
from tools import find_cameras  # noqa: E402

CAMERAS = {"front": Camera(mount="gripper", source=1), "overhead": Camera(mount="fixed", source=0)}


class _Cap:
    opened: list = []

    def __init__(self, index, *a):
        _Cap.opened.append(index)
        self.index = index

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def read(self):
        return True, np.full((720, 1280, 3), 90, np.uint8)

    def release(self):
        pass


def _rig(monkeypatch, listed):
    monkeypatch.setattr(devices, "video_devices", lambda: listed)
    monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(find_cameras.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(find_cameras, "backends", SimpleNamespace(arm=lambda: SimpleNamespace(
        NAME="SO-101", config=SimpleNamespace(CAMERAS=CAMERAS))))
    import tools._common as common
    monkeypatch.setattr(common, "backends", find_cameras.backends)
    monkeypatch.setattr(cv2, "VideoCapture", _Cap)
    _Cap.opened = []


def test_the_list_opens_nothing_and_names_the_computers_own(monkeypatch, capsys):
    _rig(monkeypatch, {0: "MacBook Pro Camera", 1: "USB Camera", 2: "HD USB Camera"})
    monkeypatch.setattr(sys, "argv", ["find_cameras.py"])
    find_cameras.main()
    out = capsys.readouterr().out
    assert _Cap.opened == []
    assert "MacBook Pro Camera" in out and "this computer's own: never opened" in out
    assert "for 2 index(es): enough" in out


def test_a_picture_only_while_the_rig_is_plugged_in(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(find_cameras, "OUT", tmp_path)
    _rig(monkeypatch, {0: "MacBook Pro Camera", 1: "MacBook Pro Desk View Camera"})
    monkeypatch.setattr(sys, "argv", ["find_cameras.py", "--picture", "0"])
    with pytest.raises(SystemExit, match="Not opening a camera"):
        find_cameras.main()
    assert _Cap.opened == []                                               # the rig unplugged: nothing opened
    _rig(monkeypatch, {0: "MacBook Pro Camera", 1: "USB Camera", 2: "HD USB Camera"})
    find_cameras.main()
    assert _Cap.opened == [0]
    assert [p.name for p in tmp_path.glob("*/*.jpg")] == ["index_0.jpg"]
    assert "CAMERAS 'overhead'" in capsys.readouterr().out
