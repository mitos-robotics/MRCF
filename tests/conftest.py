"""
Keep framework configuration from leaking between tests or into deployments,
and skip — not fail — a test that needs OpenCV where it is not installed.
"""

import importlib.util

import pytest

from mrcf import action_registry
from mrcf import manipulation
from mrcf import safety
from mrcf import vlm


@pytest.fixture(autouse=True)
def restore_framework_configuration(monkeypatch):
    for module, names in ((manipulation, ("_policies", "_vision", "_scenario")),
                          (safety, ("_bounds", "_thresholds", "_sequence_validators", "_monitor")),
                          (vlm, ("_server",))):      # the model server: configure()'s, else the environment's
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    # The registries are restored in place, never rebound: vlm and every
    # `from vlm import ACTIONS` hold the same dict as action_registry.
    saved = [(reg, dict(reg)) for reg in (action_registry.ACTIONS, action_registry.CONDITIONS)]
    yield
    for reg, before in saved:
        reg.clear()
        reg.update(before)


# ── No real camera ────────────────────────────────────────────────────────────
# A test never opens a camera on the machine it runs on: by index, the rig's
# cameras unplugged, that is the computer's own (2026-11-17: a UR5 test made
# its env from the rig's settings, opened index 0 — the Mac's camera — and the
# user saw its light come on). A test that means to open one patches
# cv2.VideoCapture itself (tests/test_real_robot_env.py's fakes do).

@pytest.fixture(autouse=True)
def no_real_camera(monkeypatch):
    if importlib.util.find_spec("cv2") is None:
        yield
        return
    import cv2

    def refused(*args, **kwargs):
        raise RuntimeError(f"a test opened a real camera (cv2.VideoCapture{args}): make the env offline, "
                           "or give it a fake source")
    try:
        monkeypatch.setattr(cv2, "VideoCapture", refused)
    except (AttributeError, TypeError):           # an OpenCV stub without it (the no-extras run)
        pass
    yield


# ── Nor lists this computer's cameras, nor finds faces with a model it fetched ─
# What camera a test sees is the test's (mrcf/devices.py: devices= given);
# whether faces are found is too (privacy.faces patched), so a run is the
# same on a computer with the face model as on CI without it.

@pytest.fixture(autouse=True)
def no_computer_specifics(monkeypatch):
    from mrcf import devices, privacy

    monkeypatch.setattr(devices, "video_devices", lambda: {})
    monkeypatch.setattr(privacy, "_DETECTOR", [None])
    yield


# ── OpenCV ─────────────────────────────────────────────────────────────────────
# Much of the framework reaches OpenCV lazily (mrcf/perception/, the motion guard, a
# pick's lighting check), so `pip install -e ".[dev]"` alone runs a test until
# it gets there. Such a test is skipped, saying why. A test file that imports
# it uses pytest.importorskip("cv2"); a test that needs it without importing
# it (a frame the flight recorder only writes with it) a skipif of its own.
# CI runs the suite both ways.

HAS_CV2 = importlib.util.find_spec("cv2") is not None
WHY = "needs OpenCV: pip install -e '.[measure]'"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    exc = call.excinfo
    if (not HAS_CV2 and report.failed and exc is not None and exc.errisinstance(ModuleNotFoundError)
            and getattr(exc.value, "name", None) == "cv2"):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {WHY}")
