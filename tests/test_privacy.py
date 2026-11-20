"""
No pictures of people kept, nothing kept beyond the work area, and no camera
of the computer's own opened (mrcf/privacy.py, mrcf/devices.py). User,
2026-11-17: "make sure there are no pictures of people or anything that
isn't essential etc. Especially when testing cameras".
"""

import re
from pathlib import Path

import numpy as np
import pytest

from mrcf import devices, privacy
from mrcf.safety import WorkspaceBounds

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = WorkspaceBounds(x_min=0.1, x_max=0.5, y_min=-0.25, y_max=0.25, z_min=0.0, z_max=0.4)


def _looking_down_at(x, y, height, f=500.0, w=640, h=480):
    """A camera model {K, T_world_cam} straight above (x, y), looking down."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])                     # camera z (its view) is world -z
    T[:3, 3] = [x, y, height]
    return {"K": K, "T_world_cam": T}


def _looking_level_away(height=0.3, f=500.0, w=640, h=480):
    """A camera at the base, level, looking out along -x: away from the table."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]])  # view along world -x, image down = world -z
    T[:3, 3] = [-0.3, 0.0, height]
    return {"K": K, "T_world_cam": T}


# ── Cameras of the computer's own ─────────────────────────────────────────────

def test_a_computers_own_cameras_are_known_by_name():
    for name in ("MacBook Pro Camera", "FaceTime HD Camera", "iPhone Camera", "MacBook Pro Desk View Camera",
                 "Capture screen 0", "Integrated Camera"):
        assert privacy.personal(name), name
    for name in ("USB Camera", "HD USB Camera", "Intel(R) RealSense(TM) Depth Camera 435 RGB", None):
        assert not privacy.personal(name), name


def test_on_a_mac_nothing_opens_unless_every_rig_camera_is_plugged_in():
    own = {0: "MacBook Pro Camera", 1: "MacBook Pro Desk View Camera", 2: "Capture screen 0"}
    with pytest.raises(devices.PersonalCamera, match="only 0 plugged in"):     # the rig unplugged
        devices.check_sources({"overhead": 0}, own, system="Darwin")
    one = {**own, 3: "USB Camera"}
    devices.check_sources({"overhead": 0}, one, system="Darwin")
    with pytest.raises(devices.PersonalCamera):                                 # the wrist camera unplugged
        devices.check_sources({"front": 1, "overhead": 0}, one, system="Darwin")
    devices.check_sources({"front": 1, "overhead": 0}, {**one, 4: "HD USB Camera"}, system="Darwin")


def test_on_linux_an_index_is_its_device_and_is_checked_as_such():
    listed = {0: "Integrated Camera", 2: "USB Camera"}
    with pytest.raises(devices.PersonalCamera, match="/dev/video0"):
        devices.check_sources({"overhead": 0}, listed, system="Linux")
    devices.check_sources({"overhead": 2}, listed, system="Linux")


def test_paths_urls_and_unlistable_computers_pass():
    devices.check_sources({"overhead": "rtsp://cam/stream", "side": "/dev/video4"}, {0: "FaceTime HD Camera"})
    devices.check_sources({"overhead": 0}, {})                                   # Windows: nothing to go by


def test_the_arms_refuse_before_opening(monkeypatch):
    from mrcf.backends.real_robot_env import RealRobotEnv

    monkeypatch.setattr(devices, "video_devices", lambda: {0: "MacBook Pro Camera"})
    monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
    from types import SimpleNamespace

    pytest.importorskip("cv2")
    env = SimpleNamespace()
    with pytest.raises(devices.PersonalCamera):
        RealRobotEnv._open_camera(env, "overhead", 0)    # conftest's cv2 guard would raise if it got there


# ── Faces ─────────────────────────────────────────────────────────────────────

def test_a_face_found_is_covered_whole_and_nothing_else(monkeypatch):
    monkeypatch.setattr(privacy, "faces", lambda img: [(100, 80, 40, 40)])
    img = np.full((240, 320, 3), 200, np.uint8)
    out, n = privacy.cover_people(img)
    assert n == 1 and out is not img and img.mean() == 200                   # a copy; the frame as it was
    assert (out[90:110, 110:130] == privacy.COVER_RGB).all()                 # the face
    assert (out[200:, :60] == 200).all()                                     # the table
    monkeypatch.setattr(privacy, "faces", lambda img: [])
    assert privacy.cover_people(img)[0] is img


def test_every_way_a_picture_is_written_covers_faces(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(privacy, "faces", lambda img: [(20, 20, 30, 30)])
    img = np.full((120, 160, 3), 220, np.uint8)
    privacy.imwrite(tmp_path / "a.png", img)
    privacy.save_rgb(tmp_path / "b.png", img)
    for name in ("a.png", "b.png"):
        back = cv2.imread(str(tmp_path / name))
        assert back[35, 35].max() <= 45, name
    ok, png = cv2.imencode(".png", img)
    covered = cv2.imdecode(np.frombuffer(privacy.encoded(png.tobytes()), np.uint8), cv2.IMREAD_COLOR)
    assert covered[35, 35].max() <= 45


def test_without_the_model_no_faces_are_looked_for_and_it_is_said():
    assert privacy.faces(np.zeros((100, 100, 3), np.uint8)) == []              # conftest: no model
    assert not privacy.has_face_model()


def test_the_face_model_is_checked_before_it_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("MRCF_HOME", str(tmp_path))
    assert not privacy.face_model_ok()
    bad = tmp_path / "models" / privacy.FACE_MODEL_NAME
    bad.parent.mkdir()
    bad.write_bytes(b"not the model")
    assert not privacy.face_model_ok()
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import fetch_face_model
    with pytest.raises(SystemExit, match="not the face model"):
        fetch_face_model.install(bad)


def test_pictures_kept_for_later_can_be_turned_off(monkeypatch):
    assert privacy.keep_pictures()
    monkeypatch.setenv("MRCF_KEEP_PICTURES", "0")
    assert not privacy.keep_pictures()


# ── The work area ─────────────────────────────────────────────────────────────

def test_a_camera_over_the_table_keeps_the_table_and_loses_the_room():
    mask = privacy.work_area_mask(_looking_down_at(0.3, 0.0, 1.0), 640, 480, BOUNDS)
    assert mask is not None and mask[240, 320]                                # straight down: the table
    assert not mask[5, 5] or not mask[475, 635]                               # far corners: beyond it
    assert 0.2 < mask.mean() < 0.95


def test_a_camera_looking_away_from_the_table_keeps_nothing():
    mask = privacy.work_area_mask(_looking_level_away(), 640, 480, BOUNDS)
    assert mask is not None and not mask.any()
    img = np.full((480, 640, 3), 180, np.uint8)
    assert privacy.work_area_only(img, _looking_level_away(), BOUNDS).max() == 0


def test_without_bounds_or_a_model_the_frame_is_kept_as_it_is():
    img = np.full((48, 64, 3), 99, np.uint8)
    assert privacy.work_area_only(img, None, BOUNDS) is img
    assert privacy.work_area_mask(_looking_down_at(0.3, 0, 1), 64, 48, WorkspaceBounds()) is None


def test_the_recorders_frames_are_kept_to_the_work_area_and_faces_covered(monkeypatch):
    calls = []
    monkeypatch.setattr(privacy, "faces", lambda img: calls.append(img.copy()) or [])

    class Env:
        def camera_model(self, cam, w, h):
            return _looking_level_away(w=w, h=h, f=w * 0.8) if cam == "front" else None
    keep = privacy.frame_guard(Env(), BOUNDS)
    img = np.full((480, 640, 3), 150, np.uint8)
    assert keep("front", img).max() == 0 and calls[-1].max() == 0              # masked before faces are looked for
    assert keep("other", img) is img or (keep("other", img) == img).all()      # no model: whole, faces covered


# ── Every saved picture goes through here ─────────────────────────────────────

def test_no_picture_is_written_past_the_guard():
    # A frame written straight with cv2.imwrite or PIL's save keeps whoever was
    # in it: every one goes through privacy.imwrite / save_rgb (or the flight
    # recorder, which guards its own).
    allowed = {ROOT / "mrcf" / "privacy.py", ROOT / "mrcf" / "flight_recorder.py"}
    bad = []
    for f in [*ROOT.glob("mrcf/**/*.py"), *ROOT.glob("deployments/**/*.py"), *ROOT.glob("tools/**/*.py"),
              ROOT / "run.py"]:
        if f in allowed or "__pycache__" in f.parts:
            continue
        text = f.read_text(encoding="utf-8")
        if "cv2.imwrite(" in text or re.search(r"Image\.fromarray\([^\n]*\)\.save\((?!buf)", text):
            bad.append(str(f.relative_to(ROOT)))
    assert not bad, f"pictures written past mrcf/privacy.py: {bad}"


def test_no_tool_sweeps_the_cameras():
    # 2026-11-14 an index sweep opened an iPhone's camera: a camera is opened by
    # the index a rig names, or one asked for, never by trying them in turn.
    for f in ROOT.glob("tools/**/*.py"):
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"for \w+ in range\([^)]*\):\s*\n\s*[^\n]*(grab|VideoCapture)\(", text), f
