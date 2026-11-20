"""
tools/calibrate_camera_by_hand.py end to end on a fake arm in freedrive: it
holds still at one pose after another (a UR5's kinematics), its camera renders
a real AprilTag where a tag on link 4 would be, and the recording is saved,
detected and solved from disk; tools/show_workspace.py then draws from the
solution. No arm, no camera.
"""

import json
import sys
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.insert(0, "tests")

from mrcf.backends.ur5 import kinematics  # noqa: E402
from mrcf.perception import camera_from_arm as cfa  # noqa: E402
from mrcf.perception import fiducials  # noqa: E402

import test_camera_from_arm as synth  # noqa: E402

W, H = 1280, 720
TAG_ID = 1


def _render(T_cam_tag, size_m) -> np.ndarray:
    """A grey frame with tag TAG_ID drawn where it is (its black square warped onto its projected corners)."""
    img = np.full((H, W, 3), 150, np.uint8)
    cells, px_per = 10, 24                                   # 8 cells of tag + a white cell all round
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(fiducials.TAG_DICTIONARY), TAG_ID,
                                           8 * px_per)
    sheet = np.full((cells * px_per, cells * px_per), 255, np.uint8)
    sheet[px_per:9 * px_per, px_per:9 * px_per] = marker
    obj = fiducials.tag_object_points(size_m)
    Pc = (T_cam_tag[:3, :3] @ obj.T).T + T_cam_tag[:3, 3]
    dst, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), synth.K, synth.D)
    m, s = px_per, 8 * px_per
    src = np.float32([[m, m], [m + s, m], [m + s, m + s], [m, m + s]])
    Hm = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
    warped = cv2.warpPerspective(sheet, Hm, (W, H), borderValue=0)
    mask = cv2.warpPerspective(np.full_like(sheet, 255), Hm, (W, H), borderValue=0) > 0
    img[mask] = np.repeat(warped[mask][:, None], 3, axis=1)
    return img


class FreedriveArm:
    """
    Held still at each pose until a picture of it has been taken, then (a
    moment later, as a hand would) moved to the next; renders the tag; counts
    freedrive on/off. Paced by the pictures, not the clock, so a slow machine
    takes the same pictures.
    """
    MOVE_AFTER_S = 0.1

    def __init__(self, poses_q):
        self.qs, self.i, self.pictured_at, self.torque = poses_q, 0, None, []

    def _q(self):
        if self.pictured_at is not None and time.monotonic() - self.pictured_at > self.MOVE_AFTER_S:
            self.i, self.pictured_at = min(self.i + 1, len(self.qs) - 1), None
        return self.qs[self.i]

    def link_frames(self):
        return kinematics.frames(self._q())

    def render_image(self, camera, width, height):
        T_cam_tag = cfa.inv(synth.CAMERA) @ kinematics.frames(self._q())[4] @ synth.TAG
        image = _render(T_cam_tag, synth.SIZE_M)
        # The hand moves it a moment after the picture is taken, however long the
        # drawing took. Timed from before the drawing, on CI's slower runners
        # (2026-11-13) the arm had moved before the tool read it again, every
        # picture was taken for one mid-move, and none was kept.
        self.pictured_at = time.monotonic()
        return image

    def freedrive(self, on):
        self.torque.append("off" if on else "on")


def _visible_poses(n, seed=1):
    out = []
    rng = np.random.default_rng(seed)
    while len(out) < n:                                      # the same sampler, keeping the joints this time
        q = synth.Q_NOW + np.radians(rng.uniform(-1, 1, 6) * [30, 20, 20, 25, 20, 30])
        T_cam_tag = cfa.inv(synth.CAMERA) @ kinematics.frames(q)[4] @ synth.TAG
        to_cam = -T_cam_tag[:3, 3] / np.linalg.norm(T_cam_tag[:3, 3])
        # Clearly readable, not just readable: CI (Linux, 2026-11-13) missed a tag 59 deg
        # off the camera and 17 px across that a Mac read, and one pose short, the
        # recording never got its 10. About the recording and the solve, not the detector's edge.
        if T_cam_tag[:3, 2] @ to_cam < np.cos(np.radians(45.0)):
            continue
        Pc = (T_cam_tag[:3, :3] @ fiducials.tag_object_points(synth.SIZE_M).T).T + T_cam_tag[:3, 3]
        px, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), synth.K, synth.D)
        px = px.reshape(4, 2)
        if not ((px > 20) & (px < [W - 20, H - 20])).all() or np.linalg.norm(px[0] - px[2]) / np.sqrt(2) < 22:
            continue
        out.append(q)
    return out


def test_record_then_solve_from_disk(tmp_path, monkeypatch):
    import tools.calibrate_camera_by_hand as tool
    monkeypatch.setattr(tool, "STILL_S", 0.12)
    monkeypatch.setattr(tool, "TIMEOUT_S", 20.0)
    monkeypatch.setattr(tool, "lens_for", lambda camera, size: (synth.K, synth.D, "the test's lens"))
    arm = FreedriveArm(_visible_poses(10))
    poses = tool.record(arm, "overhead", (W, H), want=10, run=tmp_path)
    assert arm.torque == ["off", "on"]                       # freedrive switched on, and off again at the end
    assert sum(bool(p["tags"]) for p in poses) == 10
    assert len(list(tmp_path.glob("pose_*.jpg"))) == len(poses)

    monkeypatch.setattr(sys, "argv", ["calibrate_camera_by_hand.py", "--tag-mm", "40"])
    tool.solve_run(tmp_path)
    sol = json.loads((tmp_path / "solution.json").read_text())
    assert sol["link"] == 4 and sol["tag_id"] == TAG_ID
    T = np.asarray(sol["model"]["T_world_cam"])
    assert np.linalg.norm(T[:3, 3] - synth.CAMERA[:3, 3]) < 0.03
    assert sol["rms_px"] < 1.5

    # And the drawing: the box's floor corners land where the true camera would put them.
    import tools.show_workspace as show
    model = {k: np.asarray(v, float) if k != "size" else v for k, v in sol["model"].items()}
    out = show.render(np.zeros((H, W, 3), np.uint8), model, ((-0.4, 0.4), (0.0, 0.6), (0.0, 0.8)), 0.0)
    assert out.shape == (H, W, 3) and out.any()
    truth, _ = cfa.project([(-0.4, 0.0, 0.0), (0.4, 0.6, 0.0)], synth.CAMERA, synth.K, synth.D)
    got, _ = cfa.project([(-0.4, 0.0, 0.0), (0.4, 0.6, 0.0)], model["T_world_cam"], model["K"], model["D"])
    assert np.linalg.norm(got - truth, axis=1).max() < 4.0


def test_a_solution_becomes_the_arms_calibration_keeping_what_else_it_held(tmp_path, monkeypatch):
    """Before, solution.json was all it wrote: the UR5's live calibration was filled in by hand, and a re-run did nothing."""
    import tools.calibrate_camera_by_hand as tool
    from mrcf import overhead_calibration as oc

    monkeypatch.setattr(oc, "PATH", tmp_path / "overhead_calibration.json")
    oc.save({"T_arm_cam": np.eye(4).tolist(), "T_gf_tag": [[1.0, 0, 0, 0.01]] + np.eye(4)[1:].tolist()})
    box = ((-0.4, 0.4), (0.0, 0.6), (0.0, 0.8))
    monkeypatch.setattr(tool, "gripper_box", lambda: box)
    monkeypatch.setattr(tool.rc, "TABLE_Z", 0.0)
    solution = {"model": {"size": [W, H], "K": synth.K.tolist(), "D": list(synth.D), "T_world_cam": synth.CAMERA.tolist()},
                "tag_id": TAG_ID, "link": 4, "n": 20, "rms_px": 0.6, "time": "2026-11-12T20:00:00"}
    record = tool.calibration_record(solution, tmp_path / "run")
    assert record["T_gf_tag"][0][3] == 0.01                   # the gripper tag's pose: the arm's, kept
    assert np.allclose(record["T_arm_cam"], synth.CAMERA) and np.allclose(record["lens"]["K"], synth.K)

    # The homography takes the table's pixels (at the pipeline's render size) to the arm's (x, y).
    xy = np.array([[0.1, 0.3], [-0.2, 0.45], [0.3, 0.1]])
    px, _ = cfa.project(np.c_[xy, np.zeros(3)], synth.CAMERA, synth.K, synth.D)
    px = px * [tool.rc.RENDER_W / W, tool.rc.RENDER_H / H]
    h = (np.asarray(record["homography"]) @ np.c_[px, np.ones(3)].T).T
    assert np.abs(h[:, :2] / h[:, 2:] - xy).max() < 0.015

    oc.save(record)
    namespace = {"OVERHEAD_CAM_MODEL": {"size": (640, 480), "K": np.eye(3).tolist(), "D": [0.0] * 5,
                                        "T_world_cam": np.eye(4).tolist()}}
    assert oc.apply_to(namespace)
    model = namespace["OVERHEAD_CAM_MODEL"]
    assert np.allclose(model["K"], synth.K) and model["size"] == (W, H)
    assert np.allclose(model["T_world_cam"], synth.CAMERA)


def test_a_camera_with_no_lens_starts_from_its_field_of_view(monkeypatch):
    import tools.calibrate_camera_by_hand as tool

    monkeypatch.setattr(tool, "camera_model_of", lambda camera: None)
    monkeypatch.setattr(sys, "argv", ["calibrate_camera_by_hand.py", "--camera", "side", "--fov-deg", "90"])
    K, D, source = tool.lens_for("side", (1280, 720))
    assert K[0, 0] == pytest.approx(640.0) and K[1, 1] == pytest.approx(640.0) and K[0, 2] == 640.0
    assert not np.any(D) and "90 deg" in source
    monkeypatch.setattr(sys, "argv", ["calibrate_camera_by_hand.py", "--camera", "side"])
    with pytest.raises(SystemExit, match="--fov-deg"):
        tool.lens_for("side", (1280, 720))


def test_it_records_on_a_clock_as_exact_as_linuxs(tmp_path, monkeypatch):
    # CI, 2026-11-13: Linux sleeps exactly 0.05 s, and the readings kept for "still"
    # never spanned 0.9 x STILL_S; macOS oversleeps (53.8 ms here) and hid it.
    import tools.calibrate_camera_by_hand as tool
    sleep = time.sleep                                   # the real one, before it is replaced

    def exact(s):
        end = time.monotonic() + s
        sleep(max(0.0, s - 0.003))
        while time.monotonic() < end:
            pass
    monkeypatch.setattr(tool.time, "sleep", exact)
    test_record_then_solve_from_disk(tmp_path, monkeypatch)


def test_another_cameras_fit_is_never_saved_as_the_overhead_calibration(tmp_path, monkeypatch, capsys):
    """--camera side once offered to write the side camera's pose over the overhead camera's."""
    import tools.calibrate_camera_by_hand as tool
    monkeypatch.setattr(tool, "STILL_S", 0.12)
    monkeypatch.setattr(tool, "TIMEOUT_S", 20.0)
    monkeypatch.setattr(tool, "lens_for", lambda camera, size: (synth.K, synth.D, "the test's lens"))
    saved = []
    monkeypatch.setattr(tool.overhead_calibration, "save", lambda record: saved.append(record))
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")       # it would have been yes
    tool.record(FreedriveArm(_visible_poses(10)), "side", (W, H), want=10, run=tmp_path)
    monkeypatch.setattr(sys, "argv", ["calibrate_camera_by_hand.py", "--tag-mm", "40", "--save"])
    tool.solve_run(tmp_path)
    assert saved == []
    out = capsys.readouterr().out
    assert "CAMERAS['side'] as model=" in out and '"T_world_cam"' in out
