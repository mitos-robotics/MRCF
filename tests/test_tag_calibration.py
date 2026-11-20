"""
The wrist camera's mount calibrated against a sheet of tags on a bare table
(tools/so101/calibrate_gripper_pose.py --target tags): on the SO-101's own
kinematics, pictures of the table target rendered from a known "true" mount
3 mm and 1 deg off the arm's config, and the fit finds the true one.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]


def test_a_tag_sheet_on_the_table_calibrates_the_wrist_camera(tmp_path, monkeypatch):
    from arm_baselines import Skipped, offline_env
    from mrcf import backends
    from mrcf.backends.so101.kinematics import JOINT_NAMES
    from mrcf.perception import fiducials

    sys.path.insert(0, str(ROOT))
    import tools.so101.calibrate_gripper_pose as cal
    from tools.so101.visualise_ik import _FakeFollower

    arm = backends.arm("so101"); rc = arm.config
    # The wrist camera's shift as this test was written against: the rig's own is refitted now and then
    # (2026-11-20: -5.6, -8.0 -> -8.7, -13.2 mm), and the solve's tolerance is not about it.
    monkeypatch.setattr(rc, "GRIPPER_CAM_SCAN_SHIFT_M", (-0.0056, -0.0080, -0.0025))
    try:
        env = offline_env(arm)
    except Skipped as why:
        pytest.skip(str(why))
    if rc.GRIPPER_CAM_K is None or rc.GRIPPER_CAM_POSE is None:
        pytest.skip("the SO-101's wrist camera is not calibrated in its config")

    env._robot = _FakeFollower(rc.HOME_DEGREES, rc.MAX_RELATIVE_TARGET_DEG); env._robot.calibration = {}
    kin = env._kinematics
    # The fit's own arm (visualise_ik.offline_arm) wants LeRobot's calibration file in the home folder, which CI
    # never has: the fit is handed this arm, the kinematics the views below are rendered with.
    import tools.so101.visualise_ik as vik
    monkeypatch.setattr(vik, "offline_arm", lambda: env)
    K, D = np.array(rc.GRIPPER_CAM_K, float), np.array(rc.GRIPPER_CAM_DIST, float)
    W, H = rc.RENDER_W, rc.RENDER_H
    X_cfg = np.array(rc.GRIPPER_CAM_POSE, float)
    dq_cfg = np.array([(rc.GRIPPER_CAM_JOINT_DEG or {}).get(n, 0.0) for n in JOINT_NAMES])
    # the truth: the mount 3 mm and 1 deg off what the config has, the joints read as the config says
    R_off = cv2.Rodrigues(np.radians([0.6, -0.5, 0.6]))[0]
    X_true = X_cfg.copy(); X_true[:3, :3] = X_cfg[:3, :3] @ R_off; X_true[:3, 3] += [0.002, -0.0015, 0.0015]
    surface = 0.0007
    # the sheet: 3 x 4 tags, 30 mm, ~ 59 mm pitch across, ~ 52 mm down, centred at (0.25, 0.02)
    tags = {}
    for r in range(4):
        for c in range(3):
            tid = fiducials.TABLE_TARGET_IDS[r * 3 + c]
            cx, cy = 0.25 + (r - 1.5) * 0.052, 0.02 + (c - 1) * 0.059
            h = fiducials.TABLE_TARGET_TAG_M / 2
            tags[tid] = np.array([[cx - h, cy + h, surface], [cx - h, cy - h, surface], [cx + h, cy - h, surface], [cx + h, cy + h, surface]])
    marker_px = 400
    def render(T_world_cam):
        img = np.full((H, W, 3), 225, np.uint8)
        T_cw = np.linalg.inv(T_world_cam)
        d = cv2.aruco.getPredefinedDictionary(fiducials.TAG_DICTIONARY)
        for tid, corners in tags.items():
            pc = corners @ T_cw[:3, :3].T + T_cw[:3, 3]
            if (pc[:, 2] <= 0.02).any(): continue
            px, _ = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), K, D)
            px = px.reshape(4, 2)
            area = 0.5 * np.sum(px[:, 0] * np.roll(px[:, 1], -1) - np.roll(px[:, 0], -1) * px[:, 1])
            if area < 0:                       # seen from above the paper the marker turns clockwise in the picture
                px = px[[0, 3, 2, 1]]
            if (px < -50).any() or (px[:, 0] > W + 50).any() or (px[:, 1] > H + 50).any(): continue
            m = cv2.aruco.generateImageMarker(d, tid, marker_px, borderBits=1)
            pad = marker_px // 8
            big = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            src = np.float32([[pad, pad], [pad + marker_px, pad], [pad + marker_px, pad + marker_px], [pad, pad + marker_px]])
            Hm = cv2.getPerspectiveTransform(src, px.astype(np.float32))
            warped = cv2.warpPerspective(cv2.cvtColor(big, cv2.COLOR_GRAY2RGB), Hm, (W, H), borderValue=(0, 0, 0))
            mask = cv2.warpPerspective(np.full(big.shape, 255, np.uint8), Hm, (W, H)) > 0
            img[mask] = warped[mask]
        return img
    # corner order: detect_tags gives corners in the marker's own order (TL, TR, BR, BL of the image marker);
    # the 'world' here is built to match that order in the rendered image by construction below.
    records = []
    centre = np.vstack(list(tags.values()))[:, :2].mean(0)
    for h in cal.WIDE_HEIGHTS_M:
        for dx, dy in cal.WIDE_OFFSETS_M:
            target = np.array([centre[0] + dx, centre[1] + dy, surface + h])
            found = env.approach_orientation(target)
            if found is None or not env.move_ee_to(target, orientation=found[0]):
                continue
            base = env.get_joint_degrees()
            for dflex, droll in cal.WIDE_WRIST_VARIATIONS:
                env.goto_joint_degrees(dict(base, wrist_flex=base["wrist_flex"] + dflex, wrist_roll=base["wrist_roll"] + droll))
                q = env._q_arm_deg()
                T_true = kin.forward_kinematics(np.append(q + dq_cfg, 0.0)) @ X_true
                frame = render(T_true)
                # the world corners in detect order: project each tag's corners with the truth and match by nearest
                seen = fiducials.detect_tags(frame)
                px, world = [], []
                T_cw = np.linalg.inv(T_true)
                for tid, c in seen.items():
                    if tid not in tags: continue
                    pc = tags[tid] @ T_cw[:3, :3].T + T_cw[:3, 3]
                    proj, _ = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), K, D)
                    proj = proj.reshape(4, 2)
                    c = np.asarray(c, float).reshape(4, 2)
                    order = [int(np.argmin(np.linalg.norm(proj - ci, axis=1))) for ci in c]
                    px.append(c); world.append(tags[tid][order])
                if not px: continue
                records.append({"frame": "", "q_deg": q.tolist(), "T_world_gripper": kin.forward_kinematics(np.append(q, 0.0)).tolist(),
                                "corners_px": np.vstack(px).tolist(), "world": np.vstack(world).tolist()})

    # Every third view: enough to fit, quick to render.
    data = {"target": "tags", "K": K.tolist(), "D": D.tolist(), "surface_z_m": surface, "views": records[::3]}
    assert len(data["views"]) >= 20
    (tmp_path / "views.json").write_text(json.dumps(data))
    monkeypatch.setattr(cal, "last_good_result", lambda: None)            # from the config's mount alone
    res = cal.fit([tmp_path])
    X = np.array(res["GRIPPER_CAM_POSE"])
    assert np.linalg.norm(X[:3, 3] - X_true[:3, 3]) < 0.0005
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(X[:3, :3] @ X_true[:3, :3].T)[0])) < 0.1
    assert res["GRIPPER_CAM_PLANE_Z_M"] == pytest.approx(rc.TABLE_Z, abs=0.001)   # said at TABLE_Z, as the board's
