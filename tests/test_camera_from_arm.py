"""
perception/camera_from_arm.py on synthetic sightings: a UR5's kinematics, a
camera high over the table at a slant, a tag on one of its links, the corners
projected through a wide lens with pixel noise. The solve must find the link,
the camera to millimetres, and the tag's printed size.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mrcf.backends.ur5 import kinematics  # noqa: E402
from mrcf.perception import camera_from_arm as cfa  # noqa: E402
from mrcf.perception import fiducials  # noqa: E402

K = np.array([[858.78, 0.0, 640.0], [0.0, 858.78, 360.0], [0.0, 0.0, 1.0]])
D = np.array([-0.3449, 0.0, 0.0, 0.0, 0.0])
SIZE_M = 0.040
Q_NOW = np.radians([83.71, -77.19, -93.43, -94.81, -267.53, 271.88])


def _look_at(eye, target):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = eye
    return T


def _sightings(T_world_cam, link, T_link_tag, n=24, noise_px=0.3, seed=0, lens=(K, D)):
    rng = np.random.default_rng(seed)
    T_cam_world = cfa.inv(T_world_cam)
    frames, corners = [], []
    obj_h = np.c_[fiducials.tag_object_points(SIZE_M), np.ones(4)].T
    while len(frames) < n:
        q = Q_NOW + np.radians(rng.uniform(-1, 1, 6) * [30, 20, 20, 25, 20, 30])
        f = kinematics.frames(q)
        T_cam_tag = T_cam_world @ f[link] @ T_link_tag
        Pc = (T_cam_tag @ obj_h)[:3].T
        to_cam = -T_cam_tag[:3, 3] / np.linalg.norm(T_cam_tag[:3, 3])
        if Pc[:, 2].min() <= 0.1 or T_cam_tag[:3, 2] @ to_cam < 0.3:   # behind, or facing away
            continue
        px, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), *lens)
        px = px.reshape(4, 2) + rng.normal(0, noise_px, (4, 2))
        if not ((px >= 0) & (px < [1280, 720])).all():
            continue
        frames.append(f)
        corners.append(px)
    return frames, corners


CAMERA = _look_at(np.array([0.25, 1.15, 1.35]), [0.0, 0.25, 0.25])
TAG = np.eye(4)
TAG[:3, :3] = cv2.Rodrigues(np.array([0.1, -0.15, 0.4]))[0]
TAG[:3, 3] = [0.01, -0.02, 0.065]


def test_it_finds_the_link_and_the_camera():
    frames, corners = _sightings(CAMERA, 4, TAG)
    got = cfa.best_link(frames, corners, K, D, SIZE_M * 1.02)        # the printer scaled the tag
    assert got is not None
    link, sol, scores = got
    assert link == 4, scores
    # Along its own axis the camera's distance trades with the focal length (fitted too): cm, not mm.
    assert np.linalg.norm(sol.T_world_cam[:3, 3] - CAMERA[:3, 3]) < 0.015
    assert fiducials.rotation_angle_deg(sol.T_world_cam[:3, :3], CAMERA[:3, :3]) < 0.5
    table = np.array([[x, y, 0.0] for x in (-0.4, 0.0, 0.4) for y in (0.0, 0.3, 0.6)])
    truth, _ = cfa.project(table, CAMERA, K, D)
    got, _ = cfa.project(table, sol.T_world_cam, sol.K, sol.D)
    assert np.linalg.norm(got - truth, axis=1).max() < 2.5        # the corners, beyond the swept region, worst
    assert abs(sol.tag_size_m - SIZE_M) < 0.0005
    assert sol.rms_px < 0.6
    assert min(v for k, v in scores.items() if k != 4) > 3 * sol.rms_px   # the other links fit far worse


def test_a_lens_borrowed_from_another_rig_still_puts_the_table_in_the_right_place():
    # The same camera model on another rig, 2.5% off in focal length: the focal
    # length and the camera's distance trade against each other, so the camera
    # itself can be a couple of cm out along its axis; what the fit keeps right
    # is where the workspace lands in the image, which is what drawing a box or
    # checking a position needs.
    true_K = K.copy()
    true_K[0, 0] = true_K[1, 1] = 858.78 * 1.025
    true_D = D + [0.02, 0, 0, 0, 0]
    frames, corners = _sightings(CAMERA, 4, TAG, n=30, lens=(true_K, true_D))
    sightings = [cfa.Sighting(f[4], c) for f, c in zip(frames, corners)]
    gx, gy = np.meshgrid(np.linspace(-0.5, 0.5, 11), np.linspace(-0.2, 0.7, 10))
    table = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    truth, front = cfa.project(table, CAMERA, true_K, true_D)
    seen = front & (truth >= 0).all(1) & (truth < [1280, 720]).all(1)

    def table_error_px(sol):
        got, _ = cfa.project(table, sol.T_world_cam, sol.K, sol.D)
        return np.linalg.norm(got[seen] - truth[seen], axis=1)

    fixed = cfa.solve(sightings, K, D, SIZE_M, refine_lens=False)
    fitted = cfa.solve(sightings, K, D, SIZE_M)
    assert fitted is not None and fixed is not None
    assert fitted.rms_px < 0.6
    assert np.median(table_error_px(fitted)) < 2.0 and table_error_px(fitted).max() < 3.0
    assert table_error_px(fitted).max() < table_error_px(fixed).max()
    assert np.linalg.norm(fitted.T_world_cam[:3, 3] - CAMERA[:3, 3]) < 0.03


def test_it_refuses_too_few_or_unturned_sightings():
    frames, corners = _sightings(CAMERA, 4, TAG, n=4)
    assert cfa.solve([cfa.Sighting(f[4], c) for f, c in zip(frames, corners)], K, D, SIZE_M) is None
    f0, c0 = _sightings(CAMERA, 4, TAG, n=1)
    same = [cfa.Sighting(f0[0][4], c0[0])] * 10                           # ten times the same pose
    assert cfa.solve(same, K, D, SIZE_M) is None


def test_project_puts_world_points_where_the_camera_sees_them():
    frames, corners = _sightings(CAMERA, 4, TAG, n=1, noise_px=0.0)
    obj = fiducials.tag_object_points(SIZE_M)
    world = (frames[0][4] @ TAG @ np.c_[obj, np.ones(4)].T)[:3].T
    px, front = cfa.project(world, CAMERA, K, D)
    assert front.all()
    assert np.abs(px - corners[0]).max() < 1e-6
    _, front = cfa.project([CAMERA[:3, 3] - CAMERA[:3, 2]], CAMERA, K, D)    # behind the lens
    assert not front.any()


def test_a_blocks_sides_are_read_from_its_outline():
    from mrcf.perception import plane_geometry as pg
    T = _look_at(np.array([0.30, 0.45, 1.00]), [0.32, 0.30, 0.0])       # high, looking down at a slant
    view = pg.PlaneView.from_camera_model({"K": K, "D": [0.0] * 5, "T_world_cam": T, "size": (1280, 720)})
    for turn in (0.0, 17.0, 30.0, 62.0):
        a = np.radians(turn)
        corners = np.array([[np.cos(a) * x - np.sin(a) * y, np.sin(a) * x + np.cos(a) * y]
                            for x, y in ((-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02))]) + [0.32, 0.30]
        px, _ = cfa.project(np.c_[corners, np.zeros(4)], T, K, [0.0] * 5)
        mask = np.zeros((720, 1280), np.uint8)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
        got = pg.side_direction_deg(mask, view)
        assert min(abs(got - turn) % 90, 90 - abs(got - turn) % 90) < 2.0, (turn, got)


def test_a_blocks_centre_is_found_by_its_top_face():
    # The silhouette of a 30 mm cube as a slanted camera sees it: its top and the sides toward the
    # camera. Its centre, found by the top face's far edges, within 2 mm wherever it stands.
    from mrcf.perception import plane_geometry as pg
    T = _look_at(np.array([0.16, 0.67, 1.09]), [0.40, 0.35, 0.0])          # as on the rig
    view = pg.PlaneView.from_camera_model({"K": K, "D": [0.0] * 5, "T_world_cam": T, "size": (1280, 720)})
    for (x, y), turn in (((0.58, 0.375), 43.0), ((0.36, 0.22), 10.0), ((0.30, 0.45), 70.0)):
        a, h = np.radians(turn), 0.015
        corners = [(x + np.cos(a) * u - np.sin(a) * v, y + np.sin(a) * u + np.cos(a) * v, z)
                   for u in (-h, h) for v in (-h, h) for z in (0.0, 0.030)]
        px, _ = cfa.project(np.array(corners), T, K, [0.0] * 5)
        mask = np.zeros((720, 1280), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(px).astype(np.int32)), 1)
        got = pg.box_centre_xy(mask, view, 30.0, 30.0, turn)
        assert np.linalg.norm(got - [x, y]) < 0.002, (x, y, turn, got)
