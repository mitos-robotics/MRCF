"""Finding a wrist camera's mount from views of a flat grid — synthetic arm poses, no robot."""

import numpy as np

try:
    import cv2
except ImportError:          # the framework's maths needs OpenCV; skip where it is absent
    import pytest
    pytest.skip("needs OpenCV", allow_module_level=True)

from mrcf.perception import hand_eye as he

K = np.array([[343.0, 0, 341], [0, 342.0, 263], [0, 0, 1]])
D = np.array([0.085, -0.124, 0.0, 0.0, 0.037])
PITCH = 0.0364


def gripper_pose(rng, centre):
    """A gripper hovering over the grid, approach axis (+z) down, leant and rolled at random."""
    down = np.diag([1.0, -1.0, -1.0])
    lean = cv2.Rodrigues(np.r_[rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), 0.0])[0]
    roll = cv2.Rodrigues(np.r_[0.0, 0.0, rng.uniform(-np.pi, np.pi)])[0]
    T = np.eye(4)
    T[:3, :3] = lean @ down @ roll
    T[:3, 3] = np.r_[centre + rng.uniform(-0.06, 0.06, 2), rng.uniform(0.10, 0.17)]
    return T


def views(rng, X, grid_world, n=30, noise_px=0.3):
    gi = np.array([[i, j] for j in range(7) for i in range(7)], float)
    out = []
    while len(out) < n:
        T = gripper_pose(rng, grid_world[:, :2].mean(0))
        px = he.project_world(grid_world, T @ X, K, D)
        vis = np.isfinite(px).all(1) & (px[:, 0] > 5) & (px[:, 0] < 635) & (px[:, 1] > 5) & (px[:, 1] < 380)
        if vis.sum() < 9:
            continue
        ids = np.nonzero(vis)[0]
        ij = gi[ids] - gi[ids].min(0)            # a patch, indexed from wherever the detector started
        if rng.random() < 0.5:
            ij = ij[:, ::-1]
        out.append(he.GridView(T, px[ids] + rng.normal(0, noise_px, (len(ids), 2)), ij))
    return out


def test_mount_recovered_to_a_tenth_of_a_millimetre():
    rng = np.random.default_rng(3)
    # The camera above and behind the jaws, looking forward and down.
    X = he._T(cv2.Rodrigues(np.array([0.6, 0.0, 0.0]))[0], [0.0, -0.05, -0.03])
    gi = np.array([[i, j] for j in range(7) for i in range(7)], float)
    grid_world = np.c_[gi * PITCH + [0.11, -0.11], np.zeros(49)]
    vs = views(rng, X, grid_world)
    X0, plane_z, d0 = he.initial_from_plane(vs, K, D, PITCH)
    assert abs(plane_z) < 0.006 and np.linalg.norm(X0[:3, 3] - X[:3, 3]) < 0.008
    X1, d1 = he.refine(vs, X0, grid_world, K, D)
    turned = np.degrees(np.linalg.norm(cv2.Rodrigues(X1[:3, :3] @ X[:3, :3].T)[0]))
    assert np.linalg.norm(X1[:3, 3] - X[:3, 3]) < 0.0005 and turned < 0.1
    assert d1["reproj_mean_px"] < 0.6


def _rot(axis, deg):
    v = np.zeros(3)
    v["xyz".index(axis)] = np.radians(deg)
    return he._T(cv2.Rodrigues(v)[0], np.zeros(3))


def _move(x=0.0, z=0.0):
    return he._T(np.eye(3), [x, 0.0, z])


def toy_arm(q_deg):
    """A five-joint arm (pan, lift, elbow, wrist flex, wrist roll): its gripper, approach axis +z, points
    down when lift + elbow + flex = 90 deg."""
    pan, lift, elbow, flex, roll = q_deg
    return (_rot("z", pan) @ _move(z=0.40) @ _rot("y", lift) @ _move(x=0.12) @ _rot("y", elbow)
            @ _move(x=0.13) @ _rot("y", flex) @ _move(x=0.06) @ _rot("y", 90.0) @ _rot("z", roll))


def test_joint_readings_a_few_degrees_off_are_found_with_the_mount():
    # The arm reads pan, lift, elbow and flex a little off, the same way every
    # time: no mount alone explains the views; with the corrections fitted
    # too, mount and corrections both come back — and hold on views the fit
    # did not see. From the mount refine() makes of a start some mm off.
    rng = np.random.default_rng(5)
    X = he._T(cv2.Rodrigues(np.array([0.25, 0.0, 0.0]))[0], [0.0, -0.04, -0.04])
    bias = np.array([1.2, 1.0, 4.0, -0.8, 0.0])           # true = read + bias
    gi = np.array([[i, j] for j in range(7) for i in range(7)], float)
    grid_world = np.c_[gi * PITCH + [0.148 - 3 * PITCH, -3 * PITCH], np.zeros(49)]
    vs, q_read = [], []
    while len(vs) < 36:
        lift, elbow = rng.uniform(5, 45), rng.uniform(20, 70)    # varied: too alike, and pairing goes wrong
        q = np.array([rng.uniform(-15, 15), lift, elbow, 90 - lift - elbow + rng.uniform(-15, 15),
                      rng.uniform(-30, 30)])
        px = he.project_world(grid_world, toy_arm(q + bias) @ X, K, D)
        vis = np.isfinite(px).all(1) & (px[:, 0] > 5) & (px[:, 0] < 635) & (px[:, 1] > 5) & (px[:, 1] < 380)
        if vis.sum() < 16:
            continue
        ids = np.nonzero(vis)[0]
        vs.append(he.GridView(toy_arm(q), px[ids] + rng.normal(0, 0.3, (len(ids), 2)), gi[ids] - gi[ids].min(0)))
        q_read.append(q)
    X0 = he._T(cv2.Rodrigues(np.array([0.28, 0.02, 0.0]))[0], [0.005, -0.045, -0.035])
    X1, d1 = he.refine(vs, X0, grid_world, K, D)
    assert d1["reproj_mean_px"] > 3.0
    X3, dq, shift, d3 = he.refine_with_joints(vs, np.array(q_read), toy_arm, X1, grid_world, K, D,
                                              free=[True, True, True, True, False])
    assert np.allclose(dq, bias, atol=0.15) and dq[4] == 0.0
    turned = np.degrees(np.linalg.norm(cv2.Rodrigues(X3[:3, :3] @ X[:3, :3].T)[0]))
    assert np.linalg.norm(X3[:3, 3] - X[:3, 3]) < 0.001 and turned < 0.2
    assert np.linalg.norm(shift) < 0.001
    assert d3["reproj_mean_px"] < 0.6 and d3["held_out_mean_px"] < 0.8
    # Started from an earlier calibration's answer (dq0), it stays there.
    _, dq_again, _, d_again = he.refine_with_joints(vs, np.array(q_read), toy_arm, X3, grid_world, K, D,
                                                    free=[True, True, True, True, False], dq0=dq)
    assert np.allclose(dq_again, dq, atol=0.05) and d_again["reproj_mean_px"] < 0.6
