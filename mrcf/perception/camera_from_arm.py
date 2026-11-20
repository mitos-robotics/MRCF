"""
camera_from_arm.py — where a fixed camera is, from a tag stuck on the arm and
seen at several poses. No board, no target on the table: the arm's own joints
say where the link is each time, the camera says where the tag is, and the two
unknowns that tie them — the camera in the arm's frame, and the tag on its
link — come out of all the sightings together (OpenCV's robot-world/hand-eye
solve, then a fit to the tag's corner pixels).

What a sighting needs: the link's frame in the arm's world frame at that pose
(RobotEnv.link_frames) and the tag's four corners in the image (fiducials.
detect_tags). Which link the tag is on need not be known: solve() is run for
each candidate and the one the sightings agree with best wins (best_link).

The fit also takes the tag's printed size (a printer scales it) and, if asked,
the lens's focal length and first radial term, starting from a lens model of
the same camera (the size of an image does not change the lens; a mount
does not either). The poses must turn the tag about more than one axis: a
sweep that only slides the arm, or only turns the base, leaves the solve
underdetermined, and it says so (rank of the rotations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from mrcf.perception import fiducials

# Below this many sightings the solve is refused: 6 unknowns each side, and
# the tag's two-fold ambiguity, want a margin.
MIN_SIGHTINGS = 6
# The poses must turn the tag this much about at least two different axes.
MIN_SPREAD_DEG = 15.0


@dataclass
class Sighting:
    """One still pose: the link's frame (world <- link, 4x4) and the tag's corners (4x2 px)."""
    T_world_link: np.ndarray
    corners: np.ndarray


@dataclass
class Solution:
    T_world_cam: np.ndarray            # camera -> the arm's world frame
    T_link_tag: np.ndarray             # tag -> its link
    tag_size_m: float                  # the black square's edge, as fitted
    K: np.ndarray
    D: np.ndarray
    rms_px: float                      # mean corner reprojection error
    per_sighting_px: list = field(default_factory=list)
    n: int = 0

    def camera_model(self, size) -> dict:
        """As a robot_config camera model: {size, K, D, T_world_cam} (lists)."""
        return {"size": tuple(int(v) for v in size), "K": np.round(self.K, 3).tolist(),
                "D": np.round(self.D, 5).tolist(), "T_world_cam": np.round(self.T_world_cam, 6).tolist()}


def inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _T(rvec, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def rotation_spread_deg(sightings: list) -> "tuple[float, float]":
    """
    How much the link turned across the sightings about its two strongest
    independent axes (degrees): the hand-eye solve needs both well above zero.
    """
    R0 = sightings[0].T_world_link[:3, :3]
    vecs = [cv2.Rodrigues(R0.T @ s.T_world_link[:3, :3])[0].ravel() for s in sightings[1:]]
    if not vecs:
        return 0.0, 0.0
    sv = np.linalg.svd(np.asarray(vecs), compute_uv=False)
    sv = np.r_[sv, 0.0, 0.0]
    return float(np.degrees(sv[0] / np.sqrt(len(vecs)))), float(np.degrees(sv[1] / np.sqrt(len(vecs))))


def _closed_form(sightings, K, D, size_m):
    """The hand-eye solve on the tag's own poses; picks each tag's ambiguous pose by agreement."""
    cands = [[p.matrix() for p in fiducials.tag_poses(s.corners, size_m, K, D)] for s in sightings]
    keep = [i for i, c in enumerate(cands) if c]
    if len(keep) < MIN_SIGHTINGS:
        return None
    sightings = [sightings[i] for i in keep]
    cands = [cands[i] for i in keep]
    pick = [0] * len(sightings)
    X = Y = None
    for _ in range(4):
        A = [inv(c[k]) for c, k in zip(cands, pick)]                  # camera -> tag
        B = [inv(s.T_world_link) for s in sightings]                   # world -> link
        R_bw, t_bw, R_gc, t_gc = cv2.calibrateRobotWorldHandEye(
            [a[:3, :3] for a in A], [a[:3, 3] for a in A], [b[:3, :3] for b in B], [b[:3, 3] for b in B])
        X = np.vstack([np.c_[R_bw, t_bw.ravel()], [0, 0, 0, 1]])      # world -> camera (T_cam_world)
        Y = inv(np.vstack([np.c_[R_gc, t_gc.ravel()], [0, 0, 0, 1]]))  # tag -> link (T_link_tag)
        new = []
        for s, c in zip(sightings, cands):
            pred = X @ s.T_world_link @ Y
            new.append(int(np.argmin([fiducials.rotation_angle_deg(m[:3, :3], pred[:3, :3]) for m in c])))
        if new == pick:
            break
        pick = new
    return X, Y


def _residuals(p, sightings, K0, D0, size0, refine_lens):
    X, Y = _T(p[0:3], p[3:6]), _T(p[6:9], p[9:12])
    size = size0 * (1.0 + p[12])
    K, D = K0.copy(), D0.copy()
    if refine_lens:
        K[0, 0] *= 1.0 + p[13]
        K[1, 1] *= 1.0 + p[13]
        D[0] += p[14]
    obj_h = np.c_[fiducials.tag_object_points(size), np.ones(4)].T
    out = []
    for s in sightings:
        Pc = (X @ s.T_world_link @ Y @ obj_h)[:3].T
        if (Pc[:, 2] <= 0).any():
            return None
        px, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, D)
        out.append((px.reshape(-1, 2) - np.asarray(s.corners, float).reshape(4, 2)).ravel())
    r = np.concatenate(out)
    if refine_lens:                          # gentle priors: the lens moves only as far as the sightings insist
        r = np.r_[r, p[13] / 0.05, p[14] / 0.1]
    return r, K, D, size


def _refine(sightings, X, Y, K0, D0, size0, refine_lens, iters=60):
    p = np.concatenate([cv2.Rodrigues(X[:3, :3])[0].ravel(), X[:3, 3], cv2.Rodrigues(Y[:3, :3])[0].ravel(),
                        Y[:3, 3], [0.0, 0.0, 0.0]])
    n_free = 15 if refine_lens else 13
    got = _residuals(p, sightings, K0, D0, size0, refine_lens)
    if got is None:
        return None
    r = got[0]
    lam = 1e-3
    for _ in range(iters):
        J = np.zeros((r.size, p.size))
        for j in range(n_free):
            dp = np.zeros(p.size)
            dp[j] = 1e-6
            g = _residuals(p + dp, sightings, K0, D0, size0, refine_lens)
            if g is None:
                return None
            J[:, j] = (g[0] - r) / 1e-6
        with np.errstate(all="ignore"):         # Accelerate's matmul warns on finite input
            A, grad = J.T @ J, J.T @ r
        A = A + lam * np.diag(np.diag(A) + 1e-12)
        A[n_free:, n_free:] += np.eye(p.size - n_free)       # the fixed parameters stay put
        step = -np.linalg.solve(A, grad)
        trial = _residuals(p + step, sightings, K0, D0, size0, refine_lens)
        if trial is not None and trial[0] @ trial[0] < r @ r:
            p, r, lam = p + step, trial[0], lam * 0.3
            if np.abs(step).max() < 1e-10:
                break
        else:
            lam *= 10
            if lam > 1e8:
                break
    _, K, D, size = _residuals(p, sightings, K0, D0, size0, refine_lens)
    corner_r = r[:len(sightings) * 8].reshape(-1, 4, 2)
    per = np.linalg.norm(corner_r, axis=2).mean(axis=1)
    return _T(p[0:3], p[3:6]), _T(p[6:9], p[9:12]), size, K, D, float(per.mean()), per.tolist()


def solve(sightings: list, K, D, tag_size_m: float, refine_lens: bool = True) -> "Solution | None":
    """
    The camera's pose in the arm's world frame and the tag's on its link, from
    sightings of one tag on one link. None when too few sightings, too little
    turning, or nothing fits.
    """
    K0, D0 = np.asarray(K, float), np.r_[np.asarray(D, float).ravel(), np.zeros(5)][:5]
    if len(sightings) < MIN_SIGHTINGS or min(rotation_spread_deg(sightings)) < MIN_SPREAD_DEG / 3:
        return None
    start = _closed_form(sightings, K0, D0, tag_size_m)
    if start is None:
        return None
    X0, Y0 = start
    got = _refine(sightings, X0, Y0, K0, D0, tag_size_m, refine_lens)
    if got is None:
        return None
    X, Y, size, K1, D1, rms, per = got
    return Solution(T_world_cam=inv(X), T_link_tag=Y, tag_size_m=size, K=K1, D=D1, rms_px=rms,
                    per_sighting_px=per, n=len(sightings))


def best_link(frames_per_pose: list, corners_per_pose: list, K, D, tag_size_m: float,
              links: "list[int] | None" = None, refine_lens: bool = True) -> "tuple[int, Solution, dict] | None":
    """
    Which link the tag is on, and the solution for it: each candidate link
    (index into each pose's frames; default all but the base) solved, the
    lowest reprojection wins. Returns (link, solution, {link: rms px}), or None.
    """
    n_links = len(frames_per_pose[0])
    links = list(range(1, n_links)) if links is None else links
    found, scores = {}, {}
    for link in links:
        sightings = [Sighting(np.asarray(f[link], float), np.asarray(c, float))
                     for f, c in zip(frames_per_pose, corners_per_pose)]
        sol = solve(sightings, K, D, tag_size_m, refine_lens=refine_lens)
        # A tag more than half a metre from its link's origin is a fit that went wrong.
        if sol is not None and np.linalg.norm(sol.T_link_tag[:3, 3]) < 0.5:
            found[link], scores[link] = sol, sol.rms_px
    if not found:
        return None
    link = min(scores, key=lambda k: scores[k])
    return link, found[link], scores


def project(points_world, T_world_cam, K, D) -> "tuple[np.ndarray, np.ndarray]":
    """
    World points -> pixels (N x 2), and which of them the pixels mean anything
    for (N bools): in front of the camera, and inside the angle where the lens
    model still bends one way (a strong barrel term folds back beyond it, and
    a point far out of view would land inside the image).
    """
    P = np.asarray(points_world, float).reshape(-1, 3)
    T_cam_world = inv(np.asarray(T_world_cam, float))
    Pc = (T_cam_world[:3, :3] @ P.T).T + T_cam_world[:3, 3]
    front = Pc[:, 2] > 1e-3
    K, D = np.asarray(K, float), np.asarray(D, float)
    px, _ = cv2.projectPoints(np.where(front[:, None], Pc, [[0, 0, 1.0]]), np.zeros(3), np.zeros(3), K, D)
    px = px.reshape(-1, 2)
    # Toward where the lens model folds back, it squeezes ever more of the world
    # into the same pixels: past where it stretches by less than a third, the
    # pixels say little about where a point is, and beyond the fold nothing.
    d = np.r_[D.ravel(), np.zeros(5)]
    k1, k2, k3 = d[0], d[1], d[4]
    r2 = (Pc[:, 0] ** 2 + Pc[:, 1] ** 2) / np.maximum(Pc[:, 2], 1e-9) ** 2
    front &= 1 + 3 * k1 * r2 + 5 * k2 * r2 ** 2 + 7 * k3 * r2 ** 3 > 0.3
    return px, front
