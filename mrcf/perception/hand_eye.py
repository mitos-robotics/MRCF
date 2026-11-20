"""
hand_eye.py — where a wrist camera sits on the arm, from views of a flat grid.

A camera bolted to the gripper moves with it, so its pose in the world is the
gripper's pose (forward kinematics, from the joint angles) times one fixed
transform X, gripper -> camera. Once X is known, every frame the camera takes
comes with its own pose, and plane_geometry can measure objects in it on any
surface — no target needs to be in view. Finding X is a one-time calibration:
the arm visits poses over a flat grid (a chessboard) and records, at each,
its joint angles and the grid corners the camera sees.

Two stages, because a wrist camera close to the surface sees only a patch of
the grid, and which patch is ambiguous: a lattice looks the same shifted by a
square or turned by 90 degrees.

  1. From the surface alone (initial_from_plane). What a patch does pin down,
     ambiguity or not, is the camera's height above the surface and which way
     the surface's normal points in the camera. Across poses that tilt and
     roll the wrist, those give X's rotation (the normal seen by the camera
     must be the world's vertical seen by the gripper) and then its
     translation and the surface's height (each camera height is the
     gripper's height plus X's offset, turned by the gripper).
  2. Against the grid's known corners (refine), where another camera has
     located them in the world: predict each corner's pixel from the first
     answer, pair every detected corner with the nearest prediction — now
     unambiguous — and adjust X to minimise the reprojection error over all
     frames. This is what pins the camera's sideways offset and yaw to a
     millimetre, which is what aiming at an object needs.
  3. With the joints too (refine_with_joints), when the arm's kinematics are
     not exact: a correction to each joint reading, and where the other camera
     put the grid, fitted with X. A hobby arm reads its joints a few degrees
     off under load, the same way at the same pose; no one X explains views
     taken that way (an SO-101: 8-9 px, 1.8 with the joints, 2026-11-03), and
     the X that comes closest is centimetres out. Checked on views the fit
     did not see.

Units: the world frame is the arm's (metres); a grid pitch handed in must be in
that frame's units, so heights and offsets come out in it too.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GridView:
    """One calibration frame."""
    T_world_gripper: np.ndarray     # 4x4, metres, from forward kinematics at capture
    corners_px: np.ndarray          # (N, 2) detected grid corners
    grid_ij: np.ndarray             # (N, 2) their lattice indices within the detected patch


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t, float).ravel()
    return T


def patch_pose(view: GridView, K, D, pitch_m: float) -> "tuple[np.ndarray, float] | None":
    """
    (unit surface normal pointing at the camera, in camera coordinates;
    camera height above the surface, m) from one patch, or None.
    """
    if len(view.corners_px) < 6:
        return None
    obj = np.c_[np.asarray(view.grid_ij, float) * pitch_m, np.zeros(len(view.grid_ij))]
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, np.asarray(view.corners_px, np.float64), K, D,
                                             reprojectionError=2.0, iterationsCount=200,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 6:
        return None
    R = cv2.Rodrigues(rvec)[0]
    n = R[:, 2]
    t = tvec.ravel()
    h = float(abs(n @ t))
    if n @ (-t) < 0:          # orient toward the camera
        n = -n
    return n, h


def initial_from_plane(views: list, K, D, pitch_m: float) -> "tuple[np.ndarray, float, dict]":
    """
    Stage 1: (X, surface z in the world, diagnostics) from the surface alone.
    Needs poses that tilt the camera two different ways (lean and roll), or
    the rotation about the remaining axis is unobservable.
    """
    K, D = np.asarray(K, float), np.asarray(D, float)
    normals_cam, ups_grip, rows, rhs = [], [], [], []
    used = []
    for i, v in enumerate(views):
        pp = patch_pose(v, K, D, pitch_m)
        if pp is None:
            continue
        n_cam, h = pp
        Rwg, pwg = v.T_world_gripper[:3, :3], v.T_world_gripper[:3, 3]
        normals_cam.append(n_cam)
        ups_grip.append(Rwg.T @ np.array([0.0, 0.0, 1.0]))
        used.append((i, h, Rwg, pwg))
    if len(used) < 4:
        raise ValueError(f"only {len(used)} usable views — need at least 4, tilted different ways")
    a, b = np.array(normals_cam), np.array(ups_grip)
    # Wahba: rotation R_gc (camera -> gripper) best taking a_i to b_i.
    U, _, Vt = np.linalg.svd(b.T @ a)
    R_gc = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    spread = np.linalg.svd(b - b.mean(0), compute_uv=False)
    # Camera height h_i = z(p_i + R_i t) - plane_z: linear in (t, plane_z).
    for _, h, Rwg, pwg in used:
        rows.append(np.r_[Rwg[2, :], -1.0])
        rhs.append(h - pwg[2])
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    t_gc, plane_z = sol[:3], float(sol[3])
    resid = np.array(rows) @ sol - np.array(rhs)
    angle_err = np.degrees(np.arccos(np.clip(np.sum((a @ R_gc.T) * b, 1), -1, 1)))
    return _T(R_gc, t_gc), plane_z, {
        "views_used": len(used), "normal_err_deg": float(np.sqrt(np.mean(angle_err ** 2))),
        "height_rms_mm": float(np.sqrt(np.mean(resid ** 2)) * 1000),
        "tilt_spread": [float(s) for s in spread[:2]],
    }


def project_world(points_w: np.ndarray, T_world_cam: np.ndarray, K, D) -> np.ndarray:
    """(N, 2) pixels of (N, 3) world points, for a camera at T_world_cam (NaN behind it)."""
    T_cw = np.linalg.inv(T_world_cam)
    pc = points_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    px, _ = cv2.projectPoints(points_w, cv2.Rodrigues(T_cw[:3, :3])[0], T_cw[:3, 3], K, D)
    px = px.reshape(-1, 2)
    px[pc[:, 2] <= 1e-3] = np.nan
    return px


def snap(views: list, X: np.ndarray, grid_world: np.ndarray, K, D, max_frac: float = 0.35) -> list:
    """
    Pair each detected corner with a known world corner: the nearest predicted
    one, if nearer than max_frac of the local spacing and nothing else claims
    it. [(view index, corner index, world index)].
    """
    pairs = []
    for vi, v in enumerate(views):
        pred = project_world(grid_world, v.T_world_gripper @ X, K, D)
        pairs += [(vi, ci, wi) for ci, wi in _pair_view(pred, v.corners_px, max_frac)]
    return pairs


def _pair_view(pred: np.ndarray, corners_px, max_frac: float) -> list:
    """snap() for one view: [(corner index, world index)] from the world corners' predicted pixels."""
    ok = np.isfinite(pred).all(1)
    if ok.sum() < 4:
        return []
    idx = np.nonzero(ok)[0]
    p = pred[idx]
    dd = np.linalg.norm(p[:, None] - p[None], axis=2)
    np.fill_diagonal(dd, np.inf)
    spacing = np.median(dd.min(1))
    claims: dict[int, list] = {}
    for ci, c in enumerate(np.asarray(corners_px, float)):
        d = np.linalg.norm(p - c, axis=1)
        j = int(np.argmin(d))
        if d[j] < max_frac * spacing:
            claims.setdefault(int(idx[j]), []).append((float(d[j]), ci))
    return [(cl[0][1], wi) for wi, cl in claims.items() if len(cl) == 1]


def _residuals(params, views, pairs, grid_world, K, D):
    X = _T(cv2.Rodrigues(params[:3])[0], params[3:])
    out = []
    for vi in sorted({p[0] for p in pairs}):
        sel = [(ci, wi) for v_, ci, wi in pairs if v_ == vi]
        ci, wi = np.array(sel).T
        pred = project_world(grid_world[wi], views[vi].T_world_gripper @ X, K, D)
        out.append((pred - np.asarray(views[vi].corners_px, float)[ci]).ravel())
    return np.concatenate(out) if out else np.zeros(0)


def refine(views: list, X0: np.ndarray, grid_world: np.ndarray, K, D,
           rounds: int = 3, iters: int = 15) -> "tuple[np.ndarray, dict]":
    """
    Stage 2: X adjusted to minimise reprojection of the known grid corners over
    every view (Levenberg-Marquardt, robust to the odd mispaired corner).
    Returns (X, diagnostics).
    """
    K, D = np.asarray(K, float), np.asarray(D, float)
    grid_world = np.asarray(grid_world, float)
    params = np.r_[cv2.Rodrigues(X0[:3, :3])[0].ravel(), X0[:3, 3]]
    pairs = []
    for _ in range(rounds):
        X = _T(cv2.Rodrigues(params[:3])[0], params[3:])
        pairs = snap(views, X, grid_world, K, D)
        if len(pairs) < 12:
            raise ValueError(f"only {len(pairs)} corners could be paired with the grid — is X0 far off?")
        lam = 1e-3
        r = _residuals(params, views, pairs, grid_world, K, D)
        for _ in range(iters):
            # Robust weights (Huber, 1.5 px): a corner the detector invented on
            # the jaw should not pull the answer.
            w = np.ones_like(r)
            big = np.abs(r) > 1.5
            w[big] = np.sqrt(1.5 / np.abs(r[big]))
            J = np.empty((len(r), 6))
            for k in range(6):
                step = 1e-6 if k < 3 else 1e-6
                dp = np.zeros(6)
                dp[k] = step
                J[:, k] = (_residuals(params + dp, views, pairs, grid_world, K, D) - r) / step
            Jw, rw = J * w[:, None], r * w
            A = Jw.T @ Jw
            delta = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), Jw.T @ rw)
            new = _residuals(params + delta, views, pairs, grid_world, K, D)
            if np.sum((new * w) ** 2) < np.sum(rw ** 2):
                params, r, lam = params + delta, new, lam * 0.3
                if np.linalg.norm(delta) < 1e-9:
                    break
            else:
                lam *= 10
    X = _T(cv2.Rodrigues(params[:3])[0], params[3:])
    err = np.linalg.norm(r.reshape(-1, 2), axis=1)
    return X, {"corners": len(pairs), "views": len({p[0] for p in pairs}),
               "reproj_mean_px": float(err.mean()), "reproj_p95_px": float(np.percentile(err, 95))}


def _robust_lm(fun, p0: np.ndarray, steps: np.ndarray, iters: int = 30, huber_px: float = 1.5) -> np.ndarray:
    """Levenberg-Marquardt on fun(p) -> residuals (px), Huber-weighted, forward differences of `steps`."""
    def weights(r):
        w = np.ones_like(r)
        big = np.abs(r) > huber_px
        w[big] = np.sqrt(huber_px / np.abs(r[big]))
        return w

    p, lam = np.asarray(p0, float).copy(), 1e-3
    r = fun(p)
    with np.errstate(all="ignore"):     # macOS Accelerate flags spurious ones in matmul
        for _ in range(iters):
            w = weights(r)
            J = np.empty((len(r), len(p)))
            for k, h in enumerate(steps):
                dp = np.zeros(len(p))
                dp[k] = h
                J[:, k] = (fun(p + dp) - r) / h
            Jw, rw = J * w[:, None], r * w
            A = Jw.T @ Jw
            while lam < 1e8:
                delta = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), Jw.T @ rw)
                new = fun(p + delta)
                if np.sum((new * weights(new)) ** 2) < np.sum(rw ** 2):
                    p, r, lam = p + delta, new, lam * 0.3
                    break
                lam *= 10
            if lam >= 1e8 or np.linalg.norm(delta) < 1e-10:
                break
    return p


def refine_with_joints(views: list, q_deg, fk, X0: np.ndarray, grids, K, D, free=None, shift: bool = True,
                       dq0=None, shift0=None, rounds: int = 2, max_frac: float = 0.35) -> "tuple[np.ndarray, np.ndarray, np.ndarray, dict]":
    """
    Stage 3: X together with a correction to each joint reading and, with
    shift, where the grid is (the other camera's placing of it, a few mm out
    in the arm's frame). fk(q_deg) is the arm's forward kinematics, -> 4x4
    T_world_gripper; the corrections are added to each view's readings q_deg
    (N, joints) before it. grids is one (M, 3) array for every view, or one per
    view (captures made with the board in different places). Start X0 from
    refine(). free says which joints get a correction (default all): leave out
    a joint nothing separates from the mount — the last one before the camera,
    when it only turns the camera about the axis it sits on (a wrist roll): its
    correction and X's rotation trade one for one. dq0 and shift0 start the
    corrections and the shift somewhere other than zero: from an earlier
    calibration's answer, when views close to the surface leave refine() too
    far out to pair the corners right (an SO-101 at 8-12 cm: 18 px from its
    own start, 3.5 from the last calibration's).

    Returns (X, correction per joint in degrees (0 for those not free), grid shift in metres,
    diagnostics). The diagnostics' held_out_* are measured on views the fit did
    not see — fitted on alternate views, measured on the others, both ways —
    which is what says whether the corrections describe the arm or only these
    frames.
    """
    K, D = np.asarray(K, float), np.asarray(D, float)
    q = np.asarray(q_deg, float)
    free_idx = np.arange(q.shape[1]) if free is None else np.nonzero(np.asarray(free, bool))[0]
    n = len(free_idx)
    if isinstance(grids, (list, tuple)):
        grids = [np.asarray(g, float) for g in grids]
    else:
        grids = [np.asarray(grids, float)] * len(views)
    n_shift = 3 if shift else 0
    steps = np.r_[np.full(6, 1e-6), np.full(n, 1e-4), np.full(n_shift, 1e-6)]

    def unpack(p):
        X = _T(cv2.Rodrigues(p[:3])[0], p[3:6])
        return X, p[6:6 + n], (p[6 + n:6 + n + 3] if shift else np.zeros(3))

    def cam(p, vi):
        X, dq, s = unpack(p)
        qv = q[vi].copy()
        qv[free_idx] += dq
        return fk(qv) @ X, s

    def pair(p, idx):
        out = []
        for vi in idx:
            T, s = cam(p, vi)
            out += [(vi, ci, wi) for ci, wi in _pair_view(project_world(grids[vi] + s, T, K, D),
                                                          views[vi].corners_px, max_frac)]
        return out

    def residuals(p, pairs):
        out = []
        for vi in sorted({v for v, _, _ in pairs}):
            sel = np.array([(ci, wi) for v, ci, wi in pairs if v == vi])
            T, s = cam(p, vi)
            pred = project_world(grids[vi][sel[:, 1]] + s, T, K, D)
            out.append((pred - np.asarray(views[vi].corners_px, float)[sel[:, 0]]).ravel())
        r = np.concatenate(out) if out else np.zeros(0)
        return np.nan_to_num(r, nan=50.0)

    def solve(idx, p0):
        p, pairs = p0, pair(p0, idx)
        for _ in range(rounds):
            if len(pairs) < 12:
                raise ValueError(f"only {len(pairs)} corners could be paired with the grid — is X0 far off?")
            p = _robust_lm(lambda pp: residuals(pp, pairs), p, steps)
            pairs = pair(p, idx)
        return p, pairs

    start_dq = np.zeros(q.shape[1]) if dq0 is None else np.asarray(dq0, float)
    p0 = np.r_[cv2.Rodrigues(X0[:3, :3])[0].ravel(), X0[:3, 3], start_dq[free_idx],
               (np.zeros(3) if shift0 is None else np.asarray(shift0, float))[:n_shift]]
    every = list(range(len(views)))
    held = []
    if len(views) >= 8:
        for train, test in ((every[0::2], every[1::2]), (every[1::2], every[0::2])):
            p, _ = solve(train, p0)
            held.append(np.linalg.norm(residuals(p, pair(p, test)).reshape(-1, 2), axis=1))
    p, pairs = solve(every, p0)
    err = np.linalg.norm(residuals(p, pairs).reshape(-1, 2), axis=1)
    X, dq_free, s = unpack(p)
    dq = np.zeros(q.shape[1])
    dq[free_idx] = dq_free
    held_err = np.concatenate(held) if held else np.zeros(0)
    return X, dq, np.asarray(s, float), {
        "corners": len(pairs), "views": len({v for v, _, _ in pairs}),
        "reproj_mean_px": float(err.mean()), "reproj_p95_px": float(np.percentile(err, 95)),
        "held_out_mean_px": float(held_err.mean()) if len(held_err) else None,
        "held_out_p95_px": float(np.percentile(held_err, 95)) if len(held_err) else None,
    }
