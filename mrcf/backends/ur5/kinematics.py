"""
kinematics.py (mrcf/backends/ur5) — the UR5's joints to its flange and back, from
its nominal DH parameters (Universal Robots' published table, the same frames
as the controller's: the base frame getActualTCPPose() reports in, the flange
at a zero TCP).

What each is for here:

  - fk / frames: where the links are now, from getActualQ(), for the motion
    guard (env.arm_shape) — ur_rtde gives the TCP, not the links;
  - ik / nearest: whether a pose is reachable and the joints that reach it,
    without a round trip to the controller (OrientationSearch asks for many),
    and the offline controller's joints (offline.py).

The controller's own kinematics are calibrated per arm, so its answer can
differ from these by a few millimetres; the moves themselves are the
controller's (moveL, moveJ), so that difference never reaches a target.
tools/ur5/check_connection.py prints how far they differ on the real arm.
"""

from __future__ import annotations

import numpy as np

# UR5 (CB3), metres and radians.
D = (0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823)
A = (0.0, -0.425, -0.39225, 0.0, 0.0, 0.0)
ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)
REACH_M = 0.85                       # the flange from the shoulder's axis, as Universal Robots states it
JOINT_LIMIT_RAD = 2 * np.pi          # every joint turns +-360 deg on a UR5

# Where the body sits off the DH line: the upper arm 135.85 mm along the
# shoulder axis from it, the forearm 16.15 mm (135.85 - 119.7), and the wrist
# at D[3]. The motion guard wants the body, not the DH line.
UPPER_ARM_OFFSET_M = 0.13585
FOREARM_OFFSET_M = 0.01615


def dh(i: int, q: float) -> np.ndarray:
    """Joint i's transform (standard DH: rot z q, trans z d, trans x a, rot x alpha)."""
    c, s = np.cos(q), np.sin(q)
    ca, sa = np.cos(ALPHA[i]), np.sin(ALPHA[i])
    return np.array([[c, -s * ca, s * sa, A[i] * c],
                     [s, c * ca, -c * sa, A[i] * s],
                     [0.0, sa, ca, D[i]],
                     [0.0, 0.0, 0.0, 1.0]])


def frames(q) -> list:
    """The base and each joint's frame after it, 7 of them, in the base frame (4x4)."""
    out = [np.eye(4)]
    for i, qi in enumerate(np.asarray(q, float)):
        out.append(out[-1] @ dh(i, qi))
    return out


def fk(q) -> np.ndarray:
    """The flange in the base frame, 4x4, for joints q (radians)."""
    return frames(q)[-1]


def body_points(q) -> list:
    """
    Points along the arm's body from base to flange, each with a radius
    (metres), consecutive points joined: generous, as the motion guard wants.
    """
    f = frames(q)
    o = [t[:3, 3] for t in f]
    z = [t[:3, 2] for t in f]
    return [
        (o[0], 0.075),                                     # the base, on its mount
        (o[1], 0.075),                                     # the shoulder
        (o[1] + UPPER_ARM_OFFSET_M * z[1], 0.065),         # the upper arm, from the shoulder ...
        (o[2] + UPPER_ARM_OFFSET_M * z[2], 0.06),          # ... to the elbow
        (o[2] + FOREARM_OFFSET_M * z[2], 0.06),            # the forearm, from the elbow ...
        (o[3] + FOREARM_OFFSET_M * z[3], 0.05),            # ... to wrist 1
        (o[4], 0.05),                                      # wrist 2
        (o[5], 0.05),                                      # wrist 3
        (o[6], 0.045),                                     # the flange
    ]


def _wrap(q: float) -> float:
    return float((q + np.pi) % (2 * np.pi) - np.pi)


def ik(T, tol_m: float = 1e-6) -> list:
    """
    Every joint solution (up to 8, each joint in [-pi, pi)) that puts the
    flange at T (4x4, base frame), checked against fk. [] when T is out of
    reach. Near a singularity (wrist 2 at 0) the free joint is set to 0.
    """
    T = np.asarray(T, float)
    p06 = T[:3, 3]
    p05 = p06 - D[5] * T[:3, 2]
    r = float(np.hypot(p05[0], p05[1]))
    if r < D[3]:
        return []
    out: list = []
    phi1, phi2 = np.arctan2(p05[1], p05[0]), np.arccos(D[3] / r)
    for s1 in (1.0, -1.0):
        q1 = phi1 + s1 * phi2 + np.pi / 2
        c5 = (p06[0] * np.sin(q1) - p06[1] * np.cos(q1) - D[3]) / D[5]
        if abs(c5) > 1.0 + 1e-9:
            continue
        for s5 in (1.0, -1.0):
            q5 = s5 * float(np.arccos(np.clip(c5, -1.0, 1.0)))
            sin5 = np.sin(q5)
            if abs(sin5) < 1e-9:
                q6 = 0.0
            else:
                x60, y60 = T[0, :3], T[1, :3]      # base x and y seen from the flange: rows of R06
                q6 = np.arctan2((-x60[1] * np.sin(q1) + y60[1] * np.cos(q1)) / sin5,
                                (x60[0] * np.sin(q1) - y60[0] * np.cos(q1)) / sin5)
            # Joints 2 and 3 are a planar two-link arm in frame 1's x-y plane:
            # p14 = Rz(q2) (A1 + A2 cos q3, A2 sin q3).
            t14 = np.linalg.inv(dh(0, q1)) @ T @ np.linalg.inv(dh(4, q5) @ dh(5, q6))
            p14 = t14[:3, 3]
            c3 = (p14[0] ** 2 + p14[1] ** 2 - A[1] ** 2 - A[2] ** 2) / (2 * A[1] * A[2])
            if abs(c3) > 1.0 + 1e-9:
                continue
            for s3 in (1.0, -1.0):
                q3 = s3 * float(np.arccos(np.clip(c3, -1.0, 1.0)))
                q2 = np.arctan2(p14[1], p14[0]) - np.arctan2(A[2] * np.sin(q3), A[1] + A[2] * np.cos(q3))
                t34 = np.linalg.inv(dh(1, q2) @ dh(2, q3)) @ t14
                q4 = np.arctan2(t34[1, 0], t34[0, 0])
                q = np.array([_wrap(v) for v in (q1, q2, q3, q4, q5, q6)])
                if np.linalg.norm(fk(q)[:3, :4] - T[:3, :4]) < tol_m * 10 and \
                        not any(np.allclose(q, s, atol=1e-6) for s in out):
                    out.append(q)
    return out


def nearest(T, q_now, limit_rad: float = JOINT_LIMIT_RAD) -> "np.ndarray | None":
    """
    The solution for T closest to q_now joint by joint, each joint taken the
    way round (+-2 pi) nearest where it is now that stays within +-limit_rad;
    None when T has none. What a joint move to T (moveJ) should go to: it
    swings a joint the long way round only where the short way would pass its
    limit. 2026-11-13: the park pose leaves wrist 3 at 285 deg; the jaws
    turned 145 deg the short way would pass 360, and that solution was dropped
    for one swinging the base 139 deg, which the gripper's box refused, every try.
    """
    q_now = np.asarray(q_now, float)
    best, best_cost = None, np.inf
    for q in ik(T):
        short = q_now + (q - q_now + np.pi) % (2 * np.pi) - np.pi      # each joint the short way round
        other = short - 2 * np.pi * np.sign(short - q_now)            # ... or the other way, past it
        q = np.where(np.abs(short) <= limit_rad, short, other)
        if np.any(np.abs(q) > limit_rad):
            continue
        cost = float(np.sum(np.abs(q - q_now) * (3.0, 3.0, 2.0, 1.0, 1.0, 1.0)))   # moving the base is costly
        if cost < best_cost:
            best, best_cost = q, cost
    return best
