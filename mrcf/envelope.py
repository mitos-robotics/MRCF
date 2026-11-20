"""
envelope.py — where the gripper may be: a box in the arm's frame that the
whole gripper stays inside, checked along a move before it is sent. The arm
itself may pass outside it; what it carries and grips with may not.

The gripper is a stick from the flange to the jaw tips (along the flange's z),
`radius_m` of body round it. The stick keeps that far inside the box's walls
and ceiling, and the gripper's lowest point — the tips when it points down,
the edge of its body when it leans — at least `floor_margin_m` above the
floor (the work surface). The box is convex, so a stick with both ends inside
it is inside all the way along.

A move is refused when anywhere along it the gripper would be further outside
than it is where the move starts: from inside, not outside at all. From
outside (put there by hand), only a move that comes back toward it — never
further out from one point of the path to the next, and nearer at its end —
not one along the outside at the same distance (over the teach pendant, say).

The robot's own safety system can hold its tool point to planes as well (a
UR's Boundaries): that limit and this one should be the same box, this one so
a move is refused before it is sent, that one in case this one is not run.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS_M = 1e-4


@dataclass(frozen=True)
class GripperBox:
    x: tuple                         # (min, max), metres, the arm's frame
    y: tuple
    z: tuple                         # z[0]: the floor (the work surface)
    tip_m: float                     # flange to the jaw tips, along the flange's z
    radius_m: float = 0.05           # the gripper's body round that line, generous
    floor_margin_m: float = 0.003    # the lowest point at least this far above the floor
    # Its body at least this far from the arm's base axis (at base_xy): the box
    # may reach in to the base, the gripper must not — into the base's housing,
    # or folding the arm onto itself. 0: no such limit.
    base_clear_m: float = 0.0
    base_xy: tuple = (0.0, 0.0)

    @classmethod
    def from_config(cls, box, tip_m: float, radius_m: float = 0.05, floor_margin_m: float = 0.003,
                    base_clear_m: float = 0.0, base_xy=(0.0, 0.0)) -> "GripperBox":
        (x0, x1), (y0, y1), (z0, z1) = box
        return cls((float(x0), float(x1)), (float(y0), float(y1)), (float(z0), float(z1)), float(tip_m),
                   float(radius_m), float(floor_margin_m), float(base_clear_m),
                   (float(base_xy[0]), float(base_xy[1])))

    def _parts(self, T_flange) -> "tuple[np.ndarray, np.ndarray, float]":
        T = np.asarray(T_flange, float)
        flange, axis = T[:3, 3], T[:3, 2]
        lean = float(np.sqrt(max(0.0, 1.0 - axis[2] ** 2)))     # 0 pointing straight down or up
        return flange, flange + self.tip_m * axis, lean

    def outside(self, T_flange) -> "tuple[float, str]":
        """How far the gripper is outside (metres, summed over what it crosses; 0 inside), and which limits."""
        flange, tip, lean = self._parts(T_flange)
        r, total = self.radius_m, 0.0
        worst: dict = {}                          # each wall crossed: how far, at the worst of flange and tips
        for name, k, lo, hi in (("x", 0, *self.x), ("y", 1, *self.y)):
            for p in (flange, tip):
                for wall, by in ((f"{name} < {lo:+.2f}", (lo + r) - p[k]), (f"{name} > {hi:+.2f}", p[k] - (hi - r))):
                    if by > EPS_M:
                        total += by
                        worst[wall] = max(worst.get(wall, 0.0), by)
        if self.base_clear_m > 0:
            # The stick's nearest point to the base axis, in plan.
            b, a2 = np.asarray(self.base_xy, float), tip[:2] - flange[:2]
            u = float(np.clip(((b - flange[:2]) @ a2) / (a2 @ a2), 0.0, 1.0)) if a2 @ a2 > 1e-12 else 0.0
            near = float(np.linalg.norm(flange[:2] + u * a2 - b))
            by = (self.base_clear_m + r) - near
            if by > EPS_M:
                total += by
                worst[f"within {self.base_clear_m + r:.2f} m of the arm's base"] = by
        top = max(flange[2], tip[2]) + r - self.z[1]
        if top > EPS_M:
            total += top
            worst[f"above the ceiling {self.z[1]:.2f} m"] = top
        low = (self.z[0] + self.floor_margin_m) - (min(flange[2], tip[2]) - r * lean)
        if low > EPS_M:
            total += low
            worst["too low over the table"] = low
        return total, "; ".join(f"{wall} by {by * 100:.1f} cm" for wall, by in worst.items())

    def refuses(self, path) -> "str | None":
        """
        Why a move along path (flange poses, 4x4, start first) is refused; None
        when the gripper stays inside, or never gets further out than at the start.
        """
        path = list(path)
        start, _ = self.outside(path[0])
        if start > EPS_M:
            last, why = start, ""
            for T in path[1:]:
                out, what = self.outside(T)
                if out > last + EPS_M:
                    return (f"the gripper is outside its box already ({self.outside(path[0])[1]}), and this move "
                            f"would take it further out ({what}): jog it back inside on the pendant first")
                last = out
            if last >= start - EPS_M and len(path) > 1:
                return (f"the gripper is outside its box already ({self.outside(path[0])[1]}), and this move does "
                        "not bring it back in: jog it back inside on the pendant first")
            return None
        worst, why = start, ""
        for T in path[1:]:
            out, what = self.outside(T)
            if out > worst + EPS_M:
                worst, why = out, what
        return None if worst <= start + EPS_M else f"the gripper would leave its box: {why}"


def linear_path(T0, T1, n: int = 24) -> list:
    """
    Flange poses along a straight move from T0 to T1: the point in a line, the
    orientation turned evenly about one axis (as a UR's moveL does).
    """
    T0, T1 = np.asarray(T0, float), np.asarray(T1, float)
    rvec = cv2.Rodrigues(T0[:3, :3].T @ T1[:3, :3])[0].ravel()
    out = []
    for s in np.linspace(0.0, 1.0, max(2, n)):
        T = np.eye(4)
        T[:3, :3] = T0[:3, :3] @ cv2.Rodrigues(rvec * s)[0]
        T[:3, 3] = T0[:3, 3] + s * (T1[:3, 3] - T0[:3, 3])
        out.append(T)
    return out


def joint_path(q0, q1, fk, n: "int | None" = None) -> list:
    """Flange poses along a joint move from q0 to q1 (each joint evenly, as a UR's moveJ does): fk(q) -> 4x4."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    steps = n if n is not None else int(np.clip(np.degrees(np.max(np.abs(q1 - q0))) / 2.0, 2, 180))
    return [fk(q0 + s * (q1 - q0)) for s in np.linspace(0.0, 1.0, steps + 1)]
