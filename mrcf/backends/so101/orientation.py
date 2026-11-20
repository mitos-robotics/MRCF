"""
mrcf/backends/so101/orientation.py — which way the gripper points to reach a
target: straight down where it can, tilted where it must (near the base, far
out), the jaws turned square to the object; each candidate checked with the
solver offline, before the arm moves.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import numpy as np

from mrcf.backends.so101.follower import Follower
from mrcf.backends.so101.kinematics import JOINT_NAMES, _rot_z

# Gripper tilts from vertical that approach_orientation() will try, in order.
# Straight down is preferred; larger tilts unlock points near the base that a
# vertical wrist simply cannot reach, and the far edge. Up to 80 since
# 2026-11-04 (was 60): the arm picks almost level, and only so reaches a chess
# board's far rank (0.43-0.46 m: 60-80 deg in its model). What a shallow lean
# sweeps the gripper over is the policy's to check (ReachMixin.sweep_clearance).
_APPROACH_TILTS_DEG = tuple(range(0, 81, 5))
_IK_PROBE_ITERS = 120   # offline convergence check, no motion
# How much calibrated range a joint must have left over for approach_orientation
# to call a solution reachable. Accepting a solution that sits exactly ON a limit
# means commanding a pose the servo cannot quite hold: the arm stops short and
# the skill silently grasps next to the object instead of failing honestly.
_JOINT_MARGIN_DEG = 1.0

def _tilts_from(prefer_deg: "float | None" = None) -> list:
    """
    The tilts approach_orientation_for tries, in order: straight down first,
    then leaning further — or, given a preferred lean (a deployment's approach
    rule), nearest to it first. The least lean that reaches is at the edge of
    the joints' range, where the real arm stalls; a rule asks for some margin.
    """
    if prefer_deg is None:
        return list(_APPROACH_TILTS_DEG)
    want = abs(float(prefer_deg))
    return sorted(_APPROACH_TILTS_DEG, key=lambda t: (abs(t - want), t))


class OrientationMixin(Follower):
    """SO101Env's choice of gripper orientation (see env.py)."""

    def _grasp_point(self, q6: np.ndarray, gap_m: "float | None" = None) -> np.ndarray:
        """Where the jaws close, in the base frame, for a joint configuration (with the grasp correction)."""
        t = self._kinematics.forward_kinematics(q6)  # type: ignore[union-attr]  # callers _require_kinematics()
        here = t[:3, 3] + t[:3, :3] @ self._tcp(gap_m)
        return here + self._grasp_correction(t[:3, :3], here)

    def approach_orientation(self, target_pos: np.ndarray,
                             gap_m: "float | None" = None,
                             tol: float = 0.01) -> "tuple[np.ndarray, float] | None":
        """
        Pick a gripper orientation that can actually reach target_pos, preferring
        straight down and tilting only as far as needed.

        Insisting on a perfectly vertical gripper throws away a large part of the
        workspace for no reason. On this arm, points near the base are reachable
        only with the wrist leaning in — demanding vertical declares them
        unreachable when the arm manages them easily by hand. This searches tilts
        in increasing order and returns the first whose IK solution respects every
        joint's calibrated range, so a vertical grasp is still used whenever it
        is available.

        `tol` must be the tolerance the CALLER will hold the move to. A looser
        one approves an orientation whose own IK solution sits short of what the
        move accepts, and the move then stalls having done exactly what was
        approved. A probe looser than the move it predicts is worse than no
        probe.

        `gap_m` is the jaw opening the arm will arrive at, so the probe aims the
        same point the move will (see _tcp); default is the measured opening.

        Pure kinematics — sends nothing. Returns (rotation, tilt_degrees), or
        None if no tried orientation works.
        """
        return self.approach_orientation_for([target_pos], gap_m=gap_m, tols=[tol])

    def approach_orientation_for(self, targets: "list[np.ndarray]",
                                 gap_m: "float | None" = None,
                                 tols: "list[float] | None" = None,
                                 flip: bool = False,
                                 prefer_tilt_deg: "float | None" = None,
                                 margin_deg: "float | None" = None,
                                 turn_deg: float = 0.0,
                                 ) -> "tuple[np.ndarray, float] | None":
        """
        One gripper orientation that reaches EVERY target in order, preferring
        straight down and tilting only as far as the hardest target needs.

        For a vertical sequence over one spot — hover, descend, lift — so the
        wrist holds still while the jaws are near a piece. Choosing per target
        (approach_orientation's own behaviour) lets the tilt change between
        hover and grasp: at rank 6 that is 5-10 deg of wrist rotation during
        the final descent, swinging the jaw tips sideways by up to ~1 cm with a
        piece already between them.

        Each target is probed from the previous one's solution, the way the
        moves will actually chain; the first from the arm's measured pose (see
        approach_orientation for why the seed matters). `tols` is one per
        target, the tolerance each move will be held to. `flip` turns the
        gripper half a turn about its approach axis (see tilted_rotation).
        `prefer_tilt_deg`: try the tilts nearest this one first (a deployment's
        approach rule) instead of straight down first (see _tilts_from).
        `margin_deg`: approve only solutions that leave every joint at least
        this far from its limit at every target (default _JOINT_MARGIN_DEG) —
        the least lean that reaches leaves one joint at its limit, where the
        real arm, sagging and off its zeros by a degree or two, stalls short.
        `turn_deg`: the jaws turned that far about the approach axis (see
        tilted_rotation).

        Pure kinematics — sends nothing. (rotation, tilt_degrees) or None.
        """
        self._require_kinematics()
        targets = [np.asarray(t, dtype=float) for t in targets]
        tols = list(tols) if tols is not None else [0.01] * len(targets)
        if gap_m is None:
            gap_m = self._grasp_gap()
        tcp = self._tcp(gap_m)
        assert self._kinematics is not None     # _require_kinematics() above
        n_joints = len(self._kinematics.joint_names)
        # Seed from the arm's MEASURED pose, the same seed move_ee_to's loop
        # starts from. placo's IK is a local solver, so the seed selects which
        # branch it converges to: probing from zeros could approve an
        # orientation while the loop, starting from where the arm actually is,
        # converged somewhere else entirely and drove into a joint stop. The
        # probe is only worth anything if it predicts the motion that follows.
        seed = np.zeros(n_joints)
        seed[: len(JOINT_NAMES)] = self._q_arm_deg()

        # A preferred lean is a direction too: toward the base or away from it.
        signs = (-1.0, 1.0) if (prefer_tilt_deg or 0.0) < 0 else (1.0, -1.0)
        for tilt in _tilts_from(prefer_tilt_deg):
            for sign in (signs if tilt else (1.0,)):
                rot = self.tilted_rotation(sign * tilt, targets[0], flip=flip, turn_deg=turn_deg)
                q = seed.copy()
                for target, tol in zip(targets, tols):
                    q = self._probe(rot, target, tcp, gap_m, q, tol, margin_deg)
                    if q is None:
                        break
                else:
                    return rot, float(sign * tilt)
        return None

    def _x_along_arm(self) -> bool:
        """
        Whether the jaws open along self._jaw_axis with the gripper frame's x
        pointing along the arm (its y across it), rather than rolled a quarter
        turn from that.

        The jaws open along the gripper axis self._jaw_slide_axis names — this
        gripper's along y, not the x its meshes suggest (robot_config,
        JAW_SLIDE_AXIS) — so on it x along the arm opens them ACROSS the arm.
        """
        axis = np.asarray(self._jaw_slide_axis, float)
        opens_along_x = abs(axis[0]) >= abs(axis[1])
        return (self._jaw_axis == "radial") == opens_along_x

    def tilted_rotation(self, tilt_deg: float, target: "np.ndarray | None" = None,
                        flip: bool = False, turn_deg: float = 0.0) -> np.ndarray:
        """
        The gripper orientation for reaching `target`: straight down with the
        jaws opening along self._jaw_axis, leaned tilt_deg in the arm's own
        vertical plane.

        In the arm's plane because that is the only tilt a 5-joint arm has:
        shoulder_lift, elbow_flex and wrist_flex all pitch about one axis, so the
        approach axis can lean toward or away from the base and nowhere else. A
        lean fixed in the world is only achievable on the line straight out from
        the base; anywhere else the solver meets the position and quietly gives
        up on the orientation, so the tilt the arm actually takes is not the one
        approved.

        The jaw axis decides what that lean does to a piece standing between the
        jaws. Radial jaws open in the same plane as the lean, so the top of a
        tall piece swings toward one jaw — for a king, by more than the slack
        round it on most of the board. Tangential jaws open across it, the lean
        runs along the pad faces, and the piece stays parallel to them. See
        JAW_AXIS in robot_config.

        `flip` turns the gripper half a turn about its approach axis first. The
        jaws still open along the same line, but everything mounted on one side
        of the gripper — the wrist camera — is on the other: the way to keep
        the camera clear of a neighbouring piece. The wrist rolls a full turn,
        so both are always within its range.

        `turn_deg` turns the jaws about the approach axis, counter-clockwise
        seen from above, from where they would open: square across something
        lying on its side, whichever way it lies (mrcf/policies/stand_up.py).
        """
        rot = self._tilted(tilt_deg, target, flip)
        if not turn_deg:
            return rot
        # About the gripper's own z, which runs along its approach: pointing
        # down, a turn counter-clockwise from above is a negative one about it.
        return rot @ _rot_z(np.deg2rad(turn_deg) * (-1.0 if rot[2, 2] < 0 else 1.0))

    def _tilted(self, tilt_deg: float, target, flip: bool) -> np.ndarray:
        """tilted_rotation without a turn."""
        a = np.deg2rad(tilt_deg)
        c, s = np.cos(a), np.sin(a)
        if target is None:
            psi = 0.0
        else:
            v = np.asarray(target, dtype=float)[:2] - self._pan_centre
            psi = float(np.arctan2(v[1], v[0]))
        base = _rot_z(psi) @ self._r_down          # straight down, gripper x along the arm
        if flip:
            base = base @ _rot_z(np.pi)
        if not self._x_along_arm():
            # Rolled a quarter turn: gripper x now points across the arm,
            # which makes it the pitch axis — so the lean is about x.
            base = base @ _rot_z(np.pi / 2)
            return base @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return base @ np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def gripper_tilt_deg(self, rot: "np.ndarray | None" = None) -> float:
        """
        How far the gripper leans from straight down, in tilted_rotation's
        degrees (signed, in the arm's own plane); `rot` defaults to now. Read
        from which way "down" points in the gripper's frame, so it does not
        depend on where the arm is panned to, or on the wrist's half turn.
        """
        rot = self.get_ee_rotation() if rot is None else np.asarray(rot, float)
        down = rot.T @ np.array([0.0, 0.0, -1.0])
        if not self._x_along_arm():
            return float(np.degrees(np.arctan2(down[1], down[2])))
        return float(np.degrees(np.arctan2(-down[0], down[2])))

    def _half_turn_roll(self, rot: np.ndarray, q: np.ndarray) -> "float | None":
        """
        The wrist_roll, half a turn from q's, that `rot` needs when it faces the
        other way round from q's gripper (their jaw axes opposed); None when it
        does not, or when neither half turn is within the roll's limits.

        A goal half a turn away is one IK cannot find by itself: the solver
        starts from the arm's own wrist angle, and from there the error has no
        preferred direction to turn in. It has to start on the right side.
        """
        i = JOINT_NAMES.index("wrist_roll")
        x_now = self._kinematics.forward_kinematics(np.asarray(q, float))[:3, 0]  # type: ignore[union-attr]  # callers _require_kinematics()
        if float(x_now @ np.asarray(rot, float)[:, 0]) >= 0.0:
            return None
        lo, hi = self._ik_limits.get("wrist_roll", (-180.0, 180.0))
        for turned in sorted((q[i] + 180.0, q[i] - 180.0), key=lambda r: abs(r)):
            if lo + _JOINT_MARGIN_DEG <= turned <= hi - _JOINT_MARGIN_DEG:
                return float(turned)
        return None

    def _roll_seed_toward(self, rot: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """seed, with its wrist turned half a turn when `rot` faces the other way round."""
        turned = self._half_turn_roll(rot, seed)
        q = np.asarray(seed, float).copy()
        if turned is not None:
            q[JOINT_NAMES.index("wrist_roll")] = turned
        return q

    def _probe(self, rot: np.ndarray, target: np.ndarray, tcp: np.ndarray, gap_m: float,
               seed: np.ndarray, tol: float, margin_deg: "float | None" = None) -> "np.ndarray | None":
        """IK solution reaching target at rot inside the joint margins (margin_deg, else _JOINT_MARGIN_DEG), or None."""
        margin = _JOINT_MARGIN_DEG if margin_deg is None else max(float(margin_deg), _JOINT_MARGIN_DEG)
        t = np.eye(4)
        t[:3, :3] = rot
        t[:3, 3] = target - self._grasp_correction(rot, target) - rot @ tcp
        q = self._roll_seed_toward(rot, seed)
        for _ in range(_IK_PROBE_ITERS):
            q = self._kinematics.inverse_kinematics(q, t)  # type: ignore[union-attr]  # callers _require_kinematics()
        if float(np.linalg.norm(self._grasp_point(q, gap_m) - target)) > tol:
            return None
        if all(lo + margin <= q[i] <= hi - margin
               for i, name in enumerate(JOINT_NAMES)
               for lo, hi in [self._ik_limits.get(name, (-1e9, 1e9))]):
            return q
        return None
