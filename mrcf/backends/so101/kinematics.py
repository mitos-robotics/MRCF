"""
mrcf/backends/so101/kinematics.py — the SO-101's kinematic model, apart from the
arm: its joints and their limits (the calibration's, the URDF's, and the
tighter of the two the solver is held to), each joint's zero correction, the
straight-down orientation, and where the jaws close in the gripper's frame.

Functions and data only: nothing here talks to the arm, so offline tools
(tools/so101/visualise_ik.py) use exactly what the live arm uses.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Order matches the SO-101/SO-100 motor bus wiring (lerobot's SOFollower).
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Where the jaws actually close, in gripper_frame_link coordinates (metres).
#
# IK places the FRAME on the target, but the frame origin is not the point the
# fingers pinch — and, the part that matters here, that point MOVES as the jaws
# open. Only one jaw swings on this gripper: in the URDF `gripper` is a revolute
# joint driving moving_jaw_so101_v1_link, while gripper_frame_joint is fixed. So
# the fixed pad stays rigid with the frame and the midpoint between the pads
# slides toward the moving jaw at half the opening rate:
#
#     grasp_x(gap) = _FIXED_JAW_PAD_X - gap / 2
#
# Measured off the URDF's own meshes — locate the two pad faces where they meet
# at the closed joint limit, then carry the moving pad's vertices rigidly through
# the joint range and take the midpoint:
#
#     gap  3.1 mm -> centre  -1.0 mm       gap 31.0 mm -> centre -15.0 mm
#     gap 16.8 mm -> centre  -7.9 mm       gap 45.0 mm -> centre -22.1 mm
#
# which is a straight line of slope -1/2, as the midpoint of a chord must be.
#
# A fixed offset is right at one opening only, and wrong at both ends of a pick
# in opposite directions: with the jaws shut (touching calibration corners) and
# with them open for the descent, where aiming toward the fixed jaw drives that
# pad into the piece instead of down beside it.
_FIXED_JAW_PAD_X = 0.00035
# y and z stay at zero deliberately. They run ALONG the pad faces, where a piece
# can sit anywhere, so the geometry does not pin them down — only x, the pad-face
# normal, is forced. (The meshes put the closed-jaw contact ~4 mm along +z, but
# that is just where this one vertex pair happens to touch; GRASP_HEIGHT_M was
# tuned on the rig with z at zero, so moving it would trade a measured number
# for an arbitrary one.)
#
# The sign convention is the gripper frame's, so this rotates with the wrist —
# it is applied through the chosen approach rotation, never as a fixed offset in
# the base frame.
_TCP_OFFSET = np.array([_FIXED_JAW_PAD_X, 0.0, 0.0])   # jaws CLOSED

def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def calibrated_degree_limits(calibration: dict) -> dict[str, tuple[float, float]]:
    """
    Per-joint (min, max) in degrees from a LeRobot calibration, for JOINT_NAMES.

    These are the physical Min/Max_Position_Limit values the arm was calibrated
    with, which write_calibration() already pushed into the servos' firmware.
    Nothing in lerobot's DEGREES normalisation clamps to them on write (unlike
    its 0-100%/±100% modes), so an unconstrained IK solution can ask for an
    angle the servo will physically refuse — it just sits there stalled against
    its own limit. Clamp before ever sending.

    Accepts either lerobot's MotorCalibration objects or plain dicts parsed
    straight from the calibration JSON, so offline tools (visualise_ik.py) can
    derive exactly the same limits the live arm uses without a serial port.
    """
    MAX_RES = 4095  # sts3215: 4096 ticks/revolution
    limits: dict[str, tuple[float, float]] = {}
    for name in JOINT_NAMES:
        cal = calibration.get(name)
        if cal is None:
            continue
        lo = cal["range_min"] if isinstance(cal, dict) else cal.range_min
        hi = cal["range_max"] if isinstance(cal, dict) else cal.range_max
        mid = (lo + hi) / 2
        limits[name] = ((lo - mid) * 360 / MAX_RES, (hi - mid) * 360 / MAX_RES)
    return limits


def urdf_degree_limits(urdf_path: "str | Path") -> dict[str, tuple[float, float]]:
    """Per-joint (min, max) in degrees from the URDF's own <limit> tags."""
    text = Path(urdf_path).read_text(encoding="utf-8")
    limits: dict[str, tuple[float, float]] = {}
    for name in JOINT_NAMES:
        joint = re.search(rf'<joint name="{name}"[^>]*>(.*?)</joint>', text, re.S)
        if not joint:
            continue
        lim = re.search(r'<limit[^>]*lower="([-\d.eE+]+)"[^>]*upper="([-\d.eE+]+)"', joint.group(1))
        if lim:
            limits[name] = (float(np.degrees(float(lim.group(1)))),
                            float(np.degrees(float(lim.group(2)))))
    return limits


def within_limits(limits: dict, caps: "dict | None") -> dict[str, tuple[float, float]]:
    """limits, each joint narrowed to its range in caps where caps has one."""
    out = dict(limits)
    for name, (lo, hi) in (caps or {}).items():
        if name in out:
            out[name] = (max(out[name][0], float(lo)), min(out[name][1], float(hi)))
    return out


def effective_ik_limits(calibrated: dict, urdf: dict, beyond_urdf=()) -> dict[str, tuple[float, float]]:
    """
    The range an IK solution can ACTUALLY occupy: the tighter of the two —
    except for the joints in beyond_urdf, which get their whole calibrated
    range (robot_config.IK_CALIBRATED_RANGE_JOINTS; the solver is then given
    it too, by apply_solver_limits).

    These differ, and conflating them hides failures. placo enforces the URDF
    during the solve, while SO101Env clamps commands to the calibration before
    sending — so on this arm wrist_flex saturates at the URDF's 95 deg while
    the servo still has 6 deg of calibrated travel left. Judging reachability
    against the calibration alone therefore reports a comfortable margin for a
    solution the solver has already pinned to a stop, which is exactly how an
    unreachable hover gets approved and then stalls short.

    Used ONLY for reachability margin checks. Command clamping stays on the
    calibration (_clamp_joint_degrees), because that is the hardware's real
    limit and the URDF's is just a model.
    """
    out: dict[str, tuple[float, float]] = {}
    for name, (lo, hi) in calibrated.items():
        u = None if name in beyond_urdf else urdf.get(name)
        out[name] = (max(lo, u[0]), min(hi, u[1])) if u else (lo, hi)
    return out


def apply_solver_limits(kinematics, ik_limits: dict) -> None:
    """
    Make the IK SOLVER obey the effective limits, not just the reachability
    check. placo defaults to the URDF's range, which on this arm is wider than
    the calibration for shoulder_lift and elbow_flex — so it would return a
    pose the servo cannot reach, _clamp_joint_degrees would trim it, and the
    next iteration would re-solve to the same impossible angle. The loop then
    cannot converge, and stalls short of the target with a joint pinned at the
    URDF stop.

    Shared with visualise_ik's offline arm, so it previews the solutions the
    live solver returns.
    """
    for name, (lo, hi) in ik_limits.items():
        kinematics.robot.set_joint_limits(name, float(np.deg2rad(lo)), float(np.deg2rad(hi)))
    kinematics.solver.enable_joint_limits(True)


class OffsetKinematics:
    """
    RobotKinematics with per-joint corrections: forward_kinematics(q) is the
    model's at q + offsets, and inverse_kinematics answers in the same raw
    angles the motors report. Measured by tools/so101/calibrate_arm_offsets.py
    (JOINT_OFFSETS_DEG): a few degrees of joint zero error at the shoulder
    and elbow put a board's far corner 28 mm below its near ones.
    """

    def __init__(self, kinematics, offsets_deg: np.ndarray):
        self._kin = kinematics
        self._offsets = np.asarray(offsets_deg, dtype=float)

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        return self._kin.forward_kinematics(np.asarray(joint_pos_deg, float) + self._offsets)

    def inverse_kinematics(self, current_joint_pos: np.ndarray, desired_ee_pose: np.ndarray,
                           **kwargs) -> np.ndarray:
        q = self._kin.inverse_kinematics(np.asarray(current_joint_pos, float) + self._offsets,
                                         desired_ee_pose, **kwargs)
        return np.asarray(q, float) - self._offsets

    def __getattr__(self, name):
        return getattr(self._kin, name)


def straight_down_rotation(kinematics) -> np.ndarray:
    """
    The gripper orientation whose approach axis points at the table.

    "Gripper pointing straight down" is a fixed orientation whenever
    shoulder_lift + elbow_flex + wrist_flex == +90 deg (planar 3-link chain);
    derive it from FK at such a pose rather than hardcoding the matrix.

    NOT at the zero pose: there the sum is 0 and gripper_frame_link's approach
    axis (its local +z) points along base +x — HORIZONTAL, 90 deg off. Taken
    as "down", that asks IK for a gripper pointing sideways at the table,
    which only solves by folding the arm onto its joint stops (shoulder_lift
    pinned at the URDF's -100 deg limit), and makes _APPROACH_TILTS_DEG mean
    "degrees from horizontal" — so with a 60 deg cap no vertical grasp at all.

    Raises if the result is not actually down, so a URDF or joint-order change
    cannot silently bring that back.
    """
    q_down = np.zeros(len(kinematics.joint_names))
    q_down[JOINT_NAMES.index("shoulder_lift")] = 90.0
    r_down = kinematics.forward_kinematics(q_down)[:3, :3].copy()
    approach = r_down[:, 2]
    if not np.allclose(approach, [0.0, 0.0, -1.0], atol=1e-3):
        raise RuntimeError(
            f"gripper_frame_link approach axis is {np.round(approach, 3).tolist()}, "
            "expected [0, 0, -1] — the straight-down reference pose no longer "
            "matches this URDF."
        )
    return r_down
