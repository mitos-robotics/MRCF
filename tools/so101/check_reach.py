"""
check_reach.py — is the table actually reachable with this arm's calibration?

Pure kinematics: loads the URDF and your LeRobot calibration file and asks,
for a grid of points on the table, whether an IK solution exists that puts the
gripper straight down there WITHOUT exceeding any joint's calibrated range.
Touches no hardware and sends no commands, so it's safe to run any time.

This answers a question that otherwise looks like a perception or planner bug:
if no point on the table is reachable, then pick/place can never work no
matter how good the camera, the homography, or the VLM is. The symptom on real
hardware is an IK loop that keeps demanding an angle outside the calibrated
range, gets clamped, and thrashes the arm without converging.

Two different failure modes it distinguishes:
  - "no IK": the pose is unreachable for this arm's geometry at all, usually
    because the point is too close to the base to get the gripper vertical
    above it. Moving the workpiece further out is the only fix.
  - "needs +N deg": kinematically fine, but your calibration doesn't cover the
    required angle. Re-run lerobot-calibrate and sweep that joint through its
    FULL travel.

For the full picture — every board square, every phase of a pick and a place,
per piece, offline or on the live arm — use deployments/chess/tests/reach_map.py.

Usage:
    python tools/so101/check_reach.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from tools.so101._common import calibration_path, rc as cfg, task_config  # noqa: E402
from mrcf.backends.so101 import JOINT_NAMES, calibrated_degree_limits  # noqa: E402

CAL_PATH = calibration_path()
IK_TOL_M = 0.02


def main() -> None:
    if not CAL_PATH.is_file():
        print(f"No calibration file at {CAL_PATH} — run lerobot-calibrate first.")
        return
    if not Path(cfg.URDF_PATH).is_file():
        print(f"No URDF at {cfg.URDF_PATH} — run fetch_urdf.py first.")
        return

    # The real approach_orientation on the offline arm, not a re-derivation:
    # this used to build "straight down" from FK at the zero pose — which is
    # horizontal, 90 deg wrong (see env.straight_down_rotation) — and to aim
    # the frame origin rather than the grasp point. So it reported a
    # reachability that matched neither the arm nor the solver that drives it.
    import contextlib
    import io
    from tools.so101.visualise_ik import offline_arm

    limits = calibrated_degree_limits(json.loads(CAL_PATH.read_text()))
    with contextlib.redirect_stdout(io.StringIO()):
        env = offline_arm()

    def solve(target: np.ndarray) -> "int | None":
        """
        Smallest gripper tilt from vertical, degrees, that reaches the target
        within the effective joint limits — what the robot will actually
        attempt, rather than a vertical-only idealisation, which wrongly rules
        out a large, usable part of the workspace. None if nothing works.
        """
        chosen = env.approach_orientation(target, tol=IK_TOL_M)
        return None if chosen is None else int(round(abs(chosen[1])))

    print("Calibrated joint ranges:")
    for name in JOINT_NAMES:
        lo, hi = limits[name]
        print(f"  {name:15s} [{lo:7.1f}, {hi:6.1f}]  = {hi - lo:5.1f} deg total")

    z = cfg.TABLE_Z + task_config().GRASP_HEIGHT_M
    ys = (-0.10, -0.05, 0.0, 0.05, 0.10)
    xs = (0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.34)

    print(f"\nReach at grasp height z={z:.3f} "
          f"(TABLE_Z {cfg.TABLE_Z} + GRASP_HEIGHT_M {z - cfg.TABLE_Z:.3f}):\n")
    print(f'{"x (m)":>7} | ' + " ".join(f"y={y:+.2f}".rjust(11) for y in ys))
    print("-" * (10 + 12 * len(ys)))

    reachable_x = []
    for x in xs:
        cells, any_ok = [], False
        for y in ys:
            tilt = solve(np.array([x, y, z]))
            if tilt is None:
                cells.append("unreachable".rjust(11))
            else:
                cells.append(("straight down" if tilt == 0 else f"tilt {tilt} deg").rjust(11))
                any_ok = True
        if any_ok:
            reachable_x.append(x)
        print(f"{x:7.2f} | " + " ".join(cells))

    print("\nCells show the smallest gripper tilt from vertical that reaches the point.")
    print("'straight down' is ideal; a tilt still works but approaches more sideways,")
    print("which is worth knowing if neighbouring pieces could be knocked on the way in.")
    print("'unreachable' means no tried orientation reaches it within calibrated limits.")
    if reachable_x:
        print(f"\nReachable band: x = {min(reachable_x):.2f} to {max(reachable_x):.2f} m from the base.")
        print("Put the board inside that band, and set WORKSPACE_BOUNDS to match.")
    else:
        print("\nNO POINT ON THE TABLE IS REACHABLE. pick/place cannot work until this is "
              "fixed: check the joint ranges above against lerobot-calibrate, and TABLE_Z.")


if __name__ == "__main__":
    main()
