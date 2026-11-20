"""
arm_only.py — the policy library for an arm with no gripper on it: it moves,
hovers over things and goes home as ParallelJaw does, and refuses, before
moving, everything that needs jaws (pick, place, stand up, open, close).

    ArmOnly(GripperSpec(...))      # an arm's make_policies() when its rig has no gripper

The spec is still a ParallelJaw's: move_above takes its hover heights from it.
"""

from __future__ import annotations

from mrcf.policies.base import Outcome
from mrcf.policies.parallel_jaw import ParallelJaw

NO_GRIPPER = "there is no gripper on this arm"


class ArmOnly(ParallelJaw):
    """ParallelJaw's moves without its grasps. See the module docstring."""

    def _refuse(self, what: str) -> Outcome:
        return Outcome(False, f"cannot {what}: {NO_GRIPPER}", refused=True)

    def pick(self, env, target, obstacles=(), vision=None) -> Outcome:
        return self._refuse("pick")

    def place(self, env, at, held=None, grasp=None, obstacles=(), tilt_deg=None, surface_z=None,
              faces_deg=None) -> Outcome:
        return self._refuse("place")

    def stand_up(self, env, target, obstacles=(), vision=None) -> Outcome:
        return self._refuse("stand it up")

    def open_gripper(self, env) -> Outcome:
        return self._refuse("open the gripper")

    def close_gripper(self, env) -> Outcome:
        return self._refuse("close the gripper")

    def holding(self, env) -> bool:
        return False
