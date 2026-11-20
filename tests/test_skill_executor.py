"""skill_executor: what every skill run gets — here, backing off after a press."""

import pytest

from mrcf import skill_executor
from mrcf.interlock import Interlock, MotionStopped
from mrcf.policies.base import Outcome

PRESS = "contact force 47.5 N (limit 45.0 N, on the final approach)"


class _Env:
    """An arm that can back off (capabilities.BackOff), and says when it does."""
    contact_force_limits: dict = {}

    def __init__(self, ok=True):
        self.interlock, self.backed, self.ok = Interlock(), [], ok

    def back_off(self, distance_m):
        self.backed.append(distance_m)
        return self.ok

    def get_contact_force(self):
        return 0.0


def _stopped_by(reason, source="safety-monitor:pick"):
    def skill(env, **_):
        env.interlock.trip(reason, source=source)          # as the safety monitor's check does
        raise MotionStopped(reason)
    return skill


def test_a_press_backs_the_arm_off_then_the_stop_goes_on():
    # UR5, 2026-11-13: held where it stopped, it went on pressing until the power was cut.
    env = _Env()
    with pytest.raises(MotionStopped, match="contact force"):
        skill_executor.run_policy(_stopped_by(PRESS), "pick", env)
    assert env.backed == [skill_executor.BACK_OFF_M]


def test_no_back_off_while_anything_else_holds_the_arm_or_for_another_stop(capsys):
    env = _Env()
    env.interlock.trip("something moving in view of the overhead camera", source="motion-guard")   # a hand
    with pytest.raises(MotionStopped):
        skill_executor.run_policy(_stopped_by(PRESS), "pick", env)
    assert env.backed == [] and "Not backing off" in capsys.readouterr().out
    env = _Env()
    with pytest.raises(MotionStopped):
        skill_executor.run_policy(_stopped_by("stopped by the user"), "pick", env)
    assert env.backed == []


def test_a_back_off_that_does_not_get_there_is_said():
    env = _Env(ok=False)
    assert not skill_executor.back_off_after_press(env, MotionStopped(PRESS))
    assert skill_executor.back_off_after_press(_Env(), MotionStopped(PRESS))
    assert not skill_executor.back_off_after_press(object(), MotionStopped(PRESS))      # an arm that cannot


def test_a_skill_that_is_not_stopped_is_left_alone():
    env = _Env()
    assert skill_executor.run_policy(lambda env, **_: Outcome(True, "done"), "pick", env).ok
    assert env.backed == []
