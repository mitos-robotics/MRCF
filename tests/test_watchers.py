"""Watchers: perception on its own thread while the arm moves, able to stop it, and fail-safe."""

import threading
import time

import pytest

from fake_arm import FakeArm
from mrcf.interlock import MotionStopped
from mrcf.watchers import Watcher


def wait_for(condition, timeout_s=2.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.002)
    return False


def test_a_watcher_gets_frames_on_its_own_thread():
    arm, seen = FakeArm(), []
    w = Watcher(arm, "overhead", lambda frame, model: seen.append((threading.current_thread(), frame.shape)),
                hz=200).start()
    assert wait_for(lambda: len(seen) >= 3)
    w.stop()
    assert not w.running
    assert seen[0][0] is not threading.current_thread() and seen[0][1] == (480, 640, 3)


def test_a_hand_seen_mid_move_stops_the_arm():
    arm, hand = FakeArm(cycle_s=0.002), [False]

    def hands(frame, model):
        if hand[0]:
            arm.interlock.trip("hand near the arm", source="hands")
        else:
            arm.interlock.clear("hands")

    w = Watcher(arm, "overhead", hands, hz=200).start()
    threading.Timer(0.03, lambda: hand.__setitem__(0, True)).start()
    try:
        with pytest.raises(MotionStopped):
            arm.move_ee_to(arm.ee + [0.0, 0.0, -0.5], max_iters=100000)
        assert arm.cycles < 50
        hand[0] = False                       # the hand goes away; the watcher clears it
        assert wait_for(lambda: not arm.interlock.tripped())
    finally:
        w.stop()


def test_a_failing_safety_watcher_stops_the_arm_until_it_recovers():
    arm, broken = FakeArm(), [True]

    def look(frame, model):
        if broken[0]:
            raise RuntimeError("camera unplugged")

    w = Watcher(arm, "overhead", look, hz=200, fail_safe=True, name="hands").start()
    try:
        assert wait_for(arm.interlock.tripped)
        with pytest.raises(MotionStopped):
            arm.move_ee_to(arm.ee + [0.0, 0.0, -0.05])
        broken[0] = False
        assert wait_for(lambda: not arm.interlock.tripped())
    finally:
        w.stop()


def test_stopping_a_safety_watcher_is_not_a_failure():
    arm = FakeArm()

    def look(frame, model):
        raise RuntimeError("no camera")

    w = Watcher(arm, "overhead", look, hz=200, fail_safe=True).start()
    assert wait_for(arm.interlock.tripped)
    w.stop()
    assert not arm.interlock.tripped()
