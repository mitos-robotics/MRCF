"""The interlock: a trip from anywhere stops the next motion command; it stays tripped until cleared."""

import threading
import time

import pytest

from fake_arm import FakeArm
from mrcf.interlock import Interlock, MotionStopped


def test_a_trip_holds_until_every_source_clears():
    il = Interlock()
    il.check()
    il.trip("hand near the arm", source="hands")
    il.trip("button pressed", source="estop")
    with pytest.raises(MotionStopped) as stop:
        il.check()
    assert "hand near the arm" in stop.value.reason and "button pressed" in stop.value.reason
    il.clear("hands")
    assert il.tripped()
    il.clear("estop")
    il.check()
    assert not il.tripped() and il.reason is None


def test_except_exception_does_not_swallow_a_stop():
    il = Interlock()
    il.trip("stop")
    caught_as_exception = False
    with pytest.raises(MotionStopped):
        try:
            il.check()
        except Exception:
            caught_as_exception = True
    assert not caught_as_exception


def test_a_per_cycle_check_trips_and_stays_tripped_until_removed():
    il = Interlock()
    below = [False]
    il.add_check("floor", lambda: "below the floor" if below[0] else None)
    il.check()
    below[0] = True
    with pytest.raises(MotionStopped):
        il.check()
    below[0] = False
    with pytest.raises(MotionStopped):
        il.check()                          # latched: it does not clear itself
    il.remove_check("floor")
    il.check()


def test_wait_clear_returns_when_cleared_from_another_thread():
    il = Interlock()
    il.trip("hand", source="hands")
    threading.Timer(0.05, il.clear, args=("hands",)).start()
    assert il.wait_clear(timeout=2.0)


def test_a_trip_from_another_thread_stops_a_move_where_it_is():
    arm = FakeArm(cycle_s=0.002)
    target = arm.ee + [0.0, 0.0, -0.5]       # 50 cycles of travel

    def trip_soon():
        while arm.cycles < 5:
            time.sleep(0.001)
        arm.interlock.trip("hand near the arm", source="hands")

    threading.Thread(target=trip_soon, daemon=True).start()
    with pytest.raises(MotionStopped):
        arm.move_ee_to(target, max_iters=100000)
    assert arm.holds == 1
    stopped_at = arm.cycles
    assert stopped_at < 50

    # Nothing moves while it is tripped...
    with pytest.raises(MotionStopped):
        arm.move_ee_to(target)
    assert arm.cycles == stopped_at
    # ...and everything does once it clears.
    arm.interlock.clear("hands")
    assert arm.move_ee_to(target, max_iters=100)


def test_on_trip_runs_once_each_time_it_trips():
    il, calls = Interlock(), []
    il.on_trip(calls.append)
    il.trip("hand", source="hands")
    il.trip("hand again", source="hands")
    il.trip("too low", source="floor")      # already tripped: nothing new to do
    assert calls == ["hand"] and il.trip_count == 1
    il.clear()
    il.trip("hand", source="hands")
    assert calls == ["hand", "hand"] and il.trip_count == 2


def test_a_trip_while_the_arm_is_at_rest_holds_it_only_when_a_command_comes():
    # Holding an arm at rest re-commands it where gravity has sagged it to:
    # it would creep lower with every trip. So a trip between moves waits.
    arm = FakeArm()
    arm.may_be_moving = lambda: False
    arm.interlock.trip("hand near the arm", source="hands")
    assert arm.holds == 0
    with pytest.raises(MotionStopped):
        arm.move_ee_to(arm.ee + [0.0, 0.0, -0.05])
    with pytest.raises(MotionStopped):
        arm.move_ee_to(arm.ee + [0.0, 0.0, -0.05])
    assert arm.holds == 1


# ── Pausing: a hand over the work surface pauses the move, which carries on ────

def test_a_pausing_trip_holds_the_move_and_it_carries_on_when_cleared():
    # The user, 2026-10-22: "I want the sentry to restart automatically after the movement".
    arm = FakeArm(cycle_s=0.002)
    arm.interlock.pause_only("motion")
    target = arm.ee + [0.10, 0.0, 0.0]
    started = threading.Event()

    def hand_in_and_out():
        while len(arm.path) < 3:
            time.sleep(0.001)
        arm.interlock.trip("a hand over the board", source="motion")
        started.set()
        time.sleep(0.15)
        arm.interlock.clear("motion")

    threading.Thread(target=hand_in_and_out, daemon=True).start()
    t0 = time.monotonic()
    assert arm.move_ee_to(target, tol=1e-6)               # no MotionStopped: it waited, then finished
    assert started.is_set() and arm.holds == 1
    assert arm.paused_s >= 0.1
    assert arm.motion_clock() - (t0 - 0.0) < time.monotonic() - t0   # the pause is left out of its clock


def test_any_other_trip_still_stops_and_one_during_a_pause_ends_it():
    arm = FakeArm()
    arm.interlock.pause_only("motion")
    arm.interlock.trip("button pressed", source="estop")
    with pytest.raises(MotionStopped):
        arm.guard()
    arm.interlock.clear()

    arm.interlock.trip("a hand over the board", source="motion")
    threading.Timer(0.05, lambda: arm.interlock.trip("camera gone", source="motion:failed")).start()
    with pytest.raises(MotionStopped) as stop:
        arm.guard()
    assert "camera gone" in stop.value.reason


def test_a_pause_that_never_clears_ends_the_command(monkeypatch):
    arm = FakeArm()
    monkeypatch.setattr(type(arm), "PAUSE_MAX_S", 0.1)
    arm.interlock.pause_only("motion")
    arm.interlock.trip("something keeps moving", source="motion")
    with pytest.raises(MotionStopped) as stop:
        arm.guard()
    assert "still there after" in stop.value.reason


def test_only_pausing_sources_pause():
    il = Interlock()
    il.pause_only("motion")
    assert not il.only_pausing()                              # not tripped at all
    assert not il.stopping()
    il.trip("hand", source="motion")
    assert il.only_pausing() and not il.stopping()
    il.trip("floor", source="workspace")
    assert not il.only_pausing() and il.stopping()


def test_a_pause_clearing_while_the_arm_is_held_is_not_a_stop():
    # The hold takes the motor bus; the sentry may clear in that time. The
    # move then carries on — it was a pause when it was refused.
    arm = FakeArm()
    arm.may_be_moving = lambda: False                         # so guard() is what holds it
    arm.hold = lambda: arm.interlock.clear("motion")
    arm.interlock.pause_only("motion")
    arm.interlock.trip("a hand over the board", source="motion")
    arm.guard()                                               # returns: no MotionStopped


def test_stops_are_counted_apart_from_pauses():
    from mrcf.interlock import Interlock
    lock = Interlock()
    lock.pause_only("motion")
    lock.trip("a hand", source="motion")
    lock.clear("motion")
    assert (lock.trip_count, lock.stop_count) == (1, 0)
    lock.trip("a press", source="contact")
    lock.trip("still", source="contact")                 # the same stop again: one
    assert (lock.trip_count, lock.stop_count) == (2, 1)


def test_a_hold_that_failed_as_the_interlock_tripped_is_tried_again():
    """Marked held before hold() ran, one that raised on the sentry's thread was never retried: the move went on."""
    import threading
    import time

    from fake_arm import FakeArm

    arm = FakeArm()
    calls = []

    def hold():
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise RuntimeError("the controller did not answer")

    arm.hold = hold
    arm.interlock.pause_only("motion")
    arm.interlock.trip("a hand", source="motion")             # the on_trip hold raises (said, swallowed)
    assert len(calls) == 1
    threading.Timer(0.35, lambda: arm.interlock.clear("motion")).start()
    arm.guard()                                               # waits the pause out, holding again meanwhile
    assert len(calls) == 2
