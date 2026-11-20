"""The motion guard: a hand moving in view freezes the arm at once; the arm's own motion does not."""

import threading
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs OpenCV: pip install -e '.[measure]'")

from fake_arm import FakeArm  # noqa: E402
from mrcf.interlock import MotionStopped  # noqa: E402
from mrcf.motion_guard import MotionGuard, _fold_radius, arm_mask, surface_xy  # noqa: E402

W, H = 320, 180
SKIN = (220, 170, 140)


class SceneArm(FakeArm):
    """FakeArm seen from 0.6 m straight above, over a blocky table, with hands to put in view."""

    def __init__(self, **kw):
        super().__init__(**kw)
        rng = np.random.default_rng(0)
        blocks = rng.integers(60, 200, (H // 6 + 1, W // 6 + 1, 3)).astype(np.uint8)
        self.table = cv2.resize(blocks, (W + 6, H + 6), interpolation=cv2.INTER_NEAREST)[:H, :W]
        self.hands: list = []          # (x, y, radius_m) on the table
        self.light = np.zeros(3)       # added to every pixel, per channel: exposure, white balance
        self.hold_threads: list = []
        self.driven = False            # may_be_moving: commanded lately
        self.lag: list = []            # poses the camera has yet to show, oldest first
        T = np.eye(4)
        T[:3, :3] = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T[:3, 3] = [0.2, 0.0, 0.6]
        self.model = {"K": np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]]),
                      "D": np.zeros(5), "T_world_cam": T, "plane_z_m": 0.0, "scale": 1.0}

    def hold(self):
        super().hold()
        self.hold_threads.append(threading.current_thread().name)

    def may_be_moving(self):
        return self.driven

    def camera_model(self, camera, width, height):
        return dict(self.model)

    def body(self):
        """Where the arm really is: what is drawn."""
        ee, base = self.ee.copy(), np.array([0.0, 0.0, 0.05])
        return [(base, 0.04), ((base + ee) / 2 + [0.0, 0.0, 0.1], 0.03), (ee, 0.03)]

    def arm_shape(self):
        return self.body()

    def _px(self, p):
        T_cw = np.linalg.inv(self.model["T_world_cam"])
        pc = T_cw[:3, :3] @ np.asarray(p, float) + T_cw[:3, 3]
        return (int(round(300 * pc[0] / pc[2] + W / 2)), int(round(300 * pc[1] / pc[2] + H / 2))), pc[2]

    def render_image(self, camera, width=W, height=H):
        img = self.table.copy()
        shape = self.body()
        if self.lag:                   # a camera behind the joints shows an older pose
            self.lag.append(self.ee.copy())
            ee = self.lag.pop(0)
            base = np.array([0.0, 0.0, 0.05])
            shape = [(base, 0.04), ((base + ee) / 2 + [0.0, 0.0, 0.1], 0.03), (ee, 0.03)]
        for (a, ra), (b, _) in zip(shape[:-1], shape[1:]):
            (pa, da), (pb, _) = self._px(a), self._px(b)
            cv2.line(img, pa, pb, (30, 30, 30), 2 * int(300 * ra / da))
        for x, y, r in self.hands:
            p, d = self._px([x, y, 0.0])
            cv2.circle(img, p, int(300 * r / d), SKIN, -1)
        return np.clip(img.astype(int) + self.light, 0, 255).astype(np.uint8)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def guard_on(arm, **kw):
    guard = MotionGuard(arm, "overhead", width=W, height=H, verbose=False, **kw)
    guard._now = clock = Clock()
    return guard, clock


def look(guard, arm, clock, dt=1 / 15):
    clock.t += dt
    guard._look(arm.render_image("overhead"), arm.camera_model("overhead", W, H))


HAND_XY = (0.1, -0.15)       # well clear of the arm's path below


def sweep(arm, guard, clock, targets=([0.32, 0.08, 0.15], [0.2, -0.05, 0.12], [0.2, 0.0, 0.15])):
    for target in targets:
        while np.linalg.norm(arm.ee - target) > 1e-6:
            arm.ee = arm.ee + np.clip(np.asarray(target) - arm.ee, -0.01, 0.01)
            look(guard, arm, clock)
            if arm.interlock.tripped():
                return


def test_the_arm_moving_by_itself_does_not_stop_it():
    arm = SceneArm()
    guard, clock = guard_on(arm)
    sweep(arm, guard, clock)
    assert guard.trips == 0 and not arm.interlock.tripped()


def test_the_same_motion_stops_a_guard_told_the_arm_is_somewhere_else():
    # The control for the test above: the arm's motion is plenty to stop it,
    # and only the arm's outline keeps it from doing so.
    arm = SceneArm()
    parked = arm.body()
    arm.arm_shape = lambda: [(p + [0.0, 0.3, 0.0], r) for p, r in parked]
    guard, clock = guard_on(arm)
    sweep(arm, guard, clock)
    assert arm.interlock.tripped()


def test_a_hand_coming_into_view_stops_the_arm_and_clears_once_it_has_gone():
    arm = SceneArm()
    guard, clock = guard_on(arm, quiet_s=1.0)
    for _ in range(5):
        look(guard, arm, clock)
    arm.hands.append((*HAND_XY, 0.04))
    look(guard, arm, clock)
    assert arm.interlock.tripped() and "moving" in arm.interlock.reason
    assert abs(guard.last["regions"][0]["xy_m"][0] - HAND_XY[0]) < 0.03
    with pytest.raises(MotionStopped):
        arm.move_ee_to(arm.ee + [0.05, 0.0, 0.0])
    arm.hands.clear()                                  # it goes away...
    look(guard, arm, clock)
    for _ in range(10):                                # ...nothing moves for under a second
        look(guard, arm, clock, dt=0.09)
    assert arm.interlock.tripped()
    for _ in range(3):                                 # ...and then for longer
        look(guard, arm, clock, dt=0.1)
    assert not arm.interlock.tripped()
    assert arm.move_ee_to(arm.ee + [0.05, 0.0, 0.0])


def test_a_hand_on_the_arm_while_it_stands_still_is_seen():
    arm = SceneArm()
    guard, clock = guard_on(arm)
    for _ in range(5):
        look(guard, arm, clock)
    arm.hands.append((*arm.ee[:2], 0.04))              # right on the gripper
    look(guard, arm, clock)
    assert arm.interlock.tripped()


def test_small_changes_and_a_change_of_light_do_not_stop_it():
    arm = SceneArm()
    guard, clock = guard_on(arm)
    for _ in range(5):
        look(guard, arm, clock)
    arm.hands.append((*HAND_XY, 0.012))                # ~4.5 cm²: a chess piece, not a hand
    arm.light = np.array([25.0, 25.0, 25.0])           # and the room gets brighter...
    for _ in range(5):
        look(guard, arm, clock)
    arm.light = np.array([35.0, 20.0, 0.0])            # ...and the camera's white balance shifts
    for _ in range(5):
        look(guard, arm, clock)
    assert not arm.interlock.tripped()


def test_a_hand_mid_move_freezes_the_arm_from_the_guards_own_thread():
    arm = SceneArm(cycle_s=0.02)
    arm.driven = True                                  # it is being commanded: a trip holds it at once
    guard = MotionGuard(arm, "overhead", width=W, height=H, hz=50, compare_s=0.04,
                        arm_window_s=0.15, quiet_s=0.2, resume=False, verbose=False).start()
    try:
        threading.Timer(0.3, lambda: arm.hands.append((*HAND_XY, 0.04))).start()
        with pytest.raises(MotionStopped, match="moving"):
            arm.move_ee_to(arm.ee + [0.0, 0.0, -0.5], max_iters=100000)
        assert guard.trips == 1
        # Held once, at the trip — by the guard's thread, not at the arm's next command.
        assert arm.hold_threads == ["watcher-motion"]
        arm.hands.clear()
        end = time.monotonic() + 2.0
        while arm.interlock.tripped() and time.monotonic() < end:
            time.sleep(0.01)
        assert not arm.interlock.tripped()
    finally:
        guard.stop()


def test_the_arms_outline_covers_its_sweep_between_two_readings():
    arm = SceneArm()
    a = (0.0, *[np.array([p for p, _ in arm.arm_shape()]), np.array([r for _, r in arm.arm_shape()])])
    arm.ee = arm.ee + [0.15, 0.0, 0.0]                 # far, between one reading and the next
    b = (0.1, *[np.array([p for p, _ in arm.arm_shape()]), np.array([r for _, r in arm.arm_shape()])])
    mask = arm_mask([a, b], arm.model, (H, W), margin_m=0.0)
    mid, _ = arm._px(arm.ee - [0.075, 0.0, 0.0])
    assert mask[mid[1], mid[0]]
    assert not mask[arm._px([*HAND_XY, 0.0])[0][::-1]]


def test_it_will_not_start_without_a_camera_model_or_the_arms_shape():
    arm = FakeArm()                                    # neither
    with pytest.raises(RuntimeError, match="camera model"):
        MotionGuard(arm, "overhead").start()
    scene = SceneArm()
    scene.arm_shape = lambda: None
    with pytest.raises(RuntimeError, match="arm_shape"):
        MotionGuard(scene, "overhead").start()


def test_a_guard_that_cannot_see_stops_the_arm():
    arm = SceneArm()

    def broken(*_a, **_k):
        raise TimeoutError("latest frame is too old")

    arm.render_image = broken
    guard = MotionGuard(arm, "overhead", width=W, height=H, hz=100, warmup_s=0.05, verbose=False).start()
    try:
        end = time.monotonic() + 2.0
        while not arm.interlock.tripped() and time.monotonic() < end:
            time.sleep(0.005)
        assert "cannot watch" in arm.interlock.reason
    finally:
        guard.stop()
    assert not arm.interlock.tripped()


def test_a_black_picture_stops_the_arm_until_there_is_one_again():
    """2026-11-12, UR5: the webcam dropped, its number opened the MacBook's own camera (lid shut), and
    the guard watched a black picture — no change, so nothing would ever have stopped the arm."""
    arm = SceneArm()
    seeing = arm.render_image
    black = {"on": True}
    arm.render_image = lambda *a, **k: np.zeros((H, W, 3), np.uint8) if black["on"] else seeing(*a, **k)
    guard = MotionGuard(arm, "overhead", width=W, height=H, hz=100, warmup_s=0.05, verbose=False).start()
    try:
        end = time.monotonic() + 2.0
        while not arm.interlock.tripped() and time.monotonic() < end:
            time.sleep(0.005)
        assert "cannot watch" in arm.interlock.reason and "black" in arm.interlock.reason
        black["on"] = False
        end = time.monotonic() + 2.0
        while arm.interlock.tripped() and time.monotonic() < end:
            time.sleep(0.005)
        assert not arm.interlock.tripped()
    finally:
        guard.stop()


def test_a_zone_watches_only_its_part_of_the_surface():
    arm = SceneArm()
    beside = [(-0.15, 0.20), (0.45, 0.20), (0.45, -0.10), (-0.15, -0.10)]   # leaves HAND_XY out
    guard, clock = guard_on(arm, zone=beside)
    for _ in range(5):
        look(guard, arm, clock)
    arm.hands.append((*HAND_XY, 0.04))                 # feet shuffling beside the table
    look(guard, arm, clock)
    assert not arm.interlock.tripped()
    arm.hands.append((0.25, 0.12, 0.04))               # a hand over the work area
    look(guard, arm, clock)
    assert arm.interlock.tripped()


def test_pixels_past_a_wide_lens_fold_are_on_no_surface():
    # The SO-101's overhead lens: k1 = -0.34 folds back at ~0.98, inside the
    # picture's corners (0.86 distorted ~ beyond what the model can reach).
    D = np.array([-0.3449, 0.0, 0.0, 0.0, 0.0])
    assert 0.95 < _fold_radius(D) < 1.0 and _fold_radius(np.zeros(5)) == float("inf")
    arm = SceneArm()
    model = {**arm.model, "K": np.array([[858.78, 0, 640.0], [0, 858.78, 360.0], [0, 0, 1]]), "D": D}
    xy = surface_xy(model, np.array([[640.0, 360.0], [900.0, 500.0], [0.0, 0.0], [1280.0, 360.0]]))
    assert np.allclose(xy[0], [0.2, 0.0], atol=1e-6)
    # Where it can say, it is right: that point projects back onto its pixel.
    back, _ = cv2.projectPoints(np.array([[*xy[1], 0.0]]), *[cv2.Rodrigues(np.linalg.inv(model["T_world_cam"])[:3, :3])[0],
                                np.linalg.inv(model["T_world_cam"])[:3, 3]], model["K"], D)
    assert np.allclose(back.ravel(), [900.0, 500.0], atol=0.05)
    assert np.isnan(xy[2:]).all()                      # the corner and the edge's middle: no answer


def test_it_waits_for_the_camera_before_watching():
    arm, calls = SceneArm(), [0]
    render = arm.render_image

    def warming_up(*a, **k):
        calls[0] += 1
        if calls[0] < 5:
            raise TimeoutError("latest frame is too old")
        return render(*a, **k)

    arm.render_image = warming_up
    guard = MotionGuard(arm, "overhead", width=W, height=H, hz=100, verbose=False).start()
    try:
        time.sleep(0.1)
        assert guard.runs > 0 and guard.errors == 0 and not arm.interlock.tripped()
    finally:
        guard.stop()


def test_a_camera_behind_the_joints_does_not_see_the_arm_arriving_as_a_hand():
    # The joints have stopped; the picture, 0.4 s behind, still shows the arm
    # moving in. Driven, or still within arm_window_s of moving, it is the arm.
    def run(**kw):
        arm = SceneArm()
        guard, clock = guard_on(arm, **kw)
        arm.lag = [arm.ee.copy()] * 6                  # 6 frames at 15 Hz: 0.4 s
        for _ in range(12):
            arm.ee = arm.ee + [0.01, 0.005, 0.0]
            look(guard, arm, clock)
        for _ in range(12):                            # stopped; the camera catches up
            look(guard, arm, clock)
        return arm.interlock.tripped()

    assert not run()
    assert run(arm_window_s=0.3)                       # the control: too short a memory of the arm


def test_while_the_arm_is_driven_its_outline_is_ignored_even_if_the_joints_read_still():
    arm = SceneArm()
    guard, clock = guard_on(arm)
    still = arm.body()
    arm.arm_shape = lambda: still                      # the joints say it has not moved...
    arm.driven = True                                  # ...but it is being commanded
    for _ in range(5):
        look(guard, arm, clock)
    arm.ee = arm.ee + [0.0, 0.0, -0.03]                # and wobbles, on its gears, in its outline
    look(guard, arm, clock)
    assert not arm.interlock.tripped()


def test_an_idle_arm_twitching_on_its_servos_is_still_the_arm():
    # Holding a pose, the SO-101's servos twitch it a millimetre or two; its tag
    # and wires make that a big change in the picture. Any change in the joint
    # readings puts its outline back.
    arm = SceneArm()
    guard, clock = guard_on(arm)
    for _ in range(20):
        look(guard, arm, clock)
    assert guard.last["arm"] is None                   # idle: nothing ignored, a hand on it would be seen
    arm.ee = arm.ee + [0.0005, 0.0, 0.0]               # half a millimetre
    look(guard, arm, clock)
    assert guard.last["arm"] is not None
    for _ in range(20):                                # a second later it is idle again
        look(guard, arm, clock)
    assert guard.last["arm"] is None


def test_a_hand_mid_move_pauses_the_arm_and_it_carries_on_once_nothing_moves():
    # The user, 2026-10-22: "I want the sentry to restart automatically after the movement".
    arm = SceneArm(cycle_s=0.02)
    arm.driven = True
    guard = MotionGuard(arm, "overhead", width=W, height=H, hz=50, compare_s=0.04,
                        arm_window_s=0.15, quiet_s=0.2, verbose=False).start()
    done: list = []
    try:
        threading.Timer(0.3, lambda: arm.hands.append((*HAND_XY, 0.04))).start()
        mover = threading.Thread(target=lambda: done.append(arm.move_ee_to(arm.ee + [0.0, 0.0, -0.5],
                                                                          max_iters=100000)), daemon=True)
        mover.start()
        end = time.monotonic() + 3.0
        while not arm.interlock.tripped() and time.monotonic() < end:
            time.sleep(0.01)
        assert arm.interlock.tripped() and not done      # paused mid-move, not abandoned
        mover.join(5.0)                                   # the hand stops moving: it carries on by itself
        assert done == [True] and guard.trips == 1 and arm.holds == 1
        assert arm.paused_s > 0.1
    finally:
        guard.stop()


def test_it_remembers_everywhere_it_saw_a_hand_not_only_where_it_tripped():
    """A hand trips the guard at the zone's edge and goes on to the object: the controller asks where it went."""
    arm = SceneArm()
    guard, clock = guard_on(arm, quiet_s=1.0)
    for _ in range(5):
        look(guard, arm, clock)
    start = clock.t
    arm.hands.append((*HAND_XY, 0.04))
    look(guard, arm, clock)                            # trips here
    arm.hands[0] = (HAND_XY[0] + 0.08, HAND_XY[1], 0.04)
    look(guard, arm, clock)                            # and reaches on, still tripped
    seen = guard.moved_since(start)
    assert any(abs(x - HAND_XY[0]) < 0.03 for x, _, _ in seen)
    assert any(abs(x - HAND_XY[0] - 0.08) < 0.03 for x, _, _ in seen)
    assert all(r > 0 for _, _, r in seen)
    assert guard.moved_since(clock.t + 1.0) == []


def test_a_hand_that_came_and_went_is_forgotten_once_it_clears():
    # UR5 tabletop, 2026-11-13: a hand passed by the block a place was going to,
    # touching nothing, and the arm went back still holding its block.
    arm = SceneArm()
    guard, clock = guard_on(arm, quiet_s=1.0)
    for _ in range(5):
        look(guard, arm, clock)
    start = clock.t
    arm.hands.append((*HAND_XY, 0.04))
    look(guard, arm, clock)
    arm.hands[0] = (HAND_XY[0] + 0.05, HAND_XY[1], 0.04)
    look(guard, arm, clock)
    assert guard.moved_since(start)                     # while it lasts: there
    arm.hands.clear()                                   # gone, the table as it was
    for _ in range(15):
        look(guard, arm, clock, dt=0.1)
    assert not arm.interlock.tripped()
    assert guard.moved_since(start) == []               # came and went: nothing to look again for


def test_a_hand_that_moved_something_is_remembered_after_it_clears():
    arm = SceneArm()
    guard, clock = guard_on(arm, quiet_s=1.0)
    for _ in range(5):
        look(guard, arm, clock)
    start = clock.t
    arm.hands.append((*HAND_XY, 0.04))
    look(guard, arm, clock)
    arm.hands.clear()
    x, y = arm._px([*HAND_XY, 0.0])[0]
    arm.table[y - 8:y + 8, x - 8:x + 8] = (20, 200, 20)    # where it was, something is now (a block put down)
    for _ in range(15):
        look(guard, arm, clock, dt=0.1)
    assert not arm.interlock.tripped()
    seen = guard.moved_since(start)
    assert seen and any(abs(sx - HAND_XY[0]) < 0.03 for sx, _, _ in seen)


def test_a_hand_where_the_arm_hid_the_table_is_not_taken_to_have_come_and_gone():
    # UR5, 2026-11-13: the user took the block just stacked while the arm was still
    # over it; its spot, left out as the arm's, counted as unchanged, and the next
    # block went on where it had been.
    arm = SceneArm()
    arm.ee = np.array([*HAND_XY, 0.15])                 # the arm over the spot
    guard, clock = guard_on(arm, quiet_s=1.0)
    for _ in range(5):
        look(guard, arm, clock)
    start = clock.t
    arm.hands.append((*HAND_XY, 0.04))                  # a hand there, under it
    look(guard, arm, clock)
    assert arm.interlock.tripped()
    arm.hands.clear()
    arm.ee = np.array([0.32, 0.08, 0.15])               # the arm goes on, the spot in view again
    for _ in range(15):
        look(guard, arm, clock, dt=0.1)
    assert not arm.interlock.tripped()
    assert guard.moved_since(start)                     # it cannot say it came and went: kept


def test_it_says_it_cannot_tell_for_a_time_before_it_watched_or_too_long_ago():
    arm = SceneArm()
    guard, clock = guard_on(arm, quiet_s=1.0)
    assert guard.moved_since(clock.t) is None                 # not looked yet
    look(guard, arm, clock)
    assert guard.moved_since(clock.t - 5.0) is None           # before its first look
    assert guard.moved_since(clock.t) == []
    clock.t += guard.KEEP_MOVED_S + 1.0
    assert guard.moved_since(clock.t - guard.KEEP_MOVED_S - 0.5) is None      # forgotten


class _Glitchy(SceneArm):
    """A camera that now and then sends one frame at half its exposure (2026-11-13: USB 2 shared with a RealSense)."""

    def __init__(self):
        super().__init__()
        self.dim = 0

    def render_image(self, camera, width=W, height=H):
        img = super().render_image(camera, width, height)
        if self.dim:
            self.dim -= 1
            return (img * 0.45).astype(np.uint8)
        return img


def test_one_frame_far_darker_than_those_round_it_is_the_camera_not_something_moving():
    arm = _Glitchy()
    guard, clock = guard_on(arm)
    for _ in range(5):
        look(guard, arm, clock)
    arm.dim = 1                                        # one frame
    for _ in range(4):
        look(guard, arm, clock)
    assert not arm.interlock.tripped() and guard.glitches == 1
    arm.hands.append((*HAND_XY, 0.04))                 # and a hand after it is still seen at once
    look(guard, arm, clock)
    assert arm.interlock.tripped()


def test_a_change_of_the_whole_picture_that_lasts_is_compared_one_frame_later():
    arm = _Glitchy()
    guard, clock = guard_on(arm)
    for _ in range(5):
        look(guard, arm, clock)
    arm.dim = 10                                       # it stays dark: the scene, not one bad frame
    look(guard, arm, clock)
    assert len(guard._frames) and guard._held is not None          # held back, once
    look(guard, arm, clock)
    assert guard._held is None and guard.glitches == 0             # then taken, with the one held


def test_movement_beside_the_gripper_each_time_it_carries_on_stops_the_move(monkeypatch):
    """2026-11-19: a plug caught the tub's wall; paused, carried on, paused ... and the tub was pushed off."""
    import mrcf.motion_guard as mg

    arm = SceneArm()
    arm.interlock.pause_only("motion")
    arm.driven = True

    def push(guard, clock, at):
        """Something moves at `at`, a second after the guard carried on; then nothing for a while."""
        guard._resumed_at = clock.t - 1.0
        guard._trip({"xy_m": at, "area_cm2": 30.0})
        guard._tripped = False
        arm.interlock.clear("motion")

    for on, near, stops in ((False, True, False), (True, False, False), (True, True, True)):
        monkeypatch.setattr(mg, "STOP_PUSHING", on)
        arm.interlock.clear("motion:pushing")
        guard, clock = guard_on(arm)
        beside = (arm.ee[0] + 0.05, arm.ee[1]) if near else (arm.ee[0] + 0.4, arm.ee[1])
        push(guard, clock, beside)
        assert not arm.interlock.stopping()                       # once could be a hand: a pause
        push(guard, clock, beside)
        assert arm.interlock.stopping() is stops, (on, near)
    with pytest.raises(MotionStopped):
        arm.move_ee_to(arm.ee + [0.05, 0.0, 0.0])                 # stopped, not paused: the move is given up
