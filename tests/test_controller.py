"""The controller loop every deployment shares: plans run step by step, missing names looked for, looks again after a trip."""

import numpy as np

from mrcf import controller
from mrcf import vlm
from mrcf.action_registry import ACTIONS
from mrcf.vlm import SceneObject, SceneState, register_action

_RAN = []


@register_action("_test_step_ok", description="test", always=True)
def _step_ok(env, scene_state, **_):
    _RAN.append("ok")
    return True


@register_action("_test_step_fails", description="test", always=True)
def _step_fails(env, scene_state, **_):
    _RAN.append("fails")
    return False


class _Env:
    contact_force_limits: dict = {}

    def __init__(self):
        self.homes = 0
        self.interlock = _Interlock()

    def on_next_move(self, announce):
        pass

    def is_viewer_running(self):
        return True

    def go_home(self):
        self.homes += 1


def test_a_plan_stops_at_its_first_failed_step():
    _RAN.clear()
    failed = controller.execute_plan(
        [{"action": "_test_step_ok"}, {"action": "_test_step_fails"}, {"action": "_test_step_ok"}],
        _Env(), SceneState())
    assert failed is True
    assert _RAN == ["ok", "fails"]


def test_a_plan_that_works_reports_no_failure():
    _RAN.clear()
    assert controller.execute_plan([{"action": "_test_step_ok"}], _Env(), SceneState()) is False


def test_a_name_not_in_view_is_looked_for_with_the_deployments_own_scan_then_its_look_around():
    calls = []
    cube = SceneObject("red_cube", "a red cube", position=np.array([0.2, 0.0, 0.0]))

    def scan(env, state):
        calls.append("scan")
        return True

    def look_around(env, state, name):
        calls.append(("look_around", name))
        state.objects.append(cube)
        return True

    deployment = controller.Deployment(scan=scan, observe_pose=lambda env: calls.append("park"),
                                       look_around=look_around)
    state = SceneState()
    controller.ensure_visible([{"action": "pick", "object_name": "red_cube"}], state, _Env(), deployment)
    assert calls == ["park", "scan", ("look_around", "red_cube")]
    assert state.find("red_cube") is cube


def test_a_command_that_moved_the_arm_leaves_it_there_unless_the_deployment_wants_it_home(monkeypatch):
    monkeypatch.setattr(vlm, "perceive", lambda env: "a table")
    monkeypatch.setattr(vlm, "plan", lambda *a, **k: ([{"action": "_test_step_ok"}], ""))
    for home_after, expected in ((False, []), (True, ["home"])):
        _RAN.clear()
        homes, history, env = [], [], _Env()
        deployment = controller.Deployment(home=lambda env: homes.append("home"), task_planner=False,
                                           home_after_command=home_after)
        controller.handle_command("do the thing", env, SceneState(), history, deployment)
        assert _RAN == ["ok"] and homes == expected and env.homes == 0
        assert history == [{"command": "do the thing", "outcome": "succeeded"}]


def test_the_same_words_with_the_same_objects_run_the_plan_made_for_them_before(monkeypatch):
    """User 2026-11-19: "its the same objects in the same context - so it shouldn't be wasting time"."""
    planned, kept = [], {}
    monkeypatch.setattr(vlm, "perceive", lambda env: "a table")
    monkeypatch.setattr(vlm, "plan", lambda *a, **k: planned.append(1) or ([{"action": "_test_step_ok"}], ""))
    deployment = controller.Deployment(task_planner=False,
                                       remembered_plan=lambda words, state: kept.get(words),
                                       plan_made=lambda words, actions: kept.__setitem__(words, actions))
    _RAN.clear()
    controller.handle_command("put them away", _Env(), SceneState(), [], deployment)
    assert planned == [1] and kept == {"put them away": [{"action": "_test_step_ok"}]} and _RAN == ["ok"]
    controller.handle_command("put them away", _Env(), SceneState(), [], deployment)
    assert planned == [1] and _RAN == ["ok", "ok"]                      # run again, not planned again
    controller.handle_command("put them back", _Env(), SceneState(), [], deployment)
    assert planned == [1, 1]                                            # other words: planned


def test_a_deployment_that_can_plan_the_words_itself_skips_the_planner_and_others_do_not(monkeypatch):
    """2026-11-19: "put these four in the tub" planned in 54-240 s; the planner stays for everything else."""
    planned = []
    monkeypatch.setattr(vlm, "perceive", lambda env: "a table")
    monkeypatch.setattr(vlm, "plan", lambda *a, **k: planned.append(1) or ([{"action": "_test_step_ok"}], ""))
    quick = lambda words, env, state: [{"action": "_test_step_ok"}] if words.startswith("put") else None  # noqa: E731
    _RAN.clear()
    controller.handle_command("put it away", _Env(), SceneState(), [],
                              controller.Deployment(task_planner=False, quick_plan=quick))
    assert planned == [] and _RAN == ["ok"]                                 # from the words: no planner
    controller.handle_command("stack the blocks", _Env(), SceneState(), [],
                              controller.Deployment(task_planner=False, quick_plan=quick))
    assert planned == [1]                                                   # anything else: planned
    controller.handle_command("put it away", _Env(), SceneState(), [], controller.Deployment(task_planner=False))
    assert planned == [1, 1]                                                # no hook: planned, as before


class _Interlock:
    """An interlock whose every trip was a stop."""
    trip_count = 0
    trips_by_source: dict = {}

    @property
    def stop_count(self):
        return self.trip_count


def test_the_scene_is_looked_at_again_only_once_something_tripped_the_interlock_since():
    # 2026-10-30: nothing but the arm moves anything between commands unless
    # the motion guard sees a hand; looking again every time costs a scan.
    looks, env = [], _Env()
    env.interlock = _Interlock()
    deployment = controller.Deployment(scan=lambda env, state: looks.append("scan") or True,
                                       observe_pose=lambda env: looks.append("park"))
    state = SceneState()
    assert not controller.look_again_if_disturbed(env, state, deployment)   # never looked: the start's job
    deployment.look(env, state)
    assert state.looked_at_trip == 0 and looks == ["park", "scan"]
    controller.execute_plan([{"action": "_test_step_ok"}], env, state, deployment)
    assert looks == ["park", "scan"]                                         # nothing tripped
    env.interlock.trip_count = 1                                             # a hand over the board
    controller.execute_plan([{"action": "_test_step_ok"}], env, state, deployment)
    assert looks == ["park", "scan"] * 2 and state.looked_at_trip == 1
    controller.execute_plan([{"action": "_test_step_ok"}], env, state, deployment)
    assert looks == ["park", "scan"] * 2


def test_a_piece_in_the_gripper_is_not_carried_off_to_look_again_after_a_pause():
    # 2026-11-03: a pause during the f1 bishop's pick had the arm lift the
    # bishop away to the park pose and back, to look, before putting it down.
    import numpy as np
    from mrcf.vlm import SceneObject
    looks, env = [], _Env()
    env.interlock = _Interlock()
    deployment = controller.Deployment(scan=lambda env, state: looks.append("scan") or True,
                                       observe_pose=lambda env: looks.append("park"))
    bishop = SceneObject("bishop", "a bishop", position=np.zeros(3))
    state = SceneState(objects=[bishop])
    deployment.look(env, state)
    env.interlock.trip_count = 1                                             # paused mid-pick
    bishop.position = None                                                   # and now held
    assert not controller.look_again_if_disturbed(env, state, deployment)
    assert looks == ["park", "scan"]
    bishop.position = np.zeros(3)                                            # put down
    assert controller.look_again_if_disturbed(env, state, deployment)
    assert looks == ["park", "scan"] * 2


def test_a_scan_the_plan_asked_for_counts_as_a_look(monkeypatch):
    monkeypatch.setitem(ACTIONS, "observe", dict(ACTIONS["_test_step_ok"]))
    looks, env, state = [], _Env(), SceneState()
    env.interlock = _Interlock()
    env.interlock.trip_count = 1
    deployment = controller.Deployment(scan=lambda env, s: looks.append("scan") or True)
    controller.execute_plan([{"action": "observe"}], env, state, deployment)
    assert state.looked_at_trip == 1
    controller.execute_plan([{"action": "_test_step_ok"}], env, state, deployment)
    assert looks == []


def test_a_debugging_command_runs_one_action_without_any_model(monkeypatch):
    _RAN.clear()
    monkeypatch.setattr(vlm, "plan", lambda *a, **k: (_ for _ in ()).throw(AssertionError("planned")))
    controller.handle_command("/_test_step_ok", _Env(), SceneState(), [], controller.Deployment())
    assert _RAN == ["ok"]


@register_action("_test_step_refused", description="test", always=True)
def _step_refused(env, scene_state, **_):
    from mrcf.skill_executor import report
    _RAN.append("refused")
    report("_test_step_refused", False, "the spot is taken by red_bottle", refused=True)
    return False


def _plans(monkeypatch, *plans):
    """vlm.plan answering with each plan in turn (the last one after that); the failure contexts it was given."""
    monkeypatch.setattr(vlm, "perceive", lambda env: "a table")
    contexts: list = []

    def plan(command, scene, state, history=None, failure_context=None):
        contexts.append(failure_context)
        return [{"action": a} for a in plans[min(len(contexts), len(plans)) - 1]], ""

    monkeypatch.setattr(vlm, "plan", plan)
    _RAN.clear()
    return contexts


def test_a_refused_step_asked_for_again_first_thing_is_not_run(monkeypatch):
    # Refused before moving, and nothing done since: it would be refused again
    # (before: eight times over).
    contexts = _plans(monkeypatch, ["_test_step_refused"])
    history: list = []
    deployment = controller.Deployment(scan=lambda env, state: True, task_planner=False)
    controller.handle_command("put it on the shelf", _Env(), SceneState(), history, deployment)
    assert _RAN == ["refused"]
    assert contexts[0] is None and "'_test_step_refused' was refused without touching anything: the spot is taken" in contexts[1]
    assert history[-1]["outcome"] == "failed"


def test_a_refused_step_gets_one_more_go_after_something_first(monkeypatch):
    # Home first (the SO-101's reach depends on where the arm starts), or what
    # was in the way moved: once, so a spot that is simply taken costs one try.
    contexts = _plans(monkeypatch, ["_test_step_refused"],
                      ["_test_step_ok", "_test_step_refused"],
                      ["_test_step_ok", "_test_step_ok", "_test_step_refused"])
    deployment = controller.Deployment(scan=lambda env, state: True, task_planner=False)
    controller.handle_command("put it on the shelf", _Env(), SceneState(), [], deployment)
    assert _RAN == ["refused", "ok", "refused"]                   # the third plan asked a third time: none of it ran
    assert "will not run again" not in contexts[1] and "will not run again" in contexts[2]


@register_action("_test_step_no_room", description="test", always=True)
def _step_no_room(env, scene_state, **_):
    from mrcf.skill_executor import report
    _RAN.append("no room")
    report("_test_step_no_room", False, "jaws need 52 mm open here but red_block_m_obj7 leaves 0 mm", refused=True)
    return False


def test_going_home_does_not_make_room(monkeypatch):
    # UR5 tabletop, 2026-11-13: refused for room beside a block's shadow, the
    # planner went home and asked for the same place again.
    _plans(monkeypatch, ["_test_step_no_room"], ["home", "_test_step_no_room"])
    deployment = controller.Deployment(scan=lambda env, state: True, task_planner=False)
    controller.handle_command("put it on the red block", _Env(), SceneState(), [], deployment)
    assert _RAN == ["no room"]                                    # the home-then-again plan did not run
    assert controller._asks_again([{"action": "_test_step_ok"}, {"action": "_test_step_no_room"}],
                                  (("_test_step_no_room", ()), "x leaves 0 mm"), set()) == ""   # moving something might


def test_the_same_failing_plan_is_tried_twice_not_a_third_time(monkeypatch):
    # A failure (not a refusal) may not repeat — a pick that missed can hit on
    # a fresh look — so it gets one more go, and no more.
    monkeypatch.setattr(vlm, "perceive", lambda env: "a table")
    monkeypatch.setattr(vlm, "plan", lambda *a, **k: ([{"action": "_test_step_fails"}], ""))
    _RAN.clear()
    deployment = controller.Deployment(scan=lambda env, state: True, task_planner=False)
    controller.handle_command("do the thing", _Env(), SceneState(), [], deployment)
    assert _RAN == ["fails", "fails"]


class _Sentried:
    """An interlock that pauses for the sentry, and stops for anything else."""
    trip_count = 0
    stop_count = 0
    trips_by_source: dict = {}


def _sentried_scene():
    import numpy as np
    from mrcf.vlm import SceneObject
    looks, env = [], _Env()
    env.interlock, env.moved = _Sentried(), []
    env.motion_seen_since = lambda t: None if env.moved is None else list(env.moved)
    env.sentry_sources = lambda: {"motion"}           # with motion_seen_since: capabilities.Sentry
    deployment = controller.Deployment(scan=lambda env, state: looks.append("scan") or True,
                                       observe_pose=lambda env: looks.append("park"))
    state = SceneState(objects=[SceneObject("red_block_u", "a block", position=np.array([0.57, 0.38, 0.0]))])
    deployment.look(env, state)
    return looks, env, deployment, state


def test_a_pause_for_movement_far_from_everything_does_not_send_the_arm_off_to_look():
    """2026-11-12, the UR5: 3 s of movement at the table's far edge cost a 40 s look between two places."""
    looks, env, deployment, state = _sentried_scene()
    env.interlock.trip_count, env.moved = 1, [(0.56, 0.78, 0.045)]            # 40 cm from the block
    assert not controller.look_again_if_disturbed(env, state, deployment)
    assert looks == ["park", "scan"] and state.looked_at_trip == 1


def test_movement_near_an_object_is_looked_at_again():
    looks, env, deployment, state = _sentried_scene()
    env.interlock.trip_count, env.moved = 1, [(0.56, 0.78, 0.045), (0.60, 0.45, 0.05)]   # a hand reaching it
    assert controller.look_again_if_disturbed(env, state, deployment)
    assert looks == ["park", "scan"] * 2


def test_a_stop_or_a_pause_with_nothing_seen_is_looked_at_again():
    looks, env, deployment, state = _sentried_scene()
    env.interlock.trip_count, env.interlock.stop_count, env.moved = 1, 1, [(0.56, 0.78, 0.045)]   # a press, say
    assert controller.look_again_if_disturbed(env, state, deployment)
    env.interlock.trip_count, env.moved = 2, None                             # paused, and it cannot say where
    assert controller.look_again_if_disturbed(env, state, deployment)
    assert looks == ["park", "scan"] * 3


def test_a_pause_for_a_hand_that_came_and_went_does_not_look_again(capsys):
    # 2026-11-13, the UR5: every pause sent the arm off to look (or the plan to
    # be made again), a minute each. The sentry now forgets what left no change.
    looks, env, deployment, state = _sentried_scene()
    env.interlock.trip_count, env.moved = 1, []
    assert not controller.look_again_if_disturbed(env, state, deployment)
    assert looks == ["park", "scan"] and "came and went" in capsys.readouterr().out


def test_an_error_or_ctrl_c_in_a_command_ends_the_command_not_the_session(monkeypatch, capsys):
    """A camera that stalled mid-command raised out of the REPL: the session ended, and sent the arm home."""
    env, holds, said = _Env(), [], iter(["first", "second", "quit"])
    from mrcf.interlock import Interlock
    env.interlock = Interlock()
    env.hold = lambda: holds.append(1)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(said))
    calls = []

    def handle(user_input, *a, **k):
        calls.append(user_input)
        raise (RuntimeError("Camera 'overhead' has sent nothing for 3 s") if user_input == "first"
               else KeyboardInterrupt())

    monkeypatch.setattr(controller, "handle_command", handle)
    controller.repl(env, SceneState(), controller.Deployment())
    assert calls == ["first", "second"] and len(holds) == 2
    out = capsys.readouterr().out
    assert "stopped on an error" in out and "Ctrl-C" in out and "Goodbye" in out


def test_a_camera_that_went_blind_since_the_look_is_looked_at_again_however_far_the_movement():
    """The SO-101's camera reopening pauses the arm (a pausing source): blind meanwhile, nothing says what moved."""
    from mrcf.interlock import Interlock

    looks, env, deployment, state = _sentried_scene()
    env.interlock = Interlock()
    env.interlock.pause_only("motion")
    env.interlock.pause_only("camera:overhead")
    env.sentry_sources = lambda: {"motion"}
    deployment.look(env, state)
    env.interlock.trip("a hand at the edge", source="motion")
    env.interlock.clear("motion")
    env.moved = [(0.56, 0.78, 0.045)]                                      # far off
    assert not controller.look_again_if_disturbed(env, state, deployment)  # only that: no look
    env.interlock.trip("the camera reopening", source="camera:overhead")
    env.interlock.clear("camera:overhead")
    assert controller.look_again_if_disturbed(env, state, deployment)      # blind for a while: look


def test_keys_typed_while_a_command_ran_are_dropped_before_the_next_prompt(monkeypatch):
    """An Enter pressed while the arm moved came back as an empty command: "You: You:" (2026-11-13)."""
    from mrcf.interlock import Interlock

    env, events, said = _Env(), [], iter(["stack them", "quit"])
    env.interlock = Interlock()
    monkeypatch.setattr(controller, "_drop_typed_ahead", lambda: events.append("drop"))
    monkeypatch.setattr("builtins.input", lambda prompt="": events.append("prompt") or next(said))
    monkeypatch.setattr(controller, "handle_command", lambda *a, **k: events.append("command"))
    controller.repl(env, SceneState(), controller.Deployment())
    assert events == ["drop", "prompt", "command", "drop", "prompt"]


def test_dropping_typed_ahead_input_is_harmless_without_a_terminal(monkeypatch):
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("left in the pipe\n"))
    controller._drop_typed_ahead()                       # a pipe: nothing to flush, nothing raised
    assert sys.stdin.read() == "left in the pipe\n"


# ── Saying when the arm moves (for demos: it thinks for a minute first) ───────

def test_the_next_move_is_announced_once_as_the_arm_is_first_commanded():
    from fake_arm import FakeArm

    arm, said = FakeArm([]), []
    arm.on_next_move(lambda: said.append("now"))
    assert said == []                                  # set, not yet moved
    arm.guard()
    arm.guard()
    assert said == ["now"]                             # once, at the first command
    arm.on_next_move(lambda: said.append("again"))
    arm.on_next_move(None)                             # forgotten before any move
    arm.guard()
    assert said == ["now"]


def test_a_plan_says_moving_now_as_its_first_step_moves_and_names_each_step(capsys):
    from fake_arm import FakeArm
    from mrcf.action_registry import ACTIONS, register_action

    arm = FakeArm([])

    @register_action("_demo_move", description="test", parameters={"object_name": "what"})
    def _demo_move(env, scene_state, object_name="", **_):
        print("before it moves")
        env.guard()                                    # what every motion command does first
        return True

    @register_action("_demo_still", description="test")
    def _demo_still(env, scene_state, **_):
        return True

    from mrcf.scene_state import SceneObject

    scene = SceneState(objects=[SceneObject(n, n, position=np.array([0.3, y, 0.0]))
                                for n, y in (("green_block_7", 0.1), ("blue_block_d", -0.1))])
    try:
        steps = [{"action": "_demo_move", "object_name": "green_block_7"},
                 {"action": "_demo_move", "object_name": "blue_block_d"}]
        assert controller.execute_plan(steps, arm, scene) is False
        out = capsys.readouterr().out
        assert out.count("MOVING NOW") == 1 and "\a" in out
        assert out.index("before it moves") < out.index("MOVING NOW")      # at the move, not before
        assert "▶ Step 1/2: _demo_move green_block_7" in out and "▶ Step 2/2: _demo_move blue_block_d" in out
        assert controller.execute_plan([{"action": "_demo_still"}], arm, scene) is False
        assert "MOVING NOW" not in capsys.readouterr().out                  # nothing moved: nothing said
        arm.guard()
        assert "MOVING NOW" not in capsys.readouterr().out                  # and nothing left to say later
    finally:
        ACTIONS.pop("_demo_move", None)
        ACTIONS.pop("_demo_still", None)


# ── A deployment's own start, from the first look ─────────────────────────────

def test_a_deployments_start_runs_after_a_first_look_and_never_without_one(monkeypatch, capsys):
    """Chess reads the first look to carry on its saved game or begin one (2026-11-16): never on a look not had."""
    from types import SimpleNamespace

    from mrcf import flight_recorder
    from mrcf.interlock import MotionStopped

    monkeypatch.setattr(flight_recorder, "RECORDER", SimpleNamespace(
        start=lambda **k: None, stop=lambda: None, watch_cameras=lambda *a, **k: None, watch_sensors=lambda *a: None,
        failure=lambda *a, **k: None, timed=lambda *a, **k: None, event=lambda *a, **k: None))
    env = SimpleNamespace(go_home=lambda: None, list_cameras=lambda: [])

    def session(looked, start=None):
        events = []

        class Dep(controller.Deployment):
            def look(self, env, scene_state):
                events.append("look")
                if looked == "stopped":
                    raise MotionStopped("a hand over the board")
                return looked

        monkeypatch.setattr(controller, "repl", lambda *a: events.append("prompt"))
        controller.run(env, SceneState(), Dep(on_start=start or (lambda e, s: events.append("start"))))
        return events

    assert session(True) == ["look", "start", "prompt"]
    assert session(False) == ["look", "prompt"]
    assert session("stopped") == ["look", "prompt"]

    def broken(e, s):
        raise ValueError("no saved game file")

    assert session(True, broken) == ["look", "prompt"]               # said, and the session goes on
    assert "start failed — ValueError: no saved game file" in capsys.readouterr().out


# ── Pick-and-put units (Deployment.defer_failed) ──────────────────────────────

def _units_scene(monkeypatch, fails: dict):
    from mrcf import skill_executor

    calls = []

    def dispatch(action, env, state):
        name = controller._object_of(action)
        calls.append((action["action"], name))
        if action["action"] == "pick" and fails.get(name, 0) > 0:
            fails[name] -= 1
            skill_executor.report("pick", False, "closed on nothing")
            return False
        return True
    monkeypatch.setattr(controller, "dispatch", dispatch)
    monkeypatch.setattr(controller.safety, "validate_plan", lambda actions, state, **k: (actions, []))
    objs = [SceneObject(n, n, position=np.array([0.2 + 0.05 * i, 0.0, 0.0])) for i, n in enumerate("abc")]
    for i, o in enumerate(objs):
        o.track_id = f"obj-{i}"
    state = SceneState(objects=objs)

    def rescan(env, st):                           # the next scan names b otherwise: found again by its track
        st.objects[1].name = "b_now"
        return True
    deployment = controller.Deployment(scan=rescan, defer_failed=True)
    plan = []
    for n in "abc":
        plan += [{"action": "pick", "object_name": n}, {"action": "place_in", "container": "tub"}]
    return calls, state, deployment, plan


def test_a_failed_pick_is_set_aside_the_rest_go_on_and_it_is_tried_again_after_a_look(monkeypatch, capsys):
    calls, state, deployment, plan = _units_scene(monkeypatch, {"b": 1})
    assert controller.execute_plan(plan, _Env(), state, deployment) is False
    picks = [n for a, n in calls if a == "pick"]
    assert picks == ["a", "b", "c", "b_now"]                  # c not held up by b; b again, by its new name
    assert controller.LAST_PLAN_SETTLED
    assert "Done: a, c, b_now." in capsys.readouterr().out



def test_one_pushed_and_tracked_afresh_is_found_again_by_its_name_near_where_it_was(monkeypatch, capsys):
    """2026-11-19 16:23: the plug, pushed 4 cm by its failed pick, named again under a new track, never retried."""
    def pushed(env, st):                           # b moved 4 cm: a new track, its name the same; a far one named b
        st.objects[1].track_id = "obj-9"
        st.objects[1].position = st.objects[1].position + [0.0, 0.04, 0.0]
        st.objects.append(SceneObject("b_obj2", "b", position=np.array([0.60, -0.20, 0.0])))
        st.objects[-1].track_id = "obj-10"
        return True

    for near in (None, 0.10):
        calls, state, deployment, plan = _units_scene(monkeypatch, {"b": 1})
        deployment.scan, deployment.refind_near_m = pushed, near
        controller.execute_plan(plan, _Env(), state, deployment)
        picks = [n for a, n in calls if a == "pick"]
        out = capsys.readouterr().out
        if near is None:
            assert picks == ["a", "b", "c"] and "not on the table where the scan can find it now" in out
        else:
            assert picks == ["a", "b", "c", "b"], picks                   # the near one, not b_obj2 40 cm off
            assert "b: no longer tracked, but b is 4 cm from where it was" in out
            assert "Done: a, c, b." in out


def test_one_found_again_by_its_name_retires_its_old_track_not_kept_clear_of_beside_itself(monkeypatch):
    """2026-11-20 09:15: the plug found again 5 cm off, refused for "white_plug (missed...; last seen there)"."""
    calls, state, deployment, plan = _units_scene(monkeypatch, {"b": 1})
    old = state.objects[1].track_id
    retired = []
    state.memory = type("M", (), {"superseded": lambda self, tid, by, why: retired.append((tid, by))})()

    def pushed(env, st):
        st.objects[1].track_id = "obj-9"
        st.objects[1].position = st.objects[1].position + [0.0, 0.04, 0.0]
        return True
    deployment.scan, deployment.refind_near_m = pushed, 0.10
    controller.execute_plan(plan, _Env(), state, deployment)
    assert retired == [(old, "obj-9")]


def test_one_that_fails_again_is_said_not_planned_again(monkeypatch, capsys):
    calls, state, deployment, plan = _units_scene(monkeypatch, {"b": 1, "b_now": 1})
    planned = []
    monkeypatch.setattr(vlm, "perceive", lambda env: "a table")
    monkeypatch.setattr(vlm, "plan", lambda *a, **k: planned.append(1) or (plan, ""))
    deployment.task_planner = False
    controller.handle_command("put them all in the tub", _Env(), state, [], deployment)
    assert len(planned) == 1                                   # no re-plan after the units ran out
    out = capsys.readouterr().out
    assert "not done: b_now (closed on nothing)" in out and "Scheduling recovery re-plan" not in out


def test_a_plan_that_is_not_units_runs_as_before(monkeypatch):
    calls, state, deployment, plan = _units_scene(monkeypatch, {"a": 1})
    assert controller._units(plan[:2]) is None and controller._units(plan[:3]) is None
    assert controller.execute_plan(plan[:2], _Env(), state, deployment) is True
    assert calls == [("pick", "a")]


# ── A deployment's own tasks, and no planner (Deployment.task_plan, .planner) ──

def _no_model_calls(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the model was asked")
    for name in ("perceive", "perceive_from_frames", "plan", "decompose_task", "render_all_cameras"):
        monkeypatch.setattr(vlm, name, refuse)


def test_a_deployments_own_task_runs_as_planned_for_it_and_nothing_else_is_planned(monkeypatch, capsys):
    """User 2026-11-19: "skip the planner - so it just does the task we pre plan for it"."""
    calls, state, deployment, plan = _units_scene(monkeypatch, {})
    _no_model_calls(monkeypatch)
    deployment.task_plan = lambda words, env, st: plan if words == "tidy up" else None
    deployment.task_names, deployment.planner = ("tidy up",), False
    controller.handle_command("tidy up", _Env(), state, [], deployment)
    assert [n for a, n in calls if a == "pick"] == ["a", "b", "c"]
    out = capsys.readouterr().out
    assert "Its own task, as planned for it: 6 step(s), no planning." in out
    assert "Planned for these words before" not in out              # 2026-11-20: said of a task, wrongly
    calls.clear()
    controller.handle_command("dance a jig", _Env(), state, [], deployment)
    assert calls == []
    assert 'is none of its tasks ("tidy up"). Nothing moved.' in capsys.readouterr().out


def test_without_a_planner_a_failed_task_is_said_not_planned_again(monkeypatch, capsys):
    calls, state, deployment, plan = _units_scene(monkeypatch, {"a": 5})
    _no_model_calls(monkeypatch)
    steps = [{"action": "look_for", "object_name": "a"}] + plan[:2]           # not units: stops at a failure
    deployment.task_plan = lambda words, env, st: steps
    deployment.planner = False
    controller.handle_command("go", _Env(), state, [], deployment)
    out = capsys.readouterr().out
    assert "not planned again" in out and "Scheduling recovery re-plan" not in out


def test_with_a_planner_the_task_is_still_its_own_and_other_words_are_planned(monkeypatch):
    calls, state, deployment, plan = _units_scene(monkeypatch, {})
    planned = []
    monkeypatch.setattr(vlm, "perceive", lambda env: "a table")
    monkeypatch.setattr(vlm, "plan", lambda *a, **k: planned.append(1) or (plan[:2], ""))
    deployment.task_planner = False
    deployment.task_plan = lambda words, env, st: plan if words == "tidy up" else None
    controller.handle_command("tidy up", _Env(), state, [], deployment)
    assert planned == []
    controller.handle_command("put a in the tub", _Env(), state, [], deployment)
    assert planned == [1]
