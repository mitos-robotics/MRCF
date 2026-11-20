"""The flight recorder: one timeline, a rolling record, and incidents that can be read afterwards."""

import json

import numpy as np
import pytest

from mrcf import flight_recorder as fr
from mrcf.object_memory import Detection, ObjectMemory


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    rec = fr.FlightRecorder().start(tmp_path)
    monkeypatch.setattr(fr, "RECORDER", rec)
    yield rec
    rec.stop()


def frame(level=120):
    return np.full((48, 64, 3), level, np.uint8)


def test_nothing_is_written_until_a_session_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "ROOT", tmp_path)
    rec = fr.FlightRecorder()
    rec.event("policy", "say", text="hello")
    with rec.action("pick", object="pawn") as act:
        act.problem("missed")
    assert not list(tmp_path.iterdir())


def test_an_action_that_went_wrong_leaves_an_incident_with_what_led_up_to_it(recorder):
    pytest.importorskip("cv2")
    recorder.frame("overhead", frame())
    with recorder.action("place", object="queen", target="e3") as act:
        recorder.sample("read", {"shoulder_lift": 12.5})
        fr.event("sentry", "pause", image=frame(200), area_cm2=24.0, xy_m=[0.17, -0.22])
        fr.event("policy", "say", text="[Place] Camera at the release: 0.7 mm")
        act.done(ok=True, message="placed object")
        act.problem("not seen where it was put")
    (inc,) = list((recorder.directory / "incidents").iterdir())
    summary = json.loads((inc / "summary.json").read_text(encoding="utf-8"))
    assert summary["why"] == "not seen where it was put" and summary["action"]["name"] == "place"
    kinds = [json.loads(line)["kind"] for line in (inc / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kinds[:4] == ["begin", "pause", "say", "end"]
    assert (inc / "samples.jsonl").read_text(encoding="utf-8").count("shoulder_lift") == 1
    names = {p.name.split("_")[0] for p in (inc / "frames").iterdir()}
    assert names == {"overhead", "event"}                     # the rolling frames, and the sentry's own
    story = (inc / "story.txt").read_text(encoding="utf-8")
    assert "sentry" in story and "xy_m=[0.17, -0.22]" in story
    assert fr.main([str(inc)]) == 0


def test_a_stop_through_an_action_is_recorded_and_passed_on(recorder):
    from mrcf.interlock import MotionStopped
    with pytest.raises(MotionStopped):
        with recorder.action("pick", object="pawn"):
            raise MotionStopped("hand at the arm")
    (inc,) = list((recorder.directory / "incidents").iterdir())
    assert "hand at the arm" in json.loads((inc / "summary.json").read_text(encoding="utf-8"))["why"]


def test_an_action_that_went_well_writes_no_incident(recorder):
    with recorder.action("pick", object="pawn") as act:
        act.done(ok=True)
    assert not list((recorder.directory / "incidents").iterdir())
    kinds = [json.loads(line)["kind"] for line in (recorder.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kinds == ["begin", "end"]


def test_a_piece_the_arm_put_down_and_the_next_scan_does_not_find_is_an_incident(recorder):
    # 2026-10-28: a queen fell out on the way to e3 and rolled off the board;
    # the place said "placed", and nothing tied that to the next scan.
    m = ObjectMemory()
    (queen,), _ = m.observe([Detection(position=(0.25, 0.03), place="d4", foot_mm=27.0)])
    m.picked(queen.id)
    with recorder.action("place", object="queen", target="e3") as act:
        act.done(ok=True)
    m.placed(queen.id, (0.215, -0.013), "e3")
    m.observe([])
    (inc,) = list((recorder.directory / "incidents").iterdir())
    summary = json.loads((inc / "summary.json").read_text(encoding="utf-8"))
    assert "not seen where it was put (e3)" in summary["why"] and summary["action"]["name"] == "place"


def test_recording_never_raises_into_its_caller(recorder):
    class Unprintable:
        def __repr__(self):
            raise RuntimeError("no")
    recorder.event("arm", "move", goal=Unprintable())          # kept or not, but no exception
    recorder.frame("overhead", "not an image")
    recorder.sample("read", {"x": "nan?"})


def test_the_rigs_sensors_are_kept_on_the_rolling_record(recorder):
    import time

    class _Env:
        def list_sensors(self):
            return {"wrench": "force (N)", "pad": "pressure"}

        def read_sensor(self, name):
            return np.array([1.0, 2.0, 3.0]) if name == "wrench" else 0.5

    recorder.watch_sensors(_Env(), hz=100.0)
    time.sleep(0.1)
    kept = [values for _t, stream, values in list(recorder._samples) if stream == "sensors"]
    assert kept and kept[-1] == {"wrench[0]": 1.0, "wrench[1]": 2.0, "wrench[2]": 3.0, "pad": 0.5}


def test_the_joints_are_kept_with_the_sensors(recorder):
    # UR5, 2026-11-13: the gripper's cable wound round the wrist, and nothing
    # kept said how far the wrist had turned.
    import time

    class _Env:
        def list_sensors(self):
            return {}

        def get_joint_angles(self):
            return {"shoulder_pan": 1.0, "wrist_3": 4.5}

    recorder.watch_sensors(_Env(), hz=100.0)
    time.sleep(0.1)
    kept = [values for _t, stream, values in list(recorder._samples) if stream == "sensors"]
    assert kept and kept[-1] == {"joint.shoulder_pan": 1.0, "joint.wrist_3": 4.5}


def test_a_picture_kept_with_the_run_is_written_whole_whatever_happens_after(tmp_path):
    pytest.importorskip("cv2")
    from mrcf.flight_recorder import FlightRecorder

    rec = FlightRecorder()
    assert rec.keep_picture("placed_block_side", np.zeros((720, 1280, 3), np.uint8)) is None    # not running
    rec.start(tmp_path)
    try:
        path = rec.keep_picture("placed_block_side", np.full((720, 1280, 3), 90, np.uint8))
        import cv2
        assert path is not None and path.parent.name == "pictures" and cv2.imread(str(path)).shape == (720, 1280, 3)
    finally:
        rec.stop()


def test_a_model_call_is_kept_whole(tmp_path):
    # UR5 tabletop, 2026-11-13: a plan of "home, then the same refused place", and
    # nothing kept showed what the planner had been given.
    import base64
    import json as _json

    from mrcf.flight_recorder import FlightRecorder
    assert FlightRecorder().keep_exchange("plan", model=None, system=None, prompt="", reply="") is None  # not recording
    rec = FlightRecorder().start(tmp_path)
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 8).decode()
    jpg = base64.b64encode(b"\xff\xd8\xff" + b"\0" * 8).decode()
    path = rec.keep_exchange("plan", model="gemma4:31b", system="you plan", prompt="Known objects: red_block_m",
                             reply='{"actions": []}', images_b64=[png, jpg], seconds=1.234,
                             extra={"few_shot": [{"role": "user", "content": "an example"}]})
    again = rec.keep_exchange("perceive", model="gemma4:31b", system=None, prompt="what is there", reply="a table")
    rec.stop()
    data = _json.loads(path.read_text(encoding="utf-8"))
    assert (data["tag"], data["system"], data["prompt"], data["reply"], data["seconds"]) == \
        ("plan", "you plan", "Known objects: red_block_m", '{"actions": []}', 1.23)
    assert data["few_shot"][0]["content"] == "an example"
    assert [p.rsplit(".", 1)[1] for p in data["pictures"]] == ["png", "jpg"]
    assert (path.parent / data["pictures"][0]).read_bytes()[:4] == b"\x89PNG"
    assert path.name.startswith("001_") and again.name.startswith("002_")      # in the order they were made


# ── The session's record: what it was, what it printed, where the time went ──

def test_a_session_says_what_it_was_and_a_second_start_adds_to_it(tmp_path):
    rec = fr.FlightRecorder().start(tmp_path, session={"arm": {"name": "so101"}, "deployment": {"name": "tidy"}})
    rec.start(session={"cameras_open": ["front", "overhead"]})                 # the controller's, the same run
    rec.stop()
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    info = json.loads((runs[0] / "session.json").read_text())
    assert info["arm"]["name"] == "so101" and info["deployment"]["name"] == "tidy"
    assert info["cameras_open"] == ["front", "overhead"] and "started" in info and "stopped" in info
    again = fr.FlightRecorder()
    again.start(tmp_path)
    again.stop()
    again.start(tmp_path)                                                      # stopped: a new run
    again.stop()


def test_everything_printed_is_kept_with_the_run_and_the_console_given_back(tmp_path, capsys):
    import sys

    before = sys.stdout
    rec = fr.FlightRecorder().start(tmp_path, console=True)
    print("  [Pick] closed on nothing at its waist")
    print("a warning", file=sys.stderr)
    rec.stop()
    assert sys.stdout is before
    log = (rec.directory / "console.log").read_text()
    assert "[Pick] closed on nothing at its waist" in log and "! a warning" in log
    assert "[Pick]" in capsys.readouterr().out                                  # still shown as it was


def test_where_the_time_went_is_counted_by_action_step_and_model_call(tmp_path, capsys):
    rec = fr.FlightRecorder().start(tmp_path)
    with rec.action("pick") as act:
        act.done(ok=False)
    with rec.action("pick"):
        pass
    rec.timed("command", 12.5, failed=False)
    rec.event("vlm", "call", tag="plan", seconds=21.0)
    rows = rec.timings()
    assert rows["pick"]["count"] == 2 and rows["pick"]["failed"] == 1
    assert rows["model:plan"]["total_s"] == 21.0 and rows["command"]["max_s"] == 12.5
    rec.stop()
    assert json.loads((rec.directory / "timings.json").read_text())["model:plan"]["count"] == 1
    assert "Where the time went" in capsys.readouterr().out
    assert fr.main(["--timings", str(rec.directory)]) == 0
    table = capsys.readouterr().out
    assert table.index("model:plan") < table.index("command") < table.index("pick")   # the most time first


def test_a_failure_that_is_no_actions_is_an_incident_with_its_traceback(tmp_path):
    rec = fr.FlightRecorder().start(tmp_path)
    try:
        raise ValueError("the overhead camera sent nothing")
    except ValueError as exc:
        where = rec.failure("the command 'pick the pawn' stopped on an error", exc=exc, command="pick the pawn")
    rec.stop()
    story = (where / "story.txt").read_text()
    assert "stopped on an error" in story and "Traceback" in story and "ValueError: the overhead camera" in story


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_crash_nothing_caught_on_a_thread_is_an_incident(tmp_path):
    import threading

    rec = fr.FlightRecorder().start(tmp_path, crashes=True)

    def boom():
        raise RuntimeError("the sentry's thread died")
    t = threading.Thread(target=boom, name="sentry")
    t.start()
    t.join()
    rec.stop()
    found = [p for p in (rec.directory / "incidents").iterdir()]
    assert found and "sentry" in (found[0] / "story.txt").read_text()
    assert threading.excepthook is not None and rec._hooks == []                # hooks given back


def test_pictures_kept_for_later_can_be_turned_off_but_not_an_incidents(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setenv("MRCF_KEEP_PICTURES", "0")
    rec = fr.FlightRecorder().start(tmp_path)
    assert rec.keep_picture("scan_overhead", frame()) is None
    rec.frame("overhead", frame())
    with rec.action("pick") as act:
        act.problem("missed")
    rec.stop()
    incident = next((rec.directory / "incidents").iterdir())
    assert list((incident / "frames").glob("overhead_*.jpg"))


def test_kept_frames_go_through_the_cameras_guard(tmp_path):
    pytest.importorskip("cv2")
    seen = []

    class Env:
        def render_image(self, cam, width, height):
            return np.full((height, width, 3), 200, np.uint8)
    rec = fr.FlightRecorder().start(tmp_path)
    rec.watch_cameras(Env(), ["front"], hz=50.0, width=64, height=48,
                      keep=lambda cam, img: seen.append(cam) or np.zeros_like(img))
    import time
    time.sleep(0.15)
    rec.stop()
    assert seen and set(seen) == {"front"}
