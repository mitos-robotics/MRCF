"""
run.py's choice of arm: asked first, before anything reads MRCF_ARM; Enter
takes the arm plugged in; --arm chooses without asking. And each arm's
present(), the check behind it, which connects to nothing.
"""

from __future__ import annotations

import socket
import sys
from types import SimpleNamespace

import pytest

import run
from mrcf import backends


@pytest.fixture(autouse=True)
def _mrcf_arm_put_back(monkeypatch):
    # run.py sets MRCF_ARM itself; monkeypatch puts back only what it changed,
    # and a delenv of a variable not set records nothing. Set then delete: the
    # set records it, so whatever run.py leaves is undone after each test.
    monkeypatch.setenv("MRCF_ARM", "so101")
    monkeypatch.delenv("MRCF_ARM")


def _arms(monkeypatch, **arms):
    """Fake arms for run.py to list: name -> (found, how), or an Exception it fails to load with."""
    def arm(name=None):
        made = arms[name]
        if isinstance(made, Exception):
            raise made
        found, how = made

        def present():
            print(f"{name} was imported")         # what an arm's import prints (the SO-101's notes)
            return found, how
        return SimpleNamespace(NAME=name.upper(), present=present)
    monkeypatch.setattr(backends, "available", lambda: list(arms))
    monkeypatch.setattr(backends, "arm", arm)


def _answer(monkeypatch, reply):
    asked = []

    def ask(prompt=""):
        asked.append(prompt)
        if reply is None:
            raise AssertionError("asked")
        return reply
    monkeypatch.setattr("builtins.input", ask)
    return asked


def test_enter_takes_the_arm_plugged_in_over_a_mrcf_arm_left_set(monkeypatch, capsys):
    _arms(monkeypatch, so101=(True, "follower on /dev/x"), ur5=(False, "no answer"))
    monkeypatch.setenv("MRCF_ARM", "ur5")                       # left exported from a UR5 session
    _answer(monkeypatch, "")
    assert run._choose_arm(None) == "so101"
    assert backends.os.environ["MRCF_ARM"] == "so101"           # what the deployment's config will read
    out = capsys.readouterr().out
    assert "MRCF_ARM=ur5 is set in this terminal; Enter takes so101, the one plugged in" in out
    assert "plugged in" in out and "not found" in out


def test_the_arm_flag_chooses_without_asking_even_one_not_found(monkeypatch, capsys):
    _arms(monkeypatch, so101=(True, "follower"), ur5=(False, "no answer from 192.0.2.21"))
    _answer(monkeypatch, None)
    assert run._choose_arm("ur5") == "ur5"
    assert backends.os.environ["MRCF_ARM"] == "ur5"
    assert "though it was not found" in capsys.readouterr().out


def test_only_the_chosen_arms_import_output_is_shown(monkeypatch, capsys):
    _arms(monkeypatch, so101=(True, "follower"), ur5=(False, "no answer"))
    _answer(monkeypatch, "2")
    assert run._choose_arm(None) == "ur5"
    out = capsys.readouterr().out
    assert "ur5 was imported" in out and "so101 was imported" not in out


def test_an_arm_that_does_not_load_is_listed_but_not_driven(monkeypatch, capsys):
    _arms(monkeypatch, so101=(True, "follower"), ur5=ImportError("no module named rtde"))
    _answer(monkeypatch, "ur5")
    with pytest.raises(SystemExit) as stop:
        run._choose_arm(None)
    assert stop.value.code == 1
    assert "does not load here — ImportError: no module named rtde" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run._choose_arm("ur5")
    with pytest.raises(SystemExit):
        run._choose_arm("ur3")                                  # no such arm


def test_with_none_or_both_plugged_in_enter_takes_mrcf_arm_then_the_default(monkeypatch):
    none = [run._Arm("so101", False, ""), run._Arm("ur5", False, "")]
    both = [run._Arm("so101", True, ""), run._Arm("ur5", True, "")]
    assert run._default_arm(none) == backends.DEFAULT == "so101"
    assert run._default_arm(both) == "so101"
    monkeypatch.setenv("MRCF_ARM", "ur5")
    assert run._default_arm(none) == run._default_arm(both) == "ur5"
    assert run._default_arm([run._Arm("ur5", True, "", loads=False), run._Arm("so101", False, "")]) == "so101"


def test_the_flags_are_taken_and_the_rest_passed_on(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "--arm=ur5", "--deployment", "tabletop", "--verbose"])
    assert run._parse_run_args() == ("tabletop", "ur5", ["--verbose"])
    monkeypatch.setattr(sys, "argv", ["run.py", "--arm", "so101"])
    assert run._parse_run_args() == (None, "so101", [])


def test_the_arm_is_set_before_any_deployment_is_loaded(monkeypatch):
    _arms(monkeypatch, so101=(False, "no follower"), ur5=(True, "answering"))
    monkeypatch.setenv("MRCF_ARM", "so101")
    monkeypatch.setattr(sys, "argv", ["run.py"])
    _answer(monkeypatch, "")
    seen = {}

    def discover(arm=None):
        seen["arm"] = backends.os.environ["MRCF_ARM"]
        return [], []
    monkeypatch.setattr(run, "_discover", discover)
    monkeypatch.setattr(run, "_show_rig", lambda arm, ask=True: {})
    with pytest.raises(SystemExit):                             # no deployments: stops before launching
        run.main()
    assert seen == {"arm": "ur5"}


def test_only_the_chosen_arms_deployments_are_offered():
    # A deployment is one arm, one layout and one task: the UR5 is offered its
    # own, and the SO-101's are listed with the arm to choose for them.
    valid, skipped = run._discover("ur5")
    assert [d.name for d, _ in valid] == ["tabletop", "toolbench"]
    why = {d.name: reason for d, reason in skipped}
    assert why["chess"] == why["tidy"] == "the so101's: python run.py --arm so101"
    valid, skipped = run._discover("so101")
    assert "toolbench" not in [d.name for d, _ in valid]
    assert {d.name: reason for d, reason in skipped}["toolbench"] == "the ur5's: python run.py --arm ur5"


def test_a_deployment_named_alone_brings_its_arm(monkeypatch):
    assert run._deployment_arm(run.DEPLOYMENTS_DIR / "tidy") == "so101"
    assert run._deployment_arm(run.DEPLOYMENTS_DIR / "tabletop") == "ur5"
    _arms(monkeypatch, so101=(False, "no follower"), ur5=(True, "answering"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--deployment", "tidy"])
    _answer(monkeypatch, None)                                  # asks nothing
    seen = {}

    def discover(arm=None):
        seen["arm"] = arm
        return [], []
    monkeypatch.setattr(run, "_discover", discover)
    monkeypatch.setattr(run, "_show_rig", lambda arm, ask=True: {})
    with pytest.raises(SystemExit):
        run.main()
    assert seen == {"arm": "so101"}                             # not the UR5 plugged in


# ── The arms' own checks: nothing connected to ───────────────────────────────

def test_the_so101_is_there_when_its_serial_port_is(monkeypatch, tmp_path):
    from mrcf.backends import so101
    port = tmp_path / "tty.usbmodemTEST"
    monkeypatch.setattr(so101.rc, "FOLLOWER_PORT", str(port))
    assert so101.present()[0] is False
    port.touch()
    assert so101.present() == (True, f"follower on {port}")


def test_the_ur5_is_there_when_its_controller_answers(monkeypatch):
    from mrcf.backends import ur5
    monkeypatch.setattr(ur5.config, "HOST", None)
    assert ur5.present()[0] is False                           # no address: it would run in memory
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    monkeypatch.setattr(ur5.config, "HOST", "127.0.0.1")
    try:
        assert ur5.present(port=port) == (True, "controller answering at 127.0.0.1")
    finally:
        listener.close()
    found, how = ur5.present(port=port, timeout_s=0.2)           # nothing listening now
    assert found is False and how.startswith("no answer from 127.0.0.1")


def test_new_is_offered_for_the_arm_and_the_deployment(monkeypatch, capsys):
    monkeypatch.delenv("MRCF_DEPLOYMENT", raising=False)
    monkeypatch.setenv("MRCF_ARM", "ur5")
    valid = [(type("F", (), {"name": "tabletop"})(), None)]
    made = (type("F", (), {"name": "sorting"})(), None)
    monkeypatch.setattr(run, "_new_deployment", lambda arm: made)
    _answer(monkeypatch, "n")
    assert run._pick_deployment(valid, "ur5") is made
    out = capsys.readouterr().out
    assert "[n] new" in out and "on the ur5" in out
    _answer(monkeypatch, "")
    assert run._pick_deployment(valid, "ur5")[0].name == "tabletop"           # Enter: the arm's own


def test_a_new_deployment_from_the_launcher_is_made_on_the_chosen_arm(monkeypatch, tmp_path):
    import shutil

    for name in ("tidy", "tabletop"):
        shutil.copytree(run.DEPLOYMENTS_DIR / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(run, "DEPLOYMENTS_DIR", tmp_path)
    replies = iter(["2", "sort_parts", "N"])                                  # from tidy; its name; not run now
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    with pytest.raises(SystemExit) as stop:
        run._new_deployment("ur5")
    assert stop.value.code == 0
    assert run._deployment_arm(tmp_path / "sort_parts") == "ur5"
    assert (tmp_path / "sort_parts" / "profile.py").is_file()


def test_the_rig_is_shown_between_the_arm_and_the_job(monkeypatch, capsys):
    from mrcf import rig_report

    monkeypatch.setattr(rig_report, "report", lambda arm, config=None, listed=None: {
        "arm": {"name": "so101", "label": "SO-101", "found": True, "how": "follower on /dev/x"},
        "gripper": {"opens_mm": 80.0}, "cameras": [], "camera_check": {"plugged_in": None, "why": "",
        "rig_cameras_listed": [], "own_cameras_listed": []}, "sensors": [],
        "privacy": {"faces_covered": True, "pictures_kept": True}})
    run._show_rig("so101", ask=False)
    out = capsys.readouterr().out
    assert "The SO-101 (so101): plugged in" in out and "faces covered" in out


def test_the_whole_start_arm_rig_job_recorded_nothing_launched(monkeypatch, tmp_path, capsys):
    # run.main() end to end in this process: the arms faked (nothing asked of
    # the network or a port), the deployments the real ones, the launch caught.
    from mrcf import flight_recorder

    monkeypatch.setattr(run, "_survey", lambda: [run._Arm("so101", True, "follower on /dev/fake"),
                                                 run._Arm("ur5", False, "no answer")])
    monkeypatch.setattr(flight_recorder, "ROOT", tmp_path)
    monkeypatch.setattr(flight_recorder, "RECORDER", flight_recorder.FlightRecorder())
    launched = []
    monkeypatch.setattr(run, "_launch", lambda folder, args: launched.append(folder.name))
    monkeypatch.delenv("MRCF_DEPLOYMENT", raising=False)
    monkeypatch.setattr(sys, "argv", ["run.py"])
    replies = iter(["", "tidy"])                                 # Enter: the SO-101 plugged in; then tidy
    monkeypatch.setattr("builtins.input", lambda prompt="": print(prompt) or next(replies))
    run.main()
    out = capsys.readouterr().out
    assert launched == ["tidy"]
    assert "The SO-101 (so101)" in out and "camera    front" in out and "pictures" in out   # the rig, shown
    assert "[n] new" in out and "tabletop" in out and "the ur5's" in out                   # another arm's: said
    run_dir = next(tmp_path.iterdir())
    import json
    session = json.loads((run_dir / "session.json").read_text())
    assert session["arm"]["name"] == "so101" and session["deployment"]["name"] == "tidy"
    assert "cameras" in session["rig"] and "version" in session and "stopped" in session
    assert (run_dir / "console.log").is_file()
