"""
The tabletop deployment — the UR5's letter blocks — offline: the table task's
places and naming (mrcf/tabletop) as plain functions, and its skills end to
end on the UR5 in memory (mrcf/backends/ur5/offline.py) — a real pick, then
place_beside, a marked spot, putting it back.

Its skills register into the framework's one action registry, which the chess
deployment's tests also fill: each test here imports the deployment and the
table task fresh, under MRCF_ARM=ur5 (its arm, as run.py sets it), and puts
the registry back as it was after.
"""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import cv2  # noqa: F401 — the deployment's naming crops with it
except ImportError:
    pytest.skip("needs OpenCV", allow_module_level=True)

from mrcf.safety import WorkspaceBounds

BOUNDS = WorkspaceBounds(x_min=-0.70, x_max=0.70, y_min=-0.70, y_max=0.70, z_min=-0.005, z_max=0.6)


def _drop_tabletop_modules():
    # The deployment only: imported again, its config binds the table task anew
    # (tabletop.use), and the task's skills are made again for it.
    for name in [m for m in sys.modules if m == "deployments.tabletop" or m.startswith("deployments.tabletop.")]:
        del sys.modules[name]


@pytest.fixture
def tabletop(monkeypatch, tmp_path):
    from mrcf import manipulation, overhead_calibration
    from mrcf.action_registry import ACTIONS

    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    monkeypatch.setenv("MRCF_ARM", "ur5")
    saved_actions = dict(ACTIONS)
    saved_install = (manipulation._policies, manipulation._vision, manipulation._scenario)
    _drop_tabletop_modules()
    cfg = importlib.import_module("deployments.tabletop.config")
    monkeypatch.setattr(cfg, "PLACES_PATH", tmp_path / "places.json")
    monkeypatch.setattr(cfg, "MEMORY_PATH", tmp_path / "memory.json")
    try:
        importlib.import_module("deployments.tabletop.skills")          # binds the table task to it
        skills = importlib.import_module("mrcf.tabletop.skills")
        yield SimpleNamespace(cfg=cfg, skills=skills, places=importlib.import_module("mrcf.tabletop.places"),
                              naming=importlib.import_module("mrcf.tabletop.naming"), actions=ACTIONS)
    finally:
        ACTIONS.clear()
        ACTIONS.update(saved_actions)
        manipulation._policies, manipulation._vision, manipulation._scenario = saved_install
        _drop_tabletop_modules()


# ── Memory ────────────────────────────────────────────────────────────────────

def test_memory_starts_fresh_unless_kept_across_restarts(tabletop, monkeypatch):
    from mrcf.object_memory import Detection, ObjectMemory

    ObjectMemory(tabletop.cfg.MEMORY_PATH).observe([Detection(position=(0.4, 0.3), label="block")])
    assert tabletop.cfg.REMEMBER_ACROSS_RESTARTS is False
    assert tabletop.skills.memory().tracks == {}

    monkeypatch.setattr(tabletop.skills, "_MEMORY", None)
    ObjectMemory(tabletop.cfg.MEMORY_PATH).observe([Detection(position=(0.4, 0.3), label="block")])
    monkeypatch.setattr(tabletop.cfg, "REMEMBER_ACROSS_RESTARTS", True)
    assert len(tabletop.skills.memory().tracks) == 1


# ── Places ────────────────────────────────────────────────────────────────────

def test_sides_are_the_viewers(tabletop):
    p = tabletop.places
    cup = np.array([0.0, -0.45])                     # straight out along -y from the base
    facing = {s: p.beside(cup, 0.03, 0.02, s, [], BOUNDS, seen_from="facing") for s in ("left", "right", "front", "behind")}
    assert facing["front"][1] < cup[1] and facing["behind"][1] > cup[1]     # front: toward the person, away from the arm
    assert facing["left"][0] < cup[0] < facing["right"][0]                  # the person faces +y: their left is -x
    arm = {s: p.beside(cup, 0.03, 0.02, s, [], BOUNDS, seen_from="arm") for s in ("left", "right", "front")}
    assert arm["front"][1] > cup[1] and arm["left"][0] > cup[0]             # the arm looks along -y: its left is +x
    for spot in facing.values():
        assert np.linalg.norm(spot - cup) == pytest.approx(0.03 + 0.02 + 0.025)


def test_beside_keeps_clear_of_neighbours_and_the_edge(tabletop):
    p = tabletop.places
    cup = np.array([0.0, -0.45])
    right = p.beside(cup, 0.03, 0.02, "right", [], BOUNDS)
    blocked = [(right, 0.03)]                                                # something stands there
    assert p.beside(cup, 0.03, 0.02, "right", blocked, BOUNDS) is None      # no room that side
    other = p.beside(cup, 0.03, 0.02, "any", blocked, BOUNDS)               # "any" goes round to the left
    assert np.allclose(other, p.beside(cup, 0.03, 0.02, "left", [], BOUNDS))
    near = [(right + np.array([0.0, 0.05]), 0.02)]                          # beside it, not on it: turned away from
    turned = p.beside(cup, 0.03, 0.02, "right", near, BOUNDS)
    assert turned is not None and not np.allclose(turned, right)
    assert np.linalg.norm(turned - near[0][0]) >= 0.02 + 0.02 + 0.025 - 1e-9
    walled = [(cup + 0.08 * np.array([np.cos(a), np.sin(a)]), 0.03) for a in np.radians(np.arange(0, 360, 20))]
    assert p.beside(cup, 0.03, 0.02, "any", walled, BOUNDS) is None
    edge = np.array([0.0, -0.66])                                            # behind it (for the person) is off the table
    assert p.beside(edge, 0.03, 0.02, "front", [], BOUNDS) is None
    assert p.beside(cup, 0.03, 0.02, "left", [], BOUNDS, reachable=lambda xy: False) is None
    with pytest.raises(ValueError):
        p.beside(cup, 0.03, 0.02, "above", [], BOUNDS)


def test_marked_spots_round_trip(tabletop, tmp_path):
    p = tabletop.places
    path = tmp_path / "spots.json"
    assert p.load(path) == {}
    p.save(path, {p.clean_name("Parking Spot"): (0.1, -0.4)})
    assert p.load(path) == {"parking_spot": (0.1, -0.4)}
    (target,) = p.targets(p.load(path), 0.01)
    assert target.name == "parking_spot" and np.allclose(target.position, [0.1, -0.4, 0.01])


# ── Naming ────────────────────────────────────────────────────────────────────

def test_names_are_colour_and_what_it_is(tabletop):
    n = tabletop.naming
    track = SimpleNamespace(id="obj-7", label=None, attributes={}, height_mm=None, feet_mm=[40.0])
    assert n.object_name(track) == "object_7" and "not yet named" in n.describe(track)
    track.label, track.attributes = "coffee mug", {"colour": "blue"}
    assert n.object_name(track) == "blue_coffee_mug"
    track.label = "red bottle"
    track.attributes = {"colour": "red"}
    assert n.object_name(track) == "red_bottle"                             # not red_red_bottle
    assert n.clean_what("A Toy-Car!! with wheels and more") == "a toy car"


def _each(answer, asked=None):
    """A stand-in vision model that gives every numbered thing on a sheet this answer (and keeps each call's schema)."""
    def fake_vlm(prompt, image_b64=None, system=None, json_mode=True, _tag="", model=None, schema=None):
        if asked is not None:
            asked.append(schema)
        keys = list((schema or {}).get("properties", {}))
        if not (keys and keys[0] == "1"):
            return dict(answer)
        said = (f"{answer['colour']} {answer['what']}" if answer.get("is_object") else "nothing")
        return {k: said for k in keys}
    return fake_vlm


def test_the_model_is_asked_once_per_object_and_held_to_its_form(tabletop, monkeypatch):
    from mrcf.object_memory import Detection, ObjectMemory
    from mrcf import vlm

    n = tabletop.naming
    asked: list = []
    monkeypatch.setattr(vlm, "call_vlm", _each({"is_object": True, "what": "Glue Stick", "colour": "purple"}, asked))
    mem = ObjectMemory(None, namer=n.object_name)
    tracks, _ = mem.observe([Detection(position=(0.0, -0.45), foot_mm=30.0, appearance={"a": 0.0, "b": 0.0}),
                             Detection(position=(0.2, -0.45), foot_mm=30.0, appearance={"a": 40.0, "b": 0.0})])
    frame = np.zeros((120, 160, 3), np.uint8)
    mask = np.zeros((120, 160), bool)
    mask[40:70, 60:90] = True
    found = [{"mask": mask, "measurement": SimpleNamespace(base_radius_mm=15.0, height_mm=90.0)}] * 2
    n.identifier(mem)(frame, found, tracks)
    assert mem.name(tracks[0]) == "purple_glue_stick" and mem.name(tracks[1]) == "purple_glue_stick"
    assert len(asked) == 1                                                  # both in one call
    assert list(asked[0]["properties"]) == ["1", "2"]
    assert asked[0]["properties"]["1"] == {"type": "string"}
    n.identifier(mem)(frame, found, tracks)
    assert len(asked) == 1                                                  # named once

    monkeypatch.setattr(vlm, "call_vlm", lambda *a, **k: {"is_object": False, "what": "", "colour": "black"})
    assert n.ask("png", 30.0) == (n.NOT_AN_OBJECT, "black")
    monkeypatch.setattr(vlm, "call_vlm", lambda *a, **k: {})
    assert n.ask("png", 30.0) is None


# ── The skills, on the UR5 in memory ──────────────────────────────────────────

def _scene(tabletop, objects):
    from mrcf.object_memory import Detection
    from mrcf.vlm import SceneObject, SceneState

    mem = tabletop.skills.memory()
    # Each object as a scan from an angled overhead camera gives it: foot and height.
    tracks, _ = mem.observe([Detection(position=tuple(xy), foot_mm=foot, height_mm=height,
                                       appearance={"a": 0.0, "b": 0.0}) for _, xy, foot, height in objects])
    state = SceneState(memory=mem)
    for (label, xy, foot, _), track in zip(objects, tracks):
        colour, what = label.split(" ", 1)
        mem.set_label(track.id, what, "user", attributes={"colour": colour})
        obj = SceneObject(mem.name(track), label, position=np.array([xy[0], xy[1], tabletop.cfg.TABLE_Z]))
        obj.track_id, obj.geometry = track.id, {"base_diameter_m": foot / 1000.0}
        obj.measured_shape = track.shape() if track.shape_current() else None
        state.objects.append(obj)
    return state


def _env(tabletop):
    # These scenes stand where the arm in memory starts, outside the rig's box
    # (robot_config GRIPPER_BOX): they are about the skills, not the box.
    return tabletop.cfg.ARM.make_env(host=None, speed_m_s=4.0, joint_speed_rad_s=6.0, control_hz=1000.0,
                                     home_joints_deg=(0.0, -90.0, 90.0, -90.0, -90.0, 0.0), gripper_box=None)


# Pointing down over (0.35, 0.35, 0.45): inside the rig's box.
IN_BOX_Q_DEG = (212.3, -90.5, 87.8, -87.3, -90.0, -57.7)


def _env_in_box(tabletop):
    """The arm in memory started inside the rig's box, the box on: as the real rig runs."""
    from mrcf.backends.ur5.arm import URArm
    from mrcf.backends.ur5.offline import OfflineController, OfflineGripper, connect_offline
    rc = tabletop.cfg.ARM.config
    arm = URArm(None, gripper=OfflineGripper(rc.GRIPPER_MAX_OPEN_M),
                connect_with=connect_offline(OfflineController(q_deg=IN_BOX_Q_DEG)), **rc.ARM_TUNING)
    arm.connect()
    env = tabletop.cfg.ARM.make_env(sdk=arm, sources={}, motion_guard_zone=None,     # no camera, no sentry
                                    speed_m_s=4.0, joint_speed_rad_s=6.0, control_hz=1000.0,
                                    home_joints_deg=IN_BOX_Q_DEG)
    assert env._gripper_box is not None
    return env


def _do(tabletop, env, state, action, **params):
    from mrcf.action_registry import dispatch
    return dispatch({"action": action, **params}, env, state)


def test_pick_then_place_beside_then_put_it_back(tabletop):
    env = _env_in_box(tabletop)
    state = _scene(tabletop, [("red bottle", (0.42, 0.30), 40.0, 180.0), ("blue mug", (0.30, 0.48), 45.0, 95.0)])
    env._sdk.gripper.object_m = 0.040                       # the bottle, between the jaws when they close
    assert _do(tabletop, env, state, "pick", object_name="red_bottle") is True
    assert state.held_object().name == "red_bottle"
    assert any(t.name == "where_red_bottle_was" for t in state.targets)     # somewhere to put it back

    assert _do(tabletop, env, state, "place_beside", object_name="blue_mug", side="left") is True
    bottle, mug = state.find("red_bottle"), state.find("blue_mug")
    assert state.held_object() is None and bottle.position is not None
    gap = np.linalg.norm(bottle.position[:2] - mug.position[:2])
    assert gap == pytest.approx(0.020 + 0.0225 + tabletop.cfg.BESIDE_GAP_M, abs=0.002)
    assert np.linalg.norm(env.get_ee_pos()[:2] - bottle.position[:2]) < 0.01   # the arm put it there


def test_a_marked_spot_is_somewhere_to_put_things(tabletop):
    env = _env_in_box(tabletop)
    state = _scene(tabletop, [("green box", (0.45, 0.35), 30.0, 60.0)])
    assert env.move_ee_to(np.array([0.30, 0.50, 0.15]))
    assert _do(tabletop, env, state, "mark_place", name="Tray") is True
    assert tabletop.places.load(tabletop.cfg.PLACES_PATH)["tray"] == pytest.approx((0.30, 0.50), abs=1e-3)
    env._sdk.gripper.object_m = 0.030
    assert _do(tabletop, env, state, "pick", object_name="green_box") is True
    assert _do(tabletop, env, state, "place", target_name="tray") is True
    assert np.allclose(state.find("green_box").position[:2], (0.30, 0.50), atol=1e-3)
    assert _do(tabletop, env, state, "forget_place", name="tray") is True and not tabletop.places.load(tabletop.cfg.PLACES_PATH)


def test_identify_takes_a_name_a_colour_and_a_height(tabletop):
    env = _env(tabletop)
    state = _scene(tabletop, [("red bottle", (-0.45, -0.15), 40.0, 180.0)])
    assert _do(tabletop, env, state, "identify", object_name="red_bottle", what="green glue stick 9 cm tall") is True
    obj = state.find("green_glue_stick")
    assert obj is not None and obj.measured_shape["height_mm"] == pytest.approx(90.0)


def test_the_scan_drops_what_is_not_an_object_and_offers_the_marked_spots(tabletop, monkeypatch):
    from mrcf import scene_scan

    env = _env(tabletop)
    state = _scene(tabletop, [("red bottle", (-0.45, -0.15), 40.0, 180.0), ("black smudge", (-0.40, 0.10), 20.0, None)])
    mem = tabletop.skills.memory()
    smudge = state.find("black_smudge")
    mem.set_label(smudge.track_id, tabletop.naming.NOT_AN_OBJECT, "user")
    tabletop.places.save(tabletop.cfg.PLACES_PATH, {"tray": (-0.3, 0.3)})
    monkeypatch.setattr(scene_scan, "observe", lambda *a, **k: True)        # the scan itself needs a camera
    assert tabletop.skills.scan_scene(env, state) is True
    assert [o.name for o in state.objects] == ["red_bottle"] and [t.name for t in state.targets] == ["tray"]


def test_run_py_accepts_the_deployment(tabletop):
    from pathlib import Path

    import run
    ok, why, _ = run._validate(Path(tabletop.cfg.__file__).parent)
    assert ok, why


def test_tools_take_the_arms_own_task_and_refuse_another_arms(tabletop, monkeypatch):
    from tools import _common

    monkeypatch.setattr(sys, "argv", ["tool"])
    monkeypatch.delenv("MRCF_DEPLOYMENT", raising=False)
    monkeypatch.setattr(_common, "_last_run", lambda: None)                     # none launched before
    assert _common.task_config().__name__ == "deployments.tabletop.config"       # the UR5's DEPLOYMENT
    monkeypatch.setenv("MRCF_DEPLOYMENT", "chess")
    with pytest.raises(SystemExit, match="the so101's"):                        # chess is the SO-101's
        _common.task_config()
    monkeypatch.setenv("MRCF_ARM", "so101")
    assert _common.task_config().__name__ == "deployments.chess.config"


def test_run_py_asks_which_of_the_arms_deployments_and_enter_takes_its_own(tabletop, monkeypatch, capsys):
    # Two deployments on the SO-101: `python run.py` asks, even on an arm with one of its own.
    import run

    replies = []

    def answer(prompt=""):
        print(prompt)
        return replies.pop(0)
    monkeypatch.setattr("builtins.input", answer)
    valid = [(type("F", (), {"name": n})(), None) for n in ("chess", "tidy")]
    monkeypatch.delenv("MRCF_DEPLOYMENT", raising=False)
    monkeypatch.setenv("MRCF_ARM", "so101")
    replies[:] = [""]
    assert run._pick_deployment(valid)[0].name == "chess"                    # the SO-101's DEPLOYMENT
    assert "Enter for chess" in capsys.readouterr().out
    replies[:] = ["2"]
    assert run._pick_deployment(valid)[0].name == "tidy"
    replies[:] = ["tidy"]
    assert run._pick_deployment(valid)[0].name == "tidy"
    monkeypatch.setenv("MRCF_DEPLOYMENT", "tidy")
    replies[:] = [""]
    assert run._pick_deployment(valid)[0].name == "tidy"
    assert any(line.split()[1:2] == ["tidy"] and line.endswith("(MRCF_DEPLOYMENT)")
               for line in capsys.readouterr().out.splitlines())


def test_run_py_with_no_task_of_its_own_enter_chooses_nothing(tabletop, monkeypatch):
    import run

    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setenv("MRCF_DEPLOYMENT", "sorting")                         # not one of those found
    valid = [(type("F", (), {"name": n})(), None) for n in ("chess", "tidy")]
    with pytest.raises(SystemExit):
        run._pick_deployment(valid)


# ── Stacking: a block on a block ──────────────────────────────────────────────

def _lowest_z(env, run):
    """The lowest grasp point the arm was sent to while run() ran."""
    sent, move = [], env.move_ee_to

    def logged(target, *a, **k):
        sent.append(float(np.asarray(target, float)[2]))
        return move(target, *a, **k)

    env.move_ee_to = logged
    try:
        assert run() is True
    finally:
        env.move_ee_to = move
    return min(sent)


# Where the camera found the letter blocks on the rig, 2026-11-12: inside the box.
BLOCKS = [("red block", (0.44, 0.37), 40.0, 40.0), ("blue block", (0.39, 0.46), 40.0, 40.0),
          ("green block", (0.26, 0.48), 40.0, 40.0), ("yellow block", (0.32, 0.56), 40.0, 40.0)]


def test_blocks_stack_three_high_and_come_off_again(tabletop):
    env = _env_in_box(tabletop)
    state = _scene(tabletop, BLOCKS)
    env._sdk.gripper.object_m = 0.040
    table = tabletop.cfg.TABLE_Z
    # Put down on the table, for the height a place lets go at.
    assert _do(tabletop, env, state, "pick", object_name="yellow_block") is True
    on_table = _lowest_z(env, lambda: _do(tabletop, env, state, "place", target_name="where_yellow_block_was"))

    assert _do(tabletop, env, state, "pick", object_name="red_block") is True
    on_blue = _lowest_z(env, lambda: _do(tabletop, env, state, "place_on", object_name="blue_block"))
    red, blue = state.find("red_block"), state.find("blue_block")
    assert red.on == blue.name and red.position[2] == pytest.approx(table + 0.040)
    assert np.allclose(red.position[:2], blue.position[:2])
    assert on_blue - on_table == pytest.approx(0.040, abs=0.002)    # let go one block higher

    # Onto blue again: blue's stack has red on top, so green goes on red.
    assert _do(tabletop, env, state, "pick", object_name="green_block") is True
    on_red = _lowest_z(env, lambda: _do(tabletop, env, state, "place_on", object_name="blue_block"))
    green = state.find("green_block")
    assert green.on == red.name and green.position[2] == pytest.approx(table + 0.080)
    assert on_red - on_table == pytest.approx(0.080, abs=0.002)
    assert "[ON TOP OF red_block]" in str(green)

    # Not from under what stands on it; the top one comes off, from its own height.
    assert _do(tabletop, env, state, "pick", object_name="blue_block") is False
    assert state.held_object() is None
    off_top = _lowest_z(env, lambda: _do(tabletop, env, state, "pick", object_name="green_block"))
    from mrcf.manipulation import policies
    tips = policies().spec.tip_below_grasp_m
    assert off_top - tips > table + 0.080                  # the jaw tips never went down into the red block
    assert state.held_object().name == "green_block" and state.held_object().on is None
    track = tabletop.skills.memory().get(red.track_id)
    assert track.on == blue.track_id and track.raised_mm == pytest.approx(40.0)


def test_a_held_block_is_squared_by_the_least_turn_of_the_wrist(tabletop):
    # UR5, 2026-11-13 15:09: red's faces lay 44 deg from the arm's way at red,
    # and putting blue on it turned the wrist -87 deg where +7 squared them too:
    # the gripper's cable wound round the wrist, and blue landed 9 mm off red.
    # The run again (its blocks, faces and the park pose it started from), in memory.
    from mrcf.backends.ur5.arm import URArm
    from mrcf.backends.ur5.offline import OfflineController, OfflineGripper, connect_offline
    rc = tabletop.cfg.ARM.config
    park = rc.PARK_JOINTS_DEG
    arm = URArm(None, gripper=OfflineGripper(rc.GRIPPER_MAX_OPEN_M),
                connect_with=connect_offline(OfflineController(q_deg=park)), **rc.ARM_TUNING)
    arm.connect()
    env = tabletop.cfg.ARM.make_env(sdk=arm, sources={}, motion_guard_zone=None,
                                    speed_m_s=4.0, joint_speed_rad_s=6.0, control_hz=1000.0)
    blocks = [("blue block", (0.50806, 0.4679), 30.0, 30.0, 75.8), ("red block", (0.58638, 0.29363), 30.0, 30.0, 73.0),
              ("green block", (0.26112, 0.39046), 30.0, 30.0, 5.4)]
    state = _scene(tabletop, [b[:4] for b in blocks])
    for obj, b in zip(state.objects, blocks):
        obj.faces_deg = b[4]
    wrist, run = [float(park[5])], env._run

    def watched(send, budget, limit_n=None):
        run(send, budget, limit_n)
        wrist.append(float(np.degrees(env._sdk.joints())[5]))
    env._run = watched
    env._sdk.gripper.object_m = 0.030
    for action, name in (("pick", "blue_block"), ("place_on", "red_block"), ("pick", "green_block"),
                         ("place_on", "blue_block")):
        assert _do(tabletop, env, state, action, object_name=name) is True, (action, name)
    turns = np.abs(np.diff(wrist))
    assert turns.max() < 45.0, turns.round(1)                       # was 87 deg
    assert max(abs(w - park[5]) for w in wrist) < 90.0               # was 142 deg from park


def test_put_back_after_a_tower_goes_back_where_each_block_stood(tabletop):
    # UR5, 2026-11-13 15:47: "put the blocks back" picked blue off the tower, and
    # where_blue_block_d_was became the tower's spot; the place there was refused.
    env = _env_in_box(tabletop)
    state = _scene(tabletop, BLOCKS)
    env._sdk.gripper.object_m = 0.040
    start = {n: state.find(n).position[:2].copy() for n in ("red_block", "green_block")}
    for action, name in (("pick", "red_block"), ("place_on", "blue_block"),
                         ("pick", "green_block"), ("place_on", "red_block")):
        assert _do(tabletop, env, state, action, object_name=name) is True, (action, name)
    plan = [{"action": "pick", "parameters": {"object_name": "green_block"}},
            {"action": "place", "parameters": {"target_name": "where_green_block_was"}},
            {"action": "pick", "parameters": {"object_name": "red_block"}},
            {"action": "place", "parameters": {"target_name": "where_red_block_was"}}]
    from mrcf import controller
    assert controller.execute_plan(plan, env, state, None) is False          # every step ran
    for name in ("green_block", "red_block"):
        obj = state.find(name)
        assert obj.on is None and np.allclose(obj.position[:2], start[name], atol=0.002), name
        assert np.allclose(state.find(f"where_{name}_was").position[:2], start[name], atol=1e-6)


def test_place_at_an_object_puts_it_on_top(tabletop):
    env = _env_in_box(tabletop)
    state = _scene(tabletop, BLOCKS[:2])
    env._sdk.gripper.object_m = 0.040
    assert _do(tabletop, env, state, "pick", object_name="red_block") is True
    assert _do(tabletop, env, state, "place", target_name="blue_block") is True
    assert state.find("red_block").on == "blue_block"


def test_a_stack_is_planned_with_the_gripper_rule(tabletop):
    from mrcf.manipulation import gripper_sequence_validator
    state = _scene(tabletop, BLOCKS[:3])
    plan = [{"action": "pick", "object_name": "red_block"}, {"action": "place_on", "object_name": "blue_block"},
            {"action": "pick", "object_name": "green_block"}, {"action": "place_on", "object_name": "red_block"}]
    assert not [v for v in gripper_sequence_validator(plan, state) if v.severity == "block"]


def test_a_block_has_the_size_a_block_is_known_to_have(tabletop):
    from mrcf.object_memory import Detection
    mem = tabletop.skills.memory()
    (block, bottle), _ = mem.observe([Detection(position=(0.3, 0.3), foot_mm=35.0),
                                      Detection(position=(0.5, 0.3), foot_mm=60.0)])
    mem.set_label(block.id, "block p", "vlm", attributes={"colour": "blue"})
    mem.set_label(bottle.id, "bottle", "vlm", attributes={"colour": "red"})
    height, width = tabletop.cfg.KNOWN_SIZES_MM["block"]
    tabletop.skills.known_sizes(mem)
    assert block.shape()["height_mm"] == pytest.approx(height) and block.shape_current()
    assert block.shape()["widths_mm"][20] == pytest.approx(width)
    assert bottle.shape() is None                          # nothing known: the camera's, or nothing
    block.shape_doubted = True                             # moved by hand: what stands there may be another
    tabletop.skills.known_sizes(mem)
    assert block.shape_current() and block.shape()["height_mm"] == pytest.approx(height)


def _jaw_deg(env) -> float:
    """Which way the jaws open now, deg from x, mod 90 (the flange's y: JAW_SLIDE_AXIS)."""
    axis = env.gripper_frame()[:3, 1]
    return float(np.degrees(np.arctan2(axis[1], axis[0])) % 90.0)


def _mod90_apart(a, b) -> float:
    d = abs(a - b) % 90.0
    return min(d, 90.0 - d)


def test_a_turned_block_is_gripped_across_its_sides_and_squared_on_the_one_below(tabletop):
    env = _env_in_box(tabletop)
    state = _scene(tabletop, BLOCKS[:2])
    env._sdk.gripper.object_m = 0.040
    red, blue = state.find("red_block"), state.find("blue_block")
    red.faces_deg, blue.faces_deg = 30.0, 10.0
    assert _do(tabletop, env, state, "pick", object_name="red_block") is True
    assert _mod90_apart(_jaw_deg(env), 30.0) < 1.0                 # closed across its sides
    assert _do(tabletop, env, state, "place_on", object_name="blue_block") is True
    assert _mod90_apart(_jaw_deg(env), 10.0) < 1.0                 # let go squared on blue's
    assert _mod90_apart(state.find("red_block").faces_deg, 10.0) < 1.0


def test_a_block_held_at_a_width_far_from_its_known_size_is_said(tabletop, capsys):
    env = _env_in_box(tabletop)
    state = _scene(tabletop, BLOCKS[:1])
    tabletop.skills.known_sizes(tabletop.skills.memory())
    red = state.find("red_block")
    red.measured_shape = tabletop.skills.memory().get(red.track_id).shape()
    known = tabletop.cfg.KNOWN_SIZES_MM["block"][1] / 1000.0
    env._sdk.gripper.object_m = known + 0.010             # the block is 10 mm wider than it is known to be
    assert _do(tabletop, env, state, "pick", object_name="red_block") is True
    assert "[Shape] red_block is" in capsys.readouterr().out


def test_a_block_taken_for_another_by_its_look_is_told_apart_by_its_letter(tabletop, monkeypatch):
    """2026-11-12: green 7 taken off a stack by hand came back named blue_block_z, and a plan moved it twice."""
    from mrcf import vlm
    from mrcf.object_memory import Detection, ObjectMemory

    n = tabletop.naming
    asked: list = []
    fake_vlm = _each({"is_object": True, "what": "block 7", "colour": "green"}, asked)

    def block(x, y, foot, a, b):
        return Detection(position=(x, y), foot_mm=foot, appearance={"a": a, "b": b})

    mem = ObjectMemory(None, namer=n.object_name)
    g, z, r = mem.observe([block(0.35, 0.42, 22.0, -12, -18), block(0.60, 0.27, 37.0, -15, -12),
                           block(0.58, 0.37, 34.0, 9, -14)])[0]
    mem.set_label(g.id, "block 7", "vlm", {"colour": "green"})
    mem.set_label(z.id, "block z", "vlm", {"colour": "blue"})
    mem.set_label(r.id, "block u", "vlm", {"colour": "red"})
    mem.observe([block(0.35, 0.42, 22.0, -12, -18), block(0.58, 0.37, 34.0, 9, -14)])
    mem.picked(g.id)
    mem.placed(g.id, (0.58, 0.37), on=r.id, raised_mm=30.0)
    tracks, _ = mem.observe([block(0.58, 0.37, 34.0, 9, -14), block(0.40, 0.436, 34.0, -13, -18)])
    assert mem.name(tracks[1]) == "blue_block_z"                            # by its look alone

    monkeypatch.setattr(vlm, "call_vlm", fake_vlm)
    frame = np.zeros((120, 160, 3), np.uint8)
    mask = np.zeros((120, 160), bool)
    mask[40:70, 60:90] = True
    found = [{"mask": mask, "measurement": SimpleNamespace(base_radius_mm=17.0, height_mm=None)}] * 2
    n.identifier(mem)(frame, found, tracks)
    assert len(asked) == 1                                  # only the guess: red U stayed where it was
    assert tracks[1].id == g.id and mem.name(tracks[1]) == "green_block_7"
    assert mem.get(z.id).state == "missing" and mem.name(mem.get(z.id)) == "blue_block_z"
    n.identifier(mem)(frame, found, tracks)
    assert len(asked) == 1                                  # settled: not asked again this scan


def test_what_the_sheet_calls_nothing_is_asked_again_by_itself_and_a_long_tile_is_kept_close(tabletop, monkeypatch):
    """2026-11-19, the SO-101's table: the scoop's tile showed half the table, the sheet said "nothing"."""
    from mrcf import vlm
    from mrcf.object_memory import Detection, ObjectMemory

    n = tabletop.naming
    frame = np.zeros((400, 400, 3), np.uint8)
    mask = np.zeros((400, 400), bool)
    mask[150:250, 100:300] = True                                           # 200 px long
    assert n.crop_tile(frame, mask, side=64) is not None
    import cv2
    monkeypatch.setattr(cv2, "resize", lambda c, size, **k: c)              # the tile as cut, not shrunk
    monkeypatch.setattr(n.cfg, "NAME_TILE_PAD_MAX_PX", None)
    assert n.crop_tile(frame, mask).shape[0] == 2 * 218                     # 199 px + 0.6 of it each side
    monkeypatch.setattr(n.cfg, "NAME_TILE_PAD_MAX_PX", 40)
    assert n.crop_tile(frame, mask).shape[0] == 2 * 139                     # ... + 40 px each side
    monkeypatch.undo()

    asked: list = []

    def fake_vlm(prompt, image_b64=None, system=None, json_mode=True, _tag="", model=None, schema=None):
        asked.append(_tag)
        keys = list((schema or {}).get("properties", {}))
        if keys and keys[0] == "1":
            return {k: "nothing" for k in keys}
        return {"is_object": True, "what": "scoop", "colour": "blue"}

    monkeypatch.setattr(vlm, "call_vlm", fake_vlm)
    found = [{"mask": mask, "measurement": SimpleNamespace(base_radius_mm=47.0, height_mm=None)}]

    def look():
        mem = ObjectMemory(None, namer=n.object_name)
        tracks, _ = mem.observe([Detection(position=(0.2, 0.15), foot_mm=94.0, appearance={"a": 0.0, "b": -40.0})])
        n.identifier(mem)(frame, found, tracks)
        return mem.get(tracks[0].id).label

    monkeypatch.setattr(n.cfg, "NAME_NOTHING_TWICE", False)
    assert look() == n.NOT_AN_OBJECT and asked == ["describe_numbered"]   # off: the sheet's word stands
    asked.clear()
    monkeypatch.setattr(n.cfg, "NAME_NOTHING_TWICE", True)
    assert look() == "scoop" and asked == ["describe_numbered", "name"]
    monkeypatch.setattr(n.cfg, "NAME_NOTHING_TWICE_MAX", 0)
    asked.clear()
    assert look() == n.NOT_AN_OBJECT and asked == ["describe_numbered"]


def test_a_moved_block_the_model_reads_the_same_keeps_its_name_and_colour(tabletop, monkeypatch):
    from mrcf import vlm
    from mrcf.object_memory import Detection, ObjectMemory

    n = tabletop.naming
    monkeypatch.setattr(vlm, "call_vlm", _each({"is_object": True, "what": "block d", "colour": "white"}))
    mem = ObjectMemory(None, namer=n.object_name)
    (d,), _ = mem.observe([Detection(position=(0.36, 0.22), foot_mm=34.0, appearance={"a": -7, "b": -33})])
    mem.set_label(d.id, "block d", "vlm", {"colour": "blue"})
    tracks, report = mem.observe([Detection(position=(0.45, 0.30), foot_mm=34.0, appearance={"a": -7, "b": -33})])
    assert [t.id for t, _ in report["moved"]] == [d.id]
    mask = np.zeros((120, 160), bool)
    mask[40:70, 60:90] = True
    found = [{"mask": mask, "measurement": SimpleNamespace(base_radius_mm=17.0, height_mm=None)}]
    n.identifier(mem)(np.zeros((120, 160, 3), np.uint8), found, tracks)
    assert mem.name(tracks[0]) == "blue_block_d"            # the letter agrees: one odd colour does not rename it

    monkeypatch.setattr(vlm, "call_vlm", _each({"is_object": True, "what": "block p", "colour": "white"}))
    tracks, _ = mem.observe([Detection(position=(0.50, 0.35), foot_mm=34.0, appearance={"a": -7, "b": -33})])
    n.identifier(mem)(np.zeros((120, 160, 3), np.uint8), found, tracks)
    assert mem.name(tracks[0]) == "blue_block_d"            # another face, another letter: nothing to hand it to


def test_freedrive_is_refused_on_an_arm_that_falls_when_let_go(tabletop):
    """An SO-101 has disable_torque, and with it off the arm drops: that is not freedrive."""
    skills = importlib.import_module("deployments.tabletop.skills")

    class Limp:
        torque = []

        def disable_torque(self):
            self.torque.append("off")

        def enable_torque(self):
            self.torque.append("on")

    class Holds:
        calls = []

        def freedrive(self, on):
            self.calls.append(on)

    limp, holds = Limp(), Holds()
    assert skills.skill_freedrive(limp, None, seconds=1) is False and limp.torque == []
    assert skills.skill_freedrive(holds, None, seconds=1) is True and holds.calls == [True, False]


def test_a_height_the_user_gives_is_not_replaced_by_the_known_size(tabletop):
    """A block said to be 40 mm stays 40 mm, though blocks are known to be 30: the user is trusted over the config."""
    env = _env(tabletop)
    state = _scene(tabletop, [("blue block", (-0.45, -0.15), 34.0, 30.0)])
    skills = importlib.import_module("deployments.tabletop.skills")
    assert _do(tabletop, env, state, "identify", object_name="blue_block", what="block z 40 mm tall") is True
    track = state.memory.get(state.find("block_z").track_id)
    skills.known_sizes(state.memory)
    assert track.height_mm == pytest.approx(40.0)


def test_a_name_the_user_gives_keeps_its_number(tabletop):
    env = _env(tabletop)
    state = _scene(tabletop, [("white cube", (-0.45, -0.15), 20.0, 20.0)])
    assert _do(tabletop, env, state, "identify", object_name="white_cube", what="white dice 6") is True
    assert state.find("white_dice_6") is not None


def test_a_sheets_answers_read_as_what_and_colour(tabletop):
    n = tabletop.naming
    assert n._read("Blue block P") == ("block p", "blue")
    assert n._read("green block 7") == ("block 7", "green")
    assert n._read("red coffee mug 2") == ("coffee mug 2", "red")
    assert n._read("nothing") == (n.NOT_AN_OBJECT, "")
    assert n._read("shadow, not an object") == (n.NOT_AN_OBJECT, "")
    assert n._read("teapot") == ("teapot", "")
    assert n._read(None) is None
    assert n._read("block P, blue") == ("block p", "blue")                  # the colour anywhere
    assert n._read("dark green block 7") == ("block 7", "green")
    assert n._read("gray mug") == ("mug", "grey")
    assert n._read("wooden letter block p") == ("letter block p", "")       # its letter kept
    assert n._read("shadow") == (n.NOT_AN_OBJECT, "")
    assert n._read("a mark on the table") == (n.NOT_AN_OBJECT, "")
    assert n._read("yellow tape measure") == ("tape measure", "yellow")


def test_an_arm_that_looks_for_heights_takes_no_size_from_a_name(tabletop, monkeypatch):
    # 2026-11-16, the SO-101: a 55 mm speaker called a "black block" was taken for the UR5's
    # 30 mm letter blocks — measured, so never looked at, and aimed at as a 30 mm cube.
    import dataclasses

    from mrcf.object_memory import Detection
    from mrcf.policies.parallel_jaw import ParallelJaw

    looks = ParallelJaw(dataclasses.replace(tabletop.skills.POLICIES.spec, height_from_views=True))
    monkeypatch.setattr(tabletop.skills, "POLICIES", looks)
    mem = tabletop.skills.memory()
    (speaker,), _ = mem.observe([Detection(position=(0.34, 0.04), foot_mm=55.0)])
    mem.set_label(speaker.id, "block", "vlm", attributes={"colour": "black"})
    tabletop.skills.known_sizes(mem)
    assert speaker.shape() is None and tabletop.skills.boxed(speaker) is None
    assert tabletop.skills.sizes_by_name() == {}


def test_the_planner_is_told_what_is_wider_than_the_gripper_opens(tabletop):
    from mrcf.object_memory import Detection

    mem = tabletop.skills.memory()
    (speaker, ball), _ = mem.observe([Detection(position=(0.34, 0.04), foot_mm=55.0),
                                      Detection(position=(0.27, 0.12), foot_mm=25.0)])
    mem.set_label(speaker.id, "speaker", "vlm", attributes={"colour": "black"})
    mem.set_label(ball.id, "ball", "vlm", attributes={"colour": "red"})
    widest = tabletop.skills.jaws_open_mm()
    assert widest is not None and 25.0 < widest < 55.0
    assert f"wider across than the gripper opens ({widest:.0f} mm)" in tabletop.skills.describe(speaker)
    assert "wider" not in tabletop.skills.describe(ball)


# ── Outlined again by what it is, before a pick by its wall ───────────────────

def test_a_scrap_of_an_outline_is_replaced_by_the_whole_found_by_name(tabletop, monkeypatch):
    # 2026-11-17: under a lamp turned on, the scan outlined a blue scoop as an 8 mm scrap.
    from mrcf.perception import grounding
    from mrcf.vlm import SceneObject

    skills = tabletop.skills
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    T[:3, 3] = (0.30, 0.0, 0.50)
    model = {"K": np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]), "D": np.zeros(5), "T_world_cam": T,
             "plane_z_m": 0.0}
    asked = {}

    class _Env:
        def camera_model(self, camera, w, h):
            return model

        def render_image(self, camera, width, height):
            return np.zeros((height, width, 3), np.uint8)

    def find(frame, what, **_):
        asked["what"], asked["shape"] = what, frame.shape[:2]
        m = np.zeros(frame.shape[:2], bool)
        m[300:420, 560:740] = True                         # the whole scoop
        return grounding.Found(box["now"], m, 0.93), ""
    box = {"now": (560, 300, 740, 420)}
    monkeypatch.setattr(grounding, "find", find)
    monkeypatch.setattr(skills, "goto_observe_pose", lambda env: True)
    w, h = tabletop.cfg.CAPTURE_W_OVERHEAD, tabletop.cfg.CAPTURE_H_OVERHEAD
    scrap = {tabletop.cfg.OBSERVE_CAMERA: {"size": [w, h], "px": [[640, 360], [645, 360], [642, 365]]}}
    scoop = SceneObject("blue_scoop_obj4", "blue scoop", position=np.array([0.30, 0.0, 0.0]), pixel_xy=[640, 360])
    scoop.geometry = {"outlines": scrap}
    assert skills.outline_again(_Env(), None, scoop) is True
    assert asked["what"] == "the blue scoop" and asked["shape"] == (h, w)
    px = np.asarray(scoop.geometry["outlines"][tabletop.cfg.OBSERVE_CAMERA]["px"])
    assert px[:, 0].min() >= 559 and px[:, 0].max() <= 740 and px[:, 1].min() >= 299
    # and it is where that outline's middle is: (650, 360) px, 10 px off the camera's centre
    assert scoop.position[:2] == pytest.approx((0.30 + 0.010 * 0.5 / 0.9, 0.0), abs=0.002)
    # A box somewhere else is some other blue scoop: the scan's outline stays.
    box["now"] = (100, 100, 200, 200)
    other = SceneObject("blue_scoop", "blue scoop", position=np.array([0.30, 0.0, 0.0]), pixel_xy=[640, 360])
    other.geometry = {"outlines": scrap}
    assert skills.outline_again(_Env(), None, other) is False and other.geometry["outlines"] is scrap


# ── Looking past the arm ──────────────────────────────────────────────────────

def test_a_look_parks_the_arm_only_where_it_hides_the_table(tabletop, monkeypatch):
    skills = tabletop.skills
    parked = []

    class _Env:
        def go_park(self):
            parked.append(True)
            return True

        def go_home(self):                      # looked from first (hiding as much here): then parked
            return True
    for share, spot, should_park in ((0.03, False, False), (0.2, False, True), (0.03, True, True), (None, False, True)):
        parked.clear()
        monkeypatch.setattr(skills, "arm_hides", lambda env, around=None, radius_m=0.05: (share, spot))
        assert skills.goto_observe_pose(_Env(), around=(0.2, 0.0)) is True
        assert bool(parked) is should_park, (share, spot)
    monkeypatch.setattr(tabletop.cfg, "SCAN_PAST_ARM", False)
    monkeypatch.setattr(skills, "arm_hides", lambda env, around=None, radius_m=0.05: (0.0, False))
    parked.clear()
    skills.goto_observe_pose(_Env())
    assert parked                                       # switched off: parked as before



def test_a_tool_run_with_no_deployment_named_takes_the_one_last_launched(monkeypatch, tmp_path):
    """2026-11-19: the SO-101's rest tool checked its path against the chess board after a tidy session."""
    from mrcf import backends
    from tools import _common

    monkeypatch.setattr(sys, "argv", ["tool"])
    monkeypatch.delenv("MRCF_DEPLOYMENT", raising=False)
    monkeypatch.setenv("MRCF_ARM", "so101")
    monkeypatch.setattr(backends.arm("so101").config, "RIG_DIR", tmp_path)
    assert _common.task_config().__name__ == "deployments.chess.config"          # none launched: the arm's own
    (tmp_path / "last_deployment.txt").write_text("tidy\n")
    assert _common.task_config().__name__ == "deployments.tidy.config"           # the one launched last
    (tmp_path / "last_deployment.txt").write_text("gone\n")
    assert _common.task_config().__name__ == "deployments.chess.config"          # one no longer there: ignored
