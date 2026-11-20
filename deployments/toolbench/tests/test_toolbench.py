"""
The toolbench deployment — the UR5 putting hand tools into a box — offline:
one arm, one layout, one task. Its arm is the UR5 whatever MRCF_ARM says; its
rules (profile.py) reach the table task's policies, none of them one that
needs a wrist camera (the UR5 has none); no sizes are written in by hand; its
kept state is its own; and the UR5's block stacking (deployments/tabletop)
gets none of its rules.
"""

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")          # the table task's naming crops with it

HERE = Path(__file__).resolve().parents[1]


def _drop(prefixes=("deployments.toolbench", "deployments.tabletop")):
    # The deployments only: imported again, a config binds the table task anew
    # (tabletop.use), and the task's skills are made again for it.
    for name in [m for m in sys.modules if m in prefixes or m.startswith(tuple(p + "." for p in prefixes))]:
        del sys.modules[name]


@pytest.fixture
def bound(monkeypatch):
    """Import a table deployment fresh, its skills registered; the registry and the manipulation hooks put back."""
    from mrcf import manipulation
    from mrcf.action_registry import ACTIONS

    saved_actions = dict(ACTIONS)
    saved_install = (manipulation._policies, manipulation._vision, manipulation._scenario)
    _drop()

    def bind(name):
        importlib.import_module(f"deployments.{name}.skills")
        return importlib.import_module(f"deployments.{name}.config"), importlib.import_module("mrcf.tabletop.skills")
    try:
        yield bind
    finally:
        ACTIONS.clear()
        ACTIONS.update(saved_actions)
        manipulation._policies, manipulation._vision, manipulation._scenario = saved_install
        _drop()


def test_its_arm_is_the_ur5_whatever_mrcf_arm_says(bound, monkeypatch):
    from mrcf import backends

    monkeypatch.setenv("MRCF_ARM", "so101")
    cfg, skills = bound("toolbench")
    assert cfg.ARM_NAME == "ur5" and cfg.ARM is backends.arm("ur5")
    assert skills.cfg.ARM is cfg.ARM


def test_its_rules_pick_tools_by_handle_and_across_and_put_them_into_a_container(bound):
    cfg, skills = bound("toolbench")
    spec = skills.POLICIES.spec
    assert spec.lying_pick and spec.handle_first and spec.close_across_long and spec.place_in
    assert not spec.grip_low_from_above                       # only within side looks: never runs on the UR5
    assert skills.containers_on() and "place_in" in skills.ACTIONS
    assert cfg.DEFER_FAILED and cfg.RELOCATE_AFTER_MISS
    from mrcf import manipulation
    assert manipulation._vision.relocate is not None          # a miss is found again by an overhead look
    assert spec.max_open_m == pytest.approx(cfg.ARM.config.GRIPPER_MAX_OPEN_M)   # the Hand-E's own


def test_none_of_its_rules_needs_a_wrist_camera(bound):
    cfg, skills = bound("toolbench")
    assert cfg.ARM.config.CAMERA_NAME is None
    spec = skills.POLICIES.spec
    assert not (spec.height_from_views or spec.approach_look or spec.wrist_at_grasp or spec.held_by_camera
                or spec.held_while_still)
    assert not (cfg.HELD_BY_MODEL or cfg.DIAGNOSE_MISSES)


def test_no_sizes_are_written_in(bound):
    _, skills = bound("toolbench")
    assert skills.sizes_by_name() == {}                       # not the letter blocks' 30 mm


def test_its_kept_state_is_its_own_and_the_rigs_camera_photo_is_shared(bound):
    cfg, _ = bound("toolbench")
    rig = Path(cfg.ARM.config.RIG_DIR)
    for setting, kind in (("SURFACE_PATH", "surface"), ("MEMORY_PATH", "memory"), ("PLACES_PATH", "places"),
                          ("CONTAINERS_PATH", "containers"), ("CATALOGUE_PATH", "catalogue")):
        assert getattr(cfg, setting) == rig / f"toolbench_{kind}.json"
    assert Path(cfg.SURFACE_REFERENCE_PATH).parent == rig      # one table, one camera: the rig's


def test_the_ur5s_block_stacking_gets_none_of_its_rules(bound):
    cfg, skills = bound("tabletop")
    assert cfg.ARM_NAME == "ur5" and cfg.PROFILE is None
    spec = skills.POLICIES.spec
    assert not (spec.place_in or spec.handle_first or spec.close_across_long or spec.lying_pick)
    assert skills.sizes_by_name() == {"block": (30.0, 30.0)}


def test_run_py_offers_it_on_the_ur5_and_launches_the_table_controller(bound):
    import run

    ok, why, _ = run._validate(HERE)
    assert ok, why
    assert run._deployment_arm(HERE) == "ur5"
    controller = importlib.import_module("deployments.toolbench.llm_controller")
    assert controller.main.__module__ == "deployments.toolbench.llm_controller"
    from mrcf.tabletop import controller as table
    assert "toolbench" in table.DEPLOYMENT.name


def _screwdriver_track():
    from types import SimpleNamespace
    return SimpleNamespace(label="screwdriver", attributes={"colour": "orange"}, height_mm=None, feet_mm=[73.0],
                           extra={"length_mm": 200.0, "widest_mm": 28.0, "narrowest_mm": 6.0})


def test_a_tool_is_told_to_the_planner_by_how_narrow_it_is_across_not_its_round_foot(bound):
    """2026-11-20 14:36: the screwdriver, "73 mm across" by its foot, was too wide for the 52 mm jaws: nothing picked."""
    _, skills = bound("toolbench")
    text = skills.describe(_screwdriver_track())
    assert "lying along the table, 200 mm long, 6 mm across at its narrowest" in text
    assert "cannot be picked up" not in text and "73 mm" not in text


def test_the_ur5s_block_stacking_still_describes_by_the_foot(bound):
    _, skills = bound("tabletop")
    text = skills.describe(_screwdriver_track())
    assert "73 mm across" in text and "cannot be picked up" in text


def test_its_firmer_squeeze_reaches_the_ur5s_gripper(monkeypatch):
    from deployments.toolbench import config
    from mrcf.backends import ur5

    seen = {}
    real = ur5.make_arm
    monkeypatch.setattr(ur5, "make_arm", lambda host, gripper, rc=None: seen.setdefault("rc", rc) and real(host, gripper, rc))
    ur5.make_env(profile=config.PROFILE, host=None)
    assert seen["rc"].ROBOTIQ_FORCE == 40 and ur5.config.ROBOTIQ_FORCE == 10      # its rule; the arm's fact unchanged


def _tool_table(skills, monkeypatch, box_kept=True):
    """Today's table (2026-11-20 15:08) as the scan named it: a hammer, two screwdrivers, a tape measure, the box."""
    import numpy as np

    from mrcf.object_memory import Detection, ObjectMemory
    from mrcf.vlm import SceneObject, SceneState

    mem = ObjectMemory(load=False) if "load" in ObjectMemory.__init__.__code__.co_varnames else ObjectMemory()
    monkeypatch.setattr(skills, "memory", lambda: mem)
    things = [((0.314, 0.348), 51.0, "silver hammer", "silver_hammer"),
              ((0.468, 0.413), 20.0, "red screwdriver", "red_screwdriver"),
              ((0.232, 0.385), 95.0, "red screwdriver", "red_screwdriver_obj8"),
              ((0.155, 0.282), 62.0, "yellow tape measure", "yellow_tape_measure")]
    tracks = mem.observe([Detection(position=p, foot_mm=f, label=n, label_source="vlm") for p, f, n, _ in things])[0]
    state = SceneState()
    state.memory = mem
    for t, (_, _, _, name) in zip(tracks, things):
        o = SceneObject(name, name, interactable=True, position=np.array([*t.position, 0.0]))
        o.track_id, o.geometry = t.id, {"base_diameter_m": t.foot_mm / 1000.0}
        state.objects.append(o)
    box = SceneObject("brown_box", "a box", interactable=False, position=np.array([0.55, 0.30, 0.0]))
    if box_kept:
        state.targets = [box]
    looked = []

    def find_container(env, st, what, park=True):
        looked.append(what)
        st.targets.append(box)
        return True
    monkeypatch.setattr(skills, "find_container", find_container)
    monkeypatch.setattr(skills, "_keeper", lambda: type("K", (), {"holds": lambda self, st, o: None})())
    monkeypatch.setattr(skills, "look_for", lambda env, st, name: False)
    return state, looked


def test_its_task_puts_the_hammer_and_every_screwdriver_into_the_box_with_no_planner(bound, monkeypatch):
    """User 2026-11-20: "lets maybe consider also allow a non planner method - so we pre plan that the tools go into the box"."""
    cfg, skills = bound("toolbench")
    assert cfg.PLANNER is False and "put the tools away" in cfg.TASKS
    importlib.import_module("deployments.toolbench.llm_controller")
    deployment = importlib.import_module("mrcf.tabletop.controller").DEPLOYMENT
    assert deployment.task_plan is skills.task_plan and deployment.planner is False
    state, _ = _tool_table(skills, monkeypatch)
    for words in ("put the tools away", "go"):
        plan = skills.task_plan(words, None, state)
        picked = [a["parameters"]["object_name"] for a in plan if a["action"] != "place_in"]
        assert sorted(picked) == ["red_screwdriver", "red_screwdriver_obj8", "silver_hammer", "yellow_tape_measure"]
        assert {a["parameters"]["container"] for a in plan if a["action"] == "place_in"} == {"brown_box"}
        assert all(a["action"] == "pick" for a in plan if a["action"] != "place_in")     # the lying pick, each


def test_on_its_first_run_the_box_is_looked_for_by_name(bound, monkeypatch):
    _, skills = bound("toolbench")
    state, looked = _tool_table(skills, monkeypatch, box_kept=False)
    plan = skills.task_plan("go", object(), state)
    assert looked == ["box"] and plan and plan[1] == {"action": "place_in", "parameters": {"container": "brown_box"}}


def test_an_unmeasured_containers_walls_are_taken_as_tall_as_its_box_may_be(bound):
    cfg, skills = bound("toolbench")
    assert cfg.CONTAINER_TALLEST_M == pytest.approx(0.30) and cfg.TALLEST_OBJECT_M == pytest.approx(0.15)
    assert skills._keeper().tallest_m == pytest.approx(0.30)


def test_once_its_task_is_done_it_looks_again_for_more_of_what_it_names(bound, monkeypatch):
    """2026-11-20 17:41: a screwdriver lying along the black tape, unseen by the first look, left on the table."""
    _, skills = bound("toolbench")
    state, _ = _tool_table(skills, monkeypatch)
    assert skills.task_plan("go", None, state)
    state.objects = [o for o in state.objects if o.name not in ("silver_hammer", "red_screwdriver")]  # in the box
    more = skills.task_more(None, state, ["silver_hammer", "red_screwdriver"])
    picked = [a["parameters"]["object_name"] for a in more if a["action"] == "pick"]
    assert sorted(picked) == ["red_screwdriver_obj8", "yellow_tape_measure"]


def test_a_second_one_found_by_the_same_name_as_one_done_is_done_too(bound, monkeypatch):
    """2026-11-20 17:51: the second screwdriver, found again as "screwdriver", was left out as done."""
    _, skills = bound("toolbench")
    state, _ = _tool_table(skills, monkeypatch)
    assert skills.task_plan("go", None, state)
    state.objects = [o for o in state.objects if o.name not in ("silver_hammer", "red_screwdriver")]  # in the box
    for o in state.objects:
        if o.name == "red_screwdriver_obj8":
            o.name = "red_screwdriver"                     # found again under the done one's name
    more = skills.task_more(None, state, ["silver_hammer", "red_screwdriver"])
    picked = [a["parameters"]["object_name"] for a in more if a["action"] == "pick"]
    assert any(p.startswith("red_screwdriver") for p in picked)
