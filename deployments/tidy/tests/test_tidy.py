"""
The tidy deployment — the SO-101 putting things away into a tub — offline: it
is one arm, one layout and one task. Its arm is the SO-101 whatever MRCF_ARM
says, its rules (profile.py) reach the table task's policies, its kept state
is its own, and the UR5's deployment of the same task gets none of its rules.
"""

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")          # the table task's naming crops with it

HERE = Path(__file__).resolve().parents[1]


def _drop(prefixes=("deployments.tidy", "deployments.tabletop")):
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


def test_its_arm_is_the_so101_whatever_mrcf_arm_says(bound, monkeypatch):
    from mrcf import backends

    monkeypatch.setenv("MRCF_ARM", "ur5")
    cfg, skills = bound("tidy")
    assert cfg.ARM_NAME == "so101" and cfg.ARM is backends.arm("so101")
    assert skills.cfg.ARM is cfg.ARM


def test_its_rules_drive_its_arm_and_the_rest_are_the_arms(bound):
    cfg, skills = bound("tidy")
    spec = skills.POLICIES.spec
    assert spec.place_in and spec.rim_pick and spec.height_from_views and spec.held_by_camera
    assert cfg.GRIPPER_HOLD_PCT == 10.0                                  # its profile's
    assert cfg.GRIPPER_MAX_OPEN_M == cfg.ARM.config.GRIPPER_MAX_OPEN_M   # the arm's own: not chess's 45
    assert spec.max_open_m == pytest.approx(cfg.ARM.config.GRIPPER_MAX_OPEN_M)
    assert skills.containers_on() and "place_in" in skills.ACTIONS
    assert cfg.COLOURS[0] == "red"                                       # the table task's default
    assert skills.sizes_by_name() == {}                                  # it measures heights; no letter blocks


def test_the_same_task_on_the_ur5_gets_none_of_its_rules(bound):
    cfg, skills = bound("tabletop")
    assert cfg.ARM_NAME == "ur5" and cfg.PROFILE is None
    spec = skills.POLICIES.spec
    assert not (spec.place_in or spec.rim_pick or spec.height_from_views or spec.held_by_camera)
    assert not skills.containers_on()
    assert skills.sizes_by_name() == {"block": (30.0, 30.0)}              # its letter blocks


def test_its_kept_state_is_its_own(bound):
    cfg, _ = bound("tidy")
    rig = Path(cfg.ARM.config.RIG_DIR)
    for setting, kind in (("SURFACE_PATH", "surface"), ("MEMORY_PATH", "memory"), ("PLACES_PATH", "places"),
                          ("CONTAINERS_PATH", "containers")):
        assert getattr(cfg, setting) == rig / f"tidy_{kind}.json"


def test_run_py_offers_it_on_the_so101_and_launches_the_table_controller(bound):
    import run

    ok, why, _ = run._validate(HERE)
    assert ok, why
    assert run._deployment_arm(HERE) == "so101"
    controller = importlib.import_module("deployments.tidy.llm_controller")
    assert controller.main.__module__ == "deployments.tidy.llm_controller"
    from mrcf.tabletop import controller as table
    assert table.DEPLOYMENT.name == "SO-101 tidy (real hardware)"


def test_its_grip_counts_as_a_miss_only_when_the_jaws_never_closed(bound):
    cfg, skills = bound("tidy")
    # The arm's own 30 mm is chess's: here a plug is gripped at 55 mm (2026-11-18 16:08).
    assert skills.POLICIES.spec.miss_width_m == pytest.approx(cfg.ARM.config.GRIPPER_MAX_OPEN_M - 0.002)
    assert cfg.ARM.config.GRASP_MISS_WIDTH_M == pytest.approx(0.030)


def test_it_keeps_a_catalogue_of_its_kinds_and_the_ur5s_deployment_does_not(bound):
    cfg, skills = bound("tidy")
    assert cfg.CATALOGUE and cfg.CATALOGUE_PATH == Path(cfg.ARM.config.RIG_DIR) / "tidy_catalogue.json"
    from mrcf import manipulation
    assert manipulation._scenario.recipe is not None and manipulation._scenario.picked is not None
    cfg, skills = bound("tabletop")
    assert not cfg.CATALOGUE and skills.catalogue() is None


def test_a_plan_kept_for_its_words_runs_again_only_while_its_objects_are_all_out(bound, monkeypatch, tmp_path):
    """User 2026-11-19: "its the same objects in the same context - so it shouldn't be wasting time"."""
    import numpy as np

    from mrcf.vlm import SceneObject, SceneState

    from mrcf import skill_executor

    cfg, skills = bound("tidy")
    assert cfg.PLAN_MEMORY and cfg.HOME_AFTER_COMMAND and not skill_executor.DESCRIBE_FAILURES
    monkeypatch.setattr(cfg, "PLANS_PATH", tmp_path / "plans.json")
    plan = [{"action": "pick", "parameters": {"object_name": "red_cap"}},
            {"action": "place_in", "parameters": {"container": "plastic_tub"}}]
    skills.plan_made("Put the red cap into the plastic tub!", plan)
    state = SceneState()
    state.objects = [SceneObject("red_cap", "a red cap", interactable=True, position=np.array([0.2, 0.03, 0.0]))]
    state.targets = [SceneObject("plastic_tub", "a tub", interactable=False, position=np.array([0.2, -0.15, 0.0]))]
    inside: list = []
    monkeypatch.setattr(skills, "_keeper", lambda: type("K", (), {"holds": lambda self, st, o: inside[0] if inside
                                                                  else None})())
    assert skills.remembered_plan("put the red cap into the plastic tub", state) == plan       # same words
    assert skills.remembered_plan("put the red cap on the shelf", state) is None
    inside.append("plastic_tub")
    assert skills.remembered_plan("put the red cap into the plastic tub", state) is None       # in already
    inside.clear()
    state.objects[0].position = None
    assert skills.remembered_plan("put the red cap into the plastic tub", state) is None       # held
    cfg, skills = bound("tabletop")
    assert not cfg.PLAN_MEMORY and not cfg.HOME_AFTER_COMMAND and skill_executor.DESCRIBE_FAILURES


class _Arm:
    """Where the arm goes for a look: home and the park pose, counted."""

    def __init__(self):
        self.went = []

    def go_home(self):
        self.went.append("home")
        return True

    def go_park(self):
        self.went.append("park")
        return True


def test_a_look_the_arm_is_in_the_way_of_is_taken_from_home_before_the_park_pose(bound, monkeypatch):
    cfg, skills = bound("tidy")
    hidden = iter([(0.20, True), (0.05, False)])                  # over the spot; then home, clear of it
    monkeypatch.setattr(skills, "arm_hides", lambda env, around=None: next(hidden))
    arm = _Arm()
    assert skills.goto_observe_pose(arm, around=(0.3, 0.1)) is True
    assert arm.went == ["home"]
    hidden = iter([(0.20, True), (0.20, True)])                   # home hides it too: parked
    monkeypatch.setattr(skills, "arm_hides", lambda env, around=None: next(hidden))
    arm = _Arm()
    skills.goto_observe_pose(arm, around=(0.3, 0.1))
    assert arm.went == ["home", "park"]


def test_what_a_pick_shows_is_learned_and_its_twin_is_picked_by_it(bound, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from mrcf.catalogue import Catalogue
    from mrcf.object_memory import Detection, ObjectMemory
    from mrcf.policies.base import Diagnosis, Outcome

    cfg, skills = bound("tidy")
    mem, cat = ObjectMemory(), Catalogue(tmp_path / "cat.json")
    monkeypatch.setattr(skills, "memory", lambda: mem)
    monkeypatch.setattr(skills, "_CATALOGUE", [cat])
    figure = {"foot_mm": 22.0, "long_mm": 24.0, "short_mm": 22.0, "L": 60.0, "a": 30.0, "b": 20.0}
    one, two, plug = mem.observe([Detection(position=(0.2, -0.1), foot_mm=22.0, label="bottle cap", label_source="vlm"),
                                  Detection(position=(0.33, 0.07), foot_mm=22.5, label="ball", label_source="vlm"),
                                  Detection(position=(0.2, 0.15), foot_mm=51.0, label="plug", label_source="vlm")])[0]
    one.extra, two.extra = {"sig": figure}, {"sig": dict(figure, foot_mm=22.8, a=27.0)}
    plug.extra = {"sig": {"foot_mm": 51.0, "long_mm": 55.0, "short_mm": 47.0, "L": 226.0, "a": 4.0, "b": -11.0}}
    obj = lambda t: SimpleNamespace(name=t.label, track_id=t.id)               # noqa: E731
    state = SimpleNamespace(memory=mem)
    # The first figure: a close over it, asked why ("too high"), then held 16 mm up.
    skills._picked_hook(obj(one), None, Outcome(True, "grasped", grasp=SimpleNamespace(height_m=0.06), attempts=[
        {"attempt": 1, "result": "closed_on_nothing", "grip_m": 0.033, "diagnosis": "too_high"},
        {"attempt": 2, "result": "ok", "grip_m": 0.016}]))
    assert len(cat.kinds) == 1 and cat.kinds[0].name == "bottle cap"
    grips = {g["grip_at_m"]: (g["held"], g["missed"]) for g in cat.kinds[0].pose("upright").grips}
    assert grips == {0.033: (0, 1), 0.016: (1, 0)}
    # The other figure, named something else by the vision model, is of that kind: picked by that grip.
    got = skills._recipe(state, obj(two))
    assert got["grip_at_m"] == pytest.approx(0.016) and got["max_height_m"] == pytest.approx(0.07)   # top + margin
    # The jaws opened past what its kind is across (RECIPE_OPENS_TO_KIND): 2026-11-20, the plug's foot read 44
    # mm from above, its kind 59 across, and one jaw came down on it.
    assert cfg.RECIPE_OPENS_TO_KIND and got["across_m"] == pytest.approx(0.022)
    monkeypatch.setattr(cfg, "RECIPE_OPENS_TO_KIND", False)
    assert "across_m" not in skills._recipe(state, obj(two))
    monkeypatch.setattr(cfg, "RECIPE_OPENS_TO_KIND", True)
    assert skills._recipe(state, obj(plug)) is None                         # a first meeting
    # Picked by that recipe again (nothing measured): its top stays 60 mm, not the 70 it was planned by —
    # 2026-11-19, the plug's crept 72, 82 ... 112 mm, 10 mm a reuse.
    for _ in range(3):
        skills._picked_hook(obj(two), None, Outcome(True, "grasped", grasp=SimpleNamespace(height_m=0.07), attempts=[
            {"attempt": 1, "result": "ok", "grip_m": 0.016}]))
    assert skills._recipe(state, obj(two))["max_height_m"] == pytest.approx(0.07)
    # A miss whose cause was its aim says nothing about its grip; one found lying is learned as lying.
    skills._picked_hook(obj(plug), None, Outcome(False, "missed", grasp=SimpleNamespace(height_m=0.07), attempts=[
        {"attempt": 1, "result": "closed_on_nothing", "grip_m": 0.021, "diagnosis": "aim_off"}]))
    assert len(cat.kinds) == 1                                              # not learned: never held, not lying
    skills._picked_hook(obj(two), None, Outcome(False, "lying", diagnosis=Diagnosis("lying", "on its side"), attempts=[
        {"attempt": 1, "result": "closed_on_nothing", "grip_m": 0.016, "diagnosis": "lying"}]))
    assert cat.kinds[0].pose("lying") is not None and two.extra["pose"] == "lying"


def test_a_kind_known_resting_a_way_not_met_before_is_gripped_low_when_its_longest_side_lets_it(bound, monkeypatch,
                                                                                                 tmp_path):
    """User 2026-11-19: "the plug should be liftable in any orientation"."""
    from types import SimpleNamespace

    from mrcf.catalogue import Catalogue
    from mrcf.object_memory import Detection, ObjectMemory

    cfg, skills = bound("tidy")
    mem, cat = ObjectMemory(), Catalogue(tmp_path / "cat.json")
    monkeypatch.setattr(skills, "memory", lambda: mem)
    monkeypatch.setattr(skills, "_CATALOGUE", [cat])
    plug_sig = {"foot_mm": 55.0, "long_mm": 62.0, "short_mm": 59.0, "L": 241.0, "a": -2.0, "b": -6.0}
    plug, _ = cat.add("plug", "white", plug_sig)
    cat.learn(plug, "upright", plug_sig, grip_at_m=0.0177, top_m=0.072, held=True)
    scoop_sig = {"foot_mm": 91.0, "long_mm": 126.0, "short_mm": 72.0, "L": 90.0, "a": 10.0, "b": -50.0}
    scoop, _ = cat.add("scoop", "blue", scoop_sig)
    cat.learn(scoop, "upright", scoop_sig, grip_at_m=0.031, top_m=0.06, held=True)
    on_back, upside_down = mem.observe([
        Detection(position=(0.16, 0.07), foot_mm=42.0, label="plug", label_source="vlm", attributes={"colour": "white"}),
        Detection(position=(0.25, 0.10), foot_mm=60.0, label="scoop", label_source="vlm",
                  attributes={"colour": "blue"})])[0]
    on_back.extra = {"sig": dict(plug_sig, foot_mm=42.0, long_mm=45.0, short_mm=40.0)}       # not its outline
    upside_down.extra = {"sig": dict(scoop_sig, foot_mm=60.0, long_mm=70.0, short_mm=65.0)}
    state = SimpleNamespace(memory=mem)
    got = skills._recipe(state, SimpleNamespace(name="white_plug", track_id=on_back.id))
    assert got["grip_at_m"] == pytest.approx(0.0177) and got["max_height_m"] == pytest.approx(0.062 + 0.010)
    assert "not met before" in got["said"]
    # 126 mm long: resting any way it may stand taller than the gripper clears gripping low — no recipe.
    assert skills._recipe(state, SimpleNamespace(name="blue_scoop", track_id=upside_down.id)) is None


def test_outlined_whole_again_a_fragment_of_a_figure_is_the_figure_its_catalogue_knows(bound, monkeypatch, tmp_path):
    """2026-11-19 14:03: a figure the scan outlined as 16 mm of its 23 matched nothing and was gripped off its middle."""
    from types import SimpleNamespace

    import numpy as np

    from mrcf.catalogue import Catalogue
    from mrcf.object_memory import Detection, ObjectMemory

    cfg, skills = bound("tidy")
    assert cfg.OUTLINE_BEFORE_PICK
    mem, cat = ObjectMemory(), Catalogue(tmp_path / "cat.json")
    monkeypatch.setattr(skills, "_CATALOGUE", [cat])
    figure = {"foot_mm": 22.0, "long_mm": 24.0, "short_mm": 22.0, "L": 60.0, "a": 30.0, "b": 20.0}
    kind, _ = cat.add("cap", "red", figure)
    (track,), _ = mem.observe([Detection(position=(0.2, 0.17), foot_mm=16.0, label="red bottle cap",
                                         label_source="vlm")])
    track.extra = {"sig": dict(figure, foot_mm=16.0, long_mm=17.0, short_mm=15.0)}
    assert cat.match(track.extra["sig"]) is None                                # the fragment: nothing
    mask = np.zeros((100, 100), np.uint8)
    mask[30:54, 40:63] = 1                                                      # the whole of it, 24 x 23

    class _View:
        def to_plane(self, px):
            return np.asarray(px, float)

    skills._resign(SimpleNamespace(memory=mem), SimpleNamespace(name="red_bottle_cap", track_id=track.id), mask,
                   _View())
    assert track.extra["kind"] == kind.id and track.extra["sig"]["foot_mm"] >= 20


def test_what_the_scan_outlined_inside_an_object_outlined_whole_is_part_of_it(bound):
    """2026-11-19: a plug on its side outlined as its body and, apart, its pins — both "white block"."""
    import numpy as np

    from mrcf.vlm import SceneObject, SceneState

    cfg, skills = bound("tidy")
    body = SceneObject("white_block_obj7", "body", interactable=True, position=np.array([0.26, 0.12, 0.0]),
                       pixel_xy=[40, 40])
    pins = SceneObject("white_block", "pins", interactable=True, position=np.array([0.29, 0.13, 0.0]),
                       pixel_xy=[60, 42])
    away = SceneObject("red_ball", "a figure", interactable=True, position=np.array([0.33, 0.03, 0.0]),
                       pixel_xy=[90, 90])
    state = SceneState()
    state.objects = [body, pins, away]
    mask = np.zeros((100, 100), bool)
    mask[30:50, 30:70] = True                                         # the whole plug, pins and all
    skills._fold_parts(state, body, mask, 100, 100)
    assert [o.name for o in state.objects] == ["white_block_obj7", "red_ball"]


def test_put_these_into_the_tub_is_planned_from_the_words_each_found_by_what_the_user_calls_it(bound, monkeypatch,
                                                                                                tmp_path):
    """2026-11-19 14:25: the plug looked for as "silver cylinder"; the re-plan timed out and came back cut off."""
    import numpy as np

    from mrcf.catalogue import Catalogue
    from mrcf.object_memory import Detection, ObjectMemory
    from mrcf.vlm import SceneObject, SceneState

    cfg, skills = bound("tidy")
    assert cfg.QUICK_PUT_INTO
    assert importlib.import_module("mrcf.tabletop.controller").DEPLOYMENT.quick_plan is skills.quick_plan
    mem, cat = ObjectMemory(), Catalogue(tmp_path / "cat.json")
    monkeypatch.setattr(skills, "memory", lambda: mem)
    monkeypatch.setattr(skills, "_CATALOGUE", [cat])
    scoop, _ = cat.add("scoop", "blue", {"foot_mm": 91.0, "long_mm": 126.0, "short_mm": 72.0, "L": 90, "a": 10, "b": -50})
    scoop.picked_by = "pick_by_rim"
    tracks = mem.observe([Detection(position=p, foot_mm=f, label=n, label_source="vlm") for p, f, n in
                          [((0.17, 0.16), 23.0, "red cap"), ((0.33, 0.03), 21.0, "red ball"),
                           ((0.24, 0.05), 64.0, "blue scoop"), ((0.26, 0.12), 26.0, "silver cylinder")]])[0]
    tracks[2].extra = {"kind": scoop.id}
    state = SceneState()
    state.memory = mem
    for t, name in zip(tracks, ["red_cap", "red_ball", "blue_scoop", "silver_cylinder"]):
        o = SceneObject(name, name, interactable=True, position=np.array([*t.position, 0.0]))
        o.track_id, o.geometry = t.id, {"base_diameter_m": t.foot_mm / 1000.0}
        state.objects.append(o)
    state.targets = [SceneObject("plastic_tub", "a tub", interactable=False, position=np.array([0.2, -0.15, 0.0]))]
    monkeypatch.setattr(skills, "_keeper", lambda: type("K", (), {"holds": lambda self, st, o: None})())
    looked = []

    def look_for(env, st, name):                                            # found whole by the user's words
        looked.append(name)
        st.objects = [o for o in st.objects if o.name != "silver_cylinder"]
        st.objects.append(SceneObject(name, "found by name", interactable=True, position=np.array([0.27, 0.12, 0.0])))
        return True
    monkeypatch.setattr(skills, "look_for", look_for)
    plan = skills.quick_plan("put the red cap, the red ball, the white plug and the blue scoop into the plastic tub",
                             None, state)
    assert looked == ["white_plug"]
    assert [(a["action"], a["parameters"].get("object_name") or a["parameters"].get("container")) for a in plan] == [
        ("pick", "red_cap"), ("place_in", "plastic_tub"), ("pick", "red_ball"), ("place_in", "plastic_tub"),
        ("pick", "white_plug"), ("place_in", "plastic_tub"), ("pick_by_rim", "blue_scoop"), ("place_in", "plastic_tub")]
    assert skills.quick_plan("stack the blocks", None, state) is None            # anything else: the planner
    # Two called alike: the one the catalogue knows as that kind, not the power strip off the table's edge.
    plug, _ = cat.add("plug", "white", {"foot_mm": 55.0, "long_mm": 62.0, "short_mm": 59.0, "L": 241, "a": -2, "b": -6})
    strip, body = mem.observe([Detection(position=(0.20, 0.34), foot_mm=40.0, label="plug", label_source="vlm"),
                               Detection(position=(0.24, 0.13), foot_mm=55.0, label="plug", label_source="vlm")])[0]
    body.extra = {"kind": plug.id}
    for t, name in ((strip, "white_plug"), (body, "white_plug_obj3")):
        o = SceneObject(name, name, interactable=True, position=np.array([*t.position, 0.0]))
        o.track_id = t.id
        state.objects.append(o)
    assert skills._thing_called(state, "the white plug", set()).name == "white_plug_obj3"
    assert skills.quick_plan("put the red cap on the shelf", None, state) is None
    cfg, skills = bound("tabletop")
    assert not cfg.QUICK_PUT_INTO                                               # the UR5's: planned, as before
    importlib.import_module("deployments.tabletop.llm_controller")
    assert importlib.import_module("mrcf.tabletop.controller").DEPLOYMENT.quick_plan is None


def test_only_an_outline_that_looks_wrong_is_looked_at_again_before_its_pick(bound, monkeypatch, tmp_path):
    """2026-11-19 (filmed): the re-outline before every pick sent the arm home and up to its park pose."""
    import numpy as np

    from mrcf.catalogue import Catalogue
    from mrcf.object_memory import Detection, ObjectMemory
    from mrcf.vlm import SceneObject, SceneState

    cfg, skills = bound("tidy")
    mem, cat = ObjectMemory(), Catalogue(tmp_path / "cat.json")
    monkeypatch.setattr(skills, "_CATALOGUE", [cat])
    figure = {"foot_mm": 22.0, "long_mm": 24.0, "short_mm": 22.0, "L": 60.0, "a": 30.0, "b": 20.0}
    kind, pose = cat.add("cap", "red", figure)
    known, stranger, fragment = mem.observe([Detection(position=p, foot_mm=f, label="x", label_source="vlm")
                                             for p, f in (((0.20, 0.10), 23.0), ((0.30, 0.0), 40.0),
                                                          ((0.25, 0.18), 12.0))])[0]
    known.extra = fragment.extra = {"kind": kind.id, "pose": pose.name}
    state = SceneState()
    state.memory = mem
    objs = []
    for t in (known, stranger, fragment):
        o = SceneObject(t.id, "x", interactable=True, position=np.array([*t.position, 0.0]))
        o.track_id, o.geometry = t.id, {"base_diameter_m": t.foot_mm / 1000.0}
        objs.append(o)
    state.objects = list(objs)
    assert skills._outline_suspect(state, objs[0]) is False          # whole, known: picked as the scan saw it
    assert skills._outline_suspect(state, objs[1]) is True           # its kind not known
    assert skills._outline_suspect(state, objs[2]) is True           # 12 mm of a 22 mm kind: a fragment
    part = SceneObject("part", "x", interactable=True, position=np.array([0.21, 0.10, 0.0]))
    state.objects.append(part)
    assert skills._outline_suspect(state, objs[0]) is True           # another outline against it: split


def test_a_snapshot_keeps_a_picture_from_each_camera_and_says_where(bound, monkeypatch, capsys, tmp_path):
    import numpy as np

    from mrcf import flight_recorder

    cfg, skills = bound("tidy")
    kept = []
    monkeypatch.setattr(flight_recorder, "keep_picture", lambda stem, img: kept.append(stem) or tmp_path / f"{stem}.jpg")

    class _Cams:
        def render_image(self, cam, width=None, height=None):
            return np.zeros((height or 48, width or 64, 3), np.uint8)
    assert skills.skill_snapshot(_Cams()) is True
    assert kept == [f"snapshot_{cfg.OBSERVE_CAMERA}", f"snapshot_{skills.POLICIES.spec.camera}"]
    assert "[Snapshot]" in capsys.readouterr().out


def test_a_sighting_of_a_kind_learned_is_named_by_it_and_the_vision_model_not_asked(bound, monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import numpy as np

    from mrcf import catalogue as catalogue_mod
    from mrcf.object_memory import Detection, ObjectMemory
    from mrcf.perception.plane_geometry import PlaneView

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
    from test_place_in import down_camera

    cfg, skills = bound("tidy")
    model = down_camera([0.25, 0.05, 0.6])
    env = SimpleNamespace(camera_model=lambda cam, w, h: model)
    mask = np.zeros((480, 640), np.uint8)
    mask[200:230, 300:330] = 1
    colour = {"L": 60.0, "a": 30.0, "b": 20.0}
    cat = catalogue_mod.Catalogue(tmp_path / "cat.json")
    sig = catalogue_mod.signature(mask, PlaneView.from_camera_model(model), 22.0, colour)
    kind, _ = cat.add("wooden figure", "red", sig)
    mem = ObjectMemory()
    track = mem.observe([Detection(position=(0.25, 0.05), foot_mm=22.0)])[0][0]
    found = [{"mask": mask, "measurement": SimpleNamespace(base_radius_mm=11.0), "appearance": colour}]
    skills._match_kinds(cat, env, np.zeros((480, 640, 3), np.uint8), found, [track], mem)
    assert track.label == "wooden figure" and track.extra["kind"] == kind.id and track.extra["pose"] == "upright"
    assert track.label_source == "catalogue"                 # named: the vision model skips it


def test_an_object_the_scan_missed_is_found_by_name_and_its_parts_folded_into_it(bound, monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import numpy as np

    from mrcf.object_memory import ObjectMemory
    from mrcf.perception import grounding
    from mrcf.vlm import SceneObject, SceneState

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
    from test_place_in import down_camera

    cfg, skills = bound("tidy")
    model = down_camera([0.25, 0.10, 0.6])
    mem = ObjectMemory()
    monkeypatch.setattr(skills, "memory", lambda: mem)
    monkeypatch.setattr(skills, "goto_observe_pose", lambda env, around=None: True)
    monkeypatch.setattr(skills, "_overhead", lambda env: (np.zeros((480, 640, 3), np.uint8), model))
    mask = np.zeros((480, 640), bool)
    mask[220:260, 300:350] = True                                  # the plug, as the vision model boxed it
    monkeypatch.setattr(grounding, "find", lambda frame, what, **k: (
        SimpleNamespace(mask=mask, box=(300, 220, 350, 260), score=0.9), "") if "plug" in what else (None, "none"))
    pin = SceneObject("silver_screw", "a pin", position=np.array([0.25, 0.10, 0.0]), pixel_xy=[320, 240])
    other = SceneObject("blue_scoop", "a scoop", position=np.array([0.32, 0.14, 0.0]), pixel_xy=[100, 100])
    state = SceneState(objects=[pin, other])
    assert skills.look_for(None, state, "white_plug") is True
    names = [o.name for o in state.objects]
    assert "white_plug" in names and "silver_screw" not in names and "blue_scoop" in names
    plug = state.find("white_plug")
    assert plug.track_id and mem.get(plug.track_id).label == "white plug" and "sig" in mem.get(plug.track_id).extra
    assert plug.geometry["outlines"][cfg.OBSERVE_CAMERA]["px"]
    assert skills.look_for(None, state, "red_teapot") is False
    from mrcf.tabletop import controller as table
    assert table.DEPLOYMENT.look_around is skills.look_for


def _tidy_table(skills, monkeypatch, tmp_path):
    """Three of tidy's four on the table, the tub kept; the white plug not to be found."""
    import numpy as np

    from mrcf.catalogue import Catalogue
    from mrcf.object_memory import Detection, ObjectMemory
    from mrcf.vlm import SceneObject, SceneState

    mem, cat = ObjectMemory(), Catalogue(tmp_path / "cat.json")
    monkeypatch.setattr(skills, "memory", lambda: mem)
    monkeypatch.setattr(skills, "_CATALOGUE", [cat])
    scoop, _ = cat.add("scoop", "blue", {"foot_mm": 91.0, "long_mm": 126.0, "short_mm": 72.0, "L": 90, "a": 10, "b": -50})
    scoop.picked_by = "pick_by_rim"
    tracks = mem.observe([Detection(position=p, foot_mm=f, label=n, label_source="vlm") for p, f, n in
                          [((0.17, 0.16), 23.0, "red cap"), ((0.33, 0.03), 21.0, "red ball"),
                           ((0.24, 0.05), 64.0, "blue scoop")]])[0]
    tracks[2].extra = {"kind": scoop.id}
    state = SceneState()
    state.memory = mem
    for t, name in zip(tracks, ["red_cap", "red_ball", "blue_scoop"]):
        o = SceneObject(name, name, interactable=True, position=np.array([*t.position, 0.0]))
        o.track_id, o.geometry = t.id, {"base_diameter_m": t.foot_mm / 1000.0}
        state.objects.append(o)
    state.targets = [SceneObject("plastic_tub", "a tub", interactable=False, position=np.array([0.2, -0.15, 0.0]))]
    monkeypatch.setattr(skills, "_keeper", lambda: type("K", (), {"holds": lambda self, st, o: None})())
    monkeypatch.setattr(skills, "look_for", lambda env, st, name: False)
    return state


def test_tidy_runs_its_own_task_planned_in_advance_and_plans_nothing_itself(bound, monkeypatch, tmp_path, capsys):
    """User 2026-11-19: "skip the planner - so it just does the task we pre plan for it"."""
    cfg, skills = bound("tidy")
    assert "tidy up" in cfg.TASKS and cfg.PLANNER is False
    importlib.import_module("deployments.tidy.llm_controller")
    deployment = importlib.import_module("mrcf.tabletop.controller").DEPLOYMENT
    assert deployment.task_plan is skills.task_plan and deployment.planner is False
    assert deployment.task_names == ("tidy up",)
    state = _tidy_table(skills, monkeypatch, tmp_path)
    for words in ("Tidy up!", "go", "run"):
        plan = skills.task_plan(words, None, state)
        assert [(a["action"], a["parameters"].get("object_name") or a["parameters"].get("container")) for a in plan] == [
            ("pick", "red_cap"), ("place_in", "plastic_tub"), ("pick", "red_ball"), ("place_in", "plastic_tub"),
            ("pick_by_rim", "blue_scoop"), ("place_in", "plastic_tub")]
    assert '"the white plug": not found on the table — left out.' in capsys.readouterr().out
    assert skills.task_plan("dance a jig", None, state) is None


def test_a_task_written_as_steps_is_laid_out_on_the_table_as_it_is(bound, monkeypatch, tmp_path, capsys):
    cfg, skills = bound("tidy")
    state = _tidy_table(skills, monkeypatch, tmp_path)
    monkeypatch.setattr(cfg, "TASKS", {"scoop and plug away": [
        {"action": "pick", "object_name": "the blue scoop"}, {"action": "place_in", "container": "the plastic tub"},
        {"action": "pick", "object_name": "the white plug"}, {"action": "place_in", "container": "the plastic tub"},
        {"action": "home"}]})
    plan = skills.task_plan("scoop and plug away", None, state)
    assert [(a["action"], a["parameters"]) for a in plan] == [
        ("pick_by_rim", {"object_name": "blue_scoop"}), ("place_in", {"container": "plastic_tub"}), ("home", {})]
    assert '"the white plug": not found on the table — left out.' in capsys.readouterr().out
    assert skills.task_plan("go", None, state) is not None                      # one task: "go" runs it


def test_a_table_deployment_without_tasks_plans_as_before(bound):
    cfg, skills = bound("tabletop")
    assert not getattr(cfg, "TASKS", None) and cfg.PLANNER is True
    importlib.import_module("deployments.tabletop.llm_controller")
    deployment = importlib.import_module("mrcf.tabletop.controller").DEPLOYMENT
    assert deployment.task_plan is None and deployment.planner is True


def test_its_tasks_words_are_names_as_written_not_kinds(bound, monkeypatch, tmp_path):
    """TASK_WORDS_BY_KIND (2026-11-20, the UR5's toolbench) is not tidy's: "the cap" is not the red_cap."""
    cfg, skills = bound("tidy")
    assert not cfg.TASK_WORDS_BY_KIND and cfg.CONTAINER_TALLEST_M is None
    state = _tidy_table(skills, monkeypatch, tmp_path)
    assert skills._thing_called(state, "the cap", set()) is None
    assert skills._thing_called(state, "the red cap", set()).name == "red_cap"
