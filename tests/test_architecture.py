"""
The layers' import rules, checked on the source. Every framework module is in
mrcf/; an import of one is read by its name there (`from mrcf import safety`
and `mrcf.policies.base` are `safety` and `policies`).

    mrcf/policies/   geometry in, Outcome out: never the world model, the object
                     memory, the planner, or any deployment
    mrcf/perception/ frames and camera models in, measurements out: never the
                     world model, the memory, the planner, the arm, or a policy
    the framework    never imports a deployment, nor names an arm
    the table task   (mrcf/tabletop) likewise: it works for the deployment bound to it
    deployments      each is one arm, one layout and one task: it names its arm
                     (ARM_NAME) and reaches it through backends.arm(ARM_NAME),
                     never a backend's modules; never another deployment
    tools/           likewise: the arm that runs, never one by name
    backend envs     drive hardware: nothing that sits above them
    object_memory    knows objects, not how the arm moves
    the repository   holds no module outside mrcf/ but the run.py script

These are what let a policy be reused in a new scenario, and a scenario on a
new arm. A failure here means something reached across a line it should not.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MRCF = ROOT / "mrcf"

DEPLOYMENT_MODULES = {p.stem for p in ROOT.glob("deployments/*/*.py")} - {"__init__"}
STATE_AND_PLANNER = {"object_memory", "vlm", "scene_state", "action_registry", "manipulation",
                     "skill_executor", "scene_scan", "watchers", "safety", "controller"}


def _framework(dotted: str) -> str:
    """A module's name, a framework one's as it is inside mrcf/ ("mrcf.policies.base" -> "policies.base")."""
    return dotted[len("mrcf."):] if dotted.startswith("mrcf.") else dotted


def imported_modules(path: Path) -> set:
    """Full dotted names of every module a file imports, anywhere in it (a framework one's inside mrcf/)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {_framework(a.name) for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names |= ({a.name for a in node.names} if node.module == "mrcf" else {_framework(node.module)})
    return names


def _arm_packages(path: Path) -> set:
    """The arms (backends/<arm>/ packages) a file names in its imports."""
    return {m.split(".")[1] for m in imported_modules(path)
            if m.startswith("backends.") and (MRCF / "backends" / m.split(".")[1] / "__init__.py").exists()}


def imported(path: Path) -> set:
    """Top-level names of every module a file imports, anywhere in it (a framework one's inside mrcf/)."""
    return {name.split(".")[0] for name in imported_modules(path)}


def test_policies_see_geometry_not_state():
    for f in sorted((MRCF / "policies").glob("*.py")):
        bad = imported(f) & (STATE_AND_PLANNER | DEPLOYMENT_MODULES | {"backends", "deployments"})
        assert not bad, f"mrcf/policies/{f.name} imports {sorted(bad)}"


def test_perception_sees_frames_not_state_or_motion():
    # So it can be used with any camera, memory or arm: what it needs comes in
    # as frames, masks and camera models (a RobotEnv only as something that has
    # render_image and camera_model, never imported).
    for f in sorted((MRCF / "perception").glob("*.py")):
        bad = imported(f) & (STATE_AND_PLANNER | DEPLOYMENT_MODULES
                             | {"backends", "deployments", "policies", "robot_env", "interlock", "motion_guard"})
        assert not bad, f"mrcf/perception/{f.name} imports {sorted(bad)}"


def test_the_framework_never_imports_a_deployment():
    for f in sorted([*MRCF.glob("*.py"), *(MRCF / "policies").glob("*.py"), *(MRCF / "perception").glob("*.py"),
                     *(MRCF / "tabletop").glob("*.py")]):
        bad = imported(f) & (DEPLOYMENT_MODULES | {"deployments"})
        assert not bad, f"{f.relative_to(ROOT)} imports {sorted(bad)}"


def test_the_table_task_names_no_arm():
    # Its arm is the bound deployment's (cfg.ARM): it imports no backend at all.
    for f in sorted((MRCF / "tabletop").glob("*.py")):
        bad = {m for m in imported_modules(f) if m.split(".")[0] == "backends"}
        assert not bad, f"{f.relative_to(ROOT)} imports {sorted(bad)}"


def test_the_framework_names_no_arm():
    # What runs on every arm cannot import one: it is handed the env and the policies.
    for f in sorted([*MRCF.glob("*.py"), *(MRCF / "policies").glob("*.py"), *(MRCF / "perception").glob("*.py")]):
        bad = {m for m in imported_modules(f) if m.split(".")[0] == "backends"}
        assert not bad, f"{f.relative_to(ROOT)} imports {sorted(bad)}"


def test_each_deployment_names_its_one_arm():
    # A deployment is one arm, one layout and one task (DEPLOYMENT_SPEC.md):
    # the same task on another arm is another deployment. 2026-11-17: the
    # SO-101's rules for the table, filed inside the UR5's deployment as
    # profiles/so101.py, were the UR5's to miss. Its config says which arm
    # (ARM_NAME), and it reaches it by backends.arm(ARM_NAME), not by importing
    # the backend's modules. Its tests/ and tools may be arm-specific.
    from mrcf import backends

    configs = sorted(ROOT.glob("deployments/*/config.py"))
    assert len(configs) >= 3
    for config in configs:
        tree = ast.parse(config.read_text(encoding="utf-8"))
        named = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "ARM_NAME" for t in n.targets)
                 and isinstance(n.value, ast.Constant)]
        assert len(named) == 1 and named[0] in backends.available(), \
            f"{config.relative_to(ROOT)} names no arm it can run on (ARM_NAME = one of {backends.available()})"
        assert not (config.parent / "profiles").exists(), \
            f"{config.parent.relative_to(ROOT)}/profiles: a deployment has one arm, and one profile.py"
    for f in sorted(ROOT.glob("deployments/*/*.py")):
        imported_arms = _arm_packages(f)
        assert not imported_arms, \
            f"{f.relative_to(ROOT)} imports the {sorted(imported_arms)} backend: use backends.arm(ARM_NAME)"
        others = {m.split(".")[1] for m in imported_modules(f) if m.startswith("deployments.")} - {f.parent.name}
        assert not others, f"{f.relative_to(ROOT)} imports another deployment ({sorted(others)})"


def test_the_frameworks_tools_measure_whichever_arm_runs():
    # tools/ is for every arm (tools/_common: backends.arm()); a tool that needs
    # one arm's motors or poses belongs in tools/<backend>/ (tools/so101/).
    for f in sorted((ROOT / "tools").glob("*.py")):
        named = _arm_packages(f)
        assert not named, f"tools/{f.name} imports the {sorted(named)} arm by name: use tools._common"


def test_backend_envs_import_nothing_above_them():
    so101_parts = [f for f in sorted((MRCF / "backends/so101").glob("*.py"))
                   if f.name not in ("__init__.py", "robot_config.py")]   # env.py and the parts it is built from
    assert len(so101_parts) >= 8
    for f in [*so101_parts, MRCF / "backends/real_robot_env.py",
              MRCF / "backends/rig.py", MRCF / "backends/contract.py",
              MRCF / "streaming.py"]:                                     # what backends' streams are built on
        bad = imported(f) & (STATE_AND_PLANNER | DEPLOYMENT_MODULES | {"policies", "deployments"})
        assert not bad, f"{f.relative_to(ROOT)} imports {sorted(bad)}"


def test_the_object_memory_knows_nothing_of_motion():
    bad = imported(MRCF / "object_memory.py") & (
        {"policies", "manipulation", "robot_env", "vlm", "skill_executor", "backends"} | DEPLOYMENT_MODULES)
    assert not bad, f"object_memory imports {sorted(bad)}"


def test_the_flight_recorder_imports_nothing_of_the_framework():
    # Every layer writes to it — policies, perception's users, backends, the
    # memory — so it can sit under none of them.
    bad = imported(MRCF / "flight_recorder.py") - {"__future__", "base64", "collections", "contextlib", "json", "os", "sys",
                                                   "threading", "time", "dataclasses", "datetime", "pathlib", "traceback",
                                                   "numpy", "cv2"}
    assert not bad, f"flight_recorder imports {sorted(bad)}"


def test_every_framework_module_is_in_the_one_package():
    # pyproject installs mrcf/ and nothing else: a module beside it imports fine
    # from the checkout and is missing once installed — and would put a name of
    # its own into everyone's site-packages.
    import re
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'include = \["mrcf\*"\]', text) and "py-modules" not in text
    scripts = {"run"}                                   # entry points, not importable modules
    assert {p.stem for p in ROOT.glob("*.py")} - scripts == set()
    stale = [str(p.relative_to(ROOT)) for name in ("backends", "policies", "perception")
             for p in (ROOT / name).rglob("*.py")]
    assert not stale, f"modules left where the framework was before mrcf/: {stale}"


# ── Asking an env for something by name ───────────────────────────────────────
# hasattr(env, "x") or getattr(env, "x", default) in code every arm runs: on an
# arm without x, or one where a change renamed it, the shared code quietly
# does without, and nothing fails. What every arm has is on RobotEnv, called
# as it is; what only some have is a capability (capabilities.py), asked with
# isinstance — and which capabilities each arm has is recorded
# (tests/baselines/<arm>/env.json), so losing one fails. Asked by name only
# where the thing is one arm's own and a tool shows it when it is there:

ASKED_BY_NAME = {
    ("tools/_common.py", "approach_orientation"): "the SO-101's one-target search, before OrientationSearch",
    ("tools/look_tour.py", "get_joint_degrees"): "the SO-101's joints in degrees, kept with a look where it has them",
    ("tools/calibrate_overhead_camera.py", "disable_torque"):
        "hand-guided: an SO-101 with its torque off, a UR5 in freedrive (UR5Env.disable_torque)",
    ("tools/check_motion.py", "current_error"): "a UR5 controller's error, shown when there is one",
    ("tools/check_motion.py", "speed_scaling"): "a UR5's speed slider, shown when there is one",
    ("deployments/chess/skills.py", "go_to_look_around_pose"): "the SO-101's look-around poses (LOOK_AROUND_POSES)",
}


def _shared_code() -> list:
    """Code every arm runs: mrcf/ but each arm's own folder, the deployments (not their tests), tools/ but tools/<arm>/."""
    arms = {p.name for p in (MRCF / "backends").iterdir() if p.is_dir()}
    code = [p for p in MRCF.rglob("*.py")
            if not (p.parent.parent == MRCF / "backends" and p.parent.name in arms)]
    code += [p for p in (ROOT / "deployments").rglob("*.py") if "tests" not in p.parts]
    code += list((ROOT / "tools").glob("*.py"))
    return sorted(code)


def _asked_by_name(path: Path) -> list:
    """(line, name) of every hasattr(env, "name") and getattr(env, "name", default) in path: env, self.env, self._env."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and (node.func.id == "hasattr" and len(node.args) == 2
                     or node.func.id == "getattr" and len(node.args) == 3)):
            continue
        who, what = node.args[0], node.args[1]
        is_env = (isinstance(who, ast.Name) and who.id == "env"
                  or isinstance(who, ast.Attribute) and who.attr in ("env", "_env"))
        if is_env and isinstance(what, ast.Constant) and isinstance(what.value, str) \
                and not what.value.startswith("_"):
            found.append((node.lineno, what.value))
    return found


def test_shared_code_asks_an_env_what_it_can_do_by_capability_not_by_name():
    asked = {(str(p.relative_to(ROOT)).replace("\\", "/"), name, line)
             for p in _shared_code() for line, name in _asked_by_name(p)}
    bad = sorted(f"{f}:{line} asks for {name!r}" for f, name, line in asked if (f, name) not in ASKED_BY_NAME)
    assert not bad, ("Shared code asks an env for something by name (hasattr / getattr with a default): call it "
                     "directly if every arm has it (RobotEnv), else make it a capability (capabilities.py) and "
                     "ask isinstance.\n  " + "\n  ".join(bad))
    stale = sorted(f"{f}: {name}" for f, name in ASKED_BY_NAME if not any((f, name) == a[:2] for a in asked))
    assert not stale, f"ASKED_BY_NAME lists what nothing asks for any more: {stale}"


def test_the_rule_finds_a_name_asked_for(tmp_path):
    import textwrap

    src = textwrap.dedent("""
        def f(env, self):
            a = hasattr(env, "frame_scale")
            b = getattr(self.env, "move_refusals", 0)
            c = getattr(env, "frame_scale")          # no default: raises, like env.frame_scale
            d = hasattr(env, "_private")             # an arm's own insides: a contract check's business
            e = hasattr(robot, "probe")              # not an env
    """)
    path = tmp_path / "probe.py"
    path.write_text(src, encoding="utf-8")
    assert _asked_by_name(path) == [(3, "frame_scale"), (4, "move_refusals")]
