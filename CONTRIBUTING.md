# Contributing

Contributions are welcome: new deployments, new skills, better IK, more robot
arms and new backends.

## Reporting a problem

Open an issue from one of the templates: a bug report, a feature request or a
new robot.

Two kinds of problem never go in a public issue: a security hole, and the
robot moving when it should not have (for example, the safety layer, the
interlock, the motion guard or the workspace bounds letting a move through).
Report those privately, as [SECURITY.md](SECURITY.md) explains.

Everyone who takes part follows the [code of conduct](CODE_OF_CONDUCT.md).

## Repo structure

The repo has three tiers, plus the tools. A change normally touches exactly
one. A rig's own files (calibrations, photos, downloads, tool output, flight
records) belong to none of them. They live in `rig/<backend>/`, `reports/` and
`runs/`, and never go into git (`mrcf/rig_state.py`).

- **Framework** (`mrcf/`: `robot_env.py`, `capabilities.py`, `interlock.py`,
  `watchers.py`, `motion_guard.py`, `manipulation.py`, `controller.py`, `vlm.py`,
  `scene_state.py`, `action_registry.py`, `skill_executor.py`, `safety.py`,
  `object_memory.py`, `scene_scan.py`, `policies/`, `perception/`, …). It is
  not tied to any robot or task, and it is the only package `pip install`
  installs. It takes `env: RobotEnv` everywhere and knows nothing else about
  the arm. A task that more than one deployment performs also lives here, once
  (`mrcf/tabletop/`, the table task), and each deployment hands it its config.
  A new module goes inside `mrcf/`, never beside it (a test checks), and you
  import it as `from mrcf import x`. `mrcf/perception/` takes only frames and
  camera models, never the world model, the memory, the planner, the arm or a
  policy (a test checks that too). Policies take geometry, never the world
  model or the memory. `tests/test_architecture.py` checks these import rules.
- **Backends** (`mrcf/backends/`). One backend per robot, and `MRCF_ARM`
  chooses which one runs (`backends.arm()`). The folder also holds
  `rig.py` (what every arm with cameras on its rig shares: camera models, the
  grasp point, the gripper tag, the wrist servo), `contract.py` (the checks
  every arm's env passes offline), `real_robot_env.py` (a base class for
  hardware driven through a plain SDK) and `_template/` (a new arm to copy,
  kept working by `tests/test_template.py`). `mrcf/backends/so101/` is a full
  example: its `env.py` drives the arm with LeRobot and placo IK (a `RigEnv`),
  and its `robot_config.py` describes the physical setup. The tools for what
  only the SO-101 has (its poses, servos and calibrations) are in
  `tools/so101/`.
- **Tools** (`tools/`). These measure and check the rig (cameras, tags, the
  overhead camera's calibration, the motion guard) on whichever arm runs,
  through `tools/_common.py`. A test keeps them from naming an arm.
- **Deployments** (`deployments/*/`). Each is one arm, one layout and one
  task. Its `config.py` names its arm (`ARM_NAME`) and serves that arm's
  `robot_config` through `backends.arm(ARM_NAME)` instead of restating
  hardware values. Its `profile.py`, if it has one, says how it drives that arm.
  `deployments/chess/` is the worked example of one with code of its own, and
  `deployments/tidy/` of one that hands its config to a framework task.

To add a capability, a task or an arm, see `DEPLOYMENT_SPEC.md`.

## The rules that keep the structure

A test checks most of these rules and names the rule when it fails. A
failure means some code crossed a line. Move the code to where it belongs, and
never loosen the test. If a rule turns out to be wrong, change it on purpose
in its own pull request, here, in `DEPLOYMENT_SPEC.md` and in its test
together.

| Rule | Checked by |
|---|---|
| A deployment is one arm, one layout and one task. It names its arm once (`ARM_NAME` in `config.py`) and reaches the arm only through `backends.arm(ARM_NAME)`. One flat `profile.py`, if it needs rules of its own, says how it drives that arm, with no `profiles/` folder and no file or folder named after an arm. The same task on another arm or another layout is a new deployment. | `test_each_deployment_names_its_one_arm` |
| A deployment never imports another. Code that two deployments need moves into the framework once, as the table task did (`mrcf/tabletop/`), and is never copied from one to the other. | `test_each_deployment_names_its_one_arm` |
| The framework, the tasks in it and `tools/` name no arm and import no deployment. They are given the env and the policies, or ask `backends.arm()` for the arm that runs. Anything only one arm needs goes in `mrcf/backends/<arm>/` or `tools/<arm>/`. | `tests/test_architecture.py` |
| A policy gets geometry, never the world model or the memory. Perception gets only frames and camera models. A backend's env imports nothing above it. | `tests/test_architecture.py` |
| A new module goes in `mrcf/`, a deployment or `tools/`, never at the top of the repo beside `run.py`. | `test_every_framework_module_is_in_the_one_package` |
| Each value lives in one place, where it belongs: what was measured on an arm in its `robot_config.py`, how a task drives that arm in the deployment's `profile.py`, and a rig's calibrations and saved state in `rig/<arm>/`. A deployment never copies a hardware value. | review |
| What only some arms can do is a capability (`mrcf/capabilities.py`). Shared code asks for it with `isinstance`, never with `hasattr` or `getattr` with a default. | `test_shared_code_asks_an_env_what_it_can_do_by_capability_not_by_name` |
| Shared code changed while working on one arm leaves every other arm's recording as it was. Otherwise the change goes behind a switch that is on only for that arm ([A change for one arm](#a-change-for-one-arm)). Re-record only the arm you meant to change. | `tests/test_arm_baselines.py` |
| Every picture written to disk goes through `mrcf/privacy.py`. A camera is opened only by the index its rig names, after `devices.check_sources`, never by trying indexes in turn. | `tests/test_privacy.py` |

## Adding a new skill

A skill has two parts. The **policy** is how the arm does it: it takes
geometry, returns an `Outcome` and knows nothing of the world model. The
**verb** is what the planner can ask for (`manipulation.object_action`).
[POLICIES.md](POLICIES.md) takes you from an idea to a verb, step by step,
planned or learned, with a worked example and its tests on
`tests/fake_arm.py`. The verb's options, and actions that are not about one
object (`register_action`, conditions), are in
[DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#adding-a-capability).

## Adding a new arm

These steps are for a real arm with a gripper and cameras, in the order a
factory would use: the arm, then its sensors, then the job.

1. Create it. Run `python tools/new_arm.py <arm>`, or answer `n` when
   `run.py` asks for the arm. This copies `mrcf/backends/_template` to
   `mrcf/backends/<arm>/`: an arm that exists only in memory and already meets
   the backend contract. Nothing outside that folder changes.
2. Fill it in. Replace every `YOUR ARM` mark. `env.py` talks to the arm's SDK
   (a `backends.rig.RigEnv`). `robot_config.py` holds every measured
   value, and every camera and sensor the rig has (`CAMERAS`, `SENSORS`).
   "Adding an arm" in DEPLOYMENT_SPEC.md says what to implement, and
   `mrcf/backends/ur5/` is a second complete example. HARDWARE.md says what
   the rig needs.
3. Check it offline. In a test like the one in `tests/test_template.py`, make
   `check_backend()` (`mrcf/backends/contract.py`) pass on an offline
   instance. Then record the arm's baselines:
   `MRCF_UPDATE_BASELINES=1 pytest tests/test_arm_baselines.py -k <arm>`.
4. Set it up on the rig. `python run.py --arm <arm>` shows what the arm has
   before anything runs: plugged in or not, its gripper, each camera (plugged
   in, calibrated) and its sensors. Then follow "First-time setup on another
   arm" in COMMANDS.md, in order: the arm moving, the cameras, the overhead
   calibration, the motion guard.
5. Keep what is only this arm's with it. Scripts only this arm needs go in
   `tools/<arm>/`, and what you see it do goes in
   `mrcf/backends/<arm>/NOTES.md`.
6. Give it a job: a deployment of its own (below). An arm never runs
   another arm's deployment, and `run.py` lists those under the other arm.

Any other robot implements the `RobotEnv` abstract interface:

```python
from mrcf.robot_env import RobotEnv
import numpy as np

class MyEnv(RobotEnv):
    def step(self, n: int = 1) -> None: ...
    def get_ee_pos(self) -> np.ndarray: ...
    def get_joint_angles(self) -> dict[str, float]: ...
    def get_body_pos(self, name: str) -> np.ndarray | None: ...
    def get_movable_body_names(self) -> list[str]: ...
    def get_static_body_names(self, exclude_keywords=None) -> list[str]: ...
    def render_image(self, camera, width, height) -> np.ndarray: ...
    def pixel_to_world(self, u, v, z_world, camera, render_w, render_h) -> np.ndarray: ...
    # Manipulation (optional — only for robots with a controllable end-effector):
    def move_ee_to(self, target_pos, tol, max_iters=300, step_size=0.0, max_step_deg=None, orientation=None) -> bool: ...
    def set_gripper(self, width: float, n_steps: int = 200, max_step_pct=None) -> None: ...
    def get_gripper_width(self) -> float: ...
    def go_home(self) -> None: ...
    # Stopping: call self.guard() before every motor command, and stop where it is here.
    def hold(self) -> None: ...
    # The connection (both optional): open the hardware without moving it; close it, even half-open.
    def connect(self) -> None: ...
    def stop(self) -> None: ...
```

`mrcf/backends/so101/` (`env.py` lists its parts) is a complete real-hardware
example. It shows how to cope when the optional kinematics (placo and the
URDF) are not installed, and how to send every command through one guarded,
locked `_send`.

## Adding a new deployment

A deployment is one arm, one layout and one task, and it names its arm
(`ARM_NAME`).

1. Make sure it is new. More of what a deployment already does (new things
   to do with the same objects, on the same layout, with the same arm) goes
   into that deployment. Another arm, layout or goal needs a new one (see "A
   new deployment, or more of one there is?" in DEPLOYMENT_SPEC.md).
2. Start from the nearest one. Run
   `python tools/new_deployment.py NAME --from SOURCE [--arm ARM]`, or answer
   `n` when `run.py` asks for the deployment. This copies the source's code
   onto the chosen arm. The source's rules go into the new `profile.py`
   commented out, each with its reason. Turn on only those whose reason holds
   for the new work. The source's tests, notes, history and saved state are
   not copied.
3. Make it its own. Give it its own settings in `config.py` (the arm serves
   the hardware values, so never restate them), its own actions in
   `skills.py` and its own prompts. Its saved state is named after it
   (`rig/<arm>/<name>_*.json`), so it never shares another deployment's memory
   or surface.
4. Test and document it. Its tests go in `deployments/<name>/tests/`, where
   `pytest` finds them on its own. Give it a `NOTES.md` and a `HISTORY.md`.
5. Run it. `python run.py --deployment NAME` runs it on its own arm.

`DEPLOYMENT_SPEC.md` lists the required files and has a checklist.
`deployments/chess/` is a complete reference implementation on real hardware.
`deployments/tidy/` is a deployment of the framework's table task
(`mrcf/tabletop/`), made of a config and a profile.

## When something goes wrong

1. Read what was recorded. Every run keeps a folder, `runs/flight/<session>/`,
   with `session.json` (the arm, the deployment, the cameras, the commit),
   `console.log` (everything printed), `incidents/` (each failure with its
   frames and traceback) and `timings.json`. `python -m mrcf.flight_recorder`
   tells the latest incident's story, `--list` lists the incidents, and
   `--timings` shows where the time went.
2. Narrow it down with the tools in "When something's wrong" in COMMANDS.md.
   Each tests a smaller piece than the last, with no model and no planner.
3. Fix it in the tier the problem belongs to, and only there.

   | The problem is about | The fix goes in |
   |---|---|
   | One arm's measurement or calibration | Its `robot_config.py`, or a re-run of the calibration ("What to re-run after a change" in COMMANDS.md) |
   | How one task drives its arm (how wide to open, how hard to hold, what counts as held) | That deployment's `profile.py` |
   | One task's objects, layout or prompts | That deployment |
   | Code every arm runs | The framework. A change tuned on one arm goes behind a switch that is on for that arm only, and the baselines show what it does to each arm ([A change for one arm](#a-change-for-one-arm)) |
   | Only this rig (its light, its table, its room) | Its calibration in `rig/<arm>/`, and a line in `rig/<arm>/notes.md` |

   Never add a special case for one arm or one task inside shared code, copy
   one deployment's file into another to patch it, re-record another arm's
   baselines to make the suite pass, or loosen a test in
   `tests/test_architecture.py` or `tests/test_privacy.py`.
4. Write down what you learned. What the code cannot fix yet goes in that
   tier's `NOTES.md` (Seen, Cause, Workaround, Status). How you found a value
   or a fix goes in the `HISTORY.md` beside the code. "What you learn, and
   where it goes" in DEPLOYMENT_SPEC.md has the details.
5. Report it as an issue ([Reporting a problem](#reporting-a-problem)), with
   the incident's story. Before you attach a frame, check that it shows
   nothing private.

## Code style

- Use Python 3.10+ type hints throughout.
- Give every public function a docstring.
- Comments say what a value or a step is for, and why, in the present tense.
  The story of how it was found (the date, the run, what went wrong) goes in
  the `HISTORY.md` beside the code, under the file and the name it explains.
- Use `print()` for runtime diagnostics (no external logging framework). A
  run's console is kept with its flight record (`console.log`). Also report a
  failure with `flight_recorder.RECORDER.failure(why, exc=...)`, which records
  an incident with the traceback. Time anything worth timing with
  `RECORDER.timed(name, seconds)` or an `action()` span.
- Write every picture to disk through `mrcf/privacy.py` (`imwrite`,
  `save_rgb`, or the flight recorder). It covers faces and keeps only the work
  area in a camera's frames. Open a camera only by the index its rig names,
  after `devices.check_sources`, never by trying indexes in turn.
  `tests/test_privacy.py` enforces both rules.
- Add no external dependencies beyond those in `requirements.txt` and
  `pyproject.toml`.

## Tests and lint

```bash
pip install -e ".[dev]"
pytest              # framework tests, plus every deployment's offline ones
ruff check .        # errors and likely bugs only — see [tool.ruff] in pyproject.toml
mypy                # types, over mrcf/ — see [tool.mypy] in pyproject.toml
```

That runs most of the suite. A test that needs OpenCV, Stable-Baselines3,
LeRobot or placo skips itself when that library is missing, and says which.
To run more of them:

```bash
pip install -e ".[dev,measure,sb3]"   # OpenCV, SAM2's transformers, Stable-Baselines3
```

CI (`.github/workflows/tests.yml`) runs both installs, and mypy. The core
install runs on Linux with Python 3.10 and 3.15, and on Windows and macOS with
3.15, and nothing may fail there. The full install runs on Linux with Python
3.15. The SO-101 tests that need LeRobot and placo run on every push, in
`.github/workflows/so101.yml`. Before you send a change to
`mrcf/backends/so101/`, run them in a LeRobot environment too.

## A change for one arm

The arms share most of the code: the policies, `RigEnv`, `wiring.py`, the
`GripperSpec` defaults, safety, perception and the controller. A change you
make with one arm on the desk reaches every other arm through it, and the
shared code's own tests run on the test arm (`tests/fake_arm.py`), which is
none of the real arms. So:

- Each arm is recorded. `tests/test_arm_baselines.py` builds every arm
  offline, the way the shared code does, and compares it with
  `tests/baselines/<arm>/`: its settings (the `GripperSpec` its
  `make_policies()` builds, the keywords `wiring.py` passes to its
  `make_env()`), its env (its capabilities, its constants, the state it starts
  in) and its plans (standard scenes laid out in its own workspace, each
  planned without moving anything: the grasp, the room, the approaches and
  why not the others, the waypoints). A failure names the arm and what
  changed. If the change was meant for that arm too, re-record it:

  ```bash
  MRCF_UPDATE_BASELINES=1 pytest tests/test_arm_baselines.py -k so101
  python tests/arm_baselines.py so101 --diff      # the same differences, from the command line
  ```

  Then commit `tests/baselines/so101/` with the change, so the diff shows
  what it does to that arm. Nothing connects or moves during recording. The
  SO-101's env and plans need LeRobot, placo and its URDF, and the UR5's need
  OpenCV. Without them, those checks are skipped, but never in the CI jobs set
  to check them (`MRCF_BASELINES_REQUIRE`). A new arm fails here until it is
  first recorded.
- What only some arms can do is a capability. Shared code never asks an env
  for something by name (`hasattr`, or `getattr` with a default): if one arm
  renames it, the other quietly goes without. What every arm has is on
  `RobotEnv`, and shared code calls it directly. What only some have is a
  Protocol in `mrcf/capabilities.py`, asked for with `isinstance` and
  recorded per arm with the env. `tests/test_architecture.py` enforces this.
- A change tuned on one arm belongs to that arm until another has run it.
  Say you find a new step in a pick, a different default or a new check on
  the UR5. Put it behind a switch: a `GripperSpec` field (set from the arm's
  config through `wiring.SPEC_SETTING_FOR`) or a config setting. Its default
  is what the code did before, and it is on in the arm that tried it. The
  other arm's recording stays as it was. Once the change has run on the other
  arm, turn it on there and record that arm's baselines. Once every arm has
  it on, make it the default and remove the switch.

## Pull requests

- Make one logical change per PR.
- Found a quirk or a limit the code cannot fix yet? Add an entry to the
  `NOTES.md` of the framework, robot or task it is about
  ([DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#what-you-learn-and-where-it-goes)).
- Update `README.md` / `DEPLOYMENT_SPEC.md` / `POLICIES.md` if the public API
  changes.
- For a change users will notice, add a line to [CHANGELOG.md](CHANGELOG.md)
  under "Unreleased" (Added, Changed, Fixed or Removed).
- Run the tests and the linter (above). The suite runs without hardware,
  so also verify a change to motion on a real arm, or on a vendor's simulator
  of one (URSim for a UR), and describe how you tested it.

The pull request template has these as a checklist.

MRCF is licensed under Apache-2.0 ([LICENSE](LICENSE)). Anything you submit is
licensed under it too (section 5), including its patent grant, so no one who
contributes can later assert a patent on their contribution against those who
use it.

## Releasing

This is for maintainers. MRCF follows [semantic versioning](https://semver.org).
While the version is 0.x, a minor version (0.1 to 0.2) may break the API.

1. Set the new version in `mrcf/__init__.py` (`__version__`). `pyproject.toml`
   reads it from there.
2. In `CHANGELOG.md`, turn "Unreleased" into the new version with today's date
   (`## [0.2.0] — 2026-11-02`), and put an empty "Unreleased" above it. Then
   update the links at the bottom: "Unreleased" compares from the new tag, and
   the new version from the one before. For 0.1.0, whose section is already
   there marked `unreleased`, only set its date.
3. Commit that to `main`, then tag it and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The release workflow (`.github/workflows/release.yml`) checks that the tag
matches the version and that the changelog section has a date. It then builds
the sdist and the wheel, and publishes a GitHub release with both attached and
the version's section of `CHANGELOG.md` as its notes. Nothing goes to PyPI.
