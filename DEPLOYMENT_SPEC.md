# Extending the framework

## How the pieces fit

```
Camera(s) ──► Vision processing ◄──► Memory / State
                     │                     │
                     ├───────────► VLM ◄───┘ ◄── Human instruction
                     │              │
                     └──► Policy library ──► Robot hardware
```

| Box | Where |
|---|---|
| Camera(s), Robot hardware | `mrcf/backends/<arm>/`, a `RobotEnv` |
| Vision processing | `mrcf/perception/` (any camera: measuring, outlines, the surface scan, tags, hand-eye, light); `mrcf/scene_scan.py` (turns a scan into memory); a deployment's own perception (`chessboard.py`) |
| Memory / State | `mrcf/object_memory.py` (which object is which), `mrcf/scene_state.py` (`SceneState`, the world model the planner sees) |
| VLM | `mrcf/vlm.py` (perceive, decompose, plan); `mrcf/action_registry.py` (the actions it plans with, and dispatch); `mrcf/controller.py` (the loop from a command to the arm) |
| Policy library | `mrcf/policies/`: how an arm does things, from geometry |

The most important rule: **a policy never sees the world model or the
memory.** It is given geometry: where an object's foot is, how tall it is, and
what stands next to it (`policies.Target`, `Obstacle`). It returns an
`Outcome`. `mrcf/manipulation.py` is the only code that knows both sides. It
turns the planner's names into Targets, runs the policy, and writes what
happened back into the world model and the memory. That is what lets the same
pick run in a new scenario, and a scenario run on a new arm.

`tests/test_architecture.py` checks these rules on the source:

- `mrcf/policies/` imports no state, planner or deployment module
- nothing at the framework level imports a deployment
- a backend's env imports nothing above it
- `object_memory` knows nothing about motion

## Three tiers

| Tier | Where | Knows about |
|---|---|---|
| **Framework** | `mrcf/` | Nothing specific to an arm or deployment. A task that more than one deployment does (the table task, `mrcf/tabletop/`) lives here once, and each hands it its config. |
| **Backend** | `mrcf/backends/<arm>/` | One robot, whatever it does: how to drive it, how it is set up and calibrated, its jaws, its reach, how to check it. What every arm shares (cameras and gripper geometry, the contract it keeps) is in `mrcf/backends/rig.py` and `mrcf/backends/contract.py`. |
| **Deployment** | `deployments/<name>/` | One arm, one layout, one task: which arm (`ARM_NAME`), what the objects are and where they stand, how to find and name them, and what to tell the models. Also how it drives its arm (`profile.py`: how wide to open, how hard to hold, what a disagreeing look means), laid over the arm's facts (`mrcf/profiles.py`). |

A recalibration changes one file, not every deployment on the arm. A second
deployment on the same arm gets the tools, the measured poses and the policies
for free. It does not get another deployment's rules, which belong to each
deployment (see its profile in [Adding a task](#adding-a-task)). A deployment
names its arm (`ARM_NAME`, reached by `backends.arm(ARM_NAME)`), and the same
task on another arm, or on another layout, is another deployment. The
framework, the tasks in it (`mrcf/tabletop/`) and the tools never name an arm.
They work with the arm a deployment brings, or the one that is running
(`MRCF_ARM`, which `run.py` sets to the deployment's).

Pick your situation:

- [Adding a capability](#adding-a-capability): a new thing the planner can ask
  for
- [Adding a learned policy](#adding-a-learned-policy): a trained model
  (imitation or RL) as part of one
- [POLICIES.md](POLICIES.md): writing a policy, planned or learned, step by
  step
- [Adding a task](#adding-a-task): the same robot, different objects or goal
- [Adding an arm](#adding-an-arm): different hardware
- [Stopping the arm, and running things alongside it](#stopping-the-arm-and-running-things-alongside-it)
- [What you learn, and where it goes](#what-you-learn-and-where-it-goes): the
  quirks every robot and task turns up

---

## What you learn, and where it goes

Every robot and task turns up things its code and config cannot say: a grip
that slips on one kind of object, a detection that doubles up, noon light that
washes out half the table. Write each down next to what it is about, in the
same tiers as the code:

| It is about | Where | In git |
|---|---|---|
| The framework, on every robot and task | `mrcf/NOTES.md` | yes |
| One robot, on any rig and any task | `mrcf/backends/<backend>/NOTES.md` | yes, it travels with the backend |
| One task, on any robot | `deployments/<task>/NOTES.md` | yes, it travels with the task |
| Only this rig: its table, its light, its room | `rig/<backend>/notes.md` | no, it would be wrong anywhere else |
| How the stages fail, on every robot and task | [SKILL_LIBRARY.md](SKILL_LIBRARY.md), "Failure modes by stage" | yes |
| Why the code is as it is: the dated runs behind a value or a fix | `HISTORY.md` next to the code (`mrcf/`, `mrcf/policies/`, `mrcf/backends/<backend>/`, `deployments/<task>/`) | yes |

Each `NOTES.md` starts with its **Limits**: what it cannot do, by design or
not yet (no force sensing, a reach smaller than the table). Give each a
sentence or two, and delete it once lifted. A project's known limitations live
here, next to what they are about, not in the README. Below them come the
things seen, one entry per issue, in the same shape everywhere, so they read
and search alike:

```markdown
### What happens, in a few words
- **Seen:** the date; what was seen; the flight recorder's incident, if there is one
- **Cause:** what is behind it, or "not known yet" and the best guess
- **Workaround:** what to do until it is fixed
- **Status:** open, or fixed (the commit, or where the fix lives)
```

A comment in the code says what a value or a step is for, and why, in the
present tense. The story of how it was found (the date, the run, the piece
that fell) goes in the `HISTORY.md` next to it, under the file and the name it
explains. A fixed `NOTES.md` entry that is worth keeping moves there too.

A note is for what cannot be code. If a quirk can be fixed or checked with a
test, a refusal or a hook, do that, then mark the entry fixed or delete it
(git keeps it). Measured values and the reasons behind them stay where they
are used (`robot_config.py`, next to the value). The flight recorder already
keeps what happened at every failure (`runs/flight/<session>/incidents/`, each
with its traceback where there is one), what the run was (`session.json`: arm,
deployment, cameras, commit), everything printed (`console.log`) and where the
time went (`timings.json`; `python -m mrcf.flight_recorder --timings`). A note
says what it meant. `backends/_template` has a `NOTES.md` to start from.

---

## Adding a capability

A capability has two pieces, in two places: the **policy** (how the arm does
it, from geometry) and the **action** (what the planner calls, which connects
the policy to the world model).
[POLICIES.md](POLICIES.md#a-planned-policy-step-by-step) walks through a
fuller example, with a test on the in-memory arm.

**1. The policy** goes in `mrcf/policies/` if any arm with that kind of
end-effector could use it, or in your backend if it is specific to one arm. It
takes `env` and geometry, talks to the arm only through `RobotEnv`, and
returns an `Outcome`:

```python
import numpy as np

from mrcf.policies.base import Outcome, Target

def nudge(env, target: Target, distance_m: float = 0.02, table_z: float = 0.0) -> Outcome:
    goal = np.array([target.position[0] + distance_m, target.position[1], table_z + 0.02])
    if not env.move_ee_to(goal):
        return Outcome(False, f"IK did not converge ({distance_m * 100:.0f} cm nudge)")
    return Outcome(True, f"nudged {distance_m * 100:.0f} cm", at_xy=goal[:2])
```

Check what `move_ee_to` returns. It reports whether the arm arrived. A policy
that ignores it can report success from the wrong place. If the policy
declines before moving (no room, out of reach), say so with `refused=True`.
The failure is then reported without asking the vision model to describe a
scene where nothing happened.

Do not catch stops. When the interlock trips, the backend raises
`MotionStopped` from inside the move. It derives from `BaseException`, so
`except Exception` lets it through, as it should.

**2. The action** registers the policy with the planner. For a verb on one
named object, `manipulation.object_action` does all the work. It finds the
object and hands the policy its geometry and neighbours through `run_policy`
(the safety monitor, the failure description). With `moves_object=True`, it
moves the object to the Outcome's `at_xy` in the world model and the memory:

```python
from mrcf import manipulation

manipulation.object_action(
    "nudge", lambda pol, env, target, obstacles, **_: nudge(env, target),
    description="Push an object a short distance without grasping it.",
    moves_object=True)
```

(`pol` is the installed policy library, for a verb that is one of its
methods.) For anything else, write your own `register_action` that does the
same by hand:

```python
from mrcf import manipulation
from mrcf.skill_executor import run_policy
from mrcf.vlm import register_action

@register_action(
    "nudge",
    description="Push an object a short distance without grasping it.",
    parameters={"object_name": "Exact name of the object to nudge."},
    conditions={"pre": ["object_exists", "nothing_carried"]},
)
def skill_nudge(env, scene_state, object_name, **_):
    obj = scene_state.find(object_name)
    if obj is None or obj.position is None:
        print(f"  [Error] '{object_name}' not located. Try 'observe' first.")
        return False
    outcome = run_policy(nudge, "nudge", env, target=manipulation.target_of(obj))
    if outcome.ok:
        scene_state.update_position(obj.name, np.r_[outcome.at_xy, obj.position[2]])
    return outcome.ok
```

Return the outcome. `dispatch()` passes it back to the controller, which is
how a failed step stops the rest of the plan and gets re-planned. To the
controller, an action that returns nothing looks as if it might have worked.

The `description` is the planner's only clue about when to choose this action.
Write it for a model choosing between a dozen actions, not for a developer.

Pick, place, move_above, home and open/close_gripper already exist:
`manipulation.install()` registers them (see [Adding a task](#adding-a-task)).
`register_action` and `dispatch` live in `mrcf/action_registry.py`, and `vlm`
re-exports them.

### Conditions

`conditions` checks an action against the world model before it runs (`pre`)
and cross-checks afterwards (`post`). The framework provides:

`object_exists` · `object_not_carried` · `something_carried` ·
`nothing_carried` · `target_exists`

Register any task-specific condition yourself:

```python
from mrcf.vlm import register_condition

@register_condition("board_visible")
def _board_visible(params, scene_state):
    return bool(scene_state.targets), "no board squares in view"
```

A condition is a pure check on `(params, scene_state)` that returns
`(ok, reason)`. It must not move the robot. An unknown condition name prints a
warning and is skipped, so a typo shows up instead of silently letting
everything through.

`post` conditions are advisory. Instead of failing the action, they report
when the robot and the world model disagree (a pick that returned success
while nothing is marked carried).

---

## Adding a learned policy

A learned policy is a trained model (ACT, diffusion, SmolVLA or pi0 from
demonstrations, or a policy trained by reinforcement learning). It runs on the
arm's stream for one segment of a skill: a planned move to where its
demonstrations started, the learned part, then a look to see what it did. It
is trained elsewhere. This repository only runs it.

[POLICIES.md](POLICIES.md#a-learned-policy-step-by-step) takes one from an
idea to a verb, step by step:

- the arm's stream (`capabilities.ActionStreaming`; `tools/show_stream.py`
  prints what a model must be trained on)
- training on it elsewhere
- packaging it with the stream it was trained on (`mrcf/policies/packages.py`)
- checking that it fits before anything moves
- making it a verb (`LearnedPolicy`, `chain`, chess's `LEARNED_SKILLS`)
- what its refusals mean
- a model format of your own
- the plugin seams a policy store would use (sources, formats, arms)

The short version:

```python
from mrcf.policies.learned import LearnedPolicy, chain, hover_over
from mrcf.policies.lerobot_model import LeRobotModel

upright = LearnedPolicy(LeRobotModel(".../checkpoints/last/pretrained_model", task="stand the pawn up"),
                        name="upright", max_s=20)
manipulation.object_action("upright", chain(hover_over(0.06), upright),
                           description="Stand a fallen piece back up on its base.",
                           after=look_again)   # your deployment's (env, scene_state, obj, outcome): a fresh scan
```

Like any policy, it never sees the world model or the memory and returns an
`Outcome`. Before anything moves, it refuses a model trained on another
stream, or one whose library this machine does not have.

---

## Adding a task

### A new deployment, or more of one there is?

A deployment is one arm, one layout and one task. New things to do with the
same objects, on the same layout, with the same arm, toward the same goal (a
new opening for chess, another container for tidy) go into that deployment as
a skill, a prompt or a rule in its profile. A different arm, surface, set of
objects or goal (a desk to tidy, a dishwasher to load, the table task on the
UR5 instead of the SO-101) is a new deployment. Start it from the nearest
one's code, and take over only the rules whose reasons hold for the new work:

    python tools/new_deployment.py tidy_desk --from tidy             # its code and arm; its rules as candidates
    python tools/new_deployment.py tidy_ur5 --from tidy --arm ur5    # the same task on another arm
    python tools/new_deployment.py --list chess                      # a deployment's rules, each with why

Code two deployments share belongs in the framework, not copied into each. The
table task is `mrcf/tabletop/`. `tabletop` (the UR5's letter blocks) and
`tidy` (the SO-101 putting things into a tub) are each a config, a profile and
a few lines that hand them to it (`mrcf/tabletop/__init__.py`).

Its rules are its own (`profile.py`, below). A rule tuned for one task can
fail on another, and rules for one arm do not reach another arm doing the same
task. So each task on each arm is a deployment of its own.

### Its files

Create `deployments/<name>/` (or let `tools/new_deployment.py` copy one):

```
deployments/my_task/
├── __init__.py         REQUIRED — empty; makes the folder the package deployments.my_task
├── config.py           REQUIRED — its arm (ARM_NAME), its own settings; serves the arm's
├── skills.py           REQUIRED — installs the standard actions, adds its own
├── llm_controller.py   REQUIRED — entry point: a main() that runs the controller
├── perception.py       optional — its own perception, like chessboard.py
├── profile.py          how it drives its arm (its rules, each with why): mrcf/profiles.py
└── NOTES.md            what it has shown (What you learn, and where it goes)
```

A deployment of a task the framework has (the table task) is smaller: a
`config.py` that names its arm, sets what is its own and ends with
`tabletop.serve(globals())`, a `skills.py` that calls `tabletop.use(config)`,
and an `llm_controller.py` whose `main()` is the table task's. See
`deployments/tidy/`.

**A task planned in advance, and no planner.** A deployment that does one job
need not ask a language model how to do it. In its `config.py` (for a table
task, `mrcf/tabletop/settings.py` has the defaults):

```python
TASKS = {"tidy up": "put the red cap, the red ball, the white plug and the blue scoop into the plastic tub"}
PLANNER = False
```

Typing a task's name runs it (with one task, so do `go`, `run` and `start`). A
task is a sentence, which the table task turns into steps, or the steps
written out in the planner's format, with objects named the way anyone would:

```python
TASKS = {"scoop away": [{"action": "pick", "object_name": "the blue scoop"},
                        {"action": "place_in", "container": "the plastic tub"},
                        {"action": "home"}]}
```

Each object is found on the table when the task runs: by name, by its kind in
the catalogue, or else by looking for those words. `pick` picks it the way its
kind is picked. An object that is not found is left out, and the rest is done.
With `PLANNER = False`, nothing else is planned. A command that is none of its
tasks (and, for the table task, not a "put ... into ...") is answered with the
tasks it has. A failed step is reported, not planned again. Framework-wide,
these are `controller.Deployment.task_plan`, `task_names` and `planner`.
`tools/new_deployment.py` starts a copy with its own `TASKS` empty and the
planner on, with the source's kept commented out for reference.

A deployment is a package. Its modules import each other relatively
(`from . import config as cfg`, `from .perception import find_blocks`).
Elsewhere they are `deployments.my_task.config` and so on, so one task's
`config` never shadows another's, and the folder never needs to be on
`sys.path`. Name the folder as you would a Python module (letters, digits,
underscores).

`run.py` asks which arm, then offers its deployments: any folder with the four
required files whose `config.py` names that arm. It imports
`deployments.<name>.config` to check it loads, then calls `main()` in
`deployments.<name>.llm_controller` to run the controller.
`python -m deployments.my_task.llm_controller`, run from the repository root,
does the same without the picker.

**`config.py`** names its arm and holds what is true of the *deployment*. It
serves the arm's settings live, read from `robot_config` each time, never
copied at import. Some change while the controller runs: a scan that finds the
camera has moved re-poses it, and a copy would keep the old homography until a
restart:

```python
from pathlib import Path

from mrcf import backends, profiles

ARM_NAME = "so101"                           # its one arm
ARM = backends.arm(ARM_NAME)
PROFILE = profiles.load(Path(__file__).parent)   # profile.py, or None
_OWN = profiles.settings(PROFILE)

def __getattr__(name):                       # its rule if it has one, else the arm's setting, now
    if name in _OWN:
        return _OWN[name]
    if name in backends.RIG_SETTINGS or (name.isupper() and hasattr(ARM.config, name)):
        return getattr(ARM.config, name)
    raise AttributeError(name)

OBJECT_TYPES = ("cube", "cylinder")
TALLEST_OBJECT_M = 0.06
DISCOVER_GUIDANCE = "This is a tray of coloured blocks, viewed from above. ..."
WATCHERS: list = []      # see "Stopping the arm"
```

Do not copy hardware values into it. The port, cameras, measured poses,
homography, workspace bounds and the jaws' geometry belong to the arm.
Restating them here means a recalibration silently fixes one deployment and
not the other. How it drives the arm (how wide to open among its objects, how
hard to hold them, what to make of a look that disagrees) is the deployment's,
in its profile:

```python
# deployments/my_task/profile.py — names as its arm's robot_config spells them
# Objects stand apart here: open up to 80 mm, each pick narrowed to its neighbours.
GRIPPER_MAX_OPEN_M = 0.08
# Soft things: hold 3 % past contact, not chess's 10 (it crushed a wrapped sweet).
GRIPPER_HOLD_PCT = 3.0
```

The profile is laid over the arm's config wherever the deployment makes the
arm's env and policies (`ARM.make_env(profile=PROFILE)`,
`ARM.make_policies(profile=PROFILE)`), and recorded per arm in
`tests/baselines/<arm>/deployments.json`. A rule another deployment learned is
taken over only with its reason (`tools/new_deployment.py`). Its code reads
only the rig settings (`backends.RIG_SETTINGS`: camera names and sizes,
`TABLE_Z`, `WORKSPACE_BOUNDS`, the overhead camera's calibration, ...). Every
arm has those, so the same task on another arm (a deployment started from this
one) runs as it is. A name only one arm has belongs in the profile or the
backend.

**`skills.py`** installs the standard manipulation actions with the arm's
policies, and tells them what only the task knows:

```python
from mrcf import manipulation
from mrcf.policies.base import Vision

from .config import ARM, PROFILE, TALLEST_OBJECT_M

manipulation.install(
    ARM.make_policies(profile=PROFILE),   # the arm's policies, with this task's rules for it
    vision=Vision(),                      # the framework's measuring at the hover; or your own
                                          # (Vision.relocate: your scan, to find an object again
                                          # after a pick's miss pushed it)
    scenario=manipulation.Scenario(
        max_height=lambda obj: TALLEST_OBJECT_M,   # how tall an unmeasured object could be
        describe=lambda track: track.label or "object",
        picked=my_logger, placed=my_logger,        # optional
        approach_tilt=lambda obj: None,            # optional: how far to lean the gripper over obj
                                                   # (your heuristics; chess: config.APPROACH_RULES)
    ),
)
```

Then it registers anything else the task needs (chess adds `identify`,
`observe`, `answer`, ...).

**`llm_controller.py`** has the `main()` that `run.py` calls. It configures
the models and the safety layer, then runs the framework's controller
(`mrcf/controller.py`) with the task's hooks:

```python
from mrcf import controller

from . import skills  # noqa: F401 — registers the actions as a side-effect
from .config import ARM, WORKSPACE_BOUNDS, ...

def main():
    _safety.configure(bounds=WORKSPACE_BOUNDS, thresholds=SAFETY_THRESHOLDS,
                      sequence_validators=[manipulation.gripper_sequence_validator])
    _vlm.configure(model_server=MODEL_SERVER, model_url=MODEL_URL, model=MODEL, planning_model=PLANNING_MODEL,
                   observe_cameras=[OBSERVE_CAMERA], render_w=RENDER_W, render_h=RENDER_H,
                   work_plane_z=TABLE_Z, plan_few_shot=[...],
                   discover_extra_system=DISCOVER_GUIDANCE)

    env = ARM.make_env(profile=PROFILE)    # the arm's own construction, with this task's rules
    env.connect()
    try:
        controller.run(env, SceneState(memory=memory), controller.Deployment(
            name="my task", scan=my_scan, observe_pose=park_out_of_view, home=POLICIES.home,
            examples=["put the red cube in tray 2"], watchers=WATCHERS))
    finally:
        env.stop()

if __name__ == "__main__":      # python -m deployments.my_task.llm_controller
    main()
```

The loop does the rest. It perceives and decomposes the command, plans, and
runs `safety.validate_plan()` before any step. It dispatches step by step,
stops at the first failed step and re-plans from a fresh look, and catches
`MotionStopped`. A stop abandons the command, but the arm's own pauses (the
motion guard) only wait. Between commands, the arm stays where the last one
left it (`home_after_command=True` sends it home after each one). The scene is
looked at again only if the interlock has tripped since the last look (a hand
over the work surface, a stop). Until then, nothing but the arm has moved
anything. Every hook is optional. Without `scan`, it uses the vision model's
`discover_scene`. With a fixed camera, the observe pose can be
`env.go_park()`, which takes each arm to its own pose out of that camera's
view. `deployments/chess/llm_controller.py` is a complete example.

### Remembering objects

A scan says what is where now. `object_memory.ObjectMemory` says which of
those things is which. Hand each scan to `observe()` as `Detection`s, each
with a position on the work surface and whatever you measured (foot, height,
colour attributes, a label and who gave it). It returns one stable track per
object, kept when the object is moved by hand, marked missing when it leaves,
and recognised when it comes back. The standard actions report the arm's own
picks, places and measurements to it. Report what a person says with
`set_label(..., source="user")`.

```python
memory = ObjectMemory(Path(__file__).parent / "object_memory.json", namer=my_namer)
scene_state = SceneState(memory=memory)        # summary() then carries its recent events
tracks, changes = memory.observe([Detection(position=(x, y), place="tray_2", foot_mm=31.0,
                                            attributes={"colour": "red"})])
obj.track_id = tracks[0].id                    # on the SceneObject you build from it
obj.geometry = {"base_diameter_m": 0.031}      # what the policies read (manipulation.target_of)
```

Nothing in it knows about any task. `place` is whatever you call a location (a
chess square, a tray slot, or None), and labels and attributes are free text.
It is safe to share between threads.

With no perception of your own,
`scene_scan.observe(env, scene_state, memory, camera, ...)` does the whole
step. It outlines and measures whatever stands on the surface, and rebuilds
`scene_state.objects` from the memory, including `geometry` and
`measured_shape`. It needs the backend's `camera_model` and, on a patterned
surface, a photo of the empty surface (`surface_scan.save_reference`). Only
trust motion to a measured shape while `track.shape_current()` holds, because
the object may have been swapped in place since.
`deployments/chess/chessboard.py` (`discover`) is the worked example.

### Prompts

A small model may copy any example in a prompt word for word, instead of
looking at the image. Use placeholders in the schema:

```python
'Respond JSON only: {"pieces": [{"square": "XX", "colour": "white"}]}'
```

A literal `"d3"` there came back as the answer in 3 of 3 runs, on a board
whose only piece was on e5. Few-shot *pairs* (a user turn plus the matching
assistant turn) are fine. A bare answer template is what gets copied.

When the answer is one of a known set, give the model a JSON schema instead of
a template. `vlm.label(looks, {"type": [...], "colour": [...]}, prompt)` holds
each field to its list, and keeps an answer only where several looks agree.
`vlm.call_vlm(..., schema=...)` takes any schema. The planner's reply is
already held this way (`action_registry.plan_schema`): only registered
actions, and only names the scene has. Let the model name and choose, and
measure positions and sizes with the cameras' geometry.

---

## Adding an arm

Start from the template (or answer `n` to run.py's arm question):

```bash
python tools/new_arm.py my_arm     # mrcf/backends/my_arm/, a copy of mrcf/backends/_template
```

```
mrcf/backends/my_arm/
├── __init__.py       the arm's interface: NAME, config, make_env(), make_policies()
├── env.py            the RobotEnv — for an arm with cameras on its rig, a backends.rig.RigEnv
├── robot_config.py   port, cameras, measured poses, limits, gripper tuning — and the rig settings
└── NOTES.md          what this robot has shown (What you learn, and where it goes)
```

Scripts that measure and check what only this arm has go in `tools/my_arm/`,
not in the package (the SO-101's are in `tools/so101/`).

Then run the tools on it with `MRCF_ARM=my_arm`, and give it a deployment of
its own: a task it does on its layout, started from the nearest one
(`python tools/new_deployment.py my_table --from tidy --arm my_arm`). Run it
with `python run.py --arm my_arm` once its cameras have indexes and the
overhead camera is calibrated (until then the controller stops and says so).

As copied, it is an arm that exists only in memory, and keeps the backend
contract (`tests/test_template.py`). Every place to change is marked
`YOUR ARM`. Its `env.py` talks to the arm through a handful of SDK calls (its
`InMemoryArm`, each with the ur_rtde call it stands for). Replace that with
your SDK's client, then measure the rig ([HARDWARE.md](HARDWARE.md) for what
it needs, [COMMANDS.md](COMMANDS.md#first-time-setup-on-another-arm) for the
order).

Nothing in the framework names an arm. It is handed the env and the policies,
or asks `backends.arm()` for the one that is running. A deployment names only
its own arm (`ARM_NAME`, through `backends.arm(ARM_NAME)`), never the
backend's modules. `tests/test_architecture.py` enforces this.

### The arm's package

```python
# mrcf/backends/my_arm/__init__.py
from mrcf.backends.my_arm import robot_config as config
from mrcf.backends.my_arm.env import MyArmEnv

NAME = "My Arm"

def make_env(profile=None, **overrides) -> MyArmEnv:   # wired from config, a task's profile over it; connect() opens the arm
    ...

def make_policies(profile=None, **overrides):          # its policies.base.Policies, a task's profile over config
    ...

def describe() -> str:                       # optional: what it connects to, for the start-up line
    ...

def present() -> tuple[bool, str]:           # optional: plugged in? checked without connecting (run.py's arm menu)
    ...

def tour_and_calibrate(env) -> dict:         # optional: re-measure its fixed camera by itself
    ...
```

Its `config` defines the **rig settings**, `backends.RIG_SETTINGS`.
Deployments and the framework read these names without knowing which arm it
is:

- every camera and sensor on the rig (`CAMERAS`, `SENSORS`)
- the names of the two cameras with jobs (`CAMERA_NAME` on the gripper,
  `OVERHEAD_CAMERA_NAME` fixed over the surface), and their capture sizes and
  calibrations
- `RENDER_W`/`RENDER_H`, `TABLE_Z`, `WORKSPACE_BOUNDS`,
  `SURFACE_REFERENCE_PATH`
- `MOTION_GUARD_ZONE` and `MOTION_GUARD_CAMERAS`, `GRIPPER_TAG_ID`,
  `OVERHEAD_CALIBRATION_PATH`

A camera or calibration the rig does not have is `None`.
`tests/test_backends.py` checks that every arm defines them all. Everything
else in the config is the arm's own business.

`make_env()` and `make_policies()` take what every arm shares from the config
by name, through `mrcf/backends/wiring.py`. They list only what is the arm's
own, as the template's do:

```python
from mrcf import profiles
from mrcf.backends import wiring

def make_env(profile=None, **overrides):
    rc = profiles.layered(config, profile)                   # a task's rules for this arm over its facts
    kwargs = dict(wiring.camera_kwargs(rc, overrides),       # the cameras a RealRobotEnv opens, their sizes
                  **wiring.rig_kwargs(rc),                   # RigEnv's: cameras' jobs and calibrations, the tag,
                  sdk=..., home_xyz=rc.HOME_XYZ, ...)        #   the sentry, the jaw model, GRIPPER_BOX
    kwargs.update(overrides)
    return MyArmEnv(**kwargs)

def make_policies(profile=None, **overrides):
    return ParallelJaw(wiring.gripper_spec(profiles.layered(config, profile), **overrides))
```

`wiring.SPEC_SETTING_FOR` names the setting for each `GripperSpec` field
(`JAW_AXIS`, `JAW_SLACK_M`, `MEASURE_AT_HOVER`, ...). Set one in the config
and the policies use it. Leave it out and the policy's default stands. Put in
the config only what is true of the arm whatever task it does: its jaws, its
reach, what an empty close reads. How a task drives the arm belongs in that
task's profile (`deployments/<name>/profile.py`, see
[Adding a task](#adding-a-task)).

`JAW_AXIS` is read by both sides. The env turns the jaws that way on a
straight-down move (`RigEnv.down_rotation`), and the policies plan the room
around an object for it. That is why it is one setting, not two: when the UR5
had two that disagreed, it planned room on the wrong sides.

`GRIPPER_BOX` keeps the whole gripper inside a box (`mrcf/envelope.py`) on an
arm whose moves check it. Such an arm sets `CHECKS_GRIPPER_BOX = True` and
sends each move's path through `_refused_by_box`, as the template and the UR5
do. An arm that does not check one refuses to be given one.

Optionally, `prepare_gripper(ask) -> bool` gets the gripper ready before
`tools/check_motion.py` moves it (the UR5's activates its Robotiq, asking
first), and says whether to use it.

### Cameras and sensors

A rig can have any number of each, and the arm's config lists them all:

```python
from mrcf.backends.rig import Camera, Sensor

CAMERAS = {
    "wrist":    Camera(mount="gripper", source=0, size=(640, 480)),
    "overhead": Camera(mount="fixed", source=1, size=(1280, 720)),
    "side":     Camera(mount="fixed", source=2, size=(1280, 720),
                       model={...}, homography=[...]),     # tools/calibrate_overhead_camera.py --camera side
}
CAMERA_NAME, OVERHEAD_CAMERA_NAME = "wrist", "overhead"   # the two with jobs (None: none)
MOTION_GUARD_CAMERAS = ["overhead", "side"]               # a guard each

SENSORS = {
    "wrench": Sensor(kind="force", unit="N, N·m", shape=(6,), read="tcp_wrench"),   # an env method
}
```

`RigEnv` (`_init_rig(cameras=, sensors=, motion_guard_cameras=)`) does the
rest, the same on every arm:

- A camera without a job gets `camera_model()` and `pixel_to_world()` from its
  entry: a fixed camera from its calibration, one on the gripper from its pose
  there and the current joint angles.
- Every camera in `MOTION_GUARD_CAMERAS` is watched by its own guard while the
  arm is connected.
- Every sensor is read by `read_sensor()` and listed by `list_sensors()`. A
  `"force"` sensor becomes `get_contact_force()`, so the safety monitor's
  force limits work. A `"gripper_contact"` sensor is there for
  `gripper_loaded()` to read.

The flight recorder keeps every camera's frames (`list_cameras()`) and every
sensor's readings, and a stream's spec carries the sensors
(`streaming.sensor_specs(env)`). Whatever opens the cameras opens all of them:
`RealRobotEnv` from each `Camera`'s source and size (the template's
`make_env()`), and the SO-101 through LeRobot, alongside its own two.
HARDWARE.md says what each is for, and what a rig loses without it. The
contract checks that every sensor reads, at the shape it declares
(`check_sensors`).

A rig's own files (its calibrations, the photo of its empty surface, a
downloaded URDF, what its tools save to fit again) are kept out of the code
and out of git, in `rig/<backend>/` (`mrcf/rig_state.py`). `MRCF_HOME` moves
the folder that holds `rig/`. The config asks for its folder by its own name,
and refuses to start if any of them is still in its code folder:

```python
from mrcf import rig_state as _rig_state
RIG_DIR = _rig_state.rig_dir(_HERE.name, code_dir=_HERE)       # rig/<backend>/
SURFACE_REFERENCE_PATH = RIG_DIR / "surface_reference.png"
```

With a fixed camera, the config also points `mrcf/overhead_calibration.py`
(the live calibration each scan checks and re-poses) at the arm's own file
there:

```python
OVERHEAD_CALIBRATION_PATH = RIG_DIR / "overhead_calibration.json"

from mrcf import overhead_calibration as _overhead_calibration
_overhead_calibration.PATH = OVERHEAD_CALIBRATION_PATH
_overhead_calibration.apply_to(globals())      # the latest measurement over the values above
```

### Which base class

| | Use when | Example |
|---|---|---|
| **`backends.rig.RigEnv`** | The arm has a two-finger gripper and cameras on its rig: a wrist camera, a fixed one over the table, and a tag on the gripper. You give it the gripper's pose from your kinematics (`gripper_frame()`) and drive the motors. It gives you, as on the SO-101: `camera_model` for both cameras, the fixed camera's `pixel_to_world`, the grasp point and held-width model behind `get_ee_pos`, the gripper tag's sighting (`overhead_gripper_offset`), the wrist servo (`visual_servo_xy`), `wait_still`, re-posing the fixed camera while running, and the sentry on connect. | `mrcf/backends/so101/env.py` |
| **`RealRobotEnv`** | You want the framework to own the cameras: OpenCV capture (each camera on a reader thread that keeps its newest frame), calibration files, back-projection and serial discovery. It combines with `RigEnv` as `class MyArmEnv(RigEnv, RealRobotEnv)`. `RealRobotEnv` captures, and `RigEnv` does the geometry. | `mrcf/backends/_template/env.py` (in memory; no real arm yet) |
| **`RobotEnv` directly** | Anything else: a simulator, or an embodiment that does not grasp. | `tests/fake_arm.py` |

`SO101Env` is a `RigEnv` but not a `RealRobotEnv`, because LeRobot opens its
cameras and runs their reader threads. It reimplements `render_image` and the
lifecycle methods. Around a plain SDK, `RigEnv` + `RealRobotEnv` does most of
the backend's work.

### What you must implement

`RobotEnv` requires `step`, `get_ee_pos`, `get_joint_angles`, `get_body_pos`,
`get_movable_body_names`, `get_static_body_names`, `render_image` and
`pixel_to_world`. `RigEnv` provides `get_ee_pos` and the fixed camera's
`pixel_to_world` from `gripper_frame()`.

The connection: every program calls `connect()` after `make_env()` and before
the arm first moves, and `stop()` once at the end. `stop()` is called after an
error too, so it must cope with an arm that connected only partly. Both do
nothing by default. `connect()` opens the hardware and must not move the arm.
On a `RigEnv`, the sentry starts as the arm's own `connect()` returns and
stops as its `stop()` begins, with nothing for the arm to do.

Optional, for arms with a gripper: `move_ee_to`, `set_gripper`,
`get_gripper_width`, `squeeze_gripper`, `go_home` and `go_park` (out of the
fixed camera's view, for a scan; home by default). These raise
`NotImplementedError` by default, so an embodiment that does not grasp leaves
them and registers different actions. The policies pass:

- `move_ee_to` `tol`, `orientation` (one held through a sequence of moves,
  from `OrientationSearch`) and `max_step_deg` (go gently: on the SO-101, the
  most a joint moves in a cycle, which an arm that commands speeds reads as a
  fraction of its usual step)
- `set_gripper` `n_steps` and `max_step_pct` (close gently)

On a `RigEnv`, `set_gripper` also sets `self._held_width_m = None` as the jaws
let go.

Optional, for the policies to do better: the extras in `mrcf/capabilities.py`,
such as choosing one orientation for a whole move (`OrientationSearch`),
sensing a grip by load (`GripSensing`), seeing the gripper from a fixed camera
(`OverheadSighting`) or servoing from a wrist camera (`WristServo`). Each is a
`Protocol`. A backend has one by having its methods, and a policy checks
`isinstance(env, OverheadSighting)` before using it. `RigEnv` provides the
camera ones. `OrientationSearch`, `GripperTilt` and `GripSensing` are the
arm's own.

Optional, for the tools: `disable_torque()` and `enable_torque()`, which let
you move the arm by hand (torque off on the SO-101, teach mode on a UR).
`tools/calibrate_overhead_camera.py` uses them, and asks for the arm's own
hand-guiding when they are missing.

Optional, for measuring objects: `camera_model(camera, width, height)`, the
camera's lens and its pose in the world right now (a fixed camera's from its
calibration, a wrist camera's from the joint angles). With it,
`plane_geometry` and `surface_scan` measure objects from that camera on any
surface. The SO-101 fits its overhead camera with
`tools/calibrate_overhead_camera.py` and its wrist camera with
`calibrate_gripper_pose.py` (`mrcf/perception/hand_eye.py`). If the arm's
frame reads short or long of the real world, say so in `frame_scale`. `RigEnv`
also provides `camera_model_at(camera, width, height, at, rotation)`
(`capabilities.CameraAtPose`): where a wrist camera would be with the jaws at
a pose, with nothing moved. A pick plans its looks for an object's height with
it (`GripperSpec.height_from_views`).

Get these contracts right. Each has caused a bug in this codebase:

- `get_ee_pos()` must return the point the gripper actually closes on, not a
  convenient link origin. On the SO-101 the two are 8.5 mm apart, which is
  enough to close the jaws beside a chess piece every time. With one jaw
  moving, the point also slides as the jaws open (`RigEnv._tcp`).
- `move_ee_to()` must return whether it arrived. Callers branch on it.
- Call `self.guard()` before every command you send the motors, and implement
  `hold()` to stop the arm where it is. That lets a trip of the interlock stop
  the arm within one control cycle, mid-move. Route every command through one
  function that guards (`SO101Env._send`). A single unguarded loop is a move
  nothing can stop. Do not hold the bus lock while `guard()` waits out a
  pause. The watcher that ends the pause reads the arm through that lock, so
  the pause never ends (this happened on the SO-101).
- `render_image()` and `camera_model()` must be safe to call from another
  thread while the arm moves, because watchers call them. Do not read frames
  through a call that also talks to the motor bus, and put the bus under a
  lock.
- A frame must be the camera's newest, never one its driver queued before the
  last move. The wrist servo and the motion guard act on it as if it showed
  the present.

### Checking it before it moves: the contract

`mrcf/backends/contract.py` drives an offline instance of your env, with its
motors and cameras faked, the way the framework does. It fails on any of the
problems above that it can see from outside:

- a `connect()` or `stop()` that takes arguments, or that, on a `RigEnv`, does
  not start and stop the sentry
- a move that does not say whether it arrived
- a command that runs with the interlock tripped
- a pause that raises, or does not carry on
- frames of the wrong size
- a camera model that is not a rigid pose
- a keyword argument the policies pass that the env does not take
- a read of the arm from another thread that fails mid-move

```python
from mrcf.backends.contract import check_backend

def test_my_arm_keeps_the_contract():
    found = check_backend(make_offline_env, reachable=[(0.40, 0.10, 0.05)], unreachable=[(1.2, 0.0, 0.1)])
    print(found)        # the capabilities it has
```

`tests/test_backends.py` runs it on the SO-101 (offline) and the test arm.
`tests/test_rig.py` runs it on a ~60-line `RigEnv` arm whose controller moves
the grasp point itself, as a UR's does, and `tests/test_template.py` on the
template, the starting point. The contract cannot check the arm itself: that
`get_ee_pos()` is where the jaws really close, or that `hold()` really stops
the motors. The arm's own tools check that, on the rig.

### The arm's policy library

Give the backend a `make_policies()` that returns its
`policies.base.Policies`. An arm with a two-finger gripper working from above
can use the framework's `policies.parallel_jaw.ParallelJaw` with its own
`GripperSpec`: its jaw geometry, how fast to close, how to tell a grip from a
miss, and the heights it can reach straight down
(`mrcf/backends/so101/__init__.py` is the example). A different end-effector,
such as suction or a hand, implements `Policies` itself, taking the same
Targets, so every deployment's actions work on it unchanged.

Where an object stands changes how the arm picks it. `ParallelJaw` approaches
an object in different ways close to the base, in the middle of the reach, and
far out (`near_base`, `top_down`, `reach_out`; see `Approach` in
`mrcf/policies/approaches.py`). A new way to approach an object is a new
`Approach` in `GripperSpec.approaches`. Its `align` list of strategies says
how it lines the jaws up over the object once there
(`mrcf/policies/aligners.py`: the foot a located wrist camera finds, the fixed
camera's sighting, the wrist servo, ...). A new strategy is an `Aligner`
subclass given to `register_aligner()`. The pick's phases (hover, align,
measure, descend, close, lift) are `ParallelJaw` methods that share a
`PickContext`. `tests/test_pick_paths.py` shows a custom approach end to end.

Give the env a `base_xy` (`RobotEnv.base_xy`, the axis the arm swings about),
and the spec the zone edges (`near_base_m`, `far_reach_m`) and
`base_keep_out_m`, from your own arm's reach map. Without them, every object
counts as "mid-reach" and gets `top_down` first. (Close to the base it gets
`top_down` first anyway, wherever that reaches.) An env with
`approach_orientation_for(..., margin_deg=)` lets `reach_out` keep its joints
off their limits.

### Then

The tools in `tools/` work on whichever arm is running. They cover:

- the cameras (`check_camera.py`, `check_exposure.py`)
- the overhead camera's calibration (`calibrate_overhead_camera.py`,
  `tag_tour.py`, `check_overhead_correct.py`, `check_targeting.py`)
- the tags (`make_tag_sheets.py`, `check_tags.py`)
- the sentry (`check_motion_guard.py`)
- the empty surface (`capture_surface_reference.py`)
- the models (`check_perception.py`)

They reach the arm through `tools/_common.py` (`backends.arm()`'s
`make_env()`, `config` and `make_policies()`) and the env's public methods. A
test stops them from naming an arm.
[COMMANDS.md](COMMANDS.md#first-time-setup-on-another-arm) runs them in order
on a new arm.

What only one arm has goes in its own `tools/`. For the SO-101 that is its
poses, servos, joint offsets, gripper width table and its wrist camera's
calibrations (`calibrate_gripper_lens.py`, `_pose.py` and `_servo.py`, which
move it in its own joint space, and `check_lighting.py`, at its servo's pose).
An arm's own tools copy the shape of `tools/so101/_common.py`: the arm's
`make_env()`, plus the shared `deployment_config()` and `task_config()`, which
keep robot-level tools from hardcoding one task's numbers.

### Worked example: a UR5

What follows is the plan for this codebase's second arm. Everything above the
backend carries over unchanged: the policies, chess, perception, the object
memory, the controller, the motion guard and the overhead camera's self-check.
The work is `mrcf/backends/ur5/`.

It is now written (a CB3 on PolyScope 3.15) and runs on a real arm: since
2026-11-12 it has stacked letter blocks (`deployments/tabletop`) and put hand
tools into a box (`deployments/toolbench`). It also runs offline, on
`offline.py`'s UR5, and on URSim 3.15. The plan below was written for a
Robotiq 2F-85. The arm has a Robotiq Hand-E instead (52 mm stroke,
`GRIPPER_MAX_OPEN_M`, `JAW_SLIDE_AXIS` in its `robot_config.py`), and the
backend differs from the plan in four more ways:

- `go_home()` and `go_park()` go to measured joint poses (`HOME_JOINTS_DEG`)
  and refuse until those are set.
- A move that turns the jaws turns them in place first.
- `set_gripper` waits for the fingers to stop.
- The controller is behind `arm.URArm`, the one file that names ur_rtde's
  calls.

Its NOTES.md lists what is still to check on the arm.

Start from the template (`cp -r mrcf/backends/_template mrcf/backends/ur5`).
Its `InMemoryArm` has this table's calls, one method each, and its
`TemplateEnv` is already the `RigEnv` + `RealRobotEnv` below.

**`env.py`: `class UR5Env(RigEnv, RealRobotEnv)`.** `RealRobotEnv` opens the
cameras. `RigEnv` provides their geometry, the wrist servo and the gripper
tag. `UR5Env` talks to the controller over Ethernet with
[ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/) (`RTDEControlInterface`,
`RTDEReceiveInterface`). Poses are `[x, y, z, rx, ry, rz]`, ending in a
rotation vector that `cv2.Rodrigues` turns into a matrix. Keep it under one
lock: the control interface is not safe from two threads, and `hold()` comes
from the tripping thread.

| What the framework calls | On a UR5 | Notes |
|---|---|---|
| `gripper_frame()` | `getActualTCPPose()` as a 4x4, with the controller's TCP at the flange (`setTcp([0]*6)`) | Leave the jaws' offset to `RigEnv` (`tcp_offset`, `jaw_centre_slide` 0: both fingers of a Robotiq 2F-85 move), so the jaw model lives in one place. |
| `move_ee_to(target, tol, orientation, max_step_deg)` | `guard()`, then `moveL(pose, speed, accel, asynchronous=True)`, then every cycle `guard()` and `getActualTCPPose()` until within `tol` or `getAsyncOperationProgress()` says it ended | No IK loop or stall window: the controller plans the line. `orientation` None: straight down, the jaws turned to the arm's radial (or tangential) axis, as `tilted_rotation(0, target)` does on the SO-101. `max_step_deg` means a slower `speed`. Return False when `getInverseKinematics` finds no solution or `isPoseWithinSafetyLimits` says no. |
| `hold()` | `stopL(a)` | A UR slows to a stop by itself. No need to command it again. |
| `may_be_moving()` | `not isSteady()` | |
| `get_joint_angles()` | `getActualQ()` | Already in radians. |
| `set_gripper`, `get_gripper_width`, `squeeze_gripper`, `gripper_loaded` | A Robotiq gripper's URCap socket (port 63352): `POS`, `SPE`, `FOR`. `OBJ` says whether it stopped on an object | `gripper_loaded` from `OBJ`, not a servo's load. `squeeze_gripper` can stay the default. |
| `approach_orientation_for` | Straight down is reachable over almost all of the 0.85 m reach. Return `(rotation, 0.0)` when `getInverseKinematics(pose, qnear)` solves every waypoint | Six joints need no lean in the arm's plane, so `GripperTilt` is 0 and the SO-101's lean retries never run. |
| `connect()`, `stop()` | Open `RTDEControlInterface` and `RTDEReceiveInterface` (and the gripper's socket). Stop with `stopScript()` and `disconnect()` | `RigEnv` starts the sentry after `connect()` and stops it before `stop()`. |
| `disable_torque()`, `enable_torque()` | `teachMode()`, `endTeachMode()` | Hand-guiding, for the touches in `tools/calibrate_overhead_camera.py`. |
| `get_contact_force()` | `getActualTCPForce()` (the norm of its force part) | The safety monitor's force limits then work, which they cannot on the SO-101. It is rough on a CB3 (from joint currents). An e-Series has a sensor. |
| `arm_shape()` | The links' frames from the UR5's DH parameters | ur_rtde gives the TCP, not the links, and the motion guard needs the body. |
| `go_home()`, `go_park()` | `moveJ` to measured joint poses | |
| `stream_spec`, `stream_observation()`, `stream_action()` (`ActionStreaming`) | In: joints from `getActualQ()` and the gripper's position. Out: joint targets through `servoJ` (or the TCP pose in and `servoL` targets out). At 30 Hz or more | The UR5 has its own stream, so models trained on the SO-101 are refused on it, not run wrongly. Its policies are trained elsewhere, on demonstrations with these features. |

A protective stop (`isProtectiveStopped()`) is the arm's own safety system
stopping it. `move_ee_to` then raises `MotionStopped` with that as the reason,
as for a trip. Set the UR's safety planes on the pendant to match
`WORKSPACE_BOUNDS`, so the two stop at the same place.

**`__init__.py`: `make_policies()`** returns `ParallelJaw` with the 2F-85's
`GripperSpec`: `max_open_m` 0.085, its fingertips' depth and width,
`grasp_step_deg`/`close_step_pct` as fractions of speed,
`descent_lean_retries` 0, `near_base_m`/`far_reach_m` None (every object is
mid-reach until a UR5 reach map says otherwise), and `servo_tilt_deg` None.

These stay with the SO-101: servo register tuning, the IK loop's deadband lead
and stall window, `JOINT_OFFSETS_DEG`, the gripper width table, grip sensing
by load, the lean retries, the radial/tangential lean geometry, and
`ARM_FRAME_SCALE` (a UR's frame is exact). Each exists because of hobby servos
and a five-joint wrist.

These are measured again on the new rig, in the order of
[COMMANDS.md](COMMANDS.md#first-time-setup-on-another-arm):

- `TABLE_Z` (the closed jaws on the table)
- `WORKSPACE_BOUNDS` and `MOTION_GUARD_ZONE`
- the park pose
- the overhead camera: `tools/calibrate_overhead_camera.py` touches the
  board's corners in freedrive, which also fits its lens, and then
  `tools/tag_tour.py` refines it (it needs that lens)
- the gripper tag's pose
- the empty-board photo

The wrist camera's lens and pose, and the wrist servo's pixel and Jacobian,
are measured again too, once their calibrations are ported from the SO-101's
tools. Until then, the policies leave the wrist camera out.

Before the real arm, run the contract on an offline `UR5Env` (ur_rtde's
interfaces faked). Then use URSim, Universal Robots' simulator, which speaks
the same RTDE protocol. `check_backend` against it moves, stops and pauses a
simulated UR5 through the real backend. Then the real arm, slowly, with the
pendant's speed slider low.

---

## Stopping the arm, and running things alongside it

Every env has an **interlock** (`mrcf/interlock.py`). Anything that should
stop the arm trips it, from any thread:

```python
env.interlock.trip("hand near the arm", source="hands")
env.interlock.clear("hands")
```

A trip holds the arm where it is at once, from the thread that tripped it, if
the arm was commanded in the last second. An arm at rest is left alone,
because holding it again where it has sagged to would creep it down. The
backend also checks the interlock before every command, so every motion
command raises `MotionStopped` until each source that tripped it has cleared.
The safety monitor's limits (the workspace floor and height ceiling) are
checked the same way, every cycle, while an action runs.

A **watcher** (`mrcf/watchers.py`) runs a function on a camera's newest frame,
on its own thread, at a rate you choose, while plans and moves carry on. A
deployment lists the watchers it wants in `config.WATCHERS`, and the
controller starts them before the arm first moves:

```python
from mrcf.watchers import Watcher

def hand_guard(env):
    def look(frame, model):
        if hand_near_arm(frame, model):          # your detector
            env.interlock.trip("hand near the arm", source="hands")
        else:
            env.interlock.clear("hands")
    return Watcher(env, "overhead", look, hz=10, fail_safe=True).start()

WATCHERS = [hand_guard]
```

With `fail_safe=True`, a watcher that crashes or loses its camera trips the
interlock itself until it recovers. A safety watcher that cannot see is not
watching. The time to stop is the watcher's period plus its function's run
time, so keep a safety watcher's function fast.

The framework provides one: `MotionGuard` (`mrcf/motion_guard.py`) trips the
interlock when anything other than the arm moves in a calibrated fixed
camera's view. It needs the backend's `camera_model()` for that camera and
`arm_shape()` (where the arm's body is, from its joint angles), and refuses to
start without them. Give it a `zone`: the part of the work surface to watch,
as world (x, y) corners. Without one it watches everything in view, including
the floor beside the table, where shuffling feet read as movement on the
surface. Its trips pause the move, which carries on once nothing has moved for
2 s (`resume=True`).

The SO-101 backend starts one itself on every `connect()` (`robot_config`'s
`MOTION_GUARD` and `MOTION_GUARD_ZONE`), so chess's `WATCHERS` is empty. On a
backend that does not, a deployment starts it like any other watcher, but only
one per camera: two guards share the interlock source `"motion"`, and each
would clear the other's trips.

```python
def start_motion_guard(env):
    from mrcf.motion_guard import MotionGuard
    return MotionGuard(env, OVERHEAD_CAMERA_NAME, zone=MOTION_GUARD_ZONE).start()

WATCHERS = [start_motion_guard]
```

From its thread, a watcher may read cameras, trip and clear the interlock, and
update the object memory. It must not move the arm, because only one thread
drives it. A watcher feeding the memory from the overhead camera while the arm
is over the surface sees the arm, not what is under it. Skip those frames, or
mask the arm out.

---

## What the framework provides

These work on any robot. Do not fork them for a deployment.

| Module | What it gives you |
|---|---|
| `mrcf/robot_env.py` | `RobotEnv`, the interface every backend implements, with its stop (`interlock`, `guard`, `hold`) |
| `mrcf/interlock.py` | `Interlock`, `MotionStopped`: stop the arm from anywhere, mid-move |
| `mrcf/watchers.py` | `Watcher`: perception on its own thread, alongside the arm; fail-safe for safety watchers |
| `mrcf/capabilities.py` | What a backend can do beyond `RobotEnv`, one `Protocol` each: the policies' optional extras |
| `mrcf/backends/__init__.py` | `arm()` (the arm that runs, `MRCF_ARM`), `available()`, `RIG_SETTINGS` (the settings every arm defines) |
| `mrcf/backends/rig.py` | `RigEnv`: an arm's cameras and gripper geometry, from its kinematics (camera models, the grasp point, the gripper tag, the wrist servo, the sentry) |
| `mrcf/backends/contract.py` | `check_backend()`: what every arm's env must do, checked on an offline instance before it drives the real arm |
| `tests/arm_baselines.py`, `tests/baselines/` | Each arm as the shared code builds it (settings, capabilities, plans for standard scenes), recorded, so a change made for one arm that changes another fails (`tests/test_arm_baselines.py`) |
| `mrcf/streaming.py` | `StreamSpec`: what an arm's stream carries, and so what its models were trained on |
| `mrcf/policies/rl.py` | `VectorCodec` (how a vector policy reads and drives a stream), `SB3Model` (a trained Stable-Baselines3 policy as a `learned.Model`) |
| `mrcf/policies/learned.py`, `mrcf/policies/lerobot_model.py` | `LearnedPolicy` (a trained model run on the arm's stream for one segment of a skill), `chain`, `hover_over`, and `LeRobotModel` (a LeRobot checkpoint: ACT, diffusion, SmolVLA, pi0, SAC) |
| `mrcf/overhead_calibration.py` | A fixed camera's calibration as live state: checked at every scan, re-posed from one frame, a tour when the arm disagrees |
| `mrcf/rig_state.py` | `rig_dir(backend)`: where a rig's own files live (`rig/<backend>/`, under `MRCF_HOME`), out of the code and out of git |
| `mrcf/policies/` | `Target`, `Obstacle`, `Outcome`, `Vision` and the `Policies` interface. `parallel_jaw`: pick, place, move_above and home for a two-finger gripper (and `height_looks`: looking for an unmeasured object's height first). `approaches` and `aligners`: the ways it comes at an object and lines up over it |
| `mrcf/manipulation.py` | The standard actions (pick, place, move_above, home, open/close_gripper), from the world model to the policies and back. `object_action` for a verb of your own. `gripper_sequence_validator` |
| `mrcf/scene_state.py` | `SceneObject`, `SceneState`: the world model the planner works from |
| `mrcf/action_registry.py` | `register_action`, `register_condition`, `dispatch`: the actions a planner may use |
| `mrcf/vlm.py` | Every model call: `perceive`, `discover_scene`, `decompose_task`, `plan` (re-exports the two above) |
| `mrcf/model_server.py` | Where the models run: `Ollama`, `OpenAICompatible`, or a plugin's `ModelServer` (`mrcf.model_servers` entry points). Chosen with `MRCF_MODEL_SERVER` and `MRCF_MODEL_URL` |
| `mrcf/controller.py` | `run(env, scene_state, Deployment(...))`: the command → plan → safety → dispatch → recover loop that every deployment shares |
| `mrcf/object_memory.py` | `ObjectMemory`: one identity per physical object across scans, the arm's own moves and restarts, labels ranked by trust, and what to tell the planner |
| `mrcf/perception/plane_geometry.py` | `PlaneView`, `measure`, `outline_objects`, `measure_near`: an object's foot (position, width), height and width profile from its outline, for any camera that knows where it is relative to the work surface |
| `mrcf/perception/shape_fit.py` | An object's height from its outlines in several calibrated views at once: a solid (straight sides, rounded top, dome, ball, taper) fitted on its footprint, and how settled the fit is |
| `mrcf/perception/segment.py` | SAM2 outlines of whatever is under a set of points, any colour, any background (optional dependency) |
| `mrcf/perception/surface_scan.py` | `scan`, `change_map`, references: every object standing on the surface, found and measured with no board or object list, and told apart from the surface in any light |
| `mrcf/scene_scan.py` | `observe`: a surface scan, matched to the memory and put in the world model |
| `mrcf/perception/fiducials.py`, `mrcf/perception/lighting.py` | Tags and bullseyes (where a camera, gripper or board is). Whether the light is good enough, area by area, in words |
| `mrcf/perception/outlines.py`, `mrcf/perception/tones.py` | One outline per object out of a segmenter's many, and an outline's middle. Where a colour falls between two of the surface's own colours, whatever the exposure |
| `mrcf/perception/hand_eye.py` | Where a wrist camera sits on the arm, from views of a flat grid: the maths behind a backend's camera calibration |
| `mrcf/skill_executor.py` | `run_policy()` / `run_skill()`: the safety monitor, and a failure description |
| `mrcf/safety.py` | `configure()`, `validate_plan()`, `ExecutionMonitor`, `bounds()`: the work volume |
| `mrcf/profiles.py` | A deployment's rules for its arm (`profile.py`), laid over the arm's facts |
| `mrcf/tabletop/` | The table task (find, name, pick, place, stack, put into), which a deployment hands its config to (`use`, `serve`). `tabletop` and `tidy` are two |
| `mrcf/flight_recorder.py` | The timeline, incidents (with tracebacks), `session.json`, `console.log`, `timings.json`; `python -m mrcf.flight_recorder [--list | --timings]` |
| `mrcf/privacy.py` | Guards every kept picture: a calibrated camera's frames are limited to the work area (`frame_guard`) and faces are covered (YuNet: `tools/fetch_face_model.py`). `imwrite`, `save_rgb` |
| `mrcf/devices.py` | This computer's cameras and ports, by name, without opening any. `check_sources` refuses to open a camera that is the computer's own |
| `mrcf/rig_report.py` | What an arm brings to a job (plugged in, gripper, cameras, sensors, pictures guarded), reported before a run and kept with it |
| `run.py` | The launcher. It asks for the arm (or a new one), shows what it has, offers its deployments (or a new one), and starts the run's recording |

---

## Checklist

**New capability**
- [ ] The policy takes geometry, returns an `Outcome`, and checks what `move_ee_to` returned
- [ ] The policy imports nothing from the world model, memory or planner (`tests/test_architecture.py`)
- [ ] The action uses `manipulation.object_action`, or is registered with `@register_action`,
      runs the policy through `run_policy()`, records the outcome, and returns `outcome.ok`
- [ ] Any non-standard condition is registered with `register_condition()`
- [ ] The `description` is written for a model choosing between actions

**New deployment**
- [ ] It has an `__init__.py`, and its modules import each other relatively (`from . import config as cfg`)
- [ ] `config.py` names its one arm (`ARM_NAME`, `backends.arm(ARM_NAME)`) and serves hardware settings live from it,
      without restating or copying them. Its code reads only `backends.RIG_SETTINGS` (no arm's own names)
- [ ] `skills.py` calls `manipulation.install()` with the arm's `make_policies(profile=PROFILE)` (or, for a table
      deployment, `tabletop.use(config)`), and does not import `backends.<arm>`
- [ ] It is a new deployment, not more of an existing one ("A new deployment, or more of one there is?"). It started
      from the nearest (`tools/new_deployment.py NAME --from SOURCE [--arm ARM]`), and each rule in its `profile.py`
      came over with its reason. Code shared with another deployment is in the framework, not copied
- [ ] `llm_controller.py`'s `main()` calls `_safety.configure()` and `_vlm.configure()`, then `controller.run()`
      with the task's `Deployment` (its scan, observe pose, home and `WATCHERS`)
- [ ] Prompt examples use placeholders, not plausible literal values
- [ ] No imports from another deployment

**New arm**
- [ ] Started from `mrcf/backends/_template` (`tools/new_arm.py`), with every `YOUR ARM` replaced or confirmed
- [ ] `mrcf/backends/<arm>/__init__.py` exports `NAME`, `config`, `make_env()`, `make_policies()`
- [ ] Its config defines every `backends.RIG_SETTINGS` name (`tests/test_backends.py`)
- [ ] `check_backend()` passes on an offline instance of its env (`mrcf/backends/contract.py`)
- [ ] Its `make_env()` builds it offline with nothing connected, and its settings, env and plans are recorded (`MRCF_UPDATE_BASELINES=1 pytest tests/test_arm_baselines.py -k <arm>`; CONTRIBUTING.md, "A change for one arm")
- [ ] `connect()` opens the hardware without moving the arm, and `stop()` closes it, even half-connected
- [ ] `MOTION_GUARD = True` once the overhead camera is calibrated and `tools/check_motion_guard.py` agrees
- [ ] `get_ee_pos()` returns the grasp point
- [ ] `move_ee_to()` returns whether it arrived
- [ ] Every motor command goes through `guard()`, and `hold()` stops the arm where it is, from any thread
- [ ] `may_be_moving()` says when a trip should hold it at once (default: always)
- [ ] `arm_shape()` gives the arm's body from its joint angles, for the motion guard
- [ ] The extras in `mrcf/capabilities.py` that the arm can do are implemented, with their signatures
- [ ] `render_image()` and `camera_model()` are safe to call from another thread while it moves, and each frame is the newest
- [ ] `go_park()` takes it out of the fixed camera's view
- [ ] For learned policies: `ActionStreaming`, with its stream's spec, observation and action, through its own safety checks
- [ ] `robot_config.py` holds every measured value, and nothing is hardcoded in `env.py`
- [ ] `make_env()` and `make_policies()` are exported, so tools and deployments share one construction
- [ ] The rig is measured in the order of COMMANDS.md, "First-time setup on another arm". What only this arm
      needs (its poses, its reach) is measured with its own `tools/`
