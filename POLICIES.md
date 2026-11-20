# Adding a policy

A **policy** is how the arm does one thing: pick, place, push, or stand a
fallen piece up. The planner never moves the arm itself. It asks for a verb
("slide the white pawn"), and the verb runs a policy on that object's geometry.
This guide takes a new policy from an idea to a verb the planner can ask for,
on any arm.

- [Planned or learned?](#planned-or-learned)
- [A planned policy, step by step](#a-planned-policy-step-by-step)
- [A learned policy, step by step](#a-learned-policy-step-by-step)
- [Your own model format](#your-own-model-format)
- [When it is refused, or fails](#when-it-is-refused-or-fails)
- [Reference](#reference): the stream, how a model is run, reinforcement learning, packages, plugins
- [Checklists](#checklists)

[DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#how-the-pieces-fit) explains how
policies fit with the world model, the planner and the safety layer. One rule
holds throughout: **a policy is given geometry and returns an `Outcome`. It
never sees the world model, the object memory or the planner.**
`tests/test_architecture.py` checks this.

---

## Planned or learned?

| | Planned | Learned |
|---|---|---|
| What it is | Code that works out the moves from geometry: where the object is, how big, what stands next to it | A trained model run on the arm's stream, a tick at a time: what the arm sees and feels goes in, a goal for its joints comes out |
| Good for | Anything you can describe in geometry: pick, place, push, hover, stand up | Contact that is hard to write down: sliding, regrasping, anything a demonstration shows better than a formula |
| Where it comes from | You write it, in this repository | Trained **elsewhere** (on demonstrations or by reinforcement learning) and only run here |
| Test it without the arm | Yes: `tests/fake_arm.py` | Its fit and plumbing, yes. Its behaviour, only on the arm |
| Examples | `mrcf/policies/parallel_jaw/` (pick, place), `mrcf/policies/stand_up.py` | an ACT checkpoint from `lerobot-train`; a Stable-Baselines3 network |

Most skills use both. A planned move takes the arm to where the learned part
starts, the learned part handles the contact, and a fresh look shows what it
did (`chain()`, [step 6](#6-make-it-a-verb)).

This repository deliberately has no simulator, recorder or trainer, so it stays
light enough to run on the robot. Training happens on another machine, and the
trained model comes back as a folder.

---

## A planned policy, step by step

The example is `slide`, which pushes an object a couple of centimetres without
grasping it. MRCF already has a real verb that does this, `nudge`
(`mrcf/policies/nudge.py`). The example uses another name so you can follow
it step by step without touching that one.

### 1. Decide where it lives

- `mrcf/policies/` if any arm with that kind of gripper could use it. Code
  there talks to the arm only through `RobotEnv` (`mrcf/robot_env.py`) and the
  optional extras in `mrcf/capabilities.py`.
- `mrcf/backends/<arm>/` if only one arm could.

### 2. Write it

A policy is a function, or a method of the arm's policy library. It takes the
env and the object's geometry and returns an `Outcome`:

```python
# mrcf/policies/slide.py
import numpy as np

from mrcf.policies.base import Outcome, Target


def slide(env, target: Target, obstacles=(), direction=(1.0, 0.0), distance_m: float = 0.02,
          table_z: float = 0.0, clearance_m: float = 0.03) -> Outcome:
    """
    Push an object distance_m along direction (x, y) without grasping it: the
    closed jaws come down beside it, push it low on its body, and lift away.
    """
    height = target.height_m or target.max_height_m
    if height is None:
        return Outcome(False, "not knowing how tall it is, the jaws could land on it", refused=True)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    xy = np.asarray(target.position[:2], float)
    start = xy - d * ((target.foot_m or 0.03) / 2 + 0.015)       # beside it, on the side it is pushed from
    for ob in obstacles:                                         # never come down on a neighbour
        if np.linalg.norm(np.asarray(ob.position[:2], float) - start) < (ob.radius_m or 0.015) + 0.01:
            return Outcome(False, f"{ob.label or 'something'} stands where the jaws would come down",
                           refused=True)
    above, low = table_z + height + clearance_m, table_z + min(0.5 * height, 0.02)
    end = start + d * distance_m
    env.set_gripper(0.0)
    if not env.move_ee_to(np.r_[start, above]) or not env.move_ee_to(np.r_[start, low]):
        env.move_ee_to(np.r_[env.get_ee_pos()[:2], above])
        return Outcome(False, "could not get down beside it")
    pushed = env.move_ee_to(np.r_[end, low])
    env.move_ee_to(np.r_[env.get_ee_pos()[:2], above])
    if not pushed:
        return Outcome(False, "stopped short while pushing")
    return Outcome(True, f"pushed it {distance_m * 1000:.0f} mm", at_xy=xy + d * distance_m)
```

What it is given (`mrcf/policies/base.py`):

| | |
|---|---|
| `Target.position` | (x, y, z): the centre of the object's foot, on the surface, in the arm's frame, metres |
| `Target.foot_m`, `height_m`, `widths_mm` | what a scan measured: foot diameter, height, and width up the object (`None`: not measured) |
| `Target.max_height_m` | height not measured: the tallest it could be (the task says) |
| `Target.lying` | fallen on its side: which way it lies (`mrcf/perception/lying.py`) |
| `obstacles` | `Obstacle`s: what else stands nearby, with `position`, `radius_m` and `label` |

It returns `Outcome(ok, message, refused=, at_xy=, notes=, ...)`. If the
policy moved the object, `at_xy` is where it is now. The message needs no name
in front: the verb adds its own when it runs (`[OK] slide: pushed it 20 mm`).

### 3. The rules every policy keeps

Each of these has cost a session on the real arm.

- Check what `move_ee_to` returns. It says whether the arm got there. Ignoring
  it is how a policy reports success from the wrong place.
- Set `refused=True` only when nothing has moved (no room, height not known,
  out of reach). A refusal is reported without asking the vision model to
  describe a scene where nothing happened.
- Never catch a stop. When the sentry sees a hand, or someone stops the arm,
  the backend raises `MotionStopped` from inside the move. It derives from
  `BaseException`, so `except Exception` lets it through, as it should.
- Check for optional hardware first. An arm may lack a wrist camera, a force
  sensor or a load sensor in the gripper. Test with the protocols in
  `mrcf/capabilities.py`, and do without what is absent:
  `if isinstance(env, GripSensing): held = env.gripper_loaded()`.
- Say what happened. The user, the log and the re-planner read the message.
  Put the detail in `notes`.

### 4. Test it without the arm

`tests/fake_arm.py` is an arm in memory. Its grasp point moves in straight
steps, objects are upright cylinders, and `arm.path` holds every point it went
through.

```python
# tests/test_slide.py
import numpy as np
import pytest

from fake_arm import FakeArm
from mrcf.policies.base import Obstacle, Target
from mrcf.policies.slide import slide


def test_a_slide_comes_down_beside_the_object_and_pushes_it_low():
    arm = FakeArm()
    got = slide(arm, Target(np.array([0.25, 0.0, 0.0]), foot_m=0.03, height_m=0.04))
    assert got.ok and np.allclose(got.at_xy, [0.27, 0.0])
    assert min(p[2] for p in arm.path) == pytest.approx(0.02)        # low on its body
    assert arm.get_ee_pos()[2] == pytest.approx(0.07)                 # and lifted clear after


def test_a_slide_that_would_land_on_a_neighbour_moves_nothing():
    arm = FakeArm()
    rook = Obstacle(np.array([0.22, 0.0]), radius_m=0.015, label="the rook")
    got = slide(arm, Target(np.array([0.25, 0.0, 0.0]), foot_m=0.03, height_m=0.04), obstacles=[rook])
    assert got.refused and "the rook" in got.message and arm.path == []
```

`pytest tests/test_slide.py` runs it. For a policy that needs cameras or a
stream, copy the template arm instead: `backends.arm("_template").make_env()`
is an arm in memory with the rig's geometry, and `tests/test_template.py` uses
it.

You can test the verb ([step 5](#5-make-it-something-the-planner-can-ask-for))
offline too. Put an object in a world model by hand, as a scan would, and
dispatch the verb on it, as the planner does.

```python
# tests/test_slide_verb.py
import numpy as np

from mrcf import backends
from mrcf import manipulation
from fake_arm import FakeArm
from mrcf.policies.slide import slide
from mrcf.vlm import SceneObject, SceneState, dispatch


def test_slide_is_a_verb_that_moves_the_object_in_the_world_model():
    manipulation.install(backends.arm("_template").make_policies())    # a policy library: every verb needs one
    manipulation.object_action("slide", lambda pol, env, target, obstacles, **_: slide(env, target, obstacles),
                               description="Push an object about 2 cm without picking it up.", moves_object=True)
    pawn = SceneObject("white_pawn_e4", "a white pawn", position=np.array([0.25, 0.0, 0.0]))
    pawn.geometry = {"base_diameter_m": 0.03}                           # what a scan measures: its foot,
    pawn.measured_shape = {"height_mm": 40.0, "widths_mm": {0: 30, 40: 12}}    # its height and widths
    state = SceneState(objects=[pawn])
    assert dispatch({"action": "slide", "object_name": "white_pawn_e4"}, FakeArm(), state) is True
    assert np.allclose(state.find("white_pawn_e4").position[:2], [0.27, 0.0])
```

A `Target`'s measurements come from the object in the world model
(`manipulation.target_of`). `geometry["base_diameter_m"]` becomes `foot_m`, and
`measured_shape` gives `height_m` and `widths_mm`. An object with neither has
only the task's `max_height_m`, if the task gives one.

### 5. Make it something the planner can ask for

It takes one call in the deployment's `skills.py`. Chess's
(`deployments/chess/skills.py`) already has `ARM` from its config and has
installed the arm's policy library:

```python
from mrcf import manipulation
from mrcf.policies.slide import slide

manipulation.object_action(
    "slide",
    lambda pol, env, target, obstacles, **_: slide(env, target, obstacles, table_z=ARM.config.TABLE_Z),
    description="Push an object about 2 cm without picking it up: to free it from a neighbour.",
    moves_object=True)
```

In your own deployment, set up the arm and its policy library first. Every
verb runs with one installed, even one that does not use it:

```python
from mrcf import backends
ARM = backends.arm("so101")                        # the deployment's one arm (its config's ARM_NAME)
manipulation.install(ARM.make_policies())          # once, before the verbs
```

`object_action` finds the named object in the world model, gives the policy its
geometry and neighbours, and runs it under the safety monitor. With
`moves_object=True`, it also moves the object to the Outcome's `at_xy` in the
world model and memory. `pol` is the arm's policy library, for a verb that is
one of its methods. `after=` runs something once it is done (chess's learned
skills scan again).

The `description` is the planner's only clue about when to use it. Write it for
a model choosing between a dozen verbs, not for a developer. For a verb that is
not about one object, `register_action` does the same by hand
([DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#adding-a-capability), with the
conditions an action can check first).

### 6. Try it

At the controller's `You:` prompt (COMMANDS.md, "Run it"):

```
/actions                       every verb, with its parameters
/slide white_pawn_e4           run it on one object: no planner, the same checks
slide the white pawn on e4     through the planner
```

### Changing how the pick works

A pick is itself a planned policy with two extension points
(`mrcf/policies/parallel_jaw/`):

- An **approach** is how it comes at an object in one part of the reach
  (`mrcf/policies/approaches.py`). A new one is data,
  `MY_WAY = Approach("my_way", zones=("far",), align=("fixed_camera", "kinematics"))`,
  given to the arm's library: `make_policies(approaches=(TOP_DOWN, NEAR_BASE, REACH_OUT, MY_WAY))`.
- An **aligner** is how the jaws line up over the object once there
  (`mrcf/policies/aligners.py`). It is a subclass with `applies()` and
  `run()`, registered with `register_aligner()` and named in an approach's
  `align`.

Each module's docstring has a worked example. `deployments/chess/tests/reach_map.py`
shows offline which approach each square gets.

---

## A learned policy, step by step

The example is `upright`, which stands a fallen pawn up using the wrist camera
and joints. The same steps hold for any trained model.

### 1. Give the arm a stream

A learned policy drives the arm a tick at a time through its **stream**
(`capabilities.ActionStreaming`). It carries what the arm observes, and the
goals it is sent through the arm's own safety checks.

- The SO-101 has one: its six motors (degrees, gripper %) and the wrist
  camera, at 30 Hz (`mrcf/backends/so101/demos.py`).
- A new arm copied from `mrcf/backends/_template` has one: the grasp point
  (x, y, z, metres) and the jaws' opening at `STREAM_FPS`, plus the wrist
  camera when it is open. In `TemplateEnv`'s three stream members, change it
  to what your arm's demonstrations will record, such as a UR's joints through
  `servoJ`. Set `ROBOT_TYPE` in its `robot_config.py`, and `NAME` in its
  `__init__.py` (the name messages use).
- Any other arm needs the three members:

```python
@property
def stream_spec(self) -> StreamSpec:          # streaming.StreamSpec: rate, state and action names, cameras
def stream_observation(self) -> dict:         # {"observation.state": float32 vector, "observation.images.wrist": RGB}
def stream_action(self, action) -> tuple:     # send one goal through the arm's checks: (as sent, why held back)
```

`stream_action` must call `self.guard()` first, so a stop stops it. It must
check every goal before it goes (inside the workspace, above the table, within
the joints' limits) and hold back a goal that fails instead of raising.
`mrcf/backends/_template/env.py` is a complete one. To check it offline,
`backends.contract.check_backend()` runs `check_streaming` on any arm that
streams (`tests/test_template.py` shows how).

### 2. See exactly what a model for it must be trained on

This prints the arm's stream without connecting or moving anything:

```bash
python tools/show_stream.py
```

Run it **on the robot, with its config as it runs**. The stream carries only
the cameras the arm's config opens, and leaves out a camera with no source. A
laptop with another config prints another stream and fingerprint.

```
The SO-101's stream — what a model for it is trained on, and given here:
  rate         30 Hz
  robot_type   so101_follower
  fingerprint  1ef67dacb3261faa   (what a model depends on: StreamSpec.fingerprint)
  features (a LeRobot dataset's keys; shapes as (height, width, channels) for a camera):
    observation.state         float32 (6,)            shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos
    action                    float32 (6,)            shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos
    observation.images.wrist  video   (480, 640, 3)   from the camera 'front'
```

The demonstrations must have **exactly these keys, names, shapes, units and
rate**, and the same `robot_type`. A model that differs is refused before
anything moves, but the checks miss what they cannot see: a camera pointed
somewhere else, or joints in other units under the same names.
`--json` prints it as data for a training script.
`StreamSpec.from_dict(json.load(f))` reads it back, and its `features()` are a
LeRobot dataset's.

On the SO-101, the overhead camera is not in the stream. It is moved between
sessions, and a policy that relied on it would not survive the next move. The
planned steps around a learned one use it instead.

### 3. Record demonstrations and train — elsewhere

Train on a machine with a GPU, with LeRobot or whatever trains your model. To
use the stream there (`StreamSpec`, and `VectorCodec` for a Stable-Baselines3
policy), install this repository on it too, with `pip install -e .` and no
extras (numpy, pillow and requests). Then save the stream from the robot:

```bash
python tools/show_stream.py --json > stream.json         # on the robot
```

```python
import json
from mrcf.streaming import StreamSpec

spec = StreamSpec.from_dict(json.load(open("stream.json")))   # on the training machine
spec.features()                                                 # the dataset's features, exactly
```

**With LeRobot (ACT, diffusion, SmolVLA, pi0) on the SO-101**, move a leader
arm by hand and the follower mirrors it:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: <its index>, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=<LEADER_PORT> --teleop.id=my_leader \
  --dataset.repo_id=<you>/so101_upright --dataset.single_task="stand the pawn up" \
  --dataset.fps=30 --dataset.num_episodes=50 --dataset.push_to_hub=false

lerobot-train --dataset.repo_id=<you>/so101_upright --policy.type=act \
  --output_dir=outputs/train/upright --policy.device=cuda --policy.push_to_hub=false
```

- Name the camera `wrist`, use 640x480, and record only that camera. The
  dataset's key is then `observation.images.wrist`, as the stream's is. A
  different name or size, or an extra camera, is refused when the model runs.
- While it records, LeRobot drives the arm, not this framework. The sentry, the
  interlock, the jaw-tip floor and the stream's joint limits are not running.
  Record in a cleared area, within reach of the power switch.
- First make the leader and follower read the same pose alike. On an SO-101
  pair, the wrist roll's zero is wherever the wrist happened to be turned
  during `lerobot-calibrate`. On the rig this was built on, they read about
  100° apart, which sends the follower's wrist into its stop: it overloads and
  drops off the bus near -95°. Recalibrate the leader with its wrist turned as
  the follower's was. Before the first take, move the leader's wrist alone,
  slowly.
- LeRobot 0.4 records the follower in degrees (`use_degrees`, its default), as
  the stream does. Begin each take where the skill will start it: the hover
  over the piece ([step 6](#6-make-it-a-verb)).

The trained model is the checkpoint folder
`outputs/train/upright/checkpoints/last/pretrained_model`: its config, its
weights, and the normalisation it was trained with.

**On another arm**, record its stream while you drive it by hand (freedrive,
a teleop device) into a dataset with the stream's features. LeRobot's
`LeRobotDataset.create(repo_id, spec.fps, features=spec.features(),
robot_type=spec.robot_type)` takes them as they are. Then train as above.

**With Stable-Baselines3**, a policy reads a vector and answers with a step in
[-1, 1] per action. `policies.rl.VectorCodec` maps that onto the stream, and
the environment you train in must read and act through the same codec:

```python
from mrcf.policies.rl import VectorCodec                    # this repository, installed where you train

codec = VectorCodec(state=spec.state, action=spec.action,
                    step=(2.0, 2.0, 2.0, 2.0, 2.0, 0.0),   # a full step: 2 degrees a tick; the gripper not at all
                    environment=3,                          # the goal's length (x, y, z), if it is given one
                    mid=(...), half=(...))                  # normalise the observation: (x - mid) / half
obs = codec.observation(observation)                    # in your gymnasium env's step() and reset()
action = codec.action_for(observation, network_step)   # the stream's action for the network's step
# after training:
model.save("run/policy.zip"); codec.save("run/codec.json")
Path("run/meta.json").write_text('{"algo": "sac"}')     # or "td3", "ppo": which class loads it
```

The folder `run/` is what the robot runs ([Reference](#reinforcement-learning)).

### 4. Bring it to the robot

Copy the folder over and use it as it is (a LeRobot checkpoint, step 6). Or,
better for sharing and for a full check on the robot, make it a **policy
package**: the model's files plus a `policy.json` manifest naming the stream it
was trained on.

```python
from mrcf import backends
from mrcf.policies import packages

spec = backends.arm().make_env().stream_spec            # on the robot, its cameras connected (step 2)
packages.pack("outputs/train/upright/checkpoints/last/pretrained_model", "packages/upright",
              packages.Manifest(name="upright", format="lerobot", stream=spec, version="1.0.0",
                                description="Stand a fallen piece back up on its base.",
                                task="stand the pawn up", start_height_m=0.06, max_s=20))
```

| Manifest field | What it is |
|---|---|
| `name`, `version` | the skill's name, and which version this is |
| `format` | how its files are loaded: `"lerobot"`, `"sb3"`, or [your own](#your-own-model-format) |
| `stream` | the stream it was trained on. Its names, shapes, rate and robot type are checked against the arm's |
| `model` | its files, inside the package (`"model"`) |
| `description` | for the planner: when to use it |
| `task` | the instruction a language-conditioned model (SmolVLA, pi0) is given |
| `start_height_m` | how high over the object to start: where the demonstrations began |
| `max_s` | the longest it may run |
| `license`, `extra` | the licence is recorded, not enforced. `extra`: anything else |

You install a package by reference: a folder, `hf:owner/name@revision` (the
Hugging Face hub), or a scheme a plugin adds ([Plugins](#plugins)). A relative
folder is taken from the directory the program runs in (the repository's root,
for `run.py` and the tools), so an absolute one is safer.

### 5. Check it before it moves

These say whether it fits the arm's stream, and whether this machine can run
it:

```bash
python tools/show_stream.py --package packages/upright
python tools/show_stream.py --checkpoint outputs/train/upright/checkpoints/last/pretrained_model
```

For example:

```
Package 'upright' 1.0.0 (format 'lerobot', from packages/upright),
trained on fingerprint 1ef67dacb3261faa at 30 Hz.
It fits this arm's stream.
It cannot run on this machine: it needs lerobot and torch, not installed here (pip install lerobot).
```

A package is checked against its whole stream (names, shapes, rate, robot
type). Then, if this machine can load it, the model's own `check()` runs, as in
`LearnedPolicy` before anything moves. A bare checkpoint carries only its
shapes, so only they are checked: another arm with as many joints would pass.
Package it (step 4) to have the names checked too.

Running a model needs its library on the robot. A LeRobot model needs
`pip install lerobot` (the SO-101's `.[so101]` brings it), and a
Stable-Baselines3 one `pip install -e ".[sb3]"`.

### 6. Make it a verb

**In chess**, add one entry to `LEARNED_SKILLS` in `deployments/chess/config.py`:

```python
LEARNED_SKILLS = [
    {"name": "upright", "policy": "packages/upright"},                       # a package: its manifest says the rest
    # or a checkpoint directly, saying it all here:
    {"name": "upright", "description": "Stand a fallen piece back up on its base.",
     "checkpoint": "outputs/train/upright/checkpoints/last/pretrained_model",
     "task": "stand the pawn up", "start_height_m": 0.06, "max_s": 20},
]
```

| Key | |
|---|---|
| `name` | the verb |
| `policy` *or* `checkpoint` *or* `model` | a package reference, a LeRobot checkpoint folder, or any `learned.Model` object |
| `description` | for the planner (a package's manifest gives it) |
| `task` | the instruction, for a language-conditioned model |
| `start_height_m` | hover this high over the object first. Leave it out to start where the arm is |
| `max_s` | the longest it runs (30 s) |

Any key beside `policy` overrides the manifest. At start-up each entry becomes
a verb the planner is offered, or is left out with the reason: a package that
cannot be fetched, a malformed entry, or a model whose library this machine
lacks (`its model needs lerobot and torch, not installed here
(pip install lerobot)`). A chess skill runs its model for `max_s`, then scans
again to see where the piece ended up. It gives the model no goal, so a
goal-conditioned policy needs a verb of your own (below).

**In your own deployment**, do the same in code:

```python
from mrcf import manipulation
from mrcf.policies import packages
from mrcf.policies.learned import LearnedPolicy, chain, hover_over
from mrcf.policies.lerobot_model import LeRobotModel

upright = LearnedPolicy(LeRobotModel("outputs/train/upright/checkpoints/last/pretrained_model",
                                     task="stand the pawn up"),
                        name="upright", max_s=20)


def look_again(env, scene_state, obj, outcome):
    """After it: a learned policy cannot say where it left the object, so look (chess scans again)."""
    ...                                            # your deployment's scan


manipulation.object_action("upright", chain(hover_over(0.06), upright),
                           description="Stand a fallen piece back up on its base.", after=look_again)
```

`chain()` runs its steps in order on one object and stops at the first that
fails. `hover_over(h)` is the arm's planned `move_above`. For a package's
model, with the stream it was trained on checked too:

```python
entry = packages.skill_entry({"name": "upright", "policy": "packages/upright"})
upright = LearnedPolicy(entry["model"], name="upright", max_s=entry["max_s"], trained_on=entry["trained_on"])
```

Or run it alone, without the planner, on an arm in memory or with a hand on
the power:

```python
outcome = upright.run(env)                     # or upright.run(env, target) for a goal-conditioned model
print(outcome.ok, outcome.refused, outcome.message)
```

### 7. Run it, and read what happened

```
/upright white_pawn_e4         at the prompt: hover, the model, a fresh scan
stand the pawn on e4 up        through the planner
```

The first run of a session loads the model, taking seconds, and warms the
network up before its clock starts. The outcome says how it ended ([When it is
refused, or fails](#when-it-is-refused-or-fails)).

### Tuning a learned segment

| `LearnedPolicy(...)` | |
|---|---|
| `until=f` | `f(env, observation) -> True` when the segment is done. Without one, running `max_s` is success |
| `max_s` | the longest it runs. Pauses (the sentry) do not count |
| `max_held_back_s` | stop once the arm has held back all the model asked for this long, without a break (1 s) |
| `goal=f` | `f(target) -> vector`: a goal-conditioned model is told it each tick (`observation.environment_state`) |
| `trained_on` | the stream it was trained on (a package's): checked before anything moves, names and robot type too |

Start where the demonstrations started: the model has seen nothing else. After
a pause (a hand over the table), the model starts again from what it sees: what
it had queued was for the scene before.

---

## Your own model format

Anything that answers an observation with an action can run: an ONNX export, a
network of your own, a scripted stand-in. A model is an object with:

| | |
|---|---|
| `reset()` | a new episode: forget anything queued (an action chunk) or remembered |
| `act(observation) -> vector` | the next action, in the stream's action order |
| `check(spec) -> [problems]` *(optional)* | why it cannot run on this arm's stream, or `[]` when it can |
| `unavailable() -> str or None` *(optional)* | why it cannot run on this machine at all (a library it needs is missing), without importing it |

```python
# my_models.py — anywhere on the robot's Python path
import numpy as np


class LinearModel:
    """A network of one layer, saved as weights.npz (W, b): the arm's state in, its action out."""

    def __init__(self, path):
        self.path, self._wb = path, None

    def _weights(self):
        if self._wb is None:                  # loaded when first run, not when made
            data = np.load(self.path)
            self._wb = data["W"], data["b"]
        return self._wb

    def unavailable(self):
        return None                           # numpy only; a model needing a library: needs(...)

    def check(self, spec):
        w, _ = self._weights()
        problems = []
        if w.shape[1] != len(spec.state):
            problems.append(f"it reads {w.shape[1]} values, the arm's state has {len(spec.state)}")
        if w.shape[0] != len(spec.action):
            problems.append(f"it sends {w.shape[0]} values, the arm takes {len(spec.action)}")
        return problems

    def reset(self):
        pass                                  # nothing remembered between episodes

    def act(self, observation):
        w, b = self._weights()
        return w @ np.asarray(observation["observation.state"], np.float64) + b


def load_linear(folder, manifest):
    """The "linear" format: a package's weights.npz as a LinearModel."""
    return LinearModel(folder / manifest.model)
```

Keep your own modules out of this repository's root. Every `.py` file there is
taken for framework code, and `tests/test_architecture.py` holds it to the
framework's rules. Put them in your deployment's folder or, for a format
others will use, in a package of their own.

Run it directly (`LearnedPolicy(LinearModel("weights.npz"), name="drift",
max_s=5)`) or make it a package format. `register_format` does that for the
program it is called in, such as your script or your deployment's `skills.py`
(before it installs its skills):

```python
packages.register_format("linear", load_linear)
packages.pack("weights.npz", "packages/drift", packages.Manifest(
    name="drift", format="linear", model="weights.npz", stream=spec, description="Drift along x."))
```

For every program (`tools/show_stream.py`, a deployment started by `run.py`),
the format must come from an installed package that names it in its entry
points. A folder beside the repository is enough:

```
my_models/
  pyproject.toml
  my_models.py          the code above
```

```toml
# my_models/pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "my-models"
version = "0.1.0"
dependencies = ["numpy"]

[tool.setuptools]
py-modules = ["my_models"]

[project.entry-points."mrcf.model_formats"]
linear = "my_models:load_linear"
```

```bash
pip install -e my_models/        # into the same environment as this repository
```

Without it, a package of that format is reported instead of run:
`No model format 'linear' (known: lerobot, sb3; a plugin adds one through the
'mrcf.model_formats' entry point group).`

A format's `load` only makes the model, cheaply. A deployment makes every
skill's model as it starts, and the weights load when it first runs. A model
that needs a library says so without importing it:

```python
from mrcf.policies.learned import needs

def unavailable(self):
    return needs("onnxruntime", install="pip install onnxruntime")
```

Such a model is then refused before anything moves: at start-up in chess, and
by `LearnedPolicy` and `chain()` anywhere. A quick stand-in needs none of
this: `learned.CallableModel(act, reset=, check=)` wraps a function.

---

## When it is refused, or fails

A **refusal** happens before anything moves. A **failure** happens while it
runs, and the arm is held where it stopped.

| The message says | What happened | What to do |
|---|---|---|
| `Learned skill 'x' left out: its model needs lerobot and torch, not installed here (pip install lerobot)` | at start-up: this machine lacks the model's library | install it, or run without the skill |
| `Learned skill 'x' left out: ...` (anything else) | at start-up: the package could not be fetched or loaded, or the entry is malformed | the reason follows. Fix the entry or the reference |
| `No policy package at packages/x: no such folder (from /where/it/ran)` | a relative reference, from another directory | an absolute path, or run from where it is relative to |
| `x: cannot run on this machine — needs ...` | the same, in code you wrote | as above |
| `x: the model was trained on another stream than this arm's: ...` | names, shapes, rate or robot type differ. Each difference is listed | train on this arm's stream (`tools/show_stream.py`) |
| `x: ... takes no stream of actions (capabilities.ActionStreaming)` | the arm has no stream | [step 1](#1-give-the-arm-a-stream) |
| `x: it is told where to go, and was given no target` | a goal-conditioned policy run with no object | run it on an object |
| `trained with a goal of 3 values, given 0 (give LearnedPolicy a goal)` | a Stable-Baselines3 policy trained with a goal, given none | `LearnedPolicy(goal=lambda target: target.position)` |
| `x: done — ...` / `x: ran its 20 s — ...` | finished: `until` said so, or (no `until`) it ran its time | |
| `x: not done after 20 s — ...` | `until` never said done | more time, a better start, or more demonstrations |
| `x: stopped: the arm held back what it asked for 1 s (jaw tips 6 mm below the floor)` | the model kept asking for goals the arm will not go to: into the table, out of the workspace | look at where it was heading. Usually the start was unlike the demonstrations' |
| `x: stopped: the model gave (7,) values, the arm takes 6` | the model's output is the wrong size | the model or its adapter is wrong |

Each message ends with how many ticks it ran and how long the model took per
tick. A model slower than the stream's rate is reported as slow ticks. Each
run is also on the flight recorder's timeline (`python -m mrcf.flight_recorder`).

---

## Reference

### The arm's stream

What a learned policy sees and sends is the arm's own: its joints in its own
units, its wrist camera at its own size, its rate.

```python
env.stream_spec                        # streaming.StreamSpec: fps, state names, action names, cameras, robot_type
env.stream_observation()               # {"observation.state": vector, "observation.images.wrist": RGB frame, ...}
env.stream_action(action)              # through the arm's own checks: (as sent, why any was held back)
```

It is also what a model must have been trained on. `StreamSpec.features()` is
a LeRobot dataset's features, so the model is given exactly what it was shown.
On the SO-101, `stream_action` is `stream_joints`. It clips to the joints'
ranges, holds back goals below the jaw-tip floor or outside the workspace, and
goes through the interlock and the sentry. On the template it takes the grasp
point, held back outside the workspace or out of reach. A rig's extra sensors
(a force sensor's wrench) are in the stream too (`observation.sensors.<name>`).

Two checks hold a model against a stream. `StreamSpec.mismatches(inputs,
outputs)` compares a model's own shapes, which are all a bare LeRobot
checkpoint carries. `streaming.differences(trained, arm)` compares the whole
stream it was trained on (shapes, names, rate and robot type), and
`packages.fits()` and `LearnedPolicy(trained_on=)` use it.
`StreamSpec.fingerprint()` is the short id of what a model depends on: robot
type, rate, and every feature's name, type and shape, but not which device a
camera is. Two arms with the same fingerprint run each other's models.

### How a model is run

`LearnedPolicy` runs the model at the stream's rate until `until` says the
segment is done, its time is up, or the arm has held back everything the model
asked for over `max_held_back_s` (a model pushing where the arm may not go). A
pause (the sentry) does not count against its time, and after one the model
starts again from what it sees. It holds the arm when it ends.

Before anything moves, it refuses a model whose library is missing
(`unavailable()`), an arm with no stream, or a stream other than the one it
was trained on (`trained_on`, then the model's own `check`). `refusal(env,
target)` asks all of that without running it. A `chain()` asks it of every
step before any of them moves the arm, so a skill whose learned part would be
refused does not make its planned hover first.

The first time it is needed, `LeRobotModel` loads the checkpoint and its pre-
and post-processors and warms the network up, on the best device there is
(CUDA, then Apple's MPS, then the CPU). It runs each tick through LeRobot's
own `predict_action`.

### Reinforcement learning

At run time, a policy trained with reinforcement learning is like any other.

A LeRobot SAC checkpoint (HIL-SERL, `lerobot/rl`) is a `LeRobotModel`.
HIL-SERL trains on the real arm through LeRobot's own robot classes, so the
sentry and the interlock do not run while it trains. Train in a cleared area,
within reach of the power. Its observations are processed (images cropped and
resized, an end-effector pose added), and its actions may be end-effector
steps through IK. To run such a policy here, put the same processing in a
`Model` wrapped round `LeRobotModel`, or set up the training to use the arm's
own stream.

A Stable-Baselines3 policy is a `policies.rl.SB3Model`, loaded from a folder
holding `policy.zip` (the network), `codec.json` and `meta.json`
(`{"algo": "sac"}`). The codec (`VectorCodec`) is how the network reads and
drives the stream. An observation is the arm's state and the task's goal,
normalised. An action is a step in [-1, 1] for each action's goal, from where
it is now. Its `check()` refuses a stream with other names.
`LearnedPolicy(goal=lambda target: target.position)` gives it its goal, the
planner's target. As a package its format is `"sb3"`. The package's stream
holds the goal as its `environment`, which belongs to the task and is not
checked against the arm's.

### Packages

A package (`mrcf/policies/packages.py`) is a folder: the model's files and
`policy.json`.

```python
manifest, folder = packages.open_package("hf:me/upright-so101@v1")    # fetched, read
packages.fits(manifest, env.stream_spec)                              # [] when this arm can run it
model = packages.load_model(folder, manifest)                         # a learned.Model, by its format
entry = packages.skill_entry({"name": "upright", "policy": "..."})    # what a deployment's entry becomes
```

`skill_entry` makes the model and fills in an entry from the manifest: the
description, task, start height, time limit, `embodiment` (the fingerprint),
`version` and `trained_on` (the stream). The entry's own values win.

### Plugins

Three registries take additions from code or from an installed Python package's
entry points, with no change to this repository:

| Entry point group | Names | What it adds |
|---|---|---|
| `mrcf.model_formats` | a format (`"onnx"`) | `load(folder, manifest) -> learned.Model`: how a package's files run. It only makes the model, and the weights load when first run. `unavailable()` names a missing library |
| `mrcf.policy_sources` | a scheme (`"store"`) | `fetch(rest) -> local folder`: where packages come from. Sign-in, licence check, download and cache are the plugin's own. The framework sees only the folder |
| `mrcf.arms` | an arm (`"ur5"`) | an arm's package, as `mrcf/backends/<arm>/` would be (`backends.arm()`) |

```toml
[project.entry-points."mrcf.policy_sources"]
store = "my_store_client:fetch"          # LEARNED_SKILLS: {"name": "upright", "policy": "store:upright@1.2"}
```

So a policy store is a source plugin plus a catalogue filed by fingerprint. An
arm or a model format can be shared the same way, installed with `pip` beside
the framework. `packages.register_source()` and `register_format()` do the
same from code.

---

## Checklists

**A planned policy**
- [ ] In `mrcf/policies/` (any arm with that gripper) or `mrcf/backends/<arm>/` (one arm)
- [ ] Takes geometry (`Target`, `Obstacle`s) and returns an `Outcome`. Imports nothing of the world model, memory or planner
- [ ] Checks what every `move_ee_to` returned. `refused=True` only when nothing moved. Never catches `MotionStopped`
- [ ] Uses an optional extra only after `isinstance(env, <capability>)`
- [ ] Tested on `tests/fake_arm.py` (or a template copy)
- [ ] A verb: `manipulation.object_action(...)` with a description written for the planner; `moves_object=True` if it moves the object

**A learned policy**
- [ ] The arm streams (`ActionStreaming`) and passes `check_backend()`
- [ ] Trained elsewhere, on demonstrations with exactly `tools/show_stream.py`'s features, rate and robot type
- [ ] Packaged with the stream it was trained on (or a checkpoint, checked by shape only)
- [ ] `tools/show_stream.py --package` says it fits and can run here
- [ ] Its library installed on the robot (`pip install lerobot`, `.[sb3]`)
- [ ] A start (a planned step before it, where the demonstrations began), an end (`until`, or `max_s`), and a look after (`after=`)
- [ ] Tried at the prompt (`/name object`) with a hand near the power
