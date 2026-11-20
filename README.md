# MRCF — Modular Robotics Control Framework

Control a robot arm in plain English, with local models. You type "move the
knight to f6". The framework works out what is on the table, plans a sequence
of actions, checks the plan against safety rules, and runs it through
deterministic skill controllers that check their own success.

The language model never moves the arm itself. It picks actions from a fixed
list (`pick`, `place`, `observe`, …), and policies carry them out. Policies
never see object names or memories, so the same pick works in any task, and a
task works on any arm that has its own policies.

A policy is planned or learned. Planned policies, such as pick and place, are
deterministic and work from geometry. A learned policy is a model trained
elsewhere, on demonstrations (ACT, through LeRobot) or by reinforcement
learning. It runs here on the arm's own stream of joints and camera frames,
usually for one segment of a skill between planned steps. Nothing in this
repository trains a model, so the framework stays light enough to deploy on
the robot. Everything runs on your machine: [Ollama](https://ollama.com)
runs the models, and an optional segmentation model (SAM2) measures objects
from camera images.

**Status.** Alpha. The framework is robot-agnostic: nothing in it names an
arm. A deployment is one arm, one layout and one task, and names its arm.
`chess` drives a real [SO-101](https://github.com/TheRobotStudio/SO-ARM100)
arm through [LeRobot](https://github.com/huggingface/lerobot) and plays with
chess pieces. `tabletop` stacks letter blocks with a UR5 (`mrcf/backends/ur5`).
`tidy` has the SO-101 put everyday things away into a tub, and `toolbench` has
the UR5 put hand tools into a box. The last three share one table task (find,
name, pick, place), which belongs to the framework (`mrcf/tabletop`). Each of
them gives it its own arm, layout and rules.

You set up a new automation in the order a factory would: the arm, then its
sensors, then the job.

1. **The arm.** Use one the framework has (`mrcf/backends/so101`, `ur5`), or
   add one: answer `n` at run.py's first question, or run
   `python tools/new_arm.py NAME`. This copies `mrcf/backends/_template`, which
   runs as an arm in memory until you add its SDK
   ([DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#adding-an-arm)).
2. **Its sensors.** List its cameras and other sensors in its
   `robot_config.py` (`CAMERAS`, `SENSORS`). Find them by name with
   `tools/find_cameras.py` and calibrate them
   ([COMMANDS.md](COMMANDS.md#first-time-setup-on-another-arm)). run.py shows
   them all before each session.
3. **The job.** Answer `n` at run.py's second question to start a deployment
   (one arm, one layout, one task) from the nearest existing one. It gets its
   own config, its rules for the arm (`profile.py`), and the skills and
   policies it needs ([POLICIES.md](POLICIES.md) for a new one, planned or
   learned).
4. **Run it, and read the record.** Every run keeps its failures and timings
   (see the flight recorder, below), which show what to fix and what to make
   faster.

---

## How it works

```
You: "move the knight to f6"
        │
        ▼
[scan]       overhead camera → board, squares, and every object on it:
             where its foot is, how wide, what colour        (chessboard.py,
             matched against the object memory, so each        mrcf/perception/,
             piece keeps its identity and what is known        object_memory.py)
             about it across scans, moves and restarts
        │
        ▼
[perceive + decompose]   free-text scene description; split into sub-goals  (vlm.py)
        │
        ▼
[plan]       sub-goal + world model + recent events → JSON action list       (vlm.py)
        │
        ▼
[validate]   workspace bounds, object existence, sequence rules,            (safety.py)
             per-action pre-conditions — a blocked plan does not run
        │
        ▼
[execute]    each action turns names into geometry and runs the       (manipulation.py,
             arm's policy: retries; the gripper camera measures the     mrcf/policies/)
             piece from above before the grasp; a failed step stops
             the plan and triggers a re-plan from a fresh look
        │
        ▼
[outcome]    written back to the world model and the object memory   (manipulation.py)

alongside    watchers on their own threads — the motion guard: a hand  (watchers.py,
everything   moving near the arm — can trip the interlock; the arm      interlock.py,
             freezes where it is at once, and carries on once it has    motion_guard.py)
             gone (any other stop abandons the command)
```

[DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#how-the-pieces-fit) has the full
diagram and the rules that keep its parts separate.

The model never produces coordinates. Positions come from camera geometry: a
calibrated overhead camera, and a gripper camera whose pose is worked out from
the arm's joint angles. So their accuracy depends on the calibration, not on a
language model.

---

## Repository layout

The code has three tiers, so a new task or a new arm touches only one of
them. Each deployment names its arm in its config (`ARM_NAME`). The tools
drive the arm `MRCF_ARM` names (the SO-101 by default). The framework is one
package, `mrcf/`, and the only thing `pip install` installs
(`from mrcf import manipulation`). The scripts beside it use it from the
repository.

```
mrcf/                   ── the framework: the one installed package ──
  robot_env.py          RobotEnv — the interface every backend implements
  interlock.py          stops the arm from anywhere, mid-move
  watchers.py           runs perception on its own thread while the arm works
  motion_guard.py       freezes the arm when anything but the arm moves near it
  flight_recorder.py    what happened, in order: one timeline, the last minute of joints and
                        frames, and an incident report when an action goes wrong
  capabilities.py       what a backend can do beyond RobotEnv, one Protocol each
  overhead_calibration.py  a fixed camera's calibration: checked every scan, re-posed when it moves
  rig_state.py          where each rig's own files live: rig/<backend>/, under MRCF_HOME
  streaming.py          an arm's stream: what learned policies are trained on and run on
  manipulation.py       the standard actions: planner names → policies → world model
  NOTES.md              what it cannot do yet, and what it has shown (each backend and task has one too)
  HISTORY.md            the dated runs behind its values and fixes (so do policies/, each backend, each task)
  controller.py         the loop: a command → plan → safety → dispatch → recover
  vlm.py                every model call: perception, scene discovery, planner
  model_server.py       where the models run: Ollama, any OpenAI-compatible API, or a plugin's
  scene_state.py        the world model the planner sees
  action_registry.py    the actions a planner may use, their conditions, and dispatch
  skill_executor.py     run_policy(): safety monitor + failure description
  safety.py             plan-time validation, execution-time monitoring
  object_memory.py      one identity per physical object, across scans and restarts
  scene_scan.py         puts a surface scan into the object memory and the world model
  profiles.py           a deployment's rules for its arm (its profile.py), laid over the arm's facts
  rig_report.py         what an arm brings to a job, shown before a run: gripper, cameras, sensors
  devices.py            this computer's cameras and ports, by name, without opening any
  privacy.py            guards every kept picture: keeps only the work area, covers faces

  tabletop/             the table task (find, name, pick, place, stack, put into) that a
                        deployment hands its config to (deployments/tabletop, tidy, toolbench)

  policies/             how an arm does things. Planned from geometry: parallel_jaw/ (pick,
                        place, …); approaches.py and aligners.py (the ways a pick comes at an
                        object); stand_up.py (stands a fallen object back up). Or learned:
                        learned.py (a trained model on the arm's stream); lerobot_model.py
                        (a LeRobot checkpoint); rl.py (a Stable-Baselines3 one); packages.py
                        (a model with the manifest it travels with, and the plugin seams for
                        sources and formats)

  perception/           ── seeing the work surface, from any calibrated camera ──
    plane_geometry.py   measures an object (foot, height, widths) from its outline
    shape_fit.py        how tall an object is: fits a solid to its outlines in several views at once
    segment.py          SAM2 outlines of whatever is under a set of points
    surface_scan.py     finds and measures everything on the work surface, in any light
    fiducials.py        AprilTags and bullseyes: where a camera, a gripper or a board is
    hand_eye.py         where a wrist camera sits on the arm, from views of a grid
    lighting.py         whether the light is good enough to see by (washed out, too dark), per area
    outlines.py         one outline per object from a segmenter's many; an outline's middle
    tones.py            where a colour falls between two of the surface's own colours (any light)
    lying.py            an object on its side: which way it lies, which end it stands on, and
                        where the knob at its top is, from a fixed camera or a wrist one

  backends/             ── how to drive a robot ──
    __init__.py         arm(): the arm that runs (MRCF_ARM); RIG_SETTINGS: what every arm's config has
    rig.py              RigEnv: an arm's cameras and gripper geometry from its kinematics
                        (camera models, grasp point, gripper tag, wrist servo, sentry)
    contract.py         check_backend(): what every arm's env must do, checked offline
    so101/              the SO-101 follower: env.py (SO101Env, built from follower.py,
                        kinematics.py, gripper.py, orientation.py, motion.py, cameras.py,
                        stream.py), robot_config.py (this rig's measured values: the arm's
                        facts, whatever the task), make_policies()
    ur5/                the UR5 (CB3) with a Robotiq Hand-E, over ur_rtde: env.py (UR5Env),
                        arm.py, gripper.py, kinematics.py, robot_config.py, make_policies()
    _template/          a new arm to copy: a RigEnv over an SDK, its config, its policies
                        (it runs in memory as it is, and passes the contract checks)
    real_robot_env.py   base class for an arm driven by your own SDK (it owns the cameras)

run.py                  deployment launcher

deployments/            ── what to ask it to do ──
  chess/                chess pieces on a board: board scan, naming, controller. A package,
                        deployments.chess (python -m deployments.chess.llm_controller)
  tabletop/             the UR5's letter blocks on its table: the table task (mrcf/tabletop)
  tidy/                 the SO-101 putting everyday things away into a tub: the table task
  toolbench/            the UR5 putting hand tools into a box: the table task
  <name>/profile.py     how that deployment drives its one arm (ARM_NAME), laid over the arm's facts

tools/                  ── measuring and checking the rig, on whichever arm runs ──
                        cameras, tags, the overhead camera's calibration, the motion guard.
                        new_deployment.py: a new deployment from the nearest one's code, on
                        its arm or another, with its rules offered as candidates only.
                        new_arm.py: a new arm from mrcf/backends/_template. find_cameras.py:
                        lists the cameras by name without opening any. fetch_face_model.py,
                        clean_pictures.py: covering faces
  so101/                the SO-101's own tools: its poses, servos, joint and gripper calibrations

rig/<backend>/          ── this rig's own files, never in git (rig_state.py) ──
                        the overhead camera's calibration, the photo of the empty surface,
                        the URDF, and what the calibration tools save for refitting

tests/                  framework tests (pytest)
```

---

## Install

```bash
git clone https://github.com/mitos-robotics/MRCF.git
cd MRCF
python -m venv .venv && source .venv/bin/activate     # or conda; Python 3.10 or newer
pip install -e ".[dev]"                                # the framework and its tests: no hardware, no models
```

It runs on Linux, Windows and macOS. The framework works on Python 3.10 to
3.15. To drive a real SO-101 or UR5, use Python 3.10 to 3.14: their libraries
(LeRobot's placo and torch, and ur_rtde) have no builds for 3.15 yet. The commands in these docs are written
for a macOS or Linux shell. On Windows, activate with `.venv\Scripts\activate`
and copy folders with `xcopy /E` or Explorer. Where the docs say
`export NAME=value`, set the variable with `$env:NAME = "value"` (PowerShell).

Then add what you need:

| | |
|---|---|
| Run the tests, or contribute | `.[dev]`, or `.[dev,measure,sb3]` for the whole suite ([CONTRIBUTING.md](CONTRIBUTING.md#tests-and-lint)) |
| Run a learned policy | For a LeRobot policy (ACT, diffusion, …): `pip install lerobot` (`.[so101]` includes it). For a Stable-Baselines3 policy: `.[sb3]` |
| Drive an SO-101 | A LeRobot environment, then `pip install -e ".[so101]"` ([Quick start: SO-101](#quick-start-so-101)) |
| Drive another arm | `.[rig,dev]` and the arm's SDK ([Quick start: another arm](#quick-start-another-arm)) |
| Talk to it in plain English | [Ollama](https://ollama.com), then `ollama pull gemma4:31b` (~20 GB). Or use models on another machine or a hosted API ([COMMANDS.md](COMMANDS.md#models-on-another-machine-or-a-hosted-api)) |

---

## Quick start: SO-101

You need an SO-101 follower arm already calibrated with LeRobot, a USB camera
on the gripper, and a second camera fixed above the table.

```bash
# In your lerobot environment
pip install placo                        # Cartesian IK
pip install "transformers>=4.56,<5"      # optional: measuring objects (SAM2)
python tools/so101/fetch_urdf.py

ollama pull gemma4:31b                   # vision and planning (~20 GB)
```

Point the code at your arm (or edit `mrcf/backends/so101/robot_config.py`):

```bash
export SO101_FOLLOWER_PORT=/dev/tty.usbmodemXXXXXXXX   # lerobot-find-port finds it
export SO101_FOLLOWER_ID=my_follower     # your lerobot calibration id
# export MRCF_ARM=so101                  # the default; another arm is MRCF_ARM=<its mrcf/backends/ folder>
```

The port name depends on the OS: `/dev/tty.usbmodem…` on macOS, `/dev/ttyACM0`
on Linux (your user must be in the `dialout` group), and `COM3` on Windows.

Then measure the rig. `robot_config.py` ships with **one rig's measured values**
(poses, camera calibrations, workspace bounds), and you must re-measure every
one of them on yours. [COMMANDS.md](COMMANDS.md#first-time-setup-in-order) lists
the steps in order. You measure the arm's poses and how wide its jaws really
open, correct its joint readings and calibrate the overhead camera from
touches on a chessboard, check reach, and calibrate the gripper camera.

Run it:

```bash
python run.py
```

run.py first asks which arm to use, with the one it finds plugged in as the
default (Enter). Next it shows what that arm has, without opening or connecting
anything: whether the arm is plugged in, its gripper, each camera (calibrated
or not, and whether the rig's cameras are plugged in), its other sensors, and
whether kept pictures are guarded. Then it asks which of that arm's
deployments to run, with the arm's own (chess) as the default. Answer `n` at
either question to start a new arm or a new deployment
([Extending it](#extending-it)). `python run.py --arm so101 --deployment chess`
skips both questions. The arm then homes, parks out of the overhead camera's
view, scans, and gives you a prompt:

```
You: pick up the pawn on e2
You: put it on e4
You: that piece on c3 is the white bishop
You: what can you do?
You: /objects                  ← slash commands skip the models: /pick, /home, /status, …
```

Stay within reach of the power switch for the first session. The safety layer
enforces workspace bounds and per-step joint limits, but nothing replaces
watching the arm the first few times it moves on its own. When you are done,
`python tools/so101/rest.py` parks the arm low and cuts torque.

---

## Quick start: another arm

Copy the template, make it drive your arm, and check it offline before it
moves:

```bash
pip install -e ".[rig,dev]"                   # cameras, tests; and your arm's SDK (a UR: pip install ur_rtde)
cp -r mrcf/backends/_template mrcf/backends/my_arm      # then the places marked YOUR ARM
export MRCF_ARM=my_arm
```

[DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#adding-an-arm) explains what to
implement and the contract that checks it, with a UR5 worked through.
[HARDWARE.md](HARDWARE.md) covers the cameras, board and tags the rig needs,
and [COMMANDS.md](COMMANDS.md#first-time-setup-on-another-arm) the
calibration, in order. The wrist camera's calibration is still SO-101-only, so
another arm picks with the overhead camera until it is ported.

---

## Framework modules

### `mrcf/robot_env.py` — the backend interface

All framework code talks to a `RobotEnv`. A backend implements `step`,
`get_ee_pos` (the point the jaws close on), `get_joint_angles`, `get_body_pos`,
`get_movable_body_names`, `get_static_body_names`, `render_image` and
`pixel_to_world`. Arms with a gripper add `move_ee_to` (which returns whether
it arrived), `set_gripper`, `get_gripper_width` and `go_home`. A backend that
knows where its cameras are adds `camera_model(camera, width, height)`, which
gives a camera's lens and pose, so the measurement modules below can work
with it.

Every backend calls `guard()` before each command it sends to the motors, and
implements `hold()`. That is how a stop reaches the arm mid-move. Cameras must
be readable from other threads while the arm moves. Every program calls
`connect()` before the arm first moves and `stop()` at the end. On an arm built
on `backends.rig.RigEnv`, the motion guard runs from `connect()` to `stop()`,
whatever the arm's own `connect()` does.

A rig can have any cameras and sensors. The arm's config lists them
(`CAMERAS`, `SENSORS`) and names the two cameras that have jobs: one on the
gripper and one fixed over the surface. Either can be absent.
`list_cameras()`, `camera_model()` and `read_sensor()` give access to the rest.
A force sensor is read with `get_contact_force()`, which the safety monitor's
force limits use. [HARDWARE.md](HARDWARE.md#cameras-and-sensors-what-each-is-for)
explains what each one is for.

### `mrcf/policies/` and `mrcf/manipulation.py` — doing things

A policy drives the arm to do one thing, from geometry. It takes a `Target`
(where an object's foot is, how tall and wide it is), the `Obstacle`s beside
it, and optionally a `Vision` hook for looking while it works. It returns an
`Outcome`: whether it worked, what it measured, and how it is holding what it
picked. It never sees a name, the world model or the memory.
`policies.parallel_jaw.ParallelJaw` is the two-finger pick and place.
Each arm configures it with its gripper's measurements (see `make_policies()`
in `mrcf/backends/so101`).

How a pick comes at an object depends on where the object stands, because a
different limit takes over at each distance from the arm's base. Each of the
three approaches is the first choice in its own zones
(`GripperSpec.near_base_m`, `far_reach_m`), with the others as fallbacks:

| Approach | Where | What it does differently |
|---|---|---|
| `top_down` | Mid-reach, and close to the base wherever it can reach | Hovers, lines up with the wrist servo from its calibrated height, and goes straight down |
| `near_base` | Close to the base, where `top_down` cannot reach | Hovers low instead of climbing to the servo's height, lines up with the fixed camera, and lifts only as high as it can reach. The arm's own base counts as an obstacle |
| `reach_out` | Far out | Uses the lean the arm needs to reach at all. The fixed camera lines up once that lean is far from the servo's, and a little joint margin is kept. With jaws that open along the arm, the grasp goes higher up the object so the tipped jaws clear its top |

Each approach is checked against the arm's reach before anything moves. A
retry tries the next one that can be done, and if none can, the pick is
refused without moving. `Outcome.approach` and `Outcome.attempts` say which
approach was used and why each attempt was retried. Approaches are data
(`mrcf/policies/approaches.py`). Each lines the jaws up over the object with
its list of aligners (`mrcf/policies/aligners.py`): the foot as seen by a
located wrist camera, the fixed camera's sighting, or the wrist servo. So a
new way of picking is a new `Approach`, plus one `Aligner` if it lines up in a
new way.

A **learned** policy (`policies.learned.LearnedPolicy`) is a trained model,
such as ACT, diffusion, SmolVLA or a SAC policy from HIL-SERL. It runs on the
arm's stream until its segment is done (`capabilities.ActionStreaming`: the
arm's observation and action, one tick at a time, through the arm's own safety
checks). `chain()` puts it between planned steps. It travels as a package
(`mrcf/policies/packages.py`): the model, plus a manifest naming the stream
it was trained on, whose fingerprint says which arms can run it. You install
it by reference: a folder, the Hugging Face hub, or a source an installed
plugin adds, such as a store's.

[POLICIES.md](POLICIES.md) takes a new policy of either kind from an idea to a
verb the planner can use, step by step. For a planned one, that means writing
and testing it. For a learned one, it means choosing what to train it on
(`tools/show_stream.py`), packaging it, checking that it fits, running it, and
reading its refusals.

`manipulation.install(policies, vision, scenario)` registers the standard
actions: `pick`, `place`, `move_above`, `home`, `open_gripper` and
`close_gripper`. Each finds the named object in the world model, runs the
policy, and writes the outcome back. The `Scenario` hooks carry what only the
task knows: how tall an unmeasured object could be, how to describe one, what
to log, and, optionally, how far to lean the gripper over a particular spot.
That last is an approach rule: where the arm's search for a lean starts.
Chess has `config.APPROACH_RULES`, now empty because the approaches above
choose the lean. `manipulation.object_action` adds a task's own verb on one
named object, such as a push or a nudge, with the same bookkeeping.

**Standing a fallen object up** (`mrcf/policies/stand_up.py`, `ParallelJaw.stand_up`)
is one such verb. Given how the object lies (`Target.lying`: its axis, length,
and the knob at its top, from `mrcf/perception/lying.py`), the jaws come
straight down, turned square across it. They grip the knob low, with the fixed
jaw beside it, and lift straight up. The knob turns between the pads, the
heavier foot slides in underneath, and the object is let down standing. After
a miss, the wrist camera looks again from above the object and the next try
aims at what it sees, without the arm going back to park
(`capabilities.WristSurface`: where the wrist camera's pixels land, from the
servo's calibration). Chess registers it as `stand_up`.

### `mrcf/interlock.py` and `mrcf/watchers.py` — stopping, and running alongside

```python
from mrcf.watchers import Watcher

def hands(frame, model):
    if hand_near_arm(frame, model):                       # your detector
        env.interlock.trip("hand near the arm", source="hands")
    else:
        env.interlock.clear("hands")

Watcher(env, "overhead", hands, hz=10, fail_safe=True).start()
```

A trip holds the arm where it is at once, from the tripping thread, without
waiting for the next command. It raises `MotionStopped` out of whatever was
moving the arm, and nothing moves again until every source has cleared. A
`Watcher` runs your function on a camera's newest frames on its own thread,
while plans and moves carry on. With `fail_safe=True`, a watcher that crashes
or loses its camera stops the arm itself. A deployment lists its watchers in
`config.WATCHERS`.

`mrcf/flight_recorder.py` writes everything that happens to one timeline: what
each policy decided, each move and grip, each pause, and what each scan
changed. Underneath, it keeps the last minute of joint readings and camera
frames. When a standard action or a verb fails, needs another attempt, or is
contradicted by the next scan, it leaves an incident under `runs/flight/`:
the timeline from just before it, the frames, and a story to read
(`python -m mrcf.flight_recorder`). Any other failure leaves one too, with its
traceback: a command's error, a deployment's start, a crash on any thread.
Each run's folder also keeps:

- `session.json`: which arm and deployment ran, with which cameras and
  sensors, at which commit
- `console.log`: everything printed, time-stamped
- `timings.json`: every command, plan, step, action and model call, counted
  and timed, failures included

`python -m mrcf.flight_recorder --timings` shows where a run's time went,
largest first.

No picture of a person is kept (`mrcf/privacy.py`). A calibrated camera's
frames keep only what lies toward the arm's work volume, so the room, the
ceiling and anyone standing at the table are blacked out. Every face found is
covered, using OpenCV's YuNet face model (fetch it once with
`python tools/fetch_face_model.py`). The computer's own cameras (a laptop's,
an iPhone's) are never opened. An arm refuses to open its cameras while fewer
of the rig's cameras are plugged in than it names (`mrcf/devices.py`), and
`tools/find_cameras.py` lists cameras by name without opening any.

One watcher ships with the framework: `MotionGuard` (`mrcf/motion_guard.py`).
It watches a calibrated fixed camera and trips the interlock when anything
other than the arm moves near it, such as a hand reaching in. It tells the
arm's own motion apart by drawing the arm into the image from its joint angles
(`RobotEnv.arm_shape`) and ignoring changes there. Its trips pause the move
instead of abandoning it (`Interlock.pause_only`): once nothing has moved for
2 s, the guard clears and the move carries on from where it stopped. Every arm
built on `backends.rig.RigEnv` starts one on each camera in its
`MOTION_GUARD_CAMERAS` (the overhead camera, and any other fixed camera that
sees what it cannot) every time it connects (`robot_config.MOTION_GUARD`). So
every program that moves the arm is watched. `check_motion_guard.py` shows
what the guard sees.

```python
from mrcf.motion_guard import MotionGuard

guard = MotionGuard(env, "overhead", zone=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).start()   # fail-safe Watcher
```

### `mrcf/vlm.py`, `mrcf/scene_state.py`, `mrcf/action_registry.py`, `mrcf/controller.py` — models, world model, actions

`mrcf/vlm.py` talks to the models (perceive, discover, decompose, plan).
`mrcf/scene_state.py` is the world model they read. `mrcf/action_registry.py`
holds the actions a plan may use and runs them (`dispatch`). `vlm` re-exports
both. `mrcf/controller.py` is the loop every deployment runs. It decomposes a
command, plans it, checks the plan and dispatches it step by step, re-planning
from a fresh look after a failed step. The deployment supplies hooks
(`controller.Deployment`): how it scans, where the arm waits during a scan,
and where home is.

```python
from mrcf import vlm
from mrcf.vlm import SceneState, register_action, register_condition, plan, dispatch

vlm.configure(model_server="ollama",          # or "openai", at model_url=...
              model="gemma4:31b", planning_model="gemma4:31b",
              observe_cameras=["overhead"])

@register_action("nudge", description="Push an object a short distance.",
                 parameters={"object_name": "Exact name of the object."},
                 conditions={"pre": ["object_exists", "nothing_carried"]})
def skill_nudge(env, scene_state, object_name, **_):
    ...
    return success            # dispatch() passes this back to the controller
```

The `description` is the planner's only clue about when to use an action.
Pre-conditions block an action. Post-conditions warn when the world model and
the robot disagree afterwards.

### `mrcf/skill_executor.py` and `mrcf/safety.py`

`run_policy(fn, name, env, **kwargs)` runs a policy under the execution
monitor, whose limits are checked every control cycle through the interlock.
When the policy fails, it asks the vision model to describe the scene (but
not when it refused before moving, or was stopped). `run_skill` does the same for a function that returns `(success, message)`.
`safety.validate_plan(actions, scene_state)` checks a plan before anything
moves. The controller runs it on every plan. If you run skills yourself,
outside the controller, call it yourself.
`safety.configure()` sets the workspace bounds, force thresholds and any
sequence rules (for example, no `pick` while already holding something).

### `mrcf/object_memory.py` — who each object is

A scan says what is where now. The memory says which of those things is
which. It matches each scan against what it remembers. An object moved by
hand is recognised as having moved. One taken away is kept as missing, and
recognised when it comes back. The arm's own picks and places are recorded
with certainty. Each label records who gave it: the vision model's guess, a
shape match, a taught layout, or the user. A label from a better source is
never overwritten by one from a worse source. The memory is saved to disk, so
it survives restarts. Its recent events go into the planner's prompt, so "put
it back" has a "back".

### `mrcf/perception/` — measuring objects, from any calibrated camera

Frames and camera models go in, and measurements come out. Nothing in
`mrcf/perception/` knows which arm holds a camera, what the task is, or what
the objects are called. A test checks that it imports no world model, memory,
planner, arm or policy.

Given a camera that knows where it is relative to the table, an object's
outline (from SAM2, via `mrcf/perception/segment.py`) becomes measurements:
where its foot stands in the arm's frame, how wide it is, and, from a slanted
view, its height and its width at every height. `surface_scan.scan()` does
this for everything on the surface, and `scene_scan.observe()` feeds the
results to the memory without knowing what the objects are. On a patterned
surface, a photo of the empty surface tells objects apart from the pattern.
The two are compared after levelling the light, so a change of exposure or
daylight does not count as a change of scene.

### `mrcf/perception/hand_eye.py` — locating a wrist camera

It works out where a camera bolted to the gripper sits relative to the
gripper, from views of a flat grid at several arm poses. Once that is known,
every frame from that camera comes with its pose from the joint angles. The
SO-101's `tools/so101/calibrate_gripper_pose.py` uses it.

---

## Extending it

[POLICIES.md](POLICIES.md) covers adding a policy, planned or learned, step by
step. [DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md) covers adding a task or an arm,
with checklists. [CONTRIBUTING.md](CONTRIBUTING.md#the-rules-that-keep-the-structure)
lists the rules that keep the structure, each with the test that checks it.
The main points:

- **A capability.** Write a policy in `mrcf/policies/` (geometry in, `Outcome`
  out), and give it to the planner with `manipulation.object_action` or your
  own `@register_action`. For a new way to pick, add an `Approach` and maybe
  an `Aligner`. A learned capability is trained elsewhere and runs on the
  arm's stream.
- **A deployment.** One arm, one layout, one task, in `deployments/<name>/`.
  It is a package (`__init__.py`) whose modules import each other relatively
  (`from . import config as cfg`), with:
  - `config.py`: names its arm (`ARM_NAME`) and serves the arm's
    `robot_config` live instead of restating it
  - `skills.py`: calls `manipulation.install()` with the arm's policies
  - `llm_controller.py`: short; its `main()` runs `controller.run()` with the
    deployment's hooks
  - `profile.py` (optional): its rules for its arm

  `run.py` offers the deployment when its arm is chosen. The same task on
  another arm or layout is another deployment. A task two deployments share
  belongs to the framework (`mrcf/tabletop/`), so a table deployment is only a
  config, an optional profile and a few lines. To start one from the nearest, run
  `python tools/new_deployment.py NAME --from SOURCE [--arm ARM]`. It copies
  the code and offers the source's rules as candidates.
- **An arm.** Copy `mrcf/backends/_template` to `mrcf/backends/<arm>/`. It
  exports `NAME`, `config`, `make_env()` and `make_policies()`, and has
  `env.py`, `robot_config.py` (the rig settings every arm has, plus its own)
  and, for what only it needs, `tools/`. Run the tools with `MRCF_ARM=<arm>`,
  and give the arm its own deployment. Build the env on `backends.rig.RigEnv`:
  you give it the gripper's pose and drive the motors, and the cameras'
  geometry, the wrist servo and the sentry come with it. Add `RealRobotEnv` if
  the framework should own the cameras. `check_backend()`
  (`mrcf/backends/contract.py`) checks the arm offline before it moves. `mrcf/capabilities.py` lists the optional features the
  policies use when an arm has them.
- **A model server.** For an API that is neither Ollama's nor
  OpenAI-compatible, write a `model_server.ModelServer` whose `send(request)`
  returns the answer's text, in a package of your own. `post()` sends HTTP and
  raises an error for what the status means: `Unreachable`, `TimedOut`, `Busy`
  (retried), `Refused`, or `FormatRefused` (retried without the schema). The
  key goes as `self.headers()`'s bearer token unless you send it your own way.
  Name the server in the package's `pyproject.toml`, then run with
  `MRCF_MODEL_SERVER=my_api`:

  ```toml
  [project.entry-points."mrcf.model_servers"]
  my_api = "my_package.server:MyAPI"
  ```

  ```python
  from mrcf.model_server import ModelServer, Request

  class MyAPI(ModelServer):
      name = "My API"
      default_url = "https://api.example.com/v1/answer"

      def send(self, request: Request) -> str:
          # request.model, .system, .turns (the last is the question), .images (base64,
          # shown with the last turn), .json_mode, .schema, .max_tokens, .temperature
          body = {"model": request.model, "instructions": request.system, "input": request.turns[-1]["content"],
                  "pictures": request.images, "schema": request.schema}
          return self.post(self.url, body, request).json()["text"]
  ```

[SKILL_LIBRARY.md](SKILL_LIBRARY.md) explains how to debug the pipeline one
stage at a time.

---

## Documentation

| | |
|---|---|
| [COMMANDS.md](COMMANDS.md) | Every command, the first-time setup in order (for the SO-101 and for another arm), and what to re-run after a change |
| [HARDWARE.md](HARDWARE.md) | The rig: the cameras, board and tags, where they go, and the SO-101 rig as built |
| [POLICIES.md](POLICIES.md) | Adding a policy, planned or learned, step by step |
| [DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md) | Adding a skill, a task or an arm, and where to record what you learn about each |
| `mrcf/NOTES.md`, `mrcf/backends/<backend>/NOTES.md`, `deployments/<task>/NOTES.md` | What the framework, each robot and each task cannot do yet, and what each has shown that its code cannot say: quirks, failures and workarounds |
| `HISTORY.md` beside the code (`mrcf/`, `mrcf/policies/`, each backend, each task) | Why the code is the way it is: the dated runs behind its values and fixes |
| [SKILL_LIBRARY.md](SKILL_LIBRARY.md) | The action registry, and debugging one stage at a time |
| [CONTRIBUTING.md](CONTRIBUTING.md) | The rules that keep the structure, step-by-step guides for a new arm or deployment, and what to do when something goes wrong |
| [CHANGELOG.md](CHANGELOG.md) | What changed in each release |
| [SECURITY.md](SECURITY.md) | Reporting a security hole, or the robot moving when it should not. Both are reported privately |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions are
accepted under the same license.
