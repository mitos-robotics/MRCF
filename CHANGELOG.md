# Changelog

All notable changes to MRCF are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and MRCF follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, a minor release may change the API.

## [Unreleased]

## [0.1.0] — unreleased

The first public release. MRCF lets you control a real robot arm in plain
English, using models that run on your own machine. The language model only
chooses actions from a fixed list. It never moves the arm itself. Planned or
learned policies do the moving, and a safety layer checks every plan first.

### Control

- Type a command in plain English. MRCF splits it into steps, plans them as
  actions from the fixed list, checks the plan for safety (workspace, joint
  limits, order of steps) and runs it one step at a time. If a step fails, it
  looks again and plans the rest, knowing why the step failed.
- Use Ollama on this computer or another one, or any OpenAI-compatible server
  or hosted API (`mrcf/model_server.py`).
- Give a deployment its own tasks planned in advance (`TASKS`), and turn the
  planner off if you don't need it (`PLANNER`).
- Start everything with `python run.py`. It asks for the arm, shows what the
  arm has (cameras, sensors, calibrations, and anything not yet checked on the
  rig), then asks for the job. You can also start a new arm or a new
  deployment from it.

### Arms

- The SO-101, through LeRobot (`mrcf/backends/so101`).
- The UR5 (CB3, PolyScope 3.x) with a Robotiq Hand-E gripper, through ur_rtde
  (`mrcf/backends/ur5`). Every move's path is checked against an allowed box,
  contact forces are limited, and it has its own setup and calibration tools.
- Any other arm, starting from a template (`mrcf/backends/_template`,
  `tools/new_arm.py`). A contract test (`check_backend()`) checks the new arm
  offline before it moves. Optional abilities are declared as capabilities
  (`mrcf/capabilities.py`), and each arm's settings and plans are recorded as
  test baselines.
- Facts that a grasp depends on, such as which way the jaws open, must be
  checked on the real arm before it will pick anything (`mrcf/arm_facts.py`,
  `tools/check_jaw_direction.py`).

### Picking and placing

- Pick and place with a parallel-jaw gripper. The arm comes at each object in
  the way that suits where it stands (`top_down`, `near_base`, `reach_out`),
  and it can stand a fallen object back up.
- Objects that need more than a simple squeeze: picked by a handle, by one
  wall (a cup or a bowl), or lying flat. A long tool is gripped by the handle
  next to its heavy end, or else at its balance point. Before the jaws come
  down, MRCF checks that one finger will land on each side.
- Putting things into containers. Containers are found and remembered by
  name, and the opening is told apart from the walls and flaps by its shade.
  Things are lowered in, or dropped from just above the walls, and a long
  thing is laid along the container.
- Recovering grips: the arm can look to find an object's height, correct for
  where its jaws really close (`mrcf/grasp_correction.py`), find a missed
  object again and retry, and use a catalogue of the kinds of object a
  deployment handles.

### Learned policies

- Run trained models on the arm: a LeRobot checkpoint (ACT, diffusion, …) or
  a Stable-Baselines3 model, fed the arm's joints and camera frames. Each
  model comes with a manifest that says which arms can run it. MRCF does not
  train models.

### Seeing

- The overhead camera is calibrated by guiding the arm by hand or by a tour
  of tags, and it is checked on every scan. If the camera has moved, the scan
  stops.
- Objects are measured from their outlines (SAM2). An object memory keeps
  track of each object across scans and restarts. The wrist camera lines the
  jaws up with the object at the grasp, and scans can see past the arm.

### Safety and privacy

- The motion guard watches through a fixed camera and pauses the arm when
  something other than the arm, or what it is carrying, moves near it. The
  move carries on once the area is clear. It is a guard only. It is not a
  certified safety system.
- Every arm command's path is checked against the table and the allowed box
  before it runs.
- No pictures of people are kept. Faces are covered, and calibrated cameras
  are cropped to the work area. The computer's own cameras are never opened.
  Rig cameras are found by name.

### Records

- The flight recorder keeps each run's session, console output, timings and
  failures under `runs/flight/`, plus an incident report when an action goes
  wrong (`python -m mrcf.flight_recorder`).
- Each rig's own files (calibrations, photos, downloads) live outside git, in
  `rig/<backend>/` under `MRCF_HOME`.

### Deployments

A deployment is one arm, one layout and one task, with its own rules for that
arm (`profile.py`). `tools/new_deployment.py` starts a new one from the
closest existing one.

- chess (SO-101): scans the board, names the pieces and moves them. It can
  play openings and a game against Stockfish.
- tabletop (UR5): picks, places and stacks letter blocks.
- tidy (SO-101): puts everyday things away into a tub.
- toolbench (UR5): puts hand tools into a box.

### The project

- Runs on Linux, Windows and macOS, with Python 3.10 or newer.
- Licensed under Apache-2.0 (`LICENSE`, `NOTICE`).
- CI runs on every push: the offline tests on three operating systems, with
  and without the optional libraries, the type check, and a separate SO-101
  job. Pushing a `v*` tag builds a release.
- Includes a contributing guide, a code of conduct, a security policy, and
  issue and pull request templates.

[Unreleased]: https://github.com/mitos-robotics/MRCF/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mitos-robotics/MRCF/releases/tag/v0.1.0
