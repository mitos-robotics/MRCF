# The rig

This page covers what the framework expects around any arm, and how the SO-101
rig it was built on is laid out. The software side of a new arm is in
[DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#adding-an-arm). The calibration order
is in [COMMANDS.md](COMMANDS.md#first-time-setup-on-another-arm).

```
                 overhead camera (fixed), ~0.5 m up,       (more fixed cameras:
                 looking back at the arm, ~60° down         any number, anywhere)
                              ◣
          ┌───────────────────────────────┐   ← board tags on the border
          │                               │
          │       work surface /          │
          │       chessboard              │
          │                               │
          └───────────────────────────────┘
                        ▲
                 gripper (+ wrist camera, gripper tag)
                        │
                     arm base
```

## What it assumes

- One flat work surface, level enough that every object stands on the same
  plane (`TABLE_Z`).
- An arm with a two-finger gripper working from above. Its backend can say
  where the gripper is from the joints (`gripper_frame()`), stop where it is
  when told (`hold()`), and say whether a move arrived. The backend hides
  whether the arm has six joints or five, servos or an industrial controller.
  Another end-effector (suction, a hand) needs its own policy library.
- As many cameras and sensors as the rig has, each listed by name in the arm's
  config (`CAMERAS`, `SENSORS`). The tasks so far rely on a fixed camera over
  the work surface; everything else is optional. The next section says what
  each does and what you lose without it.
- A computer that runs the models through Ollama: this one, or one it can
  reach over the network. Or a hosted API
  ([COMMANDS.md](COMMANDS.md#models-on-another-machine-or-a-hosted-api)), which
  is then sent the camera's pictures.

## Cameras and sensors: what each is for

Every camera is listed in the arm's `CAMERAS` as a `backends.rig.Camera`: fixed
or on the gripper, its source and size, and, for any but the two below, its
calibration. Every sensor is in `SENSORS` as a `backends.rig.Sensor`: its kind,
unit, shape and how the env reads it. Two cameras have jobs the framework knows
by name, and either can be `None`.

| | What it is used for | Without it |
|---|---|---|
| **A fixed camera over the surface** (`OVERHEAD_CAMERA_NAME`) | Scans: finding, measuring and naming everything. Sighting the gripper tag to correct where the jaws are. Checking its own calibration every scan. The motion guard | The chess task has nothing to scan with. With no camera to watch through, an arm with `MOTION_GUARD` on refuses to connect |
| **A camera on the gripper** (`CAMERA_NAME`) | Measuring an object from right above it before the grasp. Lining the jaws up over it (the wrist servo). The stand-up's second look after a miss | Picks line up with the fixed camera alone, from its slanted view |
| **More fixed cameras** | The motion guard watches through each one listed in `MOTION_GUARD_CAMERAS` (one camera cannot see under the arm or behind it). The flight recorder keeps their frames. Their geometry (`camera_model`, `pixel_to_world`) serves perception and your own deployments | — |
| **More cameras on the gripper** | Their pose from the joints, for perception. The flight recorder keeps their frames | — |
| **A gripper tag** (`GRIPPER_TAG_ID`) | The fixed camera's correction of where the jaws are, and the tag tour | The jaw position comes from the kinematics alone |
| **A force sensor** (kind `"force"`: a force, or a wrench) | The safety monitor's contact-force limits (`get_contact_force()`) | Those limits never fire. The floor and height limits still do |
| **A gripper contact sensor** (kind `"gripper_contact"`) | Telling a grip from a miss (`gripper_loaded()`) | A grip is judged from how far the jaws closed |
| **Any sensor** | `read_sensor()`. The flight recorder keeps its readings, and a learned policy's stream carries them | — |

A camera is named by its OpenCV index, and an index is only a place in today's
list. Unplug the rig's camera, and index 0 on a laptop is its own camera,
looking at you. So an arm opens its cameras only while as many rig cameras are
plugged in as it names (`mrcf/devices.py`). The computer's own cameras (the
laptop's, an iPhone's Continuity Camera, Desk View) are known by name and never
opened. On macOS the names come from `ffmpeg`, which must be installed.
`run.py` says before every session which cameras are plugged in.
`tools/find_cameras.py` lists them by name without opening any.

Not yet: a scan from several fixed cameras at once (one does the scan), a depth
camera's depth, and force-guided grasping. Each is a perception or policy
feature to build once a rig has the hardware; the cameras and sensors are
already there to read.

## The arm and its gripper

- The controller's own safety comes first: a stop button within reach and, on
  an industrial arm, safety planes and a speed limit set on the pendant to
  match `WORKSPACE_BOUNDS`, so the two stop at the same place.
- The gripper's geometry goes in the arm's config, from its drawing: how wide
  it opens, where the jaws meet when closed (`JAW_CLOSED_POINT_M`, in the
  flange's frame), whether one jaw moves or both (`JAW_CENTRE_SLIDE`), and the
  fingers' size. `get_ee_pos()` must return the point the jaws close on. On the
  SO-101, an 8.5 mm error there closed the jaws beside every piece.
- A gripper that senses its grip (a Robotiq's `OBJ`) tells a hold from a miss
  more reliably than an opening can.

## The overhead camera

- Put it on the far side of the work surface, looking back at the arm, about
  half a metre above it and 60° down. It must see the gripper's outer end in
  every pick pose. On the SO-101 it first hung above the arm's base, looking
  down the forearm. The wrist then hid the gripper body in every pick pose, so
  a tag on it was never seen.
- Mount it rigidly. Every scan checks the camera against the board's tags and
  re-poses it if it has moved (`mrcf/overhead_calibration.py`). But a camera
  that sways while the arm works is wrong between scans.
- It should see the whole work surface with a margin, and as little else as
  possible. The motion guard watches a zone of the surface
  (`MOTION_GUARD_ZONE`). With the floor beside the table in view, shuffling
  feet read as movement.
- On the SO-101 it captures 1280x720. The pipeline works at 640x480
  (`RENDER_W`, `RENDER_H`).
- Use even, diffuse light. Auto exposure on the SO-101's cameras clipped the
  light squares to white (35% of them on one afternoon), and a cream piece on a
  clipped maple square is invisible. `python tools/check_exposure.py --show`
  measures it. Those cameras ignore manual exposure, so a lamp or a curtain was
  the fix. A camera that takes manual exposure is worth having.

Calibrate any further fixed camera like the overhead one
(`tools/calibrate_overhead_camera.py --camera NAME`) and put its calibration in
its `CAMERAS` entry.

## The wrist camera (optional)

- Mount it on the gripper body, so it moves with the jaws and sees them, with
  nothing jointed in between.
- Its sharp zone is narrow. Focus it at the height the jaws hover over an
  object. On the SO-101, `tools/so101/check_lighting.py` reads sharpness live
  at the wrist servo's pose.
- Its calibration tools are still the SO-101's. On another arm, leave it
  uncalibrated at first. Picks then line up with the overhead camera alone
  ([COMMANDS.md](COMMANDS.md#first-time-setup-on-another-arm)).

## The board and the tags

- The chessboard is both the calibration target and the chess deployment's
  board: a printed or wooden 8x8 board inside the arm's reach, with its squares
  measured (`CHESSBOARD_SQUARE_SIZE_M`, 40 mm on the SO-101 rig). For a
  different board, change `deployments/chess/config.py` (squares, inner
  corners).
- Board tags are optional, for the closed loop: six AprilTags on the board's
  border, printed to scale by `python tools/make_tag_sheets.py`. Each tag's
  position is measured from a frame, so you need not place them exactly. They
  need only be flat, glued down and off the squares.
- The gripper tag is optional too: one AprilTag on the gripper body, facing
  the overhead camera in every pick pose, with nothing jointed between it and
  the jaws. The tag tour measures where it sits. On the SO-101 it is the 24 mm
  one, on the back of the wrist camera's mount. A bigger one reads better from
  further away.
- Print at 100% ("actual size") and check the sheet's 100 mm ruler.

## The computer

- Linux, Windows or macOS, and Python 3.10 or newer. The SO-101 runs in a
  LeRobot environment, and another arm in whatever environment its SDK needs:
  `pip install -e ".[rig,dev]"` (the cameras, and pytest for its contract
  test), plus the SDK as its own extra in `pyproject.toml` (for the UR5,
  `pip install -e ".[ur5,dev]"`).
- [Ollama](https://ollama.com) with a vision model: `ollama pull gemma4:31b`
  (about 20 GB in memory). The chess deployment uses it for every call (seeing,
  labelling and planning), on the machine beside the arm or one it can reach
  over the network. A Raspberry Pi cannot hold it. Ollama uses an NVIDIA GPU
  (CUDA), an Apple-silicon Mac's (Metal), or, more slowly, the CPU. The SO-101
  rig runs it on an Apple-silicon Mac. Instead of Ollama, any OpenAI-compatible
  server (vLLM, LM Studio, llama.cpp) or a hosted API works
  ([COMMANDS.md](COMMANDS.md#models-on-another-machine-or-a-hosted-api)).
- Optionally, SAM2-tiny for outlining objects (`pip install -e ".[measure]"`,
  ~150 MB on first use).
- Training a learned policy needs a GPU machine and is done there. Running one
  does not ([POLICIES.md](POLICIES.md#a-learned-policy-step-by-step)).

## The SO-101 rig, as built

| | |
|---|---|
| Arm | SO-101 follower (LeRobot) |
| Gripper | the SO-101's own: one moving jaw, 45 mm open |
| Wrist camera | USB, 640x480 at 30 fps, on the gripper body beside the jaws |
| Overhead camera | USB, 1280x720 at 30 fps, ~0.5 m above the board's far side, looking back at the arm |
| Board | 8x8, 40 mm squares, wooden; six 26 mm board tags on its border |
| Gripper tag | 24 mm, tag36h11 id 12, on the back of the wrist camera's mount |
| Computer | an Apple-silicon Mac running Ollama and the arm |

## The UR5 rig, being set up

Its backend drives the real arm. Since 2026-11-12 it has stacked letter
blocks (`deployments/tabletop`), and on 2026-11-20 it put hand tools into a
box (`deployments/toolbench`). It also runs offline and on URSim. The control
box has a power-up fault, first seen on 2026-11-05; the workaround is in
`mrcf/backends/ur5/NOTES.md`. The order to bring the arm up is in
[COMMANDS.md](COMMANDS.md). What is still to measure is in the same notes.

| | |
|---|---|
| Arm | UR5 (CB3, PolyScope 3.15), 850 mm reach, bolted to the table |
| Gripper | Robotiq Hand-E (URCap installed): two jaws, 50 mm stroke, 1.07 kg |
| Link | Ethernet, direct to the computer, on its own address range (the arm's address in `rig/ur5/host`) |
| Wrist camera | none |
| Overhead camera | over the table, looking down at an angle; calibrated by guiding the arm by hand (`tools/calibrate_camera_by_hand.py`) |
| Tasks | tabletop (letter blocks, picked, placed and stacked) and toolbench (hand tools put into a box) |

## Before the first session on a new rig

- The arm's controller stops it at its own limits, and its stop is within
  reach.
- `WORKSPACE_BOUNDS` and the controller's safety planes agree.
- The overhead camera is calibrated, with `MOTION_GUARD = True` set again
  after its first calibration. `tools/check_motion_guard.py` shows the guard
  stops the arm for a hand and not for the arm.
- The first moves run slowly (a UR's speed slider low), with someone at the
  stop.

The motion guard sees movement, not hands, through the fixed cameras in `MOTION_GUARD_CAMERAS`. It stops the arm
0.1-0.15 s after a hand appears. It is a guard, not a certified safety system.