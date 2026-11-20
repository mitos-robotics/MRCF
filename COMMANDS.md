# Command reference

Commands come in three kinds:

- `tools/…` is about the **rig**, on whichever arm runs: its cameras, the
  tags, the overhead camera's calibration, the motion guard. They know nothing
  of the task beyond the board they calibrate against.
- `tools/so101/…` is about the **SO-101** itself: its poses, servos, joint
  calibration and wrist camera calibration.
- `deployments/chess/…` and `run.py` are about the **task**.

A few tools need task values (grasp heights, the calibration target's size).
They take `--deployment NAME`. With only one deployment present, they use it.

All paths are from the repo root. On the SO-101 everything runs in your
**lerobot** environment. Unless noted, scripts take **no arguments**. Most of
them connect to the arm, and many move it.

`python run.py` asks which arm to drive. Enter takes the one plugged in
(`--arm so101` chooses without asking). Then it asks which of its deployments
to run; Enter takes the arm's own (`--deployment NAME` chooses one and brings
its arm).

A deployment is one arm, one layout and one task, and it names its arm: `chess`
and `tidy` are the SO-101's, `tabletop` and `toolbench` the UR5's. A deployment
run directly drives its own arm. The `tools/` drive the arm that `MRCF_ARM`
names: `so101` by default, or another arm's folder name under `mrcf/backends/`.
A `MRCF_ARM` left exported in a terminal quietly picks the arm for them
(`unset MRCF_ARM`). To set up another arm, see
[First-time setup on another arm](#first-time-setup-on-another-arm).

---

## Run it

| Command | What it does |
|---|---|
| `python run.py` | The normal way in. Asks which arm, listing each as plugged in or not (Enter takes the one that is; `n` a new one). Shows what the arm has, opening nothing: gripper, cameras (calibrated? the rig's plugged in?), sensors, whether kept pictures are guarded. Then asks which deployment (chess or tidy on the SO-101, tabletop or toolbench on the UR5; Enter takes the arm's own; `n` a new one, from the nearest). Records the run from there in `runs/flight/<time>/`. |
| `python run.py --arm so101` | Chooses the arm without asking. |
| `python run.py --deployment NAME` | Runs that deployment on its arm without asking (tidy: the SO-101). Also takes `--arm`. |
| `python -m deployments.chess.llm_controller` | Starts the controller directly, without `run.py`, on the deployment's arm (chess: the SO-101). Run it from the repository root, since the deployment is a package (`deployments.chess`). |
| `python run.py --deployment tidy` (the SO-101) | Puts everyday things away by plain English: "put the red figure in the tub", "put everything into the plastic tub", "put it back". It finds the tub by name (`find`, or by itself the first time you ask for it). The wrist camera measures the tub's walls on the first place into it; they are kept across restarts in `rig/so101/tidy_containers.json`. An open-topped thing too wide to close the jaws round (a cup, a bowl, a scoop) is picked by one wall (`pick_by_rim`). `place_in container=... drop=true` drops a sturdy thing from above the walls instead of lowering it in. The first start learns where the jaws are in the wrist camera's picture (`learn_jaw_view`, ~30 s over a clear table), and again after the camera is moved. It works from the bare table as touched (`rig/so101/tidy_surface.json`), not the chessboard's top. |

On startup the controller homes the arm, moves to the parking pose, scans the
board, prints what it found and leaves you at a `You:` prompt. On exit it homes
and disconnects.

---

## At the `You:` prompt

Type plain English and a planner turns it into the actions below.
`pick up the pawn` and `move the pawn to e5` both work.

A deployment with tasks planned in advance (`TASKS` in its `config.py`) runs
one when you type its name, with no planning: `tidy up` on the tidy table, or
`go` where there is one task. With `PLANNER = False` (as in tidy), those tasks
and `put ... into ...` are all it does. Anything else is answered with its
tasks, and a failed step is reported, not planned again. The prompt lists its
tasks at the start. See DEPLOYMENT_SPEC.md, "A task planned in advance".

| Action | Parameters | What it does |
|---|---|---|
| `pick` | `object_name` | Pick up an object from its last observed position. |
| `place` | `target_name` | Place the held object at a named target. |
| `move_above` | `object_name`, `height` | Hover over an object without grasping. Height defaults to `HOVER_HEIGHT`. |
| `stand_up` | `object_name` | Stand a piece that has fallen on its side back up where it lies. The arm parks and the overhead camera measures how the piece lies. Then, from home, the jaws grip its top and lift it upright. After a miss the wrist camera looks again from above and the arm retries there, up to 3 times, without parking. Refused (the arm only parks) if the camera cannot see it lying whole (a cream head on a maple square), if it may be standing, or if its top points straight at the arm. |
| `nudge` | `object_name`, `target_name` | Push an object a short way (at most 4 cm) with the jaws shut, without picking it up: onto `target_name`, or if none, back to the middle of its square. The push is low on its foot, raised for a tall piece so the gripper clears its top, but never more than halfway up the shortest it could be. Refused before moving if the fingers have no room behind it or it would be pushed into something. `/nudge white_bishop_f1`. |
| `move` | `from_square`, `to_square` | Chess: move the piece on `from_square`, whatever the scan named it, to `to_square`. A piece on `to_square` is taken off the board first. `/move e2 e4`. |
| `play_opening` | `opening`, `rounds` | Chess: play a book opening (`/openings`) for both sides from the starting position, checking every move with a scan, then take every move back; repeat `rounds` times. `all` plays every line that captures nothing and is in reach. If the board is not as the rules say, it stops and moves nothing more. Each move is logged to `deployments/chess/logs/openings.jsonl`. `/play_opening kings_pawn 3`. |
| `openings` | — | Chess: list the book, with what each line needs that is out of reach. |
| `new_game` | `arm_plays`, `strength` | Chess: start a game against the arm ([Play a game](#play-a-game)). The board is checked at the starting position and the arm records what every piece is. If the arm plays white, it moves first. `strength` is the engine's Elo (default 1500). `/new_game`, `/new_game white 1800`. |
| `my_move` | `move` | Chess game: tell the arm you have made your move. Without `move`, the arm finds it with a look; with it (`e5`, `Nxe4`, `O-O`, `e7e5`), the look checks it. Then the arm replies. Also use it for a move the arm asked you to make for it. `/my_move`, `/my_move e5`. |
| `game` | — | Chess game: the board as the game has it, the moves, whose move it is, the engine's view. |
| `hint` | — | Chess game: the engine's best move for you. |
| `resign` / `end_game` | — | Chess game: `resign` resigns; `end_game` stops the game without a result. |
| `resume_game` | — | Chess game: after a restart, carry on with the game saved in `deployments/chess/logs/game.json`, checking the board against it first. |
| `home` | — | Return to the safe home pose. It is the reset button after anything goes wrong. |
| `open_gripper` | — | Open the jaws fully. |
| `close_gripper` | — | Close the jaws. |
| `observe` | — | Re-scan the board and update object positions. |
| `identify` | `object_name`, `what` | Record what a piece is when you say so ("that's the queen on d5"). Remembered through moves and restarts. |
| `forget` | — | Forget every remembered piece and rescan (after you reset or rearrange the whole board). |
| `list_objects` | — | Print known objects and their last observed positions. |
| `list_actions` | — | Print every available action. |
| `status` | — | Live sensor readings: joint angles, end-effector position, gripper opening. |
| `answer` | `question` | Answer a question about the scene from the camera. |
| `calculate` | `expression` | Evaluate maths. `sqrt`, `sin`, `log`, `pi`, … |

Commands starting with `/` run one action directly, with no VLM: `/actions`,
`/objects`, `/pick white_pawn_d4`, `/move_above white_pawn_d4 0.08`.
[SKILL_LIBRARY.md](SKILL_LIBRARY.md) shows how they isolate a failure.

`quit`, `exit` or `q` ends the session.

**A pick refused for lack of room makes room first.** When a neighbour leaves
the jaws too little room ("jaws need 29 mm open here but white_king_e1 leaves 6
mm"), before moving anything the arm weighs pushing the piece itself a little,
lifting the neighbour onto a nearby empty square, and pushing the neighbour
clear. It makes the first that works (`[Room] Making room: ...`), picks, and
puts the neighbour back after the place. Known piece types (from
`/starting_position`, `identify` or a game's `/new_game`) give it far more to
work with: their height is known, so the jaws can turn and a push can go low.

**Positions are only measured when a scan runs**: at startup, on `observe`, and
after a failed skill. If you move a piece by hand mid-session, say `observe`
before asking the arm to pick it, or the arm goes to where the piece used to
be.

---

## Play a game

Play chess against the arm. You move your pieces by hand, a chess engine
(Stockfish) chooses the arm's moves, and the arm makes them. A scan checks
every move. Install the engine once: `brew install stockfish` (Linux:
`apt install stockfish`). Any UCI engine works, named by `MRCF_CHESS_ENGINE` or
by `ENGINE_PATH` in `deployments/chess/config.py`. That file also sets the
default strength (`ENGINE_ELO`) and thinking time (`ENGINE_MOVETIME_MS`).

```
You: /new_game                 ← board at the starting position; the arm (white) plays 1. e4
     (you play e7-e5 on the board)
You: done                      ← or /my_move, or /my_move e5: the arm looks, then replies 2. Nf3
You: what should I play?       ← /hint
You: /game                     ← the board, the moves, whose turn
```

- The arm plays the side nearest it (`ROBOT_PLAYS`, white). Your pieces start
  on the far ranks, at the edge of its reach. When you capture, take its piece
  off the board by hand.
- The arm finds your move from which squares have a piece. If two moves leave
  the same squares filled (a knight that could have captured on either of two
  squares), it asks: tell it with `/my_move Nxe4`. A promotion is taken to be a
  queen unless you say otherwise (`/my_move e2e1n`).
- If the arm cannot make a move itself (out of reach, a promotion, or a failed
  carry), it asks you to make it on the board. Say `done` once it is made;
  making your own move as well is fine. If a carry leaves the board other than
  the game expects, the arm prints the board as it should be. Set it up that
  way and say `done`.
- The game ends on checkmate, stalemate, too little material, fifty moves or a
  third repetition. Every move is saved.
- You need not say which game. At startup, the arm's first look carries on the
  saved game if the board shows it, and pieces set up at the start begin a new
  game (say `go` or `done` for the arm's first move). Mid-game, set the pieces
  up again and say `done`: a new game starts and the arm plays first.
  `/new_game` and `/resume_game` still work for a board that shows neither.
- A `/` command skips the language model: `/my_move` answers at once, while
  `done` takes about a minute for the model to read it.

---

## First-time setup, in order

This is the SO-101's. For another arm, see
[First-time setup on another arm](#first-time-setup-on-another-arm).

```bash
pip install placo                  # Cartesian IK — without it pick/place won't run
ollama pull gemma4:31b             # every model call: board reading, scene description, planning (~20 GB)
```

Or run the models elsewhere:
[Models on another machine, or a hosted API](#models-on-another-machine-or-a-hosted-api).

| Step | Command | Produces |
|---|---|---|
| 1 | `python tools/so101/fetch_urdf.py` | The SO-101 URDF and meshes (~19 MB, not committed), into `rig/so101/urdf/`. All IK needs them. Files already there are not fetched again. |
| 2 | `lerobot-calibrate` | The joint calibration, which this framework reads as is (lerobot's own tool). |
| 3 | `python tools/so101/find_poses.py` | `HOME_DEGREES`, `TABLE_Z`, `OBSERVE_DEGREES`, `CALIBRATION_PARK_DEGREES`. |
| 4 | `python tools/so101/calibrate_gripper_width.py` | **The arm moves** (it holds the gripper up while the jaws step). `GRIPPER_WIDTH_TABLE`: the real jaw opening at each gripper motor position. You measure the gap at the jaw ends with a ruler at six openings. Without it every width is a straight-line guess, badly wrong for a hinged jaw. |
| 5 | `python tools/so101/calibrate_arm_offsets.py --deployment chess` | `JOINT_OFFSETS_DEG` (corrections to the arm's joint readings) and, from the same fit, `TABLE_Z`, `OVERHEAD_HOMOGRAPHY`, `OVERHEAD_CAM_MODEL`, `ARM_FRAME_SCALE`. With torque off and the board clear, you touch 9 board corners with the closed jaws. |
| 6 | `python tools/so101/check_reach.py` | `WORKSPACE_BOUNDS`. |

Closed-loop control is optional, but it is the only thing that corrects an
absolute frame error (see [SKILL_LIBRARY.md](SKILL_LIBRARY.md)):

| Step | Command | Produces |
|---|---|---|
| 7 | `python tools/make_tag_sheets.py` | To-scale PDFs in `tools/tag_sheets/`: six AprilTags for the board's border, the gripper tag in three sizes (on the SO-101 it goes on the back of the wrist camera's mount), and the jaw marks. Each sheet shows where its tags go. |
| 8 | `python tools/check_tags.py` | Whether the cameras read the tags: which board tags are found, where the overhead camera sits relative to the board (useful while aiming it), the gripper tag, the jaw marks. No arm motion. |
| 9 | `python tools/tag_tour.py`, then `--calibrate` | **The arm moves.** `OVERHEAD_HOMOGRAPHY`, `OVERHEAD_CAM_MODEL` (pose), `GRIPPER_TAG_POSE`, `GRIPPER_TAG_READS_AS_M` and the chess board cache, with no touches. They replace the overhead values step 5 wrote. The arm hovers over ~26 squares while the overhead camera reads the board and gripper tags, and stops if the gripper tag goes out of sight. Then `check_overhead_correct.py` checks that the pick's camera correction puts the gripper over each square (~1 mm). |
| 9b | `python tools/calibrate_grasp.py --squares e2,e4,e6` | Where the jaws really close at each lean, against the jaw model: `grasp_correction.json` in the rig folder, which every connect applies. Stand a pawn on each square. At every lean it reaches there, the arm closes on the pawn at offsets along the arm, out from the middle to the first miss either side. It finds the pawn again after every miss (a miss pushes it along the arm). `--show` prints the table; `--dry` measures without saving. Once tag_tour has run, it measures through the overhead camera's correction. |

Gripper-camera servoing corrects the hover position during a pick. The lens
must be in focus first, and the sharp zone is narrow: `check_lighting.py` reads
sharpness live at the servo pose.

| Step | Command | Produces |
|---|---|---|
| 10 | `python tools/so101/calibrate_gripper_lens.py` | **The arm moves.** `GRIPPER_CAM_K`, `GRIPPER_CAM_DIST`: the lens model the piece detector needs. Fully automatic, ~3 min of arm motion. |
| 11 | `python tools/so101/calibrate_gripper_servo.py` | **The arm moves.** `GRIPPER_CAM_GRASP_PX`, `GRIPPER_CAM_JACOBIAN`, `GRIPPER_CAM_REF_HEIGHT_M`, `GRIPPER_CAM_REF_TILT_DEG` (the tilt the pixel was taken at; the servo corrects for any other tilt a pick holds). Asks you to stand one piece between the open jaws, then confirm each detection. Run steps 4 and 5 first: the target pixel depends on the jaw opening and on what "straight down" really is. `--probe X,Y` moves it to a clear spot; `--grasp-px U,V` reuses a measured pixel and only re-measures the Jacobian. |
| 12 | `python tools/so101/calibrate_gripper_pose.py` | **The arm moves.** `GRIPPER_CAM_POSE`, `GRIPPER_CAM_JOINT_DEG`, `GRIPPER_CAM_PLANE_Z_M`, `GRIPPER_CAM_SCAN_SHIFT_M`, `GRIPPER_CAM_SCAN_SHIFT_AT`: where the camera sits on the gripper, so every gripper frame comes with the camera's pose from the joint angles, and where the overhead scan's positions fall in its frame. A pick then looks for its piece among the others the scan found, finds its foot on any surface, and shifts the overhead camera's aim by where it stands against the scan (`AIM_AT_MEASURED_FOOT`), instead of servoing a pixel. Re-run after the overhead camera is re-posed (a start-up warning says so). Board flat and clear of pieces; fully automatic, ~70 poses, 3–4 min. `--wide` instead visits a pick's hover heights over the whole board: ~90 poses, 5 min. It fits a correction to each joint reading with the mount (`GRIPPER_CAM_JOINT_DEG`), since on this arm the mount alone is centimetres out. It saves its frames; `--fit <folder> [<folder> ...]` re-fits without the arm, several captures together. With no board (a deployment on the bare table, `tidy`): `--target tags`, with the table target sheet (`tools/tag_sheets/table_target_a4.pdf`, from `make_tag_sheets.py`) printed at 100% and laid flat, uncut, in the overhead camera's view where the arm reaches. `--deployment tidy` takes its measured surface. It starts from the last good calibration, so it needs one. |

To scan surfaces other than the board (optional):

| Step | Command | Produces |
|---|---|---|
| 13 | `python tools/capture_surface_reference.py` | `rig/<backend>/surface_reference.png`: the empty work surface, so a scan with no board can tell objects from a patterned surface. A plain table needs none. |

Each prints values to paste into
[`mrcf/backends/so101/robot_config.py`](mrcf/backends/so101/robot_config.py),
the arm's configuration, shared by every deployment. Task settings (piece
vocabulary, grasp heights, prompts) go in
[`deployments/chess/config.py`](deployments/chess/config.py) instead.

**`find_poses.py`** disables torque so you can move the arm by hand. Pose the
arm, type a label, press Enter. Four labels map to config constants: `home`,
`table` (fingertips touching the table), `observe`, `calib` (parked clear of
the overhead camera). Any other label (`left`, `right`, …) becomes a
`LOOK_AROUND_POSES` entry.

**`calibrate_arm_offsets.py`** is the board calibration to use. A board is
flat, square and a known size, so nine touches on its corners show where the
arm's own readings are wrong. On this rig, a few degrees at the shoulder put
the far edge 23 mm below the near one and made the 40 mm squares read 31-39 mm.
It fits the joint corrections with the board held level, then places all 49
corners for the overhead camera. It prints the touch residual before and after
(12.3 mm -> 3.4 mm here) and warns about a touch that fits badly, usually the
wrong corner. `--fit <touches.json>` re-fits from saved touches.

**`tools/calibrate_overhead_camera.py`** is the quick alternative. It is fine
only while the arm's readings are already consistent (always so on an
industrial arm, where it is that arm's step). It finds the chessboard corners
automatically, then asks you to touch three of them. **Close the jaws and touch
with the point where the fingertips meet**, not with one finger. `get_ee_pos()`
reports that midpoint, so touching with a single finger bakes an ~8.5 mm offset
into the homography, at whatever wrist angle you happened to be holding.

---

## First-time setup on another arm

This assumes the arm's package exists (`mrcf/backends/<arm>/`, copied from
`mrcf/backends/_template`; see
[DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#adding-an-arm)) and the rig is built
([HARDWARE.md](HARDWARE.md)), with every camera and sensor listed in the arm's
`CAMERAS` and `SENSORS`. Set `MRCF_ARM=<arm>` first, so `run.py` and every
command under `tools/` drive that arm. Each step's values go into
`mrcf/backends/<arm>/robot_config.py`.

Start with `MOTION_GUARD = False` in the arm's config. The guard watches
through the overhead camera's calibration, so until step 4 makes one, an arm
with the guard on refuses to connect. Then, before anything moves, set the
controller's own speed limit low (a UR's pendant slider) and keep a hand near
its stop:

| Step | Command | Produces |
|---|---|---|
| 1 | `pytest` on a test of your own like `tests/test_template.py`'s contract test, the arm's SDK faked | The backend contract, offline: guard, hold, pause, arrival, threads, `connect()`/`stop()` and the sentry. Then the same against the vendor's simulator, if it has one (URSim for a UR). |
| 2 | the arm's own pendant or SDK | `HOME_XYZ` and `PARK_XYZ` (park clear of the overhead camera's view of the surface). The jaw geometry: `JAW_*` from the gripper's drawing, and `JAW_AXIS`, which way the jaws open (the env turns them to it and the policies plan for it). `WORKSPACE_BOUNDS` and, for an arm that can reach past the table, `GRIPPER_BOX`, with the controller's own safety planes set to match. |
| 2b | `python tools/check_motion.py` | **The arm moves.** Its first moves, asking before each: 3 cm lines, the jaws turned 20°, a stop and a pause by the interlock (how far it runs on after each), the gripper (readied first if the arm's package has `prepare_gripper`), freedrive if it has it, home and park. The most force read in each step, for `CONTACT_FORCE_LIMITS`. Results in `rig/<arm>/check_motion/`. No camera needed: you are the guard. |
| 2c | `python tools/check_jaw_direction.py` | **The arm moves** (home, the wrist turns, the jaws open and close). Which way the jaws really open (`JAW_SLIDE_AXIS`), checked by eye and recorded in `rig/<arm>/checked_facts.json`. Until then, every pick on the real arm is refused (`mrcf/arm_facts.py`): a wrong axis grips anything not square along its length. |
| 3 | `python tools/check_camera.py` | Every camera in the arm's `CAMERAS` opens on the right source and is in focus. No motion. |
| 4 | `python tools/calibrate_overhead_camera.py` | `OVERHEAD_HOMOGRAPHY`, `OVERHEAD_CAM_MODEL` and `TABLE_Z`, from three touches on the board's corners with the closed jaws. You guide the arm by hand: the env's `disable_torque()` if it has one, else the arm's own freedrive. Paste them in (an arm with an exact frame has no use for the `ARM_FRAME_SCALE` it also prints), then set `MOTION_GUARD = True` again. From here on, every program that connects is watched. Any other fixed camera: `--camera NAME`, with its calibration in its `CAMERAS` entry. |
| 5 | `python tools/check_motion_guard.py`, then `--move` | `--move`: **the arm moves.** The sentry stops the arm for a hand and not for the arm's own cable or shadow; tune `MOTION_GUARD_ZONE`. With more than one camera in `MOTION_GUARD_CAMERAS`, `--camera NAME` checks each. |
| 6 | `python tools/check_targeting.py` | Whether the arm lands where the camera says a square is, and if not, which calibration to redo. |
| 7 | `python tools/make_tag_sheets.py`, `python tools/check_tags.py` | Optional, for the closed loop: the board's tags and a gripper tag (`GRIPPER_TAG_ID`), read by both cameras. |
| 8 | `python tools/tag_tour.py`, then `--calibrate`; then `python tools/check_overhead_correct.py` | **The arm moves.** The overhead camera and the gripper tag's pose, from ~26 hovers with no touches. Then the pick's camera correction is checked square by square. `--camera NAME` tours through another fixed camera and prints its pose for its `CAMERAS` entry. |
| 9 | `python tools/capture_surface_reference.py` | Optional: the empty surface, for scanning without the board. |

Then give the arm a job of its own, started from the nearest deployment:
`python tools/new_deployment.py NAME --from SOURCE --arm <arm>`. Another arm's
deployment never runs on this one. Run it slowly for the first session:
`python run.py --deployment NAME`.

### On the UR5

The backend is `mrcf/backends/ur5/` (a CB3 on PolyScope 3.15, over Ethernet
with ur_rtde). Its own tools come first, before step 2 above.
`MRCF_UR5_HOST=offline` runs any of them on a UR5 in memory.

```bash
export MRCF_ARM=ur5
pip install -e ".[ur5,dev]"        # on macOS, ur_rtde builds from source: mrcf/backends/ur5/NOTES.md
```

| Step | Command | Produces |
|---|---|---|
| 0 | `pytest tests/test_ur5.py` (and `MRCF_URSIM_HOST=127.0.0.1` for URSim, started as the test's docstring says) | The backend contract offline, and through the real ur_rtde against Universal Robots' simulator. No arm. |
| a | the pendant: Setup Robot > Setup Network; this computer's network settings | The arm cabled straight to this computer. On the pendant, a static address (read it there if the arm already has one), mask `255.255.255.0`. Set this computer's Ethernet adapter by hand to another address on the same range, mask `255.255.255.0`, no router. The arm's address goes in `rig/ur5/host`, one line, not in git (`MRCF_UR5_HOST` overrides it). Use a range that no Wi-Fi this computer joins uses: with the same range on both, macOS sends the arm's traffic out over Wi-Fi. On macOS, the app the tools run from (Terminal, VS Code) needs Local Network on in System Settings > Privacy & Security, then a relaunch. Without it, every send to the arm fails at once with "No route to host". The pendant's "Not connected to network!" with no router is harmless: PolyScope cannot reach its DNS server. |
| b2 | `MRCF_ARM=ur5 python tools/check_jaw_direction.py` | **The arm moves** (home, the wrist turns, the jaws open and close). Which way the jaws really open, checked by eye and recorded (`rig/<arm>/checked_facts.json`). Picks on the real arm are refused until then (`mrcf/arm_facts.py`). Any arm. |
| b | `python tools/ur5/check_connection.py` | **No motion.** Checks the network, the controller's ports, PolyScope's version, the robot and safety modes, the speed slider, this repository's kinematics against the controller's, and the Robotiq's state. Says what to fix at each. |
| c | `python tools/ur5/check_motion.py` (step 2b's `tools/check_motion.py` on the UR5) | **The arm moves.** The first moves, asking before each: 3 cm lines, the jaws turned 20°, a stop and a pause by the interlock (how far it runs on), the gripper (it offers to activate the Robotiq first), 15 s of freedrive, home and park. Results in `rig/ur5/check_motion/`. Set the slider to 10-25% and keep a hand on the emergency stop. |
| d | jog to each pose on the pendant, then `python tools/ur5/check_connection.py --pose home` (and `--pose park`) | `HOME_JOINTS_DEG`, `PARK_JOINTS_DEG`. Until they are set, `go_home()` and `go_park()` refuse rather than guess. |
| e | the steps above (another arm's), from step 3 | The overhead camera, mounted above the far side of the table looking down at an angle (so it measures heights), calibrated on a checkerboard laid on the table (`mrcf/tabletop/settings.py` says which; a deployment may set its own). The tools take the tabletop deployment's values by themselves (the UR5's `DEPLOYMENT`). |
| f | `python run.py --deployment tabletop` | Picks up, stacks and puts down the letter blocks by plain English: "put the Z block on the P block", "stack the blocks", "put the red bottle next to the mug", "this spot is the tray", "put it back". It does not put things into containers. On the UR5 that is the toolbench deployment's (`PLACE_IN` in its profile, row g). |
| g | `python run.py --deployment toolbench` | Puts hand tools into a box. Its one task, `put the tools away` (or `go`), puts the hammer, every screwdriver, the tape measure and the tape into the box. It is planned in advance, with the planner off (set `PLANNER = True` in its config for plain English). A long tool lying flat is gripped across it, by the handle next to its heavy end or else at its balance point. The box is found by name, and its opening by its darker inside. There is no wrist camera to measure the box's walls, so it uses a height kept for the box (`rig/ur5/toolbench_containers.json`), else the tallest a container may be (`CONTAINER_TALLEST_M`). It lowers a tool inside or drops it from above the walls, its long edge along the box's. On the arm, the hammer and one screwdriver go into the box; the tape measure and the tape are untried (`deployments/toolbench/NOTES.md`). Its rules: `deployments/toolbench/profile.py`. |

**Not yet for another arm:** the wrist camera's calibrations
(`calibrate_gripper_lens.py`, `_pose.py`, `_servo.py`) move the SO-101 in its
own joint space and are still its tools. Until they are ported, leave
`GRIPPER_CAM_K` None. The template's `make_policies()` then keeps the wrist
camera's measuring and servoing off, and picks line up with the overhead camera
alone.

---

## When something's wrong

Work down this list. Each command isolates a smaller part than the last, with
no VLM and no planner involved. Those under `tools/` work on any arm. Those
under `tools/so101/` are the SO-101's; another arm needs its own way to answer
the same question.

| Command | Isolates |
|---|---|
| `python tools/check_camera.py` | Are the cameras open, at the right index, and in focus? Saves frames. |
| `python tools/check_exposure.py --show --area 0.11,0.43,-0.15,0.17` | A live exposure meter for a camera. Moves nothing. Shows how much of the surface is at the sensor's maximum and how bright its lightest tenth reads, with clipped pixels red in the preview. `--camera wrist` (or any name in `CAMERAS`) picks another camera. Auto exposure clips the light squares (35% of them on one afternoon), and a clipped cream piece is the same white as the maple under it. Use it to judge a lamp, a curtain, or a new camera or placement. This rig's two cameras ignore manual exposure and every other image control (checked over USB), so a camera-control app cannot fix it. `--area` is the part of the surface to judge (arm frame, m); those numbers are this board's squares. |
| `python tools/so101/check_lighting.py` | A live brightness and sharpness meter at the visual servoing pose, with a live "would the detector find the piece?" readout. Use it to tune a lamp without re-running a calibration to find out. |
| `python tools/so101/check_motion.py` | Does the arm reach `go_home()` / `go_observe()`? Pure joint space, no IK. |
| `python tools/so101/check_gripper.py` | Do the jaws open and close through their full range? |
| `python tools/so101/check_reach.py` | Is the table reachable at all with this calibration? Pure kinematics, no gravity. For per-square, per-piece answers use `reach_map.py` (below). |
| `python tools/so101/check_hold.py` | Can the arm actually HOLD a reach near the board, or does it stall under its own weight? The torque check `check_reach.py` can't do: it has no motor model. |
| `python tools/check_perception.py` | Do the camera and VLM find objects? Writes an annotated overlay. |
| `python tools/check_tags.py` | Can the cameras read the tags? Every camera in `CAMERAS` (`--camera NAME` for one). For a fixed one: board tags found and how reliably, the gripper tag, and where the camera sits relative to the board. For one on the gripper: the jaw marks. No arm motion; saves annotated frames. If a pick's camera correction gives up with "gripper tag not in view", start here. Usually a tag has lifted or slid. |
| `python tools/check_overhead_correct.py` | **The arm moves.** Does the overhead camera put the gripper over a square? Hovers over ten squares, lets the pick's correction move it, then measures again independently. ~1 mm here; a few mm means the gripper tag moved (re-run `tag_tour.py`). |
| `python tools/check_targeting.py` | The arm reaches its target but not the piece: is it a constant offset, a wrong scale, or a skew? Presents the closed fingertips at squares across the board, fits where they land and tells you which calibration to redo. `--squares a1,h1,a6,h6,d3,e4` chooses your own. The defaults stop at rank 6 because rank 8 is past the SO-101's practical reach. |
| `python tools/check_motion_guard.py` | Does the motion guard stop the arm for a hand, and only for a hand? With the arm still, wave a hand over the board: STOP, then CLEAR 2 s after it has gone. With nothing moving, it prints the noise floor. `--move` hovers the arm over the board three times with the guard on. Stand clear: any STOP is then a false one (the arm's cable or shadow), and its picture in `reports/motion_guard/` shows where. Tune with `--zone`, `--margin`, `--min-area`, `--diff`. Put a zone that works in `robot_config.MOTION_GUARD_ZONE` (the guard every connection starts), and the rest in `MotionGuard`'s defaults. |
| `python tools/so101/unlock.py` | **The arm goes slack (hold it), then moves.** Frees a torque-locked arm so you can move it by hand, then drives it home. |
| `python tools/so101/rest.py` | **The arm moves**, then goes slack. Parks the arm folded and low, then cuts torque, so it is safe to leave for hours without the servos heating. Of its rest poses (the base turned right, left, ahead, half-right, half-left), it uses the first whose path keeps the jaws clear of what the deployment's memory has standing on the table, and says why the others will not do. The base turns at home's height before the arm folds. `--pose NAME` asks for one (still checked; `--force` skips the check). `--check` prints the check from home without connecting. `--wake` checks the way home, re-enables torque, unfolds and homes. Use this between sessions: a servo held against gravity all afternoon is what trips the gripper's overload. |
| `python tools/find_cameras.py` | Which cameras this computer has, by name, and which may be the rig's, opening none. The computer's own (the laptop's, an iPhone's, Desk View) are never opened. `--picture N` takes one picture from index N (faces covered), only while every rig camera the arm names is plugged in. |
| `python tools/fetch_face_model.py` | Once per computer: the face model that covers faces in every kept picture (OpenCV's YuNet, 232 KB, checked by its SHA-256). `--check` says whether it is there. |
| `python tools/clean_pictures.py` | Lists the pictures already kept (runs/, rig/, reports/, deployments' logs) with a face in them. `--cover` covers the faces in place; `--delete` deletes the pictures. Look at the list first: a few are a piece's head close up. |
| `python -m mrcf.flight_recorder --timings` | Where the last run's time went: every command, plan, step, action and model call, counted, failed and timed, largest first (or pass a run's folder). `--list` lists the incidents; with no argument, it tells the latest incident's story. |
| `python tools/new_arm.py NAME` | Starts a new arm: `mrcf/backends/NAME/` from `_template`, an arm in memory until its SDK is in. Prints the order: the arm, its sensors, the job. (`n` at run.py's arm question does the same.) |
| `python tools/new_deployment.py NAME --from SOURCE [--arm ARM]` | Starts a new deployment (`deployments/NAME/`) from the nearest one's code, on the source's arm or `--arm`'s (the same task on another arm is another deployment). The source's rules go into the new `profile.py` as commented-out candidates, each with why the source has it and what the arm does without it, to take over one by one where the reason holds (DEPLOYMENT_SPEC.md, "Adding a task"). `--list SOURCE` prints a deployment's rules. Writes files only. |
| `python tools/so101/check_bus.py` | Which servos are answering? Tells a loose cable (silence) from a tripped overload latch (error status); lerobot's "Missing motor IDs" looks the same for both. |
| `python tools/so101/teleop_keyboard.py` | Drive the arm from the keyboard, with the model out of the loop. Answers "is this grasp possible at all?". `k` marks a position, giving a measured coordinate for a piece to compare with what the scan claimed. Takes `--square d4`, `--xy X Y`, `--step 0.002`. |

**Arm stuck holding a pose and won't budge?** Use `unlock.py`.

**Gripper reaching the wrong place?** Check `board_overlay.png` first: it shows
which square the scan thinks each piece is on. If the labels are right but the
arm still misses, it's kinematics; go to `visualise_ik.py`.

---

## What every scan measures, and where it is logged

With the optional segmenter installed (`pip install -e ".[measure]"`;
SAM2-tiny, ~150 MB, downloads on first use), every board scan also outlines
each piece in the overhead frame and measures, in real millimetres, where its
foot actually stands (not only which square), how wide it is, and its colour.
Without it, scans work exactly as before.

To check the measurements without the planner, and with `--heights` to teach
the arm every piece type's height before it picks anything:

| Command | What it does |
|---|---|
| `python deployments/chess/tests/scan_board.py` | **The arm moves.** Parks the arm, scans, measures every piece's foot and colour, and logs them. No vision model, so pieces print as `white_piece_e4`. |
| `... scan_board.py --classify` | Also names types and colours with the vision model (Ollama must be running), as the controller does. |
| `... scan_board.py --heights` | **The arm moves.** Then hovers over each piece and measures its height and width profile with the gripper camera, which the overhead camera cannot. It is logged to `logs/shapes.jsonl` and kept with the piece in the object memory, which follows it through later scans and moves; its picks start from it. Returns home between pieces. |
| `... scan_board.py --heights --squares e2,d7` | Only those pieces. |

Every pick also measures the piece from its hover and takes the grasp height,
grip and jaw opening from its shape. The pads close where its sides are
straight, nearest `GRASP_HEIGHT_FRACTION` of its height. Straight means the
width changes by at most `GRASP_FLAT_MM` over the pads' height, so not on a
flared base, a collar or a taper. Until a piece has been measured, the pick
only needs a hover that clears the tallest piece (`TALLEST_PIECE_M`). One
measurement can be wrong, so a shape from only one look is looked at again on
the next pick (from a hover that still clears the tallest piece) until two
agree. The memory keeps every look and goes by the height most agree on.

Every scan ends with a `[Light]` line, so you know before a pick: ok, or a
WARNING naming the light squares washed out at the camera's maximum (where a
cream piece can vanish or be mis-measured) and the pieces standing among them,
or a board too dark. The camera's auto exposure cannot be turned off; the fix
is the room's light (a curtain, a lamp). Each pick also warns when the wrist
camera's view is washed out or dark.

Colour is read against the board's own squares in the same frame. Each piece
is placed on the colour line (Lab a and b) between the light and the dark
squares: 0 at the light squares, 1 at the dark. Below 0.5 it is white, above
0.65 it is black, and in between it gets no colour. So a change of light or
exposure moves the squares and the pieces together.

| File | What's in it |
|---|---|
| `deployments/chess/board_state.json` | The latest scan: every piece, its square, measured foot position and diameter, colour, type, and whether only the camera's outline reported it or the vision model's label too. For you to look at; the skills work from the same scan in memory. |
| `deployments/chess/logs/scans.jsonl` | Every scan, one line each. |
| `deployments/chess/logs/frames/` | The overhead frame each scan measured from. |
| `deployments/chess/logs/picks.jsonl` | Every pick and place (target, measured geometry, jaw opening, outcome), and refusals with the reason. A pick's `notes` are what it reported on the way: why each attempt was retried, the servo's result, its timing. |
| `deployments/chess/logs/shapes.jsonl` | Every gripper-camera measurement: height, width at each height, foot. |
| `deployments/chess/object_memory.json` | Which piece is which across scans, moves and restarts: colour, type and who said so, measured shape, square, history. The planner sees its recent events. `forget` at the prompt clears it. |
| `deployments/chess/board_overlay.png` | Labelled squares plus each measured piece's outline and foot. |

Before opening the jaws, pick and place check the scanned neighbours. A piece
whose neighbours leave no room for a finger is refused and logged rather than
knocked over. Height is measured by the gripper camera from the hover. The
overhead camera looks down too steeply to read a height.

---

## Map where each piece can be picked

`deployments/chess/tests/reach_map.py` runs a full pick and place on every
square for each piece type, using the approach the arm's pick would choose
there (**T** top_down, **N** near_base, **R** reach_out; see the README's
`mrcf/policies/` section), with its waypoints, tolerances and held orientation.
The answer depends on the piece: a king's hover has to clear its crown, and the
higher the hover, the less of the board the arm reaches straight down.

| Command | What you get |
|---|---|
| `python deployments/chess/tests/reach_map.py` | **Predicted**, offline, every piece. ~10 s per piece. Optimistic (no sag), so a fail is certain and a pass only probable. |
| `... reach_map.py --piece king` | One piece. |
| `... reach_map.py --piece unmeasured` | Any piece before it has been measured, planned as the tallest piece, as the pick does. Shows where an object nobody has named can be picked at all. |
| `... reach_map.py --live --piece pawn` | **The arm moves.** **Measured** on the real arm over an **empty board**, ~10 s a square. Also counts the board corners the gripper camera sees at each hover; near the edge it may see too little board to servo on. |
| `... reach_map.py --live --piece queen --squares a1,h1,e8` | Only those squares. |
| `... reach_map.py --report` | Rebuild `reports/reach_map.html` from everything recorded. |

Every run writes `reports/reach_<predicted|live>_<piece>.json` and refreshes
`reports/reach_map.html`. A live run saves after every square, so Ctrl-C keeps
what it measured.

Verdicts per square, worst first:

- **fail**: no approach reaches it, or a phase does not converge.
- **clash** (radial jaws only): the tilt leans the piece into a jaw, even at
  the grasp height reach_out raises it to.
- **blind** (live): too few board corners where the wrist servo looks from.
- **tilted**: the wrist servo aligns at a lean more than 10° from the one it
  was calibrated at.
- **marginal**: a joint within 2° of its limit.
- **ok**.

The grid shows each square's approach and held tilt (`R45` = reach_out at 45°),
with a mark for anything short of ok.

Piece sizes come from `PIECE_SIZES_M` in
[`deployments/chess/config.py`](deployments/chess/config.py). The grasp the arm
plans for a piece of that size (hover, grasp height, jaw opening) comes from
its policy ([`mrcf/policies/parallel_jaw/`](mrcf/policies/parallel_jaw/)),
tuned in
[`mrcf/backends/so101/robot_config.py`](mrcf/backends/so101/robot_config.py),
which also holds `JAW_AXIS`, the way the jaws open.

---

## See what the wrist camera sees from a pick's hover

**The arm moves.** `python tools/look_tour.py --objects e2,d2,b1` moves over each object as a
pick's first attempt does (the approach a pick would choose, the overhead
camera's correction), then down to the wrist servo's height. **Nothing is
gripped.** At each it keeps wrist-camera frames, the joint readings and an
overhead frame. The deployment's scan runs before and after, so a nudged piece
shows. A piece a pick would refuse (no room for the jaws) gets a look from the
hover only. With no `--objects`, it visits `--count` objects (10) spread over
the surface. Saved to `rig/<arm>/look_tours/<stamp>/`, for working offline on
what the wrist camera can tell apart.

## Inspect the IK / preview a plan

`visualise_ik.py` has the most flags of any script. It runs **entirely
offline**: no serial port, no cameras, nothing commanded. It drives the real
`approach_orientation` and `move_ee_to` against a simulated servo.

| Command | What you get |
|---|---|
| `python tools/so101/visualise_ik.py` | Joint limits and reachability maps (the default view). |
| `... visualise_ik.py --limits` | Per joint: URDF limit against calibrated limit, and which one actually binds. |
| `... visualise_ik.py --workspace` | Reachability across the table at each phase height, with the smallest tilt that works. |
| `... visualise_ik.py --plan --point 0.261 0.029` | Preview a pick at those coordinates, waypoint by waypoint. |
| `... visualise_ik.py --plan --square e5` | The same, using a board square. A scan must have run first. |
| `... visualise_ik.py --plan --square e5 --to d4` | Preview the place as well. |
| `... visualise_ik.py --all --html` | Everything, plus an HTML report with maps and joint-trajectory plots. |

Extra flags: `--step 0.03` sets the workspace grid resolution in metres
(smaller is finer and slower; the full run takes ~13 s at 0.03). `--html PATH`
writes somewhere other than `rig/so101/ik_report.html`. `--deployment NAME`
picks which task's grasp heights to preview when there is more than one.

**Reading the plan view:** the column that matters is **tightest joint
margin**. A phase can converge and still be one degree from a joint stop, which
is how an unreachable hover gets approved and then quietly stalls short. Rows
under ~2° are marked `<-`. The residual column is optimistic: the simulated
servo is ideal, with no backlash, friction or sag. Trust the margin, not the
residual.

---

## Run a learned policy

Policies are trained elsewhere (ACT and the rest with `lerobot-train`, on a GPU
machine); this repository only runs them. One for the SO-101 must have been
trained on its stream: the follower's six motors, the wrist camera and the goal
it was sent, 30 times a second (`mrcf/backends/so101/demos.py`).
[POLICIES.md](POLICIES.md#a-learned-policy-step-by-step) has every step, from
recording to reading what went wrong.

| Command | What it does |
|---|---|
| `python tools/show_stream.py` | What a model for the running arm must be trained on: its stream's rate, names, shapes, cameras and fingerprint. Nothing connects. `--json` for a training script. |
| `python tools/show_stream.py --package <folder or hf:owner/name@rev>` | Whether a policy package fits the arm's stream (names, shapes, rate, robot type), and whether this machine has its library. |
| `python tools/show_stream.py --checkpoint <.../pretrained_model>` | The same for a bare LeRobot checkpoint, by its shapes alone (loading it needs LeRobot). |

Give it an entry in `deployments/chess/config.py`'s `LEARNED_SKILLS`: a policy
package by reference (a folder, `hf:owner/name@rev`), or a LeRobot checkpoint's
`pretrained_model` folder with its name, description, task, start height and
time limit. The planner can then ask for it ("stand the pawn on e4 up"), or you
can call it at the prompt: `/upright white_pawn_e4`. The arm hovers over the
piece where the demonstrations began, runs the model on the wrist camera and
the joints for up to the time limit, and scans again. A model trained on
another camera size or another arm is refused before anything moves. One that
keeps asking for goals the arm holds back (into the board, out of the
workspace) is stopped after a second. The first run of a session loads the
model (a few seconds). A LeRobot model needs LeRobot installed
(`pip install lerobot`; the SO-101's `.[so101]` brings it). Without it,
start-up leaves the skill out and gives that command. What each refusal means:
[POLICIES.md](POLICIES.md#when-it-is-refused-or-fails).

---

## Models on another machine, or a hosted API

Every model call goes to one model server (`mrcf/model_server.py`). Ollama on
this machine is the default, needing nothing below.

**Ollama on another machine** (a GPU box on the network):

```bash
export MRCF_MODEL_URL=http://gpu-box:11434      # Ollama's port; /api/chat is added
```

**Any OpenAI-compatible API**, hosted or your own (vLLM, LM Studio, llama.cpp's
`llama-server`, Ollama's own `/v1`):

```bash
export MRCF_MODEL_SERVER=openai
export MRCF_MODEL_URL=https://api.openai.com/v1  # the base URL, up to /v1; /chat/completions is added
export MRCF_MODEL_API_KEY=...                    # from the provider; read from here only
export ROBOTICS_VLM_MODEL=<a vision model>       # the server's names for its models: one that
export ROBOTICS_PLANNING_MODEL=<a model>         # takes images for the first and the label model
export ROBOTICS_LABEL_MODEL=<a vision model>
python run.py
```

At start-up the controller prints where its models are served, and that a
server off this machine is sent the camera's pictures and the scene.

- A thinking model spends the answer's token cap on thinking, then answers
  nothing; the log says so. Turn its thinking off or down with
  `MRCF_MODEL_EXTRA_BODY`, in the provider's terms. gemma4 on Ollama's `/v1`
  answered with `'{"reasoning_effort": "none"}'` and ignored
  `{"think": false}`. Ollama's own API already has it off.
- A JSON schema holds the planner's answer to the scene's actions and names,
  and a label to its lists. A server that enforces schemas (Ollama, vLLM, LM
  Studio, llama.cpp) keeps to it; a hosted API may take it as a hint. A server
  that refuses a schema is asked for plain JSON instead, then plain text.
  Either way the answer is checked against the lists afterwards.
- Rate limits and overload (429, 502-504, 529) are waited out (for the
  server's `Retry-After`, if it gives one) and retried up to twice. A refused
  key (401, 403) is not retried.
- For another kind of API (a provider's own, a cloud SDK), write a server of
  your own ([README.md, "Extending it"](README.md#extending-it)).

## Environment variables

All optional. Set them your shell's way (`export NAME=value`; in Windows
PowerShell, `$env:NAME = "value"`). `MRCF_ARM` is read by `backends.arm()`.
The SO-101's two have their defaults in
[`mrcf/backends/so101/robot_config.py`](mrcf/backends/so101/robot_config.py),
the UR5's in [`mrcf/backends/ur5/__init__.py`](mrcf/backends/ur5/__init__.py),
and the models' in [`mrcf/model_server.py`](mrcf/model_server.py).

| Variable | Default | Purpose |
|---|---|---|
| `MRCF_ARM` | `so101` | Which arm runs: the name of its folder under `mrcf/backends/`. |
| `MRCF_KEEP_PICTURES` | on | `0` keeps none of the pictures the flight recorder keeps only for reading later (`pictures/`). An incident's frames are kept either way, with faces covered and calibrated cameras limited to the work area. |
| `MRCF_HOME` | the repository | The folder that holds `rig/<backend>/` (each rig's own files: its calibrations, surface photo, URDF, what its tools save) and `runs/` (the flight recorder). Never in git. |
| `MRCF_NO_IMAGE_OPEN` | unset | Set to stop the calibration tools opening the frames they save in the system viewer. |
| `SO101_FOLLOWER_PORT` | `/dev/tty.usbmodem5B610326071` (this rig's, on macOS) | Serial port. Find it with `lerobot-find-port`: `/dev/ttyACM0` on Linux, `COM3` on Windows. |
| `SO101_FOLLOWER_ID` | `my_follower` | Which lerobot calibration file to load. |
| `MRCF_UR5_HOST` | `rig/ur5/host` | The UR5 controller's address. `offline` runs a UR5 in memory, with no controller. |
| `MRCF_URSIM_HOST` | unset | The address of a URSim to run `tests/test_ur5.py`'s simulator test against. |
| `MRCF_CHESS_ENGINE` | `stockfish` on the PATH | The chess engine for a game: any UCI engine (`deployments/chess/engine.py`). |
| `MRCF_MODEL_SERVER` | `ollama` | Where the models run: `ollama`, or `openai` for any OpenAI-compatible chat API ([Models on another machine, or a hosted API](#models-on-another-machine-or-a-hosted-api)). |
| `MRCF_MODEL_URL` | the server's own | Its address: an Ollama host (`http://gpu-box:11434`), or an OpenAI-compatible base URL (`https://api.openai.com/v1`, the default for `openai`). |
| `MRCF_MODEL_API_KEY` | unset | The server's API key, sent only to that server as `Authorization: Bearer`. Never printed or recorded. |
| `MRCF_MODEL_EXTRA_BODY` | unset | A JSON object added to every request, for the provider's own settings: `'{"reasoning_effort": "none"}'`. |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama's endpoint when `MRCF_MODEL_URL` is not set. |
| `ROBOTICS_VLM_MODEL` | `gemma4:31b` | Vision model, for scene descriptions. On a remote server, its name there. |
| `ROBOTICS_PLANNING_MODEL` | `gemma4:31b` | Planning model. |
| `ROBOTICS_LABEL_MODEL` | `gemma4:31b` | Says what each piece a scan outlines is. |

---

## What to re-run after a change

The easiest thing to forget: these calibrations depend on each other.

| After you… | Re-run |
|---|---|
| `lerobot-calibrate` | Everything in First-time setup from step 3 on, and `calibrate_gripper_servo.py`: the joint zeros and the gripper's motor range both move. |
| Move or re-aim the overhead camera | `tag_tour.py`, then `tag_tour.py --calibrate` (use `check_tags.py` while aiming it: it shows where the camera sits). Then `calibrate_gripper_pose.py` and `capture_surface_reference.py` if you use them. Check the arm still parks out of the camera's view (`find_poses.py`, label `calib`). |
| Re-stick or replace the gripper tag | `tag_tour.py`, then `tag_tour.py --calibrate`: `GRIPPER_TAG_POSE` is where the tag sits on the gripper. |
| Move the board or the table | `calibrate_arm_offsets.py` |
| Change `JOINT_OFFSETS_DEG`, `JAW_AXIS`, `JAW_SLIDE_AXIS`, `PREGRASP_OPEN_M` or `GRIPPER_WIDTH_TABLE` | `calibrate_gripper_servo.py`: its target pixel is where a piece appears between the jaws at that opening, gripper angle and wrist turn. (`JAW_AXIS` and `JAW_SLIDE_AXIS` together set the wrist turn: a change that leaves it where it was, such as a renaming, needs nothing.) |
| Change what the work surface looks like | `capture_surface_reference.py` |
| Move the arm's base | Everything in First-time setup from step 3 on. |
| Update a rig set up before 2026-10-29 (its calibrations beside the code) | `python tools/migrate_rig_state.py` lists what moves to `rig/<backend>/`; `--apply` moves it, checking every file. The arm will not start until it has. |
| Change `HOVER_HEIGHT` / `LIFT_HEIGHT` / `GRASP_HEIGHT_M` (`robot_config.py`) | `visualise_ik.py --plan`, to confirm the new heights still have joint margin. |
| Change a piece in `PIECE_SIZES_M`, or the grasp tuning in `robot_config.py` | `reach_map.py --piece <it>`, then `--live` for that piece. |
| Change `JAW_AXIS` | `reach_map.py` for every piece, then `--live`: it changes the tilt each square needs and what the gripper camera sees. |
| Change `NEAR_BASE_M`, `FAR_REACH_M` or `BASE_KEEP_OUT_M` (which approach where) | `reach_map.py` for every piece: the grid shows which approach each square now gets. |
| Swap the gripper | `_TCP_OFFSET` in [`mrcf/backends/so101/kinematics.py`](mrcf/backends/so101/kinematics.py), then `calibrate_gripper_width.py`, `calibrate_arm_offsets.py` and `calibrate_gripper_servo.py`. |
| Refocus or bump the gripper camera | `calibrate_gripper_lens.py`, then `calibrate_gripper_servo.py` and `calibrate_gripper_pose.py`. |
| Change `_TCP_OFFSET` or the jaw geometry | `calibrate_arm_offsets.py`: the corner touches recorded the old grasp point, so the fit carries the old convention. |
| Change `SERVO_*` tuning in `robot_config.py` | `check_hold.py`: it reports the commanded-vs-held offset per joint, which is what `I_Coefficient` is meant to close. |
| Add a new task on this arm | Nothing: `mrcf/backends/so101/` carries over. Add `deployments/<name>/`. |
| Add a new arm | `mrcf/backends/<arm>/`, copied from `mrcf/backends/_template` ([DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md#adding-an-arm)). Run `check_backend()` on it offline, then follow [First-time setup on another arm](#first-time-setup-on-another-arm). Run with `MRCF_ARM=<arm>`. |
| Move a piece by hand mid-session | Say `observe` at the prompt. Positions, sizes and the neighbour check all come from the last scan. The memory records it as that piece moving, not as a new one. |
| Reset or rearrange the whole board | Say `forget`: the memory starts afresh and rescans. |

---

## Other useful lerobot commands

These come with lerobot, not this repo.

| Command | What it does |
|---|---|
| `lerobot-find-port` | Identifies which serial port the arm is on. |
| `lerobot-find-cameras` | Lists cameras lerobot can see. |
| `lerobot-calibrate` | Recalibrates the arm's joint ranges. |
| `lerobot-find-joint-limits` | Sweeps and reports each joint's travel. |

**Camera indices can change between launches**, on macOS especially, and on any
OS when a USB camera is plugged into another port. If a camera opens on the
wrong device, run `python tools/find_cameras.py`. It lists the cameras by name
without opening any, and says which are the rig's and which are the
computer's own. To see which camera an index opens, use its `--picture N`,
which takes a picture only when every rig camera is plugged in. Do not try
each index with OpenCV: that opens the computer's own cameras too (the
laptop's, or an iPhone's through Continuity Camera). On Linux,
`v4l2-ctl --list-devices` also names them. On macOS, `ffmpeg`'s device listing
numbers cameras differently from OpenCV (see the `CAMERA_DEVICE_NAME` comment
in `mrcf/backends/so101/robot_config.py`).