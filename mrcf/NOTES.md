# MRCF — notes

What the framework does not do yet, and what it has shown on every robot and
task that its code cannot say. What belongs where:
[DEPLOYMENT_SPEC.md](../DEPLOYMENT_SPEC.md#what-you-learn-and-where-it-goes).
How each stage of a command fails, and what to check first:
[SKILL_LIBRARY.md](../SKILL_LIBRARY.md), "Failure modes by stage".

## Limits

What it does not do, by design or not yet. Delete a limit once it is lifted
(git keeps it).

- **One flat work surface, and stacks the arm built.** Objects stand on one
  plane, or on another object where the arm put them (`place_on`, for a task
  whose `Scenario.carries` allows it). A stack built by hand is one tall object
  to a scan; a stack is read only from above (what is under it is where the
  arm left it). Run on the UR5 three blocks high (2026-11-12). A wrist camera
  measures against the table's plane, not the raised top, so a measurement
  of a stacked object there reads it wrong: no rig stacks with one yet.
- **Same-looking objects are told apart by their look, or the vision model.**
  The memory matches a scan by colour, foot and height; a set of one size and
  colour is a guess between them, and only a re-read by the vision model
  (`ObjectMemory.reidentify`, agreed by the camera's colour) moves a name.
- **Faces are covered, not people.** Kept pictures lose what lies outside
  the work area (a calibrated camera's) and every face the face model finds;
  a hand, an arm or a back over the table stay, and a face the model misses
  (turned away, half in the frame) stays too. Without the model fetched
  (`tools/fetch_face_model.py`) no face is looked for; run.py says so.
- **Which index is which camera, on macOS, is not known before opening.**
  ffmpeg's listing does not number them as OpenCV does: an arm refuses its
  cameras only by count (fewer of the rig's plugged in than it names), so a
  rig camera moved to another index still opens as the wrong one —
  `tools/find_cameras.py --picture N` shows which.
- **Positions are as fresh as the last scan:** at start-up, on `observe`, and
  after a failed step. Say `observe` after moving something by hand.
- **A swap the camera cannot see.** An object replaced by hand with a different
  one of the same foot size, in the same place, keeps the old one's identity
  until a wrist camera measures it again.
- **Heights of flat-topped objects read high,** by up to the top's radius ×
  the tangent of the viewing angle, where one look measures them
  (plane_geometry.measure: the hover's look). That is safe for an approach, but
  it is a bias. A height looked for from several sides
  (parallel_jaw/height_looks.py) is fitted instead and has no such bias.
- **Looking for a height needs a located wrist camera and a fixed one's
  outline.** Without the wrist camera an unmeasured object is refused rather
  than guessed (where the arm has height_from_views on); without the scan's
  outline it is fitted to the wrist camera's alone, from the scan's foot, less
  well. Only upright, roughly convex objects fit one of the solids tried (a
  straight side, a rounded top, a dome, a ball, a taper); a scoop or a mug's
  handle fits worst, and its height is the solid's. Neighbours are kept clear
  of by what their outline from above allows (could_be_mm), not looked at; one
  right below the fixed camera is taken as tall as anything could be. The
  survey's looks cost about 1.5-2.5 s each to plan on the SO-101, plus the
  moves. Run live on the SO-101's table since 2026-11-16 (deployments/tidy).
- **A knock is seen by the sentry, not by the arm.** After a hand or a falling
  stack, the arm does not come down where it saw movement, and a plan stops
  to look again (controller.execute_plan). Movement hidden from the overhead
  camera by the arm itself (a stack toppling right under the gripper) is
  not seen until the next look. A piece knocked over is not stood up by
  itself: stand_up is the planner's to choose (chess).
- **Ambiguous names resolve; they do not ask.** "the pawn", with several pawns
  in view, picks one deterministically and prints a warning.
- **Standing a fallen object up from the overhead camera** (`policies/stand_up.py`)
  is refused, after only parking, in two cases. A pale top on a pale surface is
  not outlined whole. An object lying with its top pointing straight away from
  the camera looks like one standing. Once stood up, it is let down where it
  hangs, not on a target's centre. The wrist camera's second look is placed by
  the wrist servo's calibration (one linear step, good to a few mm round the
  jaws). That look has not yet been run on an arm.
- **The motion guard sees movement, not hands.** Anything moving near the arm
  stops it, and a hand held perfectly still for 2 s lets it move again. While
  the arm moves, and for a second after any joint moves, the 5 cm round its
  outline is not watched, because its cable and shadow move there too. A hand
  is seen on its way in. Each camera it watches through (`MOTION_GUARD_CAMERAS`)
  cannot see under the arm or beyond its own view. It is a guard, not a
  certified safety system.
- **Extra cameras and sensors are read, recorded and watched through, but not
  fused.** A scan uses one fixed camera. Another calibrated fixed camera's
  view is kept at each scan (`look_frame["views"]`) and the tabletop's check
  after a place compares with it (`placed_check.look_volume`): by change
  alone, not colour, and only that check. A depth camera's depth and a force
  sensor's wrench (beyond the safety limit) are there for a policy or a
  deployment to use; none uses them yet.
- **Scanning while the arm moves sees the arm.** Watchers can feed the object
  memory from the overhead camera mid-move, but the arm hides what is under it
  and is itself in view. Nothing masks it out of a watcher's frames yet (a
  scan with the arm still in view does: `scene_scan.observe(past_arm=)`). A wrist camera's pose is
  read when its frame is taken, so its frames are placed accurately only while
  the arm is still.
- **Two real arms, built two ways.** The SO-101 is its own `RigEnv`; the UR5
  is `RigEnv` with `RealRobotEnv` (since 2026-11-12), as the template is. What
  the UR5 learned about a controller that plans its own moves (the move loop,
  resending after a pause, waiting for the gripper to settle) is in
  `backends/ur5/env.py`, not yet shared with the template: the template's
  move loop does not stop the arm when it is left mid-move (Ctrl-C, an error),
  checks no force on its moves, and its set_gripper returns before the jaws
  have stopped. The gripper box (`envelope.py`) is checked by the UR5's and
  the template's moves, not the SO-101's. The wrist camera's calibration tools are
  the SO-101's (`tools/so101/`); the by-hand camera calibration needs
  `capabilities.Freedrive` and `link_frames()`, which only the UR5 has.
- **Models off this machine: built, not yet run on a rig.** The `openai`
  model server (any OpenAI-compatible API) has answered a scene description,
  a label held to a schema and a plan through Ollama's own `/v1` on this Mac
  (2026-11-16, qwen2.5vl:7b and gemma4:31b), never a hosted API, and never a
  whole command on an arm. The token caps (`vlm._NUM_PREDICT`) are sized for
  a model that answers at once. A thinking model needs its thinking turned
  off (`MRCF_MODEL_EXTRA_BODY`), or it answers nothing. A hosted API that only
  takes the schema as a hint can name an action or an object the scene does
  not have. Those are dropped after the answer, not prevented.

## Open

Things seen that are not understood or fixed yet. Each entry: **Seen** (the
date, what was seen, the flight recorder's incident), **Cause**,
**Workaround**, **Status**.

### A measured object wider than the jaws open is planned as a pick

- **Seen:** 2026-11-14, offline, recording the arms' plans
  (`tests/baselines/ur5/plans.json`, scene `too_wide`): a measured object
  62 mm across, beside the UR5's 52 mm jaws, is planned as a top-down pick,
  `pick_check` ok, the jaws opening their widest. The template and the SO-101
  plan theirs the same.
- **Cause:** `GraspPlanning.problems` says so ("opening 52 mm does not clear
  the 62 mm width"), but no pick calls it: only the chess reach map does. An
  arm with a wrist camera may measure it narrower from over it; one without
  (the UR5) comes down onto it.
- **Workaround:** none in the pick; the deployment's sizes (tabletop's blocks,
  chess's pieces) are all narrower than the jaws.
- **Status:** open. A pick could refuse before moving when `problems` names
  an opening that does not clear the width, on an arm with no wrist camera to
  measure it again; the baselines would show that change on each arm.

### A measured shape is dropped when a neighbour swells the scan's foot

- **Seen:** 2026-11-11 17:16, chess, the f1 bishop pressed against the e1
  king: measured 62.1 mm at 17:14 (foot 19.7 mm then), it was planned as never
  measured a minute later, as an 82 mm king, and refused.
- **Cause:** `object_memory.Track.shape_current` distrusts a shape once the
  scan's foot differs from the foot at measuring by more than
  `SHAPE_FOOT_TOL_MM` (4 mm). Beside the king its outline read 23.7-23.9 mm:
  4.2 mm over. Two outlines touching swell one another; the object had not
  changed.
- **Workaround:** name the piece (`identify`, `/starting_position`, a game's
  `/new_game`): a known kind is planned at its own height
  (`GraspPlanning.height_known`).
- **Status:** open. A shape whose object's kind is trusted, and whose height
  agrees with that kind's, could be kept through a foot change.
