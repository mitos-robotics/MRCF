# MRCF framework — history

Why the code is as it is: the runs on the rig behind its values and fixes,
with their dates, by file and by name. The comments in the code say what each
value or step is for; the story of how it was found is here. What is still
wrong is in NOTES.md, not here.

## policies/lying_pick.py

### `module` (also: `GripperSpec.lying_pick`, `ParallelJaw.pick`, `plane_geometry.long_extent_mm`, `scene_scan.detections`' length/widths, tabletop `ACROSS_FOR_LONG`)

- **2026-11-20 14:34** — The UR5's first toolbench run picked nothing. The
  hammer's height unknown (its overhead camera too steep to measure heights,
  no wrist camera), the pick planned it for the tallest anything could be,
  150 mm, and closed 116 mm above it; the retry re-aimed at the same height.
  Its outline was a 10 mm piece of it (silver on the light table); the
  screwdriver, "73 mm across" by the round foot an upright thing is measured
  by, was "wider than the gripper opens (52 mm)", and the planner gave up.
  HANDLE_FIRST, in force there, was read only inside rim_pick: a plain pick
  never used it. Now a long thing lying along the table is gripped across —
  by its handle next to the part it joins (the user: "the hammer is one side
  heavy so account for that in the pick"; a hammer's head is that part), else
  at its balance point — closed from no higher than it is wide there (it lies
  on its broad side), lower until held, and lifted 2 cm first with the grip
  checked. The scan keeps each outline's length and narrowest width across,
  and a deployment may describe long things to the planner by them. Off
  except in deployments/toolbench.
- **2026-11-20 14:56** — The next run raised before the arm moved, every
  pick: "UR5Env.approach_orientation_for() got an unexpected keyword argument
  'prefer_tilt_deg'". The call was handle_pick's, which had only run on the
  SO-101; the UR5's search takes no lean (capabilities.OrientationSearch: a
  backend may leave arguments out). Both now go through
  handle_pick.straight_down, which passes only what the arm's search takes;
  and a test runs the lying pick on the UR5's own offline env, not only the
  fake arm (which searches no orientations). The same run's whole outline of
  the hammer (outlined again by name) had "no handle": one 3 mm slice of the
  flared end of its grip read 42 mm, over handle_max_m, and the walk from the
  tip stopped there; find_handle(join_min_len=) now passes a wider stretch
  shorter than spec.lying_join_min_m (15 mm) as part of the handle. Replayed
  on that outline: gripped by its handle 76 mm from the head's outer edge.

## scene_scan.py

### `observe(past_arm=)` (also: `arm_in_view`, `arm_hides`, `ObjectMemory.observe(hidden=)`)

- **2026-11-17** — Every look on the tabletop parked the arm out of the
  overhead camera's view first, and a missed grip parked it, scanned and sent
  it home before the next try: seconds each, many times a pick. The user:
  "why does the camera only work when the arm is out of the way ... It makes
  it slow and inefficient". The arm's body is now masked out of the picture
  from its own shape (motion_guard.arm_mask), and a remembered object under
  it is hidden, not gone — nor matched to a sighting elsewhere as moved.

## containers.py

### `Keeper.remembered` (also: `Keeper.refresh`, `_save_rim`)

- **2026-11-18 16:06** — The tub's walls, measured the day before, were on
  disk, but the tub was only looked for when a place named it: the first
  place of the session parked the arm high, holding the figure, to find it,
  then measured the walls on the way (user: "surely it should have measured
  the height first"). A container kept on disk is now looked for at every
  scan until found, its walls as kept; what it was asked for as is kept too.

## catalogue.py

### `module` (also: `tabletop/skills._match_kinds`, `_recipe`, `_picked_hook`, `Target.grip_at_m`)

- **2026-11-18** — The SO-101's table, four objects all day: each was looked
  at from the side for its height (25-48 s) about ten times, a different
  answer each time, and every miss was a grip 1-2 cm too high. The vision
  model's names changed from scan to scan (one figure: "bottle cap",
  "bottle", "shaker", "ball", "red bottle"), so a height kept by name (first
  tried that morning: ObjectMemory.like, now gone) was never found again. The
  chessboard had told the arm what each piece was; the table needs what the
  arm learns kept instead: kinds matched by how they look from above, how each
  rests, the grip that held it. User: "the deployment is the same objects in
  different places".

## controller.py

### `_run_units` (also: `Deployment.defer_failed`, `LAST_PLAN_SETTLED`)

- **2026-11-18** — "put the four things into the tub" stopped at the first
  failed pick and was planned again from scratch, 45-60 s a time; one re-plan
  went for a figure already in the tub. A plan of pick-and-put units now runs
  each in turn, sets a failed one aside, tries it again after a fresh look
  (found by its track: its name may have changed), and says what was not done.

## wrist_shift.py

### `module` (also: `aligners.FootAligner`'s "wrist_foot" event, `tools/calibrate_wrist_on_objects.py`)

- **2026-11-18** — The SO-101's wrist camera put feet 5 mm off the overhead's
  for two days after the overhead was re-posed, and nothing said so but a
  line at start-up; picks took its outlines for "a part of it". Fitted again
  from the objects on the table (the tool), and every pick's look is now
  logged as a reading of it; a drift is said at a session's start.

## tabletop/skills.py

### `goto_observe_pose` (home before the park pose)

- **2026-11-18 16:08** — A relocate after a miss parked the arm (folded up
  and back), scanned, then sent it home: two long moves for one look. With
  the arm over the spot, it now looks from home first, and parks only if
  home hides it too.

## perception/held_view.py

### `module` (also: `ParallelJaw.holding`, `_note_grip`, `GripperSpec.held_by_camera`)

- **2026-11-17** — The blue scoop held by its 1 mm wall: the jaws stopped at
  1.5 mm under load 120, then the hold goal (10% past contact) squeezed the
  plastic to 0.1 mm over the lift, and the pick said "it left the jaws on the
  way up" with it in them (user: "it had it - why did it let go????"). The
  wrist camera had its bowl in the same place in every frame of the lift, the
  table edge sliding away: replayed, 0.97% of the picture carried against
  0.06-0.22% with nothing in the jaws.

## privacy.py

### `module` (also: `frame_guard`, `cover_people`, `devices.check_sources`, `tools/find_cameras.py`, `tools/clean_pictures.py`)

- **2026-11-17** — User: "make sure there are no pictures of people or
  anything that isn't essential etc. Especially when testing cameras". The
  rigs' 18,871 kept pictures held faces: the SO-101's wrist camera, raised in
  a failed pick, kept the user's (11-17 15:18); find_cameras.py's sweep of
  indexes 0-5 kept one from the Mac's own camera (11-13). OpenCV's Haar face
  models, tried first, were rejected: of 93 "faces" in the wrist camera's
  frames, 91 were the figures and pieces being picked (precision 2%), and
  18% of overhead frames had one (the back rank). YuNet (opencv_zoo, 232 KB)
  found the real faces at 0.82-0.92 and mistook 11 of 7,020 wrist frames and
  2 of 2,000 overhead ones (0.81-0.84). So: the work area first — every pixel
  whose line of sight misses the work volume blacked out, no model needed —
  then YuNet at 0.8. An eye check on Haar's faces dropped the real ones too.
- **2026-11-17** — `devices.check_sources`: ffmpeg's camera listing does not
  number cameras as OpenCV does on the SO-101's Mac (robot_config,
  CAMERA_DEVICE_NAME), so an index cannot be named there; the check counts
  instead: fewer cameras that may be the rig's than indexes to open, nothing
  opened. With the rig unplugged this Mac lists only its own (MacBook Pro
  Camera, Desk View, a screen capture): both arms refuse.

## flight_recorder.py

### `session_info`, `failure`, `timed`, the console (also: run.py `_record`, `controller.repl`, `action_registry.dispatch`)

- **2026-11-17** — User: "Any and all failures need to be logged - and
  possibly every task to then see if we can make it more efficient". The
  recorder started only once the arm was connected; a command's error was an
  event without its traceback; a deployment's start, homing and the last go
  home failed in print only; only pick, place and a few verbs had spans; and
  nothing said which arm and deployment a run was. Now: run.py starts it as
  a deployment is chosen (session.json), the console is kept, every failure is
  an incident with its traceback, an uncaught exception on any thread too,
  and every command, plan, step, action and model call is timed.

## profiles.py

### `module` (also: `make_env(profile=)`, `make_policies(profile=)`, `deployments/<name>/profile.py`, `tools/new_deployment.py`)

- **2026-11-17, evening** — The tabletop's rules for the SO-101 were filed
  inside it as profiles/so101.py, and the UR5, which runs the same task, had
  none of them: a task's rules keyed by arm, nested in the task. The user:
  "why inside the tabletop folder is there an so101 isn't this the kind of
  nesting we want to avoid?", then "each deployment should absolutely combine
  both the arm and the layout etc". A deployment is now one arm, one layout
  and one task: it names its arm (ARM_NAME) and has one profile.py. The
  SO-101's table work became deployments/tidy; the table task's code, which
  both run, moved to mrcf/tabletop (mrcf/tabletop/HISTORY.md). run.py offers
  the chosen arm's deployments only.
- **2026-11-17** — The SO-101's tabletop failed one way after another on a
  tub, a scoop, a plug and a sweet: a 50 mm plug refused as wider than the
  jaws open (45 mm, among chess squares), a scoop's 1 mm wall let go as
  "nothing held" (3.4 mm, the narrowest chess piece), a figure 44 mm from the
  side refused as "another object" for reading 21 from above with nothing
  within 11 cm (a crowded board's rule), a sweet crushed (the hold on wooden
  pieces). Every one was a rule learned on chess, kept in the arm's
  robot_config, which the tabletop built its policies from. The user: "why is
  there so much contamination between chess and this - this should have been
  recognised as a new task and deployment and should create rules for it".
  The arm's config now holds what is true of the arm whatever it does; each
  task's rules for an arm are its profile, laid over the arm's facts when the
  task makes the env and policies. Chess's moved unchanged (its spec the same
  but for the tabletop's 11-16/17 rules, which had reached it untried); the
  tabletop's own moved beside them, with the chess rules whose reasons hold on
  a table copied over and those that do not listed with why not. A new task
  starts from the nearest one's code, its rules offered as candidates only
  (tools/new_deployment.py).

## action_registry.py

### `_check_conditions`

- Previously an unknown condition was silently skipped, so a typo or a
  condition the deployment forgot to register looked exactly like a guard that
  passed. It is now said, instead of quietly allowed.

### `dispatch`

- Declared post-conditions were previously accepted by register_action,
  documented, and never evaluated. dispatch now checks them, advisory only.
- **2026-11-12** — A pick refused because a block stood on it also printed
  "pick reported done but not carrying any object". It had reported nothing
  of the kind. Post-conditions are now checked only after a step that did
  not fail.

## capabilities.py

### `DownRotation`, `MoveRefusals`, `GripperFits`, `Sentry`, `FixedViews`, `FrameAge` (and no asking by name)

- **2026-11-14** — Two arms on two branches, each tuned on its own rig, and
  changes for one kept reaching the other with every test passing. About 70
  places in the shared code asked an env for something by name (`hasattr`,
  `getattr` with a default): `getattr(env, "frame_scale", 1.0)` in the reach,
  `getattr(env, "move_refusals", 0)` in the pick and place, the sentry's
  `motion_seen_since` in the controller and safety. A rename in one arm, or
  in `RigEnv` for one arm's sake, and the other would have aimed 9% off (the
  SO-101's frame scale) or retried a refused move, quietly. What every arm has
  is now called as it is (on `RobotEnv`); what only some have is one of these
  Protocols; `tests/test_architecture.py` refuses a new name asked for; and
  each arm's capabilities are recorded (`tests/baselines/<arm>/env.json`), so
  one lost fails. The six were added for what `RigEnv` gives every rig arm;
  each arm's recorded settings and plans were the same before and after.

### `BackOff` (also: `skill_executor.back_off_after_press`, `BACK_OFF_M`)

- **2026-11-13** — UR5 tabletop: a pick 15-17 mm off its block came down with
  a finger on it and was stopped at 47.5 N, as it should be. But a stop holds
  the arm where it is, so it went on pressing until the user cut the power.
  Now a stop for a contact force backs the arm 20 mm straight up first, unless
  something else holds the interlock (a hand in view, an e-stop), and then
  the stop goes on. Only the UR5 has it so far.

### `Freedrive`

- **2026-11-12** — The tabletop's freedrive skill and the by-hand camera
  calibration went by disable_torque, which an SO-101 has too: with its
  torque off it falls. Being pushed by hand and holding where let go (a UR's
  teach mode) is a capability of its own now.


## controller.py

### `Deployment.on_start`

- **2026-11-16** — Every chess session began with `/new_game` or
  `/resume_game` typed, even when the first look plainly showed which: the
  saved game standing on the board, or the pieces set up again. A deployment
  now reads that first look itself, after a look that worked and before the
  prompt. It moves nothing, and a start that fails is reported without ending
  the session.

### `moved_only_far_off`'s inf (and a plan made again: what it is not told)

- **2026-11-13** — A pause for a hand that came and went (the sentry now
  forgets those) no longer sends the arm off to look again: moved_only_far_off
  says inf for nothing left.
- **2026-11-13** — Tried and taken out the same afternoon: telling a plan
  made again which steps of the last one had run ("That happened … do not undo
  it"). The user took red off the tower while blue was in the jaws, and the
  planner trusted the line over the objects. It put blue on green and stopped,
  red left off. Replaying that call against gemma4: with the line, 2 steps and
  red forgotten (2/2); without it, 4 steps and the tower finished (3/3). The
  run that had prompted it (a block taken off and put back) came from the
  console-only refusal, now told (manipulation._refuse), and the impossible
  order, now not run (stack_sequence_validator).

### `_asks_again` (`_FOR_ROOM`; also: the plan's recovery prompt in `vlm.plan`)

- **2026-11-13** — UR5 tabletop: a place refused for room ("jaws need 52 mm
  open here but red_block_m_obj7 leaves 0 mm", the block's shadow outlined as a
  block) was planned again after going home, since the recovery prompt said
  "home first" for any refusal and home counted as doing something first. It
  was refused the same way, and the user saw an arm going home for no reason.
  Home or a look before a step refused for room no longer counts. The prompt
  now says home is for reach and only moving the object named makes room.

### `_drop_typed_ahead` (also: `repl`)

- **2026-11-13** — The UR5's prompt came back as "You: You:": an Enter
  pressed while the tower was being built had waited in the terminal, and
  the prompt took it as an empty command. What is typed while a command runs
  is now dropped before the next prompt; a line typed then would have run
  unseen.

### `repl`, `handle_command`'s recovery, `ensure_visible` (the 2026-11-12 evening review)

- **2026-11-12** — A review of the command loop found: an error in a command
  (a camera stall) or Ctrl-C ended the session, whose end sent the arm home
  with whatever it held — now the command ends, the arm held, and the
  session goes on; a failed step still followed the planner's follow-up as if
  it had worked; a sub-goal given up on let the next ones run; a plan the
  safety check blocked sent the arm off to look although nothing had moved;
  a step that failed before it said why left the step before's "ok" as the
  reason; a held object counted as missing and was looked for with the arm;
  and the one object in view was taken for whatever the plan named that was
  not there. Each fixed; and a place now finds a spot before an object, an
  object only by its exact name (a gone "where_X_was" matched X itself).

### `execute_plan`: stopped between steps by something moving near the table's objects (also: `moved_near_scene`, `safety.moved_near`)

- **2026-11-12** — Asked: the UR5 halfway through a stack, a hand knocks it
  down — what then? The sentry paused the arm while the hand moved, and
  then the plan carried on as written: the next block would have been let go
  where the knocked top had stood. Now, after a step, movement the sentry saw
  near an object or a marked spot stops the plan, and the controller looks
  again and plans the goal afresh. A plan the safety check blocks says why
  to the next one (a pick planned with a block still in the jaws, which the
  real planner did, until the tabletop's examples showed it putting the
  block back first).

### `moved_only_far_off` (also: `look_again_if_disturbed`, `MotionGuard.moved_since`, `Interlock.stop_count`)

- **2026-11-12** — A three-block tower in one command waited 72 s between
  its first place and its second pick: 30 s planning the second sub-goal,
  and a full look again because the sentry had paused the arm for 3 s, for
  something moving at the table's far edge, 35 cm from any block. The sentry
  now keeps where it saw movement, all of it (a hand trips it at the edge
  and goes on to the block), and a pause whose movement stayed 10 cm clear
  of every object and marked spot is not looked at again. A stop of any
  other kind, or the sentry gone blind, still is.

### `handle_command` (also: `execute_plan`'s `LAST_FAILED_STEP`, `skill_executor.report`)

- A refused place (a bottle stood on the spot) was re-planned as "home, then
  place there" eight times over: the recovery prompt told the planner only
  what the camera showed, and to home and retry. Seen 2026-11-02, the tabletop
  deployment's planner on the offline UR5. Now the failed step's own reason
  goes to the planner (skill_executor.report, which run_policy fills); a plan
  asking again for a step just refused does not run at all when that step comes
  first (nothing changed, so it would be refused again), and gets one more go
  when the plan does something before it — home, since the SO-101 searches its
  reach from where the arm starts, or moving what was in the way; and an
  identical plan that fails twice is not tried a third time.

## envelope.py

### `GripperBox`

- **2026-11-12** — For the UR5 (it can reach well past the table and into
  itself), the user chose a box the gripper must stay inside — x and y
  0-0.75 m, up to 0.8 m — while the arm itself may pass outside it. The
  gripper is a stick from flange to tips with 5 cm of body round it, checked
  along a move's whole path before it is sent; a move from outside may only
  come back in.


## flight_recorder.py

### `watch_sensors`: the joints too

- **2026-11-13** — A UR5 run wound the gripper's cable round its wrist, and
  what had been kept was forces only: nothing said how far the wrist had
  turned, so the run had to be replayed in memory to find out. The joints
  are now on the "sensors" stream ("joint.<name>"), written with every
  incident.

### `keep_exchange` (also: `vlm._log_vlm`)

- **2026-11-13** — UR5 tabletop: a plan of "home, then the same refused
  place", and the timeline kept only that each model call happened and how
  long its answer was. Every call is now kept whole in the run's models/
  folder: system, prompt, reply, the pictures shown, and for a plan the worked
  examples and the schema. A naming sheet's call keeps about 30 KB.

### `module`

- **2026-10-28** — A queen fell out of the jaws on the way to e3 and rolled off
  the board. The sentry saw it go and paused the arm, but only said so on the
  console; the place opened the jaws on nothing and reported success; the next
  scan marked the queen missing. Each part saw some of it, nothing put it
  together, and the account came from the person watching. Why the recorder
  exists.

## grasp_correction.py

### `module`

- **2026-10-30** — On one pawn over four runs, grasps at a lean of 0 to -5
  deg gripped 6 times in 7, at -15 to -20 deg 2 in 8. In the misses the wrist
  camera showed the pawn beyond the fingertips while the overhead camera and
  the wrist servo both had the jaws over it: the overhead camera corrects
  where the gripper is, not how it is turned, ~90 mm above the jaws.

## interlock.py

### `Interlock.pause_only`

- **2026-10-22** — The user: "I want the sentry to restart automatically after
  the movement". So a watcher whose cause goes away by itself — a hand over the
  work surface — pauses the arm rather than stops it.

## perception/shape_fit.py

### `module`

- **2026-11-16** — Written for the SO-101 tabletop, whose picks closed in the
  air over balls nobody had measured (mrcf/policies/HISTORY.md,
  parallel_jaw/height_looks.py). One outline traced back to the vertical through
  the foot (plane_geometry.measure) reads a flat top high by its radius x cot of
  the slant: on synthetic views at 35-40 deg a 30 mm box with a 40 mm top read
  38-42. Fitted to the outlines of every view at once, a solid on the footprint
  read it 30-31, a 25 mm ball 25, a 70 mm bottle 70-73.

### `OutlineFootprint`

- **2026-11-16** — First an ellipse with the outline's spread: a square box's
  cross-section is not one, and its 40 mm sides read 46 mm across corners'
  worth. And the outline from above is pushed away from the camera by the
  object's height (7 mm on a 30 mm box 0.12 m from below the camera). Carried
  to the heights where each solid is widest and intersected, it is the box's
  own 40 mm.

### `SEES_HEIGHT_PX_PER_MM`

- **2026-11-16** — On synthetic views from above only (the overhead camera and
  the wrist camera at a hover), a 70 mm bottle, its neck unlike every solid
  tried, fitted a straight 50 mm with a narrow spread: "resolved", and wrong. No
  view there saw height (0.2-0.3 px a mm). Resolved now needs one that does.

### `ShapeFit.spread_mm` (also: `height_looks.BOUNDED_MM`)

- **2026-11-16** — Outlines 2 px grown in one view and shrunk in another (4 px
  apart) let a 40 mm solid with a rounded top fit a flat 30 mm box nearly as
  well (0.939 against 0.954): a spread of 28-40. A pick does not need it
  settled to grip it: as the shortest, the hover clear of the tallest
  (bounded).

## perception/plane_geometry.py

### `box_centre_xy` (also: `side_direction_deg`, `scene_scan.observe`'s `boxed`)

- **2026-11-12** — On the UR5 a green block put on red U fell off it: the
  scan had U's middle 11 mm off. The foot is fitted as a circle by an
  outline's near rim, which is right for round things; a block seen at a
  slant shows its sides, which pull that circle toward the camera (D 5 mm,
  a cream block 0.6). A box's middle now comes from its top face, the
  outline carried up to the top's height, the edge away from the camera on
  each of its axes; within 2 mm on cubes rendered from that camera's angle.
  Its sides' direction (minAreaRect, mod 90) squares the jaws to it.

### `cut_off`

- **2026-11-03** — From beside the d1 queen a wrist camera outlined her only
  as far as the frame's edge and read her 28 mm tall instead of 69. Only an
  outline into the jaw had been marked occluded.

## manipulation.py

### `stack_sequence_validator` (also: `_refuse`)

- **2026-11-13** — UR5 tabletop: "stack the three blocks into a tower" was
  planned blue onto green, then green onto red. The arm put blue on green,
  and the pick of green was refused ("blue is on top of green") on the
  console only, so the planner heard "it failed". The plan is now checked
  against what stands on what as each step leaves it, and is not run when a
  pick would be refused. Replayed against gemma4 with that reason, the run's
  own planning call came back bottom-up (green on red, blue on green) twice
  out of two. Every refusal of a pick or a place before it moves now tells
  the planner why (_refuse).
- **2026-11-13** — gripper_sequence_validator did not count place_beside as
  letting go: a plan putting a block beside another and then picking again
  was blocked as "pick planned while already holding an object".
- **2026-11-18 16:37** — Nor place_in: "put the four things into the tub",
  planned pick, place_in, pick, ..., was blocked at every pick after the
  first, and planned again into the same block. Nor did pick_by_rim count
  as a grip.

### `Scenario.carries` (also: `_place`, `_place_on`)

- **2026-11-12** — With stacking added for the UR5's blocks, a place at an
  object was made a place on top of it. In chess an occupied square is no
  place target, and "e4" matches white_pawn_e4 loosely: a piece would have
  gone on a pawn where the room check used to refuse it. Whether anything
  may go on top of an object is now the task's to say; chess says nothing,
  and place_on is not offered to its planner.

### `_check_grip_width` (also: `GRIP_WIDTH_MISMATCH_MM`)

- **2026-11-12** — The UR5's blocks were taken for 40 mm (one guess) and are
  30: the jaws closed at 30, and a place onto another block let go 1.5 cm
  above it. Said now when the jaws hold an object far from its shape's width.

### `_make_room` (also: `_restore_room`, `Scenario.aside_spots`, `Scenario.home_of`, the `nudge` action)

- **2026-11-11** — The user, after the f1 bishop was slid into the e1 king:
  "it could even move the king out of the way slightly then put it back — it
  has full power to get the job done". A pick refused for room now weighs, by
  the policy's own checks and before anything moves, pushing the object
  itself, lifting the neighbour onto a nearby spot, and pushing the
  neighbour; the first that works is made, and a neighbour moved is put back
  after the next place. Two same-height feet always crowd each other both
  ways, so lifting the neighbour comes from jaw turns or different shapes.
  Offline only; not yet on the arm.
- **2026-11-11, review** — Two guards. (1) Nothing is set aside on, or
  pushed toward, where the picked object goes next (`SceneState.headed_for`,
  set by `execute_plan` from the plan's next place and by chess's carry): in
  1. e4 e5 2. Bd3, the king's best free square was e3, beside d3. (2) An
  object moved aside that is no longer where the arm left it
  (`RESTORE_MOVED_M`) is not put back: in a game that is the person's move.
  The aside check now counts the neighbour where it will stand, not gone.

### `MISSED_KEEP_CLEAR_S`

- A dark piece on a dark square came and went from scan to scan on the rig.

### `obstacles_around`

- On the rig, pieces the vision model invented on empty squares refused picks
  next to them.
- A missing piece, kept clear of where it had stood, refused the pick of the
  very piece standing where it had been.
- **2026-11-03** — The d1 queen was refused on the c1 bishop's foot, though the
  fingers pass only its slimmer top. Since then each neighbour carries its
  measured shape (Obstacle.widths_mm). A pick's own measurements of its
  neighbours are kept as theirs (`_remember_around`), so the next pick beside
  them knows the room before it moves.

### `object_action`

- **2026-10-28** — A stand_up that missed on the rig left no incident, and its
  wrist view at the grip was dropped with the Outcome. Since then a verb of a
  deployment's own is recorded like pick and place.

## motion_guard.py

### `_forget_what_came_and_went` (`LASTING_CM2`, `LASTING_PAD`; also: `moved_since`)

- **2026-11-13** — UR5 tabletop: a hand passed within 1 cm of the block a
  place was going to and touched nothing. The arm went back, still holding
  its block, and the look after found nothing moved. Every pause by the
  user at the table's edge also sent the arm to look again, or the plan to
  be made again (a minute each). Once a pause clears, where the sentry saw
  movement is compared with the frame from before it. It is kept only where
  the scene still differs, by 4 cm² or more (a 30 mm block's top is 9), and
  at the spot where it differs. On that run's frames, the block's spot was
  unchanged after the hand; the one lasting change was the user still at
  the edge, 15 cm away. Movement it cannot place on the surface still
  counts as "cannot say" (moved_since None): a look.
- **2026-11-13**, the next run — The user took the block just stacked while
  the arm was still over the stack. The arm's own outline was left out of the
  comparison, and with it the stack, so the pause "came and went" and the
  next block went on where the taken one had been. A region more than a
  quarter behind the arm (UNSEEN_SHARE), in the frame before or now, is not
  checked: what moved there is kept.

### `MotionGuard.GLITCH_L`

- **2026-11-13** — With the UR5's RealSense streaming on the same USB 2 hub
  as the overhead webcam, the webcam now and then sent a single frame far
  darker or brighter (75 or 202 between frames at 136). Each read as
  1800-3600 cm² moving: pauses with the arm still, and a pick refusing to
  come down "where something moved". One frame like that is held back and
  dropped when the next is like those before it.

### `controller` phase messages (`_when_it_moves`, `RobotEnv.on_next_move`)

- **2026-11-13** — Recording demos, the user could not tell when the arm
  would move after a minute of planning. THINKING, then a bell and MOVING
  NOW as guard() first runs for the plan, each step, then DONE.

### `MotionGuard._look`: a black picture

- **2026-11-12** — On the UR5 the overhead webcam dropped off USB; its number
  then opened the MacBook's own camera, lid shut, and the guard watched a
  black picture, in which nothing ever changes, so nothing would have
  stopped the arm. A black picture now counts as not watching (the
  fail-safe holds the arm), as one at opening is refused (real_robot_env).

### `MotionGuard.__init__`

- **2026-10-22** — The user: "I want the sentry to restart automatically after
  the movement" — so a stop for a hand pauses the move, and it carries on by
  itself once nothing has moved for quiet_s (resume=True; Interlock.pause_only,
  RobotEnv.guard).
- **2026-10-22** — A check for "still there" (the scene against the frame
  before the trip) was tried and dropped at the user's word the same day: the
  zone takes in the floor beside the table, and a foot moved and rested there
  held the arm for minutes.
- still_m: at 3 mm, still_m once let three of the SO-101's servo twitches
  (1-2 mm every few seconds, a thousand changed pixels each) stop the arm.
- zone: on the SO-101's first live run, a 0.55 m disc round the base took in
  the floor both sides of the table: all three stops were the user's feet.

## object_memory.py

### `Track.on`, `Track.raised_mm` (also: `scene_scan.observe`, the `place_on` action)

- **2026-11-12** — The user wanted a UR5 demo stacking 4 cm letter blocks by
  typed command ("put Z on P"). Everything stood on the table: a place let
  go at table height plus the grasp, a pick reached down to it, and a scan
  seeing a stack as one 80 mm object would have made the top block 80 mm tall
  and called the one under it missing. Now the arm records what it put a
  block on; a scan of the stack is the top one less what is under it, and
  the ones under it stay (not missing) while it is seen; ParallelJaw raises
  its surface for that pick or place. Offline: three high and off again on
  the UR5 in memory, release heights 23/63/103 mm.

### `ObjectMemory.reidentify` (also: `guessed`, `checked`, tabletop `naming.identifier`)

- **2026-11-12** — Green block 7, put on red U by the arm and taken off by
  hand before a restart, came back named blue_block_z: a greenish block gone
  half an hour. 7's own remembered foot was mostly older readings (median
  22 mm against 34 now), which ruled it out, and Z's look was the nearest
  left. Told "now stack the blue on top of the green tower", the splitting
  step took both "blue" names, and the plan moved the green a second time;
  the pick was refused, as the blue was on it by then. Now a sighting the
  scan took by look alone is a guess, and the deployment's vision model can
  give it to the remembered object it reads. The camera's colour must agree
  too: on the same day the model read red U, lying on its side, as D, and at
  full size as P.

### `SAME_PLACE_MAX`

- The first live scan called a black rook white: one attribute misread, and
  still the same object.

### `Detection.confirmed`

- **2026-10-21** — A vision model put pieces on five empty squares; each
  blocked picks as a neighbour, and matched as a "move" of the real piece,
  which then lost its identity to it.

### `ObjectMemory.observe`

- **2026-10-28** — Objects just put down by the arm were judged like any other
  sighting: one read as 8 mm, not the queen's 27, became a new object, and the
  queen, "missing" from that square, then kept a place
  there refused for its 15 minutes; one landed 18 mm off and read white, and
  the queen's name went to a fallen rook across the board. (also:
  `PLACED_NEAR_M`, `Track.refoot`)
- **2026-10-28** — The place reported success for a queen that had fallen out
  on the way and rolled off the board. The first scan after a place that does
  not see the object now writes an incident.

### `Track.shape_doubted`

- **2026-10-29** — A 42 mm pawn stood on e4, where a 52 mm piece had stood the
  day before and was then missing. Its foot was within 3 mm of that piece's,
  so the memory took it for that piece back again, with its 52 mm. The wrist
  camera's look at the hover was too dark to trust, and the pick gripped by
  the 52 mm.

### `ObjectMemory.placed`

- **2026-10-28** — A scan read a 69 mm queen's foot as 27 mm on d4 and 17 on
  e5, and that difference alone dropped its shape and had the next pick hover
  over it as a piece nobody had measured. (also: `Track.refoot`)

## overhead_calibration.py

### `apply_to`'s lens (also: `tools/calibrate_camera_by_hand.py`'s `calibration_record`)

- **2026-11-12** — The UR5's camera was calibrated by hand with a tag on its
  arm, and the tool wrote only its report: the pose, the table homography
  and the refitted lens went into the rig's file and robot_config by hand,
  and apply_to dropped the lens, so a second run would have changed nothing.
  The tool now offers its fit as the arm's calibration, keeping what else the
  file held, and apply_to loads the lens too.

### `module`

- **2026-10-22** — The camera was moved overnight and the board with it —
  every square read 25-35 mm off in the arm's frame, a pick spent 22-26 mm of
  correction at each hover and missed, and nothing said why. Since then a scan
  checks the camera against the board in every clean frame it takes.

### `MOVED_MM`

- **2026-10-22** — Between two still frames of the tour the board's pose
  repeated to 0.05 mm and 0.006 deg. (also: `TURNED_DEG`)

### `REPOSE_MAX_MM` (also: `REPOSE_MAX_DEG`)

- **2026-10-31** — A scan during a pick's re-find put the camera 360.4 mm and
  38.45 deg from its calibration from one clean frame of the board
  (homography 1.2 mm mean), re-posed and saved it; the arm hovered by it, the
  motion guard took the arm for a hand 40 times over two minutes, and the pick
  ended on the height ceiling. The next scan re-posed it back to within 1 mm
  of the 10-28 pose. A real knock (10-22) had put squares 25-35 mm off.
  Incident: `runs/flight/20261031_103651/incidents/104037_pick_motionstopped__end_point_z_0_417_m_above/`.

### `ARM_AGREE_MM`

- **2026-10-22** — The arm and the camera agreed on where the gripper is to
  ~1.5 mm median after a tour (check_overhead_correct.py).

### `check_camera`

- **2026-10-22** — The camera moved with nothing noticing: every square read
  25-35 mm off, and the picks failed (the user: "the live run needs to be able
  to identify this discrepancy and run a rescan and update systems").

## rig_state.py

### `STARTUP_STATE`

- **2026-10-22** — The failure it guards against: what a backend reads as it
  starts, left in its code folder by an older layout, passed over for
  robot_config's older values without a word — every square 25-35 mm off.

## robot_env.py

### `RobotEnv.PAUSE_MAX_S`

- **2026-10-22** — None since then: a 120 s cap abandoned a pick while the user
  worked beside the board for over two minutes ("why doesn't it restart?").

## safety.py

### `approach_contact_limit` (also: `descend_onto`, `RobotEnv.contact_force_limits`, `zero_contact_force`)

- **2026-11-12** — The UR5's pick of a block landed a finger on it (the jaws
  then opened only 38 mm): 19 -> 70 -> 140 N in 0.2 s, stopped at 109 N
  against a 100 N limit that free-air moves need (its estimate reads 35-74 N
  touching nothing). A tighter limit, 30 N, now holds for the last
  descent onto an object, where its slow moves read 1-10 N. Applied first
  over the whole approach, it stopped a gripper touching nothing at 32 N:
  the reading had drifted over the long move there; it is counted from
  each descent's own start. Then each descent, first and retried, came to
  have it (descend_onto): the place's retried descents had run at 100 N, the
  pick's moves up and across between them at 30 N from a stale start.

### `ExecutionMonitor.check`

- Passing near_miss_force_n as the near-miss's limit_value made every
  near-miss report 0% margin (observed >= trigger by construction), which also
  made the "margin < 20" print gate always fire. It now passes the hard stop,
  max_contact_force_n.

### `SafetyThresholds.max_contact_force_n` (also: `near_miss_force_n`)

- These were 5 N and 3.5 N for every arm, set by the task. The SO-101 reads
  0 N, so they never fired. A UR5 (CB3) estimates its force from its joint
  currents and is off by several newtons at rest, so they would have stopped
  every UR5 skill. Found by reading the code before the arm's first session,
  2026-11-02. Now None by default, meaning the arm's own limits
  (env.contact_force_limits), else 5 N and 3.5 N.

## scene_scan.py

### `observe`'s `identify`

- Added 2026-11-02 for the tabletop deployment, which names each object with
  the vision model from a crop of its outline: scan() returned the frame and the
  outlines, but observe() kept neither.

## scene_state.py

### `SceneState.find`

- Searching objects and targets both, "white_piece_e4" — a piece the scan had
  since renamed — matched the empty square e4, and the arm went down on an
  empty square. Hence `among`.

## vlm.py

### `decompose_task`'s object names

- **2026-11-12** — "Stack all the blocks" did nothing on the UR5: splitting
  it into sub-goals with no names to go by, the model invented "pick up
  block 1 and put it on block 2", and the planner, which knew no block 1,
  asked instead of moving. It now gets the scene's names and uses only those.

### `label_numbered`

- **2026-11-03** — The chess scan named 32 pieces one crop a call, two looks
  each: about 4.5 minutes a start. All of them numbered on one sheet, one call:
  16-27 s, and better answers (the chess HISTORY.md's `label_pieces`). The
  answer is most of the call's time — 524 tokens for a field each, 248 for
  "white pawn" — hence one short answer a number.

## backends/rig.py

### `RigEnv.down_rotation`, `_flange_for`, the gripper's box (`CHECKS_GRIPPER_BOX`, `_refused_by_box`)

- **2026-11-13** — Moved here from the UR5's env so a third arm copied from
  the template has them. The template's straight-down moves had the jaws
  pointing one fixed way across the table while its policies planned room
  for jaws turned to the line from the base: right only along one axis, the
  UR5's jaw-axis bug again. down_rotation now takes the slide axis as
  mounted (the UR5's assumed the flange's y; the same rotations for it, every
  axis, flip and turn checked). The box was the UR5's only; an arm whose moves
  do not check one is now refused one rather than given one it would ignore.

## backends/wiring.py

- **2026-11-13** — Written after the 11-12 sweep found the UR5's
  make_policies had never passed JAW_AXIS: the template and the UR5 each
  copied about forty settings into make_env and a dozen into make_policies by
  hand. The three arms' keywords and specs were recorded before the change
  and compared after: the same, but for the template's new jaw axis. The
  UR5's pick choices (no hover measuring, no servoing, no lean retries, its
  miss and unknown widths) moved from make_policies into its config.

### `RigEnv._start_sentry`

- **2026-10-22** — The user's rule: "any movement with the arm must run the
  sentry protocol so it stops if there is a hand in the way". So every arm on a
  rig is watched while it is connected, whether or not its own connect()
  remembers to. (also: `RigEnv.__init_subclass__`)

### `RigEnv.wait_still`

- The servos creeping on toward their last goal, up to 10 mm, were seen by the
  overhead camera's gripper tag on the SO-101's first tag tour
  (tools/tag_tour.py).

### `RigEnv.visual_servo_xy`

- The first working pick found the pawn 105 px out against a fixed 110 px
  radius — one bad scan from failing. Hence acquire, then track.
- look(): from a 60 mm hover the detector found the board's pose in two frames
  of three, and quitting on the first miss turned that into "target lost" on
  every other look.
- A 36 mm correction asked for in one move stalled the solver partway on every
  try (the scan was that far off). Hence max_step_m.

## perception/fiducials.py

### `BULLSEYE_LEVELS`

- On the rig one jaw's sticker ran into a light square along one edge, and the
  other's dot was half washed out by glare. (also: `find_bullseyes`)

## perception/lighting.py

### `module`

- **2026-10-27** — Checked: the SO-101's two cameras ignore every UVC image
  control.

### `TOO_DARK_MEAN`

- **2026-10-15** — Picks failed at a wrist-camera mean of 41-67; 102-136
  worked.

## perception/lying.py

### `LYING_MIN_CHANGED`

- **2026-10-28** — Two frames: a walnut square with a pawn's foot on it read
  41 mm, 1.5 times as long as wide, one end wider — 0.57 of it changed; the
  pawn itself 0.90-0.92.

### `UPRIGHT_LINE_DEG`

- **2026-10-28** — An upright pawn's head over its foot made one end the
  narrower: 28 mm, 1.6 times as long as wide, 0 deg off the line from the
  camera's foot through it.

### `measure_lying`

- The knob's width, taken over a knob's width back from the end, on a pawn
  reached its collar: 16.5-17 mm for a 13 mm head, and its middle put 2.5 mm
  short of where two cameras saw it. Hence the top END_SPAN's widest.

### `find_lying`

- **2026-10-28** — A lying cream pawn across a walnut and a maple square came
  out as its walnut half alone (22 mm), the whole pawn (38-41 mm) only as a
  less sure outline. Hence the longest, not the surest. The same pawn with its
  head on a maple square was never outlined whole — head and maple are 7-14
  apart in Lab, and the segmenter and the change map both lost it.
- An upright pawn's outline with its shadow read as one lying (34 mm, its ends
  unequal) beside the outline of the pawn alone.

### `find_lying_in_frame`

- **2026-10-28** — Cream on maple and walnut: close up, the segmenter outlined
  a lying pawn in parts — its head, its foot — and never whole.

## perception/outlines.py

### `keep_outlines`

- **2026-10-17** — With the light behind the board and the camera behind the
  arm, the surest outline put one piece 0.209-0.222 m out over four scans, foot
  32-34 mm, while a smaller outline of the piece alone read 0.221-0.222 every
  time. Not always with a narrower foot: on one scan both read the same width,
  the shadow only on the near side.
- Keeping the largest outline made one chess pawn's foot read 21, 33 and 34 mm
  on three scans of the same board.
- **2026-11-04** — The full starting position: the d1 queen's outline was
  replaced by her head's, inside it, whose foot read 23 mm off, past the
  board's edge, so the scan lost her on every frame; once the king's the same
  way, 20 mm off. Shadow-free replacements moved the foot 0.4-10 mm on the same
  frames. Why ALT_MAX_FOOT_SHIFT_MM — and why a wider foot still replaces it
  (11-03: a 4 mm sliver of a pawn, replaced by its whole foot 21 mm away).
- **2026-11-04** — One outline took in the f1 bishop and the f2 pawn beside it;
  the bishop's own outline then replaced it, and the pawn was lost. Why an
  outline round two objects is split (`_parts`) before either replaces it.
  Replayed on 19 frames of 11-03/04: no piece lost, the queen and f2 found.

## perception/placed_check.py

### `look` (also: `Scenario.confirm_placed`, tabletop `skills.confirm_placed`, `SceneState.look_frame`)

- **2026-11-12** — The user: "maybe we should add a quick look to see if it
  succeeded?" A place said "placed" for a block that had fallen off the one
  under it, and the next step went on as if it stood there. One frame, a
  fraction of a second, after letting go: the patch where its top should be.
  Colour alone was not enough — on the UR5's frame a patch of bare table 6 cm
  from each of 9 blocks read as that block 5 times — so it is also compared
  with the last scan's frame: there, bare table read as unchanged 8 times of
  8, and blocks moved between two real frames as changed.

### `look_volume` (also: `RigEnv.fixed_views`, `scene_scan`'s kept views)

- **2026-11-13** — The UR5's overhead camera cannot see a stack's top while
  the gripper is over it ("the arm is in the camera's way there"), and a
  finished tower that fell went unnoticed. A RealSense D435 was put low at
  the table's side; OpenCV gives its infrared view only (greyscale, its
  projector's dots; on USB 2 no colour camera). Calibrated by hand with a tag
  on the gripper after it was moved twice off the windows' glare, which had
  cut the tag's contrast to a third of the overhead camera's. VOLUME_CHANGED
  from real pictures: new blocks 43-81, unchanged spots at most 27.

## perception/segment.py

### `module`

- Every colour rule tried on the first rig (chess) separated a piece from the
  board by colour, and a boxwood piece on a maple square is the same colour as
  the square: the residual detector only ever saw the part of a cream piece
  standing over a walnut square. Hence a learned segmenter.

## perception/surface_scan.py

### `drift`, `check_drift`, `CameraMoved` (also: `scan`'s `max_drift_px`)

- **2026-11-13** — UR5 tabletop: the overhead camera's new mount crept after
  its calibration, measured against the empty-table photo taken just after
  it: 0.3 px at calibration, 0.5 at 14:10, 8.5 at the run's start (14:25),
  12.8 at the pick that landed on its block (about 14 mm on the table).
  Features matched between the frame and the photo, one slide and turn by
  RANSAC, 15-60 ms. With a limit, a scan stops before outlining anything
  (a test case: 12.9 px, stopped; the calibration's own frames, 0.2-0.3 px).

### `without_shadow` (also: `plane_geometry.outline_objects`' `trim`)

- **2026-11-13** — The shadow test removes outlines that are mostly shadow,
  but a block's own outline took in a strip of its shadow (30% of it, 1401 px
  against 995), and box_centre_xy put the block 9.1 mm toward the shadow (by
  its coloured border). Measured without the shadow's pixels: 2.9 mm, in
  three replays of the run's frame.

### `SHADOW_SHARE` (`SHADOW_AB`, `SHADOW_DEEPEST`, `compare`, `is_shadow`)

- **2026-11-13** — UR5 tabletop, overhead camera just moved closer: every
  block had a twin about 20 mm away, which refused picks and places for room.
  Replayed offline on a live frame, the twin was the block's soft shadow,
  72% changed from the empty table, outlined and then named after the block
  beside it by the naming sheet. Yesterday's one-crop naming named it a block
  too, and drawing its outline on the tile fixed one shadow of two. Of the
  changed pixels in each outline, the share only darker (by less than 40
  levelled) in the table's own colour (within 4): shadows 72-84%, the blocks
  0-1% (their coloured borders and letters), the arm's black gripper 7-13%. At
  0.5, three replays kept each block once and no shadow.

### `CHANGED_LAB`

- **2026-10-22** — Measured on 21 overhead frames of a chessboard, a cream pawn
  on it (d4-d6, e4, h2), against the empty board built from 9 of them (the
  median of frames with the pawn elsewhere — reference_from_frames): at 10 and
  0.4 the pawn was found in every frame, 0.60-0.97 of its outline changed, and
  no bare square came out at all; the grid-line rule it replaces had missed the
  pawn near e4's corner in four. Cream on maple is only 7-14 apart, so 18 (the
  old value) was too coarse to see it. (also: `MIN_CHANGED`)

### `LIGHT_SCALE_PX`

- **2026-10-27** — On the overhead frames, compared raw, the bare board came
  out changed over 22% of its area two hours after the photo, 45% by the
  afternoon and 43-58% five days later (2026-10-27, where it let patches of
  board either side of a queen through as pieces); levelled, 0-9% on all of
  them, with the pieces at 67-99%.

## perception/tiles.py

### `upright`

- **2026-11-03** — Crops framed round the overhead camera's outline of a chess
  piece take in its foot and cut off its top: the tall pieces lean away up the
  frame, and a king showed no crown. Framed from below the foot to where the
  tallest piece's top projects, turned upright, the white back rank read as a
  person would read it. Why the frame is cut by the view, not by the outline.

## containers.py

### `Container.opening_xy` (also: `overlap`)

- **2026-11-17** — First written to carry the overhead camera's outline of a
  container up to its rim height, on the reasoning that this errs inward on
  both sides. It does on the far side; on the near side the outline is the
  wall's foot, and carried up it moves toward the camera — out of the
  container when the camera is outside it on that side, as the SO-101's
  overhead is beside the tub (16 mm out in the synthetic test). The opening is
  now the overlap of the outline carried at the surface and at the rim.

## perception/grounding.py

### `CONVENTIONS`, `is_container`

- **2026-11-17** — The SO-101's overhead camera, 1280 x 720: qwen2.5vl:7b
  boxed the clear plastic tub at [167, 38, 515, 567] (by eye 170-510, 40-555)
  in 0.5-3.5 s and said found: false for a thing that was not there;
  gemma4:31b gave the same box as [y, x, y, x] in thousandths. Asked "is it a
  container" in the same reply, qwen boxed the tub when asked for "the red
  figure"; so locating and judging are two questions. On crops, gemma4:31b
  told the tub and the scoop from the plug and a figure — but once called a
  figure's tight crop a container, live, and it was kept as one: hence two
  crops, both to say so.

## controller.py: `ensure_visible` and a container not yet found

- **2026-11-17** — The plan schema holds every `_name` parameter to the
  scene's names, so a container nobody had found could not be named: asked to
  put the red ball in an unlisted tub, gemma4:31b planned
  `place_in(target_name="pink_ball")`. `place_in` takes `container` in the
  user's words instead and looks for it itself (`Scenario.find_opening`):
  the same request then planned `place_in(container="plastic tub")`. A first
  try used `Deployment.look_around`, which runs for any name a plan uses,
  and looked for a pick's misnamed object as a container.

- **2026-11-19** — tabletop/skills._picked_hook: a pick planned by the
  catalogue's recipe (nothing measured) learned the recipe's top plus its
  10 mm margin as the top it cleared, so each reuse raised it 10 mm (the
  plug's 72 mm became 112 over four runs, its hover out of reach from 34 cm).
  Now kept as it was. The tidy catalogue's two ratcheted tops were put back
  to what they were first learned as (plug 72 mm, cap 69 mm).

- **2026-11-19** — catalogue.py: one kind per object (same name and colour,
  merged on load and on add — the scoop had been kept twice, its better grip
  under the other); each pose keeps the range its sizes were seen at (the
  plug 42-57 mm across); a colour this far off rules a match out (an
  upside-down blue scoop, plug-sized, was taken for the white plug).
  tabletop/skills: a kind found by its name and colour, resting a way not met
  before, is gripped low with the hover clear of its longest side, where that
  is short enough (user: "the plug should be liftable in any orientation");
  the new way it rests is learned as its own pose. rim_pick: a wall felt under
  load is held while the jaws stay where they stopped (HELD_WHILE_STILL) —
  three times on 2026-11-19 the scoop was called "left the jaws" and carried
  home in them.

- **2026-11-19** — policies/handle_pick.py (GripperSpec.handle_first): a thing
  with a handle — the long narrow part leading from its widest part, found
  along its outline's long axis — gripped across it, the jaws turned to it,
  closed from high down in handle_step_m steps until they stop on it under
  load; rim_pick tries it before a wall. manipulation.PARTS_INSIDE_ARE_ITS_OWN:
  what the scan put inside the object being picked is part of it. The pick's
  Outcome carries the height it gripped at (a lean raises it), for the place.

- **2026-11-19** — controller.Deployment.quick_plan: a deployment may plan a
  command from its words itself (the table task's "put these into that",
  tabletop/skills.quick_plan, behind QUICK_PUT_INTO); None, and the planner
  plans as before — every deployment that does not set it. catalogue.Kind.
  picked_by: the action that held it.

- **2026-11-19** — place_in: GripperSpec.container_lean_above_walls — going
  down inside a container, one lean over the spot (as far as that height
  takes) and another, within container_max_lean_deg, once the jaws are among
  its walls; spots it can go down into first. pick: descent_turn_first — a
  descent re-chosen to lean otherwise turns over the object first, and one
  that went off its line goes back over the grasp. quick_reseat — the wrist
  camera's reseat unsettled on its way up and across. All off but for tidy.
  containers.Keeper keeps each container's outline on disk (outline_xy, seen)
  for a tool without a camera; tools/_common.task_config takes the deployment
  run.py launched last (rig/<arm>/last_deployment.txt) before the arm's own.

- **2026-11-19** — GripperSpec.refused_descent_ends: a pick's descent the
  arm's own checks refused (MoveRefusals) ends the attempt instead of leaning
  further. controller.Deployment.refind_near_m: a unit set aside whose object
  is tracked afresh is the one of its name nearest where it was (tabletop
  REFIND_NEAR_M). Both off but for tidy.

- **2026-11-19** — controller.Deployment.task_plan / task_names / planner:
  a deployment's own tasks planned in advance, tried first; planner False and
  nothing else is planned — a command none of them is answered with the tasks,
  a failed step is said, not planned again. The table task: TASKS (a sentence
  or written steps, each object found when it runs, one not found left out)
  and PLANNER in mrcf/tabletop/settings.py; quick_plan leaves out what it does
  not find when there is no planner. tools/new_deployment.py starts a copy's
  TASKS empty, the source's commented. User: "i want to have the option now
  for any new deployment that we can skip the planner".
