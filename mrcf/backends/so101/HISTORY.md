# SO-101 — history

Why the code is as it is: the runs on the rig behind its values and fixes,
with their dates, by file and by name. The comments in the code say what each
value or step is for; the story of how it was found is here. What is still
wrong is in NOTES.md, not here.

## robot_config.py

### `module` (task rules out: deployments/<task>/profiles/so101.py)

- **2026-11-17** — ~30 settings tuned on chess pieces, and 14 the tabletop
  added, moved into the tasks' profiles (mrcf/HISTORY.md, profiles.py). What
  stays is the arm's: its jaws, its reach, its calibrations, what an empty
  close reads, the grasp its calibrations were measured with.
  GRIPPER_MAX_OPEN_M is now the arm's usable 80 mm (the jaws open to 100);
  chess narrows it to 45.

## motion.py

### `_move_ee_to` near stall (also: `_IK_NEAR_STALL_ITERS`, `_IK_NEAR_PROGRESS_M`)

- **2026-11-18 16:15** — Reaching a figure 36 cm out, a descent hung 4-8 mm
  off its goal, the loaded shoulder ~1.2 deg short: inside the stall check's
  2 x tol, so nothing ended it but the 20 s backstop, the arm shaking the
  whole time (user: "just shaking"), then stopped 8 mm short. A move within
  2 x tol that gets no closer by 0.5 mm in 20 ticks is now settled onto the
  solver's solution once, and ends there. Offline, servos frozen 7 mm over
  the goal: 401 commands before, under 120 now.

## follower.py

### `Follower._below_floor` (also: `FloorRefused`, `FLOOR_Z`, `_FLOOR_SLACK_M`, `motion._reach_error`, `_REACH_SLACK_M`)

- **2026-11-17 14:27** — A height look on the tabletop rose toward a transit
  40.6 cm up — the folded-up park pose's height, after a scan — that the arm
  could not reach with the gripper held for the look; it strained there,
  stalled, and the move on down to the look put the gripper into the table
  (user: "you just smashed into the table"). Nothing checked a command's
  path against the table: envelope.py's floor check was the UR5's only. Every
  arm command's joint path is now checked (jaw tips at both finger edges and
  the wrist above FLOOR_Z; lower than now and under the floor: not sent), and
  a goal held to an orientation the IK cannot bring within 2 cm of is refused
  before anything moves. The guard first had no slack: a scoop's wall grip,
  the jaws 16 mm up, was refused for 1 mm; 3 mm now. The reach check first
  used the orientation search's joint margins, and refused the lift back out
  of the tub it had just come down from.

## follower.py

### `Follower._setup_kinematics`

- A copy of this (for the offline arm) that missed the joint corrections mapped
  reach for a different arm — why the offline arm is built by this same method.
- Taken through the joint offsets, the straight-down rotation (`_r_down`) came
  out 12.7 deg from vertical; it is taken from the uncorrected model.

## kinematics.py

### `_TCP_OFFSET` (also: `_FIXED_JAW_PAD_X`)

- This replaces a FIXED -8.5 mm, i.e. this same formula frozen at a 17.7 mm
  opening — so it was wrong at both ends of a pick, in opposite directions:
  - touching overhead-calibration corners with the jaws SHUT (gap ~0, true
    offset ~0) it reported the grasp point 8.5 mm off along the gripper's x, at
    whatever wrist angle the corner was touched at;
  - descending with the jaws OPEN at PREGRASP_OPEN_M (gap 30 mm, true offset
    -14.7 mm) it aimed 6.2 mm too far toward the fixed jaw, leaving that pad
    8.9 mm from the piece's centre — inside the ~10 mm radius of a chess piece.
    The jaw therefore drove INTO the piece instead of down beside it, while the
    moving jaw still had ~12 mm of slack on the other side.

  The second of those is why a descent that converged to its target still
  bumped the piece rather than straddling it.

### `apply_solver_limits`

- Before the solver was held to the effective limits, the IK loop could not
  converge: observed as "IK stalled 1.3 cm from target" with elbow_flex pinned
  at the URDF stop, and the gripper closing a jaw-width to the side of the
  piece. Seeded from a pose near the board it was 14.5 cm out.
- visualise_ik's offline arm's solver ran without these limits, and previewed
  solutions the live solver never returns.

### `straight_down_rotation`

- "Down" taken at the zero pose (approach axis horizontal, 90 deg off) made
  approach_orientation ask IK for a gripper pointing sideways at the table,
  which only solves by folding the arm onto its joint stops (shoulder_lift
  pinned at the URDF's -100 deg limit), and made _APPROACH_TILTS_DEG mean
  "degrees from horizontal" — so with a 60 deg cap it could never express a
  vertical grasp at all. The function raises if the result is not actually down
  so that a URDF or joint-order change cannot silently reintroduce this.

## gripper.py

### `GripperMixin.set_gripper`

- With one send_action and the 8.0 this deployment uses, asking for 100% from
  8% yielded 16%, i.e. jaws that travelled 3.7 mm -> 7.1 mm and never opened.
  Every "grasp" then closed on air, and the ~4 mm readback was close enough to
  a real grip to be reported as success. Why it is a loop.

### `GripperMixin.drive_gripper_pct`

- After the ramped close this used to make a FULL un-ramped jump to `pct`,
  bypassing step_pct and falling back to lerobot's arm-wide max_relative_target
  (8.0, more than 5x GRASP_CLOSE_STEP_PCT's 1.5) — one large, sudden squeeze
  just as the jaws made contact, plausibly what popped a smooth, round piece
  sideways out of the grip. Replaced by a few more steps at the same gentle rate.
- This used to wait all n_steps whatever the jaws did — 0.75-1 s after every
  opening and close, by which time they had long since arrived or stopped on
  the object.

### `GripperMixin._min_hold_m`

- **2026-10-22** — Load alone stopped telling a held piece from an empty close
  once grip pads were fitted: the pads meet at 0.9-1.0 % of travel (1.7-1.9 mm
  on the width table) and press each other at the torque cap — load 160-200 of
  200 on three empty closes, where the bare jaws went to 0.1 % at ~30.

### `GripperMixin.squeeze_gripper`

- With set_gripper's ramped close an EMPTY close was measured stopping anywhere
  from 0.6% to 7.8% of span, overlapping the 4-12% a held pawn or knight stops
  at, and a held pawn read load 25, the same as an empty close. Overshooting:
  0.1% on every empty trial, a piece blocking the jaws at 3.7-6.2% with the
  servo at its torque cap. Pressing at lerobot's 500 tripped the servo.
- The opening alone stopped separating the two once the torque was capped — an
  empty close read up to 0.48 mm and a knight held on the board 0.74 mm — so
  `loaded`, the servo's load, became the grip signal.

## orientation.py

### `OrientationMixin.approach_orientation`

- `tol` used to be hardcoded at 15 mm while move_ee_to's default was 10 mm, so
  this would approve an orientation whose own IK solution sat 10-15 mm short —
  and the move that followed then reported "IK stalled 1.0 cm from target"
  having done exactly what was approved.

### `OrientationMixin._x_along_arm`

- **2026-10-21** — Until this day "radial" meant x along the arm whatever the
  jaws did, and the policy modelled the fingers and the lean on the wrong axis.

### `OrientationMixin.tilted_rotation`

- This used to ask for a lean fixed in the world; anywhere off the line
  straight out from the base the solver met the position and quietly gave up
  on the orientation, so the tilt the arm actually took was not the one
  approved.
- **2026-10-28** — Live: -22 deg about the gripper's own z turned the jaws
  22 deg counter-clockwise seen from above — the sign `turn_deg` uses.

## motion.py

### `_MIN_LEAD_DEG`

- **2026-10-19** — With the command re-based on the measured angle every tick,
  a joint the solver moved by a fraction of a degree per tick was never asked
  to move further than that, and did not move at all — shoulder_lift held at
  -18.2 deg for a whole descent commanded from -19.4 to -17.4, the elbow and
  wrist did the moving, and the jaws went 3 cm outward instead of 5 cm down
  ("it doesn't bend the shoulder forward").

### `MotionMixin._move_ee_to`

- Waiting for each intermediate goal to be reached (a _settle_to per
  iteration) is what made the motion visibly jerky. Settling per iteration was
  there originally because per-iteration progress measured servo lag, read as a
  stall; the windowed stall check handles that instead.
- The stall check's misfire near convergence (with a 4 mm tolerance no
  iteration inside ~6 mm can improve on the best by 2 mm) is what killed the
  gripper-servo calibration's x+ probe at a pose whose IK solution is exact
  (verified offline, 0.0 mm out). Since then a stall is only called far from
  the target.
- **2026-10-19** — Asking a joint the deadband holds a little further each tick
  (what the solver asks added to what was last sent) is what made descents
  reach.
- **2026-10-20** — `max_iters` at 60 (3 s) cut off long moves from home on these
  gently-accelerating servos 2-4 cm out, still closing. Now 400 (20 s).
- **2026-10-20** — Each joint was clipped to the step on its own. From a 70 mm
  hover (wrist at its limit) to a 27 mm grasp the elbow opens ~40 deg while the
  shoulder needs ~12: clipped apart, the shoulder ran ahead and lifted the jaws
  5 cm, 10 cm from the target, before the elbow brought them down — and the
  stall check stopped it there, "descent stalled 8-10 cm short". The step is
  scaled as a whole since.
- **2026-10-20** — Running the command on past a joint that was moving wound up
  against the servos' lag — on travel moves pan swung 15 -> 24 -> 15 deg on a
  0.7 s cycle, the jaws circling 13-22 mm off the target until the stall check
  gave up. Since then, on moves that are not gentle, a joint is run ahead only
  while it is stuck.
- Holding the loaded joints back to "only when stuck" on gentle moves too
  stalled a place's descent at d5, the arm reaching out, twice, 5-6 cm short.
  So a descent still runs every tick while the joint is asked the same way.
- **2026-10-28** — Streamed in Cartesian space all the way, the joints drifted
  apart on the real servos — the shoulder ran past its command, the elbow
  trailed it by ~6 deg — and the jaws came DOWN before they came out: 3-5 cm
  below the hover, still 3-4 cm short of it, where the neighbour toward the arm
  stands, and the stall check ended the move there (c5 and d5 from the park
  pose). Since then a long way sideways goes in joint space first. (also:
  `_joint_line`, `_JOINT_SPACE_FROM_M`)
- **2026-10-28** — Leaning 15-20 deg, two nudge corrections at a grasp stalled
  with the jaws swung 7.6 mm round the queen, and the close found nothing (d5
  and e5). Since then a nudge is solved and settled onto. (also: `_NUDGE_M`)
- **2026-10-28** — At a grasp on d5 the jaws went 7.7 mm round the queen after
  two stalled corrections of 2-3 mm: the last command, up to `lead` ahead of
  the joints and left standing, carried the arm on after the move had given
  up. Since then a stall holds the arm where it is.

### `_PUSH_PAST_GOAL_DEG` (in `MotionMixin._move_ee_to`)

- **2026-11-04** — The user saw the arm sway side to side and bob up and down
  with a light piece. Recorded at a grasp on e5 (10 mm, held to 1 mm, 3 deg
  steps): shoulder_pan swung -6.5 -> -2.6 -> -6.5 deg every 0.6 s, ~2 cm at
  0.32 m, its command swinging -2 -> -7 with it; every joint reversed dozens
  of times a minute while moving. The push meant for a joint held by its
  deadband ran the command on past the solver's goal while the joint was
  moving, and the joint overshot and was pushed back.

### `MotionMixin.go_observe`

- Reaching the observe pose through Cartesian IK previously left the arm
  stalled against a servo limit instead of actually reaching a pose that could
  see the table (see `OBSERVE_DEGREES` under robot_config.py).

### `MotionMixin._joint_line`

- **2026-10-28** — Descents tried in joint space on the rig, in 1 cm pieces:
  the jaws went 16-17 mm sideways on 4 cm descents round a queen and the closes
  after them missed — worse than the Cartesian descent, so descents stay there.

### `MotionMixin._settle_to` (also: `_SETTLE_PUSH`, `_SETTLE_PUSH_MAX_DEG`)

- tol_deg used to be 2.0, chosen because check_hold.py had found elbow_flex
  sitting at a flat 1.6-1.8 deg offset from commanded at every pose, unchanged
  after 8 s of settling — asking for better was asking for precision the arm
  did not have. That offset was not mechanical. lerobot sets I_Coefficient = 0,
  so the servos had no integral term and could not wind out their own
  steady-state error; _tune_servos now sets it (robot_config's
  SERVO_I_COEFFICIENT). Re-running check_hold.py after that change:

  ```
  elbow_flex   1.6-1.8 deg at every pose  ->  0.1 deg at every pose
  shoulder_lift, held 1 s                 ->  0.0-0.2 deg
  ```

  So 0.5 deg became comfortably inside what the arm can hold. At ~3.2 mm of
  grasp-point travel per degree, 2.0 deg was letting the arm start a grasp up
  to 6 mm off.
- **2026-10-28** — 20 settles traced: waiting on the integrator alone took
  0.6-1.5 s per move — over half of a pick's moving time: ~0.3 s for the servos
  to catch up, then the shoulder or elbow parked ~1 deg short under its load
  while the integral crept it in, often to the timeout. Hence the push on a
  joint that has stopped short.

### `MotionMixin.goto_joint_degrees` (also: `Follower.STALL_S`, `STALL_FAR_DEG`, `STALL_MOVE_DEG`)

- The wrist roll, blocked, was pushed for the whole timeout: stuck at -90 deg
  through 107 commands home, it tripped its overload protection and then
  stopped answering on the bus.

## cameras.py

### `resolve_camera_index`

- macOS AVFoundation indices were observed empirically on this project to swap
  even with nothing unplugged, e.g. when a Continuity Camera device (an iPhone)
  appeared or disappeared from the list and shifted every index after it.

### `CamerasMixin.frame_age_s`

- **2026-10-26** — The wrist camera stopped sending for seconds at a time and
  then came back by itself.

### `CamerasMixin._reopen` (also: `REOPEN_EVERY_S`, `_reader_died`)

- **2026-10-31** — Three of four runs ended with a camera's reader thread
  dead: the wrist camera's mid-scan; the overhead camera's (the motion guard
  froze the arm and it never went home), then the wrist camera's; and the
  wrist camera's again during a pick's third attempt, with the iPhone's
  Continuity Camera gone from the list. lerobot's reader gives up for good
  after eleven failed reads in a row, with no pause between them; each time,
  a plain `cv2.VideoCapture` read the same camera without a failure straight
  after the run.
- **2026-11-14** — The chess controller died at startup. The wrist camera
  (on its own Mac port since 11-11) dropped off USB three times in 8 s
  (10:32:42-50) during the self-check's hover, and the reopen closed it. The
  next joint command, the move to the park pose, raised "SOFollower is not
  connected": lerobot counts the follower connected only while every camera in
  its `cameras` is, and checks on each `send_action` and `disconnect`. The
  shutdown's disconnect was refused too, which left torque on (as configured
  anyway). Fix: while closed, the camera is held out of the follower's
  `cameras` (`_closed_cameras`), under the bus lock, and goes back once it
  reads; `Follower.stop` disconnects piece by piece if lerobot refuses.
  Tests: `_CheckedFollower` in test_so101_env.py.
- **2026-11-14** — It dropped again at 11:03, mid-game, and hung; a restart
  would have failed in connect until it was replugged. `_connect_cameras_again`
  now sets aside a camera that still sends nothing after its retries, unless
  it is the overhead camera or one the sentry watches through, and the arm
  starts without it.

## stream.py

### `StreamMixin.stream_joints` (also: `_stream_watch_stall`, `UNBLOCK_DEG`)

- **2026-10-26** — The wrist roll pushed into whatever stops it near -95 deg
  until it overloaded and dropped off the bus (as on 10-15).

### `StreamMixin.STREAM_CEILING_MARGIN_M`

- **2026-10-26** — A stream held high ended the session at 0.417 m.

### `StreamMixin._stream_watch_stall`

- **2026-10-26** — Comparing a goal that had just jumped with the last second's
  stillness took a shoulder lifted briskly off its rest for a blocked one, and
  held it down for the rest of the run.

### `StreamMixin._stream_refusal`

- Refusing every goal outside the bounds left an arm that started outside (the
  overhead parking pose is behind the workspace) stuck where it was.

## demos.py

### module

- The overhead camera moved twice in a week — why a learned policy is not
  given it.

## robot_config.py

### `OVERHEAD_CAMERA_W` (also: `OVERHEAD_CAMERA_H`, `OVERHEAD_CAMERA_FPS`)

- An earlier probe against the wrong index wrongly suggested 640x480 was broken
  on this camera; it isn't.
- The one scan that ran at 640x480 failed to see a board that the same camera,
  same pose, same light had resolved 15 minutes earlier.
  calibrate_overhead_camera.py had always forced 1280x720.

### `OVERHEAD_HOMOGRAPHY`

- **2026-10-20** — Fitted with no touches, by tools/tag_tour.py --calibrate (run
  20261020_120552), after the camera moved to the far side of the board: the
  board's lattice in a clean frame gives camera -> board, the gripper tag over
  24 hovers gives board -> arm (the arm's kinematics agree with it to ~1 mm).
  0.83 mm mean, 2.55 mm max over the 49 corners. The fit:

  ```
  [[-0.00014362025891703875, 0.0012067225110859309, 0.0493532270873168],
   [0.0013942090754752042, 0.00012177166858961595, -0.4522776196372171],
   [2.518395154436265e-06, 0.0004794682238674929, 0.9999999999999999]]
  ```
- **2026-10-22** — Re-fitted (run 20261022_095819, 23 hovers): the camera had
  moved overnight — the whole frame, arm base and curtain too, ~40 px — and
  every square read 25-35 mm off in the arm frame, which a pick on d6 then spent
  22-26 mm of hover correction on. 0.90 mm mean, 3.00 max over the corners.
  The value in robot_config.py.

### `GRIPPER_CAM_GRASP_PX`

- **2026-10-19** — (334.8, 251.8), measured on a rook at (0.22, 0.05).
- **2026-10-22** — Re-measured at 12:43 (calibrate_gripper_servo.py --height
  0.07), because the servo had been 11-24 mm off all morning, always toward the
  base: the overhead camera had moved overnight, and the gripper was handled
  for grip pads (fitted and taken off). The value in robot_config.py.

### `GRIPPER_CAM_JACOBIAN`

- **2026-10-19** — Kept when the grasp pixel was re-measured: that run's probes
  fitted 1.22 mm/px, physically wrong for this view (~0.3) — the rook's outline
  changes between viewpoints 12 mm apart. The Jacobian kept was from the probes
  at the 70 mm hover, after the joint offsets; its u column the less certain
  (one x probe dipped 6 mm), which costs ~15 px of prediction on a correction
  the servo looks at again anyway. It matched the camera's geometry within
  5-10 deg and 10%.
- **2026-10-22** — Re-measured with the grasp pixel; the pawn tracked cleanly
  on all four probes (0.34 mm/px at 71 mm). Within 10-20% of the one before:
  `[[7.805e-05, -2.826e-04], [3.409e-04, 1.352e-04]]`.

### `DESCEND_UNCONFIRMED`

- **2026-11-16** — True, for a demo: a pick with no camera to confirm the jaws
  comes down at the scan's position instead of retrying the same look. The d1
  queen, beside the base, was never picked: its foot was hidden from the tilted
  wrist camera, and the gripper's tag was off the top of the overhead camera's
  frame. The reasons are in `mrcf/policies/HISTORY.md`, `unconfirmed_descent`.

### `HEIGHT_FROM_VIEWS` (also: `HEIGHT_LOOKS_MAX`, `APPROACH_LOOK`)

- **2026-11-16** — True, after the tabletop's first run on this arm: with no
  height measured, every pick planned for the tallest object there could be
  (150 mm) and closed 117 mm up over 25 mm balls. From the hover the wrist
  camera saw 0.21 px per mm of a ball's height; from 6-12 cm beside it,
  leaning in, 0.9-1.5. Built and tried offline only. The reasons are in
  `mrcf/policies/HISTORY.md`, `parallel_jaw/height_looks.py`.

### `BACK_OFF_TO_ATTEMPT_HOVER`

- **2026-11-16** — True: on the tabletop, 32 cm out, backing off to the
  plan's 165 mm hover was out of reach, and ground on 24 s each retry, 17 mm
  short. Offline only so far. The reasons are in `mrcf/policies/HISTORY.md`,
  `_back_off`.

### `GRIPPER_CAM_REF_HEIGHT_M`

- **2026-10-22** — 0.0696 before this day, 0.0709 since.

### `GRIPPER_CAM_REF_TILT_DEG`

- **2026-10-19** — -5.2, measured with the grasp pixel.
- **2026-10-22** — -5.1, measured with the grasp pixel.
- Before the servo stepped for the difference (SO101Env._servo_tilt_offset) it
  aimed as calibrated at every tilt, and every logged success was near the
  calibration point.

### `SERVO_POINT_HEIGHT_FRACTION` (also: `GRIPPER_CAM_REF_POINT_HEIGHT_M`)

- **2026-10-22** — The calibration used the cream pawn (40 mm); before, the
  black rook (54 mm).

### `ARM_FRAME_SCALE`

- **2026-10-17** — It was 1.0979 — the arm's frame read ~9% short — until
  JOINT_OFFSETS_DEG corrected the joint readings: the 40 mm board then measures
  40.0 mm in the arm frame, so the shortfall was the joint zeros all along.

### `OVERHEAD_CAM_MODEL`

- **2026-10-17** — Lens fitted from one photo of the board
  (calibrate_arm_offsets.py, 0.17 px RMS).
- **2026-10-20** — The pose that day:
  `[[-0.103415, 0.970139, -0.219400, 0.378503], [0.994556, 0.103705, -0.010224, 0.026110], [0.012834, -0.219262, -0.975582, 0.533888]]`.
- The lens was still right from the camera's new place: it reprojected the
  board's lattice to 0.24 px, and freeing the focal length in a joint fit with
  the gripper tag moved it under 1%.
- **2026-10-22** — Pose re-fitted with tools/tag_tour.py --calibrate, as
  OVERHEAD_HOMOGRAPHY (the camera had moved; the lens still reprojects the
  lattice to 0.22 px).

### `GRIPPER_CAM_POSE` (also: `GRIPPER_CAM_POSITION_M`)

- **2026-10-15** — The one fit, before JOINT_OFFSETS_DEG, reprojected to
  ~10.6 px, not ~1 — the uncorrected joint readings were not consistent enough
  for one fixed mount — and in use put a knight's foot 31 mm from where it
  stood; so it is not set. GRIPPER_CAM_POSITION_M comes from this fit (76 mm
  along -x, as an overhead photo had it).
- **2026-11-03** — Re-run with JOINT_OFFSETS_DEG in place (capture
  rig/so101/gripper_pose_frames/20261103_101927, 70 poses, board clear): the
  mount alone still reprojects to 8.2 px, and the error is systematic, not
  noise. Each view's own grid fits, and cleaning the frames changed nothing;
  freeing a correction per joint takes it to 2.2 px, on views the fit did not
  see as well. The corrections, on top of JOINT_OFFSETS_DEG: shoulder_pan
  +1.2, shoulder_lift +1.0, elbow_flex +4.0, wrist_flex -0.8 deg (wrist_roll
  cannot be told from the mount). The elbow is the joint the 10-17 touches
  could not pin down. The overhead camera's board sat 7 mm from the arm's.
  A simulation with those errors gives the same 8 px, and its mount-only
  answer is 19 mm and 4 deg out.
- **2026-11-03** — `--wide` (20261103_104117, 90 poses at 8-12 cm, near the
  base to rank 7). Its own fit failed (18 px). The first guess from views that
  close to the board pairs corners with the wrong neighbours. Started from
  the morning's answer it fits at 3.5 px. Both captures together, 148 views:
  3.1 px on views the fit did not see, with shoulder_pan +1.19,
  shoulder_lift +0.97, elbow_flex +3.41 and wrist_flex +0.11 deg. The
  overhead camera's board was 8 mm from the arm's in both. So one set of
  corrections holds from 8 to 17 cm over the whole board. The tool now also
  starts from the last good calibration. Saved in 20261103_104117/result.json.
  Not set in robot_config: it waits for the cross-camera matching.
- **2026-11-03** — Set, with the matching. Live the same day: e2, d2 and b1
  each gripped at the first attempt (the policies' HISTORY.md, FootAligner).

### `GRIPPER_CAM_SCAN_SHIFT_AT`

- **2026-11-03 17:10** — The overhead camera was re-aimed and the board moved
  2 cm; the tag tour re-measured both. The shift went stale: the d1 queen, c1
  and e1 were rejected as 6-13 mm from where the scan had them. Re-fitted from
  the tour's 26 wrist frames against the 17:11 scan (64 feet, median spread
  3.4 mm): (-8.3, +2.9) became (-6.5, -3.2) mm. The next pick found the
  queen's foot 2.1 mm from the scan.

- **2026-11-03** — The overhead camera's self-check re-posed it at 12:23,
  between two of the day's picks, 1.5 mm and 0.18 deg from where it was at the
  wrist camera's calibration: the same pose measured again. The warning, then
  on any re-pose, said to redo the calibration. Why it warns only past
  overhead_calibration's MOVED_MM / TURNED_DEG.

### `GRIPPER_TAG_ID`

- **2026-10-20** — The 24 mm tag went on the arm, the biggest that fitted.

### `GRIPPER_TAG_POSE` (also: `GRIPPER_TAG_READS_AS_M`)

- **2026-10-20** — Measured by tools/tag_tour.py (run 20261020_131056, after
  re-sticking the tag with nothing over its face: 26 hovers, 1.27 px; with the
  arm's rotation, its position agreed with the arm to 0.8 mm median, 2.7 max
  at held-out squares). It read 23.1 mm through the overhead camera, printed
  24.0 (ruler); it read the same with clear tape over the tag and without. The
  fit, reading 0.02308:

  ```
  [[0.248521, 0.757107, -0.604174, -0.069614],
   [0.968368, -0.179799, 0.173017, 0.019807],
   [0.022362, -0.628062, -0.777842, -0.053146]]
  ```
- **2026-10-22** — Re-fitted with the camera's new pose (run 20261022_095819,
  23 hovers, 1.68 px): 5 mm further along the gripper's y and its face turned a
  few degrees from 2026-10-20's — grip pads went on the jaws that morning, by
  hand, close to the tag. The value in robot_config.py.

### `CALIBRATION_PARK_DEGREES`

- **2026-10-17** — Measured: folded back below the board's near edge, placed by
  hand.
- **2026-10-20** — Still clear with the camera on the board's far side: scans
  from there find all 64 squares.

### `OBSERVE_DEGREES`

- An unconstrained IK solution for the observe pose asked for an angle outside
  the arm's calibrated range (e.g. shoulder_lift beyond ±57°), the servo
  stalled against its own limit instead of reaching a pose that could see the
  table, and that is exactly what caused discover_scene() to find 0 objects.

### `TABLE_Z`

- **2026-10-17** — Setup moved: taken from calibrate_arm_offsets.py's fit, the
  board's surface level in the corrected arm frame. It was 0.000 before the
  move, in the uncorrected frame.

### `JOINT_OFFSETS_DEG`

- **2026-10-17** — Fitted by tools/calibrate_arm_offsets.py from 9 closed-jaw
  touches on the board's corners.

### `IK_CALIBRATED_RANGE_JOINTS`

- **2026-10-21** — Allowed by the user ("try pushing the wrist to 101"), after
  the check that nothing on the gripper comes near the forearm at full bend.

### `GRIPPER_WIDTH_TABLE` (also: `GRIPPER_MAX_OPEN_M`)

- **2026-10-16** — Measured with calibrate_gripper_width.py (gap at the jaw
  ENDS, by ruler). Until then GRIPPER_MAX_OPEN_M was the opening at 100% of a
  straight-line map. The straight line it replaced put 100% at 45 mm. In fact
  69.5% is 100 mm, so the "30 mm" every approach asked for was ~96 mm — the
  jaws nearly wide open (user: "why is the jaw approaching fully open?"), with
  the close taking most of the pick's time and the open fingers sweeping the
  neighbours. Every width in robot_config.py is real millimetres from here on.

### `JAW_CENTRE_SLIDE` (also: `JAW_SLIDE_AXIS`, `JAW_CLOSED_POINT_M`)

- **2026-10-17** — 1.04 along -x.
- **2026-10-20** — Experiments slid the model along x (1.04, 0, 0.5); each saw
  the servo move the jaws back by exactly the slide: along x the true slide is
  zero. 0.0 along -x from then to 2026-10-21.
- **2026-10-21** — Along y it was never modelled, and it showed: a pick aimed by
  the overhead camera alone (near_base on e2, aligned to 1.2 mm, jaws 39 mm)
  came down with the pawn against the FIXED jaw, the moving one ~20 mm away —
  half the opening — and closed on nothing. The same day showed the closed-jaw
  point is not where the model had it either: 0.5 along -y alone overshot — the
  pawn then sat against the MOVING jaw. All three sightings fit one geometry:
  the jaws close ~10 mm along +y (toward the fixed jaw) from the old closed
  point (env._TCP_OFFSET, now kinematics._TCP_OFFSET), and open from there at
  0.5 along -y:

  ```
  slide 0,   jaws 38.5 mm open: aim 10 mm toward the fixed jaw      (seen ~10)
  slide 0.5 from the old point: aim  9 mm toward the moving jaw     (seen ~11)
  a held 16 mm pawn placed at slide 0: off ~3 mm across the jaws    (seen  3)
  ```

  Read off the wrist camera's grasp frames (logs/grasp_frames/20261021_124011_*,
  20261021_125551_*) and the overhead scans: good to ~3 mm.
- **2026-10-28** — JAW_CLOSED_POINT_M's y +10 -> -2 mm. Every place since the
  gripper tag was re-fitted (2026-10-22, its y on the gripper 20 -> 14.4 mm)
  landed 10-14 mm toward the MOVING jaw: e7 -13.5, d4 -12.2, f7 -10.1, e4 -12
  (overhead scans, confirmed by eye in the frames) — whatever the piece (pawn,
  rook, queen) and the lean. The same offset showed at the pick: descents aimed
  by the overhead camera put the queen against the FIXED jaw (grasp frames
  20261027_111625, 20261028_093554_try1) and one closed on nothing. It moved by
  the places' mean, across the jaws only; along them they landed within ~3 mm.

### `JAW_AXIS`

- **2026-10-14** — The reach_map.py study (a king's top meets a jaw on 43 of 64
  squares with the jaws along the arm, on none across it) was run on estimated
  piece sizes, under the old names.
- **2026-10-15** — Radial (now the name for it; then called "tangential") was
  tried: with the roll at about -92 deg everywhere, the wrist roll overloaded
  at -96 deg mid-pick twice and dropped off the bus.
- **2026-10-21** — The NAMES were swapped until this day: this setting was
  "radial", which then meant "gripper x along the arm" on the belief that the
  jaws open along x. They open along y (JAW_SLIDE_AXIS), so "radial" held them
  across the arm while the policy checked fingers and the lean along it.
  Renaming it moved nothing: the same wrist roll, so no calibration taken at it
  needs redoing. Older notes that say "radial" for this orientation mean it.
- Tangential is the setting picks worked with (f40cd62).

### `GRIPPER_CAM_BODY_M`

- **2026-10-15** — With the jaws radial (then called "tangential") the wrist
  camera hung out sideways, and from e3 it reached the pieces on g3 and g4.

### `MIN_HOLD_WIDTH_M`

- This sat at 0.003 — which is EXACTLY the width the jaws report when they have
  closed on nothing at all, so whether a total miss got caught came down to
  sensor noise in the last tenth of a millimetre. Observed: a miss reported
  "[OK] pick: grasped object (width=3 mm)" and the world model then believed it
  was carrying a piece it had never touched.
- It then sat at 0.010 on the belief that the reading is true millimetres, so a
  ~20 mm piece would read ~20 mm. It is not: get_gripper_width is a linear map
  of the motor's 0-100% onto a hinged jaw, and HELD pieces read 2-6 mm. Every
  real grasp was therefore judged "fully closed — missed" and released before
  the lift, which is why a pick could fail three times without the pawn moving.
- The "3 mm floor" was set_gripper's stall detector firing early, not the jaws
  meeting. Since then it is read after squeeze_gripper(), which closes the jaws
  the rest of the way.
- **2026-10-22** — Grip pads went on: the pads now meet at 0.9-1.0 % (1.7-1.9 mm)
  and press each other at the torque cap — three empty closes read load 160-200
  of 200, a grip by load alone. Since then it is half of the grip signal, with
  the servo's load.

### `HOVER_HEIGHT` (also: `LIFT_HEIGHT`)

- The previous 0.08 / 0.10 were both past the ceiling of a vertical grasp, so
  every hover and lift stalled ~1.9 cm short with wrist_flex pinned.

### `PREGRASP_OPEN_M`

- It was 0.030 on the old straight-line map, really ~96 mm.

### `GRASP_CLOSE_STEP_PCT`

- On the old straight-line map 1.5 implied ~13 mm/s near closed; on
  GRIPPER_WIDTH_TABLE it is ~40-60 mm/s — and it has gripped fine.

### `GRASP_HEIGHT_FRACTION` (also: `GRASP_FLAT_MM`)

- Before 2026-10-21: 0.45, the pawn's rig-tuned 20 mm of its ~43.
- **2026-10-21** — 0.6, the user's call: "you can pick up the piece at a higher
  point". At 0.45 a 41 mm pawn was grasped at 18 mm, the jaw tips down at 11 mm
  on its flared base, and the close squeezed it out toward the arm's base on
  every try at d2 — even with the jaws 1.2 mm from it. At 0.6 (~25 mm, the tips
  above the flare, the pads on the straight neck): 4 of 4 first-attempt grips
  there, one 1.1 mm off. Leaning 20-30 deg (d1, d7) it still slipped: see the
  grip pads note in that day's notes.
- **2026-10-27** — A PREFERENCE, not the rule: what mattered at 0.6 was that the
  pads closed on the pawn's straight neck, not its flare, so the policy now
  grasps where the measured profile is straight (GRASP_FLAT_MM). Every pawn
  grasp logged since 10-14 lands within 1 mm of where 0.6 put it; a king's
  collar ridge at 0.6 of its height moves down onto its neck.

### `NEAR_BASE_M` (also: `FAR_REACH_M`)

- **2026-10-21** — The offline reach study behind both distances.
- **2026-10-21** — On the arm, e2 (0.16 m, so "near"): near_base, aligned by the
  overhead camera alone, closed on nothing on every attempt — the scan's foot
  was ~9 mm off along the arm — while top_down, which reached e2 at -5 deg,
  gripped on every retry. So near the base top_down goes first where it
  reaches.

### `SERVO_TOL_PX`

- At 5 px (visual_servo_xy's own default) a servo that had closed a 10 mm
  error to 3 mm reported failure, and the pick then descended at the
  uncorrected position. 15 px is ~4 mm at 0.29 mm/px.
- **2026-10-16** — 8 since: aligned to 13.5 px the jaws closed on nothing (they
  stopped within 2 mm of the aim — the aim was off); at 4.5 px the next attempt
  gripped. The failure the old note describes is gone: a servo that does not
  converge now retries the approach instead of descending uncorrected.

### `GRASP_TOL_M`

- **2026-10-16** — 8 mm let an aligned pick stop 4 mm short — most of the slack
  either side of a piece.

### `MAX_SERVO_SHIFT_M`

- **2026-10-16** — The scan sat 20-26 mm short of the arm near the board's
  middle, and 20 mm refused every aligned pick there. It was 45 mm while that
  scan error was ~37 mm.
- **2026-10-17** — Back to 35 mm after the recalibration.

### `OVERHEAD_FIRST` (also: `SERVO_AGREE_M`)

- **2026-10-30** — A white pawn moved square to square by the arm, 10 picks:
  4 gripped at the first attempt. Five closed on nothing with the overhead
  camera putting the jaws within 2 mm of where the pawn was remembered; every
  retry, aligned by the wrist camera, found it 9-15 mm away (+7 to +13 mm
  along the arm, -6 to -9 across) and gripped it. The wrist camera's frames
  before each close showed the pawn beyond the fingertips. With the wrist
  servo on the first attempt, the 10 mm SERVO_AGREE_M would still have sent
  the d4 pick, 9.1 mm off, to the remembered spot. Incident:
  `runs/flight/20261030_094816/incidents/094907_pick_pick_took_2_attempts/`.

### `GRIPPER_HOLD_PCT`

- **2026-10-30** — Held with the goal 3 % past the jaws, a fraction of the
  squeeze's pressure, pawns came out of the grip three times in two days: on
  a swing from the park pose (10-29), in a lift (e3), and 0.2-0.4 s into a
  carry (d4, the gripper's reading falling from 8.0 to 4.1 %). Every pawn that
  held was gripped at the same spot, 16 mm across the neck, so where it gripped
  did not tell them apart. Incident:
  `runs/flight/20261030_100652/incidents/100732_place_place_failed__the_object_was_no_longer_i/`.

### `GRIPPER_TORQUE_LIMIT`

- Closing on nothing at lerobot's 500, the gripper servo latched off and
  dropped off the bus until power-cycled — three times in one session.

### `WORKSPACE_BOUNDS`

- An earlier x_min of 0.24 came from assuming a vertical gripper and wrongly
  ruled out the near half of the board.

### `STREAM_JOINT_LIMITS_DEG`

- **2026-10-15** — The wrist roll overloaded and dropped off the bus mid-pick at
  -96 deg, twice.
- **2026-10-26** — Following the leader to -97 deg, it overloaded and dropped
  off the bus again.

### `MOTION_GUARD` (also: `MOTION_GUARD_MARGIN_M`, `MOTION_GUARD_ZONE`)

- Moved to robot_config.py from the chess deployment, which ran it only inside
  its controller.
- **2026-10-22** — The user's rule: "any movement with the arm must run the
  sentry".
- **2026-10-22** — A 0.55 m disc round the base took in the floor both sides,
  and all three of its first live stops were the user's feet. Since then it
  watches the table the arm reaches.

### `REMEASURE_KNOWN`

- **2026-10-29** — A pawn (42 mm, measured that way all afternoon) was stood on
  e4 by hand; the scan's object memory paired it with obj-115, a black piece
  last seen on e4 days before, and the pick took obj-115's 52 mm as measured:
  jaws 28 mm, gripped under the pawn's head. The jaws stopped at 16 mm on it,
  the squeeze drove on to 0 mm, and the pawn fell over. Incident:
  `runs/flight/20261029_170333/incidents/170446_pick_pick_failed__pick_failed_after_3_attempt/`.

### `motion._descend_by_joint_line` (also: `descent_line_above_m`, `DESCENT_LINE_ABOVE_M`)
2026-11-19: a held-orientation descent's upper part as one line in joint space
to the solver's own answer descent_line_above_m over the goal, settled onto;
the Cartesian loop the rest. At 32 cm and -20 deg the loop, re-steering from
joints lagging 2-3 deg under load, ran the shoulder past its goal while the
elbow trailed: the jaws went up and out (run 37) or the command dipped into
the table (run 39). Off (None) unless a deployment sets it (tidy: 0.025).
Tested on the real kinematics with deadbanded servos (test_so101_env).
