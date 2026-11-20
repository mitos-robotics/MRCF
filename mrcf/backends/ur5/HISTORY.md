# UR5 — history

Why the code is as it is: the runs behind its values and fixes, with their
dates, by file and by name. The comments in the code say what each value or
step is for; the story of how it was found is here. What is still wrong is in
NOTES.md, not here.

## The package

- 2026-11-02: built from `mrcf/backends/_template` before the arm was on the
  network, following DEPLOYMENT_SPEC.md, "Worked example: a UR5". The arm is a
  CB3 UR5 (made May 2016) on PolyScope 3.15, and it is the only UR5 this has
  been written for. Tested offline only.

### `present`

- **2026-11-16** — `python run.py` now asks which arm, and lists each as
  plugged in or not. The UR5's check opens a TCP connection to the RTDE port
  (30004, the one ur_rtde's interfaces use) and closes it at once, sending
  nothing: the controller does nothing for it, and an arm that answers there
  is one ur_rtde can reach. 0.5 s at most, so an arm powered off costs the
  menu half a second. From a process macOS keeps off the local network it is
  not found, and could not be driven from there either.

## robot_config.py

### `HOST`

- First a default in this file, a range picked on 2026-11-02 before the arm
  was on the network. On 2026-11-05 the pendant showed the arm already on a
  static address of its own, also clear of the rig's Wi-Fi, so the pendant
  was left alone. The address is the rig's, not the code's, so it moved out
  of git, into rig/ur5/host (MRCF_UR5_HOST still wins); with neither, the
  tools run the UR5 in memory and say why.

### `GRIPPER_BOX` (also: `OVERHEAD_CAM_MODEL`, `WORKSPACE_BOUNDS`, `RigEnv._refused_by_box`)

- **2026-11-20** — x out to 0.90 m (was 0.75). The toolbench's cardboard box
  stood at x 0.60-0.83 m, its far half outside the box, and the planner had
  it out of reach; the user: "just expand the safety box - it can definitely
  reach it". Pointing straight down the arm reaches 0.89 m out with the grasp
  point 5 cm up, 0.79 m at 30 cm, 0.67 m at 45 cm (kinematics.ik). The
  sentry's zone grows with it (to x 0.96 m). The baselines' standard scenes,
  placed from the box's middle, moved with it (tests/baselines/ur5/plans.json).

- 2026-11-12: the user, before any task: "really make it super safety
  compliant because this robot could destroy the things around it and
  itself". The controller's own safety settings were still the factory's (no
  planes, no orientation limit, 1.5 m/s, 150 N), and nothing here checked a
  move's path, only where it ended. With the overhead camera calibrated by
  hand (tools/calibrate_camera_by_hand.py: the wrist tag on link 5, 20 poses,
  0.56 px) the user chose a box on the camera's picture
  (tools/show_workspace.py) and the rule: the gripper stays inside it, the arm
  may pass outside. Every move is now checked along its way first: a joint
  move's sweep, a line's, a streamed step. WORKSPACE_BOUNDS is the same box.
  The pendant's planes are to match once its safety password is had.

### `CAMERAS` (also: `make_env`)

- The overhead camera had no source until one was mounted. On 2026-11-12 a USB
  webcam went up high over the table at a slant: OpenCV index 0 on the rig's
  Mac. Giving it a source meant every make_env() opened it, the in-memory UR5's
  too, so tests and offline runs reached for a real camera (none in CI; on
  macOS a process without camera access hangs). The offline UR5 now opens no
  cameras unless the caller passes sources=.

## kinematics.py

### `nearest`

- 2026-11-13 13:03-13:07: the blue block's pick failed in under a second,
  three tries, twice, and the planner planned again each time. From park
  (wrist 3 at 285 deg) the jaws had to turn 145 deg; the short way passes
  wrist 3's 360 deg limit, and that solution was dropped for one swinging
  the base 139 deg, which the gripper's box refused. Now a joint goes the
  other way round where the short way would pass its limit. Offline, the
  same pick then turns the wrist alone and comes down.

### `ik`

- The planar step (joints 2 and 3) first followed a published derivation
  whose frames put the arm in frame 1's x-z plane. With these DH frames the arm
  lies in x-y, and every solution failed the forward check. Now the standard
  two-link solution in x-y. 3000 random poses round-trip to 4e-12 m.

## env.py

### `UR5Env._transit_joints` (`TRANSIT_*`; also: `move_ee_to`)

- **2026-11-13** — The user asked why the wrist twist does not happen on the
  way. A move across turned the wrist in place, then ran a line at 5 cm/s
  (SPEED_M_S). It now goes as one joint move at joint speed, turning on the
  way, when it is 5 cm or more sideways with both ends 5 cm or more up, and
  only where the joint path keeps the jaws within 10° of down, the gripper in
  its box, and the grasp point no more than 1 cm below the lower end. A UR's
  jaws stay pointing down along a joint move whose ends both do (wrist 2
  alike). Else the line, as before; descents are lines. In memory at the rig's
  own speeds, a three-block tower took 63.3 s, then 39.4 s with this.

### `UR5Env.back_off` (also: `_run`'s `limit_n`, `BACK_OFF_SPEED_M_S`, `BACK_OFF_FORCE_N`)

- **2026-11-13** — After a pick stopped at 47.5 N with a finger on its block,
  the arm held there, pressing, until the power was cut (capabilities.BackOff).
  It now lifts straight up at 20 mm/s, the jaws as they are. A move's force is
  counted from its start, which here is mid-press, so the press letting go
  reads as a force as large as the press: the back-off stops only at the press
  plus 25 N. Not yet run on the arm.

### `_run`, `hold`, `set_gripper` (the 2026-11-12 evening safety review)

- **2026-11-12** — A review of the live path found: a move left by anything
  but a stop (Ctrl-C, an error) went on to its goal unwatched, the
  controller running it by itself — now stopped on the way out; no software
  force limit on home, park or a look's trip (the controller's factory 150 N
  only) — now every move stops over CONTACT_FORCE_LIMITS' max; a hold did not
  stop the fingers, which close at their own force with a hand at the jaws —
  now halted while set_gripper waits on them (never once they have stopped
  on an object: a halt is no grip), and sent on after the pause; every
  get_ee_pos asked the gripper its width under the arm's lock, though both
  fingers move and the point does not — now not asked (RigEnv._tcp); the
  box let a move from outside run along it — now only back in; and nothing
  kept the gripper off the arm's own base — GRIPPER_BASE_CLEAR_M.

### `UR5Env._settled_wrench` (also: `_run`'s force zero, `zero_contact_force`, robot_config `FORCE_SETTLE_S`)

- 2026-11-13 11:22: a pick's final descent onto a green block stopped at 31.9 N
  against the 30 N approach limit, touching nothing: it read 25-30 N from its
  first moment (the console's near-misses). The tool force in the flight
  recorder: the jaws had just turned 34° in place over the block, and the
  estimate swung from (+15, -12, -22) to (+10, +22, +5) N over 0.25 s after
  the turn — the CB3's estimate is from its joint currents, and depends on the
  pose. The descent's zero was one reading taken as it was sent, mid-change
  (-2, -3, -1). A zero is now the median of 0.1 s of readings, 0.35 s after
  the last move ended; tests/test_ur5.py replays those readings. Not yet run
  on the arm.

### `UR5Env._turn_in_place`

- Written on the guess that a UR runs a moveL's turn at the tool speed in
  rad/s, which would take 21 s for 60 deg at 0.05 m/s. URSim 3.15.8 timed it
  on 2026-11-02 (wall times over its ~1.7x emulation lag, a 10 cm line taking
  3.4 s for 2): a 60 deg turn in place took ~5 s at 0.05 m/s and ~1.5 s at
  0.25. That is about 4x the tool speed in rad/s, and a line that also turns
  takes as long as the turn. A wrist-3 joint move does it in ~2.6 s, so the
  turn in place stays: twice as fast, and every line holds its orientation, so
  a descent never twists. offline.py's moveL is timed the same way.

### `UR5Env.contact_wrench` (also: robot_config `CONTACT_FORCE_LIMITS`)

- First the force sensor read getActualTCPForce() as it was, against the
  framework's 5 N stop (sized for the SO-101, which reads 0, and a real force
  sensor). A CB3's estimate is off by several newtons at rest, so every skill
  would have stopped at once on the real arm; URSim and offline.py both read 0
  and could not show it. Found by reading the safety monitor's thresholds
  before the first session (2026-11-02). Now the change since each move
  began, with limits of the arm's own (env.contact_force_limits), which the
  safety monitor takes for any task that does not set its own. (First wired
  into the chess deployment's thresholds; moved into the framework once the
  user said the UR5 would not run chess.)
- 2026-11-13: the approach limit 30 -> 45 N, the user agreeing: with the
  zero settled (`_settled_wrench`), descents touching nothing still read
  24-27 N after a wrist turn. A finger landing on a block rose 19 -> 70 -> 140 N
  in 0.2 s (11-12), so 45 N stops it ~0.05 s after 30 N would have.

## arm.py

- 2026-11-12, on the arm: touching nothing, the change in the estimate swung
  35-74 N in check_motion's 3 cm lines and home/park moves (slider 25-33%),
  with the payload (1.1 kg) and its centre of gravity set by every connect
  (setPayload: the control script's setTcp() would otherwise leave an unset
  centre of gravity at the flange). At 25 N every skill would have stopped
  itself: the stop is now 100 N, the near-miss 80. The joints' current against
  the controller's model (current_error) sat within ~0.6 A in the same moves
  and 0.01 A at rest: the gentler sense to try next.

### `ROBOT_MODES`, `JOINT_MODES`

- Robot mode 4 was first labelled "powered on, brakes on". On the arm
  (2026-11-05) it sat in 4 with every joint still booting: 4 is UR's
  POWER_ON, the joints coming up; brakes-on is 5, IDLE. Relabelled, and the
  joints' own modes added so the tools can say which state each is in.

### `URArm.speed_scaling`

- First read getSpeedScaling(), which is the trajectory's actual scaling and
  reads 0 while nothing runs. Against URSim the slider read 0%. Now
  getTargetSpeedFraction(), the slider itself.

### `_ur_rtde`

- URSim fresh from Docker refused the control interface ("Failed to start RTDE
  data synchronization"). Until its safety setup is confirmed on the pendant,
  the controller answers RTDE's version request with a text message, and
  ur_rtde read that as version 0.0. The error now says to confirm the safety
  configuration.

### `URArm.reachable`, `URArm._wait_ready`

- The contract's thread check failed on URSim now and then: a move to where
  the arm already stood was refused as unreachable. The controller's own IK
  query (getInverseKinematicsHasSolution, with a qnear and ur_rtde's default
  1e-10 tolerances) said no to the pose the arm stood in on its first call in
  5 of 8 sessions, and yes when asked again; with 1e-4/1e-3 tolerances, or
  without a qnear, it said yes every time. reachable() now solves the joints
  locally (kinematics.py matched URSim to 0.0 mm) and asks the controller
  only whether the pose and those joints are inside its safety limits, which
  answered right 8 of 8 (in reach and out). connect() waits until the
  controller says the arm's own pose and joints are inside them.

### `URArm.move_ended`

- A move first counted as ended once it had been seen moving and stopped, or
  had not started within 0.5 s. URSim (2026-11-02) showed the controller's own
  signal: getAsyncOperationProgress() alternates -1 and -2 as each move ends,
  for a stopped move and one to where the arm already is too. That is now the
  test, with the timing kept as a fallback (2 s).

## tools/ur5/check_connection.py

- 2026-11-05, first run on the real controller: the arm would not finish
  powering up, and the check said only "robot mode: powered on, brakes on".
  It now prints each joint's state and the arm's voltage while it powers up,
  tells "no joint has answered" (all their readings 0) from "they answer but
  have not finished booting", and prints the 48 V supply, flagged below
  46.5 V: this arm read 47.2-47.4 V on the days it worked and 45.0-45.8 V on
  the day it would not (NOTES.md).

## tools/ur5/check_motion.py

- 2026-11-12, first run on the real arm (every step passed): an answer had to
  be a bare Enter, s or q, and anything else asked again without saying why,
  so "go" looked like a hang. It now also takes go/y/yes, skip/n, quit/stop,
  and says what it takes when it gets something else.
- 2026-11-13: made the shared tools/check_motion.py, for any arm; this one
  now runs that with MRCF_ARM=ur5. Its Robotiq activation became the UR5
  package's prepare_gripper(ask), and the wrist step turns the jaws 20° about
  their own axis through move_ee_to (which turns them in place by a joint
  move) rather than driving wrist 3 by name. Without the gripper it now makes
  the arm with make_arm(gripper=None), which sets the payload, as the old
  bare URArm did not. Run offline end to end; not yet on the arm.

## gripper.py

### `RobotiqSocket._ask`

- 2026-11-12, the first connect with the Hand-E activated: it hung, then
  timed out, on the first SET. The URCap answers a GET with a line ("POS 3",
  newline) but a SET with a bare "ack" and no newline, and the reader waited
  for one. The tests' fake socket had answered "ack" with a newline — an
  assumption, never checked against the rig — so they passed. Both now as
  the rig does; a reply that does not come says which line it was for.

### `RobotiqSocket.settled` (also: `UR5Env.set_gripper`, `URArm.gripper_settled`)

- 2026-11-12, the first real pick of a block: set_gripper waited 20 ms, then
  took "not moving" for done; the Hand-E, slow to take a new position over the
  URCap's socket, still said its last move was done (OBJ 3), so the close read
  48 mm, no load: a miss, though the fingers then closed on the block. The
  retry lifted it, opened, dropped it, and came down on it where it had stood
  (pick.py, and the person at the e-stop). Done now means the gripper echoes
  the position asked for (PRE) and has stopped.

