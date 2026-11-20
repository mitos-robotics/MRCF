# Policies — history

Why the code is as it is: the runs on the rig behind its values and fixes,
with their dates, by file and by name. The comments in the code say what each
value or step is for; the story of how it was found is here. What is still
wrong is in the NOTES.md of the robot or task it is about, not here.

## parallel_jaw/pick.py

### `_wrist_at_grasp` (also: `gap_px`, `GripperSpec.wrist_at_grasp`)

- **2026-11-18** — The SO-101's table: the jaws came down ~1 cm to one side of
  a figure and of a plug, the overhead camera saying they were within 2 mm of
  their aim — its model of where the jaws are, from the gripper's tag. At the
  grasp the wrist camera sees the object and the jaw tips in one picture
  (user: "you need the arm camera for minor adjustment"): both carried onto
  the plane at the grip's height through the same camera, so its own placing
  errs alike for both, and the jaws reseated by the difference. The jaws'
  learned view (wrist_jaw_view.npz, 11-17 13:17) had gone stale by then — the
  moving jaw drawn ~50 px off, after the wrist camera was replugged several
  times — and is learned again before relying on it.

### `_diagnose` (also: `Vision.diagnose`, `Diagnosis`, `perception/grasp_diagnosis.py`, `_gripped_at`)

- **2026-11-18** — On the SO-101's table a close on nothing was taken for an
  aim that was off, every time: up, a look, down beside where it had been.
  Three closes over a figure lying on its side (the jaws over its 10 mm neck,
  16 mm up), and over a plug lying on its back gripped 21 mm up; each
  re-aimed, each pushed it further. The wrist camera's pictures showed why
  each time. Now asked of the vision model, its answer one of a few causes,
  and the next attempt changes that: lower, found again, or not picked as it
  stands. A deployment's catalogue gives the grip that held (Target.grip_at_m).

## parallel_jaw/height_looks.py

### `low_from_above` (also: `GripperSpec.grip_low_from_above`)

- **2026-11-18 16:43** — The SO-101's table: side looks took 25-36 s an
  object, and made a plug lying on its side 36-51 mm tall; gripped by that
  the jaws closed on nothing twice and pushed it away, where gripped as low
  as they go it had held. An object whose outline from above alone caps it
  short enough to grip at the jaws' lowest under the gripper's body is
  gripped so, with no side looks (off unless a deployment turns it on).

## base.py

### `Target.shape_looks`

- One look can be wrong: a 41 mm pawn read 19.5 and 59.6 mm on two picks, two
  frames agreeing each time. (also: GripperSpec.trust_shape_after,
  GraspPlanning.shape_unsettled)

## approaches.py

### `TOP_DOWN`

- **2026-10-21** — Near the base, top_down before near_base wherever it
  reaches: on the rig (e2) near_base closed on nothing every time, ~9 mm off
  along the arm where the scan put the foot, and the top_down retry gripped
  every time.

### `NEAR_BASE`

- **2026-10-21** — Its hover is the object's own: a pawn left 99 mm from the
  pan axis was out of reach at the 60 mm floor and in reach at the 56 it needs.
  (also: Approach.own_hover)

## aligners.py

### `unconfirmed_descent` (also: `GripperSpec.descend_unconfirmed`)

- **2026-11-16** — In a live game the arm could not make 2. Qa4: the queen on
  d1, beside the arm's base, was never picked. The wrist camera was tilted
  25 deg (straight down is out of reach there) and 3 cm above the crown. It saw
  the queen's body but not its foot, which ran off the bottom of the frame
  behind a jaw. The gripper's tag was off the top of the overhead camera's
  frame. Each pick looked three times from the same place (two picks, six
  looks), refused to descend blind, and the game asked for the move by hand.
  Incidents: `runs/flight/20261116_103341/incidents/103856_*` and `103921_*`.
  Over the 45 picks since 2026-11-03 where the wrist camera found the foot, its
  sighting moved the scan's aim by 2.0 mm (median), 4.4 mm at the 90th
  percentile, and 5.4 mm at most. At d1 the open jaws had ~4 mm to spare each
  side. The user wanted a pick that goes, for a demo. descend_unconfirmed
  (default False; True on the SO-101) comes down at the scan's position when
  no camera confirms the jaws, in all three aligners that refused: the foot,
  the servo and the fixed camera. It never does so after a close that may have
  moved the object.
  Verified offline only.

### `FootAligner`

- A fixed "object appears here" pixel could neither look for the object where
  it should be nor find its foot whatever its height: it locked onto a tall
  neighbour instead.
- **2026-11-03** — On a full chessboard, from a pick's hover, the e1 king stood
  in front of the e2 pawn and the wrist servo locked onto the king. With the
  camera located, the measurement nearest the expected pixel outlined the c1
  bishop from over b1. Now it is told who stands around the object
  (`known_around`, `perception/expected_view.py`), finds the foot without a
  height where the view cannot resolve one (`find_foot`), and decides who it
  is in the scan's frame. Replayed on a look tour, it found every piece within
  0.1-3.5 mm. It aims by the overhead camera's sighting moved by that, not by
  the camera's mount and jaw model. Those put the jaws ~11 mm from the
  hand-measured grasp pixel, and the joint corrections fitted with the mount
  are not the arm's own (the SO-101's NOTES.md). It falls back to the overhead camera's aim when it
  cannot find the foot, where it used to refuse to descend. On the arm the same
  day: three picks on the full board, each at the first attempt at -15 deg; the
  jaw model's aim would have been 7-20 mm off across.

### `OverheadFirstAligner`

- **2026-10-22** — With the fixed camera calibrated, it aimed every pick that
  gripped, while the wrist servo — calibrated on the 19th, the gripper handled
  since — was 4-17 mm off or lost the piece, and twice took the jaws to an
  empty spot 14-17 mm away. (also: GripperSpec.overhead_first)

### `WristServoAligner`

- "Aligned" from a 97 mm hover put the jaws 33 mm off a pawn: the camera looks
  at an angle, so from any height but the one its target pixel was set at the
  object shows up somewhere else. (also: GripperSpec.servo_height_m)

### `WristServoAligner.run`

- **2026-10-20** — The wrist camera loses pieces at the board's edge and on a
  square of their own colour, where the fixed one still sees both: why, when
  the wrist camera cannot align, the pick comes down where the fixed camera
  put the jaws. (also: GripperSpec.descend_on_overhead)
- Descending on the kinematics alone is what landed a jaw on a pawn's head and
  knocked pieces over; on the rig, uncorrected descents gripped 0 of 3 while
  aligned ones gripped 5 of 8.
- **2026-10-31** — d3, a pawn the arm had just put there: the servo led the
  jaws 13-15 mm out towards it at both hovers and gave up 13-16 px short; the
  pick came down at the remembered spot, where the fixed camera had put the
  jaws, and closed on nothing. The scan after found the pawn ~10 mm further
  out than it was put, where the servo had been heading. Why a servo that
  cannot settle well away from the object has it found again first.
- **2026-10-31** — c2, just scanned: the servo settled 11-13 mm outward of the
  pawn (+10 to +12 mm along the arm) on all three attempts, each followed as
  "well away from the scan", each a miss; the scans after found the pawn
  where the first had it. Why a scan decides any disagreement over
  servo_agree_m, settled or not (`_disputed`), and stands for the rest of the pick.

### `WristServoAligner._down_to_servo_height`

- An object already measured goes straight to the servo, as the pick that
  worked did. Re-measuring it from up here re-planned a 50 mm piece as 60.
- Taller than the servo's hover clears: what worked on the rig, for every
  piece, is aligning from there anyway.

### `WristServoAligner._check_where_it_aligned`

- **2026-10-21** — The servo's estimate held a bias near the base: on d2 it
  moved the jaws 4-8 mm outward and across, every time, from where a person
  then saw the pawn — and the close cammed it out toward the base. Hence a
  servo within servo_agree_m of the scan descends at the scan's estimate.
  (also: GripperSpec.servo_agree_m)
- A servo that locked onto a neighbour moved 43.6 mm and closed on one. But
  the scan and the arm disagree by 20-26 mm near the middle of this board, and
  the pick that worked trusted the servo. (also: GripperSpec.max_servo_shift_m)

## parallel_jaw/spec.py

### `GripperSpec.fast_descent_above_top_m` (also: `PickMixin._fast_to_above_top`)

- **2026-11-13** — The UR5's picks came down 15 cm at the gentle grasp step
  (1.8 cm/s), 8.6 s of every pick. With the object's height known, the descent
  now goes at full speed until the jaw tips are 15 mm above its top, then
  gently, under the approach's contact limit throughout. With the height
  unknown, it is planned as the tallest object there could be, and all of it
  stays gentle. Off by default (the SO-101 unchanged). In memory, the tower:
  39.4 s → 31.8 s.
- **2026-11-13**, its first run on the arm — First this was under the
  approach's 45 N limit too. A CB3's force estimate read 24-54 N on the way
  down at 5 cm/s, touching nothing, and stopped the pick of green with the
  tips 47 mm up, 17 mm above the block's top (from the incident's joints and
  wrench). The fast stretch is now a move like any other, under the arm's own
  limit. The gentle stretch starts from a settled zero under 45 N, as before.

### `ParallelJaw.face_turns` (also: `Target.faces_deg`, `_place_turn_deg`)

- **2026-11-12** — Built for the UR5's letter blocks, which the first scan
  found lying at 9-64° to the arm's axes: gripped as the arm happens to
  turn the jaws, a cube can be taken across its corners, wider than the
  jaws open. A block's sides are seen (plane_geometry.side_direction_deg)
  and the jaws turned to close across a pair of them, the least turn first;
  a place squares the held block to the one under it. Live: blue P at 27°
  (jaws turned -38°), green 7 at 28°.
- **2026-11-13** — "The least turn" was counted from the arm's own way at the
  target, not from where the jaws pointed. Putting blue (faces 66°) on red
  (73°), red's faces lay 44° from the arm's way there. The policy chose
  -43.6° (the wrist swung -87° in place, then -16°) over +46.4° (7° from
  where the jaws were). The gripper's cable wound round the wrist, and blue
  landed 9.2 mm off red; with green on top, the two fell off. Now the turns
  are ordered by how far the wrist turns from now, either way round
  (_wrist_turn; _hold's as_now chooses the way round). Replayed in memory
  from park: largest single wrist move 87° → 39°, the run's range from park
  142° → 74°, the turn home 114° → 24°. A tabletop test keeps the run.

### `PickMixin._start_attempt`: no blind retry after a miss (also: `GripperSpec.blind_retry_after_miss`)

- 2026-11-12, the UR5 (no wrist camera, no gripper tag): a close misread as a
  miss had held the block; the retry opened and dropped it, and the next
  attempt came down where it had stood, onto it — 100.9 N before the contact
  stop, and the person pressed the e-stop. After a close that may have moved
  the object, a pick that can neither look again with a wrist camera nor find
  it afresh (Vision.relocate) now stops and asks for a scan; the controller
  looks again after a failed step. blind_retry_after_miss keeps the old way for
  the tests of the retries themselves.

### `GripperSpec.stand_up_tip_clearance_m`

- **2026-10-28** — Jaw tips 3.6 mm up over a 13 mm pawn's head let the moving
  jaw squeeze it out beneath the fixed one; 1 mm held it. The same run as
  stand_up.py, `module`.

### `GripperSpec.stand_up_overshoot_m`

- **2026-10-28** — The fixed camera read a 13 mm head 16-17 mm, which alone
  takes 4 mm off the lift: why the lift errs high.

## parallel_jaw/planning.py

### `GraspPlanning.usable_widths`

- A 41 mm pawn's profile read 0.7 and 2.8 mm at 8-12 mm, then 31: an outline
  cut short, not the pawn.

### `GraspPlanning.plan` (also: `Target.min_height_m`, `Target.assumed_widths_mm`)

- **2026-11-03** — Live, full chess board: the d1 queen could not be measured
  from anywhere the wrist camera reaches (`PickMixin._look_around`). So she was
  planned as possibly the 82 mm king: gripped at 27 mm and counted as her
  27 mm foot all the way up, and refused. Every piece is 45-82 mm. Gripped at
  35 mm, as high as the shortest allows, and counted as 0.87 of her foot from
  the tips up (the deployment's 90th percentile over 138 measured pieces),
  she fitted before the arm moved. Picked first try in 15.8 s and put on f3
  1.7 mm from its centre (17:44).

### `GraspPlanning.height_known`

- **2026-11-11** — Live, near-full chess board: the f1 bishop, named as one,
  was still gripped by the shortest-piece rule — at 52 mm, 10 mm under its
  62 mm top, on the tapering mitre. With the shortest and tallest it could be
  agreeing, its height is known: planned from its kind's assumed shape as a
  measured one is, gripped at 37 mm (offline replay of the 17:16 refusal on
  the rig's kinematics).

### `GraspPlanning.grasp_height_for`

- **2026-10-21** — A 41 mm pawn gripped at 18 mm, on its flared base, was
  squeezed out toward the arm on every try at d2; at 25 mm, on its straight
  neck, it held 4 of 4. (also: GripperSpec.grasp_flat_mm)

## parallel_jaw/reach.py

### `ReachMixin.half_width_at` (also: `jaw_room`'s tips_m, `Obstacle.widths_mm`)

- **2026-11-03** — On the full chess board, live: "pick up d1" (the queen) was
  refused twice before moving. The jaws needed 35 mm open (its 27 mm foot plus
  slack), and the c1 bishop's 24 mm foot left 27. Every neighbour counted as
  its foot all the way up. Yet the finger tips come down no lower than 34 mm
  for a measured queen, and from there up the profiles of earlier picks make
  a queen 22 mm across (30 mm needed) and a bishop 19 (33 mm left). Even a
  measured queen was refused that way.

### `ReachMixin._pick_reach`

- A grasp the servo moved 10 mm out was past what the held tilt could reach:
  the descent stalled 19 mm short of it. (also: GripperSpec.servo_reach_margin_m)

### `ReachMixin._back_off`

- **2026-10-21** — Opening the jaws beside the object and moving off to the
  next attempt's hover swept the fingers through it at its own height: three
  missed closes on d2 left a pawn 19.5 mm nearer the arm's base, as three on
  e2 had left one ~30 mm.
- A pick that failed left the jaws there for whatever moved next.

### `ReachMixin._move_again_if_short`

- **2026-10-28** — From home the arm stopped 4.4 and 5.6 cm short of hovers
  over c5 and d5 that the next attempts reached from there in 1.6 s — each
  costing an attempt and a change of approach.

### `ReachMixin._stalled_at_start` (also: `PickMixin._lean_down_again`, the place's descent)

- **2026-11-04** — A pick on e4, mid-board: the descent at -15 deg stalled
  before it moved (7.3 of 7.2 cm short). Leaning further, the retry reached at
  -25, and the close squeezed the pawn out of the jaws; it landed 10 cm away.
  Leaning further is for the far squares, where the arm stalls near the bottom.

### `ReachMixin._move_through`

- **2026-10-28** — Each settle held the arm still for most of a second, at
  every waypoint.

## parallel_jaw/sighting.py

### `SightingMixin.wrist_sending` (also: `WRIST_STALE_S`, `_wrist_frame`)

- **2026-11-14** — Mid-game, the wrist camera dropped off USB and would not
  open again, and both tries of 3. Nc4 raised "OpenCVCamera(1) has not
  captured any frames yet" out of the pick, though the overhead camera had the
  jaws over the knight within 3 mm. A camera that has sent nothing for
  WRIST_STALE_S no longer counts as located (camera_located, _servo_ready),
  so the pick aims by the overhead camera alone, as it did before the wrist
  camera was located. A frame that does not come mid-look is a look that
  found nothing. Tests: SilentWristArm in test_pick_paths.py.

### `SightingMixin.measure_neighbours`

- **2026-11-03** — Replayed on the morning's look tour (real hover frames, real
  piece_vision measure, no arm), it measured 5 of 9 rank-2 neighbours along
  the jaws: 39-46 mm tall, 15-16 mm across from 32 mm up, feet within 1 mm of
  the scan's. The others came back "cannot tell its height" (seen from nearly
  straight above), "a part of it" (an outline 10-15 mm off its foot), or "the
  gripper hides part of it". Each is kept as its foot, so the pick is refused.

### `SightingMixin._visual_servo`

- Servoing at 36 mm against a 30 mm calibration put the aim 3 mm past a piece,
  and the narrower jaws that came down then caught its crown with the fixed one
  and knocked it over. The pick that worked servoed at the calibration opening
  always. (also: GripperSpec.servo_open_m)

### `SightingMixin._overhead_at_grasp`

- **2026-10-21** — What was left of the hover's correction plus what the
  joints do under load on the way down added up to 4-6 mm on d2 — with the
  pawn that much toward the base of the fingers' middle, the closing jaw
  cammed it out toward the base, 2 cm, every time. Seen from the side, held
  there with the jaws open, the pawn was where the camera said: "slightly
  closer to the base, the fixed jaw very close but not touching". So the jaws
  are put on their aim at the grasp. (also: GripperSpec.grasp_settle_tol_m)
- **2026-10-22** — At its limits the arm stalls and lurches: on d6 three
  stalled 3 mm corrections left the jaws 8.5 mm off where they had been 3.4.
  Hence the correction stops at the first one it cannot make.
- **2026-10-28** — Every queen close 2.5-2.7 mm toward the base missed, three
  of three, and those within 2 mm held, seen frame by frame in the flight
  recorder. (also: GripperSpec.grasp_settle_along_m)
- The arm stops short of a small correction (d7: asked 6 mm, went ~3): why the
  correction is asked for again, up to overhead_max_iters times.
- A small correction that stalls no longer carries the arm on: move_ee_to
  holds it.

### `SightingMixin._other_object`

- One look measured 54 mm over a 41 mm pawn, and the jaws then closed on the
  neighbour.

### `SightingMixin._unlike_scan`

- **2026-10-28** — Leaning 40 deg over a rook on f7, both frames read 29 mm
  tall on an 8 mm foot where the scan had 18 mm; gripped at 17 mm, low on a
  top-heavy piece, it fell over when set down. (also:
  GripperSpec.measure_foot_ratio)

### `SightingMixin._fit_measured`

- **2026-10-19** — Narrowed to 31 mm after a 35 mm alignment, the jaws closed
  on nothing.

## parallel_jaw/pick.py

### `PickMixin.pick`

- Before, a pick nothing could reach went ahead with each move choosing its own
  orientation, and failed partway.
- The log had only free text to go on for why an attempt was retried: hence the
  short code in Outcome.attempts.

### `PickMixin._start_attempt`

- **2026-10-21** — After a miss, near_base's second try on d2 came down on the
  spot the first had pushed the pawn off.
- **2026-10-31** — d3: found again after a top_down miss, the retry took the
  next approach along, near_base, and missed at the fresh position too. The
  grasp correction (`tools/calibrate_grasp.py`) was measured that day on d3
  with the approach the first attempt used: the same within its 4 mm step as
  on e2 and e4. Why a retry at a position found afresh comes the same way.
  (also: `_Choices.named`)

### `PickMixin._room_by_looking` (also: `_settle_room`, `GripperSpec.slimmest_above_foot`, `Refuse`)

- **2026-11-03** — For the d1 refusal above. A pick its neighbours' feet crowd
  goes over the piece and measures it and them, then decides; it is refused
  before moving only where the feet would leave no room even if everything
  were half as wide above them. On the 13:39 scan of the full board (the real
  SO-101 policy, no arm) that turns 12 of 32 pieces from refused into "looks
  first": c1, d1, e1, c2, b7, c7 and b8-g8.

### `PickMixin._look_around` (also: `GripperSpec.look_aside_m`, `look_around_max`)

- **2026-11-03** — Live, full board, d1 queen. From over her the wrist camera
  could tell neither her height nor the c1 bishop's (a part of it, 13 mm off).
  Looking from the spots beside her measured c1 (60 mm). But the camera sits
  on the arm's side of the jaws and looks outward: from beside her the back
  rank lies along the bottom of its frame, cut off by the jaws, and she was
  either not outlined or read 28 mm tall. Turning the wrist half a turn, to
  face the camera back at her, is beyond this wrist's ±80 deg. So the arm
  cannot measure a tall back-rank piece from where it reaches;
  `plane_geometry.cut_off` now keeps a height cut off at the frame's edge from
  being trusted.

### `PickMixin._found_again`

- **2026-10-30** — Two zig-zags across the board stopped at the same pick, d3
  at a lean of -5 to -10: the first close missed and pushed the pawn 16-27 mm
  toward the base; the retries looked for it from the old spot with the wrist
  camera, chased it 42 mm and lost it. The grasp calibration the same day had
  found pawns pushed 2-14 mm by a miss, and a scan finding them again every
  time. (also: `Vision.relocate`, the chess deployment's `_relocate_piece`)

### `PickMixin._third_look`

- **2026-10-30** — On d3 the coarse look from high up read a 39 mm pawn as
  17 mm; the look at the servo's height had the gripper over the pawn's top
  and refused; the pick went by the 17 and closed on the pawn's foot.

### `PickMixin._hover`

- **2026-10-20** — Near the wrist's limit the servos stop 1-2 cm short of where
  they are sent, and every such attempt was thrown away; a hover that stopped
  short now goes on once the fixed camera has put the jaws over the object.
- The proximity test alone passed a hover that had stalled 1.3 cm out, because
  its tolerance (2 x approach_tol_m = 3 cm) is looser than the miss — and
  descending from 1.3 cm off puts a jaw on top of the object instead of beside
  it, which is what sent pieces flying rather than grasping. A stalled hover
  passed on proximity alone put a jaw on the object.

### `PickMixin._descend`

- **2026-10-19** — 23 mm sideways at 0.28 m is a plane 5 deg round, and holding
  the old orientation the descent stalled 4-5 cm short, five times running.

### `PickMixin._aim_at_anvil`

- **2026-10-28** — Centred in jaws opened wide for the camera's aim, the moving
  jaw pushed a piece (open - width) / 2 across the board before it was held:
  7 mm for a queen, 11 for a rook, which tipped and slid out under the fixed
  jaw — two misses running on d3, and a queen dropped on the way to e3.
  (also: GripperSpec.anvil_margin_m)

### `PickMixin._lean_down_again`

- **2026-10-22** — Leaning further is what got a place down on e7, where the
  model's -35 deg stalled 7.7 cm above the release and -45 reached it (see
  parallel_jaw/place.py, `PlaceMixin.place`).
- **2026-10-27** — Far out the real arm reaches less than its model: rank 7
  picks (f7 at 0.38 m) reached the hover at -40 deg and stalled 6-12 cm above
  the grasp, and a stall ended the attempt. (also:
  GripperSpec.descent_lean_retries, GripperSpec.descent_lean_step_deg)

### `PickMixin._lift`

- Aiming the lift at target_xy dragged the gripped object sideways across the
  surface while rising.

## parallel_jaw/place.py

### Not coming down where something moved (also: `PickMixin._descend`, `PickContext.disturbed`)

- **2026-11-12** — A pick or a place no longer makes its final descent where
  the sentry saw something move since it set out: a pick looks again
  (Vision.relocate: chess) or, with nothing to look with, stops and asks for
  a scan; a place keeps hold of the object.

### `PlaceMixin.place`'s descents (also: `PickMixin._descend`, `_lean_down_again`, `safety.descend_onto`)

- **2026-11-12** — Only the place's first descent ran under the arm's
  approach limit; its retried descents (after a stall, at a further lean)
  ran at the 100 N limit, as the pick's did whose 100.9 N press on a block
  ended in the e-stop. Every descent of either now goes through
  safety.descend_onto, counted from its own start.

### `PlaceMixin.place`'s touchdown (also: `GripperSpec.touchdown_m`)

- **2026-11-04** — Taking back 1...e5: a pawn picked at -20 deg was carried to
  e7 at -40 (the least that reaches there). 8 mm above the release height the
  jaws closed from 16 to 10 mm; the slip watch stopped the arm holding it, and
  the pawn went with the arm and stood on e6. The wrist frames show it between
  the jaws just over the board, then gone. At a 40 deg lean a pawn's foot edge
  meets the board ~7 mm before its centre would: it touched down and pivoted
  between the pads.

### `PlaceMixin._watching_grip` (also: `GripperSpec.slip_m`)

- **2026-10-30** — A pawn slipped out of the jaws 0.2-0.4 s into the carry
  from d4 to e3: the gripper's reading fell from 8.0 % to 4.1 % right then.
  The place carried on over e3, came down, and found the jaws empty at the
  release 9 s later. Incident:
  `runs/flight/20261030_100652/incidents/100732_place_place_failed__the_object_was_no_longer_i/`.

### `PlaceMixin.place`

- **2026-10-20** — Picked at 5 deg and set down at 15-20 a pawn landed 7-9 mm
  further out; picked at 15 and set down at 10, short: why a place leans as
  the object was picked.
- **2026-10-22** — Far out the arm reaches less than its model: on e7 a -35 deg
  descent the model approved stalled 7.7 cm above the release.
- **2026-10-28** — On the way to e3 a queen fell out of the jaws and rolled off
  the board; the place went on, opened them on nothing, and called it placed
  on e3.
- **2026-10-28** — Nothing checked the descent, and leaning 20 deg one drifted
  out to land a queen 16 mm past the centre of d5.
- **2026-10-28** — One read loaded at 35 mm round a 21 mm queen; the place
  refused to retreat, and the next move swept the open jaws through the queen.
  (also: _LOAD_SETTLE_S)
- Without the fixed camera's correction a place had nothing to correct it: one
  on e5 landed 27 mm short, the arm's error there.

## parallel_jaw/reach.py: `_back_off` (also: `GripperSpec.back_off_to_attempt_hover`)

- **2026-11-16** — The SO-101 tabletop, a block 32 cm out: after each miss the
  arm backed straight up to the plan's hover, 165 mm, clear of the tallest
  object there could be, while the attempt had come down from 62 mm. Out of
  reach that far out, each back-off ground on 24 s and stopped 17 mm short:
  half of a 100 s pick. With the switch (on for the SO-101), no higher than the
  hover the attempt reached.

## parallel_jaw/height_looks.py

### `module` (also: `GripperSpec.height_from_views`, `PickMixin.pick`)

- **2026-11-16** — The tabletop's first run on the SO-101 ("pick and place all
  these small objects into the black bowl"). Five picks, none placed. Nothing had
  measured any height, so each pick planned for the tallest object there could be
  (TALLEST_OBJECT_M, 150 mm) and closed the jaws 117 mm up, over balls about
  25 mm across: closed on nothing, three times a pick. The ball by the base
  never moved, because a hover clear of 150 mm is out of reach there. The look at the hover
  could not help: offline, from the recorded joints, the wrist camera was 24.5 cm
  up and 8 deg from straight over the ball, where a mm of its height moves its
  top 0.21 px (measure() needs 0.8). From poses the arm reaches beside it,
  leaning in, the same camera sees 0.9-1.5 px a mm. So the height is looked for
  first, from there, and the pick is refused while it is not settled. Shown on
  the fake arm (tests/test_height_looks.py); the views planned on the SO-101's own
  kinematics, and the outline and fit on the run's recorded frames (one oblique
  frame: 22 mm, 16-26 fit as well). Not yet live.

### `_next_view`

- **2026-11-16** — Offline on the SO-101's kinematics: of the views scored best,
  the arm could hold none as planned. Its orientation search and down_rotation
  mean opposite things by flip (the orientation it held was the planned one
  turned 179 deg about the approach), and a camera off to one side then looks
  elsewhere: the view scored 0. And the best-scored leaned away from the base,
  which a 5-joint arm cannot. Now each position is asked of the arm, either sign
  and either way round, and the view from what it can hold is scored. Planning a
  look takes 1.3-2.5 s there, and finds three for the red ball (0.9-1.1 px a mm)
  and for the ball by the base (0.65-0.95).

### `look_obstacles`, `could_be_mm` (also: `Obstacle.outlines`, `ShapeFit.bound_mm`)

- **2026-11-16** — Neighbours not measured were first kept clear of as the arm's
  default 80 mm, and a sweep check skips an object of no height at all: both
  under the 150 mm a tabletop object may be. Counted as 150 mm, the jaws could
  look from nowhere low near the base (offline, 0 of 144 poses). Their outlines
  from above bound them: offline, the overhead camera's outline alone, at a 0.10
  drop, was above the truth for every object tried (25 mm ball, 40x30 box,
  70 and 140 mm bottles, a 150 mm pole), the ball by the base bounded at 48-60 mm;
  at 0.05 a 140 mm bottle read 132. Right under the camera it bounds nothing.

### `_look_clear`'s camera body

- **2026-11-16** — body_clearance goes by feet alone (made for a hover low
  among chess pieces), and refused 81 poses near the base whose jaw tips were
  well over the neighbour's top. The camera's body is above the tips: only what
  the tips do not clear counts.

### `_keep_look`

- **2026-11-16** — The first live survey (a white plug, 4 looks, 28 s) was
  refused with anything from 8 to 49 mm fitting, and a refused pick opens no
  incident: nothing was left to see why its outlines matched so poorly. Every
  look's frame is kept now, drawn on, in the run's pictures/, and its numbers
  in events.jsonl (height_look).

### `_wrist_silhouette`'s jaws

- **2026-11-16** — On the run's frame at the grasp the segmenter outlined the
  ball and the moving jaw under it as one. The jaws' pixels count neither way
  (off the valid ones), so an outline is set aside only when too little of it
  is off the jaws, not when it touches them.

## stand_up.py

### `module`

- **2026-10-28** — First stood a pawn up on the rig, on the second try; the
  first missed with the jaw tips 3.6 mm up, level with the head's middle, and
  the moving jaw rolled it out under the fixed one. (also: _try,
  GripperSpec.stand_up_tip_clearance_m)

### `_try`

- **2026-10-28** — Opening moved the grasp point the arm reports (one jaw
  moves), and aiming at the old one swung the fingers 2 mm sideways past the
  piece just stood up (offline).

## lerobot_model.py

### `LeRobotModel.load`

- **2026-10-27** — A network's first pass took 1.6 s for an ACT on the CPU
  (torch setting itself up).

## nudge.py

### `module` (also: `GripperSpec.push_*`, `PickMixin.pick_check`, `Outcome.blocked_by`)

- **2026-11-11** — Live: three missed closes slid the f1 bishop 11 mm into
  the e1 king (the jaws close along the rank), and the retry was refused —
  "jaws need 29 mm open here but white_king_e1 leaves 6 mm". The user asked
  for a push to help recovery, and for the arm to move the king aside and put
  it back. Replayed offline on the rig's kinematics: the jaws turned -30 deg
  left 45 mm, but no orientation reached with the bishop planned as an 82 mm
  king (it was not named); named, the pick went ahead. Pushed from the free
  board-edge side, the bishop could be moved 10 mm to where its pick fits —
  but only 26 mm up it, to keep the gripper body over a possible king, which
  would tip a pawn: so the push is kept no higher than is safe for the
  shortest the object could be, and an unnamed piece there is not pushed.
  Built and tested offline only (tests/test_nudge.py); not yet on the arm.
- **2026-11-11, review** — Where the fingers come down is now kept clear of
  the object's widest point from the finger tips up to its top, not over the
  finger's 9 mm thickness: they pass all of it on the way down, and a top
  wider than the foot would have been landed on.

## place_in.py

### `module` (also: `Policies.place_in`, `Opening`, `GripperSpec.place_in`, `container_*`, `ParallelJaw._clear_of`)

- **2026-11-17** — Asked to put objects into a clear plastic tub on the
  SO-101 tabletop. There was no way to put anything *in* a container: the
  11-16 run planned `place_on(black_bowl)`, a stack on top of it. The place
  carries at the grasp's lift height (7 cm), not over anything, and SO-101
  moves across in joint space, which does not keep its height: so the held
  object is first lifted straight up over the walls. `_own_hover` lowered a
  place's retreat to the held object's own top, inside the tub: `_clear_of`
  keeps every hover of the call over the walls. Offline on the rig's
  kinematics, with the 11-17 tub's outline: walls taken as 150 mm (the
  tabletop's tallest object), every spot was refused — leaning in, the gripper
  would pass 11-65 mm too low over the near wall — and at 100 mm the place
  went in, the jaws 137 mm up crossing the wall line and down to 35 mm inside.
  Hence the rim measured before the place (`look_at_rim`). On the fake arm and
  offline only; the tub was found live, but no object was picked to put in it.

### `look_at_rim` (also: `perception/plane_sweep.py`)

- **2026-11-17** — Two wrist-camera looks along a wall cannot tell its height
  (a straight edge looks the same from anywhere along it: every height fitted
  in the first synthetic test), so they are taken across it. Offline on the
  SO-101's kinematics and wrist camera's model, a rim drawn at 90 mm read
  88 (86-92 fitted), one at 60 read 56 (52-64); the walls are then passed at
  the top of what fitted, and 10 mm more.

## learn_jaw_view.py

### `module` (also: `perception/jaw_view.py`, `ParallelJaw.jaw_view`, `GripperSpec.jaw_view_*`)

- **2026-11-17** — The height looks drew the jaws from the wrist camera's
  model and put them 50-90 px right of where they are in every look (a
  mount fitted to see the table at 8-17 cm, ~11 mm off at the jaws: 11-03).
  So a candy hidden behind the fixed jaw was "84 mm tall" — the jaw itself
  outlined as it — and views were planned that the jaws hid. Learned from
  the pictures instead (dark, from the bottom, in every place): live, 5
  openings from 3 spots in 29 s, the masks on the jaws at every opening.
  With them the same looks outlined the figures whole.

## parallel_jaw/height_looks.py: `_transit_z`, `_go_look`

- **2026-11-17** — After every scan the SO-101 is parked folded up, 41 cm
  high, and the way to each look crossed at the height the arm was at: out of
  its reach with the gripper held for the look. Each look strained for it and
  stalled; heights came out of whatever views it got (a 42 mm figure read
  17-27 mm 34 cm out), and once the move on down from the stall swung the
  gripper into the table. The crossing is now the height that clears what
  stands on the way and no higher, and a look whose way does not arrive is
  given up. Live after: 39-46 mm for that figure.

## parallel_jaw/rim_pick (mrcf/policies/rim_pick.py): `hollow_behind`, the grip height

- **2026-11-17** — The blue scoop: the wall point first chosen was on its
  handle's edge, low and flat, and the jaws shut on the table beside it; a
  wall is now gripped only where a hollow (rim_hollow_m) stands behind it.
  Its looks put the lowest its walls could be at 4, 22, 27 and 13 mm on four
  tries, its best fit at 40-odd each time: it is gripped by the best fit (too
  high is a close on air beside the wall), and refused only where even that
  leaves no bite (rim_min_bite_m).

## parallel_jaw/height_looks.py: `_height_capped` (also: `GripperSpec.grip_low_when_capped`)

- **2026-11-17** — The SO-101's wooden figures (22 mm across, ~35 mm tall):
  seen from 3-12 cm off and 10-14 cm up, anything from 5 to 31 mm fitted
  (0.60-0.76), and every pick was refused as not settled. Gripped as low as
  the jaws go (16 mm) and hovered over the most it could be (+10 mm), any
  height up to that is right; one shorter than the grip is a miss. Live: the
  picks went ahead (and missed for another reason: tabletop NOTES).

## parallel_jaw/pick.py: `_relocate` (also: `GripperSpec.step_after_sideways_push`)

- **2026-11-17** — Six closes on the wooden figures, 19-23 cm out: the jaws
  round the head and neck, the body beyond the tips; each close pushed the
  figure 13-17 mm along the jaws' line, which the miss steps out along the arm
  did not count as evidence (only an unmoved object did). Now a push mostly
  along the jaws' line and not toward the base counts. Not yet seen to help:
  in the next run the pushes were also 6-10 mm toward the base.
