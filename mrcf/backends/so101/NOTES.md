# SO-101 — notes

What this robot has shown that its code and config cannot say: quirks, failure
modes, what to watch for, what was tried. One entry per issue; delete an
entry once it is fixed (git keeps it). What belongs where:
[DEPLOYMENT_SPEC.md](../../../DEPLOYMENT_SPEC.md#what-you-learn-and-where-it-goes).
Measured values and the reasons behind them stay in `robot_config.py`, beside
the values.

Each entry: **Seen** (the date, what, the flight recorder's incident),
**Cause**, **Workaround**, **Status** (open, or fixed and where).
The framework's own limits, on every robot, are in [mrcf/NOTES.md](../../NOTES.md).

## Limits

- **No force sensing.** `get_contact_force()` returns 0, so the safety
  monitor's force limits never fire. The descent floor and the height ceiling
  are live. Whether a grasp holds is judged from the gripper servo's load.
- **Its reach is smaller than a chess board.** Reaching straight down, it gets
  about 0.33 m from its base. `tools/so101/check_reach.py` maps the reach.
- **From the park pose, the solver finds no straight-down approach** anywhere on
  the board. A move begun there leans 5–20°, where the same move from home
  reaches straight down. The stand-up goes home first; a pick after a scan
  still leans.
- **Its servos twitch while it holds a pose,** about 40 % of the time. Each
  twitch is a joint movement, so the motion guard leaves the 5 cm round the arm
  unwatched for a second after it. On this rig the guard stops the arm 0.1–0.15 s
  after a hand appears: up to one frame (~70 ms) plus the camera's lag (~70 ms,
  measured).

## Open

### A pawn gripped at a 15 deg lean can be tipped over by the close
- **Seen:** 2026-10-29 at 15:17, the first pick after a scan, a 42 mm pawn on e5
  (0.30 m out): "Straight down is out of reach here — holding a -15 deg tilt".
  The jaws came down 0.2 mm from their aim, the wrist camera shows the pawn's
  head against the fixed jaw before the close, and on the hover after it the
  pawn lies on its side. The two retries took it for standing and pushed it
  another 17 and 31 mm. Picks at e4 (0.25 m, a -5 deg lean) the same day gripped
  first time, twice. Incident:
  `runs/flight/20261029_151555/incidents/151752_pick_pick_failed__pick_failed_after_3_attempt/`
  (`frames/front_0017.91.jpg`, `front_0021.48.jpg`).
- **Cause:** not known yet. Likely the lean: the close pushes the head, high on
  a short light piece, over the fixed jaw's pad.
- **Workaround:** keep picks of short pieces where the lean is 5 deg or less
  (ranks 2-4 on this rig); a pick after a scan leans more (see Limits).
- **Status:** open. Also: after a miss the retries do not check whether the
  object now lies on its side (perception/lying.py could tell them). Seen
  again 2026-10-30 on e3 at a -10 deg lean: gripped, lost in the lift, and the
  wrist camera could not find the pawn after (`runs/flight/20261030_094816/`).
- **Seen again:** 2026-11-04 at 10:25, the pawn on e4 (0.27 m out): the descent
  at -15 deg stalled before it had moved at all (7.3 of 7.2 cm short, every
  joint within 3 deg of its command), the retry leaned to -25, and the close
  squeezed the pawn out of the jaws — they stopped at 10.4 mm, then the squeeze
  ran them shut — and it landed 10 cm away, between d1 and e2. Incident:
  `runs/flight/20261104_102507/incidents/102600_pick_pick_failed__pick_failed_after_3_attempt/`
  (`frames/front_0021.82.jpg` between the jaws, `front_0023.34.jpg` gone).
  Since then a descent that stalls before a quarter of the way down is tried
  again at its own lean before leaning further (`ReachMixin._stalled_at_start`);
  not yet seen on the arm.

### Grasps at a lean of 10 deg or more close in front of a pawn
- **Seen:** 2026-10-30, 26 grasp attempts on one pawn over four runs: at a
  lean of 0 to -5 deg 6 of 7 gripped, at -10 4 of 7, at -15 to -20 2 of 8. In
  the misses the wrist camera's frames at the grasp show the pawn beyond the
  fingertips, along the arm, also when the wrist servo and the overhead camera
  had both just agreed the jaws were over it (zigzag 9, d4:
  `runs/flight/20261030_110214/incidents/110303_pick_pick_took_2_attempts/`,
  `frames/front_0024.16.jpg`). A first attempt leans more because it chooses
  its lean from a higher hover; the retries that gripped came from a 60 mm
  hover at -5 or 0 deg.
- **Cause:** not known yet. Where the fingertips are taken to be at a lean —
  the jaw model, or the servo's step for a lean other than its calibrated
  -5 deg — is the suspect; the same few mm either way decide hit or miss.
- **Workaround:** none yet. Supersedes the entry below, whose explanation (the
  piece standing away from where it was put) these frames rule out.
- **Status:** open. Built 2026-10-30, not yet measured on the arm: a grasp
  correction by lean and reach (`mrcf/grasp_correction.py`), measured by
  `tools/calibrate_grasp.py` with the jaws on pawns standing free. Its first
  run on free pawns measured nothing: every lean pushed the pawn 3-14 mm
  toward the base. It now finds the pawn again after every miss.
- **Seen again:** 2026-10-31, d3 at a -10 deg lean, the grasp correction on:
  two attempts, each with the jaws 0.2-0.3 mm from where the scan had the
  pawn, closed on nothing. Each time the pick found the pawn again by a scan,
  16-17 mm away the same way (+10/+11 mm along the arm, +14/+12 across): the
  wrist camera shows it beyond the fixed jaw's tip as the jaws close, and the
  moving jaw flicks it out sideways. Incident:
  `runs/flight/20261031_095321/incidents/095702_pick_pick_failed__skill_raised_exception__ope/`
  (`frames/front_0034.20.jpg`).
  But the first attempt aimed where the place had put the pawn, not where a
  scan saw it: at the hover the wrist servo moved the jaws 13-15 mm out
  towards it (twice), stopped 13-16 px short, and the pick went back to the
  remembered spot ("Visual servo failed — using kinematics alone"). The second
  aimed at the scan's fresh position, but by near_base, which the grasp
  correction does not measure.
- **Measured:** 2026-10-31, `calibrate_grasp.py --squares d3`: 0, +4, +4, +4,
  +2 mm at 0, -5, -10, -15, -20 deg — within the probes' 4 mm step of e2's
  and e4's. The correction does not differ on d3; its catch window is 4-8 mm
  wide there.
- **Seen again:** 2026-10-31, c2 (zigzag 16, just scanned, leans 0 to +0):
  the wrist servo settled 11-13 mm outward of the pawn (+10 to +12 mm along
  the arm, "stepped 4-6 mm" for its tilt against the calibration's -5) on all
  three attempts; each close missed; each scan after found the pawn where the
  scan before had it. Incident:
  `runs/flight/20261031_103210/incidents/103428_pick_pick_failed__pick_failed_after_3_attempt/`. The retry
  after the first miss kept top_down ("Again by top_down") and the scans
  found it 0 and 9 mm away. The wrist camera did not drop once.
- **Built** 2026-10-31, offline-tested, not yet on the arm: the servo leading
  the jaws more than `SERVO_AGREE_M` from where the object was thought to be,
  settled or not, has a scan find it first; after that the scan's position
  stands for the pick. A retry at a position found afresh keeps the approach
  before (`WristServoAligner._disputed`, `PickMixin._start_attempt`).
- **Measured** 2026-11-03, offline, from a look tour over a full board
  (`rig/so101/look_tours/20261103_110243`) with the wrist camera located
  (`GRIPPER_CAM_POSE`). The camera puts every piece's foot within 0.1-3.5 mm of
  the scan's. But in the arm's frame, through its mount and the jaw model, it
  puts each piece 5-12 mm further out along the arm than the overhead camera's
  aim, at leans of -10 and -15 deg. With the jaw centre moved to where the hand-measured
  `GRIPPER_CAM_GRASP_PX` has it, it is still 3-10 mm further out. That move is
  -1.8, +10.9 mm in gripper x, y, the same within 0.4 mm at three looks: the
  jaw model is off ~11 mm along the jaws' opening, as this camera sees them.
  The same way as these misses. Picks with the located camera say both aims,
  each time ("by its own jaw model it would aim ..."): the data to settle which
  is right.
- **Seen** 2026-11-03, on the arm, the full starting position: e2, d2 and b1
  each gripped at the first attempt at a -15 deg lean. Each aimed where the
  overhead camera put the jaws, moved by where the wrist camera found the foot
  against the scan (0.6, 0.3, 5.1 mm); no wrist servo. The overhead camera at
  the grasp had the jaws 0.9-2.3 mm off that aim. The camera's jaw model would
  have aimed +4.5/+2.2/+5.6 mm along the arm and +7.0/+10.6/+19.8 mm across:
  off the pawn by the across part alone, more the further round the arm was
  (b1 the most). The overhead camera's aim is the right one; the jaw model,
  as the wrist camera sees it, is not. With 3 picks this is a workaround, not
  yet a cause: the misses above aimed by the wrist servo, which these did not
  use (chess `logs/picks.jsonl`, 12:23-12:55).

### Tall pieces on d1 and e1 are out of reach
- **Seen:** 2026-11-03, offline: the real policy on the 13:39 scan of the full
  board, no arm. It is predicted for d1 by the 11-03 look tour's offline run
  too. On the arm d1 has not got this far: it was refused for room first
  (chess NOTES, "The back-rank pieces cannot be picked").
- **Cause:** d1 and e1 are 96 and 103 mm from the pan axis. A piece not yet
  measured is planned as the tallest it may be (82 mm), hovering at 97 mm, and
  no lean reaches that hover, the 27 mm grasp and the lift there. A measured
  69 mm queen (hover 84, grasp 41) reaches only at -30 deg with the lift no
  higher than the hover and the wrist 1 deg from its limit. No approach allows
  that: top_down wants the full 91 mm lift, near_base checks the grasp 15 mm
  nearer the base, and reach_out keeps every joint 3 deg from its limit. Also,
  probed alone from the arm's pose, the hover reaches only at -30/-40 and the
  grasp only at -20 to +20; from the grasp's own solution, -30 reaches both.
  The search's seed matters here.
- **Workaround:** none yet.
- **Status:** open.

### A pick on e6 tries reach_out first, which stalls there
- **Seen:** 2026-10-31, a pawn on e6 (0.34 m out): reach_out at -30 deg
  stalled 15 cm short of the grasp, then at -40 with the wrist at its limit;
  the second attempt, top_down at -20 deg, gripped it. 52 s of the pick's
  86 s went to the two stalls. Incident:
  `runs/flight/20261031_095321/incidents/095514_pick_pick_took_2_attempts/`.
- **Cause:** the plan finds the -30 deg lean in reach (every joint 3 deg from
  its limit); the arm does not get there, as `tools/calibrate_grasp.py` found
  on e6 at -25 and -30 the day before.
- **Workaround:** none needed: the retry grips.
- **Status:** open.

### Its cameras' readers die
- **Seen:** 2026-10-31, in three of four runs (HISTORY.md, `CamerasMixin._reopen`).
- **Cause:** not known. A camera that stops sending for a moment (the USB
  link, macOS rebuilding its camera list as an iPhone comes and goes) uses up
  lerobot's eleven tries at once.
- **Workaround:** a camera whose reader has died is opened again
  (`CamerasMixin._reopen`), the arm paused meanwhile if the motion guard
  watches through it.
- **Status:** open. Seen on the rig 2026-10-31 (the d3 calibration): the
  reopen ran each time, and the wrist camera opened but sent no frame within
  lerobot's 1 s warmup and failed again; a fresh process read it without a
  failure. The cable, at the wrist or elbow, is the next suspect.
- **Seen again:** 2026-11-03 and 11-04, at start-up: the wrist camera sent no
  frame in lerobot's 1 s warmup and the controller exited before the arm moved.
  Since 11-04 `SO101Env.connect` closes and opens such a camera again, up to
  `CONNECT_CAMERA_RETRIES` (2) times, before giving up; not yet seen on the arm.

### A piece the arm puts down can stand 9-15 mm from where it was put
- **Seen:** 2026-10-30, a white pawn moved square to square, 10 picks: five
  first attempts closed on nothing with the overhead camera putting the jaws
  within 2 mm of where the pawn was remembered. Each retry, aligned by the
  wrist camera, found it 9-15 mm away, always the same way (+7 to +13 mm along
  the arm, -6 to -9 across), mostly on the d and e files, and gripped it. One
  of the five came straight after a scan, not a place. Incident:
  `runs/flight/20261030_094816/incidents/094907_pick_pick_took_2_attempts/`
  (`frames/front_0010.98.jpg`: the pawn beyond the fingertips before the close).
- **Cause:** not known yet. The same direction every time points at something
  systematic: where the overhead camera puts the jaws, or the piece, near the
  board's middle (the scan already sits short of the arm there:
  `MAX_SERVO_SHIFT_M`).
- **Workaround:** the first attempt aligns with the wrist camera too
  (`OVERHEAD_FIRST = False`, `SERVO_AGREE_M` 8 mm).
- **Status:** superseded by the entry above. Tried 2026-10-30: the wrist servo
  on the first attempt agreed with the overhead camera within 2-5 mm, and
  the grasp still closed in front of the pawn.

### A tall piece held high on a tilt can slip out while the arm swings
- **Seen:** 2026-10-29. A 42 mm chess piece, gripped 25 mm up (its width changes
  2.5 mm over the pads there) with the gripper tilted 10° from straight down,
  held through a scan at the park pose; about 11 s into the place, as the arm
  swung from park toward the square, the jaws closed further (the gripper read
  6.4 % then 4.0 %): it had slipped out. `place` reported "the object was no
  longer in the jaws at the release — dropped on the way". Incident:
  `runs/flight/20261029_115509/incidents/115747_place_place_failed__the_object_was_no_longer_i/`
  (`samples.jsonl` has the gripper through it).
- **Cause:** not known yet. Likely the grip on a tapering part of the piece and
  the swing's acceleration with the piece high in the jaws. To try: a lower
  grasp height, a firmer squeeze, a gentler joint-space swing while holding.
- **Workaround:** keep a held piece out of long swings — no scan while holding
  (chess no longer rescans to put a piece back on its own square).
- **Seen again:** 2026-10-30, a pawn gripped by the neck under its head (24 mm
  up on a 43 mm pawn, a -5 deg lean), no swing: 0.2-0.4 s into the carry from
  d4 to e3 the gripper read 8.0 % falling to 4.1 % and the pawn fell on its
  side. The wrist camera's frames show it (`front_0005.20.jpg` held,
  `front_0005.71.jpg` gone). Incident:
  `runs/flight/20261030_100652/incidents/100732_place_place_failed__the_object_was_no_longer_i/`.
  The place only found out at the release, 9 s later.
- **Status:** open. Built 2026-10-30, not yet tried on the arm: the jaws hold
  with the squeeze's pressure (`GRIPPER_HOLD_PCT` 10, was 3), and a place
  stops the arm the moment the jaws close on a slip (`GripperSpec.slip_m`).
