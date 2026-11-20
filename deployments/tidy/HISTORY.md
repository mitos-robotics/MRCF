# Tidy — history

The SO-101 putting everyday things away into a tub on the bare table: the
runs behind this deployment's own values, with their dates. The table task's
code it runs is the framework's (mrcf/tabletop), with its own history
(mrcf/tabletop/HISTORY.md: the SO-101's table runs of 2026-11-16/17 are there
too, where they changed that code). What is still wrong is in NOTES.md.

## The deployment

- 2026-11-16/17: the SO-101 ran the UR5's tabletop deployment on the bare
  table, the chessboard taken away: wooden figures, a plug, a scoop and a
  sweet, into a clear plastic tub (all four in by 15:52 on 11-17). Its rules
  for the arm were filed inside that deployment, as profiles/so101.py.
- 2026-11-17: its own deployment (user: "each deployment should absolutely
  combine both the arm and the layout"). ARM_NAME "so101"; its rules are
  profile.py; its kept state renamed rig/so101/tidy_surface.json and
  tidy_memory.json. It builds the same spec as before; the profile lost only
  GRIPPER_MAX_OPEN_M 0.08, the arm's own value.

## profile.py (was deployments/tabletop/profiles/so101.py)

### `module`

- **2026-11-17** — Its rules for the SO-101 had been the arm's config, tuned
  on chess pieces (mrcf/HISTORY.md, profiles.py). Now its own: what it learned
  on the table (place_in, rim_pick, relocate after a miss, close across a
  lying object, the jaw reseat, gripping low when capped, height from the
  side, the 80 mm opening, "another object" only with one near), and from
  chess only the hold, the grasp fraction, the 4 mm grasp tolerance and the
  wrist camera's first look — the rest of chess's left out with why (its
  docstring). Its spec lost chess's descend_unconfirmed, remeasure_known and
  the fixed-pixel servo's tolerances.
- **2026-11-18 16:08** — GRASP_MISS_WIDTH_M 0.078: the plug, closed on at
  55 mm with the servo loaded, was called "too wide — missed" by the arm's
  30 mm (chess's: no piece is wider), backed up with it in the jaws, parked,
  and looked for on the table it was no longer on. It went into the tub from
  the gripper at 16:13.
- **2026-11-18 16:15** — HEIGHT_FROM_ITS_KIND (name-keyed heights): the
  second figure, its twin measured minutes before, was looked at from the
  side for 32 s (user: "surely all the heights and data should be stored in
  memory at this point?"). The names changed scan to scan, and it never
  helped; replaced the same evening by CATALOGUE.
- **2026-11-18 evening** — CATALOGUE, DIAGNOSE_MISSES, DEFER_FAILED: after a
  day of misses on the same four objects (user: "lets step back and think
  carefully why this is so hard"): the kinds learned and kept, a miss asked
  why, a failed pick set aside instead of re-planning (mrcf/HISTORY.md).
- **2026-11-18 16:43** — GRIP_LOW_FROM_ABOVE: the side looks made the plug
  36-51 mm tall; gripped by that the jaws closed on nothing twice and pushed
  it off its spot (user: "the plug kept getting pushed"). Gripped as low as
  the jaws go at 16:08 it had held. Where the overhead outline alone caps an
  object under 71 mm (the jaws' lowest grip and the gripper's free depth),
  it is gripped so with no side looks.
- **2026-11-19 10:55** — NAME_TILE_PAD_MAX_PX 40, NAME_NOTHING_TWICE: the
  table reset for a timed run, the blue scoop's naming tile showed half the
  table round it (room round a tile a share of the outline's 19 cm), the
  model called it "nothing", and it was left out of the scene: the plan
  could not have had it. The tile's room now at most 40 px; a "nothing" on
  the sheet asked again by itself before it is let go.
- **2026-11-19 11:07** — PLACE_IN_OVER_CONTENTS: the scoop, 128 mm across,
  was refused by place_in — no spot 83 mm in from the walls and clear of the
  figure put first in the tub's middle — and the plan stopped with it in the
  jaws. What will not fit clear of the contents is let go from over the
  walls onto them, where the jaws themselves fit.
- **2026-11-19 11:09** — HELD_WHILE_STILL: the same scoop, seen carried as it
  was lifted, was "not held" two minutes later: the arm at home, the wrist
  camera looking out across the room past it, the vision model said "the
  jaws are empty" (a reworded question said the same). Once seen carried, it
  is held while the jaws stand where they were (no more than 0.4 mm nearer
  shut; had it slipped out, the squeezing jaws would have closed on).
- **2026-11-19 11:18** — CONTAINER_MAX_LEAN_DEG 20, CONTAINER_DESCENT_RETRY
  False: the plug taken to the tub's far corner, 36 cm out (the roomiest
  spot clear of the cap and scoop), reached leaning 35 deg; the gripper met
  the far wall going down, the descent was tried again, and it pushed the
  tub across the table (user: "you are pushing the tub"). It then said
  "lowered inside". The spots that went in cleanly leaned 10-20 deg.
- **2026-11-19 11:30** — TRANSIT_OVER_WALLS: the drag of 11:18 was the move
  after the release (user: "it didn't recognise the tub wall and didn't lift
  the jaws above the walls so it dragged the tub with it"): up to 112 mm in
  the tub, then straight to the next object's 61 mm hover across the wall.
  A pick's way to its hover that crosses a container's walls now rises over
  them first and crosses at that height.
- **2026-11-19 11:30** — FAR_FROM_LEAN_DEG 15: the plug at 32 cm was
  "mid-reach" by chess's 29 cm and went top_down; its leaned descents
  stalled three times (user: "pick up things at an angle not just top down
  for slightly further away things"). A spot straight down would need more
  than 15 deg for goes to reach_out, the leaning pick.
- **2026-11-19 11:30** — TRANSIT_SETTLE False (user: "make the whole motions
  smoother - its very jerky"): the joint record of 11:18:22-50 had 62 stops
  and starts in 29 s, the arm moving 12 of them — guard pauses for the
  arm's cable dragged across the table, each move's 1-1.5 s creep onto its
  goal, and many short correcting moves. Moves that only travel no longer
  wait to settle.
- **2026-11-19 12:00** — LEAN_CLEARS_FLOOR, LEAN_FLOOR_CLEAR_M 0.016: a grip
  held leaning is raised as far as the jaw tips then reach below it (9-10 mm
  at 20-30 deg) and room for the arm's sag far out — 4 mm at 10 deg, 16 at
  30, in proportion — but never after a miss the wrist camera saw closing
  above it. 12:01 run: the ball at 34 cm picked first try leaning 20 deg;
  the plug, raised to 24-25 mm at 10-20 deg by a flat 16 mm, slipped off an
  edge (user: "the arm needs to go down slightly more"): made proportional.
- **2026-11-19 12:06** — RIM_GRIP_DEPTH_M 0.020, RIM_SWING_SHARE 0.33: the
  scoop's looks put its rim at 53 mm (it stands ~45); gripped 12 mm under
  that, 41 mm up, it hung by the very tip of its wall, swung on the way
  across and hit the tub's wall (user: "it picked it up on the very very
  tip ... hence why it hit the wall"). Gripped 20 mm under, and crossed a
  third of its width higher.
- **2026-11-19 12:01-12:06** — all four into the tub in 4 min 59 s from the
  command: planning 60 s, cap 23 s, ball 29 s, plug 70 s failed (slipped)
  then 56 s on the retry after a fresh look, scoop 52 s (25 s of side looks:
  lying otherwise, the catalogue did not know it).
- **2026-11-19 12:36** — RIM_GRIP_DEPTH_M 0.020 taken out again: gripped 20
  mm under the rim (27 mm up) the jaws closed on nothing, then 10 mm lower
  on nothing again — the scoop's bowl tapers, its wall further in lower
  down — and the scoop was left upside down. PLAN_MEMORY, HOME_AFTER_COMMAND
  (user: "same objects in the same context ... why isn't the arm ending in
  its main position?"): 12:33 run, planned in 0 s, home at the end; cap 25 s,
  ball 24 s, plug picked on the retry (34 cm, then pushed nearer), 3 of 4 in
  3 min 57 s.
- **2026-11-19 12:42** — DESCRIBE_FAILURES False, HELD_TURNS: run from the
  kept plan (started in 3 s): cap 35 s and ball 24 s (both 33-34 cm out,
  leaning), plug 17 s first try (held 27 mm across, off its middle: lifted
  for its 20 mm grip it caught the tub's far wall — user: "it didn't lift it
  high enough"), scoop gripped by its wall and carried home, called "left
  the jaws on the way up" by the hold check; put in by hand-steered moves at
  12:46. Three in by 2 min 33 s.
- **2026-11-19 12:50** — GUARD_STOPS_PUSHING; HELD_TURNS by where it was
  gripped (user: "based on where it grips the object it needs to lift it a
  different height to pass the tub walls"; "why can't the camera see
  something is wrong when this happens and stop it?"): crossing the walls,
  a thing gripped off its middle lifts by its farthest part from the grip;
  and movement beside the gripper that starts again each time the arm
  carries on stops the move rather than pausing it. The catalogue's recipe
  top no longer creeps 10 mm a reuse (mrcf/HISTORY.md).
- **2026-11-19 13:50** — HANDLE_FIRST, PARTS_INSIDE_ARE_ITS_OWN,
  CONTAINER_CROSS_EXTRA_M 0.015 (13:33 run: the cap and plug in; a figure
  outlined as two refused twice for its own head; the scoop, held by its
  wall, called lost a fourth time — its jaws creep 0.9 % to 0.07 % shut on
  the flexing wall as it is carried, so nothing can tell held from slipped;
  the cap clipped the tub's wall, its grip 3 mm higher than the place
  reckoned). The scoop now by its handle (user: "why not grasp a more
  secure thing ... like maybe grab the handle??"), closed across it from
  high down until the jaws stop on it under load.
- **2026-11-19 14:10** — OUTLINE_BEFORE_PICK, RELOOK_AFTER_MISS: 14:03, a
  figure outlined by the scan as 16 mm of its 23 matched nothing in the
  catalogue, was gripped at that fragment's middle 5 mm off its own, tipped
  by the jaws, came down on again blind and rolled off the table (user: "how
  is it getting worse"). Every plain pick now outlines its object again by
  what it is first — and the whole outline is matched to the catalogue — and
  after a close on nothing it is looked for afresh before another try. The
  snapshot also draws what the scan outlined, to check a scan before a run.
- **2026-11-19 14:25** — the scan checked before a run (/snapshot now draws
  what it outlined): the scoop whole, the figures whole, the plug on its side
  split into its body and its pins (both "white block"). An object outlined
  again by name (OUTLINE_BEFORE_PICK) now takes what the scan outlined inside
  it as part of it.
- **2026-11-19 14:50** — QUICK_PUT_INTO (user: "its fine for this - but i
  guess its still needed in the framework so make sure it doesn't break the
  other deployments"): 14:25, planned in 240 s, the plug looked for as
  "silver cylinder" (the scan's name for its pins), the re-plan timed out
  and came back cut off; both figures in, first try. "Put these into the
  tub" is now planned from the words: each thing found by what the user calls
  it (by name, by its kind in the catalogue, else looked for by those words),
  picked as its kind was held (Kind.picked_by), put in; the planner for
  anything else.
- **2026-11-19 14:58** — CONTAINER_MAX_DROP_LEAN_DEG 45: 14:55 run, planned
  from the words in 0 s; the red ball in (18 s pick); the cap refused (every
  lean to it would sweep the gripper over the scoop standing in front); the
  plug pushed on a reseat and lost; the scoop picked by its HANDLE first try
  (21 mm, 34 mm up, held under load) — then its place refused: crossing high
  for its swing (~180 mm), every spot leaned 35-40 deg, past the 20 that is
  for lowering inside. Letting go from over the walls may lean to 45.
- **2026-11-19 15:07** — 15:05 run, from the words in 0 s: the cap (16 s
  pick), the red ball (20 s), the plug (19 s, across its body) all in first
  try, 1 min 26 s. The scoop's handle found again; on the way from the tub to
  it the arm came down across the tub's wall and pushed it ~7 cm — the guard
  stopped it on the second push (GUARD_STOPS_PUSHING, right this time). The
  wall and handle picks went straight to their hover: they now rise over a
  container's walls first, as the plain pick does.
- **2026-11-19 15:14-15:17** — ALL FOUR INTO THE TUB, NOTHING DONE BY HAND, 2 min
  35 s from the command: planned from the words in 0 s; the cap 54 s (its
  first descent stalled; dropped from over the walls), the red ball 36 s,
  the plug 41 s (held off its middle, dropped from 179 mm for its swing), the
  scoop 24 s by its handle (up over the tub's walls first, dropped from 177
  mm); the arm home at the end.
- **2026-11-19 15:50** — from the user's film of the 15:27 run: the arm went
  home and straight up to its park pose before the plug, to re-outline it
  past itself; a re-outline is now only for an outline that looks wrong (its
  kind not known from it, a fragment of its kind's size, another outline
  against it) and never moves the arm. The scoop's handle (user: "too
  precarious - move the jaw down slightly to make sure you get it first
  try"): where it held before, closed first 4 mm under that, not a step
  above it (handle_bite_m), the height that held still the one learned.
  The scan finding one figure twice by name; a "too high" miss the wrist
  camera saw, not given up when it cannot be found again from above.
- **2026-11-19 17:30** — before the next run, offline: (1) the rest tool knew
  nothing of the tub and checked its fold against the chess board: a tool now
  takes the deployment run.py launched last (rig/so101/last_deployment.txt),
  and the tub's outline is kept on disk beside its walls' height, which the
  rest keeps clear of. (2) 6 of 9 put-ins in runs 36-37 let go 13-18 cm up:
  over the tub, that high and 25-32 cm out, the arm reaches only leaning
  30-40 deg, while at the bottom 10-15 deg does; it now leans as far as being
  over the spot takes while the jaws are above the walls, and no more than
  20 deg among them (CONTAINER_LEAN_ABOVE_WALLS), and a spot it can lower
  into comes before one it can only drop into. (3) The "bobbing" in the
  user's film (15:28): the descent re-chosen at -15 deg, turning and coming
  down at once, went 4 cm up and 3 cm out; it now turns over the object
  first, and one that goes off its line goes back over the grasp, not up
  from where it went (DESCENT_TURN_FIRST). (4) "close 5.1 s" was mostly the
  wrist camera's reseat (2.9 s, three moves each settled to 2 mm): its way
  up and across no longer settles (QUICK_RESEAT); the close itself, 1.9 s,
  is as it was. None of it run on the arm yet.
- **2026-11-19 16:21-16:25** — run 38, 2 of 4 in 3 min 45 s (worse; user: "a
  much much worse run"). Lowered into the tub at last: the red ball let go 40
  mm up (25 deg over it, 20 among the walls), the scoop by its handle 85 mm
  up. The cap: two approaches refused for sweeping over the plug 9 cm off, one
  closed in front of it (it then measured 23 mm tall: knocked over); at the
  end every descent was refused by the table guard (tips 3-9 mm into the
  table) and each was leaned further, lower still. The plug: no outline from
  the wrist camera, came down on the overhead camera's aim, 2 cm off at the
  grasp, closed on its edge and pushed it 4 cm; under a new track it was
  never tried again. QUICK_RESEAT made the wrist camera's reseat useless
  (6.6 -> 7.8 -> 6.6 mm): off again. Built: REFUSED_DESCENT_ENDS (a refused
  descent ends the attempt), REFIND_NEAR_M 0.10 (a set-aside object found
  again by its name near where it was).
- **2026-11-19 16:37-17:30** — run 39 stopped by the user after the cap: its
  descents refused by the table guard, and REFUSED_DESCENT_ENDS (built off run
  38's lying cap) ended each attempt — but leaning further, re-planning the
  grip higher, is how run 36's cap got down: off again. Why it was refused:
  offline, the refused command put the jaws 27 mm under the grip and 15 mm
  short; the solver's own answer at -20 deg clears the table by 25 mm. The
  step-by-step descent, re-steering from joints lagging 2-3 deg under load,
  drove the shoulder past its goal while the elbow (36 deg to open) trailed.
  Simulated on the arm's kinematics with such servos it ends 7 cm off, as
  run 37's bobbing did. DESCENT_LINE_ABOVE_M 0.025: the descent most of the
  way as one joint line to the solver's answer 25 mm over the grasp, the
  rest as before — in simulation it arrives (3.5 mm) and strays no more near
  the object over the cap, ball and plug. Not yet run on the arm. Also ruled
  out with the frames: the wrist camera had not moved (the fixed jaw where it
  was at 15:28).
- **2026-11-19 17:06-17:10** — run 40: 3 of 4 in 3 min 33 s (the cap, the
  scoop by its handle, the ball at the end), all three lowered into the tub.
  The descent line ran six times and arrived every time (1-3 mm). The plug:
  the wrist camera's reseats were refused on the way down by the table guard
  — a stuck joint's command pushed 3 deg past its goal — so the jaws stayed
  15 mm up, the next look (taken as if at the grip) misread the offset, and
  the close missed and pushed it. FLOOR_CAPS_PUSH: such a push sent as the
  plain step. Replayed offline (13 moves from runs 37-40, the arm's kinematics,
  its floor guard, servos 1-2.6 deg short): the sim matches 9 of 13 live
  outcomes; it shows far-out grips under ~20 mm are erratic (arrive/refuse
  flips with 3 mm of grip height), the next thing to measure on the arm.
- **2026-11-19 17:40** — the servos measured from the flight recorder (22
  windows, runs 35-40, no arm moved): far out and low a joint stops short of
  its command by (75th percentile) elbow 2.8 deg moving + / 0.9 moving -,
  shoulder lift 1.6 / 1.6, wrist flex 0.4 / 2.8, pan and roll ~0.5, and moves
  at most ~30 deg/s. The replay with that model, 110 descents (6 spots far and
  mid, leans 10-30, grips 16-28 mm): the old descent 72 arrived (18 refused,
  20 short, worst 103 mm off); DESCENT_LINE_ABOVE_M + FLOOR_CAPS_PUSH 102
  arrived (none refused; the 8 short all at a 16 mm grip, 4-10 mm off); a joint
  line all the way to the grip 110 arrived but swung the jaws up to 18 mm
  sideways near the object — as the real arm once did (16-17 mm) — so not used.
- **2026-11-19 18:40** — tools/so101/rest.py looks before it moves: the
  overhead camera's look (surface_scan against the empty table, no names)
  decides which remembered objects are still where they were, adds whatever
  it sees that they are not, and the walls kept; a rest or a wake's path is
  checked against that (table_now). Without the rig's cameras it connects
  without them and checks against the last scan. 18:20: the user moved the
  scoop and a figure off the left rest's path, and every rest was refused for
  them still. Not yet run on the rig (unplugged when built).
- **2026-11-19 19:00** — TASKS {"tidy up": the four into the tub} and PLANNER
  False: "tidy up" or "go" runs it, no planning; "put ... into ..." as before;
  anything else answered with its task; a thing not found is left out and the
  rest done (run 41 lost 2+ min to the planner over "the red ball").
- **2026-11-20 09:12-09:15** — run 43, morning light, "tidy up" (its task, no
  planner): 3 of 4 in 3 min 07 s. The scan named all four right; the cap and
  the ball first try (~22 s each, lowered in 38-41 mm), the scoop by its handle
  first try (let go from over the walls: no room left to lower it). The plug:
  the jaws opened 52 mm for a plug 59 across (its foot read 44 from above) and
  closed on its edge (user: "the jaws need to be a bit wider for the plug and
  your method will work"); found again 5 cm off by its name, then refused for
  its own old track beside it. Built: RECIPE_OPENS_TO_KIND (the jaws open past
  its kind's width across: ~71 mm for the plug), the re-found object's old
  track retired (ObjectMemory.superseded). The wrist camera still placed only
  1 of 5 feet in good light — not the light, then: its shift against the
  overhead camera (tools/calibrate_wrist_on_objects.py) is the next suspect.
  The rest tool's live look worked on the rig (saw the scoop moved off the
  wake's path).
- **2026-11-20 09:24** — run 44, the plug alone ("put the white plug into the
  plastic tub"): the jaws opened 71 mm (RECIPE_OPENS_TO_KIND: its kind 63
  across). Attempt 1 got it between the jaws but slipped out on the lift;
  attempt 2 met it at 58 mm, the squeeze went on to 43 with no load, and it
  slid out — then the jaws went on to 29 mm after the lift, and the wrist
  camera's 0.5% of the picture "moving with them" was taken for it carried:
  "put into" the tub, still on the table (user: "no it isn't - the pick didn't
  work"). HELD_CLOSED_M 0.010: jaws that closed that much further than they
  shut on it hold nothing, whatever the camera seems to see. The plug's ~60 mm
  smooth sides are near what this gripper holds; it held before across ~27 mm.
- **2026-11-20 09:56-10:00** — the wrist camera's shift fitted from the objects
  (tools/calibrate_wrist_on_objects.py: two figures agreed within 3 mm, the
  white plug on the white table not a ruler): (-5.6, -8.0) -> (-8.7, -13.2) mm,
  applied. Run 46, the plug alone: first try, held under full load at 53 mm,
  lowered in, 34 s.
- **2026-11-20 10:25-10:28** — run 46, "tidy up": ALL FOUR INTO THE TUB, 2 min
  56 s, nothing by hand (user: "yup all good"). The figures as "red cap" twice:
  "the red ball" found by its words (look_for). The ball 21 s and the scoop by
  its handle first try; the cap refused at first (out of reach), picked at the
  end after the look; the plug's first close rightly called empty (jaws 30.2 ->
  19.4 mm, HELD_CLOSED_M), found again lower down, held across its ~27 mm
  thickness and let go from over the walls. Every hold but the plug's last
  felt under full load.
- **2026-11-20 10:32-10:34** — run 46 again, "tidy up": ALL FOUR, EVERY ONE
  FIRST TRY, 1 min 50 s (the best yet; run 36's 2:35). Every hold felt under
  full load (23, 24, 49 mm; the scoop by its handle at 17); the figures lowered
  in 37-38 mm up; the plug and the scoop let go from ~180 mm — no room left to
  lower them clear of the figures already in (the figures went 54-59 mm in
  from the walls, not to one end).
