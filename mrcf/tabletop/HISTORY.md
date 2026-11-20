# The table task — history

Why the code is as it is: the runs behind its values and fixes, with their
dates. Each deployment that does this task has its own history too
(deployments/tabletop: the UR5's letter blocks; deployments/tidy: the SO-101
putting things away). What is still wrong is in NOTES.md and theirs.

## The table task

- 2026-11-02: written for the UR5 as deployments/tabletop. The user wanted it
  to pick and place "a variety" of everyday objects, not play chess. They
  chose named spots and spots beside objects as the places, and upright
  objects first. Built from the framework's surface scan, object memory and
  standard pick and place, with the chess controller's shape.
- 2026-11-16/17: run on the SO-101 too, as the same deployment, with that
  arm's rules filed inside it (profiles/so101.py), which left the UR5 with none
  of them.
- 2026-11-17: moved here from deployments/tabletop when a deployment became
  one arm, one layout and one task (user: "each deployment should absolutely
  combine both the arm and the layout"). The SO-101's table work is now
  deployments/tidy; both hand this their config (tabletop.use). What the
  deployments serve was compared before and after: settings, the policies'
  spec, the actions and the planner's examples the same, but for the kept
  state's names (rig/so101/tidy_*.json) and the letter blocks' size, left to
  the UR5's deployment.

## settings.py (was deployments/tabletop/config.py)

### `SURFACE_Z`, `serve()` (also: `ENV_OVERRIDES`, `POLICY_OVERRIDES["table_z"]`, rig/so101/tidy_surface.json)

- **2026-11-17** — On the SO-101 every pick closed round the objects' heads
  or in the air: twelve closes over four picks, none held; heights read
  short. The arm's TABLE_Z was the chessboard's top, and the tabletop ran on
  the bare table 16 mm lower (user: "there is something fundamental wrong
  here"). The bare table, touched with the jaw tips (the arm limp, read
  through its kinematics), is the tabletop's surface; the figures were then
  gripped first try. Touching it with the arm's own descent first pushed the
  tub aside and overloaded the shoulder (touch_surface: off).

### `CAMERA_DRIFT_PX` (also: `skills.scan_scene`)

- **2026-11-13** — The overhead camera crept 12.8 px (about 14 mm) in the 15
  minutes after its calibration, and a pick landed on its block. Each scan
  now checks the camera against the empty-table photo and stops at 4 px (the
  check reads 0.2-1.1 px standing still). Stopped, the objects are cleared,
  so no plan acts on what the moved camera saw.

### `REMEMBER_ACROSS_RESTARTS` (also: `ObjectMemory(load=)`)

- **2026-11-13** — The object memory started each run with the last run's
  blocks. Rearranged by hand between runs, they came back as things "missed
  by the last scan; last seen there", which refused picks and places ("blue_block
  ... leaves 0 mm") and opened runs with "not seen where it was put". The user
  began three runs that day with "forget everything". Now each start begins
  empty and the first scan says what is there. Named places are kept (a file
  of their own), and chess still remembers its board.

## naming.py (was deployments/tabletop/naming.py)

### The colour a block is named by (`_SHEET_SYSTEM`, `_prompt`)

- **2026-11-13** — With the overhead camera re-mounted to look almost
  straight down, the letter blocks' white tops filled the crops. The model
  named them by their main colour: "white block M", "white block R",
  "white block D". The user calls them red M, green R and blue D, and the
  same blocks came back blue or white from one scan to the next. The colour
  is now the print's on a white or plain object with a coloured letter,
  number or border, else the main colour. Every naming sheet kept that day
  (11), named again: every block by its print, twice out of two. The two
  white objects by the arm's base stayed white, and every "nothing" stayed.

### `identifier` (also: `ObjectMemory.reidentify`)

- **2026-11-12** — Letter blocks look alike to the memory. After a restart,
  green 7, taken off a stack by hand, came back as blue_block_z, and a plan
  moved it twice. An object matched by its look alone is now read again;
  its name moves only when the reading names a remembered object and the
  camera's colour agrees, since on the same frames gemma4 read red U, on its
  side, as D and as P, and called green 7 blue.
- **2026-11-12** — Names end with the letter or number on the object
  ("block P"): the user tells the blocks apart that way ("put Z on P").
- **2026-11-12** — Named all on one sheet, one call, one string each ("blue
  block P"): on a frame of 9 objects, 40.6 s one call each, 19.9 s on a sheet
  as three fields each, 3.7-7.7 s as strings. Colours as good or better (red
  U as red, not pink); letters worse — it wrote "block P" for D and for 7.
  The user: colour is enough for now.

## skills.py (was deployments/tabletop/skills.py)

### `outline_again` (also: `Scenario.outline_again`, `OUTLINE_AGAIN_PX`)

- **2026-11-17** — With a lamp turned on the surface scan outlined the blue
  scoop as an 8 mm scrap, and every look for its walls' height aimed at the
  scrap ("it is hidden there", four looks, refused). Before a pick by a wall
  it is now found by name with the grounding model and outlined in its box,
  its position moved to that outline's middle (64-69 mm from the scan's). On
  a 320 px crop qwen2.5vl's box came back at 372-911 px: the whole frame.

### `_vacated`: a put-back keeps the spot it goes back to

- **2026-11-13** — "put the blocks back" after a tower: the planner got it
  right (pick blue, place at where_blue_block_d_was, its table spot then).
  But the pick of blue off the tower made that spot the tower's, at table
  height, and the place there was refused ("green leaves 0 mm"), in the plan
  made again too. A pick now keeps the spot when its next step in the plan
  goes there (headed_for), and when it comes off a stack (a spot inside the
  stack could never be put on).

### `skill_identify`, `known_sizes`

- **2026-11-12** — A height the user gave was replaced by the known size at
  the next scan, and a name lost its digits ("dice 6" became "dice"), unlike
  the vision model's. Both fixed in the review that day.

### `skill_freedrive`

- **2026-11-12** — Went by disable_torque, which on an SO-101 lets the arm
  fall. Now only on an arm with capabilities.Freedrive (a UR).

## controller.py (was deployments/tabletop/llm_controller.py)

### The planner's stacking guidance and examples

- **2026-11-13** — Two examples: finishing a tower already part built, and
  finishing one with a block in the jaws. Replayed against gemma4 on the day's
  own calls: with red taken off and blue in the jaws, 4 steps finishing the
  tower (3/3). A fresh tower, whose call had come back in an impossible order
  (blue on green, then green on red), came back bottom-up (3/3). With blue
  already on green, it still moves blue aside and rebuilds (6 steps, valid;
  2 would do).
- **2026-11-12** — "Put the green block on the red block and then the blue
  block on top" was split into two sub-goals, a planning call each (35 s and
  24 s, the second after the first had run). A sequence of moves is one
  sub-goal now: one plan of all four, 22 s.
- **2026-11-12** — Written with stacking: a goal about a group ("stack the
  blocks") is kept as one sub-goal, so its plan is made whole, and the
  examples show "put Z on P", a whole tower, and the top block taken off. Offline with gemma4:31b: a
  12-action tower; and "put the green block on the red block and then the
  blue block on top", "make a tower with red at the bottom, then green, then
  blue" and "stack the green and the blue blocks on the red one" all planned
  right (24-36 s a plan).
