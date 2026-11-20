# Chess — notes

What this task has shown that its code and config cannot say: quirks, failure
modes, what to watch for, what was tried. One entry per issue; delete an
entry once it is fixed (git keeps it). What belongs where:
[DEPLOYMENT_SPEC.md](../../DEPLOYMENT_SPEC.md#what-you-learn-and-where-it-goes).

Each entry: **Seen** (the date, what, the flight recorder's incident),
**Cause**, **Workaround**, **Status** (open, or fixed and where).

## Open

### Black's back rank is out of the arm's reach, so most openings cannot be played
- **Seen:** 2026-11-04, `play.py`'s book on the starting position: 14 of the
  21 openings need a black knight or bishop from rank 8 (b8, g8, c8, f8),
  0.43-0.46 m from the arm's base with the board 2 cm out since 11-03; rank 8
  is past `WORKSPACE_BOUNDS` (x_max 0.42) but for a8/b8. Rank 7 is at
  0.395-0.42 m: on 11-03 the f7 pick needed a -60 deg lean and two places on
  f7 stalled.
- **Cause:** the board is 32 cm deep and the SO-101 reaches from about 0.13 to
  0.40-0.42 m. Moved out to reach d1 and e1, the far edge left its reach.
- **Workaround:** `/openings` lists what is out of reach; `/play_opening`
  leaves those lines out. Played in reach: kings_pawn, queens_pawn, reti and
  the captures queens_gambit_accepted, kings_gambit. In a game against the
  arm (`game.py`) the engine is offered only the moves whose squares are in
  reach, so it plays weaker than its strength by passing up captures on
  ranks 7-8; with no move in reach, the person makes its move for it.
- **Status:** open — the board's place is the user's choice. Predicted
  offline (`reach_map.py`'s simulated arm, optimistic far out): with the board
  1 cm nearer 20 of 21 lines are in reach, 3 cm nearer all 21 (rank 8 at
  45-60 deg leans); d1/e1 then go out of reach for the king and queen, which
  no book line here moves.

### A piece standing between two squares can be reported on both
- **Seen:** 2026-10-29. One white piece, about 17 mm from d3's centre toward d2,
  came out of a scan as two: `white_piece_d3` (outlined by the camera, its foot
  measured) and `white_piece_d2` — "not outlined by the camera". The next scan
  dropped d2's ("missed by the last scan; only the vision model ever reported
  it").
- **Cause:** the second came only from the vision model's square-by-square
  occupancy check: the piece reaches into d2's part of the image.
- **Seen again:** 2026-10-29 at 15:10, one real piece on e5. A piece only the
  vision model reported came out of the scans on a5, then "moved a5 -> a4 by
  hand". Asked "move the pawn on e5 back to e4", the planner chose it
  (`white_piece_a4`); three picks at a4 closed on nothing, and the recovery
  went on to scan again. Incident:
  `runs/flight/20261029_145747/incidents/151057_pick_pick_failed__pick_failed_after_3_attempt/`
- **Workaround:** stand pieces on their squares' centres, and name the piece
  (`/pick white_piece_e5`) rather than describe it. The doubtful one is not
  steered around, but it is offered to the planner and can be picked.
- **Status:** fixed 2026-10-29, offline (not yet on the arm): only the camera's
  outlines put a piece on the board; the vision model only labels outlined pieces
  (`chessboard.merge_sources`, `label_pieces`).

### The planner can ask for a piece that is not there
- **Seen:** 2026-10-29. "move the pawn on e5 back to e4", with the piece named
  `white_piece_e4` (its name from where it was first seen) standing on e5: the
  planner (qwen2.5:14b) asked to pick `white_piece_c3`. The safety layer blocked
  the plan before anything moved.
- **Cause:** the planning model's naming; likely made worse by a piece whose
  name says one square while it stands on another.
- **Workaround:** name the piece (`/pick white_piece_e4`).
- **Status:** fixed 2026-10-29: the planner's reply is held to a JSON schema of
  the scene's actions and names (`action_registry.plan_schema`). Offline, the
  same step against qwen2.5:14b named the right piece 5 times in 5.

### The back-rank pieces cannot be picked from the starting position
- **Seen:** 2026-11-03 at 13:34 and 13:38, the full board after e2-e4: "pick up
  d1 and put it on f3" was refused twice before moving ("jaws need 35 mm open
  here but white_piece_c1 leaves 27 mm"). The planner then asked for the same
  pick a third time, and the controller handed back to the user.
- **Cause:** two things. First, every neighbour counted as its foot all the
  way up, and an unmeasured piece as its foot too. From 34 mm up, where the
  finger tips stop for a queen, a queen is about 22 mm across and a bishop 19:
  there is room (mrcf/policies/HISTORY.md, `half_width_at`). Second, behind
  that, d1 and e1 are out of reach for the pick's first plan (SO-101 NOTES,
  "Tall pieces on d1 and e1 are out of reach").
- **Status:** the room part fixed 2026-11-03, offline (not yet on the arm). A
  pick the feet crowd now goes over the piece to measure it and the neighbours
  in the jaws' way, then decides, and the neighbours' shapes are kept. On the
  13:39 scan, c1, c2, b7, c7 and b8-g8 now go over to look instead of being
  refused. d1 and e1 are still refused before moving, now for reach. Neighbours
  the wrist camera cannot measure from the hover (4 of 9 in a replay) count as
  their feet, and the pick is then refused after looking.
- **Seen live:** 2026-11-03 at 17:1x. The board was moved 2 cm out and the
  overhead camera re-aimed. The startup's self-check took that for a moved
  camera, hovered once (23.4 mm apart) and ran the tag tour by itself over the
  full board: 26 hovers, no trouble. The wrist camera's scan shift then had to
  be fitted again from the tour's wrist frames (SO-101 robot_config). With
  that, "pick d1" went over the queen and found her foot 2.1 mm from the scan.
  But from straight over her the view cannot tell her height, and it outlines
  only part of the c1 bishop. Both stay as the worst case (planned as a king,
  gripped at 27 mm; c1 as its foot), and it was refused after looking twice:
  "c1 leaves 24-29 mm", 35 needed. Still open: a crowded piece has to be
  measured from beside it, where the look tour measured neighbours' heights.
- **Status:** fixed 2026-11-03 at 17:44, on the arm: d1 to f3, picked first
  try in 15.8 s, put down 1.7 mm from f3's centre. Measuring from beside her
  was built and tried (it measured c1, 60 mm), but this wrist camera cannot
  measure a tall back-rank piece from anywhere it reaches. What did it: a
  piece not yet measured is gripped as high as the shortest piece allows
  (`config.SHORTEST_PIECE_M`, 35 mm), and counted as slim as measured pieces
  are from there up (`config.UNMEASURED_SHAPE_SHARE`). It fits before the arm
  moves. Still refused at 17:4x, by 1-4 mm: c1 (beside the queen) and e8.

## Fixed

### A scan after a move named the squares from a scrambled lattice
- **Seen:** 2026-11-04 at 10:06, the first opening run: after e2-e4 the scan
  read 41.5 mm squares (40.0 before) at a 10.2 mm grid residual, outlined h3,
  h4 and two corner tags as pieces, and the run stopped.
- **Cause:** with the e-pawn moved, the corner finder found all 49 corners in
  the wrong order. The lattice passed the a1-dark test and was cached, so the
  next scans reused it.
- **Status:** fixed 2026-11-04: `chessboard.board_lattice` takes the board's
  tags first and refuses any lattice off a regular grid (HISTORY.md). On the
  arm at 10:25: 40.0 mm squares, 1.0 mm residual, 32 pieces.

### The d1 queen, and sometimes f1 or f2, dropped out of scans
- **Seen:** 2026-11-04, the starting position: the queen was missing on every
  frame replayed with the right lattice, f2 or f1 on some.
- **Cause:** `keep_outlines` let an outline of her head, inside her outline,
  replace it; its foot read 23 mm off, past the board's edge. And one outline
  took in f1 and f2 together (mrcf/HISTORY.md).
- **Status:** fixed 2026-11-04, offline: 32 of 32 on the three frames of that
  morning, nothing lost on 16 frames of 11-03.

### On a full board the wrist camera took a neighbour for the piece
- **Seen:** 2026-11-03, `move e2 to e4` on the starting position. From over
  e2 the wrist camera sees the whole first rank. The e1 king stands in front
  of the e2 pawn, and the wrist servo locked onto it: it pulled the arm 1.5-3 cm
  toward the base until it stalled. The look measured a 25 mm "pawn".
- **Cause:** the servo and the measurement each took the outline nearest
  where they expected the piece, knowing nothing of the others.
- **Status:** fixed 2026-11-03, on the arm. The wrist camera is located
  (`GRIPPER_CAM_POSE`) and told where the scan has every piece
  (`perception/expected_view.py`). From a pick's hover it found the right
  piece within 0.1-3.5 mm on all nine looked at, b1 included, which before
  measured the c1 bishop. An outline of only part of the piece, such as a
  pawn's head, is not taken for it. Look tour `20261103_110243`. Then on the
  starting position, e2-e4, d2-d4 and Nb1-c3 each gripped at the first attempt,
  at a -15 deg lean, 16-20 s a pick: the foot found 0.6, 0.3 and 5.1 mm from
  the scan's (`logs/picks.jsonl`, 12:23-12:55).

### Every start spent 4-6 minutes naming the pieces, and named most of them pawn
- **Seen:** 2026-11-03, the full starting position: each start asked the
  vision model about every piece twice, one crop a call (64 calls, about 4.5
  minutes). The 12:55 scan called every white back-rank piece but b1 a pawn.
  Scored against the known position on that frame (one look, 190 s): 10 of 32
  types right, 23 of 32 colours, and 8 real pieces "not a piece".
- **Cause:** a vision model's turn costs seconds whatever it is shown, and each
  crop was framed round the outline the overhead camera sees: mostly the foot.
  The tall pieces lean away from their feet in the frame, and their tops were
  cut off. The names were asked again at every start, though the memory had them.
- **Status:** fixed 2026-11-03, offline (not yet on the arm). The scan matches
  the memory first and asks only about pieces it cannot name (new, moved by
  hand, back, never named): a restart on the board as it was asks nothing, and
  the scan took 3.4 s. The ones it asks about go on one sheet, each cut upright
  from its foot to the tallest piece's top (`perception/tiles.py`), in one
  call: 32 pieces in 16-27 s, 30-32 colours right, 19-22 types, no piece
  called "not a piece" in six runs. "Not a piece" now needs a second look to
  agree. The log says what it called which squares in a few lines. The types
  are still often wrong (kings and queens called rooks): they stay the vision
  model's guess, which names and picks do not trust. The old one-crop guesses
  were cleared from the memory (`object_memory.json.bak-20261103-labels`).
- **Seen live:** 2026-11-03 at 13:29: one 19 s call for 32 pieces, and the
  restart at 13:29:50 asked nothing. But every piece came back a pawn. On that
  frame the model answers so in 9 of 9 calls; the model server's log shows it
  took the image, and a blank sheet comes back all "not a piece". So the types
  from the overhead camera are no better than calling every piece a pawn
  (16 of 32). Still open: where the types should come from.

### Finding a piece again after a miss took 4.5 minutes on a full board
- **Seen:** 2026-11-03, `move e2 to e4` on the full starting position: after
  each miss the arm sat parked for about 4.5 minutes.
- **Cause:** the re-find (`skills._relocate_piece`) ran the full scan, and the
  vision model looked twice at each of the 32 pieces (about 4.3 s a look), only
  for the re-find to use the positions.
- **Status:** fixed 2026-11-03, offline: the re-find scans with
  `classify=False`. The object memory keeps the names.

### A full board was "not detected" after the camera's self-check re-posed it
- **Seen:** 2026-11-03, the first scan of a full starting position. The
  self-check found the camera about 9 mm and 1 deg from 10-31 (the board had
  likely shifted while it was set up), re-posed it and dropped the corner
  cache. Pieces hid the inner corners, and the tags' lattice was refused:
  "Board not detected", then "no empty-board photo", so there was no scan.
- **Cause:** `board_tags.measure_board` judged a1 dark by one pixel at a1's
  centre against one at h1's centre. With the pieces set up, a cream rook
  stands on each.
- **Status:** fixed 2026-11-03, offline: a1's colour is now judged by every
  square of each colour (`board_tags._a1_dark`). The failed frame, replayed:
  lattice from the tags, the photo lined up again (6.9 px), 32 pieces on the
  right squares. Not yet seen on the arm.

### Putting a piece back on its own square rescanned with the piece in the jaws
- **Seen:** 2026-10-29. `/pick white_piece_d3` then `/place d3`: "[Recover] 'd3'
  not in the current scene — re-scanning...", so the arm went to park and back
  holding the piece, and it slipped out on the way (the SO-101's NOTES.md).
- **Cause:** a scan lists only empty squares as places to put a piece, so the
  square a pick had just emptied was not one until the next scan.
- **Status:** fixed 2026-10-29: a pick now offers the square it emptied again,
  where the last scan put it (`manipulation.Scenario.vacated`, `skills._vacated`).
  Checked on the arm the same day: `/pick white_piece_e4`, `/place e4` placed it
  with no re-scan.
