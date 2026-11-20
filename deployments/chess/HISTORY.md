# Chess — history

Why the code is as it is: the runs on the rig behind its values and fixes,
with their dates, by file and by name. The comments in the code say what each
value or step is for; the story of how it was found is here. What is still
wrong is in NOTES.md, not here.

## profile.py (was profiles/so101.py)

### `module`

- **2026-11-17** — Chess's rules for the SO-101 (how wide to open among the
  squares, how hard to hold a wooden piece, the servo's tolerances on a cream
  pawn, the memory's piece swaps, the 11-16 demo's descend_unconfirmed) moved
  here from the arm's robot_config, values unchanged, so another task no
  longer takes them on (mrcf/HISTORY.md, profiles.py). The tabletop's rules of
  11-16/17, which had reached chess through the arm's config untried, no
  longer apply to it.

## board_log.py

### `piece_name`

- The `<colour>_<type>_<square>` name is the one pieces had before there was a
  memory, kept so plans and few-shot examples read the same.
- The first live scan's guesses called a king and a rook "pawn". Why the type
  is in the name only once something better than the vision model's guess
  vouches for it.

## chessboard.py

### `module`

- On this rig the generic pipeline's vision model named the robot and the
  furniture while missing the pawn.

### `_tile_prompt`

- The template's placeholders read `"colour": "white", "type": "pawn"` — and
  the first live scan called five of six pieces, kings and black rooks among
  them, white pawns. A small model copies a literal from the template.

### `_TILE_STRIDE`

- The first version of the occupancy check showed the model ONE photo of the
  WHOLE labelled board and asked it to list every occupied square. Measured
  live, against a board with a single piece confirmed (by eye) on c4, that
  answered "e4" 4 times out of 5 — not noise around the right answer, but a
  specific wrong one: e4 is chess's single most common opening square, and the
  model was reproducing that prior instead of reading the small printed text.
  Rewording the prompt to warn against exactly that, and separately making the
  labels far more legible (solid white backing box instead of red-on-board),
  did NOT fix it — still wrong most of the time either way. (also:
  classify_tile, discover)
- What did fix it: showing a crop small enough that it no longer looks like a
  recognisable chess position. The same "read the label" task, run on a
  roughly-3x3-square tile instead of all 64, answered correctly 8/8 on the
  piece's own tile and correctly reported no piece 12/12 across several empty
  and board-edge tiles (the latter checks the board's decorative wooden border
  — visible at the edge tiles — doesn't itself get misread as a label).
- A classical-CV colour pass was tried for localising pieces before cropping
  around them, and dropped: a wooden chess piece and a light wood square are
  close enough in colour that no threshold tried separated them, and the
  board's own decorative border skewed every edge square's baseline regardless
  of whether it held a piece. Tiling sidesteps needing that step at all. It
  costs more calls than the old one-shot design for a sparse board, fewer
  tokens per call, and was the one actually tested as reliable.

### `classify_tile`

- Answers outside the vocabulary used to become "white" and "pawn", and on the
  first live scan five of six pieces — a king, a black rook, a knight among
  them — came back as white pawns. A wrong type was worse than none: it set the
  hover height, and a king approached at a pawn's hover has the jaws at its
  shoulders.

### `_WHITE_BELOW`

- **2026-10-20** — From then on every scan logged the cream pawn black: from
  the camera's old place the cream read 30-43 deg of hue, the brown 13-26; from
  the new one auto white balance put the cream at 0-15, and washed out to near
  white it has no hue at all. Why not a fixed hue. (also: _BLACK_ABOVE,
  piece_darkness, colour_of)
- **2026-10-27** — The thresholds rest on the saved frames since the camera
  moved (2026-10-17) and the live scan of 2026-10-27: cream pawn -0.2 to 0.42,
  red-brown queen, rook and others 0.77 to 1.57.
- **2026-10-27** — The brown queen's lit top read L 167 and the cream pawn 163,
  both lighter than the dark squares (125). Why not lightness.

### `_WASHED_SQUARE`

- **2026-10-22** — 8% of the light squares' pixels at the camera's maximum one
  morning, 24-28% that afternoon. (also: _OVEREXPOSED, lighting_report,
  lighting_warning)
- **2026-10-27** — 33-36%: 12-13 light squares more than 30% washed out, all
  at the window end, the cream pawn on b7 among them outlined as a scrap.

### `realign_stale_reference`

- **2026-10-27** — The camera had crept 2.2 mm (1.3 px), and the scans without
  the empty-board photo took a queen's shadow for a piece.

### `measure_pieces`

- Before measure_pieces, the scan used the centre of the square the tile
  classifier named as the piece's position: a hand-placed piece is rarely
  centred, and nothing about its size was known at all. Without the segmenter
  the scan is still exactly that.
- **2026-10-22** — The user: pieces stand anywhere; the squares' names are only
  a convenience. Why an outline that has not changed from the empty-board photo
  is the board, wherever it is and whatever its shape, with no rules about
  squares.
- **2026-10-27** — The squares either side of a queen standing on the d4/e4
  line differed from the photo over 44% and 68% of their outlines (the queen
  88%), were kept as pieces, and one took the queen's place in its square. Why
  the grid-line rule still runs on an outline the photo does not match.

### `label_pieces` (also: `label_unknown`, `LABEL_KINDS`)

- **2026-11-03** — On the full starting position, one crop a call, two looks a
  piece, took 4.5 minutes a start, and on the 12:55 frame got 10 of 32 types,
  23 of 32 colours, and 8 real pieces "not a piece". Every piece on one upright
  sheet, one call (scratchpad eval on the 12:18 and 12:55 frames): 16-27 s,
  30-32 colours, 19-22 types, no "not a piece" in six runs; a sheet of crops
  framed round the outline (tops cut off) did no better on type than one crop.
  Asking for "white pawn" rather than a field each halved the answer (248
  tokens against 524) and the time. Why one sheet, short answers, a second
  look only for "not a piece", and only what the memory cannot name.

### `merge_sources`

- The classifier read a black rook as white. Why the camera's colour, from the
  piece's own pixels, wins over the classifier's.
- **2026-10-21** — The vision model put pieces on empty squares e1, e2, h5 and
  a4, a pawn the only piece on the board; the camera's outlines did not. Why a
  classifier-only piece is not confirmed when the outlines ran.

### `discover`

- A single knight was enough to fail a scan whose board was fully in frame: the
  corner finder needs all 49 internal corners, and a tall piece seen obliquely
  hides the one behind it. Why the last clean lattice is cached.
- **2026-10-19** — A piece on e6 hid the far row and the board's border stood
  in for the near one: the corner finder found a 7 x 7 grid one row off. Why a
  lattice that puts a1 on a light square gives way to the last good one.
- **2026-10-27** — No corner cache (the camera's self-check deletes it when the
  camera has moved), and a queen on d4/e4 hiding corners: no board at all. Why
  the board's tags find the lattice when there is no cache.

### `board_lattice` (also: `_lattice_ok`, `_LATTICE_MAX_RMS_M`)

- **2026-11-04** — After e2-e4 on the full board the corner finder found all
  49 corners, in the wrong order: 41.5 mm squares, 10.2 mm off a regular grid.
  It passed the a1-dark test, went into the cache, and named the squares of
  every scan after it: two empty squares and two corner tags outlined as
  pieces. The board's tags gave the lattice on all 19 frames of 11-03/04,
  within 0.1-0.5 mm of the corner finder's when that was right, in 0.1 s
  against 1-2 s; the corner finder failed on most full-board frames. Why the
  tags come first, and nothing that does not fit a regular grid is used or kept.

## config.py

### `__getattr__`

- **2026-10-22** — A copy of the robot settings taken here at import kept the
  old homography placing every square until a restart, after
  overhead_calibration had re-posed a moved camera. Since then config.X is
  robot_config.X at the moment it is read. (also: _ROBOT_SETTINGS)

### `ROBOT_PLAYS`

- Squares used to be named "rank 1 nearest, files across" with the files
  running the other way — a mirror image of the board no player ever sees, so
  "e4" meant the square a player would call d4.

### `BOARD_TAG_IDS`

- **2026-10-20** — The tags went on by hand, half a turn from the printed
  diagram: 2 at the a1 corner, 3 at h1, 5 on the a-file side, 4 on the h-file
  side; 0 and 1 at the rank-8 corners. (also: BOARD_TAG_ROUGH_XY_M)

### `TALLEST_PIECE_M`

- The classifier named a king a pawn on the first live scan, and a pawn's hover
  puts the jaws at a king's shoulders.

### `APPROACH_RULES`

- **2026-10-21** — Emptied: the arm's approaches (mrcf/policies/parallel_jaw.py,
  Approach; robot_config NEAR_BASE_M / FAR_REACH_M) now choose how to come at a
  square by its distance from the base — near_base, top_down, reach_out — and
  keep joint margin themselves. What the rules here did:
  `{"ranks": (8, 8), "tilt_deg": 40}, {"ranks": (7, 7), "tilt_deg": 30},
  {"ranks": (6, 6), "tilt_deg": 20}, {"ranks": (1, 2), "files": "bcdefg", "tilt_deg": 20}`.
  The far ones asked for about the least lean that reaches anyway (offline:
  rank 6 20-30, 7 30-45, 8 45-60); the near one held 20 deg on rank 2 where 0-5
  reaches once the hover is low (and with jaws that open along the arm, which
  these do not — robot_config JAW_AXIS — more lean is more of a tall piece's top
  swung toward a jaw).

### `ARM_KEYWORDS`

- On this rig the VLM named 'robot' and 'chair' — the arm itself and the
  furniture behind the table — while the actual pawn went unreported.

### `WATCHERS`

- **2026-10-22** — Since then the arm runs the motion guard (the sentry) itself,
  on every connection (its robot_config's MOTION_GUARD; RigEnv._start_sentry,
  from SO101Env.connect) — so the tours, checks and rest.py are watched too,
  not just this controller.

## game.py

### `on_start` (also: `set_up_for_new_game`, `_reset_instead`, `RESET_MISSES`, `RESET_FROM_SQUARES`)

- **2026-11-16** — The user, mid-demo: "surely when you do the scan and see the
  chess board is reset — surely I don't need to tell it to restart?" Before,
  `/resume_game` on a board set up again listed every square that differed
  and waited for the old game's position, and "done" said the look was no
  legal move. Now the session's first look carries on the saved game when the
  board shows it (or one move on, or with the arm's owed move made). Pieces at
  the start begin a new game: the same side and strength, the arm's first move
  waiting for "go" or "done". Mid-game, "done" and `/resume_game` on pieces
  set up again begin a new game and make the arm's first move. A look may miss
  two pieces (RESET_MISSES). Only a game two moves' worth from the start
  (RESET_FROM_SQUARES 4 squares) counts as set up again: after 1. e4 a board
  at the start is more likely the pawn put back, and is set up as the game has
  it. A legal move that brings every piece home (1. Nf3 Nf6 2. Ng1 Ng8) is a
  move. Verified offline only.

## play.py

### `module`

- **2026-11-04** — The first opening run on the arm: e2-e4 placed 0.4 mm from
  its aim in 17 s, then the look found pieces on h3 and h4 (the scrambled
  lattice above) and the run stopped without moving anything more, as meant.
  Names had swapped between tracks again that morning ("white_piece_d1 moved
  e4 -> d1"): why moves go by square, not by name.

### `make`'s dropped piece

- **2026-11-04** — The e5 pawn, carried back to e7, left the jaws 8 mm above
  the board and stood on e6, 16 mm off its centre; the run stopped with the
  board one square out. Why a place that loses its piece is looked after, and
  a piece standing near its square carried on to it.

### `DISPLACED_MM`

- **2026-11-04** — A pawn squeezed out of the jaws on e4 landed between d1 and
  e2, 21 mm from d1's centre; the queen there was not outlined, so d1 still
  read "occupied" and the occupancy alone looked right but for e4. Pieces
  stood by hand or put down by the arm read 1-11.5 mm off.

## piece_geometry.py

### `GRID_EDGE_PIECE_MAX`

- **2026-10-22** — A cream pawn near e4's corner: 0.35 of a square, 0.47 on the
  grid, 13.6 from the square's colour (Lab, 0-255) — and missed by four scans
  in a row. Quarters of bare squares cut from the same frame: 0.25 of a square,
  0.56-0.58 on the grid, 1.0-3.6 from their own colour. (also:
  PIECE_MAX_SQUARE_FRACTION, UNLIKE_SQUARE_LAB)

## piece_vision.py

### `module`

- The circle detector this replaced found pieces by Hough circles: "a piece
  seen from above is round". That holds for a sharp pawn seen straight down and
  for nothing else on this rig — the camera looks at an angle, so a pawn is a
  head on a body, a knight is not round from anywhere, and once the lens
  drifted out of focus nothing was round. It returned nothing on every real
  servo attempt. A colour rule was also tried and fails for the reason the
  original notes gave: a wooden piece is roughly a blend of the two board
  colours, so in colour space it sits ON the line between them.
- Also tried, and lost: a plain homography from the detected corners, no lens
  model — right near the corners, drifting half a square off by the frame edge.
  The drift was corrupted corners (a piece or the jaw) skewing the fit, not
  distortion — the lens turned out to be mild.
- Also tried, and lost: one frame-wide reference per square colour — lost
  pieces in 12 of 34 frames, because the frame's edges are darker than its
  middle.

### `_UNIFORM_SHARE`

- **2026-10-19** — A sunlit patch at the top of the frame failed the reference
  by 49 and, rejected, left the piece beside it without the board round it to
  be judged. (also: _LIT_BESIDE)

### `_PLAIN_UNIFORM`

- A varnished dark square mirroring the window came out bluish grey, L 138,
  nearer the light squares — judged as covered, the whole square was flagged
  and pulled the piece's centre 65 px.

### `_JAW_TOUCH_PX`

- The jaw's own edge or shadow, alone in view, were servoed onto. (also:
  _JAW_TOUCH_MAX)

### `_board_pose`

- On one frame findChessboardCornersSB returned a 3x3, 3x6 or 6x3 grid from
  call to call, even single-threaded, which made the servo see a piece on one
  call and nothing on the next.

### `_pose_from_corners`

- When a piece corrupted a whole row of corners, a bare homography found a
  self-consistent fit one full square off.

### `_residual_and_pose`

- Unpacking a negative square key with // and % returned the wrong square —
  with the wrong light/dark parity, so it was judged against the other colour.
  Why the key is offset.
- Taken over every square in view, the typical colour included the virtual
  squares extended over the dark border and the table; on a view with plenty of
  border that dragged "typical light square" to L=87 against a real 199, every
  light square failed the test, and a piece standing on one could never be
  found.
- Skipping a square whose median a piece had dragged toward itself hid the very
  piece standing on it: a pawn filling a dark square at servo height was
  invisible.
- Judged against a frame-wide colour, the darker corners of the frame read as
  objects, and where the jaw covered a square its edges and shadow did the
  same. Why a covered square is judged against its diagonal neighbours, and not
  where the jaw covers it.
- **2026-10-19** — own_colour: a square's plain median was dragged toward a
  piece over part of it, and the square's own visible part, judged against
  that, came out as piece: the "aligned" centre sat beside the piece and the
  jaws closed on nothing.
- **2026-10-19** — From the servo's hover a rook on e6 covered 90% of its
  square, and skipping that square hid the piece completely. (also:
  _COVERED_BESIDE, _COVERED_DIAGONAL, _COVERED_MAX_JAW)
- **2026-10-19** — The neighbours' colour, 20+ off a covered square under
  uneven light, flagged the visible corner of the square with the piece and
  pulled the "aligned" centre off the piece. Why a covered square is judged
  against what shows of it, when enough does. (also: _REMNANT_NEAR,
  _REMNANT_MIN_PX, _REMNANT_MIN_SHARE)

### `_blobs`

- Returning one fragment of a piece or the other at random made the reported
  point jump ~50 px between frames. Why fragments are grouped.
- Judged by colour, the jaw rejected nearly every piece: dark grain on the dark
  squares has the jaw's colour too. Why the jaw is found by place (jaw_mask).
- The ring check is what rejected the empty-board false detection below the
  board edge.

### `measure_piece`

- Before the backend's camera model, the piece under the jaws was measured from
  the board's own pattern only: sizes, but no position.

## skills.py

### `scan_scene`

- **2026-10-27** — Without a photo of the empty board, a surface scan took the
  board's squares for objects — 70 of them, each remembered, and every pick
  refused for want of room between them.

### `_OVERHEAD_SUSPECT_MM`

- **2026-10-22** — Every hover needed 22-26 mm, where a calibrated one needs
  1-3. (also: _calibration_suspect, _log_pick)

### `_log_pick`

- picks.jsonl recorded only pass/fail, and a failure could not be traced. Why
  each attempt's retry reason and the timing are logged.

### `_vacated`

- **2026-10-29** — Putting the piece back on the square it was picked from made
  the controller rescan first — to park and back, the piece in the jaws — and
  on that swing one slipped out (NOTES.md, "Putting a piece back on its own
  square rescanned with the piece in the jaws").

### `_stand_up`

- **2026-10-28** — Out of the park pose the policy held a 10 deg lean at e3 and
  the head squeezed out of the jaws; from home it comes straight down there, as
  the first stand-up did.

## tests/capture_piece_frames.py

### `main`

- Closing the jaws to centre the piece leaned on get_gripper_width, which reads
  ~0.7 mm for a held 20 mm piece, and put the label ~35 mm off. Why the label
  is the open gap's centre, read without moving the jaws.

## play.py

### `carry`: a look before a place tried again where something moved

- **2026-11-12** — A carry tries a refused place again from where the arm is.
  When the sentry saw something move by the square (a piece knocked onto
  it), it now looks at the board first; a square taken now has the piece put
  back where it came from, not set down beside it to grip again (_relay).
