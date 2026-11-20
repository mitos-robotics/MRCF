# Toolbench — history

The UR5 putting hand tools into a box on its table: the runs behind this
deployment's own values, with their dates. The table task's code it runs is
the framework's (mrcf/tabletop), with its own history
(mrcf/tabletop/HISTORY.md). What is still wrong is in NOTES.md.

## The deployment

- 2026-11-20: made from deployments/tabletop (tools/new_deployment.py), the
  UR5's letter blocks, on the same table and camera. The user's choices:
  tools with a handle (screwdrivers, a hammer) and compact boxy ones (a tape
  measure); put into a box (a cardboard one on the table); every command in
  plain English, planned by the model (PLANNER, no TASKS); named toolbench
  ("tools" is the repository's folder of rig scripts).
- 2026-11-20: the blocks' KNOWN_SIZES_MM (30 x 30 mm) not taken over: no
  sizes are written in by hand; what a tool measures comes from the cameras
  or from the user saying so.
- 2026-11-20 (the rig, before any toolbench run): the overhead camera had been
  moved since 2026-11-13 — the wrist tag 191 px (~10 cm) off where the
  calibration put it, the picture slid 187 px and turned 6.8 deg against the
  empty-table photo. Calibrated again by hand (tools/
  calibrate_camera_by_hand.py, reports/camera_by_hand/20261120-141424): tag 1
  on link 5 (the wrist's cap), 25 poses, 1.14 px; left one out at a time, the
  table moves 0.3 px typically, 2.6 px at worst; a fresh look at a new pose
  0.7 px. Empty-table photo taken again (14:19; the old one kept as
  rig/ur5/surface_reference.20261113.png).

- 2026-11-20 14:32 (runs/flight/20261120_143205): the first run, "pick and
  place all the tools into the box" — nothing picked. The jaws closed 116 mm
  above the hammer twice (no height: planned for 150 mm); it was outlined as
  a 10 mm piece of itself and then lost; the planner then read every tool as "wider
  than the gripper opens" by its round foot, and stopped. The tools and the box
  lay along the camera's bottom edge, the hammer's handle out of view; the
  camera sent torn frames (every sentry pause) and 19 level jumps in 6 min.
- 2026-11-20 14:53 (runs/flight/20261120_145351): the second run — the arm
  never moved: every lying pick raised on the UR5's orientation search
  (mrcf/HISTORY.md, policies/lying_pick.py). What it showed otherwise: the
  camera, plugged into the Mac directly, sent no glitched frames; outlined
  again by name, the hammer (248 mm) and a screwdriver (176 mm) came out
  whole; the planner, told the tools' widths across, planned them all.
- 2026-11-20 15:07 (runs/flight/20261120_150709): the third run — "pick up
  all the tools and put them in the box" — the planner answered "Can the
  brown_box be reached?" and stopped: the scan put the box (by its far flap) at
  x 0.78 m, out of reach, and most of the box stands outside the arm's box
  (GRIPPER_BOX, 0-0.75 m). Then (the user): a task planned in advance, the
  planner off — TASKS "put the tools away", PLANNER False; TASK_WORDS_BY_KIND
  and CONTAINER_TALLEST_M 0.40 in the profile.
- 2026-11-20 15:24-18:06 (runs/flight/20261120_152453 to _180626): about
  twenty runs of "put the tools away". What they showed, in order (the
  framework's changes: CHANGELOG.md, mrcf/HISTORY.md):
  - The jaws came down along the hammer's handle, onto it, at 50 N: the
    UR5's JAW_SLIDE_AXIS had been a guess, a quarter turn wrong. Corrected
    and checked on the rig (tools/check_jaw_direction.py, 16:07); a pick is
    now refused on any arm until that is checked (mrcf/arm_facts.py), and a
    lying grip's finger footprints are checked against the outline first
    (mrcf/policies/grasp_sense.py).
  - The box's wall height, saved as a guess, was read back as measured and
    the hammer lowered inside on it; fixed in containers.Keeper. The user
    gave the walls as about 25 cm.
  - Carried out of and into the box through its wall: TRANSIT_OVER_WALLS.
  - The box outlined with its flaps: the hammer let go over a flap, and once
    missed the box. Its opening is now found by its darker inside
    (CONTAINER_OPENING_BY_SHADE).
  - The place refused for room — "clear of what is in it", "half its length
    from every wall". The user: "its a box ... there may be things inside",
    "ideally so the long edge of the object is on the long edge of the box".
    Now what is in it is ignored (PLACE_IN_OVER_CONTENTS) and a long tool is
    laid in by its outline along the box, or dropped from over the walls,
    aligned, where it cannot lie flat.
  - A false contact force (120 N, halted) with the elbow nearly straight,
    out at the box's far side: the UR5 refuses those orientations
    (ELBOW_STRAIGHT_MARGIN_DEG). The motion guard stopped the arm for the
    hammer it carried: capabilities.Carries.
  - From about 17:00 the hammer and a screwdriver went into the box, run
    after run; the second screwdriver was never picked (task_more since).
  - 18:04: the overhead camera 13 px off its empty-table photo; every scan
    stopped and nothing moved (NOTES.md).

## profile.py

### `module`

- **2026-11-20 (the afternoon's runs)** — LYING_MIN_ASPECT 1.0 (the tape
  measure to the lying pick, not the plain pick planned for 150 mm);
  OUTLINE_ONLY_PARTIAL; TRANSIT_OVER_WALLS; CONTAINER_GRIPPER_INSIDE_M;
  CONTAINER_OPENING_BY_SHADE; PLACE_IN_OVER_CONTENTS; CAMERA_DRIFT_PX 8;
  HOME_AFTER_COMMAND; CONTAINER_TALLEST_M 0.40 to 0.30 (the box about
  25 cm). Each with its reason in the profile.

- **2026-11-20 (after the first run)** — LYING_PICK, ACROSS_FOR_LONG,
  OUTLINE_BEFORE_PICK and ROBOTIQ_FORCE 40 taken on; GRIP_LOW_FROM_ABOVE
  dropped: it runs only within the side looks (a wrist camera), so it never
  ran here. HANDLE_FIRST now reaches a plain pick, through lying_pick.

- **2026-11-20** — Its rules for the UR5, taken from the SO-101's table work
  (deployments/tidy/profile.py) where their reason holds here and the arm
  needs no wrist camera for them: HANDLE_FIRST, CLOSE_ACROSS_LONG,
  GRIP_LOW_FROM_ABOVE, RELOCATE_AFTER_MISS, PLACE_IN, DEFER_FAILED. The rest,
  and why not, in the profile's docstring; CATALOGUE, QUICK_PUT_INTO,
  OUTLINE_BEFORE_PICK and PARTS_INSIDE_ARE_ITS_OWN left as candidates. None
  run on the UR5 yet. tests/baselines/ur5/deployments.json recorded with it.
