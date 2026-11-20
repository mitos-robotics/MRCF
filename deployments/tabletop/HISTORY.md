# Tabletop — history

The UR5's letter blocks on its table: the runs behind this deployment's own
values, with their dates. The table task's code it runs is the framework's
(mrcf/tabletop), with its own history (mrcf/tabletop/HISTORY.md). What is
still wrong is in NOTES.md.

## The deployment

- 2026-11-02: written for the UR5. The user wanted it to pick and place "a
  variety" of everyday objects, not play chess. They chose named spots and spots
  beside objects as the places, and upright objects first. Built from the
  framework's surface scan, object memory and standard pick and place, with
  the chess controller's shape.
- 2026-11-17: one arm, one layout and one task — the UR5's. Its code moved to
  the framework (mrcf/tabletop); it is now a config (ARM_NAME "ur5", the
  letter blocks' size) handed to it. The SO-101's table work, which had run
  as this deployment with its rules in profiles/so101.py, is deployments/tidy.

## config.py

### `KNOWN_SIZES_MM` (also: `skills.known_sizes`, `skills.boxed`)

- **2026-11-12** — The UR5's first pick of a block planned for a 150 mm
  object (TALLEST_OBJECT_M) and closed 10 cm above it: its overhead camera
  is too steep to measure heights (0.2-0.34 px per mm of height, against the
  0.8 needed). Blocks got a known size, 40 mm from a guess; the jaws then
  closed at 30 and a place onto another block let go 1.5 cm high, so 30 mm
  (the ruler's reading still to come). The known size is a box's too, which
  the scan squares and centres by (scene_scan.observe's boxed).
