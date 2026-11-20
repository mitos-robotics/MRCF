# Tabletop — notes

What this deployment, the UR5's letter blocks, has shown that its code and config cannot say. One entry per
issue; delete an entry once it is fixed (git keeps it). What belongs where:
[DEPLOYMENT_SPEC.md](../../DEPLOYMENT_SPEC.md#what-you-learn-and-where-it-goes).

Each entry: **Seen** (the date, what, the flight recorder's incident),
**Cause**, **Workaround**, **Status** (open, or fixed and where).

The table task's own limits, on any deployment: mrcf/tabletop/NOTES.md.

## Limits

- **Letter blocks only, since 2026-11-12.** Picks, places and three-high stacks
  by typed command, with the overhead camera's scan. Everyday objects
  (bottles, boxes) have not been picked on the UR5 yet (hand tools have, in
  deployments/toolbench).
- **No containers.** place_in on the UR5 is toolbench's
  (deployments/toolbench/profile.py); pick by a wall and the wrist camera's
  looks are tidy's rules for the SO-101 (deployments/tidy/profile.py), not
  tried on the UR5. This deployment has no profile: none of them.
- **On the UR5, heights come from known sizes, not the camera.** The overhead
  camera measures a height only when it sees enough of an object's side for
  its height (MIN_HEIGHT_PX_PER_MM). The UR5's looks 14-29° from vertical at
  the table, too steep for that: a kind in KNOWN_SIZES_MM gets its size from
  there; anything else plans for TALLEST_OBJECT_M, and grips a short object
  above its top. A height the user gives (identify) is kept over both. The
  SO-101 looks for an unmeasured object's height with its wrist camera
  (tidy's HEIGHT_FROM_VIEWS); the UR5 has no wrist camera.
