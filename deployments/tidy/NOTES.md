# Tidy — notes

What this deployment, the SO-101 putting things away, has shown that its code and config cannot say. One entry per
issue; delete an entry once it is fixed (git keeps it). What belongs where:
[DEPLOYMENT_SPEC.md](../../DEPLOYMENT_SPEC.md#what-you-learn-and-where-it-goes).

Each entry: **Seen** (the date, what, the flight recorder's incident),
**Cause**, **Workaround**, **Status** (open, or fixed and where).

The table task's own limits, on any deployment: mrcf/tabletop/NOTES.md.

## Limits

- **Run since 2026-11-16 (as the tabletop deployment until 11-17).** Two
  wooden figures, a 3-pin plug and a measuring scoop (by its wall,
  pick_by_rim) picked and put into a clear plastic tub (2026-11-17), the
  figures first try once the surface was measured. Soft things and things
  lying down: Open, below.
- **Heights from the wrist camera.** An unmeasured object's height is looked
  for from beside it (HEIGHT_FROM_VIEWS), refused while the looks disagree.
- **The catalogue is new (2026-11-18), run live since 2026-11-19.** Kinds are matched
  by foot, outline and colour from above; two kinds alike in all three would
  be taken for one. A kind found lying is not stood up: its pick gives up,
  and the task loop sets it aside (stand_up is not wired here yet).
- **The scoop is picked by its handle** (HANDLE_FIRST, 2026-11-19): held by
  its 1 mm wall it read empty while held (2026-11-18), so the handle is tried
  first and the wall only without one. Nothing is ever finished by hand-steered
  moves: a failure is reported and fixed in code (user, 2026-11-19).
- **The wrist camera's shift goes stale.** 2026-11-19 pm to 11-20 am every
  wrist look was thrown away ("a part of it, not all of it"), in any light:
  its shift against the overhead camera (GRIPPER_CAM_SCAN_SHIFT_M) had moved
  6 mm since 11-16. Refitted from the objects on the table
  (tools/calibrate_wrist_on_objects.py --deployment tidy, then --fit ...
  --apply), the next run put all four in first try. When the controller says
  "[Calibration] ... fit it again" at its start, do that first. The light
  still matters to the overhead scan's names and outlines (evening lamps split
  the plug and the scoop), less than it seemed.
- **A white thing on the white table** (the plug) is not outlined by the wrist
  camera from above and is no ruler for the calibration; it is aimed by the
  overhead camera and checked between the jaws at the grasp.
- **Far out and low is marginal.** 31-34 cm out, gripped under ~19 mm, a
  descent arrives or not on 3 mm of grip height (tools/so101/replay_descents.py:
  102 of 110 replayed descents arrive as tidy has it set, the misses all at a
  16 mm grip). A figure lying on its side there (23 mm tall) is not picked.
- **Its task, planned in advance** (config.py TASKS, PLANNER False): "tidy up"
  or "go"; "put ... into ..." too; nothing else is planned, and a failed step
  is not planned again. A thing the scan has not named as the task calls it is
  left out (two figures both named "red cap", 2026-11-19 18:13).
- **No board to calibrate against.** Its layout is the bare table: the wrist
  camera's shift is fitted from the objects on it
  (tools/calibrate_wrist_on_objects.py --deployment tidy, or --from-runs);
  its mount against the table target sheet
  (tools/so101/calibrate_gripper_pose.py --wide --target tags --deployment
  tidy; the sheet tools/tag_sheets/table_target_a4.pdf, printed at 100%, laid
  flat and uncut). That fit starts from the last good calibration (a board's
  or a sheet's): the arm's first one is still on the board.

## Open

- **Into a container: let go higher than it need be, and things land lying down.**
  - **Seen:** 2026-11-17, the SO-101, four objects into a clear plastic tub
    (runs/flight/20261117_14*, _15*). The walls read 82, 92 and 106 mm by the
    wrist camera (place_in's rim look), and once not at all; the overhead
    camera puts the rim at about 70. Lowered inside, the figures were let go
    28-30 mm over the floor and lay down; one place's lowering stalled at its
    start and it was let go from 177 mm; the plug, the walls unmeasured that
    run, from 196.
  - **Cause:** the rim look reads high by design (the higher of what fits, a
    lip's edges lining up a little higher too); the gripper's body is kept
    over the rim (free_depth_m above the jaws), so the jaws go no lower than
    rim + clearance - free depth; the stalled lowering was the slow,
    contact-limited descent of a grasp (an ordinary move now).
  - **Workaround:** none needed for sturdy things.
  - **Status:** mostly done. Its walls' height is kept across restarts
    (containers.Keeper, rig/so101/tidy_containers.json), and since 2026-11-19
    it goes down inside leaning as far as being over the spot takes above the
    walls, no more than 20 deg among them (CONTAINER_LEAN_ABOVE_WALLS): runs
    38-41 let the figures go 37-40 mm up, the scoop and the plug — which can
    swing below the jaws — 85-90 mm up, where before 13-18 cm.

- **A soft thing is crushed.**
  - **Seen:** 2026-11-17 12:09, a wrapped sweet: squeezed flat and held fast,
    the jaws opened only by writing the gripper's goal by hand.
  - **Cause:** GRIPPER_HOLD_PCT 10 (profile.py, from chess's wooden
    pieces): the goal stays 10% of travel past contact, and a soft thing
    yields until it gets there.
  - **Workaround:** do not ask it to pick soft things.
  - **Status:** open: a hold that eases off what gives way.

- **A thing knocked onto its side is not picked.**
  - **Seen:** 2026-11-17 14:20-14:28, the green figure, knocked over by a
    missed close 34 cm out: closed on along its length from straight above
    twice; then the looks for its height, near the table's edge, drove the arm
    into the table (since stopped by the SO-101's floor guard).
  - **Cause:** the pick plans for an upright thing: its jaws on the arm's own
    line, its height from looks that fit an upright solid.
  - **Workaround:** stand it up by hand.
  - **Status:** open; the jaws turned across a long footprint is built
    (CLOSE_ACROSS_LONG), not tried on the arm.

- **Built, not yet run on the arm.**
  - 2026-11-17: held as the wrist camera sees it, when the gripper reads empty
    (HELD_BY_CAMERA; replayed on the scoop's lift, offline). Watch [Held].
  - 2026-11-19: a push into the table sent as the plain step (FLOOR_CAPS_PUSH;
    the plug's reseats, run 40); the rest tool's rest without cameras
    (tools/so101/rest.py connected).
  - Run on the arm since (2026-11-20): its task with no planner ("tidy up"),
    a set-aside thing found again by its name, the rest tool's look before it
    moves, the jaws opened to the kind's width, an empty close after the lift
    called empty (HELD_CLOSED_M), the lower grip (MIN_GRASP_HEIGHT_M 11 mm).
