# toolbench: notes

What running this task has shown (DEPLOYMENT_SPEC.md, "What you learn, and
where it goes"). Started from tabletop (tools/new_deployment.py) on its code;
its rules are this deployment's own (profile.py). The runs behind its values:
HISTORY.md.

## Limits

- On the arm (2026-11-20, afternoon): the hammer, gripped by its handle next
  to the head, and one screwdriver, into the box, in several runs. The second
  screwdriver is looked for again after the first pass (task_more) but has
  not been seen picked. The tape measure and the roll of tape were not on
  the table for those runs: both untried.
- The roll of tape has no handle and lies flat wider than the Hand-E opens:
  it wants a grip by its wall (RIM_PICK, not yet run on the UR5).
- The hammer (about 250 mm) is longer than the box's opening (about 235 mm):
  it cannot lie flat in it, so it is dropped from over the walls, its long
  edge along the box's, and lands slanted (about 12 deg off, as logged).
- No wrist camera: the box's wall height is not measured by the arm (the
  user gave about 25 cm, kept as its height in
  rig/ur5/toolbench_containers.json; a box not kept is taken as
  CONTAINER_TALLEST_M at most), and nothing looks at a grip from close by —
  held or not is the Hand-E's own feel.
- The overhead camera moved twice in one day (knocked; USB replugs): by
  18:04 its picture was 13 px (about 16 mm) off the empty-table photo, over
  CAMERA_DRIFT_PX, and every scan stopped. After it moves: calibrate again
  (tools/calibrate_camera_by_hand.py) and take the empty-table photo again
  (tools/capture_surface_reference.py).
- The camera's view ends near the box's far edge: keep the tools and the box
  where it sees them whole.

## Seen

- 2026-11-20 18:05 (runs/flight/20261120_180411): with every scan stopped by
  the camera's drift, the hammer was found by name while planning and again
  before the plan ran (controller.ensure_visible); the plan's "hammer" no
  longer resolved and the plan was blocked before anything moved. Open: a
  second look by name should keep the object the plan names.
