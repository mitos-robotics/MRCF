"""
toolbench's own rules for its arm, the ur5 — how toolbench drives it, laid over the
arm's facts (mrcf/backends/ur5/robot_config.py) wherever toolbench makes the
arm's env and policies (mrcf/profiles.py).

Started from tabletop (tools/new_deployment.py), which has no profile. The
rules below are taken from the SO-101's table work (deployments/tidy/
profile.py) where their reason holds for this one too: tools on a bare
table, put into a box, by an arm with NO WRIST CAMERA — so each needs only
the overhead camera and the gripper's own feel; and, after the first run
(2026-11-20 14:32: nothing picked), the ones made for it — LYING_PICK,
ACROSS_FOR_LONG, a firmer squeeze. Each was a change for this arm to try
live; with them the hammer and a screwdriver have since gone into the box
(HISTORY.md).

Not taken, and why:
  need a wrist camera     HEIGHT_FROM_VIEWS, HEIGHT_LOOKS_MAX, APPROACH_LOOK,
                          WRIST_AT_GRASP, HELD_BY_CAMERA, HELD_WHILE_STILL,
                          HELD_BY_MODEL, DIAGNOSE_MISSES, GRIP_LOW_WHEN_CAPPED
                          and GRIP_LOW_FROM_ABOVE (both only within the side
                          looks: on 2026-11-20 the second was in force here
                          and could never run)
  the SO-101's own        GRIPPER_HOLD_PCT, GRASP_TOL_M, GRASP_RESEAT_ALONG_M,
                          MIN_GRASP_HEIGHT_M, HELD_CLOSED_M, the lean rules
                          (CONTAINER_*_LEAN_*, FAR_FROM_LEAN_DEG,
                          LEAN_CLEARS_FLOOR...), DESCENT_*, TRANSIT_SETTLE,
                          TOUCH_SURFACE (its servo overloads; the UR5's table
                          is its base plane)
  the UR5 has it already  GRASP_MISS_WIDTH_M (its max opening less 2 mm)
  its objects'            RIM_PICK (thin-walled scoop), GRIP-by-wall rules
Candidates for later are at the end, commented out.
"""

# ── Picking a tool ────────────────────────────────────────────────────────────

# A tool lying along the table, its height not known, gripped across where it
# holds best and closed from no higher than it is wide there, lower each time
# it closes on nothing (GripperSpec.lying_pick, mrcf/policies/lying_pick.py):
# a thing at rest lies on its broad side, so it stands no taller than it is
# wide. 2026-11-20 14:34, the first run: the hammer, its height unknown (the
# overhead camera is too steep to measure heights, and there is no wrist
# camera), was planned for the tallest anything could be (150 mm) and the jaws
# closed 116 mm above it, twice. Lifted 2 cm first and the grip checked.
LYING_PICK = True
# ... and a compact thing too, not only a long one (LYING_MIN_ASPECT 1,
# GripperSpec's 2): with no wrist camera and no height from the steep overhead
# camera, gripped low and stepped down from its width is the grip there is —
# the tape measure otherwise went to the plain pick, planned for 150 mm.
LYING_MIN_ASPECT = 1.0

# ... by its handle first, where it has one, gripped next to where it joins the
# wider part (GripperSpec.handle_first, handle_pick.find_handle): the wider
# part of a hammer is its heavy head, and gripped beside it the hammer hangs in
# the jaws rather than turning out of them (user, 2026-11-20: "the hammer is
# one side heavy so account for that in the pick"). Else at its balance point.
# Before 2026-11-20 this was read only by a pick by the rim (handle_pick inside
# rim_pick): a plain pick never used it, and the first run's did not.
HANDLE_FIRST = True

# A Hand-E squeeze of ~46 N (0-255 over 20-185 N), not the arm's gentle ~26 N
# (ROBOTIQ_FORCE 10): steel tools, gripped near but not at their balance point,
# twist in a light grip. Nothing here is soft.
ROBOTIQ_FORCE = 40

# A long thing is told to the planner by how long it is and how narrow across,
# where the jaws close on it — not "wider than the gripper opens" by the round
# foot an upright thing is measured by (ACROSS_FOR_LONG, mrcf/tabletop/
# settings.py). 2026-11-20 14:36: the screwdriver, "73 mm across" by its foot,
# and the planner stopped with nothing picked.
ACROSS_FOR_LONG = True

# Before a pick, the tool outlined again by what it is — the vision model's box
# round it, the segmenter in that (OUTLINE_BEFORE_PICK, ~4 s; qwen2.5vl:7b).
# 2026-11-20 14:33: the silver hammer on the light table was outlined as a
# 10 mm piece of itself, the pick aimed at that, and the next look lost it.
OUTLINE_BEFORE_PICK = True
# ... only where the scan's outline looks like a piece of it (OUTLINE_ONLY_PARTIAL):
# 6 s with the arm still before every pick, over outlines that were whole (user,
# 2026-11-20: "it doesn't need to reoutline - nothing else has moved").
OUTLINE_ONLY_PARTIAL = True

# A tool longer one way than the other is closed on across it, the jaws turned
# to it (GripperSpec.close_across_long): for a plain pick, where lying_pick
# does not take it (its height known, no outline from above, or one too small
# to say). With LYING_MIN_ASPECT 1, one where no stretch fits is refused, not
# handed on.
CLOSE_ACROSS_LONG = True

# After a close on nothing, found again by a fresh overhead look, and the next
# attempt aimed there (RELOCATE_AFTER_MISS, mrcf/tabletop/skills.py). The UR5
# has no wrist camera to see where it went: without this, a miss that may
# have moved the tool ends the pick with "Scan again before another try"
# (GripperSpec.blind_retry_after_miss off, 2026-11-12).
RELOCATE_AFTER_MISS = True

# ── Putting it away ───────────────────────────────────────────────────────────

# Put what is held into a container — the box — over its walls, let go inside
# it (GripperSpec.place_in, mrcf/policies/place_in.py); the box found by name
# (perception/grounding.py) where the scan does not outline it. The walls'
# height is measured by the wrist camera where there is one; here there is
# none, so, their height not kept as measured (CONTAINER_GRIPPER_INSIDE_M,
# below), it lets go from over the walls at the most they may be
# (Opening.rim_m): the tool drops in. Sturdy tools; watch the hammer.
PLACE_IN = True

# "Put the tools in the box": a pick that fails is set aside and tried again
# at the end after a fresh look, the rest going on — not the whole goal
# planned again (DEFER_FAILED, mrcf/tabletop/settings.py). On the SO-101
# (2026-11-18) each failed pick stopped the plan and re-planned for 45-60 s,
# and one re-plan went for a thing already in the tub.
DEFER_FAILED = True

# ── Its task's words ──────────────────────────────────────────────────────────

# "the hammer" is any hammer, "the screwdrivers" each screwdriver, and the box
# looked for by name the first time (TASK_WORDS_BY_KIND, mrcf/tabletop/
# settings.py): 2026-11-20 the tools were named red, then orange screwdriver,
# grey, then silver hammer, run to run, and no box had been found before.
TASK_WORDS_BY_KIND = True

# The box's walls are not measured (no wrist camera), so taken as up to 300 mm
# — carried over at that, let go there (CONTAINER_TALLEST_M): the user,
# 2026-11-20, "the height of the box is closer to 25cm", flaps up; the table
# task's 150 mm otherwise, and a tool carried over at 150 would meet its wall.
# No higher than needed: out at the box (0.6-0.8 m from the base) the arm
# reaches less the higher it is (0.79 m out at 30 cm up, 0.67 m at 45 cm).
CONTAINER_TALLEST_M = 0.30

# From inside the box, or across it, up over its walls first, across at that
# height until past them, then on (GripperSpec.transit_over_walls), as tidy does
# for its tub. 2026-11-20 16:25: the hammer let go in the box, the way from there
# to the screwdriver went straight down across the box's wall — 86-93 N, a
# protective stop.
TRANSIT_OVER_WALLS = True

# The box's walls measured (the user: "closer to 25cm"; rig/ur5/toolbench_containers.json,
# 260 mm), the gripper goes down among them over the spot, its flange kept over
# the rim (CONTAINER_GRIPPER_INSIDE_M; the Hand-E is 146 mm flange to grasp
# point): a tool let go ~15 cm over the box's floor, not dropped from over the
# walls. 2026-11-20 16:35: the hammer, let go 336 mm up, "dropped quite harshly".
CONTAINER_GRIPPER_INSIDE_M = 0.12

# The box's opening is the inside, in its own shade, not the box with its flaps
# folded out (CONTAINER_OPENING_BY_SHADE, containers.open_top). 2026-11-20 16:47:
# the whole outline taken for the opening, a screwdriver let go over a flap
# (user: "it should realise not to drop it on the flap").
CONTAINER_OPENING_BY_SHADE = True

# A scan goes on while the overhead camera has slid up to 8 px (~10 mm) from its
# empty-table photo, not 4 (CAMERA_DRIFT_PX, the table task's): after its two
# replugs on 2026-11-20 it sat 5-6 px off, every scan stopped and the hammer,
# set aside, could not be found again. Tools are gripped with 10 mm and more
# to spare each side. Calibrate the camera again and retake the photo to go
# back to 4.
CAMERA_DRIFT_PX = 8.0

# What is in the box already is lain on, not kept clear of (PLACE_IN_OVER_CONTENTS,
# as tidy's tub): tools go in on top of one another (user, 2026-11-20: "its a
# box - there may be things inside - why would that be a reason to reject").
PLACE_IN_OVER_CONTENTS = True

# Home at the end of each command that moved the arm (HOME_AFTER_COMMAND), as
# tidy does: it was left over the box (user, 2026-11-20: "why is it stuck?").
HOME_AFTER_COMMAND = True

# ── Candidates, not in force ──────────────────────────────────────────────────
# Take one over once the rules above have run live and its reason shows here.

# The kinds of tool learned by the arm and kept (mrcf/catalogue.py,
# rig/ur5/toolbench_catalogue.json): matched by how each looks from above, not
# by the vision model's name; picked by the grip that held it before. The
# SO-101's names churned ("figure" = "bottle cap"/"shaker"/"ball"); worth it
# once a pick of each tool has worked.
# CATALOGUE = True

# "Put <things> into <container>" laid out from the words themselves, the
# planner only for anything else (QUICK_PUT_INTO): ~35 s less a command. The
# command is still plain English; only its planning is skipped.
# QUICK_PUT_INTO = True

# A part the scan outlined apart (a screwdriver's coloured handle and its
# steel shaft) whose centre lies inside the one picked is part of it, not an
# obstacle (PARTS_INSIDE_ARE_ITS_OWN). Only for parts within the other's foot.
# PARTS_INSIDE_ARE_ITS_OWN = True
