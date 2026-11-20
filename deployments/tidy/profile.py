"""
Tidy's own rules for its arm, the SO-101 — picking up everyday things on the
bare table (figures, a plug, a scoop, sweets) and putting them away into a
tub — laid over the arm's facts (mrcf/backends/so101/robot_config.py) wherever
tidy makes the arm's env and policies (mrcf/profiles.py). The surface it works
on is measured, not set here (config.SURFACE_Z: rig/so101/tidy_surface.json).
Until 2026-11-17 these were deployments/tabletop/profiles/so101.py: the SO-101
running the UR5's deployment.

Its own, learned on the table (2026-11-16/17); then the rules chess learned
whose reasons hold here too, copied with why. Not copied from chess
(deployments/chess/profile.py), and why not:

    REMEASURE_KNOWN          the memory pairing a piece with another's track on
                             the same square: there are no squares here
    DESCEND_UNCONFIRMED      a demo's workaround for one hidden queen: here a pick
                             the cameras cannot confirm retries instead
    UNKNOWN_OPEN_M           "a tall chess piece's" opening: the framework's
    UNMEASURED_OBSTACLE_RADIUS_M, SERVO_TOL_PX, SERVO_AGREE_M, MAX_SERVO_SHIFT_M,
    VISUAL_SERVO_ENABLED     the fixed-pixel servo, calibrated on a cream pawn
                             on the board: picks here aim by the located wrist
                             camera and the overhead (AIM_AT_MEASURED_FOOT), the
                             framework's values otherwise
    JAW_SLACK_M, NEIGHBOUR_CLEARANCE_M, GRASP_FLAT_MM,
    PLACE_RELEASE_M, MEASURE_AT_HOVER, AIM_AT_MEASURED_FOOT, MAX_AIM_SHIFT_M
                             the framework's own values are chess's: nothing to copy
    GRIPPER_MAX_OPEN_M 45    among 40 mm squares: here objects stand apart, and
                             the jaws open as wide as the arm's own 80 mm
                             (robot_config GRIPPER_MAX_OPEN_M), each pick still
                             narrowed to the room its neighbours leave.
                             2026-11-17: under chess's 45, a 50 mm plug and a
                             53-64 mm scoop were refused as wider than the jaws open
"""

# ── Its own ──────────────────────────────────────────────────────────────────

# Here, a look at the hover whose height disagrees with the one
# known is another object only with one standing within this of where it was
# found (GripperSpec.other_object_near_m); else a part of it, not trusted.
# 2026-11-17: a figure 44 mm from the side read 21 from over it in a dim room,
# nothing within 11 cm, and the pick gave up. The chessboard, crowded, keeps
# every disagreement as another piece.
OTHER_OBJECT_NEAR_M = 0.04

# Pick a thin-walled open thing by one wall, a jaw either side of it
# (GripperSpec.rim_pick, mrcf/policies/rim_pick.py). 2026-11-17: the blue
# measuring scoop, its bowl wider than the jaws were let open (user: "it can be
# lifted by just grasping one wall").
RIM_PICK = True

# After a close on nothing that pushed it sideways, along the jaws' line, not
# toward the base (GripperSpec.step_after_sideways_push). 2026-11-17, a wooden
# figure 23 cm out: three closes each pushed it 13-16 mm sideways (+5 along),
# the jaws round its head and short of its body; never stepped out.
STEP_AFTER_SIDEWAYS_PUSH = True

# An object longer one way than the other, as the overhead camera sees its
# footprint, is closed on across it, the jaws turned to it
# (GripperSpec.close_across_long). 2026-11-17: a wooden figure knocked onto
# its side was closed on along its length, from straight above, three times.
CLOSE_ACROSS_LONG = True

# How tall an object nobody has measured is, looked for before a pick plans by
# it (GripperSpec.height_from_views, mrcf/policies/parallel_jaw/height_looks.py):
# the wrist camera is sent where the object's height shows, and its shape is
# fitted to its outlines in every view, the overhead camera's from the scan
# among them (mrcf/perception/shape_fit.py). Refused, with why, while unsettled.
# Only for an object the scan outlined from above (Target.outlines: the
# table task's scene_scan); chess's board scan gives none, and is unchanged.
# 2026-11-16, on this table: planned for the tallest it could be (150 mm), every
# pick closed 117 mm up over 25 mm balls. From the hover the wrist camera saw
# 0.21 px per mm of a ball's height; from 6-12 cm beside it, leaning, 0.9-1.5.
HEIGHT_FROM_VIEWS = True

HEIGHT_LOOKS_MAX = 4

# ... and when the looks agree only on how tall it can be at most, and that is
# short enough to grip it as low as the jaws go, picked so rather than refused
# (GripperSpec.grip_low_when_capped). 2026-11-17: the wooden figures, 5-31 mm
# fitted from four looks, were refused every time.
GRIP_LOW_WHEN_CAPPED = True

# ... and before any look from the side, where the overhead camera's outline
# alone caps it short enough (GripperSpec.grip_low_from_above): gripped as low
# as the jaws go, no side looks. 2026-11-18: the side looks took 25-36 s an
# object, and made the plug 36-51 mm tall; gripped at that it was closed on
# nothing twice and pushed off its spot (user: "the plug kept getting pushed").
# Gripped low it had held (16:08).
GRIP_LOW_FROM_ABOVE = True

# ... and on the way down, one more look from just above its top, the grasp
# planned again by it (GripperSpec.approach_look).
APPROACH_LOOK = True

# After a failed attempt, back straight up no higher than the hover that
# attempt reached (GripperSpec.back_off_to_attempt_hover). 2026-11-16, on this
# table: 32 cm out, a back-off to 165 mm (clear of the tallest object there
# could be) is out of reach, and ground on 24 s each retry, 17 mm short.
BACK_OFF_TO_ATTEMPT_HOVER = True

# Put what is held into a container — a tub, a bowl — over its walls, let go
# inside it (GripperSpec.place_in, mrcf/policies/place_in.py). The container is
# found by name (perception/grounding.py) where the scan cannot outline it: the
# clear plastic tub on 2026-11-17 was never among the scan's objects.
PLACE_IN = True

# After a close on nothing, the object is found again by a fresh
# overhead look (its Vision.relocate), and the next attempt aims there — out
# along the arm by MISS_STEPS_ALONG_M if it had not moved. Without it a miss
# with the wrist camera unable to see the foot ended the pick (2026-11-17, the
# red figure: closed short of it, then twice "not descending blind").
RELOCATE_AFTER_MISS = True

# The surface a task works on found by touching it with the shut jaws
# (GripperSpec.touch_surface, mrcf/policies/touch_surface.py): this arm stalls
# gently on contact, as a pick's descent onto something does. TABLE_Z above is
# the chessboard's top; 2026-11-17 this ran on the bare table, the board
# (2-3 cm) gone, and every grasp closed round the objects' heads. Tidy keeps
# what it touches (rig/so101/tidy_surface.json) and works from that; chess
# keeps TABLE_Z.
# Off until the touch stops on joint load and only where the wrist camera sees bare surface:
# 11:43 it came down on the tub (not yet found that run), pushed it aside, and the shoulder
# servo latched its overload protection.
TOUCH_SURFACE = False

# Jaws still further than this along the arm from the object's middle after the
# overhead camera's correction at the grasp: up, over and down again before
# closing (GripperSpec.grasp_reseat_along_m). 2026-11-17, a wooden figure: the
# correction stalled 4.8 mm off and the closing jaws squeezed its body out.
GRASP_RESEAT_ALONG_M = 0.003


# Held is also what the wrist camera sees: once the jaws have shut on something
# and the gripper reads empty, what was between them should still be there,
# carried, while the table slides by (GripperSpec.held_by_camera,
# perception/held_view.py). 2026-11-17: a scoop held by its 1 mm wall, squeezed
# to 0.1 mm, was called "left the jaws on the way up" with it in them.
HELD_BY_CAMERA = True
# ... and once seen carried, held while the jaws stand still
# (GripperSpec.held_while_still). 2026-11-19 11:09: the scoop, seen carried
# as it was lifted, was "not held" two minutes later at home — the model
# shown the room past it — and its place refused.
HELD_WHILE_STILL = True

# Jaws stopped wider than this after closing = nothing grasped
# (GripperSpec.miss_width_m): the arm's 80 mm opening less 2 mm, as on the UR5.
# The arm's own 30 mm is chess's — no piece is wider — and here things run to
# the jaws' full opening; the servo's load says held or not, so only jaws that
# never closed are a miss. 2026-11-18 16:08: the plug, closed on at 55 mm and
# loaded, was called "too wide — missed", lifted in the jaws and looked for on
# the table.
GRASP_MISS_WIDTH_M = 0.078

# The kinds of object it works with, learned by the arm and kept
# (mrcf/catalogue.py, rig/so101/tidy_catalogue.json): matched by how each looks
# from above, not by the vision model's name for it; picked by the grip that
# held it before, with no looks for its height; each pick's result learned.
# 2026-11-18: the same four objects were looked at from the side 25-48 s each,
# about ten times in a day, a different height each time, and the vision model
# named one figure "bottle cap", "bottle", "shaker", "ball" and "red bottle".
CATALOGUE = True

# A close on nothing is asked why, of the vision model on the wrist camera's
# pictures, and the next attempt changes that (mrcf/tabletop/settings.py
# DIAGNOSE_MISSES). 2026-11-18: three closes over a figure lying on its side,
# and over a plug's body gripped too high, each re-aimed and pushed further.
DIAGNOSE_MISSES = True

# "Put these in the tub": a failed pick is set aside and tried again at the
# end, the rest going on; not planned again from scratch (mrcf/tabletop/
# settings.py DEFER_FAILED). 2026-11-18: each failed pick stopped the plan and
# re-planned for 45-60 s, and one re-plan went for a figure already in the tub.
DEFER_FAILED = True

# At the grasp, before closing, the object and the gap between the jaw tips
# found in the same wrist-camera picture, and the jaws reseated onto it by
# their difference (GripperSpec.wrist_at_grasp). 2026-11-18: the jaws came down
# ~1 cm beside a figure and a plug, the overhead camera putting them "within
# 2 mm" of their aim, and closed on nothing (user: "you need the arm camera for
# minor adjustment").
WRIST_AT_GRASP = True

# Jaws that stop well short of shut with no load felt have something between
# them — lifted, and the hold check decides (GripperSpec.stopped_short_holds).
# 2026-11-19 09:54: a plug on its back, the jaws shut on its smooth pins at
# 28.5 mm, "no load", and called a miss.
STOPPED_SHORT_HOLDS = True

# Gripped as low as the jaws go, opened as wide as the neighbours let them
# (GripperSpec.open_wide_when_low). 2026-11-19: the same plug's body, 55 x
# 47 mm: at ~60 mm open the fingers came down on it, the descent stalled.
OPEN_WIDE_WHEN_LOW = True

# A grip the servo cannot feel, asked of the vision model on the wrist
# camera's picture (mrcf/tabletop/settings.py HELD_BY_MODEL). 2026-11-18 17:16
# and 2026-11-19 10:38: the scoop held by its 1 mm wall, plainly in the
# picture, called "left the jaws on the way up" — and its place refused.
HELD_BY_MODEL = True

# Naming what the scan found (mrcf/tabletop/settings.py NAME_TILE_PAD_MAX_PX,
# NAME_NOTHING_TWICE). 2026-11-19 10:55: the blue scoop's tile showed half the
# table round it, the model called it "nothing", and it was left out of the
# scene — no plan could have put it in the tub.
NAME_TILE_PAD_MAX_PX = 40
NAME_NOTHING_TWICE = True

# A tub with something in it already: what will not fit clear of it is let go
# over the walls onto it (GripperSpec.place_in_over_contents). 2026-11-19
# 11:07: the scoop refused — no spot 83 mm in from the walls clear of the
# figure put in the middle — and the plan stopped, the scoop in the jaws.
PLACE_IN_OVER_CONTENTS = True

# Into the tub: no spot reached leaning more than 20 deg, and a way down
# inside that stops short is not pushed again (GripperSpec.
# container_max_lean_deg, container_descent_retry). 2026-11-19 11:18: the
# plug taken to the tub's far corner, 36 cm out, leaning 35 deg; the gripper
# met the far wall, the descent was tried again, and pushed the tub across
# the table. The spots that went in cleanly leaned 10-20 deg.
CONTAINER_MAX_LEAN_DEG = 20.0
CONTAINER_DESCENT_RETRY = False
# ... but let go from over the walls leaning up to 45 deg: nothing goes down
# inside there. 14:57: the scoop, held by its handle and crossing high for its
# swing, was refused at every spot of the tub for leaning 35-40 deg.
CONTAINER_MAX_DROP_LEAN_DEG = 45.0
# ... and going down inside, leaning as far as being over the spot takes while
# the jaws are above the walls, no more than 20 deg among them; a spot it can
# go down into chosen before one it can only drop into (GripperSpec.
# container_lean_above_walls). Runs 36-37, 2026-11-19: 6 of 9 let go 13-18 cm
# up — over the spot, 12-18 cm up and 25-32 cm out, the arm reaches only
# leaning 30-40 deg; at the bottom 10-15 deg does. One lean all the way down,
# the top decided it, and every time that was a drop.
CONTAINER_LEAN_ABOVE_WALLS = True

# A pick's descent leaning other than its hover did turns to that lean over
# the object first; one that goes off its line goes back over the object at
# its lean, not up from where it went (GripperSpec.descent_turn_first).
# 2026-11-19 15:28, run 37 (the "bobbing" in the user's film): measured from
# the hover, the descent was re-chosen at -15 deg; turning and coming down at
# once, 3 deg a step, the arm went 4 cm up and 3 cm out and stopped; "up"
# from there stalled 2.4 cm short in 4.1 s, and down from there 13.5 cm
# short. The next attempt, its hover at -20 and down at -15, arrived.
DESCENT_TURN_FIRST = True

# A held-orientation descent (a pick's, onto its object) goes most of the way
# down as one line in joint space, to the solver's own answer 25 mm over the
# grasp, and the last 25 mm as before (the SO-101's DESCENT_LINE_ABOVE_M).
# Run 39, 16:38, and run 37's "bobbing": leaning 20 deg 32 cm out, the step-
# by-step descent drove the shoulder past its goal while the lagging elbow
# trailed — the jaws went up and out, or the command dipped into the table
# and was refused. Simulated on the arm's own kinematics with servos 2-3 deg
# short under load: that descent ends 7 cm off; the line then the steps, 3.5
# mm, and over the ball and the plug they stray no more than before.
DESCENT_LINE_ABOVE_M = 0.025

# A command the arm's controller pushes past its step (a joint stuck in its
# deadband, a settle closing a load offset) that the floor guard would refuse
# is sent as the step itself (the SO-101's FLOOR_CAPS_PUSH): at worst the move
# stops a few mm short, instead of being refused and abandoned. Run 40, 17:07:
# the plug gripped 18 mm up; the wrist camera's reseat came down 15 mm, a
# stuck joint was pushed 3 deg past its goal, the guard refused it, the jaws
# stayed 15 mm up — and the next look, taken from there as if at the grip,
# read the offset wrong (5.7, 7.5, 6.1 mm): the jaws closed off it and missed.
FLOOR_CAPS_PUSH = True

# A pick's way to its hover that crosses the tub's walls goes up over them
# first (GripperSpec.transit_over_walls). 2026-11-19 11:18 (user: "it didn't
# recognise the tub wall and didn't lift the jaws above the walls so it
# dragged the tub with it"): from the tub, 112 mm up, straight to the next
# object's 61 mm hover, coming down across the wall.
TRANSIT_OVER_WALLS = True

# Far is also where top_down would lean more than this (GripperSpec.
# far_from_lean_deg): reach_out, the leaning pick, there (user: "pick up
# things at an angle not just top down for slightly further away things").
# 2026-11-19: the plug at 32 cm was "mid-reach" by chess's 29 cm, went
# top_down at -20 deg, and stalled on the way down three times. The tub's
# clean spots took 10-20 deg; past 15 the wrist servo's lean is left behind.
FAR_FROM_LEAN_DEG = 15.0

# A grasp held leaning is raised as far as the jaw tips then reach below it
# (GripperSpec.lean_clears_floor). 2026-11-19: the plug 30 cm out, gripped
# 18 mm up as learned straight down, leaning 20-30 deg: the floor guard
# stopped every descent, the tips 3-8 mm under the table.
# The tips sit 9-10 mm under the grip at 20-30 deg (the arm's model); the
# guard saw the commands 12-17 mm lower than the grasp — the arm sags that far
# out, and its moves command past it. Room for that: 16 mm over the table.
LEAN_CLEARS_FLOOR = True
LEAN_FLOOR_CLEAR_M = 0.016
# ... in proportion to the lean: the bare 4 mm at 10 deg, all of it at 30.
# 2026-11-19 12:06 (user: "for the plug - the arm needs to go down slightly
# more and it won't be so hard"): 16 mm at 10-20 deg raised the plug's grip
# to 24-25 mm; it closed on an edge, slipped, and dropped.

# Held by its wall: crossed into the tub a third of its width higher
# (GripperSpec.rim_swing_share). 2026-11-19 12:05 (user: "it picked it up on
# the very very tip ... hence why it hit the wall"): the scoop's looks put
# its rim at 53 (it stands ~45), it was gripped 41 mm up, and it swung into
# the tub's wall. Gripping it deeper (20 mm under the rim, 12:36) closed on
# nothing twice — its bowl tapers, the wall further in lower down — and
# flipped it over: the depth is the rim's height to get right, not this.
RIM_SWING_SHARE = 0.33

# Moves that only travel — up, across, back up, into and out of the tub, over
# its walls — go on without settling at their end (GripperSpec.
# transit_settle). User 2026-11-19: "make the whole motions smoother - its
# very jerky". The joint record of 11:18:22-50: 62 stops and starts in 29 s,
# each move ending in 1-1.5 s of creeping onto its goal.
TRANSIT_SETTLE = False

# ── From chess, where the reason holds here too ──────────────────────────────

# How hard the jaws hold once squeeze_gripper() has found something: the goal
# this many % of travel past where they stopped. Chess's 10, from its wooden
# pieces: the wooden figures, the plug and the scoop's wall held at it
# (2026-11-17), where 3 let a pawn slip out at the carry's start. It CRUSHED a
# wrapped sweet (2026-11-17 12:09): a soft thing yields until the goal. Until a
# hold that eases off something giving way is built, soft things want less.
GRIPPER_HOLD_PCT = 10.0

# Grasp at this fraction of a measured height, where the profile is straight
# (ParallelJaw.grasp_height_for): chess's 0.6 put the pads on a pawn's neck
# above its flared foot. The wooden figures here are the same turned shape —
# a flared body under a narrower neck and head.
GRASP_HEIGHT_FRACTION = 0.6

# How close a descent onto an object must get to its aim: chess's 4 mm, "most
# of the slack either side of a piece". The figures here are piece-sized (22-23
# mm across in jaws opened ~30).
GRASP_TOL_M = 0.004

# The first attempt aligns with the wrist camera as well, not only by the
# overhead camera (chess: the overhead checks the jaws against where the
# object was remembered, and cannot see it under the gripper). The same here.
OVERHEAD_FIRST = False

# The same request, the same objects on the table: the plan made for it
# before, not planned again (mrcf/tabletop/settings.py PLAN_MEMORY); and home
# when it is done (HOME_AFTER_COMMAND). 2026-11-19 (user: "its the same
# objects in the same context - so it shouldn't be wasting time looking for
# stuff ... why isn't the arm ending in its main position?"): 54-65 s of
# planning each time, and the arm left over the tub.
PLAN_MEMORY = True
HOME_AFTER_COMMAND = True

# A failed pick not described by the vision model (mrcf/tabletop/settings.py
# DESCRIBE_FAILURES): 5-16 s each on 2026-11-19, said and not used.
DESCRIBE_FAILURES = False

# Into the tub, what is held crosses the walls as if it had turned in the jaws
# and hung by its smaller size (GripperSpec.held_turns). 2026-11-19 12:43
# (user: "it didn't lift it high enough"): the plug, held 27 mm across its
# 54 mm body, lifted to 112 mm for its 20 mm grip, caught the far wall.
HELD_TURNS = True

# The guard stops the arm pushing something rather than pausing and carrying
# on (mrcf/tabletop/settings.py GUARD_STOPS_PUSHING). 2026-11-19 12:43: the
# plug caught the tub's wall, the guard paused four times, each a second after
# carrying on, and the tub went toward the table's edge until the user moved
# it (user: "why can't the camera see something is wrong ... and stop it?").
GUARD_STOPS_PUSHING = True

# Into the tub: its walls crossed 15 mm higher than the 20 mm clearance
# (GripperSpec.container_cross_extra_m). 2026-11-19 13:35 (user: "i think you
# have some measurement wrong - the bottom of the object hit the tub"): a
# figure carried over the wall from 33 cm out, its grip 3 mm higher than the
# place reckoned and the arm sagging there, clipped it.
CONTAINER_CROSS_EXTRA_M = 0.015

# A figure the scan outlined as two (its head 1 cm from its body) is picked as
# one (mrcf/tabletop/settings.py PARTS_INSIDE_ARE_ITS_OWN): 13:35, refused twice
# for "multicoloured_toy leaves 0 mm".
PARTS_INSIDE_ARE_ITS_OWN = True

# A thing with a handle picked by it, not by the edge of its wall (GripperSpec.
# handle_first, mrcf/policies/handle_pick.py). User 13:38: "with the scoop why
# not grasp a more secure thing - its right on the very very edge - like
# maybe grab the handle??".
HANDLE_FIRST = True

# Every plain pick outlines its object again by what it is first, and after a
# close on nothing it is looked for afresh before the next try (mrcf/tabletop/
# settings.py OUTLINE_BEFORE_PICK; GripperSpec.relook_after_miss). 2026-11-19
# 14:03 (user: "how is it getting worse - the item rolled off the table"): a
# figure outlined as 16 mm of its 23 was gripped 5 mm off its middle, tipped,
# then come down on again blind and pushed off the table.
OUTLINE_BEFORE_PICK = True
RELOOK_AFTER_MISS = True

# "Put these into the tub" planned from the words themselves, each thing found
# by what the user calls it (mrcf/tabletop/settings.py QUICK_PUT_INTO); the
# planner for anything else. 2026-11-19 14:25: planned in 240 s, the plug
# looked for as "silver cylinder" (the scan's name for its pins), the re-plan
# timed out and came back cut off. User: "its fine for this - but i guess its
# still needed in the framework so make sure it doesn't break the other
# deployments".
QUICK_PUT_INTO = True

# A pick by the catalogue's recipe opens the jaws past what the object's kind
# is across, not only the scan's foot (mrcf/tabletop/settings.py
# RECIPE_OPENS_TO_KIND). 2026-11-20 09:13, morning light (user: "the jaws need
# to be a bit wider for the plug and your method will work"): the plug's foot
# read 44 mm from above, its kind 59 across; the jaws opened 52 and one came
# down on it — "closed above it", and the grip lost on the way up.
RECIPE_OPENS_TO_KIND = True

# Not held when the jaws have closed 10 mm further than they shut on it,
# whatever the wrist camera seems to see (GripperSpec.held_closed_m). 2026-11-20
# 09:25 (user: "no it isn't - the pick didn't work"): the plug slipped out as it
# was lifted, the jaws went on from 42.8 to 28.8 mm, the camera took 0.5% of its
# picture for it carried, and nothing was "put into" the tub.
HELD_CLOSED_M = 0.010

# The jaws may close as low as 11 mm over the table, their tips ~4 mm off it
# (GripperSpec.min_grasp_height_m; the framework's 16 is chess's, its pieces
# on a board). 2026-11-20 09:25 (user: "it should be able to pick up in any
# orientation"): the plug lying flat, gripped 17.7 mm up — on its rounded top
# edge — slipped out twice; "gripped lower next" had nowhere to go, 16 being
# the floor. The table guard still stops any move that would go into it.
MIN_GRASP_HEIGHT_M = 0.011

# A pick set aside whose object the next look tracks afresh is the one of its
# name nearest where it was, within 10 cm (mrcf/tabletop/settings.py
# REFIND_NEAR_M). Run 38, 16:23: the plug, pushed 4 cm by its failed pick,
# named white_plug again under a new track, was "not on the table where the
# scan can find it" and never tried again.
REFIND_NEAR_M = 0.10
