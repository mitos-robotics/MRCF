"""
mrcf/policies/parallel_jaw/spec.py — what a two-finger gripper is
(GripperSpec: its geometry and tuning, filled in by the arm's backend) and what
one grasp of one object is (GraspPlan: how high, how wide, how far to lift).

The runs behind its values and fixes, with their dates: mrcf/policies/HISTORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from mrcf.policies.approaches import (  # noqa: F401 — re-exported
    DEFAULT_ALIGN,
    NEAR_BASE,
    REACH_OUT,
    TOP_DOWN,
    Approach,
    ApproachFit,
    _Choices,
)


@dataclass(frozen=True)
class GripperSpec:
    """A two-finger gripper on an arm: its geometry and how to drive it. Metres."""
    # ── Geometry (measured on the gripper) ─────────────────────────────────────
    max_open_m: float                     # widest the jaws open
    tip_below_grasp_m: float              # jaw tips below the grasp point (the pads' centre)
    free_depth_m: float                   # how far an object can stand above the grasp point
                                          # before it meets the gripper body
    finger_thickness_m: float             # each finger, beyond the gap, along the jaw axis
    finger_half_width_m: float            # each finger, across the jaw axis, half
    jaw_axis: str = "radial"              # how the jaws open relative to the arm's base:
                                          # "radial" (toward it) or "tangential" (across)
    camera_body_m: "tuple | None" = None  # what sticks out of the gripper beside the jaws
                                          # (a wrist camera): (x, y, radius) in the gripper
                                          # frame, from the grasp point — kept off neighbours
    camera_clearance_m: float = 0.010     # wanted between it and a neighbour
    sweep_half_width_m: float = 0.025     # leaning: the gripper's half width across its line, where it
                                          # passes over what stands toward the base (ReachMixin.sweep_clearance)
    sweep_margin_m: float = 0.003         # ... and how far over their tops it must stay
    sweep_free_width_m: float = 0.026     # ... but between the open fingers, this wide, up to free_depth_m
                                          # from the jaws, what stands there passes between them
    grip_over_top_m: float = 0.010        # gripped higher to pass over it, no nearer the object's top than this
    # ── Where it works ─────────────────────────────────────────────────────────
    table_z: float = 0.0                  # the work surface, world z
    camera: "str | None" = None           # the wrist camera's name, if it has one
    render_w: int = 640
    render_h: int = 480
    # ── Heights ────────────────────────────────────────────────────────────────
    hover_height_m: float = 0.06          # least hover above the surface
    lift_height_m: float = 0.07           # least carry height
    lift_clearance_m: float = 0.05        # carry this far above the grasp height
    top_clearance_m: float = 0.008        # jaw tips above an object's top, at the hover
    release_clearance_m: float = 0.005    # an object's foot above the surface, at release
    # ── Sizing a grasp ─────────────────────────────────────────────────────────
    grasp_height_fraction: float = 0.45   # grasp a measured object this far up it, where its
                                          # sides are straight enough there (grasp_height_for)
    min_grasp_height_m: float = 0.016     # and never lower (jaw tips off the surface)
    grasp_flat_mm: float = 2.5            # the most an object's width may change over the pads'
                                          # height where they close: more is a taper or a ridge,
                                          # which the closing jaws cam the object out along
    trust_shape_after: int = 2            # a remembered shape backed by fewer agreeing looks than
                                          # this is looked at again (Target.shape_looks)
    jaw_slack_m: float = 0.004            # room either side of the width, opening
    neighbour_clearance_m: float = 0.003  # least gap between a finger and an obstacle
    unknown_open_m: float = 0.036         # opening for an object not measured yet
    unmeasured_obstacle_radius_m: float = 0.016   # an obstacle whose foot nobody measured
    slimmest_above_foot: float = 0.5      # an object nobody has measured may be this share of its foot
                                          # across where the fingers pass: a pick its feet crowd goes
                                          # over it to measure when it could fit then (_room_by_looking)
    look_aside_m: float = 0.04            # what the camera cannot measure from over the object it
                                          # looks at from this far either side of it, then twice
                                          # as far (_look_around)
    look_around_max: int = 6              # ... from at most this many spots (0: never)
    jaw_turns_deg: tuple = (15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0, 90.0, -90.0)
                                          # where the neighbours leave the jaws too little room, the
                                          # turns about the approach tried, in order (_turned_for_room)
    default_max_height_m: float = 0.08    # the tallest an object could be, when nothing says
    # ── Telling a grip from a miss ─────────────────────────────────────────────
    miss_width_m: float = 0.030           # wider than this after closing: nothing grasped
    min_hold_width_m: float = 0.0008      # narrower than this: closed on nothing
    slip_m: "float | None" = 0.004        # carrying: jaws closing this much past where they held
                                          # it mean it has slipped out — the arm stops there (None: not watched)
    touchdown_m: float = 0.015            # ... unless within this of the release height: the object's
                                          # foot met the surface early (a lean tips it below the jaws'
                                          # model), and it is let go there (PlaceMixin.place)
    # ── Motion ─────────────────────────────────────────────────────────────────
    approach_tol_m: float = 0.015         # a hover counts as reached within twice this
    travel_tol_m: float = 0.01            # move_ee_to tolerance: hover, lift, transit, retreat
    grasp_tol_m: float = 0.008            # descent onto an object
    release_tol_m: float = 0.005          # descent with an object in the jaws
    grasp_step_deg: float = 3.0           # joint step on the final descent
    # The final descent at full speed (a move like any other, under the arm's
    # own contact limit) until the jaw tips are this far above the object's top,
    # its height known, then at grasp_step_deg under the approach's limit; None:
    # all of it at grasp_step_deg. The UR5's came down 15 cm at 1.8 cm/s, 8.6 s
    # of every pick (2026-11-13).
    fast_descent_above_top_m: "float | None" = None
    close_step_pct: float = 1.5           # jaw travel per cycle when closing onto something
    retries: int = 3
    # After a close that may have moved the object (a miss, too wide, the grip
    # lost), come down again where it was, though nothing can look for it
    # again (no wrist camera, no Vision.relocate). Off: the pick stops and asks
    # for a scan (policies/HISTORY.md). On only for a test of the retries themselves.
    blind_retry_after_miss: bool = False
    # ── Looking before descending ──────────────────────────────────────────────
    measure_at_hover: bool = True         # measure the object with the wrist camera
    aim_at_measured_foot: bool = True     # camera located: aim at the foot it finds, not
                                          # the scan's position (only if it is the more accurate)
    max_aim_shift_m: float = 0.02         # further than this from expected: not the same object
    max_servo_shift_m: float = 0.035      # the most the servo may move the jaws from the scan's
                                          # estimate (and never nearer another object's)
    servo_agree_m: float = 0.010          # servo within this of the scan, and the fixed camera seeing
                                          # the jaws: descend at the scan's estimate, not the servo's
    remeasure_known: bool = False         # look again at an object whose shape is known: seconds
                                          # a pick, but it also tells an unlisted neighbour by height
    back_off_to_attempt_hover: bool = False  # after a failed attempt, straight up no higher than the hover
                                          # that attempt reached, not the plan's: far out, the SO-101 cannot
                                          # get up to a hover clear of the tallest thing there could be, and
                                          # backing off to it ground on 24 s a retry (2026-11-16)
    # ── Looking for how tall it is (height_looks.py) ───────────────────────────
    height_from_views: bool = False       # not measured, its kind's height not known, and outlined from above
                                          # (Target.outlines): the wrist camera looks at it from where its
                                          # height shows, and the shape is fitted to its outlines in every
                                          # view; refused while still unsettled
    height_looks_max: int = 4             # ... at most this many looks a pick
    grip_low_when_capped: bool = False    # ... not settled nor bounded, but no taller than lets it be gripped as low
                                          # as the jaws go under the gripper's body: picked so, hovering clear of
                                          # that (height_looks._height_capped); off: refused
    stopped_short_holds: bool = False     # jaws that stop well short of shut (stopped_short_m) with no load felt
                                          # have something between them (smooth or rigid: a plug's pins): lifted,
                                          # and the hold check decides; off: a miss
    stopped_short_m: float = 0.004
    open_wide_when_low: bool = False      # gripped as low as the jaws go (a recipe, or lowered after a miss):
                                          # opened as wide as the neighbours let them, round a wide flat body
    wrist_at_grasp: bool = False          # at the grasp, before closing: the object and the gap between the jaw
                                          # tips found in the same wrist-camera picture (the jaws' learned view),
                                          # and the jaws reseated onto it by their difference (pick._wrist_at_grasp)
    wrist_at_grasp_tol_m: float = 0.0025  # ... left as they are within this
    wrist_at_grasp_max_m: float = 0.020   # ... and not moved by more: a difference that large is a wrong outline
    grip_low_from_above: bool = False     # ... and before any look from the side: where its outline from above
                                          # alone caps it short enough for that (height_looks.low_from_above),
                                          # gripped as low as the jaws go, no side looks; off: looked at
    height_looks_agree: int = 2           # ... settled once the fit is resolved and this many looks from
                                          # the side see it
    height_view_clearance_m: float = 0.04  # ... looking from beside, the jaws at least this far from any
                                          # object's column (each as tall as it could be) below its top
    # ── Picking by a wall (mrcf/policies/rim_pick.py) ──────────────────────────
    rim_pick: bool = False                # pick a thin-walled open thing (a cup, a bowl, a scoop) by one wall,
                                          # one jaw inside, one outside. Off: refused
    rim_grip_depth_m: float = 0.012       # ... the jaws closed this far below the lowest its rim may stand
    rim_jaw_gap_m: float = 0.012          # ... each jaw this far from the wall as they come down
    rim_max_misalign_deg: float = 25.0    # ... where the wall faces within this of the jaws' own line
    rim_tries: int = 8                    # ... the best this many points of it tried for reach
    handle_first: bool = False            # ... by its handle first, where it has one (policies/handle_pick.py):
                                          # the long narrow part leading away from its widest part, closed across
    handle_min_m: float = 0.008           # ... a handle this wide across at least ...
    handle_max_m: float = 0.040           # ... and no wider (wider: the part it joins) ...
    handle_min_len_m: float = 0.025       # ... for this long
    handle_from_join_m: float = 0.010     # ... gripped this far out from where it joins (and the fingers' half width)
    handle_open_m: float = 0.012          # ... the jaws this much wider than it each side, coming down
    handle_step_m: float = 0.008          # ... closed on nothing: opened, and this much lower, until it holds
    handle_bite_m: float = 0.004          # ... where it held before known (Target.grip_at_m): closed first that
                                          # much lower, a deeper bite of it (that height still learned)
    rim_min_hold_m: float = 0.0008        # ... its wall held when the jaws stop at least this far apart under load:
                                          # thinner than any piece (an empty close reads up to 0.5 mm)
    rim_hollow_m: float = 0.03            # ... a wall only where it has a hollow this wide behind it (not a handle)
    rim_min_bite_m: float = 0.004         # ... and gripped no less than this below the lowest its rim may be
    # ── Picking a thing lying along the table, its height not known (mrcf/policies/lying_pick.py) ──
    lying_pick: bool = False              # a long thing (a tool) gripped across, by its handle (handle_first) next to
                                          # the heavier part it joins, else at its balance point; closed from no
                                          # higher than it is wide there, lower until it holds. Off: the plain pick
    lying_min_aspect: float = 2.0         # ... long: at least this many times as long as it is widest across
    lying_side_clear_m: float = 0.005     # ... the jaws open at least this much wider than it each side
    lying_test_lift_m: float = 0.02       # ... lifted this far first and held checked, before the full lift
    lying_join_min_m: float = 0.015       # ... a handle's joined part wider for at least this long (shorter: a lip,
                                          # the flared end of a hammer's grip, passed over)
    held_by_camera: bool = False          # the gripper reading empty after shutting on something, the wrist camera
                                          # asked whether it still carries it (perception/held_view.py)
    held_open_m: float = 0.003            # ... until the jaws open wider than they shut by this
    held_closed_m: "float | None" = None  # ... and not when the jaws have since closed this much further than they
                                          # shut on it: had it been between them they could not have (None: the
                                          # camera may say carried however far they closed)
    held_turns: bool = False              # what is held may turn in the jaws: a place into a container crosses
                                          # its walls by where it was gripped — through its middle, as far as it
                                          # was gripped above its foot; off it, as far as its farthest part is
                                          # from the grip (place_in._turned_hang)
    rim_swing_share: float = 0.0          # held by a wall, it may swing under the jaws: a place into a container
                                          # crosses its walls this share of its width higher (more if the arm
                                          # reaches; else as before, and said). 0: as gripped
    held_while_still: bool = False        # ... once the camera or the model has seen it carried: still held while
                                          # the jaws stand where they were then, no more than held_still_m nearer
                                          # shut — slipped out, squeezing jaws close on — and that is at least
                                          # rim_min_hold_m apart (nearer shut, an empty close looks the same)
    held_still_m: float = 0.0004
    # ── Touching the surface to measure it (mrcf/policies/touch_surface.py) ───
    touch_surface: bool = False           # find a surface's height by lowering the shut jaws onto it until they
                                          # stop: for an arm that stalls gently on contact. Off: refused
    touch_step_m: float = 0.002           # ... this far down a step
    # ── The wrist camera's view of the jaws (mrcf/policies/learn_jaw_view.py) ──
    jaw_view_openings_m: tuple = (0.0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1)   # the openings it learns them at
                                          # (each no wider than max_open_m)
    jaw_view_height_m: float = 0.12       # ... looking down from this high over the surface
    jaw_view_spots_min: int = 3           # ... from at least this many places (what is behind them differs)
    # ── Putting things into containers (mrcf/policies/place_in.py) ─────────────
    place_in: bool = False                # put what is held into a container (a tub, a bowl): over its walls,
                                          # let go inside. Off: refused (not yet tried on this arm)
    container_wall_clearance_m: float = 0.02  # ... what is held, and the jaws, at least this far over and in from
                                          # its walls
    container_cross_extra_m: float = 0.0  # ... and crossing them, this much higher again (the arm sagging and
                                          # lagging its commands carries lower than its model says)
    container_floor_m: float = 0.01       # ... its floor no higher than this above the surface it stands on, when
                                          # not known: let go that far up, at most
    container_spot_step_m: float = 0.02   # ... spots inside it tried this far apart
    container_max_lean_deg: "float | None" = None   # ... no spot the gripper must lean further than this from
                                          # straight down to reach (None: any it reaches). Leaning far out, its
                                          # body meets the far wall
    container_max_drop_lean_deg: "float | None" = None  # ... letting go from over its walls (nothing going down
                                          # inside to meet a wall), leaning no further than this (None: any)
    container_lean_above_walls: bool = False  # ... going down inside, leaning as far as being over the spot takes
                                          # (up to container_max_drop_lean_deg) while the jaws are above the walls,
                                          # and no further than container_max_lean_deg once among them; and a spot
                                          # it can go down inside chosen before one it can only drop into. Off: one
                                          # lean all the way down
    container_descent_retry: bool = True  # ... a way down inside that stopped short tried once more; off: it
                                          # backs straight up and lets go from over the walls (stopped short is
                                          # something in the way, and pushing again pushes the container)
    transit_settle: bool = True           # moves that only travel — up, across, back up, in a place into a
                                          # container and over its walls — waited on to settle at their end
                                          # (most of a second still, a slow creep onto the goal); off: on at once
    transit_over_walls: bool = False      # a pick's way to its hover that crosses a container's walls (or starts
                                          # inside them): up over its rim first, across until past them, then on
                                          # (ParallelJaw.walls gives them). Off: straight there
    place_in_over_contents: bool = False  # ... a thing held by its outline (GraspPlan.held_outline) is laid in
                                          # over what is there, not kept clear of it; anything else that finds no
                                          # clear spot is let go from over its walls, as far in as the jaws' own
                                          # room allows, onto what is there. Off: kept clear of, else refused
    container_spots_max: int = 8          # ... and at most this many, before it is refused
    container_gripper_inside_m: "float | None" = None  # ... the walls measured, the gripper this far down among
                                          # them, over the spot (None: no further than free_depth_m: what is held
                                          # under the jaws, the gripper's body over the rim)
    container_outline_clearance_m: float = 0.010  # ... what is held, by its own outline (GraspPlan.held_outline),
                                          # kept this far in from the walls at least
    container_look_baseline_m: float = 0.04   # ... its walls' height not measured: two wrist-camera looks this far
                                          # apart by a turn about the arm's base, at a wall the turn crosses
                                          # (perception/plane_sweep.sweep_edges)
    container_look_settle_s: float = 0.4  # ... each look after the arm has stopped this long
    container_look_reach_m: float = 0.08  # ... its edges counted along this much of the wall either side
    container_look_band_px: int = 40      # ... and this close to the wall's line, at any height it may have
    container_rim_min_m: float = 0.02     # ... a container's walls are at least this tall: edges lining up lower
                                          # are its floor, or what is in it
    container_rim_margin_m: float = 0.01  # ... and passed this much higher than measured
    approach_look: bool = False           # on the way down, one more look from just above its top, and the
                                          # grasp planned again by it (height_looks: _approach_look)
    approach_look_above_top_m: float = 0.02   # ... the jaw tips this far over the tallest it may be
    approach_look_shift_m: float = 0.006  # ... the aim moved by what it sees, no further than this
    servo_reach_margin_m: float = 0.015   # the held orientation must also reach this much further
                                          # out: the servo moves the grasp after it is chosen
    measure_agree_mm: float = 6.0         # two looks' heights must agree this closely
    far_measure_agree_mm: float = 15.0    # ... from a hover above the servo's height, where heights
                                          # are coarser (the taller of the two is used)
    measure_agree_foot_mm: float = 3.0    # and their feet this closely
    same_height_tol_m: float = 0.012      # a look at the object must agree with the height known
                                          # of it — else the camera is on something else
    other_object_near_m: "float | None" = None   # ... and one that does not is another object only with another
                                          # standing this near where it was found (beyond their foot); None:
                                          # always. Nothing near: a part of it seen — the look is not trusted
    measure_foot_ratio: tuple = (0.6, 1.7)   # the look's foot over the fixed camera's (Target.foot_m):
                                             # outside this, the look outlined something else
    servo: bool = True                    # centre the jaws from the wrist camera, if calibrated
    servo_tol_px: float = 15.0
    servo_height_m: "float | None" = None  # the hover the servo's target pixel was set at:
                                           # it aims right only there (None: anywhere)
    servo_open_m: "float | None" = None    # the jaw opening it was set at: where the arm says
                                           # the jaws are slides with the opening (one jaw
                                           # moves), so it servos at this one (None: any)
    servo_point_height_fraction: float = 0.4   # how far up an object the servo's detector puts
                                               # its point: told to the servo when the object's
                                               # height is known (where it aims depends on it)
    # ── Correcting from a fixed camera that sees the gripper ───────────────────
    overhead_correct: bool = True         # at the pick's hover and the place's transit, move until
                                          # a fixed camera sees the gripper over the target
                                          # (the backend's overhead_gripper_offset), not just
                                          # the kinematics; then come down with the same correction
    overhead_tol_m: float = 0.003         # close enough
    overhead_max_iters: int = 3
    overhead_max_step_m: float = 0.03     # one correction no bigger than this
    overhead_max_total_m: float = 0.05    # nor all of them together: more is a wrong measurement
    grasp_settle_tol_m: float = 0.003     # at the grasp, before closing: jaws further than this from
                                          # their aim, as the fixed camera sees them, are moved onto it
    grasp_settle_along_m: float = 0.0015  # ... or further than this along the arm (_overhead_at_grasp)
    grasp_reseat_along_m: "float | None" = None   # ... still further than this along the arm after that: up, over and
                                          # down again (pick._reseat); None: closed where they are
    grasp_reseats: int = 2                # ... at most this many times, then retried without closing
    grasp_reseat_lift_m: float = 0.015    # ... up this far for it
    grasp_settle_max_m: float = 0.008     # ... but never more than this in all, round the object
    anvil_margin_m: "float | None" = 0.004   # descend with the object's widest part this far off the
                                             # jaw that does not move (_aim_at_anvil); None: centred
    descend_on_overhead: bool = True      # when the wrist camera cannot align, come down where the
                                          # fixed camera saw the jaws over the object (it measured
                                          # both) instead of retrying; never without that sighting
    descend_unconfirmed: bool = False     # when no camera can confirm the jaws over the object (the
                                          # wrist camera cannot find it, the fixed one cannot see the
                                          # gripper), come down at the scan's position on the kinematics
                                          # anyway instead of retrying the same look: a pick that goes,
                                          # at the risk of a knocked object. Never after a close that
                                          # may have moved it (aligners.unconfirmed_descent)
    # After a close that missed an object it had not moved (found again within
    # unmoved_m of where it was): the aim, not the object, was off. The next
    # attempts aim this much further out along the arm (positive: away from
    # the base), one step per such miss; () aims at the same spot again.
    miss_steps_along_m: tuple = ()
    step_after_sideways_push: bool = False   # ... also when the miss pushed it along the jaws' line and not toward
                                          # the base (the closing jaw caught its edge: the jaws were beside it)
    close_across_long: bool = False       # an object longer one way than the other on the surface (lying on its
                                          # side, a pen): the jaws turned to close across it (ReachMixin.long_way_deg)
    long_ratio: float = 1.5               # ... longer by this much, and by long_margin_m, than it is wide
    long_margin_m: float = 0.01
    unmoved_m: float = 0.004
    descent_lean_retries: int = 2         # a pick's or a place's descent that stalls is tried again
    descent_lean_step_deg: float = 10.0   # ... leaning this much further each time (far out, more lean
                                          # reaches lower)
    descent_turn_first: bool = False      # a pick's descent leaning other than the hover did: turned to it over
                                          # the object first, then straight down; and one that went off its line
                                          # (ended further from the grasp than it set out) back over the grasp at
                                          # its lean and down again, not up from where it went. Off: turned on the
                                          # way down, and up from where it stalled
    overhead_first: bool = True           # the fixed camera has the jaws over the object: the first
                                          # attempt comes down there, without the wrist servo, which
                                          # looks again only once a miss may have moved the object
    # ── Approaches: how it comes at an object, by where the object stands ──────
    approaches: tuple = (TOP_DOWN, NEAR_BASE, REACH_OUT)   # in order of preference within a zone
    near_base_m: "float | None" = None    # nearer the arm's base (RobotEnv.base_xy) than this: "near"
    far_reach_m: "float | None" = None    # further than this: "far" (None, None: everywhere "mid")
    relook_after_miss: bool = False       # a close on nothing found again (looked for afresh) before the next
                                          # attempt, whatever the miss was put down to: it may have been touched
                                          # and tipped. Off: one closed above it is gripped lower where it was
    lean_clears_floor: bool = False       # a grasp held leaning: raised, before the descent, as far as the
                                          # gripper's lowest point then reaches below it (GripperBelowGrasp) and
                                          # lean_floor_clear_m more — capped grip_over_top_m under its top. Off: as
                                          # planned (the floor guard refuses what would go below the table)
    lean_floor_clear_m: float = 0.004     # ... at lean_floor_clear_full_deg and beyond (far out the arm sags, and
                                          # its commands run that far lower); lean_floor_clear_min_m at
                                          # lean_clears_floor_from_deg, and between in proportion. After a miss
                                          # the wrist camera saw closing above it: the least, as lowered
    lean_floor_clear_min_m: float = 0.004
    lean_clears_floor_from_deg: float = 10.0   # ... only leaning at least this far
    lean_floor_clear_full_deg: float = 30.0
    far_from_lean_deg: "float | None" = None  # ... and "far" too where straight down over it — the hover, the
                                          # grasp and the lift — needs more lean than this: reach_out's leaning
                                          # pick, not top_down's (None: by distance alone)
    base_keep_out_m: "float | None" = None   # the arm's own base, as an obstacle this wide (radius,
                                             # about base_xy): no finger or wrist camera inside it
    servo_tilt_deg: "float | None" = None    # the lean the wrist servo was calibrated at (the
                                             # backend's tilt convention), for servo_max_lean_change_deg
    # ── Standing a fallen object up (stand_up.py) ─────────────────────────────
    stand_up_margin_m: float = 0.006      # the knob this far off the fixed jaw as the jaws come down:
                                          # wider than anvil_margin_m — a round knob on its side rolls
    stand_up_tip_clearance_m: float = 0.001   # jaw tips this far above the surface at the grip: under
                                              # the knob's middle, or the moving jaw squeezes it out
                                              # beneath the fixed one
    stand_up_overshoot_m: float = 0.006   # lift the knob this much past where it is with the object
                                          # stood up: it hangs clear and drops onto its foot. Erring
                                          # high is cheap, low lets it go still leaning: the fixed
                                          # camera reads a 13 mm head 16-17 mm, which alone takes
                                          # 4 mm off the lift
    stand_up_grip_tol_m: float = 0.0015   # the descent to the grip ends this close to it
    stand_up_step_m: float = 0.003        # the lift goes up in steps this tall
    stand_up_step_deg: float = 2.0        # ... no joint moving more than this a control cycle
    stand_up_min_knob_m: float = 0.006    # a knob measured narrower than this is not one to grip
    stand_up_turn_slack_deg: float = 15.0  # the jaws may grip the knob this far off square to it, where
                                           # the wrist turns no further (_plan's turns, nearest first)
    stand_up_attempts: int = 3            # tries in all; after a miss the wrist camera looks again
                                          # from over it (WristSurface), not the arm parked for a scan
    stand_up_relook_shift_m: float = 0.025    # a miss moves it this far at most: a knob the wrist
                                              # camera finds further off is not taken for it
    # ── Pushing an object across the surface (mrcf/policies/nudge.py) ─────────
    push_max_m: float = 0.04              # the furthest one push moves an object (further: pick it up)
    push_tip_clearance_m: float = 0.004   # the closed jaw tips this far above the surface as they push
    push_gap_m: float = 0.006             # they come down this far short of the object
    push_body_margin_m: float = 0.006     # the gripper body above the fingers this far over what it passes
    push_max_contact_share: float = 0.5   # pushed no higher up an object than this share of its height
    push_step_m: float = 0.003            # the push goes in steps this long
    push_step_deg: float = 1.5            # ... no joint moving more than this a control cycle
    push_back_off_m: float = 0.003        # back from it this far before lifting away
    push_short_tol_m: float = 0.004       # a push that ends this much short still counts as done


@dataclass(frozen=True)
class GraspPlan:
    """
    How to grasp one object: its size where the jaws meet it, and the heights
    of each phase, metres above the surface.
      height_m          surface to its top
      width_m           widest part the open jaws pass on the way down
      grasp_height_m    where the pads close
      grip_width_m      its width there — decides where the held object's centre is
      pregrasp_open_m   jaw opening on the way down
    """
    height_m: float
    width_m: float
    grasp_height_m: float
    grip_width_m: float
    pregrasp_open_m: float
    hover_height_m: float
    lift_height_m: float
    release_height_m: float
    # What is held, from above, in the jaws' own frame once gripped: [(u, v)] m from
    # the grasp point, u along the way the jaws open, v square to it (policies/
    # lying_pick.py). None: not known — a place keeps clear by width_m all round.
    held_outline: "tuple | None" = None
