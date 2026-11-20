"""
mrcf/policies/parallel_jaw/reach.py — where the arm can take the jaws: the
room between neighbours, the waypoints of a pick and a place, one wrist
orientation held through them (and leaning further when it must), which
approach to use, and the small moves (backing off, moving again when short).

The runs behind its values and fixes, with their dates: mrcf/policies/HISTORY.md.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from mrcf.capabilities import DownRotation, GripperRotation, GripperTilt, OrientationSearch
from mrcf.policies.approaches import (  # noqa: F401 — re-exported
    DEFAULT_ALIGN,
    NEAR_BASE,
    REACH_OUT,
    TOP_DOWN,
    Approach,
    ApproachFit,
    _Choices,
)
from mrcf.policies.base import Obstacle
from mrcf.policies.parallel_jaw.planning import GraspPlanning, _takes
from mrcf.policies.parallel_jaw.spec import GraspPlan

# A lift (or a place's carry) that cannot go straight up as high as planned is
# tried this much lower at a time, down to the hover.
_LIFT_STEP_M = 0.010
# How far from the servo's calibrated height a hover can be and still servo.
_SERVO_HEIGHT_TOL_M = 0.005
# A descent that stalls before it has come this share of the way down did not
# meet the arm's reach (that stalls near the bottom): it is tried again at its
# own lean before leaning further (_stalled_at_start).
_START_STALL_SHARE = 0.25


BASE_LABEL = "the arm's own base"      # the Obstacle _with_base adds for it

class ReachMixin(GraspPlanning):
    """ParallelJaw's reach and approach (see the package docstring)."""

    def jaw_axis(self, env, xy, turn_deg: float = 0.0) -> np.ndarray:
        """
        Unit vector, surface plane, along which the jaws open when reaching
        xy — turned turn_deg about the approach, counter-clockwise from above
        (OrientationSearch's turn_deg).
        """
        radial = np.asarray(xy[:2], float) - self._base_xy(env)
        radial /= np.linalg.norm(radial) or 1.0
        axis = np.array([-radial[1], radial[0]]) if self.spec.jaw_axis == "tangential" else radial
        if not turn_deg:
            return axis
        a = np.radians(turn_deg)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ axis

    def face_turns(self, env, xy, faces_deg: "float | None") -> "list[float] | None":
        """
        For flat sides facing faces_deg (Target.faces_deg), the jaw turns at xy
        that close the jaws across a pair of them: across their corners the
        fingers can land on top (a 40 mm cube is 57 mm corner to corner, more
        than many grippers open). None: no sides, any turn. First the one the
        wrist turns least to from where the jaws point now (_wrist_turn), then
        the least from the arm's way at xy. UR5, 2026-11-13: a block's faces
        lay 44 deg from that way, and it turned -83 deg where +7 squared them
        too; turning a held block that far wound the gripper's cable and threw
        the block 9 mm off the one below.
        """
        if faces_deg is None:
            return None
        axis = self.jaw_axis(env, xy)
        t = (float(faces_deg) - float(np.degrees(np.arctan2(axis[1], axis[0]))) + 45.0) % 90.0 - 45.0
        turns = [t, t + 90.0, t - 90.0]
        wrist = self._wrist_turn(env, xy)
        if wrist is not None:
            turns.sort(key=lambda turn: round(wrist(turn), 3))        # stable: a tie keeps the order above
        return turns

    def long_way_deg(self, env, target) -> "float | None":
        """
        Which way target lies longest on the surface — deg from the world x
        axis, mod 180 — when it is longer than wide by spec.long_ratio and
        spec.long_margin_m; None: about as wide every way, or not known. From a
        fixed camera's outline of it (Target.outlines): what of that outline,
        carried down to the surface and up to its top, both cover — a standing
        thing's outline stretches away from the camera by its height, and that
        is taken out. Its top: its height, measured; else the most it may be
        (Target.max_height_m).
        """
        import cv2

        from mrcf.perception.plane_geometry import PlaneView

        s = self.spec
        top = target.height_m if target.height_m is not None else target.max_height_m
        if not target.outlines or top is None:
            return None
        for camera, outline in target.outlines.items():
            size = tuple(outline.get("size") or ())
            px = np.asarray(outline.get("px") or [], float).reshape(-1, 2)
            model = env.camera_model(camera, int(size[0]), int(size[1])) if len(size) == 2 and len(px) >= 3 else None
            if model is None:
                continue
            view = PlaneView.from_camera_model(model)
            raised = PlaneView(view.K, view.D, view.R, view.t + view.R @ np.array([0.0, 0.0, float(top) * 1000.0 / view.scale]),
                               scale=view.scale, origin_m=view.origin_m)
            low, high = view.to_plane(px), raised.to_plane(px)            # mm on the plane, 1 px a mm below
            if not (np.isfinite(low).all() and np.isfinite(high).all()):
                continue
            corner = np.minimum(low.min(axis=0), high.min(axis=0)) - 2.0
            span = np.maximum(low.max(axis=0), high.max(axis=0)) - corner + 2.0
            if (span > 1000.0).any():
                continue
            shape = (int(span[1]) + 1, int(span[0]) + 1)
            a, b = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
            cv2.fillPoly(a, [np.round(low - corner).astype(np.int32)], 1)
            cv2.fillPoly(b, [np.round(high - corner).astype(np.int32)], 1)
            found, _ = cv2.findContours(a & b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not found:
                return None
            _, (w, h), angle = cv2.minAreaRect(max(found, key=cv2.contourArea))
            short, long = sorted((float(w), float(h)))
            if long < s.long_ratio * short or long - short < s.long_margin_m * 1000.0:
                return None
            # The rectangle's long side, from the plane's x axis — the world's.
            return float((angle if w >= h else angle + 90.0) % 180.0)
        return None

    def across_turn(self, env, xy, long_deg: float) -> float:
        """The jaw turn at xy (OrientationSearch's turn_deg, -90..90) that closes them across something lying long_deg."""
        axis = self.jaw_axis(env, xy)
        now = float(np.degrees(np.arctan2(axis[1], axis[0])))
        return (float(long_deg) + 90.0 - now + 90.0) % 180.0 - 90.0

    def _wrist_turn(self, env, xy):
        """
        turn_deg -> how far (0-90 deg) the wrist turns from where the jaws point
        now to the jaws at xy turned turn_deg, either way round being as good
        (_hold holds them the way round they are now, as_now). None when the
        arm cannot say where they point (GripperRotation, down_rotation).
        """
        if not isinstance(env, GripperRotation) or not isinstance(env, DownRotation):
            return None
        now = np.asarray(env.get_ee_rotation(), float)[:2, 0]
        if float(np.linalg.norm(now)) < 1e-6:
            return None
        a_now = float(np.degrees(np.arctan2(now[1], now[0])))
        at = np.r_[np.asarray(xy, float)[:2], 0.0]

        def turn(turn_deg: float) -> float:
            x = np.asarray(env.down_rotation(at, turn_deg=turn_deg), float)[:2, 0]
            d = (float(np.degrees(np.arctan2(x[1], x[0]))) - a_now) % 180.0
            return min(d, 180.0 - d)
        return turn

    def faces_at(self, env, xy, turn_deg: float) -> float:
        """Which way flat sides held across the jaws face, put down at xy with the jaws turned turn_deg (mod 90)."""
        axis = self.jaw_axis(env, xy, turn_deg)
        return float(np.degrees(np.arctan2(axis[1], axis[0])) % 90.0)

    def jaw_room(self, env, xy, obstacles, tips_m: "float | None" = None,
                 turn_deg: float = 0.0) -> "tuple[float, str | None]":
        """
        The widest the jaws can open at xy without a finger landing on an
        obstacle: (opening_m, label of the obstacle that limits it).

        Each finger is a box finger_thickness_m deep beyond the gap and
        2 x finger_half_width_m across, kept neighbour_clearance_m along the
        jaw axis from every obstacle that reaches into its width — by as far
        as it reaches, so one off to the side of the fingers' path takes only
        what it reaches into it. At f1 on the full chess board (2026-11-03) the
        jaws open 21 deg off the rank: the g1 knight, 33 mm along their line
        and 17 mm aside, reaches 1 mm into a finger's width, and counted as
        standing on the line it left 20 mm where it leaves 32. tips_m: how far above the
        surface the finger tips come down to. An obstacle whose shape is
        measured is then as wide as it is from there up (half_width_at); else,
        and without tips_m, it is its foot circle, the widest it can be.
        turn_deg: the jaws turned (jaw_axis).
        """
        s = self.spec
        axis = self.jaw_axis(env, xy, turn_deg)
        across_axis = np.array([-axis[1], axis[0]])
        # Positions are in the arm's frame; feet and fingers are real metres.
        scale = float(env.frame_scale)
        room, limit = s.max_open_m, None
        for ob in obstacles:
            v = (np.asarray(ob.position[:2], float) - np.asarray(xy[:2], float)) * scale
            r = self.half_width_at(ob, tips_m)
            aside = abs(v @ across_axis) - s.finger_half_width_m
            if aside >= r:
                continue
            # How far toward the object it reaches, over the finger's width.
            reach = r if aside <= 0 else float(np.sqrt(r * r - aside * aside))
            gap = 2 * (abs(v @ axis) - reach - s.finger_thickness_m - s.neighbour_clearance_m)
            if gap < room:
                room, limit = gap, ob.label or "an obstacle"
        return room, limit

    def half_width_at(self, ob, tips_m: "float | None" = None) -> float:
        """
        How far an obstacle reaches out from its centre where the fingers
        pass, from tips_m above the surface up: its measured shape's widest
        there, else its foot. A shape that stops short of tips_m does not
        count. A partial outline (a pawn's head alone) is short and narrow,
        and sizing a neighbour by it would land a finger on its foot.

        On the full chess board (2026-11-03) the c1 bishop's 24 mm foot
        refused the d1 queen, though the fingers come down no lower than
        34 mm, where a bishop is 19 mm across.
        """
        s = self.spec
        foot = ob.radius_m if ob.radius_m is not None else s.unmeasured_obstacle_radius_m
        if tips_m is None or not ob.widths_mm:
            return foot
        above = [w for z, w in self.usable_widths(ob.widths_mm) if z >= tips_m * 1000 - 2.0]
        return max(above) / 2000 if above else foot

    def fit(self, env, plan: GraspPlan, xy, obstacles,
            turn_deg: float = 0.0) -> "tuple[GraspPlan, str | None, float]":
        """
        plan with its opening narrowed to the room the obstacles leave at xy
        down to where its jaw tips go, the jaws turned turn_deg: (plan,
        problem or None, room_m). A problem when not even the width plus
        slack fits.
        """
        s = self.spec
        width, opening = plan.width_m, plan.pregrasp_open_m
        room, limit = self.jaw_room(env, xy, obstacles, plan.grasp_height_m - s.tip_below_grasp_m, turn_deg)
        if room < min(opening, width + 2 * s.jaw_slack_m):
            return plan, (f"jaws need {(width + 2 * s.jaw_slack_m) * 1000:.0f} mm open here but "
                          f"{limit} leaves {max(room, 0) * 1000:.0f} mm"), room
        opening = min(opening, room, s.max_open_m)
        return dataclasses.replace(plan, pregrasp_open_m=opening), None, room

    # ── Waypoints ───────────────────────────────────────────────────────────────
    # Shared with the reach-mapping tools, which drive these exact waypoints to
    # map where a pick can work: a map made with looser tolerances would approve
    # places the pick then fails at.

    def pick_waypoints(self, xy, plan: GraspPlan) -> list:
        """[(phase, target, tol, max_step_deg)] for the vertical moves of a pick, in order."""
        s, (x, y) = self.spec, (float(xy[0]), float(xy[1]))
        return [
            ("hover", np.array([x, y, s.table_z + plan.hover_height_m]), s.travel_tol_m, None),
            ("grasp", np.array([x, y, s.table_z + plan.grasp_height_m]), s.grasp_tol_m, s.grasp_step_deg),
            ("lift",  np.array([x, y, s.table_z + plan.lift_height_m]), s.travel_tol_m, None),
        ]

    def place_waypoints(self, xy, plan: GraspPlan) -> list:
        """[(phase, target, tol, max_step_deg)] for the vertical moves of a place, in order."""
        s, (x, y) = self.spec, (float(xy[0]), float(xy[1]))
        return [
            ("transit", np.array([x, y, s.table_z + plan.lift_height_m]), s.travel_tol_m, None),
            ("release", np.array([x, y, s.table_z + plan.release_height_m]), s.release_tol_m,
             s.grasp_step_deg),
            ("retreat", np.array([x, y, s.table_z + plan.hover_height_m]), s.travel_tol_m, None),
        ]

    def hold_orientation(self, env, waypoints: list, tag: str = "", obstacles=(),
                         tilt_deg: "float | None" = None,
                         margin_deg: "float | None" = None, turn_deg: float = 0.0) -> "np.ndarray | None":
        """
        One gripper orientation for a whole vertical sequence, or None to let
        each move choose its own (the fallback). waypoints: [(phase, target, tol), ...].

        A two-finger gripper can grip from either of two wrist angles half a
        turn apart — the jaws open along the same line — but whatever is
        mounted beside the jaws (spec.camera_body_m: a wrist camera) is on
        opposite sides in the two. Both are tried; of those the arm can reach,
        the one that keeps that body clear of the obstacles is held; then the
        one that tilts least; then the one that needs no turn of the wrist from
        how it is now (turning sweeps the camera round — not something to do
        with an object in the jaws when it is not needed).

        tilt_deg: a lean the scenario's rules ask for here (Target.tilt_deg).
        The arm's search starts from it rather than from straight down, and
        "nearest to it" replaces "tilts least". margin_deg: hold the least lean
        that leaves every joint this far from its limit, where the backend can
        say (Approach.lean_margin_deg); else the least that reaches. turn_deg:
        the jaws turned that far about the approach axis from how the arm
        opens them (OrientationSearch).
        """
        rot, _, checked = self._hold(env, waypoints, tag, obstacles, tilt_deg, margin_deg, turn_deg=turn_deg)
        if rot is None and checked:
            self._say(tag, "No single orientation reaches every phase here — "
                           "letting each move choose its own.")
        return rot

    @staticmethod
    def _orientation(env, targets, tols, flip, tilt_deg, margin_deg, turn_deg: float = 0.0):
        """
        env.approach_orientation_for with what this backend takes of the lean
        asked for, the joint margin and the flip: (rot, tilt) or None. Each one
        a backend does not take is dropped, the margin first. A turn is never
        dropped: jaws not turned as asked are none (callers check the backend
        takes turn_deg before asking).
        """
        prefers = (tilt_deg, None) if tilt_deg is not None else (None,)
        margins = (margin_deg, None) if margin_deg is not None else (None,)
        for prefer in prefers:
            for margin in margins:
                kwargs = {"tols": tols, **({"flip": True} if flip else {}),
                          **({"turn_deg": turn_deg} if turn_deg else {}),
                          **({} if prefer is None else {"prefer_tilt_deg": prefer}),
                          **({} if margin is None else {"margin_deg": margin})}
                try:
                    return env.approach_orientation_for(targets, **kwargs)
                except TypeError:
                    continue
        return None

    def _hold(self, env, waypoints: list, tag: str = "", obstacles=(), tilt_deg: "float | None" = None,
              margin_deg: "float | None" = None, quiet: bool = False,
              tilt_why: str = "the approach rule here asks for", turn_deg: float = 0.0,
              open_m: "float | None" = None) -> tuple:
        """
        hold_orientation's choice as (rot, tilt_deg, checked). (None, None,
        False): the backend cannot check orientations, so each move chooses
        its own. (None, None, True): it checked, and nothing reaches every
        waypoint. quiet: say nothing (a feasibility check, not a move).
        open_m: how far apart the fingers are coming down (sweep_clearance).
        """
        self._swept = None
        if not isinstance(env, OrientationSearch):
            return None, None, False
        say = (lambda *_: None) if quiet else self._say
        targets, tols = [w[1] for w in waypoints], [w[2] for w in waypoints]
        options, margined = [], False
        for margin in ((margin_deg, None) if margin_deg is not None else (None,)):
            for flip in (False, True):
                chosen = self._orientation(env, targets, tols, flip, tilt_deg, margin, turn_deg)
                if chosen is not None:
                    rot, tilt = chosen
                    options.append((flip, rot, tilt, self.body_clearance(env, rot, targets[0], obstacles)))
            if options:
                margined = margin is not None
                if margin is None and margin_deg is not None:
                    say(tag, f"No lean leaves every joint {margin_deg:.0f} deg from its limit here — "
                             "holding the least that reaches.")
                break
        if not options:
            return None, None, True
        # A lean that would sweep the gripper into what stands toward the base
        # is not one it can hold: leaning far over, the fingers pass low over
        # the next square in.
        low = min(targets, key=lambda t: float(np.asarray(t, float)[2]))
        swept = [(o, self.sweep_clearance(env, o[1], low, obstacles, open_m)) for o in options]
        clear_of = [o for o, sw in swept if sw is None or sw[0] >= self.spec.sweep_margin_m]
        if not clear_of:
            gap, near = min((sw for _, sw in swept if sw is not None), key=lambda g: g[0])
            self._swept = (self.spec.sweep_margin_m - gap, near)
            say(tag, f"Every lean that reaches it would sweep the gripper {(self.spec.sweep_margin_m - gap) * 1000:.0f} "
                     f"mm too low over {near}.")
            return None, None, True
        options = clear_of
        want = self.spec.camera_clearance_m
        now = env.get_ee_rotation() if isinstance(env, GripperRotation) else None

        def preference(option):
            flip, rot, tilt, clear = option
            as_now = now is None or float(now[:, 0] @ rot[:, 0]) >= 0.0
            # Clear enough beats everything; then the least tilt (or nearest the
            # one asked for); then no turn; then unturned.
            lean = -round(abs(abs(tilt) - abs(tilt_deg))) if tilt_deg is not None else -round(abs(tilt))
            return (clear is None or clear[0] >= want, clear[0] if clear and clear[0] < want else 0.0,
                    lean, as_now, not flip)

        flip, rot, tilt, clear = max(options, key=preference)
        if tilt_deg is not None:
            say(tag, f"Holding a {tilt:+.0f} deg tilt ({tilt_why} ~{abs(tilt_deg):.0f}).")
        elif margined:
            say(tag, f"Holding a {tilt:+.0f} deg tilt: the least that leaves every joint "
                     f"{margin_deg:.0f} deg from its limit.")
        elif abs(tilt) > 0.5:
            say(tag, f"Straight down is out of reach here — holding a {tilt:+.0f} deg tilt.")
        if clear is not None:
            gap, near = clear
            if flip:
                say(tag, f"Wrist turned half a turn: keeps the camera {gap * 1000:.0f} mm "
                         f"clear of {near}.")
            elif gap < want:
                say(tag, f"The camera passes {gap * 1000:.0f} mm from {near} either way round.")
        return rot, float(tilt), True

    def sweep_clearance(self, env, rot, at, obstacles, open_m: "float | None" = None) -> "tuple[float, str] | None":
        """
        Leaning, how far the gripper's underside passes over the tops of what
        stands under its line back toward the wrist, with the jaws at `at` in
        orientation `rot`: (metres, the nearest one's label) — negative, it
        would hit it — or None when it passes over nothing (straight down, or
        nothing under it, or no heights known). What stands beside the jaws,
        across the line, is jaw_room's.

        The gripper's line rises from the jaws along its approach axis
        (rot's third column, pointing at the jaws); the underside is
        finger_thickness_m below it, sweep_half_width_m to either side. What
        stands on the line, narrow enough where the fingers pass to go between
        them (open_m apart, else sweep_free_width_m; a neighbour not measured
        counts as wide as its assumed shape, the widest pieces are there)
        and within their length (free_depth_m) of the jaws, passes between
        them: the jaws open across the line only with tangential jaws. An obstacle counts from its foot's near edge, at its
        height_m (measured, or the tallest it may be). 2026-11-04: rank 8
        reaches at 60-80 deg, with the rank-7 pawn in front 40 mm in.
        """
        s = self.spec
        back = -np.asarray(rot, float)[:, 2]
        across = float(np.linalg.norm(back[:2]))
        if across < 1e-3 or back[2] <= 1e-6:
            return None
        d = back[:2] / across
        rise = back[2] / across
        at = np.asarray(at, float)
        scale = float(env.frame_scale)
        best = None
        for ob in obstacles:
            if ob.height_m is None:
                continue
            rel = (np.asarray(ob.position[:2], float) - at[:2]) * scale
            along = float(rel @ d)
            r = ob.radius_m if ob.radius_m is not None else s.unmeasured_obstacle_radius_m
            lateral = abs(float(rel @ np.array([-d[1], d[0]])))
            if along <= r or lateral > s.sweep_half_width_m + r:
                continue
            under = at[2] + (along - r) * rise - s.finger_thickness_m
            free = open_m if open_m is not None else s.sweep_free_width_m
            if (s.jaw_axis == "tangential" and along / across < s.free_depth_m
                    and lateral + self.half_width_at(ob, max(under - s.table_z, 0.0)) <= free / 2):
                continue                       # between the fingers
            gap = under - (s.table_z + float(ob.height_m))
            if best is None or gap < best[0]:
                best = (gap, ob.label or "an obstacle")
        return best

    def body_clearance(self, env, rot, at, obstacles) -> "tuple[float, str] | None":
        """
        How far spec.camera_body_m clears the nearest obstacle with the gripper
        at `at` in orientation `rot`: (metres, that obstacle's label), or None
        with no body or no obstacles. Negative: it would touch.
        """
        body = self.spec.camera_body_m
        if body is None or not obstacles:
            return None
        x, y, radius = body
        where = np.asarray(at, float)[:2] + (np.asarray(rot, float) @ np.array([x, y, 0.0]))[:2]
        scale = float(env.frame_scale)
        best = None
        for ob in obstacles:
            r = ob.radius_m if ob.radius_m is not None else self.spec.unmeasured_obstacle_radius_m
            gap = float(np.linalg.norm((np.asarray(ob.position[:2], float) - where) * scale)) - r - radius
            if best is None or gap < best[0]:
                best = (gap, ob.label or "an obstacle")
        return best

    # ── Approaches ──────────────────────────────────────────────────────────────

    @staticmethod
    def _base_xy(env) -> np.ndarray:
        """Where the arm stands (RobotEnv.base_xy): what "near the base" and "far out" are measured from."""
        return np.asarray(env.base_xy, float)[:2]

    def _with_base(self, env, obstacles) -> list:
        """
        obstacles, and the arm's own base as one more (spec.base_keep_out_m):
        a finger must not land on it, and the wrist camera must not be turned
        into it. Turned half a turn over rank 1-2, the camera on this rig
        would hang 18-57 mm from the pan axis — inside the shoulder.
        """
        r = self.spec.base_keep_out_m
        if r is None:
            return list(obstacles)
        return [*obstacles, Obstacle(self._base_xy(env), float(r), BASE_LABEL)]

    def zone(self, env, xy) -> str:
        """"near", "mid" or "far": where xy stands from the arm's base (spec.near_base_m, far_reach_m)."""
        s = self.spec
        r = float(np.linalg.norm(np.asarray(xy[:2], float) - self._base_xy(env)))
        if s.near_base_m is not None and r < s.near_base_m:
            return "near"
        if s.far_reach_m is not None and r > s.far_reach_m:
            return "far"
        return "mid"

    def approach_order(self, env, xy, plan: "GraspPlan | None" = None) -> list:
        """
        spec.approaches to try at xy: those for its zone first, then the rest,
        each in listed order. plan given: a "mid" spot that straight down needs
        more lean than spec.far_from_lean_deg for counts as "far" (_leans_far).
        """
        z = self.zone(env, xy)
        if z == "mid" and plan is not None and self._leans_far(env, xy, plan):
            z = "far"
        return ([a for a in self.spec.approaches if z in a.zones]
                + [a for a in self.spec.approaches if z not in a.zones])

    def _leans_far(self, env, xy, plan: GraspPlan) -> bool:
        """
        Whether, over xy, the hover, the grasp and the lift need more lean than
        spec.far_from_lean_deg — or cannot be had at all in one orientation.
        Far by distance was set on chess pieces (45 mm, a 70 mm hover); a plug
        hovers at 97 mm, and the same distance needs more lean. 2026-11-19, the
        SO-101: a plug at 32 cm, "mid-reach", went top_down, the wrist servo's
        held lean; its descents stalled, three times. Leaning is reach_out's.
        """
        s = self.spec
        if s.far_from_lean_deg is None or not isinstance(env, OrientationSearch):
            return False
        xy = np.asarray(xy[:2], float)
        points = [np.array([xy[0], xy[1], s.table_z + h])
                  for h in (plan.hover_height_m, plan.grasp_height_m, plan.lift_height_m)]
        try:
            found = env.approach_orientation_for(points, prefer_tilt_deg=0.0)
        except TypeError:
            return False
        return found is None or abs(float(found[1])) > s.far_from_lean_deg

    # A deployment's containers, () -> [Opening]: what a pick's way to its hover keeps the jaws above
    # (spec.transit_over_walls). Set by the deployment, as Sighting.ask_held is.
    walls: "Callable[[], list] | None" = None

    def _over_walls(self, env, to, tag: str = "", rot=None) -> bool:
        """
        Before going to `to`: where the straight way there crosses a container's
        walls, or starts inside them, up over its rim first (the open jaws'
        tips clear of it), across at that height until past them, then on.
        False: a step of it did not get there (nothing moved on from there).
        2026-11-19 11:18, the SO-101: the plug let go of in the tub, the arm up
        to 112 mm, and from there straight to the next object's hover at 61 mm
        across the tub's wall — coming down as it crossed, the jaws caught the
        wall and dragged the tub across the table.
        """
        from mrcf.policies.place_in import _polygon, inside_by

        s = self.spec
        if not s.transit_over_walls or self.walls is None:
            return True
        try:
            openings = list(self.walls() or [])
        except Exception:                                   # noqa: BLE001 — none known: straight there
            return True
        here, to = np.asarray(env.get_ee_pos(), float), np.asarray(to, float)
        a, b = here[:2], to[:2]
        span = float(np.linalg.norm(b - a))
        n = max(2, int(span / 0.01) + 1)
        # Near enough the walls to meet them: the fingers' own thickness and the clearance. Not the
        # whole open span: from home, every pick's way passed within that of the tub, and rose for it.
        reach = s.container_wall_clearance_m + s.finger_thickness_m
        need, last = None, None
        for op in openings:
            try:
                poly = _polygon(op)
            except ValueError:
                continue
            near = [k for k in range(n + 1) if inside_by(poly, a + (b - a) * k / n) > -reach]
            if not near:
                continue
            z = (s.table_z + float(op.rim_m) + s.container_wall_clearance_m + s.container_cross_extra_m
                 + s.tip_below_grasp_m + 0.01)
            need = z if need is None else max(need, z)
            last = max(near) if last is None else max(last, max(near))
        if need is None or (here[2] >= need and to[2] >= need):
            return True
        assert last is not None
        past = min(1.0, (last + 1) / n + (reach / span if span > 0 else 1.0))
        exit_xy = a + (b - a) * past
        self._say(tag, f"Up over the container's walls ({(need - s.table_z) * 1000:.0f} mm) before moving across.")
        steps = []
        if here[2] < need - s.travel_tol_m:
            steps.append(np.r_[a, need])
        steps.append(np.r_[exit_xy, max(need, float(to[2]))])
        for p in steps:
            if s.transit_settle:
                env.move_ee_to(p, tol=s.travel_tol_m, orientation=rot)
            else:
                self._move_through(env, p, s.travel_tol_m, rot)
            if not self._near(env, p, 2 * s.travel_tol_m):
                self._say(tag, "Did not get up over the container's walls — not moving across.")
                return False
        return True

    def _lift_heights(self, plan: GraspPlan) -> list:
        """The heights a "highest" lift tries, highest first: the plan's, then lower, down to the hover."""
        out, h = [plan.lift_height_m], plan.lift_height_m - _LIFT_STEP_M
        while h > plan.hover_height_m + 1e-9:
            out.append(h)
            h -= _LIFT_STEP_M
        if plan.hover_height_m < plan.lift_height_m:
            out.append(plan.hover_height_m)
        return out

    def _held_for(self, env, approach: Approach, xy, plan: GraspPlan, obstacles=(), tilt_deg=None,
                  tag: str = "", quiet: bool = True, start=None, servo_z: "float | None" = None,
                  lift_h: "float | None" = None, turn_deg: float = 0.0) -> tuple:
        """
        The orientation `approach` holds for a pick at xy planned as `plan`:
        (rot, tilt, checked, lift_height_m), as _hold. start: where the
        sequence starts instead of the hover (a re-hold mid-pick); servo_z: the
        wrist servo's hover, world z, when it aligns from there; lift_h: the
        lift height, else as approach.lift says — "highest" tries the plan's,
        then lower, down to the hover.
        """
        s = self.spec
        xy = np.asarray(xy[:2], float)
        (_, hover, hover_tol, _), (_, grasp, grasp_tol, _), (_, _, lift_tol, _) = self.pick_waypoints(xy, plan)
        first = hover if start is None else np.asarray(start, float)
        if lift_h is not None:
            lifts = [lift_h]
        elif approach.lift == "highest":
            lifts = self._lift_heights(plan)
        else:
            lifts = [plan.lift_height_m]
        for height in lifts:
            lift = np.array([xy[0], xy[1], s.table_z + height])
            points = self._pick_reach(first, hover_tol, grasp, grasp_tol, lift, lift_tol, env,
                                      inward=approach.reach_margin == "in")
            if servo_z is not None and start is None:
                points.insert(1, ("servo hover", np.array([xy[0], xy[1], servo_z]), hover_tol))
            rot, tilt, checked = self._hold(env, points, tag, obstacles, tilt_deg, approach.lean_margin_deg,
                                            quiet=quiet, turn_deg=turn_deg, open_m=plan.pregrasp_open_m)
            if not checked:
                return None, None, False, lifts[0]
            if rot is not None:
                return rot, tilt, True, height
        return None, None, True, None

    def fit_approach(self, env, approach: Approach, xy, plan: GraspPlan, obstacles=(),
                     tilt_deg: "float | None" = None, servo_ready: bool = False,
                     widths_mm: "dict | None" = None, measured: bool = False,
                     room: "float | None" = None, turn_deg: float = 0.0) -> "ApproachFit | str":
        """
        `approach` worked out for an object at xy planned as `plan`: the
        orientation it would hold, how high it can lift, whether the wrist
        servo aligns — or, as a str, why it cannot be done here. Pure
        kinematics: nothing moves. servo_ready: this rig's wrist servo can run
        (_servo_ready). widths_mm, measured: the object's shape, when it has
        been measured (keep_top_clear re-plans the grasp from it). room: the
        widest the neighbours let the jaws open (fit), for _widened.

        The waypoints are the pick's own (hover, the servo's hover when it
        aligns, grasp, the grasp's reach margin, lift), so an approach approved
        here is one the pick can drive.
        """
        s = self.spec
        xy = np.asarray(xy[:2], float)
        plan = self._own_hover(approach, self._widened(plan, room))
        low = self._servo_height_for(np.array([0.0, 0.0, s.table_z + plan.hover_height_m]))
        servo_z = s.table_z + low if low is not None else None
        use_servo = approach.servo and servo_ready
        gated = use_servo and approach.servo_max_lean_change_deg is not None and s.servo_tilt_deg is not None
        rot, tilt, checked, lift_h = self._held_for(env, approach, xy, plan, obstacles, tilt_deg,
                                                    servo_z=servo_z if use_servo and not gated else None,
                                                    turn_deg=turn_deg)
        over = None
        swept = getattr(self, "_swept", None)
        if checked and rot is None and swept is not None:
            # Every lean that reaches it sweeps the gripper into what stands in
            # front: gripped higher, the whole gripper passes higher.
            higher = self._grip_over(plan, swept[0])
            if higher is not None:
                again = self._held_for(env, approach, xy, higher, obstacles, tilt_deg,
                                       servo_z=servo_z if use_servo and not gated else None, turn_deg=turn_deg)
                if again[0] is not None:
                    plan, over = higher, higher.grasp_height_m
                    rot, tilt, checked, lift_h = again
        if checked and rot is None:
            what = "the servo's hover, " if use_servo and not gated and servo_z is not None else ""
            return (f"{approach.name}: no orientation reaches its hover, {what}grasp "
                    f"({plan.grasp_height_m * 1000:.0f} mm) and lift")
        skipped = servo_ready and not approach.servo
        if gated and tilt is not None:
            if abs(tilt - s.servo_tilt_deg) > approach.servo_max_lean_change_deg:
                # The servo's target pixel holds at the lean it was set at; at
                # 20-50 deg more, where "aligned" leaves the jaws rests on a
                # model, and the camera looks along the board at the object.
                use_servo, skipped = False, True
            elif servo_z is not None:
                with_servo = self._held_for(env, approach, xy, plan, obstacles, tilt_deg, servo_z=servo_z,
                                            turn_deg=turn_deg)
                if with_servo[0] is not None:
                    rot, tilt, checked, lift_h = with_servo
                else:
                    use_servo, skipped = False, True
        if tilt is not None:
            leaned = self.pads_clear_at(plan, tilt)
            if leaned is not plan:
                again = self._held_for(env, approach, xy, leaned, obstacles, tilt_deg,
                                       servo_z=servo_z if use_servo else None, turn_deg=turn_deg)
                if again[0] is not None or not again[2]:
                    plan = leaned
                    rot, tilt, checked, lift_h = again
        if approach.keep_top_clear and measured and tilt is not None:
            clear = self._top_clear(plan, widths_mm, tilt)
            if clear is not plan:
                again = self._held_for(env, approach, xy, clear, obstacles, tilt_deg,
                                       servo_z=servo_z if use_servo else None, turn_deg=turn_deg)
                if again[0] is not None or not again[2]:
                    plan = clear
                    rot, tilt, checked, lift_h = again
        return ApproachFit(approach, plan, rot, tilt,
                           lift_h if lift_h is not None else plan.lift_height_m,
                           servo=use_servo, checked=checked, servo_skipped=skipped, grasp_over_m=over)

    def _grip_over(self, plan: GraspPlan, short_m: float) -> "GraspPlan | None":
        """
        plan, gripped higher by what the gripper passed short_m too low over
        the piece in front (sweep_clearance), and a few mm more for the lean
        to change with it — or None when that is above the highest grasp the
        object allows (its height less spec.grip_over_top_m). The user's
        suggestion, 2026-11-04: a back-rank piece with a pawn in front, gripped
        higher, is picked over it.
        """
        s = self.spec
        g = plan.grasp_height_m + short_m + 0.004
        top = plan.height_m - s.grip_over_top_m
        if g > top:
            return None
        return dataclasses.replace(plan, grasp_height_m=float(np.ceil(g * 1000) / 1000))

    def choose_approach(self, env, xy, plan: GraspPlan, obstacles=(), tilt_deg: "float | None" = None,
                        servo_ready: bool = False, widths_mm: "dict | None" = None,
                        measured: bool = False, room: "float | None" = None,
                        turn_deg: float = 0.0) -> "tuple[ApproachFit | None, list]":
        """
        The approach a pick at xy starts with: (fit, why the ones before it in
        approach_order cannot be done). (None, reasons): none can. For tools
        that show what the pick would do (the reach maps).
        """
        choices = _Choices(self.approach_order(env, xy, plan), plan,
                           lambda a, p: self.fit_approach(env, a, xy, p, obstacles, tilt_deg, servo_ready,
                                                          widths_mm, measured, room, turn_deg))
        return choices.feasible(0), choices.why_not

    def _own_hover(self, approach: Approach, plan: GraspPlan) -> GraspPlan:
        """plan, hovering only as high as the object needs, when the approach says so (Approach.own_hover)."""
        if not approach.own_hover:
            return plan
        s = self.spec
        need = plan.height_m + s.tip_below_grasp_m + s.top_clearance_m
        return plan if need >= plan.hover_height_m else dataclasses.replace(plan, hover_height_m=need)

    def _pick_reach(self, hover, hover_tol, grasp, grasp_tol, lift, lift_tol, env=None,
                    inward: bool = False) -> list:
        """
        The waypoints a pick's held orientation must reach: its own, and its
        grasp spec.servo_reach_margin_m further from the base (inward: nearer
        it). The orientation is chosen before the servo or the fixed camera
        moves the grasp, and a grasp they move out can be past what that tilt
        reaches. Further from the base is where reach runs out in the middle
        and far out; nearer it, close to the base.
        """
        away = np.asarray(grasp[:2], float) - self._base_xy(env)
        away = away / (np.linalg.norm(away) or 1.0) * self.spec.servo_reach_margin_m * (-1.0 if inward else 1.0)
        shifted = np.array([grasp[0] + away[0], grasp[1] + away[1], grasp[2]])
        return [("hover", hover, hover_tol), ("grasp", grasp, grasp_tol),
                ("grasp, nearer the base" if inward else "grasp, further out", shifted, grasp_tol),
                ("lift", lift, lift_tol)]

    def _back_off(self, env, plan: GraspPlan, tag: str, reached=None) -> None:
        """
        Straight up to the hover height, the gripper held as it is, when an
        attempt left the jaws down beside the object. Opening them there and
        moving off to the next attempt's hover (another lean, another spot)
        sweeps the fingers through it at its own height and pushes it. A pick
        that fails backs off too, rather than leave the jaws there for
        whatever moves next. reached: the hover the attempt came down from;
        with spec.back_off_to_attempt_hover no higher than that — it cleared
        the object, and the arm got there.
        """
        s = self.spec
        here = env.get_ee_pos()
        up = s.table_z + plan.hover_height_m
        if s.back_off_to_attempt_hover and reached is not None:
            up = min(up, float(np.asarray(reached, float)[2]))
        if here[2] >= up - s.travel_tol_m:
            return
        rot = env.get_ee_rotation() if isinstance(env, GripperRotation) else None
        self._say(tag, f"Backing straight up to {(up - s.table_z) * 1000:.0f} mm before anything else moves.")
        self._move_through(env, np.array([here[0], here[1], up]), s.travel_tol_m, rot)

    def _move_again_if_short(self, env, goal, tol: float, rot, arrived) -> bool:
        """
        A long move that stopped well short (further than a hover's 2 x
        approach_tol_m), asked for once more from where it stopped; whether
        it arrived. A long move can stop centimetres short of a hover that the
        same move reaches from there in a second or two; given up on, each
        would cost an attempt and a change of approach.
        """
        if arrived is not False or self._near(env, goal, self.spec.approach_tol_m * 2):
            return arrived
        return env.move_ee_to(goal, tol=tol, orientation=rot)

    @staticmethod
    def _move_through(env, goal, tol: float, rot) -> bool:
        """
        A move nothing is done at the end of — a lift, a retreat, a back-off —
        not waited on to settle where the backend can skip that (its
        move_ee_to takes settle=): each settle holds the arm still for most of
        a second.
        """
        if _takes(env.move_ee_to, "settle"):
            return env.move_ee_to(goal, tol=tol, orientation=rot, settle=False)
        return env.move_ee_to(goal, tol=tol, orientation=rot)

    @staticmethod
    def _tilt_of(env, rot) -> "float | None":
        """How far rot leans from straight down, when the env can say."""
        return env.gripper_tilt_deg(rot) if rot is not None and isinstance(env, GripperTilt) else None

    def _stalled_at_start(self, env, top_z: float, goal) -> bool:
        """
        Whether a descent from top_z to goal stalled before it had come
        _START_STALL_SHARE of the way down, said aloud if so. 2026-11-04, a pick
        on e4 at -15 deg: stalled 7.3 cm short of a 7.2 cm descent — every
        joint within 3 deg of its command, none moving. Leaning further, it
        reached, and the close at -25 squeezed the pawn out of the jaws.
        """
        z = float(env.get_ee_pos()[2])
        drop = float(top_z - goal[2])
        if drop <= 0 or (top_z - z) > _START_STALL_SHARE * drop:
            return False
        self._say("[Reach]", f"The descent stalled before it got going ({(top_z - z) * 1000:.0f} of "
                             f"{drop * 1000:.0f} mm down) — up, and down again at the same lean.")
        return True

    def _leaned_further(self, env, tilt: float, waypoints: list, goal, tag: str, obstacles, margin_deg,
                        turn_deg: float = 0.0) -> tuple:
        """
        For a descent that stalled short of goal at tilt: the orientation
        spec.descent_lean_step_deg further from straight down that holds over
        waypoints, and its tilt, said aloud. (None, None) when nothing leans
        at least a degree further.
        """
        s = self.spec
        want = tilt - s.descent_lean_step_deg if tilt <= 0 else tilt + s.descent_lean_step_deg
        new_rot, new_tilt, _ = self._hold(env, waypoints, tag, obstacles, want, margin_deg, quiet=True,
                                          turn_deg=turn_deg)
        if new_rot is None or abs(new_tilt) < abs(tilt) + 1.0:
            return None, None
        stalled = float(np.linalg.norm(env.get_ee_pos() - goal))
        self._say(tag, f"The descent stalled {stalled * 100:.1f} cm short at {tilt:+.0f} deg — up, and down "
                       f"again leaning {new_tilt:+.0f}.")
        return new_rot, new_tilt

    # ── Sensing ─────────────────────────────────────────────────────────────────

    def _near(self, env, target, tol) -> bool:
        return float(np.linalg.norm(env.get_ee_pos() - target)) < tol

    def _servo_height_for(self, hover) -> "float | None":
        """
        The height (above the surface) to servo at, when it is not the hover the
        arm is at: spec.servo_height_m, if set and more than a few mm away.
        """
        s = self.spec
        if s.servo_height_m is None or abs(hover[2] - s.table_z - s.servo_height_m) <= _SERVO_HEIGHT_TOL_M:
            return None
        return s.servo_height_m
