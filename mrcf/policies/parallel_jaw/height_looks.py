"""
mrcf/policies/parallel_jaw/height_looks.py — how tall the object is, found out
by looking: the wrist camera sent to where the object's height shows, and the
object's shape fitted to its outlines in every calibrated view at once
(mrcf/perception/shape_fit.py). Before a pick plans its grasp (survey_height),
and once more from close above on the way down (_approach_look).

Why it looks from the side: 2026-11-16, the SO-101 tabletop. Nothing had
measured the balls, so every pick planned for the tallest object there could
be (150 mm) and closed the jaws 117 mm up, over a ball 25 mm across; and the
look at the hover could not help — 8 deg from straight down, a mm of the ball's
height moved its top 0.21 px. From 6-12 cm beside it, leaning in, the same
camera sees 0.9-1.5 px a mm (offline, the rig's own kinematics and camera).

How a look is chosen: poses round the object (how far, which side, how high,
how far the gripper leans, the camera either side of it), scored by what the
camera would see there — the object in frame and clear of the jaws, its foot
not hidden behind a neighbour, a mm of its height moving a pixel or more, and
from a side not yet looked from — before any of them is asked of the arm's
kinematics. The best few are then checked for reach and for the gripper's room
among the objects (each as tall as it could be, where not measured).

What it gives: the object's height and its width up it, and the spread of
heights that fit as well; settled only when the fit is resolved and at least
two looks from the side see it. Not settled, the pick is refused, with why:
it does not close on a guess.

The runs behind its values and fixes, with their dates: mrcf/policies/HISTORY.md.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from mrcf import flight_recorder
from mrcf.capabilities import CameraAtPose, DownRotation, GripperRotation, OrientationSearch
from mrcf.policies.base import Measured, Obstacle, Target
from mrcf.policies.parallel_jaw.sighting import SightingMixin

if TYPE_CHECKING:
    from mrcf.perception.shape_fit import ShapeFit

# ── Where a look may be taken from ─────────────────────────────────────────────
# The jaws this far from the object, toward each of these sides of it (0: from
# the arm's base), this high above the surface, the gripper leaning this far in
# the arm's own plane, the camera either side.
LOOK_DISTANCES_M = (0.03, 0.045, 0.06, 0.09, 0.12)   # the near ones from over its top: near the base the
                                                      # far ones are out of reach or over the base
LOOK_SIDES_DEG = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
LOOK_HEIGHTS_M = (0.07, 0.10, 0.14, 0.18, 0.22, 0.26)    # the high ones clear neighbours not yet measured
LOOK_LEANS_DEG = (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0)
TRY_BUDGET_S = 3.0          # how long the best-scored poses are checked for reach, best first, before giving up
GOOD_ENOUGH = 0.9           # ... and the first that sees this well (px per mm of height, as scored) is taken
SAME_SIDE_COS = 0.8         # a look within ~37 deg of one already taken counts as the same side
SEEING_IOU = 0.6            # a look counts toward settling the height when the fit matches it this well ...
SEEING_PX_PER_MM = 0.5      # ... and a mm of the object's height moves its top this many px in it
OUTLIER_IOU = 0.5           # a look whose outline matches the others' fit worse than this is set aside
BOUND_DROP = 0.10           # how tall an object not measured can be, from what has been seen of it:
BOUND_MARGIN_MM = 10.0      # ... the fit's bound_mm at this drop, and this much more (ShapeFit.bound_mm)
# Not resolved, but bounded: the solids that fit about as well differ in height by
# no more than this (or this share of the shortest), each seen from the side — a
# flat 30 mm box and a 40 mm one with a rounded top, outlines 4 px apart. Gripped
# as the shortest is, the hover clear of the tallest: right for every one of them.
BOUNDED_MM = 15.0
BOUNDED_SHARE = 0.5


@dataclass
class HeightLooks:
    """What a pick has seen of the object's height: its outlines so far, in every calibrated view, and the fit."""
    footprint: object                     # shape_fit.Footprint or OutlineFootprint, the fit's frame
    silhouettes: list                     # shape_fit.Silhouette, the fixed camera's first
    known: list                           # expected_view.KnownObject, the scan's frame: "target" first
    shift: np.ndarray                     # a scan position in the wrist camera's frame, less in the scan's
    height_range_mm: tuple
    fit: "ShapeFit | None" = None
    looked_from: list = field(default_factory=list)   # unit (x, y) from the object toward each wrist look
    tried: list = field(default_factory=list)         # look poses that came to nothing
    reach: dict = field(default_factory=dict)         # what the arm can hold at each look pose asked about
    models: list = field(default_factory=list)        # each wrist silhouette's camera model, in order

    @property
    def target_xy(self) -> np.ndarray:
        return np.asarray(self.known[0].foot_xy_m, float)[:2]

    def refit(self) -> "ShapeFit | None":
        from mrcf.perception.shape_fit import fit_shape
        self.fit = fit_shape(self.silhouettes, self.footprint, self.height_range_mm)
        return self.fit

    def wrist_looks(self) -> int:
        return sum(1 for x in self.silhouettes if not x.fixed)

    def seeing(self) -> int:
        """The wrist looks the fit matches that see the object's height (SEEING_IOU, SEEING_PX_PER_MM)."""
        f = self.fit
        if f is None:
            return 0
        return sum(1 for sil, iou, pp in zip(self.silhouettes, f.ious, f.px_per_mm)
                   if not sil.fixed and iou >= SEEING_IOU and pp >= SEEING_PX_PER_MM)


class HeightLookMixin(SightingMixin):
    """ParallelJaw's looks for an object's height (see the module docstring)."""

    _height_looks: "HeightLooks | None" = None

    # ── Whether it can, and what it starts from ─────────────────────────────────

    def height_unknown(self, target: Target) -> bool:
        """Nothing says how tall target is: not measured, and the deployment does not know it by its kind."""
        return not target.measured and not self.height_known(target)

    def cannot_look_for_height(self, env) -> "str | None":
        """Why this arm cannot look for an object's height (None: it can)."""
        s = self.spec
        if s.camera is None:
            return "there is no wrist camera"
        if not (isinstance(env, CameraAtPose) and isinstance(env, OrientationSearch)
                and isinstance(env, DownRotation)):
            return "this arm cannot work out where its wrist camera would be"
        if not self.camera_located(env):
            return "the wrist camera is not located, or not sending"
        return None

    def _known(self, target: Target, obstacles) -> list:
        """The object, keyed "target", and what stands round it (KnownObject), each as tall as it is or could be."""
        from mrcf.perception.expected_view import KnownObject
        from mrcf.policies.parallel_jaw.reach import BASE_LABEL

        s = self.spec
        tallest = target.max_height_m or s.default_max_height_m
        out = [KnownObject("target", np.asarray(target.position[:2], float),
                           (target.foot_m or 2 * s.unmeasured_obstacle_radius_m) / 2, target.height_m or tallest)]
        for i, ob in enumerate(obstacles):
            if ob.label == BASE_LABEL:
                continue
            out.append(KnownObject(f"{i}:{ob.label}", np.asarray(ob.position[:2], float),
                                   ob.radius_m or s.unmeasured_obstacle_radius_m, ob.height_m or tallest))
        return out

    def _height_start(self, env, target: Target, obstacles) -> "HeightLooks | str":
        """What a survey starts from: the fixed camera's outline of it (Target.outlines), where it stands, its neighbours."""
        from mrcf.perception import shape_fit

        why = self.cannot_look_for_height(env)
        if why:
            return why
        s = self.spec
        model = env.camera_model(s.camera, s.render_w, s.render_h)
        shift = np.zeros(2) if model is None or model.get("scan_to_camera") is None \
            else np.asarray(model["scan_to_camera"], float)[:2]
        sils, footprint = [], None
        for camera, outline in (target.outlines or {}).items():
            got = self._fixed_silhouette(env, camera, outline, shift)
            if got is not None:
                sils.append(got[0])
                footprint = footprint or got[1]
        if footprint is None:
            footprint = shape_fit.Footprint.circle(np.asarray(target.position[:2], float) + shift,
                                                   target.foot_m or 2 * s.unmeasured_obstacle_radius_m)
        lo = max(4.0, (target.min_height_m or 0.0) * 1000.0 - 3.0)
        hi = max(lo + 10.0, (target.max_height_m or s.default_max_height_m) * 1000.0)
        return HeightLooks(footprint, sils, self._known(target, obstacles), shift, (lo, hi))

    @staticmethod
    def _fixed_silhouette(env, camera: str, outline: dict, shift):
        """(Silhouette, OutlineFootprint) of a fixed camera's outline of it ({"size", "px"}), in the wrist camera's frame."""
        import cv2

        from mrcf.perception import shape_fit
        from mrcf.perception.plane_geometry import PlaneView

        try:
            w, h = (int(v) for v in outline["size"])
            px = np.asarray(outline["px"], float).reshape(-1, 2)
        except (KeyError, TypeError, ValueError):
            return None
        model = env.camera_model(camera, w, h)
        if model is None or len(px) < 3:
            return None
        view = shape_fit.shifted(PlaneView.from_camera_model(model), shift)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
        return (shape_fit.Silhouette(view, [mask.astype(bool)], None, name=camera, fixed=True),
                shape_fit.OutlineFootprint(view, px))

    # ── The survey ──────────────────────────────────────────────────────────────

    def survey_height(self, env, target: Target, obstacles=(), tag: str = "[Height]") -> "tuple[Target, HeightLooks | None, str]":
        """
        How tall target is, looked for (module docstring): (target with its
        height and widths, the looks, ""), or (target as given, the looks so
        far or None, why it could not be settled). The arm ends clear above
        where it last looked from.
        """
        s = self.spec
        looks = self._height_start(env, target, obstacles)
        if isinstance(looks, str):
            return target, None, looks
        self._height_looks = looks
        looks.refit()                                       # what the outline from above allows, before any look
        tallest = target.max_height_m or s.default_max_height_m
        for k in looks.known[1:]:                           # the neighbours as tall as they can be, drawn so
            i = int(k.key.split(":", 1)[0])
            ob = obstacles[i]
            if ob.height_m is None or ob.height_m >= tallest - 1e-6:
                k.height_m = self.could_be_mm(env, ob.outlines, k.foot_xy_m, ob.radius_m, tallest, looks) / 1000.0
        everything = self.look_obstacles(env, target, obstacles, looks)
        started = time.monotonic()
        self._say(tag, "Not measured: looking at it from the side to see how tall it is.")
        self._look_here(env, looks, tag, quiet=True)       # free, when the camera already sees it from here
        moved = False
        for _ in range(s.height_looks_max):
            if self._height_settled(looks):
                break
            everything = self.look_obstacles(env, target, obstacles, looks)
            view = self._next_view(env, looks, everything)
            if view is None:
                self._say(tag, "No pose the arm can reach sees it from another side.")
                break
            point, rot, label = view
            if not self._go_look(env, point, rot, everything):
                looks.tried.append(point)
                self._say(tag, f"Could not get to the look {label} — trying another.")
                continue
            moved = True
            if not self._look_here(env, looks, tag, where=label):
                looks.tried.append(point)
        if moved:
            self._clear_up(env, everything)
        fit = looks.fit
        took = time.monotonic() - started
        if fit is None:
            return target, looks, f"no look outlined it ({took:.0f} s)"
        if self._height_settled(looks):
            self._say(tag, f"{fit.height_mm:.0f} mm tall ({fit.profile}; {fit.spread_mm[0]:.0f}-"
                           f"{fit.spread_mm[1]:.0f} fit as well), {fit.footprint.width_mm:.0f} mm across, from "
                           f"{looks.seeing()} looks from the side and {sum(1 for x in looks.silhouettes if x.fixed)} "
                           f"from above ({took:.0f} s).")
            return dataclasses.replace(target, height_m=fit.height_mm / 1000.0, widths_mm=dict(fit.widths_mm),
                                       shape_looks=looks.seeing()), looks, ""
        if self._height_bounded(looks):
            lo, hi = fit.spread_mm
            self._say(tag, f"Between {lo:.0f} and {hi:.0f} mm tall ({fit.height_mm:.0f} fits best), from "
                           f"{looks.seeing()} looks from the side ({took:.0f} s): gripping it as the shortest of "
                           "those would be, the hover clear of the tallest.")
            # One look's worth (shape_looks 1): the plan hovers clear of max_height_m
            # (planning._plan_for, an unsettled shape), and the memory is not told it.
            return dataclasses.replace(target, height_m=lo / 1000.0, widths_mm=fit.widths_up_to(lo),
                                       max_height_m=max(hi / 1000.0, lo / 1000.0), shape_looks=1), looks, ""
        cap = self._height_capped(looks)
        if cap is not None:
            self._say(tag, f"No taller than {cap:.0f} mm ({fit.spread_mm[0]:.0f}-{fit.spread_mm[1]:.0f} fit, from "
                           f"{looks.seeing()} looks from the side; {took:.0f} s): short enough to grip as low as "
                           "the jaws go, the hover clear of that.")
            # Not measured (the memory is not told), the tallest it may be known: the plan grips it
            # as low as an object that tall allows (planning._plan_for).
            return dataclasses.replace(target, height_m=None, widths_mm=None, max_height_m=cap / 1000.0,
                                       shape_looks=None), looks, ""
        why = fit.why if not fit.resolved else (
            f"only {looks.seeing()} look{'s' if looks.seeing() != 1 else ''} from the side "
            f"matched it, {s.height_looks_agree} needed")
        return target, looks, f"{why} ({took:.0f} s)"

    def low_from_above(self, env, target: Target, tag: str = "[Height]") -> "Target | None":
        """
        target as tall as its outline from above alone lets it be (could_be_mm),
        where gripped as low as the jaws go (spec.min_grasp_height_m) the
        gripper's body still clears that (spec.free_depth_m): planned so, with
        no look from the side (spec.grip_low_from_above) — the grip is right for
        any height it could have, the worst a close above something shorter
        than the jaws' lowest, a miss. None: the outline allows it taller, or
        gives no bound. 2026-11-18, the SO-101's table: the side looks took
        25-36 s an object, and made the plug 36-51 mm tall; gripped by that
        it was closed on nothing twice and pushed away, where gripped low at
        16:08 it held.
        """
        s = self.spec
        if not target.outlines:
            return None
        tallest = target.max_height_m or s.default_max_height_m
        radius = (target.foot_m or 0.0) / 2.0
        cap_m = self.could_be_mm(env, target.outlines, np.asarray(target.position, float)[:2], radius, tallest,
                                 None) / 1000.0
        if cap_m >= tallest - 1e-6 or cap_m > s.min_grasp_height_m + s.free_depth_m:
            return None
        self._say(tag, f"From above it is no taller than {cap_m * 1000:.0f} mm: gripped as low as the jaws go "
                       f"({s.min_grasp_height_m * 1000:.0f} mm up), which clears that — no looks from the side.")
        return dataclasses.replace(target, max_height_m=cap_m)

    def _remembered_height(self, env, target: Target, obstacles) -> "HeightLooks | None":
        """
        For an object whose height is remembered, what its outline from above
        makes of it, over every height it could have — a memory can be wrong
        by any amount (another object's track): what the look from close above
        on the way down (_approach_look) is fitted with. None when the arm
        cannot look so, or the outline gives no fit.
        """
        looks = self._height_start(env, target, obstacles)
        if isinstance(looks, str) or not looks.silhouettes or target.height_m is None:
            return None
        lo, hi = looks.height_range_mm
        looks.height_range_mm = (lo, max(hi, 1.5 * target.height_m * 1000.0))
        return looks if looks.refit() is not None else None

    def look_obstacles(self, env, target: Target, obstacles, looks: "HeightLooks | None" = None) -> list:
        """
        What a look must keep the gripper off: the object itself, its neighbours
        and the arm's base. Each not measured is as tall as it could be: as
        anything here can be (Target.max_height_m), or less where what has been
        seen of it says so (could_be_mm) — never left out, as a sweep check
        leaves an object with no height.
        """
        from mrcf.policies.parallel_jaw.reach import BASE_LABEL

        s = self.spec
        tallest = target.max_height_m or s.default_max_height_m
        out = []
        for ob in obstacles:
            if ob.label == BASE_LABEL:
                out.append(ob)
                continue
            h = ob.height_m
            if h is None or h >= tallest - 1e-6:          # not measured: the tallest it may be
                h = self.could_be_mm(env, ob.outlines, np.asarray(ob.position, float)[:2], ob.radius_m,
                                     tallest, looks) / 1000.0
            out.append(dataclasses.replace(ob, height_m=h))
        own = target.height_m
        if own is None:
            own = tallest
            if looks is not None and looks.fit is not None:
                own = min(tallest, (looks.fit.bound_mm(BOUND_DROP) + BOUND_MARGIN_MM) / 1000.0)
        out.append(Obstacle(np.asarray(target.position[:2], float),
                            (target.foot_m or 2 * s.unmeasured_obstacle_radius_m) / 2, "it", own))
        return self._with_base(env, out)

    def could_be_mm(self, env, outlines, xy, radius_m, tallest_m: float, looks: "HeightLooks | None") -> float:
        """
        How tall an object not measured can be, mm: what its outline from above
        allows (ShapeFit.bound_mm at BOUND_DROP, and BOUND_MARGIN_MM more), at
        most tallest_m; tallest_m when nothing outlined it. Seen from right
        under the camera an outline allows anything, away from it a compact one
        does not: a 25 mm ball near the SO-101's base is bounded at 48-60 mm.
        """
        from mrcf.perception import shape_fit

        if not outlines:
            return tallest_m * 1000.0
        shift = np.zeros(2) if looks is None else looks.shift
        sils, footprint = [], None
        for camera, outline in outlines.items():
            got = self._fixed_silhouette(env, camera, outline, shift)
            if got is not None:
                sils.append(got[0])
                footprint = footprint or got[1]
        if footprint is None:
            return tallest_m * 1000.0
        fit = shape_fit.fit_shape(sils, footprint, (4.0, tallest_m * 1000.0))
        if fit is None or fit.score < shape_fit.MIN_FIT_IOU:
            return tallest_m * 1000.0
        return float(min(tallest_m * 1000.0, fit.bound_mm(BOUND_DROP) + BOUND_MARGIN_MM))

    def _height_settled(self, looks: HeightLooks) -> bool:
        f = looks.fit
        return f is not None and f.resolved and looks.seeing() >= self.spec.height_looks_agree

    def _height_bounded(self, looks: HeightLooks) -> bool:
        """Not resolved, but every solid that fits about as well is within BOUNDED_MM of the others, each seen from the side."""
        from mrcf.perception.shape_fit import MIN_FIT_IOU, SEES_HEIGHT_PX_PER_MM

        f = looks.fit
        if f is None or f.score < MIN_FIT_IOU or looks.seeing() < self.spec.height_looks_agree:
            return False
        if max((pp for pp, iou in zip(f.px_per_mm, f.ious) if iou >= SEEING_IOU), default=0.0) < SEES_HEIGHT_PX_PER_MM:
            return False
        lo, hi = f.spread_mm
        return hi - lo <= max(BOUNDED_MM, BOUNDED_SHARE * lo)

    def _height_capped(self, looks: HeightLooks) -> "float | None":
        """
        Not resolved nor bounded, but how tall it can be at most is known, mm
        (the spread's top or the fit's bound, whichever is more, and
        BOUND_MARGIN_MM) — and gripped as high as a top that tall needs to
        clear the gripper's body (free_depth_m above the grasp; no lower than
        spec.min_grasp_height_m), the jaws are still on it at the upper end of
        what fits (spec.grip_over_top_m under its top): safe for every height
        up to the cap, the worst a close above a shorter one's top, a miss.
        2026-11-17, a 3-pin plug fitting 24-59 mm, bound 87: refused while the
        grip had to be on the shortest that fits; gripped at 32 mm it clears
        the tallest and holds any from 42 mm up. None otherwise (spec.grip_low_when_capped
        off). 2026-11-17, the SO-101's wooden figures (22 mm across, ~35 mm
        tall): seen from 3-12 cm off, 10-14 cm up, anything from 5 to 31 mm
        fitted, and every pick of them was refused.
        """
        from mrcf.perception.shape_fit import MIN_FIT_IOU, SEES_HEIGHT_PX_PER_MM

        s = self.spec
        f = looks.fit
        if not s.grip_low_when_capped or f is None or f.score < MIN_FIT_IOU \
                or looks.seeing() < s.height_looks_agree:
            return None
        if max((pp for pp, iou in zip(f.px_per_mm, f.ious) if iou >= SEEING_IOU), default=0.0) < SEES_HEIGHT_PX_PER_MM:
            return None
        cap = max(float(f.spread_mm[1]), float(f.bound_mm(BOUND_DROP))) + BOUND_MARGIN_MM
        # Gripped high enough that a top this tall clears the gripper's body (as planning._plan_for
        # grips an object not measured) — safe whatever its height. Gone ahead with while that grip is
        # on it at the upper end of what fits: on a shorter one the jaws close above it, a miss.
        grip = max(s.min_grasp_height_m * 1000.0, cap - s.free_depth_m * 1000.0)
        on_tall = max(s.min_grasp_height_m * 1000.0, float(f.spread_mm[1]) - s.grip_over_top_m * 1000.0)
        if grip > on_tall + 1e-6:
            self._say("[Height]", f"No taller than {cap:.0f} mm, but the gripper's body clears that only gripped "
                                  f"{grip:.0f} mm up, above where even the tallest that fits "
                                  f"({f.spread_mm[1]:.0f} mm) could be held.")
            return None
        if grip > max(s.min_grasp_height_m * 1000.0, float(f.spread_mm[0]) - s.grip_over_top_m * 1000.0) + 1e-6:
            self._say("[Height]", f"Gripping it {grip:.0f} mm up so a top {cap:.0f} mm tall clears the gripper's body: "
                                  f"were it as short as {f.spread_mm[0]:.0f} mm the jaws would close above it — a miss.")
        return cap

    def height_measured(self, looks: HeightLooks) -> "Measured | None":
        """The fit as a Measured, for the pick's Outcome (and so the object memory)."""
        f = looks.fit
        if f is None or not f.resolved:
            return None
        return Measured(height_mm=round(f.height_mm, 1), widths_mm=dict(f.widths_mm), foot_mm=round(f.foot_mm, 1),
                        slant_deg=0.0, looks=max(1, looks.seeing()))

    def _look_here(self, env, looks: HeightLooks, tag: str, quiet: bool = False, where: str = "here") -> bool:
        """The wrist camera's outline of the object from where it is, added and the shape fitted again. Whether it had one."""
        s = self.spec
        time.sleep(0.3)                      # let the arm stop before judging the outline
        frame = self._wrist_frame(env, tag)
        model = env.camera_model(s.camera, s.render_w, s.render_h)
        if frame is None or model is None:
            return False
        got = self._wrist_silhouette(env, frame, model, looks)
        n = len(looks.models) + 1
        if isinstance(got, str):
            if not quiet:
                self._say(tag, f"From {where}: {got}.")
                _keep_look(frame, f"height_{n}_none", None, looks, [f"from {where}", got])
            return False
        looks.silhouettes.append(got)
        looks.models.append(model)
        cam = np.asarray(model["T_world_cam"], float)[:3, 3]
        d = cam[:2] - (looks.target_xy + looks.shift)
        looks.looked_from.append(d / (np.linalg.norm(d) or 1.0))
        self._views.append((f"height_{len(looks.models)}", frame))
        fit = looks.refit()
        self._drop_outliers(looks, tag)
        fit = looks.fit
        if fit is not None:
            self._say(tag, f"From {where}: {fit.height_mm:.0f} mm tall so far "
                           f"({fit.spread_mm[0]:.0f}-{fit.spread_mm[1]:.0f} fit as well; this view moves "
                           f"{fit.px_per_mm[-1]:.2f} px per mm of height, matched {fit.ious[-1]:.2f}).")
            flight_recorder.event("policy", "height_look", look=n, where=where, **fit.as_dict())
        _keep_look(frame, f"height_{n}", got, looks,
                   [f"from {where}"] + ([] if fit is None else [
                       f"{fit.height_mm:.0f} mm {fit.profile}, {fit.spread_mm[0]:.0f}-{fit.spread_mm[1]:.0f} fit",
                       f"this view {fit.px_per_mm[-1]:.2f} px/mm, matched {fit.ious[-1]:.2f}"]))
        return True

    def _drop_outliers(self, looks: HeightLooks, tag: str) -> None:
        """Set aside a wrist look whose outline the others' fit does not match (a neighbour, a shadow), once there are three."""
        f = looks.fit
        wrist = [i for i, x in enumerate(looks.silhouettes) if not x.fixed]
        if f is None or len(wrist) < 3:
            return
        worst = min(wrist, key=lambda i: f.ious[i])
        if f.ious[worst] >= OUTLIER_IOU:
            return
        self._say(tag, f"One look's outline matched the rest poorly ({f.ious[worst]:.2f}) — set aside.")
        del looks.silhouettes[worst]
        k = wrist.index(worst)
        del looks.models[k]
        del looks.looked_from[k]
        looks.refit()

    # ── The object's outline in a wrist frame ───────────────────────────────────

    def _wrist_silhouette(self, env, frame, model, looks: HeightLooks):
        """
        shape_fit.Silhouette of the object in one wrist-camera frame, or why
        there is none. The segmenter is asked about points just above its foot
        (inside any object over ~15 mm tall), only where it should show; an
        outline that takes in another object's foot, or is mostly the jaws,
        is not its. Where the jaws or something standing in front could hide
        it counts neither way.
        """

        from mrcf.perception import segment, shape_fit
        from mrcf.perception.expected_view import ExpectedView
        from mrcf.perception.plane_geometry import PlaneView

        view = PlaneView.from_camera_model(model)
        h, w = frame.shape[:2]
        known = [dataclasses.replace(k, foot_xy_m=np.asarray(k.foot_xy_m, float)[:2] + looks.shift) for k in looks.known]
        if looks.fit is not None:
            # As tall as the fit allows, not as the tallest anything could be: a column drawn
            # 150 mm tall over a 25 mm ball reaches past a camera looking from close above.
            known[0] = dataclasses.replace(known[0], height_m=looks.fit.spread_mm[1] / 1000.0)
        ev = ExpectedView(view, known, frame.shape)
        if not ev.drawn("target"):
            return "it is out of the camera's view"
        jaws = self._jaw_zone(env, view, model, (h, w))
        centre = (np.asarray(looks.fit.centre_xy_m if looks.fit is not None else looks.target_xy + looks.shift)
                  * 1000.0)
        rise = [4.0, 8.0, 12.0]
        if looks.fit is not None and looks.fit.height_mm > 16.0:
            rise.append(0.5 * looks.fit.height_mm)
        visible = ev.visible("target")
        prompts = []
        for u, v in view.project(np.array([view.lifted(centre, z) for z in rise])):
            if 2 <= u < w - 2 and 2 <= v < h - 2 and visible[int(v), int(u)] and (jaws is None or not jaws[int(v), int(u)]):
                prompts.append((float(u), float(v)))
        if not prompts:
            return "it is hidden there (behind the jaws or something in front)"
        others = []
        for k in known[1:]:
            p = view.project(view.lifted((np.asarray(k.foot_xy_m, float) - view.origin_m) * 1000.0, 4.0)[None])[0]
            if np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h:
                others.append((int(p[1]), int(p[0])))
        found: list = []
        for masks, scores in segment.segment(frame, prompts):
            for m, sc in zip(masks, scores):
                m = np.asarray(m, bool)
                area = int(m.sum())
                if float(sc) < 0.5 or area < 30 or area > 0.25 * h * w:
                    continue
                if any(m[r, c] for r, c in others):
                    continue                          # takes in a neighbour's foot: not its alone
                if jaws is not None and int((m & ~jaws).sum()) < max(30, 0.2 * area):
                    continue                          # the jaws themselves (what of it is the jaws counts for nothing)
                if any((m & f).sum() > 0.9 * max(area, int(f.sum())) for _, f in found):
                    continue
                found.append((float(sc), m))
        if not found:
            return "nothing of its own was outlined there"
        found.sort(key=lambda x: -x[0])
        hidden = np.zeros((h, w), bool) if jaws is None else jaws.copy()
        mine = ev.keys.index("target")
        if "target" in ev.masks:
            hidden |= ev.masks["target"] & (ev.owner >= 0) & (ev.owner != mine)
        return shape_fit.Silhouette(view, [m for _, m in found[:4]], ~hidden, name=f"wrist{len(looks.models) + 1}")

    def _jaw_zone(self, env, view, model, shape) -> "np.ndarray | None":
        """
        Where the jaws and the gripper's body are in the wrist frame (model's), a
        margin round them; None when it cannot say. As the camera has learned to
        see them (jaw_view, perception/jaw_view.py) where it has; else drawn from
        the camera's model, which puts them ~1 cm off on the SO-101.
        """
        import cv2

        tcp = np.asarray(env.get_ee_pos(), dtype=float)
        gap = float(env.get_gripper_width())
        learned = self.jaw_view()
        if learned is not None:
            seen = learned.at(gap, shape[:2]).astype(np.uint8)
            return cv2.dilate(seen, np.ones((21, 21), np.uint8)).astype(bool)
        if not isinstance(env, GripperRotation):
            return None
        rot = np.asarray(env.get_ee_rotation(), dtype=float)
        h, w = shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for hull in self._jaws_in(env, view, model, tcp, rot, gap, (h, w)):
            cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
        mask = cv2.dilate(mask, np.ones((21, 21), np.uint8))
        return mask.astype(bool)

    def _jaws_in(self, env, view, model, tcp, rot, gap: float, shape) -> list:
        """The jaws' and body's outlines (convex hulls, px) as a camera (view, model) sees them: grasp point at tcp, gripper at rot, open gap."""
        import cv2

        s = self.spec
        h, w = shape[:2]
        down = rot[:, 2] / (np.linalg.norm(rot[:, 2]) or 1.0)               # along the approach, to the jaws
        axis = np.r_[self.jaw_axis(env, tcp[:2]), 0.0]
        axis = axis - (axis @ down) * down
        axis /= np.linalg.norm(axis) or 1.0
        across = np.cross(down, axis)
        boxes = []
        for side in (-1.0, 1.0):
            c0 = tcp + side * (gap / 2 + s.finger_thickness_m / 2) * axis
            boxes.append([c0 + a * axis + b * across + u * down
                          for a in (-s.finger_thickness_m / 2, s.finger_thickness_m / 2)
                          for b in (-s.finger_half_width_m, s.finger_half_width_m)
                          for u in (s.tip_below_grasp_m, -s.free_depth_m)])
        half = gap / 2 + s.finger_thickness_m
        boxes.append([tcp + a * axis + b * across + u * down for a in (-half, half) for b in (-0.02, 0.02)
                      for u in (-s.free_depth_m, -s.free_depth_m - 0.04)])
        to_arm = model.get("to_arm")
        inv = None if to_arm is None else np.linalg.inv(np.asarray(to_arm, float))
        plane_z = float(model.get("plane_z_m", 0.0))
        hulls = []
        for box in boxes:
            pts = np.asarray(box, float)
            if inv is not None:
                pts = (inv @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
            plane = (pts - np.array([0.0, 0.0, plane_z])) * 1000.0
            cam = plane @ view.R.T + view.t
            if (cam[:, 2] <= 1.0).any():
                continue
            px = view.project(plane)
            if not np.isfinite(px).all() or np.abs(px).max() > 20 * max(w, h):
                continue
            hulls.append(cv2.convexHull(px.astype(np.float32)).reshape(-1, 2))
        return hulls

    # ── Where to look from ──────────────────────────────────────────────────────

    def _leaned(self, env, at, lean_deg: float, flip: bool) -> np.ndarray:
        """Straight down at `at` (down_rotation), leant lean_deg in the arm's own vertical plane: toward the base, positive."""
        down = np.asarray(env.down_rotation(np.asarray(at, float), flip=flip), float)
        r = np.asarray(at, float)[:2] - self._base_xy(env)
        r /= np.linalg.norm(r) or 1.0
        a = np.array([-r[1], r[0], 0.0])
        t = np.radians(lean_deg)
        k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
        return (np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)) @ down

    def _view_score(self, env, model, looks: HeightLooks, at, rot, gap: float) -> float:
        """How well a camera (model) would see the object's height from a look pose; 0: it would not."""
        from mrcf.perception.expected_view import ExpectedView
        from mrcf.perception.plane_geometry import PlaneView

        if model is None:
            return 0.0
        s = self.spec
        w, h = s.render_w, s.render_h
        view = PlaneView.from_camera_model(model)
        centre = looks.target_xy * 1000.0                 # planned in the arm's frame: a few mm do not matter here
        top = 30.0 if looks.fit is None else float(min(looks.fit.spread_mm[1], 60.0))
        pts = np.array([view.lifted(centre, 0.0), view.lifted(centre, top)])
        if ((pts @ view.R.T + view.t)[:, 2] <= 1.0).any():
            return 0.0
        px = view.project(pts)
        mx, my = 0.08 * w, 0.08 * h
        if not (np.isfinite(px).all() and (px[:, 0] >= mx).all() and (px[:, 0] <= w - mx).all()
                and (px[:, 1] >= my).all() and (px[:, 1] <= h - my).all()):
            return 0.0
        rate = view.px_per_mm_of_height(centre)
        if rate < SEEING_PX_PER_MM:
            return 0.0
        import cv2

        learned = self.jaw_view()
        if learned is not None:
            near = cv2.dilate(learned.at(gap, (h, w)).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
            if any(near[int(v), int(u)] for u, v in px):
                return 0.0                 # behind the jaws as the camera sees them, or within 12 px of them
        else:
            for hull in self._jaws_in(env, view, model, np.asarray(at, float), rot, gap, (h, w)):
                for u, v in px:
                    if cv2.pointPolygonTest(hull.reshape(-1, 1, 2), (float(u), float(v)), True) > -12.0:
                        return 0.0             # behind the jaws, or within 12 px of them
        cam = np.asarray(model["T_world_cam"], float)[:3, 3]
        if float(np.linalg.norm(cam[:2] - looks.target_xy)) < 0.03 and cam[2] - s.table_z < 0.12:
            return 0.0                                     # right over it and close: it sees the top only
        ev = ExpectedView(view, looks.known, (h, w))
        seen = ev.foot_visible("target")
        if seen < 0.6:
            return 0.0
        d = cam[:2] - looks.target_xy
        d = d / (np.linalg.norm(d) or 1.0)
        fresh = 0.5 if any(float(d @ prev) > SAME_SIDE_COS for prev in looks.looked_from) else 1.0
        return float(min(rate, 2.0) * seen * fresh)

    def _next_view(self, env, looks: HeightLooks, everything) -> "tuple | None":
        """
        The best look pose not yet tried: (grasp point, rotation, words for it),
        or None. Scored first by what the camera would see there (_view_score),
        then the best few checked for reach (the orientation the arm can hold
        must lean as planned) and for the gripper's room (_look_clear).
        """
        s = self.spec
        xy = looks.target_xy
        base = self._base_xy(env)
        toward = base - xy
        toward /= np.linalg.norm(toward) or 1.0
        gap = float(env.get_gripper_width())
        ranked = []
        for dist in LOOK_DISTANCES_M:
            for side in LOOK_SIDES_DEG:
                a = np.radians(side)
                way = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ toward
                for z in LOOK_HEIGHTS_M:
                    p = np.r_[xy + dist * way, s.table_z + z]
                    if any(float(np.linalg.norm(p - np.asarray(t, float))) < 0.01 for t in looks.tried):
                        continue
                    if not self._look_clear(env, p, None, everything, gap):
                        continue
                    for lean in LOOK_LEANS_DEG:
                        for flip in (False, True):
                            rot = self._leaned(env, p, lean, flip)
                            if not self._look_clear(env, p, rot, everything, gap):
                                continue           # the gripper's line or its camera over something
                            sc = self._view_score(env, env.camera_model_at(s.camera, s.render_w, s.render_h, p, rot,
                                                                           gap_m=gap), looks, p, rot, gap)
                            if sc > 0.0:
                                ranked.append((sc, p, rot, lean, flip, dist, side, z))
        ranked.sort(key=lambda r: -r[0])
        # Each position once, by the best view planned there; then the arm asked what it
        # can hold there near that lean (either sign, either way round: what those mean is
        # the backend's), and the view from what it can hold is what counts.
        positions: list = []
        seen: set = set()
        for sc, p, _rot, lean, flip, dist, side, z in ranked:
            where_at = tuple(np.asarray(p, dtype=float).round(4).tolist())
            if where_at not in seen:
                seen.add(where_at)
                positions.append((sc, p, lean, flip, dist, side, z))
        best = None
        until = time.monotonic() + TRY_BUDGET_S
        for sc, p, lean, flip, dist, side, z in positions:
            if time.monotonic() > until:
                break
            here = None
            for prefer in (lean, -lean):
                for f in (flip, not flip):
                    ask = (tuple(np.asarray(p, dtype=float).round(4).tolist()), round(float(prefer), 1), bool(f),
                           round(gap, 3))
                    if ask not in looks.reach:
                        got = env.approach_orientation_for([p], gap_m=gap, tols=[s.travel_tol_m], flip=f,
                                                           prefer_tilt_deg=prefer)
                        looks.reach[ask] = None if got is None else np.asarray(got[0], float)
                    held = looks.reach[ask]
                    if held is None:
                        continue
                    if not self._look_clear(env, p, held, everything, gap):
                        continue
                    real = self._view_score(env, env.camera_model_at(s.camera, s.render_w, s.render_h, p, held,
                                                                     gap_m=gap), looks, p, held, gap)
                    if real > 0.0 and (here is None or real > here[0]):
                        here = (real, held)
            if here is None:
                continue
            where = {0.0: "the base's side", 180.0: "the far side"}.get(side, f"{abs(side):.0f} deg round "
                                                                         f"{'left' if side > 0 else 'right'}")
            found = (here[0], p, here[1], f"{dist * 100:.0f} cm off on {where}, {z * 100:.0f} cm up")
            if best is None or found[0] > best[0]:
                best = found
            if here[0] >= min(0.7 * sc, GOOD_ENOUGH):
                break                                  # good enough: no need to look further
        return None if best is None else best[1:]

    def _look_clear(self, env, at, rot, everything, gap: float) -> bool:
        """
        Whether the gripper at `at` (rotation rot; None: whatever it holds) has
        room among everything: its jaws spec.height_view_clearance_m clear of
        each object's column unless their tips are above its top (each as tall
        as it could be, where not measured), its line back to the wrist over
        their tops (sweep_clearance), the camera off them (body_clearance), and
        the arm's base kept out of.
        """
        from mrcf.policies.parallel_jaw.reach import BASE_LABEL

        s = self.spec
        tips = float(at[2]) - s.tip_below_grasp_m
        if tips < s.table_z + 0.01:
            return False
        scale = float(env.frame_scale)
        for ob in everything:
            r = ob.radius_m if ob.radius_m is not None else s.unmeasured_obstacle_radius_m
            d = float(np.linalg.norm((np.asarray(at, float)[:2] - np.asarray(ob.position, float)[:2]) * scale)) - r
            if ob.label == BASE_LABEL:
                if d < s.height_view_clearance_m:
                    return False
                continue
            top = s.table_z + (ob.height_m if ob.height_m is not None else s.default_max_height_m)
            if d < s.height_view_clearance_m and tips < top + s.top_clearance_m:
                return False
        if rot is not None:
            swept = self.sweep_clearance(env, rot, at, everything, open_m=gap)
            if swept is not None and swept[0] < s.sweep_margin_m:
                return False
            # The camera's body sits above the jaw tips: what their tips clear, it clears.
            # body_clearance itself goes by feet alone, for a hover low among chess pieces.
            higher = [ob for ob in everything
                      if ob.label == BASE_LABEL or ob.height_m is None
                      or tips < s.table_z + float(ob.height_m) + s.top_clearance_m]
            body = self.body_clearance(env, rot, at, higher)
            if body is not None and body[0] < s.camera_clearance_m:
                return False
        return True

    def _transit_z(self, env, here, to, everything) -> float:
        """
        A height to cross from here to `to` at: the jaw tips over everything
        standing near the way across — no higher than that and `to` need. Not
        the height the arm happens to be at: after a scan it is folded up out
        of the camera's view, 41 cm up, which no look's place reaches with the
        gripper held for the look (2026-11-17: every look strained for it, and
        one swung the gripper into the table).
        """
        s = self.spec
        a, b = np.asarray(here, float)[:2], np.asarray(to, float)[:2]
        ab = b - a
        n = float(ab @ ab)
        z = float(to[2])
        for ob in everything:
            p = np.asarray(ob.position, float)[:2]
            t = 0.0 if n < 1e-9 else float(np.clip((p - a) @ ab / n, 0.0, 1.0))
            r = ob.radius_m if ob.radius_m is not None else s.unmeasured_obstacle_radius_m
            if float(np.linalg.norm(a + t * ab - p)) * float(env.frame_scale) - r < s.height_view_clearance_m + 0.02:
                top = s.table_z + (ob.height_m if ob.height_m is not None else s.default_max_height_m)
                z = max(z, top + s.tip_below_grasp_m + s.top_clearance_m + 0.01)
        return z

    def _go_look(self, env, at, rot, everything) -> bool:
        """
        To a look pose: up to a height that clears what stands on the way,
        across, and down to it. Whether it got there. A step of the way that
        does not get there ends it: going on from wherever the arm stopped —
        straining at a height out of its reach — swung the gripper into the
        table (2026-11-17, a transit 40 cm up).
        """
        s = self.spec
        here = np.asarray(env.get_ee_pos(), float)
        transit = self._transit_z(env, here, at, everything)
        steps = []
        if here[2] < transit - 0.005:
            steps.append((np.r_[here[:2], transit], None))
        if transit > float(at[2]) + 0.005:
            steps.append((np.r_[np.asarray(at, float)[:2], transit], rot))
        steps.append((np.asarray(at, float), rot))
        for goal, held in steps:
            env.move_ee_to(goal, tol=s.travel_tol_m, orientation=held)
            if not self._near(env, goal, max(s.approach_tol_m, 2 * s.travel_tol_m)):
                return False
        return True

    def _clear_up(self, env, everything) -> None:
        """Straight up from the last look, to where the jaw tips clear everything near it."""
        here = np.asarray(env.get_ee_pos(), float)
        z = self._transit_z(env, here, here, everything)
        if here[2] < z - 0.005:
            env.move_ee_to(np.r_[here[:2], z], tol=self.spec.travel_tol_m)


def _keep_look(frame, stem: str, sil, looks: HeightLooks, words: list) -> None:
    """
    A look's frame kept with the run (flight_recorder.keep_picture), drawn on:
    where the object could not be seen darkened, its outlines (the one the fit
    matches best green, the others yellow), the fitted solid's outline red, and
    words. A look that found nothing is kept too, with why: 2026-11-16 a survey
    that could not settle a white plug's height left nothing to see why by.
    """
    try:
        import cv2

        from mrcf.perception import shape_fit

        img = np.ascontiguousarray(frame).copy()
        best = None
        fit = looks.fit
        if sil is not None:
            if sil.valid is not None:
                img[~sil.valid] = (img[~sil.valid] * 0.45).astype(np.uint8)
            cast = None
            if fit is not None:
                pts = shape_fit.solid_points(sil.view, fit.footprint, fit.height_mm, fit.profile)   # as fitted: moved, scaled
                cast = None if pts is None else shape_fit.cast(sil.view, pts, img.shape)
            ious = [0.0 if cast is None else float((m & cast).sum()) / max(1, int((m | cast).sum())) for m in sil.masks]
            best = int(np.argmax(ious)) if ious else None
            for i, m in enumerate(sil.masks):
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, contours, -1, (0, 220, 0) if i == best else (230, 200, 0), 2 if i == best else 1)
            if cast is not None:
                contours, _ = cv2.findContours(cast.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, contours, -1, (230, 40, 40), 1)
        for k, line in enumerate(words):
            org = (8, 20 + 18 * k)
            cv2.putText(img, str(line)[:90], org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(line)[:90], org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        flight_recorder.keep_picture(stem, img)
    except Exception as exc:                 # noqa: BLE001 — a picture not kept is not a pick lost
        print(f"  [Height] look not kept: {exc}")


def _turn_deg(a, b) -> float:
    """How far one rotation (3x3) is turned from another, degrees."""
    d = np.asarray(a, float).T @ np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip((np.trace(d) - 1.0) / 2.0, -1.0, 1.0))))


def _angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
