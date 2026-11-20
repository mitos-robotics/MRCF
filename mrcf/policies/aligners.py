"""
mrcf/policies/aligners.py — how a two-finger pick lines the jaws up over the object.

Once over the object, a pick has to put the jaws where it really is: the
scan's estimate is a few millimetres to a few centimetres off, and so is the
arm. Each way of doing that is an Aligner. An Approach names the ones it may
use, in order (Approach.align); the first whose `applies` holds runs, and its
outcome stands — it sets where the jaws close (PickContext.target_xy), or ends
the attempt with Retry. One that may fall back on another does so itself.

    wrist_camera_foot   a located wrist camera finds the object's foot, in the arm's frame
    overhead_first      a fixed camera already has the jaws over the object (first attempt)
    wrist_servo         the wrist camera centres the jaws over it (from its own height)
    fixed_camera        an approach that does not use the wrist servo: the fixed camera, or nothing
    kinematics          no camera to align by: the arm's own model

A new strategy is a subclass registered under its name:

    class Touch(Aligner):
        name = "touch"
        def applies(self, pol, env, ctx): return hasattr(env, "probe")
        def run(self, pol, env, ctx): ctx.target_xy = env.probe(ctx.object_xy)

    register_aligner(Touch())
    TOUCH_FIRST = Approach("touch_first", align=("touch", "kinematics"))

They use the policy's helpers (pol: policies.parallel_jaw.ParallelJaw) and the
pick's state (ctx: its PickContext); they never move on to the descent
themselves.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw, PickContext


class Retry(Exception):
    """The attempt under way is over: why, as a short code (Outcome.attempts' "result"), and what to say."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class GiveUp(Exception):
    """No further attempt can help: what to say. The pick then fails."""


class Refuse(GiveUp):
    """
    Nothing can be done here, and nothing has been touched: the pick ends
    refused (Outcome.refused) for this reason, as one refused before moving.
    """


def unconfirmed_descent(pol: "ParallelJaw", ctx: "PickContext", why: str) -> bool:
    """
    GripperSpec.descend_unconfirmed: with no camera to confirm the jaws over
    the object, come down at the scan's position on the kinematics rather
    than retry a look that will see the same. True when it does (said); False
    leaves the aligner to retry. After a close that may have moved the object,
    the scan's position is stale and it never does.

    2026-11-16, Qa4: the queen on d1, beside the arm's base, was never picked.
    The wrist camera, tilted 25 deg (straight down is out of reach there) and
    3 cm above its crown, saw its body but not its foot. The gripper's tag was
    off the top of the overhead camera's frame. Six identical looks found
    nothing, and the game asked for the move by hand. The scan's position is
    good enough most of the time: over 45 picks the wrist camera's sighting
    moved the aim 2 mm (median), more than 4 mm one time in five.
    """
    if not pol.spec.descend_unconfirmed or ctx.missed:
        return False
    pol._say(ctx.tag, f"{why} — descending at the scan's position anyway, nothing having confirmed the "
                      "jaws over it (descend_unconfirmed).")
    ctx.target_xy = ctx.object_xy.copy()
    return True


class Aligner:
    """One way to line the jaws up over the object. Subclass, name it, register it."""
    name = ""

    def applies(self, pol: "ParallelJaw", env, ctx: "PickContext") -> bool:
        raise NotImplementedError

    def run(self, pol: "ParallelJaw", env, ctx: "PickContext") -> None:
        """Set ctx.target_xy (and what was measured on the way), or raise Retry."""
        raise NotImplementedError


class FootAligner(Aligner):
    """
    The wrist camera knows where it is, so it looks for the object where it
    should be — the scan's position, projected into the frame, with what
    stands around it — and finds its foot, whatever the object's height. A
    fixed "object appears here" pixel can do neither, and can lock onto a tall
    neighbour instead.

    It decides who the object is and where it stands against the scan: there
    the two cameras agree to a few mm (2026-11-03, a full board: 0.1-3.5). The
    jaws then go where the fixed camera put them over the scan's position,
    moved by that much. Not by where the camera's mount and the jaw model say
    the jaws are: on the SO-101 that put the centre between the open jaws
    ~11 mm from where a piece centred there by hand appears (GRIPPER_CAM_GRASP_PX).
    That aim is said beside it, for the record.
    """
    name = "wrist_camera_foot"

    def applies(self, pol, env, ctx) -> bool:
        return pol.camera_located(env)

    def run(self, pol, env, ctx) -> None:
        s = pol.spec
        # Told who stands around it (known_around), so a neighbour's outline
        # is not taken for it: from over b1 on a full board the c1 bishop's was.
        found = pol.find_foot(env, ctx.vision, ctx.tag, pol.known_around(ctx))
        ctx.located = found.measured
        if found.aim_xy is None or found.scan_xy is None:
            if pol._overhead_err_m is not None and not ctx.missed:
                # The fixed camera has put the jaws over it: as good as the
                # pick was before the wrist camera was located.
                pol._say(ctx.tag, f"The wrist camera cannot find its foot ({found.why}) — descending where the "
                                  f"overhead camera put the jaws ({pol._overhead_err_m * 1000:.1f} mm from it).")
                return
            if unconfirmed_descent(pol, ctx, f"The wrist camera cannot find its foot ({found.why})"):
                return
            raise Retry("foot_not_found", f"The wrist camera cannot find its foot ({found.why}) — "
                                          "not descending blind, retrying")
        # Who it is, in the scan's frame: there the two cameras agree to a
        # few mm (2026-11-03, a full board from a pick's hover: 0.6-3.5).
        shift = float(np.linalg.norm(found.scan_xy - np.asarray(ctx.target.position[:2], float)))
        if shift > s.max_aim_shift_m:
            raise Retry("foot_too_far", f"Foot found {shift * 1000:.0f} mm from where the scan put it — "
                                        "too far to be the same object; retrying")
        other = pol._other_object(ctx.target, ctx.located, ctx.obstacles) if ctx.located is not None else None
        if other:
            raise Retry("other_object", f"The object under the camera is {other} — not the one sent for; "
                                        "retrying")
        part = pol._part_seen(ctx.target, ctx.located, ctx.obstacles) if ctx.located is not None else None
        if part:
            pol._say(ctx.tag, f"The wrist camera reads it {part} — aiming by its foot, but not by that height.")
            ctx.located = None
        out = ctx.target_xy - pol._base_xy(env)
        out = out / (np.linalg.norm(out) or 1.0)
        across = np.array([-out[1], out[0]])
        moved = found.scan_xy - np.asarray(ctx.target.position[:2], float)
        # Where the wrist camera puts a foot, less where the scan does: every pick's look is a
        # reading of the shift between the two cameras (tools/calibrate_wrist_on_objects.py
        # --from-runs fits it; 2026-11-18 it had drifted 5 mm unnoticed since the overhead moved).
        from mrcf import flight_recorder
        flight_recorder.event("calibration", "wrist_foot", off_m=[round(float(v), 5) for v in moved],
                              at=[round(float(v), 4) for v in np.asarray(ctx.target.position[:2], float)])
        by_jaws = np.asarray(found.aim_xy, float) - ctx.target_xy
        ctx.jaw_along_m = float(by_jaws @ out)
        record = (f"by its own jaw model it would aim {float(by_jaws @ out) * 1000:+.1f} along the arm, "
                  f"{float(by_jaws @ across) * 1000:+.1f} across")
        if s.aim_at_measured_foot:
            pol._say(ctx.tag, f"Foot found {shift * 1000:.1f} mm from where the scan saw it "
                              f"({float(moved @ out) * 1000:+.1f} along the arm, {float(moved @ across) * 1000:+.1f} "
                              f"across) — the overhead camera's aim moved by that ({record}).")
            ctx.target_xy = ctx.target_xy + moved
        else:
            pol._say(ctx.tag, f"It is there: foot {shift * 1000:.1f} mm from where the scan saw it — "
                              f"aiming by the scan ({record}).")


class OverheadFirstAligner(Aligner):
    """
    The scan saw the object through the fixed camera that has just put the
    jaws over it, and checks them again at the grasp; until a miss may have
    moved it, that is a better aim than the wrist servo's
    (GripperSpec.overhead_first).
    """
    name = "overhead_first"

    def applies(self, pol, env, ctx) -> bool:
        return (ctx.fit.servo and pol.spec.overhead_first and pol._overhead_err_m is not None
                and not ctx.missed)

    def run(self, pol, env, ctx) -> None:
        pol._say(ctx.tag, f"Descending where the overhead camera put the jaws "
                          f"({pol._overhead_err_m * 1000:.1f} mm from the object); the wrist camera "
                          "looks again only after a miss.")


class WristServoAligner(Aligner):
    """
    The wrist camera centres the jaws over the object (the backend's
    visual_servo_xy), from the height its target pixel was set at — the
    camera looks at an angle, so from any other height the object shows up
    somewhere else, and "aligned" can leave the jaws centimetres off it.
    An object of unknown height is first centred and measured from the hover
    that clears the tallest it could be.

    What it found is checked before anything descends on it, and when it
    cannot find the object the fixed camera's sighting stands in — unless a
    miss may have moved it since.
    """
    name = "wrist_servo"

    def applies(self, pol, env, ctx) -> bool:
        return ctx.fit.servo and pol._servo_ready(env, ctx.vision)

    def run(self, pol, env, ctx) -> None:
        s = pol.spec
        low = pol._servo_height_for(ctx.hover)
        if low is not None:
            self._down_to_servo_height(pol, env, ctx, low)
        looked = ctx.measured or ctx.far
        height = looked.height_mm / 1000 if looked is not None else ctx.target.height_m
        if pol._visual_servo(env, ctx.vision, ctx.tag, ctx.rot, ctx.room, height_m=height):
            self._check_where_it_aligned(pol, env, ctx)
        elif ctx.missed:
            # Where the fixed camera would put the jaws is where the scan
            # saw the object before a close that may have pushed it.
            raise Retry("lost_after_miss", "The wrist camera cannot find it after the miss — not coming down "
                                           "where it was; looking again")
        elif (shift := self._disputed(pol, env, ctx)) is not None:
            # It could not settle, but it led the jaws well away from where the
            # object was thought to be. 2026-10-31, d3: a pawn the arm had put
            # down stood ~10 mm further out than it was put; the servo led the
            # jaws 13-15 mm towards it, gave up 13-16 px short, and the close
            # at the remembered spot missed.
            ctx.doubted = True
            raise Retry("servo_disagrees", f"The wrist camera led the jaws {shift * 1000:.0f} mm from where it "
                                           "was thought to be and could not settle — finding it again first")
        elif s.descend_on_overhead and pol._overhead_err_m is not None:
            # The wrist camera could not align, but the fixed camera saw the
            # jaws over the object — measured both, as the scan measured the
            # object — so come down there. The wrist camera loses pieces at
            # the board's edge and on a square of their own colour, where
            # the fixed one still sees both.
            pol._say(ctx.tag, f"Descending where the overhead camera put the jaws "
                              f"({pol._overhead_err_m * 1000:.1f} mm from the object).")
        elif not unconfirmed_descent(pol, ctx, "The wrist camera could not centre on it"):
            # Nothing has measured where the jaws are relative to the object,
            # and descending on the kinematics alone can land a jaw on its top
            # or knock it over. Look again from the hover instead.
            raise Retry("servo_failed", "Not descending blind — retrying the approach")

    @staticmethod
    def _disputed(pol, env, ctx) -> "float | None":
        """
        How far the servo has led the jaws from where the object was thought
        to be, when that is more than servo_agree_m and a scan has yet to
        decide between them this pick (Vision.relocate) — else None. Either
        can be the one that is off: 2026-10-31, a pawn put down on d3 stood
        ~10 mm from where it was put and the servo was right; on c2, just
        scanned, the servo settled 11-13 mm outward of it three times and
        every close missed.
        """
        if ctx.vision.relocate is None or ctx.refound:
            return None
        shift = float(np.linalg.norm(env.get_ee_pos()[:2] - ctx.object_xy))
        return shift if shift > pol.spec.servo_agree_m else None

    @staticmethod
    def _down_to_servo_height(pol, env, ctx, low: float) -> None:
        s = pol.spec
        # Only for an object of unknown height: one already measured goes
        # straight to the servo, a look from up here being the coarser
        # (far_measure_agree_mm). Measured by one look only, it may be taller
        # than that look said: as unknown.
        if ctx.measured is None and (not ctx.target.measured or pol.shape_unsettled(ctx.target)):
            pol._visual_servo(env, ctx.vision, ctx.tag, ctx.rot, ctx.room)
            ctx.far = pol.measure_at_hover(env, ctx.vision, ctx.tag, expected_xy=ctx.object_xy,
                                           agree_mm=s.far_measure_agree_mm, take_taller=True)
            if ctx.far is not None:
                pol._replan(env, ctx, ctx.far)
        if ctx.gp.height_m > low - s.tip_below_grasp_m:
            # Taller than the servo's hover clears: align from there anyway —
            # the open jaws straddle its top, the gripper body is well above.
            pol._say(ctx.tag, f"Up to {ctx.gp.height_m * 1000:.0f} mm tall: aligning from the servo's "
                              f"{low * 1000:.0f} mm with its top between the open jaws.")
        pol._set_waypoints(ctx, ctx.object_xy, ctx.gp, pol._lift_for(ctx.approach, ctx.gp, ctx.lift_h),
                           hover_height=low)
        ctx.rot, ctx.tilt, checked = pol._hold(env, pol._pick_reach_of(env, ctx), ctx.tag, ctx.obstacles,
                                               ctx.target.tilt_deg, ctx.approach.lean_margin_deg)
        if checked and ctx.rot is None:
            raise Retry("servo_height", "No single orientation reaches down from the servo's height here — "
                                        "retrying")
        if (not env.move_ee_to(ctx.hover, tol=ctx.hover_tol, orientation=ctx.rot)
                or not pol._near(env, ctx.hover, s.approach_tol_m * 2)):
            dist = float(np.linalg.norm(env.get_ee_pos() - ctx.hover))
            raise Retry("servo_height", f"Could not come down to the servo's height (residual "
                                        f"{dist * 100:.1f} cm) — retrying")

    @staticmethod
    def _check_where_it_aligned(pol, env, ctx) -> None:
        s = pol.spec
        ctx.target_xy = env.get_ee_pos()[:2].copy()
        # A servo that locked onto a neighbour must not be descended on. But
        # the scan and the arm disagree by 20-26 mm near the middle of this
        # board, every time, the same way — the servo exists to correct that.
        # So: not further than max_servo_shift_m, and never nearer another
        # object's estimate than this one's; both are off by the same scan.
        shift = float(np.linalg.norm(ctx.target_xy - ctx.object_xy))
        # Kept, square by square: how far the arm's model and the scan disagree
        # here — what a place, which has no camera to correct it, will be off by.
        out = ctx.object_xy - pol._base_xy(env)
        out = out / (np.linalg.norm(out) or 1.0)
        along = float((ctx.target_xy - ctx.object_xy) @ out)
        across = float((ctx.target_xy - ctx.object_xy) @ np.array([-out[1], out[0]]))
        pol._say(ctx.tag, f"The servo put the jaws {shift * 1000:.0f} mm from the scan's estimate: "
                          f"{along * 1000:+.0f} mm along the arm, {across * 1000:+.0f} across.")
        nearer = [o.label or "another object" for o in ctx.obstacles
                  if float(np.linalg.norm(ctx.target_xy - np.asarray(o.position[:2], float))) < shift]
        if shift > s.max_servo_shift_m or nearer:
            why = f"nearer {nearer[0]}" if nearer else f"more than {s.max_servo_shift_m * 1000:.0f} mm"
            raise Retry("servo_too_far", f"The servo moved the jaws {shift * 1000:.0f} mm from where the "
                                         f"scan put it — {why}, so probably not the same object; not "
                                         "descending, retrying")
        if WristServoAligner._disputed(pol, env, ctx) is not None:
            ctx.doubted = True
            raise Retry("servo_disagrees", f"The servo and where it was thought to be are {shift * 1000:.0f} mm "
                                           "apart — finding it again to tell which is right")
        if ctx.refound and shift > s.servo_agree_m:
            pol._say(ctx.tag, f"A scan found it there this pick: going by the scan, not the servo's "
                              f"{shift * 1000:.0f} mm.")
            ctx.target_xy = ctx.object_xy.copy()
            return
        if pol._overhead_err_m is not None and shift <= s.servo_agree_m and not ctx.missed:
            # Two estimates a few mm apart: the scan's is measured through the
            # camera that just put the jaws over it (and checks them again at
            # the grasp), so its errors cancel. The servo's can hold a bias of
            # 4-8 mm near the base, outward and across, which the close then
            # cams the object out along. Further apart, the object is not where
            # the scan saw it, and the servo wins.
            pol._say(ctx.tag, f"Within {s.servo_agree_m * 1000:.0f} mm of the scan, which the fixed "
                              "camera measures the jaws against — descending at the scan's estimate.")
            ctx.target_xy = ctx.object_xy.copy()


class FixedCameraAligner(Aligner):
    """
    An approach that does not use the wrist servo here (near the base its
    height is out of reach; far out, its lean is far from the one it was
    calibrated at): the fixed camera aligns, or nothing — and with a camera
    that could have, it does not come down blind.
    """
    name = "fixed_camera"

    def applies(self, pol, env, ctx) -> bool:
        return ctx.fit.servo_skipped or not ctx.approach.servo

    def run(self, pol, env, ctx) -> None:
        if pol._overhead_err_m is not None:
            pol._say(ctx.tag, f"Descending where the overhead camera put the jaws "
                              f"({pol._overhead_err_m * 1000:.1f} mm from the object); {ctx.approach.name} "
                              "does not use the wrist servo here.")
        elif (pol._can_overhead(env) or pol._servo_ready(env, ctx.vision)) and not unconfirmed_descent(
                pol, ctx, f"The overhead camera did not confirm the jaws over it ({ctx.approach.name})"):
            raise Retry("blind", "The overhead camera did not confirm the jaws over the object, and "
                                 f"{ctx.approach.name} does not use the wrist servo here — not descending "
                                 "blind, retrying")
        # A rig with neither camera to align by: on the kinematics, as always.


class KinematicsAligner(Aligner):
    """No camera to align by: descend where the arm's own model puts the object (saying, once, why not the servo)."""
    name = "kinematics"

    def applies(self, pol, env, ctx) -> bool:
        return True

    def run(self, pol, env, ctx) -> None:
        pol._visual_servo(env, ctx.vision, ctx.tag, ctx.rot, ctx.room)     # says once why it cannot


ALIGNERS: dict = {}


def register_aligner(aligner: Aligner) -> Aligner:
    """Make `aligner` available to Approach.align under its name (replacing one of that name)."""
    if not aligner.name:
        raise ValueError("an Aligner needs a name")
    ALIGNERS[aligner.name] = aligner
    return aligner


for _a in (FootAligner(), OverheadFirstAligner(), WristServoAligner(), FixedCameraAligner(), KinematicsAligner()):
    register_aligner(_a)


__all__ = ["Aligner", "ALIGNERS", "register_aligner", "Retry", "GiveUp", "FootAligner", "OverheadFirstAligner",
           "WristServoAligner", "FixedCameraAligner", "KinematicsAligner"]
