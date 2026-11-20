"""
mrcf/policies/parallel_jaw/sighting.py — looking on the way in: measuring the
object with the wrist camera at the hover, the wrist servo, the fixed camera's
sighting of the gripper, and whether what is measured is the object that was
asked for.

The runs behind its values and fixes, with their dates: mrcf/policies/HISTORY.md.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from mrcf.capabilities import FrameAge, GripSensing, OverheadSighting, Settling, WristServo
from mrcf.policies.approaches import (  # noqa: F401 — re-exported
    DEFAULT_ALIGN,
    NEAR_BASE,
    REACH_OUT,
    TOP_DOWN,
    Approach,
    ApproachFit,
    _Choices,
)
from mrcf.policies.base import Measured, Target, Vision
from mrcf.policies.parallel_jaw.reach import ReachMixin
from mrcf.policies.parallel_jaw.spec import GraspPlan

if TYPE_CHECKING:
    from mrcf.robot_env import RobotEnv


def _in_camera_frame(model: dict, xy) -> np.ndarray:
    """A point on the surface, from the frame the arm moves by into the one the camera is placed in (model["to_arm"])."""
    T = model.get("to_arm")
    xy = np.asarray(xy, float)[:2]
    if T is None:
        return xy
    z = float(model.get("plane_z_m", 0.0))
    return (np.linalg.inv(T) @ np.r_[xy, z, 1.0])[:2]


def _in_arm_frame(model: dict, xy) -> np.ndarray:
    """A point on the surface the camera found, in the frame the arm moves by (model["to_arm"])."""
    T = model.get("to_arm")
    xy = np.asarray(xy, float)[:2]
    if T is None:
        return xy
    z = float(model.get("plane_z_m", 0.0))
    return (T @ np.r_[xy, z, 1.0])[:2]


def _scan_to_camera(model: dict, xy) -> np.ndarray:
    """A position the fixed camera's scan measured, in the frame the camera is placed in (model["scan_to_camera"])."""
    shift = model.get("scan_to_camera")
    return np.asarray(xy, float)[:2] + (0.0 if shift is None else np.asarray(shift, float)[:2])


def _camera_to_scan(model: dict, xy) -> np.ndarray:
    shift = model.get("scan_to_camera")
    return np.asarray(xy, float)[:2] - (0.0 if shift is None else np.asarray(shift, float)[:2])


@dataclasses.dataclass
class FootFound:
    """What find_foot found: where to put the jaws, where the object stands, and its measure if the view had it."""
    aim_xy: "np.ndarray | None"           # its foot, in the frame the arm moves by (what to aim at)
    scan_xy: "np.ndarray | None"          # its foot, in the frame the scan measured it in (who it is)
    measured: "Measured | None" = None    # when the view also resolved its height
    why: str = ""                         # why not found


def _takes_known(measure) -> bool:
    """Whether a Vision.measure takes the objects around (known=, target=, need_height=)."""
    import inspect
    try:
        params = inspect.signature(measure).parameters
    except (TypeError, ValueError):
        return False
    return "known" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


class SightingMixin(ReachMixin):
    """ParallelJaw's looks during a pick (see the package docstring)."""

    # A wrist camera whose newest frame is older than this has stopped sending.
    # 2026-11-14 it dropped off USB mid-game and would not open again, and every
    # pick raised "has not captured any frames yet"; the picks now go by the
    # overhead camera alone, as before the wrist camera was located.
    WRIST_STALE_S = 2.0

    def wrist_sending(self, env) -> bool:
        """
        False when the backend says the wrist camera has sent nothing for
        WRIST_STALE_S (frame_age_s), said once until it sends again; True
        where the backend cannot say. No wrist camera: False.
        """
        s = self.spec
        if s.camera is None:
            return False
        if not isinstance(env, FrameAge):
            return True
        try:
            age = float(env.frame_age_s(s.camera))
        except Exception:                    # noqa: BLE001 — a camera that cannot say how old its frame is sends none
            age = float("inf")
        sending = age <= self.WRIST_STALE_S
        if not sending and not self.__dict__.get("_wrist_silence_said"):
            since = "ever" if age == float("inf") else f"{age:.0f} s"
            print(f"  [Pick] The wrist camera has sent nothing for {since} — the jaws aim by the overhead "
                  "camera alone until it does.")
        self.__dict__["_wrist_silence_said"] = not sending
        return sending

    def _wrist_frame(self, env, tag: str):
        """The wrist camera's newest frame; None (said) when it has none to give."""
        s = self.spec
        try:
            return env.render_image(s.camera, width=s.render_w, height=s.render_h)
        except (RuntimeError, TimeoutError, ConnectionError) as exc:
            self._say(tag, f"The wrist camera gave no frame ({exc}) — going by the overhead camera.")
            return None

    def holding(self, env) -> bool:
        """
        True when the gripper is holding something. Sensed from the servo's load
        where the backend can (a held knight and an empty close differ by under
        a millimetre of opening, but several-fold in load); else from the opening.
        Where the gripper reads empty, and the jaws have not opened since they
        shut on something (_note_grip), the wrist camera is asked too
        (spec.held_by_camera): what it saw between the jaws then, still there.
        """
        if isinstance(env, GripSensing):
            felt = env.gripper_loaded()
        else:
            w = env.get_gripper_width()
            felt = self.spec.min_hold_width_m < w < self.spec.miss_width_m
        if felt or self._grip_view is None:
            return felt
        if self._closed_on_nothing(env):
            return False
        if self._still_gripped(env):
            return True
        held = self._seen_holding(env) or self._asked_holding(env)
        if held and self._grip_view is not None and self.spec.held_while_still:
            self._grip_view["held_gap"] = float(env.get_gripper_width())
        return held

    def _closed_on_nothing(self, env) -> bool:
        """
        spec.held_closed_m: the jaws have closed that much further since they
        shut on it — nothing is between them, whatever a picture seems to
        show; said once, and the grip forgotten. 2026-11-20 09:25, the SO-101:
        the plug slipped out as it was lifted, the jaws went on from 42.8 mm to
        28.8, and the wrist camera's 0.5% of the picture "moving with them" was
        taken for it carried — it was "put into" the tub, still on the table.
        """
        s, grip = self.spec, self._grip_view
        if s.held_closed_m is None or grip is None:
            return False
        gap = float(env.get_gripper_width())
        if gap > float(grip["gap"]) - float(s.held_closed_m):
            return False
        self._say("[Held]", f"The jaws closed from {float(grip['gap']) * 1000:.1f} mm to {gap * 1000:.1f} after "
                            "shutting on it: nothing is between them.")
        self._grip_view = None
        return True

    def _held_at(self, env, gap_m: float) -> None:
        """A grip felt under load, the jaws stopped gap_m apart: held while they stay there (_still_gripped)."""
        if self._grip_view is None:
            self._grip_view = {"frame": None, "model": None, "gap": float(gap_m), "said": False}
        self._grip_view["held_gap"] = float(gap_m)

    def _still_gripped(self, env) -> bool:
        """
        spec.held_while_still: seen carried before (the camera's check, or the
        model's), and the jaws have not moved since — had it slipped out they
        would have closed on. 2026-11-19 11:09, the SO-101: the scoop held by
        its wall at 1.1 mm, seen carried as it was lifted; two minutes later
        the arm at home, its camera looking out across the room past the
        scoop, the model said "the jaws are empty", and the place was refused
        with the scoop plainly in them.
        """
        s, grip = self.spec, self._grip_view
        if not s.held_while_still or grip is None or grip.get("held_gap") is None:
            return False
        gap = float(env.get_gripper_width())
        if gap > grip["gap"] + s.held_open_m:
            self._grip_view = None                          # opened since: whatever it held is let go
            return False
        return grip["held_gap"] >= s.rim_min_hold_m and gap >= grip["held_gap"] - s.held_still_m

    # A deployment's question to a vision model, frame -> (carrying, what it saw) | None: asked when the
    # gripper reads empty after shutting on something and the camera's own check does not see it carried.
    ask_held: "Callable | None" = None
    HELD_ASK_S = 30.0                  # an answer stands this long for the same grip

    def _asked_holding(self, env) -> bool:
        """holding()'s last word: the vision model asked, once a grip (HELD_ASK_S), whether it is carried."""
        grip = self._grip_view
        if self.ask_held is None or grip is None or self.spec.camera is None:
            return False
        now = time.monotonic()
        if grip.get("asked") is not None and now - grip["asked"][0] < self.HELD_ASK_S:
            return grip["asked"][1]
        try:
            frame = env.render_image(self.spec.camera, width=self.spec.render_w, height=self.spec.render_h)
            got = self.ask_held(np.asarray(frame)) if frame is not None else None
        except Exception:                                   # noqa: BLE001 — no answer is no
            got = None
        carrying = bool(got and got[0])
        grip["asked"] = (now, carrying)
        if carrying and got is not None:
            self._say("[Held]", f"The gripper reads empty, but the vision model sees it carried ({got[1]}).")
        return carrying

    def _note_grip(self, env) -> None:
        """
        The wrist camera's picture as the jaws shut on something, and the
        opening then: what holding() looks for the gripper still carrying when
        its own reading says empty (spec.held_by_camera). Kept until the jaws
        open wider.
        """
        self._grip_view = None
        s = self.spec
        if not s.held_by_camera or s.camera is None:
            return
        try:
            frame = env.render_image(s.camera, width=s.render_w, height=s.render_h)
            model = env.camera_model(s.camera, s.render_w, s.render_h)
        except Exception:                                   # a camera that gave nothing: the servo decides
            return
        if frame is None or model is None:
            return
        self._grip_view = {"frame": np.asarray(frame), "model": model, "gap": float(env.get_gripper_width()),
                           "said": False}

    def _seen_holding(self, env) -> bool:
        """Whether the wrist camera sees the gripper carrying what it shut on (perception/held_view.py)."""
        from mrcf.perception.held_view import held_view

        s, grip = self.spec, self._grip_view
        assert grip is not None
        gap = float(env.get_gripper_width())
        if gap > grip["gap"] + s.held_open_m:
            self._grip_view = None                          # opened since: whatever it held is let go
            return False
        if grip.get("frame") is None:
            return False                                    # no picture as it shut: nothing to compare with
        jaws = self.jaw_view()
        if jaws is None:
            return False
        try:
            frame = env.render_image(s.camera, width=s.render_w, height=s.render_h)
            model = env.camera_model(s.camera, s.render_w, s.render_h)
        except Exception:
            return False
        if frame is None or model is None:
            return False
        frame = np.asarray(frame)
        seen = held_view(grip["frame"], grip["model"], frame, model, jaws.at(gap, frame.shape[:2]))
        if seen.held and not grip["said"]:
            grip["said"] = True
            self._say("[Held]", f"The gripper reads empty ({gap * 1000:.1f} mm), but the wrist camera sees it carried "
                                f"between the jaws ({seen.carried * 100:.1f}% of the picture moving with them).")
        return bool(seen.held)

    def camera_located(self, env) -> bool:
        """
        True when the wrist camera can locate an object's foot: the backend
        gives its pose and the segmenter can run. Both, because a pick that
        waits for a foot the camera can never find would refuse to descend on
        every attempt.
        """
        s = self.spec
        if s.camera is None:
            return False
        if env.camera_model(s.camera, s.render_w, s.render_h) is None:
            return False
        if not self.wrist_sending(env):
            return False
        try:
            from mrcf.perception import segment
        except ImportError:
            return False
        return segment.available()

    def measure_at_hover(self, env, vision: Vision, tag: str = "", expected_xy=None,
                         agree_mm: "float | None" = None, take_taller: bool = False,
                         known=None) -> "Measured | None":
        """
        The object under the jaws, measured from two wrist-camera frames that
        agree, or None (and why, printed) when it cannot be measured; the
        caller then uses what it had. expected_xy is where the scan put its foot.
        agree_mm: how far apart the two heights may be (spec.measure_agree_mm);
        take_taller: report the taller look rather than their mean — what to
        plan clearance by when the looks are coarse. known: the object, keyed
        "target", and those around it (known_around), for a measure that can
        tell them apart (Vision.measure).
        """
        s = self.spec
        if s.camera is None:
            return None
        if vision.measure is None:
            from mrcf.perception import plane_geometry
            measure = plane_geometry.measure_in_frame
        else:
            measure = vision.measure
        expected = (env.wrist_servo_target_px if isinstance(env, WristServo) else None) or (s.render_w / 2, s.render_h / 2)
        found, frames = [], []
        for _ in range(2):
            time.sleep(0.3)                  # let the arm stop before judging the outline
            frame = self._wrist_frame(env, tag)
            if frame is None:
                return None
            # Read with the frame, the arm still: a wrist camera's pose is the
            # joints' at the moment the frame was taken.
            model = env.camera_model(s.camera, s.render_w, s.render_h)
            frames.append(frame)
            if len(frames) == 1:
                # Said with the pick, not found out from it: a washed-out or dark
                # view is where junk heights and lost outlines come from.
                from mrcf.perception import lighting
                light = lighting.warning(lighting.reading(frame), "the wrist camera's view")
                if light:
                    self._say(tag, f"Light: {light}")
            r = measure(frame, expected_px=expected, model=model, **self._looking_for(model, measure, expected_xy, known))
            if not r["ok"]:
                self._say(tag, f"Could not measure the object ({r['why']}) — using what the scan gave.")
                return None
            if r.get("occluded"):
                # Its outline runs into the gripper, so the top may be hidden.
                self._say(tag, "The gripper hides part of the object — not trusting its height.")
                return None
            found.append(r["measurement"])
        heights = [m.height_mm for m in found]
        if abs(heights[0] - heights[1]) > (s.measure_agree_mm if agree_mm is None else agree_mm):
            self._say(tag, f"Two looks disagree on the height ({heights[0]:.0f} vs "
                           f"{heights[1]:.0f} mm) — using what the scan gave.")
            return None
        m = found[int(np.argmax(heights))] if take_taller else found[0]
        out = Measured(height_mm=round(float(max(heights) if take_taller else np.mean(heights)), 1),
                       widths_mm={int(k): round(float(v), 1) for k, v in m.widths_mm.items()},
                       foot_mm=round(2 * m.base_radius_mm, 1), slant_deg=round(m.slant_deg, 1),
                       frames=frames)
        feet = [f.base_world_m for f in found if f.base_world_m is not None]
        if model is not None:
            feet = [_in_arm_frame(model, f) for f in feet]
        if len(feet) == 2:
            apart = float(np.linalg.norm(feet[0] - feet[1])) * 1000
            if apart <= s.measure_agree_foot_mm:
                out.foot_xy_m = np.round(np.mean(feet, 0).astype(float), 5)
            else:
                self._say(tag, f"Two looks put the foot {apart:.1f} mm apart — not aiming by it.")
        return out

    def _looking_for(self, model, measure, expected_xy, known, target: str = "target") -> dict:
        """
        Vision.measure's expected_xy_m (and known=, target=), in the frame the
        camera is placed in. expected_xy is in the frame the arm moves by;
        known (known_around) in the one the scan measured them in — the
        wrist camera's calibration measured where those stand for it.
        target: the key in known of the one to measure.
        """
        if model is None:
            return {"expected_xy_m": expected_xy}
        out: dict = {"expected_xy_m": None if expected_xy is None else _in_camera_frame(model, expected_xy)}
        if known and _takes_known(measure):
            out.update(known=[dataclasses.replace(k, foot_xy_m=_scan_to_camera(model, k.foot_xy_m)) for k in known],
                       target=target)
        return out

    def can_measure_around(self, env, vision: Vision) -> bool:
        """Whether the wrist camera can measure the objects around the one under it (measure_neighbours)."""
        if not self.spec.measure_at_hover or not self.camera_located(env):
            return False
        if vision.measure is None:
            return True                      # plane_geometry.measure_in_frame takes them
        return _takes_known(vision.measure)

    def measure_neighbours(self, env, vision: Vision, tag: str, known, which: dict,
                           why_not: "dict | None" = None, frames: "list | None" = None) -> dict:
        """
        Objects standing around the one under the wrist camera, measured from
        where it is: {key: Measured} for those of which (key in known: label)
        two frames agree on the height. Not one of them is taken for another:
        each is measured as known's target in turn. Each it cannot measure is
        said, with why — or, given why_not, kept there instead. frames: the
        frames it took are added to it.

        What the jaws' room needs to know of a neighbour is how wide it is
        where the fingers pass (half_width_at), and from over a crowded spot
        the camera sees the neighbours from the side, where their height and
        shape show.
        """
        s = self.spec
        if not which or not self.can_measure_around(env, vision):
            return {}
        if vision.measure is None:
            from mrcf.perception import plane_geometry
            measure = plane_geometry.measure_in_frame
        else:
            measure = vision.measure
        looks: dict = {key: [] for key in which}
        for _ in range(2):
            time.sleep(0.3)                  # let the arm stop before judging the outlines
            frame = self._wrist_frame(env, tag)
            model = env.camera_model(s.camera, s.render_w, s.render_h)
            if frame is None or model is None:
                return {}
            if frames is not None:
                frames.append(frame)
            for key in which:
                r = measure(frame, model=model, **self._looking_for(model, measure, None, known, target=key))
                m = r.get("measurement") if r["ok"] else None
                if m is None or m.height_mm is None or r.get("occluded"):
                    why = r.get("why") or ("the gripper hides part of it" if r.get("occluded") else "no height")
                    looks[key].append(why)
                else:
                    looks[key].append(m)
        out = {}
        def cannot(key, label, why):
            if why_not is None:
                self._say(tag, f"Could not measure {label} from here ({why}).")
            else:
                why_not[key] = why

        for key, label in which.items():
            a, b = looks[key]
            if isinstance(a, str) or isinstance(b, str):
                cannot(key, label, a if isinstance(a, str) else b)
                continue
            if abs(a.height_mm - b.height_mm) > s.measure_agree_mm:
                cannot(key, label, f"two looks disagree on its height, {a.height_mm:.0f} vs {b.height_mm:.0f} mm")
                continue
            # Each level as the wider of the two looks: a neighbour is room taken.
            levels = {int(k): float(v) for k, v in a.widths_mm.items()}
            for k, v in b.widths_mm.items():
                levels[int(k)] = max(levels.get(int(k), 0.0), float(v))
            out[key] = Measured(height_mm=round(float(max(a.height_mm, b.height_mm)), 1),
                                widths_mm={k: round(v, 1) for k, v in sorted(levels.items())},
                                foot_mm=round(2 * max(a.base_radius_mm, b.base_radius_mm), 1),
                                slant_deg=round(a.slant_deg, 1))
        return out

    def find_foot(self, env, vision: Vision, tag: str, known) -> FootFound:
        """
        Where the object's foot is, from two wrist-camera frames that agree on
        it. known: the object, keyed "target", and the ones around it
        (known_around), so a neighbour's outline is not taken for it.

        The foot is what aiming needs, and it is there to see from straight
        above, where an object's height is not (2026-11-03: from a pick's hover
        over a full board, the outlines of e2, b1 and c7 had their feet within
        3.5 mm of the scan's, and no height). Where it stands is said both ways:
        in the scan's frame, which tells who it is, and in the frame the arm
        moves by, the aim — the second keeps its place relative to the jaws,
        which the camera is bolted beside, whatever the arm's joints read.
        """
        s = self.spec
        if s.camera is None:
            return FootFound(None, None, why="no wrist camera")
        if vision.measure is None:
            from mrcf.perception import plane_geometry
            measure = plane_geometry.measure_in_frame
        else:
            measure = vision.measure
        if not _takes_known(measure):
            target = next(k for k in known if k.key == "target")
            m = self.measure_at_hover(env, vision, tag, expected_xy=target.foot_xy_m)
            if m is None or m.foot_xy_m is None:
                return FootFound(None, None, why="it could not be measured")
            return FootFound(m.foot_xy_m, m.foot_xy_m, m)
        seen, found, frames = [], [], []
        for _ in range(2):
            time.sleep(0.3)                  # let the arm stop before judging the outline
            frame = self._wrist_frame(env, tag)
            if frame is None:
                return FootFound(None, None, why="the wrist camera gave no frame")
            model = env.camera_model(s.camera, s.render_w, s.render_h)
            frames.append(frame)
            if model is None:
                return FootFound(None, None, why="the camera is not located")
            if len(frames) == 1:
                from mrcf.perception import lighting
                light = lighting.warning(lighting.reading(frame), "the wrist camera's view")
                if light:
                    self._say(tag, f"Light: {light}")
            r = measure(frame, model=model, **self._looking_for(model, measure, None, known), need_height=False)
            if not r["ok"]:
                return FootFound(None, None, why=r["why"])
            m = r["measurement"]
            seen.append((model, np.asarray(m.base_world_m, float)[:2]))
            found.append((m, bool(r.get("occluded"))))
        apart = float(np.linalg.norm(seen[0][1] - seen[1][1])) * 1000
        if apart > s.measure_agree_foot_mm:
            return FootFound(None, None, why=f"two looks put its foot {apart:.1f} mm apart")
        aim = np.round(np.mean([_in_arm_frame(md, f) for md, f in seen], 0).astype(float), 5)
        scan = np.round(np.mean([_camera_to_scan(md, f) for md, f in seen], 0).astype(float), 5)
        heights = [m.height_mm for m, occluded in found if m.height_mm is not None and not occluded]
        measured = None
        if len(heights) == 2 and abs(heights[0] - heights[1]) <= s.measure_agree_mm:
            m = found[0][0]
            measured = Measured(height_mm=round(float(np.mean(heights)), 1),
                                widths_mm={int(k): round(float(v), 1) for k, v in m.widths_mm.items()},
                                foot_mm=round(2 * m.base_radius_mm, 1), slant_deg=round(m.slant_deg, 1),
                                foot_xy_m=aim, frames=frames)
        return FootFound(aim, scan, measured)

    def _servo_ready(self, env, vision: Vision) -> bool:
        """True when wrist-camera servoing can run: enabled, a detector, a calibration."""
        return (self.spec.servo and vision.find is not None and isinstance(env, WristServo)
                and env.wrist_servo_target_px is not None
                and (self.spec.camera is None or self.wrist_sending(env)))

    def _visual_servo(self, env: "RobotEnv", vision: Vision, tag: str, orientation=None, room: "float | None" = None,
                      height_m: "float | None" = None) -> bool:
        """
        Nudge the hover position using the wrist camera, if it is calibrated.
        True when it aligned the jaws over the object. height_m: the object's
        height, when known — where the camera's point on it is, which decides
        where "aligned" leaves the jaws once the gripper is tilted.

        The jaws are opened to the servo's calibration opening first (within
        `room`). Its target pixel marks the centre of THAT gap, and the arm
        reports the centre of the gap as it is now: with one moving jaw the two
        differ by half the difference in opening.

        Advisory on purpose. Servoing is an accuracy improvement, not a
        precondition: when it cannot run, or cannot see the object, the pick
        proceeds on kinematics exactly as it did before. Failing the pick here
        would turn an optional refinement into a new way to not pick anything up.
        """
        if not self.spec.servo or vision.find is None or not isinstance(env, WristServo):
            return False
        if self.spec.servo_open_m is not None:
            want = self.spec.servo_open_m if room is None else min(self.spec.servo_open_m, room)
            if abs(env.get_gripper_width() - want) > 0.001:
                env.set_gripper(want, n_steps=10)
        point = None if height_m is None else self.spec.servo_point_height_fraction * height_m
        ok, msg = env.visual_servo_xy(vision.find, tol_px=self.spec.servo_tol_px, orientation=orientation,
                                      point_height_m=point)
        if ok:
            self._say(tag, f"Visual servo: {msg}")
        elif msg.startswith("gripper camera not calibrated"):
            # Once per session: an uncalibrated rig would otherwise print this on
            # every attempt of every pick.
            if not self._servo_warned:
                self._say(tag, f"Visual servo unavailable ({msg}) — using kinematics alone.")
                self._servo_warned = True
        else:
            # Every time: a detection that fails on one attempt and not the next is
            # exactly what a failed pick has to be diagnosed from.
            self._say(tag, f"Visual servo failed ({msg}) — using kinematics alone.")
        return ok

    def _overhead_correct(self, env: "RobotEnv", at: np.ndarray, tol: float, rot, tag: str) -> np.ndarray:
        """
        Move until a fixed camera sees the gripper over `at` — the xy correction
        to the kinematics' target that does it (zeros when it cannot be measured).

        The kinematics only know where they think the gripper is. A camera that
        sees it, and saw the target, measures both the same way, so its own pose
        errors cancel: what is left is the arm's (joint offsets, a URDF off, a
        stall) plus the camera's noise. The correction found holds for the rest
        of the move at that spot — the descent and the climb out.
        """
        s = self.spec
        corr = np.zeros(2)
        self._overhead_err_m = None
        self._overhead_off = None
        self._overhead_first_err_m = None
        if not s.overhead_correct or not isinstance(env, OverheadSighting):
            return corr
        for i in range(s.overhead_max_iters):
            if isinstance(env, Settling):
                env.wait_still()
            off = env.overhead_gripper_offset()
            if off is None:
                if i == 0:
                    self._say(tag, "The camera cannot see the gripper's tag here — kinematics alone.")
                break
            err = (np.asarray(at, float) - (env.get_ee_pos() + off))[:2]
            if i == 0:
                # Before any correction: how far apart the camera and the arm's
                # own model are here. Kept per attempt (Outcome.attempts,
                # "overhead_err_mm"): consistently large, the camera's
                # calibration has gone stale — a deployment re-measures it.
                self._overhead_first_err_m = float(np.linalg.norm(err))
            if np.linalg.norm(err) < s.overhead_tol_m:
                self._overhead_err_m = float(np.linalg.norm(err))
                self._overhead_off = np.asarray(off, float)[:2].copy()
                if i or np.linalg.norm(corr):
                    self._say(tag, f"Camera: over the target within {np.linalg.norm(err) * 1000:.1f} mm "
                                   f"(corrected {np.linalg.norm(corr) * 1000:.1f} mm).")
                else:
                    self._say(tag, f"Camera: over the target within {np.linalg.norm(err) * 1000:.1f} mm.")
                return corr
            step = err * min(1.0, s.overhead_max_step_m / float(np.linalg.norm(err)))
            if np.linalg.norm(corr + step) > s.overhead_max_total_m:
                self._say(tag, f"Camera says the gripper is {np.linalg.norm(err) * 1000:.0f} mm off after "
                               f"{np.linalg.norm(corr) * 1000:.0f} mm of correction — more than "
                               f"{s.overhead_max_total_m * 1000:.0f} mm in all; keeping the kinematics.")
                return np.zeros(2)
            self._say(tag, f"Camera: gripper {np.linalg.norm(err) * 1000:.1f} mm off the target "
                           f"({err[0] * 1000:+.1f}, {err[1] * 1000:+.1f}) — correcting.")
            corr = corr + step
            goal = np.asarray(at, float).copy()
            goal[:2] += corr
            env.move_ee_to(goal, tol=tol, orientation=rot)
        return corr

    def _overhead_at_grasp(self, env, scan_xy, aim_xy, grasp, rot, tag: str, what: str = "grasp") -> "dict | None":
        """
        Where the fixed camera sees the jaws at the grasp, before they close,
        and the jaws moved onto their aim when they are more than
        spec.grasp_settle_tol_m off it (never more than grasp_settle_max_m in
        all). aim_xy: where the pick chose to close, in the camera's frame —
        the scan's object, moved by whatever the wrist camera found.

        Returns {"from_scan_mm", "drift_mm", "aim_before_mm", "aim_after_mm"},
        each [along the arm, across it] (drift None without a sighting at the
        hover), or None when the camera cannot see the jaws.

        The hover is corrected by the camera; the descent is not, and what
        is left of the hover's correction plus what the joints do under load
        on the way down can add up to 4-6 mm — with the object that much
        toward the base of the fingers' middle, the closing jaw cams it out
        toward the base. So the jaws are put on their aim here, where the
        error is last measurable, rather than trusting the descent.

        Across the arm, along the jaws' travel, not for less than 3 mm
        (grasp_settle_tol_m): the open jaws leave 6-10 mm of slack there.
        Along the arm, along the pads, from 1.5 mm (grasp_settle_along_m):
        a piece 2.5 mm toward the base of the pads is wedged out that way by
        the closing jaw, and one within 2 mm holds. A small correction that
        stalls does not carry the arm on: move_ee_to holds it.
        """
        s = self.spec
        if not self._can_overhead(env):
            return None

        def look():
            if isinstance(env, Settling):
                env.wait_still()
            seen = env.overhead_gripper_offset()
            return None if seen is None else np.asarray(seen, float)[:2]

        off = look()
        if off is None:
            self._say(tag, f"Camera: cannot see the gripper's tag at the {what}.")
            return None
        out = np.asarray(scan_xy, float)[:2] - self._base_xy(env)
        out = out / (np.linalg.norm(out) or 1.0)
        across = np.array([-out[1], out[0]])

        def axes(v):
            return [round(float(v @ out) * 1000, 1), round(float(v @ across) * 1000, 1)]

        miss = np.asarray(scan_xy, float)[:2] - (env.get_ee_pos()[:2] + off)
        drift = None if self._overhead_off is None else off - self._overhead_off
        err = np.asarray(aim_xy, float)[:2] - (env.get_ee_pos()[:2] + off)
        got = {"from_scan_mm": axes(miss), "drift_mm": None if drift is None else axes(drift),
               "aim_before_mm": axes(err)}
        seen = "object as the scan saw it" if what == "grasp" else "target"
        msg = (f"Camera at the {what}: the {seen} is {np.linalg.norm(miss) * 1000:.1f} mm "
               f"from the jaws ({got['from_scan_mm'][0]:+.1f} along the arm, {got['from_scan_mm'][1]:+.1f} across)")
        if drift is not None:
            msg += (f"; the descent moved them {np.linalg.norm(drift) * 1000:.1f} mm off the kinematics "
                    f"({got['drift_mm'][0]:+.1f} along, {got['drift_mm'][1]:+.1f} across)")
        self._say(tag, msg + ".")

        start = env.get_ee_pos()[:2].copy()
        moved = np.zeros(2)                    # as the arm says it went, not as asked: near its limits
        best = (float(np.linalg.norm(err)), env.get_ee_pos().copy(), err)
        for _ in range(s.overhead_max_iters):  # again: it can stop short of a small move
            if abs(float(err @ out)) <= s.grasp_settle_along_m and abs(float(err @ across)) <= s.grasp_settle_tol_m:
                break
            if np.linalg.norm(moved + err) > s.grasp_settle_max_m:
                self._say(tag, f"Camera at the {what}: the jaws are {np.linalg.norm(err) * 1000:.1f} mm from their "
                               f"aim, and moving them there would take them more than "
                               f"{s.grasp_settle_max_m * 1000:.0f} mm round the object; closing where they are.")
                break
            here = env.get_ee_pos()
            arrived = env.move_ee_to(np.array([here[0] + err[0], here[1] + err[1], grasp[2]]), tol=0.001,
                                     max_step_deg=s.grasp_step_deg, orientation=rot)
            moved = env.get_ee_pos()[:2] - start
            seen = look()
            if seen is None:
                break
            err = np.asarray(aim_xy, float)[:2] - (env.get_ee_pos()[:2] + seen)
            if np.linalg.norm(err) < best[0]:
                best = (float(np.linalg.norm(err)), env.get_ee_pos().copy(), err)
            if arrived is False:
                # At its limits the arm stalls and lurches, and stalled
                # corrections can leave the jaws further off than they started.
                # Stop at the first one it cannot make.
                break
        if np.linalg.norm(err) > best[0] + 0.0005:
            self._say(tag, f"Camera at the {what}: that left the jaws {np.linalg.norm(err) * 1000:.1f} mm off — "
                           f"back to where they were {best[0] * 1000:.1f} mm off.")
            env.move_ee_to(best[1], tol=0.001, max_step_deg=s.grasp_step_deg, orientation=rot)
            moved, err = env.get_ee_pos()[:2] - start, best[2]
        got["aim_after_mm"] = axes(err)
        if np.any(np.abs(moved) > 1e-4):
            self._say(tag, f"Camera at the {what}: moved the jaws {np.linalg.norm(moved) * 1000:.1f} mm onto their "
                           f"aim; now {np.linalg.norm(err) * 1000:.1f} mm off it ({got['aim_after_mm'][0]:+.1f} "
                           f"along, {got['aim_after_mm'][1]:+.1f} across).")
        return got

    def _other_object(self, target: Target, measured: Measured, obstacles=()) -> "str | None":
        """
        Why what the wrist camera measured cannot be the target — a height
        that disagrees with the one it measured before, with something else
        near enough to be it (spec.other_object_near_m) — or None.

        In a crowded spot the camera sees several objects, and a detector that
        takes the nearest blob can lock onto a neighbour, and the jaws then
        close on the neighbour. Height only: a foot from the overhead scan is
        not comparable — it reads a knight's irregular base 20 mm where the
        wrist camera reads ~30.
        """
        if self.shape_unsettled(target):
            # The height it is held to is one look, which may be the wrong one:
            # disagreeing with it is what a second look is for (_measure).
            return None
        off = self._height_off(target, measured)
        if off is None or not self._neighbour_near(target, measured, obstacles):
            return None
        return off

    def _height_off(self, target: Target, measured: Measured) -> "str | None":
        """How a look's height disagrees with the one known for target (beyond same_height_tol_m), or None."""
        if target.height_m is not None and abs(measured.height_mm / 1000 - target.height_m) > self.spec.same_height_tol_m:
            return f"{measured.height_mm:.0f} mm tall, not {target.height_m * 1000:.0f}"
        return None

    def _part_seen(self, target: Target, measured: Measured, obstacles=()) -> "str | None":
        """
        Why a look disagreeing with target's known height is not trusted
        (nothing else stands near enough to be what it saw: a part of it, in
        poor light or half hidden), or None when it is trusted or is
        _other_object's. 2026-11-17: a wooden figure, 44 mm from three looks
        from the side, read 21 from over it in a dim room — three attempts
        retried for "another object" with nothing within 11 cm.
        """
        if self.shape_unsettled(target):
            return None
        off = self._height_off(target, measured)
        if off is None or self._neighbour_near(target, measured, obstacles):
            return None
        return f"{off} — but nothing else stands near: a part of it"

    def _neighbour_near(self, target: Target, measured: Measured, obstacles=()) -> bool:
        """Whether something besides target stands within spec.other_object_near_m of where a look found it (True: no limit set)."""
        from mrcf.policies.parallel_jaw.reach import BASE_LABEL

        s = self.spec
        if s.other_object_near_m is None:
            return True
        at = np.asarray(measured.foot_xy_m if measured.foot_xy_m is not None else target.position, float)[:2]
        return any(ob.label != BASE_LABEL and ob.position is not None
                   and float(np.linalg.norm(np.asarray(ob.position, float)[:2] - at))
                   < s.other_object_near_m + (ob.radius_m or s.unmeasured_obstacle_radius_m)
                   for ob in obstacles)

    def _too_short(self, target: Target, measured: Measured) -> "str | None":
        """
        Why a look cannot have outlined the whole object — shorter, by more
        than same_height_tol_m, than the shortest it could be
        (Target.min_height_m) — or None. 2026-11-03, the full chess board:
        looks from beside the d1 queen and the f1 bishop read them 28 and
        22 mm tall, every piece being 45 mm or more, and the grasps planned
        from them went 20 mm lower.
        """
        if target.min_height_m is None:
            return None
        if measured.height_mm / 1000 < target.min_height_m - self.spec.same_height_tol_m:
            return (f"{measured.height_mm:.0f} mm tall, shorter than the shortest it could be "
                    f"({target.min_height_m * 1000:.0f}) — a part of it")
        return None

    def _unlike_scan(self, target: Target, measured: Measured) -> "str | None":
        """
        Why a look at the hover is not of the object the fixed camera sees —
        its foot out of proportion with the scan's (spec.measure_foot_ratio) —
        or None.

        Two frames can agree on the same wrong outline, and a grasp re-planned
        from it is wrong too. Over 22 good looks the two feet were 0.74-1.54 of
        each other (the scan reads a tall piece's foot short or long by its
        lean in the view).
        """
        if not target.foot_m or not measured.foot_mm:
            return None
        ratio = measured.foot_mm / (target.foot_m * 1000)
        lo, hi = self.spec.measure_foot_ratio
        if lo <= ratio <= hi:
            return None
        return f"{measured.foot_mm:.0f} mm across, where the fixed camera measured {target.foot_m * 1000:.0f}"

    def _fit_measured(self, env, measured: Measured, room: float, hover, tag: str) -> GraspPlan:
        """The grasp re-planned from what the wrist camera measured, opened within `room`."""
        s = self.spec
        shaped = self.plan_from_shape(measured.height_mm / 1000, measured.widths_mm, measured.foot_mm / 1000)
        opening = max(min(shaped.pregrasp_open_m, room), shaped.grip_width_m + 2 * s.jaw_slack_m)
        tips = hover[2] - s.table_z - s.tip_below_grasp_m
        if tips < shaped.height_m:
            self._say(tag, f"Note: the hover left the jaw tips {(shaped.height_m - tips) * 1000:.0f} "
                           "mm below the object's top — later picks of it will hover higher.")
        plan = dataclasses.replace(shaped, pregrasp_open_m=opening)
        # Wider if they are already wider, never narrower: once the servo has
        # aligned, the jaws are at its opening (within the room the neighbours
        # leave), and narrowing them then moves where the arm thinks their
        # centre is by a model of the opening (JAW_CENTRE_SLIDE) — and takes
        # millimetres of slack off a grasp aligned to a few.
        if env.get_gripper_width() < opening - 0.002:
            env.set_gripper(opening, n_steps=10)
        _, why = self.grasp_height_for(measured.height_mm / 1000, measured.widths_mm)
        self._say(tag, f"Measured {measured.height_mm:.0f} mm tall, foot "
                       f"{measured.foot_mm:.0f} mm — grasping at {plan.grasp_height_m * 1000:.0f} mm "
                       f"({why}), jaws {max(opening, env.get_gripper_width()) * 1000:.0f} mm.")
        return plan

    def _can_overhead(self, env) -> bool:
        """True when a fixed camera that sees the gripper can correct a hover (see _overhead_correct)."""
        return self.spec.overhead_correct and isinstance(env, OverheadSighting)
