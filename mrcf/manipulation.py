"""
manipulation.py — the standard manipulation actions: the planner's names turned
into the policies' geometry, and what happened written back.

The planner says "pick white_knight_g1". A policy (mrcf/policies/) takes geometry —
where the object's foot is, how tall it is, what stands next to it — and
returns an Outcome. This is the layer between them:

  1. find the name in the world model (vlm.SceneState)
  2. turn the object into a policies.Target, its neighbours into Obstacles
  3. run the arm's policy (skill_executor.run_policy: the safety monitor, and a
     description of the scene when it fails)
  4. record the Outcome: in the world model (held, put down), in the object
     memory (picked, placed, measured), and through the scenario's own hooks

A stop (MotionStopped) mid-action ends the policy without an Outcome. The
gripper is then asked what it holds, and the world model is made to agree
before the stop is passed on: a pick stopped after the jaws closed has the
object in the gripper, and believing it is still on the table would have the
next pick open the jaws and drop it.

The policies never see the world model or the memory, and the world model
never sees a policy's insides; only this module knows both. So the same
policies serve any scenario, and a scenario's actions work on any arm whose
policy library implements policies.base.Policies.

A deployment installs it once, with its arm's policy library:

    from mrcf import backends
    from mrcf import manipulation

    manipulation.install(backends.arm().make_policies(), vision=Vision(...), scenario=Scenario(...))

which registers pick, place, move_above, home, open_gripper and close_gripper
for the planner. gripper_sequence_validator is the matching plan-time rule,
for safety.configure().

A verb of the deployment's own on one object — a push, a nudge — is one call
to object_action, with the same bookkeeping; the policy library gains the
method that does it, taking the same Target and Obstacles.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable

import numpy as np

from mrcf import flight_recorder
from mrcf import safety
from mrcf.interlock import MotionStopped
from mrcf.policies.base import Obstacle, Policies, Target, Vision
from mrcf import skill_executor
from mrcf.skill_executor import run_policy
from mrcf.action_registry import ACTIONS
from mrcf.vlm import SceneObject, SceneState, register_action

if TYPE_CHECKING:
    from mrcf.robot_env import RobotEnv


@dataclass
class Scenario:
    """
    What the deployment knows that neither the framework nor the policies do.
    Every hook is optional.

      max_height(obj) -> float | None   the tallest an object nobody has measured
                                        could be (the scenario's tallest object)
      min_height(obj) -> float | None   and the shortest
      assumed_shape(obj) -> dict | None the widest it could be up its height,
                                        {height_mm: width_mm}: how slim its kind is
                                        above its foot, as the deployment has measured
      describe(track) -> str            a remembered object's description, as
                                        scene_scan.observe takes it
      picked(obj, start_xy, outcome)    after every pick, whatever happened — a log
      vacated(scene_state, obj, start_xy)
                                        after a pick took obj from start_xy: the spot
                                        is free, and the scenario may offer it as a
                                        place target again (chess: its square), so
                                        putting the object back needs no fresh scan
      placed(obj, target, outcome)      after every place
      approach_tilt(obj) -> float | None
                                        how far the gripper should lean from straight
                                        down to pick obj or place on it, by the
                                        scenario's own rules (chess: by rank); where
                                        the arm's search for an orientation starts
      aside_spots(scene_state, obj, for_obj, headed_for) -> [SceneObject]
                                        free spots near obj to set it down on for a
                                        moment while for_obj is picked (make-room
                                        recovery), best first, as place targets (chess:
                                        empty squares) — not on or beside headed_for,
                                        the place target for_obj goes to next (None:
                                        not known); none: obj is only pushed aside
      home_of(scene_state, obj, xy) -> (name, xy) | None
                                        the spot an object standing at xy belongs on: its
                                        name as a place target, and its middle (chess: that
                                        square). What a nudge with no target pushes it back
                                        to, and where an object moved aside goes back
      carries(obj) -> bool              whether another object may be put on top of obj
                                        (a block on a block: place_on). Not given: nothing
                                        is stacked — a place at an object is refused for
                                        room, as it stands there (a chess piece on a piece)
      confirm_placed(env, scene_state, obj, at, on) -> bool | None
                                        a quick look, once let go of, at whether obj stands
                                        with its foot at `at` (on `on`): False, it is not
                                        (fell, slid off) and the place counts as failed, so a
                                        plan stops and looks again; None, it cannot tell
                                        (perception/placed_check.py)
      opening(scene_state, name) -> (Opening, SceneObject) | str | None
                                        the container called name (a tub, a bowl) as the
                                        policies take it (policies.base.Opening), and the scene's
                                        entry for it; a str: it is a container but cannot be put
                                        into now, and why; None: not a container. place_in needs
                                        it, and a place or a place_on at a container goes in it
      holds(scene_state, obj) -> str | None
                                        the name of the container obj stands in, if any: a pick
                                        out of one is refused (its walls are not passed yet)
      find_opening(env, scene_state, name) -> bool
                                        look for the container called name that is not known
                                        yet (a tub nobody has asked for), and keep it so opening
                                        knows it: whether it was found. What place_in does with a
                                        container it is told by the user's words
      opening_measured(scene_state, name, rim_m)
                                        a place_in measured how high the container's walls
                                        stand (rim_m, metres above the surface): kept, so the
                                        next one need not look
      recipe(scene_state, obj) -> dict | None
                                        how to pick obj as it was picked before (a catalogue of
                                        the deployment's kinds): {"grip_at_m", "max_height_m",
                                        "said"} — gripped there, the hover clear of that top,
                                        no looks for its height. None: a first meeting
      outline_again(env, scene_state, obj) -> bool
                                        obj outlined afresh, found by what it is, before a pick by
                                        its wall (pick_by_rim), whose plan is its outline: the
                                        scan's may be a scrap of it. Whether it was (nothing
                                        changed when not)
    """
    max_height: "Callable | None" = None
    min_height: "Callable | None" = None
    assumed_shape: "Callable | None" = None
    describe: "Callable | None" = None
    picked: "Callable | None" = None
    vacated: "Callable | None" = None
    placed: "Callable | None" = None
    approach_tilt: "Callable | None" = None
    aside_spots: "Callable | None" = None
    home_of: "Callable | None" = None
    carries: "Callable | None" = None
    confirm_placed: "Callable | None" = None
    opening: "Callable | None" = None
    holds: "Callable | None" = None
    find_opening: "Callable | None" = None
    opening_measured: "Callable | None" = None
    outline_again: "Callable | None" = None
    # Whether the scan's outline of obj is one to look at again before a plain pick (OUTLINE_BEFORE_PICK):
    # (scene_state, obj) -> bool. None: every one.
    outline_suspect: "Callable | None" = None
    recipe: "Callable | None" = None


_policies: "Policies | None" = None
_vision = Vision()
_scenario = Scenario()


def policies() -> Policies:
    """The installed policy library."""
    if _policies is None:
        raise RuntimeError("manipulation.install() has not been called")
    return _policies


def vision() -> Vision:
    """The installed Vision: what the deployment's perception gives a pick."""
    return _vision


# ── From the world model to geometry ───────────────────────────────────────────

def target_of(obj: SceneObject) -> Target:
    """An object in the world model, as what a policy needs to know about it."""
    shape, geom = obj.measured_shape, obj.geometry
    return Target(
        position=np.zeros(3) if obj.position is None else np.asarray(obj.position, float),
        foot_m=geom["base_diameter_m"] if geom else None,
        height_m=shape["height_mm"] / 1000.0 if shape else None,
        widths_mm=shape["widths_mm"] if shape else None,
        max_height_m=_scenario.max_height(obj) if _scenario.max_height else None,
        min_height_m=_scenario.min_height(obj) if _scenario.min_height and not shape else None,
        assumed_widths_mm=_scenario.assumed_shape(obj) if _scenario.assumed_shape and not shape else None,
        tilt_deg=_approach_tilt(obj),
        shape_looks=shape.get("looks") if shape else None,
        grasp_above_m=getattr(obj, "grasp_above_m", None),
        faces_deg=getattr(obj, "faces_deg", None),
        surface_z_m=float(obj.position[2]) if obj.on and obj.position is not None else None,
        outlines=geom.get("outlines") if geom else None)


def _approach_tilt(obj) -> "float | None":
    """The lean the scenario's rules ask for over obj (Scenario.approach_tilt), if any."""
    return _scenario.approach_tilt(obj) if _scenario.approach_tilt else None


# A scan that does not see an object is not proof it has gone: a dark object on
# a dark surface can come and go from scan to scan. So for this long after
# it was last seen, an object the memory has as missing is kept clear of where
# it stood. Longer than that, it has more likely really been taken away.
MISSED_KEEP_CLEAR_S = 15 * 60

# An object the scan put inside the very object being picked — its centre within
# that one's foot — is part of it, not something beside it: two cannot stand in
# one place. Not kept clear of. A deployment turns it on (mrcf/tabletop:
# PARTS_INSIDE_ARE_ITS_OWN). 2026-11-19, the SO-101: a wooden figure outlined as
# two, its head 1 cm from its body, refused twice — "jaws need 30 mm open here
# but multicoloured_toy leaves 0 mm".
PARTS_INSIDE_ARE_ITS_OWN = False


def _inside_foot(o, of) -> bool:
    """Whether o's centre stands within of's foot (half its base diameter)."""
    if of is None or o.position is None or getattr(of, "position", None) is None or not getattr(of, "geometry", None):
        return False
    r = float(of.geometry.get("base_diameter_m") or 0.0) / 2
    return float(np.linalg.norm(np.asarray(o.position[:2], float) - np.asarray(of.position[:2], float))) < r


def obstacles_around(scene_state: SceneState, exclude=None, quiet: bool = False) -> list:
    """
    Every object standing in the scene except `exclude`, as an Obstacle — and
    every one the memory has as missing but saw within MISSED_KEEP_CLEAR_S,
    where it was last seen.

    Not kept clear of, and said so: an object only a vision model has seen
    (SceneObject.confirmed, object_memory.Track.confirmed — objects it invents
    on an empty surface would refuse picks next to them), and a missing object
    whose spot an object the camera measures now stands on: two cannot stand in
    one place, so it is not there (kept clear of, it can refuse the pick of the
    very object standing where it had been).
    """
    excluded = list(exclude) if isinstance(exclude, (list, tuple, set)) else [exclude]
    standing = [o for o in scene_state.objects if o.position is not None]
    out, ignored = [], []
    for o in standing:
        if any(o is e for e in excluded):
            continue
        part_of = next((e for e in excluded if PARTS_INSIDE_ARE_ITS_OWN and _inside_foot(o, e)), None)
        if part_of is not None:
            ignored.append(f"{o.name} (inside {part_of.name}'s own foot: part of it)")
            continue
        if not o.confirmed:
            ignored.append(f"{o.name} (only the vision model reported it)")
            continue
        shape = getattr(o, "measured_shape", None) or {}
        widths = shape.get("widths_mm") or (_scenario.assumed_shape(o) if _scenario.assumed_shape else None)
        height, below = _height_of(o), stack_below(scene_state, o)
        if height is not None and below:         # on a stack: in the way up to the stack's top
            height += float(o.position[2]) - float(below[-1].position[2])
        out.append(Obstacle(np.asarray(o.position, float),
                            o.geometry["base_diameter_m"] / 2 if o.geometry else None, o.name, height,
                            widths_mm=widths or None, outlines=(o.geometry or {}).get("outlines")))
    feet = [(np.asarray(o.position[:2], float), o.geometry["base_diameter_m"] / 2)
            for o in standing if o.confirmed and o.geometry]
    memory = scene_state.memory
    if memory is not None:
        now = datetime.now()
        for t in memory.in_state("missing"):
            if t.position is None or any(e is not None and t.id == getattr(e, "track_id", None) for e in excluded):
                continue
            try:
                age = (now - datetime.fromisoformat(t.last_seen)).total_seconds()
            except ValueError:
                continue
            if age > MISSED_KEEP_CLEAR_S:
                continue
            name, foot = memory.name(t), t.foot_mm
            where = np.asarray(t.position[:2], float)
            if not t.confirmed:
                ignored.append(f"{name} (missed by the last scan; only the vision model ever reported it)")
                continue
            if foot and any(np.linalg.norm(where - p) < foot / 2000.0 + r for p, r in feet):
                ignored.append(f"{name} (missed by the last scan; something else stands where it stood)")
                continue
            out.append(Obstacle(where, foot / 2000.0 if foot else None,
                                f"{name} (missed by the last scan; last seen there)",
                                t.height_mm / 1000.0 if getattr(t, "height_mm", None) else None))
    if ignored and not quiet:
        print(f"  Not keeping clear of {'; '.join(ignored)}.")
    return out


def stack_below(scene_state: SceneState, obj) -> list:
    """What obj stands on, all the way down its stack (nearest first); [] on the surface."""
    out: list = []
    at = obj
    while at.on:
        # SceneObject.on is the scene's own name for it (scene_scan.observe): exact, not find()'s
        # loose match; and by identity — two objects' == compares their numpy positions.
        under = next((o for o in scene_state.objects if o.name == at.on), None)
        if under is None or under is obj or any(under is o for o in out) or under.position is None:
            break
        out.append(under)
        at = under
    return out


def on_top_of(scene_state: SceneState, obj) -> list:
    """What stands directly on obj."""
    return [o for o in scene_state.objects if o.position is not None and o.on == obj.name]


def stack_top(scene_state: SceneState, obj):
    """The object on top of obj's stack: obj itself when nothing stands on it."""
    seen = [obj]
    while True:
        above = on_top_of(scene_state, seen[-1])
        if not above or any(above[0] is o for o in seen):
            return seen[-1]
        seen.append(above[0])


def _check_grip_width(env, obj, outcome) -> None:
    """
    Said when the jaws closed on the object at a width far from the one its
    shape gives where they hold it: its size is wrong — a known size that is
    not the object's, or a misread — and a place that goes by its height lets
    go at the wrong height (HISTORY.md).
    """
    shape = getattr(obj, "measured_shape", None) or {}
    widths = shape.get("widths_mm") or {}
    grasp = getattr(outcome, "grasp", None)
    at_mm = getattr(grasp, "grasp_height_m", None)
    try:
        held_mm = float(env.get_gripper_width()) * 1000.0
    except Exception:                        # noqa: BLE001 — a reading, not a reason to fail the pick
        return
    if not widths or at_mm is None or not np.isfinite(held_mm):
        return
    near = min(widths, key=lambda h: abs(float(h) - at_mm * 1000.0))
    expected = float(widths[near])
    if abs(held_mm - expected) > GRIP_WIDTH_MISMATCH_MM:
        print(f"  [Shape] {obj.name} is {held_mm:.0f} mm across the jaws, not the {expected:.0f} mm its shape says: "
              "its size is wrong (a known size, or a misread) — a place by its height lets go at the wrong height.")


# How far the jaws' width on a held object may be from its shape's before it is said.
GRIP_WIDTH_MISMATCH_MM = 5.0


def _height_of(obj) -> "float | None":
    """How tall obj is: as the wrist camera measured it, else the tallest the scenario says it may be."""
    shape = getattr(obj, "measured_shape", None)
    if shape and shape.get("height_mm"):
        return float(shape["height_mm"]) / 1000.0
    return _scenario.max_height(obj) if _scenario.max_height else None


def _track(scene_state: SceneState, obj):
    memory = scene_state.memory
    return memory, (memory.get(obj.track_id) if memory is not None else None)


def _remember_around(scene_state: SceneState, around: dict) -> None:
    """
    The neighbours a pick measured on the way (Outcome.around, by obstacle
    label: the object's name), kept as each one's own measurement: the next
    pick beside them knows the room they leave before it moves.
    """
    memory = scene_state.memory
    if memory is None:
        return
    for label, m in (around or {}).items():
        obj = next((o for o in scene_state.objects if o.name == label), None)
        track = memory.get(obj.track_id) if obj is not None else None
        if obj is None or track is None:
            continue
        memory.measured(track.id, m.height_mm, m.widths_mm, m.foot_mm)
        obj.measured_shape = track.shape() if track.shape_current() else None


def _retire_where(memory, track, start, obj) -> None:
    """
    A missing object last seen where the one just picked stood was it, under
    an older track (the scan saw only one there since): retired, so it is not
    kept clear of on the spot it is about to be put back on. 2026-11-04: the
    b8 knight, picked under a new track, could not be put back on b8 — "jaws
    need 29 mm, black_knight_obj84 (missed by the last scan; last seen there)
    leaves 0 mm".
    """
    foot = (obj.geometry or {}).get("base_diameter_m") or 0.02
    for t in memory.in_state("missing"):
        if t.id != track.id and t.position is not None and \
                float(np.linalg.norm(np.asarray(t.position[:2], float) - start[:2])) <= foot / 2 + 0.01:
            memory.superseded(t.id, by=track.id, why="picked from where it was last seen")


def _holding_now(env) -> "bool | None":
    """Read the physical hold after an action or stop; None if it cannot say."""
    try:
        return bool(policies().holding(env))
    except Exception:
        return None


# ── The actions ────────────────────────────────────────────────────────────────

def _jaws_hold_something(env: "RobotEnv") -> bool:
    """
    The gripper's own sensor says it holds something, its jaws closed on it:
    a pick opens them first, and would drop it. Only a load sensor counts
    (capabilities.GripSensing), and only narrower than an empty close — jaws
    left part-open after a place are no hold.
    """
    from mrcf.capabilities import GripSensing

    spec = getattr(_policies, "spec", None)
    if spec is None or not isinstance(env, GripSensing):
        return False
    try:
        return bool(env.gripper_loaded()) and float(env.get_gripper_width()) < float(spec.miss_width_m)
    except Exception:                  # noqa: BLE001 — a sensor that cannot be read says nothing
        return False


def _refuse(skill: str, why: str) -> bool:
    """
    Said, and told to the planner (skill_executor.report), refused before
    anything moved: False. 2026-11-13, the UR5: "blue is on top of green" went
    to the console only, the planner heard "it failed", and it planned the
    whole tower again from the start, undoing what it had built.
    """
    print(f"  [Refused] {why[0].upper()}{why[1:]}.")
    skill_executor.report(skill, False, why, refused=True)
    return False


# Before a plain pick, the object outlined again by what it is (Scenario.outline_again), as pick_by_rim
# does: the scan's outline is the segmenter's alone, and a fragment of the object aims the pick at the
# fragment. A deployment turns it on (mrcf/tabletop: OUTLINE_BEFORE_PICK). 2026-11-19 14:03, the SO-101: a
# wooden figure outlined as 16 mm of its 23, gripped at that fragment's middle 5 mm off its own, tipped by
# the jaws and pushed off the table by the next try (user: "the item rolled off the table").
OUTLINE_BEFORE_PICK = False


def _pick(env, scene_state, object_name, _by: str = "pick", **_):
    if OUTLINE_BEFORE_PICK and _by == "pick" and _scenario.outline_again is not None:
        obj = scene_state.find(object_name, among="objects")
        if (obj is not None and obj.position is not None and not _jaws_hold_something(env)
                and (_scenario.outline_suspect is None or _scenario.outline_suspect(scene_state, obj))):
            # Looked at from where the arm is, not moved for it: 2026-11-19 15:29 (filmed), before the
            # plug the arm went home and then straight up to its park pose, to see past itself.
            _scenario.outline_again(env, scene_state, obj, move=False)
    if _jaws_hold_something(env):
        why = ("the gripper already holds something, though nothing is known to be in it — put it down "
               "(open_gripper over a clear spot) before another pick, which would drop it")
        print(f"  [Refused] {why}.")
        skill_executor.report("pick", False, why, refused=True)
        return False
    obj = scene_state.find(object_name, among="objects")
    if obj is None or obj.position is None:
        return _refuse("pick", f"'{object_name}' is not in the scene or its position is unknown — observe first")
    above = on_top_of(scene_state, obj)
    if above:
        return _refuse("pick", f"{above[0].name} is on top of {obj.name}: move {above[0].name} off first")
    inside = _scenario.holds(scene_state, obj) if _scenario.holds else None
    if inside:
        return _refuse("pick", f"{obj.name} is inside {inside}: picking things out of a container is not "
                               "built yet (its walls would be in the way)")
    print(f"  Picking {obj.name}{f' off {obj.on}' if obj.on else ''}...")
    start = np.asarray(obj.position[:2], float).copy()
    memory, track = _track(scene_state, obj)
    with flight_recorder.RECORDER.action("pick", object=obj.name, at=start, track=obj.track_id) as act:
        try:
            target = target_of(obj)
            known = _scenario.recipe(scene_state, obj) if _scenario.recipe and not target.measured else None
            if known and known.get("grip_at_m") is not None:
                print(f"  [Catalogue] {obj.name}: {known.get('said') or 'picked as before'}.")
                target = dataclasses.replace(target, grip_at_m=float(known["grip_at_m"]),
                                             max_height_m=known.get("max_height_m") or target.max_height_m)
                across = known.get("across_m")
                if across is not None and float(across) > float(target.foot_m or 0.0):
                    # The jaws opened past what its kind is across, not only the scan's foot: 2026-11-20 09:13,
                    # the plug's foot read 44 mm, it is 59 across, the jaws opened 52 and one came down on it.
                    print(f"  [Catalogue] {obj.name}: its kind is {float(across) * 1000:.0f} mm across, the scan "
                          f"made its foot {float(target.foot_m or 0.0) * 1000:.0f} — the jaws open for "
                          f"{float(across) * 1000:.0f}.")
                    target = dataclasses.replace(target, foot_m=float(across))
            outcome = run_policy(getattr(policies(), _by), "pick", env, target=target,
                                 obstacles=obstacles_around(scene_state, [obj] + stack_below(scene_state, obj)),
                                 vision=_vision)
        except MotionStopped:
            if _holding_now(env):
                obj.position = None
                if track is not None:
                    memory.picked(track.id)
                print(f"  [Stop] Stopped with {obj.name} in the gripper — the world model has it held.")
            raise
        _judge(act, outcome, "pick")
        m = outcome.measured
        if m is not None and track is not None:
            memory.measured(track.id, m.height_mm, m.widths_mm, m.foot_mm, looks=m.looks)
            # The object's next pick plans by this, not only the next scan.
            obj.measured_shape = track.shape() if track.shape_current() else None
        _remember_around(scene_state, outcome.around)
        # A failed lift (or an exception after closing) may still leave the object
        # in the jaws. Do not let recovery start another pick and drop it.
        if outcome.ok or (not outcome.refused and _holding_now(env)):
            _check_grip_width(env, obj, outcome)
            obj.grasp = outcome.grasp
            obj.position = None                  # now in the gripper
            obj.on = None
            if track is not None:
                memory.picked(track.id)
                _retire_where(memory, track, start, obj)
            if _scenario.vacated:
                _scenario.vacated(scene_state, obj, start)
        if _scenario.picked:
            _scenario.picked(obj, start, outcome)
    if outcome.refused and outcome.blocked_by and not _ROOM["busy"]:
        made = _make_room(env, scene_state, obj, outcome)
        if made is not None:
            return made
    return outcome.ok


def _place(env, scene_state, target_name, **_):
    # A place target first, by find()'s loose match ("E5", "square e5"); an
    # object only by its exact name. Loosely, a spot that has gone since the
    # plan was made ("where_blue_block_was", dropped by a scan) matches the
    # object it was named for, and the held one would be stacked on it.
    tgt = scene_state.find(target_name, among="targets") or next(
        (o for o in scene_state.objects if o.name == target_name), None)
    if tgt is None or tgt.position is None:
        return _refuse("place", f"'{target_name}' is not in the scene or its position is unknown — observe first")
    # The scene's own name for it, not the planner's spelling ("E5", "square e5").
    target_name = tgt.name
    held = scene_state.held_object()
    if held is None:
        return _refuse("place", "nothing is held — pick something first")
    if _opening(scene_state, target_name) is not None:
        return _place_in(env, scene_state, target_name)          # a container: into it
    if any(tgt is o for o in scene_state.objects) and _carries(tgt):
        # Put down where an object stands is on top of it: anywhere lower, in it.
        return _place_on(env, scene_state, target_name)
    print(f"  Placing {held.name} at {target_name}...")
    return _put_held(env, scene_state, held, tgt, np.asarray(tgt.position, float),
                     obstacles=obstacles_around(scene_state, held), tilt=_approach_tilt(tgt))


def _carries(obj) -> bool:
    """The scenario lets another object be put on top of obj (Scenario.carries)."""
    return _scenario.carries is not None and bool(_scenario.carries(obj))


def _place_on(env, scene_state, object_name, **_):
    """The held object put down on top of another (stacked): on the top of that one's stack."""
    held = scene_state.held_object()
    if held is None:
        return _refuse("place_on", "nothing is held — pick something first")
    if _opening(scene_state, object_name) is not None:
        return _place_in(env, scene_state, object_name)          # "on" the tub is into it
    base = scene_state.find(object_name, among="objects")
    if base is None or base.position is None:
        return _refuse("place_on", f"'{object_name}' is not in the scene or its position is unknown — observe first")
    if base is held:
        return _refuse("place_on", f"{held.name} is the object in the gripper: it cannot go on itself")
    top = stack_top(scene_state, base)
    if not _carries(top):
        print(f"  [Refused] Nothing is stacked on {top.name} here (the task does not put one object on another).")
        return False
    if top is not base:
        print(f"  {top.name} is on top of {base.name}'s stack: putting {held.name} on {top.name}.")
    height = _height_of(top)
    if height is None:
        print(f"  [Refused] How tall {top.name} is has not been measured, so its top is not known: "
              "scan the table again, or say how tall it is.")
        return False
    surface = float(top.position[2]) + height
    at = np.array([float(top.position[0]), float(top.position[1]), surface])
    below = stack_below(scene_state, top)
    up = surface - float((below or [top])[-1].position[2])
    print(f"  Placing {held.name} on {top.name}, {up * 1000:.0f} mm up"
          + (f" (a stack of {len(below) + 1})" if below else "") + "...")
    return _put_held(env, scene_state, held, top, at, surface_z=surface, on=top, faces_deg=top.faces_deg,
                     obstacles=obstacles_around(scene_state, [held, top] + stack_below(scene_state, top)))


def _put_held(env, scene_state, held, tgt, at, *, obstacles, tilt=None, surface_z=None, on=None,
              faces_deg=None) -> bool:
    """
    The place itself, and what the world model learns from it: held put down
    with its foot at `at` — on the surface, or on `on` (another object) whose
    top is surface_z.
    """
    raised = {k: v for k, v in (("surface_z", surface_z), ("faces_deg", faces_deg)) if v is not None}
    with flight_recorder.RECORDER.action("place", object=held.name, target=tgt.name, at=at,
                                         track=held.track_id, **({"on": on.name} if on is not None else {})) as act:
        try:
            outcome = run_policy(policies().place, "place", env, at=at, held=target_of(held), grasp=held.grasp,
                                 obstacles=obstacles, tilt_deg=tilt, **raised)
        except MotionStopped:
            # The jaws open only at the release height, so empty jaws now mean it
            # was put down where it was meant to go.
            if _holding_now(env) is False:
                _put_down(scene_state, held, tgt, at, on)
                print(f"  [Stop] Stopped after letting go of {held.name} — the world model has it on {tgt.name}.")
            raise
        _judge(act, outcome, "place")
        # A place that needs the object held higher: its next pick grips it so.
        held.grasp_above_m = outcome.regrip_above_m
        if outcome.ok or outcome.released:
            _put_down(scene_state, held, tgt, at, on)
            if outcome.faces_deg is not None:         # which way its sides face now, as it was put down
                held.faces_deg = outcome.faces_deg
            if outcome.ok and _confirmed(env, scene_state, held, at, on) is False:
                where = f"on {on.name}" if on is not None else f"at {tgt.name}"
                why = f"{held.name} is not {where} as the camera sees it — it may have fallen or slid off"
                print(f"  [Check] {why}.")
                act.problem(f"place: {why}")
                skill_executor.report("place", False, why)
                outcome = dataclasses.replace(outcome, ok=False, message=why)
        elif _holding_now(env) is False:
            # Not put down, and not in the jaws either: dropped on the way.
            _lost(scene_state, held, f"it left the jaws on the way to {tgt.name}")
        if _scenario.placed:
            _scenario.placed(held, tgt, outcome)
    if (outcome.ok or outcome.released) and _ROOM["restore"] and not _ROOM["busy"]:
        _restore_room(env, scene_state)
    return outcome.ok


def _confirmed(env, scene_state, held, at, on) -> "bool | None":
    """Scenario.confirm_placed's word on a place; None without one, or when it could not look (an error is said)."""
    if _scenario.confirm_placed is None:
        return None
    try:
        return _scenario.confirm_placed(env, scene_state, held, np.asarray(at, float), on)
    except MotionStopped:
        raise
    except Exception as exc:          # noqa: BLE001 — a check that cannot run does not fail the place
        print(f"  [Check] Could not look at {held.name}: {exc}")
        return None


def _opening(scene_state: SceneState, name: str):
    """Scenario.opening's answer for name: (Opening, SceneObject), a str why not, or None (not a container)."""
    return _scenario.opening(scene_state, name) if _scenario.opening else None


def _place_in(env, scene_state, container, drop=False, **_):
    """
    The held object put into a container (Scenario.opening): let go inside it,
    over its walls. container: its name, or what the user calls it — one not
    known yet is looked for first (Scenario.find_opening).
    """
    held = scene_state.held_object()
    if held is None:
        return _refuse("place_in", "nothing is held — pick something first")
    got = _opening(scene_state, container)
    if got is None and _scenario.find_opening is not None and _scenario.find_opening(env, scene_state, container):
        got = _opening(scene_state, container)
    if got is None:
        return _refuse("place_in", f"'{container}' is not a container that can be found here")
    if isinstance(got, str):
        return _refuse("place_in", got)
    opening, tgt = got
    # Not kept clear of: the container itself (its walls are the opening's) and what is held.
    obstacles = [o for o in obstacles_around(scene_state, [held, tgt]) if o.label != tgt.name]
    print(f"  Putting {held.name} into {tgt.name}...")
    with flight_recorder.RECORDER.action("place_in", object=held.name, target=tgt.name,
                                         track=held.track_id, rim_m=round(float(opening.rim_m), 3),
                                         rim_measured=opening.rim_measured) as act:
        try:
            outcome = run_policy(policies().place_in, "place_in", env, opening=opening, held=target_of(held),
                                 grasp=held.grasp, obstacles=obstacles,
                                 drop=str(drop).strip().lower() in ("true", "1", "yes", "drop"))
        except MotionStopped:
            if _holding_now(env) is False:
                print(f"  [Stop] Stopped after letting go of {held.name} — the world model has it in {tgt.name}, "
                      "where exactly not known.")
                _put_down(scene_state, held, tgt, None)
            raise
        _judge(act, outcome, "place_in")
        if outcome.measured is not None and _scenario.opening_measured:
            _scenario.opening_measured(scene_state, tgt.name, outcome.measured.height_mm / 1000.0)
        held.grasp_above_m = outcome.regrip_above_m
        if outcome.ok or outcome.released:
            xy = outcome.at_xy if outcome.at_xy is not None else np.asarray(tgt.position, float)[:2]
            surface = float(tgt.position[2]) if tgt.position is not None else 0.0
            _put_down(scene_state, held, tgt, np.array([float(xy[0]), float(xy[1]), surface]))
            track = scene_state.memory.get(held.track_id) if scene_state.memory is not None and held.track_id else None
            said = _scenario.describe(track) if _scenario.describe and track is not None else held.description
            held.description = f"{said} — in {tgt.name}"
        elif _holding_now(env) is False:
            _lost(scene_state, held, f"it left the jaws on the way into {tgt.name}")
        if _scenario.placed:
            _scenario.placed(held, tgt, outcome)
    return outcome.ok


def install_pick_by_rim() -> None:
    """Register pick_by_rim (Policies.pick_by_rim), for a deployment whose arm's policies can."""
    def pick_by_rim(env, scene_state, object_name, **kw):
        obj = scene_state.find(object_name, among="objects")
        if obj is not None and obj.position is not None and _scenario.outline_again is not None:
            _scenario.outline_again(env, scene_state, obj)
        return _pick(env, scene_state, object_name, _by="pick_by_rim", **kw)

    register_action(
        "pick_by_rim",
        description=("Pick up a cup, bowl, scoop or other thin-walled open object by gripping one of its walls, a "
                     "jaw either side of it: for one too wide to close the jaws round, or one a plain pick could "
                     "not settle the height of."),
        parameters={"object_name": "Exact name of the object to pick up."},
        conditions=dict(ACTIONS["pick"].get("conditions") or {}),
    )(pick_by_rim)


def install_place_in() -> None:
    """
    Register place_in, for a deployment whose Scenario has opening (and whose
    arm's policies can: Policies.place_in). Its own call: an action the
    planner sees is one it may plan, so a deployment without containers does
    not offer it.
    """
    register_action(
        "place_in",
        description=("Put the held object INTO a container — a tub, a bowl, a box, a tray: over its walls and let "
                     "go inside it. 'Put it in the tub' is place_in. A container not listed is looked for by "
                     "what the user calls it."),
        parameters={"container": ("The container's name as listed among the placement targets; if it is not "
                                  "listed, what the user calls it ('plastic tub') — it is then looked for."),
                    "drop": ("true to let go from over its walls instead of lowering it in: when the user says "
                             "'drop' or 'throw' it in, for a sturdy object. Default false.")},
        conditions={"pre": ["something_carried"]},
    )(_place_in)


def _judge(act, outcome, what: str) -> None:
    """What an action came to, for the flight recorder, and whether it is an incident."""
    act.done(ok=outcome.ok, message=outcome.message, refused=outcome.refused, approach=outcome.approach,
             attempts=outcome.attempts, released=outcome.released)
    if outcome.refused:
        return                               # nothing moved: said, not an incident
    if not outcome.ok:
        act.problem(f"{what} failed: {outcome.message}")
    elif len(outcome.attempts or []) > 1:
        act.problem(f"{what} took {len(outcome.attempts)} attempts")


def _put_down(scene_state: SceneState, held, tgt, at=None, on=None) -> None:
    """held is down at `at` (else tgt's position): on the surface, or on top of `on`."""
    where = np.asarray(tgt.position if at is None else at, float)
    scene_state.update_position(held.name, where)
    held.grasp = None                    # put down; the next pick grips it afresh
    held.on = on.name if on is not None else None
    memory, track = _track(scene_state, held)
    if track is not None:
        if on is not None:
            # Its foot this far above the surface the stack stands on.
            bottom = (stack_below(scene_state, on) or [on])[-1]
            raised_mm = (float(where[2]) - float(bottom.position[2])) * 1000.0
            memory.placed(track.id, where[:2], None, on=on.track_id, raised_mm=raised_mm)
        else:
            memory.placed(track.id, where[:2], tgt.name)
        # Named for where it stands now, as the next scan would name it.
        held.name = memory.name(track)
        if _scenario.describe:
            held.description = f"{_scenario.describe(track)} on {tgt.name}"


def _move_above(env, scene_state, object_name, height=None, **_):
    obj = scene_state.find(object_name)
    if obj is None or obj.position is None:
        print(f"  [Error] '{object_name}' not in scene or position unknown. Try 'observe' first.")
        return False
    # Direct dispatch must enforce the same target bounds as a planned move.
    _, violations = safety.validate_plan(
        [{"action": "move_above", "object_name": object_name, "height": height}],
        scene_state, sequence_validators=[])
    if any(v.severity == "block" for v in violations):
        return False
    print(f"  Moving above {obj.name}...")
    return run_policy(policies().move_above, "move_above", env, target=target_of(obj),
                      height_m=None if height is None else float(height)).ok


def _home(env, **_):
    print("  Moving to home position...")
    return run_policy(policies().home, "home", env).ok


def _open_gripper(env, scene_state, **_):
    print("  Opening gripper...")
    held = scene_state.held_object()
    try:
        return run_policy(policies().open_gripper, "open_gripper", env).ok
    finally:
        # A partial opening or stop can release it too. None means held in
        # SceneState, so remove the unlocated object until a scan finds it.
        if held is not None and _holding_now(env) is False:
            _lost(scene_state, held, "the gripper was opened outside a placement")


def _lost(scene_state: SceneState, held, why: str) -> None:
    """The held object has left the jaws somewhere unknown: not held, not anywhere, until a scan finds it."""
    memory, track = _track(scene_state, held)
    if track is not None:
        memory.released(track.id, why)
    held.grasp = None
    scene_state.objects = [o for o in scene_state.objects if o is not held]


def _close_gripper(env, **_):
    print("  Closing gripper...")
    return run_policy(policies().close_gripper, "close_gripper", env).ok


# ── Pushing, and making room ───────────────────────────────────────────────────
# A pick refused because a neighbour leaves the jaws no room (Outcome.blocked_by)
# is not the end of it: something can be moved. Before anything moves, every
# way is weighed by the policy's own checks (Policies.pick_check, nudge with
# check_only) — the cheapest that works first:
#   1. push the object itself a little, to where its pick fits (nothing to put back)
#   2. lift the neighbour onto a spot nearby (Scenario.aside_spots), pick, and
#      put the neighbour back once the object is put down
#   3. push the neighbour clear, pick, and put it back the same way
# The user's idea, 2026-11-11: three missed closes had slid the f1 bishop into
# the e1 king, so close neither could be gripped — "it could even move the king
# out of the way slightly then put it back; it has full power to get the job done".

ROOM_PUSHES_M = (0.010, 0.015, 0.020, 0.025, 0.030)   # how far a push to make room may go, shortest first
ROOM_DIRECTIONS = 16                                  # the directions weighed, evenly round
ROOM_FULL_CHECKS = 10        # full pick checks (with the reach, the slow part) weighed per way at most
RESTORE_TOL_M = 0.003        # an object moved aside this close to where it came from is back already
RESTORE_MOVED_M = 0.015      # one this far from where the arm left it has been moved since: not put back

_ROOM: dict = {"busy": False, "restore": []}   # busy: making room (its own picks make none); restore:
                                               # the objects moved aside, to put back after the next place


def _nudge(env, scene_state, object_name, target_name=None, **_):
    obj = scene_state.find(object_name, among="objects")
    if obj is None or obj.position is None:
        print(f"  [Error] '{object_name}' not in scene or position unknown. Try 'observe' first.")
        return False
    if target_name:
        tgt = scene_state.find(target_name)
        if tgt is None or tgt.position is None:
            print(f"  [Error] '{target_name}' not in scene. Try 'observe' first.")
            return False
        return _push(env, scene_state, obj, np.asarray(tgt.position[:2], float)).ok
    home = _scenario.home_of(scene_state, obj, obj.position[:2]) if _scenario.home_of else None
    if home is None:
        print(f"  [Error] Where should {obj.name} go? Name the spot (target_name).")
        return False
    return _push(env, scene_state, obj, np.asarray(home[1], float)[:2]).ok


def _push(env, scene_state: SceneState, obj, to_xy):
    """Push obj's foot to to_xy (Policies.nudge), and the world model and memory follow it. The Outcome."""
    print(f"  Nudging {obj.name}...")
    with flight_recorder.RECORDER.action("nudge", object=obj.name, at=obj.position[:2], to=to_xy,
                                         track=obj.track_id) as act:
        outcome = run_policy(lambda e, **k: policies().nudge(e, **k), "nudge", env, target=target_of(obj),
                             to_xy=np.asarray(to_xy, float), obstacles=obstacles_around(scene_state, obj))
        _judge(act, outcome, "nudge")
        if outcome.at_xy is not None and not outcome.refused:
            obj.position = np.array([outcome.at_xy[0], outcome.at_xy[1], obj.position[2]], float)
            memory, track = _track(scene_state, obj)
            if track is not None:
                memory.moved(track.id, outcome.at_xy, how="nudge")
    return outcome


def _xy_of(obj, fallback) -> "np.ndarray | None":
    """Where obj stands now by the world model (after a place or push), else fallback; surface xy."""
    at = obj.position if obj.position is not None else fallback
    return None if at is None else np.asarray(at, float)[:2].copy()


def _moved(obstacles: list, label: str, xy=None) -> list:
    """obstacles with the one labelled label standing at xy instead — or gone, xy None."""
    out = []
    for ob in obstacles:
        if ob.label != label:
            out.append(ob)
        elif xy is not None:
            out.append(dataclasses.replace(ob, position=np.array([xy[0], xy[1], *np.asarray(ob.position)[2:3]])))
    return out


def _directions(away: np.ndarray) -> list:
    """ROOM_DIRECTIONS unit vectors round the surface, those nearest `away` first."""
    a = np.radians(np.arange(ROOM_DIRECTIONS) * 360.0 / ROOM_DIRECTIONS)
    dirs = [np.array([np.cos(t), np.sin(t)]) for t in a]
    n = float(np.linalg.norm(away))
    return sorted(dirs, key=lambda d: -float(d @ away) / n) if n > 0 else dirs


def _plan_room(env, scene_state: SceneState, obj, blockers: list) -> "tuple | None":
    """
    The first way to make room for obj's pick that the checks pass, as
    (kind, what moves, where to, said); None when there is none. Kinds:
    "push_self" (to an xy), "aside" (to a place target), "push_other" (to an xy).
    """
    pol = policies()
    tgt = target_of(obj)
    here = np.asarray(obj.position[:2], float)
    around = obstacles_around(scene_state, obj, quiet=True)
    full = {"left": ROOM_FULL_CHECKS}
    # Where obj goes once picked: nothing is set down there, or pushed toward it.
    going = getattr(scene_state, "headed_for", None)
    going_to = next((t for t in scene_state.targets if t.name == going), None) if going else None
    going_xy = (np.asarray(going_to.position[:2], float)
                if going_to is not None and going_to.position is not None else None)

    def fits(target, obstacles) -> bool:
        if not pol.pick_check(env, target, obstacles, _vision, quick=True).ok or full["left"] <= 0:
            return False
        full["left"] -= 1
        return pol.pick_check(env, target, obstacles, _vision).ok

    # 1. The object itself, a little, away from what crowds it.
    away = sum((here - np.asarray(b.position[:2], float) for b in blockers), np.zeros(2))
    for dist in ROOM_PUSHES_M:
        for d in _directions(away):
            to = here + d * dist
            moved = dataclasses.replace(tgt, position=np.array([to[0], to[1], tgt.position[2]]))
            if not pol.pick_check(env, moved, around, _vision, quick=True).ok:
                continue
            if not pol.nudge(env, tgt, to, around, check_only=True).ok:
                continue
            if fits(moved, around):
                return "push_self", obj, to, f"pushing {obj.name} {dist * 1000:.0f} mm, where the jaws fit round it"
    for b in blockers:
        b_here = np.asarray(b.position[:2], float)
        b_around = obstacles_around(scene_state, b, quiet=True)
        # 2. The neighbour lifted onto a spot nearby for a moment.
        spots = _scenario.aside_spots(scene_state, b, obj, going) if _scenario.aside_spots else []
        spots = [t for t in spots if t.name != going and t.position is not None]
        if spots and pol.pick_check(env, target_of(b), b_around, _vision).ok:
            full["left"] = ROOM_FULL_CHECKS
            if fits(tgt, _moved(around, b.name, spots[0].position[:2])):
                return "aside", b, spots[0], f"lifting {b.name} onto {spots[0].name} for a moment"
        # 3. The neighbour pushed clear.
        full["left"] = ROOM_FULL_CHECKS
        for dist in ROOM_PUSHES_M:
            for d in _directions(b_here - here):
                to = b_here + d * dist
                if going_xy is not None and np.linalg.norm(to - going_xy) < np.linalg.norm(b_here - going_xy):
                    continue
                if not pol.pick_check(env, tgt, _moved(around, b.name, to), _vision, quick=True).ok:
                    continue
                if not pol.nudge(env, target_of(b), to, b_around, check_only=True).ok:
                    continue
                if fits(tgt, _moved(around, b.name, to)):
                    return "push_other", b, to, f"pushing {b.name} {dist * 1000:.0f} mm out of the way"
    return None


def _make_room(env, scene_state: SceneState, obj, refused) -> "bool | None":
    """
    obj's pick was refused for room (refused.blocked_by): make room the first
    way the checks pass (see above), then pick it. Whether that pick worked;
    None when no way was found, or making room failed before the pick — the
    refusal stands.
    """
    blockers = [o for o in scene_state.objects
                if o.name in refused.blocked_by and o is not obj and o.position is not None]
    print(f"  [Room] {refused.message} — looking for a way to make room.")
    _ROOM["busy"] = True
    try:
        plan = _plan_room(env, scene_state, obj, blockers)
        if plan is None:
            others = [n for n in refused.blocked_by if n not in {b.name for b in blockers}]
            print("  [Room] No way to make room was found" +
                  (f" ({', '.join(others)} is not something the arm can move)" if others else "") + ".")
            return None
        kind, what, to, said = plan
        print(f"  [Room] Making room: {said}.")
        back = {"name": what.name, "track": what.track_id, "home_xy": np.asarray(what.position, float).copy()}
        if kind == "push_self":
            if not _push(env, scene_state, obj, to).ok:
                return None
        elif kind == "aside":
            if not _pick(env, scene_state, what.name):
                return None
            if not _place(env, scene_state, to.name):
                if _holding_now(env):
                    _place(env, scene_state, _home_target(scene_state, what, back["home_xy"], back["name"]).name)
                return None
            _ROOM["restore"].append({**back, "left_xy": _xy_of(what, to.position)})
        else:
            pushed = _push(env, scene_state, what, to)
            if not pushed.refused:
                _ROOM["restore"].append({**back, "left_xy": _xy_of(what, pushed.at_xy)})
            if not pushed.ok:
                return None
        ok = _pick(env, scene_state, obj.name)
    finally:
        _ROOM["busy"] = False
    if not ok and _ROOM["restore"]:
        _restore_room(env, scene_state)
    return ok


def _restore_room(env, scene_state: SceneState) -> None:
    """Every object moved aside to make room, back where it came from: picked and put down there, else pushed."""
    _ROOM["busy"] = True
    try:
        while _ROOM["restore"]:
            held = scene_state.held_object()
            if held is not None or _holding_now(env):
                # A pick now would open the jaws on it (a put-back that failed
                # still holding, a pick that failed with it gripped): the rest
                # wait until the jaws are empty.
                print(f"  [Room] {'Holding ' + held.name if held is not None else 'The gripper holds something'}"
                      " — not putting the others back until it is down.")
                break
            r = _ROOM["restore"].pop(0)
            obj = next((o for o in scene_state.objects if r["track"] is not None and o.track_id == r["track"]),
                       None) or scene_state.find(r["name"], among="objects")
            if obj is None or obj.position is None:
                print(f"  [Room] {r['name']} was moved aside, but is not in the scene now — not put back.")
                continue
            now = np.asarray(obj.position[:2], float)
            if float(np.linalg.norm(now - r["home_xy"][:2])) <= RESTORE_TOL_M:
                continue
            left = r.get("left_xy")
            if left is not None and float(np.linalg.norm(now - left)) > RESTORE_MOVED_M:
                # Not where the arm left it: someone has moved it since (in a
                # game, the person's move), and putting it back would undo that.
                print(f"  [Room] {obj.name} has been moved since it was set aside — not putting it back.")
                continue
            print(f"  [Room] Putting {obj.name} back.")
            # Lifted first: where it belongs is a place to put it only once it
            # is off it (for chess, its square becomes a target again then).
            if _pick(env, scene_state, obj.name):
                held = scene_state.held_object()
                if held is not None and _place(env, scene_state,
                                               _home_target(scene_state, held, r["home_xy"], r["name"]).name):
                    continue
            elif not _holding_now(env):
                spot = _home_target(scene_state, obj, r["home_xy"], r["name"], add=False)
                home_xy = None if spot.position is None else np.asarray(spot.position, float)[:2]
                if home_xy is not None and policies().nudge(
                        env, target_of(obj), home_xy, obstacles_around(scene_state, obj, quiet=True),
                        check_only=True).ok and _push(env, scene_state, obj, home_xy).ok:
                    continue
            print(f"  [Room] Could not put {obj.name} back where it stood.")
    finally:
        _ROOM["busy"] = False


def _home_target(scene_state: SceneState, obj, xy, was: "str | None" = None, add: bool = True) -> SceneObject:
    """
    The place target obj goes back to from having been moved aside: where it
    belongs by the scenario (Scenario.home_of), else where it stood (xy),
    named for what it was called there (was). A place target in the scene,
    added if it is not (add False: only made). By its exact name: the scene's
    find() matches loosely, and "where_object_spare_was" is not "spare".
    """
    xy = np.asarray(xy, float)
    home = _scenario.home_of(scene_state, obj, xy[:2]) if _scenario.home_of else None
    name, at = (home[0], np.asarray(home[1], float)[:2]) if home else (f"where_{was or obj.name}_was", xy[:2])
    found = next((t for t in scene_state.targets if t.name == name), None) if add else None
    if found is not None:
        return found
    z = float(xy[2]) if len(xy) > 2 else 0.0
    tgt = SceneObject(name, f"where {obj.name} stood", interactable=False, position=np.array([at[0], at[1], z]))
    if add:
        scene_state.targets.append(tgt)
    return tgt


def object_action(name: str, policy: Callable, *, description: str, parameters: "dict | None" = None,
                  conditions: "dict | None" = None, moves_object: bool = False, after: "Callable | None" = None,
                  finds_missing: bool = False):
    """
    Give the planner a verb of your own that acts on one named object — push
    it, tip it upright, wipe round it — with the bookkeeping the standard
    actions do: the name found in the world model and turned into geometry,
    the policy run under the safety monitor, and what it did written back.

        def push(pol, env, target, obstacles, distance_m=0.03, **_):
            return pol.push(env, target, float(distance_m), obstacles=obstacles)   # -> Outcome

        manipulation.object_action("push", push, description="Push an object away from the arm.",
                                   parameters={"distance_m": "How far, metres."}, moves_object=True)

    policy(policies, env, target, obstacles, **parameters) -> Outcome, where
    policies is the installed library (install), target the named object as a
    policies.Target and obstacles what stands around it. The planner names the
    object as object_name; `parameters` are the verb's others, as the planner
    gives them (strings or numbers: convert in the policy). moves_object: the
    Outcome's at_xy is where the object stands afterwards, and the world model
    and the object memory follow it. By default the object must exist and not
    be in the gripper; `conditions` replaces that ({"pre": [...], "post": [...]}).
    after(env, scene_state, obj, outcome): what the deployment does once the
    verb has run, unless it refused before moving — a learned one
    (mrcf/policies/learned.py) can leave the object anywhere, which only a fresh
    look can say. finds_missing: the object may be one the last scan missed —
    named as the memory names it, or by the target it was last seen on
    (missing_object) — as a piece lying on its side often is.
    """
    def action(env, scene_state, object_name, **kwargs):
        obj = scene_state.find(object_name, among="objects")
        if obj is None and finds_missing:
            obj = missing_object(scene_state, object_name)
        if obj is None or obj.position is None:
            print(f"  [Error] '{object_name}' not in scene or position unknown. Try 'observe' first.")
            return False
        print(f"  {name}: {obj.name}...")
        # Recorded like pick and place, so a verb that misses leaves an
        # incident, and the policy's views (a wrist view at the grip) are kept
        # on the timeline rather than dropped with the Outcome.
        with flight_recorder.RECORDER.action(name, object=obj.name, at=obj.position[:2],
                                             track=obj.track_id) as act:
            outcome = run_policy(lambda e, **k: policy(policies(), e, **k), name, env,
                                 target=target_of(obj), obstacles=obstacles_around(scene_state, obj), **kwargs)
            for label, frame in outcome.views or ():
                flight_recorder.event("policy", "view", image=frame, label=label)
            _judge(act, outcome, name)
            if moves_object and outcome.at_xy is not None and (outcome.ok or outcome.released):
                obj.position = np.array([outcome.at_xy[0], outcome.at_xy[1], obj.position[2]], float)
                memory, track = _track(scene_state, obj)
                if track is not None:
                    memory.moved(track.id, outcome.at_xy, how=name)
            # Inside the action, so an incident shows where a fresh look found it.
            if after is not None and not outcome.refused:
                after(env, scene_state, obj, outcome)
        return outcome.ok

    register_action(
        name, description=description,
        parameters={"object_name": "Exact name of the object.", **(parameters or {})},
        conditions=conditions if conditions is not None else
        ({"pre": []} if finds_missing else {"pre": ["object_exists", "object_not_carried"]}),
    )(action)
    return action


# A target names where a missing object was last seen when it was this close.
_MISSING_NEAR_M = 0.03


def missing_object(scene_state: SceneState, name: str) -> "SceneObject | None":
    """
    An object the memory has as missing — the last scan did not find it — as
    one in the scene again, where it was last seen: named as the memory names
    it, or by the target (a chess square) it was last seen on. None when there
    is none. 2026-11-03: the f1 bishop fell over as it was put on c4, and
    lying cream on a light square the scan no longer outlined it; standing it
    up needs it by name.
    """
    memory = scene_state.memory
    if memory is None:
        return None
    spot = scene_state.find(name, among="targets")
    near = None if spot is None or spot.position is None else np.asarray(spot.position[:2], float)
    best, best_d = None, None
    for t in memory.in_state("missing"):
        if t.position is None:
            continue
        d = 0.0 if memory.name(t) == name else (
            float(np.linalg.norm(np.asarray(t.position[:2], float) - near)) if near is not None else None)
        if d is not None and d <= _MISSING_NEAR_M and (best_d is None or d < best_d):
            best, best_d = t, d
    if best is None:
        return None
    where = np.asarray(best.position, float)
    z = float(spot.position[2]) if spot is not None and spot.position is not None else (
        float(where[2]) if len(where) > 2 else 0.0)
    obj = SceneObject(memory.name(best), f"{memory.name(best)} (missed by the last scan; last seen there)",
                      position=np.array([where[0], where[1], z]))
    obj.track_id = best.id
    obj.geometry = {"base_diameter_m": best.foot_mm / 1000.0} if best.foot_mm else None
    obj.confirmed = bool(getattr(best, "confirmed", True))
    scene_state.objects.append(obj)
    print(f"  [Scene] '{name}': {obj.name}, which the last scan missed — where it was last seen.")
    return obj


def install(policy_library: Policies, vision: "Vision | None" = None,
            scenario: "Scenario | None" = None) -> None:
    """Use this policy library for the standard actions, and register them."""
    global _policies, _vision, _scenario
    _policies = policy_library
    _vision = vision or Vision()
    _scenario = scenario or Scenario()

    register_action(
        "pick",
        description="Pick up an object from its last observed position.",
        parameters={"object_name": "Exact name of the object to pick up."},
        # nothing_carried matters as much as the other two: a pick opens the
        # jaws as its first action, so picking while already holding drops the
        # held object. object_not_carried only rules out re-picking the SAME one.
        conditions={"pre": ["object_exists", "object_not_carried", "nothing_carried"],
                    "post": ["something_carried"]},
    )(_pick)
    register_action(
        "place",
        description="Place the currently held object at a named target.",
        parameters={"target_name": "Exact name of the destination."},
        conditions={"pre": ["something_carried", "target_exists"]},
    )(_place)
    if _scenario.carries is not None:
        register_action(
            "place_on",
            description=("Put the held object down on top of another object, stacking it (a block on a block): "
                         "on the top of that object's stack if something already stands on it."),
            parameters={"object_name": "Exact name of the object to put it on."},
            conditions={"pre": ["something_carried", "object_exists", "object_not_carried"]},
        )(_place_on)
    else:
        ACTIONS.pop("place_on", None)        # a task that stacks nothing: not offered to its planner
    register_action(
        "move_above",
        description="Move the end-effector above an object without grasping it.",
        parameters={
            "object_name": "Exact name of the object to hover over.",
            "height":      "Height in metres above the table surface.",
        },
    )(_move_above)
    register_action(
        "home",
        description=(
            "Move the arm to its safe home position, raised clear of the table. "
            "Use this to reset after an error or before starting a new task."
        ),
        # Offered even when nothing is in view: a reset is wanted most exactly
        # when a scan has come back empty.
        always=True,
    )(_home)
    register_action(
        "nudge",
        description=(
            "Push an object a short way across the surface with the closed jaws, without picking it up: "
            "onto a named spot, or, with none named, back to the middle of where it belongs (a chess "
            "piece knocked a little off its square)."
        ),
        parameters={"object_name": "Exact name of the object to push.",
                    "target_name": "Where to push it to (optional: back where it belongs)."},
        conditions={"pre": ["object_exists", "object_not_carried", "nothing_carried"]},
    )(_nudge)
    register_action("open_gripper", description="Open the robot gripper fully.", always=True)(_open_gripper)
    register_action("close_gripper", description="Close the robot gripper.", always=True)(_close_gripper)


# ── Plan-time rule ─────────────────────────────────────────────────────────────

def gripper_sequence_validator(actions, scene_state):
    """
    Check the plan's gripper sequence: a 'place' with nothing held, and a
    'pick' while something already is. For safety.configure(), which keeps the
    framework's safety layer robot-agnostic.

    The second case is the damaging one — a pick opens the jaws before it
    approaches, so a pick planned while carrying drops the carried object
    somewhere arbitrary.
    """
    from mrcf.safety import PlanViolation

    violations = []
    simulated_holding = scene_state.held_object() is not None

    for action in actions:
        skill = action.get("action", "")
        if skill in ("place", "place_on", "place_in") and not simulated_holding:
            violations.append(PlanViolation(
                action, f"'{skill}' planned but nothing is held — missing 'pick' action", "warn",
            ))
        if skill in ("pick", "pick_by_rim"):
            if simulated_holding:
                violations.append(PlanViolation(
                    action,
                    f"'{skill}' planned while already holding an object — the gripper "
                    "opens before approaching, which would drop it. Place it first.",
                    "block",
                ))
            simulated_holding = True
        # Each lets go of what is held. 2026-11-18: place_in was not among them, and
        # "put the four things into the tub" — pick, place_in, pick, ... — was blocked
        # at every pick after the first, and planned again into the same block.
        if skill in ("place", "place_on", "place_beside", "place_in", "open_gripper"):
            simulated_holding = False

    return violations


def stack_sequence_validator(actions, scene_state):
    """
    Check the plan against what stands on what, as each step leaves it: a
    'pick' of an object something stands on — now, or once an earlier step
    has put it there — would be refused (_pick), so the plan is not run at
    all, and the planner hears why before anything moves. 2026-11-13, the
    UR5: "stack the three blocks" was planned blue onto green, then green
    onto red. The arm put blue on green, the pick of green was refused, and
    the plan made again from there took blue off and put it back.
    """
    from mrcf.safety import PlanViolation

    on = {o.name: o.on for o in scene_state.objects}
    held = scene_state.held_object()
    holding = held.name if held is not None else None
    if holding is not None:
        on[holding] = None
    put_by: dict = {}                              # what a step stacked, by which step
    violations = []

    def above(name):
        return [n for n, under in on.items() if under == name and n != holding]

    def top(name):
        seen = {name}
        while (up := above(name)) and up[0] not in seen:
            name = up[0]
            seen.add(name)
        return name

    for step, action in enumerate(actions, 1):
        skill = action.get("action", "")
        params = action.get("parameters") if isinstance(action.get("parameters"), dict) else action
        name = params.get("object_name") or params.get("target_name")
        if skill == "pick" and name in on:
            if up := above(name):
                when = f"step {put_by[up[0]]} puts it there" if up[0] in put_by else "it is there now"
                violations.append(PlanViolation(
                    action, f"step {step} picks {name}, but {up[0]} is on top of it ({when}): it cannot be picked "
                            "until that is moved off. A tower is built from the bottom up: the block that stays at "
                            "the bottom is never picked", "block"))
            holding, on[name] = name, None
        elif skill in ("place_on", "place") and holding is not None and name in on and name != holding:
            on[holding], put_by[holding], holding = top(name), step, None
        elif skill in ("place", "place_beside", "open_gripper") and holding is not None:
            on[holding], holding = None, None
    return violations
