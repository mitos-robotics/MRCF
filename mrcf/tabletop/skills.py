"""
skills.py — the table task's actions, for the deployment it is bound to
(tabletop.use(config): the arm, its policies, its kept state are that
deployment's).

Picking and placing are the framework's standard actions (manipulation.py),
driving the arm's own policy library; install_manipulation() plugs this task
into them: how an object is described, how tall an unmeasured one could be,
and that the spot a pick emptied is somewhere to put it back. What is added
here:

  - scan_scene: the framework's surface scan (scene_scan.observe), each object
    named by the vision model (naming.py), the marked spots as places;
  - place_beside: put the held object down on free table beside another;
  - look_for / find: a thing the scan does not list, or a container to put
    things into, found by what it is called;
  - mark_place / forget_place: name a spot on the table;
  - identify: the user says what an object is (and how tall, if they like);
  - freedrive: the arm free to be pushed by hand for a while (to show a spot);
  - learn_jaw_view, measure_surface (an arm that touches the surface);
  - observe, forget, list_objects, list_actions, snapshot, status, answer.

Registered as a side effect of importing this module, which a deployment's
skills.py does after tabletop.use(config); each returns True or False for
whether it worked.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from mrcf import flight_recorder, manipulation
from mrcf.capabilities import GripperFits, OrientationSearch
from mrcf.policies.base import Vision
from mrcf.policies.parallel_jaw import ParallelJaw
from mrcf.skill_executor import report
from mrcf.robot_env import RobotEnv
from mrcf.vlm import ACTIONS, SceneObject, call_vlm, register_action, render_to_base64

from . import cfg
from . import naming, places

POLICIES = cfg.ARM.make_policies(profile=cfg.PROFILE, **cfg.POLICY_OVERRIDES)
DEFAULT_RADIUS_M = 0.02         # an object whose foot the scan has not measured
_MEMORY = None


def memory():
    """The deployment's object memory, made on first use: empty, unless REMEMBER_ACROSS_RESTARTS."""
    global _MEMORY
    if _MEMORY is None:
        from mrcf.object_memory import ObjectMemory
        _MEMORY = ObjectMemory(cfg.MEMORY_PATH, namer=naming.object_name, load=cfg.REMEMBER_ACROSS_RESTARTS)
        if not cfg.REMEMBER_ACROSS_RESTARTS:
            print("  [Memory] Starting fresh: the first scan says what is on the table.")
    return _MEMORY


# ── Looking at the table ──────────────────────────────────────────────────────

def arm_hides(env: RobotEnv, around=None, radius_m: float = 0.05) -> "tuple[float | None, bool]":
    """How much of the table the arm hides from the overhead camera, and whether round `around` (scene_scan.arm_hides)."""
    from mrcf import scene_scan

    return scene_scan.arm_hides(env, cfg.OBSERVE_CAMERA, cfg.CAPTURE_W_OVERHEAD, cfg.CAPTURE_H_OVERHEAD,
                                cfg.WORKSPACE_BOUNDS, around, radius_m)


def goto_observe_pose(env: RobotEnv, around=None) -> bool:
    """
    The overhead camera's view of the table clear enough for a scan: where the
    arm is, if it hides little of the table there (cfg.SCAN_PAST_ARM, at most
    cfg.ARM_HIDES_AT_MOST of it, and none round `around` when given) — the scan
    looks past it (scene_scan.observe(past_arm=)) — else the arm parked out of
    view. True: no need to move it back after (parked: False is not that; see
    the return of env.go_park()).
    """
    if getattr(cfg, "SCAN_PAST_ARM", False):
        share, spot = arm_hides(env, around)
        if share is not None and share <= cfg.ARM_HIDES_AT_MOST and not spot:
            print(f"  [Observe] Looking past the arm where it is (it hides {share * 100:.0f}% of the table) — "
                  "not parking it.")
            return True
        # From home, before the park pose: home hides little of the table, and is
        # where the arm goes after a look anyway; the park pose folds it up and
        # back, and the next move starts from there. 2026-11-18: a relocate after
        # a miss parked the arm, scanned, and went home — two long moves for one look.
        if env.go_home() is not False:
            share, spot = arm_hides(env, around)
            if share is not None and share <= cfg.ARM_HIDES_AT_MOST and not spot:
                print(f"  [Observe] Looking past the arm from home (it hides {share * 100:.0f}% of the table) — "
                      "not parking it.")
                return True
    reached = env.go_park()
    if reached is False:
        print("  [Observe] Did not reach the park pose — the arm may hide part of the table from the camera.")
    return reached is not False


def _in_bounds(xy) -> bool:
    b = cfg.WORKSPACE_BOUNDS
    return b.x_min <= xy[0] <= b.x_max and b.y_min <= xy[1] <= b.y_max


def _reachable(env):
    """
    An object the arm can pick at (x, y): inside WORKSPACE_BOUNDS, and with the
    gripper straight down over it inside its box (RigEnv.gripper_fits) — the
    box is the whole gripper's, so an object at its edge is out of reach.
    2026-11-13: a white thing at x 0.003 by the base was planned into a tower
    over and over, and every pick of it refused by the box.
    """
    fits = env.gripper_fits if isinstance(env, GripperFits) else None
    lift = cfg.TABLE_Z + 0.03                      # the gripper over a block, at the height it grips it

    def reachable(xy) -> bool:
        return _in_bounds(xy) and (fits is None or bool(fits((float(xy[0]), float(xy[1]), lift))))

    return reachable


def scan_scene(env: RobotEnv, scene_state) -> bool:
    """
    The whole look: the surface scan finds and measures what stands on the
    table, the memory says which object each is, the vision model names any
    new one, and the marked spots are the places to put things. What the model
    says is not an object (a mark, a shadow) is left out. False when the
    surface cannot be scanned (no calibrated overhead camera, no segmenter).
    """
    from mrcf import scene_scan
    from mrcf.perception import surface_scan

    size = (cfg.CAPTURE_W_OVERHEAD, cfg.CAPTURE_H_OVERHEAD)
    reference = surface_scan.load_reference(cfg.SURFACE_REFERENCE_PATH, size)
    b, mem = cfg.WORKSPACE_BOUNDS, memory()
    try:
        ok = scene_scan.observe(env, scene_state, mem, cfg.OBSERVE_CAMERA, *size, (b.x_min, b.x_max),
                                (b.y_min, b.y_max), reference=reference, describe=describe,
                                reachable=_reachable(env), identify=_identify(mem, env), table_z=cfg.TABLE_Z,
                                boxed=boxed, max_drift_px=cfg.CAMERA_DRIFT_PX,
                                past_arm=bool(getattr(cfg, "SCAN_PAST_ARM", False)))
    except surface_scan.CameraMoved as moved:
        # Nothing it saw before can be trusted either: no objects, so no plan acts on them.
        print(f"  [Scan] STOPPED: {moved}")
        scene_state.objects = []
        return False
    if not ok:
        from mrcf.perception import segment
        if env.camera_model(cfg.OBSERVE_CAMERA, *size) is None:
            why = ("the overhead camera is not calibrated (tools/calibrate_camera_by_hand.py, or "
                   "tools/calibrate_overhead_camera.py with a printed board)")
        elif not segment.available():
            why = f"the segmenter cannot run: {segment.why_unavailable()}"
        else:
            why = "the scan found no way to look (no camera model, no segmenter)"
        print(f"  [Scan] The table could not be scanned: {why}.")
        return False
    scene_state.objects = [o for o in scene_state.objects if not _not_an_object(mem, o)]
    held = scene_state.held_object()
    kept = [t for t in scene_state.targets if held is not None and t.name == _where_was(held.name)]
    scene_state.targets = places.targets(places.load(cfg.PLACES_PATH), cfg.TABLE_Z) + kept
    if containers_on():
        refresh_containers(env, scene_state)
    return True


def sizes_by_name() -> dict:
    """
    config.KNOWN_SIZES_MM where the arm cannot find out how tall an object is
    for itself; {} on one that can (GripperSpec.height_from_views): it looks at
    each one, and a size by name is a guess. 2026-11-16, the SO-101: the vision
    model called the black speaker, 55 mm across, a "black block", and the UR5's
    letter-block size made it a 30 mm cube — measured, so never looked at; set
    where a 30 mm cube's top face would put it, so the jaws came down beside it;
    and not refused as wider than the 45 mm the jaws open.
    """
    if isinstance(POLICIES, ParallelJaw) and POLICIES.spec.height_from_views:
        return {}
    return dict(cfg.KNOWN_SIZES_MM)


def jaws_open_mm() -> "float | None":
    """The widest the arm's gripper opens, mm (None: the policy library does not say)."""
    return POLICIES.spec.max_open_m * 1000.0 if isinstance(POLICIES, ParallelJaw) else None


def lying_along(track) -> "tuple[float, float] | None":
    """
    (length, narrowest across) mm of a thing the scan saw lying along the
    table — at least LONG_MIN_ASPECT times as long as it is widest across
    (scene_scan: track.extra length_mm, widest_mm, narrowest_mm) — when
    ACROSS_FOR_LONG; None otherwise.
    """
    if not getattr(cfg, "ACROSS_FOR_LONG", False):
        return None
    extra = getattr(track, "extra", None) or {}
    length, widest, narrowest = extra.get("length_mm"), extra.get("widest_mm"), extra.get("narrowest_mm")
    if not (length and widest and narrowest) or length < float(cfg.LONG_MIN_ASPECT) * widest:
        return None
    return float(length), float(narrowest)


def describe(track) -> str:
    """
    What the planner reads about it (naming.describe), and that the gripper
    cannot close round it when it is wider across, as the scan saw it, than the
    jaws open: 2026-11-16, asked to put "all the objects" in the bowl, the plan
    sent the SO-101 for a 55 mm speaker and a 49 mm plug with jaws that open 45.
    """
    foot: "float | None"
    lying = lying_along(track)
    if lying is not None:
        # A thing lying along the table (a tool) by how long it is and how narrow across, where the jaws
        # close on it (ACROSS_FOR_LONG): 2026-11-20, the UR5's screwdriver, "73 mm across" by its round
        # foot, was "wider across than the gripper opens (52 mm)", and the planner picked nothing.
        length, narrowest = lying
        text = naming.describe(track, size=f"lying along the table, {length:.0f} mm long, "
                                            f"{narrowest:.0f} mm across at its narrowest")
        widest, foot = jaws_open_mm(), narrowest
    else:
        text = naming.describe(track)
        widest, foot = jaws_open_mm(), (track.feet_mm[-1] if track.feet_mm else None)
    if widest and foot and foot >= widest and track.label and track.label != naming.NOT_AN_OBJECT:
        text += f" — wider across than the gripper opens ({widest:.0f} mm), so it likely cannot be picked up"
    return text


def known_sizes(mem) -> None:
    """
    The size config.KNOWN_SIZES_MM knows for what an object is called (sizes_by_name),
    recorded for each one on the table or held that has no size of its own,
    or one doubted since (moved by hand, renamed): a stack needs its blocks'
    heights, and a camera looking straight down cannot see them. Not over a
    height the user gave (identify).
    """
    for track in mem.in_state("present") + mem.in_state("held"):
        if "height_mm" in track.pinned:
            continue                                  # the user said how tall it is
        words = (track.label or "").lower().split()
        for kind, (height, width) in sizes_by_name().items():
            if kind in words:
                if not track.shape_current() or abs((track.height_mm or 0.0) - height) > 1.0:
                    mem.measured(track.id, height_mm=height,
                                 widths_mm={z: width for z in range(4, int(height), 4)})
                break


def boxed(track) -> "tuple[float, float] | None":
    """The size of a box the object is, (height, side) mm: one of sizes_by_name's kinds (they are boxes); else None."""
    words = (track.label or "").lower().split()
    return next(((float(size[0]), float(size[1])) for kind, size in sizes_by_name().items() if kind in words), None)


def _identify(mem, env=None):
    """
    scene_scan.observe's identify: each object's look from above kept
    (its catalogue signature), those of a kind the catalogue has learned
    named by it — the vision model not asked — then the vision model names
    the rest, and the known sizes go on.
    """
    name = naming.identifier(mem)

    def identify(frame, found, tracks) -> None:
        cat = catalogue()
        if cat is not None and env is not None:
            _match_kinds(cat, env, frame, found, tracks, mem)
        name(frame, found, tracks)
        known_sizes(mem)
    return identify


# ── The catalogue: the kinds of object this deployment works with ─────────────

_CATALOGUE: list = [None]


def catalogue():
    """This deployment's catalogue of kinds (mrcf/catalogue.py; cfg.CATALOGUE), made once; None when off."""
    if not getattr(cfg, "CATALOGUE", False):
        return None
    if _CATALOGUE[0] is None:
        from mrcf.catalogue import Catalogue
        _CATALOGUE[0] = Catalogue(cfg.CATALOGUE_PATH)
    return _CATALOGUE[0]


def _match_kinds(cat, env, frame, found, tracks, mem) -> None:
    """Each sighting's signature kept on its track; one not named yet, of a kind learned, named as that kind."""
    from mrcf import catalogue as catalogue_mod
    from mrcf.perception.plane_geometry import PlaneView

    h, w = np.asarray(frame).shape[:2]
    model = env.camera_model(cfg.OBSERVE_CAMERA, w, h)
    if model is None:
        return
    view = PlaneView.from_camera_model(model)
    guessed = mem.guessed()
    for f, track in zip(found, tracks):
        try:
            sig = catalogue_mod.signature(f["mask"], view, 2 * float(f["measurement"].base_radius_mm),
                                          f.get("appearance"))
        except Exception:                                  # noqa: BLE001 — no signature: matched by nothing
            continue
        track.extra = {**(track.extra or {}), "sig": sig}
        if track.label and track.id not in guessed and (track.extra or {}).get("kind"):
            continue
        got = cat.match(sig)
        if got is None:
            continue
        kind, pose, _ = got
        track.extra = {**track.extra, "kind": kind.id, "pose": pose.name}
        if track.label != kind.name and track.label_source != "user":
            mem.set_label(track.id, kind.name, "catalogue", attributes={"colour": kind.colour} if kind.colour else None)
        print(f"  [Catalogue] {naming.object_name(track)}: a {kind.colour + ' ' if kind.colour else ''}{kind.name}, "
              f"{pose.name} (learned {kind.learned})" + (" — picked by the grip that held it before" if pose.best() else "")
              + ".")


def _kind_of(mem, obj):
    """(catalogue, track, kind, pose) for obj: matched at the scan, else now by its signature; Nones where not."""
    cat = catalogue()
    track = mem.get(obj.track_id) if (cat is not None and mem is not None and obj.track_id) else None
    if track is None:
        return cat, None, None, None
    extra = track.extra or {}
    kind = cat.get(extra.get("kind"))
    pose = kind.pose(extra.get("pose")) if kind is not None else None
    if kind is None and extra.get("sig"):
        got = cat.match(extra["sig"])          # a kind learned since this scan (its twin, picked just now)
        if got is not None:
            kind, pose, _ = got
            track.extra = {**extra, "kind": kind.id, "pose": pose.name}
    if kind is None and track.label:
        # Not by its outline, but by what it is called and its colour: the same object resting another way
        # (2026-11-19, the plug on its back, pins up — its outline 42 mm across, not the 54 it was learned at).
        kind = cat.named(track.label, (track.attributes or {}).get("colour"))
        pose = None
        if kind is not None:
            track.extra = {**extra, "kind": kind.id, "pose": None}
    return cat, track, kind, pose


RECIPE_TOP_MARGIN_M = 0.010


def _recipe(scene_state, obj) -> "dict | None":
    """Scenario.recipe: the grip that held obj's kind, as it rests, before (catalogue.Pose.best)."""
    _, _, kind, pose = _kind_of(scene_state.memory, obj)
    if kind is not None and pose is None:
        return _any_way_up(kind)
    best = pose.best() if pose is not None else None
    if best is None:
        return None
    top = best.get("top_m")
    # The hover clear of the top it was planned by, and a margin: that top was what the pick had of
    # it then (2026-11-18, a figure ~42 mm tall learned as 33 by the low end of its looks).
    out = {"grip_at_m": best["grip_at_m"], "max_height_m": None if top is None else float(top) + RECIPE_TOP_MARGIN_M,
           "said": f"a {kind.name} {pose.name}, held {best['held']} time(s) gripped {best['grip_at_m'] * 1000:.0f} mm "
                   f"up — picked so, no looks for its height"}
    across = _across_m(pose)
    if across is not None and getattr(cfg, "RECIPE_OPENS_TO_KIND", False):
        out["across_m"] = across
    return out


def _across_m(pose) -> "float | None":
    """
    How wide a kind resting so is across its narrow side, at the most it has
    been seen (its outline's short side: Pose.span, else its signature) — what
    the jaws, closing across it, must open past. None: not known.
    """
    span = (getattr(pose, "span", None) or {}).get("short_mm")
    short = max(span) if span else (getattr(pose, "sig", None) or {}).get("short_mm")
    return None if not short else float(short) / 1000.0


def _any_way_up(kind) -> "dict | None":
    """
    Scenario.recipe for a kind known, resting a way not met before: resting any
    way, it is no taller than its longest side; where that is short enough to
    grip as low as the jaws go with the gripper's body clear of its top
    (free_depth_m over the lowest grip), gripped low — the lowest grip that has
    held it — and hovered clear of its longest side. No looks for its height.
    User 2026-11-19: "the plug should be liftable in any orientation".
    """
    longest = kind.longest_mm()
    held = [g for p in kind.poses for g in p.grips if g.get("held", 0) > 0]
    if longest is None or not held or not isinstance(POLICIES, ParallelJaw):
        return None
    s = POLICIES.spec
    if longest / 1000.0 > s.min_grasp_height_m + s.free_depth_m:
        return None
    low = min(float(g["grip_at_m"]) for g in held)
    return {"grip_at_m": low, "max_height_m": longest / 1000.0 + RECIPE_TOP_MARGIN_M,
            "said": f"a {kind.name} resting a way not met before: no taller than its longest side "
                    f"({longest:.0f} mm) — gripped low, {low * 1000:.0f} mm up, as it has held"}


def _picked_hook(obj, start_xy, outcome) -> None:
    """Scenario.picked: what the pick showed of obj's kind, learned (catalogue.Catalogue.learn); a kind first met added."""
    cat, track, kind, pose = _kind_of(memory(), obj)
    if cat is None or track is None or not (track.extra or {}).get("sig"):
        return
    sig = track.extra["sig"]
    lying = outcome.diagnosis is not None and outcome.diagnosis.cause == "lying"
    pose_name = "lying" if lying else (pose.name if pose is not None else "upright")
    if kind is not None and pose is None and not lying:
        pose_name = f"resting-{len(kind.poses) + 1}"     # known by its name, not its outline: a new way it rests
    if kind is None:
        if not outcome.ok and not lying:
            return                                     # nothing learned of a kind not met before
        kind, _ = cat.add(track.label or obj.name, (track.attributes or {}).get("colour", ""), sig, pose_name)
        print(f"  [Catalogue] Learned a new kind: {kind.name}" + (f" ({kind.colour})" if kind.colour else "")
              + f", {pose_name}.")
    m = outcome.measured
    tops = getattr(outcome.grasp, "height_m", None)
    best = pose.best() if (pose is not None and m is None) else None
    if tops is not None and best is not None and best.get("top_m") is not None:
        # Picked by the recipe (nothing measured): the height it planned by was the recipe's top and its
        # margin — not something seen. Kept as it was, or the margin is added again every pick: 2026-11-19,
        # the plug's hover crept 82, 92, 102 mm over three runs, too high to reach down from 34 cm out.
        tops = min(float(tops), float(best["top_m"]))
    for a in outcome.attempts:
        grip = a.get("grip_m")
        if grip is None or a.get("result") not in ("ok", "closed_on_nothing", "grip_lost"):
            continue
        held = a["result"] == "ok"
        if not held and a.get("diagnosis") != "too_high":
            continue                                   # a miss for its aim, not its grip
        cat.learn(kind, pose_name, sig, grip_at_m=grip, top_m=tops, held=held,
                  height_mm=m.height_mm if (held and m is not None) else None,
                  widths_mm=m.widths_mm if (held and m is not None) else None)
    if lying:
        cat.learn(kind, "lying", sig, grip_at_m=None, top_m=None, held=False)
    if outcome.ok:
        by = "pick_by_rim" if any(a.get("approach") in ("rim", "handle") for a in outcome.attempts) else "pick"
        if kind.picked_by != by:
            kind.picked_by = by
            cat.save()
    track.extra = {**track.extra, "kind": kind.id, "pose": pose_name}


def _not_an_object(mem, obj) -> bool:
    track = mem.get(obj.track_id) if obj.track_id else None
    return track is not None and track.label == naming.NOT_AN_OBJECT


# ── Pick and place: the framework's, with this task's hooks ───────────────────

def _where_was(name: str) -> str:
    return f"where_{name}_was"


def _vacated(scene_state, obj, start_xy) -> None:
    """
    Scenario.vacated: where a pick took obj from is somewhere to put it back
    ("put it back") — unless it is the spot this very plan puts it back on
    next (scene_state.headed_for), or obj came off a stack (something still
    stands there): its spot then is inside the stack, at the table, where
    nothing can be put. Either way the spot it had stays. 2026-11-13, the UR5:
    "put the blocks back" picked blue off the tower and its spot became the
    tower's; the place there was refused ("green leaves 0 mm"), every try.
    """
    name = _where_was(obj.name)
    had = any(t.name == name for t in scene_state.targets)
    under = [o for o in scene_state.objects if o is not obj and o.position is not None
             and float(np.linalg.norm(np.asarray(o.position[:2], float) - np.asarray(start_xy[:2], float))) < 0.02]
    if under or (had and scene_state.headed_for == name):
        return
    scene_state.targets = [t for t in scene_state.targets if t.name != name] + [
        SceneObject(name, f"where {obj.name} stood before it was picked up", interactable=False,
                    position=np.array([start_xy[0], start_xy[1], cfg.TABLE_Z]))]


# How long after letting go the quick look waits: a block that topples takes a moment to.
PLACED_SETTLE_S = 0.4


def confirm_placed(env, scene_state, obj, at, on) -> "bool | None":
    """
    Scenario.confirm_placed: one frame from each calibrated fixed camera, and
    whether obj now stands where it was put (perception/placed_check.py). The
    overhead camera: its colour where its top should be, that patch changed
    since the last look. Each other one (from the side): the space it now
    fills changed since the last look. A no from any camera that could see is
    a no; else a yes from any is a yes; None when none could say (no
    calibrated camera, no size known for it, the arm in the way).
    """
    from mrcf.perception import placed_check

    mem = scene_state.memory
    track = mem.get(obj.track_id) if mem is not None and obj.track_id else None
    box = boxed(track) if track is not None else None
    height = box[0] if box is not None else (track.height_mm if track is not None else None)
    if track is None or height is None:
        return None
    side = box[1] if box is not None else (track.foot_mm or 30.0)
    kept = scene_state.look_frame or {}
    was = (kept.get("positions") or {}).get(track.id)
    # Where it stood at that look is where it is now (put back): those frames show it there already.
    fresh = was is None or float(np.hypot(was[0] - at[0], was[1] - at[1])) > 0.025
    bottom, top = float(at[2]), float(at[2]) + float(height) / 1000.0
    cameras = env.list_cameras()
    time.sleep(PLACED_SETTLE_S)
    shape = _arm_points(env.arm_shape())
    then = _arm_points(kept.get("arm"))
    said = []                                               # (camera, verdict, why)

    size = (cfg.CAPTURE_W_OVERHEAD, cfg.CAPTURE_H_OVERHEAD)
    model = env.camera_model(cfg.OBSERVE_CAMERA, *size)
    look = track.appearance or {}
    if model is not None and cfg.OBSERVE_CAMERA in cameras and look.get("a") is not None and look.get("b") is not None:
        frame = env.render_image(cfg.OBSERVE_CAMERA, *size)
        flight_recorder.keep_picture(f"placed_{obj.name}_{cfg.OBSERVE_CAMERA}", frame)
        under = mem.get(on.track_id) if on is not None and on.track_id else None
        under_ab = (under.appearance.get("a"), under.appearance.get("b")) if under is not None and under.appearance else None
        before = kept.get("frame") if kept.get("camera") == cfg.OBSERVE_CAMERA and fresh else None
        said.append((cfg.OBSERVE_CAMERA, *placed_check.look(
            frame, model, at[:2], top, float(side) / 4000.0, (look["a"], look["b"]), under_ab,
            _mask(shape, model, frame.shape[:2]), before)))
    for name, before in (kept.get("views") or {}).items():
        h, w = before.shape[:2]
        view = env.camera_model(name, w, h)
        if view is None or name not in cameras:
            continue
        frame = env.render_image(name, w, h)
        flight_recorder.keep_picture(f"placed_{obj.name}_{name}", frame)
        said.append((name, *placed_check.look_volume(
            frame, view, at[:2], bottom, top, float(side) / 2000.0, _mask(shape, view, (h, w)),
            before if fresh else None, _mask(then, view, (h, w)))))
    if not said:
        return None
    print(f"  [Check] {obj.name}: " + "; ".join(f"{name}: {why}" for name, _, why in said) + ".")
    flight_recorder.event("check", "placed", object=obj.name, at=[float(v) for v in at],
                          said={name: [v, why] for name, v, why in said})
    verdicts = [v for _, v, _ in said]
    return False if False in verdicts else (True if True in verdicts else None)


def _arm_points(shape):
    """arm_shape()'s [(point, radius)] as motion_guard.arm_mask's pose, None when there is none."""
    if not shape:
        return None
    return (0.0, np.array([p for p, _ in shape], float), np.array([r for _, r in shape], float))


def _mask(pose, model, size):
    """Where the arm (pose: _arm_points) is in a camera's frame of size (h, w), with a 2 cm margin; None: not known."""
    from mrcf.motion_guard import arm_mask

    return arm_mask([pose], model, size, 0.02) if pose is not None else None


# ── Containers ────────────────────────────────────────────────────────────────
# Things to put objects into — a tub, a bowl — found by what the user calls
# them (mrcf/containers.py: a vision model's box, the segmenter's outline):
# the scan does not list them (2026-11-17, the SO-101: the clear plastic tub was
# never among its objects). Kept by name for the session, found again at each
# scan, and offered to the planner as places to put things INTO (place_in).
# Only on an arm whose policies can (GripperSpec.place_in).

_OVERHEAD_MODEL: list = [None]               # the overhead camera's model, as last read with the arm to hand


def containers_on() -> bool:
    """Whether this arm puts things into containers (GripperSpec.place_in)."""
    return isinstance(POLICIES, ParallelJaw) and bool(POLICIES.spec.place_in)


def _overhead(env):
    """(frame, model) from the overhead camera now; the model kept for the hooks that have no env."""
    size = (cfg.CAPTURE_W_OVERHEAD, cfg.CAPTURE_H_OVERHEAD)
    model = env.camera_model(cfg.OBSERVE_CAMERA, *size)
    if model is not None:
        _OVERHEAD_MODEL[0] = model
    return env.render_image(cfg.OBSERVE_CAMERA, width=size[0], height=size[1]), model


def _keep_container_picture(frame, found) -> None:
    import cv2

    drawn = np.ascontiguousarray(frame).copy()
    cv2.polylines(drawn, [np.asarray(found.outline_px, np.int32)], True, (0, 255, 0), 2)
    if found.box is not None:
        x1, y1, x2, y2 = (int(v) for v in found.box)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (255, 255, 0), 1)
    flight_recorder.keep_picture(f"container_{found.name}", drawn)


def _keeper():
    """The framework's container keeper (mrcf/containers.Keeper) for this table, made once."""
    from mrcf import containers

    if _KEEPER[0] is None:
        _KEEPER[0] = containers.Keeper(
            cfg.OBSERVE_CAMERA, _overhead, grounding_model=cfg.GROUNDING_MODEL, judge_model=cfg.MODEL,
            tallest_m=getattr(cfg, "CONTAINER_TALLEST_M", None) or cfg.TALLEST_OBJECT_M, table_z=cfg.TABLE_Z,
            rims_path=cfg.CONTAINERS_PATH,
            keep_picture=_keep_container_picture,
            opening_by_shade=bool(getattr(cfg, "CONTAINER_OPENING_BY_SHADE", False)))
    return _KEEPER[0]


_KEEPER: list = [None]


def container_name(what) -> str:
    """The scene's name for a container the user calls `what` (containers.name_for)."""
    from mrcf import containers

    return containers.name_for(what)


def look_for(env: RobotEnv, scene_state, name) -> bool:
    """
    Deployment.look_around, and the look_for action: an object the scan did
    not outline, found by what it is called — the vision model's box round it
    in the overhead camera's picture, the segmenter's outline inside it
    (perception/grounding.py, as a container is found) — and put in the scene
    and the memory under that name. What the scan did outline inside it (a
    part of it: a plug's pin) is folded into it, not kept as a neighbour.
    2026-11-18: a white plug on its back on the white table, the scan
    outlining only one pin of it ("silver screw", 8 mm), and every pick of it
    aimed by that.
    """
    from mrcf import catalogue as catalogue_mod
    from mrcf import containers
    from mrcf.object_memory import Detection
    from mrcf.perception import grounding
    from mrcf.perception.plane_geometry import PlaneView, measure

    words = str(name).replace("_", " ")
    words = re.sub(r"\bobj\d+\b", "", words).strip()
    goto_observe_pose(env)
    frame, model = _overhead(env)
    if model is None:
        print(f"  [Look] {words}: the overhead camera is not calibrated.")
        return False
    frame = np.ascontiguousarray(frame)
    if containers_on():
        # What is in a container already is not what is looked for on the table: each kept container's
        # outline greyed out first (2026-11-20, "the screwdrivers": the one put in the box, found again).
        import cv2
        frame = frame.copy()
        fh, fw = frame.shape[:2]
        for c in list(_keeper().kept.values()):
            pts = np.asarray(c.outline_px, float).reshape(-1, 2)
            if len(pts) >= 3 and tuple(c.size) == (fw, fh):
                cv2.fillPoly(frame, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], (128, 128, 128))
    found, why = grounding.find(frame, f"the {words}", model=cfg.GROUNDING_MODEL, ask=call_vlm)
    if found is None or found.mask is None:
        print(f"  [Look] {words}: not found ({why}).")
        return False
    mask = np.asarray(found.mask, bool)
    view = PlaneView.from_camera_model(model)
    m = measure(mask, view)
    if m is None or m.base_world_m is None:
        print(f"  [Look] {words}: outlined, but its foot cannot be measured from the outline.")
        return False
    xy = np.asarray(m.base_world_m, float)[:2]
    if not _in_bounds(xy):
        print(f"  [Look] {words}: found at ({xy[0]:+.3f}, {xy[1]:+.3f}), outside the work area.")
        return False
    h, w = frame.shape[:2]
    px = containers.outline_px(mask)
    foot = 2 * float(m.base_radius_mm)
    mem = memory()
    track = mem.add(Detection(position=(float(xy[0]), float(xy[1])), foot_mm=foot, label=words,
                              label_source="user"), why=f"found by name ({found.score:.2f})")
    track.extra = {**(track.extra or {}), "sig": catalogue_mod.signature(mask, view, foot, None)}
    obj = SceneObject(name=str(name), description=f"{words} ({foot:.0f} mm across), found by name",
                      position=np.r_[xy, cfg.TABLE_Z], track_id=track.id,
                      geometry={"base_diameter_m": foot / 1000.0,
                                "outlines": {cfg.OBSERVE_CAMERA: {"size": [int(w), int(h)],
                                                                  "px": [] if px is None else px.tolist()}}})
    parts = []
    for o in list(scene_state.objects):
        if o.position is None or o.pixel_xy is None:
            continue
        then = ((o.geometry or {}).get("outlines") or {}).get(cfg.OBSERVE_CAMERA, {}).get("size") or [w, h]
        u, v = int(float(o.pixel_xy[0]) * w / float(then[0])), int(float(o.pixel_xy[1]) * h / float(then[1]))
        if 0 <= v < h and 0 <= u < w and mask[v, u]:
            parts.append(o)
            scene_state.objects.remove(o)
            if o.track_id:
                mem.part_of(o.track_id, track.id, f"inside the outline of {words}")
    scene_state.objects.append(obj)
    print(f"  [Look] {name}: found by name at ({xy[0]:+.3f}, {xy[1]:+.3f}), {foot:.0f} mm across, outlined "
          f"{found.score:.2f}" + (f"; {', '.join(o.name for o in parts)} taken as part of it" if parts else "") + ".")
    flight_recorder.keep_picture(f"look_{name}", frame)
    return True


register_action(
    "look_for",
    description=("Look for an object the scan does not list, by what it is called ('the white plug'), and add it "
                 "to the scene: for one the same colour as the table, or only partly outlined."),
    parameters={"object_name": "What it is called, as the user says it ('white_plug')."},
    always=True,
)(lambda env, scene_state, object_name, **_: look_for(env, scene_state, object_name))


def known_container(name):
    """The kept container called name (containers.Keeper.known)."""
    return _keeper().known(name)


def find_container(env: RobotEnv, scene_state, what, park: bool = True) -> bool:
    """
    Look for `what` with the overhead camera (its view cleared first,
    goto_observe_pose, unless park is False), and keep it as a container if the vision
    model finds it and takes it for one. True when found.
    """
    if not containers_on():
        print("  [Find] This arm does not put things into containers (GripperSpec.place_in).")
        return False
    if park:
        goto_observe_pose(env)
    found, why = _keeper().find(env, scene_state, what)
    flight_recorder.event("containers", "find", what=str(what).replace("_", " "), found=found is not None, why=why)
    if found is None:
        print(f"  [Find] {str(what).replace('_', ' ')}: {why}.")
        return False
    centre = found.centre_xy(_keeper().model)
    print(f"  [Find] {found.name}: a container"
          + (f" at ({centre[0]:+.3f}, {centre[1]:+.3f}) m" if centre is not None else "")
          + f", outlined {found.score:.2f}"
          + (f"; walls {found.rim_m * 1000:.0f} mm, as measured before" if found.rim_measured else "") + ".")
    return True


# How far outside the vision model's box round an object the scan may have put
# it and it still be that object (outline_again): the scan's point is its foot's,
# at the box's edge for something tall, seen slantwise.
OUTLINE_AGAIN_PX = 40


def outline_again(env: RobotEnv, scene_state, obj, _parked: bool = False, move: bool = True) -> bool:
    """
    Scenario.outline_again: obj outlined afresh in the overhead camera (its
    view cleared, goto_observe_pose, unless move is False), found by what it is — the vision model's box round
    it, which must take in where the scan has it (OUTLINE_AGAIN_PX), then the
    segmenter in that box (perception/grounding.py) — and that outline kept for the pick
    (Target.outlines). The scan's own outline is the segmenter's alone: under a
    lamp turned on, a blue scoop's came out an 8 mm scrap of it, and every look
    for its walls' height aimed at the scrap (2026-11-17). False, the scan's
    kept, when it is not found there.
    """
    from mrcf import containers
    from mrcf.perception import grounding
    from mrcf.perception.plane_geometry import PlaneView

    if obj.pixel_xy is None:
        return False
    if move:
        goto_observe_pose(env)
    frame, model = _overhead(env)
    if model is None:
        return False
    h, w = frame.shape[:2]
    then = ((obj.geometry or {}).get("outlines") or {}).get(cfg.OBSERVE_CAMERA, {}).get("size") or [w, h]
    u, v = float(obj.pixel_xy[0]) * w / float(then[0]), float(obj.pixel_xy[1]) * h / float(then[1])
    words = re.sub(r"_obj\d+$", "", obj.name).replace("_", " ")
    found, why = grounding.find_near(np.ascontiguousarray(frame), f"the {words}", (u, v), OUTLINE_AGAIN_PX,
                                     model=cfg.GROUNDING_MODEL, ask=call_vlm)
    if found is None:
        print(f"  [Outline] {obj.name}: {why} — going by the scan's outline.")
        return False
    x0, y0, x1, y1 = (float(c) for c in found.box)
    mask = np.asarray(found.mask, bool)
    px = containers.outline_px(mask)
    if px is None:
        return False
    from mrcf import scene_scan

    arm = scene_scan.arm_in_view(env, cfg.OBSERVE_CAMERA, w, h) if getattr(cfg, "SCAN_PAST_ARM", False) else None
    if arm is not None and float((arm & mask).sum()) > scene_scan.ARM_OVERLAP * max(1.0, float(mask.sum())):
        if not move:
            print(f"  [Outline] {obj.name}: the arm is in the way of it here — going by the scan's outline.")
            return False
        if not _parked:
            print(f"  [Outline] {obj.name}: the arm is in the way of it — parking it to look again.")
            env.go_park()
            return outline_again(env, scene_state, obj, _parked=True)
        return False
    obj.geometry = {**(obj.geometry or {}), "outlines": {cfg.OBSERVE_CAMERA: {"size": [int(w), int(h)],
                                                                            "px": px.tolist()}}}
    view = PlaneView.from_camera_model(model)
    on_plane = view.to_plane(px)
    across = float(np.max(np.ptp(on_plane, axis=0))) if np.isfinite(on_plane).all() else float("nan")
    # Where it is: the middle of that outline on the surface — what the looks at it aim at. The scan's
    # point was its scrap's, at the scoop's edge: every look had it at the picture's corner.
    rows, cols = np.nonzero(mask)
    middle = view.to_plane(np.array([[cols.mean(), rows.mean()]], float))[0]
    moved = ""
    if np.isfinite(middle).all() and obj.position is not None:
        origin = np.zeros(2) if view.origin_m is None else np.asarray(view.origin_m, float)[:2]
        xy = origin + middle / 1000.0
        moved = f", its middle {float(np.linalg.norm(xy - np.asarray(obj.position, float)[:2])) * 1000:.0f} mm from the scan's"
        obj.position = np.array([xy[0], xy[1], float(obj.position[2])])
    print(f"  [Outline] {obj.name}: outlined again by what it is ({found.score:.2f}), {across:.0f} mm across{moved}.")
    _resign(scene_state, obj, mask, view)
    _fold_parts(scene_state, obj, mask, w, h)
    import cv2

    drawn = np.ascontiguousarray(frame).copy()
    cv2.polylines(drawn, [px.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.rectangle(drawn, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 1)
    flight_recorder.keep_picture(f"outline_{obj.name}", drawn)
    return True


def _outline_suspect(scene_state, obj) -> bool:
    """
    Scenario.outline_suspect: whether the scan's outline of obj is worth
    looking at again before its pick — its kind not known from it, or it much
    smaller than its kind has been seen at (a fragment), or another outline
    standing against it (it split in two). A whole outline of a known kind is
    picked as the scan saw it.
    """
    cat, mem = catalogue(), scene_state.memory
    track = mem.get(obj.track_id) if (cat is not None and mem is not None and obj.track_id) else None
    extra = (track.extra or {}) if track is not None else {}
    kind = cat.get(extra.get("kind")) if cat is not None else None
    pose = kind.pose(extra.get("pose")) if kind is not None and extra.get("pose") else None
    if kind is None or pose is None:
        return True
    foot = float((obj.geometry or {}).get("base_diameter_m") or 0.0) * 1000.0
    smallest = float((pose.span.get("foot_mm") or [pose.sig.get("foot_mm") or 0.0])[0])
    if foot and smallest and foot < 0.8 * smallest:
        return True
    if obj.position is not None:
        for o in scene_state.objects:
            if o is not obj and o.position is not None and float(np.linalg.norm(
                    np.asarray(o.position[:2], float) - np.asarray(obj.position[:2], float))) < 0.03:
                return True
    return False


def _outline_partial(scene_state, obj) -> bool:
    """
    Scenario.outline_suspect without a catalogue (OUTLINE_ONLY_PARTIAL): looked
    at again only where the scan's outline looks like a piece of it — under 40
    mm long on the table (scene_scan's length_mm), or another outline standing
    against it — not every one. 2026-11-20, the UR5: each tool outlined again,
    ~6 s with the arm still, over outlines that were whole (user: "it doesn't
    need to reoutline - nothing else has moved"); the 10 mm piece of a hammer
    that morning is what it is for.
    """
    mem = scene_state.memory
    track = mem.get(obj.track_id) if (mem is not None and obj.track_id) else None
    length = ((track.extra or {}).get("length_mm") if track is not None else None)
    if not length or float(length) < 40.0:
        return True
    if obj.position is not None:
        for o in scene_state.objects:
            if o is not obj and o.position is not None and float(np.linalg.norm(
                    np.asarray(o.position[:2], float) - np.asarray(obj.position[:2], float))) < 0.03:
                return True
    return False


def _fold_parts(scene_state, obj, mask, w: int, h: int) -> None:
    """
    What the scan outlined inside obj's whole outline (outline_again) taken
    as part of it, as look_for does: 2026-11-19, a plug on its side outlined
    as its body and, apart, its pins — both "white block", the pins kept as a
    neighbour the jaws must leave room for.
    """
    if scene_state is None:
        return
    mem = getattr(scene_state, "memory", None)
    parts = []
    for o in list(scene_state.objects):
        if o is obj or o.position is None or o.pixel_xy is None:
            continue
        then = ((o.geometry or {}).get("outlines") or {}).get(cfg.OBSERVE_CAMERA, {}).get("size") or [w, h]
        u, v = int(float(o.pixel_xy[0]) * w / float(then[0])), int(float(o.pixel_xy[1]) * h / float(then[1]))
        if 0 <= v < h and 0 <= u < w and mask[v, u]:
            parts.append(o.name)
            scene_state.objects.remove(o)
            if mem is not None and o.track_id and obj.track_id:
                mem.part_of(o.track_id, obj.track_id, f"inside the outline of {obj.name}")
    if parts:
        print(f"  [Outline] {', '.join(parts)} taken as part of {obj.name}.")


def _resign(scene_state, obj, mask, view) -> None:
    """
    obj's catalogue signature made again from its new outline (outline_again),
    and its kind matched by that: the scan's fragment of it matched nothing (a
    figure 16 mm across of its 23), the whole of it is what it is.
    """
    cat = catalogue()
    mem = getattr(scene_state, "memory", None)
    track = mem.get(obj.track_id) if (cat is not None and mem is not None and obj.track_id) else None
    if track is None:
        return
    from mrcf import catalogue as catalogue_mod

    old = dict((track.extra or {}).get("sig") or {})
    try:
        rough = catalogue_mod.signature(mask, view, 0.0, None)
    except Exception:                                      # noqa: BLE001 — kept as it was
        return
    # Its foot: the scan's, unless that was a fragment of this outline (then its narrower side).
    foot = old.get("foot_mm") or 0.0
    if not foot or foot < 0.75 * float(rough["short_mm"]):
        foot = float(rough["short_mm"])
    sig = {**rough, "foot_mm": round(float(foot), 1), "L": old.get("L"), "a": old.get("a"), "b": old.get("b")}
    got = cat.match(sig)
    track.extra = {**(track.extra or {}), "sig": sig}
    if got is not None:
        kind, pose, _ = got
        if (track.extra or {}).get("kind") != kind.id:
            print(f"  [Catalogue] {obj.name}: outlined whole, a {kind.colour + ' ' if kind.colour else ''}{kind.name}, "
                  f"{pose.name}.")
        track.extra = {**track.extra, "kind": kind.id, "pose": pose.name}


def refresh_containers(env: RobotEnv, scene_state) -> None:
    """
    Every kept container found again where it is now; one not seen is
    forgotten. With the arm in the camera's view (cfg.SCAN_PAST_ARM), one it
    covers keeps the outline it had: found again through the arm, the outline
    would take the arm in.
    """
    from mrcf import scene_scan

    keeper = _keeper()
    if not keeper.kept and not keeper.remembered():
        return
    size = (cfg.CAPTURE_W_OVERHEAD, cfg.CAPTURE_H_OVERHEAD)
    arm = scene_scan.arm_in_view(env, cfg.OBSERVE_CAMERA, *size) if getattr(cfg, "SCAN_PAST_ARM", False) else None
    # One kept from before (not found yet this session) is taken unless the arm covers a good share
    # of it (scene_scan.ARM_OVERLAP): the opening errs inward anyway. 2026-11-18: the tub's corner
    # by the arm's base, under the arm at home, and it was left unfound at the first scan.
    def covered(c) -> bool:
        most = 0.05 if c.name in keeper.kept else scene_scan.ARM_OVERLAP
        return scene_scan.share_on_arm(arm, c.outline_px, c.size) > most
    for name, said in keeper.refresh(env, scene_state, covered=covered):
        print(f"  [Containers] {name}: {said}.")


def _holds(scene_state, obj) -> "str | None":
    """Scenario.holds (containers.Keeper.holds)."""
    return _keeper().holds(scene_state, obj)


def _container_targets(scene_state) -> None:
    """The kept containers as place targets (containers.Keeper.targets)."""
    _keeper().targets(scene_state)


def _opening(scene_state, name):
    """Scenario.opening (containers.Keeper.opening)."""
    return _keeper().opening(scene_state, name)


def _opening_measured(scene_state, name, rim_m: float) -> None:
    """Scenario.opening_measured: how high its walls stand, kept on disk (containers.Keeper.measured)."""
    _keeper().measured(scene_state, name, rim_m)
    flight_recorder.event("containers", "rim", name=str(name), rim_m=round(float(rim_m), 4))


# ── Seeing its own jaws ───────────────────────────────────────────────────────
# The wrist camera learns where the jaws are in its picture (ParallelJaw.
# learn_jaw_view) over clear stretches of the table: what is behind them must
# differ, and nothing be within reach of the open jaws 12 cm up.

JAW_VIEW_CLEAR_M = 0.07          # a spot this far from every object's foot
JAW_VIEW_SPOTS = 3


def clear_spots(env: RobotEnv, scene_state, n: int = JAW_VIEW_SPOTS, clear_m: float = JAW_VIEW_CLEAR_M) -> list:
    """
    Up to n spots [(x, y)] on the table this far from every object's foot,
    outside every kept container by as much, inside the workspace, spread as
    far apart as they can be — nearest the middle of the workspace first.
    """
    from mrcf.policies.place_in import inside_by

    b = cfg.WORKSPACE_BOUNDS
    feet = [np.asarray(o.position[:2], float) for o in scene_state.objects if o.position is not None]
    walls = _keeper().floors()
    free = []
    for x in np.arange(b.x_min + 0.08, b.x_max - 0.06, 0.02):
        for y in np.arange(b.y_min + 0.04, b.y_max - 0.04, 0.02):
            xy = np.array([x, y])
            if any(np.linalg.norm(xy - f) < clear_m for f in feet):
                continue
            if any(w is not None and len(w) >= 3 and inside_by(w, xy) > -clear_m for w in walls):
                continue
            free.append(xy)
    if not free:
        return []
    middle = np.array([(b.x_min + b.x_max) / 2, 0.0])
    chosen = [min(free, key=lambda p: float(np.linalg.norm(p - middle)))]
    while len(chosen) < n:
        far = max(free, key=lambda p: min(float(np.linalg.norm(p - c)) for c in chosen))
        if min(float(np.linalg.norm(far - c)) for c in chosen) < 0.04:
            break
        chosen.append(far)
    return [tuple(map(float, p)) for p in chosen]


def learn_jaws(env: RobotEnv, scene_state) -> bool:
    """The wrist camera learns where the jaws are, over clear spots of the table; home after."""
    if not isinstance(POLICIES, ParallelJaw):
        print("  [JawView] This arm's policies have no jaws to learn.")
        return False
    # Spares beyond those needed: one out of reach at the height learned from is skipped (2026-11-18,
    # (+0.18, +0.03) by the base at 18 cm, and the jaws not learned again).
    spots = clear_spots(env, scene_state, n=POLICIES.spec.jaw_view_spots_min + 3)
    if len(spots) < POLICIES.spec.jaw_view_spots_min:
        print(f"  [JawView] Not enough clear table ({len(spots)} spot(s) {JAW_VIEW_CLEAR_M * 100:.0f} cm from "
              "everything) to learn the jaws over.")
        return False
    print(f"  [JawView] Learning where the jaws are in the wrist camera's picture, over "
          f"{', '.join(f'({x:+.2f}, {y:+.2f})' for x, y in spots)}...")
    # Above the tallest anything here could be (a container not yet asked for is not known to stand there).
    with flight_recorder.RECORDER.action("learn_jaw_view", spots=[list(s) for s in spots]) as act:
        out = POLICIES.learn_jaw_view(env, spots, height_m=max(POLICIES.spec.jaw_view_height_m,
                                                               cfg.TALLEST_OBJECT_M + 0.03))
        act.done(ok=out.ok, message=out.message, refused=out.refused)
    print(f"  [JawView] {out.message}.")
    POLICIES.home(env)
    return out.ok


def on_start(env: RobotEnv, scene_state) -> None:
    """
    Deployment.on_start: an arm that looks for heights with its wrist camera
    learns its jaws first — once, and again when the jaws may open wider than
    it learned them at.
    """
    if isinstance(POLICIES, ParallelJaw) and POLICIES.spec.camera is not None:
        from mrcf import wrist_shift

        drifted = wrist_shift.said(flight_recorder.ROOT)
        if drifted:
            print(f"  [Calibration] {drifted[0].upper()}{drifted[1:]}.")
    if not (isinstance(POLICIES, ParallelJaw) and POLICIES.spec.height_from_views
            and POLICIES.jaw_view_path is not None):
        return
    view = POLICIES.jaw_view()
    if view is None or max(view.openings_m) < POLICIES.spec.max_open_m - 0.001:
        learn_jaws(env, scene_state)


@register_action(
    "learn_jaw_view",
    description=("Let the wrist camera learn again where the gripper's own jaws are in its picture (after the "
                 "camera has been moved or knocked). Moves the arm over clear table."),
    always=True,
)
def skill_learn_jaw_view(env: RobotEnv, scene_state, **_):
    return learn_jaws(env, scene_state)


# ── The surface the objects stand on ──────────────────────────────────────────
# Touched with the shut jaws (ParallelJaw.touch_surface) at a few clear spots
# near the objects — the surface they stand on is the one that counts — and
# kept (config.SURFACE_PATH) for the next start to work from.

SURFACE_CLEAR_M = 0.06           # a touch this far from every object's foot ...
SURFACE_NEAR_M = 0.15            # ... and no further than this from one
SURFACE_AGREE_M = 0.003          # the touches agree this closely, or nothing is kept
SURFACE_SEARCH_M = (0.04, 0.05)  # from this far above the surface believed now, down to this far below it


def surface_spots(scene_state, n: int = 3) -> list:
    """Up to n spots [(x, y)] to touch: clear of every object and container, near some object, spread apart."""
    from mrcf.policies.place_in import inside_by

    b = cfg.WORKSPACE_BOUNDS
    feet = [np.asarray(o.position[:2], float) for o in scene_state.objects if o.position is not None]
    walls = _keeper().floors()
    free = []
    for x in np.arange(b.x_min + 0.08, b.x_max - 0.12, 0.02):
        for y in np.arange(b.y_min + 0.04, b.y_max - 0.04, 0.02):
            xy = np.array([x, y])
            near = [float(np.linalg.norm(xy - f)) for f in feet]
            if not near or min(near) < SURFACE_CLEAR_M or min(near) > SURFACE_NEAR_M:
                continue
            if any(w is not None and len(w) >= 3 and inside_by(w, xy) > -SURFACE_CLEAR_M for w in walls):
                continue
            free.append(xy)
    if not free:
        return []
    middle = np.mean(feet, axis=0)
    chosen = [min(free, key=lambda p: float(np.linalg.norm(p - middle)))]
    while len(chosen) < n:
        far = max(free, key=lambda p: min(float(np.linalg.norm(p - c)) for c in chosen))
        if min(float(np.linalg.norm(far - c)) for c in chosen) < 0.04:
            break
        chosen.append(far)
    return [tuple(map(float, p)) for p in chosen]


def measure_surface(env: RobotEnv, scene_state) -> bool:
    """Touch the surface at a few clear spots near the objects and keep its height for the next start."""
    import json

    if not (isinstance(POLICIES, ParallelJaw) and POLICIES.spec.touch_surface):
        print("  [Surface] This arm does not touch the surface to measure it (GripperSpec.touch_surface).")
        return False
    spots = surface_spots(scene_state)
    if not spots:
        print("  [Surface] No clear spot near the objects to touch.")
        return False
    above, below = SURFACE_SEARCH_M
    print(f"  [Surface] Touching the table with the shut jaws at "
          f"{', '.join(f'({x:+.2f}, {y:+.2f})' for x, y in spots)}, from {(cfg.TABLE_Z + above) * 1000:.0f} "
          f"down to {(cfg.TABLE_Z - below) * 1000:.0f} mm at most...")
    heights, where = [], []
    with flight_recorder.RECORDER.action("measure_surface", spots=[list(s) for s in spots]) as act:
        for xy in spots:
            out = POLICIES.touch_surface(env, xy, cfg.TABLE_Z + above, cfg.TABLE_Z - below)
            print(f"  [Surface] ({xy[0]:+.2f}, {xy[1]:+.2f}): {out.message}.")
            if out.ok and out.measured is not None:
                heights.append(out.measured.height_mm / 1000.0)
                where.append(list(xy))
        ok = len(heights) >= 2 and max(heights) - min(heights) <= SURFACE_AGREE_M
        act.done(ok=ok, message=f"touched {[round(h * 1000, 1) for h in heights]} mm")
    POLICIES.home(env)
    if not ok:
        print(f"  [Surface] Not kept: {len(heights)} touch(es), "
              + (f"{(max(heights) - min(heights)) * 1000:.1f} mm apart" if heights else "none")
              + f" ({SURFACE_AGREE_M * 1000:.0f} mm needed).")
        return False
    z = float(np.median(heights))
    cfg.SURFACE_PATH.write_text(json.dumps({
        "table_z": round(z, 5), "measured": time.strftime("%Y-%m-%d %H:%M"), "by": "touch_surface",
        "touches_m": [round(h, 5) for h in heights], "at": where, "arm_table_z": cfg.ARM.config.TABLE_Z}, indent=1))
    print(f"  [Surface] The table is at {z * 1000:+.1f} mm ({(z - cfg.ARM.config.TABLE_Z) * 1000:+.1f} from the arm's "
          f"TABLE_Z); kept in {cfg.SURFACE_PATH.name}. Start the controller again to work from it.")
    return True


if isinstance(POLICIES, ParallelJaw) and POLICIES.spec.touch_surface:
    @register_action(
        "measure_surface",
        description=("Touch the table with the shut jaws at a few clear spots to find how high it stands, and keep "
                     "it for the next start (after the surface has changed: a board taken away, another table)."),
        always=True,
    )
    def skill_measure_surface(env: RobotEnv, scene_state, **_):
        return measure_surface(env, scene_state)


# ── Finding an object again after a miss ──────────────────────────────────────
# Vision.relocate: the object a close missed, found again by a fresh look of the
# overhead camera (the arm out of its view): no further than this from where it
# was, else it was knocked away. Only where, not what: no vision model.
RELOCATE_RADIUS_M = 0.03


def _arm_clear_of(env: RobotEnv, xy) -> bool:
    """Whether the overhead camera sees round xy past the arm where it is (cfg.SCAN_PAST_ARM): no need to move it."""
    if not getattr(cfg, "SCAN_PAST_ARM", False):
        return False
    share, spot = arm_hides(env, xy, RELOCATE_RADIUS_M + 0.02)
    if share is None or spot:
        return False
    print("  [Relocate] The overhead camera sees round it past the arm: looking from here.")
    return True


def relocate(env: RobotEnv, near_xy):
    """
    Vision.relocate: the foot nearest near_xy in a fresh surface scan, within
    RELOCATE_RADIUS_M. Looked for past the arm where it is when the arm hides
    none of round it (cfg.SCAN_PAST_ARM); else from the park pose, the arm home
    after.
    """
    from mrcf.perception import surface_scan

    near = np.asarray(near_xy, float)[:2]
    stayed = _arm_clear_of(env, near)
    if not stayed:
        goto_observe_pose(env, around=near)
    size = (cfg.CAPTURE_W_OVERHEAD, cfg.CAPTURE_H_OVERHEAD)
    b = cfg.WORKSPACE_BOUNDS
    got = surface_scan.scan(env, cfg.OBSERVE_CAMERA, *size, (b.x_min, b.x_max), (b.y_min, b.y_max),
                            reference=surface_scan.load_reference(cfg.SURFACE_REFERENCE_PATH, size))
    if not stayed:
        env.go_home()
    if got is None:
        return None
    feet = [np.asarray(f["measurement"].base_world_m, float)[:2] for f in got[1]]
    best = min(feet, key=lambda p: float(np.linalg.norm(p - near)), default=None)
    if best is None or float(np.linalg.norm(best - near)) > RELOCATE_RADIUS_M:
        return None
    return best


def _find_opening(env: RobotEnv, scene_state, name) -> bool:
    """Scenario.find_opening: a container the user named that is not kept yet, looked for by those words."""
    return known_container(name) is not None or find_container(env, scene_state, name)


def _diagnose(frames, what):
    """Vision.diagnose: why a close found nothing, asked of the vision model (perception/grasp_diagnosis.py)."""
    from mrcf.perception import grasp_diagnosis
    from mrcf.policies.base import Diagnosis

    got = grasp_diagnosis.diagnose(frames, call_vlm, what, model=cfg.MODEL)
    return None if got is None else Diagnosis(*got)


def install_manipulation() -> None:
    """Register pick, place, move_above, home, open/close_gripper (manipulation.py) for the tabletop."""
    on = containers_on()
    # Found again after a miss, by a fresh overhead look: the arm's own setting (RELOCATE_AFTER_MISS),
    # off where it has not been tried — a miss then stops the pick and asks for a scan.
    again = bool(getattr(cfg, "RELOCATE_AFTER_MISS", False))
    manipulation.install(
        POLICIES,
        vision=Vision(relocate=relocate if again else None,
                      diagnose=_diagnose if getattr(cfg, "DIAGNOSE_MISSES", False) else None),
        scenario=manipulation.Scenario(
            max_height=lambda obj: cfg.TALLEST_OBJECT_M,
            describe=describe,
            vacated=_vacated,
            carries=lambda obj: True,        # anything whose top is known (place_on refuses the rest)
            confirm_placed=confirm_placed,
            opening=_opening if on else None,
            holds=_holds if on else None,
            find_opening=_find_opening if on else None,
            opening_measured=_opening_measured if on else None,
            outline_again=outline_again,
            outline_suspect=(_outline_suspect if catalogue() is not None else
                             _outline_partial if getattr(cfg, "OUTLINE_ONLY_PARTIAL", False) else None),
            recipe=_recipe if getattr(cfg, "CATALOGUE", False) else None,
            picked=_picked_hook if getattr(cfg, "CATALOGUE", False) else None,
        ),
    )
    if on:
        manipulation.install_place_in()
    if isinstance(POLICIES, ParallelJaw) and POLICIES.spec.rim_pick:
        manipulation.install_pick_by_rim()


install_manipulation()


# ── Plans kept for the words they were made for ───────────────────────────────

def _plan_key(words: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9_ ]", " ", str(words).lower()).split())


def _plans() -> dict:
    try:
        return dict(json.loads(Path(cfg.PLANS_PATH).read_text()).get("plans", {}))
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def plan_made(words: str, actions: list) -> None:
    """controller.Deployment.plan_made: kept on disk (PLANS_PATH), the last plan for these words."""
    plans = _plans()
    plans[_plan_key(words)] = {"actions": actions, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path = Path(cfg.PLANS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"plans": plans}, indent=1))
    tmp.replace(path)


def remembered_plan(words: str, scene_state) -> "list | None":
    """
    controller.Deployment.remembered_plan: the plan these words had before,
    when every object it names is on the table now — not held, not in a
    container already — and every place it names is there; else None.
    """
    kept = _plans().get(_plan_key(words))
    if not kept:
        return None
    actions = list(kept.get("actions") or [])
    targets = {t.name for t in scene_state.targets}
    for a in actions:
        params = a.get("parameters") or {k: v for k, v in a.items() if k != "action"}
        name = params.get("object_name")
        if name is not None:
            obj = next((o for o in scene_state.objects if o.name == name), None)
            if obj is None or obj.position is None or not obj.interactable:
                return None
            if containers_on() and _keeper().holds(scene_state, obj):
                return None
        place = params.get("container") or params.get("target_name")
        if place is not None and place not in targets:
            return None
    return actions or None


# ── "Put these into that", planned from the words ─────────────────────────────

_PUT_INTO = re.compile(r"^\s*(?:please\s+)?(?:put|place|move|drop|tidy)\s+(?P<things>.+?)\s+"
                       r"(?:in|into|inside)\s+(?P<into>.+?)\s*[.!]?\s*$", re.IGNORECASE)
_ARTICLES = {"the", "a", "an", "both", "my", "this", "that", "these", "those"}


def _words(text: str) -> list:
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", str(text).lower().replace("_", " ")).split()
            if w not in _ARTICLES and not re.fullmatch(r"obj\d+", w)]


def _container_called(scene_state, said: str) -> "str | None":
    """The kept container whose name the words say (all of its words in them, or they in its name)."""
    want = set(_words(said))
    for t in scene_state.targets:
        have = set(_words(t.name))
        if have and (have <= want or (want and want <= have)):
            return t.name
    return None


def _thing_called(scene_state, said: str, taken: set):
    """
    The object on the table the user means by `said` ("the white plug"), not
    yet taken by this plan: called that; else of a kind the catalogue knows by
    those words (its name and colour), seen now; else None.
    """
    want = _words(said)
    if not want:
        return None
    standing = [o for o in scene_state.objects if o.position is not None and o.interactable
                and getattr(o, "reachable", True) is not False
                and o.name not in taken and not (containers_on() and _keeper().holds(scene_state, o))]
    cat, mem = catalogue(), scene_state.memory

    def of_that_kind(o) -> bool:
        track = mem.get(o.track_id) if (mem is not None and o.track_id) else None
        kind = cat.get((track.extra or {}).get("kind")) if (cat is not None and track is not None) else None
        return kind is not None and set(_words(kind.name)) <= set(want) and (not kind.colour or kind.colour in want)

    # Called that, the one the catalogue knows as that kind first: 2026-11-19 15:21, the power strip off the
    # table's edge called "white_plug", the plug itself "white_plug_obj3".
    named = sorted((o for o in standing if _words(o.name) == want),
                   key=lambda o: (not of_that_kind(o), not _in_bounds(o.position[:2])))
    if named:
        return named[0]
    kinded = next((o for o in standing if of_that_kind(o)), None)
    if kinded is not None or not getattr(cfg, "TASK_WORDS_BY_KIND", False):
        return kinded
    # By its kind's words, any colour (TASK_WORDS_BY_KIND): "the hammer" is the grey_hammer or the silver_hammer —
    # 2026-11-20, the UR5's tools were named red, then orange screwdriver, grey, then silver hammer, run to run.
    return next((o for o in sorted(standing, key=lambda o: not _in_bounds(o.position[:2]))
                 if set(want) <= set(_words(o.name))), None)


def _each_of(scene_state, said: str) -> list:
    """
    The things `said` names, one phrase each: "the screwdrivers" — a plural,
    when TASK_WORDS_BY_KIND — as "screwdriver" once for every object whose
    name holds its words (at least once, to be looked for); else `said` itself.
    """
    words = _words(said)
    if not (getattr(cfg, "TASK_WORDS_BY_KIND", False) and words and len(words[-1]) > 3 and words[-1].endswith("s")):
        return [said]
    one = words[:-1] + [words[-1][:-1]]
    n = sum(1 for o in scene_state.objects if o.position is not None and o.interactable
            and getattr(o, "reachable", True) is not False and set(one) <= set(_words(o.name)))
    return [" ".join(one)] * max(1, n)


def _how_picked(obj) -> str:
    """
    The pick that holds it: its kind's (catalogue); else, with the lying pick on (GripperSpec.lying_pick),
    the plain pick; else by its wall where it is wider than the jaws open.
    """
    _, _, kind, _ = _kind_of(memory(), obj)              # by its outline, else by its name and colour
    if kind is not None and kind.picked_by and kind.picked_by in ACTIONS:
        return kind.picked_by
    if isinstance(POLICIES, ParallelJaw) and POLICIES.spec.lying_pick:
        # The lying pick finds for itself where the jaws fit across it, by its outline, not by the round foot
        # (2026-11-20: the UR5's tape measure, 62 mm "across" by its foot, sent to a pick by its wall).
        return "pick"
    foot = float((obj.geometry or {}).get("base_diameter_m") or 0.0)
    wide = isinstance(POLICIES, ParallelJaw) and foot > POLICIES.spec.max_open_m
    return "pick_by_rim" if wide and "pick_by_rim" in ACTIONS else "pick"


def quick_plan(words: str, env, scene_state, partial: "bool | None" = None) -> "list | None":
    """
    controller.Deployment.quick_plan for "put <things> into <container>":
    each thing found by what the user calls it (_thing_called; else looked for
    by those words, look_for), then picked as its kind is (_how_picked) and put
    into the container. None — planned as usual — for other words, a container
    not known, or a thing not found. partial (default: cfg.PLANNER off): a
    thing not found is left out, said, and the rest planned — there is no
    planner to hand it to.
    """
    if partial is None:
        partial = not bool(getattr(cfg, "PLANNER", True))
    m = _PUT_INTO.match(str(words))
    if not m or "place_in" not in ACTIONS:
        return None
    into = _container_called(scene_state, m.group("into"))
    if into is None and getattr(cfg, "TASK_WORDS_BY_KIND", False) and containers_on() and env is not None:
        # Not found yet (its first run): looked for by what it is called, as a place into it would.
        if find_container(env, scene_state, "_".join(_words(m.group("into"))) or m.group("into")):
            into = _container_called(scene_state, m.group("into"))
    if into is None:
        return None
    things = [one for t in re.split(r"\s*,\s*|\s+and\s+", m.group("things")) if _words(t)
              for one in _each_of(scene_state, t)]
    if not things:
        return None
    actions: list = []
    taken: set = set()
    picked: list = []
    places: list = []                                   # (name, xy) of each planned so far
    for said in things:
        obj = _thing_called(scene_state, said, taken)
        if obj is None:
            name = "_".join(_words(said))
            if not look_for(env, scene_state, name):
                if partial:
                    print(f"  [Plan] \"{said.strip()}\": not found on the table — left out.")
                    continue
                print(f"  [Plan] \"{said.strip()}\": not found on the table — planned as usual instead.")
                return None
            obj = scene_state.find(name, among="objects")
            if obj is None:
                if partial:
                    continue
                return None
            same = next((t for t in places if obj.position is not None
                         and float(np.linalg.norm(np.asarray(obj.position[:2]) - t[1])) < 0.02), None)
            if same is not None:
                # What those words found is what an earlier one did: 2026-11-19 15:27, one figure on the
                # table found as "the red cap" and again as "the red ball", and picked for both.
                scene_state.objects.remove(obj)
                print(f"  [Plan] \"{said.strip()}\": what is found by those words is {same[0]}, already "
                      "planned — not on the table, left out.")
                continue
        taken.add(obj.name)
        if obj.position is not None:
            places.append((obj.name, np.asarray(obj.position[:2], float)))
        how = _how_picked(obj)
        picked.append(f"{obj.name} by {how}")
        actions.append({"action": how, "parameters": {"object_name": obj.name}})
        actions.append({"action": "place_in", "parameters": {"container": into}})
    if not actions:
        return None
    print("  [Plan] " + "; ".join(picked) + f", each into {into}.")
    return actions


# A deployment with one task runs it on these words too (TASKS).
_RUN_WORDS = ("go", "run", "start", "do it", "run it", "start it", "do the task", "run the task")


def _said(text) -> str:
    """Words as compared with a task's name: lower case, no punctuation, single spaces."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())


_LAST_TASK: list = [None]


def task_more(env, scene_state, done) -> list:
    """
    controller.Deployment.task_more: the last task laid out again on the table
    as it is now, less what was done — for a task's plurals ("the
    screwdrivers"): one the first look did not see (2026-11-20, a screwdriver
    lying along the table's black tape) is found once the others are away.
    [] when there is nothing more, or no such task.
    """
    words = _LAST_TASK[0]
    if not words or not getattr(cfg, "TASK_WORDS_BY_KIND", False):
        return []
    plan = quick_plan(words, env, scene_state, partial=True) or []
    done = {str(d) for d in done}
    out, i = [], 0
    while i + 1 < len(plan):
        pick, put = dict(plan[i]), plan[i + 1]
        name = str(pick.get("parameters", {}).get("object_name"))
        # By where it is, not what it is called: one on the table still, not in a container — the second
        # screwdriver, found by name as "screwdriver" too, was left out as done (2026-11-20 17:51).
        here = [o for o in scene_state.objects if o.name == name and o.position is not None
                and not (containers_on() and _keeper().holds(scene_state, o))]
        if here:
            obj = here[-1]
            if obj.name in done:
                taken = {str(o.name) for o in scene_state.objects} | done
                k = 2
                while f"{name}_{k}" in taken:             # exact names: find() matches loosely, and never ran out
                    k += 1
                obj.name = f"{name}_{k}"                  # its own name, the done one's kept apart
            pick["parameters"] = {**pick.get("parameters", {}), "object_name": obj.name}
            out += [pick, put]
        i += 2
    return out


def task_plan(words: str, env, scene_state) -> "list | None":
    """
    controller.Deployment.task_plan from cfg.TASKS: the task these words name
    (its name as written, case and punctuation aside; with one task, also
    "go", "run", "start"), laid out on the table as it is now — a sentence by
    the quick plan, steps with each object and container found by what they
    are called. None: none of its tasks.
    """
    tasks = dict(getattr(cfg, "TASKS", None) or {})
    if not tasks:
        return None
    said = _said(words)
    name = next((t for t in tasks if _said(t) == said), None)
    if name is None and len(tasks) == 1 and said in _RUN_WORDS:
        name = next(iter(tasks))
    if name is None:
        return None
    task = tasks[name]
    _LAST_TASK[0] = task if isinstance(task, str) else None
    print(f"  [Plan] Task \"{name}\".")
    if isinstance(task, str):
        got = quick_plan(task, env, scene_state)
        if got is None:
            print(f"  [Plan] \"{name}\": nothing in it could be laid out on the table as it is.")
        return got
    return _laid_out(list(task), env, scene_state)


def _laid_out(steps: list, env, scene_state) -> "list | None":
    """A task's written steps on the table as it is: objects and containers found by what they are called."""
    actions: list = []
    taken: set = set()
    skip_put = False
    for step in steps:
        step = dict(step)
        action = str(step.pop("action"))
        params = dict(step.pop("parameters", {}) or {})
        params.update(step)
        if action in ("place_in", "place", "put_beside") and skip_put:
            skip_put = False
            continue                                       # what it was to put was not found
        skip_put = False
        if "container" in params:
            into = _container_called(scene_state, str(params["container"]))
            if into is None:
                print(f"  [Plan] \"{params['container']}\": no such container kept — the task stops here.")
                return actions or None
            params["container"] = into
        if "object_name" in params:
            said = str(params["object_name"])
            obj = _thing_called(scene_state, said, taken)
            if obj is None:
                name = "_".join(_words(said))
                obj = scene_state.find(name, among="objects") if look_for(env, scene_state, name) else None
            if obj is None:
                print(f"  [Plan] \"{said.strip()}\": not found on the table — left out.")
                skip_put = action.startswith("pick")
                continue
            taken.add(obj.name)
            params["object_name"] = obj.name
            if action == "pick":
                action = _how_picked(obj)                  # the way its kind is picked
        actions.append({"action": action, "parameters": params})
    return actions or None


def _ask_held(frame):
    """ParallelJaw.ask_held: the vision model asked whether the wrist camera sees the gripper carrying something."""
    from mrcf.perception import grasp_diagnosis

    return grasp_diagnosis.held(frame, call_vlm, model=cfg.MODEL)


if isinstance(POLICIES, ParallelJaw) and getattr(cfg, "HELD_BY_MODEL", False):
    POLICIES.ask_held = _ask_held
if isinstance(POLICIES, ParallelJaw) and containers_on():
    POLICIES.walls = lambda: _keeper().walls()      # what a move across keeps the jaws above (transit_over_walls)

from mrcf import skill_executor as _skill_executor  # noqa: E402 — the deployment's say, as the policies above

_skill_executor.DESCRIBE_FAILURES = bool(getattr(cfg, "DESCRIBE_FAILURES", True))
from mrcf import motion_guard as _motion_guard  # noqa: E402

_motion_guard.STOP_PUSHING = bool(getattr(cfg, "GUARD_STOPS_PUSHING", False))
manipulation.PARTS_INSIDE_ARE_ITS_OWN = bool(getattr(cfg, "PARTS_INSIDE_ARE_ITS_OWN", False))
manipulation.OUTLINE_BEFORE_PICK = bool(getattr(cfg, "OUTLINE_BEFORE_PICK", False))


def _refuse(skill: str, why: str) -> bool:
    """Say why a skill gave up before moving, to the user and to the planner's next try; False."""
    print(f"  [REFUSED] {skill}: {why}")
    report(skill, False, why, refused=True)
    return False


def _radius(obj) -> float:
    geom = obj.geometry
    return geom["base_diameter_m"] / 2 if geom and geom.get("base_diameter_m") else DEFAULT_RADIUS_M


_SIDE_WORDS = {"left": "left", "right": "right", "front": "front", "in front": "front", "behind": "behind",
               "back": "behind", "next": "any", "beside": "any", "near": "any", "by": "any", "any": "any"}


def _side(said) -> str:
    """The planner's word for a side, as one of places.SIDES."""
    text = str(said or "any").lower().replace("_", " ")
    for word, side in _SIDE_WORDS.items():
        if word in text:
            return side
    raise ValueError(f"'{said}' is not a side: left, right, front, behind, or any")


@register_action(
    "place_beside",
    description=(
        "Put the held object down on free table beside another object: 'put it next to the cup' (side any), "
        "'left of the box', 'in front of the bottle'. Left, right, front and behind are as the person "
        "giving the command sees them. Finds a free spot on that side, clear of everything else."
    ),
    parameters={
        "object_name": "Exact name of the object to put it beside.",
        "side": "left, right, front, behind, or any (just next to it).",
    },
    conditions={"pre": ["something_carried", "object_exists"]},
)
def skill_place_beside(env: RobotEnv, scene_state, object_name, side="any", **_):
    obj = scene_state.find(object_name, among="objects")
    held = scene_state.held_object()
    if obj is None or obj.position is None:
        return _refuse("place_beside", f"'{object_name}' is not on the table as far as the last scan saw")
    if held is None:
        return _refuse("place_beside", "not holding anything: pick something first")
    side = _side(side)
    others = [(o.position, _radius(o)) for o in scene_state.objects
              if o is not obj and o is not held and o.position is not None]

    def reachable(xy) -> bool:
        finder = env.approach_orientation_for if isinstance(env, OrientationSearch) else None
        hover = np.array([xy[0], xy[1], cfg.TABLE_Z + cfg.TALLEST_OBJECT_M])
        return finder is None or finder([hover]) is not None

    spot = places.beside(obj.position, _radius(obj), _radius(held), side, others, cfg.WORKSPACE_BOUNDS,
                         base_xy=env.base_xy, gap_m=cfg.BESIDE_GAP_M, seen_from=cfg.LEFT_RIGHT_SEEN_FROM,
                         reachable=reachable)
    if spot is None:
        return _refuse("place_beside", f"no free room {'beside' if side == 'any' else 'to the ' + side + ' of'} "
                                       f"{obj.name} that the arm can reach")
    name = f"beside_{obj.name}"
    scene_state.targets = [t for t in scene_state.targets if t.name != name] + [
        SceneObject(name, f"free table {'beside' if side == 'any' else side + ' of'} {obj.name}",
                    interactable=False, position=np.array([spot[0], spot[1], cfg.TABLE_Z]))]
    return ACTIONS["place"]["fn"](env=env, scene_state=scene_state, target_name=name)


if containers_on():
    @register_action(
        "find",
        description=(
            "Look for something the scan does not list, by what the user calls it — a tub, a bowl, a box, a tray to "
            "put things into — and keep it as a container (place_in puts things into it)."
        ),
        parameters={"what": "What the user calls it ('the plastic tub')."},
        always=True,
    )
    def skill_find(env: RobotEnv, scene_state, what, **_):
        return find_container(env, scene_state, what)


# ── Marked spots ──────────────────────────────────────────────────────────────

@register_action(
    "mark_place",
    description=(
        "Remember a spot on the table by a name, to put things there later ('this spot is the tray', "
        "'call where the cup is the parking spot'): where the gripper is now, or where an object stands."
    ),
    parameters={
        "name": "The name to call the spot ('tray').",
        "object_name": ("An object standing on the spot, only if the user named one ('where the cup is'). "
                        "For 'this spot' or 'here', leave it out: the spot is under the gripper."),
    },
    always=True,
)
def skill_mark_place(env: RobotEnv, scene_state, name, object_name=None, **_):
    name = places.clean_name(name)
    if not name:
        print("  [Error] A spot needs a name.")
        return False
    if object_name:
        obj = scene_state.find(object_name, among="objects")
        if obj is None or obj.position is None:
            return _refuse("mark_place", f"'{object_name}' is not on the table as far as the last scan saw")
        xy = np.asarray(obj.position, float)[:2]
    else:
        xy = np.asarray(env.get_ee_pos(), float)[:2]
    if not _in_bounds(xy):
        return _refuse("mark_place", f"({xy[0]:.3f}, {xy[1]:.3f}) m is outside the workspace")
    marked = places.load(cfg.PLACES_PATH)
    marked[name] = (float(xy[0]), float(xy[1]))
    places.save(cfg.PLACES_PATH, marked)
    scene_state.targets = [t for t in scene_state.targets if t.name != name] + places.targets({name: xy}, cfg.TABLE_Z)
    print(f"  Marked '{name}' at ({xy[0]:+.3f}, {xy[1]:+.3f}) m.")
    return True


@register_action(
    "forget_place",
    description="Forget a marked spot by its name.",
    parameters={"name": "The spot's name."},
    always=True,
)
def skill_forget_place(env: RobotEnv, scene_state, name, **_):
    name = places.clean_name(name)
    marked = places.load(cfg.PLACES_PATH)
    if marked.pop(name, None) is None:
        print(f"  [Error] No spot called '{name}'. Marked: {', '.join(marked) or 'none'}.")
        return False
    places.save(cfg.PLACES_PATH, marked)
    scene_state.targets = [t for t in scene_state.targets if t.name != name]
    print(f"  Forgot '{name}'.")
    return True


@register_action(
    "freedrive",
    description=(
        "Let the user move the arm by hand for a while (to show a spot before marking it): "
        "the arm floats, then holds again where it was left."
    ),
    parameters={"seconds": "How long, in seconds (default 15)."},
    always=True,
)
def skill_freedrive(env: RobotEnv, scene_state, seconds=15, **_):
    from mrcf import capabilities

    if not isinstance(env, capabilities.Freedrive):
        # Not disable_torque: an arm that goes limp falls (an SO-101).
        print("  [Error] This arm cannot be moved by hand from here: it does not hold itself up when let go.")
        return False
    seconds = min(max(float(seconds), 1.0), 120.0)
    env.freedrive(True)
    try:
        print(f"  The arm is free to move by hand for {seconds:.0f} s.")
        time.sleep(seconds)
    finally:
        env.freedrive(False)
    print("  Holding again.")
    return True


# ── What things are ───────────────────────────────────────────────────────────

_HEIGHT = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|mm|centimet\w*|millimet\w*)")


@register_action(
    "identify",
    description=(
        "Record what an object is when the user says so ('that's a marker', 'the blue thing is a cup', "
        "'the bottle is 18 cm tall'). Remembered from then on, and trusted over the camera's guess."
    ),
    parameters={
        "object_name": "Exact name of the object being identified.",
        "what": "What the user says it is, with its colour and height if they gave them.",
    },
    conditions={"pre": ["object_exists"]},
)
def skill_identify(env: RobotEnv, scene_state, object_name, what, **_):
    obj = scene_state.find(object_name)
    mem = scene_state.memory
    if obj is None or mem is None or mem.get(obj.track_id) is None:
        print(f"  [Error] '{object_name}' is not a remembered object. Try 'observe' first.")
        return False
    text = str(what).lower().replace("_", " ")
    height = _HEIGHT.search(text)
    if height:
        text = text.replace(height.group(0), " ")
    # As the vision model's names are cleaned (naming.clean_what): letters and digits ("dice 6").
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", text).split() if w not in ("tall", "high", "a", "an", "the", "is")]
    colours = [w for w in words if w in cfg.COLOURS]
    label = " ".join(w for w in words if w not in cfg.COLOURS) or None
    track = mem.get(obj.track_id)
    if label or colours:
        mem.set_label(track.id, label, "user", attributes={"colour": colours[0]} if colours else None)
    if height:
        mm = float(height.group(1)) * (10.0 if height.group(2).startswith("c") else 1.0)
        if "height_mm" not in track.pinned:
            track.pinned.append("height_mm")         # the user's: no known size replaces it (known_sizes)
        mem.measured(track.id, height_mm=mm)
        obj.measured_shape = track.shape()
    old = obj.name
    obj.name = mem.name(track)
    obj.description = describe(track) + ("" if obj.position is not None else ", held in the gripper")
    print(f"  Noted: {old} is {naming.describe(track)} — now called {obj.name}.")
    return True


@register_action(
    "forget",
    description=(
        "Forget everything remembered about the objects, then look again. Use only when the user says "
        "they have cleared or rearranged the whole table."
    ),
    always=True,
)
def skill_forget(env: RobotEnv, scene_state, **_):
    if scene_state.memory is not None:
        scene_state.memory.clear()
    print("  Forgot every remembered object.")
    return skill_observe(env, scene_state)


# ── Scene / status ────────────────────────────────────────────────────────────

@register_action("observe", description="Re-scan the table with the camera to update what is where.", always=True)
def skill_observe(env: RobotEnv, scene_state, **_):
    print("  Moving clear of the camera...")
    goto_observe_pose(env)
    print("  Looking at the table...")
    ok = scan_scene(env, scene_state)
    print(scene_state.summary())
    return ok


@register_action("list_objects", description="Print the known objects, places and positions.", always=True)
def skill_list_objects(scene_state, **_):
    print()
    print(scene_state.summary())
    print()
    return True


@register_action("list_actions", description="Print all available robot actions.", always=True)
def skill_list_actions(**_):
    print("\nAvailable actions:")
    for name, info in ACTIONS.items():
        params = list(info["parameters"].keys())
        print(f"  {name}({', '.join(params)})")
        print(f"    {info['description']}")
    print()
    return True


def _outlines_drawn(frame: np.ndarray, scene_state) -> "np.ndarray | None":
    """The overhead picture with each object's outline, as the scene has it, drawn and named (to check the scan)."""
    import cv2

    out = np.ascontiguousarray(frame).copy()
    h, w = out.shape[:2]
    drew = False
    for o in scene_state.objects:
        line = ((o.geometry or {}).get("outlines") or {}).get(cfg.OBSERVE_CAMERA)
        if not line or not line.get("px"):
            continue
        sw, sh = (line.get("size") or [w, h])[:2]
        px = np.asarray(line["px"], float).reshape(-1, 2) * [w / float(sw), h / float(sh)]
        cv2.polylines(out, [px.astype(np.int32)], True, (0, 255, 0), 2)
        x, y = px.min(axis=0)
        cv2.putText(out, o.name, (int(x), max(12, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
        drew = True
    return out if drew else None


@register_action(
    "snapshot",
    description="Save a picture from each camera as things stand now, and say where (to see what the arm did).",
    always=True,
)
def skill_snapshot(env: RobotEnv, scene_state=None, **_):
    cams = [(cfg.OBSERVE_CAMERA, cfg.CAPTURE_W_OVERHEAD, cfg.CAPTURE_H_OVERHEAD)]
    if isinstance(POLICIES, ParallelJaw) and POLICIES.spec.camera is not None:
        cams.append((POLICIES.spec.camera, POLICIES.spec.render_w, POLICIES.spec.render_h))
    kept = []
    for cam, w, h in cams:
        try:
            frame = env.render_image(cam, width=w, height=h)
        except Exception as exc:                            # noqa: BLE001 — one camera's trouble is said
            print(f"  [Snapshot] {cam}: no picture ({exc}).")
            continue
        path = flight_recorder.keep_picture(f"snapshot_{cam}", frame)
        if path is not None:
            kept.append(f"{cam}: {path}")
        if cam == cfg.OBSERVE_CAMERA and scene_state is not None:
            drawn = _outlines_drawn(np.asarray(frame), scene_state)
            if drawn is not None:
                path = flight_recorder.keep_picture("snapshot_outlines", drawn)
                if path is not None:
                    kept.append(f"what the scan outlined: {path}")
    print("  [Snapshot] " + ("; ".join(kept) if kept else "nothing kept (no recorder running, or pictures off)") + ".")
    return bool(kept)


@register_action(
    "status",
    description="Print the robot's live readings: joint angles, gripper position, gripper opening.",
    always=True,
)
def skill_status(env: RobotEnv, **_):
    ee, width = env.get_ee_pos(), env.get_gripper_width()
    print("\nRobot status (live sensor readings):")
    print(f"  Grasp point     : [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] m")
    print(f"  Gripper opening : {width * 1000:.1f} mm")
    print("  Joint angles    :")
    for name, angle in env.get_joint_angles().items():
        print(f"    {name:<16}: {np.degrees(angle):+8.2f} deg")
    print()
    return True


@register_action(
    "answer",
    description=(
        "Answer a question about the table by looking at the camera: what is there, colours, shapes, "
        "positions, anything visible."
    ),
    parameters={"question": "The question to answer from the camera's view."},
    always=True,
)
def skill_answer(env: RobotEnv, question, **_):
    try:
        img_b64 = render_to_base64(env)
    except Exception as exc:                          # noqa: BLE001 — answer without the picture, saying so
        print(f"  [Answer] No picture: {exc}")
        img_b64 = None
    prompt = (f"You are looking at a table from the robot's '{cfg.OBSERVE_CAMERA}' camera, above it at an angle.\n"
              f"Answer concisely from what you can see.\n\nQuestion: {question}")
    response = call_vlm(prompt, img_b64, json_mode=False)
    print(f"\n  {response}\n")
    return bool(response)
