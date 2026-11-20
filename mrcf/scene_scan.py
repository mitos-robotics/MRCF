"""
scene_scan.py — a surface scan (mrcf/perception/surface_scan.py) into the object
memory and the world model: the whole perception step for a deployment with no
target of its own to find objects by (no chessboard).

mrcf/perception/ measures; this decides who each measured object is (object_memory)
and what the planner sees (vlm.SceneState). Kept apart so the measuring can be
used with any camera and any memory, or none.
"""

from __future__ import annotations

import numpy as np

from mrcf import flight_recorder
from mrcf.capabilities import FixedViews
from mrcf.object_memory import Detection
from mrcf.perception.surface_scan import scan


def detections(found: list, place=None, view=None) -> list:
    """
    object_memory Detections for scan()'s results. place(xy_m) -> name, if the
    deployment names places. view (plane_geometry.PlaneView): also which way
    each outline's sides face (extra["faces_deg"]), for objects with sides,
    and how long and wide across it is on the surface (extra["length_mm"],
    ["widest_mm"], ["narrowest_mm"]: plane_geometry.long_extent_mm).
    """
    from mrcf.perception import plane_geometry as pg
    out = []
    for f in found:
        meas = f["measurement"]
        xy = meas.base_world_m
        colour = f["appearance"]
        # Matched on chroma only: lightness follows the room's light from scan
        # to scan (a piece's read 139 and 168 on two scans minutes apart).
        out.append(Detection(position=(float(xy[0]), float(xy[1])),
                             place=place(xy) if place is not None else None,
                             foot_mm=2 * meas.base_radius_mm,
                             height_mm=meas.height_mm,
                             appearance={"a": colour["a"], "b": colour["b"]},
                             extra={"lightness": colour["L"],
                                    "faces_deg": pg.side_direction_deg(f["mask"], view) if view is not None else None,
                                    # How long and how wide across, on the surface: what the jaws close
                                    # across on a thing lying along it (a tool), not the round foot above.
                                    **((pg.long_extent_mm(f["mask"], view) or {}) if view is not None else {})}))
    return out


def _outline_px(mask) -> "list | None":
    """An outline's boundary as [[u, v], ...] pixels (its largest part, simplified to within a pixel); None when empty."""
    import cv2

    m = np.asarray(mask, bool).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return [[int(u), int(v)] for u, v in c] if len(c) >= 3 else None


def _other_views(env) -> dict:
    """A frame from each of the arm's other calibrated fixed cameras (RigEnv.fixed_views), {name: RGB}: taken as the scan's was."""
    views = {}
    for name, (w, h) in (env.fixed_views() if isinstance(env, FixedViews) else {}).items():
        try:
            views[name] = env.render_image(name, int(w), int(h))
            flight_recorder.keep_picture(f"scan_{name}", views[name])     # what a later check compares with
        except Exception as exc:                 # noqa: BLE001 — a view that fails is one fewer, not a failed scan
            print(f"  [Scan] No frame from the {name} camera ({exc}): the check after a place goes without it.")
    return views


def _arm_now(env):
    """Where the arm's body is (RobotEnv.arm_shape: points and radii), None when it does not say."""
    try:
        return env.arm_shape()
    except Exception:                            # noqa: BLE001 — only for leaving the arm out of a comparison
        return None


# Looking past the arm, not waiting for it to leave the camera's view: where its
# body is in the picture (RobotEnv.arm_shape, motion_guard.arm_mask), grown by
# ARM_MARGIN_M; an outline ARM_OVERLAP or more on it is the arm, or hidden by it,
# and a remembered object under it is not seen there, not gone.
ARM_MARGIN_M = 0.02
ARM_OVERLAP = 0.3


def arm_in_view(env, camera: str, width: int, height: int, margin_m: float = ARM_MARGIN_M) -> "np.ndarray | None":
    """Where the arm's body is in the camera's picture now, bool (height, width); None when either is not known."""
    from mrcf.motion_guard import arm_mask

    shape = _arm_now(env)
    model = env.camera_model(camera, width, height)
    if not shape or model is None:
        return None
    return arm_mask([(0.0, np.array([p for p, _ in shape], float), np.array([r for _, r in shape], float))],
                    model, (height, width), margin_m)


def arm_hides(env, camera: str, width: int, height: int, bounds, around=None,
              radius_m: float = 0.05) -> "tuple[float | None, bool]":
    """
    How much of the work surface (bounds: WORKSPACE_BOUNDS, x and y) the arm
    hides from the camera where it is now — a share of that rectangle in the
    picture — and whether it hides any of the disc radius_m round `around`
    (x, y). (None, False): not known (no arm shape, no camera model). What a
    deployment decides by whether to look past the arm or move it out of view.
    """
    import cv2

    from mrcf.perception.plane_geometry import PlaneView

    arm = arm_in_view(env, camera, width, height)
    model = env.camera_model(camera, width, height)
    if arm is None or model is None:
        return None, False
    view = PlaneView.from_camera_model(model)
    origin = np.zeros(2) if view.origin_m is None else np.asarray(view.origin_m, float)[:2]

    def px(xys) -> np.ndarray:
        pts = (np.asarray(xys, float) - origin) * 1000.0
        return view.project(np.c_[pts, np.zeros(len(pts))])

    work = np.zeros(arm.shape, np.uint8)
    corners = px([(bounds.x_min, bounds.y_min), (bounds.x_max, bounds.y_min),
                  (bounds.x_max, bounds.y_max), (bounds.x_min, bounds.y_max)])
    if not np.isfinite(corners).all():
        return None, False
    cv2.fillPoly(work, [np.round(corners).astype(np.int32)], 1)
    share = float((arm & (work > 0)).sum()) / max(1.0, float(work.sum()))
    spot = False
    if around is not None:
        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = px(np.asarray(around, float)[:2] + radius_m * np.c_[np.cos(t), np.sin(t)])
        if np.isfinite(ring).all():
            disc = np.zeros(arm.shape, np.uint8)
            cv2.fillPoly(disc, [np.round(ring).astype(np.int32)], 1)
            spot = bool((arm & (disc > 0)).any())
    return share, spot


def share_on_arm(arm: "np.ndarray | None", outline_px, size) -> float:
    """The share of an outline (px, in a picture of size (w, h)) that the arm's mask (arm_in_view) covers; 0 with none."""
    import cv2

    if arm is None or outline_px is None:
        return 0.0
    m = np.zeros(arm.shape, np.uint8)
    scale = np.array([arm.shape[1] / float(size[0]), arm.shape[0] / float(size[1])])
    cv2.fillPoly(m, [np.round(np.asarray(outline_px, float) * scale).astype(np.int32)], 1)
    return float((arm & (m > 0)).sum()) / max(1.0, float(m.sum()))


def hidden_by(mask: np.ndarray, view, table_z: float):
    """hidden(xy) for ObjectMemory.observe: whether a point of the surface is under the arm in this picture."""
    origin = np.zeros(2) if view.origin_m is None else np.asarray(view.origin_m, float)[:2]
    h, w = mask.shape

    def hidden(xy) -> bool:
        p = (np.asarray(xy, float)[:2] - origin) * 1000.0
        px = view.project(np.array([[p[0], p[1], 0.0]]))[0]
        if not np.isfinite(px).all():
            return False
        u, v = int(round(px[0])), int(round(px[1]))
        return 0 <= u < w and 0 <= v < h and bool(mask[v, u])
    return hidden


def observe(env, scene_state, memory, camera: str, width: int, height: int, x_range_m, y_range_m, *,
            reference=None, surface_fraction=None, place=None, describe=None, reachable=None,
            identify=None, table_z: float = 0.0, boxed=None, max_drift_px: "float | None" = None,
            past_arm: bool = False) -> bool:
    """
    The whole perception step for a deployment with no board of its own: scan
    the surface, match what is there against the memory, and rebuild
    scene_state.objects from it — present objects at their measured feet, a
    held one (position None) kept as it was. Placement targets are left alone.

      place(xy_m) -> str | None       a deployment's name for where something stands
      describe(track) -> str          its description for the planner
      reachable(xy_m) -> bool         whether the arm can get there
      identify(frame, found, tracks)  say what each found object is (memory.set_label),
                                      from the frame and its outline (found[i]["mask"]),
                                      before it is named
      boxed(track) -> (height_mm, side_mm) | None   an object that is a box of that size (a
                                      block): which way its sides face (SceneObject.faces_deg),
                                      and where it stands by its top face (box_centre_xy)

    past_arm: the arm may be in the camera's view (arm_in_view): what is on it
    is left out, and what the memory has under it is hidden, not gone.

    False when the surface cannot be scanned (no camera model, no segmenter).
    surface_scan.CameraMoved, with nothing changed, when max_drift_px is given
    and the camera has moved further than that since the reference was taken.
    """
    from mrcf.vlm import SceneObject

    got = scan(env, camera, width, height, x_range_m, y_range_m,
               reference=reference, surface_fraction=surface_fraction, max_drift_px=max_drift_px)
    if got is None:
        return False
    frame, found = got
    from mrcf.perception import plane_geometry as pg
    model = env.camera_model(camera, width, height)
    view = pg.PlaneView.from_camera_model(model) if model is not None else None
    hidden = None
    arm = arm_in_view(env, camera, width, height) if past_arm else None
    if arm is not None and view is not None:
        kept = [f for f in found
                if float((np.asarray(f["mask"], bool) & arm).sum()) < ARM_OVERLAP * max(1.0, float(np.asarray(f["mask"], bool).sum()))]
        if len(kept) < len(found):
            print(f"  [Scan] Looking past the arm: {len(found) - len(kept)} outline(s) on it left out.")
        found = kept
        hidden = hidden_by(arm, view, table_z)
    tracks, changes = memory.observe(detections(found, place, view), hidden=hidden)
    if identify is not None:
        identify(frame, found, tracks)
    memory.restack()                             # heights learned this look: what stands on them moves
    scene_state.memory = memory
    scene_state.look_frame = {"camera": camera, "frame": frame,
                              "positions": {t.id: tuple(map(float, t.position[:2])) for t in tracks
                                            if t.position is not None},
                              "views": _other_views(env), "arm": _arm_now(env)}
    held_before = {o.track_id: o for o in scene_state.objects if o.position is None and o.track_id}
    scene_state.objects = []
    obj: SceneObject | None
    for f, track in zip(found, tracks):
        meas = f["measurement"]
        xy = meas.base_world_m
        desc = describe(track) if describe is not None else (
            f"{track.label or 'unidentified object'}, foot {2 * meas.base_radius_mm:.0f} mm")
        # On top of a stack, its foot is up on the one below.
        foot_z = table_z + (track.raised_mm / 1000.0 if track.on else 0.0)
        obj = SceneObject(memory.name(track), desc, interactable=True,
                          position=np.array([xy[0], xy[1], foot_z]), pixel_xy=list(meas.base_px))
        obj.track_id = track.id
        obj.reachable = True if reachable is None else bool(reachable(xy))
        obj.square = track.place  # type: ignore[attr-defined]  # a deployment's extra attribute, not a field
        obj.geometry = {"base_diameter_m": 2 * meas.base_radius_mm / 1000.0}
        outline = _outline_px(f["mask"])
        if outline is not None:
            # What a look for its height fits its shape to, with the wrist camera's
            # (Target.outlines, mrcf/policies/parallel_jaw/height_looks.py).
            obj.geometry["outlines"] = {camera: {"size": [int(width), int(height)], "px": outline}}
        obj.measured_shape = track.shape() if track.shape_current() else None
        size = boxed(track) if boxed is not None else None
        obj.faces_deg = track.extra.get("faces_deg") if size is not None else None
        if size is not None and obj.faces_deg is not None and view is not None:
            # A box stands where its top face says, not where a round foot fitted to
            # its outline would (HISTORY.md: plane_geometry.box_centre_xy).
            top_mm = (track.raised_mm if track.on else 0.0) + float(size[0])
            centre = pg.box_centre_xy(f["mask"], view, top_mm, float(size[1]), obj.faces_deg)
            if centre is not None:
                obj.position = np.array([centre[0], centre[1], foot_z])
                # Kept, for while something stands on it and hides it: with where
                # the memory had it then, so a move since makes it stale.
                track.extra["box_centre"] = {"at": list(track.position), "xy": [float(centre[0]), float(centre[1])]}
        scene_state.objects.append(obj)
    # Under the arm in this picture (past_arm): where the memory has them, as they were.
    for track, _ in changes.get("hidden", []):
        if track.position is None:
            continue
        desc = describe(track) if describe is not None else (track.label or "object")
        obj = SceneObject(memory.name(track), f"{desc}, hidden by the arm this look (where it was)",
                          interactable=True, position=np.array([track.position[0], track.position[1], table_z]))
        obj.track_id = track.id
        obj.reachable = True if reachable is None else bool(reachable(track.position))
        obj.square = track.place  # type: ignore[attr-defined]  # a deployment's extra attribute, not a field
        obj.geometry = {"base_diameter_m": track.foot_mm / 1000.0} if track.foot_mm else None
        obj.measured_shape = track.shape() if track.shape_current() else None
        scene_state.objects.append(obj)
    # Under a stack, out of the camera's sight: where the memory has them.
    seen_ids = {t.id for t in tracks}
    for track in memory.in_state("present"):
        if track.id in seen_ids or track.position is None or not any(
                t.on == track.id for t in memory.in_state("present")):
            continue
        desc = describe(track) if describe is not None else (track.label or "object")
        box = track.extra.get("box_centre") or {}
        xy = box["xy"] if box.get("at") == list(track.position) else track.position
        obj = SceneObject(memory.name(track), f"{desc}, under what stands on it", interactable=True,
                          position=np.array([xy[0], xy[1], table_z + track.raised_mm / 1000.0]))
        obj.track_id = track.id
        obj.reachable = True if reachable is None else bool(reachable(track.position))
        obj.square = track.place  # type: ignore[attr-defined]  # as above
        obj.geometry = {"base_diameter_m": track.foot_mm / 1000.0} if track.foot_mm else None
        obj.measured_shape = track.shape() if track.shape_current() else None
        obj.faces_deg = track.extra.get("faces_deg") if boxed is not None and boxed(track) is not None else None
        scene_state.objects.append(obj)
    for track in memory.in_state("held"):
        obj = held_before.get(track.id)
        if obj is None:
            obj = SceneObject(memory.name(track), (describe(track) if describe else track.label or "object")
                              + ", held in the gripper", interactable=True, position=None)
            obj.track_id, obj.square, obj.geometry, obj.measured_shape = track.id, track.place, None, None  # type: ignore[attr-defined]  # as above
        scene_state.objects.append(obj)
    seen = set()
    for o in scene_state.objects:
        if o.name in seen:
            o.name = f"{o.name}_{o.track_id.replace('-', '')}"
        seen.add(o.name)
    by_track = {o.track_id: o for o in scene_state.objects if o.track_id}
    for o in scene_state.objects:                # what each stands on, by the name the planner sees
        track = memory.get(o.track_id) if o.track_id else None
        o.on = by_track[track.on].name if track is not None and track.on in by_track else None
    n_new, n_moved, n_gone = len(changes["new"]), len(changes["moved"]) + len(changes["returned"]), len(changes["missing"])
    print(f"  [Surface] {len(found)} object(s) on the surface"
          + (f" — {n_new} new, {n_moved} moved, {n_gone} gone" if (n_new or n_moved or n_gone) else ""))
    for o in scene_state.objects:
        if o.position is not None:
            print(f"           {o.name} @ ({o.position[0]:+.3f}, {o.position[1]:+.3f})  "
                  f"foot {o.geometry['base_diameter_m'] * 1000:.0f} mm"
                  + ("" if o.reachable else "  [OUT OF REACH]"))
    return True
