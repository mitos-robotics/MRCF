"""
place_in.py — put what a two-finger gripper holds into a container (a tub, a
bowl, a tray): over its walls, lowered inside, let go above its floor, and
back up over the walls.

    spots    where inside to let go: points of the opening at least `room`
             in from its rim — half the jaws' opening and a finger, or half
             what is held if that is wider, and spec.container_wall_clearance_m
             — and clear of what is in it already. With
             spec.place_in_over_contents, a thing held by its outline (laid,
             below) is laid over what is there, and anything else that finds
             no clear spot is let go from over the walls onto what is there.
             The roomiest first, then those the arm comes at
             straight down before those it leans to; nearest the gripper
             among equals
    laid     what is held has an outline of its own (GraspPlan.held_outline,
             from a lying pick): each spot and wrist turn is checked by that
             outline, not by a circle as wide as it is long (laid_in). Its
             long edge goes along the container's where it fits, else as
             near along as fits, nearest the arm's base; where it fits no
             way flat, it is dropped from over the walls, still aligned
    over     straight up where the gripper is, until what it holds clears the
             walls by spec.container_wall_clearance_m, before anything moves
             sideways: the way across in joint space is not a straight line
    rim      the walls' height not measured (Opening.rim_measured): the wrist
             camera looks at a wall from two poses that differ only by a turn
             about the arm's base — every other joint the same in both, so
             where the arm thinks the camera is errs alike in both — and the
             height where the edges round that wall's line in one picture land
             on the other's is the rim's (perception/plane_sweep.sweep_edges).
             The wall looked at is one the turn carries the camera across:
             along it, a straight edge looks the same from everywhere. Not
             seen: the walls are as high as they may be (Opening.rim_m)
    in       over the spot, the walls cleared; then, the rim measured and not
             asked to drop, straight down inside until what is held is just
             over the floor — but the gripper's body (spec.free_depth_m above
             the jaws, or spec.container_gripper_inside_m when set) no lower
             than over the rim. Let go there, and straight
             back up. Asked to drop (drop=True), or the rim not measured: let
             go from over the walls (a sturdy object)

Where the floor is: Opening.floor_m, else no higher than spec.container_floor_m
above the surface the container stands on — what is held is let go that far
up at most, and drops the rest.

What it is given is geometry, as every policy is (policies.base): the opening,
what is held, and what stands in the container already, as Obstacles. Whether
it landed inside is the caller's to look at.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING

import numpy as np

from mrcf.capabilities import CameraAtPose, GripperRotation, GripperTilt, OrientationSearch
from mrcf.policies.base import Measured, Opening, Outcome, Target

if TYPE_CHECKING:
    from mrcf.policies.parallel_jaw import ParallelJaw
    from mrcf.robot_env import RobotEnv

TAG = "[PlaceIn]"


def _polygon(opening: Opening) -> np.ndarray:
    pts = np.asarray(opening.outline_xy, float).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("an opening needs at least three points of outline")
    return pts


def inside_by(polygon: np.ndarray, xy) -> float:
    """How far xy is inside the polygon (m, surface plane); negative: outside, by that much."""
    import cv2

    poly = (np.asarray(polygon, float) * 1e5).astype(np.float32).reshape(-1, 1, 2)
    return float(cv2.pointPolygonTest(poly, (float(xy[0]) * 1e5, float(xy[1]) * 1e5), True)) / 1e5


def nearest_wall(polygon: np.ndarray, xy) -> "tuple[np.ndarray, np.ndarray]":
    """The point of the polygon's edge nearest xy, and the unit normal there pointing inside it."""
    poly = np.asarray(polygon, float)
    p = np.asarray(xy, float)[:2]
    best = None
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        t = 0.0 if float(ab @ ab) == 0.0 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if best is None or d < best[0]:
            best = (d, q, ab)
    assert best is not None                     # _polygon: at least three points
    _, q, ab = best
    n = np.array([-ab[1], ab[0]]) / (np.linalg.norm(ab) or 1.0)
    if inside_by(poly, q + 0.005 * n) < inside_by(poly, q - 0.005 * n):
        n = -n
    return q, n


def spots(policy: "ParallelJaw", env, opening: Opening, room_m: float, obstacles=(), held_half_m: float = 0.02,
          step_m: float = 0.02) -> list:
    """
    Where inside the opening to let go, best first: [(x, y)] at least room_m
    in from the rim and clear of what stands in it (obstacles: half what is
    held, their own foot, and the wall clearance). The roomiest first (in
    steps of a centimetre, no roomier than room_m + 3 cm counting for more),
    then the arm's straight-down zone before its leaning ones, then nearest
    the gripper.
    """
    s = policy.spec
    poly = _polygon(opening)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    here = np.asarray(env.get_ee_pos(), float)[:2]
    zones = {name: i for i, name in enumerate(("mid", "near", "far"))}
    out = []
    for x in np.arange(lo[0] + step_m / 2, hi[0], step_m):
        for y in np.arange(lo[1] + step_m / 2, hi[1], step_m):
            xy = np.array([x, y])
            depth = inside_by(poly, xy)
            if depth < room_m:
                continue
            if any(float(np.linalg.norm(xy - np.asarray(o.position, float)[:2]))
                   < held_half_m + (o.radius_m or s.unmeasured_obstacle_radius_m) + s.container_wall_clearance_m
                   for o in obstacles):
                continue
            roomier = int(min(depth - room_m, 0.03) * 100)          # whole centimetres, up to 3
            zone = zones.get(policy.zone(env, xy), len(zones))
            out.append((-roomier, zone, float(np.linalg.norm(xy - here)), (float(x), float(y))))
    out.sort()
    return [xy for *_, xy in out]


def _orient(env, targets: list, prefer_tilt_deg: float = 0.0, turn_deg: float = 0.0):
    """
    env.approach_orientation_for with only the keyword arguments the arm's
    search takes (capabilities.OrientationSearch: a backend may leave some out
    — the UR5's takes no prefer_tilt_deg); None where the turn cannot be asked.
    """
    from mrcf.policies.parallel_jaw.planning import _takes

    fn = env.approach_orientation_for
    kw: dict = {}
    if _takes(fn, "prefer_tilt_deg"):
        kw["prefer_tilt_deg"] = prefer_tilt_deg
    if turn_deg:
        if not _takes(fn, "turn_deg"):
            return None
        kw["turn_deg"] = turn_deg
    return fn(list(targets), **kw)


TURN_STEP_DEG = 5.0      # a held thing's outline tried this many degrees of turn apart, diagonals among them


def _inside_map(poly: np.ndarray, res_m: float = 0.002, pad_m: float = 0.01):
    """xy points (N, 2) -> how far inside poly each is (m; outside: minus how far out, to pad_m; -1 beyond)."""
    import cv2

    lo = poly.min(axis=0) - pad_m - 0.01
    size = np.ceil((poly.max(axis=0) + pad_m + 0.01 - lo) / res_m).astype(int) + 1
    mask = np.zeros((int(size[1]), int(size[0])), np.uint8)
    cv2.fillPoly(mask, [np.round((poly - lo) / res_m).astype(np.int32).reshape(-1, 1, 2)], 1)
    dist = (cv2.distanceTransform(mask, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)) * res_m

    def at(pts: np.ndarray) -> np.ndarray:
        ij = np.round((np.asarray(pts, float) - lo) / res_m).astype(int)
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < size[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < size[1])
        out = np.full(ij.shape[:-1], -1.0)
        out[ok] = dist[ij[ok][:, 1], ij[ok][:, 0]]
        return out
    return at


def laid_in(policy: "ParallelJaw", env, opening: Opening, outline_uv, spots_xy: list, z: float, obstacles=(),
            prefer_jaw=None, most: int = 8, clearance_m: "float | None" = None) -> "tuple[list, list]":
    """
    Where, and at what wrist turn, what is held — its outline in the jaws'
    frame (GraspPlan.held_outline) — lies wholly inside the opening,
    spec.container_outline_clearance_m in from its walls and clear of what is
    in it: ([xy], [turn deg]), along the container first (else as near along
    as fits), then nearest the arm's base, then the roomiest (the most
    clearance to the walls), at most `most`, each as the arm really turns
    there. prefer_jaw (the way the jaws open now) is kept for the caller's
    say; it does not decide. A long tool goes in along the container or corner to corner
    (user, 2026-11-20: "it can angle it diagonally if it wants more space"),
    not where a circle as wide as it is long would fit.
    """
    from mrcf.capabilities import JawDirection

    if not (isinstance(env, OrientationSearch) and isinstance(env, JawDirection)):
        return [], []
    s = policy.spec
    need = s.container_outline_clearance_m if clearance_m is None else float(clearance_m)
    poly = _polygon(opening)
    inside = _inside_map(poly, pad_m=max(0.0, -need) + 0.01)
    pts: np.ndarray = np.asarray(outline_uv, float).reshape(-1, 2)
    if len(pts) >= 3:
        dense: list = []
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.005)))
            dense.extend(a + (b - a) * t for t in np.linspace(0.0, 1.0, n, endpoint=False))
        pts = np.asarray(dense)
    angles = np.radians(np.arange(0.0, 360.0, TURN_STEP_DEG))
    jaws = np.c_[np.cos(angles), np.sin(angles)]                            # (A, 2): the way the jaws open
    squares = np.c_[-jaws[:, 1], jaws[:, 0]]
    others = [(np.asarray(o.position, float)[:2], (o.radius_m or s.unmeasured_obstacle_radius_m)
               + s.container_wall_clearance_m) for o in obstacles if o.position is not None]

    def clearance(xy, jaw, square) -> "np.ndarray":
        """How far inside the walls, at least, what is held lies at each way the jaws open (m; -1: out)."""
        world = np.asarray(xy, float) + pts[None, :, :1] * jaw[:, None, :] + pts[None, :, 1:2] * square[:, None, :]
        least = inside(world).min(axis=1)
        for c, r in others:
            least = np.where(np.linalg.norm(world - c, axis=2).min(axis=1) >= r, least, -1.0)
        return least

    # Every spot and way that fits, ranked as below — the most clearance to the walls among them, as a person lays a
    # long tool in a box: along it or corner to corner, in the middle (2026-11-20 16:25: laid at the least turn
    # from how it was held, it only just cleared a wall). Then the arm asked, in that order, until `most`.
    # Laid along the container where it fits (user, 2026-11-20: "not aligned with the axis of the box at all"),
    # else as near along as it fits (corner to corner); then nearest the arm's base — reached comfortably, not
    # stretched out (16:35: the far side, 0.77 m out, the elbow nearly straight); then the roomiest.
    c0 = poly.mean(axis=0)
    box_axis = np.linalg.svd(poly - c0, full_matrices=False)[2][0]
    held_axis = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)[2][0]
    base = np.asarray(policy._base_xy(env), float)[:2]
    ranked: list = []
    for xy in spots_xy:
        room = clearance(xy, jaws, squares)
        for i in np.flatnonzero(room >= need):
            along = abs(float((held_axis[0] * jaws[i] + held_axis[1] * squares[i]) @ box_axis))
            ranked.append((-round(along, 1), round(float(np.linalg.norm(np.asarray(xy) - base)), 2),
                           -float(room[i]), xy, i))
    ranked.sort(key=lambda f: f[:3])
    fitting = [(f[2], f[3], f[4]) for f in ranked]
    found_xy: list = []
    found_turn: list = []
    for _, xy, i in fitting:
        if any(np.allclose(xy, f) for f in found_xy):
            continue
        base = policy.jaw_axis(env, np.asarray(xy, float))
        j = jaws[i]
        turn = float(np.degrees(np.arctan2(base[0] * j[1] - base[1] * j[0], float(base @ j))))
        turn = (turn + 90.0) % 180.0 - 90.0
        got = _orient(env, [np.array([xy[0], xy[1], z])], 0.0, turn)
        real = None if got is None else env.jaw_direction(got[0])
        if real is None:
            continue
        if float(clearance(xy, real[None, :], np.array([[-real[1], real[0]]]))[0]) >= need:
            found_xy.append(xy)                                  # as the arm will really turn it
            found_turn.append(turn)
        if len(found_xy) >= most:
            break
    return found_xy, found_turn


class _Unbounded:
    """An env whose orientation search reaches everywhere: where something would fit, reach aside."""

    def __init__(self, env):
        self._env = env

    def approach_orientation_for(self, targets, turn_deg: float = 0.0, **_):
        return self._env.down_rotation(targets[0], turn_deg=turn_deg), 0.0

    def jaw_direction(self, rotation):
        return self._env.jaw_direction(rotation)

    def __getattr__(self, name):
        return getattr(self._env, name)


def _travel(policy: "ParallelJaw", env, goal, rot) -> None:
    """A move that only travels (up, across, back up): settled onto at its end unless spec.transit_settle is off."""
    s = policy.spec
    if s.transit_settle:
        env.move_ee_to(goal, tol=s.travel_tol_m, orientation=rot)
    else:
        policy._move_through(env, goal, s.travel_tol_m, rot)


def _up(policy: "ParallelJaw", env, z: float) -> bool:
    """Straight up where the gripper is, to z, leaning as little away from how it is held as that takes."""
    s = policy.spec
    here = np.asarray(env.get_ee_pos(), float)
    if here[2] >= z - s.travel_tol_m:
        return True
    goal = np.array([here[0], here[1], z])
    rot = None
    if isinstance(env, OrientationSearch):
        if isinstance(env, GripperTilt):
            found = _orient(env, [goal], env.gripper_tilt_deg())
            rot = None if found is None else np.asarray(found[0], float)
        elif isinstance(env, GripperRotation):
            # Straight up as it is held, the wrist not turned: the search's own way of turning the jaws would
            # swing what is held round on the way up (the UR5, which searches no lean: 2026-11-20).
            rot = np.asarray(env.get_ee_rotation(), float)
        else:
            found = _orient(env, [goal])
            rot = None if found is None else np.asarray(found[0], float)
    _travel(policy, env, goal, rot)
    return float(env.get_ee_pos()[2]) >= z - 2 * s.travel_tol_m


def _turned(base, xy, angle: float) -> np.ndarray:
    """xy turned by angle (rad) about base, on the surface's plane."""
    c, s = np.cos(angle), np.sin(angle)
    d = np.asarray(xy, float)[:2] - base
    return base + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def _yawed(rot, angle: float) -> np.ndarray:
    """A gripper orientation turned by angle (rad) about the vertical."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(rot, float)


def look_at_rim(policy: "ParallelJaw", env: "RobotEnv", opening: Opening, near_xy, z: float) -> "tuple[float | None, str]":
    """
    How high the container's walls stand, from the wrist camera: (metres above
    the surface, "") or (None, why not). Two looks with the grasp point at
    height z (already clear of the walls), spec.container_look_baseline_m
    apart by a turn about the arm's base, at a wall the turn crosses (within
    45 degrees), the nearest near_xy first; swept by their edges
    (plane_sweep.sweep_edges). Moves the arm; leaves it at the second look.
    """
    from mrcf.perception import plane_sweep
    from mrcf.perception.plane_geometry import PlaneView
    from mrcf.policies.rim_pick import wall_points

    s = policy.spec
    camera = s.camera
    if camera is None or not isinstance(env, CameraAtPose) or not isinstance(env, OrientationSearch):
        return None, "no wrist camera that can be placed for a look"
    poly = _polygon(opening)
    base = policy._base_xy(env)
    tops = np.linspace(0.0, float(opening.rim_m), 4)

    def sees(pose, rot, wall) -> bool:
        model = env.camera_model_at(camera, s.render_w, s.render_h, pose, rot)
        if model is None:
            return False
        view = PlaneView.from_camera_model(model)
        pts = np.array([[wall[0] * 1000.0, wall[1] * 1000.0, h * 1000.0] for h in tops])
        if ((pts @ view.R.T + view.t)[:, 2] <= 1.0).any():
            return False
        px = view.project(pts)
        m = 0.15
        return bool(np.isfinite(px).all() and (px[:, 0] > m * s.render_w).all() and (px[:, 0] < (1 - m) * s.render_w).all()
                    and (px[:, 1] > m * s.render_h).all() and (px[:, 1] < (1 - m) * s.render_h).all())

    ranked = []
    for p, n in wall_points(poly, 0.01):
        radial = p - base
        r = float(np.linalg.norm(radial))
        if r < 1e-6:
            continue
        crosses = abs(float(n @ np.array([-radial[1], radial[0]]) / r))
        if crosses >= np.cos(np.radians(45.0)):
            ranked.append((-round(crosses, 1), float(np.linalg.norm(p - np.asarray(near_xy, float)[:2])), p, n))
    ranked.sort(key=lambda r: (r[0], r[1]))
    plan = None
    for _, _, wall, outward in ranked[:s.container_spots_max]:
        # The camera sits beside the jaws, so the wall is not in view from straight above it:
        # the looks are slid in or out from it until both see it.
        for slide in (0.0, 0.03, -0.03, 0.06, -0.06, 0.09):
            mid = wall - outward * slide
            half = s.container_look_baseline_m / 2.0 / max(float(np.linalg.norm(mid - base)), 1e-6)
            found = env.approach_orientation_for([np.r_[mid, z]])
            if found is None:
                continue
            rot = np.asarray(found[0], float)
            a = (np.r_[_turned(base, mid, -half), z], _yawed(rot, -half))
            b = (np.r_[_turned(base, mid, half), z], _yawed(rot, half))
            if sees(*a, wall) and sees(*b, wall):
                plan = (a, b, wall, outward)
                break
        if plan is not None:
            break
    if plan is None:
        return None, "no pair of looks a turn of the base apart from which the wrist camera sees its wall"
    a, b, wall, outward = plan
    along = np.array([outward[1], -outward[0]])

    def keep(model, shape):
        """Where in a view the wall's top can be — round its line at every height it may have — less the gripper."""
        import cv2

        view = PlaneView.from_camera_model(model)
        h, w = shape
        band = np.zeros((h, w), np.uint8)
        line = np.array([wall + t * along for t in np.linspace(-s.container_look_reach_m, s.container_look_reach_m, 17)])
        for top in np.linspace(0.0, float(opening.rim_m) + 0.02, 8):
            pts = np.c_[line * 1000.0, np.full(len(line), top * 1000.0)]
            front = (pts @ view.R.T + view.t)[:, 2] > 1.0
            if front.sum() < 2:
                continue
            px = view.project(pts[front])
            if np.isfinite(px).all() and np.abs(px).max() < 20 * max(w, h):
                cv2.polylines(band, [px.astype(np.int32)], False, 1, thickness=2 * s.container_look_band_px)
        jaws = policy.jaw_view()
        if jaws is not None:
            seen = jaws.at(float(env.get_gripper_width()), (h, w)).astype(np.uint8)
            pts = np.column_stack(np.nonzero(seen)[::-1]).astype(np.int32)
            if len(pts) >= 3:
                hand = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(hand, cv2.convexHull(pts), 1)       # the jaws and what hangs between them
                hand = cv2.dilate(hand, np.ones((41, 41), np.uint8))
                band[hand > 0] = 0
        return band.astype(bool)

    views = []
    for pose, rot in (a, b):
        env.move_ee_to(pose, tol=s.travel_tol_m, orientation=rot)
        time.sleep(s.container_look_settle_s)
        frame = policy._wrist_frame(env, TAG)
        model = env.camera_model(camera, s.render_w, s.render_h)
        if frame is None or model is None:
            return None, "the wrist camera gave no frame" if frame is None else "the wrist camera is not located"
        frame = np.asarray(frame)
        views.append((frame, model, keep(model, frame.shape[:2])))
    most = float(opening.rim_m) + 0.02
    heights = np.arange(0.0, most + 1e-9, 0.002)
    swept = plane_sweep.sweep_edges(views[0][0], views[0][1], views[1][0], views[1][1], heights,
                                    keep_a=views[0][2], keep_b=views[1][2], min_height_m=s.container_rim_min_m)
    if swept.height_m is None:
        return None, swept.why
    if swept.height_m >= most - 0.004:
        return None, f"its edges line up best at the top of the heights tried ({most * 1000:.0f} mm)"
    lo, hi = swept.spread_m
    policy._say(TAG, f"Its walls stand {hi * 1000:.0f} mm (the edges round its wall line up most at "
                     f"{lo * 1000:.0f} mm, last clearly at {hi * 1000:.0f}; two looks "
                     f"{s.container_look_baseline_m * 1000:.0f} mm apart, {swept.points} edge points).")
    return float(hi), ""


def _way_down(policy: "ParallelJaw", env, over, down, inside_z: float,
              turn: float = 0.0) -> "tuple[np.ndarray, list, str] | None":
    """
    spec.container_lean_above_walls: (the orientation over the spot, [(point,
    orientation)] from there down to `down`, what was done) — leaning as far
    as being over the spot takes, up to spec.container_max_drop_lean_deg,
    while the jaws are above inside_z (the height their tips come among the
    walls at), and no further than spec.container_max_lean_deg below it. None:
    not that way. 2026-11-19, runs 36-37: over the tub, 12-18 cm up and 25-32
    cm out, the SO-101 reaches only leaning 30-40 deg, at the bottom 10-15 deg;
    one lean all the way down, the top decided it, and 6 of 9 were drops.
    """
    s = policy.spec
    limit, top_limit = s.container_max_lean_deg, s.container_max_drop_lean_deg
    within = lambda got, most: got is not None and (most is None or abs(float(got[1])) <= most)  # noqa: E731
    if down[2] >= inside_z - 1e-9:
        # The jaws never among the walls: as far as going over the spot leans.
        got = _orient(env, [over, down], 0.0, turn)
        if not within(got, top_limit):
            return None
        rot = np.asarray(got[0], float)
        return rot, [(down, rot)], f"leaning {abs(float(got[1])):.0f} deg, the jaws above its walls all the way"
    entry = np.array([over[0], over[1], min(float(over[2]), inside_z)])
    inside = _orient(env, [entry, down], 0.0, turn)
    if not within(inside, limit):
        return None
    rot_in = np.asarray(inside[0], float)
    if entry[2] >= over[2] - 1e-9:
        return rot_in, [(down, rot_in)], ""
    top = _orient(env, [over], float(inside[1]), turn)   # the nearest lean to that
    if not within(top, top_limit):
        return None
    return (np.asarray(top[0], float), [(entry, rot_in), (down, rot_in)],
            f"over it leaning {abs(float(top[1])):.0f} deg, among its walls {abs(float(inside[1])):.0f}")


def _way(policy: "ParallelJaw", env, over, down, inside_z: float,
         turn: float = 0.0) -> "tuple[np.ndarray | None, list, str]":
    """
    How to let go at one spot: (the orientation over it, [(point,
    orientation)] down inside from there — [] to let go from over its walls —,
    what was done or why the spot will not do). Pure kinematics: sends nothing.
    """
    s = policy.spec
    if not isinstance(env, OrientationSearch):
        return None, ([] if down is None else [(down, None)]), ""
    xy = f"({over[0]:+.3f}, {over[1]:+.3f})"
    found = None if down is None else _orient(env, [over, down], 0.0, turn)
    limit = s.container_max_lean_deg
    if down is not None and s.container_lean_above_walls and \
            (found is None or (limit is not None and abs(float(found[1])) > limit)):
        way = _way_down(policy, env, over, down, inside_z, turn)
        if way is not None:
            return way
    if found is None:
        down = None                                          # not straight down from here: let go from over the walls
        found = _orient(env, [over], 0.0, turn)
    if found is None:
        return None, [], f"{xy} out of reach {(over[2] - s.table_z) * 1000:.0f} mm up"
    if limit is not None and abs(float(found[1])) > limit:
        # 2026-11-19 11:18, the SO-101: the tub's far corner, 36 cm out, reached leaning 35 deg; the
        # gripper met the far wall on the way down, and the retry pushed the tub across the table.
        # Over the walls, letting go there, nothing goes down inside to meet one: leaning further is
        # let (container_max_drop_lean_deg). 14:57: a scoop held by its handle, crossing high for its
        # swing, refused at every spot for leaning 35-40 deg.
        over_only = _orient(env, [over], 0.0, turn) if down is not None else found
        drop_limit = s.container_max_drop_lean_deg
        if over_only is None or drop_limit is None or abs(float(over_only[1])) > drop_limit:
            return None, [], f"{xy} reached only leaning {abs(float(found[1])):.0f} deg"
        found, down = over_only, None
    rot = np.asarray(found[0], float)
    return rot, ([] if down is None else [(down, rot)]), ""


def let_go_in(policy: "ParallelJaw", env, opening: Opening, candidates: list, clear: float, plan, name: str,
              rim_said: str, lowest: "float | None" = None, floor: float = 0.0, turns: "list | None" = None) -> Outcome:
    """
    Over the first spot the arm reaches at the height that clears the walls
    (clear, above the surface), and — lowest given — down inside to that
    grasp-point height (_way), stopping where it is stopped; the jaws opened
    there, and back up the way it went down to clear. lowest None: let go from
    over the walls (a drop).
    """
    s = policy.spec
    poly = _polygon(opening)
    hang = max(plan.grasp_height_m, s.tip_below_grasp_m)
    # Where the jaws' tips come among the walls, going down.
    inside_z = s.table_z + float(opening.rim_m) + s.container_wall_clearance_m + s.tip_below_grasp_m
    tried = []

    def ends(xy):
        over = np.array([xy[0], xy[1], s.table_z + clear])
        down = None if lowest is None or lowest >= clear - s.travel_tol_m else np.array([xy[0], xy[1], s.table_z + lowest])
        return over, down

    # Each spot with the wrist turn what is held goes in at there (laid_in), else none.
    order = list(zip(candidates, turns or [0.0] * len(candidates)))[:s.container_spots_max]
    if s.container_lean_above_walls and lowest is not None and isinstance(env, OrientationSearch):
        # A spot it can go down inside before one it can only let go over: roomiest first among each.
        goes_down = [bool(_way(policy, env, ends(xy)[0], ends(xy)[1], inside_z, t)[1]) for xy, t in order]
        order = [c for c, d in zip(order, goes_down) if d] + [c for c, d in zip(order, goes_down) if not d]
    for xy, turn in order:
        over, below = ends(xy)
        rot, steps, said = _way(policy, env, over, below, inside_z, turn)
        if rot is None and isinstance(env, OrientationSearch):
            tried.append(said)
            continue
        down = steps[-1][0] if steps else None
        here = np.asarray(env.get_ee_pos(), float)
        if here[2] > over[2] + s.travel_tol_m:
            _travel(policy, env, np.r_[over[:2], here[2]], rot)                               # across, then down
        _travel(policy, env, over, rot)
        if not policy._near(env, over, 2 * s.travel_tol_m):
            tried.append(f"did not get over ({xy[0]:+.3f}, {xy[1]:+.3f})")
            continue
        here = np.asarray(env.get_ee_pos(), float)
        if inside_by(poly, here[:2]) < s.container_wall_clearance_m:
            tried.append(f"stopped over its wall at ({here[0]:+.3f}, {here[1]:+.3f})")
            continue
        lowered = False
        if steps and down is not None:
            if said:
                policy._say(TAG, f"Down inside {name}: {said}.")
            # Free space inside the walls down to `lowest` (what is held stays over the floor, the gripper's
            # body over the rim): an ordinary move. The slow, contact-limited descent of a grasp stalled
            # before it got going here, twice on 2026-11-17, and the object was let go from over the walls.
            for point, r in steps[:-1]:
                _travel(policy, env, point, r)
            for _ in range(2 if s.container_descent_retry else 1):
                env.move_ee_to(down, tol=s.travel_tol_m, orientation=steps[-1][1])
                if float(env.get_ee_pos()[2]) <= down[2] + 2 * s.travel_tol_m:
                    break
            here = np.asarray(env.get_ee_pos(), float)
            lowered = here[2] <= down[2] + 0.03
            if not s.container_descent_retry and here[2] > down[2] + 2 * s.travel_tol_m:
                # Stopped short inside: something in the way — a wall, what is in it. Not pushed again.
                policy._say(TAG, f"The way down inside {name} stopped at {(here[2] - s.table_z) * 1000:.0f} mm, short "
                                 f"of {(down[2] - s.table_z) * 1000:.0f} — something in the way: back up, and letting "
                                 "go from over its walls.")
                for point, r in reversed(steps[:-1]):
                    _travel(policy, env, point, r)
                _travel(policy, env, over, rot)
                here, lowered, down, steps = np.asarray(env.get_ee_pos(), float), False, None, []
        up = (here[2] - s.table_z) * 1000.0
        above_floor = (here[2] - s.table_z - hang - floor) * 1000.0
        how = (f"lowered inside to {up:.0f} mm, what it holds {max(above_floor, 0.0):.0f} mm over the floor"
               if lowered else f"{up:.0f} mm up, over walls {rim_said}"
               + (f" (the way down inside got only to {up:.0f} of {(down[2] - s.table_z) * 1000:.0f} mm)"
                  if down is not None else ""))
        policy._say(TAG, f"Inside {name} at ({here[0]:+.3f}, {here[1]:+.3f}), {inside_by(poly, here[:2]) * 1000:.0f} mm "
                         f"in from its walls, {how}: letting go.")
        env.set_gripper(max(plan.pregrasp_open_m, plan.grip_width_m + 2 * s.jaw_slack_m), n_steps=15)
        if steps:
            for point, r in reversed(steps[:-1]):
                _travel(policy, env, point, r)                                                 # back up the way down
            _travel(policy, env, over, rot)
        verb = "put into" if lowered else "dropped into"
        return Outcome(True, f"{verb} {name} at ({here[0]:+.3f}, {here[1]:+.3f}), let go {up:.0f} mm up",
                       grasp=plan, at_xy=here[:2].copy(), released=True)
    why = "; ".join(dict.fromkeys(tried)) or "no spot"
    return Outcome(False, f"could not get over {name} to let go in it — {why}; still holding it", refused=True,
                   grasp=plan)


# Gripped across less than this share of its width, it was not gripped through its middle.
OFF_MIDDLE_SHARE = 0.8


def _turned_hang(plan, jaws_m: float) -> float:
    """
    How far below the jaws what is held may hang (spec.held_turns), from
    where it was gripped: gripped through its middle (the jaws closed across
    most of its width), as far as it was gripped above its foot; gripped off
    its middle — across a narrower part, or by a wall — it can turn in the
    jaws and hang by its farthest part: as far as that is from where it was
    gripped (up or down its height, and half its width across). User,
    2026-11-19: "based on where it grips the object it needs to lift it a
    different height to pass the tub walls". 12:43: a plug held 27 mm across
    its 54 mm body, lifted for its 20 mm grip, caught the tub's far wall;
    12:05, a scoop by the tip of its wall, the same.
    """
    g = float(plan.grasp_height_m)
    if jaws_m >= OFF_MIDDLE_SHARE * float(plan.width_m):
        return g
    return float(np.hypot(max(g, float(plan.height_m) - g), float(plan.width_m) / 2.0))


def place_in(policy: "ParallelJaw", env: "RobotEnv", opening: Opening, held: "Target | None" = None, grasp=None,
             obstacles=(), drop: bool = False) -> Outcome:
    """Put what is held into the container (drop: let go from over its walls): see the module docstring."""
    s = policy.spec
    if not s.place_in:
        return Outcome(False, "this arm has not been set up to put things into containers (GripperSpec.place_in)",
                       refused=True)
    if not policy.holding(env):
        return Outcome(False, "not holding an object — pick first", refused=True)
    try:
        _polygon(opening)
    except ValueError as exc:
        return Outcome(False, str(exc), refused=True)
    from mrcf.policies.parallel_jaw.spec import GraspPlan

    plan = grasp if isinstance(grasp, GraspPlan) else policy.plan(held or Target(np.zeros(3)))
    name = opening.label or "the container"
    floor = float(opening.floor_m) if opening.floor_m is not None else s.container_floor_m
    # How far below the jaws' grasp point the held thing reaches: its foot, or the jaw tips.
    hang = max(plan.grasp_height_m, s.tip_below_grasp_m)
    plain_hang = hang
    if s.held_turns:
        hang = max(hang, _turned_hang(plan, float(env.get_gripper_width())))
    if s.rim_swing_share > 0 and plan.grip_width_m <= 0.005:
        # Held by a wall (rim_pick's plan grips a 4 mm wall), it can swing under the jaws. 2026-11-19 12:05:
        # a scoop held by its rim's very tip, 41 mm up, swung on the way across and hit the tub's wall.
        hang = max(hang, plan.grasp_height_m + s.rim_swing_share * plan.width_m)

    def clear_of(rim_m: float) -> float:
        """The grasp point's least height over walls rim_m tall, above the surface the container stands on."""
        return float(rim_m) + s.container_wall_clearance_m + s.container_cross_extra_m + hang

    room = max(plan.pregrasp_open_m / 2 + s.finger_thickness_m, plan.width_m / 2) + s.container_wall_clearance_m
    turns = None
    if plan.held_outline:
        # Its own outline, turned with the wrist: where it lies inside, along or corner to corner, not a circle
        # as wide as it is long (2026-11-20, the UR5's hammer, 251 mm: "nowhere 145 mm in from its walls" in a
        # box longer than it). The jaws themselves in from the walls; what is held checked at each turn.
        from mrcf.capabilities import GripperRotation, JawDirection

        jaws_room = plan.pregrasp_open_m / 2 + s.finger_thickness_m + s.container_wall_clearance_m
        now = (env.jaw_direction(env.get_ee_rotation())
               if isinstance(env, GripperRotation) and isinstance(env, JawDirection) else None)
        z_over = s.table_z + float(opening.rim_m) + s.container_wall_clearance_m + max(plan.grasp_height_m,
                                                                                         s.tip_below_grasp_m)
        # What is in it already is lain on, not kept clear of, where things go on top of one another
        # (spec.place_in_over_contents; user, 2026-11-20: "its a box - there may be things inside") — the spots
        # tried too, not only the fit: 17:34, a hammer and a screwdriver in the box, every spot kept clear of
        # them, and the second screwdriver refused.
        inside = () if s.place_in_over_contents else obstacles
        near = spots(policy, env, opening, jaws_room, inside, held_half_m=plan.grip_width_m / 2,
                     step_m=min(s.container_spot_step_m, 0.01))
        if not near and s.place_in_over_contents:
            near = [tuple(float(v) for v in _polygon(opening).mean(axis=0))]     # its middle, whatever else
        candidates, turns = laid_in(policy, env, opening, plan.held_outline, near, z_over, inside,
                                    prefer_jaw=now, most=s.container_spots_max)
        if not candidates:
            # Not flat inside, clear of the walls (a 250 mm hammer, its head 110 mm across, in a 195 x 234 mm
            # opening: 2026-11-20 17:09): let go over the walls where it fits most nearly — overhanging them by
            # 3 cm at most — to settle in at a slant, as the hammer dropped in at 16:35 did.
            candidates, turns = laid_in(policy, env, opening, plan.held_outline, near, z_over, inside,
                                        prefer_jaw=now, most=s.container_spots_max, clearance_m=-0.03)
            if candidates:
                drop = True
                policy._say(TAG, f"It does not lie flat inside {name}, clear of its walls: letting go over them, "
                                 "where it fits most nearly, to settle in.")
        if not candidates and near:
            # Put in, whatever its fit: over the most central spot the jaws fit, dropped from over the walls
            # (user, 2026-11-20: "i'm telling you to put it in - stop making up random rules like these").
            # Its long edge along the container's long edge, as near the middle as the arm reaches (laid_in ranks
            # along the container first; any overhang let be): user, "the long edge of the object is on the long
            # edge of the box - thats the only common sense thing it should do".
            candidates, turns = laid_in(policy, env, opening, plan.held_outline, near, z_over, (),
                                        prefer_jaw=now, most=s.container_spots_max, clearance_m=-1.0)
            if not candidates:
                candidates, turns = list(near[:s.container_spots_max]), [0.0] * min(len(near), s.container_spots_max)
            drop = True
            policy._say(TAG, f"Dropping it into {name} from over its walls, its long edge along {name}'s.")
        if candidates and isinstance(env, OrientationSearch) and isinstance(env, JawDirection):
            got = _orient(env, [np.r_[candidates[0], z_over]], 0.0, turns[0])
            jaw = None if got is None else env.jaw_direction(got[0])
            if jaw is not None:
                pts = np.asarray(plan.held_outline, float)
                held_axis = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)[2][0]
                box = _polygon(opening)
                box_axis = np.linalg.svd(box - box.mean(axis=0), full_matrices=False)[2][0]
                lies = held_axis[0] * jaw + held_axis[1] * np.array([-jaw[1], jaw[0]])
                off = float(np.degrees(np.arccos(min(1.0, abs(float(lies @ box_axis))))))
                policy._say(TAG, f"Its long edge {off:.0f} deg from {name}'s long edge, as it will be let go.")
        if candidates:
            policy._say(TAG, f"Laid in {name} by its own outline, where it has the most room: the wrist turned "
                             f"{turns[0]:+.0f} deg at ({candidates[0][0]:+.3f}, {candidates[0][1]:+.3f}).")
        else:
            # Room, but not within reach (2026-11-20: a box 0.67-0.83 m out, its walls 260 mm: over them the arm is
            # nearly straight there): said so, not "no room".
            reach_all = laid_in(policy, _Unbounded(env), opening, plan.held_outline, near, z_over, obstacles, most=1)[0]
            if reach_all:
                far = float(np.linalg.norm(np.asarray(reach_all[0]) - np.asarray(policy._base_xy(env), float)[:2]))
                return Outcome(False, f"{name} has room for it, but no spot in it the arm reaches comfortably "
                                      f"{(z_over - s.table_z) * 1000:.0f} mm up, over its walls ({far:.2f} m from its "
                                      f"base): move {name} nearer the arm", refused=True, grasp=plan)
            return Outcome(False, f"no room inside {name} to let go: what is held does not lie inside its walls "
                                  "at any spot or turn, clear of what is in it", refused=True, grasp=plan)
    else:
        candidates = spots(policy, env, opening, room, obstacles, held_half_m=plan.width_m / 2,
                           step_m=s.container_spot_step_m)

    def over_contents() -> list:
        # Into a tub, things go on top of one another. 2026-11-19, the SO-101:
        # the scoop, 128 mm across, refused — nowhere 83 mm in from the walls
        # and clear of the figure already put in the middle — and the plan
        # stopped holding it. Over the walls, where the jaws themselves fit,
        # roomiest first (spots()), and let go: it lands on what is there.
        jaws = plan.pregrasp_open_m / 2 + s.finger_thickness_m + s.container_wall_clearance_m
        return spots(policy, env, opening, jaws, (), held_half_m=0.0, step_m=s.container_spot_step_m)

    if not candidates and s.place_in_over_contents:
        candidates = over_contents()
        if candidates:
            policy._say(TAG, f"No room in {name} for all of it ({room * 1000:.0f} mm in from its walls, clear of "
                             "what is in it): letting go from over its walls, onto what is there.")
            drop = True
    if not candidates:
        return Outcome(False, f"no room inside {name} to let go: nowhere {room * 1000:.0f} mm in from its walls "
                              "and clear of what is in it", refused=True, grasp=plan)

    # Up first, where it is: the move across goes in joint space, which does not keep its height.
    rim_said = f"{opening.rim_m * 1000:.0f} mm" + ("" if opening.rim_measured else " at most (not measured)")
    clear = clear_of(opening.rim_m)
    if float(env.get_ee_pos()[2]) < s.table_z + clear - s.travel_tol_m:
        policy._say(TAG, f"Up to {clear * 1000:.0f} mm, clear of {name}'s walls ({rim_said}), before moving across.")
    if not _up(policy, env, s.table_z + clear) and hang > plain_hang:
        hang = plain_hang
        clear = clear_of(opening.rim_m)
        policy._say(TAG, f"Not that high here: across at {clear * 1000:.0f} mm, as gripped.")
    if not _up(policy, env, s.table_z + clear):
        return Outcome(False, f"cannot lift it high enough to clear {name}'s walls here (got to "
                              f"{(float(env.get_ee_pos()[2]) - s.table_z) * 1000:.0f} mm of {clear * 1000:.0f}) — "
                              "still holding it", refused=True, grasp=plan)

    measured = None
    if not opening.rim_measured:
        # A wall it is about to cross: the nearest the gripper that the looks can be taken across.
        rim, why = look_at_rim(policy, env, opening, np.asarray(env.get_ee_pos(), float)[:2], s.table_z + clear)
        if rim is None:
            policy._say(TAG, f"Could not see how high {name}'s walls stand ({why}) — passing them as high as they "
                             f"may be ({opening.rim_m * 1000:.0f} mm), and letting go from there.")
        else:
            measured = Measured(height_mm=round(rim * 1000.0, 1), widths_mm={}, foot_mm=0.0, slant_deg=0.0, looks=2)
            opening = dataclasses.replace(opening, rim_m=rim + s.container_rim_margin_m, rim_measured=True)
            clear = clear_of(opening.rim_m)
            rim_said = f"{opening.rim_m * 1000:.0f} mm"

    lowest = None
    if opening.rim_measured and not drop:
        # Down to what is held just over the floor — the gripper's body kept over the rim.
        body = s.container_gripper_inside_m if s.container_gripper_inside_m is not None else s.free_depth_m
        lowest = max(floor + hang + s.release_clearance_m, opening.rim_m + s.container_wall_clearance_m - body)
    out = let_go_in(policy, env, opening, candidates, clear, plan, name, rim_said, lowest, floor, turns)
    if out.refused and s.place_in_over_contents and lowest is not None and policy.holding(env):
        # The clear spots all out of an upright reach (container_max_lean_deg): over what is in it, nearer.
        policy._say(TAG, f"No clear spot in {name} the arm reaches upright ({out.message}) — letting go from over "
                         "its walls, onto what is there.")
        out = let_go_in(policy, env, opening, over_contents(), clear, plan, name, rim_said, None, floor)
    return dataclasses.replace(out, measured=measured)
