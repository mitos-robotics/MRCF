"""
scene_state.py — the world model the planner works from: what is in the scene,
where, and what the arm holds.

    SceneObject   one object or placement target, with its position and what
                  has been measured of it
    SceneState    every object and target, name lookup that tolerates the
                  planner's wording, and the summary the planner reads

Kept current by perception (a deployment's scan, vlm.discover_scene) and by the
actions (manipulation.py); read by the planner (vlm.plan) and the safety layer.
Nothing here talks to a model or a robot.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mrcf.object_memory import ObjectMemory

# SceneState.summary marks an object "[on <target>]" within PROXIMITY_ON_M of a
# placement target, and "[near <target>, X cm]" within PROXIMITY_NEAR_M (XY,
# metres; 0 turns either off). Set through vlm.configure().
PROXIMITY_ON_M = 0.015
PROXIMITY_NEAR_M = 0.060


# ── Scene state ────────────────────────────────────────────────────────────────

@dataclass
class SceneObject:
    name: str               # exact body name in the simulator / perception system
    description: str        # human-readable, e.g. "small red cube"
    interactable: bool = True   # can the robot act on this object? (pick it up, push it, etc.)
    position: np.ndarray | None = None  # [x, y, z] world frame estimated from camera
    pixel_xy: list | None = None        # [u, v] image pixel from discover_scene (for action anchoring)
    reachable: bool = True  # within robot's effective workspace; set by caller after discover_scene
    track_id: str | None = None  # its identity in scene_state.memory (object_memory), if one is kept
    # What scans and the arm have measured about it, for the policies
    # (manipulation.py turns these into a policies.Target):
    geometry: dict | None = None        # {"base_diameter_m": ...}: its foot, from a scan
    measured_shape: dict | None = None  # {"height_mm", "widths_mm", "looks"}: from the wrist camera
                                        # ("looks": how many measurements agree on it)
    grasp: object = None                # while held: how the policy holds it (Outcome.grasp)
    faces_deg: float | None = None      # flat sides (a block): which way one faces, deg from x, mod 90
                                        # (policies.Target.faces_deg); None: round, or not known
    on: str | None = None               # standing on another object (a block on a block): that one's
                                        # name; position[2] is then the top of it, where this one's foot is
    # False: only a source that can report what is not there (a vision model)
    # has seen it — the one that measures looked and found nothing (see
    # object_memory.Detection.confirmed). Still in the scene; not kept clear of.
    confirmed: bool = True

    def __str__(self):
        if self.position is not None:
            p = self.position
            reach_tag = "" if self.reachable else "  [OUT OF REACH]"
            on_tag = f"  [ON TOP OF {self.on}]" if self.on else ""
            return f"{self.name}: {self.description}  @ [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]{on_tag}{reach_tag}"
        # No position is the convention for carried (SceneState.held_object):
        # "try observe" here had the planner scan with the object in the jaws.
        return f"{self.name}: {self.description}  (in the gripper: no position on the table)"


# Words that carry no identity in an object reference, so they must not stop a
# token match ("the pawn" -> "white_pawn_e5").
_STOPWORDS = {"the", "a", "an", "this", "that", "it", "one", "piece", "object"}


@dataclass
class SceneState:
    objects: list = field(default_factory=list)  # list[SceneObject] — interactable
    targets: list = field(default_factory=list)  # list[SceneObject] — placement zones
    # Optional object_memory.ObjectMemory: who each object is across scans,
    # what moved and what has gone. A deployment that keeps one sets it; the
    # summary then carries its recent events to the planner.
    memory: "ObjectMemory | None" = None
    # The interlock's trip_count when the scene was last looked at (None: not
    # yet). Tripped since, something other than the arm may have moved, and
    # the controller looks again (controller.look_again_if_disturbed).
    looked_at_trip: "int | None" = None
    # And with it: how many of those trips were stops, not the sentry's pauses
    # (Interlock.stop_count), and when (time.monotonic()) the look began — what
    # the sentry saw move since then says whether to look again.
    looked_at_stops: "int | None" = None
    looked_at_time: "float | None" = None
    looked_at_sources: "dict | None" = None   # Interlock.trips_by_source then
    # The look's own frame, for a quick check later of what changed since
    # (perception/placed_check.py): {"camera", "frame" (RGB), "positions"
    # (track id -> (x, y) where each object stood then), "views" (each other
    # calibrated fixed camera's frame then, {name: RGB}), "arm" (arm_shape()
    # then)}. None: no look kept.
    look_frame: "dict | None" = None
    # Where the object about to be picked is going: the name of the place
    # target the plan puts it on next (None: not known). execute_plan sets it
    # for each pick; a skill that picks and places itself (chess's carry) sets
    # it round its pick. Making room for a pick (manipulation._make_room) sets
    # nothing down there, and pushes nothing toward it.
    headed_for: "str | None" = None

    def all_objects(self):
        return self.objects + self.targets

    def all_names(self):
        return [o.name for o in self.all_objects()]

    def find(self, name: str, among: str = "all") -> "SceneObject | None":
        """
        Look up an object or target by name, exact match first. `among` is
        "all", "objects" or "targets": a pick must only ever find an object.
        Searching both, an old name ("white_piece_e4", for a piece a scan has
        since renamed) can match the empty square e4, and the arm go down on an
        empty square.

        Falls back to looser matching because the name in a plan and the name
        in the belief state come from two DIFFERENT models: discover_scene's
        vision model invents the names it reports ("white_pawn"), while the
        planner refers to whatever the free-text scene description led it to
        call the same thing ("pawn"). Requiring those two to agree character
        for character blocks correct plans over pure naming differences.

        Exact match still wins outright, so backends with a real body registry
        (a simulator) keep their precise-name semantics.
        """
        if not name:
            return None
        candidates = {"all": self.all_objects(), "objects": list(self.objects),
                      "targets": list(self.targets)}[among]

        exact = next((o for o in candidates if o.name == name), None)
        if exact is not None:
            return exact

        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())

        target = _norm(name)
        if not target:
            return None

        same = [o for o in candidates if _norm(o.name) == target]
        if same:
            return same[0]

        def _tokens(text: str) -> set[str]:
            return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t} - _STOPWORDS

        want = _tokens(name)

        # Two ways to match loosely, pooled rather than tried in sequence:
        #
        #   substring, either direction — "pawn" matches "white_pawn", and
        #   "white_pawn" matches a belief entry called just "pawn".
        #
        #   token subset — whole-string substring alone misses the most natural
        #   divergence of all: a planner saying "the pawn" for an object called
        #   "white_pawn_e5" matched nothing, because "thepawn" is not a
        #   substring of "whitepawne5" in either direction.
        #
        # Pooled, because the branches disagree about which match is better and
        # the weaker one is not always empty. "the white pawn on e5" matches
        # the empty square "e5" by substring and the piece "white_pawn_e5" by
        # tokens; running substring first and stopping returned the square.
        pool: dict[str, SceneObject] = {}
        for o in candidates:
            name_norm, name_tokens = _norm(o.name), _tokens(o.name)
            if target in name_norm or name_norm in target:
                pool[o.name] = o
            elif want and name_tokens and (name_tokens <= want or want <= name_tokens):
                pool[o.name] = o

        if pool:
            # Most shared tokens first — that is what makes "white_pawn_e5" beat
            # "e5" for "the white pawn on e5". Then closest length (most
            # specific), then name, so an ambiguous lookup is at least stable.
            def _rank(o: SceneObject):
                return (-len(_tokens(o.name) & want),
                        abs(len(_norm(o.name)) - len(target)),
                        o.name)

            best = min(pool.values(), key=_rank)
            # An ambiguous reference resolving silently is how "pick the pawn"
            # quietly operates on a different pawn than the one meant. Only
            # equally-good matches count as ambiguous — a clear winner is not.
            tied = [o.name for o in pool.values()
                    if o.name != best.name and _rank(o)[:2] == _rank(best)[:2]]
            if tied:
                print(f"  [Scene] '{name}' is ambiguous — matched {best.name} "
                      f"(also: {', '.join(sorted(tied)[:4])}"
                      f"{', …' if len(tied) > 4 else ''}). Name it exactly to disambiguate.")
            return best

        return None

    def held_object(self) -> "SceneObject | None":
        """Return the object currently being carried (position=None), or None.

        Deployments mark an object as carried by setting obj.position = None
        in their skill implementations.  This is the SceneState convention for
        "not localisable because it is attached to the robot."
        """
        return next((o for o in self.objects if o.position is None), None)

    def update_position(self, name: str, new_pos: "np.ndarray | None") -> None:
        """Incrementally update one object's position in the belief state.

        Pass None to mark the object as currently carried.  Called by skills
        after a successful action so the world model stays current without a
        full VLM rescan.
        """
        obj = self.find(name)
        if obj is not None:
            obj.position = new_pos.copy() if new_pos is not None else None

    def summary(self) -> str:
        lines = []
        # Pre-compute nearest target for each object so the planner can resolve
        # relative spatial language ("two squares forward", "next to the zone").
        def _nearest_target(obj: "SceneObject") -> "tuple[str, float] | None":
            if obj.position is None or not self.targets:
                return None
            best_name, best_dist = None, float("inf")
            for t in self.targets:
                if t.position is None:
                    continue
                d = float(np.linalg.norm(obj.position[:2] - t.position[:2]))
                if d < best_dist:
                    best_dist, best_name = d, t.name
            return (best_name, best_dist) if best_name else None

        if self.objects:
            lines.append("Objects:")
            for o in self.objects:
                line = f"  {o}"
                nearest = _nearest_target(o)
                if nearest is not None:
                    n_name, n_dist = nearest
                    if PROXIMITY_ON_M > 0 and n_dist < PROXIMITY_ON_M:
                        line += f"  [on {n_name}]"
                    elif PROXIMITY_NEAR_M > 0 and n_dist < PROXIMITY_NEAR_M:
                        line += f"  [near {n_name}, {n_dist * 100:.1f} cm]"
                lines.append(line)
        if self.targets:
            lines.append("Placement targets:")
            for t in self.targets:
                lines.append(f"  {t}")
        remembered = self.memory.describe() if self.memory is not None else ""
        if remembered:
            lines.append(remembered)
        return "\n".join(lines) if lines else "(no objects detected)"


__all__ = ["SceneObject", "SceneState", "PROXIMITY_ON_M", "PROXIMITY_NEAR_M"]
