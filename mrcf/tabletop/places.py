"""
places.py — where the table task can put something down.

    load(path), save(path, places)   the spots marked once and named, {name: (x, y)}
    targets(places, table_z)         them as placement targets for the world model
    beside(...)                      free table beside an object, on the side asked for

Sides are the viewer's: "front" is toward whoever looks, "left" their left.
The viewer is the arm (looking out from its base) or a person facing it from
across the table (the deployment's LEFT_RIGHT_SEEN_FROM).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIDES = ("left", "right", "front", "behind", "any")
_TURNS_DEG = (0.0, 20.0, -20.0, 40.0, -40.0)     # tried round each side, in order
_FURTHER_M = (0.0, 0.01, 0.02)                   # and a little further out


def load(path) -> dict:
    """{name: (x, y)} from path; {} when there is none yet."""
    p = Path(path)
    if not p.exists():
        return {}
    return {str(k): (float(v[0]), float(v[1])) for k, v in json.loads(p.read_text()).items()}


def save(path, places: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: [round(float(v[0]), 4), round(float(v[1]), 4)] for k, v in sorted(places.items())},
                            indent=2))


def clean_name(name: str) -> str:
    """A place's name as the planner sees it: lower case, words joined by underscores."""
    return "_".join(str(name).lower().replace("_", " ").split())


def targets(places: dict, table_z: float) -> list:
    """The marked spots as placement targets (vlm.SceneObject)."""
    from mrcf.vlm import SceneObject

    return [SceneObject(name, "a marked spot on the table", interactable=False,
                        position=np.array([xy[0], xy[1], table_z])) for name, xy in sorted(places.items())]


def _directions(obj_xy, base_xy, seen_from: str) -> dict:
    """Unit (x, y) for each side of an object at obj_xy, as the viewer sees it."""
    out = np.asarray(obj_xy, float)[:2] - np.asarray(base_xy, float)[:2]
    out = out / (np.linalg.norm(out) or 1.0)                 # away from the arm
    toward_viewer = -out if seen_from == "arm" else out
    # The viewer's right, looking at the object (along -toward_viewer).
    look = -toward_viewer
    right = np.array([look[1], -look[0]])
    return {"front": toward_viewer, "behind": -toward_viewer, "right": right, "left": -right}


def beside(obj_xy, obj_radius_m: float, held_radius_m: float, side: str, others: list, bounds,
           base_xy=(0.0, 0.0), gap_m: float = 0.025, seen_from: str = "facing", reachable=None):
    """
    A free spot (x, y) to stand an object of held_radius_m beside the one at
    obj_xy, on `side` (SIDES; "any": the first free of right, left, front,
    behind): clear of each of `others` [(xy, radius_m)] by gap_m, inside
    bounds (safety.WorkspaceBounds) by the held object's radius, and
    reachable(xy) when that is given. Tried straight out on the side first,
    then turned a little each way, then a little further out. None: no room.
    """
    if side not in SIDES:
        raise ValueError(f"side is one of {', '.join(SIDES)}, not {side!r}")
    obj_xy = np.asarray(obj_xy, float)[:2]
    dirs = _directions(obj_xy, base_xy, seen_from)
    order = ["right", "left", "front", "behind"] if side == "any" else [side]
    for name in order:
        d = dirs[name]
        for further in _FURTHER_M:
            r = obj_radius_m + held_radius_m + gap_m + further
            for turn in _TURNS_DEG:
                a = np.radians(turn)
                v = np.array([np.cos(a) * d[0] - np.sin(a) * d[1], np.sin(a) * d[0] + np.cos(a) * d[1]])
                spot = obj_xy + r * v
                if _inside(spot, bounds, held_radius_m) and _clear(spot, held_radius_m, others, gap_m) \
                        and (reachable is None or reachable(spot)):
                    return spot
    return None


def _inside(xy, bounds, margin: float) -> bool:
    if bounds is None:
        return True
    return (bounds.x_min + margin <= xy[0] <= bounds.x_max - margin
            and bounds.y_min + margin <= xy[1] <= bounds.y_max - margin)


def _clear(xy, radius: float, others: list, gap: float) -> bool:
    return all(float(np.linalg.norm(np.asarray(o, float)[:2] - xy)) >= r + radius + gap for o, r in others)
