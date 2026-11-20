"""
grasp_correction.py — where a gripper's jaws really close, measured, against
where the arm's model puts its grasp point: a correction by lean and reach.

An arm's model puts its grasp point from the joint angles, and a fixed camera
watching a tag on the gripper corrects where the gripper is — but not how it
is turned. On an arm whose joints sit a degree or two off, or sag, that leaves
the jaws, ~90 mm below the tag, off by however far the turn swings them: more,
the further the gripper leans. tools/calibrate_grasp.py measures it with the
jaws themselves, at each lean the arm comes down at and at a few reaches: it
closes them on an object held down to the surface at offsets along the arm,
and the middle of the span where they catch it is where they really close.

    table = GraspCorrection.load(path)            # empty without a file
    along_m, across_m = table.at(lean_deg, reach_m)
    table.offset(lean_deg, at_xy, base_xy)        # the same, as a world (x, y, 0) vector

"Along" is away from the arm's base, "across" a quarter turn counter-clockwise
from it, seen from above; the values are the true grasp point minus the
model's. RigEnv applies it (RigEnv._grasp_correction); an empty table changes
nothing.

Between the measured points it interpolates linearly: in lean among the
points measured at one reach (each within REACH_BAND_M of the next), then
in reach between the two reaches either side. Past the ends of either it
holds the last value measured rather than guess beyond it; a measured point
is its own value exactly.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Points this close in reach to the next are one reach, interpolated by lean:
# an object pushed along while it is measured spreads its points out (on the
# SO-101, 2026-10-30, one pawn's over 117-140 mm), and they stay one reach.
REACH_BAND_M = 0.02


@dataclass
class Point:
    """One measurement: where the jaws closed at this lean and reach."""
    lean_deg: float
    reach_m: float
    along_m: float                      # true grasp point minus the model's, away from the base
    across_m: float = 0.0               # ... a quarter turn counter-clockwise from that
    span_m: "list | None" = None        # the offsets along the arm that caught the object [first, last]
    note: str = ""


@dataclass
class GraspCorrection:
    points: list = field(default_factory=list)       # [Point]
    measured: str = ""

    def __bool__(self) -> bool:
        return bool(self.points)

    def at(self, lean_deg: float, reach_m: float) -> "tuple[float, float]":
        """(along_m, across_m) at this lean and reach; (0, 0) with no points."""
        if not self.points:
            return 0.0, 0.0
        reaches, values = [], []
        for band in self._bands():
            band = sorted(band, key=lambda p: p.lean_deg)
            leans = [p.lean_deg for p in band]
            reaches.append(float(np.mean([p.reach_m for p in band])))
            values.append((float(np.interp(lean_deg, leans, [p.along_m for p in band])),
                           float(np.interp(lean_deg, leans, [p.across_m for p in band]))))
        order = np.argsort(reaches)
        r = np.asarray(reaches)[order]
        along = float(np.interp(reach_m, r, [values[i][0] for i in order]))
        across = float(np.interp(reach_m, r, [values[i][1] for i in order]))
        return along, across

    def _bands(self) -> list:
        """The points grouped by reach: each within REACH_BAND_M of the one before it."""
        bands: list = []
        for p in sorted(self.points, key=lambda p: p.reach_m):
            if bands and p.reach_m - bands[-1][-1].reach_m <= REACH_BAND_M:
                bands[-1].append(p)
            else:
                bands.append([p])
        return bands

    def offset(self, lean_deg: float, at_xy, base_xy=(0.0, 0.0)) -> np.ndarray:
        """at(), as a world (x, y, 0) vector for a grasp point at at_xy from an arm based at base_xy."""
        out = np.asarray(at_xy, float)[:2] - np.asarray(base_xy, float)[:2]
        reach = float(np.linalg.norm(out))
        if not self.points or reach < 1e-6:
            return np.zeros(3)
        u = out / reach
        along, across = self.at(lean_deg, reach)
        return np.array([along * u[0] - across * u[1], along * u[1] + across * u[0], 0.0])

    # ── Kept on disk ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"measured": self.measured, "points": [asdict(p) for p in self.points]}

    @classmethod
    def from_dict(cls, data: dict) -> "GraspCorrection":
        known = set(Point.__dataclass_fields__)
        return cls(points=[Point(**{k: v for k, v in p.items() if k in known}) for p in data.get("points", [])],
                   measured=str(data.get("measured", "")))

    @classmethod
    def load(cls, path: "str | Path | None") -> "GraspCorrection":
        """The table saved at path; empty when there is none (or it cannot be read, which it says)."""
        if path is None or not Path(path).is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            print(f"  [Grasp] Could not read {path} ({exc}) — no grasp correction.")
            return cls()

    def save(self, path: "str | Path") -> None:
        self.measured = self.measured or time.strftime("%Y-%m-%d %H:%M")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")


__all__ = ["GraspCorrection", "Point", "REACH_BAND_M"]
