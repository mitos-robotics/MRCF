"""
wrist_shift.py — how far the wrist camera puts objects' feet from where the
overhead scan does, read from what every pick logged (flight recorder,
"calibration" / "wrist_foot": parallel_jaw aligners, FootAligner).

The shift between the two cameras (an arm's GRIPPER_CAM_SCAN_SHIFT_M) holds
while neither moves. 2026-11-18 it had drifted 5 mm on the SO-101, two days
after the overhead camera was re-posed, and nothing said so: picks took the
wrist camera's outlines for "a part of it" and came down on the overhead's
aim alone. Each pick's look is a reading of it; their median is the drift.

    from_runs(runs_dir, since=None, last=None) -> [[dx, dy], ...]   metres
    drift(offs, at_least) -> (median (dx, dy) m, spread mm) | None
    said(runs_dir) -> str | None     a line for a session's start when it has drifted
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DRIFT_SAY_MM = 2.5           # a median this far off is said at a session's start
AT_LEAST = 8                 # ... from at least this many picks' looks
LAST_SESSIONS = 5


def from_runs(runs_dir, since: "str | None" = None, last: "int | None" = None) -> list:
    """Every pick's wrist-foot offset (m, x y) in the recorder's sessions: since `since` (YYYY-MM-DD), the last `last`."""
    sessions = sorted(Path(runs_dir).glob("*/events.jsonl"))
    if since:
        sessions = [s for s in sessions if s.parent.name[:8] >= since.replace("-", "")]
    if last:
        sessions = sessions[-last:]
    offs = []
    for events in sessions:
        try:
            lines = events.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if '"wrist_foot"' not in line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("source") == "calibration" and e.get("kind") == "wrist_foot" and e.get("off_m"):
                offs.append([float(v) for v in e["off_m"]][:2])
    return offs


def drift(offs: list, at_least: int = AT_LEAST) -> "tuple[np.ndarray, float] | None":
    """(median offset m, spread mm) of the picks' wrist-foot offsets; None with fewer than at_least."""
    if len(offs) < at_least:
        return None
    d = np.asarray(offs, float)
    med = np.median(d, axis=0)
    return med, float(np.median(np.linalg.norm(d - med, axis=1)) * 1000.0)


def said(runs_dir, last: int = LAST_SESSIONS) -> "str | None":
    """A line for a session's start when the last sessions' picks put feet DRIFT_SAY_MM or more off the scan's."""
    offs = from_runs(runs_dir, last=last)
    got = drift(offs)
    if got is None:
        return None
    med, spread = got
    off = float(np.linalg.norm(med)) * 1000.0
    if off < DRIFT_SAY_MM:
        return None
    return (f"the wrist camera has put objects' feet {np.round(med * 1000, 1).tolist()} mm off the overhead "
            f"camera's over the last {len(offs)} picks (spread {spread:.1f} mm) — fit it again: "
            "python tools/calibrate_wrist_on_objects.py --from-runs --apply")
