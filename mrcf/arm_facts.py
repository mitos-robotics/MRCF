"""
arm_facts.py — which of an arm's facts have been checked on the rig, not only
written down.

An arm's robot_config holds what is known about it: some measured by the rig's
own tools, some read off a datasheet, some a first guess marked MEASURE. A
grasp turns the jaws by some of them, and a guess there is a grasp that cannot
work, every time, on every object that is not square. 2026-11-20, the UR5's
first tool picks: JAW_SLIDE_AXIS — which way the Hand-E's fingers open in the
flange's frame — had been written as the flange's y and never looked at; the
letter blocks, square, were picked either way for a week; then the jaws came
down along the hammer's handle, on it, at 50 N (user: "there should be some
basic common sense ... so the next time we have a new deployment it doesn't
do something as dumb as this").

GRASP_FACTS are the facts a pick depends on. Each is checked once on the rig
(tools/check_jaw_direction.py for the jaws) and recorded here with its value:
rig/<arm>/checked_facts.json. Changed in the config since, it is unchecked
again. An arm whose config says <FACT>_CHECKED = True (the reason beside it)
counts as checked: the SO-101's jaws, by months of picks. A new arm (backends/
_template) has none checked. The env of a real arm lists what is unchecked
(RigEnv.unchecked_facts), and a pick that depends on one is refused with the
tool that checks it; an offline arm has nothing physical to be wrong.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

GRASP_FACTS = ("JAW_SLIDE_AXIS",)
FILE = "checked_facts.json"
# The tool that checks each, for the refusal's message.
CHECKED_BY = {"JAW_SLIDE_AXIS": "python tools/check_jaw_direction.py"}


def path(config) -> Path:
    return Path(config.RIG_DIR) / FILE


def _plain(value):
    try:
        return [round(float(v), 6) for v in value]
    except TypeError:
        return value


def load(config) -> dict:
    if getattr(config, "RIG_DIR", None) is None:
        return {}                                     # no rig folder: nothing recorded
    try:
        return json.loads(path(config).read_text())
    except (OSError, ValueError):
        return {}


def checked(config, name: str) -> bool:
    """Whether the arm's `name` has been checked on the rig as it is set now (or its config says so)."""
    if getattr(config, f"{name}_CHECKED", False):
        return True
    got = load(config).get(name) or {}
    return bool(got) and got.get("value") == _plain(getattr(config, name, None))


def unchecked(config, names=GRASP_FACTS) -> tuple:
    """Those of `names` the config sets that have not been checked on the rig."""
    return tuple(n for n in names if hasattr(config, n) and not checked(config, n))


def record(config, name: str, how: str) -> Path:
    """`name` checked on the rig as it is set now, by `how`: kept in rig/<arm>/checked_facts.json."""
    where = path(config)
    facts = load(config)
    facts[name] = {"value": _plain(getattr(config, name)), "how": how,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(json.dumps(facts, indent=1) + "\n")
    return where


def forget(config, name: str) -> None:
    """`name` no longer checked (seen to be wrong): dropped from rig/<arm>/checked_facts.json."""
    facts = load(config)
    if facts.pop(name, None) is not None:
        path(config).write_text(json.dumps(facts, indent=1) + "\n")


def refusal(names) -> str:
    """What to say, refusing a pick that depends on the unchecked `names`."""
    names = list(names)
    tools = sorted({CHECKED_BY.get(n, "") for n in names} - {""})
    return (f"this arm's {', '.join(names)} (which way its jaws open) has not been checked on the rig, and a pick "
            "turns the jaws by it — a guess there grips everything that is not square along its length. "
            + (f"Check it first: {' / '.join(tools)} (about a minute; the arm turns its wrist at home)." if tools
               else "Check it on the rig first."))
