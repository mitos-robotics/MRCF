"""
Each arm, as the shared code builds it, is as it was recorded
(tests/baselines/<arm>/, by tests/arm_baselines.py): its settings, its env
(capabilities, constants, state) and its plans for standard scenes.

A change made while working on one arm that changes another fails here, for
the arm it reaches, saying what changed. Meant for that arm too, record it:

    MRCF_UPDATE_BASELINES=1 pytest tests/test_arm_baselines.py -k so101

and commit tests/baselines/so101/ with the change. The SO-101's env and plans
need lerobot, placo and its URDF (CI's so101 job has them); the UR5's need
OpenCV. Where they are missing those sections are skipped, not passed — and
fail instead for the arms MRCF_BASELINES_REQUIRE names (CI sets it in the
jobs meant to check them).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import arm_baselines as ab
from mrcf import backends

UPDATE_VAR = "MRCF_UPDATE_BASELINES"
UPDATE = os.environ.get(UPDATE_VAR, "").strip() not in ("", "0")
# Arms whose every section must be recorded here, not skipped: CI's jobs that
# have their libraries (so101.yml: the SO-101; tests.yml's full: the UR5).
REQUIRE_VAR = "MRCF_BASELINES_REQUIRE"
REQUIRED = {a.strip() for a in os.environ.get(REQUIRE_VAR, "").split(",") if a.strip()}
MAX_LINES = 40


def _arms() -> list:
    """The arms in this repository (backends/<name>/, the template too): not installed plugins, which keep their own."""
    here = Path(backends.__file__).resolve().parent
    return sorted(p.name for p in here.iterdir() if (p / "__init__.py").is_file() and not p.name.startswith("."))


ARMS = _arms()


@pytest.fixture(scope="module")
def recorded():
    """Every arm recorded once, all at once (each in a process of its own)."""
    pool = ThreadPoolExecutor(max_workers=len(ARMS))
    futures = {arm: pool.submit(ab.record, arm) for arm in ARMS}
    yield lambda arm: futures[arm].result()
    pool.shutdown(wait=True)


def _changed(arm: str, section: str, lines: list) -> str:
    name = "the template" if arm.startswith("_") else f"the {backends.arm(arm).NAME}"
    shown = lines[:MAX_LINES] + ([f"... and {len(lines) - MAX_LINES} more"] if len(lines) > MAX_LINES else [])
    return "\n".join([
        f"{name[0].upper()}{name[1:]}'s {section} changed ({len(lines)} difference{'s' if len(lines) != 1 else ''}, "
        f"recorded -> now):", *("  " + line for line in shown), "",
        f"Working on {name} and meant it: record it, and commit tests/baselines/{arm}/ with the change:",
        f"    {UPDATE_VAR}=1 pytest tests/test_arm_baselines.py -k {arm}",
        f"Working on another arm: the change reaches {name} too. Keep it to the arm it was made for (its",
        f"config, or a GripperSpec field that arm sets), or behind a switch left off for {name} until it",
        "has been tried there (CONTRIBUTING.md, \"A change for one arm\").",
    ])


@pytest.mark.parametrize("section", ab.SECTIONS)
@pytest.mark.parametrize("arm", ARMS)
def test_the_arm_is_as_recorded(arm, section, recorded):
    got = recorded(arm)[section]
    if set(got) == {"skipped"}:
        if arm in REQUIRED:
            pytest.fail(f"{arm} {section}: {got['skipped']}, and {REQUIRE_VAR} says it is checked here.")
        pytest.skip(f"{arm} {section}: {got['skipped']}")
    if UPDATE:
        ab.save(arm, section, got)
        return
    want = ab.load(arm, section)
    if want is None:
        pytest.fail(f"No baseline of the {arm}'s {section}: record it with "
                    f"{UPDATE_VAR}=1 pytest tests/test_arm_baselines.py -k {arm}")
    lines = ab.differences(want, got, ab.TOLERANCE[section])
    if lines:
        pytest.fail(_changed(arm, section, lines), pytrace=False)


def test_every_baseline_is_of_an_arm_there_is():
    kept = sorted(p.name for p in ab.BASELINES.iterdir() if p.is_dir()) if ab.BASELINES.is_dir() else []
    assert [name for name in kept if name not in ARMS] == []


# ── The comparison itself ─────────────────────────────────────────────────────

def test_a_number_that_moves_is_a_difference_and_one_within_tolerance_is_not():
    assert ab.differences({"a": 1.0}, {"a": 1.0 + 1e-12}, 1e-9) == []
    assert ab.differences({"a": 1.0}, {"a": 1.001}, 1e-9) == ["a: 1.0 -> 1.001"]
    assert ab.differences({"x": [10.0, 20.0]}, {"x": [10.04, 20.0]}, 0.05) == []


def test_what_is_added_or_gone_is_named():
    assert ab.differences({"a": 1, "b": 2}, {"a": 1, "c": 3}, 0.0) == ["b: 2 -> (gone)", "c: (new) -> 3"]


def test_a_message_whose_rounded_number_moves_by_one_is_the_same_and_a_different_message_is_not():
    assert ab.differences("the jaws have 23 mm here", "the jaws have 24 mm here", 0.05) == []
    assert ab.differences("the jaws have 23 mm here", "the jaws have 31 mm here", 0.05) != []
    assert ab.differences("would pick it by top_down", "would pick it by near_base", 0.05) != []


def test_a_bool_is_not_a_number():
    assert ab.differences({"ok": True}, {"ok": 1}, 1.0) == ["ok: True -> 1"]


def test_paths_into_the_recording_folders_are_named_not_spelled_out():
    got = ab._replace_paths({"p": "/tmp/x/home/rig/so101/urdf/a.urdf"}, {"/tmp/x/home": "<MRCF_HOME>", "/tmp/x": "<tmp>"})
    assert got == {"p": "<MRCF_HOME>/rig/so101/urdf/a.urdf"}
