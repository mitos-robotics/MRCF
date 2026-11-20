"""The wrist camera's drift against the overhead, read from what picks logged (mrcf/wrist_shift.py)."""

import json

import numpy as np

from mrcf import wrist_shift as ws


def _session(root, name, offs):
    d = root / name
    d.mkdir(parents=True)
    lines = [json.dumps({"t": 0, "source": "arm", "kind": "move"})]
    lines += [json.dumps({"t": 1, "source": "calibration", "kind": "wrist_foot", "off_m": o, "at": [0.2, 0.1]})
              for o in offs]
    (d / "events.jsonl").write_text("\n".join(lines) + "\n")


def test_every_picks_look_is_a_reading_and_their_median_the_drift(tmp_path):
    _session(tmp_path, "20261117_100000", [[0.010, 0.0]] * 3)
    _session(tmp_path, "20261118_100000", [[0.003, -0.002]] * 5 + [[0.004, -0.003]] * 4)
    assert len(ws.from_runs(tmp_path)) == 12
    assert len(ws.from_runs(tmp_path, since="2026-11-18")) == 9 and len(ws.from_runs(tmp_path, last=1)) == 9
    med, spread = ws.drift(ws.from_runs(tmp_path, last=1))
    assert np.allclose(med, [0.003, -0.002], atol=0.0011) and spread < 2.0
    assert ws.drift([[0.0, 0.0]] * 3) is None                       # too few to say


def test_it_is_said_at_the_start_only_when_it_has_drifted(tmp_path):
    _session(tmp_path, "20261118_100000", [[0.0005, 0.0005]] * 10)
    assert ws.said(tmp_path) is None
    _session(tmp_path, "20261118_110000", [[0.004, -0.003]] * 30)
    line = ws.said(tmp_path)
    assert line is not None and "--from-runs --apply" in line
