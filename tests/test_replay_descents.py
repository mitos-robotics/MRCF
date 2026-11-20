"""tools/so101/replay_descents.py: the servos fitted from the flight recorder, and the offline arm on them."""

import json

import numpy as np
import pytest

from tools.so101 import replay_descents as rd


def test_how_far_a_joint_stops_short_is_read_from_the_recorders_samples(tmp_path):
    rows, t = [], 1000.0
    joints = dict.fromkeys(rd.JOINTS, 0.0)
    for k in range(40):
        t += 0.05
        if k == 0:
            rows.append({"t": t, "stream": "sent", **{**joints, "elbow_flex": 10.0, "wrist_flex": -5.0}})
        held = {**joints, "elbow_flex": 7.5, "wrist_flex": -4.0}          # held 2.5 short going +, 1.0 going -
        rows.append({"t": t + 0.01, "stream": "read", **held, "gripper": 30.0})
    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    bands = rd.fit([path])
    assert bands["elbow_flex"][0] == pytest.approx(2.5) and np.isnan(bands["elbow_flex"][1])
    assert bands["wrist_flex"][1] == pytest.approx(1.0) and np.isnan(bands["wrist_flex"][0])


def test_the_servo_model_stops_short_by_its_band_each_way_and_moves_at_most_its_step():
    follower = rd._Follower({j: 0.0 for j in rd.JOINTS}, 50.0)
    for _ in range(40):
        follower.send_action({"elbow_flex.pos": 10.0, "wrist_flex.pos": -10.0})
    assert follower.bus.positions["elbow_flex"] == pytest.approx(10.0 - rd.SERVO_BANDS["elbow_flex"][0])
    assert follower.bus.positions["wrist_flex"] == pytest.approx(-10.0 + rd.SERVO_BANDS["wrist_flex"][1])
    follower = rd._Follower({j: 0.0 for j in rd.JOINTS}, 50.0)
    follower.send_action({"shoulder_pan.pos": 30.0})
    assert follower.bus.positions["shoulder_pan"] == pytest.approx(rd.SERVO_STEP_DEG)
