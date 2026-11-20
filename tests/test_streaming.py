"""
An arm's stream (streaming.StreamSpec): what a learned policy was trained on
and is given — on the SO-101, and on any arm.
"""

import numpy as np
import pytest

from mrcf.backends.so101 import demos
from mrcf.backends.so101.kinematics import JOINT_NAMES
from mrcf.streaming import StreamSpec

SO101_NAMES = [f"{n}.pos" for n in JOINT_NAMES] + ["gripper.pos"]


def test_the_so101s_demonstrations_record_what_they_always_have():
    # A model trained on the datasets recorded before the stream had a spec
    # of its own must still run.
    vector = {"dtype": "float32", "shape": (6,), "names": SO101_NAMES}
    assert demos.features((480, 640, 3)) == {
        "observation.state": vector, "action": vector,
        "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    }
    assert list(demos.features((480, 640, 3))) == ["observation.state", "action", "observation.images.wrist"]


def _act_like(state=6, action=6, image=(3, 480, 640)):
    return {"observation.state": (state,), "observation.images.wrist": image}, {"action": (action,)}


def test_a_model_trained_on_this_stream_fits_it_and_one_trained_on_another_does_not():
    spec = demos.spec("front", 640, 480)
    assert spec.mismatches(*_act_like()) == []
    assert spec.mismatches(*_act_like(image=(480, 640, 3))) == []              # channels last too
    assert "frames" in spec.mismatches(*_act_like(image=(3, 240, 320)))[0]
    assert "action" in spec.mismatches(*_act_like(action=7))[0]
    assert "does not have" in spec.mismatches({"observation.images.top": (3, 480, 640)}, {"action": (6,)})[0]
    assert "cannot take" in spec.mismatches({}, {"action.ee": (3,)})[0]


def test_any_arm_says_what_its_stream_is():
    spec = StreamSpec(fps=50, state=("x", "y", "z", "gripper"), action=("dx", "dy", "dz", "gripper"),
                      cameras={"wrist": ("cam0", 128, 96), "side": ("cam1", 64, 48)}, robot_type="ur5")
    f = spec.features()
    assert f["observation.state"]["names"] == ["x", "y", "z", "gripper"] and f["action"]["shape"] == (4,)
    assert f["observation.images.wrist"]["shape"] == (96, 128, 3) and f["observation.images.side"]["shape"] == (48, 64, 3)


def test_a_model_from_an_arm_with_as_many_joints_but_other_ones_does_not_fit():
    from dataclasses import replace
    from mrcf.streaming import differences
    so101 = demos.spec("front", 640, 480)
    assert differences(so101, so101) == []
    other = replace(so101, state=tuple(f"joint_{i}" for i in range(6)), action=tuple(f"joint_{i}" for i in range(6)),
                    robot_type="ur5")
    assert so101.mismatches({"observation.state": (6,), "observation.images.wrist": (3, 480, 640)},
                            {"action": (6,)}) == []                    # its shapes alone would let it through
    found = " | ".join(differences(other, so101))
    assert "'ur5'" in found and "state: the model was trained on ['joint_0'" in found and "action:" in found
    assert "Hz" in " ".join(differences(replace(so101, fps=10), so101))
    reach = replace(so101, environment=("goal.0", "goal.1", "goal.2"))
    assert differences(reach, so101) == []                             # a goal is the task's, not the arm's


# ── The SO-101's stream is its demonstrations' ────────────────────────────────

def _so101():
    from test_so101_env import offline_so101
    env = offline_so101()
    env._camera_capture_wh = (640, 480)
    env._stream_tip_floor_z, env._stream_bounds = None, None
    return env


def test_the_so101_streams_what_its_demonstrations_recorded():
    env = _so101()
    spec = env.stream_spec
    assert spec.state == spec.action == tuple(demos.NAMES) and spec.fps == demos.FPS
    assert spec.cameras == {"wrist": ("front", 640, 480)}
    got, want = env.stream_observation(), demos.observe(env, "front", 640, 480)
    assert got.keys() == want.keys()
    assert all(np.array_equal(got[k], want[k]) for k in got)


def test_an_so101_action_goes_out_through_its_stream_checks():
    env = _so101()
    action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 40.0])
    sent, why = env.stream_action(action)
    assert why is None and np.allclose(sent, action)
    assert env._robot.sent[-1] == {f"{n}.pos": v for n, v in zip([*JOINT_NAMES, "gripper"], action)}
    with pytest.raises(ValueError, match="6 values"):
        env.stream_action(np.zeros(5))


# ── Sensors and cameras a rig may or may not have ─────────────────────────────

def test_a_stream_carries_the_rigs_sensors_at_their_shapes():
    from mrcf.backends.rig import Sensor
    from mrcf.streaming import sensor_observation, sensor_specs

    class _Env:
        _sensors = {"wrench": Sensor(kind="force", shape=(6,)), "pad": Sensor(kind="pressure")}

        def read_sensor(self, name):
            return np.arange(6.0) if name == "wrench" else 0.5

    spec = StreamSpec(fps=30, state=("a",), action=("a",), sensors=sensor_specs(_Env()))
    features = spec.features()
    assert features["observation.sensors.wrench"]["shape"] == (6,) and features["observation.sensors.pad"]["shape"] == (1,)
    got = sensor_observation(_Env(), spec)
    assert got["observation.sensors.pad"].dtype == np.float32 and got["observation.sensors.pad"].shape == (1,)
    assert np.array_equal(got["observation.sensors.wrench"], np.arange(6.0, dtype=np.float32))
    # A model that wants a sensor this stream does not carry is refused, as a camera is.
    assert "observation.sensors.wrench" in StreamSpec(fps=30, state=("a",), action=("a",)).mismatches(
        {"observation.state": (1,), "observation.sensors.wrench": (6,)}, {"action": (1,)})[0]


def test_an_so101_with_no_wrist_camera_streams_its_joints_alone():
    spec = demos.spec(None, 640, 480)
    assert spec.cameras == {} and "observation.images.wrist" not in spec.features()
    env = _so101()
    env._camera_name = None
    assert env.stream_spec.cameras == {}
    assert set(env.stream_observation()) == {"observation.state"}
