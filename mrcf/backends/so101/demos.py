"""
demos.py — the SO-101's stream (streaming.StreamSpec): the demonstrations a
policy for this arm was trained on, as a LeRobot dataset, and so what it is
given and sends when it runs here — one definition (SO101Env.stream_spec,
stream_observation and stream_action are this), so a policy at run time sees
exactly what it was shown. Policies are trained outside this repository, on
datasets with these features.

    observation.state         the follower's six motors: degrees, gripper %
    observation.images.wrist  the gripper camera, RGB
    action                    the goal the follower was sent that tick

Not the overhead camera. It gets moved, and a learned policy that relied on it
would not survive the next move; the scripted steps around a learned one use
it, through their own calibration, instead.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import numpy as np

from mrcf.backends.so101.kinematics import JOINT_NAMES
from mrcf.streaming import StreamSpec

MOTORS = (*JOINT_NAMES, "gripper")
CAMERA_KEY = "wrist"
FPS = 30     # the gripper camera's own rate (robot_config CAMERA_FPS)
NAMES = [f"{m}.pos" for m in MOTORS]
ROBOT_TYPE = "so101_follower"


def spec(camera: "str | None", width: int, height: int, sensors: "dict | None" = None) -> StreamSpec:
    """
    The SO-101's stream, with the wrist camera `camera` at width x height (None:
    a rig with none) and the rig's sensors (streaming.sensor_specs).
    """
    cameras = {} if camera is None else {CAMERA_KEY: (camera, width, height)}
    return StreamSpec(fps=FPS, state=tuple(NAMES), action=tuple(NAMES), cameras=cameras,
                      robot_type=ROBOT_TYPE, sensors=dict(sensors or {}))


def features(frame_shape: tuple) -> dict:
    """The dataset's features, for frames of frame_shape (height, width, 3)."""
    return spec("", frame_shape[1], frame_shape[0]).features()


def observe(env, camera: "str | None", width: int, height: int, spec: "StreamSpec | None" = None) -> dict:
    """The follower now: {"observation.state": ..., "observation.images.wrist": ..., and the spec's sensors}."""
    from mrcf.streaming import sensor_observation

    pos = env._motor_pos()
    out = {"observation.state": np.array([pos[m] for m in MOTORS], np.float32)}
    if camera is not None:
        out[f"observation.images.{CAMERA_KEY}"] = env.render_image(camera, width, height)
    if spec is not None:
        out.update(sensor_observation(env, spec))
    return out


def action(goal: dict[str, float]) -> np.ndarray:
    """A {motor: value} goal as the dataset's action vector."""
    return np.array([goal[m] for m in MOTORS], np.float32)


def goal(action_vector) -> dict[str, float]:
    """An action vector as the {motor: value} goal stream_joints takes — action()'s inverse."""
    values = np.asarray(action_vector, float).ravel()
    if values.shape != (len(MOTORS),):
        raise ValueError(f"an SO-101 action is {len(MOTORS)} values ({', '.join(NAMES)}), not {values.shape}")
    return {m: float(v) for m, v in zip(MOTORS, values)}
