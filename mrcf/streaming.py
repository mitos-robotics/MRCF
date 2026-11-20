"""
streaming.py — an arm driven one tick at a time by a learned policy: what it
observes and what it is sent (StreamSpec).

The spec is the arm's (capabilities.ActionStreaming: env.stream_spec — on the
SO-101, mrcf/backends/so101/demos.py): its joints and gripper in its own units, its
cameras (the wrist camera, if it has one), its sensors (a wrench, if it has a
force sensor: SENSORS, backends.rig.Sensor), its rate. A model is trained
elsewhere, on demonstrations with these features (features()), and is given
exactly that here. A model trained on one arm's stream runs only where the
stream is the same; mismatches() says what differs from a model's inputs and outputs,
and differences() from the whole stream it was trained on (names and robot
type too), before anything moves.

    spec = env.stream_spec
    observation = env.stream_observation()            # {"observation.state": ..., "observation.images.wrist": ...}
    sent, why = env.stream_action(action)             # the action as the arm sent it; why any was held back

Nothing here knows any arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATE = "observation.state"
ENVIRONMENT = "observation.environment_state"
ACTION = "action"
IMAGE_PREFIX = "observation.images."
SENSOR_PREFIX = "observation.sensors."


@dataclass(frozen=True)
class StreamSpec:
    """What an arm's stream carries, a tick at a time. See the module docstring."""
    fps: float                       # ticks a second: the rate it was recorded at and is run at
    state: tuple                     # observation.state's names, in order ("shoulder_pan.pos", ...)
    action: tuple                    # the action vector's names, in order
    cameras: dict = field(default_factory=dict)   # dataset key ("wrist") -> (the env's camera, width, height)
    robot_type: str = ""             # the dataset's robot_type ("so101_follower")
    environment: tuple = ()          # observation.environment_state's names: what the TASK adds to each
                                     # observation, not the arm (a goal: where to reach) — empty for most
    sensors: dict = field(default_factory=dict)   # dataset key -> (the env's sensor, its shape): sensor_specs()

    def features(self) -> dict:
        """The LeRobot dataset features: the state and action vectors, and a video per camera."""
        def vector(names):
            return {"dtype": "float32", "shape": (len(names),), "names": list(names)}
        out = {STATE: vector(self.state), ACTION: vector(self.action)}
        if self.environment:
            out[ENVIRONMENT] = vector(self.environment)
        for key, (_camera, width, height) in self.cameras.items():
            out[IMAGE_PREFIX + key] = {"dtype": "video", "shape": (int(height), int(width), 3),
                                       "names": ["height", "width", "channels"]}
        for key, (_sensor, shape) in self.sensors.items():
            out[SENSOR_PREFIX + key] = {"dtype": "float32", "shape": tuple(shape), "names": None}
        return out

    def to_dict(self) -> dict:
        """The spec as plain data, for a file (a policy package keeps the stream its model was trained on)."""
        return {"fps": float(self.fps), "state": list(self.state), "action": list(self.action),
                "cameras": {k: [camera, int(w), int(h)] for k, (camera, w, h) in self.cameras.items()},
                "robot_type": self.robot_type, "environment": list(self.environment),
                "sensors": {k: [name, list(shape)] for k, (name, shape) in self.sensors.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "StreamSpec":
        """to_dict's inverse."""
        return cls(fps=float(data["fps"]), state=tuple(data["state"]), action=tuple(data["action"]),
                   cameras={k: (c, int(w), int(h)) for k, (c, w, h) in (data.get("cameras") or {}).items()},
                   robot_type=data.get("robot_type", ""), environment=tuple(data.get("environment") or ()),
                   sensors={k: (n, tuple(s)) for k, (n, s) in (data.get("sensors") or {}).items()})

    def fingerprint(self) -> str:
        """
        What a model trained on this stream depends on, as a short id: the robot
        type, the rate, and every feature's name, type and shape — not which
        device a camera is, nor the env's own name for a sensor. Two arms with the
        same fingerprint run each other's models; a policy store can file the
        policies it offers under the fingerprint of the stream each was trained on.
        """
        import hashlib
        import json

        what = {"robot_type": self.robot_type, "fps": round(float(self.fps), 3),
                "features": {k: [v["dtype"], list(v["shape"]), v.get("names")] for k, v in sorted(self.features().items())}}
        return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:16]

    def mismatches(self, inputs: dict, outputs: dict) -> list:
        """
        Why a model with these inputs and outputs ({key: shape}, a LeRobot
        policy's input_features / output_features) cannot run on this stream;
        [] when it can. Images are taken channels first or last.
        """
        problems = []
        shapes = {k: tuple(v["shape"]) for k, v in self.features().items()}
        for key, shape in inputs.items():
            shape = tuple(int(n) for n in shape)
            want = shapes.get(key) if key != ACTION else None
            if want is None:
                problems.append(f"the model wants {key!r}, which this arm's stream does not have "
                                f"(it has {sorted(k for k in shapes if k != ACTION)})")
            elif key.startswith(IMAGE_PREFIX):
                if shape not in (want, (want[2], want[0], want[1])):
                    problems.append(f"{key}: the model was trained on {shape} frames, the arm gives {want}")
            elif shape != want:
                problems.append(f"{key}: the model takes {shape}, the arm gives {want}")
        for key, shape in outputs.items():
            shape = tuple(int(n) for n in shape)
            if key != ACTION:
                problems.append(f"the model outputs {key!r}, which this arm's stream cannot take")
            elif shape != shapes[ACTION]:
                problems.append(f"action: the model outputs {shape}, the arm takes {shapes[ACTION]}")
        return problems


def differences(trained: StreamSpec, arm: StreamSpec) -> list:
    """
    Why a model trained on the stream `trained` cannot run on an arm streaming
    `arm`; [] when it can. More than mismatches(), which has only a model's
    shapes: the robot type, the rate, and the state's and action's names — an
    arm with as many joints, but other ones or in other units, is another arm.
    A goal the model was trained with (environment) is the task's, given as it
    runs (LearnedPolicy(goal=)), not the arm's: it is not held against it.
    """
    from dataclasses import replace

    arm = replace(arm, environment=trained.environment)
    features = trained.features()
    problems = arm.mismatches({k: v["shape"] for k, v in features.items() if k != ACTION},
                              {ACTION: features[ACTION]["shape"]})
    if trained.robot_type and arm.robot_type and trained.robot_type != arm.robot_type:
        problems.append(f"the model was trained on a {trained.robot_type!r}, this arm is a {arm.robot_type!r}")
    if abs(float(trained.fps) - float(arm.fps)) > 1e-6:
        problems.append(f"the model was trained at {trained.fps:g} Hz, this arm streams at {arm.fps:g} Hz")
    for what, had, has in (("state", trained.state, arm.state), ("action", trained.action, arm.action)):
        if len(had) == len(has) and tuple(had) != tuple(has):
            problems.append(f"{what}: the model was trained on {list(had)}, the arm's is {list(has)}")
    return problems


def sensor_specs(sensors) -> dict:
    """
    StreamSpec.sensors for every sensor of a rig — its SENSORS ({name:
    backends.rig.Sensor}), or an env made with them: {name: (name, shape)}.
    """
    if not isinstance(sensors, dict):
        sensors = getattr(sensors, "_sensors", None) or {}
    return {name: (name, tuple(s.shape) or (1,)) for name, s in sensors.items()}


def sensor_observation(env, spec: StreamSpec) -> dict:
    """The spec's sensors now, as an observation takes them: {"observation.sensors.<key>": float32 array}."""
    return {SENSOR_PREFIX + key: np.asarray(env.read_sensor(name), np.float32).reshape(shape)
            for key, (name, shape) in spec.sensors.items()}

