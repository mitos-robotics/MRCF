"""
capabilities.py — what a backend can do beyond RobotEnv's required methods.

RobotEnv (robot_env.py) is what every backend must implement. A policy uses
more where the backend has it, and does without where it does not; each such
extra is one Protocol here, so a backend author has a checklist and a policy
asks one question:

    if isinstance(env, OverheadSighting):
        off = env.overhead_gripper_offset()

They are runtime-checkable: a backend has a capability by having its methods
(and, for WristServo, the property), whatever it inherits from. Nothing here
knows about any robot; it imports only the standard library.

  OrientationSearch   choose one gripper orientation that reaches a sequence of
                      points — hover, grasp, lift — so the wrist holds still
                      through it (pure kinematics, nothing moves)
  Carries             what the jaws hold, as part of the arm's shape for the motion guard
  JawDirection        the way the jaws open on the surface, held a given way
  GripperRotation     the gripper's orientation now
  GripperTilt         how far the gripper leans from straight down
  GripperBelowGrasp   how far the gripper reaches below its grasp point, held a given way
  GripSensing         whether the jaws are pressing on something (a load sensor)
  HeldWidth           tell the grasp-point model how wide the held object is
  WristServo          centre the jaws over an object from a wrist camera
  WristSurface        where the wrist camera's pixels land on the surface, from here
  OverheadSighting    where a fixed camera sees the gripper, against the kinematics
  Settling            hold still and wait until the arm has stopped
  OverheadCalibration the fixed camera's calibration, read and replaced while running
                      (overhead_calibration.py)
  ActionStreaming     driven a tick at a time by a learned policy, with what it
                      observes and takes (streaming.StreamSpec; mrcf/policies/learned.py)
  Freedrive           pushed by hand, holding itself up and staying where it is left
                      (a UR's teach mode) — not an arm that goes limp (RobotEnv has no
                      such method; an SO-101 with its torque off falls)
  BackOff             straight up slowly after a press stopped it
  DownRotation        the gripper's rotation straight down at a point, the jaws as the
                      arm opens them there
  MoveRefusals        how many moves its own checks refused before sending them, and why
                      the last one was
  GripperFits         whether the gripper straight down at a point is where it may go
  Sentry              what a watcher of the work surface saw move, other than the arm
  FixedViews          the other calibrated fixed cameras a scan keeps a frame from
  FrameAge            how old a camera's newest frame is
  CameraAtPose        where a camera would be, and what it would see of the surface, with
                      the jaws at a given point and rotation (pure geometry, nothing moves)

mrcf/backends/rig.py gives an arm the camera ones (WristServo, OverheadSighting,
Settling, OverheadCalibration, HeldWidth, CameraAtPose) from its kinematics, and
GripperRotation, JawDirection, DownRotation, MoveRefusals, GripperFits, Sentry and
FixedViews.

Shared code never asks an env for something by name (hasattr, getattr with a
default): what every arm has is on RobotEnv and called as it is; what only
some arms have is a Protocol here. tests/test_architecture.py holds it to
that. A name asked for can be renamed in one arm and the other then quietly
goes without; a capability an arm loses changes what tests/baselines/<arm>/
recorded it has (tests/test_arm_baselines.py), and that fails.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class OrientationSearch(Protocol):
    def approach_orientation_for(self, targets: list, gap_m: "float | None" = None,
                                 tols: "list | None" = None, flip: bool = False,
                                 prefer_tilt_deg: "float | None" = None,
                                 margin_deg: "float | None" = None,
                                 turn_deg: float = 0.0) -> "tuple | None":
        """
        (rotation 3x3, tilt_deg) of one gripper orientation that reaches every
        target in order within its tolerance, or None when none does. flip:
        the gripper turned half a turn about its approach axis. turn_deg: the
        jaws turned that far about it, counter-clockwise seen from above, from
        where the arm would open them there (to square them with something
        lying on its side).
        prefer_tilt_deg: start the search at this lean. margin_deg: approve
        only solutions this far from every joint limit. A backend may leave
        out the keyword arguments it cannot honour; callers retry without them.
        """


@runtime_checkable
class Carries(Protocol):
    def carrying(self, outline_uv, grip_width_m: float) -> None:
        """
        What the jaws now hold, from above in their own frame ([(u, v)] m from
        the grasp point, u along the way they open; None: nothing): part of the
        arm's shape (arm_shape) for the motion guard until the jaws open wider
        than grip_width_m. 2026-11-20, the UR5: a hammer carried across the
        table paused the arm every few centimetres, seen as something moving.
        """


@runtime_checkable
class JawDirection(Protocol):
    def jaw_direction(self, rotation):
        """
        The way the jaws open on the surface (unit x, y) with the gripper at
        `rotation` (3x3, world): as the gripper is mounted (an arm's
        JAW_SLIDE_AXIS). None where it opens straight up or down.
        """


@runtime_checkable
class GripperRotation(Protocol):
    def get_ee_rotation(self):
        """The gripper's orientation now, world frame: a 3x3 whose columns are its x, y, z."""


@runtime_checkable
class GripperTilt(Protocol):
    def gripper_tilt_deg(self, rot=None) -> float:
        """How far the gripper (now, or in orientation `rot`) leans from straight down, signed, degrees."""


@runtime_checkable
class GripperBelowGrasp(Protocol):
    def below_grasp_m(self, rot, gap_m: float) -> float:
        """
        How far the gripper's lowest point (a jaw tip's edge, or the wrist end)
        stands below its grasp point, held in orientation rot with the jaws
        gap_m apart: metres, never negative. Leaning, it is more than straight down.
        """


@runtime_checkable
class GripSensing(Protocol):
    def gripper_loaded(self) -> bool:
        """Whether the jaws are pressing on something now (sensed, not inferred from the opening)."""


@runtime_checkable
class HeldWidth(Protocol):
    def hold_width(self, width_m: float) -> None:
        """How wide the object between the jaws is, until they next open: where its centre is."""


@runtime_checkable
class WristServo(Protocol):
    # The pixel the object shows at in the wrist camera when the jaws are over
    # it, or None when the servo has not been calibrated.
    wrist_servo_target_px: "tuple[float, float] | None"

    def visual_servo_xy(self, detect: Callable, tol_px: float = 5.0, orientation=None,
                        point_height_m: "float | None" = None) -> "tuple[bool, str]":
        """
        Move the gripper (holding its height and `orientation`) until
        detect(frame, expected_px, radius_px) -> (u, v) | None finds the object
        at wrist_servo_target_px: (aligned, message). Leaves the arm where it
        is when it cannot.
        """


@runtime_checkable
class WristSurface(Protocol):
    def wrist_to_surface(self, px, height_m: float = 0.0):
        """
        Where wrist-camera pixels px ((N, 2), at the camera's render size) land
        on a plane height_m above the work surface, with the arm as it is now:
        (N, 2) world metres, or None when it cannot say. For a wrist camera
        with no located pose — local to the view round the jaws, the better
        the nearer the height its calibration was made at. Take the frame and
        call this with the arm still.
        """


@runtime_checkable
class OverheadSighting(Protocol):
    def overhead_gripper_offset(self):
        """
        Where a fixed camera sees the gripper minus where the kinematics put it
        (world frame, metres, (3,)); None when it cannot see it. Take it with
        the arm still.
        """


@runtime_checkable
class Settling(Protocol):
    def wait_still(self) -> float:
        """Hold the arm where it is and wait for it to stop; the last step it moved, mm."""


@runtime_checkable
class OverheadCalibration(Protocol):
    # The fixed camera's pose now, camera -> world 4x4 (metres), or None when
    # it has no calibrated model.
    overhead_camera_pose: "object | None"

    def set_overhead_calibration(self, T_world_cam=None, homography=None, gripper_tag_pose=None,
                                 gripper_tag_size_m: "float | None" = None) -> None:
        """
        Re-pose the fixed camera on a running arm: its pose (camera -> world),
        its pixels -> table homography, and where the gripper's tag sits on the
        gripper and the size it reads at. None leaves that one as it is.
        """


@runtime_checkable
class ActionStreaming(Protocol):
    # What its observations and actions are (streaming.StreamSpec): what a
    # model trained for this arm was trained on.
    stream_spec: "object"

    def stream_observation(self) -> dict:
        """
        The arm now, as stream_spec says: {"observation.state": float32 vector,
        "observation.images.<key>": RGB frame (height, width, 3) uint8, ...}.
        """

    def stream_action(self, action) -> "tuple[object, str | None]":
        """
        Send one action — a vector in stream_spec.action's order — through the
        arm's own checks, and the interlock: (the action as it was sent, why
        any of it was held back, or None). A goal it may not reach is held
        back, not raised.
        """


@runtime_checkable
class Freedrive(Protocol):
    def freedrive(self, on: bool) -> None:
        """
        on: the arm may be pushed by hand, and holds itself up and stays where
        it is let go. off: it holds where it is, under its own control again.
        """


@runtime_checkable
class BackOff(Protocol):
    def back_off(self, distance_m: float) -> bool:
        """
        The grasp point straight up by distance_m, slowly, the jaws as they
        are: after a press stopped the arm (skill_executor), so it is not left
        pushing on what it came down on. True when it got there. Only ever
        asked while nothing holds the interlock.
        """
        ...


@runtime_checkable
class DownRotation(Protocol):
    def down_rotation(self, at, flip: bool = False, turn_deg: float = 0.0):
        """
        The gripper's rotation (3x3) straight down at `at` with the jaws opening
        as the arm opens them there (its jaw axis), turned turn_deg about the
        approach, counter-clockwise seen from above; half a turn more when flip.
        """


@runtime_checkable
class MoveRefusals(Protocol):
    # How many moves the arm's own checks have refused before sending them (the
    # gripper's box), and why the last one was (None: none yet). A policy whose
    # move came to nothing tells a refusal from a move that fell short by them.
    move_refusals: int
    last_refusal: "str | None"


@runtime_checkable
class GripperFits(Protocol):
    def gripper_fits(self, at) -> bool:
        """Whether the gripper straight down with its grasp point at `at` (x, y, z) is where it may go (its box)."""


@runtime_checkable
class Sentry(Protocol):
    def motion_seen_since(self, t: float) -> "list | None":
        """
        Where something other than the arm moved on the work surface since t
        (time.monotonic): [(x, y, radius_m)]. None when nothing watches, so
        nothing can say what did not move.
        """

    def sentry_sources(self) -> set:
        """The interlock sources the watchers trip by when they see movement (not their own failures)."""


@runtime_checkable
class FixedViews(Protocol):
    def fixed_views(self) -> dict:
        """The calibrated fixed cameras other than the overhead one, {name: capture size}: a scan keeps a frame of each."""


@runtime_checkable
class FrameAge(Protocol):
    def frame_age_s(self, camera: str) -> float:
        """How old the camera's newest frame is, seconds; inf when it has none."""


@runtime_checkable
class CameraAtPose(Protocol):
    def camera_model_at(self, camera: str, width: int, height: int, at, rotation,
                        gap_m: "float | None" = None) -> "dict | None":
        """
        RobotEnv.camera_model as it would be with the grasp point at `at` (x, y, z)
        and the gripper turned to `rotation` (3x3), the jaws open gap_m (None: as
        they are): for a camera on the gripper, where that pose would put it; a
        fixed camera's as it is. None when the camera cannot be placed. Nothing
        moves: what a policy plans a look by.
        """


__all__ = ["OrientationSearch", "GripperRotation", "GripperTilt", "GripperBelowGrasp", "GripSensing", "HeldWidth",
           "WristServo", "OverheadSighting", "Settling", "OverheadCalibration", "ActionStreaming", "Freedrive",
           "BackOff", "DownRotation", "MoveRefusals", "GripperFits", "Sentry", "FixedViews", "FrameAge",
           "CameraAtPose"]
