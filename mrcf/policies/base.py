"""
mrcf/policies/base.py — what every policy library takes and returns.

A policy drives the arm to do one thing — pick an object, place it, hover over
it — from GEOMETRY: where the object's foot is, how tall and wide it is, what
stands next to it. It does not know the object's name, who it is, what the
scene is for, or what anyone remembers about it; whoever calls it turns those
into a Target, and records what the Outcome says happened. That is the line
between the policy library and the rest:

    world model / memory ──► Target, Obstacles ──► policy ──► RobotEnv
                       ◄──────── Outcome ◄────────┘

so the same pick runs in any scenario, and a scenario runs on any arm whose
policies take the same Targets.

A policy may use vision while it works (the wrist camera at a hover): that is
the Vision hook, which the caller passes in.

Stopping is not the policy's job either: the backend checks the interlock
before every command (robot_env.py, "Stopping"), and a stop arrives as
MotionStopped, which a policy lets through.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Lying:
    """
    An object fallen on its side, as geometry: world (x, y) metres. From a
    fixed camera's outline of it (mrcf/perception/lying.py), or the scenario.
    """
    foot_xy: np.ndarray                   # the end it stands on, upright, where it meets the surface
    top_xy: np.ndarray                    # the middle of the knob at its other end: what to grip
    length_m: float                       # foot to top: its height, stood up
    top_width_m: float                    # the knob's width
    widths: tuple = ()                    # ((m from the foot end, width m), ...) along it, when measured

    @property
    def axis(self) -> np.ndarray:
        """Unit (x, y) along it, from its foot toward its top."""
        d = np.asarray(self.top_xy, float)[:2] - np.asarray(self.foot_xy, float)[:2]
        return d / (np.linalg.norm(d) or 1.0)

    def widest_near_top(self, span_m: float) -> float:
        """Its widest within span_m either side of the knob's middle (top_width_m when not measured)."""
        knob = self.length_m - self.top_width_m / 2
        near = [w for at, w in self.widths if abs(at - knob) <= span_m]
        return max([self.top_width_m, *near])


@dataclass
class Target:
    """An object to act on, as geometry. World frame, metres."""
    position: np.ndarray                  # (x, y, z): the centre of its foot, on the surface
    foot_m: "float | None" = None         # foot diameter, as a scan measured it
    height_m: "float | None" = None       # measured height
    widths_mm: "dict | None" = None       # measured width up the object: {height_mm: width_mm}
    max_height_m: "float | None" = None   # not measured: the tallest it could be
    min_height_m: "float | None" = None   # not measured: the shortest it could be
    assumed_widths_mm: "dict | None" = None  # not measured: the widest it could be up its height,
                                             # {height_mm: width_mm}, as the deployment knows its kind
    tilt_deg: "float | None" = None       # how far to lean the gripper from straight down here,
                                          # when the scenario has a rule for it: where the arm's
                                          # search starts (None: as little as reaches)
    shape_looks: "int | None" = None      # how many separate measurements agree on height_m and
                                          # widths_mm (None: not said — taken as settled). One look
                                          # can be wrong, even with its two frames agreeing
    lying: "Lying | None" = None          # fallen on its side: how it lies (stand_up needs it)
    grasp_above_m: "float | None" = None  # grip it no lower than this above the surface: what a place
                                          # needs it held by (Outcome.regrip_above_m)
    grip_at_m: "float | None" = None      # grip it at this height above the surface: a recipe that held
                                          # it before (a deployment's catalogue). Planned by that, no
                                          # look for its height first; max_height_m still sets the hover
    faces_deg: "float | None" = None      # it has flat sides (a block): the direction of one, deg from the
                                          # world x axis, mod 90; the jaws close across them (None: round)
    surface_z_m: "float | None" = None    # it stands on something raised (a block on a block): the top
                                          # of that, world z, which its heights are above (None: the
                                          # work surface)
    outlines: "dict | None" = None        # its outline as each fixed camera saw it when it was found:
                                          # {camera: {"size": (w, h), "px": [[u, v], ...]}} — what a
                                          # look for its height fits the shape to, with the wrist
                                          # camera's (parallel_jaw/height_looks.py)

    @property
    def measured(self) -> bool:
        return self.height_m is not None


@dataclass
class Obstacle:
    """Something the gripper must not land on: an object standing at `position`."""
    position: np.ndarray                  # (x, y[, z]), world frame, m
    radius_m: "float | None" = None       # its foot's radius; None when not measured
    label: str = ""                       # for messages: what it is
    height_m: "float | None" = None       # how tall, or the tallest it may be; None when unknown
    widths_mm: "dict | None" = None       # its measured width up its height, {height_mm: width_mm}:
                                          # the fingers pass it only from their tips up
    outlines: "dict | None" = None        # its outline as each fixed camera saw it (Target.outlines): how tall
                                          # it can be at most, where its height is not measured


@dataclass
class Opening:
    """
    A container to put things into (a tub, a bowl, a tray), as geometry: what
    a place into it must stay inside, and how high its walls are. World frame,
    metres. Where it came from (a camera's outline of it) is the caller's.
    """
    outline_xy: np.ndarray                # (N, 2): its rim as seen from above, on the surface's plane — no wider
                                          # than it is (the caller errs inward where it cannot tell)
    rim_m: float                          # how high its walls stand above the surface: measured, or the tallest
                                          # they may be (rim_measured says which)
    rim_measured: bool = False
    label: str = ""                       # for messages: what it is
    floor_m: "float | None" = None        # its floor's height above the surface, when known; None: about the
                                          # surface's (a tub standing on the table)


@dataclass
class Measured:
    """What a policy's own camera measured about the object it was working on."""
    height_mm: float
    widths_mm: dict                       # {height_mm: width_mm}
    foot_mm: float
    slant_deg: float
    foot_xy_m: "np.ndarray | None" = None  # where its foot is, when the camera is located
    frames: list = field(default_factory=list)   # the RGB frames it was measured from
    looks: int = 1                        # separate looks that agree on it (a height looked for from
                                          # several sides counts each that saw it)

    def as_dict(self) -> dict:
        out = {"height_mm": self.height_mm, "widths_mm": self.widths_mm,
               "foot_mm": self.foot_mm, "slant_deg": self.slant_deg}
        if self.foot_xy_m is not None:
            out["base_world_m"] = [round(float(v), 5) for v in self.foot_xy_m]
        return out


@dataclass
class Outcome:
    """What a policy did. `ok` covers the whole action; `released` records a
    confirmed placement even when the subsequent retreat did not succeed.
    """
    ok: bool
    message: str
    refused: bool = False                 # declined without touching anything (no room, out of
                                          # reach): before moving, or after only looking from over it
    measured: "Measured | None" = None
    grasp: object = None                  # how it holds what it picked: give it back to place()
    at_xy: "np.ndarray | None" = None     # where it acted
    released: bool = False               # confirmed release at the placement target, even if retreat failed
    notes: list = field(default_factory=list)    # what it reported on the way, in order: why each
                                                 # attempt was retried, what it measured, its timing
    views: list = field(default_factory=list)    # [(label, RGB frame)] it took on the way, for the log:
                                                 # the wrist camera at each grasp, before closing
    approach: "str | None" = None         # how it came at the object (the strategy's name), when it says
    attempts: list = field(default_factory=list)  # one per attempt, in order: {"attempt", "approach",
                                                  # "tilt_deg", "result"} — result "ok" or why it was
                                                  # retried, as a short code ("closed_on_nothing", ...)
    around: dict = field(default_factory=dict)    # {obstacle label: Measured}: the neighbours its own
                                                  # camera measured on the way (to see the room they leave)
    regrip_above_m: "float | None" = None  # a place refused because of how the object is held: gripped
                                           # this high (above the surface) instead, it could be done
    blocked_by: list = field(default_factory=list)  # refused for room: the labels of the obstacles that each
                                                    # leave the jaws too little (what moving would free)
    diagnosis: "Diagnosis | None" = None  # a pick that missed: why its last close found nothing (Vision.diagnose)
    faces_deg: "float | None" = None      # a place of an object with flat sides: which way they face as
                                          # it was put down (Target.faces_deg's sense)


@dataclass
class Vision:
    """
    What a policy may ask of vision processing while it works.

      measure(frame_rgb, expected_px=None, model=None, expected_xy_m=None) -> dict
          The object nearest where it is expected, in one wrist-camera frame:
          {"ok", "why", and when ok "measurement" (plane_geometry's
          ObjectMeasurement) and optionally "occluded" (the outline runs into
          the gripper, so its top may be hidden)}. None: the framework's
          plane_geometry.measure_in_frame, which needs the camera located.
          One that also takes known=, target= and need_height=
          (plane_geometry.measure_near's) is given the objects around it, so
          it measures the one sent for and no neighbour.
      find(frame_rgb, expected_px, radius_px) -> (u, v) | None
          The object's pixel, for visual servoing. None: no servoing.
      lying(frame_rgb, to_surface, near_px, expected) -> (Lying | None, why)
          The object lying on its side near pixel near_px in one wrist-camera
          frame, measured through to_surface (capabilities.WristSurface's
          wrist_to_surface); expected: the Lying it was before. None: the
          framework's perception.lying.find_lying_in_frame.
      diagnose(frames, what) -> Diagnosis | None
          Why a close found nothing, from the wrist camera's frames at the
          grasp ([(label, RGB)], the jaws open over it and then shut) and what
          the object is called: policies.base.Diagnosis. None (or no hook):
          not said, the retry as before.
      between(frame_rgb, jaws_mask, gap_px) -> bool mask | None
          What lies between the jaw tips in a wrist-camera picture at the
          grasp. None (no hook): the segmenter, at and round the gap.
      relocate(env, near_xy) -> (x, y) | None
          Where the object last seen at near_xy (Target.position's frame) stands
          now, found afresh — by a deployment's scan, the arm moved out of its
          view — with the arm left clear of the work (home); None when nothing
          is there. Asked after a close that missed, which can push a
          free-standing object along: the retry comes at it where it stands.
          None: a retry looks for it with the wrist camera instead.
    """
    measure: "Callable | None" = None
    find: "Callable | None" = None
    lying: "Callable | None" = None
    relocate: "Callable | None" = None
    diagnose: "Callable | None" = None
    between: "Callable | None" = None


# Why a close found nothing (Vision.diagnose), and what the next attempt changes for it.
MISS_CAUSES = {
    "too_high": "the jaws closed above it: gripped lower next",
    "aim_off": "the jaws came down beside it: found again and aimed afresh",
    "lying": "it is lying on its side: not picked upright",
    "moved": "it was pushed away: found again",
    "unclear": "not clear from the pictures",
}


@dataclass
class Diagnosis:
    """Vision.diagnose's answer: one of MISS_CAUSES, and what the pictures showed, said."""
    cause: str
    said: str = ""


class Policies(ABC):
    """
    An arm's policy library. Every method takes the env first, returns an
    Outcome, and leaves recording what happened to the caller.
    """

    @abstractmethod
    def pick(self, env, target: Target, obstacles=(), vision: "Vision | None" = None) -> Outcome:
        """Pick up the object at target. Outcome.grasp describes the hold, for place()."""

    @abstractmethod
    def place(self, env, at, held: "Target | None" = None, grasp=None, obstacles=(),
              tilt_deg: "float | None" = None, surface_z: "float | None" = None,
              faces_deg: "float | None" = None) -> Outcome:
        """
        Put down what is held, its foot at `at` (x, y[, z]). grasp: pick()'s
        Outcome.grasp. tilt_deg: as Target.tilt_deg, for the place.
        surface_z: put it down on something raised (on top of another
        object): that top, world z; None: the work surface. faces_deg: turn
        what is held so its flat sides face so (a block squared on the one
        under it), as Target.faces_deg.
        """

    @abstractmethod
    def move_above(self, env, target: Target, height_m: "float | None" = None) -> Outcome:
        """Hover over target without touching it; height_m above the surface, or clear of it."""

    @abstractmethod
    def home(self, env) -> Outcome:
        """Go to the arm's safe resting pose."""

    @abstractmethod
    def holding(self, env) -> bool:
        """Whether the gripper holds something right now."""

    def pick_check(self, env, target: Target, obstacles=(), vision: "Vision | None" = None,
                   quick: bool = False) -> Outcome:
        """
        Whether pick() would go ahead here, worked out as it starts — nothing
        moved, nothing said: ok, or the Outcome refusing it. quick: the room
        between the obstacles only, not whether an approach reaches. What a
        caller checks a rearrangement by before making it (manipulation's
        make-room recovery).
        """
        return Outcome(False, f"{type(self).__name__} cannot check a pick before making it", refused=True)

    def nudge(self, env, target: Target, to_xy, obstacles=(), check_only: bool = False) -> Outcome:
        """
        Push target a short way across the surface, its foot from where it is
        to to_xy, without gripping it. check_only: whether it could, nothing
        moved. Outcome.at_xy: where it should stand now.
        """
        return Outcome(False, f"{type(self).__name__} has no nudge", refused=True)

    def place_in(self, env, opening: Opening, held: "Target | None" = None, grasp=None,
                 obstacles=(), drop: bool = False) -> Outcome:
        """
        Put what is held down inside a container (opening): over its walls,
        let go above its floor somewhere inside, clear of the walls and of what
        is already in it (obstacles), and back up over the walls. Outcome.at_xy:
        where it was let go. Its walls' height is looked for when
        Opening.rim_measured is False; Outcome.measured then has it (height_mm)
        for the caller to keep. drop: let go from over the walls instead of
        lowering it inside (a sturdy object).
        """
        return Outcome(False, f"{type(self).__name__} cannot put things into containers", refused=True)

    def pick_by_rim(self, env, target: Target, obstacles=(), vision: "Vision | None" = None) -> Outcome:
        """
        Pick target — a thin-walled open thing: a cup, a bowl, a scoop — up by
        one of its walls, a jaw either side of it. Outcome.grasp as pick()'s.
        """
        return Outcome(False, f"{type(self).__name__} cannot pick things up by a wall", refused=True)

    def open_gripper(self, env) -> Outcome:
        return Outcome(False, f"{type(self).__name__} has no open_gripper", refused=True)

    def close_gripper(self, env) -> Outcome:
        return Outcome(False, f"{type(self).__name__} has no close_gripper", refused=True)
