"""
safety.py — Independent safety oversight layer.

Operates in two modes, both architecturally separate from the VLM and skill executor:

  1. Plan-time  — validate_plan() checks the proposed action sequence before any
                  execution begins: object/target existence and workspace bounds,
                  plus any robot-specific sequence rules registered via configure().
                  Blocked actions are removed; warnings are printed.

  2. Execution-time — ExecutionMonitor polls contact force and the end-point's
                  height during skill execution. run_skill() registers it as one
                  of the interlock's per-cycle checks (interlock.py), so a hard
                  limit stops the arm within one control cycle, mid-move;
                  approaches are recorded as near-miss events.

Near-miss logging is a first-class feature.  When the robot approaches a constraint
boundary without crossing it, the event is recorded with full state context and
printed when the margin is small, so near-misses are visible without halting.

Module-level configuration
--------------------------
Call safety.configure() once at startup (from your entry-point script) to set
robot-specific workspace bounds, force thresholds, and sequence validation rules.
Without a configure() call the safety layer runs with permissive defaults (no
workspace constraints, no sequence rules) so unrelated robots are unaffected.

    safety.configure(
        bounds=WorkspaceBounds(x_min=-0.4, x_max=0.4, z_min=0.43, z_max=0.72),
        thresholds=SafetyThresholds(skill_overrides={"pick": {"max_contact_force_n": 4.0}}),
        sequence_validators=[my_gripper_state_validator],
    )

Module-level monitor
--------------------
A module-level singleton monitor is set by skill_executor at the start of each skill
and cleared at the end. The interlock polls it every control cycle; a skill that
is not run through a backend's guarded commands can also poll it itself.

    safety.set_monitor(monitor)   # called at start of run_skill
    safety.check()                # True = halt; for skills polling by hand
    safety.clear_monitor()        # called at end of run_skill

Architectural constraint
------------------------
This module imports only robot_env (for type hints). It is NOT imported by
robot_env.py, so there is no circular dependency.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mrcf.robot_env import RobotEnv


# ── Constraint definitions ─────────────────────────────────────────────────────

@dataclass
class WorkspaceBounds:
    """
    Cartesian limits for the controlled end-point in world frame (metres).

    Defaults are ±inf — no constraint — so the workspace check is a no-op
    until the caller provides actual limits via safety.configure().
    Set values appropriate for your robot and workspace.
    """
    x_min: float = -float("inf")
    x_max: float =  float("inf")
    y_min: float = -float("inf")
    y_max: float =  float("inf")
    z_min: float = -float("inf")
    z_max: float =  float("inf")


DEFAULT_CONTACT_FORCE_N = 5.0
DEFAULT_NEAR_MISS_FORCE_N = 3.5


@dataclass
class SafetyThresholds:
    """
    Dynamic constraint thresholds for execution monitoring.

    Per-skill overrides allow different limits for precision tasks vs rough moves.
    Define robot- and task-specific overrides via safety.configure() or by
    constructing a SafetyThresholds instance directly.
    """
    # None: the arm's own (env.contact_force_limits: how good its force reading
    # is decides what a stop can be — a CB3's estimate moves by more than 5 N in
    # a slow move), else DEFAULT_CONTACT_FORCE_N / DEFAULT_NEAR_MISS_FORCE_N.
    max_contact_force_n:   float | None = None    # N — hard stop, skill aborted
    near_miss_force_n:     float | None = None    # N — near-miss logged, execution continues
    max_ee_z_descent_m:    float = 0.005  # m below z_min — tolerance band before halt

    # Per-skill overrides: {skill_name: {field: value, ...}}
    # Define these in your robot's config, not here.
    skill_overrides: dict = field(default_factory=dict)

    def for_skill(self, skill_name: str) -> SafetyThresholds:
        """Return a copy with per-skill overrides applied."""
        overrides = self.skill_overrides.get(skill_name, {})
        if not overrides:
            return self
        t = copy.copy(self)
        for k, v in overrides.items():
            if hasattr(t, k):
                setattr(t, k, v)
        return t


# ── Module-level configuration ────────────────────────────────────────────────
# Permissive defaults — framework adds no constraints unless the deployment configures them.

_bounds:               WorkspaceBounds       = WorkspaceBounds()
_thresholds:           SafetyThresholds      = SafetyThresholds()
_sequence_validators:  list[Callable]        = []


def configure(
    bounds:              WorkspaceBounds | None  = None,
    thresholds:          SafetyThresholds | None = None,
    sequence_validators: list[Callable] | None   = None,
) -> None:
    """
    Set robot-specific safety configuration.  Call once at startup.

    bounds              — workspace limits for the end-point.
    thresholds          — force/velocity thresholds and per-skill overrides.
    sequence_validators — list of callables, each with signature:
                            (actions: list[dict], scene_state) -> list[PlanViolation]
                          Used for action-sequence logic that is robot-specific
                          (e.g. gripper-state sequencing for manipulation robots).
    """
    global _bounds, _thresholds, _sequence_validators
    if bounds is not None:
        _bounds = bounds
    if thresholds is not None:
        _thresholds = thresholds
    if sequence_validators is not None:
        _sequence_validators = sequence_validators


# ── Near-miss events ───────────────────────────────────────────────────────────

@dataclass
class NearMissEvent:
    """
    A constraint boundary approach that did not cross into violation.

    Recorded for every approach to a limit and kept on the monitor; printed
    when the margin is under 20%.
    """
    timestamp:        datetime
    skill_name:       str
    constraint:       str           # e.g. "contact_force", "ee_z_min"
    observed_value:   float         # the measured value
    limit_value:      float         # the hard-stop threshold
    ee_pos:           list[float] | None = None
    ee_state_value:   float | None       = None  # e.g. gripper width (mm) or joint angle

    def margin_pct(self) -> float:
        """Proximity to limit as a percentage (0% = at limit, 100% = no risk)."""
        if self.limit_value == 0:
            return 0.0
        return max(0.0, 100.0 * (1.0 - self.observed_value / self.limit_value))

    def summary_line(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        return (
            f"  [{ts}] {self.skill_name} near-miss {self.constraint}="
            f"{self.observed_value:.2f} (limit {self.limit_value:.2f}, "
            f"{self.margin_pct():.0f}% margin)"
        )


# ── Plan-time validation ───────────────────────────────────────────────────────

class PlanViolation:
    """One constraint violation found during plan-time checking."""
    def __init__(
        self,
        action:   dict,
        reason:   str,
        severity: Literal["block", "warn"],
    ):
        self.action   = action
        self.reason   = reason
        self.severity = severity

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.action.get('action', '?')}: {self.reason}"


def bounds() -> WorkspaceBounds:
    """The work volume configure() set: open (±inf) until it has been."""
    return _bounds


def validate_plan(
    actions:              list[dict],
    scene_state,                               # vlm.SceneState (duck-typed to avoid circular import)
    bounds:               WorkspaceBounds | None  = None,
    sequence_validators:  list[Callable] | None   = None,
) -> tuple[list[dict], list[PlanViolation]]:
    """
    Validate a proposed action sequence before execution begins.

    Per-action checks (always run):
      - Object and target existence in scene
      - Position within workspace bounds

    Sequence checks (robot-specific, injected via sequence_validators or
    configured globally via safety.configure()):
      - Any stateful logic across the action sequence, e.g. gripper-state
        sequencing for manipulation robots

    Returns (filtered_actions, all_violations).
    Blocked actions are removed from filtered_actions.
    Warned actions pass through but are logged.
    """
    effective_bounds     = bounds              if bounds              is not None else _bounds
    effective_validators = sequence_validators if sequence_validators is not None else _sequence_validators

    violations: list[PlanViolation] = []
    filtered:   list[dict]          = []

    # Run sequence validators (stateful, cross-action logic).
    seq_violations: list[PlanViolation] = []
    for validator in effective_validators:
        seq_violations.extend(validator(actions, scene_state))

    # Build a set of action dict ids that are blocked by sequence validators.
    seq_blocked_ids: set[int] = {id(v.action) for v in seq_violations if v.severity == "block"}
    violations.extend(seq_violations)

    for action in actions:
        name   = action.get("action", "")
        params = {k: v for k, v in action.items() if k != "action"}
        if "parameters" in params and isinstance(params.get("parameters"), dict):
            nested = params.pop("parameters")
            params.update(nested)

        action_violations = _check_action(name, params, scene_state, effective_bounds)
        violations.extend(action_violations)

        blocked = (
            any(v.severity == "block" for v in action_violations)
            or id(action) in seq_blocked_ids
        )
        # Collect every violation that pertains to this action (per-action +
        # sequence validators that tagged this dict) so warnings are reported
        # regardless of whether the action is blocked or passes through.
        related_seq = [v for v in seq_violations if id(v.action) == id(action)]
        for v in action_violations + related_seq:
            if v.severity == "block":
                print(f"  [Safety:Plan] BLOCKED — {v}")
            else:
                print(f"  [Safety:Plan] WARN — {v}")
        if not blocked:
            filtered.append(action)

    n_blocked = len(actions) - len(filtered)
    if n_blocked:
        print(f"  [Safety:Plan] {n_blocked} action(s) blocked before execution.")
    return filtered, violations


def _check_action(
    name:        str,
    params:      dict,
    scene_state,
    bounds:      WorkspaceBounds,
) -> list[PlanViolation]:
    violations: list[PlanViolation] = []
    action = {"action": name, "parameters": params}

    # ── Object existence ───────────────────────────────────────────────────────
    obj_name = params.get("object_name", "")
    if obj_name:
        obj = scene_state.find(obj_name, among="objects" if name == "pick" else "all")
        if obj is None:
            violations.append(PlanViolation(
                action, f"object '{obj_name}' not found in scene", "block"
            ))
        elif obj.position is not None:
            _check_workspace(action, np.array(obj.position[:3]), bounds, violations)
            if name == "move_above" and params.get("height") is not None:
                # Objects are located by their foot on the work plane; height
                # is above that plane, not an absolute world z coordinate.
                try:
                    height = float(params["height"])
                    if not np.isfinite(height) or height < 0:
                        raise ValueError("height must be finite and non-negative")
                except (TypeError, ValueError, OverflowError):
                    violations.append(PlanViolation(action, "height must be a finite, non-negative number", "block"))
                else:
                    goal = np.asarray(obj.position[:3], float).copy()
                    goal[2] += height
                    _check_workspace(action, goal, bounds, violations)

    # ── Target existence ───────────────────────────────────────────────────────
    tgt_name = params.get("target_name", "")
    if tgt_name:
        tgt = scene_state.find(tgt_name)
        if tgt is None:
            violations.append(PlanViolation(
                action, f"target '{tgt_name}' not found in scene", "block"
            ))
        else:
            if tgt.position is not None:
                _check_workspace(action, np.array(tgt.position[:3]), bounds, violations)

    return violations


def _check_workspace(
    action:     dict,
    pos:        np.ndarray,
    bounds:     WorkspaceBounds,
    violations: list[PlanViolation],
) -> None:
    # These BLOCK rather than warn: a target outside the reachable workspace
    # can't be reached by definition, so letting it through just means an IK
    # solver chases an impossible point — on real hardware that drives joints
    # hard against their limits and thrashes the arm. If a rejected target is
    # genuinely reachable, widen WORKSPACE_BOUNDS rather than downgrading this.
    hint = " (widen WORKSPACE_BOUNDS in the robot's config if this is actually reachable)"
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if not np.isfinite([x, y, z]).all():
        violations.append(PlanViolation(action, "target position must be finite", "block"))
        return
    if not (bounds.x_min <= x <= bounds.x_max):
        violations.append(PlanViolation(
            action,
            f"target x={x:.3f} m outside workspace [{bounds.x_min}, {bounds.x_max}]" + hint,
            "block",
        ))
    if not (bounds.y_min <= y <= bounds.y_max):
        violations.append(PlanViolation(
            action,
            f"target y={y:.3f} m outside workspace [{bounds.y_min}, {bounds.y_max}]" + hint,
            "block",
        ))
    if z < bounds.z_min - 1e-9:
        violations.append(PlanViolation(
            action,
            f"target z={z:.3f} m below floor limit {bounds.z_min}",
            "block",
        ))
    if z > bounds.z_max + 1e-9:
        violations.append(PlanViolation(
            action,
            f"target z={z:.3f} m above ceiling limit {bounds.z_max}",
            "block",
        ))


# ── Execution-time monitoring ─────────────────────────────────────────────────

# A tighter contact limit while a policy makes its final approach (a pick's last
# descent onto the object, a place's onto the surface): the arm's own,
# env.contact_force_limits["approach_contact_force_n"], when it gives one. Slow
# and straight there, its force reading is quieter than over a whole move, and
# a press there is the jaws landing on the object, not beside it.
_approach_limit_n: "float | None" = None


@contextlib.contextmanager
def approach_contact_limit(env):
    """While it lasts, the monitor stops at the arm's approach limit, if it has one (see _approach_limit_n)."""
    global _approach_limit_n
    given = env.contact_force_limits.get("approach_contact_force_n")
    was = _approach_limit_n
    if given is not None:
        # Counted from the approach's start, not the last move's: a reading that
        # is a change since a move began drifts over a long one, and the tight
        # limit would take that drift for a contact (HISTORY.md).
        env.zero_contact_force()
        _approach_limit_n = float(given) if was is None else min(float(given), was)
    try:
        yield
    finally:
        _approach_limit_n = was


# The sentry's movement this near a spot (past the moving thing's own size)
# may have moved what is there.
MOVED_NEAR_M = 0.10


def moved_near(env, xy, since: float, near_m: float = MOVED_NEAR_M) -> "float | None":
    """
    How near to xy (metres, the work surface) the sentry saw something other
    than the arm move since `since` (time.monotonic), when within near_m of
    it — a hand, a block knocked over; None when nothing did, or the arm has
    no sentry to say (RigEnv.motion_seen_since).
    """
    from mrcf.capabilities import Sentry

    moved = env.motion_seen_since(since) if isinstance(env, Sentry) else None
    if not moved:
        return None
    x0, y0 = float(xy[0]), float(xy[1])
    nearest = min(float(np.hypot(x - x0, y - y0)) - r for x, y, r in moved)
    return max(nearest, 0.0) if nearest <= near_m else None


def descend_onto(env, target, **move_kw) -> bool:
    """
    One final descent — the jaws down round an object, or an object down onto
    a surface: env.move_ee_to(target, **move_kw) under the arm's approach
    limit, counted from this descent's own start. Every descent of a pick or
    place, first or retried, goes this way; the moves between them (up,
    across, a lean) are not held to it.
    """
    with approach_contact_limit(env):
        return env.move_ee_to(target, **move_kw)


class ExecutionMonitor:
    """
    Lightweight monitor polled during skill execution phases.

    Reads contact force and EE position from env each time check() is called.
    A hard limit violation sets the halt flag and returns True; callers should
    abort the current skill.  Near-miss events are accumulated for post-skill
    review and learning.

    Uses module-level defaults from safety.configure() unless explicit
    thresholds/bounds are passed at construction.

    Usage::

        monitor = ExecutionMonitor(env, "pick")
        if monitor.check():
            return False, "safety halt"
    """

    def __init__(
        self,
        env:        "RobotEnv",
        skill_name: str,
        thresholds: SafetyThresholds | None = None,
        bounds:     WorkspaceBounds  | None = None,
    ):
        self.env         = env
        self.skill_name  = skill_name
        self.thresholds  = (thresholds or _thresholds).for_skill(skill_name)
        # The force limits: the deployment's, else the arm's, else the defaults.
        arm = env.contact_force_limits
        t = self.thresholds
        self.max_contact_force_n: float = (t.max_contact_force_n if t.max_contact_force_n is not None
                                           else float(arm.get("max_contact_force_n", DEFAULT_CONTACT_FORCE_N)))
        self.near_miss_force_n: float = (t.near_miss_force_n if t.near_miss_force_n is not None
                                         else float(arm.get("near_miss_force_n", DEFAULT_NEAR_MISS_FORCE_N)))
        self.bounds      = bounds or _bounds
        self.near_misses: list[NearMissEvent] = []
        self.halt_reason: str | None = None
        self._halted     = False
        self._checks     = 0

    def poll(self) -> str | None:
        """check() as an interlock per-cycle check: why to stop, or None."""
        return self.halt_reason if self.check() else None

    def check(self) -> bool:
        """
        Poll sensors and evaluate constraints.

        Returns True if execution should halt (hard limit crossed).
        Logs near-miss events as a side-effect.

        Reads universal sensors first (EE position, contact force); then
        attempts to read end-effector state (e.g. gripper width) separately
        so robots without that sensor still get force/position monitoring.
        """
        if self._halted:
            return True
        self._checks += 1

        # ── Universal sensor reads ─────────────────────────────────────────────
        try:
            ee    = self.env.get_ee_pos()
            force = self.env.get_contact_force()
        except Exception:
            return False   # sensor read failed — don't halt on uncertainty

        # ── End-effector state (optional — read via deployment-defined sensor) ──
        ee_state_val: float | None = None
        try:
            raw = self.env.read_sensor("ee_state")
            ee_state_val = float(raw) if np.isscalar(raw) else float(raw.flat[0])
        except Exception:
            pass  # "ee_state" sensor not registered for this deployment

        # ── Contact force ──────────────────────────────────────────────────────
        limit, near = self.max_contact_force_n, self.near_miss_force_n
        if _approach_limit_n is not None and _approach_limit_n < limit:
            limit, near = _approach_limit_n, min(near, 0.8 * _approach_limit_n)    # the final approach
        if force >= limit:
            self.halt_reason = (f"contact force {force:.1f} N (limit {limit:.1f} N"
                                + (", on the final approach)" if limit < self.max_contact_force_n else ")"))
            print(f"  [Safety:Exec] HALT — {self.skill_name}: {self.halt_reason}")
            self._halted = True
            return True
        elif force >= near:
            # limit_value is the HARD stop, not the near-miss trigger — margin_pct()
            # divides by it. The trigger would make every near-miss report 0%
            # margin (observed >= trigger by construction), and the "margin < 20"
            # print gate below always fire.
            self._record_near_miss("contact_force", force,
                                   limit, ee, ee_state_val)

        # ── EE workspace floor ─────────────────────────────────────────────────
        ee_z = float(ee[2])
        floor = self.bounds.z_min - self.thresholds.max_ee_z_descent_m
        if ee_z < floor:
            self.halt_reason = f"end-point z={ee_z:.3f} m below the floor {floor:.3f} m"
            print(f"  [Safety:Exec] HALT — {self.skill_name}: {self.halt_reason}")
            self._halted = True
            return True

        if not np.isfinite(ee).all() or ee_z > self.bounds.z_max + 1e-9:
            self.halt_reason = ("end-point position is not finite" if not np.isfinite(ee).all()
                                else f"end-point z={ee_z:.3f} m above the ceiling {self.bounds.z_max:.3f} m")
            print(f"  [Safety:Exec] HALT — {self.skill_name}: {self.halt_reason}")
            self._halted = True
            return True

        return False

    def _record_near_miss(
        self,
        constraint:   str,
        value:        float,
        limit:        float,
        ee:           np.ndarray,
        ee_state_val: float | None,
    ) -> None:
        ev = NearMissEvent(
            timestamp=datetime.now(),
            skill_name=self.skill_name,
            constraint=constraint,
            observed_value=value,
            limit_value=limit,
            ee_pos=ee.tolist(),
            ee_state_value=ee_state_val,
        )
        self.near_misses.append(ev)
        margin = ev.margin_pct()
        if margin < 20:
            print(f"  [Safety:Exec] NEAR-MISS — {ev.summary_line().strip()}")

    def summary(self) -> str:
        lines = [f"ExecutionMonitor({self.skill_name}): {self._checks} checks"]
        if self._halted:
            lines.append("  HALTED due to hard constraint violation")
        if self.near_misses:
            lines.append(f"  Near-misses ({len(self.near_misses)}):")
            for ev in self.near_misses:
                lines.append(ev.summary_line())
        else:
            lines.append("  No near-misses.")
        return "\n".join(lines)


# ── Module-level monitor singleton ────────────────────────────────────────────
# skill_executor calls set_monitor/clear_monitor; safety.check() delegates to it.

_monitor: ExecutionMonitor | None = None


def set_monitor(monitor: ExecutionMonitor) -> None:
    """Set the active execution monitor (called by skill_executor.run_skill)."""
    global _monitor
    _monitor = monitor


def clear_monitor() -> ExecutionMonitor | None:
    """Clear the active monitor and return it (for near-miss summary logging)."""
    global _monitor
    m = _monitor
    _monitor = None
    return m


def check() -> bool:
    """
    Poll the active monitor.  Returns True if execution should halt.
    No-op (returns False) if no monitor is set — safe to call from tests.
    """
    if _monitor is None:
        return False
    return _monitor.check()
