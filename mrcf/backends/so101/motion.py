"""
mrcf/backends/so101/motion.py — moving the SO-101: move_ee_to (to a point,
through the IK solver, streamed to the servos), long sideways moves in joint
space, settling onto a goal the servos stop short of, and the named poses
(home, park, observe).

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import time

import numpy as np

from mrcf import flight_recorder
from mrcf.backends.so101.follower import FloorRefused
from mrcf.backends.so101.kinematics import JOINT_NAMES
from mrcf.backends.so101.orientation import OrientationMixin

# move_ee_to() gives up when end-effector error has not improved on its best
# by _IK_MIN_PROGRESS_M within this many iterations. Measured against the best
# error so far rather than the previous iteration, so it is a window, not a
# consecutive-failure count — which is what lets the loop stream goals without
# waiting for the servos to settle (see move_ee_to) and still tell a real stall
# from the arm simply being behind its commands. 16 ticks is 0.8 s at 20 Hz,
# comfortably longer than the servos take to cover one step_deg.
_IK_STALL_ITERS = 16
_IK_MIN_PROGRESS_M = 0.002
# Close to the goal (within 2 x tol, where the stall check above stands aside)
# but no closer by _IK_NEAR_PROGRESS_M within _IK_NEAR_STALL_ITERS ticks: the
# streamed commands are hunting round it, not closing in. Settled onto the
# solver's solution from there instead (_settle_to), once, and done.
# 2026-11-18 16:15, reaching 36 cm out: a descent hung 4-8 mm off its goal,
# the loaded shoulder ~1 deg short, and streamed for the whole 20 s backstop —
# the arm visibly shaking — before giving up 8 mm short.
_IK_NEAR_STALL_ITERS = 20
_IK_NEAR_PROGRESS_M = 0.0005
# How far move_ee_to's command may run ahead of where a joint measures. The
# servos sit 1-3 deg off their command under load and do not move for a
# command inside that (their deadband): with the command re-based on the
# measured angle every tick, a joint the solver moves by a fraction of a degree
# per tick is never asked to move further than that, and does not move at all —
# the other joints do the moving, and the jaws go outward instead of down.
_MIN_LEAD_DEG = 4.0
# A joint that moved less than this since the last tick is held by its deadband:
# only then does move_ee_to run its command on ahead of it.
_STUCK_DEG = 0.2
# How far past the solver's own goal a joint's command may be pushed, and only
# while the joint is stuck: the servos stop 1-3 deg short under load. A joint
# that is moving is sent no further than the goal itself — pushed on past it,
# it overshot and was pushed back the other way, and the jaws swung ~2 cm side
# to side over a piece every 0.6 s (2026-11-04, shoulder_pan at a grasp).
_PUSH_PAST_GOAL_DEG = 3.0
# _settle_to: a joint stopped short of its goal is asked this much of what it
# is short further on, up to the cap — its servo's own load offset, closed.
_SETTLE_PUSH = 1.0
_SETTLE_PUSH_MAX_DEG = 2.0
# move_ee_to goes further sideways than this in joint space first (_joint_space_to):
# every joint along one line to where the IK puts the goal.
_JOINT_SPACE_FROM_M = 0.05
_JOINT_SPACE_ABOVE_M = 0.015   # ... to this far above the goal, then straight down
_JOINT_SPACE_CEILING_MARGIN_M = 0.005   # ... never along a line that rises closer than this to the ceiling
_NUDGE_M = 0.006               # move_ee_to shorter than this: solved and settled onto, not streamed
# _descend_by_joint_line: a descent at least this much longer than the line's
# own stop over the goal, and no further than this sideways of it.
_DESCENT_LINE_MIN_M = 0.010
_DESCENT_LINE_SIDEWAYS_M = 0.010
# A goal held to an orientation is refused, before anything moves, when the
# arm's IK cannot bring the jaws within this of it (or the move's own tol).
_REACH_SLACK_M = 0.02
_REACH_ITERS = 60
_JOINT_SPACE_LEAD_DEG = 6.0   # the line moves on only while every joint is this close to it
_JOINT_SPACE_RAMP_TICKS = 6    # up to speed over this many ticks, and down over the last few degrees
_JOINT_SPACE_TOGETHER_DEG = 1.0   # how far past the slowest joint the others are sent


class MotionMixin(OrientationMixin):
    """SO101Env's moves (see env.py)."""

    def move_ee_to(self, target_pos: np.ndarray, tol: float = 0.01, max_iters: int = 400, step_size: float = 0.0,
                   max_step_deg: "float | None" = None, orientation: "np.ndarray | None" = None,
                   settle: bool = True) -> bool:
        """_move_ee_to (see there), on the flight recorder's timeline: where to, and where it got."""
        t0 = time.monotonic()
        arrived = False
        try:
            arrived = self._move_ee_to(target_pos, tol=tol, max_iters=max_iters, step_size=step_size,
                                       max_step_deg=max_step_deg, orientation=orientation, settle=settle)
            return arrived
        except FloorRefused as exc:
            self._refused_by_floor(exc)
            return False
        finally:
            try:
                off = float(np.linalg.norm(self.get_ee_pos() - np.asarray(target_pos, float)))
            except Exception:
                off = None
            flight_recorder.event("arm", "move", goal=target_pos, tol=tol, step_deg=max_step_deg, settle=settle,
                                  arrived=arrived, off_mm=None if off is None else off * 1000,
                                  s=time.monotonic() - t0)

    def _move_ee_to(
        self,
        target_pos: np.ndarray,
        tol: float = 0.01,
        # A backstop, 20 s at 20 Hz: a move that stops getting closer is ended
        # by the stall check below within 0.8 s, so this only ever cuts off one
        # still on its way. Long moves from home on these gently-accelerating
        # servos take more than 3 s.
        max_iters: int = 400,
        step_size: float = 0.0,  # unused — step size is governed by max_step_deg
        max_step_deg: "float | None" = None,
        orientation: "np.ndarray | None" = None,
        settle: bool = True,
    ) -> bool:
        """
        Iterative Cartesian IK: repeatedly solve for joint angles from the robot's
        *measured* position (not the unclamped IK output), so the per-call
        max_relative_target clamp in lerobot's send_action produces a smooth,
        converging trajectory rather than one large commanded jump.

        The approach orientation is chosen per target (see approach_orientation):
        straight down where possible, tilted where that is the only way to reach.
        Pass `orientation` (from approach_orientation_for) to hold one instead,
        so a sequence of moves over one spot does not rotate the wrist between
        them.

        settle False: once within tol, go on without waiting for the servos to
        finish onto the goal (_settle_to) — for a move nothing is done at the
        end of, a lift or a retreat, which the next move carries on from.
        """
        self._require_kinematics()
        # Sample the opening ONCE and use it for both the probe and the loop, so
        # the two aim at the same point even if the jaws drift mid-move.
        gap_m = self._grasp_gap()
        if orientation is not None:
            rot = np.asarray(orientation, dtype=float)
            # Held to an orientation, a goal nothing reaches is not gone for: the
            # loop below would drive the joints at it until they stall, and the
            # way there — and on to the next goal — is anybody's. 2026-11-17: two
            # such looks, 10 and 3 cm out of reach, and the swing between them
            # put the gripper into the table.
            if self._reach_error(rot, np.asarray(target_pos, float), gap_m) > max(tol, _REACH_SLACK_M):
                why = (f"{np.round(np.asarray(target_pos, float), 3).tolist()} cannot be reached with the gripper "
                       f"held that way (within {max(tol, _REACH_SLACK_M) * 100:.0f} cm, inside the joints' range)")
                print(f"  [SO101Env] Not moving: {why}.")
                self.move_refusals += 1
                self.last_refusal = why
                return False
        else:
            chosen = self.approach_orientation(target_pos, gap_m=gap_m, tol=tol)
            if chosen is None:
                print(f"  [SO101Env] No reachable gripper orientation for "
                      f"{np.round(np.asarray(target_pos, float), 3).tolist()} within calibrated limits.")
                return False
            rot, tilt = chosen
            if abs(tilt) > 0.5:
                print(f"  [SO101Env] Straight-down approach unreachable here — "
                      f"tilting the gripper {tilt:+.0f} deg.")

        # The other way round from how the wrist is now: turn it first, on its
        # own, where the arm is — the Cartesian loop cannot find a goal half a
        # turn away (see _half_turn_roll). Moves start somewhere safe — home, or
        # over the object they are about to work on — so that is where it turns.
        q_now = np.append(self._q_arm_deg(), 0.0)
        turned = self._half_turn_roll(rot, q_now)
        if turned is not None:
            print("  [SO101Env] Turning the wrist half a turn first.")
            here_deg = {n: float(v) for n, v in zip(JOINT_NAMES, q_now)}
            if not self.goto_joint_degrees({**here_deg, "wrist_roll": turned}, tol_deg=3.0, timeout_s=8.0):
                print("  [SO101Env] The wrist did not finish turning.")
                return False

        goal = np.asarray(target_pos, dtype=float)   # where the JAWS must end up
        t_target = np.eye(4)
        t_target[:3, :3] = rot
        # Offset the commanded frame so the centre of the jaw gap, not the frame
        # origin, arrives at the target (see _TCP_OFFSET) — where the jaws
        # really close, not only where the model has them (_grasp_correction).
        t_target[:3, 3]  = goal - self._grasp_correction(rot, goal) - rot @ self._tcp(gap_m)

        q6 = np.append(self._q_arm_deg(), 0.0)
        # Stall detection: an unreachable target (outside the arm's reach, or
        # one whose IK solution needs a joint angle outside its calibrated
        # range) never converges — without this the loop would keep re-solving
        # toward the impossible point for every one of max_iters, driving
        # joints hard against their limits and visibly thrashing the arm.
        step_deg = float(max_step_deg if max_step_deg is not None
                         else self._max_relative_target_deg)

        # A long way sideways (a hover from home or the park pose, a carry to
        # the place): there in joint space first, every joint along one line
        # to where the IK puts the goal, then this loop for the last of it.
        # Streamed in Cartesian space all the way, the joints drift apart on
        # the real servos — one runs past its command, another trails it —
        # and the jaws come DOWN before they come out, short of the hover,
        # where the neighbour toward the arm stands. The joint line from
        # above comes down onto the hover from above; it ends
        # _JOINT_SPACE_ABOVE_M over the goal, where it can, so lagging servos
        # overshooting its end (12 mm in simulation) stay above the goal, and
        # this loop finishes straight down. Out of the park pose, folded back
        # 10 mm under the workspace's ceiling, every such line first rises
        # past it: this loop starts those, and hands over within
        # 2 x _JOINT_SPACE_FROM_M (_joint_line). A descent's last part stays in this loop
        # (its upper part may be one line: _descend_by_joint_line).
        #
        # A nudge of a few millimetres — the correction at a grasp — is under a
        # degree a joint, inside the servos' deadband, where this loop runs its
        # command on ahead until something gives, and the jaws swing round the
        # object. Solved instead, and settled onto: a joint short of it is
        # pushed by what it is short, and no further (_settle_to).
        if float(np.linalg.norm(self._grasp_point(q6, gap_m) - goal)) < _NUDGE_M:
            solved = self._probe(rot, goal, self._tcp(gap_m), gap_m, q6, min(tol, 0.001))
            if solved is not None:
                self._settle_to(self._clamp_joint_degrees({n: float(solved[i]) for i, n in enumerate(JOINT_NAMES)}))
                return float(np.linalg.norm(self._grasp_point(np.append(self._q_arm_deg(), 0.0), gap_m) - goal)) < tol
        line = self._joint_line(q6, rot, goal, gap_m, tol, step_deg)
        if line == "ran":
            q6 = np.append(self._q_arm_deg(), 0.0)
        elif orientation is not None and max_step_deg is not None \
                and self._descend_by_joint_line(q6, rot, goal, gap_m, step_deg):
            q6 = np.append(self._q_arm_deg(), 0.0)
        best_err = float("inf")
        err = float("inf")
        stalled_iters = 0
        near_best, near_iters = float("inf"), 0
        sent: "dict | None" = None         # the command as it has run on, per joint
        lead = max(step_deg, _MIN_LEAD_DEG)
        assert self._kinematics is not None     # _require_kinematics() above
        for _ in range(max_iters):
            q6 = self._kinematics.inverse_kinematics(q6, t_target)
            raw_action = {name: float(q6[i]) for i, name in enumerate(JOINT_NAMES)}
            clamped = self._clamp_joint_degrees(raw_action)
            # Per-call ceiling on how far any joint moves in one iteration.
            # lerobot's max_relative_target is a fixed limit for the whole
            # robot; this lets one motion be gentler than another — a descent
            # onto a 2 cm chess piece wants smaller steps than a traverse
            # across the board, because arriving fast enough to knock the
            # piece over loses it regardless of how accurate the target was.
            pos_now = self._motor_pos()
            # Scaled as a whole, not joint by joint: clipping each joint to the
            # step on its own lets the ones with little to do finish first, and
            # the jaws leave the line. From a 70 mm hover (wrist at its limit) to
            # a 27 mm grasp the elbow opens ~40 deg while the shoulder needs ~12:
            # clipped apart, the shoulder runs ahead and lifts the jaws before
            # the elbow brings them down. Scaled, every joint arrives together.
            delta = {name: v - pos_now[name] for name, v in clamped.items()}
            biggest = max((abs(d) for d in delta.values()), default=0.0)
            scale = min(1.0, step_deg / biggest) if biggest > 0 else 1.0
            plain = {name: float(pos_now[name] + d * scale) for name, d in delta.items()}
            # A joint the deadband holds still (see _MIN_LEAD_DEG) is asked a
            # little further each tick — what the solver asks of it added to what
            # was last SENT, not to where it measures — until it moves, instead
            # of the same fraction of a degree past where it already is, forever.
            # On a gentle move (a descent: steps smaller than the arm's usual),
            # every tick while it is still asked the same way: held back to
            # "only when stuck", the loaded joints of an arm reaching out stall
            # a descent short. On other moves only while it is stuck: running
            # the command on past a joint that is moving winds up against the
            # servos' lag, and the jaws circle the target until the stall check
            # gives up. A joint asked back follows the solver. At most `lead`
            # ahead either way.
            gentle = step_deg < self._max_relative_target_deg
            if sent is None:
                sent, last_pos = dict(pos_now), dict(pos_now)
            commanded = {}
            for name in clamped:
                want = plain[name] - pos_now[name]
                ahead = sent[name] - pos_now[name]
                stuck = abs(pos_now[name] - last_pos[name]) < _STUCK_DEG
                push = want * ahead > 0 and (gentle or stuck)
                cmd = sent[name] + want if push else plain[name]
                if push:
                    goal_deg = clamped[name]
                    over = _PUSH_PAST_GOAL_DEG if stuck else 0.0
                    cmd = min(cmd, goal_deg + over) if want > 0 else max(cmd, goal_deg - over)
                commanded[name] = float(np.clip(cmd, pos_now[name] - lead, pos_now[name] + lead))
            commanded = self._clamp_joint_degrees(commanded)
            commanded = self._floor_kept(commanded, plain)
            sent, last_pos = commanded, dict(pos_now)
            self._send({f"{name}.pos": v for name, v in commanded.items()})
            # One control tick, NOT _settle_to. Waiting for each intermediate
            # goal to be reached makes the motion visibly jerky: the
            # servo decelerates toward every 8 deg waypoint, then re-accelerates
            # when the next one arrives ~10 times a second. Streaming goals at a
            # fixed cadence keeps the arm in its cruise phase and the motion
            # continuous.
            #
            # Safe because `commanded` is clipped to +/-lead around the
            # MEASURED position every cycle, so the command can never lead the
            # arm by more than that however far behind it falls — and it
            # re-solves to a FIXED t_target, so it converges rather than
            # running away. Servo lag, which per-iteration progress would read
            # as a stall, is handled by the windowed stall check below, not by
            # stopping the arm.
            self.step(1)
            # Re-sync to the actually-measured position (send_action may have
            # clamped the commanded delta, and we may have clamped it further
            # above) before the next IK iteration.
            q6[: len(JOINT_NAMES)] = self._q_arm_deg()
            # gap_m, not a fresh read: the error has to be measured against the
            # same grasp point t_target was built from, or the loop converges on
            # a target that moves under it.
            here = self._grasp_point(q6, gap_m)
            err = float(np.linalg.norm(here - goal))
            if line == "later" and float(np.linalg.norm((here - goal)[:2])) < 2 * _JOINT_SPACE_FROM_M:
                line = self._joint_line(q6, rot, goal, gap_m, tol, step_deg, near_enough=True)
                if line == "ran":
                    # From where the line left the arm, as a fresh move.
                    q6 = np.append(self._q_arm_deg(), 0.0)
                    sent, best_err, stalled_iters = None, float("inf"), 0
                    near_best, near_iters = float("inf"), 0
                    continue
            if err < tol:
                # Arrived: settle where the solver says from here, not on a
                # command that has run ahead, which would carry the arm on past.
                if settle:
                    self._settle_to(plain)
                else:
                    self._send({f"{name}.pos": v for name, v in plain.items()})
                return True
            if err <= 2.0 * tol:
                if err < near_best - _IK_NEAR_PROGRESS_M:
                    near_best, near_iters = err, 0
                else:
                    near_iters += 1
                if near_iters >= _IK_NEAR_STALL_ITERS:
                    # Hunting round it (_IK_NEAR_STALL_ITERS): settled onto the solver's own
                    # solution from here — a joint short of it pushed by what it is short.
                    self._settle_to(clamped)
                    here = self._grasp_point(np.append(self._q_arm_deg(), 0.0), gap_m)
                    err = float(np.linalg.norm(here - goal))
                    if err < tol:
                        return True
                    print(f"  [SO101Env] Stopped {err * 1000:.0f} mm from target {np.round(goal, 3).tolist()} — "
                          "it was no longer getting closer; holding there.")
                    flight_recorder.event("arm", "near_stall", goal=goal, off_mm=err * 1000)
                    self.hold()
                    return False
            if err < best_err - _IK_MIN_PROGRESS_M:
                best_err = err
                stalled_iters = 0
            else:
                stalled_iters += 1
                # Only call it a stall when we are still FAR from the target.
                # This detector exists to stop the arm thrashing against a joint
                # limit for an unreachable point, which always leaves a large
                # residual. Near convergence it misfires: with a 4 mm tolerance,
                # once inside ~6 mm no iteration can improve on the best by
                # _IK_MIN_PROGRESS_M (2 mm), so a loop that is converging
                # normally trips the counter and reports a stall, even at a
                # pose whose IK solution is exact.
                # max_iters still bounds the loop, so nothing can run forever.
                if stalled_iters >= _IK_STALL_ITERS and err > 2.0 * tol:
                    # Which joint is furthest behind what it was sent: a joint
                    # at its limit, or one too loaded to follow (2026-10-30:
                    # reaching out to e6, stalls 5-13 mm short with every
                    # joint 28 deg or more inside its limits in the model).
                    behind = {n: sent[n] - float(q6[i]) for i, n in enumerate(JOINT_NAMES) if n in sent}
                    worst = max(behind, key=lambda n: abs(behind[n]), default=None)
                    lag = "" if worst is None else f"; {worst} {behind[worst]:+.1f} deg short of what it was sent"
                    print(f"  [SO101Env] IK stalled {err * 100:.1f} cm from target "
                          f"{np.round(goal, 3).tolist()} — the arm stopped getting closer{lag}; stopping.")
                    flight_recorder.event("arm", "stall", goal=goal, off_mm=err * 1000,
                                          behind_deg={n: round(v, 1) for n, v in behind.items()})
                    # Stopped where it is: the last command can be `lead` ahead
                    # of the joints, and left standing it carries the arm on
                    # after the move has given up.
                    self.hold()
                    return False
        # Ran out of iterations while still making progress. Say so: a silent
        # False here reads downstream as "did not arrive" with no distance,
        # which is indistinguishable from a stall and hides a tolerance that
        # is simply too tight for max_iters.
        print(f"  [SO101Env] IK did not converge within {max_iters} iterations — "
              f"{err * 100:.1f} cm from target {np.round(goal, 3).tolist()} "
              f"(tolerance {tol * 100:.1f} cm).")
        self.hold()
        return False

    def go_home(self) -> bool:
        """Move to the fixed joint-space home pose (works even without kinematics)."""
        return self.goto_joint_degrees(self._home_degrees, tol_deg=2.0, timeout_s=6.0)

    def go_park(self) -> bool:
        """
        RobotEnv.go_park: the measured pose out of the overhead camera's view
        (park_degrees, robot_config CALIBRATION_PARK_DEGREES), else home.
        """
        return self.goto_joint_degrees(self.__dict__.get("_park_degrees") or self._home_degrees)

    def go_observe(self) -> bool:
        """
        Move to the fixed joint-space pose used for scene scans (see robot_config.py's
        OBSERVE_DEGREES, measured with find_poses.py). Deliberately joint-space,
        not Cartesian IK — the unconstrained IK solver can ask for angles outside
        this arm's calibrated range (see _clamp_joint_degrees), which leaves the
        arm stalled against a servo limit instead of actually reaching a pose
        that can see the table.
        """
        if self._observe_degrees is None:
            raise RuntimeError(
                "OBSERVE_DEGREES not set — run find_poses.py to measure a pose where the "
                "camera can see the whole table, then pass observe_degrees to SO101Env."
            )
        # Restore the primary pixel_to_world calibration in case a prior
        # go_to_look_around_pose() call swapped in a different one.
        self._observe_center_xy = self._primary_observe_center_xy
        self._observe_height_m  = self._primary_observe_height_m
        return self.goto_joint_degrees(self._observe_degrees, tol_deg=2.0, timeout_s=6.0)

    def go_to_look_around_pose(self, pose: dict) -> bool:
        """
        Move to an alternate calibrated vantage point (robot_config.py's
        LOOK_AROUND_POSES), for when the primary observe pose's view doesn't
        contain an object the user referenced. Same joint-space approach as
        go_observe() — each entry is a directly-measured pose, not a Cartesian
        IK target — plus its own pixel_to_world calibration, since a different
        camera pose needs a different one.

        pose: {"degrees": {...}, "center_xy": (x, y), "height_m": h} — see
        find_poses.py, which can capture any number of arbitrarily-labelled
        poses in this same format.

        Caller is responsible for calling go_observe() afterwards to restore
        the primary pose/calibration — see skills.py's look_around().
        """
        self._observe_center_xy = pose["center_xy"]
        self._observe_height_m  = pose["height_m"]
        return self.goto_joint_degrees(pose["degrees"], tol_deg=2.0, timeout_s=6.0)

    # ── Extra helper: blocking joint-space move to an arbitrary pose ───────────

    def _floor_kept(self, pushed: dict, plain: dict) -> dict:
        """
        A command pushed on past the solver's step (a joint stuck in its
        deadband, a settle closing a load offset) that the floor guard would
        refuse — its path into the table — sent as the step itself instead,
        when that one keeps out of the table (the robot's floor_caps_push; off:
        pushed, and refused). 2026-11-19, run 40: at a plug gripped 18 mm up
        the wrist camera's reseats came down 15 mm, a stuck joint's command
        ran 3 deg past its goal, the guard refused it, and the reseat stayed up;
        the jaws closed 6 mm off the plug, missed, and pushed it.
        """
        if not self.__dict__.get("_floor_caps_push") or pushed == plain:
            return pushed
        if self._below_floor({f"{n}.pos": v for n, v in pushed.items()}) is None:
            return pushed
        step = self._clamp_joint_degrees(dict(plain))
        if self._below_floor({f"{n}.pos": v for n, v in step.items()}) is not None:
            return pushed                                 # the step itself goes in: the guard says so
        flight_recorder.event("arm", "push_kept_from_floor")
        return step

    def _descend_by_joint_line(self, q6, rot, goal, gap_m: float, step_deg: float) -> bool:
        """
        For move_ee_to, a held-orientation descent's upper part (the robot's
        descent_line_above_m; None: not done): one line in joint space from
        where the arm is to the solver's own answer that far over the goal,
        every joint along it together, step_deg a tick for the one with most
        to do, and settled onto; the last of it the Cartesian loop's. True
        when it ran.

        2026-11-19 (run 39, and run 37's "bobbing"): leaning 20 deg 32 cm out,
        the loop re-steering every tick from joints that lag 2-3 deg under
        load drove the shoulder past its own goal while the elbow, with 36 deg
        to open, trailed — the jaws went up and out, or its command dipped into
        the table and was refused. On the same arm in simulation (deadbanded
        servos) the line arrives where the loop does not, and strays no more
        near the goal; it ends over the goal, the loop takes the jaws round the
        object.
        """
        above_m = self.__dict__.get("_descent_line_above_m")
        if above_m is None:
            return False
        here = self._grasp_point(q6, gap_m)
        if here[2] - goal[2] < above_m + _DESCENT_LINE_MIN_M \
                or float(np.linalg.norm((here - goal)[:2])) > _DESCENT_LINE_SIDEWAYS_M:
            return False
        above = np.asarray(goal, float) + np.array([0.0, 0.0, above_m])
        solved = self._probe(rot, above, self._tcp(gap_m), gap_m, q6, 0.002)
        if solved is None:
            return False
        q0 = np.asarray(self._q_arm_deg(), float)
        q1 = np.array([float(solved[i]) for i in range(len(JOINT_NAMES))])
        n = max(1, int(np.ceil(float(np.max(np.abs(q1 - q0))) / max(step_deg, 0.5))))
        for k in range(1, n + 1):
            q = q0 + (q1 - q0) * k / n
            self._send({f"{name}.pos": v for name, v in
                        self._clamp_joint_degrees({j: float(v) for j, v in zip(JOINT_NAMES, q)}).items()})
            self.step(1)
        self._settle_to(self._clamp_joint_degrees({j: float(v) for j, v in zip(JOINT_NAMES, q1)}))
        flight_recorder.event("arm", "descent_line", goal=above, ticks=n,
                              off_mm=float(np.linalg.norm(self._grasp_point(np.append(self._q_arm_deg(), 0.0), gap_m)
                                                          - above)) * 1000)
        return True

    def _joint_line(self, q6, rot, goal, gap_m: float, tol: float, step_deg: float,
                    near_enough: bool = False) -> str:
        """
        For move_ee_to, the way to the goal in joint space (see there): "ran"
        when it went, "later" when the way would rise past the workspace's
        ceiling (stream_bounds) and the Cartesian loop should start the move
        and ask again nearer, "no" when not needed or not solvable here.

        Only a long way sideways (further than _JOINT_SPACE_FROM_M, or
        near_enough): one line to the goal's IK solution, _JOINT_SPACE_ABOVE_M
        over it where it can. Not the last of a descent: in joint space down to
        the grasp, the jaws go 16-17 mm sideways round the object (the real arm
        in 1 cm pieces; the replay, tools/so101/replay_descents.py, 18 mm) and
        the close after it misses. A descent's upper part, over the object, can
        be one line (_descend_by_joint_line, descent_line_above_m); the last of
        it is this loop's.
        """
        here = self._grasp_point(q6, gap_m)
        tcp = self._tcp(gap_m)
        if not near_enough and float(np.linalg.norm((here - goal)[:2])) <= _JOINT_SPACE_FROM_M:
            return "no"
        solved = (self._probe(rot, goal + [0.0, 0.0, _JOINT_SPACE_ABOVE_M], tcp, gap_m, q6, tol)
                  if here[2] > goal[2] else None)
        if solved is None:
            solved = self._probe(rot, goal, tcp, gap_m, q6, tol)
        if solved is None:
            return "no"
        path = [solved]
        bounds = self.__dict__.get("_stream_bounds")
        if bounds is not None:
            ceiling = bounds.z_max - _JOINT_SPACE_CEILING_MARGIN_M
            if any(self._grasp_point(a + f * (b - a), gap_m)[2] > ceiling
                   for a, b in zip([q6] + path, path) for f in np.linspace(0, 1, 11)):
                return "no" if near_enough else "later"
        self._joint_space_to(path, step_deg)
        return "ran"

    def _joint_space_to(self, path, step_deg: float) -> None:
        """
        Every arm joint along `path` — one configuration (degrees, JOINT_NAMES
        order) or a list of them, each a straight line in joint space from the
        one before. Each tick every joint is sent the same point along the
        path: one step (step_deg for the joint moving furthest there) past
        where the slowest joint has actually got — so the fast ones wait for
        it, they stay together, and the jaws keep to the path. That step grows
        over the first ticks and shrinks over the last few degrees. Returns
        when the command reaches the end (the joints may still be catching
        up: move_ee_to finishes), or when the joints stop following it.
        """
        here = {n: float(v) for n, v in self._motor_pos().items() if n in JOINT_NAMES}
        points = [here] + [self._clamp_joint_degrees({n: float(q[i]) for i, n in enumerate(JOINT_NAMES)})
                           for q in (path if isinstance(path, list) else [path])]
        legs = []                                  # (start, span, length, judged), length in degrees
        for a, b in zip(points, points[1:]):
            span = {n: b[n] - a[n] for n in JOINT_NAMES}
            length = max(abs(v) for v in span.values())
            if length > 1e-6:
                # Joints with only a little to do sit in their deadband for most
                # of the leg: not the ones to judge progress by.
                legs.append((a, span, length, [n for n in JOINT_NAMES if abs(span[n]) >= 0.3 * length]))
        total = sum(leg[2] for leg in legs)
        if not legs:
            return

        def at(dist):
            """The command `dist` degrees along the path, and its leg's index and start."""
            for i, (a, span, length, _) in enumerate(legs):
                if dist <= length or i == len(legs) - 1:
                    f = min(1.0, max(0.0, dist / length))
                    return {n: a[n] + f * span[n] for n in JOINT_NAMES}, i
                dist -= length
            raise AssertionError

        done_before = [sum(leg[2] for leg in legs[:i]) for i in range(len(legs))]
        s, tick, still, best = 0.0, 0, 0, -1.0
        while True:
            pos = self._motor_pos()
            _, i = at(s)
            a, span, length, judged = legs[i]
            got = {n: done_before[i] + length * float(np.clip((pos[n] - a[n]) / span[n], 0.0, 1.0))
                   for n in judged}
            made = min(got.values())
            tick += 1
            speed = min(1.0, tick / _JOINT_SPACE_RAMP_TICKS,
                        max(0.25, (total - made) / (_JOINT_SPACE_RAMP_TICKS * step_deg / 2)))
            s = min(total, max(s, made + speed * step_deg))
            # One that has stopped following (against something, or its limit)
            # ends this; move_ee_to then judges from where the arm is.
            still = 0 if made > best + 0.1 else still + 1
            best = max(best, made)
            if still >= _IK_STALL_ITERS:
                return
            # Every joint a step on, but one already more than
            # _JOINT_SPACE_TOGETHER_DEG ahead of the slowest is held where it
            # has got to: running on, it takes the jaws off the path.
            command, _ = at(s)
            for n, there in got.items():
                if there > made + _JOINT_SPACE_TOGETHER_DEG and s < total:
                    command[n] = at(there)[0][n]
            self._send({f"{n}.pos": v for n, v in command.items()})
            self.step(1)
            if s >= total:
                return

    def _settle_to(self, commanded: dict[str, float], tol_deg: float = 0.5,
                   timeout_s: float = 1.5) -> None:
        """
        Block until the servos actually reach `commanded` (or time out).

        Called once per move_ee_to, on arrival — not per iteration (see
        move_ee_to for why waiting between iterations made the motion jerky).
        What it buys is a settled arm before the next phase, which for a pick is
        closing the jaws onto a piece.

        With the servos' integral term set (_tune_servos, robot_config's
        SERVO_I_COEFFICIENT), check_hold.py measures elbow_flex 0.1 deg off at
        every pose and shoulder_lift 0.0-0.2 deg held 1 s. So 0.5 deg is
        comfortably inside what the arm can hold, and 1.5 s gives the
        integrator room to get there — the raw reading on arrival is still
        0.4-1.7 deg out because the integrator has not wound up yet, which is
        exactly what this method should be waiting through rather than
        passing. At ~3.2 mm of grasp-point travel per degree, 2.0 deg would let
        the arm start a grasp up to 6 mm off.

        Waiting on the integrator alone takes 0.6-1.5 s per move — over half of
        a pick's moving time: ~0.3 s for the servos to catch up, then the
        shoulder or elbow parks ~1 deg short under its load while the integral
        creeps it in, often to the timeout.
        So a joint that has stopped short is asked past its goal by what it
        is short (_SETTLE_PUSH, at most _SETTLE_PUSH_MAX_DEG): the servo
        sees twice the error and closes it. Only a stopped joint: one still
        on its way is asked for the goal itself, so it does not overshoot.
        The goal itself is sent last, to hold.
        """
        action = {f"{name}.pos": float(v) for name, v in commanded.items()}
        t0 = self.motion_clock()
        last = None
        try:
            while self.motion_clock() - t0 < timeout_s:
                self.step(1)
                pos = self._motor_pos()
                if max(abs(pos[name] - v) for name, v in commanded.items()) < tol_deg:
                    return
                # Re-send: the servo needs the goal held, and send_action re-clamps
                # relative to the position it has actually reached by now.
                pushed = {}
                for name, v in commanded.items():
                    short = float(v) - pos[name]
                    stopped = last is not None and abs(pos[name] - last[name]) < _STUCK_DEG
                    push = float(np.clip(_SETTLE_PUSH * short, -_SETTLE_PUSH_MAX_DEG, _SETTLE_PUSH_MAX_DEG))
                    pushed[f"{name}.pos"] = float(v) + (push if stopped else 0.0)
                last = dict(pos)
                kept = self._floor_kept({k.removesuffix(".pos"): v for k, v in pushed.items()}, commanded)
                self._send({f"{name}.pos": v for name, v in kept.items()})
        finally:
            self._send(action)

    def _reach_error(self, rot: np.ndarray, goal: np.ndarray, gap_m: float) -> float:
        """
        How near (metres) the jaws can come to goal with the gripper held at
        rot: the IK solved offline from where the arm is, each joint kept in its
        range as a move keeps it (_clamp_joint_degrees) — not the margins inside
        it that choosing an orientation leaves, which a move does not hold to.
        """
        assert self._kinematics is not None
        t = np.eye(4)
        t[:3, :3] = rot
        t[:3, 3] = goal - self._grasp_correction(rot, goal) - rot @ self._tcp(gap_m)
        q = self._roll_seed_toward(rot, np.append(self._q_arm_deg(), 0.0))
        best = float("inf")
        for _ in range(4):
            for _ in range(_REACH_ITERS):
                q = self._kinematics.inverse_kinematics(q, t)
            kept = self._clamp_joint_degrees({n: float(q[i]) for i, n in enumerate(JOINT_NAMES)})
            q[: len(JOINT_NAMES)] = [kept[n] for n in JOINT_NAMES]
            best = min(best, float(np.linalg.norm(self._grasp_point(q, gap_m) - goal)))
            if best <= _REACH_SLACK_M / 4:
                break
        return best

    def _refused_by_floor(self, exc: Exception) -> None:
        """A command that would have taken the gripper into the table was not sent: the arm held where it is, and why said."""
        print(f"  [Floor] Not moving: {exc}.")
        flight_recorder.event("arm", "floor_refused", why=str(exc))
        self.move_refusals += 1
        self.last_refusal = str(exc)
        self.hold()

    def goto_joint_degrees(self, target_deg: dict[str, float], tol_deg: float = 2.0, timeout_s: float = 6.0) -> bool:
        """goto_joint_degrees, refused (False, the arm held) where its path would take the gripper into the table."""
        try:
            return self._goto_joint_degrees(target_deg, tol_deg, timeout_s)
        except FloorRefused as exc:
            self._refused_by_floor(exc)
            return False

    def _goto_joint_degrees(self, target_deg: dict[str, float], tol_deg: float = 2.0, timeout_s: float = 6.0) -> bool:
        """
        Move to an arbitrary directly-measured joint pose (same safety-clamped,
        blocking pattern as go_home()/go_observe()/go_to_look_around_pose(), which
        all delegate to this). Use for one-off poses that don't need their own
        named method — e.g. a parking pose a deployment script wants the arm at
        before doing something that needs it out of a camera's view.
        """
        target_deg = self._clamp_joint_degrees(target_deg)
        action = {f"{name}.pos": float(v) for name, v in target_deg.items()}
        t0 = self.motion_clock()               # leaves out time the sentry paused it
        # A joint that is blocked (a cable pulled tight, the wrist camera against
        # something) does not move however long it is pushed, and pushing it for
        # the whole timeout trips the servo's overload protection, after which
        # it stops answering on the bus. So a joint still far off that has not
        # moved for STALL_S is let go of — commanded where it is — and the move fails.
        history: list = []
        while self.motion_clock() - t0 < timeout_s:
            self._send(action)
            self.step(1)
            pos = self._motor_pos()
            err = max(abs(pos[name] - v) for name, v in target_deg.items())
            if err < tol_deg:
                return True
            now = self.motion_clock()          # a pause is not a stall
            history.append((now, pos))
            while len(history) > 1 and now - history[1][0] >= self.STALL_S:
                history.pop(0)
            then, before = history[0]
            if now - then >= self.STALL_S:
                # Not the gripper: squeezing an object is a stall it is meant to hold.
                stuck = [name for name, v in target_deg.items() if name != "gripper"
                         and abs(pos[name] - v) > max(tol_deg, self.STALL_FAR_DEG)
                         and abs(pos[name] - before[name]) < self.STALL_MOVE_DEG]
                if stuck:
                    self._send({f"{name}.pos": float(pos[name]) for name in target_deg})
                    print(f"  [SO101Env] {', '.join(stuck)} not moving "
                          f"({', '.join(f'{pos[n]:.0f} deg, wanted {target_deg[n]:.0f}' for n in stuck)}) "
                          "— stopped pushing so the servo does not overload. Is something blocking it?")
                    return False
        return False
