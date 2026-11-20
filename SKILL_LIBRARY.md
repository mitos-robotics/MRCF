# The skill library, and how to debug it one stage at a time

"The VLM decides what to do" means that a language model is shown a JSON schema
of a **fixed registry of Python functions** and asked to return a list of calls.
Every behaviour the robot has is one of those functions. The language model
never produces motion, and it cannot invent an action. If it names one that
isn't registered, dispatch prints `Unknown action` and stops.

The motion comes from the action's policy. Some are planned from geometry
(pick, place). A learned skill, such as one in chess's
`LEARNED_SKILLS`, runs a trained model for one segment (`mrcf/policies/learned.py`).
It logs `[OK]` or `[FAILED]` with what it did, how many ticks it ran and how
long the model took per tick.

This matters for debugging. **Every failure is in one of seven stages**, and
you can run each one on its own.

---

## 1. The seven stages, and the log line each one prints

A single `You: pick up the pawn` passes through all of these stages in order.
Grep a log for the prefixes in the "Log prefix" column.

| # | Stage | Log prefix | What it does | Lives in |
|---|---|---|---|---|
| 1 | **Board scan** | `[Chess]` `[Memory]` | Finds the board, names the squares, and measures and classifies pieces into `scene_state`. Each piece is matched against the object memory so it keeps its identity. It runs at startup, on `observe` and after a failed skill, but **not** before every command. | `chessboard.py`, `mrcf/object_memory.py` |
| 2 | **Perceive** | `[VLM:perceive]` `[Perceive]` | Describes the current camera view in free text, as context for the planner only. It never produces coordinates. | `vlm.perceive` |
| 3 | **Decompose** | `[VLM:decompose]` `[Decompose]` | Splits one instruction into sub-goals. Usually it changes nothing. | `vlm.decompose_task` |
| 4 | **Plan** | `[VLM:plan]` `[Plan]` | Returns the action list. This is the only stage that chooses *which* function runs, and with what arguments. | `vlm.plan` |
| 5 | **Validate** | `[Safety]` `[PreCondition]` | Checks the plan-level safety rules, then each action's preconditions against the world model. It can drop or block actions. | `mrcf/safety.py`, `vlm.check_preconditions` |
| 6 | **Dispatch** | `[Step n/m]` | Flattens parameters, asks for anything missing, and calls the registered function. | `vlm.dispatch` |
| 7 | **Execute** | `[Pick n/m]` `[SO101Env]` `[OK]`/`[FAILED]`/`[REFUSED]` `[PostCondition]` | The action turns the object's name into geometry and runs the arm's policy: the motion, retries and success checks. Then it writes the outcome to the world model and memory. A failed step stops the rest of the plan (`[Plan] Step n failed`). | `mrcf/manipulation.py`, `mrcf/policies/parallel_jaw/`, `mrcf/backends/so101/env.py` |
| — | **Stop** | `[STOPPED]` `[Stop]` | At any time, a watcher or the safety monitor's limits (checked every control cycle) can trip the interlock. The arm holds where it is, and the whole command is abandoned without recovery or a return home. | `mrcf/interlock.py`, `mrcf/watchers.py` |

Note two things about the order. **Stage 1 does not run before stage 4.**
Positions come from the last scan, so if you move a piece by hand, the planner
works from old positions until you say `observe`. And **stage 2 does not move
the arm.** It describes wherever the arm is pointing, which is often
nowhere useful.

---

## 2. The registry

Built from `vlm.ACTIONS` after importing the chess skills (`from deployments.chess import skills`):

| Action | Parameters | Preconditions | Postconditions | Offered when scene is empty? |
|---|---|---|---|---|
| `pick` | `object_name` | `object_exists`, `object_not_carried`, `nothing_carried` | `something_carried` | no |
| `place` | `target_name` | `something_carried`, `target_exists` | — | no |
| `place_on` | `object_name` | `something_carried`, `object_exists`, `object_not_carried` | — | no |
| `nudge` | `object_name`, `target_name` (optional) | `object_exists`, `object_not_carried`, `nothing_carried` | — | no |
| `identify` | `object_name`, `what` | `object_exists` | — | no |
| `move_above` | `object_name`, `height` | — | — | no |
| `stand_up` | `object_name` | `object_exists`, `object_not_carried` | — | no |
| `home` | — | — | — | yes |
| `open_gripper` | — | — | — | yes |
| `close_gripper` | — | — | — | yes |
| `observe` | — | — | — | yes |
| `forget` | — | — | — | yes |
| `list_objects` | — | — | — | yes |
| `list_actions` | — | — | — | yes |
| `status` | — | — | — | yes |
| `answer` | `question` | — | — | yes |
| `calculate` | `expression` | — | — | yes |

`place_on` is registered only for a task that stacks things (its scenario
says what can carry what); the others never see it. The chess deployment's
controller adds its game verbs too (`new_game`, `my_move`, `game`, `hint`,
`resign`, `end_game`, `resume_game`, in `deployments/chess/game.py`).

The last column is `always=` in `@register_action`. It decides whether the
action appears in the schema the planner sees. `context_action_schema()`
includes an action if `always=True` **or** the scene has objects.

`home`, `open_gripper` and `close_gripper` are offered even when the scan
found nothing, since that is when a reset is most wanted.

### The conditions
The registered predicates are `object_exists`, `object_not_carried`,
`nothing_carried`, `something_carried` and `target_exists`.

Preconditions **block**: if one fails, the action never runs and
`[PreCondition]` is printed. Postconditions are **advisory**. They run after
the skill and print `[PostCondition] … the world model and the robot disagree`,
but leave the reported outcome alone. An unregistered condition name prints a
`[Condition]` warning instead of passing silently, which was once a real bug.

"Carried" is not a sensor reading. `scene_state.held_object()` is the object
whose `position` is `None`. The `pick` action (`mrcf/manipulation.py`) sets
`obj.position = None` after a success, and after a failed pick if the gripper
still senses a hold. A failed lift must not make the next pick drop the object.
A placement checks the release before recording it. `Outcome.released=True`
records that the object was put down even if the retreat failed, but
`ok=False` still stops the plan. Opening the gripper outside a placement
removes the held object from the scene and marks its memory track missing
until a scan finds it again.

---

## 3. Running one stage at a time

Type these at the `You:` prompt. They go straight to `dispatch()`, with the
same preconditions and safety checks, and **call no model at all**.

```
/actions                        every registered action and its parameters
/objects                        the world model, with world coordinates
/pick white_pawn_d4             dispatch that skill on that object
/move_above white_pawn_d4 0.08  positional args fill declared parameters in order
/move_above object_name=white_pawn_d4 height=0.08
/home
/status
```

Numeric arguments are converted to numbers, because skills use them directly
as coordinates.

This is the best way to isolate a stage. If `/pick white_pawn_d4` works and
`pick up the pawn` doesn't, the bug is in stages 2–4. If both fail the same
way, it is in stage 1 or 7.

### Going below the controller entirely

| Question | Command |
|---|---|
| Does the scan name the right squares? | look at `deployments/chess/board_overlay.png` after `observe` |
| Where does the scan think the piece *is*, in metres? | `/objects` |
| Is the arm reaching the coordinate it was given? | `/move_above <name> 0.06` then `/status` and compare |
| Is the target reachable at all, in pure kinematics? | `visualise_ik.py --plan --square d4` (SO-101) |
| Is the grasp physically possible at all? | `teleop_keyboard.py --square d4` (SO-101), then drive it by hand |
| What coordinate does a piece *actually* need? | `teleop_keyboard.py` (SO-101): position the jaws, then press `k` |
| Do the jaws open and close through their full range? | `check_gripper.py` (SO-101) |
| Does the arm hold a commanded pose? | `check_hold.py` (SO-101) |

The (SO-101) tools are in `tools/so101/`. Another arm answers these questions
with its own tools or its pendant.

[COMMANDS.md](COMMANDS.md) has the rest of these steps.

---

## 4. A worked example: the pick that reported success and held nothing

This trace from a real run shows why a false positive is worse than a
failure.

```
[Plan] → actions=[{"action": "pick", "parameters": {"object_name": "white_pawn_d4"}}]
Picking white_pawn_d4...
[Pick 1/3] Visual servo unavailable (gripper camera not calibrated) — using kinematics alone.
[OK] pick: grasped object (width=3 mm)
```

Stage by stage:

1. The plan named `white_pawn_d4`. No rescan line was printed, so stage 1 had
   already put it in `scene_state`.
2. The safety rules and all three preconditions passed.
3. The pick opened the jaws, hovered, descended and closed.
4. The jaws closed and reported **3 mm**.
5. The pick tested `width <= _MIN_HOLD_WIDTH_M` (now `MIN_HOLD_WIDTH_M` in
   `robot_config.py`), which was `0.003`.
6. The test returned `True`, so the action set `obj.position = None`. That is
   the only reason the `something_carried` postcondition then passed.

Step 5 is the bug. **3 mm is exactly the width the jaws report when they close
on nothing at all.** Two separate checks confirm it. The URDF's jaw
meshes still have a 3.15 mm pad gap at the gripper joint's closed limit. And
`set_gripper`'s stall detector stops the motor there, at 6.7 % of full span.
With the threshold sitting *on* that value, catching a total miss came down to
sensor noise in the last tenth of a millimetre. That is why an earlier run of
the same code correctly printed `Fully closed (0 mm) — missed`, and this one
didn't.

The fix was to stop judging by width. After the close, `squeeze_gripper()`
drives the jaws firmly shut. With nothing between them they meet. A piece
stops them, with the servo at its torque limit. The grip signal is now the
servo's **load**: about 30 on an empty close, against 200 or more on a held
piece. The widths overlapped, but the loads are far apart
(`mrcf/backends/so101/gripper.py`, `_GRIP_LOAD_FRACTION`).

A false positive cascades. The world model now believes it is carrying a pawn
it never touched, and the preconditions read that belief. So:

- The next `pick` is **blocked** by `nothing_carried`, and the message blames
  the plan instead of the grip.
- A `place` passes `something_carried`, drives to the destination and opens the
  jaws over an empty square.

Both look like planner bugs, but neither is. When a skill's success check is
wrong, the symptom shows up one or two commands later, somewhere else.

---

## 5. Failure modes by stage

| Symptom | Most likely stage | First thing to check |
|---|---|---|
| Right piece, wrong type or colour | 1 | `board_overlay.png`, then `/objects`. Its position comes from the camera's outline of it. Its type comes from the vision model, shown that one piece close up in two frames and held to the type list (`chessboard.label_pieces`). "Could not tell what it is" means the two looks disagreed. Tell it with `/identify`, and the memory keeps it. |
| Piece not in the world model at all | 1 | `observe`, then `/objects`. `board_debug.png` is written when detection fails. |
| Sensible plan, nonsense arguments | 4 | Compare `/pick <name>` with the English version. |
| `[PreCondition]` blocks something reasonable | 5 | `/objects`. Usually the world model thinks it is holding something. |
| Action never offered to the planner | 2–4 | `always=False` and an empty scene (§2). |
| Arm goes to the wrong place | 7 | `/move_above` and `/status`, then `visualise_ik.py --plan`. |
| `IK stalled N cm from target` | 7 | The solver is stuck at a fixed point and the arm is not holding the angles. Run `check_hold.py`. |
| Reports success, nothing held | 7 | The grip check in `squeeze_gripper` (§4). |
| `[Stop] …` mid-command | — | Something tripped the interlock, and the reason says what. Nothing moves until that source clears it. A watcher clears its own. See `env.interlock.reason`. |

---

## 6. Adding an action

A verb on one named object takes one call, `manipulation.object_action`.
[POLICIES.md](POLICIES.md) shows how to write its policy and test both
offline. An action of your own, written by hand:

```python
@register_action(
    "my_skill",
    description="One line the planner will read.",
    parameters={"object_name": "Exact name of the object."},
    conditions={"pre": ["object_exists"], "post": ["something_carried"]},
    always=False,
)
def skill_my_skill(env, scene_state, object_name, **_):
    obj = scene_state.find(object_name)
    if obj is None or obj.position is None:
        print(f"  [Error] '{object_name}' not in scene.")
        return False
    outcome = run_policy(my_policy, "my_skill", env, target=manipulation.target_of(obj))
    # ... record what happened in scene_state / scene_state.memory ...
    return outcome.ok
```

`my_policy(env, target, ...)` is the motion. It takes geometry, returns a
`policies.Outcome`, and reads nothing from the world model (DEPLOYMENT_SPEC.md,
"Adding a capability").

Five rules matter, because breaking them fails quietly:

1. `description` and `parameters` are all the planner knows about the action.
   If a small model misuses a skill, fix that text first.
2. Accept `**_`. `dispatch()` passes on whatever the model returned, including
   keys you never declared.
3. The policy returns an `Outcome`. `run_policy` reads it and prints its
   message. On a failure it also gets a VLM description of the scene, unless
   the policy `refused` before moving. A `MotionStopped` from the interlock
   passes straight through. Do not catch it.
4. **Make the success check tell a hold from a miss.** It is all that stands
   between a failed grasp and a world model that is wrong for every later
   command.
5. Return the outcome from the registered function. `dispatch()` passes it to
   the controller, which stops the plan and re-plans on `False`. A function
   that returns nothing is treated as "did not say", so its failure goes
   unnoticed.
