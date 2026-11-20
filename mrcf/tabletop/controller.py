"""
controller.py — the table task's natural-language controller, for the
deployment it is bound to (tabletop.use(config)): that deployment's arm, its
rules for it, its layout. A deployment's llm_controller.py runs main().

The loop is the framework's (mrcf/controller.py: decompose, perceive, plan,
safety, dispatch, recover from a fresh look). This adds the table: the surface
scan with the vision model naming each object, the arm out of the overhead
camera's view to do it where it hides the table, the marked spots,
place_beside, containers, and the planner's examples in that vocabulary.

Before the first run: the arm's first-time setup (COMMANDS.md), the overhead
camera calibrated (tools/calibrate_camera_by_hand.py, a tag on an arm with
freedrive; or tools/calibrate_overhead_camera.py, a printed board), the segmenter
(pip install -e ".[measure]") and `ollama pull` for MODEL — or a remote model server
(settings.py, "Models").
"""

import sys as _sys

if hasattr(_sys.stdout, "reconfigure") and not getattr(_sys.stdout, "line_buffering", True):
    _sys.stdout.reconfigure(line_buffering=True)

from mrcf import controller
from mrcf import manipulation
from mrcf import safety as _safety
from mrcf import vlm as _vlm
from mrcf.vlm import SceneState

from . import bound, cfg
from . import skills  # registers every @register_action skill, as a side effect
from .skills import POLICIES, goto_observe_pose, scan_scene


def _name() -> str:
    """The bound deployment's name (deployments.<name>.config)."""
    module = bound().__name__
    return module.rsplit(".", 2)[-2] if "." in module else module


DEPLOYMENT = controller.Deployment(
    name=f"{cfg.ARM.NAME} {_name()} (real hardware)",
    scan=scan_scene,
    observe_pose=goto_observe_pose,
    home=POLICIES.home,
    examples=["pick up the red bottle", "put the cup on the tray", "put the box next to the bottle",
              "put the Z block on the P block", "stack the blocks", "this spot is the tray",
              "what is on the table?"],
    task_planner=cfg.TASK_PLANNER_ENABLED,
    watchers=cfg.WATCHERS,
    on_start=skills.on_start,
    defer_failed=bool(getattr(cfg, "DEFER_FAILED", False)),
    look_around=skills.look_for,
    home_after_command=bool(getattr(cfg, "HOME_AFTER_COMMAND", False)),
    remembered_plan=skills.remembered_plan if getattr(cfg, "PLAN_MEMORY", False) else None,
    plan_made=skills.plan_made if getattr(cfg, "PLAN_MEMORY", False) else None,
    quick_plan=skills.quick_plan if getattr(cfg, "QUICK_PUT_INTO", False) else None,
    refind_near_m=getattr(cfg, "REFIND_NEAR_M", None),
    task_plan=skills.task_plan if getattr(cfg, "TASKS", None) else None,
    task_more=skills.task_more if getattr(cfg, "TASKS", None) else None,
    task_names=tuple(getattr(cfg, "TASKS", None) or ()),
    planner=bool(getattr(cfg, "PLANNER", True)),
)

# What the planner learns the vocabulary from: object names are the vision
# model's words with their colour ("red_bottle"), marked spots their own
# names, "where_<object>_was" the spot a pick emptied, "beside_<object>" made
# by place_beside.
PLAN_FEW_SHOT = [
    {"role": "user", "content": "User command: \"put the red bottle on the tray\"\n"
                                "Known objects: red_bottle, blue_mug\nPlacement targets: tray, shelf_spot"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"red_bottle\"}},"
                                     "{\"action\":\"place\",\"parameters\":{\"target_name\":\"tray\"}}],\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"put the mug next to the bottle\"\n"
                                "Known objects: red_bottle, blue_mug\nPlacement targets: tray"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"blue_mug\"}},"
                                     "{\"action\":\"place_beside\",\"parameters\":{\"object_name\":\"red_bottle\",\"side\":\"any\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"put it to the left of the box\"\n"
                                "Robot state: Currently holding: blue_mug\nKnown objects: green_box"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"place_beside\",\"parameters\":{\"object_name\":\"green_box\",\"side\":\"left\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"put it back\"\nRobot state: Currently holding: blue_mug\n"
                                "Placement targets: tray, where_blue_mug_was"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"place\",\"parameters\":{\"target_name\":\"where_blue_mug_was\"}}],"
                                     "\"continue_with\":\"\"}"},
    # Stopped mid-plan (a stack knocked down) with a block still in the jaws: it
    # goes down first, where it came from, then the goal is built again.
    {"role": "user", "content": "User command: \"put the green_block_7 on the red_block_u and then the "
                                "blue_block_d on top\"\nRobot state: Currently holding: blue_block_d\n"
                                "Known objects: red_block_u, green_block_7\nPlacement targets: where_blue_block_d_was\n"
                                "The step 'plan' failed: something moved 2 cm from red_block_u while the arm worked."},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"place\",\"parameters\":{\"target_name\":"
                                     "\"where_blue_block_d_was\"}},"
                                     "{\"action\":\"pick\",\"parameters\":{\"object_name\":\"green_block_7\"}},"
                                     "{\"action\":\"place_on\",\"parameters\":{\"object_name\":\"red_block_u\"}},"
                                     "{\"action\":\"pick\",\"parameters\":{\"object_name\":\"blue_block_d\"}},"
                                     "{\"action\":\"place_on\",\"parameters\":{\"object_name\":\"green_block_7\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"put Z on P\"\n"
                                "Known objects: green_block_z, red_block_p, blue_block_d"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"green_block_z\"}},"
                                     "{\"action\":\"place_on\",\"parameters\":{\"object_name\":\"red_block_p\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"stack the blocks into one tower\"\n"
                                "Known objects: red_block_p, blue_block_d, green_block_z, red_bottle"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"blue_block_d\"}},"
                                     "{\"action\":\"place_on\",\"parameters\":{\"object_name\":\"red_block_p\"}},"
                                     "{\"action\":\"pick\",\"parameters\":{\"object_name\":\"green_block_z\"}},"
                                     "{\"action\":\"place_on\",\"parameters\":{\"object_name\":\"blue_block_d\"}}],"
                                     "\"continue_with\":\"\"}"},
    # Part built already, or a block in the jaws (a plan made again after the user
    # took one off): finish from the table as it is. 2026-11-13, the UR5: a plan made
    # again with blue in the jaws put it down and left red off the tower.
    {"role": "user", "content": "User command: \"stack the blocks into one tower\"\n"
                                "Known objects: red_block_p, blue_block_d [ON TOP OF red_block_p], green_block_z"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"green_block_z\"}},"
                                     "{\"action\":\"place_on\",\"parameters\":{\"object_name\":\"blue_block_d\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"stack the blocks into one tower\"\n"
                                "Known objects: red_block_p, blue_block_d, green_block_z (in the gripper)\n"
                                "Robot state: Currently holding: green_block_z"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"place_on\",\"parameters\":{\"object_name\":\"red_block_p\"}},"
                                     "{\"action\":\"pick\",\"parameters\":{\"object_name\":\"blue_block_d\"}},"
                                     "{\"action\":\"place_on\",\"parameters\":{\"object_name\":\"green_block_z\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"take the top block off and put it next to the tower\"\n"
                                "Known objects: red_block_p, blue_block_d [ON TOP OF red_block_p], "
                                "green_block_z [ON TOP OF blue_block_d]"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"green_block_z\"}},"
                                     "{\"action\":\"place_beside\",\"parameters\":{\"object_name\":\"red_block_p\",\"side\":\"any\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"that one is a glue stick, it's 9 cm tall\"\nKnown objects: object_7, red_bottle"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"identify\",\"parameters\":{\"object_name\":\"object_7\","
                                     "\"what\":\"glue stick 9 cm\"}}],\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"this spot is the shelf\"\nKnown objects: blue_mug, red_bottle"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"mark_place\",\"parameters\":{\"name\":\"shelf\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"call where the mug is the parking spot\"\nKnown objects: blue_mug"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"mark_place\",\"parameters\":{\"name\":\"parking spot\","
                                     "\"object_name\":\"blue_mug\"}}],\"continue_with\":\"\"}"},
]


# Containers (an arm with GripperSpec.place_in: skills.containers_on): put INTO,
# not on; one not yet listed is named in the user's words, and looked for.
CONTAINER_FEW_SHOT = [
    {"role": "user", "content": "User command: \"put the red ball in the tub\"\n"
                                "Known objects: red_ball, blue_mug\nPlacement targets: plastic_tub"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"red_ball\"}},"
                                     "{\"action\":\"place_in\",\"parameters\":{\"container\":\"plastic_tub\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"put everything into the box\"\n"
                                "Known objects: red_ball, green_cup — in cardboard_box already, white_plug\n"
                                "Placement targets: tray, cardboard_box"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"red_ball\"}},"
                                     "{\"action\":\"place_in\",\"parameters\":{\"container\":\"cardboard_box\"}},"
                                     "{\"action\":\"pick\",\"parameters\":{\"object_name\":\"white_plug\"}},"
                                     "{\"action\":\"place_in\",\"parameters\":{\"container\":\"cardboard_box\"}}],"
                                     "\"continue_with\":\"\"}"},
    {"role": "user", "content": "User command: \"drop the mug into the basket\"\n"
                                "Known objects: blue_mug, red_ball\nPlacement targets: tray"},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"blue_mug\"}},"
                                     "{\"action\":\"place_in\",\"parameters\":{\"container\":\"basket\"}}],"
                                     "\"continue_with\":\"\"}"},
]
CONTAINER_GUIDANCE = (
    "Putting something IN or INTO a container — a tub, a bowl, a box, a basket, a tray — is place_in(container), "
    "never place_on. A container not listed among the placement targets is named in the user's words "
    "(\"plastic tub\"): it is looked for by those words. An object marked \"in <container> already\" is in it: "
    "do not pick it to put it there again.\n"
)


def configure_models_and_safety() -> None:
    """The safety layer's limits and the planner's table vocabulary."""
    _safety.configure(
        bounds=cfg.WORKSPACE_BOUNDS,
        thresholds=cfg.SAFETY_THRESHOLDS,
        sequence_validators=[manipulation.gripper_sequence_validator, manipulation.stack_sequence_validator],
    )
    _vlm.configure(
        model_server=cfg.MODEL_SERVER,
        model_url=cfg.MODEL_URL,
        model=cfg.MODEL,
        planning_model=cfg.PLANNING_MODEL,
        observe_cameras=[cfg.OBSERVE_CAMERA],
        render_w=cfg.RENDER_W,
        render_h=cfg.RENDER_H,
        work_plane_z=cfg.TABLE_Z,
        plan_few_shot=PLAN_FEW_SHOT + (CONTAINER_FEW_SHOT if skills.containers_on() else []),
        decompose_extra_system=(
            "This robot's only motion skills are: pick(object_name), place(target_name), "
            "place_on(object_name), place_beside(object_name, side), move_above(object_name), home, "
            "open_gripper, close_gripper, freedrive(seconds)"
            + (", place_in(container) — into a container" if skills.containers_on() else "") + ".\n"
            + (CONTAINER_GUIDANCE if skills.containers_on() else "") +
            "'On top of' / 'on' another OBJECT is place_on (stacking: it goes on the top of that object's "
            "stack); 'on' a marked spot is place; 'next to' is place_beside. An object marked [ON TOP OF X] "
            "stands on X: X cannot be picked until what is on it is moved.\n"
            "It also has skills that do not move the arm: mark_place(name, object_name) remembers a spot on "
            "the table by a name — where the gripper is now, or where a named object stands ('this spot is the "
            "tray', 'call where the cup is the tray'); forget_place(name); identify(object_name, what) records "
            "what the user says an object is. A command for one of these is ONE sub-goal, kept in the user's "
            "words: never turn it into a motion.\n"
            "'pick up X and put it on/next to Y' is a SINGLE atomic sub-goal — do not split it into "
            "separate 'pick X' and 'place Y' sub-goals; the lower-level planner will do that.\n"
            "A goal about a group of objects ('stack the blocks', 'make two towers by colour', 'put all the "
            "blocks in a row') is ONE sub-goal, kept in the user's words: the lower-level planner sees every "
            "object's name and works out the order.\n"
            "A sequence of moves ('put the green block on the red one and then the blue block on top', 'move X "
            "to the tray, then Y next to it') is ONE sub-goal too, kept in the user's words: the planner writes "
            "the whole sequence at once, and each place_on goes on the top of the stack as it stands then. Each "
            "extra sub-goal costs a planning call of half a minute. Split only where a later step cannot be "
            "planned before an earlier one has been seen to happen ('... then tell me what is left').\n"
            "Always preserve the EXACT destination the user specified."
        ),
        discover_extra_system=cfg.DISCOVER_GUIDANCE,
        plan_state_fn=lambda ss: (
            f"Currently holding: {held.name} ({held.description})"
            if (held := ss.held_object()) else "Gripper: empty"
        ),
    )


def main():
    if cfg.OVERHEAD_CAM_MODEL is None:
        print(f"\nThe {cfg.ARM.NAME}'s overhead camera is not calibrated (OVERHEAD_CAM_MODEL is None), so nothing on\n"
              "the table can be found or measured. Calibrate it first: tools/calibrate_camera_by_hand.py (a tag\n"
              "on an arm with freedrive) or tools/calibrate_overhead_camera.py (a printed board). COMMANDS.md,\n"
              "first-time setup.\n")
        return

    arm = cfg.ARM
    where = arm.describe() if hasattr(arm, "describe") else f"the {arm.NAME}"
    print(f"Connecting to {where}...")
    env = arm.make_env(profile=cfg.PROFILE, **cfg.ENV_OVERRIDES)
    env.connect()
    if cfg.SURFACE_Z is not None:
        print(f"  [Surface] Working on the surface touched at {cfg.SURFACE_Z * 1000:+.1f} mm "
              f"({cfg.SURFACE_PATH.name}), not the arm's TABLE_Z.")
    print("Connected.")
    try:
        scene_state = SceneState(memory=skills.memory())
        if not POLICIES.holding(env):
            for track in scene_state.memory.in_state("held"):
                scene_state.memory.released(track.id, "the gripper was empty when the controller started")
        configure_models_and_safety()
        controller.run(env, scene_state, DEPLOYMENT)
    finally:
        print("Disconnecting...")
        env.stop()

