"""
llm_controller.py — the chess deployment's natural-language controller, on
its arm, the SO-101 (config.ARM_NAME).

The loop itself is the framework's (controller.py: decompose, perceive, plan,
safety, dispatch, recover from a fresh look). What this adds is chess: the
board scan as the way to look at the scene, the arm parked out of the overhead
camera's view to do it, the piece vocabulary in the planner's examples, and
the object memory that keeps who each piece is across scans and restarts.

Before running for the first time, work through "First-time setup" in
COMMANDS.md: on the SO-101 at least find_poses.py, fetch_urdf.py (with placo
installed), tools/calibrate_overhead_camera.py, and `ollama pull` for MODEL and
PLANNING_MODEL — or a remote model server (config.py, "Models").

Usage (from the repository root; the deployment is the package deployments.chess):
    python -m deployments.chess.llm_controller
    python run.py --deployment chess        # the same, through the launcher
"""

import sys as _sys

if not __package__:   # run as a file: its relative imports below would fail with a puzzle
    _sys.exit("Run the chess controller from the repository root as a module: "
              "python -m deployments.chess.llm_controller (or python run.py).")

# Line-buffered, so progress shows as it happens when output is piped to a
# log. Only where the stream supports it (a notebook's or an IDE's may not).
if hasattr(_sys.stdout, "reconfigure") and not getattr(_sys.stdout, "line_buffering", True):
    _sys.stdout.reconfigure(line_buffering=True)

from mrcf import controller
from mrcf import manipulation
from mrcf import safety as _safety
from mrcf import vlm as _vlm
from mrcf.vlm import SceneState

from . import config
from . import skills  # also registers every @register_action skill, as a side effect
from . import play    # noqa: F401 — registers move, play_opening and openings
from . import game    # registers new_game, my_move, game, hint, resign, end_game, resume_game
from .config import ARM
from .skills import POLICIES, goto_observe_pose, look_around, scan_scene

DEPLOYMENT = controller.Deployment(
    name=f"{ARM.NAME} chess (real hardware)",
    scan=scan_scene,
    observe_pose=goto_observe_pose,
    look_around=look_around,
    home=POLICIES.home,
    examples=["pick up the pawn", "move the pawn to e5", "move the knight to f6", "open the gripper",
              "show robot status", "let's play a game of chess", "I played e5"],
    task_planner=config.TASK_PLANNER_ENABLED,
    watchers=config.WATCHERS,
    on_start=game.on_start,     # carry on the saved game, or begin one on a board set up for it
)


def configure_models_and_safety() -> None:
    """The safety layer's limits and the planner's chess vocabulary."""
    _safety.configure(
        bounds=config.WORKSPACE_BOUNDS,
        thresholds=config.SAFETY_THRESHOLDS,
        sequence_validators=[manipulation.gripper_sequence_validator],
    )
    _vlm.configure(
        model_server=config.MODEL_SERVER,
        model_url=config.MODEL_URL,
        model=config.MODEL,
        planning_model=config.PLANNING_MODEL,
        observe_cameras=[config.OBSERVE_CAMERA],
        render_w=config.RENDER_W,
        render_h=config.RENDER_H,
        work_plane_z=config.TABLE_Z,
        # Pieces are named "<colour>_<type>_<square>" and empty squares are
        # named by their algebraic coordinate — both come straight from
        # chessboard.discover(), so these examples use the real vocabulary the
        # planner will actually see in the scene summary.
        plan_few_shot=[
            {"role": "user",      "content": "User command: \"move the pawn to e5\"\nKnown objects: white_pawn_e4\nPlacement targets: e5, e6, d5"},
            {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"white_pawn_e4\"}},{\"action\":\"place\",\"parameters\":{\"target_name\":\"e5\"}}],\"continue_with\":\"\"}"},
            {"role": "user",      "content": "User command: \"pick up the knight\"\nKnown objects: black_knight_g6, white_pawn_e4"},
            {"role": "assistant", "content": "{\"actions\":[{\"action\":\"pick\",\"parameters\":{\"object_name\":\"black_knight_g6\"}}],\"continue_with\":\"\"}"},
            {"role": "user",      "content": "User command: \"place it there\"\nRobot state: Currently holding: black_knight_g6\nPlacement targets: f6, g5"},
            {"role": "assistant", "content": "{\"actions\":[{\"action\":\"place\",\"parameters\":{\"target_name\":\"f6\"}}],\"continue_with\":\"\"}"},
            # Pieces nothing has identified yet are "<colour>_piece_<square>";
            # the user saying what one is goes to identify, not to a motion.
            {"role": "user",      "content": "User command: \"the piece on c3 is the white bishop\"\nKnown objects: white_piece_c3, black_piece_f6"},
            {"role": "assistant", "content": "{\"actions\":[{\"action\":\"identify\",\"parameters\":{\"object_name\":\"white_piece_c3\",\"what\":\"white bishop\"}}],\"continue_with\":\"\"}"},
            # In a game against the arm (game.py) the person moves by hand and
            # says so; the arm looks for the move itself when none is given.
            {"role": "user",      "content": "User command: \"I played e5\"\nRobot state: Gripper: empty. Chess game in progress: the person's move (black) — 'done' or a move means my_move.\nKnown objects: white_pawn_e4, black_pawn_e7"},
            {"role": "assistant", "content": "{\"actions\":[{\"action\":\"my_move\",\"parameters\":{\"move\":\"e5\"}}],\"continue_with\":\"\"}"},
            {"role": "user",      "content": "User command: \"done\"\nRobot state: Gripper: empty. Chess game in progress: the person's move (black) — 'done' or a move means my_move.\nKnown objects: white_pawn_e4, black_pawn_e5"},
            {"role": "assistant", "content": "{\"actions\":[{\"action\":\"my_move\",\"parameters\":{}}],\"continue_with\":\"\"}"},
        ],
        decompose_extra_system=(
            "This robot's only motion skills are: pick(object_name), place(target_name), "
            "move_above(object_name), stand_up(object_name), home, open_gripper, close_gripper"
            + "".join(f", {name}(object_name)" for name in skills.LEARNED) + ".\n"
            "'pick up X and put it on Y' is a SINGLE atomic sub-goal — do not split it into "
            "separate 'pick X' and 'place Y' sub-goals; the lower-level planner will do that.\n"
            "Always preserve the EXACT destination the user specified.\n"
            "In a game of chess against the arm, 'new game', 'I played e5', 'your move', 'done', 'go' and "
            "'give me a hint' are each a SINGLE sub-goal, kept in the user's words."
        ),
        discover_extra_system=config.DISCOVER_GUIDANCE,
        plan_state_fn=lambda ss: (
            (f"Currently holding: {held.name} ({held.description})"
             if (held := ss.held_object()) else "Gripper: empty")
            + (f". {line}" if (line := game.state_line()) else "")
        ),
    )


def main():
    # Fail here rather than mid-run: without a homography the overhead camera
    # can't turn a pixel into a position, so discover_scene() finds objects but
    # every one of them has position=None, and every pick/place is rejected for
    # reasons that look nothing like the actual cause.
    if config.OBSERVE_CAMERA == config.OVERHEAD_CAMERA_NAME and config.OVERHEAD_HOMOGRAPHY is None:
        print(f"\nOVERHEAD_HOMOGRAPHY is None in the {ARM.NAME}'s robot_config — the overhead camera is\n"
              "the primary perception camera, so nothing can be located until it's calibrated.\n"
              "See COMMANDS.md, first-time setup (on any arm: tools/calibrate_overhead_camera.py).\n")
        return

    where = ARM.describe() if hasattr(ARM, "describe") else f"the {ARM.NAME}"
    print(f"Connecting to {where}...")
    env = ARM.make_env(profile=config.PROFILE)
    env.connect()
    print("Connected.")
    try:
        # Who each piece is, kept across scans and restarts (object_memory.py).
        # A piece the memory says is held, in a gripper that is empty after the
        # restart, was taken out by hand: it is somewhere, not in the jaws.
        from . import board_log
        scene_state = SceneState(memory=board_log.memory())
        if not POLICIES.holding(env):
            for track in scene_state.memory.in_state("held"):
                scene_state.memory.released(track.id, "the gripper was empty when the controller started")
        configure_models_and_safety()
        controller.run(env, scene_state, DEPLOYMENT)
    finally:
        print("Disconnecting...")
        env.stop()


if __name__ == "__main__":
    main()
