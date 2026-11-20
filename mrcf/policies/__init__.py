"""
policies — the policy library: how an arm does things, from geometry.

    base.py           Target, Obstacle, Measured, Outcome, Vision, and the
                      Policies interface every arm's library implements
    parallel_jaw/     pick / place / move_above / home for a two-finger gripper
                      working from above (GripperSpec, ParallelJaw) — planned;
                      its __init__.py lists the parts
    approaches.py, aligners.py
                      the ways a pick comes at an object, and lines up over it
    stand_up.py       stand an object fallen on its side back up (ParallelJaw.stand_up)
    place_in.py       put what is held into a container (ParallelJaw.place_in)
    rim_pick.py, handle_pick.py, lying_pick.py
                      a pick by a wall, by a handle, and of a thing lying along
                      the table; grasp_sense.py checks a grip's fingers first
    nudge.py, touch_surface.py, learn_jaw_view.py
                      a push, a touch for the surface's height, the wrist
                      camera learning its jaws
    arm_only.py       the library for an arm with no gripper
    learned.py        a trained model run on the arm's stream for one segment
                      of a skill (LearnedPolicy, Model, CallableModel); chain()
                      and hover_over() put it between planned steps
    lerobot_model.py  a LeRobot checkpoint as a learned.Model (ACT, diffusion,
                      SmolVLA, pi0, SAC)
    rl.py             a Stable-Baselines3 policy as a learned.Model (SB3Model),
                      and the codec it reads and drives the stream through
    packages.py       a learned policy as a package: its model and a manifest
                      of the stream it was trained on; formats and sources

A policy never sees the world model, the object memory or the planner: it is
given geometry (a learned one, the arm's stream) and returns an Outcome (see base.py). manipulation.py is what
connects it to the rest. tests/test_architecture.py keeps it that way.

The runs behind the policies' values and fixes, with their dates: HISTORY.md.
"""

from mrcf.policies.base import Lying, Measured, Obstacle, Outcome, Policies, Target, Vision  # noqa: F401
