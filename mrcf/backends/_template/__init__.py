"""
mrcf/backends/_template — a new arm's package, to copy:

    cp -r mrcf/backends/_template mrcf/backends/my_arm
    export MRCF_ARM=my_arm      # run.py and tools/ now drive it, once it is set up:
                                # COMMANDS.md, "First-time setup on another arm"

The interface every arm exports is mrcf/backends/__init__.py's; DEPLOYMENT_SPEC.md,
"Adding an arm", says why each part is there. Everything to change is marked
YOUR ARM. As it stands it runs on an arm that exists only in memory
(env.InMemoryArm), and tests/test_template.py runs the backend contract on it,
so a copy starts from an arm that keeps the contract and can be kept keeping
it one method at a time: point that test's make_env at the copy, with its SDK
faked, before the copy drives the real arm.

The leading underscore keeps it out of backends.available(): it becomes an
arm when it is copied to a name of its own. Its imports are relative, so the
copy needs no edits to run.

    NAME              what to call the arm in messages
    config            robot_config.py: the rig settings and the arm's own
    make_env()        env.TemplateEnv, wired from config, not yet connected
    make_policies()   ParallelJaw (mrcf/policies/parallel_jaw/), with this gripper's GripperSpec
    describe()        what it connects to, for the start-up line
    tour_and_calibrate(env)
                      re-measure the overhead camera on a running arm (a tag tour)
"""

from . import robot_config as config

NAME = "My Arm"                         # YOUR ARM

__all__ = ["NAME", "config", "make_env", "make_policies", "describe", "tour_and_calibrate"]


def describe() -> str:
    """What the arm connects to, for the start-up message."""
    return f"{NAME} at {config.HOST}" if config.HOST is not None else f"{NAME} (in memory)"


def make_env(profile=None, **overrides):
    """
    The arm's env, wired from config, not yet connected to the arm (its cameras
    open as it is made: RealRobotEnv). The settings every arm shares come from
    backends/wiring.py (the cameras' jobs and calibrations, the gripper tag, the
    sentry, the jaw model, the gripper's box); the arm's own are listed here.
    profile: a deployment's rules for this arm (mrcf/profiles.py), laid over
    config. Keyword overrides win; overhead_camera_width/_height set the overhead
    camera's capture size (tools/calibrate_overhead_camera.py asks for 1280x720).
    """
    from mrcf.backends import wiring

    from .env import TemplateEnv, make_sdk

    from mrcf import profiles

    rc = profiles.layered(config, profile)      # a deployment's rules for this arm over its facts (mrcf/profiles.py)
    kwargs = dict(
        wiring.camera_kwargs(rc, overrides),
        **wiring.rig_kwargs(rc),
        sdk=make_sdk(rc.HOST),
        control_hz=rc.CONTROL_HZ,
        speed_m_s=rc.SPEED_M_S,
        home_xyz=rc.HOME_XYZ,
        park_xyz=rc.PARK_XYZ,
        base_xy=rc.BASE_XY,
        jaw_axis=rc.JAW_AXIS,
        stream_fps=rc.STREAM_FPS,
        stream_bounds=rc.WORKSPACE_BOUNDS,
        robot_type=rc.ROBOT_TYPE,
    )
    if rc.HOST is None:
        kwargs["unchecked_facts"] = ()   # the in-memory stand-in: nothing physical to be wrong (mrcf/arm_facts.py)
    kwargs.update(overrides)
    return TemplateEnv(**kwargs)


def make_policies(profile=None, **overrides):
    """
    The arm's policy library: ParallelJaw with this gripper's GripperSpec, from
    config (wiring.gripper_spec: each setting it names that config defines).
    The rest of GripperSpec's fields keep their defaults until the arm shows
    they should not (mrcf/policies/parallel_jaw/spec.py says what each is).
    profile: a deployment's rules for this arm (mrcf/profiles.py), laid over
    config. Keyword overrides win.
    """
    from mrcf.backends import wiring
    from mrcf.policies.parallel_jaw import ParallelJaw

    from mrcf import profiles

    return ParallelJaw(wiring.gripper_spec(profiles.layered(config, profile), **overrides))


def tour_and_calibrate(env) -> dict:
    """
    overhead_calibration.check_camera's `tour`: re-measure the overhead camera
    and where the board is on the arm, with a tag tour (tools/tag_tour.py; it
    needs the board's and the gripper's tags). Leave it out if the rig has none.
    """
    from tools import tag_tour
    return tag_tour.tour_and_calibrate(env)
