"""dispatch(): a skill's reported outcome reaches the caller; a blocked or broken one is a failure."""

import numpy as np

from mrcf.vlm import SceneObject, SceneState, dispatch, register_action


@register_action("_test_succeeds", description="test")
def _succeeds(env, scene_state, **_):
    return True


@register_action("_test_fails", description="test")
def _fails(env, scene_state, **_):
    return False


@register_action("_test_reports_nothing", description="test")
def _reports_nothing(env, scene_state, **_):
    pass


@register_action("_test_raises", description="test")
def _raises(env, scene_state, **_):
    raise RuntimeError("motor bus gone")


@register_action("_test_needs_object", description="test",
                 parameters={"object_name": "an object"}, conditions={"pre": ["object_exists"]})
def _needs_object(env, scene_state, object_name, **_):
    return True


def test_outcomes_pass_through():
    state = SceneState()
    assert dispatch({"action": "_test_succeeds"}, None, state) is True
    assert dispatch({"action": "_test_fails"}, None, state) is False
    assert dispatch({"action": "_test_reports_nothing"}, None, state) is None


def test_actions_that_cannot_run_are_failures():
    state = SceneState()
    assert dispatch({"action": "_test_raises"}, None, state) is False
    assert dispatch({"action": "_no_such_action"}, None, state) is False
    assert dispatch({"action": "_test_needs_object", "object_name": "red_cube"}, None, state) is False


def test_preconditions_see_nested_parameters():
    state = SceneState(objects=[SceneObject("red_cube", "a red cube", position=np.zeros(3))])
    action = {"action": "_test_needs_object", "parameters": {"object_name": "red_cube"}}
    assert dispatch(action, None, state) is True
