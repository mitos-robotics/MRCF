"""Why a close found nothing, asked of a vision model (mrcf/perception/grasp_diagnosis.py); the model stood in for."""

import numpy as np

from mrcf.perception import grasp_diagnosis as gd


def _frames():
    return [("try1_at_grasp", np.zeros((48, 64, 3), np.uint8)), ("try1_shut", np.full((48, 64, 3), 200, np.uint8))]


def test_its_answer_is_held_to_the_causes_and_both_pictures_are_shown():
    asked = {}

    def ask(prompt, images, **kw):
        asked.update(images=images, schema=kw["schema"], prompt=prompt)
        return {"cause": "too_high", "seen": "the jaws shut  over the plug's top"}
    got = gd.diagnose(_frames(), ask, "the white plug")
    assert got == ("too_high", "the jaws shut over the plug's top")
    assert len(asked["images"]) == 2 and "the white plug" in asked["prompt"]
    assert set(asked["schema"]["properties"]["cause"]["enum"]) >= {"too_high", "aim_off", "lying", "moved"}


def test_no_pictures_or_no_usable_answer_says_nothing():
    assert gd.diagnose([], lambda *a, **k: {"cause": "lying"}) is None
    assert gd.diagnose(_frames(), lambda *a, **k: {"cause": "who knows"}) is None
    assert gd.diagnose(_frames(), lambda *a, **k: {}) is None


def test_its_causes_are_the_ones_the_pick_acts_on():
    from mrcf.policies.base import MISS_CAUSES

    assert set(gd.CAUSES) == set(MISS_CAUSES)


def test_whether_it_is_carried_is_a_yes_or_no_on_the_wrist_picture():
    asked = {}

    def ask(prompt, images, **kw):
        asked.update(n=len(images), schema=kw["schema"])
        return {"carrying": True, "seen": "a blue scoop  held by its wall"}
    assert gd.held(np.zeros((48, 64, 3), np.uint8), ask) == (True, "a blue scoop held by its wall")
    assert asked["n"] == 1 and asked["schema"]["properties"]["carrying"]["type"] == "boolean"
    assert gd.held(np.zeros((48, 64, 3), np.uint8), lambda *a, **k: {"carrying": "maybe"}) is None
    assert gd.held(None, ask) is None


def test_a_grip_the_servo_cannot_feel_is_asked_about_once_and_believed():
    import dataclasses

    from mrcf.policies.parallel_jaw import GripperSpec, ParallelJaw

    spec = GripperSpec(max_open_m=0.08, tip_below_grasp_m=0.007, free_depth_m=0.055, finger_thickness_m=0.009,
                       finger_half_width_m=0.008)
    pol = ParallelJaw(dataclasses.replace(spec, camera="wrist", render_w=64, render_h=48))
    calls = []
    pol.ask_held = lambda frame: calls.append(1) or (True, "the scoop, by its wall")

    class _Env:
        def get_gripper_width(self):
            return 0.001

        def render_image(self, cam, width=None, height=None):
            return np.zeros((48, 64, 3), np.uint8)
    pol._grip_view = {"frame": None, "model": None, "gap": 0.001, "said": False}
    pol._seen_holding = lambda env: False
    assert pol._asked_holding(_Env()) is True and pol._asked_holding(_Env()) is True
    assert len(calls) == 1                                    # the answer stands for that grip
    pol.ask_held = None
    pol._grip_view = {"frame": None, "model": None, "gap": 0.001, "said": False}
    assert pol._asked_holding(_Env()) is False                # nothing to ask: empty, as before
