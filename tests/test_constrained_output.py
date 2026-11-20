"""
What a model may answer: the planner held to the scene's actions and names
(action_registry.plan_schema), and vlm.label held to fixed lists and to its
looks agreeing. No model server: the schema is checked as data, and the calls
are stood in for.
"""

import numpy as np

from mrcf import model_server, vlm
from mrcf.action_registry import plan_schema, scene_names
from mrcf.vlm import SceneObject, SceneState, register_action


@register_action("_test_take", description="test", parameters={"object_name": "x", "speed": "y"})
def _take(env, scene_state, **_):
    return True


@register_action("_test_rest", description="test", always=True)
def _rest(env, scene_state, **_):
    return True


def _scene(*objects, targets=()):
    s = SceneState()
    s.objects = [SceneObject(n, n, interactable=i, position=np.zeros(3)) for n, i in objects]
    s.targets = [SceneObject(n, n, interactable=False, position=np.zeros(3)) for n in targets]
    return s


def _choice(schema, action):
    return next(c for c in schema["properties"]["actions"]["items"]["anyOf"]
                if c["properties"]["action"] == {"const": action})


def test_a_plan_can_only_name_what_the_scene_has():
    # 2026-10-29: asked to pick the one piece on the board, the planner named
    # white_piece_c3, which was not there.
    scene = _scene(("white_piece_e4", True), ("shadow_d2", False), targets=["e5", "e6"])
    assert scene_names(scene) == ["white_piece_e4", "e5", "e6"]          # not the obstacle-only shadow
    take = _choice(plan_schema(scene), "_test_take")
    params = take["properties"]["parameters"]
    assert params["properties"]["object_name"] == {"enum": ["white_piece_e4", "e5", "e6"]}
    assert params["properties"]["speed"] == {} and params["required"] == ["object_name"]


def test_an_action_needing_a_name_is_left_out_when_the_scene_has_none():
    schema = plan_schema(_scene(("shadow_d2", False)))
    actions = [c["properties"]["action"]["const"] for c in schema["properties"]["actions"]["items"]["anyOf"]]
    assert "_test_rest" in actions and "_test_take" not in actions


def test_a_label_is_held_to_its_lists_and_kept_only_where_the_looks_agree(monkeypatch):
    sent, replies = [], iter([{"type": "pawn", "colour": "white"}, {"type": "pawn", "colour": "black"}])
    monkeypatch.setattr(vlm, "call_vlm", lambda prompt, image, **kw: sent.append(kw["schema"]) or next(replies))
    got = vlm.label(["look1", "look2"], {"type": ["pawn", "rook"], "colour": ["white", "black"]}, "what is it?")
    assert got == {"type": "pawn", "colour": None}                     # the colours differed
    assert sent[0]["properties"]["type"] == {"enum": ["pawn", "rook"]} and sent[0]["required"] == ["type", "colour"]


def test_no_answer_or_one_off_the_list_is_no_label(monkeypatch):
    replies = iter([{}, {"type": "pawn"}, {"type": "dragon"}])
    monkeypatch.setattr(vlm, "call_vlm", lambda prompt, image, **kw: next(replies))
    assert vlm.label(["a", "b"], {"type": ["pawn"]}, "?") == {"type": None}
    assert vlm.label(["a"], {"type": ["pawn"]}, "?") == {"type": None}


def test_numbered_things_are_named_in_one_call_each_held_to_the_list(monkeypatch):
    sent = []
    monkeypatch.setattr(vlm, "call_vlm", lambda prompt, image, **kw: sent.append((image, kw["schema"]))
                        or {"1": "white pawn", "2": "purple dragon"})
    got = vlm.label_numbered("sheet", 3, ["white pawn", "black king"], "what is each?")
    assert got == ["white pawn", None, None]          # off the list, and no answer, are no label
    (image, schema), = sent
    assert image == "sheet" and schema["required"] == ["1", "2", "3"]
    assert schema["properties"]["3"] == {"enum": ["white pawn", "black king"]}


def test_every_call_asks_the_model_not_to_think_first(monkeypatch):
    # gemma4 is a thinking model: left to think, it spent the label's whole
    # token cap on it and every label came back empty; a scene description
    # came back cut off mid-sentence (2026-10-30).
    sent = []

    class _Reply:
        status_code = 200

        def json(self):
            return {"message": {"content": '{"type": "pawn"}'}}

    monkeypatch.setattr(vlm, "_server", model_server.Ollama())
    monkeypatch.setattr(model_server.requests, "post", lambda url, json=None, **kw: sent.append(json) or _Reply())
    assert vlm.call_vlm("?", "img", schema={"type": "object"}) == {"type": "pawn"}
    assert sent[-1]["think"] is False and sent[-1]["format"] == {"type": "object"}
    vlm.call_vlm("?", "img")
    assert sent[-1]["think"] is False and sent[-1]["format"] == "json"
    vlm.call_vlm("?", "img", json_mode=False)
    assert sent[-1]["think"] is False and "format" not in sent[-1]

