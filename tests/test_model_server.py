"""
Where the models run (mrcf/model_server.py): Ollama's requests as every rig
has sent them, the same questions put to an OpenAI-compatible API, which
server the environment names, and what happens when a server says no. No
model is asked: requests.post is stood in for, and once a real HTTP server on
this machine answers as each kind would.
"""

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from mrcf import flight_recorder, model_server, vlm
from mrcf.model_server import Ollama, OpenAICompatible
from mrcf.vlm import SceneObject, SceneState, register_action

PNG = "iVBORw0KGgo="          # the first bytes of a PNG, base64
JPEG = "/9j/4AAQ"             # of a JPEG
KEY = "sk-test-0123456789-SECRET"


@pytest.fixture(autouse=True)
def no_model_settings(monkeypatch):
    """Nothing from this shell's environment, no waiting between retries."""
    for name in (model_server.SERVER_ENV, model_server.URL_ENV, model_server.KEY_ENV, model_server.EXTRA_ENV,
                 "OLLAMA_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vlm, "_RETRY_DELAYS", [0.0, 0.0])
    monkeypatch.setattr(vlm, "_model", "vision-model")
    monkeypatch.setattr(vlm, "_planning_model", "plan-model")


class _Reply:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


def _posts(monkeypatch, *replies):
    """requests.post answering with `replies` in turn (an exception is raised); what was sent, kept."""
    sent, queue = [], list(replies)

    def post(url, json=None, headers=None, timeout=None):
        sent.append({"url": url, "json": copy.deepcopy(json), "headers": headers, "timeout": timeout})
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(model_server.requests, "post", post)
    return sent


def _ollama(content):
    return _Reply(body={"message": {"role": "assistant", "content": content}, "done": True})


def _openai(content, finish="stop"):
    return _Reply(body={"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}]})


def _scene():
    s = SceneState()
    s.objects = [SceneObject("red_cube", "red cube", interactable=True, position=np.zeros(3))]
    return s


# ── Ollama: unchanged ─────────────────────────────────────────────────────────

def test_ollama_is_sent_what_every_rig_has_run_on(monkeypatch):
    # The requests vlm sent Ollama before there was any other server, field for
    # field: a label held to a schema, a plan, and a warm-up.
    @register_action("_ms_take", description="test", parameters={"object_name": "x"})
    def _take(env, scene_state, **_):
        return True

    monkeypatch.setattr(vlm, "_server", Ollama())
    sent = _posts(monkeypatch, _ollama('{"type": "pawn"}'),
                  _ollama('{"actions": [{"action": "_ms_take", "parameters": {"object_name": "red_cube"}}]}'),
                  _ollama("o"))
    schema = {"type": "object", "properties": {"type": {"enum": ["pawn"]}}}
    assert vlm.call_vlm("what is it?", PNG, system="sys", _tag="label", schema=schema) == {"type": "pawn"}
    scene = _scene()
    actions, _ = vlm.plan("take the cube", "a table", scene)
    vlm.warmup(model="vision-model", planning_model="vision-model")

    label, plan, warm = sent
    assert label == {"url": "http://localhost:11434/api/chat", "headers": {}, "timeout": 90.0, "json": {
        "model": "vision-model",
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "what is it?", "images": [PNG]}],
        "stream": False, "options": {"temperature": 0.1, "num_predict": 64}, "format": schema, "think": False}}
    assert plan["json"]["model"] == "plan-model"
    assert plan["json"]["messages"][0] == {"role": "system", "content": vlm._PLAN_SYSTEM}
    assert plan["json"]["messages"][1:-1] == [*vlm._PLAN_FEW_SHOT, *vlm._plan_extra_few_shot]
    assert plan["json"]["messages"][-1]["role"] == "user" and "images" not in plan["json"]["messages"][-1]
    assert plan["json"]["format"] == vlm.plan_schema(scene)
    assert plan["json"]["options"] == {"temperature": 0.1, "num_predict": 512} and plan["json"]["think"] is False
    assert actions == [{"action": "_ms_take", "parameters": {"object_name": "red_cube"}}]
    assert warm["timeout"] == 240.0 and warm["json"] == {
        "model": "vision-model", "messages": [{"role": "user", "content": "ok"}], "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1}, "think": False}


def test_an_ollama_too_old_for_a_schema_is_asked_for_plain_json(monkeypatch):
    monkeypatch.setattr(vlm, "_server", Ollama())
    sent = _posts(monkeypatch, _Reply(400, {"error": "invalid format"}), _ollama('{"actions": []}'))
    assert vlm.plan("rest", "a table", _scene()) == ([], None)
    assert isinstance(sent[0]["json"]["format"], dict) and sent[1]["json"]["format"] == "json"


def test_an_unknown_model_is_not_taken_for_a_refused_schema(monkeypatch, capsys):
    monkeypatch.setattr(vlm, "_server", Ollama())
    sent = _posts(monkeypatch, _Reply(404, {"error": "model 'nope' not found"}))
    assert vlm.call_vlm("?", PNG, schema={"type": "object"}) == {}
    assert len(sent) == 1 and "model 'nope' not found" in capsys.readouterr().out


# ── OpenAI-compatible ─────────────────────────────────────────────────────────

def test_an_openai_compatible_server_is_asked_the_same_question_its_own_way(monkeypatch):
    monkeypatch.setattr(vlm, "_server", OpenAICompatible("https://api.example.com/v1/", api_key=KEY))
    sent = _posts(monkeypatch, _openai('{"type": "pawn"}'))
    schema = {"type": "object", "properties": {"type": {"enum": ["pawn"]}}}
    assert vlm.call_vlm("image 1 = overhead, image 2 = wrist. What is it?", [PNG, JPEG], system="sys",
                        _tag="label", schema=schema) == {"type": "pawn"}
    (call,) = sent
    assert call["url"] == "https://api.example.com/v1/chat/completions"
    assert call["headers"] == {"Authorization": f"Bearer {KEY}"} and call["timeout"] == 90.0
    assert call["json"] == {
        "model": "vision-model",
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": [
                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{JPEG}"}},
                         {"type": "text", "text": "image 1 = overhead, image 2 = wrist. What is it?"}]}],
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema,
                                                                    "strict": False}},
        "temperature": 0.1, "max_tokens": 64}


def test_a_plan_from_an_openai_compatible_server(monkeypatch):
    @register_action("_ms_take", description="test", parameters={"object_name": "x"})
    def _take(env, scene_state, **_):
        return True

    monkeypatch.setattr(vlm, "_server", OpenAICompatible())
    sent = _posts(monkeypatch, _openai('{"actions": [{"action": "_ms_take", "parameters": {"object_name": '
                                       '"red_cube"}}, {"action": "_ms_made_up"}], "continue_with": ""}'))
    scene = _scene()
    assert vlm.plan("take the cube", "a table", scene) == (
        [{"action": "_ms_take", "parameters": {"object_name": "red_cube"}}], None)   # the made-up one dropped
    body = sent[0]["json"]
    assert sent[0]["url"] == "https://api.openai.com/v1/chat/completions" and sent[0]["headers"] == {}
    assert body["messages"][1:-1] == [*vlm._PLAN_FEW_SHOT, *vlm._plan_extra_few_shot]   # worked examples as turns
    assert body["response_format"]["json_schema"]["schema"] == vlm.plan_schema(scene)
    assert body["max_tokens"] == 512 and isinstance(body["messages"][-1]["content"], str)


def test_a_server_that_cannot_hold_a_schema_is_asked_for_json_then_for_text(monkeypatch, capsys):
    monkeypatch.setattr(vlm, "_server", OpenAICompatible())
    sent = _posts(monkeypatch,
                  _Reply(400, {"error": {"message": "response_format json_schema is not supported"}}),
                  _Reply(400, {"error": {"message": "'response_format' of type 'json_object' is not supported"}}),
                  _openai('Here it is:\n```json\n{"type": "pawn"}\n```'))
    assert vlm.call_vlm("?", PNG, schema={"type": "object"}) == {"type": "pawn"}
    assert [c["json"].get("response_format", {}).get("type") for c in sent] == ["json_schema", "json_object", None]
    out = capsys.readouterr().out
    assert "plain JSON" in out and "plain text" in out


def test_a_model_that_wants_its_own_token_field_and_temperature_is_asked_its_way_from_then_on(monkeypatch):
    srv = OpenAICompatible()
    monkeypatch.setattr(vlm, "_server", srv)
    sent = _posts(
        monkeypatch,
        _Reply(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. "
                                          "Use 'max_completion_tokens' instead."}}),
        _Reply(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.1 with this model."}}),
        _openai("a table"), _openai("still a table"))
    assert vlm.call_vlm("describe", PNG, json_mode=False, _tag="perceive") == "a table"
    assert vlm.call_vlm("describe", PNG, json_mode=False, _tag="perceive") == "still a table"
    assert [("max_tokens" in c["json"], "max_completion_tokens" in c["json"], "temperature" in c["json"])
            for c in sent] == [(True, False, True), (False, True, True), (False, True, False), (False, True, False)]
    assert sent[-1]["json"]["max_completion_tokens"] == 512


def test_a_thinking_model_that_wrote_nothing_says_why(monkeypatch, capsys):
    monkeypatch.setattr(vlm, "_server", OpenAICompatible())
    _posts(monkeypatch, _openai(None, finish="length"))
    assert vlm.call_vlm("?", PNG, schema={"type": "object"}, _tag="label") == {}
    out = capsys.readouterr().out
    assert "64-token cap" in out and model_server.EXTRA_ENV in out


def test_extra_body_is_added_to_every_request(monkeypatch):
    monkeypatch.setattr(vlm, "_server", Ollama(extra_body={"options": {"num_ctx": 8192}, "keep_alive": "10m"}))
    sent = _posts(monkeypatch, _ollama("ok"))
    vlm.call_vlm("?", json_mode=False, _tag="perceive")
    assert sent[0]["json"]["options"] == {"temperature": 0.1, "num_predict": 512, "num_ctx": 8192}
    assert sent[0]["json"]["keep_alive"] == "10m"


# ── Failures ──────────────────────────────────────────────────────────────────

def test_busy_and_slow_servers_are_asked_again_refusals_and_silence_are_not(monkeypatch, capsys):
    waited = []
    monkeypatch.setattr(vlm.time, "sleep", waited.append)
    monkeypatch.setattr(vlm, "_server", OpenAICompatible(api_key=KEY))

    sent = _posts(monkeypatch, _Reply(429, {"error": {"message": "slow down"}}, {"Retry-After": "7"}), _openai("{}"))
    assert vlm.call_vlm("?") == {} and len(sent) == 2 and waited == [7.0]      # its Retry-After, not ours

    sent = _posts(monkeypatch, *[requests.exceptions.Timeout()] * 3)
    assert vlm.call_vlm("?", json_mode=False) == "" and len(sent) == 3

    sent = _posts(monkeypatch, _Reply(401, {"error": {"message": f"Incorrect API key provided: {KEY}"}}))
    assert vlm.call_vlm("?") == {} and len(sent) == 1
    assert model_server.KEY_ENV in capsys.readouterr().out

    sent = _posts(monkeypatch, requests.exceptions.ConnectionError())
    assert vlm.call_vlm("?") == {} and len(sent) == 1
    assert "Cannot reach OpenAI-compatible server at https://api.openai.com/v1/chat/completions" in \
        capsys.readouterr().out


def test_the_key_is_never_printed_or_recorded(monkeypatch, capsys, tmp_path):
    recorder = flight_recorder.FlightRecorder()
    monkeypatch.setattr(flight_recorder, "RECORDER", recorder)
    recorder.start(tmp_path)
    srv = OpenAICompatible("https://user:password@api.example.com/v1?key=" + KEY, api_key=KEY)
    monkeypatch.setattr(vlm, "_server", srv)
    _posts(monkeypatch, _Reply(401, {"error": {"message": f"bad key {KEY}"}}), _openai('{"ok": true}'))
    vlm.configure(model_server=srv)
    vlm.call_vlm("?", PNG)
    vlm.call_vlm("?", PNG)
    recorder.stop()
    kept = "".join(p.read_text(errors="replace") for p in tmp_path.rglob("*") if p.is_file())
    shown = capsys.readouterr().out + repr(srv) + srv.where()
    assert kept and "models" in {p.name for p in tmp_path.rglob("*")}
    for secret in (KEY, "password", "user:"):
        assert secret not in shown and secret not in kept


def test_a_key_sent_unencrypted_off_this_machine_is_warned_about(capsys):
    OpenAICompatible("http://10.0.0.5:8000/v1", api_key=KEY)
    assert "unencrypted" in capsys.readouterr().out
    for quiet in (OpenAICompatible("http://localhost:1234/v1", api_key=KEY),
                  OpenAICompatible("http://127.0.0.1:8000/v1", api_key=KEY),
                  OpenAICompatible("https://api.example.com/v1", api_key=KEY),
                  OpenAICompatible("http://10.0.0.5:8000/v1")):                 # no key to give away
        assert "unencrypted" not in capsys.readouterr().out, quiet


def test_where_and_off_machine():
    assert not Ollama().off_machine() and not OpenAICompatible("http://[::1]:8000/v1").off_machine()
    assert OpenAICompatible().off_machine() and Ollama("http://gpu-box:11434").off_machine()


# ── Which server ──────────────────────────────────────────────────────────────

def test_the_environment_names_the_server_its_url_key_and_extra_body(monkeypatch):
    assert isinstance(model_server.make(), Ollama) and model_server.make().url == "http://localhost:11434/api/chat"
    monkeypatch.setenv("OLLAMA_URL", "http://gpu-box:11434/api/chat")           # as before there were others
    assert model_server.make().url == "http://gpu-box:11434/api/chat"
    monkeypatch.setenv(model_server.SERVER_ENV, "OpenAI")
    monkeypatch.setenv(model_server.KEY_ENV, KEY)
    monkeypatch.setenv(model_server.EXTRA_ENV, '{"reasoning_effort": "low"}')
    srv = model_server.make()
    assert isinstance(srv, OpenAICompatible) and srv.url == "https://api.openai.com/v1/chat/completions"  # not OLLAMA_URL
    assert srv.headers() == {"Authorization": f"Bearer {KEY}"} and srv.extra_body == {"reasoning_effort": "low"}
    monkeypatch.setenv(model_server.URL_ENV, "http://localhost:8000/v1")
    assert model_server.make().url == "http://localhost:8000/v1/chat/completions"
    assert model_server.make("ollama", url="http://gpu-box:11434/").url == "http://gpu-box:11434/api/chat"

    monkeypatch.setenv(model_server.EXTRA_ENV, '["not", "an object"]')
    with pytest.raises(ValueError, match=model_server.EXTRA_ENV):
        model_server.make()
    with pytest.raises(KeyError, match="known: ollama, openai"):
        model_server.make("nonesuch")


def test_configure_chooses_the_server(monkeypatch, capsys):
    vlm.configure(ollama_url="http://gpu-box:11434/api/chat")           # the old way: an Ollama there
    assert isinstance(vlm.server(), Ollama) and vlm.server().url == "http://gpu-box:11434/api/chat"
    assert "off this machine" in capsys.readouterr().out
    vlm.configure(model_server="openai", model_url="http://localhost:8000/v1")
    assert isinstance(vlm.server(), OpenAICompatible) and "off this machine" not in capsys.readouterr().out
    vlm.configure(model="another")                                       # the server stays
    assert isinstance(vlm.server(), OpenAICompatible) and capsys.readouterr().out == ""
    mine = Ollama()
    vlm.configure(model_server=mine)
    assert vlm.server() is mine


def test_a_plugin_server_is_found_by_its_entry_point(monkeypatch):
    class Mine(model_server.ModelServer):
        name = "mine"
        default_url = "https://mine.example/ask"

        def send(self, request):
            return json.dumps({"seen": request.images, "model": request.model, "schema": request.schema})

    plugin = SimpleNamespace(name="mine", load=lambda: Mine)
    monkeypatch.setattr(model_server, "SERVERS", dict(model_server.SERVERS))
    monkeypatch.setattr(model_server, "entry_points", lambda group: [plugin] if group == model_server.PLUGIN_GROUP else [])
    assert "mine" in model_server.available()
    vlm.configure(model_server="mine")
    assert vlm.call_vlm("?", PNG, schema={"type": "object"}) == {
        "seen": [PNG], "model": "vision-model", "schema": {"type": "object"}}


def test_an_answer_wrapped_in_words_is_still_read():
    assert vlm._parse_json('{"a": 1}') == {"a": 1}
    assert vlm._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert vlm._parse_json('Sure! {"a": [1, 2]} Hope that helps.') == {"a": [1, 2]}
    assert vlm._parse_json("[1, 2]") == [1, 2]
    with pytest.raises(json.JSONDecodeError):
        vlm._parse_json("no json here")


# ── Over real HTTP ────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if self.path == "/v1/chat/completions":
            reply = {"choices": [{"message": {"content": '{"what": "cube"}'}, "finish_reason": "stop"}]}
        else:
            reply = {"message": {"content": '{"what": "cube"}'}, "done": True}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_each_server_over_real_http():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    _Handler.seen = []
    try:
        vlm.configure(model_server=OpenAICompatible(base + "/v1", api_key=KEY))
        assert vlm.call_vlm("what is it?", [PNG]) == {"what": "cube"}
        vlm.configure(model_server="ollama", model_url=base)
        assert vlm.call_vlm("what is it?", [PNG]) == {"what": "cube"}
    finally:
        httpd.shutdown()
        httpd.server_close()
    openai, ollama = _Handler.seen
    assert openai["path"] == "/v1/chat/completions" and openai["auth"] == f"Bearer {KEY}"
    assert openai["body"]["messages"][-1]["content"][0]["image_url"]["url"] == f"data:image/png;base64,{PNG}"
    assert ollama["path"] == "/api/chat" and ollama["auth"] is None and ollama["body"]["messages"][-1]["images"] == [PNG]


# ── Refusals remembered, and only real ones ───────────────────────────────────

def test_a_refused_schema_is_left_out_of_that_kinds_calls_from_then_on(monkeypatch, capsys):
    # Not two requests every call: refused once, then plain JSON for that kind of call.
    monkeypatch.setattr(vlm, "_server", OpenAICompatible())
    sent = _posts(monkeypatch,
                  _Reply(400, {"error": {"message": "response_format json_schema is not supported"}}),
                  _openai('{"type": "pawn"}'), _openai('{"type": "rook"}'), _openai('{"type": "king"}'))
    assert vlm.call_vlm("?", PNG, schema={"type": "object"}, _tag="label") == {"type": "pawn"}
    assert vlm.call_vlm("?", PNG, schema={"type": "object"}, _tag="label") == {"type": "rook"}
    assert [c["json"].get("response_format", {}).get("type") for c in sent] == ["json_schema", "json_object",
                                                                                 "json_object"]
    assert capsys.readouterr().out.count("would not hold the answer to a schema") == 1
    # Another kind of call (another schema) is still asked with its own.
    assert vlm.call_vlm("?", PNG, schema={"type": "object"}, _tag="perceive") == {"type": "king"}
    assert sent[-1]["json"]["response_format"]["type"] == "json_schema"


def test_an_error_that_was_not_about_the_schema_is_not_remembered_as_a_refusal(monkeypatch):
    # An Ollama error with a schema is taken for a refused schema; asked without, it fails as well:
    # it was not the schema (a model that ran out of memory), and the next call keeps its schema.
    monkeypatch.setattr(vlm, "_server", Ollama())
    sent = _posts(monkeypatch,
                  _Reply(500, {"error": "model runner has unexpectedly stopped"}),
                  _Reply(500, {"error": "model runner has unexpectedly stopped"}),
                  _ollama('{"type": "pawn"}'))
    assert vlm.call_vlm("?", PNG, schema={"type": "object"}, _tag="label") == {}
    assert vlm.call_vlm("?", PNG, schema={"type": "object"}, _tag="label") == {"type": "pawn"}
    assert sent[-1]["json"]["format"] == {"type": "object"}


def test_a_thinking_model_that_says_nothing_to_the_warmup_is_warm(monkeypatch):
    _posts(monkeypatch, _openai(None, finish="length"))
    OpenAICompatible().warm("thinker")               # answered: reachable, key taken, model loaded


def test_a_token_cap_too_large_for_the_model_is_not_a_reason_to_change_the_field(monkeypatch):
    # "max_tokens is too large" is about the value: switching to max_completion_tokens for good
    # broke every later call on a server that does not know that field.
    srv = OpenAICompatible(url="http://localhost:8000/v1")
    sent = _posts(monkeypatch,
                  _Reply(400, {"error": {"message": "max_tokens is too large: 4096. This model supports at most "
                                                    "2048 completion tokens."}}),
                  _openai("fine"))
    ask = model_server.Request(model="m", turns=[{"role": "user", "content": "hi"}], max_tokens=4096)
    with pytest.raises(model_server.Refused):
        srv.send(ask)
    assert srv.send(model_server.Request(model="m", turns=ask.turns, max_tokens=256)) == "fine"
    assert ["max_tokens" in c["json"] for c in sent] == [True, True]


@pytest.mark.parametrize("server", [Ollama, OpenAICompatible])
def test_a_url_without_its_scheme_is_refused_at_the_start(server):
    # "gpu-box:11434" failed every call later, and was taken for this machine.
    with pytest.raises(ValueError, match="http:// or https://"):
        server(url="gpu-box:11434")
    assert server(url="http://gpu-box:11434").off_machine()


def test_the_record_of_a_plan_says_what_it_was_really_held_to(monkeypatch, tmp_path):
    monkeypatch.setattr(vlm, "_server", OpenAICompatible())
    kept = []
    monkeypatch.setattr(flight_recorder, "keep_exchange", lambda tag, **record: kept.append(record))
    _posts(monkeypatch, _Reply(400, {"error": {"message": "response_format json_schema is not supported"}}),
           _openai('{"actions": [], "continue_with": ""}'))
    vlm.plan("do nothing", "a table", _scene())
    assert kept and kept[-1]["extra"]["schema"] == "json"
