"""
model_server.py — where the models run: the one place MRCF sends a model a
question.

vlm.py asks every question (a scene described, a piece labelled, a plan) as
one chat call: a system prompt, turns of text, pictures on the last turn, and
optionally a JSON schema the answer is held to (a Request). A ModelServer
sends that call to wherever the model runs and gives back the answer's text.
Two come with MRCF:

    ollama   Ollama's own API (/api/chat), on this machine or one it can
             reach. The default, and what every rig has run on.
    openai   Any OpenAI-compatible chat API (/v1/chat/completions): a hosted
             one, or a server of your own (vLLM, LM Studio, llama.cpp's
             llama-server, Ollama's /v1).

Any other (a provider's own API, a cloud SDK) is a ModelServer subclass in a
package of its own, named through the "mrcf.model_servers" entry point group;
README.md, "Extending it", has the skeleton.

Which one runs: MRCF_MODEL_SERVER (default "ollama"), at MRCF_MODEL_URL
(default the server's own; OLLAMA_URL still sets Ollama's), with an API key in
MRCF_MODEL_API_KEY. The key is read from the environment only, sent only to
that server, and never printed or recorded. MRCF_MODEL_EXTRA_BODY, a JSON
object, is added to every request: a provider's own settings, such as a
thinking model's reasoning turned down.

A server off this machine is sent the camera's pictures and the scene.
"""

from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass, field, replace
from importlib.metadata import entry_points
from urllib.parse import urlsplit, urlunsplit

import requests

SERVER_ENV = "MRCF_MODEL_SERVER"
URL_ENV = "MRCF_MODEL_URL"
KEY_ENV = "MRCF_MODEL_API_KEY"
EXTRA_ENV = "MRCF_MODEL_EXTRA_BODY"
PLUGIN_GROUP = "mrcf.model_servers"
DEFAULT = "ollama"


# ── What is asked ─────────────────────────────────────────────────────────────

@dataclass
class Request:
    """
    One question for a model, the same for every server.

    turns       — {"role": "user" | "assistant", "content": str}, oldest first;
                  the last is the question (earlier ones are worked examples).
    images      — base64 PNG or JPEG, shown with the last turn, in this order
                  (the prompts call them image 1, image 2, ...).
    json_mode   — the answer must be JSON; schema, if given, the JSON schema it
                  is held to (a server that cannot hold it to one says so with
                  FormatRefused, and is asked again without).
    max_tokens  — a cap on the answer's length; None leaves the server's own.
    timeout_s   — None: the server's own (ModelServer.timeout_s).
    tag         — which call this is, for the log only.
    """
    model: str
    turns: list
    system: "str | None" = None
    images: list = field(default_factory=list)
    json_mode: bool = False
    schema: "dict | None" = None
    max_tokens: "int | None" = None
    temperature: "float | None" = 0.1
    timeout_s: "float | None" = None
    tag: str = ""


# ── What can go wrong ─────────────────────────────────────────────────────────

class ModelServerError(Exception):
    """No answer came. retry: whether asking again may bring one."""
    retry = False
    retry_after: "float | None" = None


class Unreachable(ModelServerError):
    """Nothing answered at the server's address. Not retried: a refused connection stays refused."""


class TimedOut(ModelServerError):
    """The server took longer than the timeout."""
    retry = True


class Busy(ModelServerError):
    """The server said it is overloaded or rate-limited (429, 502-504, 529); retry_after, if it said when."""
    retry = True

    def __init__(self, message: str, retry_after: "float | None" = None):
        super().__init__(message)
        self.retry_after = retry_after


class Refused(ModelServerError):
    """The server refused the request (4xx, or an error it reported): a wrong key, an unknown model, a bad request."""

    def __init__(self, message: str, status: "int | None" = None, detail: str = ""):
        super().__init__(message)
        self.status = status
        self.detail = detail


class FormatRefused(Refused):
    """The server will not hold its answer to the format asked for (the schema, or JSON at all)."""


class CutOff(ModelServerError):
    """The model stopped at its token cap with nothing written: a thinking model spends the cap thinking."""


# ── A server ──────────────────────────────────────────────────────────────────

_FORMAT_WORDS = ("response_format", "json_schema", "json_object", "schema", "grammar", "structured output")
_BUSY = {429, 502, 503, 504, 529}
# What an error says when a parameter is not taken at all (not a value it does not like).
_UNSUPPORTED = ("unsupported", "not supported", "unrecognized", "unrecognised", "unknown", "not permitted",
                "not allowed", "extra inputs", "only the default")


class ModelServer:
    """
    Where the models run. A server implements send(); everything else has a
    default here.

    send(request) -> the answer's text. It raises a ModelServerError when no
    answer came; post() below sends an HTTP request and raises the right one
    from the status code, for a server that speaks HTTP. ask() is what vlm
    calls: send(), asked again without the schema and then without JSON when
    the server refuses that format.
    """
    name = "model server"
    default_url = ""
    unreachable_hint = ""

    def __init__(self, url: "str | None" = None, api_key: "str | None" = None,
                 extra_body: "dict | None" = None, timeout_s: float = 90.0):
        self.url = self.normalise_url((url or self.default_url).strip())
        self._api_key = (api_key or "").strip() or None
        self.extra_body = dict(extra_body or {})
        self.timeout_s = float(timeout_s)
        if self._api_key and _plain_http_off_machine(self.url):
            print(f"  [Models] Warning: the API key goes to {_host(self.url)} unencrypted (http://). "
                  "Use https:// for a server off this machine.")

    # Override as needed ────────────────────────────────────────────────────
    def normalise_url(self, url: str) -> str:
        """The URL requests go to, from what the user gave (a base URL, say)."""
        return url

    def send(self, request: Request) -> str:
        raise NotImplementedError

    def headers(self) -> dict:
        """Every request's HTTP headers: the key as a bearer token, when there is one."""
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def format_refused(self, status: int, detail: str, request: Request) -> bool:
        """Whether an error answer means the server will not use the format asked for (so ask() asks again without)."""
        return (request.json_mode and status in (400, 422)
                and any(w in detail.lower() for w in _FORMAT_WORDS))

    # The rest ──────────────────────────────────────────────────────────────
    _refused: frozenset = frozenset()       # (format, tag) this server would not answer in, found by ask()

    def ask(self, request: Request) -> str:
        """
        send(), held to the schema if the server can, else to plain JSON, else
        to nothing (vlm parses it). A format refused for a kind of call (its
        tag) is remembered once asking without it has brought an answer, and
        left out of that kind's calls from then on: not two requests a call,
        and not a transient error taken for a refusal for good.
        """
        if request.schema is not None and ("schema", request.tag) in self._refused:
            request = replace(request, schema=None)
        if request.json_mode and request.schema is None and ("json", request.tag) in self._refused:
            request = replace(request, json_mode=False)
        dropped: list = []
        while True:
            try:
                answer = self.send(request)
            except FormatRefused as exc:
                if request.schema is not None:
                    print(f"  [Models] {self.name} would not hold the answer to a schema ({exc}) — asking for "
                          "plain JSON.")
                    request = replace(request, schema=None)
                    dropped.append("schema")
                elif request.json_mode:
                    print(f"  [Models] {self.name} would not answer in JSON mode ({exc}) — asking in plain text.")
                    request = replace(request, json_mode=False)
                    dropped.append("json")
                else:
                    raise
                continue
            if dropped:
                self._refused = self._refused | {(what, request.tag) for what in dropped}
            return answer

    def warm(self, model: str) -> None:
        """
        Have the model loaded (and the server's address and key checked) with a
        one-token question. A thinking model spends the one token thinking and
        writes nothing (CutOff): it answered, so it is warm.
        """
        try:
            self.send(Request(model=model, turns=[{"role": "user", "content": "ok"}],
                              max_tokens=1, temperature=0.0, timeout_s=240.0, tag="warmup"))
        except CutOff:
            pass

    def where(self) -> str:
        """'<name> at <url>', the URL without any credentials in it: safe to print."""
        return f"{self.name} at {_redacted(self.url)}"

    def off_machine(self) -> bool:
        """Whether the server is on another machine (it is sent the camera's pictures and the scene)."""
        return not _is_local(_host(self.url))

    def post(self, url: str, body: dict, request: Request) -> requests.Response:
        """POST body as JSON with headers(); the response, or the ModelServerError its failure is."""
        timeout_s = request.timeout_s if request.timeout_s is not None else self.timeout_s
        try:
            resp = requests.post(url, json=_merged(body, self.extra_body), headers=self.headers(),
                                 timeout=timeout_s)
        except requests.exceptions.ConnectionError as exc:
            hint = f" — {self.unreachable_hint}" if self.unreachable_hint else ""
            raise Unreachable(f"Cannot reach {self.where()}{hint}") from exc
        except requests.exceptions.Timeout as exc:
            raise TimedOut(f"Request timed out ({timeout_s:.0f} s)") from exc
        if resp.status_code < 400:
            return resp
        detail = self._redact(_error_text(resp))
        status = resp.status_code
        if status in _BUSY:
            raise Busy(f"{self.name} is busy ({status}): {detail}", _retry_after(resp))
        if status in (401, 403):
            raise Refused(f"{self.name} refused the API key ({status}; {KEY_ENV}): {detail}", status, detail)
        if self.format_refused(status, detail, request):
            raise FormatRefused(f"{status}: {detail}", status, detail)
        raise Refused(f"{self.name} refused the request ({status}): {detail}", status, detail)

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.where()}{' (key set)' if self._api_key else ''}>"


class Ollama(ModelServer):
    """Ollama's /api/chat: the requests every rig has run on, unchanged."""
    name = "Ollama"
    default_url = "http://localhost:11434/api/chat"
    unreachable_hint = "is it running?  (ollama serve)"

    def normalise_url(self, url: str) -> str:
        url = _with_scheme(url).rstrip("/")
        return url if url.endswith("/api/chat") else url + "/api/chat"     # "http://gpu-box:11434" is enough

    def format_refused(self, status: int, detail: str, request: Request) -> bool:
        # An Ollama too old for a schema as the format refuses it with an error of
        # its own; any error that is not about the key, the model or load is taken as that.
        return request.schema is not None and status not in (401, 403, 404, *_BUSY)

    def send(self, request: Request) -> str:
        messages: list = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages += [dict(t) for t in request.turns]
        if request.images:
            messages[-1]["images"] = list(request.images)
        options: dict = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        body: dict = {"model": request.model, "messages": messages, "stream": False, "options": options}
        if request.json_mode:
            body["format"] = request.schema if request.schema is not None else "json"
        # No thinking first, on every call: a thinking model (gemma4) otherwise
        # spends the answer's token cap thinking — held to a schema it returned
        # nothing; a scene description (2026-10-30) came back cut off mid-sentence
        # after 32-46 s. One that cannot think ignores this.
        body["think"] = False
        data = self.post(self.url, body, request).json()
        content = (data.get("message") or {}).get("content") or ""
        if not content and data.get("done_reason") == "length":
            raise CutOff(_cut_off(request))
        return content


class OpenAICompatible(ModelServer):
    """
    Any OpenAI-compatible /chat/completions. MRCF_MODEL_URL is its base URL
    (".../v1"); pictures go as data URLs; a schema as response_format
    json_schema, not strict (MRCF's plan schema is not strict-mode JSON
    schema), so a server that enforces schemas (vLLM, LM Studio, llama.cpp,
    Ollama) holds the answer to it and others take it as a strong hint — the
    answer is checked against the lists afterwards either way.

    Models that refuse max_tokens (they want max_completion_tokens) or any
    temperature but their own are asked again the way they say, and are then
    asked that way from then on.
    """
    name = "OpenAI-compatible server"
    default_url = "https://api.openai.com/v1"
    strict_schema = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens_field = "max_tokens"
        self._no_temperature = False

    def normalise_url(self, url: str) -> str:
        url = _with_scheme(url).rstrip("/")
        return url if url.endswith("/chat/completions") else url + "/chat/completions"

    def send(self, request: Request) -> str:
        messages: list = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        turns = [dict(t) for t in request.turns]
        if request.images:
            parts: list = [{"type": "image_url", "image_url": {"url": f"data:{media_type(b)};base64,{b}"}}
                           for b in request.images]
            turns[-1]["content"] = [*parts, {"type": "text", "text": turns[-1]["content"]}]
        messages += turns
        body: dict = {"model": request.model, "messages": messages, "stream": False}
        if request.json_mode:
            body["response_format"] = (
                {"type": "json_schema",
                 "json_schema": {"name": "answer", "schema": request.schema, "strict": self.strict_schema}}
                if request.schema is not None else {"type": "json_object"})
        data: "dict | None" = None
        while data is None:
            if request.temperature is not None and not self._no_temperature:
                body["temperature"] = request.temperature
            else:
                body.pop("temperature", None)
            body.pop("max_tokens", None)
            body.pop("max_completion_tokens", None)
            if request.max_tokens is not None:
                body[self._tokens_field] = request.max_tokens
            try:
                data = self.post(self.url, body, request).json() or {}
            except FormatRefused:
                raise
            except Refused as exc:
                if not self._adapt(exc.detail.lower()):
                    raise
        if not isinstance(data, dict) or data.get("error"):          # some servers answer an error with 200
            raise Refused(f"{self.name} reported an error: {self._redact(_error_text_of(data))}")
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        if isinstance(content, list):                                # content as parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not content and choice.get("finish_reason") == "length":
            raise CutOff(_cut_off(request))
        return content

    def _adapt(self, detail: str) -> bool:
        """
        Change what is sent the way the server's error asks; False when it asks
        nothing this can change. Only an error that says the parameter itself is
        not taken: "max_tokens is too large" (a value, the context's length) is
        not a reason to send max_completion_tokens from then on, which a server
        that does not know it then refuses on every call.
        """
        if not any(w in detail for w in _UNSUPPORTED):
            return False
        if "max_tokens" in detail and self._tokens_field == "max_tokens":
            self._tokens_field = "max_completion_tokens"
            return True
        if "temperature" in detail and not self._no_temperature:
            self._no_temperature = True
            return True
        return False


# ── Which server ──────────────────────────────────────────────────────────────

SERVERS: dict = {"ollama": Ollama, "openai": OpenAICompatible}


def available() -> list:
    """The servers there are: the built-in ones and each installed plugin's."""
    return sorted(set(SERVERS) | {ep.name for ep in entry_points(group=PLUGIN_GROUP)})


def server_name(name: "str | None" = None) -> str:
    """`name`, else MRCF_MODEL_SERVER, else the default (ollama)."""
    return (name or os.environ.get(SERVER_ENV, "") or DEFAULT).strip().lower()


def server_class(name: str) -> type:
    """The ModelServer class for `name`: a built-in one, else an installed plugin's, loaded on first use."""
    if name not in SERVERS:
        for ep in entry_points(group=PLUGIN_GROUP):
            if ep.name == name:
                SERVERS[name] = ep.load()
                break
    if name not in SERVERS:
        raise KeyError(f"No model server '{name}' ({SERVER_ENV}; known: {', '.join(available())}; a plugin adds "
                       f"one through the {PLUGIN_GROUP!r} entry point group).")
    return SERVERS[name]


def make(name: "str | None" = None, url: "str | None" = None, api_key: "str | None" = None,
         extra_body: "dict | None" = None, **kwargs) -> ModelServer:
    """
    The server to ask: `name` (else MRCF_MODEL_SERVER, else ollama), at `url`
    (else MRCF_MODEL_URL; for Ollama then OLLAMA_URL; else the server's own),
    with `api_key` (else MRCF_MODEL_API_KEY) and `extra_body` added to each
    request (else MRCF_MODEL_EXTRA_BODY, a JSON object).
    """
    name = server_name(name)
    cls = server_class(name)
    if not url:
        url = os.environ.get(URL_ENV, "").strip() or (os.environ.get("OLLAMA_URL", "").strip()
                                                      if name == "ollama" else "") or None
    if api_key is None:
        api_key = os.environ.get(KEY_ENV, "").strip() or None
    if extra_body is None:
        extra_body = _extra_from_env()
    return cls(url=url, api_key=api_key, extra_body=extra_body, **kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def media_type(b64: str) -> str:
    """A base64 image's media type, from its first bytes (PNG unless it says otherwise)."""
    for prefix, kind in (("/9j/", "image/jpeg"), ("iVBOR", "image/png"), ("R0lG", "image/gif"),
                         ("UklG", "image/webp")):
        if b64.startswith(prefix):
            return kind
    return "image/png"


def _merged(body: dict, extra: dict) -> dict:
    """body with extra added: a dict in both is merged one level down (Ollama's "options"), anything else replaced."""
    if not extra:
        return body
    out = dict(body)
    for k, v in extra.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _extra_from_env() -> dict:
    raw = os.environ.get(EXTRA_ENV, "").strip()
    if not raw:
        return {}
    try:
        extra = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{EXTRA_ENV} is not JSON: {exc}") from None
    if not isinstance(extra, dict):
        raise ValueError(f"{EXTRA_ENV} must be a JSON object, like '{{\"reasoning_effort\": \"low\"}}'.")
    return extra


def _cut_off(request: Request) -> str:
    return (f"{request.model} wrote nothing within its {request.max_tokens}-token cap — a thinking model "
            f"spends the cap thinking: turn its thinking off or down ({EXTRA_ENV}, as "
            f"'{{\"reasoning_effort\": \"none\"}}' for gemma4 on Ollama's /v1)")


def _error_text_of(data) -> str:
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err if err is not None else data)


def _error_text(resp) -> str:
    """What an error answer says, short: its error message if it is JSON, else its text."""
    try:
        text = _error_text_of(resp.json())
    except Exception:
        text = getattr(resp, "text", "") or ""
    text = " ".join(text.split())
    return text[:300] + ("…" if len(text) > 300 else "")


def _retry_after(resp) -> "float | None":
    said = (getattr(resp, "headers", None) or {}).get("Retry-After")
    try:
        return float(said) if said is not None else None
    except ValueError:
        return None                     # an HTTP date: our own delay


def _with_scheme(url: str) -> str:
    """
    url, which must say http:// or https://. Without it ("gpu-box:11434")
    every call failed with "No connection adapters were found", and the server
    was taken for one on this machine: no warning that it is sent the camera's
    pictures, or the key unencrypted.
    """
    if urlsplit(url).scheme not in ("http", "https") or "://" not in url:
        raise ValueError(f"The model server's URL must start with http:// or https:// ({URL_ENV}): "
                         f"{_redacted(url) or url!r}")
    return url


def _host(url: str) -> str:
    return urlsplit(url).hostname or ""


def _is_local(host: str) -> bool:
    if host in ("localhost", "") or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _plain_http_off_machine(url: str) -> bool:
    return urlsplit(url).scheme == "http" and not _is_local(_host(url))


def _redacted(url: str) -> str:
    """url without a user, a password or a query (where some APIs put a key)."""
    parts = urlsplit(url)
    netloc = parts.hostname or ""
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
