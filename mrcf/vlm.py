"""
vlm.py — the vision-language models: client, perception, scene discovery and
the planner.

Every question put to a model. Where it is sent — Ollama, on this machine or
another, or any OpenAI-compatible API — is model_server.py's; the server is
chosen in configure(), else by MRCF_MODEL_SERVER. The world model it reads and writes
is scene_state.py's, the actions it plans with action_registry.py's; both are
re-exported here, so `from vlm import SceneState, register_action, dispatch`
works as it always has.
"""

import hashlib
import json
import time
from typing import Callable

import numpy as np

from mrcf import flight_recorder
from mrcf import model_server as _model_server
from mrcf import scene_state as _scene_state
from mrcf.model_server import ModelServer, ModelServerError, Request
from mrcf.action_registry import (  # noqa: F401 — re-exported
    ACTIONS, CONDITIONS, action_schema, check_postconditions, check_preconditions, context_action_schema,
    dispatch, plan_schema, register_action, register_condition,
)
from mrcf.robot_env import RobotEnv
from mrcf.scene_state import SceneObject, SceneState  # noqa: F401 — re-exported


# ── Module-level VLM / camera configuration ───────────────────────────────────
# Defaults let the module run out of the box against a local Ollama instance
# (or the server MRCF_MODEL_SERVER names). Override at startup via vlm.configure().

_server:           "ModelServer | None" = None   # None: model_server.make() on first use
_model:            str        = "llava"
_planning_model:   str        = "llava"
_observe_cameras:  list[str]  = ["overhead"]  # first entry = primary (used for pixel→world)
_render_w:         int        = 640
_render_h:         int        = 480
_work_plane_z:     float      = 0.0

# Per-tag output caps — prevents runaway generations from wedging the loop.
# Use None to leave the server's default (Ollama's: unbounded) in place.
_NUM_PREDICT: dict[str, int] = {
    "discover_diff":  256,   # tiny JSON diff in the common case
    "discover_scene": 1024,  # full enumeration (legacy / first call fallback)
    "perceive":       512,   # free-text scene description
    "plan":           512,   # one-shot plan JSON
    "decompose":      256,
    "label":          64,    # one short answer per field
    "locate":         96,    # found, and a box: perception/grounding.py
}

# Frame-hash cache: if the rendered pixel bytes are unchanged since the last
# discover_scene call we can skip the VLM call entirely — the ground-truth
# seed already reflects the current state.
_last_discover_hash: str | None = None


def server() -> ModelServer:
    """The model server every call goes to: configure()'s, else the one the environment names (model_server.make)."""
    global _server
    if _server is None:
        _server = _model_server.make()
    return _server


def _primary_camera() -> str:
    """The primary observation camera — used for pixel→world back-projection."""
    return _observe_cameras[0] if _observe_cameras else "overhead"


# ── Image capture ──────────────────────────────────────────────────────────────

def render_to_base64(env: RobotEnv, camera: str | None = None) -> str:
    """Render a single camera and return a base64-encoded PNG string.

    Uses the primary observation camera when camera is not specified.
    """
    return env.render_to_base64(camera=camera or _primary_camera(),
                                width=_render_w, height=_render_h)


def render_all_cameras(env: RobotEnv) -> list[tuple[str, str]]:
    """
    Render every configured observation camera.

    Returns a list of (camera_name, base64_png) pairs, one per camera in
    _observe_cameras.  Cameras that fail to render are silently skipped so
    a single bad camera does not abort the whole perception call.
    """
    results = []
    for cam in _observe_cameras:
        try:
            b64 = env.render_to_base64(camera=cam, width=_render_w, height=_render_h)
            results.append((cam, b64))
        except Exception as exc:
            print(f"  [Render] Camera '{cam}' failed: {exc}")
    return results


# ── Scene perception (image → plain-text description) ──────────────────────────

_PERCEIVE_SYSTEM = (
    "You are a scene perception module for a robot.\n"
    "Look at the camera image(s) and describe the scene concisely.\n"
    "Report what objects are visible, their colours and shapes, and their\n"
    "approximate positions relative to each other.\n"
    "If multiple camera views are provided, combine them into one description.\n"
    "Do NOT plan any actions. Just describe what you see.\n"
    "Respond in plain text (no JSON)."
)


def perceive(env: RobotEnv) -> str:
    """Render all configured observation cameras and return a plain-text scene description.

    This is the perception half of the split mrcf/perception/planning pipeline.
    The returned string is passed to plan() so planning uses no image — only
    the textual description plus the structured world model.

    When multiple cameras are configured, all images are sent in a single VLM
    call so the model can synthesise a unified scene description.

    Returns an empty string on failure (the caller should fall back to
    scene_state.summary()).
    """
    return perceive_from_frames(render_all_cameras(env))


def perceive_from_frames(frames: list[tuple[str, str]]) -> str:
    """
    Describe a scene from already-rendered frames.

    Separate from perceive() only so a caller that already holds frames — or
    one rendering off the main thread — does not have to re-render. perceive()
    is this plus render_all_cameras().
    """
    if not frames:
        print("  [Perceive] No cameras rendered")
        return ""
    images_b64 = [b64 for _, b64 in frames]
    cam_names = [name for name, _ in frames]
    prompt = "Describe the current scene for the robot planner."
    if len(cam_names) > 1:
        labels = ", ".join(f"image {i+1} = {n}" for i, n in enumerate(cam_names))
        prompt = f"Camera views provided: {labels}.\n{prompt}"
    result = call_vlm(
        prompt,
        image_b64=images_b64,
        system=_PERCEIVE_SYSTEM,
        json_mode=False,
        _tag="perceive",
    )
    description = result if isinstance(result, str) else ""
    print(f"  [Perceive] Scene description ({len(description)} chars) → passed to planner")
    return description


def pixel_to_world(env: RobotEnv, cam_name: str, u: float, v: float, z_world: float) -> np.ndarray:
    """Back-project pixel (u, v) to world point on the plane z = z_world."""
    return env.pixel_to_world(u, v, z_world, camera=cam_name,
                              render_w=_render_w, render_h=_render_h)


def warmup(model: str | None = None, planning_model: str | None = None) -> None:
    """
    Send tiny no-op requests to the model server so the VLM and planner models
    are loaded before the first real perception/planning cycle (and a remote
    server's address and key are checked).

    Saves the cold-start stall that otherwise happens on the first
    discover_scene / plan call.  Failures are non-fatal — if the server is
    unreachable we simply print a warning and continue; the real calls
    will surface the same error later anyway.
    """
    vision_model = model or _model
    plan_model   = planning_model or _planning_model
    targets = [("vision", vision_model)]
    if plan_model and plan_model != vision_model:
        targets.append(("planner", plan_model))

    for label, name in targets:
        print(f"  [VLM] Warming up {label} model '{name}'...", flush=True)
        t0 = time.time()
        try:
            server().warm(name)
            print(f"  [VLM] {label} ready in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"  [VLM] {label} warmup failed ({exc}) — continuing anyway")


# ── Asking the model server ────────────────────────────────────────────────────

def _log_vlm(tag: str, system: str | None, prompt: str, n_images: int, raw: str, model: str | None = None,
             seconds: "float | None" = None, images: "list | None" = None, extra: "dict | None" = None) -> None:
    """
    Print a one-line summary of the VLM call: which, what it was shown, how long its answer, and how long it took.
    The whole call — system, prompt, reply, the pictures — goes to the flight recorder (keep_exchange).
    """
    used_model = model or _model
    img_tag = f"{n_images} image{'s' if n_images != 1 else ''}" if n_images else "no image"
    took = f"  {seconds:.1f} s" if seconds is not None else ""
    print(f"  [VLM:{tag}] model={used_model}  {img_tag}  response_len={len(raw)}{took}")
    if seconds is not None:
        flight_recorder.event("vlm", "call", tag=tag, model=used_model, images=n_images, response_len=len(raw),
                              seconds=round(seconds, 2))
    flight_recorder.keep_exchange(tag, model=used_model, system=system, prompt=prompt, reply=raw,
                                  images_b64=images, seconds=seconds, extra=extra)


_RETRY_DELAYS = [1.0, 3.0]      # a timed-out or busy server is asked again after these


def _ask(request: Request, prefix: str = "VLM") -> "str | None":
    """
    The answer's text from the model server, asked again after a timeout or a
    busy answer; None when no answer came (and why is printed). Every call is
    logged and kept whole by the flight recorder (the planner's with the worked
    examples and the schema it was held to).
    """
    prompt = request.turns[-1]["content"]
    images = request.images or None
    extra = {"few_shot": request.turns[:-1], "schema": _held_to(request)} if request.tag == "plan" else None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        sent = time.time()
        try:
            content = server().ask(request)
        except ModelServerError as exc:
            _log_vlm(request.tag, request.system, prompt, len(request.images), "", model=request.model,
                     images=images, extra=extra)
            if exc.retry and attempt < len(_RETRY_DELAYS):
                delay = max(_RETRY_DELAYS[attempt], min(exc.retry_after or 0.0, 30.0))
                print(f"  [{prefix}] {exc} — retrying in {delay:.0f}s ({attempt + 1}/{len(_RETRY_DELAYS)})")
                time.sleep(delay)
                continue
            print(f"  [{prefix}] {exc}" + (" — no answer after all retries." if exc.retry else ""))
            return None
        except Exception as exc:
            _log_vlm(request.tag, request.system, prompt, len(request.images), "", model=request.model,
                     images=images, extra=extra)
            print(f"  [{prefix}] Error: {exc}")
            return None
        if extra is not None:
            extra["schema"] = _held_to(request)      # as answered: the server may have refused the schema
        _log_vlm(request.tag, request.system, prompt, len(request.images), content, model=request.model,
                 images=images, extra=extra, seconds=time.time() - sent)
        return content
    return None


def _held_to(request: Request):
    """What the answer was held to: the schema, "json", or None — less what the server refused for this kind of call."""
    refused = server()._refused
    if request.schema is not None and ("schema", request.tag) not in refused:
        return request.schema
    if request.json_mode and ("json", request.tag) not in refused:
        return "json"
    return None


def _parse_json(text: str):
    """
    The JSON in a model's answer. A server that held it to JSON gives JSON; one
    that could not (model_server.FormatRefused) may wrap it in a ```json fence
    or a sentence, which is cut away. Raises json.JSONDecodeError when there is none.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as first:
        body = text.strip()
        if body.startswith("```"):
            body = body.split("\n", 1)[1] if "\n" in body else ""
            body = body.rsplit("```", 1)[0]
        for open_, close in (("{", "}"), ("[", "]")):
            i, j = body.find(open_), body.rfind(close)
            if 0 <= i < j:
                try:
                    return json.loads(body[i:j + 1])
                except json.JSONDecodeError:
                    pass
        raise first


def call_vlm(
    prompt: str,
    image_b64: "str | list[str] | None" = None,
    system: str | None = None,
    json_mode: bool = True,
    _tag: str = "call",
    model: str | None = None,
    schema: dict | None = None,
):
    """
    Ask the model server (server(): Ollama unless configured otherwise).

    json_mode=True  (default) — constrains output to JSON; returns dict | list or {}.
    json_mode=False            — free-text output; returns str or "".
    _tag                       — label used in log output to identify call site.
    model                      — override model; defaults to MODEL (vision model).
    image_b64                  — single base64 PNG string, or a list of them for
                                 multi-camera input (all images sent in one message).
    schema                     — a JSON schema the reply is held to (json_mode only):
                                 the model can then only produce what it allows.
    """
    images = [] if not image_b64 else ([image_b64] if isinstance(image_b64, str) else list(image_b64))
    request = Request(
        model=model or _model,
        system=system,
        turns=[{"role": "user", "content": prompt}],
        images=images,
        json_mode=json_mode,
        schema=schema if json_mode else None,
        max_tokens=_NUM_PREDICT.get(_tag),
        temperature=0.1,
        tag=_tag,
    )
    content = _ask(request)
    if content is None:
        return {} if json_mode else ""
    if not json_mode:
        return content
    try:
        return _parse_json(content)
    except json.JSONDecodeError:
        print(f"  [VLM] Invalid JSON from model: {content!r}")
        return {}


def label(looks: list, fields: dict, prompt: str, system: str | None = None,
          model: str | None = None) -> dict:
    """
    What an object is, as one answer per field, each from a fixed list:
    fields = {"type": ["pawn", "rook", ..., "not_a_piece"], "colour": ["white", "black"]}.
    The model's reply is held to the lists (a JSON schema), so it can only
    answer with a value on them. Each of `looks` — a base64 image, or a list of
    them shown together — is asked on its own, and a field comes back only if
    every look gave the same answer: {field: value, or None where they differed
    or no answer came}.

    The model names what it is shown. Whether there is an object, and where,
    is the caller's to measure; this is for what it is.
    """
    schema = {"type": "object", "properties": {f: {"enum": list(v)} for f, v in fields.items()},
              "required": list(fields)}
    answers = [call_vlm(prompt, look, system=system, _tag="label", model=model, schema=schema) for look in looks]
    out: dict = {}
    for f, allowed in fields.items():
        said = [a.get(f) if isinstance(a, dict) else None for a in answers]
        out[f] = said[0] if said and said[0] in allowed and all(s == said[0] for s in said) else None
    return out


def label_numbered(image_b64: str, count: int, answers: list, prompt: str, system: str | None = None,
                   model: str | None = None) -> list:
    """
    What each of `count` things numbered 1..count in one image is — a sheet
    of them (perception/tiles.py) — in one call: [one of answers, or None
    where no answer came] in number order. The reply is held to an object
    with one key per number, each one of `answers`, so it can only name every
    one of them, and only with those.

    One call for all of them: a vision model's turn costs seconds whatever it
    is shown, and the answer, not the image, is most of the time — so the
    answers are best short ("white pawn", not a field each).
    """
    keys = [str(i + 1) for i in range(count)]
    schema = {"type": "object", "properties": {k: {"enum": list(answers)} for k in keys}, "required": keys}
    got = call_vlm(prompt, image_b64, system=system, _tag="label_numbered", model=model, schema=schema)
    got = got if isinstance(got, dict) else {}
    return [got.get(k) if got.get(k) in answers else None for k in keys]


def describe_numbered(image_b64: str, count: int, prompt: str, system: str | None = None,
                      model: str | None = None) -> list:
    """
    label_numbered with an answer in words: for each of `count` things
    numbered 1..count in one image, a short string the prompt asks for ("blue
    block P"); [it, or None where none came] in number order. One call for all
    of them, and one string each rather than fields: the answer, not the image,
    is most of the time, and a field's name is written out again for every
    thing (9 blocks: 20 s as fields, a third of the writing as strings).
    """
    keys = [str(i + 1) for i in range(count)]
    schema = {"type": "object", "properties": {k: {"type": "string"} for k in keys}, "required": keys}
    got = call_vlm(prompt, image_b64, system=system, _tag="describe_numbered", model=model, schema=schema)
    got = got if isinstance(got, dict) else {}
    return [got[k].strip() if isinstance(got.get(k), str) and got[k].strip() else None for k in keys]


# ── Scene discovery ────────────────────────────────────────────────────────────

def sim_body_pos(env: RobotEnv, name: str) -> np.ndarray | None:
    """World-frame XYZ of a named body, via env.get_body_pos()."""
    return env.get_body_pos(name)


def _backproject_entry(entry: dict, env: RobotEnv) -> np.ndarray | None:
    """
    Convert a VLM-supplied 'pixel_xy' into a world-frame [x, y, TABLE_Z] position.
    Returns None if pixel_xy is absent or back-projection fails.
    """
    pixel_xy = entry.get("pixel_xy")
    if not pixel_xy or len(pixel_xy) != 2:
        return None
    try:
        u = float(np.clip(float(pixel_xy[0]), 0, _render_w - 1))
        v = float(np.clip(float(pixel_xy[1]), 0, _render_h - 1))
        return pixel_to_world(env, _primary_camera(), u, v, _work_plane_z)
    except Exception as exc:
        print(f"  [Observe] Back-projection failed for '{entry.get('name')}': {exc}")
        return None


def _finalise_discover(scene_state: "SceneState", held_names: set[str]) -> None:
    """
    Post-process a populated scene_state: restore carried-object markers and
    print a summary line.  Shared between the VLM-diff path and the
    frame-hash-skip early return.
    """
    obj_names = [o.name for o in scene_state.objects]
    tgt_names = [t.name for t in scene_state.targets]
    print(f"  [Discover] objects={len(obj_names)}  targets={len(tgt_names)}")

    # Restore "held" state for any object that was being carried before
    # the rescan.  Without this a successful pick would immediately lose
    # its held_object() marker on the next auto-observe step.
    for obj in scene_state.objects:
        if obj.name in held_names:
            obj.position = None


def discover_scene(
    env: RobotEnv,
    scene_state: SceneState,
    exclude_keywords: "set[str] | None" = None,
):
    """
    Populate scene_state with the current belief state, then ask the VLM
    for a *diff* — only objects that have moved, disappeared, or appeared
    relative to that belief.

    Philosophy
    ----------
    Enumerating every object from scratch in every perception cycle is both
    slow and wasteful: the VLM has to generate one JSON entry per object
    (hundreds of tokens on a chess board) and the result is almost always
    identical to the previous scan.  Instead we:

        1. Seed belief from env ground truth (sim: exact positions;
           real robot: None when unavailable).
        2. Show the VLM the current image + a compact summary of that belief.
        3. Ask the VLM for a short JSON diff: missing / new / moved.
        4. Apply the diff on top of the seeded belief.

    On the first call the "diff" is effectively the whole world (nothing in
    prior belief), but even then the ground-truth seed keeps the scene
    populated so skills can run without waiting on the VLM.

    Carried-object preservation: objects already marked as carried
    (position=None in scene_state) are restored after the rescan so a
    successful pick does not lose its held_object() marker.
    """
    held_names = {o.name for o in scene_state.objects if o.position is None}
    prev_objects = {o.name: o for o in scene_state.objects}
    prev_targets = {t.name: t for t in scene_state.targets}

    free_bodies   = env.get_movable_body_names()
    static_bodies = env.get_static_body_names(exclude_keywords=exclude_keywords)

    # ── 1. Seed belief from env ground truth ──────────────────────────────
    # For sim this is exact; for real robots it's whatever the backend
    # tracker provides (position may be None).  Descriptions are carried
    # over from the previous belief so we don't lose VLM annotations.
    scene_state.objects = [
        SceneObject(
            name,
            (prev_objects[name].description if name in prev_objects else name),
            interactable=True,
            position=sim_body_pos(env, name),
            pixel_xy=(prev_objects[name].pixel_xy if name in prev_objects else None),
        )
        for name in free_bodies
    ]
    scene_state.targets = [
        SceneObject(
            name,
            (prev_targets[name].description if name in prev_targets else name),
            interactable=False,
            position=sim_body_pos(env, name),
            pixel_xy=(prev_targets[name].pixel_xy if name in prev_targets else None),
        )
        for name in static_bodies
    ]

    # ── 2. Ask the VLM for a diff against current belief ──────────────────
    frames = render_all_cameras(env)
    if not frames:
        print("  [Observe] No cameras rendered; using ground-truth positions only.")
        result: dict = {}
    else:
        images_b64 = [b64 for _, b64 in frames]
        cam_names  = [name for name, _ in frames]

        # Frame-hash skip: if the pixels are identical to the previous diff
        # call AND we already had a populated belief, the VLM can't tell us
        # anything new. The ground-truth reseed above is already current.
        global _last_discover_hash
        frame_hash = hashlib.md5(
            b"".join(b64.encode("ascii") for b64 in images_b64)
        ).hexdigest()
        if frame_hash == _last_discover_hash and prev_objects:
            print(f"  [Discover] frame unchanged — skipping VLM (objects={len(scene_state.objects)})")
            _finalise_discover(scene_state, held_names)
            return
        _last_discover_hash = frame_hash

        belief_lines = []
        for o in scene_state.objects:
            tag = o.description if o.description and o.description != o.name else ""
            belief_lines.append(f"    - {o.name}" + (f"  ({tag})" if tag else ""))

        cam_label = (
            f"Camera views: {', '.join(f'image {i+1}={n}' for i, n in enumerate(cam_names))}.\n"
            if len(cam_names) > 1
            else f"You are inspecting a robot scene from camera '{cam_names[0]}'.\n"
        )
        primary_cam_note = (
            f"The FIRST image is the primary camera ('{_primary_camera()}'). "
            f"pixel_xy coordinates must come from that image only.\n"
            if len(cam_names) > 1
            else ""
        )

        prompt = (
            cam_label + primary_cam_note + "\n"
            + "Current belief state of interactable objects:\n"
            + ("\n".join(belief_lines) if belief_lines else "    (none)")
            + "\n\n"
            + "Compare the image(s) to the belief above and report ONLY differences:\n"
            + "  - 'missing': names of belief objects that are NOT visible or have clearly\n"
            + "               been removed.\n"
            + "  - 'moved':   belief objects that are visible but at a new pixel location,\n"
            + "               with the new pixel_xy.\n"
            + "  - 'new':     extra objects visible in the image that are NOT in the belief.\n\n"
            + "Use EXACT body names from the belief list for 'missing' and 'moved'.\n"
            + f"pixel_xy is [x, y] in a {_render_w}x{_render_h} image, origin top-left.\n\n"
            + "If nothing has changed, return empty arrays. Respond JSON only:\n"
            + '{"missing": [], "moved": [{"name":"<exact name>","pixel_xy":[x,y]}], '
            + '"new": [{"name":"<exact name>","description":"<short>","pixel_xy":[x,y]}]}'
            + (f"\n\n{_discover_extra_system}" if _discover_extra_system else "")
        )

        result = call_vlm(prompt, images_b64, _tag="discover_diff")
        if not isinstance(result, dict):
            print(f"  [Discover] VLM returned non-dict: {result!r}")
            result = {}

        valid_bodies: set[str] = set(free_bodies + static_bodies)
        # A backend with no body registry at all — real hardware, where there's
        # no simulator declaring named bodies and no external tracker — can't
        # validate names: the VLM is the ONLY source of object identity, so every
        # name it reports is by definition absent from the (empty) registry.
        # Only enforce the whitelist when the backend actually has one, otherwise
        # discover_scene() can never discover anything.
        enforce_valid_bodies = bool(valid_bodies)

        by_name = {o.name: o for o in scene_state.objects}

        # Apply VLM diff, BUT only trust the VLM's positions when the
        # backend has no ground truth of its own.  For sim, env.get_body_pos
        # returned a valid seed above, so we ignore VLM position overrides
        # for entries we already have good coordinates for — the VLM's
        # pixel_xy is less accurate than the simulator's qpos.  For real
        # robots with no tracker, the VLM IS the source of truth.
        #
        # For "missing": if ground truth says the body still exists, the
        # VLM is wrong (probably occluded) — keep the seed.  If ground truth
        # has no opinion (real robot, no tracker), drop the object from the
        # scene entirely.  Setting position=None would mis-mark it as held.
        removed: set[str] = set()
        for name in result.get("missing", []) or []:
            if not isinstance(name, str) or name in held_names:
                continue
            obj = by_name.get(name)
            if obj is None:
                continue
            if sim_body_pos(env, name) is None:
                removed.add(name)
        if removed:
            scene_state.objects = [o for o in scene_state.objects if o.name not in removed]
            by_name = {o.name: o for o in scene_state.objects}

        for entry in result.get("moved", []) or []:
            if not isinstance(entry, dict):
                continue
            obj = by_name.get(entry.get("name"))
            if obj is None:
                continue
            if obj.position is not None:
                # Ground truth already set — keep it, just refresh pixel_xy
                # for any downstream consumers that want it for overlays.
                obj.pixel_xy = entry.get("pixel_xy", obj.pixel_xy)
                continue
            new_pos = _backproject_entry(entry, env)
            if new_pos is not None:
                obj.position = new_pos
                obj.pixel_xy = entry.get("pixel_xy")

        for entry in result.get("new", []) or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or name in by_name:
                continue
            if enforce_valid_bodies and name not in valid_bodies:
                continue
            # Drop background clutter the VLM names because it is visually
            # salient — the robot itself, furniture, the floor. On a backend
            # with a body registry these could never appear; with the VLM as
            # the only source of identity, this keyword filter is the only
            # thing standing between "what is in frame" and "what is a task
            # object". Without it a plan can end up targeting the robot.
            if exclude_keywords:
                haystack = f"{name} {entry.get('description', '')}".lower()
                if any(k.lower() in haystack for k in exclude_keywords):
                    print(f"  [Discover] ignoring '{name}' (matches exclude_keywords)")
                    continue
            gt_pos = sim_body_pos(env, name)
            obj = SceneObject(
                name,
                entry.get("description", name),
                interactable=True,
                position=gt_pos if gt_pos is not None else _backproject_entry(entry, env),
                pixel_xy=entry.get("pixel_xy"),
            )
            scene_state.objects.append(obj)
            by_name[name] = obj          # the model can list one object twice

    _finalise_discover(scene_state, held_names)


# ── Action planning ────────────────────────────────────────────────────────────

_PLAN_SYSTEM_BASE = (
    "You are a robot planner and general-purpose problem solver.\n"
    "Respond with ONLY valid JSON: {\"actions\": [...], \"continue_with\": \"\"}\n"
    "Each action: {\"action\": \"<name>\", \"parameters\": {...}}\n"
    "Leave continue_with empty unless an observe step is genuinely needed mid-task.\n\n"
    "RULES:\n"
    "  • Resolve colour/shape references to the best-matching body name in the scene summary.\n"
    "    The scene summary always contains the canonical names — use them directly, never ask.\n"
    "  • Objects marked [OUT OF REACH]: respond with 'answer' explaining they are unreachable.\n"
    "  • If the destination of a move is unknown, call observe first to locate it.\n"
    "  • For math → calculate. For visual/colour/count questions → answer. For status/list → single action.\n"
    "  • Only use actions that appear in the Available actions list.\n"
)
_PLAN_SYSTEM = _PLAN_SYSTEM_BASE

# Generic few-shot examples — robot-agnostic demonstrations.
# Robot-specific examples are injected at startup via vlm.configure(plan_few_shot=[...]).
_PLAN_FEW_SHOT: list[dict] = [
    # math
    {"role": "user",      "content": "User command: \"what is 2**32\""},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"calculate\",\"parameters\":{\"expression\":\"2**32\"}}],\"continue_with\":\"\"}"},
    # visual question
    {"role": "user",      "content": "User command: \"what colour is the cube\""},
    {"role": "assistant", "content": "{\"actions\":[{\"action\":\"answer\",\"parameters\":{\"question\":\"what colour is the cube\"}}],\"continue_with\":\"\"}"},
]

_DECOMPOSE_SYSTEM = (
    "You are a high-level task planner for a robot.\n"
    "Given a user goal, decompose it into an ordered list of concrete sub-goals.\n"
    "Each sub-goal should be a short English phrase that maps directly to one or\n"
    "two robot skills.\n"
    "Always preserve the EXACT destination the user specified.\n"
    "If the goal is already atomic — a single action, math question, etc. —\n"
    "return a list with just that one goal unchanged.\n"
    'Respond with JSON only: {"sub_goals": ["sub-goal 1", "sub-goal 2", ...]}'
)


# ── Module-level planner configuration ────────────────────────────────────────
# Set by the deployment's entry-point via vlm.configure().
# Defaults are empty — the framework adds no robot-specific prompt content.

_plan_extra_few_shot:    list[dict] = []
_decompose_extra_system: str        = ""
# Domain guidance injected into discover_scene's prompt. Without it the VLM
# names whatever is most visually salient in the frame — on a real rig that is
# usually the robot, the furniture and the floor, not the small task objects.
_discover_extra_system:  str        = ""
_plan_state_fn:          "Callable | None" = None
# _plan_state_fn(scene_state) → str | None.  Injected by the deployment to add
# robot-specific state context (e.g. gripper status, battery level) to each planning prompt.


def configure(
    plan_few_shot:          list[dict] | None = None,
    decompose_extra_system: str | None        = None,
    discover_extra_system:  str | None        = None,
    plan_state_fn:          "Callable | None" = None,
    proximity_on_m:         float | None      = None,
    proximity_near_m:       float | None      = None,
    model_server:           "ModelServer | str | None" = None,
    model_url:              str | None        = None,
    ollama_url:             str | None        = None,
    model:                  str | None        = None,
    planning_model:         str | None        = None,
    observe_cameras:        list[str] | None  = None,
    observe_camera:         str | None        = None,  # backward-compat alias for single camera
    render_w:               int | None        = None,
    render_h:               int | None        = None,
    work_plane_z:           float | None      = None,
    safety_rules_text:      str | None        = None,
) -> None:
    """
    Set robot-specific VLM and planner configuration.  Call once at startup.

    plan_few_shot          — list of {"role": ..., "content": ...} dicts injected
                             as conversation turns between the system prompt and the
                             live user prompt.  Use to teach the planner robot-specific
                             action names and planning patterns via demonstration.
    decompose_extra_system — additional text appended to _DECOMPOSE_SYSTEM to guide
                             the task decomposer with robot-specific skill vocabulary
                             (e.g. pick_place vs pick_stack distinctions).
    plan_state_fn          — callable(scene_state) → str | None.  Called before
                             each planning prompt to inject robot-specific state
                             context (e.g. "Gripper: empty" for a manipulation arm,
                             "Battery: 82%" for a mobile robot).  Return None to
                             omit the state section entirely.  If not set, no state
                             context is added — the framework stays robot-agnostic.
    proximity_on_m         — XY distance threshold (metres) below which an object
                             is annotated "[on <target>]" in SceneState.summary().
                             Default 0.015 m.  Set to 0 to disable.
    proximity_near_m       — XY distance threshold (metres) below which an object
                             is annotated "[near <target>, X cm]".  Default 0.06 m.
                             Set to 0 to disable.
    model_server           — where every call goes: a model_server.ModelServer, or the
                             name of one ("ollama", "openai", a plugin's), made with
                             model_url and the key in MRCF_MODEL_API_KEY. Not given:
                             MRCF_MODEL_SERVER's, else Ollama on this machine.
    model_url              — that server's URL (an Ollama host, an OpenAI-compatible
                             base URL ".../v1"); None: MRCF_MODEL_URL, else its own.
    ollama_url             — the URL of an Ollama server to use (before model_server
                             there was only Ollama): an Ollama server at this URL.
    model                  — vision model name (used for perceive / discover_scene).
    planning_model         — text model name (used for plan / decompose_task).
    observe_cameras        — ordered list of camera names to render for perception.
                             The FIRST entry is the primary camera used for pixel→world
                             back-projection.  Additional cameras provide extra context.
                             e.g. ["overhead", "wrist", "side"]
    observe_camera         — convenience alias: sets observe_cameras to [camera_name].
                             If both observe_cameras and observe_camera are given,
                             observe_cameras takes precedence.
    render_w / render_h    — image resolution for renders sent to the VLM.
    work_plane_z           — world-frame Z of the work plane for pixel back-projection.
    """
    global _plan_extra_few_shot, _decompose_extra_system, _discover_extra_system, _plan_state_fn
    global _server, _model, _planning_model
    global _observe_cameras, _render_w, _render_h, _work_plane_z
    global _PLAN_SYSTEM
    if plan_few_shot is not None:
        _plan_extra_few_shot = list(plan_few_shot)
    if decompose_extra_system is not None:
        _decompose_extra_system = decompose_extra_system
    if discover_extra_system is not None:
        _discover_extra_system = discover_extra_system
    if plan_state_fn is not None:
        _plan_state_fn = plan_state_fn
    if proximity_on_m is not None:
        _scene_state.PROXIMITY_ON_M = proximity_on_m
    if proximity_near_m is not None:
        _scene_state.PROXIMITY_NEAR_M = proximity_near_m
    if isinstance(model_server, ModelServer):
        _server = model_server
    elif model_server is not None or model_url is not None or ollama_url is not None:
        # ollama_url alone is an Ollama server there, as it was before there were others.
        name = _model_server.server_name(model_server or ("ollama" if ollama_url and not model_url else None))
        _server = _model_server.make(name, url=model_url or (ollama_url if name == "ollama" else None))
    if model_server is not None or model_url is not None or ollama_url is not None:
        srv = server()
        off = " — off this machine: it is sent the camera's pictures and the scene" if srv.off_machine() else ""
        print(f"  [VLM] Models served by {srv.where()}{off}")
    if model is not None:
        _model = model
    if planning_model is not None:
        _planning_model = planning_model
    if observe_cameras is not None:
        _observe_cameras = list(observe_cameras)
    elif observe_camera is not None:
        _observe_cameras = [observe_camera]
    if render_w is not None:
        _render_w = render_w
    if render_h is not None:
        _render_h = render_h
    if work_plane_z is not None:
        _work_plane_z = work_plane_z
    if safety_rules_text is not None:
        _PLAN_SYSTEM = _PLAN_SYSTEM_BASE + "\n" + safety_rules_text.rstrip() + "\n"


def decompose_task(user_command: str, object_names: "list[str] | None" = None) -> list[str]:
    """Level-1 planner: decompose a user goal into ordered sub-goals.

    Makes a fast text-only LLM call.  Returns a list of sub-goal strings.
    Falls back to [user_command] on any error so the caller always gets at
    least one item to plan against.

    object_names: what is in the scene, by the names the planner will see.
    Without them a goal about several objects comes back about ones it made
    up ("put block 1 on block 2"), which the planner then cannot find.
    """
    system = _DECOMPOSE_SYSTEM
    if _decompose_extra_system:
        system = system + "\n" + _decompose_extra_system
    known = ""
    if object_names:
        known = ("\nObjects in the scene: " + ", ".join(object_names) + "\n"
                 "Name objects in sub-goals only by these exact names; never by a number or a name of your own.")
    result = call_vlm(
        f'User goal: "{user_command}"{known}\n\nDecompose into sub-goals.',
        image_b64=None,
        system=system,
        _tag="decompose",
        model=_planning_model,
    )
    if isinstance(result, dict):
        sub_goals = result.get("sub_goals", [])
        if isinstance(sub_goals, list) and sub_goals:
            sub_goals = [str(g) for g in sub_goals]
            print(f"  [Decompose] '{user_command}' → {sub_goals}")
            return sub_goals
    print("  [Decompose] No valid sub_goals in response — using original command unchanged")
    return [user_command]


def plan(
    user_command: str,
    scene_description: str,
    scene_state: SceneState,
    history: list | None = None,
    failure_context: str | None = None,
) -> tuple[list, str | None]:
    """Level-2 planner: translate a sub-goal into an action sequence.

    Text-only — no image attached.  Perception is handled separately by
    perceive() which produces the scene_description string passed here.

    history       — optional list of recent {"command": str, "outcome": str} dicts
                    (oldest first) for follow-up context.
    failure_context — if the previous attempt at this command failed: why the
                      step failed (or was refused) and what the camera sees now,
                      put in the prompt as given, so the planner can choose a
                      recovery (e.g. home) or another way.

    Returns (actions, continue_with).
    """
    state_line = None
    if _plan_state_fn is not None:
        try:
            state_line = _plan_state_fn(scene_state)
        except Exception:
            pass

    history_section = ""
    if history:
        lines = "\n".join(
            f'  [{i + 1}] "{h["command"]}" → {h["outcome"]}'
            for i, h in enumerate(history)
        )
        history_section = f"Recent command history (oldest first):\n{lines}\n\n"

    failure_section = ""
    if failure_context:
        failure_section = (
            f"RECOVERY MODE — previous attempt at this command failed.\n"
            f"{failure_context}\n"
            f"If a step was refused without touching anything (no room, out of reach, something in the way), asking "
            f"for it again will be refused the same way unless the plan first changes what the refusal names. "
            f"Out of reach: home first (reach depends on where the arm starts). Another object leaves no room: "
            f"move that object first, or put the held object somewhere else; going home or looking again does "
            f"not make room. Otherwise plan another way to do what the user asked, or no actions at all so the "
            f"user is asked. If the step failed instead, plan recovery steps first (e.g. home to reset arm "
            f"orientation) then retry the command.\n\n"
        )

    prompt = (
        f"Scene description (from camera):\n{scene_description}\n\n"
        f"Known objects and positions:\n{scene_state.summary()}\n\n"
        + (f"Robot state: {state_line}\n\n" if state_line else "")
        + f"{history_section}"
        f"{failure_section}"
        f"Available actions:\n{json.dumps(context_action_schema(scene_state), indent=2)}\n\n"
        f"User command: \"{user_command}\"\n\n"
        "Return: {\"actions\": [...], \"continue_with\": \"<next step or empty>\"}."
    )

    request = Request(
        model=_planning_model,
        system=_PLAN_SYSTEM,
        turns=[*_PLAN_FEW_SHOT, *_plan_extra_few_shot, {"role": "user", "content": prompt}],
        json_mode=True,
        # Held to the scene: only registered actions, only names the scene has.
        # A server that cannot hold it to a schema is asked for plain JSON, and
        # the checks below drop what is not registered.
        schema=plan_schema(scene_state),
        max_tokens=_NUM_PREDICT.get("plan"),
        temperature=0.1,
        tag="plan",
    )
    result: dict | list = {}
    content = _ask(request, prefix="Plan")
    if content is not None:
        try:
            result = _parse_json(content)
        except json.JSONDecodeError as exc:
            print(f"  [Plan] Invalid JSON from model: {exc}")

    if not isinstance(result, dict):
        actions_out: list = []
        continue_with: str | None = None
    else:
        actions = result.get("actions")
        if isinstance(actions, list):
            actions_out = actions
        else:
            single = result.get("action")
            actions_out = [single] if isinstance(single, dict) else []

        raw_followup = result.get("continue_with", "")
        continue_with = (
            raw_followup.strip()
            if isinstance(raw_followup, str) and raw_followup.strip()
            else None
        )

    # Normalise bare action-name strings to proper dicts before validation.
    # Some models return ["answer"] instead of [{"action": "answer", ...}].
    actions_out = [
        {"action": a, "parameters": {}} if isinstance(a, str) else a
        for a in actions_out
    ]

    # Drop hallucinated action names not in the registry.
    if actions_out:
        valid   = [a for a in actions_out if isinstance(a, dict) and a.get("action") in ACTIONS]
        dropped = [a.get("action") if isinstance(a, dict) else a
                   for a in actions_out
                   if not isinstance(a, dict) or a.get("action") not in ACTIONS]
        if dropped:
            print(f"  [Plan] Ignoring unrecognised actions: {dropped}")
        actions_out = valid

    print(f"  [Plan] → actions={json.dumps(actions_out)}  continue_with={continue_with!r}")

    return actions_out, continue_with


def plan_actions(
    user_command: str,
    env: RobotEnv,
    scene_state: SceneState,
    history: list | None = None,
) -> tuple[list, str | None]:
    """Convenience wrapper: perceive then plan.

    Renders the scene with the VLM (perceive), then makes a separate text-only
    planning call (plan).
    """
    scene_description = perceive(env)
    if not scene_description:
        scene_description = scene_state.summary()
    return plan(user_command, scene_description, scene_state, history)
