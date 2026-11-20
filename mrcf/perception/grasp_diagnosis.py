"""
grasp_diagnosis.py — why a close found nothing, asked of a vision model on
the wrist camera's pictures: the jaws open over the object, then shut.

A pick that closed on nothing used to take it for an aim that was off, and
re-aimed: up, a look, down beside where it had been — and the cause was
mostly not the aim. 2026-11-18, the SO-101's table: a wooden figure lying on
its side, the jaws over its 10 mm neck closing 16 mm up, three times; a plug
lying on its back, gripped over its body; each re-aimed and pushed further.
The wrist camera's pictures showed it each time. Judged by the model with its
answer held to a few causes (CAUSES: policies.base.MISS_CAUSES says what the
pick does about each), and the pick changes that (parallel_jaw/pick.py:
_diagnose). Pictures in, an answer out: the model is asked through `ask`
(vlm.call_vlm's signature), the deployment's.

    diagnose(frames, ask, what=None, model=None) -> (cause, seen) | None
"""

from __future__ import annotations

import base64
import io

import numpy as np

# Why a close found nothing: the answers the model is held to.
CAUSES = ("too_high", "aim_off", "lying", "moved", "unclear")

SYSTEM = ("You look at pictures from a camera on a robot gripper, looking down past its two jaws at an object "
          "on a table, and say why the jaws closed on nothing. Answer only from what the pictures show.")

PROMPT = """The first picture: the gripper's two jaws open, lowered round {what}, just before they closed.
The second: the same moment after they shut — they met with nothing between them.

Why did they miss? Choose one:
  too_high  - the object is under or between the jaws in the picture, but they closed above its top
              (it is shorter than where they closed, or lower there)
  aim_off   - the object is beside the jaws, in front of or behind them, not between them
  lying     - the object is lying on its side (a figure, bottle or the like knocked over), not standing
  moved     - the object is not where the jaws came down: pushed away, or not in the pictures
  unclear   - the pictures do not show

Say in a few words what you see."""

SCHEMA = {"type": "object",
          "properties": {"cause": {"type": "string", "enum": list(CAUSES)},
                         "seen": {"type": "string"}},
          "required": ["cause", "seen"]}

PICTURE_PX = 640


def _b64(frame) -> str:
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(np.asarray(frame, np.uint8)))
    if max(img.size) > PICTURE_PX:
        img.thumbnail((PICTURE_PX, PICTURE_PX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def diagnose(frames, ask, what: "str | None" = None, model: "str | None" = None) -> "tuple[str, str] | None":
    """
    Why the close found nothing, from frames [(label, RGB)] — the open jaws
    over it first, shut last — and what it is called (None: "the object"):
    (one of CAUSES, what the model saw), or None when there are no pictures
    or no usable answer. ask: vlm.call_vlm's signature.
    """
    pictures = [f for _, f in frames if f is not None]
    if not pictures:
        return None
    if len(pictures) > 2:
        pictures = [pictures[0], pictures[-1]]
    got = ask(PROMPT.format(what=what or "the object"), [_b64(p) for p in pictures], system=SYSTEM,
              _tag="diagnose", model=model, schema=SCHEMA)
    if not isinstance(got, dict) or got.get("cause") not in CAUSES:
        return None
    return str(got["cause"]), " ".join(str(got.get("seen") or "").split())[:160]


HELD_PROMPT = """The picture is from a camera on a robot gripper, looking down past its two jaws.
Is the gripper carrying an object — held between or by its jaws, lifted off the table — or are its jaws empty,
with whatever is in view lying on the table further away? Say in a few words what you see."""

HELD_SCHEMA = {"type": "object",
               "properties": {"carrying": {"type": "boolean"}, "seen": {"type": "string"}},
               "required": ["carrying", "seen"]}


def held(frame, ask, model: "str | None" = None) -> "tuple[bool, str] | None":
    """
    Whether the wrist camera's picture shows the gripper carrying something:
    (yes or no, what the model saw), or None without a usable answer. For a
    grip the servo cannot feel — a scoop held by its 1 mm wall reads empty,
    and the camera's own check missed it (2026-11-18 17:16, 2026-11-19 10:38),
    the scoop plainly in the picture both times. ask: vlm.call_vlm's signature.
    """
    if frame is None:
        return None
    got = ask(HELD_PROMPT, [_b64(frame)], system=SYSTEM, _tag="held", model=model, schema=HELD_SCHEMA)
    if not isinstance(got, dict) or not isinstance(got.get("carrying"), bool):
        return None
    return bool(got["carrying"]), " ".join(str(got.get("seen") or "").split())[:160]
