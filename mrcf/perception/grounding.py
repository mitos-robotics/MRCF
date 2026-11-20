"""
grounding.py — find a thing by its name in one picture: a vision model draws a
box round it, and the segmenter outlines what is inside the box.

The surface scan (surface_scan.py) finds objects by outlining everything the
size of an object; what it cannot find that way it never names. Some things
are not its kind of object: a tub, a tray, a basket — bigger than an object,
thin-walled, often see-through, the table showing through them. They are
found here the other way round, by what someone calls them ("the plastic
tub"): the vision model is asked where that is, and answers with a box held
to a schema (found or not, and four numbers), so it cannot ramble or
invent a format. The box only says WHICH thing; its outline is the
segmenter's (segment.segment_box), which follows the thing's own edges.

How a model writes a box is how it was trained to, not something a prompt
changes reliably: Qwen2.5-VL writes [x1, y1, x2, y2] in the picture's pixels,
Gemma writes [y1, x1, y2, x2] scaled to 0-1000. Each family is asked in its
own convention (CONVENTIONS, convention_for), and the answer read back into
pixels. A box that is not inside the picture, or has no area, is no answer.

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np

# How a model family writes a box: the key it answers under, the prompt's
# words for it, and how to read it back as (x1, y1, x2, y2) pixels of a w x h
# picture. 2026-11-17, the SO-101's overhead camera, 1280 x 720: qwen2.5vl:7b
# boxed the clear plastic tub at [167, 38, 515, 567] (by eye 170-510, 40-555)
# in 0.5-3.5 s; gemma4:31b at [42, 131, 793, 402], the same box in its own
# convention; both said found: false for a thing that was not there.
CONVENTIONS: "dict[str, dict[str, Any]]" = {
    "pixels_xyxy": {
        "key": "bbox_2d",
        "words": "as bbox_2d [x1, y1, x2, y2] in pixels of this {w}x{h} picture",
        "read": lambda b, w, h: (b[0], b[1], b[2], b[3]),
    },
    "norm1000_yxyx": {
        "key": "box_2d",
        "words": "as box_2d [y_min, x_min, y_max, x_max], each scaled to 0-1000 of the picture's height or width",
        "read": lambda b, w, h: (b[1] * w / 1000.0, b[0] * h / 1000.0, b[3] * w / 1000.0, b[2] * h / 1000.0),
    },
}

# A box smaller than this on a side (px) is not a box round anything.
MIN_BOX_PX = 4
# The segmenter's outline is the thing only if it fills at least this share of
# the box (a box round a tub, outlined as a speck inside it, is not the tub)
# and stays inside it, to this share of its own area.
MIN_BOX_FILL = 0.2
MIN_INSIDE_BOX = 0.85
# ... and the segmenter's own estimate of it is at least this.
MIN_OUTLINE_SCORE = 0.7


def convention_for(model: str) -> str:
    """The box convention a model was trained to (CONVENTIONS): by its family, from its name."""
    name = (model or "").lower()
    return "norm1000_yxyx" if "gemma" in name or "paligemma" in name else "pixels_xyxy"


@dataclass
class Found:
    """A thing found by name in one picture."""
    box: tuple                       # (x1, y1, x2, y2), pixels: the vision model's
    mask: "np.ndarray | None"        # its outline, bool (h, w): the segmenter's, inside the box
    score: float                     # the segmenter's estimate of the outline (0 when there is none)


def _b64(frame_rgb: np.ndarray) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("the picture could not be encoded")
    return base64.b64encode(buf.tobytes()).decode()


def locate(frame_rgb: np.ndarray, what: str, *, model: str, ask, convention: "str | None" = None
           ) -> "tuple[tuple | None, str]":
    """
    Where `what` is in the picture, as the vision model boxes it: ((x1, y1,
    x2, y2) pixels, "") or (None, why not). ask: how a vision model is asked,
    vlm.call_vlm's signature — the caller's (perception does not reach the
    planner's modules).
    """
    h, w = frame_rgb.shape[:2]
    conv = CONVENTIONS[convention or convention_for(model)]
    key = conv["key"]
    schema = {"type": "object",
              "properties": {"found": {"type": "boolean"},
                             key: {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}},
              "required": ["found", key]}
    prompt = (f"Find {what} in this picture. If it is there, set found true and give the box round all of it "
              f"{conv['words'].format(w=w, h=h)}. If it is not in the picture, set found false and {key} "
              "[0, 0, 0, 0].")
    reply = ask(prompt, _b64(frame_rgb), _tag="locate", model=model, schema=schema)
    if not isinstance(reply, dict) or not reply:
        return None, "the vision model gave no answer"
    if not reply.get("found"):
        return None, f"the vision model does not see {what}"
    raw = reply.get(key)
    if not (isinstance(raw, list) and len(raw) == 4 and all(isinstance(v, (int, float)) for v in raw)):
        return None, f"the vision model's box is not four numbers ({raw!r})"
    x1, y1, x2, y2 = conv["read"]([float(v) for v in raw], w, h)
    x1, x2 = sorted((max(0.0, min(float(w), x1)), max(0.0, min(float(w), x2))))
    y1, y2 = sorted((max(0.0, min(float(h), y1)), max(0.0, min(float(h), y2))))
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None, f"the vision model's box for {what} has no area inside the picture ({raw!r})"
    return (x1, y1, x2, y2), ""


# Whether a thing is one to put things into: the model's judgement, held to a
# yes or no — not a list of container words — on a picture of it alone (the
# box, a little more round it). 2026-11-17, crops from the SO-101's overhead
# camera: gemma4:31b said the tub and the scoop are, the plug and a wooden
# figure are not; qwen2.5vl:7b called the figure one too. Asked with the box in
# one question, qwen2.5vl:7b boxed the tub when asked for "the red figure".
CONTAINER_QUESTION = ("Is the thing in this picture something other things can be put into from above — a tub, a "
                      "bowl, a box, a basket, a tray — rather than an object to be put somewhere (a ball, a bottle, a "
                      "toy, a tool, a plug)?")
# Asked of two crops — the box and a little round it, and with more of what is
# round it — and a container only when both say so: 2026-11-17, gemma4:31b
# called a wooden figure's tight crop a container once, and not another time.
CROP_PADS = (0.1, 0.6)


def is_container(frame_rgb: np.ndarray, box, *, model: str, ask) -> "bool | None":
    """
    Whether the thing in box is a container, as the vision model judges crops
    of it (CROP_PADS): True only if every crop says so; None when no answer
    came (ask: locate's).
    """
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    said = []
    for pad in CROP_PADS:
        px, py = pad * (x2 - x1), pad * (y2 - y1)
        crop = frame_rgb[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]
        if crop.size == 0:
            return None
        reply = ask(CONTAINER_QUESTION, _b64(crop), _tag="label", model=model,
                    schema={"type": "object", "properties": {"container": {"type": "boolean"}},
                            "required": ["container"]})
        if not isinstance(reply, dict) or not isinstance(reply.get("container"), bool):
            return None
        said.append(bool(reply["container"]))
        if not said[-1]:
            return False
    return all(said)


def outline(frame_rgb: np.ndarray, box, segment_box=None) -> "tuple[np.ndarray | None, float, str]":
    """
    The thing inside box, as the segmenter outlines it: (mask, score, "") or
    (None, 0, why not). Of its candidates, the best-scored that fills enough
    of the box (MIN_BOX_FILL) and stays inside it (MIN_INSIDE_BOX).
    segment_box: segment.segment_box's signature (the default), for tests.
    """
    if segment_box is None:
        from mrcf.perception.segment import segment_box
    x1, y1, x2, y2 = (float(v) for v in box)
    masks, scores = segment_box(frame_rgb, (x1, y1, x2, y2))
    h, w = frame_rgb.shape[:2]
    inside = np.zeros((h, w), bool)
    pad = 0.03 * max(x2 - x1, y2 - y1)                 # the box is the model's guess: a little slack
    inside[int(max(0, y1 - pad)):int(min(h, y2 + pad + 1)), int(max(0, x1 - pad)):int(min(w, x2 + pad + 1))] = True
    area_box = max(1.0, (x2 - x1) * (y2 - y1))
    best, best_score = None, 0.0
    for m, sc in zip(masks, scores):
        m = np.asarray(m, bool)
        area = float(m.sum())
        if area <= 0 or area < MIN_BOX_FILL * area_box or float((m & inside).sum()) < MIN_INSIDE_BOX * area:
            continue
        if sc > best_score:
            best, best_score = m, float(sc)
    if best is None:
        return None, 0.0, "the segmenter outlined nothing that fills the box"
    if best_score < MIN_OUTLINE_SCORE:
        return None, best_score, f"the segmenter's outline is unsure ({best_score:.2f})"
    return best, best_score, ""


def find(frame_rgb: np.ndarray, what: str, *, model: str, ask, convention: "str | None" = None,
         segment_box=None, judge_model: "str | None" = None) -> "tuple[Found | None, str]":
    """
    locate, then outline: (Found, "") or (None, why not). judge_model: only a
    container, as that model judges it (is_container) — no answer is a no.
    ask: locate's.
    """
    box, why = locate(frame_rgb, what, model=model, convention=convention, ask=ask)
    if box is None:
        return None, why
    if judge_model is not None and not is_container(frame_rgb, box, model=judge_model, ask=ask):
        return None, f"{what} is there, but the vision model does not take it for something to put things into"
    mask, score, why = outline(frame_rgb, box, segment_box=segment_box)
    if mask is None:
        return None, why
    return Found(box, mask, score), ""


def find_near(frame_rgb: np.ndarray, what: str, near_px, margin_px: float, *, model: str, ask,
              convention: "str | None" = None, segment_box=None) -> "tuple[Found | None, str]":
    """
    find, for a thing already known to stand near pixel near_px (u, v): the
    vision model's box round `what` must take in near_px (within margin_px of
    it) — else it is some other one, and (None, why). On the whole picture: a
    small crop came back boxed in pixels the crop does not have (qwen2.5vl,
    2026-11-17, 372-911 px in a 320 px crop). What a scan's outline is checked
    against or replaced by, before a pick that plans by it.
    """
    found, why = find(frame_rgb, what, model=model, ask=ask, convention=convention, segment_box=segment_box)
    if found is None:
        return None, why
    x0, y0, x1, y1 = (float(c) for c in found.box)
    u, v = float(near_px[0]), float(near_px[1])
    if not (x0 - margin_px <= u <= x1 + margin_px and y0 - margin_px <= v <= y1 + margin_px):
        return None, (f"the vision model boxed {what} at ({x0:.0f}-{x1:.0f}, {y0:.0f}-{y1:.0f}) px, not round "
                      f"({u:.0f}, {v:.0f})")
    return found, ""
