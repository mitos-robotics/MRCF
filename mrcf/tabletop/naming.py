"""
naming.py — what each object the surface scan finds is, in the vision model's
own words, held to a form: whether it is an object at all, what it is (a few
words), and its colour from config.COLOURS. A crop of each object's outline
from the scan's own frame, all of them numbered on one sheet and asked about
in one call (perception/tiles.py, vlm.describe_numbered: a vision model's turn
costs seconds whatever it is shown), once per object: the memory keeps the
answer across scans and restarts, and what the user says (identify) is
trusted over it. Asked again when the scan took an object by its look alone
(moved by hand, back after it had gone): blocks of one size and colour look
alike to the memory, and the letter on one can say which it is — when the
camera's colour agrees (identifier).

    identifier(memory)    scene_scan.observe's identify hook
    object_name(track)    the name the planner sees ("red_bottle")
"""

from __future__ import annotations

import base64
import io
import re

import numpy as np

from . import cfg

NOT_AN_OBJECT = "not an object"
CROP_PX = 384                    # the crop's longer side, as the model sees it
_PAD = 0.6                       # room round the outline, as a share of its size
_MIN_PAD_PX = 24
_MAX_WORDS = 3

TILE_PX = 192                    # a crop's side on the sheet: a block's letter still readable
SHEET_COLS = 4

_SYSTEM = (
    "You name objects on a table for a robot arm that will pick them up and put them down. "
    "You see one object, cropped from a camera looking down at the table at an angle. "
    "Name it the way a person would ask for it: a plain noun of one to three words "
    "(\"bottle\", \"coffee mug\", \"toy car\"), without its colour. If a letter or a number is printed "
    "on it, end the name with the one on its top face, or else the one most clearly seen (\"block P\", "
    "\"dice 6\"): that is how a person tells such objects apart. If what is in the middle of the "
    "picture is not a separate object (a mark, a shadow, a reflection, part of the table), say so."
)


def _prompt(foot_mm: float, height_mm: "float | None") -> str:
    size = f"about {foot_mm:.0f} mm across at its base" + (f" and {height_mm:.0f} mm tall" if height_mm else "")
    return (f"The object in the middle of this picture is {size}. Is it a separate object on the table; "
            "what is it; and what colour would a person call it? That is its main colour, except on a white or "
            "plain object with a letter, a number or a border printed on it in a colour, as a letter block has: "
            "then the colour of that print.")


_SHEET_SYSTEM = (
    "You name objects on a table for a robot arm that will pick them up and put them down. "
    "You see a sheet of numbered pictures, each one thing cropped from a camera looking down at the table at "
    "an angle, the thing in the middle of its picture. Look at each picture by itself. For each number answer "
    "with its colour the way a person would name it, then what it is the way a person would ask for it, one "
    "to three words (\"blue block\", \"red coffee mug\"). Its colour is its main colour, except on a white or "
    "plain object with a letter, a number or a border printed on it in a colour, as a letter block has: then "
    "the colour of that print (\"red block M\" for a white block with a red M). If a letter or a number is "
    "printed on it, end with the one on its top face, or else the one most clearly seen (\"green block 7\", "
    "\"white dice 6\"). Answer \"nothing\" for a mark, a shadow, a reflection or part of the table."
)


def _schema() -> dict:
    return {"type": "object",
            "properties": {"is_object": {"type": "boolean"},
                           "what": {"type": "string"},
                           "colour": {"enum": list(cfg.COLOURS)}},
            "required": ["is_object", "what", "colour"]}


def crop_b64(frame_rgb: np.ndarray, mask: np.ndarray) -> "str | None":
    """A PNG, base64, of frame_rgb round the outline mask, with room round it. None: an empty mask."""
    import cv2
    from PIL import Image

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    pad = max(_MIN_PAD_PX, int(_PAD * max(x1 - x0, y1 - y0)))
    h, w = frame_rgb.shape[:2]
    crop = frame_rgb[max(0, y0 - pad):min(h, y1 + pad + 1), max(0, x0 - pad):min(w, x1 + pad + 1)]
    scale = CROP_PX / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(crop)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def crop_tile(frame_rgb: np.ndarray, mask: np.ndarray, side: int = TILE_PX) -> "np.ndarray | None":
    """frame_rgb round the outline mask, with room round it, as a square side x side tile. None: an empty mask."""
    import cv2

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    cx, cy = (int(xs.min()) + int(xs.max())) // 2, (int(ys.min()) + int(ys.max())) // 2
    span = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    pad = _PAD * span
    if cfg.NAME_TILE_PAD_MAX_PX is not None:
        # A long object's room round it a share of its length took in the
        # table round it too: 2026-11-19, the scoop's tile, 2.2 times its 19 cm
        # outline, showed a figure, the plug and the table's edge beside it —
        # and the model called it "nothing". Room enough to tell it, no more.
        pad = min(pad, float(cfg.NAME_TILE_PAD_MAX_PX))
    half = max(_MIN_PAD_PX, int(span / 2 + pad))
    h, w = frame_rgb.shape[:2]
    crop = np.full((2 * half, 2 * half, 3), 255, np.uint8)       # white past the frame's edge
    x0, y0 = cx - half, cy - half
    src = frame_rgb[max(0, y0):min(h, y0 + 2 * half), max(0, x0):min(w, x0 + 2 * half)]
    crop[max(0, -y0):max(0, -y0) + src.shape[0], max(0, -x0):max(0, -x0) + src.shape[1]] = src
    return cv2.resize(crop, (side, side), interpolation=cv2.INTER_CUBIC)


def ask_all(tiles: list, sizes_mm: list) -> list:
    """
    (what, colour) for each tile, in order — (NOT_AN_OBJECT, colour) for one
    that is not an object, None where no usable answer came — from one call
    with every tile on a numbered sheet.
    """
    from mrcf import vlm
    from mrcf.perception import tiles as tl

    if not tiles:
        return []
    sheet = tl.sheet(tiles, cols=min(SHEET_COLS, len(tiles)))
    across = "; ".join(f"{i + 1}: about {mm:.0f} mm across" for i, mm in enumerate(sizes_mm))
    got = vlm.describe_numbered(
        tl.png_b64(sheet), len(tiles),
        f"{len(tiles)} numbered pictures ({across}). The colour is one of: {', '.join(cfg.COLOURS)}.",
        system=_SHEET_SYSTEM, model=cfg.LABEL_MODEL)
    return [_read(g) for g in got]


# Answers that mean "not an object", wherever in the answer.
_NOT_OBJECT_WORDS = {"nothing", "none", "shadow", "reflection", "smudge", "stain", "glare"}
# The sheet's colour words as COLOURS spells them ("gray" -> "grey").
_COLOUR_SPELLINGS = {"gray": "grey", "multicolored": "multicoloured", "multicolor": "multicoloured",
                     "multicolour": "multicoloured", "transparent": "clear"}


def _read(answer: "str | None") -> "tuple[str, str] | None":
    """
    (what, colour) from a sheet's answer ("blue block P", "block P, blue",
    "dark green block 7"): the first COLOURS word anywhere is its colour, the
    rest what it is — its letter or number kept when the words are cut to
    _MAX_WORDS ("wooden letter block p" -> "letter block p").
    """
    if answer is None:
        return None
    words = [_COLOUR_SPELLINGS.get(w, w) for w in re.sub(r"[^a-z0-9 ]", " ", answer.lower()).split()]
    said = " ".join(words)
    if (not words or set(words) & _NOT_OBJECT_WORDS or "not an object" in said or "part of the table" in said
            or "mark on" in said):
        return NOT_AN_OBJECT, ""
    colour = next((w for w in words if w in cfg.COLOURS), "")
    rest = [w for w in words if w != colour and w not in ("dark", "light", "bright", "pale", "a", "an", "the")]
    if len(rest) > _MAX_WORDS and (len(rest[-1]) == 1 or rest[-1].isdigit()):
        rest = rest[-_MAX_WORDS:]          # the letter on it is how it is told apart: keep it, lose a word before
    what = clean_what(" ".join(rest))
    return (what, colour) if what else None


def _kind(label: "str | None") -> str:
    """What it is without the letter or number on it: "block p" -> "block", "dice 6" -> "dice"."""
    words = (label or "").split()
    return " ".join(words[:-1] if len(words) > 1 and (len(words[-1]) == 1 or words[-1].isdigit()) else words)


def clean_what(what: str) -> str:
    """The model's words for it: lower case, letters, digits and spaces, at most _MAX_WORDS words."""
    words = re.sub(r"[^a-z0-9 ]", " ", str(what).lower()).split()
    return " ".join(words[:_MAX_WORDS])


def ask(look: str, foot_mm: float, height_mm: "float | None" = None) -> "tuple[str, str] | None":
    """(what, colour) for one crop; (NOT_AN_OBJECT, colour) when it is not one; None: no usable answer."""
    from mrcf.vlm import call_vlm

    got = call_vlm(_prompt(foot_mm, height_mm), look, system=_SYSTEM, _tag="name", model=cfg.LABEL_MODEL,
                   schema=_schema())
    if not isinstance(got, dict) or "is_object" not in got:
        return None
    colour = got.get("colour") if got.get("colour") in cfg.COLOURS else None
    if not got["is_object"]:
        return NOT_AN_OBJECT, colour or ""
    what = clean_what(got.get("what", ""))
    return (what, colour or "") if what else None


def _second_look(frame, found, asked, answers) -> list:
    """
    What the sheet called nothing, asked again by itself (ask(), its own crop)
    before it is let go of as not an object: an object taken for a mark is
    never picked, and nothing says it was missed. 2026-11-19, the SO-101's
    table: the blue scoop, 94 mm across, plainly in the middle of its tile,
    called "nothing" on the sheet — left out of the scene, and the plan to put
    everything in the tub would have had no scoop in it. Only the sheet's
    "nothing" answers, at most NAME_NOTHING_TWICE_MAX a look.
    """
    out = list(answers)
    left = int(cfg.NAME_NOTHING_TWICE_MAX)
    for k, ((i, _), got) in enumerate(zip(asked, answers)):
        if left <= 0 or got is None or got[0] != NOT_AN_OBJECT:
            continue
        left -= 1
        look = crop_b64(frame, found[i]["mask"])
        meas = found[i]["measurement"]
        again = ask(look, 2 * meas.base_radius_mm, getattr(meas, "height_mm", None)) if look else None
        if again is not None and again[0] != NOT_AN_OBJECT:
            print(f"  [Naming] Taken for nothing on the sheet; asked by itself: {again[1]} {again[0]}.".replace("  ", " "))
            out[k] = again
    return out


def identifier(memory):
    """
    scene_scan.observe's identify(frame, found, tracks): each object no one has
    named yet is named by the model, from a crop of its outline in the frame.
    Named once: an object the memory already has a name for (the model's, or
    the user's) is not asked about again — unless this scan took it by its
    look alone (memory.guessed()). Then, if the model calls it what an object
    not seen now is called, and the camera's colour agrees, it is that one
    (memory.reidentify; tracks[i] becomes it). Otherwise it keeps its name: a
    block shows another letter on each face, and the model misreads them.
    """
    def identify(frame, found, tracks) -> None:
        guessed = memory.guessed()
        asked = []                                 # (index, tile) of each to ask about
        for i, (f, track) in enumerate(zip(found, tracks)):
            if track.label and track.id not in guessed:
                continue
            tile = crop_tile(frame, f["mask"])
            if tile is not None:
                asked.append((i, tile))
        answers = ask_all([t for _, t in asked], [2 * found[i]["measurement"].base_radius_mm for i, _ in asked])
        if cfg.NAME_NOTHING_TWICE:
            answers = _second_look(frame, found, asked, answers)
        for (i, _), got in zip(asked, answers):
            track = tracks[i]
            if got is None:
                continue
            what, colour = got
            if what != NOT_AN_OBJECT and what != track.label:
                was = object_name(track) if track.label else None
                really = memory.reidentify(
                    track.id, what, f"the vision model calls it {colour} {what}".replace("  ", " "),
                    alike=lambda t: bool(colour) and t.attributes.get("colour") == colour and _kind(t.label) == _kind(what))
                if really is not None:
                    tracks[i] = really
                    print(f"  [Memory] That is {object_name(really)}" + (f", not {was}" if was else "")
                          + f": the vision model reads {colour} {what} on it, and the camera's colour agrees.")
                    continue
            memory.checked(track.id)
            if not track.label:
                memory.set_label(track.id, what, "vlm", attributes={"colour": colour} if colour else None)
            elif what == track.label:
                memory.set_label(track.id, what, "vlm")      # it agrees: a vote, the colour left as it is
    return identify


def object_name(track) -> str:
    """
    The name the planner sees: colour and what it is ("red_bottle"); an object
    not yet named, object_<id>. Two alike are told apart by the scan (it adds
    the id to the second).
    """
    label = (track.label or "").strip()
    if not label or label == NOT_AN_OBJECT:
        return f"object_{track.id.rsplit('-', 1)[-1]}"
    colour = str(track.attributes.get("colour") or "").strip()
    words = label.split()
    if colour and colour not in words:
        words = [colour, *words]
    return "_".join(words)


def describe(track, size: "str | None" = None) -> str:
    """What the planner reads about it; size: how big it is, in place of its foot and height."""
    label = (track.label or "").strip()
    if not label:
        return "an object not yet named"
    if label == NOT_AN_OBJECT:
        return "not an object (a mark or shadow, by the camera)"
    colour = str(track.attributes.get("colour") or "").strip()
    text = f"{colour} {label}".strip() if colour and colour not in label.split() else label
    height = track.height_mm
    foot = track.feet_mm[-1] if getattr(track, "feet_mm", None) else None
    if size is None:
        size = ", ".join(s for s in (f"{foot:.0f} mm across" if foot else "", f"{height:.0f} mm tall" if height else "")
                         if s)
    return f"{text}" + (f" ({size})" if size else "")
