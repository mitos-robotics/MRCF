"""
board_log.py — what each scan saw and what the arm did with it, on disk.

    board_state.json    the latest scan: every piece, where its foot actually
                        is, how wide, what colour and type, and which source
                        (classifier, camera, or both) reported it
    logs/scans.jsonl    every scan, one JSON object per line, append-only
    logs/frames/        the overhead frame each scan measured from, so a scan
                        can be re-measured later with better code
    logs/picks.jsonl    every pick and place: what was targeted, with what
                        geometry, and whether it worked
    logs/openings.jsonl every chess move play.py made or took back: its
                        carries, their tries and times, what the look after found
    object_memory.json  who each piece is across scans, moves and restarts
                        (object_memory.py, the framework's; memory() here)

The state file is the latest scan, for a person or a tool to look at; the
skills work from scene_state, which the same scan fills. The logs are the
record: they are how a rule tuned before a piece type had been seen (the
black-piece threshold, a grasp height) gets checked against what actually
happened.

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from mrcf import privacy

HERE = Path(__file__).parent
STATE_PATH = HERE / "board_state.json"
LOG_DIR = HERE / "logs"
MEMORY_PATH = HERE / "object_memory.json"


def _plain(value):
    """JSON-safe copy: numpy arrays and scalars to lists and floats."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _append(name: str, record: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(_plain(record)) + "\n")


def record_scan(pieces: list, empty_squares: list, meta: dict, frame_rgb=None) -> dict:
    """Write the latest board state and append it to the scan log. Returns the record."""
    stamp = datetime.now()
    record = {"time": stamp.isoformat(timespec="seconds"), **meta,
              "pieces": pieces, "empty_squares": sorted(empty_squares)}
    if frame_rgb is not None:
        try:
            import cv2
            frames = LOG_DIR / "frames"
            frames.mkdir(parents=True, exist_ok=True)
            path = frames / f"{stamp:%Y%m%d_%H%M%S}.jpg"
            privacy.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            try:
                record["frame"] = str(path.relative_to(HERE))
            except ValueError:
                record["frame"] = str(path)
        except Exception as exc:          # a log that cannot write a frame still logs
            record["frame_error"] = str(exc)
    try:
        STATE_PATH.write_text(json.dumps(_plain(record), indent=1))
        _append("scans.jsonl", record)
    except OSError as exc:
        print(f"  [Log] Could not write the board log ({exc}).")
    return record


def record_pick(kind: str, name: str, square: "str | None", success: bool, **details) -> None:
    """Append one pick or place to logs/picks.jsonl."""
    try:
        _append("picks.jsonl", {"time": datetime.now().isoformat(timespec="seconds"),
                                "kind": kind, "object": name, "square": square,
                                "success": bool(success), **details})
    except OSError as exc:
        print(f"  [Log] Could not write the pick log ({exc}).")


def record_move(**details) -> None:
    """Append one chess move the arm made (play.py: a move, a take-back, a run's summary) to logs/openings.jsonl."""
    try:
        _append("openings.jsonl", {"time": datetime.now().isoformat(timespec="seconds"), **details})
    except OSError as exc:
        print(f"  [Log] Could not write the openings log ({exc}).")


# ── Shapes measured by the gripper camera ──────────────────────────────────────
#
# One line per measurement in logs/shapes.jsonl: a piece's height and width
# profile, read at a hover (piece_vision.measure_piece), with the square it
# stood on, its colour and its foot as the overhead scan saw them. The record
# of every measurement; which piece a measurement belongs to now is the object
# memory's job (memory() below), which follows a piece when it moves.


def save_image(kind: str, name: str, rgb) -> "str | None":
    """Save an RGB frame under logs/<kind>/ as JPEG; its path, or None."""
    try:
        import cv2
        folder = LOG_DIR / kind
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{datetime.now():%Y%m%d_%H%M%S}_{name}.jpg"
        privacy.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        try:
            return str(path.relative_to(HERE))
        except ValueError:
            return str(path)
    except Exception:
        return None


def record_shape(piece_type: "str | None", colour: "str | None", square: "str | None",
                 height_mm: float, widths_mm: dict, foot_mm: "float | None", **details) -> None:
    """Append one gripper-camera measurement to logs/shapes.jsonl."""
    try:
        _append("shapes.jsonl", {"time": datetime.now().isoformat(timespec="seconds"),
                                 "type": piece_type, "colour": colour, "square": square,
                                 "height_mm": height_mm, "widths_mm": widths_mm,
                                 "foot_mm": foot_mm, **details})
    except OSError as exc:
        print(f"  [Log] Could not write the shape log ({exc}).")


# ── Who each piece is ──────────────────────────────────────────────────────────

def piece_name(track) -> str:
    """
    The name the planner sees for a remembered piece: <colour>_<type>_<square>,
    so plans and few-shot examples read the same. The type is in the name only
    once something better than the vision model's guess vouches for it — its
    guesses can call a king or a rook "pawn" — until then it is "piece", and
    the guess goes in the description. A held piece keeps the square it was
    picked from.
    """
    if "colour" not in track.attributes and track.place is None:
        # Found by the surface scan, not on a board: an object, not a piece.
        from mrcf.object_memory import default_namer
        return default_namer(track)
    colour = track.attributes.get("colour") or "unknown"
    vouched = track.label_source in ("shape", "taught", "user")
    kind = track.label if track.label and vouched else "piece"
    if track.state in ("present", "held") and track.place:
        return f"{colour}_{kind}_{track.place}"
    return f"{colour}_{kind}_{track.id.replace('-', '')}"


_MEMORY = None


def memory():
    """This deployment's object memory, loaded from MEMORY_PATH on first use."""
    global _MEMORY
    if _MEMORY is None:
        from mrcf.object_memory import ObjectMemory
        _MEMORY = ObjectMemory(MEMORY_PATH, namer=piece_name)
    return _MEMORY
