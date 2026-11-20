"""
eval_piece_labels.py — which local vision model should label the scan's pieces?

The scan's outlines say whether a piece is there and where; the vision model
only names each outlined piece. Until 2026-11-03 the scan asked about each
piece from a close crop of its outline, its answer held to fixed lists by a
JSON schema, asked of two frames that must agree; it now shows the model every
piece at once on one sheet (chessboard.label_pieces), which this does not
score. This scores the one-crop way, model by model, on labelled crops made
as the scan made them (crop_b64 and _label_prompt below, chessboard's
_LABEL_SYSTEM, vlm.label):

    wrist     piece_frames/: the light pawn and knight, and the empty board at
              the knight's labelled pixels. The two looks are the two frames
              captured at each pose (<pose>_0.png, <pose>_1.png) — at a third
              of the poses the camera gave the same frame twice, so there the
              two looks are one crop asked twice (the run says how many).
    overhead  flight-recorder frames, the scan's real input: the light pawn on
              e4 (one run) and e5 (another), before the arm reaches over it, and
              empty squares two away from it as not_a_piece negatives. The two
              looks are consecutive frames, ~0.5 s apart.

Per model and source:
    type ok       agreed on the right type (not_a_piece for an empty) / all
    WRONG agreed  both looks gave the same wrong type — the dangerous one
    disagree      the looks differed (type None: the scan leaves it unknown)
    colour ok     agreed on the right colour / real pieces
    NAP on piece  a real piece agreed to be not_a_piece (vetoed by the scan)
    empty kept    an empty NOT agreed to be not_a_piece (the scan keeps it)
    s/look        median seconds per look, the model already loaded

Talks only to Ollama — no camera, no arm.

Usage (from the repo root, with `ollama serve` running):
    python deployments/chess/tests/eval_piece_labels.py                    # all three models
    python deployments/chess/tests/eval_piece_labels.py --models gemma3:12b --per-class 4
    python deployments/chess/tests/eval_piece_labels.py --models gemma4:31b+nothink   # with "think": false
    python deployments/chess/tests/eval_piece_labels.py --source overhead --per-class 16   # all overhead items
    python deployments/chess/tests/eval_piece_labels.py --variant topdown --source overhead
    python deployments/chess/tests/eval_piece_labels.py --save-crops /tmp/crops --json /tmp/out.json
    python deployments/chess/tests/eval_piece_labels.py --crops-only --save-crops /tmp/crops   # look first
"""

import argparse
import base64
import contextlib
import io
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.resolve().parents[2]
sys.path.insert(0, str(REPO))

import cv2            # noqa: E402
import numpy as np    # noqa: E402
import requests       # noqa: E402

from mrcf import model_server, vlm                     # noqa: E402
from deployments.chess import chessboard as cb         # noqa: E402
from deployments.chess import config as cfg            # noqa: E402

DEFAULT_MODELS = ("qwen2.5vl:7b", "gemma3:12b", "gemma4:31b")
PAWN_FOOT_MM = 22.0     # the light pawn's foot, as the scan measures it

# The crop round an outline, as the scan made it until 2026-11-03: this much of
# its size again on each side (at least _LABEL_MIN_PAD_PX), scaled to
# LABEL_CROP_PX on its longer side.
_LABEL_PAD = 0.6
_LABEL_MIN_PAD_PX = 24
LABEL_CROP_PX = 384


def _label_prompt(foot_mm: "float | None") -> str:
    foot = f" Its foot measures about {foot_mm:.0f} mm across." if foot_mm else ""
    return ("This is a close-up of one object standing on a wooden chessboard, seen from above and in front."
            f"{foot}\n"
            f"type: which chess piece it is, or {cb.NOT_A_PIECE} if it is not a chess piece "
            "(a shadow, a hand, a cable, part of the board).\n"
            "colour: white for the pale-wood pieces, black for the dark-wood ones.")


def crop_b64(frame_rgb: np.ndarray, mask: np.ndarray) -> "str | None":
    """A PNG, base64, of frame_rgb round the outline mask, with room round it. None: an empty mask."""
    from PIL import Image

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    pad = max(_LABEL_MIN_PAD_PX, int(_LABEL_PAD * max(x1 - x0, y1 - y0)))
    h, w = frame_rgb.shape[:2]
    crop = frame_rgb[max(0, y0 - pad):min(h, y1 + pad + 1), max(0, x0 - pad):min(w, x1 + pad + 1)]
    scale = LABEL_CROP_PX / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Wrist frames ───────────────────────────────────────────────────────────────
# 640x480. At hover the pawn is ~80 px across and ~110 px tall (it leans down
# the image), the knight ~95 x 165. A 90 px square outline padded the way
# crop_b64 pads (0.6x its size each side) gives a ~200 px crop: about twice
# the piece, so all of a pawn and most of the knight (its head's tip, which
# leans furthest from its foot, can be cut) with a square of board round it.
WRIST_DIR = HERE / "piece_frames"
WRIST_RUNS = {"20261013_152813_pawn": ("pawn", PAWN_FOOT_MM),
              "20261013_155156_knight": ("knight", None)}
WRIST_EMPTY_RUN = "20261013_155755_empty"
WRIST_EMPTY_AT = "20261013_155156_knight"   # the empty run's crops sit at this run's labels, pose by pose
WRIST_OUTLINE_PX = 90
WRIST_COLOUR = "white"                      # the light-wood set's pieces
# Poses whose crop misses the piece, checked by eye: at o+00_-45 the label (a
# linear model, which drifts far out) is 80-100 px off the piece; at o-30_-30
# the piece is cut by the frame's top edge. Not a test of the vision model.
WRIST_SKIP_POSES = ("o+00_-45", "o-30_-30")

# ── Overhead frames ────────────────────────────────────────────────────────────
# The flight recorder keeps 640x360 JPEGs; the scan crops from 1280x720. A crop
# is padded by at least _LABEL_MIN_PAD_PX *pixels*, so a crop of the half-size
# frame would show twice as much board as the live one. Each frame is scaled
# back to 1280x720 first, so the crop shows what the live crop shows (softer:
# half the pixels, and JPEG). Squares come from board_squares.json (1280x720).
FLIGHT = REPO / "runs" / "flight"
OVERHEAD_RUNS = {
    # incident: (the pawn's square, empty squares two away from it)
    "20261029_170333/incidents/170446_pick_pick_failed__pick_failed_after_3_attempt":
        ("e4", ("c4", "g4", "e6", "e2", "c6", "g6", "c2", "g2")),
    "20261029_151555/incidents/151752_pick_pick_failed__pick_failed_after_3_attempt":
        ("e5", ("c5", "g5", "e3", "e7", "c3", "g3", "c7", "g7")),
}
OVERHEAD_CLEAR_S = 5.0      # the arm starts over the board at ~5.7 s in both
OVERHEAD_SIZE = (1280, 720)
# The pawn's outline from above at 1280x720: its 22 mm foot on a 40 mm, ~62 px
# square is ~34 px. The empties get the same outline, so the crop cannot tell.
OVERHEAD_OUTLINE_PX = 36
OVERHEAD_COLOUR = "white"
BOARD_SQUARES = REPO / "deployments" / "chess" / "board_squares.json"


# ── Variants of the crop and prompt (the live one is the baseline) ────────────

def _prompt_topdown(foot_mm):
    foot = f" The object's foot measures about {foot_mm:.0f} mm across." if foot_mm else ""
    return ("This is a close-up from a camera looking down on a wooden chessboard with 40 mm squares, "
            "centred on one spot. Seen from above, a chess piece shows as a round top over a round "
            "foot, not the familiar side view."
            f"{foot}\n"
            f"type: which chess piece stands in the middle, or {cb.NOT_A_PIECE} if the middle is bare "
            "board (squares, wood grain, a mark), a shadow, a hand or a cable.\n"
            "colour: white for the pale-wood pieces, black for the dark-wood ones.")


VARIANTS = {
    # name: (what it changes, outline scale for the crop, prompt for a foot size)
    "live": ("the scan's crop and prompt until 2026-11-03", 1.0, _label_prompt),
    # Written for the overhead camera: run it with --source overhead.
    "topdown": ("the same crop; a prompt that says the camera looks down and what a piece looks "
                "like from above", 1.0, _prompt_topdown),
}


# ── The labelled set ───────────────────────────────────────────────────────────

def _rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _crop(frame_rgb: np.ndarray, at, side_px: float) -> str:
    """crop_b64 round a filled square outline of side_px centred at `at`."""
    mask = np.zeros(frame_rgb.shape[:2], bool)
    h, w = mask.shape
    half = side_px / 2
    x0, x1 = int(round(at[0] - half)), int(round(at[0] + half))
    y0, y1 = int(round(at[1] - half)), int(round(at[1] + half))
    mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    b64 = crop_b64(frame_rgb, mask)
    if b64 is None:
        raise ValueError(f"outline at {at} falls outside the frame")
    return b64


def _take(cands: list, n: int, rng: random.Random) -> list:
    return cands if len(cands) <= n else [cands[i] for i in sorted(rng.sample(range(len(cands)), n))]


def wrist_items(per_class: int, scale: float, rng: random.Random) -> list:
    items = []

    def poses(run):
        labels = json.loads((WRIST_DIR / run / "labels.json").read_text())["frames"]
        return labels, sorted({name.rsplit("_", 1)[0] for name in labels} - set(WRIST_SKIP_POSES))

    for run, (ptype, foot) in WRIST_RUNS.items():
        labels, names = poses(run)
        for pose in _take(names, per_class, rng):
            looks = [_crop(_rgb(WRIST_DIR / run / f"{pose}_{k}.png"), labels[f"{pose}_{k}.png"],
                           WRIST_OUTLINE_PX * scale) for k in (0, 1)]
            items.append({"source": "wrist", "truth": ptype, "colour": WRIST_COLOUR, "foot": foot,
                          "looks": looks, "what": f"{run[-6:]} {pose}"})
    at_labels, _ = poses(WRIST_EMPTY_AT)
    _, names = poses(WRIST_EMPTY_RUN)
    for pose in _take(names, per_class, rng):
        looks = [_crop(_rgb(WRIST_DIR / WRIST_EMPTY_RUN / f"{pose}_{k}.png"), at_labels[f"{pose}_{k}.png"],
                       WRIST_OUTLINE_PX * scale) for k in (0, 1)]
        items.append({"source": "wrist", "truth": cb.NOT_A_PIECE, "colour": None, "foot": PAWN_FOOT_MM,
                      "looks": looks, "what": f"empty {pose}"})
    return items


def overhead_items(per_class: int, scale: float, rng: random.Random) -> list:
    squares = json.loads(BOARD_SQUARES.read_text())
    pawns, empties = [], []
    for incident, (pawn_sq, empty_sqs) in OVERHEAD_RUNS.items():
        frames = sorted((FLIGHT / incident / "frames").glob("overhead_*.jpg"))
        frames = [f for f in frames if float(f.stem.split("_")[1]) < OVERHEAD_CLEAR_S]
        pairs = [(frames[k], frames[k + 1]) for k in range(0, len(frames) - 1, 2)]
        if not pairs:
            raise FileNotFoundError(f"no overhead frames before {OVERHEAD_CLEAR_S} s in {incident}")
        for pair in pairs:
            pawns.append((incident, pawn_sq, pair))
        # Each empty square once, on a pair of its own where there are enough.
        for k, sq in enumerate(empty_sqs):
            empties.append((incident, sq, pairs[k % len(pairs)]))

    def make(cands, truth, colour, foot):
        out = []
        for incident, sq, pair in _take(cands, per_class, rng):
            at = squares[sq]["pixel"]
            looks = [_crop(cv2.resize(_rgb(f), OVERHEAD_SIZE, interpolation=cv2.INTER_CUBIC), at,
                           OVERHEAD_OUTLINE_PX * scale) for f in pair]
            out.append({"source": "overhead", "truth": truth, "colour": colour, "foot": foot,
                        "looks": looks, "what": f"{incident[:15]} {sq} {pair[0].stem[-7:]}+{pair[1].stem[-7:]}"})
        return out

    return (make(pawns, "pawn", OVERHEAD_COLOUR, PAWN_FOOT_MM)
            + make(empties, cb.NOT_A_PIECE, None, PAWN_FOOT_MM))


# ── Asking the model, timed ────────────────────────────────────────────────────

_CALLS: list = []
_LAST_IMAGE: list = [None]
_real_call_vlm = vlm.call_vlm


def _timed_call_vlm(*args, **kwargs):
    """vlm.call_vlm, timed, its answer and log kept, its printing kept quiet.
    repeat: the same image as the call before, which Ollama answers from its
    cache in a fraction of the time — such calls are left out of s/look."""
    image = args[1] if len(args) > 1 else kwargs.get("image_b64")
    repeat = image is not None and image == _LAST_IMAGE[0]
    _LAST_IMAGE[0] = image
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        answer = _real_call_vlm(*args, **kwargs)
    _CALLS.append({"s": time.perf_counter() - t0, "answer": answer, "log": buf.getvalue(), "repeat": repeat})
    return answer


vlm.call_vlm = _timed_call_vlm   # vlm.label looks it up at call time

# "<model>+nothink": the same model with Ollama's "think": false added to each
# request. call_vlm sends no "think", so a thinking model (gemma4) thinks
# first, and the label call's 64-token cap runs out before it answers.
NOTHINK = "+nothink"
_THINK_OFF = [False]


class _Requests:
    """requests, as vlm sees it, adding "think": false while _THINK_OFF is set."""
    exceptions = requests.exceptions

    @staticmethod
    def post(url, json=None, **kwargs):
        if _THINK_OFF[0] and json is not None and url.endswith("/api/chat"):
            json = {**json, "think": False}
        return requests.post(url, json=json, **kwargs)


model_server.requests = _Requests    # where vlm's calls are posted (mrcf/model_server.py)


def _unload(model: str) -> None:
    base = vlm.server().url.rsplit("/api/", 1)[0]
    with contextlib.suppress(requests.RequestException):
        requests.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)


def _mirrored(b64: str) -> str:
    """The same crop flipped left-right: a warm-up image the set does not contain."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    return base64.b64encode(cv2.imencode(".png", img[:, ::-1])[1].tobytes()).decode("utf-8")


def run_model(name: str, items: list, prompt_fn) -> dict:
    fields = {"type": [*cfg.PIECE_TYPES, cb.NOT_A_PIECE], "colour": list(cfg.PIECE_COLOURS)}
    model = name.removesuffix(NOTHINK)
    _THINK_OFF[0] = name.endswith(NOTHINK)
    vlm.configure(model=model)
    # Warm up: the first call loads the model; it is not timed.
    _CALLS.clear()
    t0 = time.perf_counter()
    vlm.label([_mirrored(items[0]["looks"][0])], fields, prompt_fn(items[0]["foot"]), system=cb._LABEL_SYSTEM)
    warm = {"s": time.perf_counter() - t0, "log": _CALLS[-1]["log"] if _CALLS else "",
            "answer": _CALLS[-1]["answer"] if _CALLS else None}
    print(f"  {name}: loaded in {warm['s']:.1f} s; first answer {warm['answer']}")
    results = []
    for n, it in enumerate(items, 1):
        _CALLS.clear()
        got = vlm.label(it["looks"], fields, prompt_fn(it["foot"]), system=cb._LABEL_SYSTEM)
        calls = list(_CALLS)
        results.append({"source": it["source"], "truth": it["truth"], "colour_truth": it["colour"],
                        "what": it["what"], "type": got["type"], "colour": got["colour"],
                        "looks": [c["answer"] for c in calls], "s": [round(c["s"], 2) for c in calls],
                        "repeat": [c["repeat"] for c in calls],
                        "errors": [c["log"].strip() for c in calls if "[VLM]" in c["log"]]})
        mark = "ok" if got["type"] == it["truth"] else ("--" if got["type"] is None else "WRONG")
        secs = " ".join(f"{c['s']:.1f}s" + ("*" if c["repeat"] else "") for c in calls)
        said = "/".join(str(c["answer"].get("type")) if isinstance(c["answer"], dict) else "?" for c in calls)
        print(f"    {n:3d}/{len(items)} {it['source']:8s} {it['truth']:11s} -> "
              f"{got['type']!s:11s} {got['colour']!s:5s} {mark:5s} {secs:12s} {it['what']}  ({said})", flush=True)
    _unload(model)
    _THINK_OFF[0] = False
    return {"model": name, "warm": warm, "results": results}


# ── Scoring ────────────────────────────────────────────────────────────────────

def _pct(k: int, n: int) -> str:
    return f"{k}/{n} {100 * k / n:3.0f}%" if n else "-"


def score(rows: list) -> dict:
    pieces = [r for r in rows if r["truth"] != cb.NOT_A_PIECE]
    empties = [r for r in rows if r["truth"] == cb.NOT_A_PIECE]
    secs = [s for r in rows for s, rep in zip(r["s"], r["repeat"]) if not rep]
    return {
        "n": f"{len(pieces)}+{len(empties)}",
        "type ok": _pct(sum(r["type"] == r["truth"] for r in rows), len(rows)),
        "WRONG agreed": str(sum(r["type"] is not None and r["type"] != r["truth"] for r in rows)),
        "disagree": _pct(sum(r["type"] is None for r in rows), len(rows)),
        "colour ok": _pct(sum(r["colour"] == r["colour_truth"] for r in pieces), len(pieces)),
        "NAP on piece": _pct(sum(r["type"] == cb.NOT_A_PIECE for r in pieces), len(pieces)),
        "empty kept": _pct(sum(r["type"] != cb.NOT_A_PIECE for r in empties), len(empties)),
        "s/look": f"{statistics.median(secs):.2f}" if secs else "-",
        "errors": str(sum(bool(r["errors"]) for r in rows)),
    }


COLS = ("n", "type ok", "WRONG agreed", "disagree", "colour ok", "NAP on piece", "empty kept", "s/look", "errors")
WIDTHS = (6, 10, 12, 10, 10, 12, 10, 6, 6)


def print_tables(runs: list) -> None:
    print("\nn = real pieces + empties. type ok counts not_a_piece on an empty as right.")
    print("WRONG agreed = both looks gave the same wrong type. empty kept = an empty not agreed "
          "to be not_a_piece.")
    print("s/look = median over looks at a new image (a repeat of the last image is answered from "
          "Ollama's cache).")
    for run in runs:
        print(f"\n{run['model']}  (load {run['warm']['s']:.1f} s)")
        print(f"  {'source':9s} " + " ".join(f"{c:>{w}s}" for c, w in zip(COLS, WIDTHS)))
        for source in ("wrist", "overhead", "all"):
            rows = [r for r in run["results"] if source in ("all", r["source"])]
            if rows:
                s = score(rows)
                print(f"  {source:9s} " + " ".join(f"{s[c]:>{w}s}" for c, w in zip(COLS, WIDTHS)))
        # What each class was called (agreed type; None = the looks differed).
        for source in ("wrist", "overhead"):
            for truth in sorted({r["truth"] for r in run["results"] if r["source"] == source}):
                said = Counter(str(r["type"]) for r in run["results"]
                               if r["source"] == source and r["truth"] == truth)
                cols = Counter(str(r["colour"]) for r in run["results"]
                               if r["source"] == source and r["truth"] == truth)
                print(f"    {source:8s} {truth:11s} type: "
                      + ", ".join(f"{k} {v}" for k, v in said.most_common())
                      + "   colour: " + ", ".join(f"{k} {v}" for k, v in cols.most_common()))
        errs = [e for r in run["results"] for e in r["errors"]]
        if errs:
            print(f"    errors, e.g.: {errs[0][:300]}")


def save_crops(items: list, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for n, it in enumerate(items):
        for k, b64 in enumerate(it["looks"]):
            name = f"{n:02d}_{it['source']}_{it['truth']}_{k}.png"
            (out / name).write_bytes(base64.b64decode(b64))
    print(f"Crops: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--per-class", type=int, default=8,
                    help="items per class per source (default 8; the pools: wrist 10 poses a class, "
                         "overhead 10 pawn frame pairs and 16 empty squares)")
    ap.add_argument("--source", choices=("wrist", "overhead", "both"), default="both")
    ap.add_argument("--variant", choices=tuple(VARIANTS), default="live")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ollama-url", default=None,
                    help="Ollama's URL (default: MRCF_MODEL_URL, else OLLAMA_URL, else this machine's)")
    ap.add_argument("--save-crops", type=Path, help="write every look's crop here")
    ap.add_argument("--json", type=Path, help="write every item's answers here")
    ap.add_argument("--crops-only", action="store_true", help="build (and --save-crops) the set; ask no model")
    a = ap.parse_args()

    what, scale, prompt_fn = VARIANTS[a.variant]
    vlm.configure(model_server="ollama", model_url=a.ollama_url, model=a.models[0], planning_model=a.models[0])
    # One draw per source, so --source overhead samples what --source both does.
    items = []
    if a.source in ("wrist", "both"):
        items += wrist_items(a.per_class, scale, random.Random(f"{a.seed}:wrist"))
    if a.source in ("overhead", "both"):
        items += overhead_items(a.per_class, scale, random.Random(f"{a.seed}:overhead"))
    counts = Counter((it["source"], it["truth"]) for it in items)
    print(f"Variant {a.variant}: {what}.")
    print(f"{len(items)} items, 2 looks each: " + ", ".join(f"{s} {t} {n}" for (s, t), n in sorted(counts.items())))
    same = Counter(it["source"] for it in items if it["looks"][0] == it["looks"][1])
    if same:
        print("Items whose two frames are identical (the camera gave the same frame twice), so the "
              "two looks are one crop asked twice: " + ", ".join(f"{s} {n}" for s, n in sorted(same.items())))
    if a.save_crops:
        save_crops(items, a.save_crops)
    if a.crops_only:
        return 0

    runs = []
    for model in a.models:
        try:
            runs.append(run_model(model, items, prompt_fn))
        except Exception as exc:   # a model that cannot run at all is a result too
            print(f"  {model}: FAILED — {exc}")
    print_tables(runs)
    if a.json:
        a.json.write_text(json.dumps({"variant": a.variant, "per_class": a.per_class, "seed": a.seed,
                                      "runs": runs}, indent=1))
        print(f"\nAnswers: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
