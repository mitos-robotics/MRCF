"""
segment.py — outline objects in a camera frame, whatever their colour.

Wraps SAM2 (Segment Anything 2, via Hugging Face transformers): give it a frame
and some points, and for each point it returns the outline of the object under
it. plane_geometry turns those outlines into measurements.

Why a learned segmenter
-----------------------
A colour rule separates an object from the surface by colour, and a boxwood
piece on a maple square is the same colour as the square: such a rule sees
only the part of a cream piece standing over a walnut square. SAM outlines the
whole object over any background: it uses edges and shading, not a colour
threshold — which is also why nothing about it is specific to chess pieces, or
to a board.

Optional: without transformers and torch (and the model download, ~150 MB, on
first use) available() is False and callers fall back to what they did before.

    pip install -e ".[measure]"        # transformers below 5, and torch

Below 5.x: lerobot pins huggingface_hub below 1.0, which 5.x needs.

One model, shared: a scan on a perception thread and a measurement inside a
pick may both ask for outlines, and they take turns (a lock), rather than
loading a second copy or running one model from two threads at once.

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

import functools
import threading

import numpy as np

# Small and fast enough for every scan (~70 ms per frame on Apple silicon).
MODEL_ID = "facebook/sam2.1-hiera-tiny"

_model = None
_processor = None
_device = None
_unavailable: "str | None" = None
_lock = threading.RLock()


def _serialised(fn):
    """Run fn holding the model's lock."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _lock:
            return fn(*args, **kwargs)
    return wrapper


@_serialised
def _load() -> bool:
    global _model, _processor, _device, _unavailable
    if _model is not None:
        return True
    if _unavailable is not None:
        return False
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except ImportError as exc:
        _unavailable = (f"{exc} — install with: pip install -e \".[measure]\" (transformers and torch)")
        return False
    _device = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
    try:
        import logging
        # The checkpoint is published as the video model; the image model loads
        # from it fine, and says so on every load.
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        _model = Sam2Model.from_pretrained(MODEL_ID).to(_device).eval()
        _processor = Sam2Processor.from_pretrained(MODEL_ID)
    except Exception as exc:     # no network on first use, a bad cache, ...
        _unavailable = f"could not load {MODEL_ID}: {exc}"
        _model = _processor = None
        return False
    return True


def available() -> bool:
    """True when the segmenter can run; why_unavailable() says why not."""
    return _load()


def why_unavailable() -> str:
    _load()
    return _unavailable or ""


_BIT_COUNTS = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def _distinct_masks(packed, scores, same_iou):
    """Global score-ordered suppression on bit-packed low-resolution masks."""
    chosen = []
    for k in np.argsort(-scores):
        if any(_BIT_COUNTS[packed[k] & packed[j]].sum()
               / max(int(_BIT_COUNTS[packed[k] | packed[j]].sum()), 1) > same_iou
               for j in chosen):
            continue
        chosen.append(int(k))
    return chosen


@_serialised
def candidates(frame_rgb: np.ndarray, points, min_score: float = 0.6,
               max_area_frac: float = 0.05, same_iou: float = 0.7, batch: int = 96) -> list:
    """
    Distinct object outlines found from a set of prompt points.

    Returns [(score, mask, point_index), ...], best first, one per object:
    every prompt yields three candidate outlines (a part, a whole, a larger
    whole), and many prompts land on the same object, so outlines below
    min_score, bigger than max_area_frac of the frame, or overlapping a better
    one by more than same_iou are dropped. What is left are distinct things;
    deciding which are pieces is the caller's job, because only the caller
    knows what a piece looks like in its geometry.

    The frame is encoded once. Batches are filtered immediately and only
    bit-packed masks retained for global score-ordered comparison. Winning
    prompts are decoded again, using the same embedding, and only their
    winning masks are scaled up. Float logits never accumulate across batches.
    """
    if batch < 1:
        raise ValueError("batch must be positive")
    if not _load():
        raise RuntimeError(why_unavailable())
    import torch

    points = [tuple(map(float, p)) for p in points]
    if not points:
        return []
    assert _model is not None and _processor is not None     # _load() made them
    inputs = _processor(images=frame_rgb, return_tensors="pt").to(_device)
    with torch.no_grad():
        embeddings = _model.get_image_embeddings(inputs["pixel_values"])
    def decode(chunk):
        prompts = _processor(images=frame_rgb, input_points=[[[list(p)] for p in chunk]],
                             input_labels=[[[1] for _ in chunk]], return_tensors="pt").to(_device)
        with torch.no_grad():
            return _model(input_points=prompts["input_points"], input_labels=prompts["input_labels"],
                          image_embeddings=embeddings, multimask_output=True)

    packed: list[np.ndarray]
    scores: list | np.ndarray
    packed, scores, owners, variants = [], [], [], []
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        out = decode(chunk)
        count = out.pred_masks.shape[2]
        binary = (out.pred_masks[0].flatten(0, 1) > 0).cpu().numpy()
        batch_scores = out.iou_scores[0].float().cpu().flatten().numpy()
        area = binary.reshape(len(binary), -1).sum(1)
        keep = np.flatnonzero((batch_scores >= min_score) & (area > 0)
                              & (area <= max_area_frac * binary[0].size))
        packed.extend(np.packbits(binary[keep].reshape(len(keep), binary[0].size), axis=1))
        scores.extend(batch_scores[keep])
        owners.extend(i + keep // count)
        variants.extend(keep % count)
        del out, binary, batch_scores
    scores = np.asarray(scores)
    chosen = _distinct_masks(packed, scores, same_iou)
    if not chosen:
        return []

    # Multiple distinct masks can come from one prompt. Decode it once, then
    # select its winning variants; preserve global score order in the result.
    wanted: dict[int, list] = {}
    for rank, k in enumerate(chosen):
        wanted.setdefault(int(owners[k]), []).append((rank, k))
    prompt_ids = list(wanted)
    result: list = [None] * len(chosen)
    for start in range(0, len(prompt_ids), batch):
        ids = prompt_ids[start:start + batch]
        out = decode([points[p] for p in ids])
        selected = [(j, rank, k) for j, p in enumerate(ids) for rank, k in wanted[p]]
        low = torch.stack([out.pred_masks[0, j, variants[k]] for j, _, k in selected]).float().cpu()
        full = _processor.post_process_masks(low[None, :, None], inputs["original_sizes"])[0]
        full = full[:, 0].numpy().astype(bool)
        for n, (_, rank, k) in enumerate(selected):
            result[rank] = (float(scores[k]), full[n], int(owners[k]))
        del out, low
    return result


@_serialised
def segment_box(frame_rgb: np.ndarray, box) -> "tuple[list, list]":
    """
    The outlines of the one thing inside a box, (x1, y1, x2, y2) pixels:
    (masks, scores), SAM's three candidates for it (a part, a whole, a larger
    whole) and its own estimate of each. A box says which thing far better
    than a point when the thing is big, thin-walled or see-through — a tub,
    a tray, a basket — where a point lands on what shows through it. What
    the box should hold is the caller's (perception/grounding.py: a vision
    model's box round a named thing).
    """
    if not _load():
        raise RuntimeError(why_unavailable())
    import torch

    assert _model is not None and _processor is not None     # _load() made them
    x1, y1, x2, y2 = (float(v) for v in box)
    inputs = _processor(images=frame_rgb, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt").to(_device)
    with torch.no_grad():
        out = _model(**inputs, multimask_output=True)
    masks = _processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0].numpy()[0]
    scores = out.iou_scores[0, 0].float().cpu().numpy()
    return [m.astype(bool) for m in masks], [float(s) for s in scores]


@_serialised
def segment(frame_rgb: np.ndarray, points) -> list:
    """
    Outlines of whatever lies under each point.

    Returns, per point and in the same order, (masks, scores): three candidate
    boolean masks of the frame's size — SAM proposes a part, a whole and a
    larger whole for every prompt — and the model's own estimate of how good
    each one is. Choosing among them is the caller's job, because only the
    caller knows what it is looking for.

    The image is encoded once however many points are asked about, and the
    points are decoded in one batch, so asking about every square of the board
    costs about the same as asking about one.
    """
    if not _load():
        raise RuntimeError(why_unavailable())
    import torch

    points = [tuple(map(float, p)) for p in points]
    if not points:
        return []
    assert _model is not None and _processor is not None     # _load() made them
    inputs = _processor(images=frame_rgb, return_tensors="pt").to(_device)
    with torch.no_grad():
        embeddings = _model.get_image_embeddings(inputs["pixel_values"])
        prompts = _processor(images=frame_rgb,
                             input_points=[[[list(p)] for p in points]],
                             input_labels=[[[1] for _ in points]],
                             return_tensors="pt").to(_device)
        out = _model(input_points=prompts["input_points"], input_labels=prompts["input_labels"],
                     image_embeddings=embeddings, multimask_output=True)
    masks = _processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0].numpy()
    scores = out.iou_scores[0].float().cpu().numpy()
    return [(masks[i].astype(bool), scores[i]) for i in range(len(points))]
