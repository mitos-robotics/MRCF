"""Global mask selection and bounded batch storage, without model downloads."""

import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from mrcf.perception import segment


def test_packed_suppression_is_global_and_preserves_nontransitive_matches():
    # A overlaps B, B overlaps C, A does not overlap C. C has the best score.
    # Suppressing B within an earlier batch would incorrectly lose A later.
    masks = np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=bool)
    packed = np.packbits(masks, axis=1)
    assert segment._distinct_masks(packed, np.array([0.8, 0.9, 0.95]), 0.5) == [2, 0]
    assert segment._distinct_masks(packed, np.array([0.8, 0.9, 0.95]), 2 / 3) == [2, 1, 0]


def fake_segmenter(monkeypatch, logits, scores):
    torch = pytest.importorskip("torch", reason="tensor integration tests need Torch, no model download")
    calls, buffers, post_sizes = [], [], []

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def __call__(self, images, return_tensors, input_points=None, input_labels=None):
            if input_points is None:
                return Inputs(pixel_values=torch.zeros(1), original_sizes=[images.shape[:2]])
            return Inputs(input_points=torch.tensor(input_points), input_labels=torch.tensor(input_labels))

        def post_process_masks(self, masks, sizes):
            post_sizes.append(masks.shape[1])
            flat = masks.flatten(0, 2)[:, None]
            resized = torch.nn.functional.interpolate(flat, size=sizes[0], mode="bilinear", align_corners=False)
            return [(resized[:, 0] > 0).reshape(masks.shape[1], 1, *sizes[0])]

    class Model:
        encodes = 0

        def get_image_embeddings(self, pixels):
            self.encodes += 1
            return pixels

        def __call__(self, input_points, input_labels, image_embeddings, multimask_output):
            # A tensor made with from_numpy holds this array alive as long as
            # any logits view survives. Previous batches must be gone already.
            assert all(ref() is None for ref in buffers), "float logits survived into the next decode batch"
            ids = input_points[0, :, 0, 0].numpy().astype(int)
            calls.append(ids.tolist())
            data = logits[ids].copy()
            buffers.append(weakref.ref(data))
            return SimpleNamespace(pred_masks=torch.from_numpy(data)[None],
                                   iou_scores=torch.from_numpy(scores[ids].copy())[None])

    model = Model()
    monkeypatch.setattr(segment, "_load", lambda: True)
    monkeypatch.setattr(segment, "_model", model)
    monkeypatch.setattr(segment, "_processor", Processor())
    monkeypatch.setattr(segment, "_device", "cpu")
    return model, calls, post_sizes


def reference_selection(logits, scores, output_size, min_score, max_area, same_iou):
    """The original all-at-once algorithm, used only on small test arrays."""
    import torch
    flat = logits.reshape(-1, *logits.shape[-2:])
    scores = scores.flatten()
    masks = flat > 0
    areas = masks.sum(axis=(1, 2))
    keep = np.flatnonzero((scores >= min_score) & (areas > 0) & (areas <= max_area * masks[0].size))
    chosen = []
    for k in keep[np.argsort(-scores[keep])]:
        if not any((masks[k] & masks[j]).sum() / (masks[k] | masks[j]).sum() > same_iou for j in chosen):
            chosen.append(k)
    if not chosen:
        return []
    full = torch.nn.functional.interpolate(torch.from_numpy(flat[chosen])[:, None],
                                           size=output_size, mode="bilinear", align_corners=False)[:, 0] > 0
    return [(float(scores[k]), full[i].numpy(), k // logits.shape[1]) for i, k in enumerate(chosen)]


def sample_masks():
    rng = np.random.default_rng(18)
    logits = -rng.uniform(0.1, 2, (7, 3, 9, 11)).astype(np.float32)
    for i in range(7):
        for v in range(3):
            y, x = (i + v) % 7, (2 * i + v) % 9
            logits[i, v, y:y + 2, x:x + 2] = rng.uniform(0.1, 2, (2, 2))
    scores = rng.uniform(0.5, 1, (7, 3)).astype(np.float32)
    # Best-scoring duplicate occurs in a later batch; preserve its owner.
    logits[6, 1] = logits[0, 0]
    scores[6, 1], scores[0, 0] = 0.99, 0.8
    logits[1, 0] = -1                     # empty mask
    logits[2, 0] = 1                      # oversized mask
    scores[3, 0] = 0.1                    # low-confidence mask
    return logits, scores


@pytest.mark.parametrize("batch", [1, 2, 4, 96])
def test_batched_results_match_global_reference_and_free_logits(monkeypatch, batch):
    logits, scores = sample_masks()
    model, calls, post_sizes = fake_segmenter(monkeypatch, logits, scores)
    expected = reference_selection(logits, scores, (18, 22), 0.6, 0.1, 0.7)
    actual = segment.candidates(np.zeros((18, 22, 3), np.uint8), [(i, 0) for i in range(7)],
                                max_area_frac=0.1, batch=batch)
    assert len(actual) == len(expected)
    for (score, mask, owner), (ref_score, ref_mask, ref_owner) in zip(actual, expected):
        assert score == ref_score and owner == ref_owner
        assert np.array_equal(mask, ref_mask)
    assert model.encodes == 1
    assert max(map(len, calls)) <= batch
    assert max(post_sizes) <= 3 * batch
    first_pass = (7 + batch - 1) // batch
    assert sorted(p for call in calls[first_pass:] for p in call) == sorted({r[2] for r in expected})


def test_rejected_batches_are_not_retained_or_decoded_again(monkeypatch):
    logits = np.ones((10, 3, 9, 11), np.float32)   # every mask too large
    scores = np.ones((10, 3), np.float32)
    model, calls, post_sizes = fake_segmenter(monkeypatch, logits, scores)
    assert segment.candidates(np.zeros((18, 22, 3), np.uint8), [(i, 0) for i in range(10)], batch=2) == []
    assert len(calls) == 5 and not post_sizes and model.encodes == 1


def test_empty_prompts_do_not_encode_or_decode(monkeypatch):
    logits, scores = sample_masks()
    model, calls, post_sizes = fake_segmenter(monkeypatch, logits, scores)
    assert segment.candidates(np.zeros((18, 22, 3), np.uint8), []) == []
    assert model.encodes == 0 and not calls and not post_sizes


@pytest.mark.parametrize("batch", [0, -1])
def test_invalid_batch_is_rejected_without_loading_a_model(monkeypatch, batch):
    def no_load():
        pytest.fail("must reject invalid batch before loading")
    monkeypatch.setattr(segment, "_load", no_load)
    with pytest.raises(ValueError, match="batch"):
        segment.candidates(np.zeros((1, 1, 3), np.uint8), [(0, 0)], batch=batch)


def test_the_empty_surface_from_photos_with_things_in_different_places():
    from mrcf.perception import surface_scan
    board = np.zeros((40, 40, 3), np.uint8)
    board[:, :20] = (205, 170, 120)                   # maple
    board[:, 20:] = (100, 55, 40)                     # walnut
    shots = []
    for x in (2, 10, 22, 30, 14):                     # a cream piece somewhere different in each
        s = board.copy()
        s[10:18, x:x + 6] = (235, 215, 180)
        shots.append(s)
    ref = surface_scan.reference_from_frames(shots)
    assert np.abs(ref.astype(int) - board.astype(int)).max() <= 1
    # Cream on maple is only ~7-14 apart in Lab (2026-10-22), and still an object.
    now = board.copy()
    now[24:34, 4:12] = (235, 215, 180)
    changed = surface_scan.change_map(now, ref)
    assert changed[26:32, 6:10].all() and not changed[:, 24:].any()


def test_a_change_of_light_is_not_a_change_of_scene():
    from mrcf.perception import surface_scan
    yy, xx = np.mgrid[0:320, 0:320]
    board = np.where(((yy // 40 + xx // 40) % 2)[..., None] == 0,
                     np.array((205, 170, 120)), np.array((100, 55, 40))).astype(np.float32)   # maple, walnut
    ref = board.astype(np.uint8)
    # Later: exposure up a fifth, and the window side half as bright again as the other.
    light = (1.2 * (0.8 + 0.4 * xx / 319.0))[..., None]
    now = np.clip(board * light, 0, 255).astype(np.uint8)
    now[140:170, 150:175] = (235, 215, 180)               # a cream piece on a walnut square
    changed = surface_scan.change_map(now, ref)
    assert changed[145:165, 155:170].mean() > 0.9
    bare = np.ones(changed.shape, bool)
    bare[120:190, 130:195] = False
    assert changed[bare].mean() < 0.05


def test_a_shadow_beside_a_block_is_not_an_object():
    # UR5 tabletop, 2026-11-13: each block's shadow was outlined and named after the block.
    from mrcf.perception import surface_scan
    rng = np.random.default_rng(0)
    table = np.clip(np.full((320, 320, 3), (205, 205, 200), np.float32) + rng.normal(0, 2, (320, 320, 3)), 0, 255)
    ref = table.astype(np.uint8)
    now = table.copy()
    region = {}

    def put(name, ys, xs, value):
        now[ys, xs] = value(now[ys, xs])
        region[name] = np.zeros((320, 320), bool)
        region[name][ys, xs] = True

    put("shadow", slice(100, 130), slice(100, 130), lambda px: px * 0.8)            # the table, darker
    block = np.full((30, 30, 3), (200, 40, 40), np.float32)                           # a coloured border ...
    block[4:-4, 4:-4] = (240, 240, 235)                                               # ... round a white face
    put("block", slice(118, 148), slice(118, 148), lambda px: block)
    put("black", slice(220, 250), slice(60, 90), lambda px: np.full_like(px, 30))     # darker than any shadow
    put("white", slice(220, 250), slice(220, 250), lambda px: np.full_like(px, 250))
    region["shadow"] &= ~region["block"]                                              # where the block covers it
    changed, shadowed = surface_scan.compare(now.astype(np.uint8), ref)
    assert changed[region["shadow"]].mean() > 0.9                                     # it does differ ...
    assert surface_scan.is_shadow(region["shadow"], changed, shadowed)                # ... as a shadow
    for thing in ("block", "black", "white"):
        assert changed[region[thing]].mean() > 0.9
        assert not surface_scan.is_shadow(region[thing], changed, shadowed), thing
    assert (surface_scan.change_map(now.astype(np.uint8), ref) == changed).all()
    # The block's own outline, with the strip of its shadow it took in: still the
    # block, and measured without the strip (it pulled a block's centre 9 mm).
    rows, cols = np.mgrid[0:320, 0:320]
    strip = region["shadow"] & (rows >= 110) & (cols < 118)
    both = region["block"] | strip
    assert not surface_scan.is_shadow(both, changed, shadowed)
    trimmed = surface_scan.without_shadow(both, shadowed)
    assert (trimmed & strip).sum() < 0.05 * strip.sum()
    assert (trimmed & region["block"]).sum() > 0.9 * region["block"].sum()
    assert surface_scan.without_shadow(region["shadow"] & (rows < 103), shadowed) is not None   # all shadow: kept as it was


def test_a_washed_out_or_dark_view_is_warned_of():
    from mrcf.perception import lighting
    view = np.full((100, 100, 3), 150, np.uint8)
    assert lighting.warning(lighting.reading(view)) is None
    view[:20] = 255                                       # a fifth at the maximum
    assert "washed out" in lighting.warning(lighting.reading(view))
    assert "dark" in lighting.warning(lighting.reading(np.full((100, 100, 3), 50, np.uint8)))


def test_a_camera_that_has_moved_is_caught_before_the_scan():
    # UR5 tabletop, 2026-11-13: the overhead camera's mount crept 12 px after its
    # calibration, and a pick came down on its block.
    import cv2
    from mrcf.perception import plane_geometry as pg
    from mrcf.perception import surface_scan
    rng = np.random.default_rng(1)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (720, 1280)).astype(np.float32), (0, 0), 2.0)
    surface = np.clip(160 + 40 * grain / grain.std(), 0, 255).astype(np.uint8)
    ref = np.dstack([surface] * 3)
    eye, at = np.array([0.3, -0.2, 0.9]), np.array([0.3, 0.3, 0.0])
    z = (at - eye) / np.linalg.norm(at - eye)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.c_[x, np.cross(z, x), z], eye
    view = pg.PlaneView.from_camera_model({"K": np.array([[870.0, 0, 640], [0, 870.0, 360], [0, 0, 1]]),
                                           "D": np.zeros(5), "plane_z_m": 0.0, "scale": 1.0, "T_world_cam": T})

    def slid(dx, dy):
        return cv2.warpAffine(ref, np.float32([[1, 0, dx], [0, 1, dy]]), (1280, 720), borderMode=cv2.BORDER_REFLECT)

    dx, dy, turn, n = surface_scan.drift(slid(-12, 3), ref)
    assert abs(dx + 12) < 0.5 and abs(dy - 3) < 0.5 and abs(turn) < 0.1 and n >= surface_scan.DRIFT_MATCHES
    surface_scan.check_drift(ref, ref, view, 4.0)                    # standing still: nothing said
    surface_scan.check_drift(slid(2, 1), ref, view, 4.0)             # within the limit
    with pytest.raises(surface_scan.CameraMoved, match="calibrate it again") as moved:
        surface_scan.check_drift(slid(-12, 3), ref, view, 4.0)
    assert abs(moved.value.px - np.hypot(12, 3)) < 0.5 and moved.value.mm is not None and moved.value.mm > 0
    assert surface_scan.drift(np.full_like(ref, 128), ref) is None    # nothing to match: it cannot say
