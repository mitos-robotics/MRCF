"""mrcf/perception/'s generic pieces: one outline per object, colour between two references, a moved photo, light by area."""

from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mrcf.perception import lighting, outlines, surface_scan, tones  # noqa: E402


def _box(y0, y1, x0, x1, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def _outline(score, mask, radius_mm):
    return (score, mask, SimpleNamespace(base_radius_mm=radius_mm))


def test_the_object_without_its_shadow_is_kept_over_the_surer_outline_with_it():
    piece = _box(20, 50, 40, 60)
    with_shadow = piece | _box(50, 65, 35, 62)        # spills towards the camera
    kept = outlines.keep_outlines([_outline(0.86, with_shadow, 16.2), _outline(0.66, piece, 13.4)])
    assert len(kept) == 1 and kept[0][1] is piece


def test_separate_objects_are_each_kept():
    a, b = _box(10, 30, 10, 30), _box(60, 80, 60, 80)
    assert len(outlines.keep_outlines([_outline(0.9, a, 12.0), _outline(0.8, b, 12.0)])) == 2


def test_a_scrap_of_an_object_or_a_doubtful_outline_does_not_replace_it():
    piece = _box(20, 60, 40, 60)
    scrap = _box(20, 28, 45, 55)                        # its top only: under 40% of it
    doubtful = _box(25, 55, 42, 58)                     # most of it, but scored 0.3
    kept = outlines.keep_outlines([_outline(0.9, piece, 12.0), _outline(0.7, scrap, 6.0),
                                   _outline(0.3, doubtful, 11.0)])
    assert len(kept) == 1 and kept[0][1] is piece


def test_the_top_of_a_tall_object_does_not_replace_its_outline():
    # 2026-11-04: the d1 queen's outline (foot on d1) was replaced by her head's,
    # whose "foot" read 23 mm off, past the board's edge — and the scan lost her.
    # A shadow-free outline of the piece reads its foot within a few mm.
    piece = _box(20, 60, 40, 60)
    head = _box(20, 40, 41, 59)
    near = _box(20, 56, 41, 59)
    whole = (0.87, piece, SimpleNamespace(base_radius_mm=13.9, base_mm=np.array([100.0, 20.0])))
    kept = outlines.keep_outlines([whole, (0.63, head, SimpleNamespace(base_radius_mm=11.0,
                                                                       base_mm=np.array([100.0, -3.0])))])
    assert len(kept) == 1 and kept[0][1] is piece
    kept = outlines.keep_outlines([whole, (0.63, near, SimpleNamespace(base_radius_mm=12.0,
                                                                       base_mm=np.array([101.0, 16.0])))])
    assert len(kept) == 1 and kept[0][1] is near
    # A sliver of the piece is replaced by its whole foot, however far that moves it.
    sliver = (0.85, piece, SimpleNamespace(base_radius_mm=4.0, base_mm=np.array([100.0, 20.0])))
    kept = outlines.keep_outlines([sliver, (0.7, near, SimpleNamespace(base_radius_mm=10.5,
                                                                       base_mm=np.array([100.0, -1.0])))])
    assert kept[0][1] is near


def test_one_outline_round_two_objects_is_split_into_theirs():
    # 2026-11-04: one outline took in the f1 bishop and the f2 pawn beside it.
    left, right = _box(20, 50, 10, 40), _box(20, 50, 50, 80)
    both = left | right | _box(30, 40, 40, 50)
    kept = outlines.keep_outlines([
        (0.79, both, SimpleNamespace(base_radius_mm=8.5, base_mm=np.array([45.0, 0.0]))),
        (0.76, left, SimpleNamespace(base_radius_mm=12.0, base_mm=np.array([25.0, 0.0]))),
        (0.72, right, SimpleNamespace(base_radius_mm=11.0, base_mm=np.array([65.0, 0.0])))])
    assert len(kept) == 2 and {id(k[1]) for k in kept} == {id(left), id(right)}
    # One object's parts, their feet close together, are not two objects: one is kept.
    kept = outlines.keep_outlines([
        (0.79, both, SimpleNamespace(base_radius_mm=14.0, base_mm=np.array([45.0, 0.0]))),
        (0.76, left, SimpleNamespace(base_radius_mm=12.0, base_mm=np.array([40.0, 0.0]))),
        (0.72, right, SimpleNamespace(base_radius_mm=11.0, base_mm=np.array([50.0, 0.0])))])
    assert len(kept) == 1


def test_an_outlines_middle_stays_off_its_edge():
    m = _box(20, 60, 20, 60)
    middle = outlines.core(m)
    assert middle.sum() < m.sum() and middle[40, 40] and not middle[21, 21]
    tiny = _box(10, 13, 10, 13)
    assert (outlines.core(tiny) == tiny).all()          # too small to have a middle: all of it


def test_a_colour_is_placed_between_two_references_by_colour_then_lightness():
    maple, walnut = (230, 133, 124), (125, 151, 118)
    assert tones.between(maple, walnut, maple) == pytest.approx(0.0)
    assert tones.between(walnut, walnut, maple) == pytest.approx(1.0)
    # Lit lighter than the dark squares, but the brown wood's colour (2026-10-27).
    assert tones.between((168, 150, 120), walnut, maple) > 0.65
    # Grey references: lightness decides; too alike: not said.
    assert tones.between((40, 128, 128), (60, 128, 128), (200, 128, 128)) > 1.0
    assert tones.between((125, 128, 128), (120, 128, 128), (130, 128, 128)) is None


def test_a_photo_of_the_empty_surface_is_moved_to_where_the_surface_is_now():
    yy, xx = np.mgrid[0:200, 0:200]
    board = (((yy // 25 + xx // 25) % 2) * 255).astype(np.uint8)
    photo = np.dstack([board] * 3)
    corners = np.array([[x, y] for y in range(25, 200, 25) for x in range(25, 200, 25)], float)
    shift = np.float32([[1, 0, 3], [0, 1, -2]])            # the camera moved: 3 px right, 2 up
    now = cv2.warpAffine(photo, shift, (200, 200), borderMode=cv2.BORDER_REPLICATE)
    lined_up, moved = surface_scan.realign_reference(photo, corners, corners + [3, -2])
    assert moved == pytest.approx(np.hypot(3, 2), abs=0.01)
    assert np.abs(lined_up[20:180, 20:180].astype(int) - now[20:180, 20:180].astype(int)).mean() < 2
    assert surface_scan.realign_reference(photo, corners[:3], corners[:3]) is None


def test_the_light_is_read_area_by_area():
    view = np.full((100, 100, 3), 120, np.uint8)
    view[:50, :50] = 255
    stats = lighting.region_stats(view, {"washed": (25, 25), "fine": (75, 75), "off": (500, 500)}, 10)
    assert stats["washed"][1] == pytest.approx(1.0) and stats["fine"][1] == 0.0 and "off" not in stats
    assert stats["fine"][0] < stats["washed"][0]
