"""
The catalogue of a deployment's kinds of object (mrcf/catalogue.py): matched
by what the fixed camera sees, not by name; how each rests; the grip that held.
"""

import numpy as np
import pytest

from mrcf import catalogue as cat

FIGURE = {"foot_mm": 22.0, "long_mm": 24.0, "short_mm": 22.0, "L": 60.0, "a": 30.0, "b": 20.0}
LYING = {"foot_mm": 22.0, "long_mm": 44.0, "short_mm": 22.0, "L": 90.0, "a": 25.0, "b": 25.0}
PLUG = {"foot_mm": 51.0, "long_mm": 55.0, "short_mm": 47.0, "L": 226.0, "a": 4.0, "b": -11.0}


class _View:
    """A camera straight down, 1 px a mm."""
    def to_plane(self, px):
        return np.asarray(px, float)


def test_its_signature_is_its_outline_on_the_surface_and_its_colour():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:54, 30:52] = 1                                   # 44 x 22 px: lying
    s = cat.signature(mask, _View(), 22.0, {"L": 90.0, "a": 25.0, "b": 25.0})
    assert s["long_mm"] == pytest.approx(43, abs=2) and s["short_mm"] == pytest.approx(21, abs=2)
    assert s["foot_mm"] == 22.0 and s["a"] == 25.0


def test_a_sighting_is_matched_by_how_it_looks_not_what_it_is_called(tmp_path):
    c = cat.Catalogue(tmp_path / "cat.json")
    figure, _ = c.add("bottle cap", "red", FIGURE)
    plug, _ = c.add("plug", "white", PLUG)
    seen = dict(FIGURE, foot_mm=23.0, long_mm=25.5, a=27.0)           # the next scan: a little different
    k, p, _ = c.match(seen)
    assert k is figure and p.name == "upright"
    assert c.match(dict(PLUG, foot_mm=52.0))[0] is plug
    assert c.match(dict(FIGURE, foot_mm=35.0)) is None                # bigger: something else
    # Colour weighs, but does not rule out: the twin figure read a 41 b 12 against 14 / 5 (2026-11-18).
    assert c.match(dict(FIGURE, L=98.0, a=41.0, b=12.0))[0] is figure
    blue, _ = c.add("block", "blue", dict(FIGURE, a=-10.0, b=-30.0))
    assert c.match(dict(FIGURE, a=-9.0, b=-29.0))[0] is blue           # alike in size: the nearer colour


def test_the_grip_that_held_is_the_one_picked_by_and_it_is_kept(tmp_path):
    c = cat.Catalogue(tmp_path / "cat.json")
    plug, _ = c.add("plug", "white", PLUG)
    c.learn(plug, "upright", PLUG, grip_at_m=0.021, top_m=0.076, held=False)
    c.learn(plug, "upright", PLUG, grip_at_m=0.016, top_m=0.060, held=True)
    c.learn(plug, "upright", PLUG, grip_at_m=0.0165, top_m=0.062, held=True)
    best = plug.pose("upright").best()
    assert best["grip_at_m"] == pytest.approx(0.016) and best["held"] == 2 and best["top_m"] == pytest.approx(0.062)
    again = cat.Catalogue(tmp_path / "cat.json")                       # a restart
    k, p, _ = again.match(PLUG)
    assert k.name == "plug" and p.best()["held"] == 2


def test_a_new_way_it_rests_is_its_own_pose_with_its_own_grip(tmp_path):
    c = cat.Catalogue(tmp_path / "cat.json")
    figure, _ = c.add("figure", "red", FIGURE)
    c.learn(figure, "upright", FIGURE, grip_at_m=0.016, top_m=0.06, held=True)
    c.learn(figure, "lying", LYING, grip_at_m=0.011, top_m=0.03, held=True)
    k, p, _ = c.match(dict(LYING, long_mm=46.0))
    assert k is figure and p.name == "lying" and p.best()["grip_at_m"] == pytest.approx(0.011)
    assert c.match(FIGURE)[1].name == "upright"
    assert figure.pose("upright").best()["grip_at_m"] == pytest.approx(0.016)


def test_none_has_held_yet_no_grip_is_offered(tmp_path):
    c = cat.Catalogue(tmp_path / "cat.json")
    plug, pose = c.add("plug", "white", PLUG)
    c.learn(plug, "upright", PLUG, grip_at_m=0.021, top_m=0.076, held=False)
    assert pose.best() is None


def test_one_object_is_one_kind_and_its_sizes_are_kept_as_the_range_it_was_seen_at(tmp_path):
    """2026-11-19: the scoop kept twice (its better grip under the other); the plug read 42-57 mm across."""
    import json

    path = tmp_path / "cat.json"
    c = cat.Catalogue(path)
    one, _ = c.add("scoop", "blue", dict(PLUG, foot_mm=91.0, long_mm=120.0, short_mm=72.0))
    c.learn(one, "upright", one.poses[0].sig, grip_at_m=0.031, top_m=0.06, held=True)
    c.kinds.append(cat.Kind("kind-9", "scoop", "blue", [cat.Pose("upright", dict(PLUG, foot_mm=110.0, long_mm=126.0,
                                                                                   short_mm=75.0),
                                                                grips=[{"grip_at_m": 0.041, "top_m": 0.06,
                                                                        "held": 1, "missed": 0}])]))
    c.save()
    again = cat.Catalogue(path)                                  # read back: the two are one
    assert [k.name for k in again.kinds] == ["scoop"]
    pose = again.kinds[0].pose("upright")
    assert pose.best()["grip_at_m"] == pytest.approx(0.031) and len(pose.grips) == 2
    assert pose.span["foot_mm"] == [91.0, 110.0]
    assert again.match(dict(PLUG, foot_mm=100.0, long_mm=123.0, short_mm=73.0))[0].name == "scoop"   # in between
    assert json.loads(path.read_text())["kinds"][0]["name"] == "scoop"

    plug, p = again.add("plug", "white", PLUG)
    for foot in (42.0, 57.0):
        again.learn(plug, "upright", dict(PLUG, foot_mm=foot), grip_at_m=0.018, top_m=0.07, held=True)
    assert again.match(dict(PLUG, foot_mm=44.0))[0] is plug                                      # seen as small
    same, _ = again.add("Plug", "white", dict(PLUG, foot_mm=30.0), "on its back")
    assert same is plug and plug.pose("on its back") is not None                                # a new way it rests
    assert again.named("plug", "white") is plug and again.named("plug", "red") is None
    assert plug.longest_mm() == pytest.approx(55.0)
