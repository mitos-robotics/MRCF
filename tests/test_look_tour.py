"""tools/look_tour.py's own choices, offline: which objects it visits, and what moved while it looked."""

from types import SimpleNamespace

import numpy as np

from tools import look_tour


def _obj(name, xy, square=None):
    return SimpleNamespace(name=name, position=np.array([*xy, 0.0]), square=square)


def test_with_none_named_it_spreads_its_visits_from_the_one_nearest_the_arm():
    row = [_obj(f"p{i}", (0.15 + 0.04 * i, 0.0)) for i in range(7)]
    chosen = look_tour.spread(row, 3, np.zeros(2))
    assert [o.name for o in chosen] == ["p0", "p6", "p3"]


def test_named_objects_are_found_by_name_or_by_where_they_stand():
    state = SimpleNamespace(objects=[_obj("white_piece_e2", (0.18, -0.02), "e2"), _obj("black_piece_e7", (0.38, 0.0), "e7")])
    got = look_tour.pick_objects(state, "e7,white_piece_e2,z9", 10, np.zeros(2))
    assert [o.name for o in got] == ["black_piece_e7", "white_piece_e2"]


def test_a_piece_whose_scan_moved_between_before_and_after_is_named():
    before = [{"track_id": "a", "place": "e2", "name": "x", "xy_m": [0.18, 0.0]},
              {"track_id": "b", "place": "d2", "name": "y", "xy_m": [0.18, 0.04]}]
    after = [{"track_id": "a", "place": "e2", "name": "x", "xy_m": [0.181, 0.0]},
             {"track_id": "b", "place": "d2", "name": "y", "xy_m": [0.17, 0.04]}]
    assert look_tour.moved_objects({"scan_before": before, "scan_after": after}) == ["d2 10 mm"]
