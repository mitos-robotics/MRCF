"""config.approach_tilt: the chess deployment's approach rules, by rank from the arm's side."""

from deployments.chess import config


def test_rules_go_by_rank_and_squares_without_one_get_none(monkeypatch):
    monkeypatch.setattr(config, "APPROACH_RULES", [{"ranks": (6, 8), "tilt_deg": 30},
                                                   {"ranks": (1, 2), "tilt_deg": 20}])
    monkeypatch.setattr(config, "ROBOT_PLAYS", "white")
    assert [config.approach_tilt(s) for s in ("e6", "H8", "a1", "c2")] == [30.0, 30.0, 20.0, 20.0]
    assert [config.approach_tilt(s) for s in ("e3", "d5", None, "white_pawn_e6", "spot")] == [None] * 5


def test_playing_black_the_rules_still_count_from_the_arms_side(monkeypatch):
    monkeypatch.setattr(config, "APPROACH_RULES", [{"ranks": (6, 8), "tilt_deg": 30}])
    monkeypatch.setattr(config, "ROBOT_PLAYS", "black")
    assert config.approach_tilt("e2") == 30.0 and config.approach_tilt("e7") is None


def test_as_shipped_only_the_far_ranks_have_one_the_arms_approaches_decide_the_rest(monkeypatch):
    # near_base / top_down / reach_out (mrcf/policies/approaches.py) choose the
    # lean by the square's distance from the base; a rule here overrides it.
    # Ranks 7-8 have one: at the arm's edge the least lean stalled (2026-11-04).
    monkeypatch.setattr(config, "ROBOT_PLAYS", "white")
    assert [config.approach_tilt(f + r) for f in "ae" for r in "78"] == [55.0] * 4
    assert all(config.approach_tilt(f + r) is None for f in "abcdefgh" for r in "123456")


def test_a_rule_can_name_files(monkeypatch):
    monkeypatch.setattr(config, "APPROACH_RULES", [{"ranks": (1, 2), "files": "cdef", "tilt_deg": 20}])
    monkeypatch.setattr(config, "ROBOT_PLAYS", "white")
    assert config.approach_tilt("d2") == 20.0 and config.approach_tilt("b2") is None
