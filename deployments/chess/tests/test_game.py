"""
A game of chess against the arm (game.py), offline: play.py's simulated board
stands in for the arm and the camera, a scripted engine for Stockfish, and
the person's moves are pieces moved on the simulated board by hand.
"""

import json
from types import SimpleNamespace

import pytest

try:
    import cv2  # noqa: F401 — the deployment's perception needs it
except ImportError:
    pytest.skip("needs OpenCV", allow_module_level=True)

from mrcf.interlock import Interlock
from mrcf.vlm import SceneState

from deployments.chess import chessboard, engine, game, play, rules
from deployments.chess import config as cfg
from test_play import Board, centre


class FakeEngine:
    """Plays the scripted moves in turn; with none left, the first move it is allowed (searchmoves)."""

    def __init__(self, script=()):
        self.script = list(script)
        self.calls: list = []
        self.elo = None
        self.score = {"cp": 20}
        self.name = "FakeFish"
        self.options: dict = {}

    def set_elo(self, elo):
        self.elo = int(elo)
        return self.elo

    def set_option(self, *_):
        pass

    def best_move(self, moves, movetime_ms=None, searchmoves=None, start_fen=None):
        self.calls.append({"moves": list(moves), "searchmoves": searchmoves, "start_fen": start_fen})
        if self.script:
            return self.script.pop(0)
        return searchmoves[0] if searchmoves else None


@pytest.fixture
def table(monkeypatch, tmp_path):
    def setup(position=None, script=(), bounds=None):
        position = position or rules.Position.start()
        b = Board(position)
        squares = {sq: {"world": list(centre(sq)), "pixel": [0.0, 0.0]} for sq in rules.SQUARES}
        monkeypatch.setattr(chessboard, "LAST_SQUARES", squares)
        monkeypatch.setattr(play, "look", b.look)
        monkeypatch.setattr(play, "_act", b.act)
        monkeypatch.setattr(play, "_holding", lambda env: b.held is not None)
        monkeypatch.setattr(play, "POLICIES", SimpleNamespace(home=lambda env: None, holding=lambda env: b.held))
        monkeypatch.setattr(play.board_log, "record_move", lambda **kw: b.__dict__.setdefault("log", []).append(kw))
        monkeypatch.setattr(cfg, "WORKSPACE_BOUNDS", bounds or
                            SimpleNamespace(x_min=0.05, x_max=0.45, y_min=-0.25, y_max=0.25))
        eng = FakeEngine(script)
        monkeypatch.setattr(engine, "shared", lambda: eng)
        monkeypatch.setattr(game, "SAVE_PATH", tmp_path / "game.json")
        monkeypatch.setattr(game, "GAME", None)
        state = SceneState()
        b.look(None, state)
        return b, state, eng
    return setup


def by_hand(b: Board, frm: str, to: str) -> None:
    """The person moves a piece from frm to to; what stood on to is taken off the board."""
    b.at.pop(to, None)
    b.at[to] = centre(to)
    b.at.pop(frm)


def sans() -> list:
    return [p["san"] for p in game.GAME.plies]


def mid_game(table, fen: str, script=()):
    """A game already under way at fen, the arm white, the board set up so."""
    pos = rules.Position.from_fen(fen)
    b, state, eng = table(pos, script)
    game.GAME = game.Game(arm="white", start_fen=pos.fen())
    return b, state, eng


# ── Starting, and the moves going back and forth ───────────────────────────────

def test_a_new_game_checks_the_board_and_the_arm_opens(table):
    b, state, eng = table(script=["e2e4"])
    assert game.skill_new_game(env=None, scene_state=state) is True
    assert sans() == ["e4"] and game.GAME.plies[0]["by"] == "arm"
    assert "e4" in b.occupied() and "e2" not in b.occupied()
    assert eng.elo == cfg.ENGINE_ELO
    saved = json.loads(game.SAVE_PATH.read_text())
    assert saved["plies"][0]["uci"] == "e2e4" and saved["arm"] == "white"


def test_a_new_game_refuses_a_board_not_at_the_start(table):
    b, state, _ = table(rules.Position.start().play(rules.Move("e2", "e4")))
    assert game.skill_new_game(env=None, scene_state=state) is True        # waits for the person
    assert game.GAME is None


def test_the_persons_move_is_found_by_a_look_and_the_arm_replies(table):
    b, state, eng = table(script=["e2e4", "g1f3"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    assert game.skill_my_move(env=None, scene_state=state) is True
    assert sans() == ["e4", "e5", "Nf3"]
    assert game.GAME.plies[1] == {"uci": "e7e5", "san": "e5", "by": "person", "how": "hand"}
    assert "f3" in b.occupied() and "g1" not in b.occupied()
    assert eng.calls[-1]["moves"] == ["e2e4", "e7e5"]                       # the whole game, every time


def test_a_said_move_is_believed_only_once_the_board_shows_it(table):
    b, state, _ = table(script=["e2e4", "g1f3"])
    game.skill_new_game(env=None, scene_state=state)
    game.skill_my_move(env=None, scene_state=state, move="e5")               # not made yet
    assert sans() == ["e4"]
    game.skill_my_move(env=None, scene_state=state, move="Ke2")              # not legal
    assert sans() == ["e4"]
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state, move="e5")
    assert sans() == ["e4", "e5", "Nf3"]


def test_with_nothing_moved_it_says_so_and_waits(table):
    b, state, eng = table(script=["e2e4"])
    game.skill_new_game(env=None, scene_state=state)
    assert game.skill_my_move(env=None, scene_state=state) is True
    assert sans() == ["e4"] and len(eng.calls) == 1


def test_a_look_that_misses_a_far_piece_still_finds_the_move(table, monkeypatch):
    b, state, _ = table(script=["e2e4", "g1f3"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    real = b.look

    def missing_a1(env, scene_state):           # the overhead camera misses a piece now and then
        real(env, scene_state)
        scene_state.objects = [o for o in scene_state.objects if o.square != "a1"]
        return True
    monkeypatch.setattr(play, "look", missing_a1)
    game.skill_my_move(env=None, scene_state=state)
    assert sans()[:2] == ["e4", "e5"]


# ── Captures, castling, and what the squares cannot tell apart ─────────────────

def test_captures_on_both_sides(table):
    b, state, _ = table(script=["e2e4", "b1c3", "c3e4"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "d7", "d5")
    game.skill_my_move(env=None, scene_state=state)
    by_hand(b, "d5", "e4")                                  # the person takes the arm's pawn off by hand
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["e4", "d5", "Nc3", "dxe4", "Nxe4"]
    assert ("e4", cfg.CAPTURE_TARGET) in b.carried         # the arm takes the pawn off first
    assert len(b.off_board) == 1 and len(game.GAME.captured) == 1
    assert game.GAME.position().pieces["e4"] == "N"


def test_castling_is_found_from_the_four_squares(table):
    b, state, _ = mid_game(table, "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
                           script=["e1g1"])
    by_hand(b, "e8", "g8")
    by_hand(b, "h8", "f8")
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["O-O", "O-O"]
    assert {"g1", "f1", "g8", "f8"} <= b.occupied() and not {"e1", "h1", "e8", "h8"} & b.occupied()


def test_a_capture_the_squares_cannot_tell_from_another_is_asked_about(table):
    b, state, _ = mid_game(table, "rnbqkb1r/ppp2ppp/5n2/3P4/4P3/8/PPP2PPP/RNBQKBNR b KQkq - 0 1",
                           script=["d1e2"])
    by_hand(b, "f6", "e4")
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == []                                     # Nxe4 or Nxd5: asked, not guessed
    game.skill_my_move(env=None, scene_state=state, move="Nxe4")
    assert sans() == ["Nxe4", "Qe2"]


# ── What the arm cannot do ────────────────────────────────────────────────────

def test_moves_out_of_reach_are_not_offered_to_the_engine(table):
    near_files = SimpleNamespace(x_min=0.05, x_max=0.45, y_min=-0.05, y_max=0.25)   # f, g, h files out of reach
    b, state, eng = table(bounds=near_files)
    game.skill_new_game(env=None, scene_state=state)
    offered = eng.calls[0]["searchmoves"]
    assert offered and not any(set(u[0] + u[2]) & set("fgh") for u in offered)
    assert game.GAME.plies[0]["uci"] == offered[0]


def test_a_move_the_arm_cannot_make_is_left_to_the_person_and_checked_on_done(table):
    b, state, _ = table(script=["e2e4", "g1f3"])
    b.fail[("pick", "e2")] = 99
    game.skill_new_game(env=None, scene_state=state)
    assert game.GAME.pending == "e2e4" and sans() == []
    assert "e2" in b.occupied()                             # the board as it was
    game.skill_my_move(env=None, scene_state=state)         # not made yet: still waiting
    assert game.GAME.pending == "e2e4"
    by_hand(b, "e2", "e4")
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["e4"] and game.GAME.plies[0]["how"] == "hand" and game.GAME.pending is None
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["e4", "e5", "Nf3"]


def test_the_arms_move_and_the_persons_made_together_are_both_found(table):
    b, state, _ = table(script=["e2e4", "g1f3"])
    b.fail[("pick", "e2")] = 99
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e2", "e4")
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["e4", "e5", "Nf3"]


def test_a_board_left_wrong_is_set_up_by_the_person_and_checked(table):
    b, state, _ = table(script=["e2e4"])
    b.drop["e4"] = "h5"                                     # slips out and rolls far off
    game.skill_new_game(env=None, scene_state=state)
    assert game.GAME.pending == "e2e4" and game.GAME.check_board
    game.skill_my_move(env=None, scene_state=state)         # still on h5: not yet
    assert game.GAME.pending == "e2e4"
    b.at.pop("h5")
    b.at["e4"] = centre("e4")
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["e4"] and not game.GAME.check_board


# ── The end of a game ──────────────────────────────────────────────────────────

def test_checkmate_ends_the_game(table):
    b, state, eng = table(script=["f2f3", "g2g4"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state)
    by_hand(b, "d8", "h4")
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["f3", "e5", "g4", "Qh4#"]
    assert game.GAME.result == "0-1" and "checkmate" in game.GAME.reason
    assert len(eng.calls) == 2                              # no search after the mate
    game.skill_my_move(env=None, scene_state=state)         # over: nothing more happens
    assert len(game.GAME.plies) == 4


def test_draws_by_material_and_by_repetition():
    bare = game.Game(start_fen=rules.Position.from_fen("8/8/8/8/8/8/8/K6k w - -").fen())
    assert game.over(bare)[0] == "1/2-1/2"
    g = game.Game()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        assert game.over(g) is None
        g.plies.append({"uci": uci, "san": "", "by": "", "how": ""})
    assert game.over(g) == ("1/2-1/2", "the same position for the third time — a draw")


def test_the_person_can_resign(table):
    b, state, _ = table(script=["e2e4"])
    game.skill_new_game(env=None, scene_state=state)
    game.skill_resign()
    assert game.GAME.result == "1-0"


# ── Saving, carrying on, showing ───────────────────────────────────────────────

def test_a_saved_game_is_carried_on_after_a_restart(table):
    b, state, _ = table(script=["e2e4", "g1f3"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state)
    before = list(game.GAME.plies)
    game.GAME = None                                        # the controller restarted
    game.skill_resume_game(env=None, scene_state=state)
    assert game.GAME.plies == before
    assert game.GAME.position() == rules.Position.start().play(rules.Move("e2", "e4")).play(
        rules.Move("e7", "e5")).play(rules.Move("g1", "f3"))


def test_a_saved_game_finds_a_board_that_differs_and_waits_for_it(table):
    b, state, _ = table(script=["e2e4"])
    game.skill_new_game(env=None, scene_state=state)
    game.GAME = None
    b.at["e2"] = b.at.pop("e4")                             # someone put the pawn back
    game.skill_resume_game(env=None, scene_state=state)
    assert game.GAME.check_board
    b.at["e4"] = b.at.pop("e2")
    by_hand(b, "e7", "e5")                                  # set up, and the person's move made too
    game.skill_my_move(env=None, scene_state=state)
    assert sans()[:2] == ["e4", "e5"]


def test_showing_the_game_and_a_hint(table, capsys):
    b, state, eng = table(script=["e2e4", "e7e5"])
    game.skill_new_game(env=None, scene_state=state)
    capsys.readouterr()
    game.skill_game()
    shown = capsys.readouterr().out
    assert "1. e4" in shown and "    8 r n b q k b n r 8" in shown and "    4 . . . . P . . . 4" in shown
    game.skill_hint()
    assert "The engine would play e5" in capsys.readouterr().out
    assert "person's move" in game.state_line()


def test_board_text_has_white_at_the_bottom():
    text = game.board_text(rules.Position.start()).splitlines()
    assert text[1] == "    8 r n b q k b n r 8" and text[8] == "    1 R N B Q K B N R 1"


def test_explain_tells_the_move_from_the_squares():
    pos = rules.Position.start()
    seen = (pos.occupied() - {"g1"}) | {"f3"}
    moves, exact = game.explain(pos, seen)
    assert exact and moves == [rules.Move("g1", "f3")]


def test_slash_commands_reach_the_game_through_the_controller(table):
    from mrcf import controller
    b, state, _ = table(script=["e2e4", "g1f3"])
    env = SimpleNamespace(is_viewer_running=lambda: True, on_next_move=lambda announce: None, interlock=Interlock(),
                          contact_force_limits={})
    controller.direct_command("/new_game white 1600", env, state, controller.Deployment())
    assert sans() == ["e4"] and game.GAME.elo == 1600
    by_hand(b, "e7", "e5")
    controller.direct_command("/my_move e5", env, state, controller.Deployment())
    assert sans() == ["e4", "e5", "Nf3"]
    controller.direct_command("/game", env, state, controller.Deployment())


@pytest.mark.skipif(__import__("shutil").which("stockfish") is None, reason="Stockfish not installed")
def test_a_few_moves_against_stockfish(table, monkeypatch):
    b, state, _ = table()
    monkeypatch.delenv("MRCF_CHESS_ENGINE", raising=False)
    monkeypatch.setattr(cfg, "ENGINE_MOVETIME_MS", 50)
    real = engine.Engine()
    monkeypatch.setattr(engine, "shared", lambda: real)
    try:
        game.skill_new_game(env=None, scene_state=state, strength=1400)
        for _ in range(10):
            if game.GAME.finished:
                break
            pos = game.GAME.position()
            assert not game.GAME.pending, game.GAME.pending
            # The person plays the first move the squares alone tell apart.
            mine = next(m for m in sorted(pos.legal_moves(), key=pos.san)
                        if game.explain(pos, pos.play(m).occupied())[0] == [m])
            after = pos.play(mine).occupied()           # castling and en passant included
            for sq in set(b.at) - after:
                b.at.pop(sq)
            for sq in after - set(b.at):
                b.at[sq] = centre(sq)
            game.skill_my_move(env=None, scene_state=state)
        assert len(game.GAME.plies) >= 20 or game.GAME.finished
        assert b.occupied() == game.GAME.position().occupied()
    finally:
        real.close()


# ── A board set up again, read without being told ─────────────────────────────

def set_up_again(b: Board) -> None:
    """The person puts every piece back on its starting square."""
    b.at = {sq: centre(sq) for sq in rules.Position.start().occupied()}


def test_done_on_a_board_set_up_again_mid_game_begins_a_new_one(table):
    b, state, eng = table(script=["e2e4", "g1f3", "d2d4"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["e4", "e5", "Nf3"]
    set_up_again(b)
    assert game.skill_my_move(env=None, scene_state=state) is True
    assert sans() == ["d4"]                          # a new game, and the arm's first move in it
    assert json.loads(game.SAVE_PATH.read_text())["plies"][0]["uci"] == "d2d4"


def test_done_with_no_game_on_a_set_up_board_begins_one(table):
    b, state, eng = table(script=["e2e4"])
    assert game.GAME is None
    assert game.skill_my_move(env=None, scene_state=state) is True
    assert sans() == ["e4"] and "e4" in b.occupied()


def test_done_with_no_game_and_pieces_not_set_up_waits(table):
    b, state, eng = table(rules.Position.start().play(rules.Move("e2", "e4")))
    assert game.skill_my_move(env=None, scene_state=state) is True
    assert game.GAME is None and eng.calls == []


def test_a_knight_gone_home_is_a_move_not_a_board_set_up_again(table):
    # 1. Nf3 Nf6 2. Ng1 Ng8: every piece home again, by a legal move — the game goes on.
    b, state, eng = table(script=["g1f3", "f3g1", "e2e4"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "g8", "f6")
    game.skill_my_move(env=None, scene_state=state)
    by_hand(b, "f6", "g8")
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["Nf3", "Nf6", "Ng1", "Ng8", "e4"]


def test_the_arms_move_owed_by_hand_and_the_board_set_up_again_begins_a_new_game(table):
    b, state, eng = table(script=["e2e4", "d2d4"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    game._commit(game.GAME, state, rules.Move("e7", "e5"), "person", "hand")
    game.GAME.pending = "g1f3"                       # the arm could not make it: the person is to
    set_up_again(b)
    game.skill_my_move(env=None, scene_state=state)
    assert game.GAME.pending is None and sans() == ["d4"]


def test_one_move_in_a_board_back_at_the_start_is_that_move_put_back_not_a_new_game(table):
    b, state, eng = table(script=["e2e4", "d2d4"])
    game.skill_new_game(env=None, scene_state=state)
    set_up_again(b)                                  # the e4 pawn back on e2
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["e4"] and len(eng.calls) == 1


def test_resume_on_a_board_set_up_again_begins_a_new_game(table):
    b, state, eng = table(script=["e2e4", "g1f3", "c2c4"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state)
    game.GAME = None                                  # a restart
    set_up_again(b)
    game.skill_resume_game(env=None, scene_state=state)
    assert sans() == ["c4"] and not game.GAME.check_board


def _restarted(b: Board, state) -> None:
    game.GAME = None
    b.look(None, state)                               # the controller's first look


def test_at_start_the_saved_game_is_carried_on_when_the_board_shows_it(table):
    b, state, eng = table(script=["e2e4", "g1f3"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state)
    calls, carried = len(eng.calls), len(b.carried)
    _restarted(b, state)
    game.on_start(None, state)
    assert sans() == ["e4", "e5", "Nf3"]
    assert len(eng.calls) == calls and len(b.carried) == carried      # nothing moved
    by_hand(b, "b8", "c6")                           # and play goes on with done
    eng.script.append("f1b5")
    game.skill_my_move(env=None, scene_state=state)
    assert sans()[-2:] == ["Nc6", "Bb5"]


def test_at_start_the_persons_move_made_meanwhile_still_carries_the_game_on(table):
    b, state, eng = table(script=["e2e4", "g1f3"])
    game.skill_new_game(env=None, scene_state=state)
    _restarted(b, state)
    by_hand(b, "e7", "e5")
    b.look(None, state)
    game.on_start(None, state)
    assert sans() == ["e4"]
    game.skill_my_move(env=None, scene_state=state)
    assert sans() == ["e4", "e5", "Nf3"]


def test_at_start_a_board_set_up_for_a_new_game_begins_one_and_waits_for_a_go(table):
    b, state, eng = table(script=["e2e4", "g1f3", "d2d4"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state)
    set_up_again(b)
    _restarted(b, state)
    calls = len(eng.calls)
    game.on_start(None, state)
    assert game.GAME is not None and game.GAME.plies == [] and game.GAME.arm == "white"
    assert len(eng.calls) == calls and b.occupied() == rules.Position.start().occupied()   # no move yet
    game.skill_my_move(env=None, scene_state=state)                                        # "go"
    assert sans() == ["d4"]


def test_at_start_a_board_like_neither_leaves_the_saved_game_for_resume(table):
    b, state, eng = table(script=["e2e4", "g1f3"])
    game.skill_new_game(env=None, scene_state=state)
    by_hand(b, "e7", "e5")
    game.skill_my_move(env=None, scene_state=state)
    _restarted(b, state)
    b.at.pop("a2"), b.at.pop("h7"), b.at.pop("d1")
    b.look(None, state)
    game.on_start(None, state)
    assert game.GAME is None


def test_set_up_for_a_new_game_forgives_a_missed_piece_or_two_but_not_one_out():
    start = rules.Position.start().occupied()
    assert game.set_up_for_new_game(start)
    assert game.set_up_for_new_game(start - {"e1", "d8"})
    assert not game.set_up_for_new_game(start - {"e1", "d8", "a2"})
    assert not game.set_up_for_new_game((start - {"e2"}) | {"e4"})
