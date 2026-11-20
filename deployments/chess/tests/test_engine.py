"""The chess engine client (engine.py): a fake UCI engine for the protocol, and Stockfish itself when installed."""

import shutil
import sys
import textwrap

import pytest

from deployments.chess import engine, rules

FAKE = textwrap.dedent('''
    import sys
    log = open(sys.argv[1], "a") if len(sys.argv) > 1 else None
    hang = len(sys.argv) > 2 and sys.argv[2] == "hang"
    twice = len(sys.argv) > 2 and sys.argv[2] == "twice"     # a stray second answer after each search
    for line in sys.stdin:
        line = line.strip()
        if log:
            log.write(line + "\\n"); log.flush()
        if line == "uci":
            print("id name FakeFish 1")
            print("option name Hash type spin default 16 min 1 max 1024")
            print("option name UCI_LimitStrength type check default false")
            print("option name UCI_Elo type spin default 1320 min 1320 max 3190")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line.startswith("go"):
            if hang:
                continue
            words = line.split()
            best = words[words.index("searchmoves") + 1] if "searchmoves" in words else "e2e4"
            print("info depth 1 score cp 31 pv " + best)
            print("info depth 2 score mate -2 pv " + best)
            print("bestmove " + best)
            if twice:
                print("bestmove a2a3")
        elif line == "quit":
            break
        sys.stdout.flush()
''')


@pytest.fixture
def fake(tmp_path):
    script = tmp_path / "fakefish.py"
    script.write_text(FAKE)
    log = tmp_path / "sent.txt"

    def start(*extra, **kw):
        return engine.Engine(command=[sys.executable, str(script), str(log), *extra], **kw)
    start.log = log
    return start


def test_handshake_reads_the_name_and_options(fake):
    eng = fake()
    try:
        assert eng.name == "FakeFish 1"
        assert eng.options["UCI_Elo"] == {"type": "spin", "default": "1320", "min": "1320", "max": "3190"}
        assert eng.alive()
    finally:
        eng.close()
    assert not eng.alive()


def test_strength_is_clamped_to_what_the_engine_offers(fake):
    eng = fake()
    try:
        assert eng.set_elo(800) == 1320
        assert eng.set_elo(1800) == 1800
    finally:
        eng.close()
    sent = fake.log.read_text()
    assert "setoption name UCI_LimitStrength value true" in sent
    assert "setoption name UCI_Elo value 1800" in sent


def test_best_move_sends_the_whole_game_and_keeps_to_searchmoves(fake):
    eng = fake()
    try:
        assert eng.best_move([], movetime_ms=10) == "e2e4"
        assert eng.best_move(["e2e4", "e7e5"], movetime_ms=10, searchmoves=["g1f3", "b1c3"]) == "g1f3"
        assert eng.score == {"mate": -2}                     # the last score it gave
    finally:
        eng.close()
    sent = fake.log.read_text().splitlines()
    assert "position startpos moves e2e4 e7e5" in sent
    assert "go movetime 10 searchmoves g1f3 b1c3" in sent


def test_a_position_from_fen_is_sent_as_one(fake):
    eng = fake()
    try:
        eng.best_move(["e7e5"], movetime_ms=10, start_fen="8/8/8/8/8/8/8/K6k b - - 0 1")
    finally:
        eng.close()
    assert "position fen 8/8/8/8/8/8/8/K6k b - - 0 1 moves e7e5" in fake.log.read_text()


def test_an_engine_that_does_not_answer_raises_rather_than_hangs(fake, monkeypatch):
    monkeypatch.setattr(engine, "SEARCH_GRACE_S", 0.3)
    eng = fake("hang")
    try:
        with pytest.raises(engine.EngineError, match="bestmove"):
            eng.best_move([], movetime_ms=10)
    finally:
        eng.close()


def test_a_hung_engine_is_ended_so_the_next_game_move_starts_a_fresh_one(fake, monkeypatch):
    monkeypatch.setattr(engine, "SEARCH_GRACE_S", 0.3)
    eng = fake("hang")
    with pytest.raises(engine.EngineError):
        eng.best_move([], movetime_ms=10)
    eng._proc.wait(timeout=2)
    assert not eng.alive()


def test_what_the_engine_sent_before_a_search_is_not_taken_for_its_answer(fake):
    # A late bestmove read as this search's would be a move for another position.
    eng = fake("twice")
    try:
        assert eng.best_move([], movetime_ms=10) == "e2e4"
        assert eng.best_move(["e2e4", "e7e5"], movetime_ms=10, searchmoves=["g1f3"]) == "g1f3"
    finally:
        eng.close()


def test_no_engine_says_how_to_get_one(monkeypatch):
    monkeypatch.delenv("MRCF_CHESS_ENGINE", raising=False)
    monkeypatch.setattr(engine.cfg, "ENGINE_PATH", None, raising=False)
    monkeypatch.setattr(engine.shutil, "which", lambda name: None)
    monkeypatch.setattr(engine, "_SEARCH_PATHS", ())
    with pytest.raises(engine.EngineError, match="brew install stockfish"):
        engine.engine_path()


def test_the_environment_names_the_engine(monkeypatch, tmp_path):
    exe = tmp_path / "myengine"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("MRCF_CHESS_ENGINE", str(exe))
    assert engine.engine_path() == str(exe)
    monkeypatch.setenv("MRCF_CHESS_ENGINE", str(tmp_path / "nothing"))
    with pytest.raises(engine.EngineError, match="no chess engine at"):
        engine.engine_path()


def test_score_text_is_from_whites_side():
    assert engine.score_text({"cp": 35}, white_view=True) == "+0.35"
    assert engine.score_text({"cp": 35}, white_view=False) == "-0.35"
    assert engine.score_text({"mate": 3}, white_view=False) == "mate in 3 for black"
    assert engine.score_text(None, white_view=True) == ""


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish not installed")
def test_stockfish_plays_a_legal_move_and_only_one_it_is_offered(monkeypatch):
    monkeypatch.delenv("MRCF_CHESS_ENGINE", raising=False)
    eng = engine.Engine(elo=1500)
    try:
        assert eng.elo == 1500
        pos = rules.Position.start().play(rules.Move("e2", "e4"))
        best = eng.best_move(["e2e4"], movetime_ms=100)
        assert pos.parse(best) in pos.legal_moves()
        assert eng.best_move(["e2e4"], movetime_ms=100, searchmoves=["a7a6"]) == "a7a6"
        assert eng.score is not None
    finally:
        eng.close()
