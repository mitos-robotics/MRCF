"""The rules of chess the arm plays by: move generation, notation, the book, and what each move carries."""

import pytest

from deployments.chess import rules
from deployments.chess.rules import CAPTURED, Move, Position, Step


def after(line: str) -> Position:
    pos, move = rules.play_line(line)[-1]
    return pos.play(move)


def perft(pos: Position, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(perft(pos.play(m, check=False), depth - 1) for m in pos.legal_moves())


def test_move_counts_match_the_known_perft_numbers():
    # The standard checks of a move generator: from the start, and "Kiwipete",
    # which has castling both ways, en passant, pins and promotions near.
    assert [perft(Position.start(), d) for d in (1, 2, 3)] == [20, 400, 8902]
    kiwipete = Position.from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert [perft(kiwipete, d) for d in (1, 2)] == [48, 2039]
    # Promotions and checks along the edge.
    assert perft(Position.from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"), 3) == 2812


def test_fen_round_trips():
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"
    assert Position.from_fen(fen).fen() == fen
    assert Position.start().play(Move("e2", "e4")).fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3"


def test_moves_are_read_from_either_notation_and_written_in_san():
    pos = Position.start()
    assert pos.parse("Nf3") == pos.parse("g1f3") == Move("g1", "f3")
    assert pos.parse("e4") == Move("e2", "e4")
    assert pos.san(Move("b1", "c3")) == "Nc3"
    with pytest.raises(ValueError):
        pos.parse("e5")
    with pytest.raises(ValueError):
        pos.parse("Ke2")
    # Two knights reach d2: the file says which.
    two = Position.from_fen("4k3/8/8/8/8/8/8/1N2KN2 w - -")
    assert two.san(Move("b1", "d2")) == "Nbd2"
    assert two.parse("Nfd2") == Move("f1", "d2")


def test_every_book_opening_is_legal_from_the_starting_position():
    for key, (_, line) in rules.OPENINGS.items():
        plies = rules.play_line(line)
        assert plies, key
        assert plies[0][0] == Position.start(), key
        # Written in the book as the rules would write it.
        assert " ".join(pos.san(m) for pos, m in plies) == line, key


def test_a_line_takes_move_numbers_and_the_book_is_found_however_it_is_spelled():
    assert [m.uci for _, m in rules.play_line("1. e4 e5 2. Nf3 Nc6")] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert rules.opening("Ruy Lopez") == rules.OPENINGS["ruy_lopez"]
    assert rules.opening("ruy-lopez") == rules.OPENINGS["ruy_lopez"]
    assert rules.opening("Queen's Gambit Declined")[1].startswith("d4 d5 c4 e6")
    with pytest.raises(KeyError):
        rules.opening("bongcloud")
    with pytest.raises(ValueError):
        rules.play_line("e4 e5 Ke3")                 # a king moves one square
    assert not rules.captures(rules.OPENINGS["italian"][1])
    assert rules.captures(rules.OPENINGS["scandinavian"][1])


def test_a_quiet_move_carries_one_piece_and_takes_back_the_other_way():
    pos = Position.start()
    move = pos.parse("Nf3")
    assert pos.steps(move) == [Step("g1", "f3", "N")]
    assert pos.take_back_steps(move) == [Step("f3", "g1", "N")]


def test_a_capture_takes_the_piece_off_first_and_puts_it_back_last():
    pos = after("e4 d5")
    move = pos.parse("exd5")
    assert pos.steps(move) == [Step("d5", CAPTURED, "p"), Step("e4", "d5", "P")]
    assert pos.take_back_steps(move) == [Step("d5", "e4", "P"), Step(CAPTURED, "d5", "p")]


def test_en_passant_takes_the_pawn_beside_not_on_the_square_moved_to():
    pos = Position.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6")
    move = pos.parse("exd6")
    assert pos.steps(move) == [Step("d5", CAPTURED, "p"), Step("e5", "d6", "P")]
    assert "d5" not in pos.play(move).pieces


def test_castling_carries_the_king_then_the_rook():
    pos = Position.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -")
    assert pos.steps(pos.parse("O-O")) == [Step("e1", "g1", "K"), Step("h1", "f1", "R")]
    after = pos.play(pos.parse("O-O"))
    assert after.pieces["f1"] == "R" and after.pieces["g1"] == "K" and after.castling == "kq"
    assert after.steps(after.parse("O-O-O")) == [Step("e8", "c8", "k"), Step("a8", "d8", "r")]
    # Not through an attacked square.
    attacked = Position.from_fen("4k3/8/8/8/8/8/5r2/R3K2R w KQ -")
    assert "O-O" not in [attacked.san(m) for m in attacked.legal_moves()]


def test_a_promotion_is_refused_as_something_the_arm_cannot_carry():
    pos = Position.from_fen("4k3/P7/8/8/8/8/8/4K3 w - -")
    with pytest.raises(ValueError):
        pos.steps(pos.parse("a8=Q"))
