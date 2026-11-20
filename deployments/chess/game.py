"""
game.py — a game of chess against the arm. The person plays one side by
hand, on the board; a chess engine (engine.py: Stockfish) chooses the arm's
moves, and the arm makes them as play.py makes any move — every carry checked
by a look, put right where it can be.

    new_game(arm_plays, strength)   the board checked at the starting position
                                    and every piece told what it is; when the arm
                                    plays white it makes the first move
    my_move(move)                   the person has moved: the move as they said it
                                    ('e5', 'Nxe4', 'O-O', 'e7e5'), or, said without
                                    one ("done", "your move"), found by a look;
                                    then the arm replies
    game                            the board as the game has it, the moves, whose
                                    move it is, what the engine makes of it
    hint                            the engine's move for the person
    resign / end_game               the person resigns; the game stops without a result
    resume_game                     the game saved in logs/game.json, after a restart

How the person's move is found: a look (play.look: a scan with no vision
model) says which squares have a piece; each legal move says which squares
should have one after it; the move whose squares the look agrees with is the
one made. Where the look agrees with no move everywhere, a move it disagrees
with only away from the move's own squares (play.around) counts — the
overhead camera misses a piece now and then — once a second look agrees.
Two moves the squares cannot tell apart (a knight that could have taken on
either of two squares) are named, and the person says which (/my_move Nxe4).
A promotion is taken to be to a queen unless the person says otherwise
(/my_move e2e1n).

What the arm cannot make itself it asks the person to make for it: the
engine chooses among the moves the arm can make (engine.py's searchmoves:
every square inside the arm's reach, and no promotion, as the arm has no
spare queen). With none of those, a carry that fails with the board as it
was, or a promotion, the move is the person's to make by hand ("pending"),
and "done" checks it was made. A carry that leaves the board other than the
game believes has the person set the board up as printed, with the move made,
checked by a look of every square.

The arm plays the side config.ROBOT_PLAYS names (the board is set up with
that side nearest the arm): its own pieces start within reach, and the
far ranks — where the person's pieces start — are at its reach's edge.

Every ply is saved to logs/game.json, and logged with play.py's moves in
logs/openings.jsonl (phase "game").

A skill that ends waiting for the person — to move, to say which move, to
put the board right — returns True: a False return makes the controller look
again and plan the command once more, which would only ask again. False is
for what went wrong in the arm or the engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

from mrcf.robot_env import RobotEnv
from mrcf.vlm import register_action

from . import board_log, engine, play, rules
from . import config as cfg

SAVE_PATH = board_log.LOG_DIR / "game.json"
TAG = "  [Game]"
START_FEN = rules.Position.start().fen()


@dataclass
class Game:
    """One game: who plays what, the moves so far, and what it waits for."""
    arm: str = "white"                      # the side the arm plays
    elo: int = 1500                         # how well the engine plays for it
    start_fen: str = START_FEN
    plies: list = field(default_factory=list)       # [{"uci", "san", "by": "arm"|"person", "how": "arm"|"hand"}]
    captured: list = field(default_factory=list)    # where each piece the arm took off was put (play.make's stack)
    pending: "str | None" = None            # the arm's move (UCI) that the person is to make for it, by hand
    check_board: bool = False               # before going on, a look must find every square as the game has
                                            # it (with pending made): the board was left other than believed
    result: "str | None" = None             # '1-0', '0-1', '1/2-1/2', or '*' (ended without one)
    reason: str = ""
    score: "dict | None" = None             # the engine's last view of it, from the arm's side
    started: str = ""

    # ── What the moves add up to ──────────────────────────────────────────

    def history(self) -> "list[rules.Position]":
        """Every position of the game, the start first and the one now last."""
        pos = rules.Position.from_fen(self.start_fen)
        out = [pos]
        for p in self.plies:
            pos = pos.play(_move(p["uci"]), check=False)
            out.append(pos)
        return out

    def position(self) -> rules.Position:
        return self.history()[-1]

    def moves_uci(self) -> "list[str]":
        return [p["uci"] for p in self.plies]

    @property
    def finished(self) -> bool:
        return self.result is not None

    @property
    def person(self) -> str:
        return "black" if self.arm == "white" else "white"

    def arm_to_move(self) -> bool:
        return self.position().turn == self.arm[0]


GAME: "Game | None" = None


def _move(uci: str) -> rules.Move:
    return rules.Move(uci[:2], uci[2:4], uci[4] if len(uci) > 4 else None)


# ── Saving ─────────────────────────────────────────────────────────────────────

def save(game: Game) -> None:
    try:
        SAVE_PATH.parent.mkdir(exist_ok=True)
        SAVE_PATH.write_text(json.dumps(asdict(game), indent=1), encoding="utf-8")
    except OSError as exc:
        print(f"{TAG} Could not save the game ({exc}).")


def load() -> "Game | None":
    """The saved game, its moves checked legal one by one; None if there is none or it does not add up."""
    try:
        data = json.loads(SAVE_PATH.read_text(encoding="utf-8"))
        game = Game(**data)
        pos = rules.Position.from_fen(game.start_fen)
        for p in game.plies:
            pos = pos.play(_move(p["uci"]))
        if game.pending:
            pos.play(_move(game.pending))
        return game
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"{TAG} No saved game to carry on with ({exc}).")
        return None


# ── The rules of the game beyond the moves ─────────────────────────────────────

def _halfmove_clock(game: Game) -> int:
    """Plies since the last capture or pawn move (the fifty-move rule counts 100)."""
    n = 0
    for pos, p in zip(game.history(), game.plies):
        move = _move(p["uci"])
        if pos.pieces[move.frm].lower() == "p" or pos.captured_square(move) is not None:
            n = 0
        else:
            n += 1
    return n


def _insufficient(pos: rules.Position) -> bool:
    """Neither side can mate: kings alone, a king and one knight or bishop, or bishops all on one colour."""
    others = [(sq, p) for sq, p in pos.board if p.lower() != "k"]
    if not others:
        return True
    if len(others) == 1 and others[0][1].lower() in "bn":
        return True
    if all(p.lower() == "b" for _, p in others):
        colours = {(rules.FILES.index(sq[0]) + int(sq[1])) % 2 for sq, _ in others}
        return len(colours) == 1
    return False


def over(game: Game) -> "tuple[str, str] | None":
    """(result, why) if the game has ended where it stands, else None."""
    hist = game.history()
    pos = hist[-1]
    if not pos.legal_moves():
        if pos._in_check(pos.turn == "w"):
            winner = "black" if pos.turn == "w" else "white"
            return ("1-0" if winner == "white" else "0-1"), f"checkmate — {winner} wins"
        return "1/2-1/2", "stalemate — a draw"
    if _insufficient(pos):
        return "1/2-1/2", "neither side has the pieces left to mate — a draw"
    if _halfmove_clock(game) >= 100:
        return "1/2-1/2", "fifty moves without a capture or a pawn move — a draw"
    key = pos.fen()
    if sum(1 for p in hist if p.fen() == key) >= 3:
        return "1/2-1/2", "the same position for the third time — a draw"
    return None


# ── Telling a move from a look ─────────────────────────────────────────────────

def touched_by(pos: rules.Position, move: rules.Move) -> "set[str]":
    """The squares a move empties or fills: its own two, the one it takes on, a castling rook's."""
    out = {move.frm, move.to}
    taken = pos.captured_square(move)
    if taken is not None:
        out.add(taken)
    if pos.pieces.get(move.frm, "").lower() == "k" and abs(ord(move.to[0]) - ord(move.frm[0])) == 2:
        rank = move.frm[1]
        out |= {"h" + rank, "f" + rank} if move.to[0] == "g" else {"a" + rank, "d" + rank}
    return out


def disagreement(pos: rules.Position, move: rules.Move, seen: "set[str]") -> "set[str] | None":
    """
    Where a look (seen: the squares with a piece) differs from the board after
    move: None if it differs on or beside the move's own squares (it is not
    that move), else the squares further off where it does (doubts about the
    look, empty when it agrees everywhere).
    """
    diff = pos.play(move, check=False).occupied() ^ seen
    if diff & play.around(touched_by(pos, move)):
        return None
    return diff


def explain(pos: rules.Position, seen: "set[str]") -> "tuple[list[rules.Move], bool]":
    """
    The legal moves a look agrees with: those it agrees with on every square,
    if any (exact True), else those it differs from only away from the move.
    A pawn reaching the last rank counts once, as a queen.
    """
    exact, loose = [], []
    for m in pos.legal_moves():
        d = disagreement(pos, m, seen)
        if d is not None:
            (loose if d else exact).append(m)
    found = exact or loose
    kept = [m for m in found if m.promotion in (None, "q")]
    return kept, bool(exact)


def _look(env, scene_state) -> "set[str]":
    play.look(env, scene_state)
    return play.occupied(scene_state)


def _differences(want: "set[str]", seen: "set[str]") -> str:
    parts = [f"nothing on {sq}" for sq in sorted(want - seen)] + [f"a piece on {sq}" for sq in sorted(seen - want)]
    return ", ".join(parts) or "nothing"


# ── A board set up again ───────────────────────────────────────────────────────

# Squares a look may miss and still have the pieces set up for a new game: the
# overhead camera misses a piece now and then (play.check's doubts).
RESET_MISSES = 2
# How far the game must have gone from the start, in squares whose occupancy
# differs, before pieces found there again are a new game: two moves' worth.
# One move from it, they are more likely that move put back (a pawn on e2
# again after 1. e4), and the board is set up as the game has it.
RESET_FROM_SQUARES = 4


class _BoardReset(Exception):
    """A look found the pieces set up for a new game, where the game under way has them elsewhere."""


def set_up_for_new_game(seen: "set[str]") -> bool:
    """Whether a look's squares are the starting position's: none extra, at most RESET_MISSES missed."""
    start = rules.Position.start().occupied()
    return not (seen - start) and len(start - seen) <= RESET_MISSES


def _reset_instead(game: Game, seen: "set[str]") -> None:
    """
    Raise _BoardReset when a look that agrees with no move of the game finds
    the pieces set up again: the person has started over, and saying so beats
    listing the 20 squares that differ from the game. Not for a game still
    near the start (RESET_FROM_SQUARES), nor one whose pieces are all home again.
    """
    gone = len(game.position().occupied() ^ rules.Position.start().occupied())
    if set_up_for_new_game(seen) and gone >= RESET_FROM_SQUARES:
        raise _BoardReset()


def _person_move(env, scene_state, game: Game, said: str) -> "rules.Move | None":
    """The person's move, as said and seen on the board, or as a look finds it; None (and why, said) if neither."""
    pos = game.position()
    if said:
        try:
            move = pos.parse(said)
        except ValueError:
            legal = ", ".join(sorted(pos.san(m) for m in pos.legal_moves()))
            print(f"{TAG} {said!r} is not a legal move here. Legal: {legal}.")
            return None
        san = pos.san(move)
        for _ in range(play.LOOK_TRIES):
            seen = _look(env, scene_state)
            if disagreement(pos, move, seen) is not None:
                return move
        want = pos.play(move, check=False).occupied()
        near = play.around(touched_by(pos, move))
        print(f"{TAG} The board does not show {san} yet: the look has "
              f"{_differences(want & near, seen & near)} — make it on the board, then say done.")
        return None
    first = None
    for _ in range(play.LOOK_TRIES):
        seen = _look(env, scene_state)
        if not (pos.occupied() ^ seen):
            print(f"{TAG} I see no move on the board yet — make yours, then say done "
                  "(or tell me it: /my_move e5).")
            return None
        moves, exact = explain(pos, seen)
        if exact and len(moves) == 1:
            return moves[0]
        if exact:
            break                                   # two moves the squares cannot tell apart: ask
        if len(moves) == 1 and first == moves[0]:
            return moves[0]                         # two looks, the same doubt away from the move
        first = moves[0] if len(moves) == 1 else None
    if not exact:
        _reset_instead(game, seen)
    if len(moves) > 1:
        names = " or ".join(sorted(pos.san(m) for m in moves))
        print(f"{TAG} The board could be {names} — which? Say it, e.g. /my_move {pos.san(moves[0])}.")
    else:
        print(f"{TAG} That does not look like a legal move from {pos.fen().split()[0]}: the look differs "
              f"from the game on {_differences(pos.occupied(), seen)}. Put it right, or tell me the move "
              "(/my_move e5).")
    return None


# ── The arm's side ─────────────────────────────────────────────────────────────

def can_make(pos: rules.Position, move: rules.Move) -> bool:
    """Whether the arm can make move itself: no promotion, and every square it carries from or to in reach."""
    try:
        steps = pos.steps(move)
    except ValueError:
        return False
    return all(play.reachable(sq) for s in steps for sq in (s.frm, s.to) if sq != rules.CAPTURED)


def _engine(game: Game) -> engine.Engine:
    """The session's engine, playing at the game's strength."""
    eng = engine.shared()
    if eng.elo != game.elo:
        game.elo = eng.set_elo(game.elo)
    return eng


def _fen_for_engine(game: Game) -> "str | None":
    return None if game.start_fen == START_FEN else game.start_fen + " 0 1"


def _commit(game: Game, scene_state, move: rules.Move, by: str, how: str) -> str:
    """move played: into the game, saved, logged, and every piece told what it now is. Its SAN."""
    pos = game.position()
    san = pos.san(move)
    game.plies.append({"uci": move.uci, "san": san, "by": by, "how": how})
    game.pending, game.check_board = None, False
    ended = over(game)
    if ended:
        game.result, game.reason = ended
    save(game)
    if how == "hand":
        board_log.record_move(phase="game", ply=len(game.plies), san=san, uci=move.uci, by=by, how=how)
    play.teach(scene_state, game.position())
    return san


def _number(pos: rules.Position, ply: int) -> str:
    return f"{ply // 2 + 1}." + ("" if pos.turn == "w" else "..")


def _arm_moves(env, scene_state, game: Game) -> bool:
    """The engine's move for the arm, made by the arm — or left to the person to make for it, said why."""
    pos = game.position()
    try:
        eng = _engine(game)
        legal = pos.legal_moves()
        doable = [m for m in legal if can_make(pos, m)]
        only = [m.uci for m in doable] if doable and len(doable) < len(legal) else None
        uci = eng.best_move(game.moves_uci(), searchmoves=only, start_fen=_fen_for_engine(game))
    except engine.EngineError as exc:
        print(f"{TAG} The chess engine failed: {exc}")
        return False
    game.score = eng.score
    if uci is None:
        print(f"{TAG} The engine has no move — the game is over?")
        return False
    try:
        move = pos.parse(uci)
    except ValueError:
        print(f"{TAG} The engine answered {uci!r}, not a legal move here.")
        return False
    san = pos.san(move)
    number = _number(pos, len(game.plies))
    if move not in doable:
        why = "a promotion needs a piece from off the board" if move.promotion else "it is out of my reach"
        game.pending = move.uci
        save(game)
        print(f"{TAG} My move: {number} {san} — {why}. Please make it for me on the board, then say done.")
        return True
    print(f"{TAG} My move: {number} {san}")
    rec = play.make(env, scene_state, pos.steps(move), game.captured, before=pos.occupied())
    board_log.record_move(phase="game", ply=len(game.plies) + 1, san=san, uci=move.uci, by="arm", **rec)
    if rec["ok"]:
        _commit(game, scene_state, move, "arm", "arm")
        _report(game)
        return True
    game.pending = move.uci
    if rec.get("board_wrong"):
        game.check_board = True
        save(game)
        print(f"{TAG} {san} went wrong ({rec['why']}), and the board is not as it should be. Please set it "
              f"up like this, with {san} made, then say done:")
        print(board_text(pos.play(move)))
        return True
    save(game)
    print(f"{TAG} I could not make {san} myself ({rec['why']}); the board is as it was. Please make it for "
          "me, then say done.")
    return True


def _settle_pending(env, scene_state, game: Game, said: str) -> "tuple[bool, rules.Move | None]":
    """
    The arm's move the person was asked to make, checked on the board: (made,
    the person's own move if they made that too). Not made: said what is
    still wanted, (False, None).
    """
    pos = game.position()
    move = _move(game.pending)
    san = pos.san(move)
    after = pos.play(move, check=False)
    if game.check_board:
        wrong, theirs = _set_up(env, scene_state, after)
        if wrong:
            _reset_instead(game, play.occupied(scene_state))
            print(f"{TAG} Not yet as it should be: {wrong}. Set it up like this, then say done:")
            print(board_text(after))
            return False, None
        _commit(game, scene_state, move, "arm", "hand")
        print(f"{TAG} The board is right — {san} is made.")
        return True, theirs
    seen = _look(env, scene_state)
    if disagreement(pos, move, seen) is not None:
        _commit(game, scene_state, move, "arm", "hand")
        print(f"{TAG} Thanks — {san} is made.")
        return True, None
    # Made, and the person's own move after it.
    if said:
        try:
            theirs = after.parse(said)
        except ValueError:
            theirs = None
        if theirs is not None and disagreement(after, theirs, seen) is not None:
            _commit(game, scene_state, move, "arm", "hand")
            return True, theirs
    else:
        moves, exact = explain(after, seen)
        if exact and len(moves) == 1:
            _commit(game, scene_state, move, "arm", "hand")
            return True, moves[0]
    _reset_instead(game, seen)
    if not (pos.occupied() ^ seen):
        print(f"{TAG} My move {san} is not on the board yet — please make it for me "
              f"({move.frm} to {move.to}), then say done.")
    else:
        want = after.occupied()
        near = play.around(touched_by(pos, move))
        print(f"{TAG} The board does not show my move {san}: the look has {_differences(want & near, seen & near)}.")
    return False, None


def _set_up(env, scene_state, want: rules.Position, moved: bool = True) -> "tuple[str, rules.Move | None]":
    """
    Whether the board stands as want, every square looked at (play.check):
    ("", None) when it does; ("", the move) when it stands as want with one
    legal move made from it — the person set it up and moved too (only if
    moved: it is the person's move); else (what is wrong, None).
    """
    found = play.check(env, scene_state, want.occupied(), set(rules.SQUARES))
    wrong = play.wrong_with(found)
    if not wrong:
        return "", None
    moves, exact = explain(want, play.occupied(scene_state))
    if moved and exact and len(moves) == 1 and not found["displaced"]:
        return "", moves[0]
    return wrong, None


def _report(game: Game) -> None:
    """After a move: the result if it ended, else the board and who is to move."""
    if game.finished:
        print(f"{TAG} Game over: {game.reason} ({game.result}). Moves: {move_list(game)}")
        return
    print(board_text(game.position()))
    if game.arm_to_move():
        print(f"{TAG} My move next.")
    else:
        last = game.plies[-1]["san"] if game.plies else ""
        print(f"{TAG} {'After ' + last + ' — y' if last else 'Y'}our move ({game.person}): make it on the board, "
              "then say done (or tell me it).")


# ── Showing it ─────────────────────────────────────────────────────────────────

def board_text(pos: rules.Position) -> str:
    """The position as text, white at the bottom: capitals are white, '.' an empty square."""
    pieces = pos.pieces
    lines = ["      a b c d e f g h"]
    for r in reversed(rules.RANKS):
        row = " ".join(pieces.get(f + r, ".") for f in rules.FILES)
        lines.append(f"    {r} {row} {r}")
    lines.append("      a b c d e f g h")
    return "\n".join(lines)


def move_list(game: Game) -> str:
    """'1. e4 e5 2. Nf3' — the moves so far in standard notation."""
    out = []
    black_first = rules.Position.from_fen(game.start_fen).turn == "b"
    for i, p in enumerate(game.plies):
        k = i + (1 if black_first else 0)
        if k % 2 == 0:
            out.append(f"{k // 2 + 1}. {p['san']}")
        elif i == 0:
            out.append(f"{k // 2 + 1}... {p['san']}")
        else:
            out.append(p["san"])
    return " ".join(out) or "none yet"


def state_line() -> str:
    """One line on the game for the planner: whether one is on, and what it waits for."""
    game = GAME
    if game is None or game.finished:
        return ""
    if game.pending:
        return (f"Chess game in progress: waiting for the person to make the arm's move "
                f"{_move(game.pending).uci} by hand and say done.")
    if game.arm_to_move():
        return "Chess game in progress: the arm's move."
    return f"Chess game in progress: the person's move ({game.person}) — 'done' or a move means my_move."


# ── Actions ───────────────────────────────────────────────────────────────────

def _side(text) -> "str | None":
    t = str(text or "").strip().lower()
    return {"white": "white", "w": "white", "black": "black", "b": "black"}.get(t)


@register_action(
    "new_game",
    description=(
        "Start a game of chess between the person and the arm ('let's play', 'new game', 'play me at 1800'): "
        "the board must be in the starting position. The person moves their pieces by hand; a chess engine "
        "chooses the arm's moves and the arm makes them. arm_plays: 'white' (the arm moves first; its "
        "pieces are the ones nearest it) or 'black'. strength: the engine's Elo rating, 1320-3190."
    ),
    parameters={"arm_plays": "The side the arm plays, 'white' or 'black' (default: the side nearest it).",
                "strength": "The engine's playing strength as an Elo rating (default 1500)."},
    always=True,
)
def skill_new_game(env: RobotEnv, scene_state, arm_plays=None, strength=None, **_):
    global GAME
    arm = _side(arm_plays) if arm_plays not in (None, "") else _side(getattr(cfg, "ROBOT_PLAYS", "white"))
    if arm is None:
        print(f"{TAG} {arm_plays!r} is not a side — 'white' or 'black'.")
        return True
    try:
        elo = int(float(strength)) if strength not in (None, "") else int(getattr(cfg, "ENGINE_ELO", 1500))
    except (TypeError, ValueError):
        print(f"{TAG} {strength!r} is not a strength — an Elo rating such as 1500.")
        return True
    if play._holding(env):
        print(f"{TAG} Something is in the jaws — put it down first.")
        return False
    try:
        eng = engine.shared()
    except engine.EngineError as exc:
        print(f"{TAG} {exc}")
        return False
    near = _side(getattr(cfg, "ROBOT_PLAYS", "white"))
    if arm != near:
        print(f"{TAG} The board is set up with {near} nearest the arm (config.ROBOT_PLAYS): playing {arm}, "
              "its pieces start at the far edge of its reach, and many of its moves will be yours to make for it.")
    if GAME is not None and not GAME.finished and GAME.plies:
        print(f"{TAG} Leaving the game in progress ({len(GAME.plies)} moves: {move_list(GAME)}).")
    start = rules.Position.start()
    found = play.check(env, scene_state, start.occupied(), set(rules.SQUARES))
    if found["extra"] or found["displaced"]:
        print(f"{TAG} The board is not at the starting position: {play.wrong_with(dict(found, missing=[]))}. "
              "Set it up, then start the game again.")
        return True
    _begin(scene_state, eng, arm, elo, found["missing"])
    return _opening(env, scene_state, GAME)


def _begin(scene_state, eng, arm: str, elo: int, missing=()) -> Game:
    """A new game, on a board a look found set up for it: every piece told what it is, saved, said."""
    global GAME
    if missing:
        print(f"{TAG} Not seen on {', '.join(missing)} — taking it as the starting position anyway.")
    play.teach(scene_state, rules.Position.start())
    GAME = Game(arm=arm, elo=elo, started=datetime.now().isoformat(timespec="seconds"))
    GAME.elo = eng.set_elo(elo)
    save(GAME)
    print(f"{TAG} New game: the arm plays {arm} at about {GAME.elo} Elo ({eng.name or 'the engine'}); "
          f"you play {GAME.person}.")
    return GAME


def _opening(env, scene_state, game: Game) -> bool:
    """A new game's first move: the arm's when it plays white, else the person is told it is theirs."""
    if game.arm_to_move():
        return _arm_moves(env, scene_state, game)
    print(f"{TAG} Your move (white): make it on the board, then say done (or tell me it: /my_move e4).")
    return True


def _start_over(env, scene_state, old: "Game | None", seen: "set[str]", move: bool = True) -> bool:
    """
    A look found the pieces set up for a new game: begin one, the same side
    and strength as old (else config's), and with move, make the arm's first
    move. The person said done, go, or resume to a board they had set up
    again, which is the go-ahead.
    """
    if old is not None and old.plies and not old.finished:
        print(f"{TAG} The pieces are set up for a new game — leaving the one in progress "
              f"({len(old.plies)} moves: {move_list(old)}).")
    else:
        print(f"{TAG} The pieces are set up for a new game.")
    if play._holding(env):
        print(f"{TAG} Something is in the jaws — put it down first.")
        return False
    try:
        eng = engine.shared()
    except engine.EngineError as exc:
        print(f"{TAG} {exc}")
        return False
    arm = old.arm if old is not None else (_side(getattr(cfg, "ROBOT_PLAYS", "white")) or "white")
    elo = old.elo if old is not None and old.elo else int(getattr(cfg, "ENGINE_ELO", 1500))
    game = _begin(scene_state, eng, arm, elo, sorted(rules.Position.start().occupied() - seen))
    if move:
        return _opening(env, scene_state, game)
    if game.arm_to_move():
        print(f"{TAG} Say go (or done) when you are ready, and I make my first move.")
    else:
        print(f"{TAG} Your move (white): make it on the board, then say done (or tell me it: /my_move e4).")
    return True


@register_action(
    "my_move",
    description=(
        "In the game of chess against the arm, the person has made their move on the board ('done', 'go', 'your "
        "move', 'I played e5', 'I took on d5', 'I castled', 'I made your move'): the arm looks at the board "
        "to see it, then makes its reply. 'move' is the person's move when they said it — 'e5', 'Nf3', "
        "'exd5', 'O-O' or 'e7e5'; leave it out when they did not."
    ),
    parameters={"move": "The person's move if they said it (e.g. 'e5', 'Nxe4', 'O-O'); omit otherwise."},
    always=True,
)
def skill_my_move(env: RobotEnv, scene_state, move="", **_):
    game = GAME
    if game is None or game.finished:
        # No game under way: a board set up for one is the go-ahead to begin it.
        if play._holding(env):
            print(f"{TAG} Something is in the jaws — put it down first.")
            return False
        seen = _look(env, scene_state)
        if set_up_for_new_game(seen):
            return _start_over(env, scene_state, game, seen)
        print(f"{TAG} No game in progress, and the pieces are not set up for one — set them up and say go "
              "(or /new_game), or /resume_game to carry on the saved one.")
        return True
    try:
        return _my_move(env, scene_state, game, move)
    except _BoardReset:
        return _start_over(env, scene_state, game, play.occupied(scene_state))


def _my_move(env, scene_state, game: Game, move) -> bool:
    """skill_my_move with a game under way; raises _BoardReset when a look finds the pieces set up again."""
    said = str(move or "").strip()
    if play._holding(env):
        print(f"{TAG} Something is in the jaws — put it down first.")
        return False
    if game.pending:
        made, theirs = _settle_pending(env, scene_state, game, said)
        if not made:
            return True
        if game.finished:
            _report(game)
            return True
        if theirs is None:
            if not said:
                _report(game)
                return True
            # Said with a move: theirs is still to be found on the board.
    elif game.check_board:
        wrong, theirs = _set_up(env, scene_state, game.position(), moved=not game.arm_to_move())
        if wrong:
            _reset_instead(game, play.occupied(scene_state))
            print(f"{TAG} Not yet as the game has it: {wrong}. Set it up like this, then say done:")
            print(board_text(game.position()))
            return True
        game.check_board = False
        save(game)
        play.teach(scene_state, game.position())
    else:
        theirs = None
    if game.arm_to_move():
        return _arm_moves(env, scene_state, game)
    pos = game.position()
    if theirs is None:
        theirs = _person_move(env, scene_state, game, said)
        if theirs is None:
            return True
    number = _number(pos, len(game.plies))
    san = _commit(game, scene_state, theirs, "person", "hand")
    print(f"{TAG} You played {number} {san}")
    if game.finished:
        _report(game)
        return True
    return _arm_moves(env, scene_state, game)


@register_action(
    "game",
    description=("In the game of chess against the arm: show the board as the game has it, the moves so far, "
                 "whose move it is and how the engine rates it ('show the game', 'whose move is it')."),
    always=True,
)
def skill_game(**_):
    game = GAME
    if game is None:
        print(f"{TAG} No game — /new_game starts one, /resume_game carries on the saved one.")
        return True
    pos = game.position()
    state = (f"over: {game.reason} ({game.result})" if game.finished else
             "waiting for you to make my move " + pos.san(_move(game.pending)) + " by hand" if game.pending else
             "my move" if game.arm_to_move() else f"your move ({game.person})")
    print(f"{TAG} You ({game.person}) v the arm ({game.arm}, about {game.elo} Elo) — {state}.")
    print(board_text(pos))
    print(f"{TAG} Moves: {move_list(game)}")
    rated = engine.score_text(game.score, white_view=game.arm == "white")
    if rated:
        print(f"{TAG} The engine's view after its last move: {rated} (+ is good for white).")
    return True


@register_action(
    "hint",
    description=("In the game of chess against the arm: the chess engine's suggestion for the person's move "
                 "('what should I play?', 'give me a hint')."),
    always=True,
)
def skill_hint(**_):
    game = GAME
    if game is None or game.finished or game.pending or game.arm_to_move():
        print(f"{TAG} A hint is for your move in a game in progress.")
        return True
    pos = game.position()
    try:
        eng = _engine(game)
        # Its best, not the strength it plays the arm's side at.
        limited = "UCI_LimitStrength" in eng.options
        skill = eng.options.get("Skill Level") if not limited else None
        if limited:
            eng.set_option("UCI_LimitStrength", False)
        elif skill is not None:
            eng.set_option("Skill Level", skill.get("max", 20))
        try:
            uci = eng.best_move(game.moves_uci(), start_fen=_fen_for_engine(game))
        finally:
            if (limited or skill is not None) and eng.alive():
                eng.set_elo(game.elo)
    except engine.EngineError as exc:
        print(f"{TAG} The chess engine failed: {exc}")
        return False
    if uci is None:
        print(f"{TAG} The engine has no move to suggest.")
        return True
    move = pos.parse(uci)
    print(f"{TAG} The engine would play {pos.san(move)} ({engine.score_text(eng.score, white_view=pos.turn == 'w')}).")
    return True


@register_action("resign", description="The person resigns the game of chess against the arm.", always=True)
def skill_resign(**_):
    game = GAME
    if game is None or game.finished:
        print(f"{TAG} No game in progress.")
        return True
    game.result = "1-0" if game.arm == "white" else "0-1"
    game.reason = f"{game.person} resigned — the arm wins"
    save(game)
    _report(game)
    return True


@register_action("end_game", description="Stop the game of chess against the arm, without a result.", always=True)
def skill_end_game(**_):
    game = GAME
    if game is None or game.finished:
        print(f"{TAG} No game in progress.")
        return True
    game.result, game.reason = "*", "ended without a result"
    save(game)
    print(f"{TAG} Game ended after {len(game.plies)} moves: {move_list(game)}")
    return True


@register_action(
    "resume_game",
    description=("Carry on with the game of chess against the arm that was saved before the controller "
                 "restarted: the board is checked against it first."),
    always=True,
)
def skill_resume_game(env: RobotEnv, scene_state, **_):
    global GAME
    game = load()
    if game is None:
        return True
    GAME = game
    if game.finished:
        print(f"{TAG} The saved game is over: {game.reason} ({game.result}). Moves: {move_list(game)}")
        return True
    print(f"{TAG} Carrying on: you ({game.person}) v the arm ({game.arm}). Moves: {move_list(game)}")
    want = game.position()
    if game.pending:
        print(f"{TAG} Waiting for you to make my move {want.san(_move(game.pending))} by hand; say done when "
              "it is made.")
        return True
    found = play.check(env, scene_state, want.occupied(), set(rules.SQUARES))
    wrong = play.wrong_with(found)
    if wrong:
        try:
            _reset_instead(game, play.occupied(scene_state))
        except _BoardReset:
            return _start_over(env, scene_state, game, play.occupied(scene_state))
        game.check_board = True
        save(game)
        print(f"{TAG} The board is not as the game has it: {wrong}. Set it up like this, then say done:")
        print(board_text(want))
        return True
    play.teach(scene_state, want)
    if game.arm_to_move():
        return _arm_moves(env, scene_state, game)
    _report(game)
    return True


def _close_to(want: "set[str]", seen: "set[str]") -> bool:
    """A look that has the board as want: the same squares, or one piece missed (a look's doubt)."""
    return not (seen - want) and len(want - seen) <= 1


def on_start(env, scene_state) -> None:
    """
    The controller's first look of a session, read for the game, moving
    nothing: the board as the saved game has it (or with the person's move
    made since, or the arm's move they were asked to make) carries that game
    on; the pieces set up for a new game begin one, the arm's first move
    waiting for a go. Anything else is left for /resume_game or /new_game.
    Before, every session began with one of those typed, even on a board that
    plainly said which (2026-11-16).
    """
    global GAME
    seen = play.occupied(scene_state)
    if not seen:
        return                                    # no board found: nothing to read
    saved = load()
    live = saved if saved is not None and not saved.finished and saved.plies else None
    if live is not None:
        pos = live.position()
        after = pos.play(_move(live.pending), check=False) if live.pending else None
        one_on = explain(after if after is not None else pos, seen)
        if (_close_to(pos.occupied(), seen) or (after is not None and _close_to(after.occupied(), seen))
                or (one_on[1] and len(one_on[0]) == 1)):
            GAME = live
            if seen == pos.occupied():
                play.teach(scene_state, pos)
            print(f"{TAG} Carrying on the saved game: you ({live.person}) v the arm ({live.arm}). "
                  f"Moves: {move_list(live)}")
            if live.pending:
                print(f"{TAG} Waiting for you to make my move {pos.san(_move(live.pending))} by hand; say done "
                      "when it is made.")
            elif live.arm_to_move():
                print(f"{TAG} My move — say go (or done) when you are ready.")
            else:
                print(f"{TAG} Your move ({live.person}): make it on the board, then say done.")
            return
    if set_up_for_new_game(seen):
        _start_over(env, scene_state, saved, seen, move=False)
    elif live is not None:
        print(f"{TAG} The board is neither as the saved game has it ({len(live.plies)} moves) nor set up for a "
              "new one: /resume_game says what differs; or set the pieces up for a new game and say go.")


__all__ = ["Game", "GAME", "explain", "disagreement", "touched_by", "can_make", "over", "board_text",
           "move_list", "save", "load", "state_line", "on_start", "set_up_for_new_game"]
