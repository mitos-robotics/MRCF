"""
rules.py — the rules of chess, as far as moving the pieces needs them.

A Position (what stands where, whose turn, castling rights, the en-passant
square), the legal moves from it, a move read from standard notation ("Nf3",
"exd5", "O-O", "e8=Q") or coordinates ("g1f3"), and what a move is physically:
which piece goes from which square to which. Castling moves two pieces; a
capture first takes one off the board, to CAPTURED; taking a move back runs
those the other way round.

OPENINGS is a small book of named openings, each a line from the starting
position; tests/test_rules.py plays every one through these rules.

No chess engine and nothing to choose moves with: the moves are given (by the
book, or by the person at the prompt), and this only says whether they are
legal and what the arm has to do to make them.
"""

from __future__ import annotations

from dataclasses import dataclass

FILES = "abcdefgh"
RANKS = "12345678"
SQUARES = [f + r for r in RANKS for f in FILES]
CAPTURED = "captured"          # off the board (config.CAPTURE_TARGET: the place a captured piece goes)

_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_NAMES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}
_KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KING = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _xy(square: str) -> "tuple[int, int]":
    return FILES.index(square[0]), RANKS.index(square[1])


def _sq(x: int, y: int) -> "str | None":
    return FILES[x] + RANKS[y] if 0 <= x < 8 and 0 <= y < 8 else None


def colour_of(piece: str) -> str:
    """'white' or 'black', for a FEN letter ('N' white knight, 'n' black)."""
    return "white" if piece.isupper() else "black"


def name_of(piece: str) -> str:
    """'white knight' for 'N'."""
    return f"{colour_of(piece)} {_NAMES[piece.lower()]}"


@dataclass(frozen=True)
class Move:
    frm: str
    to: str
    promotion: "str | None" = None     # 'q', 'r', 'b' or 'n'

    @property
    def uci(self) -> str:
        return self.frm + self.to + (self.promotion or "")


@dataclass(frozen=True)
class Step:
    """One piece carried by the arm: from a square (or CAPTURED) to a square (or CAPTURED)."""
    frm: str
    to: str
    piece: str                          # FEN letter of what is carried


@dataclass(frozen=True)
class Position:
    board: "tuple[tuple[str, str], ...]"      # (square, FEN letter), sorted
    turn: str = "w"
    castling: str = "KQkq"
    ep: "str | None" = None

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        parts = fen.split()
        rows = parts[0].split("/")
        if len(rows) != 8:
            raise ValueError(f"not a FEN position: {fen!r}")
        board = {}
        for i, row in enumerate(rows):
            y, x = 7 - i, 0
            for ch in row:
                if ch.isdigit():
                    x += int(ch)
                else:
                    board[_sq(x, y)] = ch
                    x += 1
        turn = parts[1] if len(parts) > 1 else "w"
        castling = parts[2] if len(parts) > 2 and parts[2] != "-" else ""
        ep = parts[3] if len(parts) > 3 and parts[3] != "-" else None
        return cls(tuple(sorted(board.items())), turn, castling, ep)

    @classmethod
    def start(cls) -> "Position":
        return cls.from_fen(_START_FEN)

    def fen(self) -> str:
        pieces = self.pieces
        rows = []
        for y in range(7, -1, -1):
            row, gap = "", 0
            for x in range(8):
                p = pieces.get(_sq(x, y))
                if p is None:
                    gap += 1
                    continue
                row += (str(gap) if gap else "") + p
                gap = 0
            rows.append(row + (str(gap) if gap else ""))
        return f"{'/'.join(rows)} {self.turn} {self.castling or '-'} {self.ep or '-'}"

    @property
    def pieces(self) -> "dict[str, str]":
        return dict(self.board)

    def occupied(self) -> "set[str]":
        return {sq for sq, _ in self.board}

    # ── Moves ─────────────────────────────────────────────────────────────

    def legal_moves(self) -> "list[Move]":
        """Every legal move for the side to move."""
        white = self.turn == "w"
        return [m for m in self._pseudo_moves() if not self.play(m, check=False)._in_check(white)]

    def play(self, move: Move, check: bool = True) -> "Position":
        """The position after move. check: refuse a move that is not legal (ValueError)."""
        if check and move not in self.legal_moves():
            raise ValueError(f"{move.uci} is not legal in {self.fen()}")
        board = self.pieces
        piece = board.pop(move.frm)
        captured_on = self.captured_square(move)
        if captured_on is not None:
            board.pop(captured_on, None)
        if piece.lower() == "k" and abs(_xy(move.to)[0] - _xy(move.frm)[0]) == 2:
            rook_from, rook_to = _castle_rook(move)
            board[rook_to] = board.pop(rook_from)
        board[move.to] = (move.promotion.upper() if piece.isupper() else move.promotion) if move.promotion else piece
        castling = self.castling
        for sq, rights in (("e1", "KQ"), ("e8", "kq"), ("h1", "K"), ("a1", "Q"), ("h8", "k"), ("a8", "q")):
            if sq in (move.frm, move.to):
                castling = "".join(c for c in castling if c not in rights)
        ep = None
        if piece.lower() == "p" and abs(_xy(move.to)[1] - _xy(move.frm)[1]) == 2:
            x, y = _xy(move.frm)
            ep = _sq(x, y + (1 if piece.isupper() else -1))
        return Position(tuple(sorted(board.items())), "b" if self.turn == "w" else "w", castling, ep)

    def captured_square(self, move: Move) -> "str | None":
        """Where move takes a piece from (not its destination for en passant); None if it takes none."""
        piece = self.pieces.get(move.frm, "")
        if piece.lower() == "p" and move.to == self.ep and move.to not in self.pieces:
            return move.to[0] + move.frm[1]
        return move.to if move.to in self.pieces else None

    # ── Notation ──────────────────────────────────────────────────────────

    def parse(self, text: str) -> Move:
        """A move in SAN ('Nf3', 'exd5', 'O-O', 'e8=Q+') or UCI ('g1f3', 'e7e8q'), if legal here."""
        t = text.strip().rstrip("+#!?").replace("0", "O")
        legal = self.legal_moves()
        low = t.lower()
        if len(low) in (4, 5) and low[0] in FILES and low[1] in RANKS and low[2] in FILES and low[3] in RANKS:
            move = Move(low[:2], low[2:4], low[4] if len(low) == 5 else None)
            if move in legal:
                return move
        matches = [m for m in legal if self.san(m).rstrip("+#") == t]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"{text!r} is not a legal move in {self.fen()}")

    def san(self, move: Move) -> str:
        """move in standard notation: 'Nf3', 'exd5', 'O-O', with '+' or '#' after."""
        piece = self.pieces[move.frm]
        kind = piece.upper()
        if kind == "K" and abs(_xy(move.to)[0] - _xy(move.frm)[0]) == 2:
            text = "O-O" if move.to[0] == "g" else "O-O-O"
        else:
            takes = self.captured_square(move) is not None
            if kind == "P":
                text = (move.frm[0] + "x" if takes else "") + move.to
                if move.promotion:
                    text += "=" + move.promotion.upper()
            else:
                rivals = [m for m in self._pseudo_moves() if m.to == move.to and m.frm != move.frm
                          and self.pieces[m.frm] == piece and not self.play(m, check=False)._in_check(piece.isupper())]
                which = ""
                if rivals:
                    if all(m.frm[0] != move.frm[0] for m in rivals):
                        which = move.frm[0]
                    elif all(m.frm[1] != move.frm[1] for m in rivals):
                        which = move.frm[1]
                    else:
                        which = move.frm
                text = kind + which + ("x" if takes else "") + move.to
        after = self.play(move, check=False)
        if after._in_check(after.turn == "w"):
            text += "#" if not after.legal_moves() else "+"
        return text

    # ── What the arm does ─────────────────────────────────────────────────

    def steps(self, move: Move) -> "list[Step]":
        """
        The pieces the arm carries to make move, in order: a captured piece off
        the board first (it stands where the mover is going, or beside it for en
        passant), then the mover; castling, the king then the rook. A promotion
        takes the pawn off the board — the arm has no spare queen — and the
        promoted piece is the person's to put down (ValueError instead: a book
        line never promotes).
        """
        if move.promotion:
            raise ValueError(f"{move.uci}: a promotion needs a piece from off the board — not something the arm has")
        pieces = self.pieces
        out = []
        taken = self.captured_square(move)
        if taken is not None:
            out.append(Step(taken, CAPTURED, pieces[taken]))
        out.append(Step(move.frm, move.to, pieces[move.frm]))
        if pieces[move.frm].lower() == "k" and abs(_xy(move.to)[0] - _xy(move.frm)[0]) == 2:
            rook_from, rook_to = _castle_rook(move)
            out.append(Step(rook_from, rook_to, pieces[rook_from]))
        return out

    def take_back_steps(self, move: Move) -> "list[Step]":
        """The steps that undo move, played from this position (the one before it): steps() reversed."""
        return [Step(s.to, s.frm, s.piece) for s in reversed(self.steps(move))]

    # ── Internals ─────────────────────────────────────────────────────────

    def _pseudo_moves(self):
        pieces = self.pieces
        white = self.turn == "w"
        mine = (lambda p: p.isupper()) if white else (lambda p: p.islower())
        for frm, piece in pieces.items():
            if not mine(piece):
                continue
            x, y = _xy(frm)
            kind = piece.lower()
            if kind == "p":
                yield from self._pawn_moves(frm, white)
                continue
            if kind in ("n", "k"):
                for dx, dy in (_KNIGHT if kind == "n" else _KING):
                    to = _sq(x + dx, y + dy)
                    if to is not None and not (to in pieces and mine(pieces[to])):
                        yield Move(frm, to)
                if kind == "k":
                    yield from self._castles(frm, white)
                continue
            dirs = {"b": _BISHOP_DIRS, "r": _ROOK_DIRS, "q": _BISHOP_DIRS + _ROOK_DIRS}[kind]
            for dx, dy in dirs:
                cx, cy = x + dx, y + dy
                while (to := _sq(cx, cy)) is not None:
                    if to in pieces:
                        if not mine(pieces[to]):
                            yield Move(frm, to)
                        break
                    yield Move(frm, to)
                    cx, cy = cx + dx, cy + dy

    def _pawn_moves(self, frm: str, white: bool):
        pieces = self.pieces
        x, y = _xy(frm)
        dy = 1 if white else -1
        last = 7 if white else 0
        promos = ("q", "r", "b", "n")

        def emit(to):
            if _xy(to)[1] == last:
                for p in promos:
                    yield Move(frm, to, p)
            else:
                yield Move(frm, to)

        one = _sq(x, y + dy)
        if one is not None and one not in pieces:
            yield from emit(one)
            two = _sq(x, y + 2 * dy)
            if y == (1 if white else 6) and two not in pieces:
                yield Move(frm, two)
        for dx in (-1, 1):
            to = _sq(x + dx, y + dy)
            if to is None:
                continue
            if (to in pieces and pieces[to].isupper() != white) or to == self.ep:
                yield from emit(to)

    def _castles(self, frm: str, white: bool):
        pieces = self.pieces
        home, them = ("e1", False) if white else ("e8", True)
        if frm != home or self._attacked(home, by_white=not white):
            return
        rank = home[1]
        for right, rook, between, path in (("K", "h", "fg", "fg"), ("Q", "a", "bcd", "dc")):
            if (right if white else right.lower()) not in self.castling:
                continue
            if pieces.get(rook + rank) != ("R" if white else "r"):
                continue
            if any(f + rank in pieces for f in between):
                continue
            if any(self._attacked(f + rank, by_white=them) for f in path):
                continue
            yield Move(home, path[-1] + rank)

    def _attacked(self, square: str, by_white: bool) -> bool:
        pieces = self.pieces
        x, y = _xy(square)
        theirs = (lambda p: p.isupper()) if by_white else (lambda p: p.islower())

        def at(dx, dy):
            s = _sq(x + dx, y + dy)
            return pieces.get(s) if s is not None else None

        pawn_dy = -1 if by_white else 1          # a white pawn attacks from the rank below
        if any((p := at(dx, pawn_dy)) is not None and theirs(p) and p.lower() == "p" for dx in (-1, 1)):
            return True
        if any((p := at(dx, dy)) is not None and theirs(p) and p.lower() == "n" for dx, dy in _KNIGHT):
            return True
        if any((p := at(dx, dy)) is not None and theirs(p) and p.lower() == "k" for dx, dy in _KING):
            return True
        for dirs, kinds in ((_ROOK_DIRS, "rq"), (_BISHOP_DIRS, "bq")):
            for dx, dy in dirs:
                cx, cy = x + dx, y + dy
                while (s := _sq(cx, cy)) is not None:
                    if s in pieces:
                        if theirs(pieces[s]) and pieces[s].lower() in kinds:
                            return True
                        break
                    cx, cy = cx + dx, cy + dy
        return False

    def _in_check(self, white: bool) -> bool:
        king = "K" if white else "k"
        square = next((sq for sq, p in self.board if p == king), None)
        return square is not None and self._attacked(square, by_white=not white)


def _castle_rook(move: Move) -> "tuple[str, str]":
    rank = move.frm[1]
    return ("h" + rank, "f" + rank) if move.to[0] == "g" else ("a" + rank, "d" + rank)


def play_line(text: str, position: "Position | None" = None) -> "list[tuple[Position, Move]]":
    """
    A line of moves ('1. e4 e5 2. Nf3', SAN or UCI, move numbers allowed) read
    from position (the start by default): each move with the position it is
    played from. ValueError at the first move that is not legal.
    """
    pos = position or Position.start()
    out = []
    for word in text.replace(".", ". ").split():
        if word.rstrip(".").isdigit() or word in ("...", "."):
            continue
        move = pos.parse(word)
        out.append((pos, move))
        pos = pos.play(move)
    return out


# ── A book of openings ─────────────────────────────────────────────────────────
# Name -> (what it is called, its main line). Most take nothing: a line that
# captures is marked, as taking it back puts a piece back from off the board.
OPENINGS: "dict[str, tuple[str, str]]" = {
    "italian": ("Italian Game", "e4 e5 Nf3 Nc6 Bc4 Bc5"),
    "ruy_lopez": ("Ruy Lopez", "e4 e5 Nf3 Nc6 Bb5 a6"),
    "four_knights": ("Four Knights Game", "e4 e5 Nf3 Nc6 Nc3 Nf6"),
    "vienna": ("Vienna Game", "e4 e5 Nc3 Nf6"),
    "kings_pawn": ("King's Pawn Game", "e4 e5"),
    "queens_pawn": ("Queen's Pawn Game", "d4 d5"),
    "queens_gambit_declined": ("Queen's Gambit Declined", "d4 d5 c4 e6 Nc3 Nf6"),
    "slav": ("Slav Defence", "d4 d5 c4 c6 Nf3 Nf6"),
    "london": ("London System", "d4 d5 Bf4 Nf6 e3 e6"),
    "kings_indian": ("King's Indian Defence", "d4 Nf6 c4 g6 Nc3 Bg7"),
    "nimzo_indian": ("Nimzo-Indian Defence", "d4 Nf6 c4 e6 Nc3 Bb4"),
    "dutch": ("Dutch Defence", "d4 f5 c4 Nf6 g3 e6"),
    "french": ("French Defence", "e4 e6 d4 d5 Nc3 Nf6"),
    "caro_kann_advance": ("Caro-Kann, Advance Variation", "e4 c6 d4 d5 e5 Bf5"),
    "pirc": ("Pirc Defence", "e4 d6 d4 Nf6 Nc3 g6"),
    "closed_sicilian": ("Closed Sicilian", "e4 c5 Nc3 Nc6 g3 g6"),
    "english": ("English Opening", "c4 e5 Nc3 Nf6 g3"),
    "reti": ("Réti Opening", "Nf3 d5 c4 e6"),
    # These take a piece.
    "scandinavian": ("Scandinavian Defence", "e4 d5 exd5 Qxd5"),
    "queens_gambit_accepted": ("Queen's Gambit Accepted", "d4 d5 c4 dxc4"),
    "kings_gambit": ("King's Gambit Accepted", "e4 e5 f4 exf4"),
}


def opening(name: str) -> "tuple[str, str]":
    """(title, line) for an OPENINGS key or title, however spelled ('Ruy Lopez', 'ruy-lopez'); KeyError if none."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_").replace("'", "")
    for k, (title, line) in OPENINGS.items():
        if key in (k, title.lower().replace("-", "_").replace(" ", "_").replace("'", "")):
            return title, line
    raise KeyError(f"no opening {name!r} — known: {', '.join(OPENINGS)}")


def captures(line: str) -> bool:
    """Whether a line takes any piece."""
    return any(pos.captured_square(move) is not None for pos, move in play_line(line))


__all__ = ["Position", "Move", "Step", "CAPTURED", "OPENINGS", "SQUARES", "play_line", "opening", "captures",
           "colour_of", "name_of"]
