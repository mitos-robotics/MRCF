"""
engine.py — the chess engine that chooses the arm's moves in a game
(game.py): any engine that speaks UCI, the standard text protocol chess
engines talk over stdin/stdout — Stockfish by default.

    best_move(moves_uci, movetime_ms, searchmoves)   the engine's move after
                                                     the game so far
    Engine.score                                     what its last search
                                                     thought of the position

No chess library: rules.py already knows the moves, and UCI is a handful of
text lines — `uci` / `uciok`, `setoption`, `isready` / `readyok`, `position
startpos moves e2e4 e7e5`, `go movetime 1000 searchmoves ...`, `bestmove`.
The whole game goes with every search (not just the position), so the engine
knows what has been repeated.

`searchmoves` is how the arm plays only what it can make: game.py passes the
legal moves whose every square is inside the arm's reach, and the engine
picks the best of those.

Strength: UCI_LimitStrength with UCI_Elo where the engine has them
(Stockfish: 1320-3190), clamped to what it advertises; else its "Skill Level"
(0-20), scaled from the Elo asked for.

Every read waits at most a timeout, and an engine that does not answer in
time (or exits) raises EngineError — a hung engine must not hang the
controller with the arm connected.

Finding it: MRCF_CHESS_ENGINE, else config.ENGINE_PATH, else `stockfish` on
the PATH (Homebrew's /opt/homebrew/bin and /usr/local/bin are tried too, as
an IDE's PATH can leave them out). None: EngineError saying how to install
one (`brew install stockfish`).
"""

from __future__ import annotations

import atexit
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

from . import config as cfg

HANDSHAKE_TIMEOUT_S = 10.0     # uci -> uciok, isready -> readyok
SEARCH_GRACE_S = 10.0          # a search may run this much past its movetime before it counts as hung
SKILL_ELO = (1320, 3190)       # the Elo range a "Skill Level" 0-20 is spread over, for an engine without UCI_Elo
_SEARCH_PATHS = ("/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish")


class EngineError(RuntimeError):
    """The engine could not be found or started, or did not answer in time."""


def engine_path() -> str:
    """Where the engine is: MRCF_CHESS_ENGINE, config.ENGINE_PATH, stockfish on the PATH; EngineError if none."""
    for given in (os.environ.get("MRCF_CHESS_ENGINE"), getattr(cfg, "ENGINE_PATH", None)):
        if given:
            found = shutil.which(str(given)) or (str(given) if Path(str(given)).is_file() else None)
            if found is None:
                raise EngineError(f"no chess engine at {given!r} (MRCF_CHESS_ENGINE / config.ENGINE_PATH)")
            return found
    found = shutil.which("stockfish") or next((p for p in _SEARCH_PATHS if Path(p).is_file()), None)
    if found is None:
        raise EngineError("no chess engine found — install Stockfish (macOS: brew install stockfish; "
                          "Linux: apt install stockfish), or set MRCF_CHESS_ENGINE to any UCI engine")
    return found


class Engine:
    """
    One running UCI engine. Engine(path) starts it and shakes hands; best_move
    searches; close() ends it. The options it advertised are in .options:
    {name: {"type", "default", "min", "max"}}.
    """

    def __init__(self, path: "str | None" = None, elo: "int | None" = None,
                 handshake_timeout_s: float = HANDSHAKE_TIMEOUT_S, command: "list | None" = None):
        self.path = path or (None if command else engine_path())
        self.name = ""
        self.options: dict = {}
        self.elo: "int | None" = None
        self.score: "dict | None" = None          # {"cp": int} or {"mate": int}, the side to move's view
        self._timeout = float(handshake_timeout_s)
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        try:
            self._proc = subprocess.Popen(command or [self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError as exc:
            raise EngineError(f"could not start the chess engine {self.path!r}: {exc}") from exc
        threading.Thread(target=self._read, name="chess-engine-reader", daemon=True).start()
        try:
            self._send("uci")
            for line in self._until(lambda s: s == "uciok", self._timeout, "uciok"):
                if line.startswith("id name "):
                    self.name = line[len("id name "):].strip()
                elif line.startswith("option name "):
                    self._option(line)
            if elo is not None:
                self.set_elo(elo)
            self.ready()
        except EngineError:
            self._proc.kill()            # not left running behind a failed start
            raise

    # ── Talking to it ─────────────────────────────────────────────────────

    def _read(self) -> None:
        for line in self._proc.stdout:
            self._lines.put(line.strip())
        self._lines.put(None)                      # it exited

    def _send(self, line: str) -> None:
        if self._proc.poll() is not None:
            raise EngineError(f"the chess engine has exited (code {self._proc.returncode})")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EngineError(f"the chess engine stopped taking commands: {exc}") from exc

    def _until(self, done, timeout_s: float, waiting_for: str) -> "list[str]":
        """The lines it sends up to and including the first that done() accepts; EngineError after timeout_s."""
        deadline = time.monotonic() + timeout_s
        got = []
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise EngineError(f"the chess engine did not send {waiting_for!r} within {timeout_s:.0f} s")
            try:
                line = self._lines.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise EngineError(f"the chess engine exited while waiting for {waiting_for!r}")
            got.append(line)
            if done(line):
                return got

    def _option(self, line: str) -> None:
        """One `option name <name> type <type> [default x] [min a] [max b] ...` line, into .options."""
        words = line.split()
        try:
            t = words.index("type")
        except ValueError:
            return
        name = " ".join(words[2:t])
        info: dict = {"type": words[t + 1] if t + 1 < len(words) else ""}
        for key in ("default", "min", "max"):
            if key in words[t:]:
                i = t + words[t:].index(key)
                if i + 1 < len(words):
                    info[key] = words[i + 1]
        self.options[name] = info

    def ready(self) -> None:
        self._send("isready")
        self._until(lambda s: s == "readyok", self._timeout, "readyok")

    def set_option(self, name: str, value) -> None:
        value = str(value).lower() if isinstance(value, bool) else value
        self._send(f"setoption name {name} value {value}")

    # ── Strength ──────────────────────────────────────────────────────────

    def set_elo(self, elo: int) -> int:
        """
        Play at about elo: UCI_Elo, clamped to the engine's own range, if it
        has it; else Skill Level scaled over SKILL_ELO. The Elo it plays at.
        """
        elo = int(elo)
        if "UCI_Elo" in self.options and "UCI_LimitStrength" in self.options:
            info = self.options["UCI_Elo"]
            lo, hi = int(info.get("min", elo)), int(info.get("max", elo))
            elo = max(lo, min(hi, elo))
            self.set_option("UCI_LimitStrength", True)
            self.set_option("UCI_Elo", elo)
        elif "Skill Level" in self.options:
            info = self.options["Skill Level"]
            lo, hi = int(info.get("min", 0)), int(info.get("max", 20))
            share = (elo - SKILL_ELO[0]) / float(SKILL_ELO[1] - SKILL_ELO[0])
            level = max(lo, min(hi, round(lo + share * (hi - lo))))
            self.set_option("Skill Level", level)
            elo = round(SKILL_ELO[0] + (level - lo) / float(max(1, hi - lo)) * (SKILL_ELO[1] - SKILL_ELO[0]))
        self.elo = elo
        self.ready()
        return elo

    # ── Searching ─────────────────────────────────────────────────────────

    def best_move(self, moves_uci: "list[str]", movetime_ms: "int | None" = None,
                  searchmoves: "list[str] | None" = None, start_fen: "str | None" = None) -> "str | None":
        """
        The engine's move (UCI, e.g. 'g1f3') after moves_uci played from the
        start (or from start_fen); only among searchmoves when given. None if
        it has none (the game is over). .score is what it thought of the
        position, from the side to move's view, when it said.
        """
        ms = int(movetime_ms if movetime_ms is not None else getattr(cfg, "ENGINE_MOVETIME_MS", 1000))
        where = "startpos" if not start_fen else f"fen {start_fen}"
        # Whatever it sent before (a search given up on) is read past first:
        # a late bestmove taken as this search's would be a move for another position.
        self.ready()
        self._send(f"position {where}" + (" moves " + " ".join(moves_uci) if moves_uci else ""))
        go = f"go movetime {ms}"
        if searchmoves:
            go += " searchmoves " + " ".join(searchmoves)
        self._send(go)
        try:
            lines = self._until(lambda s: s.startswith("bestmove"), ms / 1000.0 + SEARCH_GRACE_S, "bestmove")
        except EngineError:
            self._proc.kill()            # hung: the next shared() starts a fresh one
            raise
        self.score = None
        for line in lines:
            if line.startswith("info") and " score " in line:
                words = line.split()
                i = words.index("score")
                if i + 2 < len(words) and words[i + 1] in ("cp", "mate"):
                    try:
                        self.score = {words[i + 1]: int(words[i + 2])}
                    except ValueError:
                        pass
        best = lines[-1].split()[1] if len(lines[-1].split()) > 1 else ""
        return None if best in ("", "(none)", "0000") else best

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._send("quit")
                self._proc.wait(timeout=2)
            except (EngineError, subprocess.TimeoutExpired):
                self._proc.kill()

    def alive(self) -> bool:
        return self._proc.poll() is None


# ── One engine for the session ─────────────────────────────────────────────────

_SHARED: "Engine | None" = None


def shared() -> Engine:
    """The session's engine, started the first time it is asked for (and again if it has exited)."""
    global _SHARED
    if _SHARED is None or not _SHARED.alive():
        _SHARED = Engine()
    return _SHARED


def close_shared() -> None:
    global _SHARED
    if _SHARED is not None:
        _SHARED.close()
        _SHARED = None


atexit.register(close_shared)


def score_text(score: "dict | None", white_view: bool) -> str:
    """'+0.35', '-1.20', 'mate in 3 for white' — score is from the side that searched; white_view: that was white."""
    if not score:
        return ""
    sign = 1 if white_view else -1
    if "mate" in score:
        n = score["mate"] * sign
        return f"mate in {abs(n)} for {'white' if n > 0 else 'black'}"
    return f"{score['cp'] * sign / 100.0:+.2f}"


__all__ = ["Engine", "EngineError", "engine_path", "shared", "close_shared", "score_text"]
