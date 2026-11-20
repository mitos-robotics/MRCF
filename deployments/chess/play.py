"""
play.py — chess moves made by the arm, and book openings played through.

    move(from_square, to_square)    the piece standing on from_square, whatever
                                    the scan named it, to to_square; a piece
                                    standing there is taken off the board first
    play_opening(opening, rounds)   a book opening (rules.OPENINGS), both sides,
                                    from the starting position, every move checked
                                    with a scan; then every move taken back, to the
                                    starting position again; rounds times over
    openings                        the book, and which of it this board's reach allows
    starting_position               the board is set up: checked, and every piece told what it is

By square, not by name: a scan names a piece for its colour and type as the
camera and the memory have them (white pieces have come out "black", tracks
have swapped), but the square a piece stands on is the camera's own outline
on a measured lattice, which has held.

What makes a run hold up, move after move:
  - every move is checked with a look (scan_scene, classify=False: where the
    pieces stand, no vision model asked what they are) against the position the
    rules say it made; a piece put down beside its square is carried onto it,
    one put down off centre is put down again (OFF_CENTRE_MM), a piece that
    vanished from a square nothing touched stops the run
  - a carry whose pick or place fails is tried again (CARRY_TRIES), from home
    and a fresh look; a piece that will not go down where it is going is set
    down nearby and gripped again there (_relay); a place that still cannot be
    made puts the piece back where it came from, so the board stays a position
    the run knows
  - a line that cannot be finished is taken back from where it stopped, so the
    next one starts from the starting position again
  - every move goes to logs/openings.jsonl (board_log.record_move), with each
    carry's time, tries and why it failed, and what the look found

The arm goes home after each look: a pick plans its approach from where the
arm starts, and from the park pose some leans that reach from home do not.
"""

from __future__ import annotations

import time

import numpy as np

from mrcf import safety, skill_executor
from mrcf.action_registry import dispatch
from mrcf.robot_env import RobotEnv
from mrcf.vlm import SceneObject, register_action

from . import board_log, rules
from . import config as cfg
from .skills import POLICIES, goto_observe_pose, scan_scene

# A piece whose foot is within this share of a square's pitch from its centre
# stands on it, when the scan put it on no square of its own.
ON_SQUARE_SHARE = 0.45
# A piece put down further than this from its square's centre is put down
# again. Places on the rig land 1.6-2.8 mm off (2026-11-03); a scan reads a
# foot to 2-4 mm; a piece 12 mm out stands a quarter of its foot over the line,
# toward a neighbour's jaws.
OFF_CENTRE_MM = 12.0
# A piece standing further than this from its square's centre, on a square the
# move did not touch, is out of place: something knocked or dropped it there.
# Pieces stood by hand and put down by the arm read 1-11.5 mm off (2026-11-04);
# a pawn squeezed out of the jaws landed between d1 and e2 and read 21 mm off d1.
DISPLACED_MM = 16.0
# A piece found this far from where it should stand, at most (in squares), is
# taken for that piece pushed aside, and carried back.
STRAY_SQUARES = 1.6
CARRY_TRIES = 2               # pick-and-place attempts per carry, the first included
LOOK_TRIES = 2                # looks before a square the move did not touch counts as changed


# ── Squares and pieces ─────────────────────────────────────────────────────────

def _squares() -> dict:
    from . import chessboard
    return chessboard.LAST_SQUARES


def square_xy(square: str) -> "np.ndarray | None":
    entry = _squares().get(square)
    return None if entry is None else np.asarray(entry["world"], float)[:2]


def pitch_m() -> float:
    """The board's square size in the arm's frame, from the last scan's lattice (40 mm until one)."""
    a, b = square_xy("a1"), square_xy("h1")
    return float(np.linalg.norm(a - b)) / 7.0 if a is not None and b is not None else 0.040


def _on_board(scene_state) -> list:
    return [o for o in scene_state.objects if o.position is not None and getattr(o, "square", None)]


def piece_on(scene_state, square: str) -> "SceneObject | None":
    """The piece standing on square in the scene, else None; the nearest its centre if two claim it."""
    centre = square_xy(square)
    on = [o for o in _on_board(scene_state) if o.square == square]
    if centre is None:
        return on[0] if on else None
    if not on:
        near = ON_SQUARE_SHARE * pitch_m()
        on = [o for o in scene_state.objects if o.position is not None
              and float(np.linalg.norm(np.asarray(o.position[:2], float) - centre)) <= near]
    return min(on, key=lambda o: float(np.linalg.norm(np.asarray(o.position[:2], float) - centre)), default=None)


def occupied(scene_state) -> "set[str]":
    """The squares the scene has a piece on."""
    return {o.square for o in _on_board(scene_state)}


def off_centre_mm(scene_state, square: str) -> "float | None":
    obj, centre = piece_on(scene_state, square), square_xy(square)
    if obj is None or centre is None:
        return None
    return 1000.0 * float(np.linalg.norm(np.asarray(obj.position[:2], float) - centre))


def reachable(square: str) -> bool:
    """Whether square is inside the arm's workspace, as the last scan's lattice puts it."""
    xy = square_xy(square)
    b = cfg.WORKSPACE_BOUNDS
    return xy is not None and b.x_min <= xy[0] <= b.x_max and b.y_min <= xy[1] <= b.y_max


def _ensure_target(scene_state, square: str) -> bool:
    """square as a place target in the scene: an empty square the scan left out is put back, if in reach."""
    if any(t.name == square for t in scene_state.targets):
        return True
    xy = square_xy(square)
    if xy is None or not reachable(square):
        return False
    entry = _squares()[square]
    scene_state.targets.append(SceneObject(square, f"empty square {square}", interactable=False,
                                           position=np.array([xy[0], xy[1], cfg.TABLE_Z]),
                                           pixel_xy=list(entry["pixel"])))
    return True


def _captured_piece(scene_state, at_xy) -> "SceneObject | None":
    """
    The piece put off the board at at_xy, as an object to pick: the scan does
    not look past the squares, so it is where the memory has it.
    """
    memory = scene_state.memory
    at_xy = np.asarray(at_xy, float)
    best, best_d = None, 0.5 * pitch_m()
    for state in ("present", "missing"):
        for t in (memory.in_state(state) if memory is not None else []):
            if t.place != cfg.CAPTURE_TARGET or t.position is None:
                continue
            d = float(np.linalg.norm(np.asarray(t.position[:2], float) - at_xy))
            if d <= best_d:
                best, best_d = t, d
    name = f"captured_{len([o for o in scene_state.objects if o.name.startswith('captured_')])}"
    obj = SceneObject(name, "a captured piece, off the board", position=np.array([at_xy[0], at_xy[1], cfg.TABLE_Z]))
    obj.square = None
    obj.geometry = None
    obj.reachable = True
    if best is not None:
        obj.name = memory.name(best)
        obj.track_id = best.id
        obj.geometry = {"base_diameter_m": best.foot_mm / 1000.0} if best.foot_mm else None
        obj.measured_shape = best.shape() if best.shape_current() else None
    scene_state.objects = [o for o in scene_state.objects if o.name != obj.name] + [obj]
    return obj


def teach(scene_state, position: rules.Position) -> int:
    """
    Every piece the scene has on a square, told what stands there as the rules
    have it — its type ("taught": trusted over the vision model's guess) and
    its colour, pinned. Then a pick plans it at its own height
    (config.PIECE_SIZES_M; unknown, every piece is planned as the tallest, a
    king's hover, which costs reach at the far edge), and a lean knows how tall
    the pieces it passes over are. The memory keeps it through moves; taught
    again after every check, as a scan can swap two pieces' tracks. How many
    were told something new.
    """
    memory = scene_state.memory
    if memory is None:
        return 0
    taught = 0
    for obj in _on_board(scene_state):
        piece = position.pieces.get(obj.square)
        track = memory.get(getattr(obj, "track_id", None)) if piece is not None else None
        if track is None:
            continue
        kind, colour = rules.name_of(piece).split()[1], rules.colour_of(piece)
        if track.label == kind and track.label_source in ("taught", "user") and \
                track.attributes.get("colour") == colour:
            continue
        if memory.set_label(track.id, kind, "taught", attributes={"colour": colour}):
            taught += 1
    return taught


# ── Acting ────────────────────────────────────────────────────────────────────

def _act(env, scene_state, action: str, **params) -> "tuple[bool, str]":
    """One registered action; whether it worked, and what it said."""
    skill_executor.clear_report()
    try:
        ok = dispatch({"action": action, **params}, env, scene_state)
    except Exception as exc:          # noqa: BLE001 — a skill's crash is that carry's failure, said
        if type(exc).__name__ == "MotionStopped":
            raise
        return False, f"{action} raised {exc!r}"
    said = skill_executor.last_report() or {}
    return ok is not False, str(said.get("message") or "")


def _holding(env) -> bool:
    try:
        return bool(POLICIES.holding(env))
    except Exception:                  # noqa: BLE001 — no gripper to ask
        return False


def look(env, scene_state) -> bool:
    """Where the pieces stand now: parked out of the camera's view, a scan with no vision model, then home."""
    goto_observe_pose(env)
    ok = scan_scene(env, scene_state, classify=False)
    POLICIES.home(env)
    return ok


def carry(env, scene_state, frm: str, to: str, captured_at=None) -> dict:
    """
    One piece from square frm (or rules.CAPTURED, picked at captured_at) to
    square to (or rules.CAPTURED; frm itself to put it down again). Tried
    CARRY_TRIES times; a pick again after a fresh look (it ends at home), a
    place again from where the last left the arm, not swung home with the
    piece. A place that cannot be made puts the piece back on frm.
    Returns a record: ok, why, tries, seconds, and where a capture was put (put_at).
    """
    t0 = time.monotonic()
    rec: dict = {"from": frm, "to": to, "ok": False, "why": "", "tries": 0}
    to_name = cfg.CAPTURE_TARGET if to == rules.CAPTURED else to
    place_set_out = t0                    # when the last place began: what moved since may be on its square
    for attempt in range(CARRY_TRIES):
        rec["tries"] = attempt + 1
        if attempt and not _holding(env):
            look(env, scene_state)        # where it stands now, and from home again
        elif attempt and (xy := square_xy(to)) is not None and safety.moved_near(env, xy, place_set_out) is not None:
            # Holding it, and the sentry saw something move by the square it is
            # going to (a hand, a piece knocked over): the board as it is now
            # before coming down there again.
            print(f"  [Play] Something moved near {to} — looking at the board before putting it down.")
            look(env, scene_state)
            if piece_on(scene_state, to) is not None:
                rec.update(why=f"{to} is taken now", taken=True)
                break
        held = scene_state.held_object() if _holding(env) else None
        if held is None:
            obj = (_captured_piece(scene_state, captured_at) if frm == rules.CAPTURED
                   else piece_on(scene_state, frm))
            if obj is None:
                rec["why"] = f"no piece on {frm}"
                continue
            if to not in (rules.CAPTURED, frm) and piece_on(scene_state, to) is not None:
                rec["why"] = f"{to} is taken"
                break
            # Where it is going, so making room for the pick sets nothing down there.
            scene_state.headed_for = to_name if to != frm else None
            try:
                ok, said = _act(env, scene_state, "pick", object_name=obj.name)
            finally:
                scene_state.headed_for = None
            if not ok and not _holding(env):
                rec["why"] = f"pick: {said or 'failed'}"
                continue
        if to_name == cfg.CAPTURE_TARGET:
            spot = scene_state.find(to_name, among="targets")
            if spot is None:
                rec["why"] = "no room left to put a captured piece"
                break
            rec["put_at"] = [float(v) for v in spot.position[:2]]
        elif not _ensure_target(scene_state, to):
            rec["why"] = f"{to} is out of reach"
            break
        place_set_out = time.monotonic()
        ok, said = _act(env, scene_state, "place", target_name=to_name)
        if ok:
            rec.update(ok=True, why="")
            break
        rec["why"] = f"place: {said or 'failed'}"
        if not _holding(env):
            break                       # let go of somewhere: only a look says where
    if not rec["ok"] and _holding(env) and to != rules.CAPTURED and not rec.get("taken"):
        _relay(env, scene_state, frm, to, rec)
    if not rec["ok"] and _holding(env) and frm != rules.CAPTURED and _ensure_target(scene_state, frm):
        # Not where it was going: back where it came from, so the board is a
        # position the run knows, not a piece in the air.
        ok, said = _act(env, scene_state, "place", target_name=frm)
        rec["put_back"] = ok
    rec["seconds"] = round(time.monotonic() - t0, 1)
    return rec


def staging_square(scene_state, frm: str, to: str) -> "str | None":
    """
    An empty square to set a piece down on for a moment, on its way to `to`:
    in reach, not `frm` or `to` or beside a piece on either side along the way,
    the nearest `to` — and not further from the arm than it, where reach runs
    out. None if there is none.
    """
    goal = square_xy(to)
    if goal is None:
        return None
    taken = occupied(scene_state)
    base = np.zeros(2)
    best, best_d = None, None
    for t in scene_state.targets:
        sq = t.name
        if sq not in rules.SQUARES or sq in (frm, to) or sq in taken or not reachable(sq):
            continue
        xy = square_xy(sq)
        if xy is None or np.linalg.norm(xy - base) > np.linalg.norm(goal - base) + 1e-6:
            continue
        d = float(np.linalg.norm(xy - goal))
        if best_d is None or d < best_d:
            best, best_d = sq, d
    return best


def _relay(env, scene_state, frm: str, to: str, rec: dict) -> None:
    """
    A piece in the jaws that will not go down on `to`: set down on a square
    nearby (staging_square), gripped again there — at the lean that square
    needs, with a fresh grip — and carried on to `to`. The user's suggestion,
    2026-11-04; that morning a pawn picked at -20 deg on e5 and carried to e7
    at -40 slipped out on the way.
    """
    stage = staging_square(scene_state, frm, to)
    if stage is None or not _ensure_target(scene_state, stage):
        return
    print(f"  [Play] Setting it down on {stage} to grip it again for {to}.")
    rec["relay"] = stage
    ok, said = _act(env, scene_state, "place", target_name=stage)
    if not ok:
        rec["why"] += f"; set down on {stage}: {said or 'failed'}"
        return
    obj = piece_on(scene_state, stage)
    if obj is None:
        return
    # From home: a pick plans its approach from where the arm starts, and from
    # over the square it was just put down on no lean reached every phase of
    # the pick again (2026-11-04, the knight on b6, three tries).
    POLICIES.home(env)
    ok, said = _act(env, scene_state, "pick", object_name=obj.name)
    if not ok:
        rec["why"] += f"; again from {stage}: pick: {said or 'failed'}"
        rec["left_on"] = stage if not _holding(env) else None
        return
    ok, said = _act(env, scene_state, "place", target_name=to)
    if ok:
        rec.update(ok=True, why="")
    else:
        rec["why"] += f"; from {stage}: place: {said or 'failed'}"


def after_steps(occupied_before: "set[str]", steps) -> "set[str]":
    """The squares with a piece on once steps are made (rules.CAPTURED is off the board)."""
    out = set(occupied_before)
    for s in steps:
        out.discard(s.frm)
        if s.to != rules.CAPTURED:
            out.add(s.to)
    return out


def around(squares: "set[str]") -> "set[str]":
    """The squares and every square touching one of them."""
    out = set()
    for sq in squares:
        f, r = rules.FILES.index(sq[0]), int(sq[1]) - 1
        out |= {rules.FILES[f + df] + str(r + dr + 1) for df in (-1, 0, 1) for dr in (-1, 0, 1)
                if 0 <= f + df < 8 and 0 <= r + dr < 8}
    return out


def check(env, scene_state, want: "set[str]", touched: "set[str]") -> dict:
    """
    Look, and compare the board with want, the squares that should have a
    piece: those that should and have none (missing), those with one that
    should not (extra), how far the pieces on touched squares stand from their
    centres (off_mm), and the pieces elsewhere standing more than DISPLACED_MM
    from theirs (displaced). Where the arm worked — the touched squares and
    those around them — a difference counts; further off, where it did not, it
    is a doubt about the look, noted (doubts) but not counted: the overhead
    camera misses a king on e1 now and then (2026-11-04), and one look's miss
    must not stop a run. A difference that counts is believed only when
    LOOK_TRIES looks agree.
    """
    near = around(touched)
    for n in range(LOOK_TRIES):  # noqa: B007 — n is read after the loop (looks)
        look(env, scene_state)
        seen = occupied(scene_state)
        missing, extra = want - seen, seen - want
        displaced = {sq: round(d, 1) for sq in (seen - touched) & near
                     if (d := off_centre_mm(scene_state, sq)) is not None and d > DISPLACED_MM}
        if not ((missing | extra) & near - touched) and not displaced:
            break
    doubts = sorted((missing | extra) - near)
    if doubts:
        print(f"  [Play] Not counted, away from where the arm worked: the look differs on {', '.join(doubts)}.")
    off = {sq: round(d, 1) for sq in touched if sq in want and (d := off_centre_mm(scene_state, sq)) is not None}
    return {"missing": sorted(missing & near), "extra": sorted(extra & near), "off_mm": off,
            "displaced": displaced, "doubts": doubts, "looks": n + 1}


def wrong_with(found: dict) -> str:
    """What a check found wrong with the board, said; "" if nothing."""
    return ", ".join([f"nothing on {s}" for s in found["missing"]] +
                     [f"a piece on {s}" for s in found["extra"]] +
                     [f"the piece on {s} {mm:.0f} mm off its centre" for s, mm in found.get("displaced", {}).items()])


def put_right(env, scene_state, touched: "set[str]", found: dict) -> "list[dict]":
    """
    What a look found wrong on the squares a move touched, carried right: a
    piece found beside the square it should stand on (STRAY_SQUARES), onto it;
    one off its square's centre by more than OFF_CENTRE_MM, down again there.
    The carries made.
    """
    done = []
    pitch = pitch_m()
    strays = [sq for sq in found["extra"]]
    for sq in found["missing"]:
        if sq not in touched:
            continue
        centre = square_xy(sq)
        near = [s for s in strays if centre is not None and square_xy(s) is not None
                and float(np.linalg.norm(square_xy(s) - centre)) <= STRAY_SQUARES * pitch]
        if not near:
            continue
        stray = min(near, key=lambda s: float(np.linalg.norm(square_xy(s) - centre)))
        strays.remove(stray)
        print(f"  [Play] The piece meant for {sq} stands on {stray} — carrying it over.")
        done.append({"repair": "stray", **carry(env, scene_state, stray, sq)})
    for sq, mm in found["off_mm"].items():
        if mm > OFF_CENTRE_MM and sq not in found["missing"]:
            print(f"  [Play] The piece on {sq} stands {mm:.0f} mm off its centre — putting it down again.")
            done.append({"repair": "off centre", "off_mm": mm, **carry(env, scene_state, sq, sq)})
    return done


def make(env, scene_state, steps: "list[rules.Step]", captured: list, verify: bool = True,
         before: "set[str] | None" = None) -> dict:
    """
    The carries of one move (or of taking one back), then a look against the
    squares they should leave occupied, putting right what it can. A carry
    that fails has the carries before it taken back, so the board is as it was
    before the move. captured: where each captured piece went, in order, for
    taking it back (_carry_step); kept up to date. A record of it all; its
    board_wrong says the board is not as the run believes. before: the squares
    with a piece before the move, as the rules have them (else as the last
    look saw them, which may have missed one).
    """
    t0 = time.monotonic()
    rec: dict = {"carries": [], "ok": True}
    occupied_before = set(before) if before is not None else occupied(scene_state)
    done: list = []
    for i, step in enumerate(steps):
        print(f"  [Play] {rules.name_of(step.piece).capitalize()}: {step.frm} -> {step.to}")
        c = _carry_step(env, scene_state, step, captured)
        rec["carries"].append(c)
        if (not c["ok"] and i == len(steps) - 1 and step.to != rules.CAPTURED and "put_back" not in c
                and c["why"].startswith("place") and not _holding(env)):
            # Let go of on the way, or short of the square: the look below
            # finds where it stands, and a piece standing near the square is
            # carried on to it (put_right). 2026-11-04: a pawn bound for e7
            # stood on e6.
            print(f"  [Play] The piece left the jaws on the way to {step.to} — looking for it.")
            rec["dropped"] = c["why"]
            done.append(step)
            continue
        if not c["ok"]:
            rec.update(ok=False, why=f"{step.frm}->{step.to}: {c['why']}")
            if not c.get("put_back", True) and step.frm != rules.CAPTURED:
                rec["board_wrong"] = True
            break
        done.append(step)
    if not rec["ok"] and done and not rec.get("board_wrong"):
        print("  [Play] Undoing the part of the move already made.")
        for step in reversed(done):
            back = rules.Step(step.to, step.frm, step.piece)
            c = _carry_step(env, scene_state, back, captured)
            rec["carries"].append({**c, "undo": True})
            if not c["ok"]:
                rec["board_wrong"] = True
                break
            done.remove(step)
    touched = {sq for s in steps for sq in (s.frm, s.to) if sq != rules.CAPTURED}
    want = after_steps(occupied_before, done)
    if verify:
        found = check(env, scene_state, want, touched)
        if rec["ok"] and (found["missing"] or found["extra"] or
                          any(mm > OFF_CENTRE_MM for mm in found["off_mm"].values())):
            fixes = put_right(env, scene_state, touched, found)
            if fixes:
                rec["repairs"] = fixes
                found = check(env, scene_state, want, touched)
        rec["check"] = found
        wrong = wrong_with(found)
        if wrong:
            rec.update(ok=False, board_wrong=True,
                       why=(rec["why"] + "; " if rec.get("why") else "") + f"the board is not as it should be: {wrong}")
    rec["seconds"] = round(time.monotonic() - t0, 1)
    return rec


def _carry_step(env, scene_state, step: "rules.Step", captured: list) -> dict:
    """
    carry() for one rules.Step. captured: where each captured piece was put,
    in order — a stack, as moves are taken back last first: a capture pushes
    where it put the piece, taking one back pops it.
    """
    where = captured[-1] if step.frm == rules.CAPTURED and captured else None
    if step.frm == rules.CAPTURED and where is None:
        return {"from": step.frm, "to": step.to, "ok": False, "tries": 0, "piece": step.piece,
                "why": f"where the piece taken on {step.to} was put is not known"}
    c = carry(env, scene_state, step.frm, step.to, captured_at=where)
    c["piece"] = step.piece
    if c["ok"] and step.to == rules.CAPTURED:
        captured.append(c.get("put_at"))
    elif c["ok"] and step.frm == rules.CAPTURED:
        captured.pop()
    return c


# ── Openings ──────────────────────────────────────────────────────────────────

def unreachable(line: str) -> "list[str]":
    """The squares a line (and taking it back) needs the arm on that it cannot reach, by the last scan's lattice."""
    need = set()
    for pos, move in rules.play_line(line):
        for s in pos.steps(move):
            need |= {s.frm, s.to} - {rules.CAPTURED}
    return sorted(sq for sq in need if not reachable(sq))


def _resolve(spec: str) -> "list[tuple[str, str, str]]":
    """(key, title, line) for 'all', one opening, a comma list of them, or a line of moves itself."""
    spec = str(spec or "").strip()
    if spec.lower() in ("all", "book", "every", "*"):
        return [(k, t, line) for k, (t, line) in rules.OPENINGS.items() if not rules.captures(line)]
    out = []
    for part in [p for p in spec.split(",") if p.strip()]:
        try:
            title, line = rules.opening(part)
            out.append((part.strip(), title, line))
        except KeyError:
            rules.play_line(part)              # a line of moves: ValueError if it is not legal
            out.append(("line", part.strip(), part.strip()))
    return out


def play_line(env, scene_state, key: str, title: str, line: str, round_no: int = 1,
              take_back: bool = True) -> dict:
    """
    One opening from the starting position and back: each move made and
    checked, then taken back, last first. A move that fails stops the line,
    and what was made of it is taken back. A record of the run.
    """
    plies = rules.play_line(line)
    run: dict = {"opening": key, "title": title, "line": line, "round": round_no, "moves": 0, "taken_back": 0,
                 "ok": False}
    t0 = time.monotonic()
    start = plies[0][0]
    used = {sq for pos, move in plies for st in pos.steps(move) for sq in (st.frm, st.to)} - {rules.CAPTURED}
    if (start.occupied() ^ occupied(scene_state)) & around(used) or any(
            (d := off_centre_mm(scene_state, sq)) is not None and d > DISPLACED_MM for sq in start.occupied()):
        found = check(env, scene_state, start.occupied(), used)
        if wrong_with(found):
            run["why"] = "the board is not at the starting position: " + wrong_with(found)
            print(f"  [Play] {run['why']}.")
            return run
    n = teach(scene_state, start)
    if n:
        print(f"  [Play] Told {n} pieces what they are, from the starting position.")
    print(f"\n  [Play] {title}, round {round_no}: {line}")
    captured: list = []
    made: list = []
    board_bad = False
    for i, (before, move) in enumerate(plies):
        san = before.san(move)
        print(f"  [Play] {i // 2 + 1}.{'' if i % 2 == 0 else '..'} {san}")
        rec = make(env, scene_state, before.steps(move), captured, before=before.occupied())
        board_log.record_move(opening=key, round=round_no, ply=i + 1, san=san, uci=move.uci, phase="play", **rec)
        if not rec["ok"]:
            run["why"] = f"{san}: {rec['why']}"
            print(f"  [Play] {san} failed — {rec['why']}.")
            board_bad = bool(rec.get("board_wrong"))
            break
        made.append((before, move))
        run["moves"] += 1
        teach(scene_state, before.play(move))
    if take_back and not board_bad:
        for before, move in reversed(made):
            san = before.san(move)
            print(f"  [Play] Taking back {san}")
            rec = make(env, scene_state, before.take_back_steps(move), captured, before=before.play(move).occupied())
            board_log.record_move(opening=key, round=round_no, ply=plies.index((before, move)) + 1, san=san,
                                  uci=move.uci, phase="take back", **rec)
            if not rec["ok"]:
                run["why"] = (run.get("why", "") + "; " if run.get("why") else "") + \
                    f"taking back {san}: {rec['why']}"
                print(f"  [Play] Taking back {san} failed — {rec['why']}.")
                board_bad = True
                break
            run["taken_back"] += 1
            teach(scene_state, before)
    run["ok"] = run["moves"] == len(plies) and (not take_back or run["taken_back"] == len(plies))
    run["board_ok"] = not board_bad
    run["seconds"] = round(time.monotonic() - t0, 1)
    return run


@register_action(
    "play_opening",
    description=(
        "Play a chess opening through with the arm, both sides, from the starting position, checking "
        "every move, then take every move back to the starting position; 'rounds' times. 'opening' is a "
        "book opening's name (e.g. 'italian', 'ruy lopez'), several separated by commas, 'all' for every "
        "one that takes nothing, or a line of moves ('e4 e5 Nf3')."
    ),
    parameters={"opening": "Which opening(s).", "rounds": "How many times through (default 1)."},
)
def skill_play_opening(env: RobotEnv, scene_state, opening, rounds=1, **_):
    try:
        todo = _resolve(str(opening))
    except ValueError as exc:
        print(f"  [Play] {exc}")
        return False
    if not todo:
        print(f"  [Play] No opening called {opening!r}. Try /openings.")
        return False
    if _holding(env):
        print("  [Play] Something is in the jaws — put it down first.")
        return False
    if not _squares():
        look(env, scene_state)
    playable = []
    for key, title, line in todo:
        far = unreachable(line)
        if far:
            print(f"  [Play] Leaving out {title}: {', '.join(far)} out of the arm's reach.")
        else:
            playable.append((key, title, line))
    if not playable:
        return False
    POLICIES.home(env)
    runs = []
    for r in range(1, int(rounds) + 1):
        for key, title, line in playable:
            run = play_line(env, scene_state, key, title, line, r)
            runs.append(run)
            board_log.record_move(phase="summary", **run)
            if not run.get("board_ok", True) or (run["moves"] == 0 and run.get("why", "").startswith("the board")):
                print("  [Play] Stopping: the board is not a position the run knows — set it up and look again.")
                _summary(runs)
                return False
    _summary(runs)
    return all(r["ok"] for r in runs)


def _summary(runs: list) -> None:
    ok = sum(1 for r in runs if r["ok"])
    print(f"\n  [Play] {ok} of {len(runs)} opening runs played through and taken back.")
    for r in runs:
        mark = "ok  " if r["ok"] else "FAIL"
        why = f" — {r['why']}" if r.get("why") else ""
        print(f"    {mark} {r['title']} (round {r['round']}): {r['moves']} moves, {r['taken_back']} taken back, "
              f"{r.get('seconds', 0):.0f} s{why}")


@register_action(
    "move",
    description=(
        "Move the piece standing on from_square to to_square, as a chess move is made: a piece on "
        "to_square is taken off the board first. Squares as 'e2', 'e4'; the same square twice puts the "
        "piece down again in the middle of it."
    ),
    parameters={"from_square": "The square the piece stands on, e.g. 'e2'.",
                "to_square": "The square it goes to, e.g. 'e4'."},
)
def skill_move(env: RobotEnv, scene_state, from_square, to_square, **_):
    frm, to = str(from_square).strip().lower(), str(to_square).strip().lower()
    for sq in (frm, to):
        if sq not in rules.SQUARES:
            print(f"  [Move] {sq!r} is not a square.")
            return False
    if _holding(env):
        print("  [Move] Something is in the jaws — put it down first.")
        return False
    if piece_on(scene_state, frm) is None:
        print(f"  [Move] No piece on {frm} in the last look — looking again.")
        look(env, scene_state)
        if piece_on(scene_state, frm) is None:
            print(f"  [Move] Still no piece on {frm}.")
            return False
    steps = []
    if to != frm and piece_on(scene_state, to) is not None:
        steps.append((to, rules.CAPTURED))
    steps.append((frm, to))
    for a, b in steps:
        rec = carry(env, scene_state, a, b)
        board_log.record_move(phase="move", **rec)
        if not rec["ok"]:
            print(f"  [Move] {a} -> {b} failed: {rec['why']}.")
            return False
    return True


@register_action(
    "starting_position",
    description=("The board is set up in chess's starting position: look, check it is, and remember what "
                 "every piece is (its type and colour) from where it stands."),
    always=True,
)
def skill_starting_position(env: RobotEnv, scene_state, **_):
    start = rules.Position.start()
    found = check(env, scene_state, start.occupied(), set(rules.SQUARES))
    if found["extra"] or found["displaced"]:
        print(f"  [Play] Not the starting position: {wrong_with(dict(found, missing=[]))}.")
        return False
    if found["missing"]:
        print(f"  [Play] Not seen on {', '.join(found['missing'])} — taking it as the starting position anyway "
              "(a look can miss a piece; an opening's own squares are checked again before it is played).")
    print(f"  [Play] The starting position — told {teach(scene_state, start)} pieces what they are.")
    return True


@register_action("openings", description="List the book of chess openings the arm can play.", always=True)
def skill_openings(**_):
    for key, (title, line) in rules.OPENINGS.items():
        far = unreachable(line) if _squares() else []
        notes = (["takes a piece"] if rules.captures(line) else []) + (
            [f"out of reach: {', '.join(far)}"] if far else [])
        print(f"  {key:24} {title:32} {line}" + (f"   ({'; '.join(notes)})" if notes else ""))
    return True


__all__ = ["carry", "check", "make", "play_line", "piece_on", "occupied", "look", "unreachable"]
