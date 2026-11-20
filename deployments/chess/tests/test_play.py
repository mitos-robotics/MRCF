"""Chess moves and openings by the arm (play.py), offline: a simulated board stands in for the arm and camera."""

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import cv2  # noqa: F401 — the deployment's perception needs it
except ImportError:
    pytest.skip("needs OpenCV", allow_module_level=True)

from mrcf.vlm import SceneObject, SceneState

from deployments.chess import chessboard, play, rules
from deployments.chess import config as cfg

PITCH = 0.040


def centre(square: str) -> np.ndarray:
    f, r = rules.FILES.index(square[0]), int(square[1]) - 1
    return np.array([0.12 + PITCH * r, (3.5 - f) * PITCH])


class Board:
    """
    The pieces where they stand, what the jaws hold, and the arm's actions
    acting on them. fail: (action, square) -> how many more times it fails;
    knock: (action, square) -> a square that action pushes the piece onto
    instead; offset: square -> how far off centre a place there lands (m).
    """

    def __init__(self, position: rules.Position):
        self.at = {sq: np.array(centre(sq)) for sq in position.occupied()}
        self.off_board: list = []
        self.held = None
        self.fail: dict = {}
        self.stray: dict = {}
        self.offset: dict = {}
        self.drop: dict = {}             # target square -> the square a place there drops the piece on
        self.carried: list = []
        self.looks = 0

    def occupied(self) -> set:
        return set(self.at)

    def look(self, env, scene_state):
        self.looks += 1
        scene_state.objects, scene_state.targets = [], []
        for sq, xy in sorted(self.at.items()):
            obj = SceneObject(f"piece_{sq}", "a piece", position=np.array([xy[0], xy[1], cfg.TABLE_Z]))
            obj.square, obj.track_id, obj.geometry = sq, None, None
            scene_state.objects.append(obj)
        for sq in rules.SQUARES:
            if sq not in self.at:
                c = centre(sq)
                scene_state.targets.append(SceneObject(sq, f"empty square {sq}", interactable=False,
                                                       position=np.array([c[0], c[1], cfg.TABLE_Z])))
        spot = np.array([0.10, 0.20 - 0.04 * len(self.off_board)])
        scene_state.targets.append(SceneObject(cfg.CAPTURE_TARGET, "off the board", interactable=False,
                                               position=np.array([spot[0], spot[1], cfg.TABLE_Z])))
        return True

    def _fails(self, key) -> bool:
        if self.fail.get(key, 0) > 0:
            self.fail[key] -= 1
            return True
        return False

    def act(self, env, scene_state, action, object_name=None, target_name=None):
        if action == "pick":
            obj = scene_state.find(object_name, among="objects")
            sq = getattr(obj, "square", None)
            if self._fails(("pick", sq or "captured")):
                return False, "closed on nothing"
            if sq is not None:
                self.at.pop(sq)
            else:
                self.off_board = [p for p in self.off_board if np.linalg.norm(p - obj.position[:2]) > 1e-6]
            self.held, obj.position = obj, None
            return True, "grasped"
        if action == "place":
            if target_name in self.drop:
                land = self.drop.pop(target_name)
                self.at[land] = centre(land) + np.array([0.012, 0.0])
                self.held = None
                return False, "the object slipped out of the jaws on the way"
            if self._fails(("place", target_name)):
                return False, "descent stalled — still holding the object"
            tgt = scene_state.find(target_name, among="targets")
            self.carried.append((getattr(self.held, "square", None), target_name))
            if target_name == cfg.CAPTURE_TARGET:
                self.off_board.append(np.asarray(tgt.position[:2], float))
            else:
                land = self.stray.pop(target_name, target_name)
                self.at[land] = centre(land) + self.offset.pop(target_name, np.zeros(2))
            self.held.position = np.asarray(tgt.position, float)
            self.held.square = target_name
            self.held = None
            return True, "placed"
        raise AssertionError(action)


@pytest.fixture
def board(monkeypatch):
    def setup(position=None):
        position = position or rules.Position.start()
        b = Board(position)
        squares = {sq: {"world": list(centre(sq)), "pixel": [0.0, 0.0]} for sq in rules.SQUARES}
        monkeypatch.setattr(chessboard, "LAST_SQUARES", squares)
        monkeypatch.setattr(play, "look", b.look)
        monkeypatch.setattr(play, "_act", b.act)
        monkeypatch.setattr(play, "_holding", lambda env: b.held is not None)
        monkeypatch.setattr(play, "POLICIES", SimpleNamespace(home=lambda env: None, holding=lambda env: b.held))
        monkeypatch.setattr(play.board_log, "record_move", lambda **kw: b.__dict__.setdefault("log", []).append(kw))
        monkeypatch.setattr(cfg, "WORKSPACE_BOUNDS", SimpleNamespace(x_min=0.05, x_max=0.45, y_min=-0.25, y_max=0.25))
        state = SceneState()
        b.look(None, state)
        return b, state
    return setup


def test_a_line_is_played_checked_after_every_move_and_taken_back(board):
    b, state = board()
    run = play.play_line(None, state, "italian", *rules.OPENINGS["italian"])
    assert run["ok"] and run["moves"] == 6 and run["taken_back"] == 6
    assert b.occupied() == rules.Position.start().occupied()
    assert b.carried[:2] == [("e2", "e4"), ("e7", "e5")]
    assert b.carried[-1] == ("e4", "e2")
    assert b.looks >= 1 + 12                 # the first, then one after every move made or taken back
    assert [r["phase"] for r in b.log].count("take back") == 6


def test_a_capture_goes_off_the_board_and_comes_back_on_its_square(board):
    b, state = board()
    run = play.play_line(None, state, "scandinavian", *rules.OPENINGS["scandinavian"])
    assert run["ok"], run
    assert ("d5", cfg.CAPTURE_TARGET) in b.carried            # exd5 takes the pawn first
    assert b.occupied() == rules.Position.start().occupied()
    assert not b.off_board


def test_a_failed_pick_is_tried_again_after_a_look(board):
    b, state = board()
    b.fail[("pick", "g1")] = 1
    run = play.play_line(None, state, "line", "Nf3", "Nf3")
    assert run["ok"]
    made = next(r for r in b.log if r.get("san") == "Nf3" and r["phase"] == "play")
    assert made["carries"][0]["tries"] == 2


def test_a_place_that_cannot_be_made_puts_the_piece_back_and_the_line_stops_cleanly(board):
    b, state = board()
    b.fail[("place", "e5")] = 99                  # not even after setting it down nearby
    run = play.play_line(None, state, "kings_pawn", *rules.OPENINGS["kings_pawn"])
    assert not run["ok"] and run["moves"] == 1 and run["board_ok"]
    assert "e5" in run["why"]
    # e7 back on e7, and e4 taken back: the starting position again.
    assert b.occupied() == rules.Position.start().occupied()


def test_a_capture_whose_second_carry_fails_puts_the_taken_piece_back(board):
    b, state = board()
    pos = rules.Position.start()
    for word in "e4 d5".split():
        move = pos.parse(word)
        pos = pos.play(move)
    b2, state = board(pos)
    b2.fail[("place", "d5")] = play.CARRY_TRIES + 1        # the move's tries and its set-down-nearby
    rec = play.make(None, state, pos.steps(pos.parse("exd5")), [])
    assert not rec["ok"] and not rec.get("board_wrong")
    assert b2.occupied() == pos.occupied()
    assert not b2.off_board


def test_a_piece_put_down_on_the_next_square_is_carried_onto_its_own(board):
    b, state = board()
    b.stray["e4"] = "e5"
    rec = play.make(None, state, [rules.Step("e2", "e4", "P")], [])
    assert rec["ok"], rec
    assert rec["repairs"][0]["repair"] == "stray"
    assert "e4" in b.occupied() and "e5" not in b.occupied()


def test_a_piece_dropped_short_of_its_square_is_found_and_carried_on_to_it(board):
    # 2026-11-04: a pawn bound for e7 left the jaws and stood on e6.
    b, state = board(rules.Position.start().play(rules.Move("e2", "e4")).play(rules.Move("e7", "e5")))
    b.drop["e7"] = "e6"
    rec = play.make(None, state, [rules.Step("e5", "e7", "p")], [])
    assert rec["ok"], rec
    assert rec["dropped"] and rec["repairs"][0]["repair"] == "stray"
    assert b.occupied() == rules.Position.start().play(rules.Move("e2", "e4")).occupied()


def test_a_piece_put_down_off_centre_is_put_down_again(board):
    b, state = board()
    b.offset["e4"] = np.array([0.015, 0.0])
    rec = play.make(None, state, [rules.Step("e2", "e4", "P")], [])
    assert rec["ok"], rec
    assert rec["repairs"][0]["repair"] == "off centre"
    assert np.linalg.norm(b.at["e4"] - centre("e4")) < 1e-9


def test_a_piece_that_vanished_from_a_square_nothing_touched_stops_the_run(board, monkeypatch):
    b, state = board()
    real_look = b.look

    def knocked_over(env, scene_state):
        if b.carried:
            b.at.pop("d2", None)               # the e-pawn's carry knocked the d-pawn off its square
        return real_look(env, scene_state)

    monkeypatch.setattr(play, "look", knocked_over)
    run = play.play_line(None, state, "kings_pawn", *rules.OPENINGS["kings_pawn"])
    assert not run["ok"] and not run["board_ok"]
    assert "d2" in run["why"]
    assert run["taken_back"] == 0              # nothing more is moved on a board the run does not know


def test_a_piece_knocked_off_the_middle_of_a_square_nothing_touched_stops_the_run(board, monkeypatch):
    # 2026-11-04: a pawn squeezed out of the jaws landed between d1 and e2; the
    # scan put it on d1, where the queen was not outlined — d1 still "occupied".
    b, state = board()
    real_look = b.look

    def dropped_beside(env, scene_state):
        if b.carried:
            b.at["d1"] = centre("d1") + np.array([0.015, -0.012])
        return real_look(env, scene_state)

    monkeypatch.setattr(play, "look", dropped_beside)
    run = play.play_line(None, state, "kings_pawn", *rules.OPENINGS["kings_pawn"])
    assert not run["ok"] and not run["board_ok"]
    assert "d1" in run["why"] and "off its centre" in run["why"]


def test_a_line_from_a_board_not_at_the_start_does_not_move(board):
    b, state = board(rules.Position.start().play(rules.Move("e2", "e4")))
    run = play.play_line(None, state, "italian", *rules.OPENINGS["italian"])
    assert not run["ok"] and run["moves"] == 0 and not b.carried
    assert "e2" in run["why"] and "e4" in run["why"]


def test_openings_needing_squares_out_of_reach_are_left_out(board, monkeypatch):
    b, state = board()
    # Only up to rank 6 in reach: no black piece can move.
    monkeypatch.setattr(cfg, "WORKSPACE_BOUNDS", SimpleNamespace(x_min=0.05, x_max=0.12 + PITCH * 5.5,
                                                                 y_min=-0.25, y_max=0.25))
    assert "e7" in play.unreachable(rules.OPENINGS["kings_pawn"][1])
    assert not play.skill_play_opening(None, state, "kings_pawn")
    assert not b.carried


def test_move_takes_the_piece_on_the_square_it_goes_to_off_the_board_first(board):
    b, state = board(rules.Position.from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - -"))
    assert play.skill_move(None, state, "e4", "d5")
    assert b.carried == [("d5", cfg.CAPTURE_TARGET), ("e4", "d5")]
    assert b.occupied() == {"e8", "e1", "d5"}
    b.offset["d5"] = np.array([0.0, 0.014])
    assert play.skill_move(None, state, "d5", "d5")             # put down again, not taken
    assert b.carried[-1] == ("d5", "d5") and not b.off_board[1:]
    assert not play.skill_move(None, state, "a3", "a4")        # nothing there
    assert not play.skill_move(None, state, "e9", "a4")        # not a square


def test_all_means_every_book_line_that_takes_nothing():
    keys = [k for k, _, _ in play._resolve("all")]
    assert "italian" in keys and "scandinavian" not in keys
    assert [k for k, _, _ in play._resolve("italian, ruy lopez")] == ["italian", "ruy lopez"]
    assert play._resolve("e4 e5 Nf3")[0][2] == "e4 e5 Nf3"
    with pytest.raises(ValueError):
        play._resolve("e4 e4")


def test_from_the_starting_position_every_piece_is_told_what_it_is_and_keeps_it_through_a_swap(board):
    # Unknown, every piece is planned as a king (an 82 mm hover), which costs
    # reach at the far edge, and a steep lean takes the pawn in front for one.
    from mrcf.object_memory import Detection, ObjectMemory
    b, state = board()
    memory = ObjectMemory()
    tracks, _ = memory.observe([Detection(position=tuple(centre(o.square)), place=o.square, foot_mm=22.0,
                                          attributes={"colour": "black"}) for o in state.objects])
    for o, t in zip(state.objects, tracks):
        o.track_id = t.id
    state.memory = memory
    start = rules.Position.start()
    assert play.teach(state, start) == 32
    on = {o.square: memory.get(o.track_id) for o in state.objects}
    assert (on["e2"].label, on["e2"].attributes["colour"]) == ("pawn", "white")
    assert (on["g8"].label, on["d1"].label) == ("knight", "queen")
    assert play.teach(state, start) == 0                      # nothing new to say
    # Two tracks swapped by a scan: told again from where they stand.
    e2, d1 = next(o for o in state.objects if o.square == "e2"), next(o for o in state.objects if o.square == "d1")
    e2.track_id, d1.track_id = d1.track_id, e2.track_id
    assert play.teach(state, start) == 2
    assert memory.get(e2.track_id).label == "pawn" and memory.get(d1.track_id).label == "queen"


def test_a_piece_that_will_not_go_down_where_it_is_going_is_set_down_nearby_and_gripped_again(board):
    # The user's suggestion, 2026-11-04: pick it up, put it down somewhere, then
    # pick it up again and put it where it goes — a fresh grip, at the lean the
    # last stretch needs.
    b, state = board(rules.Position.start().play(rules.Move("e2", "e4")).play(rules.Move("e7", "e5")))
    b.fail[("place", "e7")] = play.CARRY_TRIES
    rec = play.carry(None, state, "e5", "e7")
    assert rec["ok"] and rec["relay"] == "e6", rec
    assert ("e5", "e6") in b.carried and ("e6", "e7") in b.carried
    assert "e7" in b.occupied() and "e6" not in b.occupied() and "e5" not in b.occupied()
    # Never further out than where it is going, nor on a square the move uses.
    for frm, to in (("e5", "e7"), ("b1", "c3")):
        stage = play.staging_square(state, frm, to)
        assert stage not in (frm, to) and stage not in play.occupied(state)
        assert np.linalg.norm(centre(stage)) <= np.linalg.norm(centre(to)) + 1e-9


def test_a_piece_the_camera_misses_away_from_where_the_arm_works_does_not_stop_a_run(board, monkeypatch):
    # 2026-11-04: the overhead camera missed the e1 king on most looks; nothing
    # touched it, and the start check refused every run.
    b, state = board()
    real_look = b.look

    def misses_h8(env, scene_state):
        real_look(env, scene_state)
        scene_state.objects = [o for o in scene_state.objects if o.square != "h8"]
        return True

    monkeypatch.setattr(play, "look", misses_h8)
    misses_h8(None, state)
    run = play.play_line(None, state, "kings_pawn", *rules.OPENINGS["kings_pawn"])
    assert run["ok"], run
    assert all("h8" in r["check"]["doubts"] for r in b.log if r.get("phase") in ("play", "take back"))


def test_a_piece_set_aside_to_make_room_is_not_set_down_on_or_beside_where_the_picked_one_goes(board):
    # 1. e4 e5 2. Bd3 with the bishop crowded by the king: e3 is the king's best
    # free square, and the king set down there would crowd the bishop off d3.
    from deployments.chess import skills
    pos = rules.Position.start().play(rules.Move("e2", "e4")).play(rules.Move("e7", "e5"))
    b, state = board(pos)
    king, bishop = play.piece_on(state, "e1"), play.piece_on(state, "f1")
    anywhere = [t.name for t in skills._aside_spots(state, king, bishop)]
    assert anywhere[0] == "e3" and "d3" in anywhere
    assert "e2" not in anywhere                               # beside f1, where the bishop is picked
    going = [t.name for t in skills._aside_spots(state, king, bishop, headed_for="d3")]
    assert going and not {"d3", "e3", "c3", "e2", "c2"} & set(going)


def test_a_piece_knocked_onto_the_square_while_carrying_is_seen_before_the_next_try(board):
    """The place refused for something moving there; the next try looks first, finds e4 taken, puts it back."""
    b, state = board()
    knocked = {"done": False}
    real_act = b.act

    def act(env, scene_state, action, object_name=None, target_name=None):
        if action == "place" and target_name == "e4" and not knocked["done"]:
            knocked["done"] = True
            b.at["e4"] = centre("e4")                              # a hand knocks a piece onto e4
            return False, "something moved 1 cm from where it was going — still holding the object"
        return real_act(env, scene_state, action, object_name=object_name, target_name=target_name)

    play._act = act
    env = SimpleNamespace(motion_seen_since=lambda t: [(*centre("e4"), 0.04)], sentry_sources=lambda: {"motion"})
    looks = b.looks
    rec = play.carry(env, state, "e2", "e4")
    assert not rec["ok"] and rec["why"] == "e4 is taken now" and rec.get("put_back") is True
    assert b.looks > looks and "e2" in b.occupied() and b.carried == [("e2", "e2")]      # no relay onto e3
