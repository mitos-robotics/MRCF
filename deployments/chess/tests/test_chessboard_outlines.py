"""The chess scan's own perception: which squares' outlines are pieces, colour, light, what the model calls them."""

import numpy as np

try:
    import cv2  # noqa: F401 — chessboard needs it
except ImportError:
    import pytest
    pytest.skip("needs OpenCV", allow_module_level=True)

from deployments.chess import chessboard


# ── merge_sources: the outlines say what is there, the model what it is ─────────


def test_only_an_outline_puts_a_piece_on_the_board():
    # 2026-10-21: one pawn on e2, outlined; a vision model shown the board also
    # put pieces on e1 and h5. Labels are only ever asked of outlines now, but a
    # label without one is still no piece.
    outlined = {"e2": {"colour": "white"}, "_view": {}}
    merged = chessboard.merge_sources({"e2": ("white", "pawn"), "e1": ("white", "pawn"), "h5": ("black", None)},
                                      outlined)
    assert set(merged) == {"e2"} and merged["e2"]["type"] == "pawn" and merged["e2"]["confirmed"]
    assert chessboard.merge_sources({"e1": ("white", "pawn")}, {}) == {}


def test_an_outline_the_model_says_is_not_a_piece_is_kept_to_keep_clear_of():
    merged = chessboard.merge_sources({"d1": (None, chessboard.NOT_A_PIECE)}, {"d1": {"colour": None}})
    assert merged["d1"]["vetoed"] and merged["d1"]["type"] is None
    assert merged["d1"]["note"] == "the vision model says it is not a piece"


def test_a_piece_the_model_could_not_name_says_so():
    merged = chessboard.merge_sources({"e2": ("white", None)}, {"e2": {"colour": "white"}, "h5": {"colour": "black"}})
    assert merged["e2"]["note"] == merged["h5"]["note"] == "the vision model could not tell what it is"
    assert chessboard.merge_sources({}, {"h5": {"colour": "black"}}, classified=False)["h5"]["note"] == ""


def _slanted_view():
    """A camera 0.4 m up, 0.2 m back, looking down at a slant onto a board at the origin."""
    from mrcf.perception.plane_geometry import PlaneView
    a = np.radians(35.0)
    z = np.array([0.0, np.sin(a), -np.cos(a)])
    x = np.array([1.0, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]
    T[:3, 3] = [0.0, -0.2, 0.4]
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    return PlaneView.from_camera_model({"K": K, "D": np.zeros(5), "T_world_cam": T, "plane_z_m": 0.0})


def _pieces(*squares):
    """measure_pieces-like records, one piece a square 40 mm apart along x, plus the view."""
    view = _slanted_view()
    out = {sq: {"measurement": type("M", (), {"base_mm": np.array([40.0 * i, 0.0]), "base_radius_mm": 11.0})(),
                "colour": None} for i, sq in enumerate(squares)}
    out["_view"] = {"view": view}
    return out, view


def test_every_piece_asked_about_is_shown_on_one_sheet_in_one_call(monkeypatch):
    import base64
    import io

    from PIL import Image

    from mrcf import vlm

    asked = []

    def label_numbered(image, count, answers, prompt, system=None, model=None):
        asked.append((image, count, answers))
        return ["white pawn", "black king"]

    monkeypatch.setattr(vlm, "label_numbered", label_numbered)
    measured, view = _pieces("a2", "e8")
    got = chessboard.label_pieces(np.zeros((720, 1280, 3), np.uint8), measured, view)
    assert got == {"a2": ("white", "pawn"), "e8": ("black", "king")}
    (image, count, answers), = asked
    assert count == 2 and answers == chessboard.LABEL_KINDS and "not a piece" in answers
    sheet = Image.open(io.BytesIO(base64.b64decode(image)))
    assert sheet.size[0] == 2 * 112                                   # two tiles, side by side


def test_not_a_piece_stands_only_where_a_second_look_says_so_too(monkeypatch):
    from mrcf import vlm

    replies = iter([["not a piece", "not a piece", "white rook"], ["not a piece", "white bishop"]])
    counts = []
    monkeypatch.setattr(vlm, "label_numbered", lambda image, count, *a, **k: counts.append(count) or next(replies))
    measured, view = _pieces("c1", "d4", "h1")
    frame = np.zeros((720, 1280, 3), np.uint8)
    again = []
    got = chessboard.label_pieces(frame, measured, view, again=lambda: again.append(1) or frame)
    assert got == {"c1": (None, chessboard.NOT_A_PIECE), "d4": ("white", None), "h1": ("white", "rook")}
    assert counts == [3, 2] and again == [1]                          # only the two it doubted, looked at again


def test_the_labelling_says_what_it_called_which_squares_in_a_few_lines():
    text = chessboard.label_summary({"a1": ("white", "rook"), "h1": ("white", "rook"), "a2": ("white", "pawn"),
                                     "e8": ("black", "king"), "x9": (None, chessboard.NOT_A_PIECE),
                                     "d8": (None, None)}, known=26, seconds=15.6)
    lines = text.splitlines()
    assert "6 pieces in one look (16 s); 26 known from before" in lines[0]
    assert lines[1].split() == ["white", "pawn", "a2", "·", "rook", "a1", "h1"]
    assert lines[2].split() == ["black", "king", "e8"]
    assert lines[3].endswith("x9") and lines[4].endswith("could not tell: d8")
    assert "nothing to ask" in chessboard.label_summary({}, known=32, seconds=0.0)


def test_only_pieces_the_memory_cannot_name_are_asked_about(monkeypatch, tmp_path):
    from mrcf.object_memory import Detection, ObjectMemory

    memory = ObjectMemory(tmp_path / "memory.json")
    measured, _ = _pieces("a1", "b1", "c1")
    asked = []

    def label_pieces(frame, some, view, again=None):
        asked.append(sorted(some))
        return {sq: ("white", "rook") for sq in some}

    monkeypatch.setattr(chessboard, "label_pieces", label_pieces)

    def scan(at):
        dets = [Detection(position=xy, place=sq, foot_mm=22.0, attributes={"colour": "white"}) for sq, xy in at]
        tracks, changes = memory.observe(dets)
        labels, ask = chessboard.label_unknown(None, None, measured, [sq for sq, _ in at], tracks, changes, memory)
        return tracks, ask

    home = [("a1", (0.10, 0.00)), ("b1", (0.10, 0.04)), ("c1", (0.10, 0.08))]
    tracks, ask = scan(home)
    assert ask == {"a1", "b1", "c1"} and all(t.label == "rook" and t.label_source == "vlm" for t in tracks)
    _, ask = scan(home)                                               # a restart, the board as it was
    assert ask == set() and len(asked) == 1
    tracks, ask = scan([("a1", (0.10, 0.00)), ("b1", (0.10, 0.04)), ("c1", (0.22, 0.08))])
    assert ask == {"c1"}                                              # moved by hand: asked again


def test_the_second_look_is_at_a_frame_that_is_not_the_first(monkeypatch):
    # A camera read straight after another can return the same image; two
    # looks at one image agree whatever the model makes of it.
    first = np.zeros((4, 4, 3), np.uint8)
    later = iter([first.copy(), first.copy(), first + 1])
    env = type("E", (), {"render_image": lambda self, *a, **k: next(later)})()
    monkeypatch.setattr(chessboard.time, "sleep", lambda s: None)
    assert (chessboard._next_frame(env, first) == 1).all()


# ── A piece near a square's corner is not part of the board ───────────────────

class _FlatView:
    """A board seen square on: 1.5 px per mm, the lattice's first corner at pixel (100, 100)."""
    pitch_mm = 40.0

    def to_plane(self, px):
        return (np.asarray(px, float) - 100.0) / 1.5

    def lifted(self, xy, height_mm):
        return np.array([xy[0], xy[1], height_mm], float)

    def project(self, pts):
        return np.asarray(pts, float)[:, :2] * 1.5 + 100.0


def _board_lab():
    lab = np.zeros((500, 500, 3), np.uint8)
    for i in range(-2, 7):
        for j in range(-2, 7):
            x0, y0 = int(100 + i * 60), int(100 + j * 60)
            colour = (200, 135, 150) if (i + j) % 2 == 0 else (90, 140, 145)     # maple, walnut
            lab[max(y0, 0):max(y0 + 60, 0), max(x0, 0):max(x0 + 60, 0)] = colour
    return lab


def test_a_pawn_cut_by_a_corners_grid_lines_is_kept_and_a_patch_of_square_is_not():
    import cv2
    from deployments.chess import piece_geometry as geo
    view, lab = _FlatView(), _board_lab()
    cell = np.zeros((500, 500), bool)
    cell[100:160, 100:160] = True                       # a maple square
    # 2026-10-22: a cream pawn near e4's corner, outlined only where it was on
    # the maple, cut by two grid lines — rejected, and missed four scans running.
    disc = np.zeros((500, 500), np.uint8)
    cv2.circle(disc, (110, 112), 20, 1, -1)
    pawn = disc.astype(bool) & cell
    lab[pawn] = (225, 128, 140)                         # paler than the maple
    assert 0.4 < geo.grid_edge_fraction(pawn, view) <= geo.GRID_EDGE_PIECE_MAX
    assert not geo.on_the_grid(pawn, view, lab)
    # A quarter of a bare square, as an outline might cut one beside a piece: the square's own colour.
    quarter = np.zeros((500, 500), bool)
    quarter[220:250, 220:250] = True
    assert geo.on_the_grid(quarter, view, lab)
    whole = np.zeros((500, 500), bool)
    whole[220:280, 220:280] = True
    assert geo.on_the_grid(whole, view, lab)
    # Washed out to the maple's own colour, it is kept when its middle has
    # changed from the empty-board photo (2026-10-22 12:28) — a patch well
    # along the grid is board even so (the squares beside a queen, 2026-10-27).
    lab[pawn] = lab[cell & ~pawn][0]
    assert geo.on_the_grid(pawn, view, lab)
    assert not geo.on_the_grid(pawn, view, lab, differs=True)
    assert geo.on_the_grid(whole, view, lab, differs=True)


def test_colour_is_read_against_the_boards_own_squares():
    # Replayed from saved overhead frames and the live scan of 2026-10-27:
    # (piece Lab, dark squares' Lab, light squares' Lab).
    cream_pawn = [((177, 143, 131), (118, 150, 118), (225, 138, 127)),
                  ((179, 142, 126), (120, 151, 119), (229, 138, 125)),
                  ((226, 139, 128), (124, 151, 119), (234, 133, 125)),
                  ((160, 142, 126), (125, 151, 118), (238, 133, 124))]
    brown = [((117, 156, 139), (163, 147, 123), (243, 126, 123)),
             ((127, 144, 151), (163, 141, 128), (242, 126, 124)),
             ((168, 150, 120), (125, 151, 118), (238, 133, 124)),     # the queen, lit top lighter than
             ((163, 147, 120), (125, 151, 118), (238, 133, 124))]     # the dark squares (2026-10-27)
    for piece, dark, light in cream_pawn:
        assert chessboard.colour_of(chessboard.piece_darkness(piece, dark, light)) == "white", piece
    for piece, dark, light in brown:
        assert chessboard.colour_of(chessboard.piece_darkness(piece, dark, light)) == "black", piece


def test_on_a_board_without_colour_lightness_decides():
    grey_dark, grey_light = (60, 128, 128), (200, 128, 128)
    assert chessboard.colour_of(chessboard.piece_darkness((40, 128, 128), grey_dark, grey_light)) == "black"
    assert chessboard.colour_of(chessboard.piece_darkness((220, 128, 128), grey_dark, grey_light)) == "white"
    assert chessboard.colour_of(chessboard.piece_darkness((125, 128, 128), grey_dark, grey_light)) is None
    assert chessboard.piece_darkness((125, 128, 128), (120, 128, 128), (130, 128, 128)) is None


def test_a_colour_the_camera_cannot_call_comes_from_the_model():
    cam = {"e4": {"colour": None}, "_view": {}}
    merged = chessboard.merge_sources({"e4": ("white", "pawn")}, cam)
    assert merged["e4"]["colour"] == "white"
    # ... and where the piece's own pixels do tell, they win.
    assert chessboard.merge_sources({"e4": ("white", "pawn")}, {"e4": {"colour": "black"}})["e4"]["colour"] == "black"


def _lit_board(washed=(), light_l=225, dark_l=120):
    """A frame of 8 x 8 squares, 40 px each, and its square map; `washed` light squares at the maximum."""
    import cv2
    files = "abcdefgh"
    frame = np.zeros((320, 320, 3), np.uint8)
    squares = {}
    grey = lambda l: cv2.cvtColor(np.full((1, 1, 3), (l, 128, 128), np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
    for r in range(8):
        for c in range(8):
            name = f"{files[c]}{r + 1}"
            light = (r + c) % 2 == 1
            frame[r * 40:(r + 1) * 40, c * 40:(c + 1) * 40] = 255 if name in washed else grey(light_l if light else dark_l)
            squares[name] = {"pixel": (c * 40 + 20, r * 40 + 20), "rc": (r + 1, c + 1)}
    return frame, squares


def test_washed_out_squares_are_named_and_pieces_among_them_warned_of():
    frame, squares = _lit_board(washed=("b7", "a8", "c8", "d7"))
    report = chessboard.lighting_report(frame, squares, occupied={"b8"})
    assert report["washed_out"] == ["a8", "b7", "c8", "d7"]
    warning = chessboard.lighting_warning(report, pieces_on=["b8", "d4"])
    assert warning.startswith("Too bright") and "ranks 7-8" in warning and "b8" in warning and "d4" not in warning


def test_good_light_says_nothing_and_a_dark_board_is_warned_of():
    frame, squares = _lit_board()
    assert chessboard.lighting_warning(chessboard.lighting_report(frame, squares)) is None
    frame, squares = _lit_board(light_l=110, dark_l=30)
    assert chessboard.lighting_warning(chessboard.lighting_report(frame, squares)).startswith("Too dark")
