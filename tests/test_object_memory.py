"""Object memory: identities kept across scans, arm actions and restarts. No task knowledge."""

import pytest

from mrcf.object_memory import Detection, ObjectMemory


def det(x, y, place=None, foot=None, colour=None, label=None, source=None, L=None):
    return Detection(position=(x, y), place=place, foot_mm=foot,
                     attributes={"colour": colour} if colour else {},
                     appearance={"L": L} if L is not None else {},
                     label=label, label_source=source)


# Six objects, roughly as the first live chess scans saw them (m, mm).
SCENE = [
    (0.213, 0.096, "b4", 29.0, "white"),
    (0.147, 0.058, "c2", 23.0, "black"),
    (0.257, 0.031, "d5", 24.0, "white"),
    (0.193, -0.019, "e3", 21.0, "white"),
    (0.199, -0.089, "g3", 28.0, "black"),
    (0.233, -0.085, "g4", 25.5, "white"),
]


def scan(jitter=0.0, foot_jitter=0.0, skip=(), extra=(), move=None):
    out = []
    for k, (x, y, sq, foot, colour) in enumerate(SCENE):
        if sq in skip:
            continue
        if move and sq == move[0]:
            x, y, sq = move[1], move[2], move[3]
        sign = 1 if k % 2 else -1
        out.append(det(x + sign * jitter, y - sign * jitter, sq, foot + sign * foot_jitter, colour))
    return out + list(extra)


def ids(tracks):
    return [t.id for t in tracks]


def test_identities_survive_rescans():
    m = ObjectMemory()
    first, report = m.observe(scan())
    assert len(report["new"]) == 6
    for _ in range(3):
        again, report = m.observe(scan(jitter=0.002, foot_jitter=1.5))
        assert ids(again) == ids(first)
        assert not any(report[k] for k in ("new", "moved", "missing", "returned"))


def test_a_hand_move_keeps_identity_and_label():
    m = ObjectMemory()
    first, _ = m.observe(scan())
    m.set_label(first[2].id, "knight", "user")
    tracks, report = m.observe(scan(move=("d5", 0.297, 0.031, "d6")))
    moved = [t for t, _ in report["moved"]]
    assert ids(moved) == [first[2].id]
    assert report["moved"][0][1] == {"from": "d5", "to": "d6", "uncertain": False}
    assert tracks[2].label == "knight" and tracks[2].place == "d6"
    assert "moved by hand from d5 to d6" in m.describe()


def test_a_different_object_on_the_same_spot_is_new():
    m = ObjectMemory()
    first, _ = m.observe(scan())
    swapped = [d if d.place != "e3" else det(d.position[0], d.position[1], "e3", 34.0, "black")
               for d in scan()]
    tracks, report = m.observe(swapped)
    assert tracks[3].id != first[3].id
    assert [t.id for t, _ in report["missing"]] == [first[3].id]
    assert report["replaced"][0][1]["replaced"] == first[3].id


def test_identical_objects_moved_together_are_still_matched():
    m = ObjectMemory()
    a = det(0.10, 0.00, "a1", 21.0, "white")
    b = det(0.10, 0.04, "a2", 21.0, "white")
    first, _ = m.observe([a, b, det(0.2, 0.0, "c1", 30.0, "black")])
    tracks, report = m.observe([det(0.14, 0.00, "b1", 21.5, "white"),
                                det(0.14, 0.04, "b2", 20.5, "white"),
                                det(0.2, 0.0, "c1", 30.0, "black")])
    assert sorted(ids(tracks[:2])) == sorted(ids(first[:2]))
    assert len(report["moved"]) == 2 and not report["new"]
    assert all(detail["uncertain"] for _, detail in report["moved"])


def test_a_guess_between_different_objects_is_not_made():
    m = ObjectMemory()
    first, _ = m.observe([det(0.10, 0.00, "a1", 24.0, "white"), det(0.10, 0.04, "a2", 24.0, "white")])
    m.set_label(first[0].id, "rook", "user")
    m.set_label(first[1].id, "knight", "user")
    # Both gone from their places; one object of their size turns up elsewhere.
    tracks, report = m.observe([det(0.20, 0.00, "c1", 24.0, "white")])
    assert tracks[0].id not in ids(first)
    assert len(report["missing"]) == 2


def test_arm_actions_are_certain():
    m = ObjectMemory()
    first, _ = m.observe(scan())
    knight = first[2]
    m.picked(knight.id)
    tracks, report = m.observe(scan(skip=("d5",)))
    assert not report["missing"] and knight.state == "held"
    m.placed(knight.id, (0.30, 0.03), "d6")
    tracks, report = m.observe(scan(move=("d5", 0.30, 0.03, "d6")))
    assert tracks[2].id == knight.id and not report["moved"]
    assert "the arm moved" in m.describe() and "from d5 to d6" in m.describe()


def test_an_object_the_arm_carried_keeps_its_shape_whatever_its_foot_reads_where_it_was_put():
    m = ObjectMemory()
    (queen,), _ = m.observe([det(0.2515, 0.0317, "d4", 27.0, "black")])
    m.measured(queen.id, 69.0, {20: 23.0, 40: 22.5}, 27.0)
    assert queen.shape_current()
    m.picked(queen.id)
    m.placed(queen.id, (0.296, -0.004), "e5")
    # The scan reads its foot 17 mm from e5 (2026-10-28): still the queen it measured.
    (t,), _ = m.observe([det(0.295, -0.005, "e5", 17.1, "black")])
    assert t.id == queen.id and t.shape_current()
    # Something else of another size standing there next time is not.
    m.observe([det(0.295, -0.005, "e5", 24.0, "black")])
    assert not queen.shape_current()


def test_an_object_the_arm_put_down_is_found_there_even_on_a_foot_misread():
    m = ObjectMemory()
    (queen,), _ = m.observe([det(0.2515, 0.0317, "d4", 27.0, "black")])
    m.picked(queen.id)
    m.placed(queen.id, (0.2148, 0.0273), "d3")
    # The scan read its foot 8 mm on d3 (2026-10-28): it is still the queen.
    (t,), report = m.observe([det(0.215, 0.027, "d3", 8.0, "black")])
    assert t.id == queen.id and not report["new"] and not report["missing"]
    assert not queen.refoot                         # only the first look after the arm put it there


def test_an_object_the_arm_put_down_a_little_off_and_misread_is_still_itself():
    m = ObjectMemory()
    (queen, rook), _ = m.observe([det(0.2515, 0.0317, "d4", 27.0, "black"),
                                  det(0.247, -0.056, "f4", 27.0, "black")])
    m.picked(queen.id)
    m.placed(queen.id, (0.2911, 0.0363), "d5")
    # Landed 18 mm off and read white on d5 (2026-10-28): its name went to the rook.
    (on_d5, on_f4), report = m.observe([det(0.307, 0.045, "d5", 29.0, "white"),
                                        det(0.247, -0.056, "f4", 27.0, "black")])
    assert on_d5.id == queen.id and on_f4.id == rook.id and not report["new"]


def test_a_missing_object_comes_back_as_itself():
    m = ObjectMemory()
    first, _ = m.observe(scan())
    m.observe(scan(skip=("g3",)))
    assert first[4].state == "missing"
    assert "last on g3" in m.describe()
    tracks, report = m.observe(scan(move=("g3", 0.16, -0.12, "h2")))
    assert tracks[4].id == first[4].id and tracks[4].state == "present"
    assert [t.id for t, _ in report["returned"]] == [first[4].id]


def test_label_trust_order_and_votes():
    m = ObjectMemory()
    (t,), _ = m.observe([det(0.1, 0.0, "a1", 22, label="pawn", source="vlm")])
    for guess in ("king", "pawn", "king", "king"):
        m.observe([det(0.1, 0.0, "a1", 22, label=guess, source="vlm")])
    assert t.label == "king"                  # 3 king, 2 pawn
    m.set_label(t.id, "queen", "user")
    m.observe([det(0.1, 0.0, "a1", 22, label="pawn", source="vlm")])
    assert (t.label, t.label_source) == ("queen", "user")
    assert not m.set_label(t.id, "rook", "shape")


def test_one_odd_colour_reading_does_not_flip_an_object():
    m = ObjectMemory()
    (t,), _ = m.observe([det(0.1, 0.0, "a1", 22, "black")])
    m.observe([det(0.1, 0.0, "a1", 22, "black")])
    m.observe([det(0.1, 0.0, "a1", 22, "white")])
    assert t.attributes["colour"] == "black"
    m.set_label(t.id, None, "user", attributes={"colour": "white"})
    for _ in range(3):
        m.observe([det(0.1, 0.0, "a1", 22, "black")])
    assert t.attributes["colour"] == "white"   # a person said so


def test_load_false_starts_empty_and_writes_over_the_file(tmp_path):
    path = tmp_path / "memory.json"
    first, _ = ObjectMemory(path).observe(scan())
    assert first

    fresh = ObjectMemory(path, load=False)
    assert fresh.tracks == {}
    tracks, _ = fresh.observe(scan(skip=("c2",)))
    assert len(tracks) == len(first) - 1
    assert len(ObjectMemory(path).tracks) == len(first) - 1     # the old run's file is gone


def test_saved_and_restored(tmp_path):
    path = tmp_path / "memory.json"
    m = ObjectMemory(path)
    first, _ = m.observe(scan())
    m.set_label(first[0].id, "king", "user")
    m.measured(first[0].id, height_mm=70.0, widths_mm={4: 29.0, 8: 28.0})
    m.picked(first[1].id)

    again = ObjectMemory(path)
    king = again.get(first[0].id)
    assert king.label == "king" and king.height_mm == 70.0 and king.widths_mm == {4: 29.0, 8: 28.0}
    assert again.get(first[1].id).state == "held"
    tracks, _ = again.observe(scan(skip=("c2",)))
    assert ids(tracks) == [i for i in ids(first) if i != first[1].id]

    again.clear()
    (new,), _ = again.observe([det(0.1, 0.0, "a1", 22)])
    assert new.id not in ids(first)           # ids are never reused


def test_a_capture_is_a_move_not_a_change_of_look():
    m = ObjectMemory()
    (knight, pawn), _ = m.observe([det(0.20, 0.00, "e5", 24.0, "black"),
                                   det(0.16, 0.04, "d4", 21.0, "white")])
    # The pawn takes on e5: e5 is still occupied, d4 is empty.
    (on_e5,), report = m.observe([det(0.201, 0.001, "e5", 21.0, "white")])
    assert on_e5.id == pawn.id
    assert report["moved"][0][1]["from"] == "d4"
    assert [t.id for t, _ in report["missing"]] == [knight.id]


def test_a_real_colour_change_wins_within_two_scans():
    m = ObjectMemory()
    (t,), _ = m.observe([det(0.1, 0.0, "a1", 22, "black")])
    m.observe([det(0.1, 0.0, "a1", 22, "black")])
    m.observe([det(0.1, 0.0, "a1", 22, "white")])
    m.observe([det(0.1, 0.0, "a1", 22, "white")])
    assert t.attributes["colour"] == "white"


def test_a_shape_is_not_trusted_once_the_foot_changes():
    m = ObjectMemory()
    (t,), _ = m.observe([det(0.1, 0.0, "a1", 21.0)])
    m.measured(t.id, height_mm=44.0)
    m.observe([det(0.1, 0.0, "a1", 22.0)])
    assert t.shape_current()
    # Something wider stands there now, but not different enough to be called
    # a new object: its height must be measured again before motion trusts it.
    m.observe([det(0.1, 0.0, "a1", 26.0)])
    assert t.id == m.observe([det(0.1, 0.0, "a1", 26.0)])[0][0].id
    assert not t.shape_current()
    m.measured(t.id, height_mm=70.0)
    assert t.shape_current() and t.height_mm == 70.0


def test_a_shape_is_not_trusted_after_the_object_was_out_of_sight(tmp_path):
    # 2026-10-29: a 42 mm pawn stood on e4, where a 52 mm piece had stood the
    # day before. Its foot was within 3 mm of that piece's, so the memory took
    # it for that piece back again, and the pick went by the 52 mm.
    m = ObjectMemory(tmp_path / "memory.json")
    (t,), _ = m.observe([det(0.1, 0.0, "e4", 23.0, "white")])
    m.measured(t.id, height_mm=52.0)
    m.observe([])
    assert t.state == "missing"
    (back,), report = m.observe([det(0.1, 0.0, "e4", 20.5, "white")])
    assert back is t and report["returned"]
    assert not t.shape_current()
    assert not ObjectMemory(tmp_path / "memory.json").get(t.id).shape_current()   # across a restart
    m.measured(t.id, height_mm=42.0)
    assert t.shape_current() and t.height_mm == 42.0                               # not 52 among them
    # Moved by hand, not by the arm: the same doubt.
    (moved,), report = m.observe([det(0.14, 0.0, "e5", 20.5, "white")])
    assert moved is t and report["moved"] and not t.shape_current()


def test_a_shape_is_not_trusted_once_its_colour_or_label_changes():
    m = ObjectMemory()
    (t,), _ = m.observe([det(0.1, 0.0, "a1", 22, "white", label="pawn", source="vlm")])
    m.measured(t.id, height_mm=42.0)
    m.observe([det(0.1, 0.0, "a1", 22, "white", label="pawn", source="vlm")])
    assert t.shape_current()
    m.observe([det(0.1, 0.0, "a1", 22, "black")])
    m.observe([det(0.1, 0.0, "a1", 22, "black")])
    assert t.attributes["colour"] == "black" and not t.shape_current()
    m.measured(t.id, height_mm=42.0)
    m.observe([det(0.1, 0.0, "a1", 22, "black", label="rook", source="shape")])
    assert t.label == "rook" and not t.shape_current()


def test_one_wild_look_does_not_set_a_shape(tmp_path):
    m = ObjectMemory(tmp_path / "memory.json")
    (t,), _ = m.observe([det(0.1, 0.0, "a1", 21.0)])
    neck = {16: 15.5, 20: 15.5, 24: 15.4}
    m.measured(t.id, height_mm=59.6, widths_mm={16: 30.9, 20: 31.2})
    assert t.shape() == {"height_mm": 59.6, "widths_mm": {16: 30.9, 20: 31.2}, "looks": 1}
    # One against one: the latest, still a single look — not their 50 mm average.
    m.measured(t.id, height_mm=41.0, widths_mm=neck)
    assert t.shape()["height_mm"] == 41.0 and t.shape()["looks"] == 1
    m.measured(t.id, height_mm=41.6, widths_mm={16: 15.0, 20: 15.2, 24: 15.0})
    shape = t.shape()
    assert shape["height_mm"] == pytest.approx(41.3) and shape["looks"] == 2
    assert shape["widths_mm"][16] < 20 and t.widths_mm == shape["widths_mm"]
    # Kept across a restart, profiles with their number keys.
    again = ObjectMemory(tmp_path / "memory.json").get(t.id)
    assert again.shape() == shape


def guess(x, y, place, colour):
    """What a vision model alone reported, the outlines having looked and found nothing there."""
    return Detection(position=(x, y), place=place, attributes={"colour": colour}, confirmed=False)


def test_a_vision_models_ghost_never_takes_a_measured_objects_identity():
    # 2026-10-21 12:39-12:43, from the scan log: a cream pawn on e2 (the user
    # said white; the overhead read it black) and a ghost on h5. The pawn was
    # picked and put back while the ghost wandered to a4. Then "the pawn
    # stayed, its colour misread" cost 1.5 and "the black ghost moved onto e2"
    # 1.0 — colour was all a ghost carries — so the pawn went missing, and its
    # stale track then refused its own next pick.
    m = ObjectMemory()
    (pawn, ghost), _ = m.observe([det(0.1695, -0.0088, "e2", 20.1, "black"),
                                  guess(0.3041, -0.1153, "h5", "black")])
    assert pawn.confirmed and not ghost.confirmed
    m.set_label(pawn.id, "pawn", "user", attributes={"colour": "white"})
    m.picked(pawn.id)
    m.observe([guess(0.2314, 0.1584, "a4", "black")])
    m.placed(pawn.id, (0.1706, -0.0101), "e2")
    (on_e2, on_h5), report = m.observe([det(0.1613, -0.0135, "e2", 20.6, "black"),
                                        guess(0.3038, -0.1154, "h5", "black")])
    assert on_e2.id == pawn.id and pawn.state == "present"
    assert on_h5.id == ghost.id and not report["missing"]


def test_a_measured_object_is_not_the_ghost_that_stood_somewhere_else():
    m = ObjectMemory()
    (ghost,), _ = m.observe([guess(0.24, 0.12, "a4", "black")])
    (real,), report = m.observe([det(0.161, -0.014, "e2", 21.0, "black")])
    assert real.id != ghost.id and real.confirmed and report["new"]


def test_a_ghost_still_moves_as_a_ghost_and_a_missed_outline_keeps_a_piece_confirmed():
    m = ObjectMemory()
    (ghost, pawn), _ = m.observe([guess(0.28, -0.12, "h5", "black"), det(0.20, 0.0, "e4", 21.0, "white")])
    # The outline misses the pawn this time; the vision model still sees it there.
    (g, p), report = m.observe([guess(0.24, 0.12, "a4", "black"), guess(0.20, 0.0, "e4", "white")])
    assert (g.id, p.id) == (ghost.id, pawn.id) and [t.id for t, _ in report["moved"]] == [ghost.id]
    assert pawn.confirmed and not ghost.confirmed


def test_the_arm_holding_or_measuring_an_object_confirms_it():
    m = ObjectMemory()
    (a, b), _ = m.observe([guess(0.20, 0.0, "e4", "white"), guess(0.20, 0.04, "d4", "white")])
    m.picked(a.id)
    m.measured(b.id, height_mm=42.0)
    assert a.confirmed and b.confirmed


def test_tracks_saved_before_confirmation_existed_load_as_confirmed(tmp_path):
    path = tmp_path / "memory.json"
    m = ObjectMemory(path)
    (ghost,), _ = m.observe([guess(0.24, 0.12, "a4", "black")])
    assert ObjectMemory(path).get(ghost.id).confirmed is False      # kept across a restart
    text = path.read_text().replace('"confirmed": false', '"_dropped": 0')
    path.write_text(text)
    assert ObjectMemory(path).get(ghost.id).confirmed is True


# ── Stacks: a block the arm put on a block ─────────────────────────────────────
# A scan sees a stack as one object where the one on top is, as tall as all of
# it; the ones under it are out of sight, not gone.

def block(x, y, height, colour="red"):
    return Detection(position=(x, y), foot_mm=40.0, height_mm=height, attributes={"colour": colour})


def _three_blocks():
    m = ObjectMemory()
    (red, blue, green), _ = m.observe([block(0.30, 0.0, 40, "red"), block(0.40, 0.1, 40, "blue"),
                                       block(0.35, -0.1, 40, "green")])
    m.picked(red.id)
    m.placed(red.id, (0.40, 0.1), None, on=blue.id, raised_mm=40.0)
    return m, red, blue, green


def test_a_stack_is_seen_as_the_one_on_top_and_the_one_under_it_stays():
    m, red, blue, green = _three_blocks()
    tracks, report = m.observe([block(0.401, 0.1, 80, "red"), block(0.35, -0.1, 40, "green")])
    assert [t.id for t in tracks] == [red.id, green.id]
    assert not report["new"] and not report["missing"]
    assert blue.state == "present" and red.on == blue.id
    assert red.height_mm == pytest.approx(40.0)              # its own height, not the stack's
    tracks, report = m.observe([block(0.40, 0.1, 81, "red"), block(0.35, -0.1, 40, "green")])
    assert tracks[0].id == red.id and not report["new"] and not report["missing"]


def test_three_high_the_two_under_it_stay():
    m, red, blue, green = _three_blocks()
    m.picked(green.id)
    m.placed(green.id, (0.40, 0.1), None, on=red.id, raised_mm=80.0)
    tracks, report = m.observe([block(0.40, 0.1, 120, "green")])
    assert [t.id for t in tracks] == [green.id] and not report["new"] and not report["missing"]
    assert red.state == blue.state == "present" and green.height_mm == pytest.approx(40.0)


def test_the_top_taken_off_by_hand_the_one_under_is_seen_again():
    m, red, blue, green = _three_blocks()
    tracks, report = m.observe([block(0.40, 0.1, 40, "blue"), block(0.35, -0.1, 40, "green")])
    assert [t.id for t in tracks] == [blue.id, green.id]
    assert red.state == "missing" and red.on is None and not report["new"]


def test_the_top_moved_by_hand_is_on_the_table_again():
    m, red, blue, green = _three_blocks()
    tracks, report = m.observe([block(0.40, 0.1, 40, "blue"), block(0.35, -0.1, 40, "green"),
                                block(0.20, 0.2, 40, "red")])
    assert [t.id for t in tracks] == [blue.id, green.id, red.id]
    assert red.on is None and red.raised_mm == 0.0 and red.position == [0.20, 0.2]


def test_picked_off_a_stack_it_stands_on_nothing():
    m, red, blue, green = _three_blocks()
    m.picked(red.id)
    assert red.on is None and red.raised_mm == 0.0


def test_a_stack_is_saved_and_restored(tmp_path):
    path = tmp_path / "memory.json"
    m = ObjectMemory(path)
    (red, blue), _ = m.observe([block(0.30, 0.0, 40, "red"), block(0.40, 0.1, 40, "blue")])
    m.picked(red.id)
    m.placed(red.id, (0.40, 0.1), None, on=blue.id, raised_mm=40.0)
    again = ObjectMemory(path)
    assert again.get(red.id).on == blue.id and again.get(red.id).raised_mm == 40.0


def test_a_height_learned_since_moves_what_stands_on_it():
    m, red, blue, green = _three_blocks()             # red on blue, put there with blue taken for 40 mm
    m.measured(blue.id, height_mm=30.0)               # blue turns out 30 mm tall
    m.restack()
    assert red.raised_mm == pytest.approx(30.0)
    m.picked(green.id)
    m.placed(green.id, (0.40, 0.1), None, on=red.id, raised_mm=80.0)
    m.measured(red.id, height_mm=30.0)
    m.restack()
    assert green.raised_mm == pytest.approx(60.0)     # down the whole stack


# ── Telling apart what looks alike ─────────────────────────────────────────────

def lab_block(x, y, foot, a, b):
    return Detection(position=(x, y), foot_mm=foot, appearance={"a": a, "b": b})


def _green_taken_for_z():
    """
    2026-11-12, live: green block 7 put on red U by the arm, taken off by hand
    before a restart, and the scan gave it to "blue block Z" — a greenish
    block gone half an hour, nearer its look than 7's old (narrower) foot.
    """
    m = ObjectMemory()
    g, z, r = m.observe([lab_block(0.35, 0.42, 22.0, -12, -18), lab_block(0.60, 0.27, 37.0, -15, -12),
                         lab_block(0.58, 0.37, 34.0, 9, -14)])[0]
    m.set_label(g.id, "block 7", "vlm", {"colour": "green"})
    m.set_label(z.id, "block z", "vlm", {"colour": "blue"})
    m.set_label(r.id, "block u", "vlm", {"colour": "red"})
    m.observe([lab_block(0.35, 0.42, 22.0, -12, -18), lab_block(0.58, 0.37, 34.0, 9, -14)])     # Z taken away
    m.picked(g.id)
    m.placed(g.id, (0.58, 0.37), on=r.id, raised_mm=30.0)
    tracks, report = m.observe([lab_block(0.58, 0.37, 34.0, 9, -14), lab_block(0.40, 0.436, 34.0, -13, -18)])
    return m, g, z, r, tracks, report


def test_a_sighting_taken_by_look_alone_is_a_guess():
    m, g, z, r, tracks, report = _green_taken_for_z()
    assert tracks[1].id == z.id and [t.id for t, _ in report["returned"]] == [z.id]     # the mistake
    assert m.guessed() == {z.id}                       # red stayed where it was: not a guess
    m.checked(z.id)
    assert m.guessed() == set()


def test_a_guess_is_given_to_the_object_it_really_is():
    m, g, z, r, tracks, _ = _green_taken_for_z()
    z_looks = len(m.get(z.id).appearances)
    really = m.reidentify(z.id, "block 7", "the vision model calls it block 7")
    assert really is m.get(g.id) and really.state == "present"
    assert really.position == [0.40, 0.436] and really.on is None
    assert really.history[-1]["event"] == "moved"      # on the table before the scan: moved by hand
    assert really.history[-2]["event"] == "placed"
    back = m.get(z.id)
    assert back.state == "missing" and back.position == [0.60, 0.27]
    assert len(back.appearances) == z_looks - 1        # the green's look taken off it again
    assert m.guessed() == set()
    assert m.reidentify(z.id, "block 7") is None       # once only


def test_a_new_sighting_of_a_remembered_object_is_that_object():
    m = ObjectMemory()
    (p,), _ = m.observe([lab_block(0.25, 0.42, 35.0, -6, -33)])
    m.set_label(p.id, "block p", "vlm", {"colour": "blue"})
    m.observe([])
    (seen,), report = m.observe([lab_block(0.47, 0.25, 22.0, -6, -33)])        # foot too different to match
    assert len(report["new"]) == 1 and seen.id in m.guessed()
    really = m.reidentify(seen.id, "block p")
    assert really is m.get(p.id) and really.history[-1]["event"] == "returned"
    assert seen.id not in m.tracks


def test_only_a_guess_is_given_away_and_only_to_an_object_not_seen():
    m, g, z, r, tracks, _ = _green_taken_for_z()
    assert m.reidentify(r.id, "block 7") is None       # red stayed put: not a guess
    assert m.reidentify(z.id, "block u") is None       # red is seen: it cannot be this one too
    assert m.reidentify(z.id, "block q") is None       # nothing remembered is called that


def test_a_misread_letter_does_not_take_another_colour_of_block():
    """Live, 2026-11-12: the vision model read red U, on its side, as D — and blue D was off the table."""
    m = ObjectMemory()
    u, d = m.observe([lab_block(0.58, 0.37, 34.0, 9, -14), lab_block(0.36, 0.22, 34.0, -7, -33)])[0]
    m.set_label(u.id, "block u", "vlm", {"colour": "red"})
    m.set_label(d.id, "block d", "vlm", {"colour": "blue"})
    tracks, report = m.observe([lab_block(0.45, 0.30, 34.0, 9, -14)])          # U moved by hand, D gone
    assert tracks[0].id == u.id and u.id in m.guessed()
    assert m.reidentify(u.id, "block d") is None
    assert m.get(u.id).state == "present" and m.get(d.id).state == "missing"


def test_an_object_the_scan_could_not_see_under_the_arm_stays_where_it_was():
    m = ObjectMemory()
    tracks, _ = m.observe(scan())
    hidden_sq = tracks[2]
    under_arm = lambda xy: abs(xy[0] - hidden_sq.position[0]) < 0.01 and abs(xy[1] - hidden_sq.position[1]) < 0.01
    again, report = m.observe(scan(skip=("d5",)), hidden=under_arm)
    assert hidden_sq.state == "present" and [t for t, _ in report["hidden"]] == [hidden_sq]
    assert report["missing"] == []
    gone, report = m.observe(scan(skip=("d5",)))                      # seen past nothing: missing
    assert hidden_sq.state == "missing"
