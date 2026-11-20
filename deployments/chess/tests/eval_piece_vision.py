"""
eval_piece_vision.py — score piece_vision against the labelled frames in piece_frames/.

Asks of every frame the question the servo's first look asks: searching
around GRIPPER_CAM_GRASP_PX with the servo's acquire radius, does find_piece
return the piece?

    hit         found, within HIT_PX of the label
    miss        returned nothing (safe: the pick falls back to kinematics)
    WRONG       returned something more than HIT_PX from the label, or
                anything at all on an empty-board frame. This is the number
                that matters: a wrong answer moves the arm.

Frames whose label lies outside the search radius are scored separately: no
detector could be asked to find them.

Usage (from the repo root):
    python deployments/chess/tests/eval_piece_vision.py            # summary
    python deployments/chess/tests/eval_piece_vision.py --sheet    # + contact sheet of failures
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.resolve().parents[2]))

import cv2            # noqa: E402
import numpy as np    # noqa: E402

from mrcf.backends.so101 import robot_config as rc   # noqa: E402
from deployments.chess import piece_vision as pv     # noqa: E402
from mrcf import privacy

# A label comes from the servo calibration's LINEAR model, which is exact at
# the grasp pixel and drifts with distance (measured: ~40 px off the pawn at
# 45 mm out). The first look only has to pick the right object — later looks
# are near the centre, where the model is good — so the tolerance grows with
# distance from the grasp pixel.
HIT_PX = 25.0           # ~7 mm at hover, at the centre
HIT_PER_PX = 0.3
ACQUIRE_RADIUS_PX = 190.0   # matches SO101Env.visual_servo_xy's first look


def evaluate(sheet: bool) -> int:
    grasp = np.asarray(rc.GRIPPER_CAM_GRASP_PX, float)
    runs = sorted((HERE / "piece_frames").glob("*/labels.json"))
    if not runs:
        print("No labelled frames yet — run capture_piece_frames.py first.")
        return 1
    by_piece = defaultdict(Counter)
    failures = []
    offsets = defaultdict(list)
    for lf in runs:
        meta = json.loads(lf.read_text())
        # Labels are offsets from the grasp pixel in force when they were
        # captured; re-base them if GRIPPER_CAM_GRASP_PX has been re-measured.
        shift = grasp - np.asarray(meta.get("grasp_px", grasp), float)
        for name, label in meta["frames"].items():
            label = None if label is None else np.asarray(label, float) + shift
            path = lf.parent / name
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            found = pv.find_piece(rgb, tuple(grasp), ACQUIRE_RADIUS_PX)
            if label is not None and np.linalg.norm(np.asarray(label) - grasp) > ACQUIRE_RADIUS_PX:
                outcome = "out of reach"
            elif label is None:
                outcome = "hit" if found is None else "WRONG"
            elif found is None:
                outcome = "miss"
            else:
                # Right object = the candidate nearest the label. Its exact point
                # is judged separately (offset column): labels come from a
                # linear model and a tall piece's outline centre sits up it,
                # so a correct detection can sit tens of px off its label.
                cands = pv.candidates(rgb)
                nearest = min(cands, key=lambda c: np.linalg.norm(np.asarray(c[:2]) - label))
                same = np.linalg.norm(np.asarray(nearest[:2]) - np.asarray(found)) < 1.0
                tol = HIT_PX + HIT_PER_PX * float(np.linalg.norm(np.asarray(label) - grasp))
                outcome = "hit" if same and np.linalg.norm(np.asarray(found) - label) <= 3 * tol else "WRONG"
                if outcome == "hit":
                    offsets[meta["piece"]].append(float(np.linalg.norm(np.asarray(found) - label)))
            by_piece[meta["piece"]][outcome] += 1
            if outcome in ("miss", "WRONG"):
                failures.append((path, label, found, outcome))

    print("WRONG = chose a different object than the piece. offset = hit's distance from its label.")
    print(f"{'piece':10s} {'hit':>5s} {'miss':>5s} {'WRONG':>6s} {'out of reach':>13s} {'median offset':>14s}")
    total = Counter()
    for piece, c in sorted(by_piece.items()):
        total.update(c)
        med = f"{np.median(offsets[piece]):.0f} px" if offsets[piece] else "-"
        print(f"{piece:10s} {c['hit']:5d} {c['miss']:5d} {c['WRONG']:6d} {c['out of reach']:13d} {med:>14s}")
    print(f"{'total':10s} {total['hit']:5d} {total['miss']:5d} {total['WRONG']:6d} {total['out of reach']:13d}")

    if sheet and failures:
        tiles = []
        for path, label, found, outcome in failures:
            img = cv2.imread(str(path))
            cv2.circle(img, tuple(int(v) for v in grasp), int(ACQUIRE_RADIUS_PX), (0, 180, 255), 1)
            if label is not None:
                cv2.drawMarker(img, tuple(int(v) for v in label), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
            if found is not None:
                cv2.drawMarker(img, tuple(int(v) for v in found), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
            cv2.putText(img, f"{outcome} {path.parent.name[-8:]}/{path.stem}", (8, 30), 0, 0.8, (255, 0, 255), 2)
            tiles.append(cv2.resize(img, (320, 240)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        out = HERE / "eval_failures.png"
        privacy.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
        print(f"\nFailures (green = label, red = detector): {out}")
    return 0 if total["WRONG"] == 0 else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    sys.exit(evaluate(ap.parse_args().sheet))
