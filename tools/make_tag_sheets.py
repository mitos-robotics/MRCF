"""
make_tag_sheets.py — print the AprilTags and jaw marks for closed-loop picking, to scale.

The plan this serves: store only what is rigid, measure everything that passes
through a servo. So the cameras measure, every frame, where the board is (tags
on its border) and where the gripper is (tags on the gripper body), and the
wrist camera measures where the jaw tips are (a mark on each jaw), instead of
trusting joint angles, a cached board lattice or a stored target pixel.

Writes two vector PDFs (exact size at any printer resolution):

  tag_sheets/board_tags_<paper>.pdf    6 board tags (+ a spare set), where they go,
                                       and where the camera goes
  tag_sheets/gripper_tags_<paper>.pdf  the gripper tag in three sizes (+ spares)
                                       and the jaw marks

and then reads every tag back out of a rasterised copy of each PDF with the
same detector the runtime will use, checking its id and its printed size.

Why these sizes and places (worked out in simulation against this rig's camera
model and the arm's own IK at every square, hover and grasp — see the notes in
GRIPPER_TAGS and BOARD_TAGS below):

  Board tags sit ON the wooden border (about 36 mm wide), not beside the board:
  the board's far corners overhang the round table, and on the board they move
  with it, which is the point. 26 mm tags with their white margin are 32.5 mm,
  so they fit the border. From the camera at least 3 of the 4 corner tags are
  visible at every arm pose, pinning the board to ~1 mm; the two mid-side tags
  are margin for glare and for the arm hiding a corner.

  The camera: above the arm base (where it was until 2026-10-20) it hangs
  almost on the shoulder-pan axis and looks down the forearm at the gripper —
  93% of the gripper body is behind the forearm and wrist, and a tag anywhere
  on the body was never seen (0 of 114 pick poses). On the FAR side of the
  board, ~7 cm past its far edge, ~47 cm above it, looking back at the arm
  ~60 deg down, it sees the gripper's outer end instead, and the board at
  ~1.4 px/mm instead of ~1.2. (Mounted as high as before, 66 cm, it still works
  but the gripper tag's error roughly doubles.)

  The gripper tag goes on something rigid to the gripper BODY, with nothing
  jointed between it and the jaws; the tag tour measures where. On the SO-101
  (gripper_link:
  the gripper servo housing and the fixed jaw — nothing jointed between them),
  never on the wrist (the wrist_roll servo sits between it and the jaws) or the
  moving jaw. From the far side the body's own faces are edge-on or hidden, but
  the wrist camera's mount is screwed to the body's moving-jaw end and its flat
  back faces outward and up — at the camera. A tag there is seen at every pick
  pose. Grasp point, with the pose flip resolved against the arm's own
  orientation and 5 frames averaged: 32 mm 2.5 mm median / 5.9 mm 90th
  percentile, 28 mm 2.8 / 6.1, 24 mm 3.9 / 9.2 (camera 0.54 m from the board
  centre; at 0.71 m a 28 mm tag gets 5.5 / 10.5). The jaw marks double as a
  check that the mount has not been knocked: the fixed jaw must stay put in the
  wrist image.

  Jaw marks: a white disc with a black centre dot, on the face of each jaw the
  wrist camera sees, near the tip. ~8 mm reads ~29 px across there.

  Jaw marks: a white disc with a black centre dot, on the face of each jaw the
  wrist camera sees, near the tip. ~8 mm reads ~29 px across there.

Usage:
    python tools/make_tag_sheets.py
    python tools/make_tag_sheets.py --paper letter
    python tools/make_tag_sheets.py --no-check     # skip the read-back

Print at 100% / "actual size" on matte paper, then measure the ruler.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrcf.perception import fiducials  # noqa: E402
from tools._common import deployment_config, rc  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "tag_sheets"

# ── What gets printed: ids and sizes are the runtime's, from the configs ─────
# AprilTag 36h11: 587 ids, Hamming distance 11 — the family the AprilTag
# authors recommend, and one OpenCV's ArUco module reads directly. Sizes are the
# edge of the BLACK square (what the pose solver uses); each sticker adds one
# cell of white margin all round (size x 10/8).
TAG_DICT = fiducials.TAG_DICTIONARY
TAG_FAMILY = "tag36h11"

_chess = deployment_config("chess")
BOARD_TAG_MM = _chess.BOARD_TAG_SIZE_M * 1000
# id -> where it goes on the printed diagram. a1 is nearest the arm on its LEFT
# (config ROBOT_PLAYS = "white"), so looking from behind the arm: 0 near-left,
# 1 near-right, 2 far-right, 3 far-left; 4 and 5 halfway up the left and right
# borders. (They went on the rig turned half a turn from this — harmless, since
# where each one sits is measured.)
BOARD_TAGS = {
    0: ("a1 corner", "near-left"),
    1: ("h1 corner", "near-right"),
    2: ("h8 corner", "far-right"),
    3: ("a8 corner", "far-left"),
    4: ("a-file side, between ranks 4 and 5", "left border, middle"),
    5: ("h-file side, between ranks 4 and 5", "right border, middle"),
}
assert tuple(BOARD_TAGS) == tuple(_chess.BOARD_TAG_IDS), "BOARD_TAGS out of step with the chess config"

# On the gripper body (on the SO-101, the back of the wrist camera's mount;
# robot_config.GRIPPER_TAG_SIZES_M):
# three sizes, one id each, so the runtime knows which was stuck on from the id
# alone. id -> black-square edge, mm.
GRIPPER_TAGS = {i: s * 1000 for i, s in rc.GRIPPER_TAG_SIZES_M.items()}

JAW_MARK_MM = (6.0, 8.0, 10.0)     # white disc diameters; centre dot is 40% of it
JAW_DOT_FRACTION = 0.4

# Where the camera goes: across the board from the arm, the closest point
# ~60 deg up that still sees the whole board with this lens (metres).
CAMERA = {"past_far_edge_m": 0.07, "height_above_board_m": 0.47,
          "distance_to_centre_m": 0.54, "down_deg": 60}

PAPER_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}
MARGIN_MM = 12.0


# ── A minimal vector PDF writer (no dependencies) ─────────────────────────────

PT = 72.0 / 25.4


class Pdf:
    """Pages of filled/stroked shapes and Helvetica text, in mm from the top-left."""

    def __init__(self, w_mm: float, h_mm: float):
        self.w, self.h = w_mm, h_mm
        self.pages: list[list[str]] = []
        self.images: list[tuple[bytes, int, int]] = []

    def page(self) -> None:
        self.pages.append([])

    def _op(self, s: str) -> None:
        self.pages[-1].append(s)

    def _xy(self, x: float, y: float) -> tuple[float, float]:
        return x * PT, (self.h - y) * PT

    @staticmethod
    def _rgb(c) -> str:
        c = (c, c, c) if isinstance(c, (int, float)) else c
        return " ".join(f"{v:.3f}" for v in c)

    def rects(self, rects, fill=0.0) -> None:
        """Many rectangles filled as ONE path: no hairline seams between cells."""
        ops = [f"{self._rgb(fill)} rg"]
        for x, y, w, h in rects:
            px, py = self._xy(x, y + h)
            ops.append(f"{px:.3f} {py:.3f} {w * PT:.3f} {h * PT:.3f} re")
        ops.append("f")
        self._op("\n".join(ops))

    def box(self, x, y, w, h, stroke=0.0, lw=0.2, dash=None) -> None:
        px, py = self._xy(x, y + h)
        d = f"[{' '.join(str(v * PT) for v in dash)}] 0 d " if dash else "[] 0 d "
        self._op(f"{d}{lw * PT:.3f} w {self._rgb(stroke)} RG {px:.3f} {py:.3f} {w * PT:.3f} {h * PT:.3f} re S")

    def line(self, x1, y1, x2, y2, stroke=0.0, lw=0.2, dash=None) -> None:
        a, b = self._xy(x1, y1), self._xy(x2, y2)
        d = f"[{' '.join(str(v * PT) for v in dash)}] 0 d " if dash else "[] 0 d "
        self._op(f"{d}{lw * PT:.3f} w {self._rgb(stroke)} RG {a[0]:.3f} {a[1]:.3f} m {b[0]:.3f} {b[1]:.3f} l S")

    def circle(self, cx, cy, r, fill=None, stroke=None, lw=0.2) -> None:
        k = 0.5523 * r
        pts = [(cx + r, cy), (cx + r, cy + k), (cx + k, cy + r), (cx, cy + r), (cx - k, cy + r),
               (cx - r, cy + k), (cx - r, cy), (cx - r, cy - k), (cx - k, cy - r), (cx, cy - r),
               (cx + k, cy - r), (cx + r, cy - k), (cx + r, cy)]
        p = [self._xy(*q) for q in pts]
        path = f"{p[0][0]:.3f} {p[0][1]:.3f} m " + " ".join(
            f"{p[i][0]:.3f} {p[i][1]:.3f} {p[i + 1][0]:.3f} {p[i + 1][1]:.3f} {p[i + 2][0]:.3f} {p[i + 2][1]:.3f} c"
            for i in range(1, 13, 3))
        if fill is not None and stroke is not None:
            self._op(f"[] 0 d {lw * PT:.3f} w {self._rgb(fill)} rg {self._rgb(stroke)} RG {path} B")
        elif fill is not None:
            self._op(f"{self._rgb(fill)} rg {path} f")
        else:
            self._op(f"[] 0 d {lw * PT:.3f} w {self._rgb(stroke if stroke is not None else 0)} RG {path} S")

    def poly(self, pts, fill=None, stroke=None, lw=0.2, dash=None) -> None:
        p = [self._xy(*q) for q in pts]
        path = f"{p[0][0]:.3f} {p[0][1]:.3f} m " + " ".join(f"{a:.3f} {b:.3f} l" for a, b in p[1:]) + " h"
        d = f"[{' '.join(str(v * PT) for v in dash)}] 0 d " if dash else "[] 0 d "
        if fill is not None and stroke is not None:
            self._op(f"{d}{lw * PT:.3f} w {self._rgb(fill)} rg {self._rgb(stroke)} RG {path} B")
        elif fill is not None:
            self._op(f"{self._rgb(fill)} rg {path} f")
        else:
            self._op(f"{d}{lw * PT:.3f} w {self._rgb(stroke if stroke is not None else 0)} RG {path} S")

    def text(self, x, y, s: str, size=8.0, bold=False, color=0.0) -> None:
        """Baseline at y. Latin-1 only (the standard fonts' WinAnsi encoding)."""
        esc = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        px, py = self._xy(x, y)
        self._op(f"BT /{'F2' if bold else 'F1'} {size:.2f} Tf {self._rgb(color)} rg "
                 f"{px:.3f} {py:.3f} Td ({esc}) Tj ET")

    def save(self, path: Path) -> None:
        objs: list[bytes] = []

        def add(b: bytes) -> int:
            objs.append(b)
            return len(objs)

        f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        pages_id = len(objs) + 1 + 2 * len(self.pages) + 1   # after page/content pairs and catalog
        kids = []
        for ops in self.pages:
            data = "\n".join(ops).encode("latin-1")
            c = add(b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")
            kids.append(add(
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {self.w * PT:.3f} {self.h * PT:.3f}] "
                f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> >> /Contents {c} 0 R >>".encode()))
        catalog = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
        assert add(f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
                   .encode()) == pages_id
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, b in enumerate(objs, 1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + b + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
        out += f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        path.write_bytes(bytes(out))


# ── Drawing the pieces ────────────────────────────────────────────────────────

def tag_cells(tag_id: int) -> np.ndarray:
    """8x8 cells of the tag (black border included), True = black."""
    d = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    img = cv2.aruco.generateImageMarker(d, tag_id, 8, borderBits=1)
    return img < 128


def draw_tag(pdf: Pdf, x: float, y: float, tag_id: int, size_mm: float,
             label: str = "", sub: str = "") -> float:
    """
    One sticker at (x, y) = its top-left (cut line). Returns its height incl. label.
    The cut line sits 0.5 mm outside the one-cell white margin, so a slightly
    generous cut still leaves a full cell of white round the black square.
    """
    cell = size_mm / 8
    quiet = cell
    cut = size_mm + 2 * quiet + 1.0
    pdf.box(x, y, cut, cut, stroke=0.65, lw=0.15, dash=(1.0, 1.0))
    ox, oy = x + 0.5 + quiet, y + 0.5 + quiet
    cells = tag_cells(tag_id)
    rects = []
    for r in range(8):            # merge each row's black runs into one rect
        c = 0
        while c < 8:
            if cells[r, c]:
                c0 = c
                while c < 8 and cells[r, c]:
                    c += 1
                rects.append((ox + c0 * cell, oy + r * cell, (c - c0) * cell, cell))
            else:
                c += 1
    pdf.rects(rects)
    if label:
        pdf.text(x, y + cut + 4.0, label, 7.5, bold=True)
    if sub:
        pdf.text(x, y + cut + 7.6, sub, 6.5, color=0.3)
    return cut + (9.5 if sub else 5.5)


def draw_jaw_mark(pdf: Pdf, cx: float, cy: float, d_mm: float) -> None:
    """White disc (cut along the grey circle) with a black centre dot."""
    pdf.circle(cx, cy, d_mm / 2, stroke=0.6, lw=0.12)
    pdf.circle(cx, cy, d_mm * JAW_DOT_FRACTION / 2, fill=0.0)


def draw_ruler(pdf: Pdf, x: float, y: float) -> float:
    """100 mm ruler, ticks every 5 mm. Returns its height."""
    pdf.text(x, y, "SCALE CHECK - measure before cutting anything", 8, bold=True)
    y0 = y + 4
    pdf.line(x, y0 + 4, x + 100, y0 + 4, lw=0.25)
    for i in range(0, 101, 5):
        h = 4 if i % 10 == 0 else 2
        pdf.line(x + i, y0 + 4 - h, x + i, y0 + 4, lw=0.2)
        if i % 10 == 0:
            pdf.text(x + i - (1.2 if i < 100 else 2.4), y0 + 8.5, str(i), 5.5)
    pdf.text(x, y0 + 13, "0 to 100 must measure exactly 100 mm.", 7)
    return 19.0


def board_diagram(pdf: Pdf, x: float, y: float, side: float) -> None:
    """Top view: board with border, tag spots, the arm, and the side camera."""
    s = side / 392.0                     # page mm per real mm (board 320 + 2 x 36)
    bx, by = x + 6, y + 4
    pdf.rects([(bx, by, side, side)], fill=(0.55, 0.38, 0.25))
    inner = 320 * s
    ix, iy = bx + 36 * s, by + 36 * s
    sq = inner / 8
    light, dark = [], []
    for r in range(8):
        for f in range(8):
            (dark if (r + f) % 2 == 0 else light).append((ix + f * sq, iy + (7 - r) * sq, sq, sq))
    pdf.rects(light, fill=(0.93, 0.87, 0.74))
    pdf.rects(dark, fill=(0.45, 0.25, 0.17))
    for f, ch in enumerate("abcdefgh"):
        pdf.text(ix + f * sq + sq / 2 - 0.8, by + side + 3.5, ch, 6)
    for r in range(8):
        pdf.text(bx - 3.5, iy + (7 - r) * sq + sq / 2 + 1, str(r + 1), 6)
    t = 32.5 * s
    mid = 18 * s
    spots = {0: (bx + mid, by + side - mid), 1: (bx + side - mid, by + side - mid),
             2: (bx + side - mid, by + mid), 3: (bx + mid, by + mid),
             4: (bx + mid, by + side / 2), 5: (bx + side - mid, by + side / 2)}
    for tid, (cx, cy) in spots.items():
        pdf.rects([(cx - t / 2, cy - t / 2, t, t)], fill=1.0)
        pdf.box(cx - t / 2, cy - t / 2, t, t, stroke=0.0, lw=0.2)
        pdf.text(cx - 1.2, cy + 1.2, str(tid), 6, bold=True)
    ax, ay = bx + side / 2, by + side + 11
    pdf.circle(ax, ay, 5.5, fill=(0.75, 0.75, 0.8), stroke=0.2)
    pdf.text(ax - 3.6, ay + 1.1, "ARM", 5.5, bold=True)
    # the camera, to scale, across the board from the arm
    cx, cy = bx + side / 2, by - CAMERA["past_far_edge_m"] * 1000 * s
    pdf.line(cx, cy + 2.5, cx, by + side / 2, lw=0.25, dash=(1.2, 1.0))
    pdf.rects([(cx - 3, cy - 2.5, 6, 5)], fill=0.15)
    tx, ty = bx + side + 8, by + 2
    pdf.line(cx + 3, cy, tx - 1, ty - 1, lw=0.15, stroke=0.4)
    pdf.text(tx, ty, "CAMERA (to scale): across the board from the arm", 7, bold=True)
    for i, line in enumerate([
            f"About {CAMERA['past_far_edge_m'] * 100:.0f} cm past the far (rank 8) edge, centred on",
            f"the board, {CAMERA['height_above_board_m'] * 100:.0f} cm above it, aimed at the board's",
            f"centre (~{CAMERA['down_deg']} deg down; 55-65 is fine). Higher still",
            "works, but every cm further away costs gripper-tag",
            "accuracy: at the old 66 cm it roughly doubles.",
            "",
            "It sees the board's near corners (0, 1) smallest and",
            "sometimes behind the arm; 4 and 5 cover for them."]):
        pdf.text(tx, ty + 3.6 * (i + 1), line, 6.5)


def board_sheet(pdf: Pdf) -> None:
    pdf.page()
    x, y = MARGIN_MM, MARGIN_MM + 4
    usable = pdf.w - 2 * MARGIN_MM
    pdf.text(x, y, f"BOARD TAGS - AprilTag {TAG_FAMILY}, {BOARD_TAG_MM:.0f} mm black square "
                   f"({BOARD_TAG_MM * 10 / 8:.1f} mm with its white margin)", 10, bold=True)
    pdf.text(x, y + 5, "Print at 100% on matte paper (gloss flares). Cut on the dotted lines - "
                       "keep the whole white margin.", 7.5)
    y += 10
    cut = BOARD_TAG_MM * 10 / 8 + 1.0
    gap = (usable - 6 * cut) / 5
    for row in (0, 1):
        hmax = 0.0
        for i, tid in enumerate(sorted(BOARD_TAGS)):
            plain = BOARD_TAGS[tid][1]
            h = draw_tag(pdf, x + i * (cut + gap), y, tid, BOARD_TAG_MM,
                         f"id {tid}" + ("" if row == 0 else "  (spare)"), plain if row == 0 else "")
            hmax = max(hmax, h)
        y += hmax + 2
    y += 2
    pdf.text(x, y, "WHERE THEY GO - top view, arm at the bottom", 8, bold=True)
    side = 70.0
    board_diagram(pdf, x, y + 2 + CAMERA["past_far_edge_m"] * 1000 * side / 392 + 4, side)
    y += side + 34
    for s, bold in [
            ("How to stick them", True),
            ("- Straight onto the wooden border, centred in the corner (4 and 5 halfway up the sides), flat and", False),
            ("  fully glued (glue stick or spray mount; no bubbles, no lifted corners). Not overlapping the squares.", False),
            ("- On the board, not the table: they must move with the board, which is what they measure.", False),
            ("- Which way up does not matter; the camera reads each tag's orientation. Where each one ended up", False),
            ("  is measured once by the camera after sticking: a tag glued to the board doesn't drift.", False),
            ("- Steady, diffuse light; autofocus, auto exposure and auto white balance off on the cameras.", False)]:
        pdf.text(x, y, s, 8 if bold else 7.5, bold=bold)
        y += 4.1
    y += 3
    draw_ruler(pdf, x, y)


def gripper_sheet(pdf: Pdf) -> None:
    pdf.page()
    x, y = MARGIN_MM, MARGIN_MM + 4
    pdf.text(x, y, f"GRIPPER TAG - AprilTag {TAG_FAMILY}, for the back of the wrist camera", 10, bold=True)
    pdf.text(x, y + 5, "Print at 100% on matte paper. Three sizes, one id each: use the BIGGEST whose sticker "
                       "sits flat; the id tells the software which.", 7.5)
    y += 10
    gap = 6.0
    for row in (0, 1):
        cx, hmax = x, 0.0
        for tid, mm in GRIPPER_TAGS.items():
            label = f"id {tid}: {mm:.0f} mm" + ("" if row == 0 else "  (spare)")
            sub = f"sticker {mm * 10 / 8:.0f} mm" if row == 0 else ""
            hmax = max(hmax, draw_tag(pdf, cx, y, tid, mm, label, sub))
            cx += mm * 10 / 8 + 1.0 + gap
        y += hmax + 3
    y += 1
    for s, bold in [
            ("Where it goes - flat on the gripper body, facing the overhead camera at every pick pose", True),
            ("- Nothing jointed between the tag and the jaws: not on the wrist, not on a moving jaw. The tag tour measures", False),
            ("  where it sits, once. On the SO-101: the flat back of the wrist camera's mount, facing away from the jaws.", False),
            ("- If the screw heads or the cable connector stop it lying flat, stick it on thin stiff card first, and keep the", False),
            ("  connector and cable free. It must not flex, tilt or shift; which way up does not matter.", False),
            ("- Expected, camera ~54 cm from the board centre: 32 mm ~2.5 mm, 28 mm ~2.8 mm, 24 mm ~3.9 mm median error at", False),
            ("  the grasp point. The camera's distance matters more than the tag: at 71 cm a 28 mm tag is twice as bad.", False)]:
        pdf.text(x, y, s, 8 if bold else 7.5, bold=bold)
        y += 4.1
    y += 5
    pdf.text(x, y, "JAW MARKS - for the wrist camera", 9, bold=True)
    y += 5
    for s in ["One on each jaw, on the face the wrist camera sees, centred ~10 mm back from the tip; not on the gripping",
              "faces. In the wrist image the FIXED jaw is on the left, the MOVING one on the right. Use the biggest that sits",
              "flat (8 mm reads ~29 px). Cut along the grey circle. The fixed jaw's mark also shows if the camera mount",
              "has been knocked: it must stay put in the wrist image."]:
        pdf.text(x, y, s, 7.5)
        y += 4.1
    y += 7
    for jaw in ("F  fixed jaw", "M  moving jaw"):
        pdf.text(x, y + 1.5, jaw, 7.5, bold=True)
        cx = x + 28
        for copy in (0, 1):                  # two of each size: one to use, one spare
            for d in JAW_MARK_MM:
                draw_jaw_mark(pdf, cx + d / 2, y, d)
                if copy == 1:
                    pdf.text(cx + d / 2 - 3, y + d / 2 + 4.5, f"{d:.0f} mm", 5.5, color=0.3)
                cx += d + 5
            cx += 6
        y += max(JAW_MARK_MM) + 9
    y += 4
    draw_ruler(pdf, x, y)


def table_sheet(pdf: Pdf) -> None:
    """
    The table target: fiducials.TABLE_TARGET_IDS in a 3 x 4 grid, to lay flat
    and uncut on whatever surface the arm works on — where there is no board to
    calibrate the wrist camera against (calibrate_gripper_pose.py --target tags).
    """
    pdf.page()
    x, y = MARGIN_MM, MARGIN_MM + 4
    mm = fiducials.TABLE_TARGET_TAG_M * 1000
    pdf.text(x, y, f"TABLE TARGET - AprilTag {TAG_FAMILY}, ids {fiducials.TABLE_TARGET_IDS[0]}-"
                   f"{fiducials.TABLE_TARGET_IDS[-1]}, {mm:.0f} mm", 10, bold=True)
    pdf.text(x, y + 5, "Print at 100% on matte paper. Do NOT cut: lay the whole sheet flat on the table, in the "
                       "overhead camera's view, where the arm reaches.", 7.5)
    y += 12
    cols, rows = 3, 4
    cut = mm + 2 * (mm / 8) + 1.0
    gap_x = (pdf.w - 2 * MARGIN_MM - cols * cut) / (cols - 1)
    gap_y = 14.0
    for r in range(rows):
        for c in range(cols):
            tid = fiducials.TABLE_TARGET_IDS[r * cols + c]
            draw_tag(pdf, x + c * (cut + gap_x), y, tid, mm, f"id {tid}")
        y += cut + gap_y
    draw_ruler(pdf, x, y)


# ── Read-back check ───────────────────────────────────────────────────────────

def read_back(pdf_path: Path, expect: dict[int, float], dpi: int = 600) -> list[str]:
    """Rasterise with Ghostscript, detect, and compare ids and printed sizes."""
    gs = shutil.which("gs")
    if gs is None:
        return ["  (no Ghostscript: skipped the read-back; check the print with a ruler)"]
    report = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pnggray", f"-r{dpi}",
                        f"-sOutputFile={tmp}/p%d.png", str(pdf_path)], check=True)
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_DICT),
                                      cv2.aruco.DetectorParameters())
        seen: dict[int, list[float]] = {}
        for png in sorted(Path(tmp).glob("p*.png")):
            img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            corners, ids, _ = det.detectMarkers(img)
            for c, i in zip(corners or [], [] if ids is None else ids.ravel()):
                q = c.reshape(4, 2)
                side = np.linalg.norm(np.diff(np.vstack([q, q[:1]]), axis=0), axis=1).mean()
                seen.setdefault(int(i), []).append(side / dpi * 25.4)
    ok = True
    for tid, mm in sorted(expect.items()):
        got = seen.get(tid, [])
        good = bool(got) and all(abs(g - mm) < 0.15 for g in got)
        ok &= good
        report.append(f"  id {tid:2d}: found {len(got)}x, black square "
                      f"{', '.join(f'{g:.2f}' for g in got) or '-'} mm (want {mm:.2f}) {'ok' if good else 'FAIL'}")
    stray = sorted(set(seen) - set(expect))
    if stray:
        ok = False
        report.append(f"  unexpected ids read: {stray}")
    report.append("  READ-BACK PASS" if ok else "  READ-BACK FAILED - do not print")
    return report


def main() -> None:
    paper = "a4"
    if "--paper" in sys.argv:
        paper = sys.argv[sys.argv.index("--paper") + 1].lower()
    if paper not in PAPER_MM:
        raise SystemExit(f"--paper must be one of {', '.join(PAPER_MM)}")
    OUT.mkdir(exist_ok=True)
    w, h = PAPER_MM[paper]

    board = Pdf(w, h)
    board_sheet(board)
    board_path = OUT / f"board_tags_{paper}.pdf"
    board.save(board_path)

    grip = Pdf(w, h)
    gripper_sheet(grip)
    grip_path = OUT / f"gripper_tags_{paper}.pdf"
    grip.save(grip_path)

    table = Pdf(w, h)
    table_sheet(table)
    table_path = OUT / f"table_target_{paper}.pdf"
    table.save(table_path)

    print(f"Wrote {board_path}\n      {grip_path}\n      {table_path}")
    if "--no-check" not in sys.argv:
        print(f"\nRead-back of {board_path.name}:")
        print("\n".join(read_back(board_path, {i: BOARD_TAG_MM for i in BOARD_TAGS})))
        print(f"Read-back of {grip_path.name}:")
        print("\n".join(read_back(grip_path, GRIPPER_TAGS)))
        print(f"Read-back of {table_path.name}:")
        print("\n".join(read_back(table_path, {i: fiducials.TABLE_TARGET_TAG_M * 1000
                                               for i in fiducials.TABLE_TARGET_IDS})))
    print(f"""
Print at 100% ("Actual size", not "Fit to page"), on {paper.upper()}, matte paper, from any PDF
viewer — or on macOS or Linux:
    lpr -o media={paper.upper()} -o fit-to-page=false {board_path}
Then check the ruler reads 100 mm before cutting.""")


if __name__ == "__main__":
    main()
