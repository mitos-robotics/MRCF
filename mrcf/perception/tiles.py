"""
tiles.py — objects cut from one camera frame and laid out for a vision model
to name all at once: each turned to stand upright, framed from below its foot
to where the tallest of them would reach, numbered on one sheet.

One crop per object, one call per crop, costs a vision model's whole turn per
object, and a crop framed round an outline from a camera looking down takes
in the object's foot and not its top: a tall object seen at a slant leans away
from its foot in the frame, and that is where what tells it apart is. The
camera's view knows where the top falls (PlaneView.lifted), so each object is
cut from below its foot to there, turned so it stands up its tile the way a
person would look at it. On the chess rig (2026-11-03) all 32 pieces on one
sheet took 16-27 s and got 30-32 of the colours right, where one crop a call
took 190 s a look, got 23, and called 8 of the pieces not a piece.

  upright(frame, view, foot_xy, radius_mm, height_mm)   one object's tile
  sheet(tiles)                                         the tiles on one image, numbered from 1
  png_b64(image)                                       an RGB image as a vision model takes it
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np

from mrcf.perception.plane_geometry import PlaneView

TILE_WH = (112, 160)        # a tile, w x h: upright objects are taller than wide
COLS = 8                    # tiles a row: a sheet of 32 is 896 x 728 px
_BELOW = 1.8                # foot radii below the foot, in the tile
_ABOVE = 0.8                # foot radii above the top
_ACROSS = 2.4               # foot radii either side
_MIN_RADIUS_PX = 4.0
_LABEL_PX = 22              # the number's band over each tile


def upright(frame: np.ndarray, view: PlaneView, foot_xy, radius_mm: float, height_mm: float,
            size: "tuple[int, int]" = TILE_WH) -> np.ndarray:
    """
    The object standing at foot_xy (the view's plane coordinates) as a tile of
    size (w, h): turned so its axis runs up the tile, its foot near the bottom,
    framed up to height_mm above the surface. An object seen from straight above,
    its axis a point, comes out as it lies in the frame.
    """
    w, h = size
    xy = np.asarray(foot_xy, float)[:2]
    foot = view.project(view.lifted(xy, 0.0)[None])[0]
    top = view.project(view.lifted(xy, height_mm)[None])[0]
    side = view.project(np.array([[xy[0] + radius_mm / view.scale, xy[1], 0.0]]))[0]
    r = max(float(np.linalg.norm(side - foot)), _MIN_RADIUS_PX)
    axis = top - foot
    length = float(np.linalg.norm(axis))
    up = axis / length if length > 1.0 else np.array([0.0, -1.0])
    across = np.array([-up[1], up[0]])
    lo, hi, half = -_BELOW * r, max(length, 2.0 * r) + _ABOVE * r, _ACROSS * r
    s = min(h / (hi - lo), w / (2.0 * half))
    bottom = h - (h - s * (hi - lo)) / 2.0
    src = np.array([foot + lo * up, foot + hi * up, foot + lo * up + half * across], np.float32)
    dst = np.array([[w / 2.0, bottom], [w / 2.0, bottom - s * (hi - lo)], [w / 2.0 + s * half, bottom]], np.float32)
    return cv2.warpAffine(frame, cv2.getAffineTransform(src, dst), (w, h), flags=cv2.INTER_CUBIC,
                          borderValue=(255, 255, 255))


def sheet(tiles: list, cols: int = COLS) -> np.ndarray:
    """The tiles in rows of cols, each in a box with its number (1, 2, ...) over it."""
    if not tiles:
        raise ValueError("no tiles")
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * (h + _LABEL_PX), min(cols, len(tiles)) * w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (h + _LABEL_PX), c * w
        out[y + _LABEL_PX:y + _LABEL_PX + h, x:x + w] = t
        cv2.putText(out, str(i + 1), (x + 4, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.rectangle(out, (x, y), (x + w - 1, y + _LABEL_PX + h - 1), (0, 0, 0), 1)
    return out


def png_b64(image_rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image_rgb)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
