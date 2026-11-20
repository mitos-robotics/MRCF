"""
tones.py — where a colour falls between two of the surface's own.

A colour read from a camera on auto exposure and auto white balance is not a
fixed thing: the same cream wood read Lab lightness 163 one day and 246
another, and a hue of 30-43 deg from one camera position, 0-15 from the next.
Two references in the same frame move with it — under the same light, through
the same exposure and white balance. So an object's colour is judged by where
it falls on the line between two colours of the surface it stands on (the two
woods of a chessboard, a printed light and dark patch):

    between = (sample - light) . (dark - light) / |dark - light|^2

0 at the light reference, 1 at the dark; by colour (Lab a, b) when the two
references differ in colour, by lightness when they differ only in that.
"""

from __future__ import annotations

import numpy as np

MIN_COLOUR_GAP = 8.0          # Lab a, b units between the references for colour to decide
MIN_LIGHTNESS_GAP = 20.0      # Lab L units between them for lightness to, when colour does not


def between(sample_lab, dark_lab, light_lab, min_colour_gap: float = MIN_COLOUR_GAP,
            min_lightness_gap: float = MIN_LIGHTNESS_GAP) -> "float | None":
    """
    Where sample falls between the references: 0 at light_lab, 1 at dark_lab
    (Lab triples, OpenCV's 0-255 scale; beyond either end is fine). By colour,
    or by lightness when the references barely differ in colour. None when
    they are not told apart either way.
    """
    p, d, l = (np.asarray(v, float) for v in (sample_lab, dark_lab, light_lab))
    axis = d[1:] - l[1:]
    if float(np.linalg.norm(axis)) >= min_colour_gap:
        return float((p[1:] - l[1:]) @ axis / (axis @ axis))
    if l[0] - d[0] < min_lightness_gap:
        return None
    return float((l[0] - p[0]) / (l[0] - d[0]))
