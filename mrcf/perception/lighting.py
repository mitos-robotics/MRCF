"""
lighting.py — is the light good enough to see by? Numbers and a warning in
words, for any camera frame.

The cameras on a rig like this one run on auto exposure that cannot be turned
off (the SO-101's two ignore every UVC image control), so the light is
whatever the room gives. Two ways it goes wrong:

  washed out   pixels at the sensor's maximum. A cream object on a light
               surface is then the same white as the surface, and nothing done
               to the picture afterwards brings the difference back: outlines
               come out as scraps, heights as nonsense, colours as the wrong one.
  too dark     everything near the bottom of the range, where a dark object on
               a dark surface stops standing out (picks fail at a wrist-camera
               mean of 41-67 and work at 102-136).

reading() gives the numbers, warning() says what is wrong, in words, or None.
A deployment that knows its surface (chess: which squares are light) can say
more — where — and does (deployments/chess/chessboard.py, lighting_report).

The runs behind its values and fixes, with their dates: mrcf/HISTORY.md.
"""

from __future__ import annotations

import cv2
import numpy as np

# A picture darker than this on average is no picture at all: a camera covered,
# or a camera number that now opens another one (a laptop's own, lid shut).
NO_PICTURE_MEAN = 3.0
CLIPPED = 250                 # any colour channel at or above this: at the sensor's maximum
WASHED_OUT = 0.08             # this share of a view at the maximum: warn
TOO_DARK_MEAN = 90            # a view's mean grey below this: warn


def no_picture(rgb: np.ndarray) -> bool:
    """All but black: nothing in it can be seen, let alone seen to move."""
    return float(np.asarray(rgb).mean()) < NO_PICTURE_MEAN


def clipped(rgb: np.ndarray) -> np.ndarray:
    """Bool mask of the pixels at the sensor's maximum in any channel."""
    return np.asarray(rgb).max(2) >= CLIPPED


def reading(rgb: np.ndarray, area: "np.ndarray | None" = None) -> dict:
    """
    Over `area` (bool mask; default the whole frame) of an RGB frame:
    clipped (share at the maximum), bright and dark (Lab lightness, 0-255, of
    its lightest and darkest tenth) and mean (grey level).
    """
    rgb = np.asarray(rgb)
    area = np.ones(rgb.shape[:2], bool) if area is None else np.asarray(area, bool)
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0][area]
    return {"clipped": float(clipped(rgb)[area].mean()), "bright": float(np.percentile(L, 90)),
            "dark": float(np.percentile(L, 10)), "mean": float(rgb[area].mean())}


def region_stats(rgb: np.ndarray, centres_px: dict, half_px: int) -> dict:
    """
    {name: (median Lab lightness, share at the maximum)} of the square patch
    half_px either side of each centre (u, v) — the light, area by area: a
    chessboard's squares, a grid over a table. Patches off the frame are left out.
    """
    rgb = np.asarray(rgb)
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    at_max = clipped(rgb)
    out = {}
    for name, (u, v) in centres_px.items():
        u, v = int(round(u)), int(round(v))
        patch = (slice(max(v - half_px, 0), v + half_px + 1), slice(max(u - half_px, 0), u + half_px + 1))
        if L[patch].size:
            out[name] = (float(np.median(L[patch])), float(at_max[patch].mean()))
    return out


def warning(r: dict, what: str = "the camera's view") -> "str | None":
    """What is wrong with the light in a reading(), in words — or None when it is fine."""
    if r["clipped"] > WASHED_OUT:
        return (f"{r['clipped']:.0%} of {what} is washed out (at the camera's maximum): light objects on light "
                "surfaces can be missed or mis-measured there. Less light, or softer light, on the work surface "
                "(a curtain, a lamp turned away).")
    if r["mean"] < TOO_DARK_MEAN:
        return (f"{what} is dark (mean {r['mean']:.0f} of 255): dark objects on dark surfaces can be missed. "
                "More light on the work surface.")
    return None
