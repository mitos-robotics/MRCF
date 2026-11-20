"""
check_exposure.py — is a camera's exposure right for telling things apart?
Moves nothing; only reads a camera.

Why it matters
--------------
Both cameras run on auto exposure, which sets the picture by the WHOLE frame:
the dark floor and curtains round the table pull it up until the light squares
of a chessboard are at the sensor's maximum. Where a pixel is at the maximum,
a cream piece and the maple under it are the same white, and nothing done to
the picture afterwards brings the difference back — on 2026-10-27, 35% of the
light squares' pixels were clipped, and cream pieces were outlined as scraps.

Setting the exposure by hand needs a camera that obeys its UVC controls. This
rig's two do not (2026-10-27): both take manual exposure, brightness,
saturation, gamma and backlight compensation, report them back, and send the
same picture whatever they are set to — so a camera-control app cannot fix it
either. What this is for: judging a change of camera, of placement or of
light. Every second, for the work surface only:

  clipped   share of its pixels at the maximum in any colour      want < 1%
  bright    lightness (Lab L, 0-255) of its lightest tenth —
            the light squares, white things                         want 190-225
  dark      of its darkest tenth                                    want > 40
  drift     how much bright moved since the last reading — with auto
            exposure still on it wanders as a hand or a cloud passes

Usage:
    python tools/check_exposure.py                   # overhead, live
    python tools/check_exposure.py --camera wrist     # or any camera in the arm's CAMERAS, by name
    python tools/check_exposure.py --show             # also a preview, clipped in red
    python tools/check_exposure.py --once             # one reading, frame saved
    python tools/check_exposure.py --area 0.11,0.43,-0.15,0.17   # just this part of the
                                                    # surface (arm frame, m; default WORKSPACE_BOUNDS) —
                                                    # here a chessboard's squares, far ranks included
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2            # noqa: E402
import numpy as np    # noqa: E402

from mrcf.perception import lighting  # noqa: E402
from tools._common import camera_model_of, camera_size, camera_source, rc, resolve_camera  # noqa: E402
from mrcf import privacy  # noqa: E402

CLIPPED = lighting.CLIPPED
WANT_CLIPPED = 0.01
WANT_BRIGHT = (190, 225)
WANT_DARK = 40


def work_area(camera: str, wh, bounds=None) -> np.ndarray:
    """
    Bool mask of the pixels that see the work surface: the overhead's view of
    bounds (x0, x1, y0, y1, arm frame, m; default WORKSPACE_BOUNDS); all of the
    wrist camera's frame.
    """
    w, h = wh
    mask = np.ones((h, w), bool)
    model = camera_model_of(camera)
    if model is None:
        return mask
    from mrcf.perception import plane_geometry as pg
    sx, sy = w / model["size"][0], h / model["size"][1]
    view = pg.PlaneView.from_camera_model(model)
    if bounds is None:
        b = rc.WORKSPACE_BOUNDS
        bounds = (b.x_min, b.x_max, b.y_min, b.y_max)
    x0, x1, y0, y1 = bounds
    corners_mm = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * 1000
    px = view.project(np.c_[corners_mm, np.zeros(4)]) * [sx, sy]
    mask[:] = False
    cv2.fillPoly(mask.view(np.uint8), [np.round(px).astype(np.int32)], 1)
    return mask


def reading(rgb: np.ndarray, area: np.ndarray) -> dict:
    """The numbers this prints, over `area` of an RGB frame (lighting.reading)."""
    return lighting.reading(rgb, area)


def verdict(r: dict) -> str:
    """What to change, in words, or "ok"."""
    if r["clipped"] > WANT_CLIPPED or r["bright"] > WANT_BRIGHT[1]:
        return "too bright — lower the exposure"
    if r["bright"] < WANT_BRIGHT[0] or r["dark"] < WANT_DARK:
        return "too dark — raise the exposure (or the gain)"
    return "ok"


def zebra(rgb: np.ndarray, area: np.ndarray) -> np.ndarray:
    """BGR preview: clipped pixels red, the work area outlined."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out[rgb.max(2) >= CLIPPED] = (0, 0, 255)
    edge = cv2.morphologyEx(area.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = (0, 255, 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--camera", default="overhead", help="overhead, wrist, or any camera in the arm's CAMERAS")
    ap.add_argument("--show", action="store_true", help="a preview window, clipped pixels in red (q to quit)")
    ap.add_argument("--once", action="store_true", help="one reading; the frame saved beside this script")
    ap.add_argument("--area", help="x0,x1,y0,y1 of the surface to judge, arm frame, metres (a calibrated fixed camera)")
    args = ap.parse_args()

    args.camera = resolve_camera(args.camera)
    index, (w, h) = camera_source(args.camera), camera_size(args.camera)
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise SystemExit(f"Could not open the {args.camera} camera (index {index}).")
    for _ in range(15):                       # let it settle
        cap.read()
    ok, bgr = cap.read()
    if not ok:
        raise SystemExit(f"The {args.camera} camera opened but sent no frame.")
    bounds = tuple(float(v) for v in args.area.split(",")) if args.area else None
    if bounds is not None and len(bounds) != 4:
        raise SystemExit("--area wants four numbers: x0,x1,y0,y1")
    area = work_area(args.camera, (bgr.shape[1], bgr.shape[0]), bounds)
    print(f"{args.camera} camera, {bgr.shape[1]}x{bgr.shape[0]}, over "
          f"{'the work surface' if camera_model_of(args.camera) is not None else 'the whole frame'}. "
          f"Want: clipped < {WANT_CLIPPED:.0%}, bright {WANT_BRIGHT[0]}-{WANT_BRIGHT[1]}, dark > {WANT_DARK}."
          + ("" if args.once else "  Ctrl-C to stop."))
    last = None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("  (no frame)")
                time.sleep(0.5)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            r = reading(rgb, area)
            drift = "" if last is None else f"  drift {r['bright'] - last:+5.1f}"
            print(f"  clipped {r['clipped']:6.1%}   bright {r['bright']:5.0f}   dark {r['dark']:5.0f}{drift}   "
                  f"{verdict(r)}", flush=True)
            last = r["bright"]
            if args.once:
                path = Path(__file__).with_name(f"check_exposure_{args.camera}.png")
                privacy.imwrite(str(path), zebra(rgb, area))
                print(f"  Frame, clipped pixels in red: {path}")
                return
            if args.show:
                cv2.imshow(f"{args.camera} — clipped in red (q to quit)", zebra(rgb, area))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
