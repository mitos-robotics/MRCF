"""
live_view.py — a live picture from one of the arm's cameras, for aiming it.

No arm: the camera is opened directly, by its source in the arm's CAMERAS.
Quit run.py (and any other tool) first: a camera in use may not open twice.

Keys, in the window:
  g        the ghost on/off: the view the camera's calibration was made for
           (the rig's surface_reference.png, for the overhead camera; --ghost
           PATH for any other picture), half see-through over the live one.
           Line it up to put the camera back where it was, or see how far a
           new aim is from the old one.
  c        crosshair and thirds on/off, for aiming
  s        save this frame (reports/cameras/live_<time>.jpg)
  q, Esc   quit

The top line: how far the picture has slid from the ghost (px, while there is
one to compare with), how bright it is, and how much of it is blown out — glare
off the windows, where a scan sees nothing. A tug on the cable while moving the
camera loses it: the window says so and keeps trying to open it again.

After a camera has moved, its calibration no longer holds: COMMANDS.md, "Move
or re-aim the overhead camera", says what to run.

    python tools/live_view.py                     the overhead camera
    python tools/live_view.py --camera side       another of CAMERAS
    python tools/live_view.py --frames 60         no window: 60 frames, the numbers, quit
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools._common import FRAMEWORK_ROOT, arg, camera_size, camera_source, chosen_camera, rc  # noqa: E402
from mrcf import privacy  # noqa: E402

OUT = FRAMEWORK_ROOT / "reports" / "cameras"
WINDOW = "live view (g ghost, c crosshair, s save, q quit)"
GLARE_LEVEL = 250          # a pixel this bright in every channel is blown out
COMPARE_EVERY = 10         # frames between shift estimates (feature matching costs ~20 ms)
COMPARE_W = 640            # the shift is measured on pictures this wide
MIN_MATCHES = 30           # features agreeing on one shift, for it to mean anything
# Phase correlation, tried first, read a 130 px move of the UR5 rig's overhead
# camera (2026-11-13) as no match at all: features matched across the two
# pictures follow a move of any size, and a turn too.
_orb = cv2.ORB_create(1500)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def open_camera(source, size: "tuple[int, int]"):
    """The camera, opened at size, or None when it will not open; never this computer's own (SystemExit)."""
    from tools._common import refuse_personal_cameras

    refuse_personal_cameras()
    if isinstance(source, int) and sys.platform == "darwin":
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


def brightness(frame: np.ndarray) -> float:
    """Mean brightness, 0-255."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def glare(frame: np.ndarray) -> float:
    """The share of the picture blown out (every channel at GLARE_LEVEL or above), 0-1."""
    return float((frame.min(axis=2) >= GLARE_LEVEL).mean())


def _small_grey(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def shift_from(ghost: np.ndarray, frame: np.ndarray) -> "tuple[float, float, float, int] | None":
    """
    Where the ghost's middle is in frame: (dx, dy) in frame pixels from frame's
    middle, the turn between them (degrees), and how many features agree on it;
    None when too few match for it to mean anything.
    """
    scale = frame.shape[1] / COMPARE_W
    a, b = _small_grey(ghost, COMPARE_W), _small_grey(frame, COMPARE_W)
    ka, da = _orb.detectAndCompute(a, None)
    kb, db = _orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < MIN_MATCHES or len(kb) < MIN_MATCHES:
        return None
    pairs = _matcher.match(da, db)
    if len(pairs) < MIN_MATCHES:
        return None
    pa = np.float32([ka[p.queryIdx].pt for p in pairs])
    pb = np.float32([kb[p.trainIdx].pt for p in pairs])
    H, inliers = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None or inliers is None or int(inliers.sum()) < MIN_MATCHES:
        return None
    middle = np.float32([[[a.shape[1] / 2, a.shape[0] / 2]], [[a.shape[1] / 2 + 50, a.shape[0] / 2]]])
    (m0, m1) = cv2.perspectiveTransform(middle, H).reshape(2, 2)
    turn = float(np.degrees(np.arctan2(m1[1] - m0[1], m1[0] - m0[0])))
    return ((m0[0] - b.shape[1] / 2) * scale, (m0[1] - b.shape[0] / 2) * scale, turn, int(inliers.sum()))


def status_line(frame: np.ndarray, shift: "tuple[float, float, float, int] | None",
                compared: bool = True) -> str:
    parts = []
    if compared:
        if shift is None:
            parts.append("no match with the ghost")
        else:
            dx, dy, turn, _ = shift
            parts.append(f"ghost's middle {np.hypot(dx, dy):.0f} px off ({dx:+.0f}, {dy:+.0f}), "
                         f"turned {turn:+.1f} deg")
    parts.append(f"brightness {brightness(frame):.0f}/255")
    parts.append(f"blown out {glare(frame) * 100:.1f}%")
    return "   ".join(parts)


def draw(frame: np.ndarray, ghost: "np.ndarray | None", show_ghost: bool, show_guides: bool,
         line: str) -> np.ndarray:
    """The picture shown: the live frame, the ghost over it, the guides, the status line."""
    out = frame.copy()
    if show_ghost and ghost is not None:
        g = ghost if ghost.shape == frame.shape else cv2.resize(ghost, (frame.shape[1], frame.shape[0]))
        out = cv2.addWeighted(out, 0.5, g, 0.5, 0)
    h, w = out.shape[:2]
    if show_guides:
        for x in (w // 3, 2 * w // 3):
            cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
        for y in (h // 3, 2 * h // 3):
            cv2.line(out, (0, y), (w, y), (255, 255, 255), 1)
        cv2.line(out, (w // 2 - 30, h // 2), (w // 2 + 30, h // 2), (0, 0, 255), 2)
        cv2.line(out, (w // 2, h // 2 - 30), (w // 2, h // 2 + 30), (0, 0, 255), 2)
    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(out, line, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def lost(size: "tuple[int, int]", source) -> np.ndarray:
    picture = np.zeros((size[1], size[0], 3), np.uint8)
    cv2.putText(picture, f"camera {source!r} lost - opening it again...", (40, size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return picture


def ghost_for(name: str) -> "np.ndarray | None":
    """--ghost PATH, else the overhead camera's surface reference; None when there is none."""
    path = arg("--ghost")
    if path is None and name == rc.OVERHEAD_CAMERA_NAME:
        path = getattr(rc, "SURFACE_REFERENCE_PATH", None)
    if path is None or not Path(path).is_file():
        return None
    return cv2.imread(str(path))


def headless(cap, n: int, ghost) -> None:
    """--frames N: grab n frames, print the numbers, quit. What a check of this tool runs."""
    got, started, frame = 0, time.monotonic(), None
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None:
            frame, got = f, got + 1
    seconds = time.monotonic() - started
    if frame is None:
        raise SystemExit("The camera opened but sent no frames.")
    print(f"{got}/{n} frames, {frame.shape[1]}x{frame.shape[0]}, {got / max(seconds, 1e-6):.0f} fps")
    print(status_line(frame, shift_from(ghost, frame) if ghost is not None else None, ghost is not None))


def main() -> None:
    name = chosen_camera()
    source, size = camera_source(name), camera_size(name)
    ghost = ghost_for(name)
    cap = open_camera(source, size)
    if cap is None:
        raise SystemExit(f"Camera '{name}' ({source!r}) does not open. Is run.py or another tool using it?")
    print(f"Camera '{name}' ({source!r}), {size[0]}x{size[1]}; ghost: "
          + ("the view it was calibrated for" if ghost is not None else "none"))
    frames = arg("--frames")
    try:
        if frames is not None:
            headless(cap, int(frames), ghost)
            return
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, *size)
        show_ghost, show_guides, shift, count, misses = False, True, None, 0, 0
        while True:
            ok, frame = cap.read() if cap is not None else (False, None)
            if not ok or frame is None:
                misses += 1
                cv2.imshow(WINDOW, lost(size, source))
                if misses % 15 == 0:                   # about every half second: open it again
                    if cap is not None:
                        cap.release()
                    cap = open_camera(source, size)
            else:
                misses = 0
                if ghost is not None and count % COMPARE_EVERY == 0:
                    shift = shift_from(ghost, frame)
                count += 1
                cv2.imshow(WINDOW, draw(frame, ghost, show_ghost, show_guides,
                                        status_line(frame, shift, ghost is not None)))
            key = cv2.waitKey(30 if misses else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                show_ghost = not show_ghost and ghost is not None
            elif key == ord("c"):
                show_guides = not show_guides
            elif key == ord("s") and ok and frame is not None:
                OUT.mkdir(parents=True, exist_ok=True)
                path = OUT / f"live_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                privacy.imwrite(str(path), frame)
                print(f"Saved {path}")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break                                  # the window's own close button
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
