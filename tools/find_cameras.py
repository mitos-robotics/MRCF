"""
find_cameras.py — the cameras on this computer, by name, without opening
any; which are the rig's and which the computer's own; and, when asked for
one index, a picture from it.

Plugging a USB camera in can renumber the others (macOS orders them as it
finds them), and an entry in the arm's CAMERAS whose source opens another
camera than it was calibrated as gives a scan in the wrong place and a
sentry watching the wrong view — or, with the rig's camera unplugged, the
computer's own camera, looking at its user. So run this after adding or
moving a camera.

It never sweeps the indexes: 2026-11-14 a sweep opened an iPhone's
Continuity Camera, and its pictures were of the room. The computer's own
cameras (the laptop's, an iPhone's, Desk View) are named and never opened
(mrcf/privacy.personal). On macOS ffmpeg's listing does not number cameras
as OpenCV does, so it says which cameras are plugged in, not which index is
which; to see which camera an index opens, ask for its picture (--picture N),
taken only when every rig camera the arm names is plugged in (no index can
then be the computer's own), faces covered (mrcf/privacy.py).

    python tools/find_cameras.py                # the list; nothing opened
    python tools/find_cameras.py --picture 0    # index 0's picture, reports/cameras/<time>/index_0.jpg

Quit anything else using the cameras first (run.py, another tool): a camera
in use may not open twice.
"""

import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tools._common import FRAMEWORK_ROOT, arg, backends, refuse_personal_cameras  # noqa: E402
from mrcf import devices, privacy  # noqa: E402

OUT = FRAMEWORK_ROOT / "reports" / "cameras"
# A camera's first frames are black or white while its exposure settles: 10
# (a third of a second) gave a new camera's picture as pure white, another's
# as black (2026-11-13). The picture kept is the last of these, ~2 s at 30 fps.
FIRST_FRAMES = 60


def grab(index: int, size=(1280, 720)) -> "np.ndarray | None":
    """A frame (BGR) from camera `index` at about size, or None when it does not open or sends nothing."""
    import cv2

    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        frame = None
        for _ in range(FIRST_FRAMES):
            ok, got = cap.read()
            if ok and got is not None:
                frame = got
        return frame
    finally:
        cap.release()


def listing(listed: "dict[int, str]", cameras: dict, system: "str | None" = None) -> list:
    """The report, as lines: every camera by name (rig's or the computer's own), then the arm's CAMERAS against them."""
    system = system or platform.system()
    if not listed:
        return ["No camera listing on this computer (macOS needs ffmpeg; Linux lists /sys/class/video4linux):",
                "nothing can say which index is which camera. Nothing was opened."]
    numbered = system == "Linux"
    lines = ["Cameras on this computer (none opened):"]
    for i, name in sorted(listed.items()):
        own = privacy.personal(name)
        where = f"/dev/video{i}" if numbered else f"listed {i}"
        what = "this computer's own: never opened" if own else "may be the rig's"
        lines.append(f"  {where:<12} {name:<40} {what}")
    rig = devices.rig_cameras(listed)
    sources = {n: c.source for n, c in cameras.items() if c.source is not None}
    lines.append("")
    lines.append("The arm's CAMERAS: " + (", ".join(f"{n} = {s!r}" for n, s in sources.items()) or "none"))
    indexes = {s for s in sources.values() if isinstance(s, int)}
    if len(rig) >= len(indexes):
        lines.append(f"  {len(rig)} camera(s) that may be the rig's, for {len(indexes)} index(es): enough.")
    else:
        lines.append(f"  Only {len(rig)} camera(s) that may be the rig's, for {len(indexes)} index(es): not all "
                     "are plugged in, and an index may now open this computer's own camera — the arm will not open them.")
    if not numbered:
        lines.append("  macOS: ffmpeg's numbers above are not OpenCV's. Which index is which: --picture N.")
    return lines


def main() -> None:
    config = backends.arm().config
    try:
        listed = devices.video_devices()
    except RuntimeError as exc:
        print(f"Could not list the cameras: {exc}")
        listed = {}
    print("\n".join(listing(listed, getattr(config, "CAMERAS", {}) or {})))
    wanted = arg("--picture", None)
    if wanted is None:
        return
    index = int(wanted)
    if platform.system() == "Linux" and privacy.personal(listed.get(index)):
        raise SystemExit(f"/dev/video{index} is '{listed[index]}', this computer's own camera: not opened.")
    refuse_personal_cameras()                     # every rig camera plugged in, or nothing opened
    print(f"\nOpening index {index} for one picture (faces covered"
          + ("" if privacy.has_face_model() else " — NOT looked for: python tools/fetch_face_model.py") + ")...")
    frame = grab(index)
    if frame is None:
        print(f"  {index}: nothing")
        return
    out = OUT / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"index_{index}.jpg"
    privacy.imwrite(str(path), frame)
    h, w = frame.shape[:2]
    using = [n for n, c in (getattr(config, "CAMERAS", {}) or {}).items() if c.source == index]
    print(f"  {index}: {w}x{h}, brightness {float(frame.mean()):.0f}/255, {path}"
          + (f"  <- CAMERAS '{using[0]}'" if using else ""))
    print("Look at the picture: the entry in CAMERAS with this index should be the view it was calibrated as.")


if __name__ == "__main__":
    main()
