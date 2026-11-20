"""
capture_surface_reference.py — photograph the empty work surface for surface_scan.

surface_scan finds objects by outlining whatever stands on the surface. On a
patterned surface — tiles, a printed mat, a chessboard — the pattern's own
parts get outlined too; this photo of the surface with nothing on it is how it
tells them apart without knowing the pattern: an outline that is mostly
unchanged from the photo is surface.

Clear the surface, run this, put things back. The arm parks out of the
overhead camera's view first. Retake whenever the surface or the camera moves,
or the light changes a lot.

Usage (from the repo root):
    python tools/capture_surface_reference.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import make_env, rc  # noqa: E402


def main() -> None:
    from mrcf.perception import surface_scan

    env = make_env()
    env.connect()
    try:
        env.go_park()
        input("Clear the work surface completely, then press Enter...")
        frame = env.render_image(rc.OVERHEAD_CAMERA_NAME, width=rc.OVERHEAD_CAMERA_W,
                                 height=rc.OVERHEAD_CAMERA_H)
        surface_scan.save_reference(frame, rc.SURFACE_REFERENCE_PATH)
        print(f"Saved {rc.SURFACE_REFERENCE_PATH}")
    finally:
        try:
            env.go_home()
        finally:
            env.stop()


if __name__ == "__main__":
    main()
