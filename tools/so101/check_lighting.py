"""
check_lighting.py — live brightness/sharpness meter at the pose that matters.

Tuning lighting by re-running a calibration is a slow loop with a bad signal:
you find out whether it worked several minutes later, from whether a detection
succeeded. This parks the arm where visual servoing actually operates and
prints the two numbers that decide it, continuously, so you can move a lamp and
watch them respond.

What the numbers mean
---------------------
brightness  mean grey level, 0-255. Every probe frame that detected reliably on
            this rig sat between 102 and 136. Below ~60 the piece stops being
            distinguishable from shadow; above ~190 highlights clip and the
            piece washes into the light squares.

sharpness   variance of the Laplacian divided by the image variance. Raw
            Laplacian variance is the usual focus metric but it conflates
            sharpness with CONTRAST — a dim frame scores low even when
            perfectly focused, which is exactly how a dark frame got
            misdiagnosed as blurry here. Dividing it out makes the number
            comparable between lighting conditions.

            Measured on this rig: ~0.21-0.43 with a piece at the jaws,
            ~0.065-0.108 with a piece on the board at hover height. The board
            is genuinely softer, so the lens is focused nearer than the plane
            that matters. Above ~0.15 at this pose is the target.

detect      whether piece_vision would find the piece right now, and with how
            many consensus votes. This is the number that actually decides
            whether servoing works; the other two explain it.

Usage:
    python tools/so101/check_lighting.py
    python tools/so101/check_lighting.py --height 0.02
"""

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2            # noqa: E402
import numpy as np    # noqa: E402

from tools.so101._common import make_env, rc, task_config  # noqa: E402
from mrcf import privacy  # noqa: E402

cfg = task_config()
HERE = rc.RIG_DIR             # this rig's folder (rig_state.py)

PROBE_XY = (0.24, 0.0)      # matches calibrate_gripper_servo.py
GOOD_BRIGHT = (100, 150)
GOOD_SHARP = 0.15


def main() -> None:
    argv = sys.argv
    height = cfg.HOVER_HEIGHT
    for i, a in enumerate(argv):
        if a == "--height" and i + 1 < len(argv):
            height = float(argv[i + 1])

    try:
        pv = importlib.import_module(f"{cfg.__package__}.piece_vision")    # the deployment's piece detector
    except ImportError:
        pv = None

    env = make_env()
    print(f"Connecting to {rc.FOLLOWER_ID} on {rc.FOLLOWER_PORT}...")
    env.connect()
    try:
        env.go_home()
        env.set_gripper(cfg.PREGRASP_OPEN_M, n_steps=10)
        target = np.array([PROBE_XY[0], PROBE_XY[1], cfg.TABLE_Z + height])
        print(f"Moving to {target.round(3).tolist()} ({height * 1000:.0f} mm above the table)...")
        if not env.move_ee_to(target, tol=0.008):
            print("  Could not reach it — measuring from wherever the arm stopped.")

        print("\nPut a piece under the jaws. Adjust the lighting and watch these.")
        print(f"Target: brightness {GOOD_BRIGHT[0]}-{GOOD_BRIGHT[1]}, "
              f"sharpness above {GOOD_SHARP:.2f}, detect = yes.")
        print("Ctrl-C when you are happy; the last frame is saved.\n")

        last = None
        while True:
            frame = env.render_image(rc.CAMERA_NAME, width=rc.RENDER_W, height=rc.RENDER_H)
            g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float64)
            bright = g.mean()
            sharp = cv2.Laplacian(g, cv2.CV_64F).var() / (g.var() + 1e-6)

            det = "n/a"
            if pv is not None:
                cands = pv.candidates(frame)
                if cands:
                    best = cands[0]
                    det = f"yes ({best[0]:.0f},{best[1]:.0f}) area={best[3]:.0f}"
                else:
                    det = "NO"

            flag_b = "ok " if GOOD_BRIGHT[0] <= bright <= GOOD_BRIGHT[1] else (
                "DIM" if bright < GOOD_BRIGHT[0] else "HOT")
            flag_s = "ok " if sharp >= GOOD_SHARP else "SOFT"
            print(f"\r  brightness {bright:5.1f} {flag_b}   "
                  f"sharpness {sharp:6.3f} {flag_s}   detect {det:28}",
                  end="", flush=True)
            last = frame
            time.sleep(0.4)
    except KeyboardInterrupt:
        print()
        if last is not None:
            out = HERE / "check_lighting.png"
            privacy.imwrite(str(out), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
            print(f"Saved {out}")
    finally:
        env.go_home()
        env.stop()
        print("Disconnected.")


if __name__ == "__main__":
    main()
