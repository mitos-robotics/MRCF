"""
privacy.py — no pictures of people kept, nothing kept that the work does not
need, and no camera that is not the rig's opened.

The arm works at a table, and people stand at tables: their hands place
things, they lean in, and a camera that moves with the gripper, or one low at
the table's edge, sees the room beyond it. 2026-11-17: the SO-101's wrist
camera, its gripper raised, kept the user's face in a failed pick's incident
frames. User: "make sure there are no pictures of people or anything that
isn't essential etc. Especially when testing cameras".

Two guards, on every picture the framework and its tools keep on disk (the
flight recorder's frames, pictures and model calls' pictures; the
calibrations' and checks' frames: imwrite, save_rgb):

  the work area  A calibrated camera's frame keeps only what lies toward the
                 arm's work volume (the safety layer's WorkspaceBounds,
                 widened by WORK_AREA_MARGIN_M): every pixel whose line of
                 sight misses that box — the room, the ceiling, whoever stands
                 at the table — is blacked out (work_area_only). The wrist
                 camera's from where it is when the frame is taken.
  faces          Any face found is covered with a solid block (cover_people):
                 OpenCV's YuNet face model (opencv_zoo, MIT licence, 232 KB),
                 fetched once by tools/fetch_face_model.py into
                 MRCF_HOME/models/; nothing is sent anywhere. Without it,
                 faces are not looked for (has_face_model(); run.py says so).
                 OpenCV's older Haar models were tried on 18,900 of the rigs'
                 own pictures and rejected: on the wrist camera 91 of 93 of
                 their "faces" were the very figures and pieces being picked.

Cameras: a computer has cameras of its own — the laptop's, an iPhone's
Continuity Camera, a Desk View — that look at the person using it. The
framework's tools never open one (personal()); tools/find_cameras.py lists
cameras by name without opening any. 2026-11-14 an index sweep opened the
iPhone's camera; 2026-11-17 a test opened the Mac's own.

MRCF_KEEP_PICTURES=0 keeps none of the pictures kept only for reading later
(the flight recorder's keep_picture); an incident's frames, what a failure is
read from, are kept either way, guarded.

This module imports only the standard library, numpy and (when present)
OpenCV, so the flight recorder can use it.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path

import numpy as np

# ── Faces ─────────────────────────────────────────────────────────────────────
FACE_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
FACE_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
                  "face_detection_yunet_2023mar.onnx")
FACE_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
# How sure YuNet must be. 2026-11-17 on the rigs' own pictures: real faces
# 0.82-0.92, its mistakes (a pawn's head close up, a cup) 0.81-0.84 and rare
# (11 of 7,020 wrist frames, 2 of 2,000 overhead); a mistake costs a patch of
# a kept picture, a miss a face kept, so the lower.
FACE_SCORE = 0.8
# A face's box grown by this much each way before it is covered: hair, ears, chin.
GROW = 0.45
COVER_RGB = (40, 40, 40)

# ── The work area ─────────────────────────────────────────────────────────────
# The work volume widened by this much each way, sideways and up: an object
# standing at its edge, the arm above it, a hand reaching in all stay.
WORK_AREA_MARGIN_M = 0.10
WORK_AREA_STEP_PX = 4            # the mask worked out on a grid this coarse, then scaled up

# Words in a camera's device name that say it looks at the person using the
# computer, not at the rig.
PERSONAL_CAMERA_WORDS = ("facetime", "iphone", "ipad", "continuity", "desk view", "macbook", "built-in",
                         "integrated camera", "integrated webcam", "obs virtual", "virtual camera", "snap camera",
                         "capture screen")

_LOCK = threading.Lock()
_DETECTOR: list = []             # [cv2.FaceDetectorYN] once loaded; [None] when it cannot be


def _home() -> Path:
    """MRCF_HOME, else the repository mrcf/ is in, else ~/.mrcf (as rig_state.home; not imported here)."""
    if os.environ.get("MRCF_HOME"):
        return Path(os.environ["MRCF_HOME"])
    repo = Path(__file__).resolve().parent.parent
    return repo if (repo / "pyproject.toml").is_file() else Path.home() / ".mrcf"


def face_model_path() -> Path:
    """Where the face model is kept: MRCF_HOME/models/ (tools/fetch_face_model.py puts it there)."""
    return _home() / "models" / FACE_MODEL_NAME


def face_model_ok(path: "Path | None" = None) -> bool:
    """Whether the face model is there and is the file it should be (its SHA-256)."""
    path = Path(path) if path is not None else face_model_path()
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == FACE_MODEL_SHA256
    except OSError:
        return False


def has_face_model() -> bool:
    """Whether faces are looked for: OpenCV with YuNet, and the model fetched."""
    return _detector() is not None


def _detector():
    with _LOCK:
        if not _DETECTOR:
            made = None
            try:
                import cv2

                if face_model_ok() and hasattr(cv2, "FaceDetectorYN"):
                    made = cv2.FaceDetectorYN.create(str(face_model_path()), "", (320, 320), FACE_SCORE, 0.3, 5000)
            except Exception as exc:          # noqa: BLE001 — a model that will not load: faces not looked for
                print(f"  [Privacy] The face model did not load ({exc}): faces are not looked for.", file=sys.stderr)
            _DETECTOR.append(made)
        return _DETECTOR[0]


def personal(device_name: "str | None") -> bool:
    """Whether a camera, by its device name, is one that looks at its user (never opened by the tools)."""
    name = (device_name or "").lower()
    return any(word in name for word in PERSONAL_CAMERA_WORDS)


def keep_pictures() -> bool:
    """Whether pictures kept only for reading later are kept at all (MRCF_KEEP_PICTURES: on unless 0/off/no)."""
    return os.environ.get("MRCF_KEEP_PICTURES", "1").strip().lower() not in ("0", "off", "no", "false")


def faces(image: np.ndarray) -> list:
    """The faces in a picture (3-channel, either colour order): [(x, y, w, h)] in its pixels; [] when not looked for."""
    det = _detector()
    if det is None or image is None or image.ndim != 3 or min(image.shape[:2]) < 32:
        return []
    import cv2

    h, w = image.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    with _LOCK:                                                        # one detector, any thread
        det.setInputSize((small.shape[1], small.shape[0]))
        _, found = det.detect(np.ascontiguousarray(small[..., :3]))
    if found is None:
        return []
    return [tuple(int(round(v / scale)) for v in row[:4]) for row in found]


def cover_people(image: np.ndarray) -> "tuple[np.ndarray, int]":
    """The picture with every face found covered (a copy; the same array when none is found), and how many."""
    found = faces(image)
    if not found:
        return image, 0
    out = np.array(image, copy=True)
    h, w = out.shape[:2]
    for (x, y, fw, fh) in found:
        gx, gy = int(fw * GROW), int(fh * GROW)
        x0, y0, x1, y1 = max(0, x - gx), max(0, y - gy), min(w, x + fw + gx), min(h, y + fh + int(gy * 1.6))
        out[y0:y1, x0:x1] = np.array(COVER_RGB, out.dtype)[: out.shape[2]]
    return out, len(found)


# ── The work area ─────────────────────────────────────────────────────────────

def work_area_mask(model: dict, width: int, height: int, bounds, margin_m: float = WORK_AREA_MARGIN_M,
                   step: int = WORK_AREA_STEP_PX) -> "np.ndarray | None":
    """
    (height, width) bool: the pixels whose line of sight passes through the work
    volume `bounds` (x/y/z_min/max, metres, the camera model's frame) widened
    by margin_m — for a camera model {K, T_world_cam} (RobotEnv.camera_model)
    at width x height. None when either is missing or the bounds are open.
    """
    try:
        K = np.asarray(model["K"], float)
        T = np.asarray(model["T_world_cam"], float)
        lo = np.array([bounds.x_min, bounds.y_min, bounds.z_min], float) - margin_m
        hi = np.array([bounds.x_max, bounds.y_max, bounds.z_max], float) + margin_m
    except (KeyError, TypeError, AttributeError, ValueError):
        return None
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) or K.shape != (3, 3) or T.shape != (4, 4):
        return None
    us, vs = np.meshgrid(np.arange(0, width, step) + step / 2.0, np.arange(0, height, step) + step / 2.0)
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], axis=-1) @ T[:3, :3].T
    origin = T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lo - origin) / rays
        t1 = (hi - origin) / rays
    near = np.nanmax(np.minimum(t0, t1), axis=-1)               # the slabs: where the ray is inside all three
    far = np.nanmin(np.maximum(t0, t1), axis=-1)
    inside_now = np.all((origin >= lo) & (origin <= hi))
    small = (far >= np.maximum(near, 0.0)) | inside_now
    full = np.repeat(np.repeat(small, step, axis=0), step, axis=1)[:height, :width]
    return full


def work_area_only(image: np.ndarray, model: "dict | None", bounds) -> np.ndarray:
    """image with everything outside the work area (work_area_mask) blacked out; as it is when that cannot be said."""
    if model is None or image is None or image.ndim < 2:
        return image
    h, w = image.shape[:2]
    mask = work_area_mask(model, w, h, bounds)
    if mask is None or mask.all():
        return image
    out = np.array(image, copy=True)
    out[~mask] = 0
    return out


def frame_guard(env, bounds):
    """
    For the flight recorder's camera frames (FlightRecorder.watch_cameras(keep=)):
    a function (camera, frame) -> the frame with only the work area kept
    (each camera's model as env gives it at that moment), then faces covered.
    bounds: the work volume (safety's WorkspaceBounds); open bounds keep the
    whole frame, faces still covered.
    """
    def keep(camera: str, frame: np.ndarray) -> np.ndarray:
        try:
            h, w = frame.shape[:2]
            model = env.camera_model(camera, w, h)
        except Exception:                     # noqa: BLE001 — no model: faces only
            model = None
        return cover_people(work_area_only(frame, model, bounds))[0]
    return keep


# ── Writing ───────────────────────────────────────────────────────────────────

def _said(path, n: int) -> None:
    if n:
        print(f"  [Privacy] {n} face(s) covered in {Path(str(path)).name}", file=sys.stderr)


def imwrite(path, image_bgr: np.ndarray, params=None) -> bool:
    """cv2.imwrite with every face covered first (the colour order does not matter to the finding)."""
    import cv2

    covered, n = cover_people(image_bgr) if image_bgr is not None and image_bgr.ndim == 3 else (image_bgr, 0)
    _said(path, n)
    return bool(cv2.imwrite(str(path), covered, params or []))


def save_rgb(path, image_rgb: np.ndarray) -> bool:
    """An RGB frame written to path (its suffix says the format), every face covered first."""
    covered, n = cover_people(image_rgb) if image_rgb is not None and image_rgb.ndim == 3 else (image_rgb, 0)
    _said(path, n)
    try:
        import cv2

        rgb = np.ascontiguousarray(covered)
        return bool(cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb))
    except ImportError:
        from PIL import Image

        Image.fromarray(np.ascontiguousarray(covered)).save(str(path))
        return True


def encoded(data: bytes) -> bytes:
    """A JPEG or PNG (bytes) with every face covered: the same bytes when there is none, or it cannot be read."""
    try:
        import cv2

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return data
        covered, n = cover_people(img)
        if not n:
            return data
        ok, buf = cv2.imencode(".jpg" if data[:2] == b"\xff\xd8" else ".png", covered)
        return buf.tobytes() if ok else data
    except Exception:                         # noqa: BLE001 — never into the caller
        return data
