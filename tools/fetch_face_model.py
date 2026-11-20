"""
fetch_face_model.py — fetch the face model every kept picture is checked
with (mrcf/privacy.py): OpenCV's YuNet, from OpenCV's own model zoo (MIT
licence, 232 KB), into MRCF_HOME/models/. Once per computer.

Until it is there, faces are not looked for in what the framework keeps — a
calibrated camera's frames still keep only the work area — and run.py says so
at the start. The file is checked against its SHA-256 before it is kept: a
download that does not match is thrown away.

    python tools/fetch_face_model.py              # download it
    python tools/fetch_face_model.py --from FILE  # a copy you already have
    python tools/fetch_face_model.py --check      # is it there, and the right file?

A download only: no camera, no arm.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrcf import privacy  # noqa: E402


def install(source: Path) -> Path:
    """Copy a model file into place after checking its SHA-256; its path. SystemExit when it does not match."""
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if digest != privacy.FACE_MODEL_SHA256:
        raise SystemExit(f"{source} is not the face model it should be (sha256 {digest}, "
                         f"expected {privacy.FACE_MODEL_SHA256}): not kept.")
    dest = privacy.face_model_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return dest


def download() -> Path:
    """The model from FACE_MODEL_URL, into a folder of its own, checked, then into place."""
    with tempfile.TemporaryDirectory(prefix="mrcf_face_model_") as tmp:
        part = Path(tmp) / privacy.FACE_MODEL_NAME
        print(f"Fetching {privacy.FACE_MODEL_URL} ...")
        with urllib.request.urlopen(privacy.FACE_MODEL_URL, timeout=60) as response, open(part, "wb") as out:
            shutil.copyfileobj(response, out)
        return install(part)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv:
        ok = privacy.face_model_ok()
        print(f"{privacy.face_model_path()}: {'there, and the right file' if ok else 'missing or not the right file'}")
        return 0 if ok else 1
    if "--from" in argv:
        i = argv.index("--from")
        if i + 1 >= len(argv):
            raise SystemExit("--from FILE")
        dest = install(Path(argv[i + 1]))
    else:
        if privacy.face_model_ok():
            print(f"Already there: {privacy.face_model_path()}")
            return 0
        dest = download()
    print(f"Kept {dest}: faces are covered in every picture the framework keeps from now on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
