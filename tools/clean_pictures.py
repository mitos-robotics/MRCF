"""
clean_pictures.py — find the faces in the pictures already kept on this
computer (the flight recorder's runs/, the rigs' rig/, the tools' reports/,
a deployment's own logs), and cover them or delete those pictures.

Before 2026-11-17 nothing guarded what was kept (mrcf/privacy.py does now):
the SO-101's wrist camera kept the user's face in a failed pick's frames,
and tools/find_cameras.py's index sweep a picture from the Mac's own camera.
None of these are in git (.gitignore); they are on this computer only.

    python tools/clean_pictures.py              # list them; nothing changed
    python tools/clean_pictures.py --cover      # cover the faces, in place
    python tools/clean_pictures.py --delete     # delete those pictures

Needs the face model (tools/fetch_face_model.py). The model also mistakes a
few things for faces (a pawn's head close up, a cup): look at the list
before --cover or --delete. No camera, no arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrcf import privacy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WHERE = ("runs", "rig", "reports", "deployments", "tools")
SUFFIXES = (".jpg", ".jpeg", ".png")


def pictures(root: Path = ROOT) -> list:
    """Every picture kept under the folders that keep them."""
    found = []
    for name in WHERE:
        base = root / name
        if base.is_dir():
            found += [p for p in base.rglob("*") if p.suffix.lower() in SUFFIXES and p.is_file()]
    return sorted(found)


def with_faces(paths: list) -> list:
    """[(path, faces)] for the pictures with a face in them."""
    import cv2

    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        found = privacy.faces(img)
        if found:
            out.append((p, found))
    return out


def cover(path: Path) -> int:
    """Cover the faces in one picture, in place; how many."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    n = len(privacy.faces(img))
    if n:
        privacy.imwrite(str(path), img)               # covers them as it writes
    return n


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not privacy.has_face_model():
        print("The face model is not here: python tools/fetch_face_model.py")
        return 1
    paths = pictures()
    print(f"Looking for faces in {len(paths)} pictures under {', '.join(WHERE)}/ ...")
    hits = with_faces(paths)
    for p, found in hits:
        print(f"  {p.relative_to(ROOT)}  ({len(found)} face(s))")
    print(f"{len(hits)} picture(s) with a face in them.")
    if "--cover" in argv:
        n = sum(cover(p) for p, _ in hits)
        print(f"Covered {n} face(s) in {len(hits)} picture(s).")
    elif "--delete" in argv:
        for p, _ in hits:
            p.unlink()
        print(f"Deleted {len(hits)} picture(s).")
    elif hits:
        print("Nothing changed. Look at the list, then --cover (faces covered in place) or --delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
