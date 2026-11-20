"""
migrate_rig_state.py — move a rig's own files out of the code and into its
folder, rig/<backend>/ (rig_state.py): the overhead camera's calibration, the
photo of the empty surface, the URDF, and everything its tools saved beside
themselves (frames, touches, fits). Once, on a rig set up before 2026-10-29: a
backend refuses to start while what it reads as it starts is still in its
code folder.

Nothing connects and nothing moves but files. Without --apply it only lists
what it would move. With it, each file is copied, checked byte for byte
(sha256) and only then removed from where it was. A file already at its new
place is left where it is and reported, never overwritten; files git tracks
are not touched.

What counts as a rig's own: every file in a backend's folder that is not
code or its docs (*.py, __pycache__/, *.md: its NOTES.md) and not in git — mrcf/backends/<backend>/, and
backends/<backend>/ at the repository's root, where the backends were before
mrcf/ (git leaves behind there what it never tracked).
backends/<backend>/tools/x lands in rig/<backend>/x, beside the rest.

Usage:
    python tools/migrate_rig_state.py            # what would move
    python tools/migrate_rig_state.py --apply    # move it
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrcf import rig_state  # noqa: E402

SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def tracked(root: Path) -> set:
    """The files git tracks under root (absolute paths); empty when it is not a git checkout."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {(root / name).resolve() for name in out.decode().split("\0") if name}


def _backend_folders(root: Path) -> list:
    """Every backend's folder: mrcf/backends/<backend>/, and backends/<backend>/ from before mrcf/."""
    parents = [d for d in (root / "mrcf" / "backends", root / "backends") if d.is_dir()]
    return sorted(p for d in parents for p in d.iterdir() if p.is_dir())


def plan(root: Path, home: Path, in_git: set) -> list:
    """[(where it is, where it goes)] for every rig file in root's backends' folders."""
    moves = []
    for code in _backend_folders(root):
        if code.name.startswith((".", "_")) or (code.parent.parent == root and not code.name.isidentifier()):
            continue
        if code.parent.parent != root and not (code / "__init__.py").is_file():
            continue                                 # in mrcf/: a backend has its code; at the root, it is left over
        for path in sorted(p for p in code.rglob("*") if p.is_file()):
            rel = path.relative_to(code)
            if (path.suffix in (".py", ".pyc", ".md") or "__pycache__" in rel.parts or path.name in SKIP_NAMES
                    or path.resolve() in in_git):
                continue
            moves.append((path, home / "rig" / code.name / Path(*rel.parts[1:] if rel.parts[0] == "tools" else rel.parts)))
    return moves


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply(moves: list) -> "tuple[list, list]":
    """Move each (copy, check, remove); (moved, kept: already at their new place). Raises if a copy differs."""
    moved, kept = [], []
    for src, dst in moves:
        if dst.exists():
            kept.append((src, dst))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if _sha256(src) != _sha256(dst):
            dst.unlink()
            raise RuntimeError(f"{dst} came out different from {src}: nothing removed; {src} is where it was.")
        src.unlink()
        moved.append((src, dst))
    return moved, kept


def prune(root: Path) -> list:
    """
    The folders the move left empty in root's backends — or holding only what
    the OS leaves behind (.DS_Store) — removed, deepest first; their paths. A
    backend's folder in mrcf/ is never removed; one left over at the root from
    before mrcf/ is, once nothing is left in it, and then backends/ itself.
    """
    removed = []

    def remove_if_empty(folder: Path) -> None:
        entries = list(folder.iterdir())
        if all(e.is_file() and e.name in SKIP_NAMES for e in entries):
            for junk in entries:
                junk.unlink()
            folder.rmdir()
            removed.append(folder)

    for code in _backend_folders(root):
        for folder in sorted((p for p in code.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
            remove_if_empty(folder)
        if code.parent.parent == root:
            remove_if_empty(code)
    if (root / "backends").is_dir():
        remove_if_empty(root / "backends")
    return removed


def _size(paths) -> str:
    total = sum(p.stat().st_size for p in paths)
    return f"{total / 1e6:.1f} MB" if total >= 1e5 else f"{total / 1e3:.0f} kB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="move them (without it: only list them)")
    args = ap.parse_args()

    home = rig_state.home()
    moves = plan(ROOT, home, tracked(ROOT))
    if not moves:
        if args.apply:
            for folder in prune(ROOT):
                print(f"Removed the empty {folder.relative_to(ROOT)}")
        print("Nothing to move: no rig files left in the backends' folders.")
        return 0
    print(f"{len(moves)} rig file(s), {_size(s for s, _ in moves)}, from the code to {home / 'rig'}:")
    for src, dst in moves:
        print(f"  {src.relative_to(ROOT)}  ->  {dst.relative_to(home) if dst.is_relative_to(home) else dst}"
              + ("   (already there: kept where it is)" if dst.exists() else ""))
    if not args.apply:
        print("\nNothing moved. --apply moves them.")
        return 0
    moved, kept = apply(moves)
    prune(ROOT)
    print(f"\nMoved {len(moved)}, each checked byte for byte.")
    if kept:
        print(f"Kept {len(kept)} where it was: a file of that name is already in the rig's folder. Compare them, "
              "keep the one you want there, and delete the other.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
