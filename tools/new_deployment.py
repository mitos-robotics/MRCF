"""
new_deployment.py — start a new deployment from one that exists, taking over
only what fits it.

A deployment is one arm, one layout and one task (DEPLOYMENT_SPEC.md, "A new
deployment"): other objects, another surface, another goal — or the same task
on another arm — is a deployment of its own. It need not start from nothing:
the source's code is mostly the framework's patterns wired up (a table task's
is a config handed to mrcf/tabletop), and is copied, its arm set to --arm.
Its RULES are not: how the source drives its arm (deployments/<source>/
profile.py — how wide to open, how hard to hold, what a look that disagrees
means) were learned on its own objects and its own arm, and are written into
the new profile as candidates only — commented out, each with its reason and
what the new deployment's arm does without it — to be taken over one by one
where the reason holds for the new work (mrcf/profiles.py). 2026-11-17: the
SO-101's table work inherited ~30 rules tuned on chess pieces, and each one
surfaced as a failure on a scoop, a plug and a sweet.

What it does not copy: the source's tests, its history and notes (the new
deployment starts its own), its kept state (object memory, marked places, a
measured surface: rig/<arm>/<source>_*.json — named after the new deployment
in the copy; copy a file over by hand where the new one shares it, as one on
the same arm and table shares its measured surface).

Usage:
    python tools/new_deployment.py NAME --from SOURCE [--arm ARM]   # ARM: the source's, unless given
    python tools/new_deployment.py --list SOURCE [--arm ARM]        # the source's rules, and why
    --root DIR    where deployments live (default: the repository's deployments/)

Writes files only; connects to nothing, moves nothing.
"""

from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrcf import profiles  # noqa: E402

# Copied from the source: its code. Not: tests, history, notes, its profile, kept state, caches.
SKIP = {"tests", "__pycache__", "logs", "HISTORY.md", "NOTES.md", profiles.PROFILE_FILE}
_ARM_NAME = re.compile(r'^ARM_NAME\s*=\s*["\']([^"\']+)["\']', re.M)
SKIP_SUFFIXES = (".json", ".npy", ".png", ".jpg", ".bak", ".pyc")


def arm_of(deployment_dir: Path) -> "str | None":
    """The arm a deployment names (ARM_NAME in its config.py), read from the file; None: none."""
    try:
        found = _ARM_NAME.search((Path(deployment_dir) / "config.py").read_text())
    except OSError:
        return None
    return found.group(1) if found else None


def rules(source_dir: Path) -> list:
    """
    The rules the source's profile sets: [(NAME, value as written, its comment
    lines)], in the file's order.
    """
    where = profiles.path(source_dir)
    if not where.is_file():
        return []
    text = where.read_text()
    lines = text.splitlines()
    out = []
    for node in ast.parse(text).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not (isinstance(target, ast.Name) and target.id.isupper()):
            continue
        start = node.lineno - 1
        comment = []
        k = start - 1
        while k >= 0 and lines[k].startswith("#") and "──" not in lines[k]:
            comment.insert(0, lines[k])
            k -= 1
        written = "\n".join(lines[start:node.end_lineno])
        out.append((target.id, written, comment))
    return out


def framework_default(name: str, arm: str) -> str:
    """What `arm` does without the rule: its own config's value, else the policy spec's default, said."""
    import dataclasses
    import importlib

    from mrcf.backends import wiring
    from mrcf.policies.parallel_jaw.spec import GripperSpec

    config = importlib.import_module(f"mrcf.backends.{arm}.robot_config")
    if hasattr(config, name):
        return f"the arm's own: {getattr(config, name)!r}"
    field = {v: k for k, v in wiring.SPEC_SETTING_FOR.items()}.get(name)
    if field is not None:
        default = {f.name: f.default for f in dataclasses.fields(GripperSpec)}.get(field, dataclasses.MISSING)
        if default is not dataclasses.MISSING:
            return f"the framework's: {default!r}"
    return "not set (the code's own default)"


def candidates(source: str, source_dir: Path, arm: str) -> str:
    """The new profile's body: every rule of the source's, commented out, with why and what `arm` does without it."""
    out = []
    for name, written, comment in rules(source_dir):
        out.append("")
        out.extend(comment)                                   # why the source has it, as it says it
        out.append(f"#     without it: {framework_default(name, arm)}")
        out.extend(f"# {line}" for line in written.splitlines())
    if not out:
        return f"\n# ({source} has no profile: nothing to take over.)\n"
    return "\n".join(out) + "\n"


def profile_text(name: str, source: str, source_dir: Path, arm: str) -> str:
    source_arm = arm_of(source_dir)
    on = f" on the {source_arm}" if source_arm and source_arm != arm else ""
    return (f'''"""
{name}'s own rules for its arm, the {arm} — how {name} drives it, laid over the
arm's facts (mrcf/backends/{arm}/robot_config.py) wherever {name} makes the
arm's env and policies (mrcf/profiles.py).

Started from {source} (tools/new_deployment.py): its rules{on} are below as
candidates, commented out, each with why {source} has it and what the {arm}
does without it. Take one over — uncomment it, and say why it holds here —
only where its reason holds for this deployment's arm, objects and surface
too. Leave the rest: what this deployment needs it learns on its own work,
and writes here.
"""
''' + "\n# ── Candidates from " + source + " (not in force) " + "─" * max(4, 46 - len(source)) + "\n"
            + candidates(source, source_dir, arm))


def notes_text(name: str, source: str) -> str:
    return f"""# {name}: notes

What running this task has shown (DEPLOYMENT_SPEC.md, "What you learn, and
where it goes"). Started from {source} (tools/new_deployment.py) on its code;
its rules are this deployment's own (profile.py).

## Limits

-

## Seen
"""


def own_tasks(config_text: str, source: str) -> str:
    """
    A copied config's tasks (TASKS, PLANNER: mrcf/tabletop/settings.py) kept
    for reference only, commented out, and the new deployment's own started
    empty, the planner on until it has them: the source's tasks name the
    source's objects, and with none of its own and no planner a deployment
    could do nothing.
    """
    tree = ast.parse(config_text)
    spans = [(n.lineno, n.end_lineno) for n in tree.body
             if isinstance(n, (ast.Assign, ast.AnnAssign))
             and any(isinstance(t, ast.Name) and t.id in ("TASKS", "PLANNER")
                     for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    if not spans:
        return config_text
    lines = config_text.splitlines()
    for start, end in spans:
        for i in range(start - 1, end or start):
            lines[i] = "# " + lines[i]
    at = max(end or start for start, end in spans)
    lines[at:at] = [f"# {source}'s tasks, above, for reference. This deployment's own (mrcf/tabletop/settings.py",
                    "# TASKS); PLANNER False once they are written, and nothing else is planned:",
                    "TASKS: dict = {}", "PLANNER = True"]
    return "\n".join(lines) + ("\n" if config_text.endswith("\n") else "")


def create(name: str, source: str, root: Path, arm: "str | None" = None) -> Path:
    """
    deployments/<name>/ from deployments/<source>/, on `arm` (the source's when
    None): see the module docstring. Refuses to overwrite.
    """
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise SystemExit(f"'{name}': name a deployment as a Python module (lower-case letters, digits, underscores)")
    src, dst = root / source, root / name
    if not (src / "config.py").is_file():
        raise SystemExit(f"No deployment '{source}' in {root} (no {source}/config.py)")
    if dst.exists():
        raise SystemExit(f"{dst} exists: not overwriting it")
    arm = arm or arm_of(src)
    if arm is None:
        raise SystemExit(f"{source} names no arm (ARM_NAME in its config.py): say which with --arm")
    dst.mkdir(parents=True)
    copied = []
    for item in sorted(src.iterdir()):
        if item.name in SKIP or item.name.startswith(".") or item.suffix in SKIP_SUFFIXES or ".bak" in item.name:
            continue
        if item.is_dir():
            continue
        text = item.read_text() if item.suffix in (".py", ".md", ".txt") else None
        if text is None:
            shutil.copy2(item, dst / item.name)
        else:
            # Its kept state named after it: rig/<arm>/<source>_memory.json and the like.
            text = re.sub(rf'(["\']){re.escape(source)}_', rf"\g<1>{name}_", text)
            if item.name == "config.py":
                text = _ARM_NAME.sub(f'ARM_NAME = "{arm}"', text, count=1)      # one arm: the new deployment's
                text = own_tasks(text, source)
            (dst / item.name).write_text(text)
        copied.append(item.name)
    (dst / "NOTES.md").write_text(notes_text(name, source))
    profiles.path(dst).write_text(profile_text(name, source, src, arm))
    return dst


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    def take(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            value = argv[i + 1] if i + 1 < len(argv) else None
            del argv[i:i + 2]
            return value
        return default

    root = Path(take("--root", str(ROOT / "deployments")))
    arm = take("--arm")
    listed = take("--list")
    if listed is not None:
        found = rules(root / listed)
        own = arm_of(root / listed)
        arm = arm or own or "so101"
        if not found:
            print(f"{listed} has no profile.")
            return
        print(f"{listed}'s rules for its arm, the {own} (deployments/{listed}/profile.py); "
              f"'without it' is the {arm}'s:")
        for name, written, comment in found:
            print(f"\n{written}")
            for line in comment:
                print(f"    {line.lstrip('# ').rstrip()}")
            print(f"    without it: {framework_default(name, arm)}")
        return
    source = take("--from")
    if not argv or source is None:
        raise SystemExit(__doc__.split("Usage:")[1].split("Writes files")[0])
    dst = create(argv[0], source, root, arm)
    print(f"Made {dst}, from {source}'s code, on the {arm_of(dst)}. Its rules are candidates in "
          f"{profiles.path(dst)}, none in force: take over only those whose reasons hold here.")


if __name__ == "__main__":
    main()
