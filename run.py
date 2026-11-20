"""
run.py — MRCF launcher: which arm, what it has, then which deployment.

As a factory sets up a new automation: the arm, then its sensors, then the
job. Each step can start something new: "n" at the arm's question copies a
new arm's backend from mrcf/backends/_template (tools/new_arm.py); "n" at the
deployment's starts a new deployment on the chosen arm from the nearest one
(tools/new_deployment.py: its code, the source's rules as candidates only).

First it asks which arm to drive. Each one is listed with whether it looks
plugged in (its backend's present(): a check, nothing connected to), and
Enter takes the one that is. That sets MRCF_ARM for everything this process
runs, before anything has read it, so a MRCF_ARM left exported in the
terminal does not pick the arm unseen. --arm NAME chooses without asking.

Then it scans the deployments/ folder for that arm's deployments, validates
each, and asks which to run; Enter takes the arm's own (its config's
DEPLOYMENT: chess on the SO-101, tabletop on the UR5). A deployment is one
arm, one layout and one task, and names its arm (ARM_NAME in its config.py):
another arm's deployments are listed as skipped, with the arm to choose for
them. --deployment NAME chooses without asking, and takes its arm with it
when --arm is not given. Only deployments that pass validation are shown as
options; skipped deployments print the reason they were excluded.

Between the two it shows what the arm has (mrcf/rig_report.py), opening and
connecting nothing: whether it is plugged in, its gripper, each camera (its
source, whether it is calibrated, whether the rig's cameras are plugged in —
never a camera of this computer's own), its other sensors, and whether the
pictures it keeps are guarded (faces covered: mrcf/privacy.py; it offers to
fetch the face model when it is missing).

The run is recorded from the moment a deployment is chosen (the flight
recorder, runs/flight/<time>/): session.json says which arm and deployment
with what (the report above, this code's version), console.log keeps what was
printed, and a crash anywhere is an incident with its traceback.

Each deployment is a package, deployments/<name>/. Validating one imports
its config as deployments.<name>.config; launching it imports
deployments.<name>.llm_controller and calls its main() in this process —
what `python -m deployments.<name>.llm_controller` does. Arguments run.py
does not take itself are passed on to the controller in sys.argv.

Usage (from the repository root):
    python run.py
    python run.py --arm so101
    python run.py --arm ur5 --deployment tabletop
    python run.py --deployment tidy          # its arm, the SO-101, with it
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

# ── Constants ─────────────────────────────────────────────────────────────────

FRAMEWORK_ROOT  = Path(__file__).resolve().parent
DEPLOYMENTS_DIR = FRAMEWORK_ROOT / "deployments"
REQUIRED_FILES  = ["__init__.py", "llm_controller.py", "config.py", "skills.py"]


# ── Arm selection ─────────────────────────────────────────────────────────────

@dataclass
class _Arm:
    name: str                 # its folder (or plugin) name: what MRCF_ARM takes
    found: "bool | None"      # present()'s answer; None: the arm cannot say
    how: str                  # what was found, or why not
    loads: bool = True        # False: its package fails to import here
    printed: str = ""         # what importing and checking it printed, shown once it is chosen


def _survey() -> "list[_Arm]":
    """
    Every arm there is, each asked whether it is plugged in. What an arm's
    import prints (the SO-101's calibration notes) is kept back, and shown
    only for the arm chosen — not an arm left unplugged.
    """
    from mrcf import backends
    arms = []
    for name in backends.available():
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                pkg = backends.arm(name)
            except Exception as exc:          # noqa: BLE001 — listed, not chosen
                arms.append(_Arm(name, False, f"does not load here — {type(exc).__name__}: {exc}", loads=False))
                continue
            label = getattr(pkg, "NAME", name)
            check = getattr(pkg, "present", None)
            try:
                found, how = check() if callable(check) else (None, "cannot tell whether it is plugged in")
            except Exception as exc:          # noqa: BLE001
                found, how = None, f"its check failed — {type(exc).__name__}: {exc}"
        arms.append(_Arm(name, found, f"{label}: {how}", printed=out.getvalue()))
    return arms


def _default_arm(arms: "list[_Arm]") -> "str | None":
    """The one arm plugged in; else MRCF_ARM's; else backends.DEFAULT — among those that load."""
    from mrcf import backends
    loads = [a.name for a in arms if a.loads]
    found = [a.name for a in arms if a.loads and a.found]
    if len(found) == 1:
        return found[0]
    for name in (os.environ.get(backends.ENV_VAR, "").strip(), backends.DEFAULT):
        if name in loads:
            return name
    return loads[0] if loads else None


def _choose_arm(forced: "str | None") -> str:
    """
    The arm to drive, set as MRCF_ARM: --arm's, else asked — Enter for
    _default_arm. Returns its name.
    """
    from mrcf import backends
    arms = _survey()
    by_name = {a.name: a for a in arms}
    if forced is not None:
        if forced not in by_name:
            print(f"No arm '{forced}'. Arms: {', '.join(by_name) or '(none)'}")
            sys.exit(1)
        chosen = by_name[forced]
        if not chosen.loads:
            print(f"Arm '{forced}' {chosen.how}")
            sys.exit(1)
    else:
        default = _default_arm(arms)
        if default is None:
            print("No arm loads here:")
            for a in arms:
                print(f"  {a.name}: {a.how}")
            sys.exit(1)
        print("Which arm?")
        for i, a in enumerate(arms, 1):
            mark = "plugged in" if a.found else ("?" if a.found is None else "not found")
            print(f"  [{i}] {a.name:<8} {mark:<11} {a.how}")
        print("  [n] new      set up another arm (a copy of mrcf/backends/_template, in memory until its SDK is in)")
        leftover = os.environ.get(backends.ENV_VAR, "").strip()
        if leftover and leftover != default:
            why = ", the one plugged in" if by_name[default].found else ""
            print(f"  ({backends.ENV_VAR}={leftover} is set in this terminal; Enter takes {default}{why}. "
                  f"`unset {backends.ENV_VAR}` clears it.)")
        try:
            raw = input(f"\nArm [1-{len(arms)}, n, or Enter for {default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if raw.lower() in ("n", "new"):
            return _choose_arm(_new_arm())
        if not raw:
            chosen = by_name[default]
        elif raw.isdigit() and 1 <= int(raw) <= len(arms):
            chosen = arms[int(raw) - 1]
        elif raw in by_name:
            chosen = by_name[raw]
        else:
            print(f"Invalid selection '{raw}'.")
            sys.exit(1)
        if not chosen.loads:
            print(f"Arm '{chosen.name}' {chosen.how}")
            sys.exit(1)
    os.environ[backends.ENV_VAR] = chosen.name
    print(chosen.printed, end="")
    if chosen.found is False:
        print(f"Driving {chosen.name}, though it was not found ({chosen.how}).")
    else:
        print(f"Driving {chosen.name}.")
    return chosen.name


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(folder: Path) -> tuple[bool, str, ModuleType | None]:
    """
    Lightweight validation of a deployment folder.

    Returns (ok, reason, config_module):
      ok=True   — deployment is ready to run; config_module is the loaded config.
      ok=False  — reason explains why it was skipped; config_module is None.
    """
    # 1. Required files must all exist.
    for fname in REQUIRED_FILES:
        if not (folder / fname).exists():
            return False, f"missing required file: {fname}", None

    # 2. llm_controller.py must not be a stub, and must have the main() run.py calls.
    src = (folder / "llm_controller.py").read_text(encoding="utf-8")
    if "raise NotImplementedError" in src:
        return False, "controller loop not yet implemented (stub — complete llm_controller.py)", None
    if "def main(" not in src:
        return False, "llm_controller.py has no main() for run.py to call", None

    # 3. The folder must be importable as a package: deployments.<name>.
    if not folder.name.isidentifier():
        return False, (f"'{folder.name}' cannot be imported as deployments.{folder.name} — "
                       "name the folder with letters, digits and underscores"), None

    # 4. config.py must import cleanly (catches missing packages / syntax errors).
    try:
        cfg = importlib.import_module(f"deployments.{folder.name}.config")
    except Exception as exc:
        return False, f"config.py failed to load: {exc}", None

    return True, "", cfg


# ── Discovery ─────────────────────────────────────────────────────────────────

def _deployment_arm(folder: Path) -> "str | None":
    """
    The arm a deployment names (ARM_NAME = "<arm>" in its config.py), read from
    the file, not imported: an arm's import is left until it is chosen. None:
    it names none, or has no config.py.
    """
    try:
        tree = ast.parse((folder / "config.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ARM_NAME" for t in node.targets)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            return node.value.value
    return None


def _discover(arm: "str | None" = None) -> tuple[list[tuple[Path, ModuleType]], list[tuple[Path, str]]]:
    """
    Scan DEPLOYMENTS_DIR for the deployments of `arm` (MRCF_ARM's when None) and return:
      valid    — list of (folder, config_module) for ready deployments
      skipped  — list of (folder, reason) for deployments that failed
                 validation, name no arm, or are another arm's
    """
    from mrcf import backends
    arm = arm or backends.arm_name()
    valid:   list[tuple[Path, ModuleType]]  = []
    skipped: list[tuple[Path, str]]         = []

    candidates = sorted(
        d for d in DEPLOYMENTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    )

    for folder in candidates:
        its_arm = _deployment_arm(folder)
        if its_arm is None and (folder / "config.py").exists():
            skipped.append((folder, "names no arm (ARM_NAME in config.py): a deployment is one arm, "
                                    "one layout and one task (DEPLOYMENT_SPEC.md)"))
            continue
        if its_arm is not None and its_arm != arm:
            skipped.append((folder, f"the {its_arm}'s: python run.py --arm {its_arm}"))
            continue
        ok, reason, cfg = _validate(folder)
        if ok:
            valid.append((folder, cfg))  # type: ignore[arg-type]
        else:
            skipped.append((folder, reason))

    return valid, skipped


# ── Deployment selection ──────────────────────────────────────────────────────

def _arms_own_task() -> "str | None":
    """MRCF_DEPLOYMENT, else the task the arm's config names as its own (DEPLOYMENT: chess on the SO-101)."""
    named = os.environ.get("MRCF_DEPLOYMENT", "").strip()
    if named:
        return named
    try:
        from mrcf import backends
        return getattr(backends.arm().config, "DEPLOYMENT", None)
    except Exception:                     # noqa: BLE001 — no arm to ask: choose by hand
        return None


def _pick_deployment(valid: list[tuple[Path, ModuleType]], arm: "str | None" = None) -> tuple[Path, ModuleType]:
    """
    The deployment to run, asked: Enter takes the arm's own (_arms_own_task),
    or the only one there is; "n" starts a new one on the arm (_new_deployment).
    Nothing is run unasked: an arm can do more than its own task. Returns
    (folder, config_module).
    """
    own = _arms_own_task()
    default = next((i for i, (d, _) in enumerate(valid) if d.name == own), 0 if len(valid) == 1 else None)
    why = ("MRCF_DEPLOYMENT" if os.environ.get("MRCF_DEPLOYMENT", "").strip() else
           "this arm's own" if default is not None and valid[default][0].name == own else "the only one")

    print("Which deployment (one arm, one layout, one task)?" if valid else "No deployment on this arm yet.")
    for i, (d, cfg) in enumerate(valid, 1):
        print(f"  [{i}] {d.name:<10} {_described(cfg)}" + (f"  ({why})" if i - 1 == default else ""))
    if arm is not None:
        print(f"  [n] new        a new deployment on the {arm}: its own layout and job, started from the nearest")

    choices = (f"1-{len(valid)}" if valid else "") + (", n" if arm is not None else "") + (
        f", or Enter for {valid[default][0].name}" if default is not None else "")
    try:
        raw = input(f"\nDeployment [{choices.lstrip(', ')}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if arm is not None and raw.lower() in ("n", "new"):
        return _new_deployment(arm)
    if not raw and default is not None:
        return valid[default]
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(valid):
            return valid[idx]

    name_map = {d.name: (d, cfg) for d, cfg in valid}
    if raw in name_map:
        return name_map[raw]

    print(f"Invalid selection '{raw}'.")
    sys.exit(1)


def _described(cfg) -> str:
    """A deployment in a line (rig_report.deployment_line); '' when it cannot be said."""
    if cfg is None:
        return ""
    try:
        from mrcf import rig_report
        return rig_report.deployment_line(rig_report.deployment(cfg))
    except Exception:                     # noqa: BLE001 — a line, not a reason to stop
        return ""


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _new_arm() -> str:
    """A new arm's backend, from mrcf/backends/_template (tools/new_arm.py); its name."""
    from tools import new_arm

    name = _ask("New arm's name, as a Python module (e.g. ur10_cell2): ").lower()
    label = _ask(f"What to call it in messages [Enter: {name.replace('_', ' ').title()}]: ") or None
    try:
        new_arm.create(name, label)
    except SystemExit as exc:
        print(exc)
        sys.exit(1)
    print(new_arm.next_steps(name))
    print()
    return name


def _new_deployment(arm: str) -> tuple[Path, ModuleType]:
    """
    A new deployment on `arm`, started from the nearest one (tools/new_deployment.py):
    its code copied, its ARM_NAME the arm's, the source's rules as candidates
    only. Run now, or left to set up first (the default).
    """
    from tools import new_deployment

    sources = sorted(d.name for d in DEPLOYMENTS_DIR.iterdir()
                     if (d / "config.py").is_file() and not d.name.startswith(("_", ".")))
    print("Start it from which deployment (its code; its rules offered as candidates, none in force)?")
    for i, name in enumerate(sources, 1):
        print(f"  [{i}] {name:<10} on the {_deployment_arm(DEPLOYMENTS_DIR / name)}")
    raw = _ask(f"From [1-{len(sources)}]: ")
    source = sources[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(sources) else raw
    if source not in sources:
        print(f"No deployment '{source}'.")
        sys.exit(1)
    name = _ask("Its name, as a Python module (e.g. sort_parts): ").lower()
    try:
        made = new_deployment.create(name, source, DEPLOYMENTS_DIR, arm)
    except SystemExit as exc:
        print(exc)
        sys.exit(1)
    print(f"\nMade {made}, on the {arm}, from {source}'s code. Next, before it runs for real:\n"
          f"  - its config.py: what is this deployment's own (objects, places, what to tell the models)\n"
          f"  - its profile.py: {source}'s rules are there as candidates, none in force — take over only those\n"
          f"    whose reasons hold for this job\n"
          f"  - its layout: the surface it works on is touched at its first start; marked places as you name them")
    if _ask("Run it now? [y/N]: ").lower() not in ("y", "yes"):
        print(f"Not run. Later: python run.py --deployment {name}")
        sys.exit(0)
    ok, why, cfg = _validate(made)
    if not ok:
        print(f"It does not load yet: {why}")
        sys.exit(1)
    return made, cfg  # type: ignore[return-value]


def _show_rig(arm: str, ask: bool = True) -> dict:
    """
    What the arm has (mrcf/rig_report.py), printed — nothing opened or
    connected; and, asked, the face model fetched when it is missing. The report.
    """
    from mrcf import backends, privacy, rig_report

    rep = rig_report.report(backends.arm(arm))
    print()
    print("\n".join(rig_report.lines(rep)))
    if ask and not privacy.has_face_model() and sys.stdin.isatty():
        if _ask("Faces are not covered in what is kept. Fetch the face model now "
                "(232 KB, OpenCV's model zoo)? [Y/n]: ").lower() in ("", "y", "yes"):
            from tools import fetch_face_model
            try:
                fetch_face_model.download()
                privacy._DETECTOR.clear()
                rep["privacy"]["faces_covered"] = privacy.has_face_model()
                print("  Faces are covered from now on.")
            except Exception as exc:          # noqa: BLE001 — said, and the run goes on
                print(f"  Not fetched ({exc}). Later: python tools/fetch_face_model.py")
    return rep


def _version() -> dict:
    """This code's commit and whether it has changes not committed (git; {} without it)."""
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=FRAMEWORK_ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain"], cwd=FRAMEWORK_ROOT, capture_output=True,
                                 text=True, timeout=5).stdout.splitlines()
        return {"commit": head, "files_changed": len(changed)}
    except Exception:                         # noqa: BLE001
        return {}


def _record(arm: str, folder: Path, cfg, rig: dict) -> None:
    """The flight recorder started for the run: which arm and deployment, with what (session.json), the console."""
    import platform

    from mrcf import flight_recorder, rig_report

    session = {"arm": rig.get("arm"), "deployment": rig_report.deployment(cfg) if cfg is not None else
               {"name": folder.name}, "rig": {k: v for k, v in rig.items() if k != "arm"},
               "version": _version(), "python": platform.python_version(), "platform": platform.platform(),
               "command": " ".join(sys.argv)}
    flight_recorder.RECORDER.start(session=session, console=True, crashes=True)


# ── Launch ────────────────────────────────────────────────────────────────────

LAST_DEPLOYMENT = "last_deployment.txt"     # in the arm's rig folder: the deployment run.py last launched on it


def _note_last(arm: str, name: str) -> None:
    """
    Which deployment ran on this arm last (rig/<arm>/last_deployment.txt), for
    a tool run after it with none named (tools/_common.task_config):
    2026-11-19, the SO-101's rest tool checked its path against the chess
    board's pieces after a tidy session, and folded the arm into the tub.
    """
    try:
        from mrcf import backends

        rig_dir = Path(getattr(backends.arm(arm).config, "RIG_DIR"))
        rig_dir.mkdir(parents=True, exist_ok=True)
        (rig_dir / LAST_DEPLOYMENT).write_text(name + "\n")
    except Exception:                                         # noqa: BLE001 — a note, never a stop
        pass


def _launch(folder: Path, extra_args: list[str]) -> None:
    """
    Run the chosen deployment's controller, in this process: import
    deployments.<name>.llm_controller and call its main(). sys.argv is what
    `python -m deployments.<name>.llm_controller <extra_args>` would give it —
    the controller's own path, then the arguments run.py did not take — and is
    set before the import, so the controller's modules see it as they load.
    """
    sys.argv = [str(folder / "llm_controller.py")] + extra_args
    controller = importlib.import_module(f"deployments.{folder.name}.llm_controller")
    controller.main()


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_run_args() -> tuple[str | None, str | None, list[str]]:
    """
    Strip run.py-specific flags from sys.argv and return:
      (forced_deployment, forced_arm, remaining_args_for_controller)
    """
    forced: dict[str, str | None] = {"--deployment": None, "--arm": None}
    remaining: list[str] = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        flag = arg.split("=", 1)[0]
        if flag in forced and "=" in arg:
            forced[flag] = arg.split("=", 1)[1]
            i += 1
        elif arg in forced and i + 1 < len(args):
            forced[arg] = args[i + 1]
            i += 2
        else:
            remaining.append(arg)
            i += 1

    return forced["--deployment"], forced["--arm"], remaining


def main() -> None:
    forced_deployment, forced_arm, passthrough_args = _parse_run_args()

    # The arm first, set as MRCF_ARM before any deployment loads: the tools
    # and anything else that asks backends.arm() get the deployment's. A
    # deployment named alone brings its arm.
    if forced_deployment and forced_arm is None:
        forced_arm = _deployment_arm(DEPLOYMENTS_DIR / forced_deployment)
    arm = _choose_arm(forced_arm)
    rig = _show_rig(arm, ask=forced_deployment is None)

    print("\nScanning deployments...\n")
    valid, skipped = _discover(arm)

    for folder, reason in skipped:
        print(f"  [skip] {folder.name}")
        print(f"         {reason}")

    if skipped:
        print()

    for folder, _ in valid:
        print(f"  [ok]   {folder.name}")

    if not valid and forced_deployment:
        print("\nNo deployments are ready to run.")
        sys.exit(1)

    print()

    # Pick deployment.
    if forced_deployment:
        match = next(((d, c) for d, c in valid if d.name == forced_deployment), None)
        if match is None:
            print(f"Deployment '{forced_deployment}' not found or not ready to run.")
            sys.exit(1)
        chosen_folder, chosen_cfg = match
    else:
        chosen_folder, chosen_cfg = _pick_deployment(valid, arm)

    _record(arm, chosen_folder, chosen_cfg, rig)
    _note_last(arm, chosen_folder.name)
    print(f"\nLaunching {chosen_folder.name}...\n")
    from mrcf import flight_recorder
    try:
        _launch(chosen_folder, passthrough_args)
    except KeyboardInterrupt:
        # Ctrl-C in the middle of a command: the controller has already sent
        # the arm home and disconnected on its way out; no traceback for that.
        print("\nStopped (Ctrl-C).")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        flight_recorder.RECORDER.failure(f"{chosen_folder.name} stopped on an error — {type(exc).__name__}: {exc}",
                                         exc=exc)
        raise
    finally:
        flight_recorder.RECORDER.stop()


if __name__ == "__main__":
    main()
