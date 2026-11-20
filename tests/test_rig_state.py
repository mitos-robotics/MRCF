"""
A rig's own files live in rig/<backend>/ (rig_state.py), not in the code: where
that is, the backends reading from it, refusing to start on files left behind,
and tools/migrate_rig_state.py moving them.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from mrcf import overhead_calibration  # noqa: E402
from mrcf import rig_state  # noqa: E402
from tools import migrate_rig_state  # noqa: E402


def test_the_rigs_folder_is_under_mrcf_home_else_this_repository(monkeypatch, tmp_path):
    monkeypatch.delenv("MRCF_HOME", raising=False)
    assert rig_state.home() == ROOT and rig_state.rig_dir("so101") == ROOT / "rig" / "so101"
    monkeypatch.setenv("MRCF_HOME", str(tmp_path))
    assert rig_state.rig_dir("so101") == tmp_path / "rig" / "so101" and not (tmp_path / "rig").exists()


def test_installed_rather_than_run_from_a_checkout_it_keeps_them_in_the_users_own_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("MRCF_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "me"))    # Windows' home: it ignores HOME
    monkeypatch.setattr(rig_state, "__file__", str(tmp_path / "site-packages" / "mrcf" / "rig_state.py"))
    assert rig_state.home() == tmp_path / "me" / ".mrcf"


def test_a_backend_will_not_start_with_its_state_left_in_its_code(tmp_path):
    (tmp_path / "overhead_calibration.json").write_text("{}")
    (tmp_path / "urdf").mkdir()
    with pytest.raises(RuntimeError, match="overhead_calibration.json, urdf.*migrate_rig_state.py --apply"):
        rig_state.rig_dir("so101", code_dir=tmp_path)
    (tmp_path / "notes.txt").write_text("not what it reads as it starts")
    (tmp_path / "overhead_calibration.json").unlink()
    (tmp_path / "urdf").rmdir()
    assert rig_state.rig_dir("so101", code_dir=tmp_path).name == "so101"


def test_nor_with_it_left_where_its_folder_was_before_mrcf(monkeypatch, tmp_path):
    monkeypatch.setenv("MRCF_HOME", str(tmp_path))
    (tmp_path / "backends" / "so101").mkdir(parents=True)
    (tmp_path / "backends" / "so101" / "surface_reference.png").write_text("png")
    with pytest.raises(RuntimeError, match="surface_reference.png"):
        rig_state.rig_dir("so101", code_dir=tmp_path / "elsewhere")


def test_the_backends_read_their_rig_files_from_their_rig_folder(monkeypatch):
    monkeypatch.setattr(overhead_calibration, "PATH", overhead_calibration.PATH)
    from mrcf.backends._template import robot_config as template
    from mrcf.backends.so101 import robot_config as so101
    for rc, name in ((so101, "so101"), (template, "_template")):
        assert rc.RIG_DIR == rig_state.home() / "rig" / name
        assert rc.OVERHEAD_CALIBRATION_PATH.parent == rc.SURFACE_REFERENCE_PATH.parent == rc.RIG_DIR
    assert so101.URDF_PATH.is_relative_to(so101.RIG_DIR)


def test_the_flight_recorder_keeps_its_runs_under_mrcf_home(tmp_path):
    out = subprocess.run([sys.executable, "-c", "from mrcf import flight_recorder; print(flight_recorder.ROOT)"], cwd=ROOT,
                         env={**os.environ, "MRCF_HOME": str(tmp_path)}, capture_output=True, text=True, check=True)
    assert Path(out.stdout.strip()) == tmp_path / "runs" / "flight"


def _old_layout(root: Path) -> None:
    """A repository as it was: a backend with its rig's files among its code."""
    code = root / "backends" / "arm1"
    (code / "tools" / "frames").mkdir(parents=True)
    (code / "tools" / "__pycache__").mkdir()
    for name, data in {"__init__.py": "", "env.py": "", "tools/check.py": "", "tools/__pycache__/check.pyc": "x",
                       ".DS_Store": "x", "tracked.json": "{}", "overhead_calibration.json": '{"T_arm_cam": 1}',
                       "tools/frames/a.png": "png", "tools/frames/.DS_Store": "x",
                       "tools/touches.json": "[1, 2]"}.items():
        (code / name).write_text(data)
    (root / "backends" / "_template").mkdir()
    (root / "backends" / "_template" / "__init__.py").write_text("")
    (root / "backends" / "_template" / "notes.png").write_text("the template's: not a rig")


def test_migration_moves_a_rigs_files_and_nothing_else(tmp_path):
    root, home = tmp_path / "repo", tmp_path / "home"
    _old_layout(root)
    code = root / "backends" / "arm1"
    moves = migrate_rig_state.plan(root, home, in_git={(code / "tracked.json").resolve()})
    rig = home / "rig" / "arm1"
    assert sorted(moves) == sorted([(code / "overhead_calibration.json", rig / "overhead_calibration.json"),
                                    (code / "tools/frames/a.png", rig / "frames/a.png"),
                                    (code / "tools/touches.json", rig / "touches.json")])
    moved, kept = migrate_rig_state.apply(moves)
    assert len(moved) == 3 and kept == []
    assert migrate_rig_state.prune(root) == [code / "tools" / "frames"]       # it held only .DS_Store
    assert (rig / "overhead_calibration.json").read_text() == '{"T_arm_cam": 1}'
    assert (rig / "frames/a.png").read_text() == "png" and (rig / "touches.json").read_text() == "[1, 2]"
    assert not (code / "overhead_calibration.json").exists() and not (code / "tools/frames").exists()
    assert (code / "env.py").exists() and (code / "tools/check.py").exists() and (code / "tracked.json").exists()
    assert migrate_rig_state.plan(root, home, in_git={(code / "tracked.json").resolve()}) == []


def test_migration_finds_a_rigs_files_in_the_package_too_and_clears_what_was_left_at_the_root(tmp_path):
    root, home = tmp_path / "repo", tmp_path / "home"
    code = root / "mrcf" / "backends" / "arm2"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text("")
    (code / "NOTES.md").write_text("# its notes: the code's, not the rig's")        # untracked, and still not moved
    (code / "surface_reference.png").write_text("png")
    left = root / "backends" / "arm2"                      # from before mrcf/: only what git never tracked
    (left / "tools" / "frames").mkdir(parents=True)
    (left / "tools" / "frames" / "a.png").write_text("png")
    (left / ".DS_Store").write_text("x")
    moves = migrate_rig_state.plan(root, home, in_git=set())
    rig = home / "rig" / "arm2"
    assert sorted(moves) == sorted([(code / "surface_reference.png", rig / "surface_reference.png"),
                                    (left / "tools/frames/a.png", rig / "frames/a.png")])
    migrate_rig_state.apply(moves)
    migrate_rig_state.prune(root)
    assert not (root / "backends").exists() and (code / "__init__.py").exists()


def test_migration_never_overwrites_what_is_already_in_the_rigs_folder(tmp_path):
    root, home = tmp_path / "repo", tmp_path / "home"
    _old_layout(root)
    rig = home / "rig" / "arm1"
    rig.mkdir(parents=True)
    (rig / "overhead_calibration.json").write_text('{"newer": true}')
    moved, kept = migrate_rig_state.apply(migrate_rig_state.plan(root, home, in_git=set()))
    assert [dst.name for _, dst in kept] == ["overhead_calibration.json"]
    assert (rig / "overhead_calibration.json").read_text() == '{"newer": true}'
    assert (root / "backends/arm1/overhead_calibration.json").read_text() == '{"T_arm_cam": 1}'
