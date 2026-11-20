"""tools/_keys.py: keys as they are pressed, no Enter — a terminal's (macOS, Linux) and Windows' console's."""

import os
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]      # the child process below puts it on its own path

from tools import _keys  # noqa: E402


def test_windows_keys_come_from_the_consoles_queue(monkeypatch):
    queue = list("x") + ["\xe0", "H", "q", "\x00", "M"]           # typed before; then up, q, right
    console = types.SimpleNamespace(kbhit=lambda: bool(queue), getwch=lambda: queue.pop(0))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", console)
    queue[:] = ["x"]
    with _keys.single_keys(arrows=True) as keys:
        assert keys.get() is None                                   # what was typed before it began is gone
        queue.extend(["\xe0", "H", "q", "\x00", "M"])
        assert [keys.get(), keys.get(), keys.wait()] == ["UP", "q", "RIGHT"]
    queue.extend(["\xe0", "H", "a"])
    with _keys.single_keys() as keys:                               # arrows not asked for: skipped
        assert keys.get() is None and keys() is None                # (the flush took them)
        queue.extend(["\xe0", "P", "a"])
        assert [keys.get(), keys.get()] == [None, "a"]


@pytest.mark.skipif(sys.platform == "win32", reason="a pseudo-terminal is POSIX's")
def test_a_terminals_keys_arrive_one_at_a_time_and_arrows_whole():
    import pty

    child = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from tools._keys import single_keys
        print("ready", flush=True)
        with single_keys(arrows=True) as keys:
            got = [keys.wait() for _ in range(3)]
        print("got", got, flush=True)
    """)
    leader, follower = pty.openpty()
    proc = subprocess.Popen([sys.executable, "-c", child], stdin=follower, stdout=follower, stderr=follower)
    os.close(follower)
    out = b""
    try:
        deadline = time.time() + 20
        while b"ready" not in out and time.time() < deadline:
            out += os.read(leader, 1024)
        time.sleep(0.3)
        os.write(leader, b"a")
        time.sleep(0.1)
        os.write(leader, b"\x1b[A")
        time.sleep(0.1)
        os.write(leader, b"q")
        while b"got" not in out and time.time() < deadline:
            out += os.read(leader, 1024)
        while b"]" not in out.split(b"got", 1)[-1] and time.time() < deadline:
            out += os.read(leader, 1024)
    finally:
        proc.wait(timeout=10)
        os.close(leader)
    assert b"got ['a', 'UP', 'q']" in out
