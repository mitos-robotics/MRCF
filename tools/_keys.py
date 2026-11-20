"""
tools/_keys.py — keys as they are pressed, no Enter, on any OS: the terminal in
cbreak mode on macOS and Linux (termios), the console's own key queue on
Windows (msvcrt).

    with single_keys() as keys:
        key = keys.get()        # one key, or None when none is waiting
        key = keys.wait()       # the next key

With arrows=True, an arrow key comes back as "UP", "DOWN", "LEFT" or "RIGHT"
(on a terminal it is three bytes, told apart from a bare Esc without waiting on
it); without, a key is what the terminal sends, a byte at a time.
"""

import os
import sys
import time
from contextlib import contextmanager

_ANSI_ARROWS = {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}
_CONSOLE_ARROWS = {"H": "UP", "P": "DOWN", "M": "RIGHT", "K": "LEFT"}


class Keys:
    def __init__(self, get):
        self.get = get

    def __call__(self) -> "str | None":
        """get(): one key, or None when none is waiting."""
        return self.get()

    def wait(self, poll_s: float = 0.01) -> str:
        """The next key, however long it takes."""
        while True:
            key = self.get()
            if key is not None:
                return key
            time.sleep(poll_s)


def check_terminal() -> None:
    """Stop, saying why, when keys cannot be read one at a time here (not a terminal, or one that will not)."""
    if not sys.stdin.isatty():
        raise SystemExit("This needs an interactive terminal — stdin is not one.\n"
                         "Run it directly rather than through a pipe or a wrapper that captures stdin.")
    if sys.platform == "win32":
        return
    import termios
    try:
        termios.tcgetattr(sys.stdin.fileno())
    except termios.error as exc:
        raise SystemExit(f"Cannot put this terminal into single-key mode ({exc}).") from exc


@contextmanager
def single_keys(arrows: bool = False):
    """
    Keys act at once, no Enter, while it lasts; yields Keys. Whatever was
    typed before it starts is thrown away. On a terminal: cbreak (not raw — so
    Ctrl-C and printing keep working), read from the file descriptor rather
    than sys.stdin, so keys that arrive together do not wait in Python's buffer
    out of select's sight.
    """
    if sys.platform == "win32":
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getwch()

        def get():
            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):                     # an arrow or function key: its code follows
                code = msvcrt.getwch()
                return _CONSOLE_ARROWS.get(code) if arrows else None
            return ch

        yield Keys(get)
        return

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    termios.tcflush(fd, termios.TCIFLUSH)

    def get():
        if not select.select([fd], [], [], 0)[0]:
            return None
        ch = os.read(fd, 1).decode(errors="ignore") or None
        if arrows and ch == "\x1b" and select.select([fd], [], [], 0.02)[0]:
            return _ANSI_ARROWS.get(os.read(fd, 2).decode(errors="ignore"), "\x1b")
        return ch

    try:
        yield Keys(get)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
