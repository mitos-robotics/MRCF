"""
devices.py — the cameras and ports this computer has, by name, without
opening any.

What run.py shows before a session (the rig's cameras, which are plugged in,
which index each is now) and what tools/find_cameras.py lists come from here.
A camera is never opened to find out what it is: macOS's AVFoundation names
its cameras in ffmpeg's device listing, Linux in /sys/class/video4linux; a
computer's own cameras (the laptop's, an iPhone's Continuity Camera) are
named, and left alone (privacy.personal). 2026-11-14 a sweep over OpenCV's
indices opened the iPhone's camera; that is never done here.

    video_devices()      {index: device name}; {} where they cannot be listed
    find(name)           the index of the one camera whose name contains name
    check_sources(srcs)  refuse to open what is, or may now be, this computer's own camera
    serial_ports()       the serial ports there are (an SO-101's follower is one)

The arms check the cameras they open (check_sources): an index is only a
place in today's list, and with the rig's camera unplugged, index 0 on a
MacBook is its own camera. 2026-11-17: with the rig unplugged a test opened
the Mac's camera through the UR5's overhead index.
"""

from __future__ import annotations

import glob
import platform
import re
import subprocess
from pathlib import Path

LIST_TIMEOUT_S = 10.0


def _avfoundation() -> "dict[int, str]":
    """macOS: {index: name} of the AVFoundation video devices, from ffmpeg's listing (nothing opened)."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                            capture_output=True, text=True, timeout=LIST_TIMEOUT_S)
    devices: "dict[int, str]" = {}
    video = False
    for line in result.stderr.splitlines():
        if "AVFoundation video devices" in line:
            video = True
            continue
        if "AVFoundation audio devices" in line:
            break
        found = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line) if video else None
        if found:
            devices[int(found.group(1))] = found.group(2).strip()
    return devices


def _video4linux() -> "dict[int, str]":
    """Linux: {index: name} from /sys/class/video4linux/video<N>/name (nothing opened)."""
    devices: "dict[int, str]" = {}
    for entry in sorted(Path("/sys/class/video4linux").glob("video*")):
        index = re.sub(r"\D", "", entry.name)
        try:
            devices[int(index)] = (entry / "name").read_text().strip()
        except (OSError, ValueError):
            continue
    return devices


def video_devices() -> "dict[int, str]":
    """
    {index: device name} of this computer's cameras, as OpenCV numbers them
    now, without opening any. {} where they cannot be listed (Windows; macOS
    without ffmpeg): then nothing can say which index is which camera.
    RuntimeError when the listing itself failed.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            return _avfoundation()
        if system == "Linux":
            return _video4linux()
    except FileNotFoundError:
        return {}
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"could not list the cameras ({exc})") from exc
    return {}


def find(name: str, devices: "dict[int, str] | None" = None) -> "int | None":
    """The index of the one camera whose device name contains `name` (any case); None for none or several."""
    devices = video_devices() if devices is None else devices
    hits = [i for i, dev in devices.items() if name.lower() in dev.lower()]
    return hits[0] if len(hits) == 1 else None


class PersonalCamera(RuntimeError):
    """A camera source that is, or may be, this computer's own camera (the laptop's, an iPhone's), not the rig's."""


def rig_cameras(devices: "dict[int, str]") -> "dict[int, str]":
    """The listed cameras that may be a rig's: not this computer's own (privacy.personal)."""
    from . import privacy

    return {i: n for i, n in devices.items() if not privacy.personal(n)}


def check_sources(sources: dict, devices: "dict[int, str] | None" = None, system: "str | None" = None) -> None:
    """
    Raise PersonalCamera rather than open a camera that is, or may now be,
    this computer's own. sources: {camera name: OpenCV index, path or URL}.

      Linux   an index is /dev/video<N>, the listing's own number: refused
              when that device is one of the computer's own cameras.
      macOS   ffmpeg's listing does not number cameras as OpenCV does (the
              SO-101 rig, 2026-11: OpenCV saw its two cameras at 0 and 1,
              ffmpeg listed the MacBook's and the iPhone's among them), so
              no index can be named. Refused when fewer of the listed
              cameras may be the rig's than there are indexes to open: one
              is unplugged, and its index may now be the Mac's own camera.
      else    (no listing) nothing is refused.

    Paths and URLs always pass.
    """
    indexes = {name: src for name, src in sources.items()
               if isinstance(src, int) and not isinstance(src, bool)}
    if not indexes:
        return
    try:
        listed = video_devices() if devices is None else devices
    except RuntimeError:
        return
    if not listed:
        return
    from . import privacy

    system = system or platform.system()
    if system == "Linux":
        for name, src in indexes.items():
            dev = listed.get(src)
            if dev and privacy.personal(dev):
                raise PersonalCamera(f"camera '{name}': /dev/video{src} is '{dev}', this computer's own camera, "
                                     "not the rig's — not opened (python tools/find_cameras.py lists them by name).")
        return
    rig = rig_cameras(listed)
    if len(rig) < len(set(indexes.values())):
        own = sorted(n for n in listed.values() if privacy.personal(n))
        raise PersonalCamera(
            f"{len(set(indexes.values()))} rig camera(s) to open ({', '.join(sorted(indexes))}) but only {len(rig)} "
            f"plugged in ({', '.join(rig.values()) or 'none'}); the rest of this computer's are its own "
            f"({', '.join(own)}). An index may now be one of those: nothing opened. Plug the rig's cameras in "
            "(python tools/find_cameras.py lists them by name).")


def check_source(source, what: str = "camera", devices: "dict[int, str] | None" = None) -> None:
    """check_sources for one camera."""
    check_sources({what: source}, devices)


def serial_ports() -> list:
    """The serial ports there are (macOS /dev/tty.*usb*, Linux /dev/ttyUSB*, /dev/ttyACM*); none opened."""
    return sorted(set(glob.glob("/dev/tty.usb*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")))
