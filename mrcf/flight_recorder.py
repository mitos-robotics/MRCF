"""
flight_recorder.py — what happened, in order, kept so that a failure can be
read afterwards in full.

Every part of the framework writes to one timeline as it works: the policy what
it decides and why, the backend each move and grip and stall, the sentry what it
saw move and where, the interlock each pause and stop, the object memory what a
scan changed. Underneath, a rolling record of the last minute: the joints as
commanded and as read, and a few frames a second from each camera.

When something goes wrong — an action fails or needs another attempt, or a
later look contradicts what an action said it did — the recorder writes an
incident: the timeline from just before the action began, the joint record,
the frames, and a story to read (`python -m mrcf.flight_recorder`).

Why: in a failure each part sees only some of it — the sentry an object
falling, the place its jaws opening, the next scan the object missing — and
without one timeline nothing puts it together; the account comes from the
person watching.

    from mrcf import flight_recorder as fr
    fr.RECORDER.start(session={...})          # a session: runs/flight/<time>/ (again: a no-op, its info added)
    fr.event("policy", "say", text="...")      # from anywhere, any thread
    with fr.RECORDER.action("pick", object="queen") as act:
        ...
        act.problem("retried")                 # write an incident when it ends

A session folder also keeps: session.json (which arm, which deployment, its
cameras, this code's version: what the run was), console.log (everything
printed, time-stamped: every failure the code says in words), and
timings.json (each action's and each model call's count, failures and
seconds: where the time goes; `python -m mrcf.flight_recorder --timings`).
An exception nothing caught, on any thread, is an incident with its
traceback.

Recording never raises into its caller: a full disk or a bad frame costs the
record, not the move. Until start() nothing is written to disk, and the
rolling record is kept in memory only — tests and offline tools leave no files.

Nothing here knows about any robot. It imports only the standard library and
numpy (and OpenCV, when present, to keep frames as JPEG), and privacy.py,
which guards every picture it keeps: faces covered, and a camera's frames
kept to the work area when the caller says how (watch_cameras(keep=)).

The runs behind its values and fixes, with their dates: HISTORY.md.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# Where the rigs' own folders are (rig_state.home — the same rule, here and not
# imported: this module imports nothing of the framework): MRCF_HOME; else the
# repository mrcf/ is in; else, installed, ~/.mrcf.
_REPO = Path(__file__).resolve().parent.parent
ROOT = (Path(os.environ["MRCF_HOME"]) if os.environ.get("MRCF_HOME") else
        _REPO if (_REPO / "pyproject.toml").is_file() else Path.home() / ".mrcf") / "runs" / "flight"
KEEP_S = 90.0                 # the rolling record: this much of the past, for an incident to show
LEAD_S = 5.0                  # an incident's timeline starts this long before its action
FRAME_WIDTH = 640             # frames are kept this wide at most
_MAX_EVENTS = 20000


def _plain(value):
    """value made JSON-safe: numpy to Python, unknowns to their repr, long lists shortened."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value[:200]]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist()) if value.size <= 200 else f"<array {value.shape}>"
    if isinstance(value, (np.floating, float)):
        return round(float(value), 5)
    if isinstance(value, (np.integer, int, bool, str)) or value is None:
        return value.item() if isinstance(value, np.generic) else value
    return repr(value)[:300]


def _jpeg(image: np.ndarray, guarded: bool = False) -> "bytes | None":
    """
    image (RGB) as JPEG bytes, at most FRAME_WIDTH wide, faces covered
    (privacy.cover_people) unless already guarded; None without OpenCV.
    """
    try:
        import cv2
    except ImportError:
        return None
    if not guarded:
        from . import privacy

        image = privacy.cover_people(image)[0]
    h, w = image.shape[:2]
    if w > FRAME_WIDTH:
        image = cv2.resize(image, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else None


@dataclass
class Action:
    """One action under way or done (FlightRecorder.action): its span and what went wrong."""
    name: str
    context: dict
    started: float
    ended: "float | None" = None
    outcome: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)

    def problem(self, why: str) -> None:
        """Mark it for an incident when it ends: why (a few words)."""
        self.problems.append(str(why))

    def done(self, **outcome) -> None:
        """What it came to: ok, message, and whatever else says how."""
        self.outcome.update(_plain(outcome))


class FlightRecorder:
    """The timeline, the rolling record and the incidents. See the module docstring."""

    def __init__(self, keep_s: float = KEEP_S):
        self.keep_s = float(keep_s)
        self._lock = threading.Lock()
        self._events: collections.deque = collections.deque(maxlen=_MAX_EVENTS)
        self._samples: collections.deque = collections.deque(maxlen=int(keep_s * 60))
        self._frames: dict = {}                  # camera -> deque[(t, jpeg bytes)]
        self._actions: collections.deque = collections.deque(maxlen=50)
        self._dir: "Path | None" = None
        self._log = None
        self._watching: list = []
        self._exchanges = 0                      # model calls kept (keep_exchange), for their order
        self._t0 = time.time()
        self._session: dict = {}
        self._timings: dict = {}                 # name -> {count, failed, total_s, max_s}
        self._tees: list = []                    # (stream name, original) while the console is kept
        self._hooks: list = []                   # the exception hooks this replaced
        self._running = False

    # ── The session ──────────────────────────────────────────────────────────

    def start(self, root: "Path | str | None" = None, session: "dict | None" = None,
              console: bool = False, crashes: bool = False) -> "FlightRecorder":
        """
        Begin writing: events to <root>/<time>/events.jsonl, incidents beside
        them; what the session is to session.json. console: everything printed
        to console.log too; crashes: an exception nothing caught, an incident
        (the controller and run.py ask for both; stop() puts them back).
        Already recording: the session's info added, and console and crashes
        taken up if asked for now.
        """
        if self._running:
            if console:
                self._keep_console()
            if crashes:
                self._catch_crashes()
            self.session_info(**(session or {}))
            return self
        self._session, self._timings = {}, {}
        try:
            base = Path(root) if root is not None else ROOT
            self._dir = base / datetime.now().strftime("%Y%m%d_%H%M%S")
            (self._dir / "incidents").mkdir(parents=True, exist_ok=True)
            self._log = open(self._dir / "events.jsonl", "a", buffering=1, encoding="utf-8")  # type: ignore[assignment]  # a text file; typing it would need an import this module may not take
            self._running = True
            if console:
                self._keep_console()
            if crashes:
                self._catch_crashes()
            print(f"  [Recorder] Recording to {self._dir}")
        except OSError as exc:
            print(f"  [Recorder] Not recording to disk ({exc}).")
            self._dir, self._log = None, None
        self.session_info(started=datetime.now().isoformat(timespec="seconds"), **(session or {}))
        return self

    def stop(self) -> None:
        for stop in self._watching:
            stop.set()
        self._watching = []
        if self._running and self._dir is not None:
            self.write_timings(print_summary=True)
            self.session_info(stopped=datetime.now().isoformat(timespec="seconds"))
        self._release_crashes()
        self._release_console()
        self._running = False
        with self._lock:
            if self._log is not None:
                self._log.close()
            self._log = None

    def session_info(self, **info) -> None:
        """What the session is (arm, deployment, cameras, version, ...), merged into its session.json."""
        try:
            with self._lock:
                self._session.update(_plain(info))
                if self._dir is not None:
                    (self._dir / "session.json").write_text(json.dumps(self._session, indent=1), encoding="utf-8")
        except Exception as exc:          # never into the caller
            print(f"  [Recorder] session not kept: {exc}", file=sys.stderr)

    # ── The console, and what nothing caught ─────────────────────────────────

    def _keep_console(self) -> None:
        """Everything printed, to console.log as well, each line with its time."""
        if self._tees or self._dir is None:
            return
        log = open(self._dir / "console.log", "a", buffering=1, encoding="utf-8")
        for name in ("stdout", "stderr"):
            original = getattr(sys, name)
            setattr(sys, name, _Tee(original, log, name == "stderr"))
            self._tees.append((name, original))
        self._tees.append(("_log", log))

    def _release_console(self) -> None:
        log = None
        for name, original in self._tees:
            if name == "_log":
                log = original
            elif isinstance(getattr(sys, name), _Tee):
                setattr(sys, name, original)
        self._tees = []
        if log is not None:
            try:
                log.close()
            except Exception:
                pass

    def _catch_crashes(self) -> None:
        """An exception nothing caught, here or on any thread: an incident with its traceback, then as before."""
        if self._hooks:
            return
        before_sys, before_thread = sys.excepthook, threading.excepthook

        def on_sys(kind, exc, tb):
            self._crash(kind, exc, tb, "main thread")
            before_sys(kind, exc, tb)

        def on_thread(args):
            if args.exc_type is not SystemExit:
                self._crash(args.exc_type, args.exc_value, args.exc_traceback,
                            getattr(args.thread, "name", "a thread"))
            before_thread(args)
        sys.excepthook, threading.excepthook = on_sys, on_thread
        self._hooks = [before_sys, before_thread]

    def _release_crashes(self) -> None:
        if self._hooks:
            sys.excepthook, threading.excepthook = self._hooks
            self._hooks = []

    def _crash(self, kind, exc, tb, where: str) -> None:
        if kind is KeyboardInterrupt:
            self.event("user", "interrupt", where=where)
            return
        self.failure(f"crash in {where}: {kind.__name__}: {exc}", exc=exc, tb=tb)

    def failure(self, why: str, exc: "BaseException | None" = None, tb=None, **data) -> "Path | None":
        """
        A failure that is not an action's (a command's error, a start that did
        not, a crash): an event, and an incident with the traceback when there
        is one. Its incident's directory, or None.
        """
        import traceback

        text = None
        if exc is not None:
            text = "".join(traceback.format_exception(type(exc), exc, tb if tb is not None else exc.__traceback__))
        self.event("failure", "failure", why=why, **data)
        extra: "dict[str, object]" = dict(data)
        if text:
            extra["traceback"] = text
        return self.incident(why, None, **extra)

    # ── Timings ──────────────────────────────────────────────────────────────

    def _time(self, name: str, seconds: float, failed: bool) -> None:
        with self._lock:
            row = self._timings.setdefault(name, {"count": 0, "failed": 0, "total_s": 0.0, "max_s": 0.0})
            row["count"] += 1
            row["failed"] += int(bool(failed))
            row["total_s"] = round(row["total_s"] + seconds, 3)
            row["max_s"] = round(max(row["max_s"], seconds), 3)

    def timed(self, name: str, seconds: float, failed: bool = False) -> None:
        """Something that took `seconds` (a command, its planning, a step), counted in the session's timings."""
        try:
            self._time(str(name), float(seconds), failed)
        except Exception:
            pass

    def timings(self) -> dict:
        """Each action's and model call's {count, failed, total_s, max_s} this session (name -> row)."""
        with self._lock:
            return {k: dict(v) for k, v in self._timings.items()}

    def write_timings(self, print_summary: bool = False) -> "Path | None":
        """timings.json in the session's folder (and, print_summary, where the time went, to the console)."""
        rows = self.timings()
        if self._dir is None or not rows:
            return None
        try:
            path = self._dir / "timings.json"
            path.write_text(json.dumps(rows, indent=1, sort_keys=True), encoding="utf-8")
            if print_summary:
                print("  [Recorder] Where the time went:\n" + "\n".join(
                    "    " + line for line in timings_table(rows).splitlines()))
            return path
        except Exception as exc:          # never into the caller
            print(f"  [Recorder] timings not kept: {exc}", file=sys.stderr)
            return None

    @property
    def directory(self) -> "Path | None":
        return self._dir

    # ── Writing ──────────────────────────────────────────────────────────────

    def event(self, source: str, kind: str, image: "np.ndarray | None" = None, **data) -> None:
        """
        One thing that happened, from any thread: who saw it (source), what
        (kind), and the details. image: a frame that shows it (kept with the
        event, written with any incident whose window it falls in).
        """
        try:
            rec = {"t": time.time(), "source": str(source), "kind": str(kind), **_plain(data)}
            if source == "vlm" and kind == "call" and isinstance(data.get("seconds"), (int, float)):
                self._time(f"model:{data.get('tag', '?')}", float(data["seconds"]), False)
            if image is not None:
                jpg = _jpeg(image)
                if jpg is not None:
                    rec["_image"] = jpg
            with self._lock:
                self._events.append(rec)
                if self._log is not None:
                    self._log.write(json.dumps({k: v for k, v in rec.items() if k != "_image"}) + "\n")
        except Exception as exc:          # never into the caller's motion
            print(f"  [Recorder] event not kept: {exc}", file=sys.stderr)

    def keep_picture(self, stem: str, image: "np.ndarray | None") -> "Path | None":
        """
        image (RGB) kept with this run at full size, in its folder's pictures/,
        whatever happens after — not only when an incident writes the window
        out: for measuring later what a check saw (the check after a place
        keeps each camera's). Faces covered (privacy.py); none kept at all
        with MRCF_KEEP_PICTURES=0. None when the recorder is not running, or
        it cannot be written.
        """
        if self._dir is None or image is None:
            return None
        try:
            import cv2

            from . import privacy

            if not privacy.keep_pictures():
                return None
            image = privacy.cover_people(np.ascontiguousarray(image))[0]
            out = self._dir / "pictures" / f"{time.strftime('%H%M%S')}_{stem}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(out), cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, 90])
            return out if ok else None
        except Exception as exc:          # never into the caller's motion
            print(f"  [Recorder] picture not kept: {exc}", file=sys.stderr)
            return None

    def keep_exchange(self, tag: str, *, model: "str | None", system: "str | None", prompt: str, reply: str,
                      images_b64: "list | None" = None, seconds: "float | None" = None,
                      extra: "dict | None" = None) -> "Path | None":
        """
        One model call kept whole with this run, in its folder's models/: what
        it was told (system) and asked (prompt), what it answered, and the
        pictures it was shown (base64 PNG or JPEG), so that afterwards a plan
        can be read against exactly what the planner was given. The timeline
        keeps only that the call happened. A UR5 tabletop run (2026-11-13)
        planned "home, then the same refused place" and nothing showed why.
        None when the recorder is not running, or it cannot be written.
        """
        if self._dir is None:
            return None
        try:
            import base64

            from . import privacy

            with self._lock:
                self._exchanges += 1
                n = self._exchanges
            out = self._dir / "models"
            out.mkdir(parents=True, exist_ok=True)
            stem = f"{n:03d}_{time.strftime('%H%M%S')}_{tag}"
            pictures = []
            for k, b64 in enumerate(images_b64 or []):
                data = privacy.encoded(base64.b64decode(b64))        # faces covered
                kind = ".jpg" if data[:2] == b"\xff\xd8" else ".png"
                path = out / f"{stem}_{k}{kind}"
                path.write_bytes(data)
                pictures.append(path.name)
            record = {"tag": tag, "model": model, "seconds": None if seconds is None else round(seconds, 2),
                      "pictures": pictures, "system": system, "prompt": prompt, "reply": reply, **_plain(extra or {})}
            path = out / f"{stem}.json"
            path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
            return path
        except Exception as exc:          # never into the caller
            print(f"  [Recorder] model call not kept: {exc}", file=sys.stderr)
            return None

    def sample(self, stream: str, values: dict) -> None:
        """A reading for the rolling record (joints commanded, joints read): high rate, kept in memory."""
        try:
            with self._lock:
                self._samples.append((time.time(), str(stream), {k: round(float(v), 2) for k, v in values.items()}))
        except Exception:
            pass

    def frame(self, camera: str, image: np.ndarray, guarded: bool = False) -> None:
        """A camera frame for the rolling record (faces covered, unless the caller has: guarded)."""
        try:
            jpg = _jpeg(image, guarded=guarded)
            if jpg is None:
                return
            now = time.time()
            with self._lock:
                ring = self._frames.setdefault(camera, collections.deque())
                ring.append((now, jpg))
                while ring and ring[0][0] < now - self.keep_s:
                    ring.popleft()
        except Exception:
            pass

    def watch_cameras(self, env, cameras, hz: float = 2.0, width: int = 1280, height: int = 720,
                      keep=None) -> None:
        """
        Keep frames from env's cameras, `hz` a second each, on a thread of its
        own, until stop(). keep: (camera, frame) -> the frame as it may be kept
        (privacy.frame_guard: the work area only, faces covered); None: faces
        covered only.
        """
        stop = threading.Event()
        self._watching.append(stop)

        def run():
            while not stop.is_set():
                for cam in cameras:
                    try:
                        img = env.render_image(cam, width=width, height=height)
                        if keep is not None:
                            self.frame(cam, keep(cam, img), guarded=True)
                        else:
                            self.frame(cam, img)
                    except Exception:
                        pass
                stop.wait(1.0 / hz)
        threading.Thread(target=run, name="flight-recorder-frames", daemon=True).start()

    def watch_sensors(self, env, hz: float = 10.0) -> None:
        """
        Keep env's sensors' readings (env.list_sensors) on the rolling record,
        `hz` a second, as the "sensors" stream, until stop(), with its joints
        (env.get_joint_angles, "joint.<name>", as read). A vector reading is
        kept a value per element. The
        joints since 2026-11-13: a UR5 run wound the gripper's cable round its
        wrist, and nothing kept showed how far the wrist had turned.
        """
        try:
            names = list(env.list_sensors())
        except Exception:
            names = []
        stop = threading.Event()
        self._watching.append(stop)

        def run():
            while not stop.is_set():
                values = {}
                for name in names:
                    try:
                        v = np.atleast_1d(np.asarray(env.read_sensor(name), float)).ravel()
                    except Exception:
                        continue
                    values.update({name: v[0]} if v.size == 1 else {f"{name}[{i}]": x for i, x in enumerate(v)})
                try:
                    values.update({f"joint.{k}": float(v) for k, v in env.get_joint_angles().items()})
                except Exception:
                    pass
                if values:
                    self.sample("sensors", values)
                stop.wait(1.0 / hz)
        threading.Thread(target=run, name="flight-recorder-sensors", daemon=True).start()

    # ── Actions ──────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def action(self, name: str, **context):
        """
        An action's span on the timeline. An exception through it (a stop
        included) is recorded and passed on; an action marked with problem()
        writes an incident as it ends.
        """
        act = Action(name, _plain(context), time.time())
        with self._lock:
            self._actions.append(act)
        self.event("action", "begin", action=name, **context)
        try:
            yield act
        except BaseException as exc:
            act.problem(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            act.ended = time.time()
            failed = bool(act.problems) or act.outcome.get("ok") is False
            self._time(name, act.ended - act.started, failed)
            self.event("action", "end", action=name, seconds=round(act.ended - act.started, 3), **act.outcome)
            if act.problems:
                self.incident("; ".join(act.problems), act)

    def last_action(self, name: "str | None" = None) -> "Action | None":
        """The latest action (named `name`, if given)."""
        with self._lock:
            for act in reversed(self._actions):
                if name is None or act.name == name:
                    return act
        return None

    # ── Incidents ────────────────────────────────────────────────────────────

    def incident(self, why: str, action: "Action | None" = None, **data) -> "Path | None":
        """
        Write what led up to `why`: from LEAD_S before `action` began (else
        the last 30 s) to now — events, the joints' record, the frames, and
        story.txt. Its directory, or None when not recording to disk.
        """
        try:
            now = time.time()
            since = (action.started if action is not None else now - 30.0) - LEAD_S
            self.event("recorder", "incident", why=why, action=action.name if action else None, **data)
            with self._lock:
                events = [e for e in self._events if e["t"] >= since]
                samples = [s for s in self._samples if s[0] >= since]
                frames = {cam: [(t, j) for t, j in ring if t >= since] for cam, ring in self._frames.items()}
            if self._dir is None:
                return None
            name = datetime.fromtimestamp(now).strftime("%H%M%S")
            label = (action.name if action is not None else "session") + "_" + "".join(
                c if c.isalnum() else "_" for c in why.lower())[:40]
            out = self._dir / "incidents" / f"{name}_{label}"
            (out / "frames").mkdir(parents=True, exist_ok=True)
            summary = {"time": datetime.fromtimestamp(now).isoformat(timespec="seconds"), "why": why,
                       "since": since, **_plain(data)}
            if action is not None:
                summary["action"] = {"name": action.name, "context": action.context, "started": action.started,
                                     "ended": action.ended, "outcome": action.outcome, "problems": action.problems}
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
            with open(out / "timeline.jsonl", "w", encoding="utf-8") as f:
                for e in events:
                    if "_image" in e:
                        shot = f"frames/event_{e['t'] - since:07.2f}_{e['source']}.jpg"
                        (out / shot).write_bytes(e["_image"])
                        e = {**{k: v for k, v in e.items() if k != "_image"}, "image": shot}
                    f.write(json.dumps(e) + "\n")
            with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
                for t, stream, values in samples:
                    f.write(json.dumps({"t": t, "stream": stream, **values}) + "\n")
            for cam, shots in frames.items():
                for t, jpg in shots:
                    (out / "frames" / f"{cam}_{t - since:07.2f}.jpg").write_bytes(jpg)
            (out / "story.txt").write_text(explain(out), encoding="utf-8")
            with open(self._dir / "incidents.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": summary["time"], "why": why, "dir": str(out),
                                    "action": action.name if action else None}) + "\n")
            print(f"  [Recorder] Incident: {why} — {out}")
            return out
        except Exception as exc:
            print(f"  [Recorder] incident not written: {exc}", file=sys.stderr)
            return None


class _Tee:
    """sys.stdout or sys.stderr, also to the session's console.log, each line with its time."""

    def __init__(self, stream, log, is_err: bool):
        self._stream, self._log, self._err = stream, log, is_err
        self._line_start = True

    def write(self, text):
        n = self._stream.write(text)
        try:
            out, stamp = [], datetime.now().strftime("%H:%M:%S.%f")[:-3] + (" ! " if self._err else "   ")
            for piece in str(text).splitlines(keepends=True):
                out.append((stamp if self._line_start else "") + piece)
                self._line_start = piece.endswith("\n")
            self._log.write("".join(out))
        except Exception:
            pass
        return n

    def flush(self):
        self._stream.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def fileno(self):              # input() still edits lines on a terminal: it asks this
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def timings_table(rows: dict) -> str:
    """{name: {count, failed, total_s, max_s}} as a table, the most time first."""
    if not rows:
        return "(nothing timed)"
    lines = [f"{'what':<28} {'count':>5} {'failed':>6} {'total s':>8} {'mean s':>7} {'max s':>7}"]
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["total_s"]):
        mean = r["total_s"] / max(1, r["count"])
        lines.append(f"{name:<28} {r['count']:>5} {r['failed']:>6} {r['total_s']:>8.1f} {mean:>7.1f} {r['max_s']:>7.1f}")
    return "\n".join(lines)


# The session's recorder: what the framework writes to.
RECORDER = FlightRecorder()


def event(source: str, kind: str, image: "np.ndarray | None" = None, **data) -> None:
    """RECORDER.event."""
    RECORDER.event(source, kind, image=image, **data)


def keep_picture(stem: str, image: "np.ndarray | None") -> "Path | None":
    """RECORDER.keep_picture."""
    return RECORDER.keep_picture(stem, image)


def keep_exchange(tag: str, **record) -> "Path | None":
    """RECORDER.keep_exchange."""
    return RECORDER.keep_exchange(tag, **record)


def sample(stream: str, values: dict) -> None:
    """RECORDER.sample."""
    RECORDER.sample(stream, values)


# ── Reading ──────────────────────────────────────────────────────────────────

_SKIP_IN_STORY = {"t", "source", "kind", "image"}


def explain(incident_dir: "Path | str") -> str:
    """An incident as text: why, the action, then every event, seconds from the start."""
    d = Path(incident_dir)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    since = summary.get("since", 0.0)
    lines = [f"Incident {summary['time']}: {summary['why']}"]
    act = summary.get("action")
    if act:
        lines.append(f"Action: {act['name']} {json.dumps(act.get('context', {}))}")
        if act.get("outcome"):
            lines.append(f"Outcome: {json.dumps(act['outcome'])[:600]}")
    if summary.get("traceback"):
        lines += ["", "Traceback:", str(summary["traceback"]).rstrip()]
    lines.append("")
    for line in (d / "timeline.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        rest = ", ".join(f"{k}={v}" for k, v in e.items() if k not in _SKIP_IN_STORY)
        shot = f"  [{e['image']}]" if "image" in e else ""
        lines.append(f"{e['t'] - since:+7.2f}s  {e['source']:<9} {e['kind']:<14} {rest}{shot}")
    frames = sorted(p.name for p in (d / "frames").glob("*.jpg"))
    if frames:
        lines += ["", f"Frames ({len(frames)}): {frames[0]} ... {frames[-1]}"]
    return "\n".join(lines)


def _latest_incident(root: Path) -> "Path | None":
    found = sorted(root.glob("*/incidents/*/summary.json"))
    return found[-1].parent if found else None


def main(argv=None) -> int:
    """
    python -m mrcf.flight_recorder [incident dir | --list | --timings [run dir]]:
    the latest incident's story, or one's; the list of incidents; where a
    run's time went (the latest run's, or one's).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--timings":
        runs = [Path(args[1])] if len(args) > 1 else sorted(p.parent for p in ROOT.glob("*/timings.json"))[-1:]
        if not runs or not (runs[0] / "timings.json").exists():
            print("No run's timings recorded yet." if not runs else f"No timings.json in {runs[0]}.")
            return 1
        print(f"{runs[0]}:\n" + timings_table(json.loads((runs[0] / "timings.json").read_text(encoding="utf-8"))))
        return 0
    if args and args[0] == "--list":
        for p in sorted(ROOT.glob("*/incidents.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                print(f"{rec['time']}  {rec.get('action') or '-':<12} {rec['why']}\n    {rec['dir']}")
        return 0
    target = Path(args[0]) if args else _latest_incident(ROOT)
    if target is None or not (target / "summary.json").exists():
        print("No incident recorded yet." if target is None else f"No incident at {target}.")
        return 1
    print(explain(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
