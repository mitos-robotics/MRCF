"""
mrcf/policies/packages.py — a learned policy as a package: the model's files and a
manifest (policy.json) saying what it is, the stream it was trained on, and how
a skill starts it. One arm's training produces one; any arm with the same
stream (StreamSpec.fingerprint) runs it; a deployment installs it by reference.
A store selling policies for different embodiments would sell these.

    upright/
      policy.json       {"name": "upright", "version": "1.0.0", "format": "lerobot", "model": "model",
                         "stream": {...StreamSpec.to_dict()...}, "embodiment": "<its fingerprint>",
                         "description": "Stand a fallen piece back up.", "task": "stand the pawn up",
                         "start_height_m": 0.06, "max_s": 20, "license": "...", "extra": {...}}
      model/            the model's own files (a LeRobot checkpoint, a Stable-Baselines3 run, ...)

    from mrcf.policies import packages
    manifest, folder = packages.open_package("hf:me/upright-so101@v1")    # fetched, read
    packages.fits(manifest, env.stream_spec)                              # [] when this arm can run it
    model = packages.load_model(folder, manifest)                         # a learned.Model

Three registries, each open to more:

  FORMATS   how a package's files become a learned.Model: "lerobot"
            (LeRobotModel), "sb3" (SB3Model); register_format() adds one.
            A format only makes the model; its weights load when first run
  SOURCES   where a package comes from, by the scheme of its reference: a
            local folder (a bare path, or "file:"), the Hugging Face hub
            ("hf:owner/name@revision"); register_source() adds one. A store
            would add its own scheme — its sign-in, its licence check, its
            download and cache are its source's business; the framework only
            ever sees the local folder it returns.
  arms      backends.arm()

An installed Python package adds to any of them without changing this
repository, through entry points (pyproject.toml's [project.entry-points]):
"mrcf.model_formats" and "mrcf.policy_sources" name a loader or a fetch
function, "mrcf.arms" an arm's package (mrcf/backends/__init__.py).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable

from mrcf.streaming import StreamSpec, differences

MANIFEST = "policy.json"


@dataclass
class Manifest:
    """A package's policy.json. See the module docstring."""
    name: str
    format: str                              # which loader: FORMATS ("lerobot", "sb3", or a plugin's)
    stream: StreamSpec                       # the stream its model was trained on
    model: str = "model"                     # the model's files, relative to the package
    version: str = "0"
    description: str = ""                    # for the planner: when to use it
    task: "str | None" = None                # the instruction a language-conditioned model is given
    start_height_m: "float | None" = None    # hover this high over the object first: where its demonstrations began
    max_s: float = 30.0                      # the most a segment may take
    license: str = ""                        # whose, and on what terms (recorded, not enforced)
    extra: dict = field(default_factory=dict)   # anything else: a store's own fields

    @property
    def embodiment(self) -> str:
        """The fingerprint of the stream it was trained on (StreamSpec.fingerprint)."""
        return self.stream.fingerprint()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["stream"] = self.stream.to_dict()
        data["embodiment"] = self.embodiment
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        known = {f for f in cls.__dataclass_fields__}
        fields = {k: v for k, v in data.items() if k in known and k != "stream"}
        return cls(stream=StreamSpec.from_dict(data["stream"]), **fields)

    def save(self, folder: "str | Path") -> Path:
        path = Path(folder) / MANIFEST
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def read(cls, folder: "str | Path") -> "Manifest":
        path = Path(folder) / MANIFEST
        if not Path(folder).is_dir():
            raise FileNotFoundError(f"No policy package at {folder}: no such folder"
                                    + ("" if Path(folder).is_absolute() else f" (from {Path.cwd()})") + ".")
        if not path.is_file():
            raise FileNotFoundError(f"No {MANIFEST} in {folder}: not a policy package.")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── Registries ─────────────────────────────────────────────────────────────────

def _lerobot(folder: Path, manifest: Manifest):
    from mrcf.policies.lerobot_model import LeRobotModel
    return LeRobotModel(folder / manifest.model, task=manifest.task)


def _sb3(folder: Path, manifest: Manifest):
    from mrcf.policies.rl import SB3Model
    return SB3Model(folder / manifest.model)


def _file(rest: str) -> Path:
    return Path(rest).expanduser()


def _huggingface(rest: str) -> Path:
    """'owner/name@revision' (revision optional) from the Hugging Face hub, into its cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("hf: packages need huggingface_hub (pip install huggingface_hub)") from exc
    repo, _, revision = rest.partition("@")
    return Path(snapshot_download(repo_id=repo, revision=revision or None))


FORMATS: "dict[str, Callable[[Path, Manifest], object]]" = {"lerobot": _lerobot, "sb3": _sb3}
SOURCES: "dict[str, Callable[[str], Path]]" = {"file": _file, "hf": _huggingface}


def register_format(name: str, load: "Callable[[Path, Manifest], object]") -> None:
    """
    A model format: load(package folder, manifest) -> a learned.Model. It only
    makes the model, cheaply — a deployment makes every skill's as it starts —
    and the weights load when it is first run; the model's unavailable() says
    which library it needs that is not installed (learned.needs).
    """
    FORMATS[name] = load


def register_source(scheme: str, fetch: "Callable[[str], Path]") -> None:
    """A package source: fetch(what follows "scheme:") -> the package's local folder."""
    SOURCES[scheme] = fetch


def _lookup(registry: dict, group: str, name: str, what: str):
    """registry[name], else an installed plugin's entry point of that name in group, registered on first use."""
    if name not in registry:
        for ep in entry_points(group=group):
            if ep.name == name:
                registry[name] = ep.load()
                break
    if name not in registry:
        found = sorted(set(registry) | {ep.name for ep in entry_points(group=group)})
        raise KeyError(f"No {what} '{name}' (known: {', '.join(found) or 'none'}; a plugin adds one "
                       f"through the {group!r} entry point group).")
    return registry[name]


# ── Using a package ────────────────────────────────────────────────────────────

def fetch(ref: "str | Path") -> Path:
    """
    The local folder of the package `ref` names: "scheme:rest" from that source
    (SOURCES), or a path. A one-letter scheme is a Windows drive ("C:\\\\..."), so a path.
    """
    if isinstance(ref, Path):
        return ref
    scheme, sep, rest = ref.partition(":")
    if not sep or len(scheme) == 1 or not scheme.replace("-", "_").isidentifier():
        return _file(ref)
    return Path(_lookup(SOURCES, "mrcf.policy_sources", scheme, "policy source")(rest))


def open_package(ref: "str | Path") -> "tuple[Manifest, Path]":
    """The package `ref` names: its manifest, and the folder it is in."""
    folder = fetch(ref)
    return Manifest.read(folder), folder


def load_model(folder: Path, manifest: Manifest):
    """
    The package's model, as a learned.Model, by its format — at once: a
    format's load only makes the model, whose weights load when it is first run.
    """
    return _lookup(FORMATS, "mrcf.model_formats", manifest.format, "model format")(Path(folder), manifest)


def fits(manifest: Manifest, spec: StreamSpec) -> list:
    """
    Why this arm's stream (spec) cannot run the package's model; [] when it can.
    Reads only the manifest: streaming.differences against the stream the model
    was trained on — its shapes, names, rate and robot type.
    """
    return differences(manifest.stream, spec)


def pack(model_files: "str | Path", out: "str | Path", manifest: Manifest) -> Path:
    """
    A package at out: model_files (a folder, or one file) copied in as
    manifest.model, and its policy.json. What a training run is published as.
    """
    out, src = Path(out), Path(model_files)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / manifest.model
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    manifest.save(out)
    return out


def skill_entry(entry: dict) -> dict:
    """
    A deployment's learned-skill entry ({"name", "policy": ref, and any
    overrides}) as a full one: the manifest's description, task, start height
    and time limit, the entry's own where it has them, and "model": the model,
    from its format (load_model: its weights load when it is first run; its
    unavailable() says now if this machine lacks a library it needs). An entry
    without "policy", or with its model already (one this has filled in),
    comes back as it is: a package is fetched once.
    """
    if "policy" not in entry or "model" in entry:
        return dict(entry)
    manifest, folder = open_package(entry["policy"])
    full = {"name": manifest.name, "description": manifest.description, "task": manifest.task,
            "start_height_m": manifest.start_height_m, "max_s": manifest.max_s,
            "embodiment": manifest.embodiment, "version": manifest.version, "trained_on": manifest.stream}
    full.update(entry)
    full["model"] = load_model(folder, manifest)
    return full
