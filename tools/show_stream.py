"""
show_stream.py — what a learned policy on the arm that runs (MRCF_ARM) is
given and sends, a tick at a time, and so exactly what a model for it must be
trained on; and whether a policy package or a LeRobot checkpoint fits it, and
can run on this machine.

Nothing connects to the arm and nothing moves. (An arm whose env opens its
cameras as it is made, as mrcf/backends/_template's does, opens them: the stream
carries the ones that are open.)

Usage:
    python tools/show_stream.py                     the stream: rate, names, shapes, cameras, fingerprint
    python tools/show_stream.py --json              the same as StreamSpec.to_dict(), for a training script
    python tools/show_stream.py --package REF       a policy package (a folder, hf:owner/name@rev, ...): does it fit?
    python tools/show_stream.py --checkpoint PATH   a LeRobot checkpoint (pretrained_model): does it fit?

Exits 1 when the arm has no stream, or what was given does not fit or cannot
run here. POLICIES.md, "A learned policy, step by step", says what to do next.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrcf.capabilities import ActionStreaming  # noqa: E402
from tools._common import ARM, make_env  # noqa: E402


def describe(spec) -> None:
    print(f"The {ARM.NAME}'s stream — what a model for it is trained on, and given here:")
    print(f"  rate         {spec.fps:g} Hz")
    print(f"  robot_type   {spec.robot_type or '(none)'}")
    print(f"  fingerprint  {spec.fingerprint()}   (what a model depends on: StreamSpec.fingerprint)")
    print("  features (a LeRobot dataset's keys; shapes as (height, width, channels) for a camera):")
    width = max(len(k) for k in spec.features())
    for key, feature in spec.features().items():
        names = feature.get("names")
        what = ", ".join(names) if key in ("observation.state", "action") else ""
        for camera_key, (camera, _w, _h) in spec.cameras.items():
            if key.endswith("." + camera_key) and key.startswith("observation.images."):
                what = f"from the camera {camera!r}"
        print(f"    {key:<{width}}  {feature['dtype']:7} {str(tuple(feature['shape'])):15} {what}")


def report(what: str, problems: list, missing: "str | None") -> bool:
    """Print whether `what` fits the stream and runs here; True when it does both."""
    if problems:
        print(f"{what} does NOT fit this arm's stream:")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"{what} fits this arm's stream.")
    print(f"It cannot run on this machine: it {missing}." if missing else "It can run on this machine.")
    return not problems and not missing


def check_package(ref: str, spec) -> bool:
    """
    The manifest's stream against the arm's (packages.fits), then — when this
    machine can load it — the model itself (its check(), which loads it), as
    LearnedPolicy does before anything moves.
    """
    from mrcf.policies import packages
    from mrcf.policies.learned import LearnedPolicy

    try:
        manifest, folder = packages.open_package(ref)
        print(f"\nPackage {manifest.name!r} {manifest.version} (format {manifest.format!r}, from {folder}),")
        print(f"trained on fingerprint {manifest.embodiment} at {manifest.stream.fps:g} Hz.")
        model = packages.load_model(folder, manifest)
    except Exception as exc:                  # noqa: BLE001 — a reference, a manifest or a format: say which
        print(f"\nThe package {ref!r} cannot be used: {exc}")
        return False
    problems = packages.fits(manifest, spec)
    missing = LearnedPolicy(model).unavailable()
    if not problems and not missing and hasattr(model, "check"):
        try:
            problems = list(model.check(spec))        # loads it
        except Exception as exc:              # noqa: BLE001 — its files: missing, or not what its format reads
            print(f"Its model cannot be loaded: {exc}")
            return False
    return report("It", problems, missing)


def check_checkpoint(path: str, spec) -> bool:
    from mrcf.policies.lerobot_model import LeRobotModel

    model = LeRobotModel(path)
    print(f"\nLeRobot checkpoint {path}:")
    missing = model.unavailable()
    if missing:
        print(f"It cannot be checked, or run, on this machine: it {missing}.")
        return False
    try:
        problems = model.check(spec)               # loads it: the weights, its processors
    except Exception as exc:                  # noqa: BLE001 — not a checkpoint, or one LeRobot cannot read
        print(f"It cannot be loaded: {exc}")
        return False
    return report("It", problems, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the stream as StreamSpec.to_dict()")
    ap.add_argument("--package", metavar="REF", help="a policy package: a folder, hf:owner/name@rev, scheme:...")
    ap.add_argument("--checkpoint", metavar="PATH", help="a LeRobot checkpoint folder (pretrained_model)")
    args = ap.parse_args()

    env = make_env()
    if not isinstance(env, ActionStreaming):
        print(f"The {ARM.NAME} takes no stream (capabilities.ActionStreaming), so no learned policy runs on it "
              'yet: POLICIES.md, "1. Give the arm a stream".')
        return 1
    spec = env.stream_spec
    if args.json:
        print(json.dumps(spec.to_dict(), indent=1))
    else:
        describe(spec)
    ok = True
    if args.package:
        ok = check_package(args.package, spec) and ok
    if args.checkpoint:
        ok = check_checkpoint(args.checkpoint, spec) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
