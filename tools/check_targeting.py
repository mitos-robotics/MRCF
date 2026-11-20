"""
check_targeting.py — is the arm missing by a constant offset, a wrong scale, or a skew?

The problem this answers
------------------------
A pick can miss for reasons that look identical from the console: the board scan
could be reporting the wrong coordinate, or the arm could be failing to reach the
coordinate it was given. Both end with the jaws beside the piece. Nothing in the
normal pipeline measures the difference, because move_ee_to closes its loop on
forward kinematics — a model against a model, with nothing observing the world.

So this measures it against the one ruler in the scene that is known to be
correct: the board itself, whose squares are 40 mm.

Why stored data cannot answer it
--------------------------------
square_map() fits a SIMILARITY (uniform scale + rotation + translation) from the
detected corner lattice to the robot frame, then generates all 64 centres from
that model. So the square pitch in board_squares.json is exactly constant by
construction, and the ~9% compression visible there is the fitted scale.

That compression is NOT by itself a reason to miss. The overhead homography was
fitted by touching corners and recording FK positions, so it maps pixels into
the SAME distorted frame move_ee_to drives in. A uniform scale therefore cancels
exactly, everywhere. What does not cancel is anything that is not a similarity:
a bad corner touch (the homography is fitted from only 3, so it is exactly
determined and cannot average out an error), or a non-uniform FK distortion from
off-centre joint zeros.

Distinguishing those needs real landings, which is what this collects.

What it does
------------
Drives the closed jaws to a spread of squares' REPORTED centres, just above the
board, and asks where the fingertips actually ended up. Jaws closed because that
is the convention calibrate_overhead_camera.py recorded its corner touches in.

Offsets are reported in SQUARES, not millimetres — "a quarter square toward h" is
something you can eyeball against the grid lines, and it sidesteps the question
of whether you mean physical millimetres or the arm's compressed ones.

Then it fits three models to the landings and prints the residual of each:

    translation only   a constant offset   -> one correction fixes every square
    scale + offset     the fitted scale is wrong
    full affine        rotation or skew    -> suspect the 3 corner touches

The model that explains the data tells you which calibration to redo.

Usage:
    python tools/check_targeting.py [--deployment chess]
    python tools/check_targeting.py --squares a1,h1,a6,h6,d3,e4
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np   # noqa: E402

from tools._common import connecting_to, make_env, task_config  # noqa: E402

cfg = task_config()

# Spread across the board rather than clustered: a constant offset and a scale
# error look the same near the middle and only separate at the edges. Corners of
# the REACHABLE region, plus two interior squares that are easy to eyeball.
#
# Not the board's own corners. a8/h8 sit at x = 0.357-0.385 m, past this arm's
# practical straight-down reach — and pure kinematics does NOT catch that:
# approach_orientation approves all 64 squares at this height, because the
# offline model has no sag, backlash or torque limit. The empirical limit is
# ~0.33 m at board height (measured on this rig; see check_reach.py), which rules
# out rank 8 and makes rank 7 marginal, so the probe set stops at rank 6.
# Override with --squares if your board sits differently.
PROBE_SQUARES = ["a1", "h1", "a6", "h6", "d3", "e4"]

# Beyond this x, straight-down reach is unreliable in practice whatever the
# kinematics says. Used only to warn — your rig overrules it.
PRACTICAL_REACH_X_M = 0.33

# Height above TABLE_Z to present the fingertips at. TABLE_Z is defined as
# fingertips touching, so a few mm clears the board while staying close enough
# to judge position against the grid lines.
PRESENT_HEIGHT_M = 0.004

FILES, RANKS = "abcdefgh", "12345678"


def grid_of(name: str) -> "np.ndarray | None":
    """(file index, rank index) for a square name, 0-based."""
    if len(name) != 2 or name[0] not in FILES or name[1] not in RANKS:
        return None
    return np.array([FILES.index(name[0]), RANKS.index(name[1])], float)


def load_squares() -> dict:
    path = Path(cfg.__file__).parent / "board_squares.json"
    if not path.is_file():
        raise SystemExit(
            f"No board map at {path}.\n"
            "Run the controller and let it scan the board first (or say 'observe').")
    raw = json.loads(path.read_text())
    return {k: np.asarray(v["world"], float) for k, v in raw.items()}


def ask_offset(name: str) -> "np.ndarray | None":
    """Where the fingertips actually landed, in squares, relative to the centre."""
    print("    Look at where the closed fingertips are pointing, relative to the")
    print(f"    CENTRE of {name}. Report it in squares (1 square = 40 mm):")
    print("      first number  + toward file h,  - toward file a")
    print("      second number + toward rank 8,  - toward rank 1")
    print("    e.g. '0.25,-0.1' = a quarter square toward h, a tenth toward rank 1.")
    reply = input("    offset (blank = dead centre, 's' = skip): ").strip()
    if reply.lower() in ("s", "skip"):
        return None
    if not reply:
        return np.zeros(2)
    try:
        a, b = reply.split(",")
        return np.array([float(a), float(b)], float)
    except ValueError:
        print("    Could not parse that — skipping this square.")
        return None


def fit_models(g: np.ndarray, e: np.ndarray) -> list:
    """
    Fit error-vs-grid-position under three models, return (name, dof, rms, params).

    g : (N,2) grid coordinates of the probed squares
    e : (N,2) measured landing error at each, in squares
    """
    out = []
    n = len(g)

    # 1. Translation only: e = b
    b = e.mean(axis=0)
    out.append(("translation only", 2, float(np.sqrt(((e - b) ** 2).sum(1).mean())),
                {"offset_squares": b}))

    # 2. Isotropic scale about the board centre + translation: e = k*(g - mean) + b
    gc = g - g.mean(axis=0)
    denom = float((gc ** 2).sum())
    k = float((gc * (e - e.mean(axis=0))).sum() / denom) if denom > 0 else 0.0
    pred = k * gc + e.mean(axis=0)
    out.append(("scale + translation", 3, float(np.sqrt(((e - pred) ** 2).sum(1).mean())),
                {"scale_error": k, "offset_squares": e.mean(axis=0)}))

    # 3. Full affine: e = A @ g + b
    if n >= 3:
        design = np.hstack([g, np.ones((n, 1))])
        sol, *_ = np.linalg.lstsq(design, e, rcond=None)
        pred = design @ sol
        out.append(("full affine", 6, float(np.sqrt(((e - pred) ** 2).sum(1).mean())),
                    {"A": sol[:2].T, "offset_squares": sol[2]}))
    return out


def main() -> None:
    squares = load_squares()

    wanted = list(PROBE_SQUARES)
    for i, arg in enumerate(sys.argv):
        if arg == "--squares" and i + 1 < len(sys.argv):
            wanted = [s.strip() for s in sys.argv[i + 1].split(",") if s.strip()]
        elif arg.startswith("--squares="):
            wanted = [s.strip() for s in arg.split("=", 1)[1].split(",") if s.strip()]

    bad = [s for s in wanted if grid_of(s) is None]
    if bad:
        raise SystemExit(f"Not square names: {', '.join(bad)}")
    probes = [s for s in wanted if s in squares]
    missing = [s for s in wanted if s not in squares]
    if missing:
        print(f"Not in the board map, skipping: {', '.join(missing)}")

    far = [s for s in probes if squares[s][0] > PRACTICAL_REACH_X_M]
    if far:
        print(f"Warning: {', '.join(far)} sit beyond x={PRACTICAL_REACH_X_M} m, past this")
        print("arm's practical straight-down reach. Pure kinematics approves them anyway")
        print("(no sag in the model), so expect a stall rather than a refusal. Press 's'")
        print("at those, or re-run with --squares to replace them.\n")

    if len(probes) < 3:
        raise SystemExit(
            f"Need at least 3 probe squares in the board map, have {len(probes)}.\n"
            "Pass your own with --squares a1,h1,a6,h6,d3,e4")

    span = np.ptp(np.array([grid_of(s) for s in probes]), axis=0)
    print(f"Probing {len(probes)} squares: {', '.join(probes)}")
    print(f"  grid span {int(span[0])} files x {int(span[1])} ranks — a scale error shows up")
    print("  as roughly span x (scale error) squares of drift across the set, so a 5%")
    print(f"  error is about {0.05 * max(span) * 40:.0f} mm at the extremes.\n")

    env = make_env()
    print(f"Connecting to {connecting_to()}...")
    env.connect()
    print("Connected.\n")

    z = cfg.TABLE_Z + PRESENT_HEIGHT_M
    results = []
    try:
        env.go_home()
        # Closed, because that is the convention the overhead corner touches used:
        # the point get_ee_pos() reports is where the two pads meet.
        env.set_gripper(0.0, n_steps=15,
                        max_step_pct=getattr(cfg, "GRASP_CLOSE_STEP_PCT", None))
        print(f"Presenting the closed fingertips at {len(probes)} squares, "
              f"{PRESENT_HEIGHT_M * 1000:.0f} mm above the board.")
        print("Keep clear of the arm between prompts.\n")

        for name in probes:
            target = np.array([squares[name][0], squares[name][1], z])
            print(f"  {name}: scan says x={target[0]:+.4f} y={target[1]:+.4f}")
            if not env.move_ee_to(target, tol=0.006):
                here = env.get_ee_pos()
                print(f"    Did not converge — stopped {np.linalg.norm(here - target) * 1000:.0f} mm "
                      f"out. Skipping (this is an arm-reach problem, not a targeting one).")
                continue
            got = ask_offset(name)
            if got is None:
                continue
            results.append((name, grid_of(name), got))
            print()

        env.go_home()
    finally:
        env.go_home()
        env.stop()
        print("\nDisconnected.")

    if len(results) < 3:
        print(f"\nOnly {len(results)} usable measurement(s) — need 3 to fit anything.")
        return

    g = np.array([r[1] for r in results])
    e = np.array([r[2] for r in results])

    print("\n" + "=" * 72)
    print("Measured landings (1 square = 40 mm)")
    print("=" * 72)
    for name, _, err in results:
        print(f"  {name}   off by ({err[0]:+.2f}, {err[1]:+.2f}) squares "
              f"= ({err[0] * 40:+.0f}, {err[1] * 40:+.0f}) mm")

    print("\nModel fits — lower residual means that model explains the data:")
    models = fit_models(g, e)
    for label, dof, rms, _ in models:
        print(f"  {label:22} {dof} dof   residual {rms:.3f} squares "
              f"({rms * 40:.1f} mm)")

    print("\n" + "=" * 72)
    off = models[0][3]["offset_squares"]
    off_mm = float(np.linalg.norm(off)) * 40
    print(f"Mean offset: ({off[0]:+.2f}, {off[1]:+.2f}) squares "
          f"= ({off[0] * 40:+.0f}, {off[1] * 40:+.0f}) mm, {off_mm:.0f} mm total")

    # Pick the CHEAPEST model that fits, judged relatively. An absolute
    # threshold cannot work here: the measurements are eyeballed to maybe
    # 0.1 squares (4 mm), so a real constant offset leaves a ~4 mm residual that
    # an absolute cutoff would read as "nothing explains it". And argmin cannot
    # work either — more parameters always fit better, so affine would always
    # win. What matters is whether the extra parameter EARNS its place: if
    # adding it barely moves the residual, the simpler model is the explanation.
    EARNS = 0.65   # an added parameter must cut the residual to below this
    r_t = models[0][2]
    r_s = models[1][2] if len(models) > 1 else r_t
    r_a = models[2][2] if len(models) > 2 else r_s

    # 0.15 squares = 6 mm: above what eyeballing to ~0.1 squares can resolve,
    # but below the ~7 mm residual even a 5% scale error leaves, so a real
    # position-dependent error still falls through to the branches below.
    if off_mm < 4.0 and r_t < 0.15:
        print("\nTARGETING LOOKS FINE — offset under ~4 mm and consistent across the")
        print("board, which is within what this eyeball method can resolve. Whatever")
        print("is making the pick fail is probably not the coordinates. Next suspects:")
        print("the grasp height, or the piece being knocked on the way down.")
    elif r_s > EARNS * r_t and r_a > EARNS * r_t:
        print("\nA CONSTANT OFFSET explains it. Neither a scale term nor a full affine")
        print(f"improves on it ({r_t * 40:.1f} -> {r_s * 40:.1f} -> {r_a * 40:.1f} mm), so the scan's")
        print("scale and orientation are sound and everything is simply shifted.")
        print("That points at the TCP convention or at all three corner touches being")
        print("off the same way — not at the board fit. This is the easy case: it is")
        print("one translation, and I can apply it directly.")
    elif r_a > EARNS * r_s:
        k = models[1][3]["scale_error"]
        print(f"\nA SCALE ERROR explains it: {k:+.4f} squares of error per square of")
        print(f"distance = {k * 100:+.1f}%, and a full affine adds nothing")
        print(f"({r_s * 40:.1f} -> {r_a * 40:.1f} mm). The board fit's scale disagrees with the arm's.")
        print("Re-run calibrate_overhead_camera.py, touching corners as FAR APART as")
        print("possible — scale is the least certain parameter when the anchors are")
        print("close together, and it is fitted from only 3 points.")
    else:
        print(f"\nOnly a full affine explains it ({r_t * 40:.1f} -> {r_s * 40:.1f} -> {r_a * 40:.1f} mm),")
        print("which means rotation or skew, not a simple shift or stretch.")
        print("With only 3 corner touches the homography is exactly determined, so it")
        print("cannot average out a bad one — ONE mis-touched corner looks exactly")
        print("like this. Re-run calibrate_overhead_camera.py, and note that touching")
        print("with a single fingertip rather than the point where the closed pads")
        print("meet is the usual way one corner ends up wrong.")

    if r_t * 40 > 12 and len(results) < 5:
        print(f"\nCaveat: {len(results)} measurements and a {r_t * 40:.0f} mm spread in them. That is")
        print("few enough that one bad reading could be driving the verdict — worth")
        print("a second run before acting on it.")
    print("=" * 72)


if __name__ == "__main__":
    main()
