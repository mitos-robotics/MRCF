"""
scan_board.py — scan the board, measure every piece, log it. No planner.

The same scan the controller runs on startup and on `observe`, on its own: the
overhead camera finds the board, outlines every piece, and measures where each
one's foot is and how wide it is (chessboard.measure_pieces). Results go to
deployments/chess/board_state.json and the logs (board_log.py), and the
labelled, outlined frame to deployments/chess/board_overlay.png.

  (default)    overhead only, no vision model: positions, feet, colours.
               Types are not named, so pieces print as e.g. white_piece_e4.
  --classify   also name each piece's type and colour with the vision model
               (needs Ollama running), exactly as the controller does.
  --heights    then hover over each piece in turn and measure its height and
               width profile with the gripper camera — what the overhead camera
               cannot see. Logged to logs/shapes.jsonl, and kept with the piece
               in the object memory (object_memory.json), which follows it
               through later scans and moves — so its picks hover just above
               it instead of at the height that clears any piece. Returns home
               between pieces, so a hover never sweeps across one.

The arm is parked out of the overhead camera's view before the scan, as the
controller does; --no-move skips that (it must already be out of view).

Usage (from the repo root):
    python deployments/chess/tests/scan_board.py
    python deployments/chess/tests/scan_board.py --classify
    python deployments/chess/tests/scan_board.py --heights
    python deployments/chess/tests/scan_board.py --heights --squares e2,d7
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))


from deployments.chess import board_log     # noqa: E402
from deployments.chess import chessboard    # noqa: E402
from deployments.chess import config as cfg  # noqa: E402


def survey_heights(env, scene_state, only=None) -> list:
    """Hover over each piece, measure it with the gripper camera, log the shape."""
    from mrcf import manipulation
    from deployments.chess import skills
    from mrcf.policies.base import Vision

    pol = skills.POLICIES
    vision = Vision(measure=skills._measure_piece)
    rows = []
    pieces = [o for o in scene_state.objects
              if o.position is not None and (only is None or getattr(o, "square", None) in only)]
    for i, obj in enumerate(pieces, 1):
        tag = f"[{i}/{len(pieces)} {obj.name}]"
        if not obj.reachable:
            print(f"  {tag} out of reach — skipped")
            continue
        # The opening a pick would use: sized from the scan, narrowed to the
        # room the neighbours leave (or, with no room, as wide as planned).
        plan, _, _ = pol.fit(env, pol.plan(manipulation.target_of(obj)), obj.position[:2],
                             manipulation.obstacles_around(scene_state, obj))
        env.go_home()
        env.set_gripper(plan.pregrasp_open_m, n_steps=10)
        (name, hover, tol, _), = pol.pick_waypoints(obj.position, plan)[:1]
        rot = pol.hold_orientation(env, [(name, hover, tol)], f"  {tag}")
        if not env.move_ee_to(hover, tol=tol, orientation=rot):
            print(f"  {tag} could not reach the hover — skipped")
            continue
        m = pol.measure_at_hover(env, vision, f"  {tag}")
        if m is None:
            continue
        frames = [board_log.save_image("hover_frames", f"look{k}", f) for k, f in enumerate(m.frames)]
        ptype = cfg.piece_type(obj.name)
        geom = getattr(obj, "geometry", None)
        board_log.record_shape(ptype, obj.name.split("_")[0], getattr(obj, "square", None),
                               m.height_mm, m.widths_mm, m.foot_mm,
                               overhead_foot_mm=geom["base_diameter_m"] * 1000 if geom else None,
                               slant_deg=m.slant_deg, frames=frames, during="survey",
                               id=obj.track_id)
        scene_state.memory.measured(obj.track_id, m.height_mm, m.widths_mm, m.foot_mm)
        obj.measured_shape = {"height_mm": m.height_mm, "widths_mm": m.widths_mm}
        shaped = pol.plan_from_shape(m.height_mm / 1000, m.widths_mm)
        rows.append((obj.name, m, shaped))
        print(f"  {tag} {m.height_mm:.0f} mm tall, foot {m.foot_mm:.0f} mm -> grasp at "
              f"{shaped.grasp_height_m * 1000:.0f} mm, grip {shaped.grip_width_m * 1000:.0f} mm, "
              f"jaws {shaped.pregrasp_open_m * 1000:.0f} mm")
    env.go_home()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--classify", action="store_true", help="name types with the vision model")
    ap.add_argument("--heights", action="store_true", help="measure each piece's height from a hover")
    ap.add_argument("--squares", help="with --heights: only these squares, e.g. e2,d7")
    ap.add_argument("--no-move", action="store_true", help="do not park the arm first")
    args = ap.parse_args()
    classify = args.classify or args.heights
    import logging
    # lerobot logs every per-step clamp of a joint move at WARNING; a park move
    # alone prints dozens, which buries the scan's own output.
    logging.getLogger().setLevel(logging.ERROR)

    from tools._common import make_env
    from mrcf.vlm import SceneState
    if classify:
        from mrcf import vlm
        vlm.configure(model_server=cfg.MODEL_SERVER, model_url=cfg.MODEL_URL, model=cfg.MODEL, planning_model=cfg.PLANNING_MODEL,
                      observe_cameras=[cfg.OBSERVE_CAMERA], render_w=cfg.RENDER_W,
                      render_h=cfg.RENDER_H, work_plane_z=cfg.TABLE_Z)

    env = make_env()
    env.connect()
    try:
        if not args.no_move:
            from deployments.chess.skills import goto_observe_pose
            goto_observe_pose(env)
        state = SceneState()
        if not chessboard.discover(env, state, save_to=HERE.parent / "board_overlay.png", classify=classify):
            raise SystemExit("Board not found — see deployments/chess/board_debug.png")
        print(f"\n  Logged to {board_log.STATE_PATH}; outlines in {HERE.parent / 'board_overlay.png'}")
        if args.heights:
            only = {s.strip().lower() for s in args.squares.split(",")} if args.squares else None
            print("\n  Measuring heights from a hover over each piece (home between each)...")
            rows = survey_heights(env, state, only)
            print(f"\n  Measured {len(rows)} piece(s); shapes logged to {board_log.LOG_DIR / 'shapes.jsonl'}.")
    finally:
        try:
            env.go_home()
        finally:
            env.stop()


if __name__ == "__main__":
    main()
