"""
wiring.py — an arm's env and policies from its config (robot_config.py), by
the setting names every arm uses, so an arm's make_env() and make_policies()
pass only what is its own.

    rig_kwargs(config)                RigEnv._init_rig's keywords: the cameras' jobs and
                                      calibrations, the gripper tag, the sentry, the jaw
                                      model, the gripper's box
    camera_kwargs(config, overrides)  RealRobotEnv's: the cameras it opens, and their sizes
    gripper_box(config)               envelope.GripperBox from GRIPPER_BOX (None: none)
    gripper_spec(config, **overrides) policies.parallel_jaw.GripperSpec from the gripper's settings

A setting the config does not define is left out, so the env's or the spec's
own default stands. The UR5's make_env once copied about forty of these by
hand and its make_policies missed JAW_AXIS: the policy planned room for jaws
opening one way while the arm turned them the other (2026-11-12). Read from
here, a setting an arm defines reaches every part that takes it.
"""

from __future__ import annotations

from types import ModuleType

# RigEnv._init_rig's keyword <- the config setting it comes from.
RIG_SETTING_FOR = {
    "camera_name": "CAMERA_NAME",
    "render_w": "RENDER_W",
    "render_h": "RENDER_H",
    "work_plane_z": "TABLE_Z",
    "overhead_camera_name": "OVERHEAD_CAMERA_NAME",
    "overhead_homography": "OVERHEAD_HOMOGRAPHY",
    "overhead_cam_model": "OVERHEAD_CAM_MODEL",
    "gripper_cam_k": "GRIPPER_CAM_K",
    "gripper_cam_dist": "GRIPPER_CAM_DIST",
    "gripper_cam_pose": "GRIPPER_CAM_POSE",
    "gripper_cam_plane_z_m": "GRIPPER_CAM_PLANE_Z_M",
    "gripper_cam_scan_shift_m": "GRIPPER_CAM_SCAN_SHIFT_M",
    "arm_frame_scale": "ARM_FRAME_SCALE",
    "gripper_cam_grasp_px": "GRIPPER_CAM_GRASP_PX",
    "gripper_cam_jacobian": "GRIPPER_CAM_JACOBIAN",
    "gripper_cam_ref_height_m": "GRIPPER_CAM_REF_HEIGHT_M",
    "gripper_cam_ref_tilt_deg": "GRIPPER_CAM_REF_TILT_DEG",
    "gripper_cam_ref_point_height_m": "GRIPPER_CAM_REF_POINT_HEIGHT_M",
    "gripper_cam_position_m": "GRIPPER_CAM_POSITION_M",
    "gripper_tag_id": "GRIPPER_TAG_ID",
    "gripper_tag_pose": "GRIPPER_TAG_POSE",
    "gripper_tag_size_m": "GRIPPER_TAG_READS_AS_M",
    "motion_guard_cameras": "MOTION_GUARD_CAMERAS",
    "cameras": "CAMERAS",
    "sensors": "SENSORS",
    "tcp_offset": "JAW_CLOSED_POINT_M",
    "jaw_centre_slide": "JAW_CENTRE_SLIDE",
    "jaw_slide_axis": "JAW_SLIDE_AXIS",
    "gripper_max_open_m": "GRIPPER_MAX_OPEN_M",
}

# GripperSpec's field <- the config setting it comes from.
SPEC_SETTING_FOR = {
    "max_open_m": "GRIPPER_MAX_OPEN_M",
    "tip_below_grasp_m": "JAW_TIP_BELOW_GRASP_M",
    "free_depth_m": "JAW_FREE_DEPTH_M",
    "finger_thickness_m": "JAW_FINGER_THICKNESS_M",
    "finger_half_width_m": "JAW_FINGER_HALF_WIDTH_M",
    "jaw_axis": "JAW_AXIS",
    "camera_body_m": "GRIPPER_CAM_BODY_M",
    "table_z": "TABLE_Z",
    "render_w": "RENDER_W",
    "render_h": "RENDER_H",
    "hover_height_m": "HOVER_HEIGHT",
    "lift_height_m": "LIFT_HEIGHT",
    "release_clearance_m": "PLACE_RELEASE_M",
    "grasp_height_fraction": "GRASP_HEIGHT_FRACTION",
    "grasp_flat_mm": "GRASP_FLAT_MM",
    "min_grasp_height_m": "MIN_GRASP_HEIGHT_M",
    "jaw_slack_m": "JAW_SLACK_M",
    "neighbour_clearance_m": "NEIGHBOUR_CLEARANCE_M",
    "unknown_open_m": "UNKNOWN_OPEN_M",
    "unmeasured_obstacle_radius_m": "UNMEASURED_OBSTACLE_RADIUS_M",
    "miss_width_m": "GRASP_MISS_WIDTH_M",
    "min_hold_width_m": "MIN_HOLD_WIDTH_M",
    "approach_tol_m": "APPROACH_TOL",
    "grasp_step_deg": "GRASP_STEP_DEG",
    "fast_descent_above_top_m": "FAST_DESCENT_ABOVE_TOP_M",
    "close_step_pct": "GRASP_CLOSE_STEP_PCT",
    "retries": "MAX_SKILL_RETRIES",
    "measure_at_hover": "MEASURE_AT_HOVER",
    "height_from_views": "HEIGHT_FROM_VIEWS",
    "back_off_to_attempt_hover": "BACK_OFF_TO_ATTEMPT_HOVER",
    "height_looks_max": "HEIGHT_LOOKS_MAX",
    "grip_low_when_capped": "GRIP_LOW_WHEN_CAPPED",
    "grip_low_from_above": "GRIP_LOW_FROM_ABOVE",
    "wrist_at_grasp": "WRIST_AT_GRASP",
    "stopped_short_holds": "STOPPED_SHORT_HOLDS",
    "open_wide_when_low": "OPEN_WIDE_WHEN_LOW",
    "approach_look": "APPROACH_LOOK",
    "place_in": "PLACE_IN",
    "place_in_over_contents": "PLACE_IN_OVER_CONTENTS",
    "transit_over_walls": "TRANSIT_OVER_WALLS",
    "transit_settle": "TRANSIT_SETTLE",
    "container_max_lean_deg": "CONTAINER_MAX_LEAN_DEG",
    "container_max_drop_lean_deg": "CONTAINER_MAX_DROP_LEAN_DEG",
    "container_lean_above_walls": "CONTAINER_LEAN_ABOVE_WALLS",
    "descent_turn_first": "DESCENT_TURN_FIRST",
    "held_closed_m": "HELD_CLOSED_M",
    "container_descent_retry": "CONTAINER_DESCENT_RETRY",
    "container_cross_extra_m": "CONTAINER_CROSS_EXTRA_M",
    "rim_pick": "RIM_PICK",
    "handle_first": "HANDLE_FIRST",
    "lying_pick": "LYING_PICK",
    "lying_min_aspect": "LYING_MIN_ASPECT",
    "lying_side_clear_m": "LYING_SIDE_CLEAR_M",
    "lying_test_lift_m": "LYING_TEST_LIFT_M",
    "lying_join_min_m": "LYING_JOIN_MIN_M",
    "container_gripper_inside_m": "CONTAINER_GRIPPER_INSIDE_M",
    "grasp_reseat_along_m": "GRASP_RESEAT_ALONG_M",
    "touch_surface": "TOUCH_SURFACE",
    "remeasure_known": "REMEASURE_KNOWN",
    "aim_at_measured_foot": "AIM_AT_MEASURED_FOOT",
    "max_aim_shift_m": "MAX_AIM_SHIFT_M",
    "max_servo_shift_m": "MAX_SERVO_SHIFT_M",
    "overhead_first": "OVERHEAD_FIRST",
    "descend_unconfirmed": "DESCEND_UNCONFIRMED",
    "servo_agree_m": "SERVO_AGREE_M",
    "grasp_tol_m": "GRASP_TOL_M",
    "servo": "VISUAL_SERVO_ENABLED",
    "servo_tol_px": "SERVO_TOL_PX",
    "servo_height_m": "GRIPPER_CAM_REF_HEIGHT_M",
    "servo_point_height_fraction": "SERVO_POINT_HEIGHT_FRACTION",
    "servo_open_m": "PREGRASP_OPEN_M",       # what tools/so101/calibrate_gripper_servo.py opens the jaws to
    "servo_tilt_deg": "GRIPPER_CAM_REF_TILT_DEG",
    "near_base_m": "NEAR_BASE_M",
    "far_reach_m": "FAR_REACH_M",
    "far_from_lean_deg": "FAR_FROM_LEAN_DEG",
    "lean_clears_floor": "LEAN_CLEARS_FLOOR",
    "relook_after_miss": "RELOOK_AFTER_MISS",
    "lean_floor_clear_m": "LEAN_FLOOR_CLEAR_M",
    "rim_grip_depth_m": "RIM_GRIP_DEPTH_M",
    "rim_swing_share": "RIM_SWING_SHARE",
    "held_turns": "HELD_TURNS",
    "base_keep_out_m": "BASE_KEEP_OUT_M",
    "miss_steps_along_m": "MISS_STEPS_ALONG_M",
    "step_after_sideways_push": "STEP_AFTER_SIDEWAYS_PUSH",
    "close_across_long": "CLOSE_ACROSS_LONG",
    "other_object_near_m": "OTHER_OBJECT_NEAR_M",
    "held_by_camera": "HELD_BY_CAMERA",
    "held_while_still": "HELD_WHILE_STILL",
    "descent_lean_retries": "DESCENT_LEAN_RETRIES",
}

_UNSET = object()


def _defined(config: ModuleType, table: dict) -> dict:
    out = {}
    for key, name in table.items():
        value = getattr(config, name, _UNSET)
        if value is not _UNSET:
            out[key] = value
    return out


def gripper_box(config: ModuleType):
    """
    The box the whole gripper stays inside, flange to jaw tips (envelope.py),
    from GRIPPER_BOX, GRIPPER_BODY_RADIUS_M, GRIPPER_BASE_CLEAR_M and BASE_XY.
    None: the config sets no box.
    """
    box = getattr(config, "GRIPPER_BOX", None)
    if not box:
        return None
    from mrcf import envelope

    return envelope.GripperBox.from_config(
        box, config.JAW_CLOSED_POINT_M[2] + config.JAW_TIP_BELOW_GRASP_M,
        getattr(config, "GRIPPER_BODY_RADIUS_M", 0.05),
        base_clear_m=getattr(config, "GRIPPER_BASE_CLEAR_M", 0.0),
        base_xy=getattr(config, "BASE_XY", (0.0, 0.0)))


def rig_kwargs(config: ModuleType) -> dict:
    """
    RigEnv._init_rig's keywords from config: each of RIG_SETTING_FOR's the
    config defines, the sentry's zone (MOTION_GUARD_ZONE when MOTION_GUARD
    is on), the grasp correction (GRASP_CORRECTION_PATH's table) and the
    gripper's box (gripper_box), each when the config has it; and the grasp
    facts not yet checked on the rig (arm_facts.unchecked).
    """
    out = _defined(config, RIG_SETTING_FOR)
    if hasattr(config, "MOTION_GUARD_ZONE"):
        out["motion_guard_zone"] = config.MOTION_GUARD_ZONE if getattr(config, "MOTION_GUARD", False) else None
    if hasattr(config, "GRASP_CORRECTION_PATH"):
        from mrcf.grasp_correction import GraspCorrection
        out["grasp_correction"] = GraspCorrection.load(config.GRASP_CORRECTION_PATH)
    box = gripper_box(config)
    if box is not None:
        out["gripper_box"] = box
    from mrcf import arm_facts
    out["unchecked_facts"] = arm_facts.unchecked(config)
    return out


def camera_kwargs(config: ModuleType, overrides: dict) -> dict:
    """
    RealRobotEnv's keywords from config: `sources`, the cameras it opens
    ({name: index or URL}, those with a source), and `capture_sizes`. Takes
    overhead_camera_width/_height out of overrides (a tool's capture size for
    the overhead camera: tools/calibrate_overhead_camera.py asks for
    1280x720), and overhead_camera_fps (the camera keeps its own rate).
    """
    sizes = {name: cam.size for name, cam in config.CAMERAS.items()}
    width = overrides.pop("overhead_camera_width", None)
    height = overrides.pop("overhead_camera_height", None)
    overrides.pop("overhead_camera_fps", None)
    if config.OVERHEAD_CAMERA_NAME is not None:
        sizes[config.OVERHEAD_CAMERA_NAME] = (width if width is not None else config.OVERHEAD_CAMERA_W,
                                              height if height is not None else config.OVERHEAD_CAMERA_H)
    return {"sources": {name: cam.source for name, cam in config.CAMERAS.items() if cam.source is not None},
            "capture_sizes": sizes}


def gripper_spec(config: ModuleType, **overrides):
    """
    policies.parallel_jaw.GripperSpec from config: each of SPEC_SETTING_FOR's
    the config defines; lift_clearance_m from LIFT_HEIGHT less GRASP_HEIGHT_M;
    the wrist camera (CAMERA_NAME) only once its lens is calibrated
    (GRIPPER_CAM_K): before that, picks line up with the overhead camera.
    Keyword overrides win.
    """
    from mrcf.policies.parallel_jaw import GripperSpec

    spec = _defined(config, SPEC_SETTING_FOR)
    if hasattr(config, "LIFT_HEIGHT") and hasattr(config, "GRASP_HEIGHT_M"):
        spec["lift_clearance_m"] = config.LIFT_HEIGHT - config.GRASP_HEIGHT_M
    if "miss_steps_along_m" in spec:
        spec["miss_steps_along_m"] = tuple(spec["miss_steps_along_m"])
    spec["camera"] = (config.CAMERA_NAME if getattr(config, "GRIPPER_CAM_K", None) is not None
                      else None)
    spec.update(overrides)
    return GripperSpec(**spec)


__all__ = ["RIG_SETTING_FOR", "SPEC_SETTING_FOR", "rig_kwargs", "camera_kwargs", "gripper_box", "gripper_spec"]
