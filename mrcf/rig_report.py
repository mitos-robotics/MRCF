"""
rig_report.py — what an arm brings to a job, said before anything runs: the
arm (plugged in or not), its gripper, its cameras (which are plugged in,
which calibrated), its other sensors, and whether the pictures it keeps are
guarded. Then, for a deployment, what it is: its layout's state and its rules.

From a factory's side (user, 2026-11-17: "think about designing the
framework from the perspective of a factory deploying an arm to a new
automation - they need to set up the arm, sensors then create policies to do
the job"): run.py shows this between choosing the arm and choosing the job,
and the flight recorder keeps it with each run (session.json), so what a run
had to work with is on its record. Nothing here connects to or opens
anything: the arm says whether it is there (its present()), the cameras are
counted by name (devices.py), the rest is the arm's config.

    report(arm)            {arm, gripper, cameras, sensors, privacy, ...}
    report(arm, config)    ... and the deployment's
    lines(report)          what run.py prints
"""

from __future__ import annotations

from pathlib import Path


def _camera(name: str, cam, rc) -> dict:
    """One camera: where it is, its source and size, and whether (and how) it is calibrated."""
    if name == getattr(rc, "OVERHEAD_CAMERA_NAME", None):
        calibrated = getattr(rc, "OVERHEAD_CAM_MODEL", None) is not None
        how = "pose and lens (OVERHEAD_CAM_MODEL)" if calibrated else "not calibrated: tools/calibrate_camera_by_hand.py"
        role = "overhead"
    elif name == getattr(rc, "CAMERA_NAME", None):
        lens = getattr(rc, "GRIPPER_CAM_K", None) is not None
        pose = getattr(rc, "GRIPPER_CAM_POSE", None) is not None
        calibrated = lens and pose
        how = ("lens and place on the gripper" if calibrated else
               "lens only: tools/so101/calibrate_gripper_pose.py" if lens else "not calibrated")
        role = "wrist"
    else:
        calibrated = getattr(cam, "model", None) is not None or getattr(cam, "K", None) is not None
        how = "calibrated" if calibrated else "not calibrated: tools/calibrate_camera_by_hand.py --camera " + name
        role = "on the gripper" if getattr(cam, "mount", "fixed") == "gripper" else "fixed"
    return {"name": name, "role": role, "mount": getattr(cam, "mount", None), "source": getattr(cam, "source", None),
            "size": list(getattr(cam, "size", ()) or ()), "calibrated": calibrated, "calibration": how}


def report(arm, config=None, listed: "dict[int, str] | None" = None) -> dict:
    """
    What `arm` (a backend package: NAME, config, present()) brings, and `config`
    (a deployment's config module) is. listed: the computer's cameras
    ({index: name}, devices.video_devices()), listed here when None.
    """
    from . import devices, privacy

    rc = arm.config
    out: dict = {"arm": {"name": str(arm.__name__).rsplit(".", 1)[-1], "label": getattr(arm, "NAME", "?")}}
    check = getattr(arm, "present", None)
    try:
        found, how = check() if callable(check) else (None, "cannot tell whether it is plugged in")
    except Exception as exc:                  # noqa: BLE001 — said, not raised
        found, how = None, f"its check failed — {type(exc).__name__}: {exc}"
    out["arm"].update(found=found, how=how)
    from . import arm_facts
    out["gripper"] = {"opens_mm": round(float(getattr(rc, "GRIPPER_MAX_OPEN_M", 0.0)) * 1000, 1),
                      "unchecked": list(arm_facts.unchecked(rc))}
    cams = getattr(rc, "CAMERAS", None) or {}
    out["cameras"] = [_camera(n, c, rc) for n, c in cams.items()]
    try:
        listed = devices.video_devices() if listed is None else listed
    except RuntimeError:
        listed = {}
    sources = {n: c.source for n, c in cams.items() if getattr(c, "source", None) is not None}
    try:
        devices.check_sources(sources, listed)
        plugged = None if not listed else True
        why = ""
    except devices.PersonalCamera as exc:
        plugged, why = False, str(exc)
    out["camera_check"] = {"plugged_in": plugged, "why": why,
                           "rig_cameras_listed": sorted(devices.rig_cameras(listed).values()),
                           "own_cameras_listed": sorted(n for n in listed.values() if privacy.personal(n))}
    out["sensors"] = [{"name": n, "kind": getattr(s, "kind", "?"), "unit": getattr(s, "unit", "")}
                      for n, s in (getattr(rc, "SENSORS", None) or {}).items()]
    out["privacy"] = {"faces_covered": privacy.has_face_model(), "pictures_kept": privacy.keep_pictures()}
    if config is not None:
        out["deployment"] = deployment(config)
    return out


def deployment(config) -> dict:
    """A deployment's config, said: its name and arm, its rules, its layout's kept state."""
    from . import profiles

    name = config.__name__.rsplit(".", 2)[-2] if config.__name__.count(".") >= 1 else config.__name__
    rules = profiles.settings(getattr(config, "PROFILE", None))
    kept = {}
    for key in ("SURFACE_PATH", "PLACES_PATH", "MEMORY_PATH", "CONTAINERS_PATH"):
        path = config.__dict__.get(key)
        if path is not None:
            kept[key] = {"path": str(path), "there": Path(path).is_file()}
    first = (config.__doc__ or "").strip().splitlines()
    return {"name": name, "arm": getattr(config, "ARM_NAME", None), "what": _what(first),
            "rules": sorted(rules), "kept": kept}


def _what(doc_lines: list) -> str:
    """A deployment's config docstring's opening sentence, without its 'config.py — the X deployment:' lead."""
    text = " ".join(line.strip() for line in doc_lines[:4])
    if "—" in text:
        text = text.split("—", 1)[1].strip()
    if ":" in text[:40]:
        text = text.split(":", 1)[1].strip()
    return text.split(". ")[0].split(", by plain English")[0].rstrip(".")


def lines(rep: dict) -> list:
    """The report, as run.py prints it."""
    arm = rep["arm"]
    state = {True: "plugged in", False: "NOT found", None: "?"}[arm.get("found")]
    out = [f"The {arm['label']} ({arm['name']}): {state} — {arm.get('how', '')}",
           f"  gripper   parallel jaws, opening {rep['gripper']['opens_mm']:.0f} mm"]
    from . import arm_facts
    unchecked = rep["gripper"].get("unchecked") or []
    if unchecked:
        tools = sorted({arm_facts.CHECKED_BY.get(n, "") for n in unchecked} - {""})
        out.append(f"            NOT CHECKED on the rig: {', '.join(unchecked)} — picks are refused until "
                   + (" / ".join(tools) if tools else "it is checked"))
    for cam in rep["cameras"]:
        src = cam["source"]
        out.append(f"  camera    {cam['name']:<10} {cam['role']:<9} source {src!r:<6} "
                   f"{'x'.join(str(v) for v in cam['size']):<9} {cam['calibration']}")
    chk = rep["camera_check"]
    if chk["plugged_in"] is False:
        out.append(f"            NOT ALL PLUGGED IN — {chk['why']}")
    elif chk["plugged_in"] is True:
        out.append(f"            plugged in: {', '.join(chk['rig_cameras_listed'])}"
                   + (f"; this computer's own, never opened: {', '.join(chk['own_cameras_listed'])}"
                      if chk["own_cameras_listed"] else ""))
    else:
        out.append("            (this computer cannot list its cameras: whether they are plugged in is not known)")
    if rep["sensors"]:
        for s in rep["sensors"]:
            out.append(f"  sensor    {s['name']:<10} {s['kind']} ({s['unit']})")
    else:
        out.append("  sensors   none beyond the joints and the gripper")
    priv = rep["privacy"]
    out.append("  pictures  " + ("faces covered" if priv["faces_covered"] else
                                 "faces NOT covered (python tools/fetch_face_model.py)")
               + ", calibrated cameras kept to the work area"
               + ("" if priv["pictures_kept"] else "; only incidents' frames kept (MRCF_KEEP_PICTURES=0)"))
    return out


def deployment_line(dep: dict) -> str:
    """One deployment, in a line: what it is, how many rules of its own, and its layout's state."""
    kept = dep.get("kept", {})
    surface = kept.get("SURFACE_PATH")
    layout = ("" if surface is None else
              "; its surface touched" if surface["there"] else "; its surface: the arm's TABLE_Z")
    what = dep["what"] or dep["name"]
    what = what if len(what) <= 70 else what[:67].rsplit(" ", 1)[0] + "..."
    return f"{what} ({len(dep['rules'])} rule(s) of its own{layout})"
