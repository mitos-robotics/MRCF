"""fiducials: tags found where they are, the right one of a tag's two poses, jaw marks."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs OpenCV: pip install -e '.[measure]'")

from mrcf.perception import fiducials as F

K = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])
D = np.zeros(5)


def _rot(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg))[0]


def _facing(axis, deg):
    """A tag turned deg about axis from facing the camera squarely."""
    return _rot(axis, deg) @ _rot([1, 0, 0], 180)


def _render_tag(tag_id, size_m, R, t, shape=(720, 1280)):
    """A tag (with its white margin) on a mid-grey frame, plus its true corner pixels."""
    cells = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(F.TAG_DICTIONARY), tag_id, 400)
    img = cv2.copyMakeBorder(cells, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)   # 1 cell margin
    margin = size_m * 10 / 8 / 2
    outer = np.array([[-margin, margin, 0], [margin, margin, 0], [margin, -margin, 0], [-margin, -margin, 0]])
    px = cv2.projectPoints(outer, cv2.Rodrigues(R)[0], t, K, D)[0].reshape(4, 2).astype(np.float32)
    # map pixel centres: the image's outer edge is at -0.5 .. 499.5
    src = np.float32([[-0.5, -0.5], [499.5, -0.5], [499.5, 499.5], [-0.5, 499.5]])
    H = cv2.getPerspectiveTransform(src, px - 0.0)
    frame = np.full(shape, 110, np.uint8)
    warped = cv2.warpPerspective(img, H, shape[::-1], flags=cv2.INTER_AREA, borderValue=0)
    mask = cv2.warpPerspective(np.full_like(img, 255), H, shape[::-1], flags=cv2.INTER_NEAREST) > 0
    frame[mask] = warped[mask]
    truth = cv2.projectPoints(F.tag_object_points(size_m), cv2.Rodrigues(R)[0], t, K, D)[0].reshape(4, 2)
    return frame, truth


def test_detect_tags_finds_the_tag_where_it_is():
    R = _facing([1, 0.3, 0], 35)
    frame, truth = _render_tag(7, 0.030, R, np.array([0.05, -0.02, 0.55]))
    found = F.detect_tags(frame)
    assert set(found) == {7}
    assert np.abs(found[7] - truth).max() < 0.5


def test_a_small_tag_is_found_with_the_default_upscale():
    R = _facing([1, 0, 0], 50)
    frame, _ = _render_tag(12, 0.024, R, np.array([0.0, 0.0, 0.65]))   # ~30 px across, steeply tilted
    assert 12 in F.detect_tags(frame)


def test_tag_pose_recovers_where_the_tag_is():
    R, t = _facing([0.2, 1, 0], 30), np.array([-0.04, 0.03, 0.5])
    frame, _ = _render_tag(3, 0.032, R, t)
    pose = F.tag_poses(F.detect_tags(frame)[3], 0.032, K, D)[0]
    assert np.linalg.norm(pose.t - t) < 0.003
    assert F.rotation_angle_deg(pose.R, R) < 2.0


def test_a_tag_seen_square_on_still_gets_its_pose():
    # OpenCV's square-tag IPPE flips this case 180 degrees; the pose must still be right
    R, t = _facing([1, 0, 0], 0), np.array([0.0, 0.0, 0.6])
    frame, _ = _render_tag(5, 0.024, R, t)
    poses = F.tag_poses(F.detect_tags(frame)[5], 0.024, K, D)
    assert poses and F.rotation_angle_deg(poses[0].R, R) < 2.0 and poses[0].err_px < 1.0
    # and exactly on the optical axis, tilted (IPPE_SQUARE returns NaN there)
    R = _facing([1, 0, 0], 40)
    frame, _ = _render_tag(5, 0.024, R, t)
    poses = F.tag_poses(F.detect_tags(frame)[5], 0.024, K, D)
    assert poses and all(np.isfinite(p.R).all() for p in poses)
    assert F.rotation_angle_deg(poses[0].R, R) < 3.0


def test_pick_pose_takes_the_solution_that_agrees_with_the_expected_turn():
    R, t = _facing([1, 0, 0], 40), np.array([0.03, 0.02, 0.6])
    frame, _ = _render_tag(5, 0.024, R, t)
    poses = F.tag_poses(F.detect_tags(frame)[5], 0.024, K, D)
    assert len(poses) == 2
    true_one = min(poses, key=lambda p: F.rotation_angle_deg(p.R, R))
    other = max(poses, key=lambda p: F.rotation_angle_deg(p.R, R))
    assert F.rotation_angle_deg(true_one.R, other.R) > 20      # a real choice, not two copies
    # a prior a few degrees off still picks the right one, whichever fits best
    assert F.pick_pose(poses, _rot([0, 1, 1], 4) @ R) is true_one
    assert F.pick_pose(poses, other.R) is other
    assert F.pick_pose(poses) is poses[0]


def test_position_with_known_rotation_finds_the_tag_from_a_far_guess():
    R, t = _facing([1, 0.2, 0], 50), np.array([0.03, -0.02, 0.45])
    frame, _ = _render_tag(12, 0.024, R, t)
    found, err = F.position_with_known_rotation(F.detect_tags(frame)[12], 0.024, R, K, D,
                                                t_guess=t + [0.02, -0.015, 0.03])
    assert np.linalg.norm(found - t) < 0.002 and err < 0.5


def test_layout_pose_from_several_tags():
    # three tags on one plane, ~0.3 m apart, the plane turned and pushed away
    s = 0.026
    layout = {0: F.tag_object_points(s) + [-0.15, -0.1, 0], 1: F.tag_object_points(s) + [0.15, -0.1, 0],
              2: F.tag_object_points(s) + [0.0, 0.12, 0]}
    R, t = _facing([1, 0, 0], 30), np.array([0.0, 0.0, 0.6])
    frame = np.full((720, 1280), 110, np.uint8)
    for i in layout:
        # each tag's own pose = the layout's pose applied to its offset
        ti = R @ (layout[i].mean(0)) + t
        f, _ = _render_tag(i, s, R, ti)
        frame = np.where(f != 110, f, frame)
    T, n, err = F.layout_pose(F.detect_tags(frame), layout, K, D)
    assert n == 3 and err < 0.5
    assert np.linalg.norm(T[:3, 3] - t) < 0.003
    assert F.rotation_angle_deg(T[:3, :3], R) < 1.0


def _disc(frame, centre, r, dot=True, value=250):
    cv2.circle(frame, centre, r, value, -1, cv2.LINE_AA)
    if dot:
        cv2.circle(frame, centre, int(r * 0.4), 40, -1, cv2.LINE_AA)


def test_find_bullseyes_finds_the_marks_and_nothing_else():
    frame = np.full((480, 640), 30, np.uint8)                 # dark jaws
    _disc(frame, (250, 380), 22)
    _disc(frame, (470, 385), 20)
    _disc(frame, (100, 100), 20, dot=False)                    # a white disc with no dot
    frame[0:60, 400:640] = 252                                 # a light board square
    cv2.circle(frame, (520, 30), 8, 40, -1)                    # a dark spot on it, not a disc
    marks = F.find_bullseyes(frame)
    assert [tuple(np.round(m.centre)) for m in marks] == [(250, 380), (470, 385)]


def test_a_bullseye_touching_something_bright_on_one_side_is_still_found():
    frame = np.full((480, 640), 30, np.uint8)
    frame[300:480, 495:640] = 215                              # a light square beside the jaw
    _disc(frame, (474, 383), 22, value=245)
    marks = F.find_bullseyes(frame)
    assert len(marks) == 1 and np.hypot(marks[0].centre[0] - 474, marks[0].centre[1] - 383) < 1.0
