"""
shape_fit: an object's height and shape from its outlines in calibrated views —
right from views that see its height, and saying it cannot tell from views that
do not (2026-11-16: from over a 25 mm ball the SO-101's wrist camera saw 0.21 px
per mm of its height, and the picks planned for 150 mm closed in the air).
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mrcf.perception import plane_geometry as pg  # noqa: E402
from mrcf.perception import shape_fit as sf  # noqa: E402

K = np.array([[480.0, 0, 320], [0, 480.0, 240], [0, 0, 1]])
SHAPE = (480, 640)
AT = np.array([0.2727, 0.121])                    # the red ball, 2026-11-16
OUT = AT / np.linalg.norm(AT)
SIDE = np.array([-OUT[1], OUT[0]])


def look_at(cam, target, up=(0.0, 0.0, 1.0)):
    cam, target = np.asarray(cam, float), np.asarray(target, float)
    z = target - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]
    T[:3, 3] = cam
    return pg.PlaneView.from_camera_model({"K": K, "D": np.zeros(5), "T_world_cam": T, "plane_z_m": 0.0})


OVERHEAD = look_at((0.359, 0.036, 0.523), (0.30, 0.06, 0.0), up=(1.0, 0.0, 0.0))   # the rig's, roughly
HOVER = look_at((0.297, 0.152, 0.245), (AT[0], AT[1], 0.0))                         # 8 deg off straight down


def beside(back, z, side=0.0):
    return look_at(np.r_[AT - back * OUT + side * SIDE, z], np.r_[AT, 0.012])


def ball(d=25.0, xy=AT):
    u, v = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 60))
    r, c = d / 2, np.asarray(xy) * 1000
    return np.c_[c[0] + r * (np.sin(u) * np.cos(v)).ravel(), c[1] + r * (np.sin(u) * np.sin(v)).ravel(),
                 r - r * np.cos(u).ravel()]


def box(w=40.0, h=30.0, xy=AT):
    g, zz = np.linspace(-w / 2, w / 2, 12), np.linspace(0, h, 12)
    X, Y, Z = np.meshgrid(g, g, zz)
    c = np.asarray(xy) * 1000
    return np.c_[c[0] + X.ravel(), c[1] + Y.ravel(), Z.ravel()]


def bottle(xy=AT):
    """30 mm across and 70 tall, narrowing to a 20 mm neck over its top 20: no solid shape_fit tries."""
    c, pts = np.asarray(xy) * 1000, []
    for z in np.linspace(0, 70, 30):
        r = 15 if z < 50 else 15 - (z - 50) * 0.5
        a = np.linspace(0, 2 * np.pi, 40)
        pts.append(np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(40, z)])
    return np.vstack(pts)


def outline(view, pts, grow_px=0):
    m = np.zeros(SHAPE, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(view.project(pts).astype(np.int32)), 1)
    if grow_px:
        k = np.ones((2 * abs(grow_px) + 1,) * 2, np.uint8)
        m = cv2.dilate(m, k) if grow_px > 0 else cv2.erode(m, k)
    return m.astype(bool)


def fit(pts, views, grow=(0,), valid=None):
    """The fit, the first view taken as the overhead camera's at the scan (its outline the footprint)."""
    sils = [sf.Silhouette(v, [outline(v, pts, grow[i % len(grow)])], None if valid is None else valid[i],
                          name=f"v{i}", fixed=i == 0) for i, v in enumerate(views)]
    cs, _ = cv2.findContours(sils[0].masks[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sf.fit_shape(sils, sf.OutlineFootprint(views[0], max(cs, key=cv2.contourArea).reshape(-1, 2)))


@pytest.mark.parametrize("pts, height, profile", [(ball(), 25.0, "ball"), (box(), 30.0, "straight"),
                                                  (bottle(), 70.0, None)])
def test_from_beside_it_the_height_is_found_and_resolved(pts, height, profile):
    got = fit(pts, [OVERHEAD, beside(0.14, 0.13), beside(0.10, 0.10, 0.08)])
    assert got.resolved, got.why
    assert abs(got.height_mm - height) <= 2.0
    assert got.spread_mm[0] - 1 <= height <= got.spread_mm[1] + 1
    if profile:
        assert got.profile == profile
    assert min(got.px_per_mm[1:]) > 1.0 and got.px_per_mm[0] < 0.5


@pytest.mark.parametrize("pts", [ball(), box(), bottle()])
def test_from_above_alone_it_says_it_cannot_tell(pts):
    # The overhead camera and the wrist camera at the hover: neither sees height, and a
    # bottle's neck, unlike every solid, once fitted a confident 50 mm. Not resolved.
    got = fit(pts, [OVERHEAD, HOVER])
    assert not got.resolved
    assert "height" in got.why


def test_a_ragged_outline_still_gives_the_height_from_two_sides():
    # SAM's edges wander a pixel or so: one view's outline grown 1 px, another's shrunk.
    got = fit(box(), [OVERHEAD, beside(0.14, 0.13), beside(0.10, 0.10, 0.08)], grow=(0, 1, -1))
    assert got.resolved, got.why
    assert abs(got.height_mm - 30.0) <= 2.0


def test_outlines_that_disagree_more_leave_a_spread_that_still_holds_the_truth():
    # 2 px each way, 4 px between the two views: a flat 30 mm box and a 40 mm one with a
    # rounded top then fit almost as well. Not resolved, but bounded, with the truth inside.
    got = fit(box(), [OVERHEAD, beside(0.14, 0.13), beside(0.10, 0.10, 0.08)], grow=(0, 2, -2))
    assert abs(got.height_mm - 30.0) <= 2.0
    assert got.spread_mm[0] <= 30.0 <= got.spread_mm[1] <= 45.0
    assert got.spread_mm[1] - got.spread_mm[0] <= 15.0


def test_what_hides_part_of_it_counts_neither_way():
    # The top half of the side view hidden (the jaws, or something in front), and the
    # outline cut there: off the valid pixels, the cut does not drag the height down.
    view = beside(0.14, 0.13)
    whole = outline(view, ball())
    ys, _ = np.nonzero(whole)
    cut = np.ones(SHAPE, bool)
    cut[: int(np.percentile(ys, 30))] = False          # the top of its outline not seen
    sils = [sf.Silhouette(OVERHEAD, [outline(OVERHEAD, ball())], name="overhead", fixed=True),
            sf.Silhouette(view, [whole & cut], cut, name="side")]
    cs, _ = cv2.findContours(sils[0].masks[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fp = sf.OutlineFootprint(OVERHEAD, max(cs, key=cv2.contourArea).reshape(-1, 2))
    hidden = sf.fit_shape(sils, fp)
    unmasked = sf.fit_shape([sils[0], sf.Silhouette(view, [whole & cut], None, name="side")], fp)
    assert hidden.height_mm > unmasked.height_mm       # the cut taken for its top only where it is not masked
    assert hidden.spread_mm[1] >= 24.0


def test_the_outline_from_above_is_carried_to_where_each_solid_is_widest():
    # A box's outline from above is its top's on the far side and its foot's on the near:
    # carried to both and intersected, it is the box's own 40 mm, not the 47 mm the outline
    # spans on the surface.
    cs, _ = cv2.findContours(outline(OVERHEAD, box()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fp = sf.OutlineFootprint(OVERHEAD, max(cs, key=cv2.contourArea).reshape(-1, 2))
    centre, ring = fp.section(1.0, 30.0, "straight")
    extent = ring.max(0) - ring.min(0)
    assert np.all(np.abs(extent - 40.0) < 2.5)
    assert np.linalg.norm(centre / 1000.0 - AT) < 0.002
    _, at_surface = fp.section(1.0, 30.0, "dome")
    assert (at_surface.max(0) - at_surface.min(0)).max() > 43.0
    # A 150 mm straight object casting the same outline would be a pole a third as wide.
    _, pole = fp.section(1.0, 150.0, "straight")
    assert (pole.max(0) - pole.min(0)).max() < 20.0


def test_the_frame_of_a_view_can_be_moved():
    moved = sf.shifted(OVERHEAD, (0.003, -0.002))
    p = np.array([[272.7, 121.0, 10.0]])
    assert np.allclose(moved.project(p + [3.0, -2.0, 0.0]), OVERHEAD.project(p), atol=1e-6)


def test_no_outline_no_fit():
    sil = sf.Silhouette(OVERHEAD, [np.zeros(SHAPE, bool)])
    assert sf.fit_shape([sil], sf.Footprint.circle(AT, 0.025)) is None
