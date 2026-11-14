import numpy as np

from conftest import plan_one
from wirebond_synth.classes import CLASSES, VIEWS, VISIBILITY
from wirebond_synth.defects import DefectSpec
from wirebond_synth.layout import make_layout
from wirebond_synth.render import render_unit


def test_good_unit_no_boxes(L):
    img, boxes = render_unit(L, [], "top", seed=1)
    assert img.shape == (640, 640, 3) and img.dtype == np.uint8 and boxes == []


def test_view_visibility_table(L):
    for c in CLASSES:
        d = plan_one(c)
        for v in VIEWS:
            _, boxes = render_unit(L, [d], v, seed=1)
            assert ([b.cls for b in boxes] == [c]) == (v in VISIBILITY[c]), (c, v)
            assert all(b.wire == 4 for b in boxes)


def test_boxes_inside_image_for_all_classes():
    big = make_layout(64, seed=9)
    for c in CLASSES:
        for w in (0, 31, 63):
            for v in VIEWS:
                for b in render_unit(big, [plan_one(c, w, extreme=True)], v, 1)[1]:
                    assert 0 <= b.x0 < b.x1 <= 639 and 0 <= b.y0 < b.y1 <= 639
                    assert b.x1 - b.x0 >= 2 and b.y1 - b.y0 >= 2


def test_render_deterministic(L):
    a = render_unit(L, [], "angled", 5)[0]
    b = render_unit(L, [], "angled", 5)[0]
    assert np.array_equal(a, b)


def test_defect_changes_pixels(L):
    good = render_unit(L, [], "top", 1)[0]
    bad = render_unit(L, [DefectSpec(4, "wire_sweep", {"amplitude_px": 20, "side": 1})], "top", 1)[0]
    assert np.abs(good.astype(int) - bad.astype(int)).sum() > 0


def test_defect_pixels_inside_box(L):
    """Changed pixels of a defect lie inside its box. Ball defects also nudge the wire start
    (physically correct), so for them only the majority of the change must be inside."""
    for c, need in (("wire_sweep", 1.0), ("broken_wire", 1.0),
                    ("off_center_ball", 0.5), ("lifted_ball", 0.5)):
        good = render_unit(L, [], "top", 1)[0].astype(int)
        bad, boxes = render_unit(L, [plan_one(c, extreme=True)], "top", 1)
        ys, xs = np.nonzero(np.abs(bad.astype(int) - good).sum(axis=2) > 30)
        b = boxes[0]
        inside = (xs >= b.x0 - 2) & (xs <= b.x1 + 2) & (ys >= b.y0 - 2) & (ys <= b.y1 + 2)
        assert inside.mean() >= need, (c, inside.mean())
