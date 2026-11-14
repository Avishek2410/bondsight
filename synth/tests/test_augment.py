import numpy as np

from conftest import plan_one
from wirebond_synth.augment import augment
from wirebond_synth.render import render_unit


def test_strength_zero_identity(L):
    img, boxes = render_unit(L, [plan_one("wire_sweep")], "top", 1)
    out, ob = augment(img, boxes, seed=3, strength=0.0)
    assert np.array_equal(out, img) and ob == boxes


def test_augment_keeps_boxes_valid_and_deterministic(L):
    img, boxes = render_unit(L, [plan_one("broken_wire", 0)], "top", 1)
    a = augment(img, boxes, 3)
    b = augment(img, boxes, 3)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    for x in a[1]:
        assert 0 <= x.x0 < x.x1 <= 639 and 0 <= x.y0 < x.y1 <= 639


def test_augment_changes_image(L):
    img, boxes = render_unit(L, [], "top", 1)
    out = augment(img, boxes, 3)[0]
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)


def test_boxes_follow_shift(L):
    """A box must move with the image: the defect's pixels stay inside the moved box."""
    good, _ = render_unit(L, [], "top", 1)
    bad, boxes = render_unit(L, [plan_one("wire_sweep", extreme=True)], "top", 1)
    for s in range(5):
        g = augment(good, [], s)[0].astype(int)
        b_img, b_boxes = augment(bad, boxes, s)
        ys, xs = np.nonzero(np.abs(b_img.astype(int) - g).sum(axis=2) > 60)
        bx = b_boxes[0]
        inside = (xs >= bx.x0 - 3) & (xs <= bx.x1 + 3) & (ys >= bx.y0 - 3) & (ys <= bx.y1 + 3)
        assert inside.mean() > 0.95
