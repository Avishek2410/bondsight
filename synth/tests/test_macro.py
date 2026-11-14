import time

import numpy as np
import pytest

from conftest import plan_one
from wirebond_synth.defects import DefectSpec
from wirebond_synth.layout import make_layout, scale_layout
from wirebond_synth.macro import STYLES, render_macro, scale_defect
from wirebond_synth.render import DEFAULT_STYLE, render_unit


def test_scale_layout_multiplies_geometry(L):
    S = scale_layout(L, 3.0)
    assert S.size == L.size * 3
    assert S.die == pytest.approx(tuple(v * 3 for v in L.die))
    w, s = L.wires[5], S.wires[5]
    assert (s.pad.x, s.pad.y, s.pad.w) == pytest.approx((w.pad.x * 3, w.pad.y * 3, w.pad.w * 3))
    assert (s.lead.w, s.lead.h) == pytest.approx((w.lead.w * 3, w.lead.h * 3))
    assert s.loop_height == pytest.approx(w.loop_height * 3)
    assert s.diameter_px == pytest.approx(w.diameter_px * 3)


def test_scale_defect_scales_pixel_params_only():
    d = DefectSpec(2, "wire_sweep", {"amplitude_px": 10.0, "side": 1})
    s = scale_defect(d, 2.5)
    assert s.params == {"amplitude_px": 25.0, "side": 1} and s.wire == 2
    b = DefectSpec(1, "broken_wire", {"t": 0.5, "dangle_px": 4.0})
    assert scale_defect(b, 2.0).params == {"t": 0.5, "dangle_px": 8.0}


def test_default_style_keeps_old_renders_identical(L):
    a = render_unit(L, [plan_one("broken_wire")], "top", 3)[0]
    b = render_unit(L, [plan_one("broken_wire")], "top", 3, style=DEFAULT_STYLE)[0]
    assert np.array_equal(a, b)


def test_dangling_break_moves_pixels_but_box_still_covers_it(L):
    base = DefectSpec(4, "broken_wire", {"t": 0.5})
    dang = DefectSpec(4, "broken_wire", {"t": 0.5, "dangle_px": 12.0, "dangle_side": 1})
    good = render_unit(L, [], "top", 1)[0].astype(int)
    a = render_unit(L, [base], "top", 1)[0].astype(int)
    b, boxes = render_unit(L, [dang], "top", 1)
    assert np.abs(a - b.astype(int)).sum() > 0
    ys, xs = np.nonzero(np.abs(b.astype(int) - good).sum(axis=2) > 30)
    bx = boxes[0]
    inside = (xs >= bx.x0 - 2) & (xs <= bx.x1 + 2) & (ys >= bx.y0 - 2) & (ys <= bx.y1 + 2)
    assert inside.mean() == 1.0


@pytest.mark.parametrize("view", ["top", "angled"])
def test_macro_crop_is_640_and_keeps_focus_defect(view):
    lay = make_layout(32, seed=4)
    d = [DefectSpec(9, "broken_wire", {"t": 0.5})]
    for seed in range(6):
        img, boxes, info = render_macro(lay, d, view, seed)
        assert img.shape == (640, 640, 3) and img.dtype == np.uint8
        assert info["focus_wire"] == 9 and 2.0 <= info["zoom"] <= 4.0
        assert [b.cls for b in boxes if b.wire == 9] == ["broken_wire"]
        for b in boxes:
            assert 0 <= b.x0 < b.x1 <= 639 and 0 <= b.y0 < b.y1 <= 639
        # the defect is big now: its box is far larger than on the 640 whole-unit image
        whole = render_unit(lay, d, view, seed)[1][0]
        big = [b for b in boxes if b.wire == 9][0]
        assert (big.x1 - big.x0) * (big.y1 - big.y0) > 3 * (whole.x1 - whole.x0) * (whole.y1 - whole.y0)


def test_macro_good_unit_has_no_boxes_and_is_deterministic():
    lay = make_layout(16, seed=2)
    a, ba, ia = render_macro(lay, [], "top", 11)
    b, bb, ib = render_macro(lay, [], "top", 11)
    assert ba == [] and np.array_equal(a, b) and ia == ib


def test_macro_styles_vary_metal_and_background():
    lay = make_layout(16, seed=2)
    metals = {render_macro(lay, [], "top", s)[2]["metal"] for s in range(30)}
    backs = {render_macro(lay, [], "top", s)[2]["style"] for s in range(30)}
    assert metals == {"gold", "al"}
    assert len(backs) >= 3 and backs <= set(STYLES)


def test_aluminium_wire_is_silver_not_gold():
    lay = make_layout(8, seed=2)
    for s in range(40):
        img, _, info = render_macro(lay, [], "top", s, augment_strength=0.0)
        if info["metal"] == "al":
            break
    b, g, r = (img.reshape(-1, 3).astype(int)).T
    goldish = ((r - b) > 90) & (r > 150)          # strongly yellow pixels
    assert goldish.mean() < 0.002


def test_macro_is_fast_enough():
    lay = make_layout(32, seed=4)
    t = time.perf_counter()
    for s in range(3):
        render_macro(lay, [DefectSpec(3, "wire_sweep", {"amplitude_px": 10, "side": 1})], "angled", s)
    assert (time.perf_counter() - t) / 3 < 1.5


def test_generate_macro_dataset(tmp_path):
    import json

    from wirebond_synth.cli import main
    from wirebond_synth.dataset import generate
    from wirebond_synth.defects import InjectionConfig
    s = generate(tmp_path / "m", 4, 16, 3, InjectionConfig("random", p=0.3), macro=True,
                 fmt="jpg")
    cfg = json.loads((tmp_path / "m/config.json").read_text())
    assert cfg["macro"] is True and s["units"] == 4
    coco = json.loads((tmp_path / "m/annotations/top.json").read_text())
    assert len(coco["images"]) == 4 and len(coco["annotations"]) == sum(s["boxes"]["top"].values())
    side = json.loads((tmp_path / "m/units/u00000.json").read_text())
    assert set(side["macro"]) == {"top", "angled"} and "zoom" in side["macro"]["top"]
    assert main(["generate", "--out", str(tmp_path / "c"), "--units", "2", "--wires", "8",
                 "--macro", "--format", "jpg"]) == 0
    assert json.loads((tmp_path / "c/config.json").read_text())["macro"] is True
