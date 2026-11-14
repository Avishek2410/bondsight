from wirebond_synth.geometry import project
from wirebond_synth.layout import make_layout

from wirebond_inspector.detector import Finding
from wirebond_inspector.wiremap import position_key


def _box_at(x, y, half=5):
    return Finding("wire_sweep", 0.9, (x - half, y - half, x + half, y + half))


def _mid(w):
    return (w.pad.x + w.lead.x) / 2, (w.pad.y + w.lead.y) / 2


def test_maps_box_on_wire_to_its_index():
    lay = make_layout(24, seed=7)
    for i in (0, 5, 17):
        x, y = _mid(lay.wires[i])
        assert position_key(_box_at(x, y), "top", (640, 640), lay.to_dict()) == f"wire:{i}"


def test_angled_projection():
    # in the angled view the wire is drawn raised (loop height), so use a point on the loop
    import numpy as np
    from wirebond_synth.geometry import loop_profile
    lay = make_layout(24, seed=7)
    w = lay.wires[5]
    x, y = _mid(w)
    z = float(loop_profile(np.array([0.5]), w.loop_height)[0])
    px, py = project(x, y, z, "angled", lay.size)
    assert position_key(_box_at(px, py), "angled", (640, 640), lay.to_dict()) == "wire:5"


def test_scales_when_image_is_not_layout_size():
    lay = make_layout(24, seed=7)
    x, y = _mid(lay.wires[5])
    assert position_key(_box_at(2 * x, 2 * y), "top", (1280, 1280), lay.to_dict()) == "wire:5"


def test_grid_without_layout():
    assert position_key(_box_at(790, 10), "top", (800, 800), None) == "cell:r0c7"
    assert position_key(_box_at(5, 795), "top", (800, 800), None) == "cell:r7c0"


def test_true_synthetic_boxes_map_to_their_wire():
    from wirebond_synth.defects import InjectionConfig, plan_unit_defects
    from wirebond_synth.render import render_unit
    lay = make_layout(24, seed=11)
    cfg = InjectionConfig("random", p=0.3)
    hits = total = 0
    for u in range(15):
        defects = plan_unit_defects(lay, u, cfg, 11)
        for view in ("top", "angled"):
            _, boxes = render_unit(lay, defects, view, u)
            for b in boxes:
                f = Finding(b.cls, 1.0, (b.x0, b.y0, b.x1, b.y1))
                hits += position_key(f, view, (640, 640), lay.to_dict()) == f"wire:{b.wire}"
                total += 1
    assert total > 50
    assert hits / total >= 0.90, f"{hits}/{total}"   # measured ~95% (top ~99%, angled ~92%)
