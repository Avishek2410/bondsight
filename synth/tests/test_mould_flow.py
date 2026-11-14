import numpy as np
import pytest

from wirebond_synth.defects import InjectionConfig, plan_unit_defects
from wirebond_synth.layout import make_layout
from wirebond_synth.render import render_unit

L32 = make_layout(32, seed=2)


def _geom(w):
    a = np.array([w.pad.x, w.pad.y]); b = np.array([w.lead.x, w.lead.y])
    u = (b - a) / np.linalg.norm(b - a)
    return np.array([-u[1], u[0]]), float(np.linalg.norm(b - a))


def test_flow_off_changes_nothing():
    a = plan_unit_defects(L32, 3, InjectionConfig("random", p=0.3), 7)
    b = plan_unit_defects(L32, 3, InjectionConfig("random", p=0.3, flow_p=0.0), 7)
    assert a == b


def test_flow_sweeps_follow_one_direction():
    d = plan_unit_defects(L32, 0, InjectionConfig("none", flow_p=1.0), 7)
    assert len(d) >= len(L32.wires) // 2 and all(x.cls == "wire_sweep" for x in d)
    assert len({x.params["flow_angle"] for x in d}) == 1
    f = np.array([np.cos(d[0].params["flow_angle"]), np.sin(d[0].params["flow_angle"])])
    for x in d:
        perp, _ = _geom(L32.wires[x.wire])
        assert x.params["side"] * (perp @ f) > 0


def test_flow_bends_crossing_wires_most():
    d = plan_unit_defects(L32, 0, InjectionConfig("none", flow_p=1.0), 7)
    f = np.array([np.cos(d[0].params["flow_angle"]), np.sin(d[0].params["flow_angle"])])
    rel = [x.params["amplitude_px"] / _geom(L32.wires[x.wire])[1] for x in d]
    cross = [abs(_geom(L32.wires[x.wire])[0] @ f) for x in d]
    ratio = np.array(rel) / np.array(cross)          # bend/length is proportional to crossing
    assert np.allclose(ratio, ratio[0]) and max(cross) > 3 * min(cross)


def test_flow_labels_by_reject_ratio():
    cfg = InjectionConfig("none", flow_p=1.0, flow_strength=30.0, sweep_reject=0.05)
    d = plan_unit_defects(L32, 0, cfg, 7)
    for x in d:
        ratio = x.params["amplitude_px"] / _geom(L32.wires[x.wire])[1]
        assert x.params["labelled"] == (ratio >= 0.05)
    assert any(x.params["labelled"] for x in d) and not all(x.params["labelled"] for x in d)


def test_flow_keeps_existing_defects():
    cfg = InjectionConfig("repetitive", wire=5, cls="missing_wire", flow_p=1.0)
    d = plan_unit_defects(L32, 0, cfg, 7)
    assert [x.cls for x in d if x.wire == 5] == ["missing_wire"]


def test_unlabelled_sweep_draws_but_no_box():
    d = [x for x in plan_unit_defects(L32, 0, InjectionConfig("none", flow_p=1.0), 7)
         if not x.params["labelled"]][:1]
    good = render_unit(L32, [], "top", 1)[0]
    img, boxes = render_unit(L32, d, "top", 1)
    assert boxes == [] and not np.array_equal(good, img)


def test_flow_validation():
    with pytest.raises(ValueError):
        plan_unit_defects(L32, 0, InjectionConfig("none", flow_p=2.0), 7)
    with pytest.raises(ValueError):
        plan_unit_defects(L32, 0, InjectionConfig("none", flow_p=1.0, sweep_reject=0.0), 7)
