from wirebond_synth.classes import CLASSES
from wirebond_synth.defects import InjectionConfig, plan_unit_defects


def test_none_mode_no_defects(L):
    assert plan_unit_defects(L, 0, InjectionConfig("none"), 7) == []


def test_random_p1_every_wire_once(L):
    d = plan_unit_defects(L, 0, InjectionConfig("random", p=1.0), 7)
    assert sorted(x.wire for x in d) == list(range(len(L.wires)))
    assert all(x.cls in CLASSES for x in d)


def test_repetitive_fraction(L):
    cfg = InjectionConfig("repetitive", wire=5, cls="wire_sweep", fraction=0.8, start_unit=10)
    hits = [any(x.wire == 5 and x.cls == "wire_sweep" for x in plan_unit_defects(L, u, cfg, 7))
            for u in range(200)]
    assert not any(hits[:10]) and 0.7 <= sum(hits[10:]) / 190 <= 0.9


def test_repetitive_overrides_random_on_same_wire(L):
    cfg = InjectionConfig("repetitive", wire=5, cls="wire_sweep", fraction=1.0, random_p=1.0)
    d = plan_unit_defects(L, 0, cfg, 7)
    assert [x.cls for x in d if x.wire == 5] == ["wire_sweep"]
    assert len({x.wire for x in d}) == len(d)


def test_planner_deterministic(L):
    cfg = InjectionConfig("random", p=0.3)
    assert plan_unit_defects(L, 3, cfg, 7) == plan_unit_defects(L, 3, cfg, 7)


def test_bad_config_rejected(L):
    import pytest
    with pytest.raises(ValueError):
        plan_unit_defects(L, 0, InjectionConfig("repetitive", wire=99, cls="wire_sweep"), 7)
    with pytest.raises(ValueError):
        plan_unit_defects(L, 0, InjectionConfig("random", p=1.5), 7)
