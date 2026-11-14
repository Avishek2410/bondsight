from wirebond_synth.layout import Layout, make_layout


def test_layout_counts_and_determinism():
    a, b = make_layout(32, seed=1), make_layout(32, seed=1)
    assert a == b and len(a.wires) == 32
    x0, y0, x1, y1 = a.die
    for w in a.wires:
        assert x0 <= w.pad.x <= x1 and y0 <= w.pad.y <= y1
        assert not (x0 <= w.lead.x <= x1 and y0 <= w.lead.y <= y1)
        assert 0 <= w.lead.x <= a.size and 0 <= w.lead.y <= a.size
    assert Layout.from_dict(a.to_dict()) == a


def test_layout_different_seed_differs():
    assert make_layout(32, 1) != make_layout(32, 2)


def test_layout_extremes_build():
    assert len(make_layout(1, 3).wires) == 1
    assert len(make_layout(64, 3).wires) == 64
