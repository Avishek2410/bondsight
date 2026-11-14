import numpy as np

from wirebond_synth.defects import DefectSpec
from wirebond_synth.geometry import project, wire_path


def max_dev(seg, a, b):
    d = (b - a) / np.linalg.norm(b - a)
    rel = seg - a
    return float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max())


def test_good_wire_top_is_straight(L):
    p = wire_path(L, L.wires[3], None, "top")
    s = p.segments[0]
    assert len(p.segments) == 1 and max_dev(s, s[0], s[-1]) < 1.0


def test_sweep_amplitude_top(L):
    d = DefectSpec(3, "wire_sweep", {"amplitude_px": 20.0, "side": 1})
    s = wire_path(L, L.wires[3], d, "top").segments[0]
    assert 18.0 <= max_dev(s, s[0], s[-1]) <= 22.0


def test_broken_has_two_segments(L):
    p = wire_path(L, L.wires[3], DefectSpec(3, "broken_wire", {"t": 0.5}), "top")
    assert len(p.segments) == 2


def test_missing_has_nothing(L):
    p = wire_path(L, L.wires[3], DefectSpec(3, "missing_wire", {}), "top")
    assert p.segments == [] and p.ball is None and p.stitch is None


def test_sagging_lower_in_angled(L):
    good = wire_path(L, L.wires[3], None, "angled").segments[0]
    sag = wire_path(L, L.wires[3], DefectSpec(3, "sagging_loop", {"height_factor": 0.2}),
                    "angled").segments[0]
    # the loop rises toward smaller y in the angled view; a sagging loop rises less
    assert np.all(sag[:, 1] >= good[:, 1] - 1e-6) and (sag[:, 1] - good[:, 1]).max() > 10


def test_wire_short_on_edge_wire(L):
    for i in (0, len(L.wires) - 1):
        p = wire_path(L, L.wires[i], DefectSpec(i, "wire_short", {}), "top")
        assert len(p.segments) == 1


def test_wire_short_bows_toward_neighbour(L):
    i = 2
    s = wire_path(L, L.wires[i], DefectSpec(i, "wire_short", {}), "top").segments[0]
    mid = s[len(s) // 2]
    nb = [wire_path(L, L.wires[j], None, "top").segments[0] for j in (i - 1, i + 1)]
    dist = min(np.linalg.norm(n[len(n) // 2] - mid) for n in nb)
    assert dist < 0.3 * np.linalg.norm(nb[0][len(nb[0]) // 2] - nb[1][len(nb[1]) // 2])


def test_off_center_ball_moves_ball(L):
    w = L.wires[3]
    p = wire_path(L, w, DefectSpec(3, "off_center_ball", {"offset": 0.5, "angle": 0.0}), "top")
    assert abs(p.ball[0] - (w.pad.x + 0.5 * w.pad.w)) < 1e-6 and abs(p.ball[1] - w.pad.y) < 1e-6


def test_project_top_identity_and_angled_lift():
    assert project(100, 200, 50, "top", 640) == (100, 200)
    x, y = project(100, 320, 50, "angled", 640)
    assert x == 100 and abs(y - (320 - 50 * np.sin(np.pi / 4))) < 1e-9
