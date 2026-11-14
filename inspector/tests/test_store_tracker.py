import pytest
from PIL import Image

from wirebond_inspector.store import Store
from wirebond_inspector.tracker import RepetitiveTracker


def _shot(cam="top", img=True, error=None):
    return {"camera": cam, "view": cam, "source": "gen:u00000",
            "image": Image.new("RGB", (32, 32), (200, 10, 10)) if img else None, "error": error}


def _f(pos="wire:5", cls="wire_sweep", cam="top"):
    return {"camera": cam, "cls": cls, "score": 0.9, "box": [1, 2, 10, 12], "position": pos}


def test_add_and_read_unit(tmp_path, db_url):
    st = Store(tmp_path, db_url)
    assert st.next_unit_no() == 0
    uid = st.add_unit(0, [_shot("top"), _shot("angled", img=False, error="no new image")],
                      [_f()], ts=1000.0)
    u = st.unit(uid)
    assert u["unit_no"] == 0 and u["ts"] == 1000.0
    top = next(s for s in u["shots"] if s["camera"] == "top")
    assert (tmp_path / top["image_path"]).is_file()
    ang = next(s for s in u["shots"] if s["camera"] == "angled")
    assert ang["error"] == "no new image" and ang["image_path"] is None
    assert u["findings"][0]["position"] == "wire:5" and u["findings"][0]["box"] == [1, 2, 10, 12]
    assert st.next_unit_no() == 1
    assert st.units()[0]["n_findings"] == 1
    assert st.unit(999) is None


def test_add_unit_is_atomic(tmp_path, db_url):
    st = Store(tmp_path, db_url)
    bad = _f()
    bad["box"] = [1, 2]                     # malformed -> whole unit must be rolled back
    with pytest.raises(Exception):
        st.add_unit(0, [_shot()], [bad], ts=1.0)
    assert st.units() == []
    assert not any((tmp_path / "images").iterdir())


def test_units_newest_first_and_paging(tmp_path, db_url):
    st = Store(tmp_path, db_url)
    ids = [st.add_unit(i, [_shot()], [], ts=float(i)) for i in range(5)]
    assert [u["id"] for u in st.units(limit=2)] == [ids[4], ids[3]]
    assert [u["id"] for u in st.units(limit=2, before=ids[3])] == [ids[2], ids[1]]


def test_same_defect_in_two_views_counts_once(tmp_path, db_url):
    st = Store(tmp_path, db_url)
    st.add_unit(0, [_shot("top"), _shot("angled")], [_f(cam="top"), _f(cam="angled")], ts=1.0)
    assert st.recent_unit_findings(10) == [{("wire:5", "wire_sweep")}]
    assert st.heatmap() == {"wire:5": {"wire_sweep": 1}}


def test_tracker_k_of_m():
    tr = RepetitiveTracker(k=3, m=10)
    hit = {("wire:5", "wire_sweep")}
    alerts = {}
    for u in range(15):
        alerts[u] = tr.update(hit if u in (0, 4, 8, 14) else set())
    assert all(alerts[u] == [] for u in range(8))
    assert alerts[8] == [{"position": "wire:5", "cls": "wire_sweep", "count": 3, "window": 10}]
    # window at unit 14 = units 5..14 -> hits at 8 and 14 only = 2 < 3
    assert alerts[14] == []


def test_random_defects_do_not_alert():
    tr = RepetitiveTracker(k=3, m=10)
    for u in range(10):
        assert tr.update({(f"wire:{u}", "broken_wire")}) == []


def test_tracker_rebuilds_from_store(tmp_path, db_url):
    st = Store(tmp_path, db_url)
    for i in range(2):
        st.add_unit(i, [_shot()], [_f()], ts=float(i))
    st2 = Store(tmp_path, db_url)                       # app restarted
    tr = RepetitiveTracker(3, 10)
    tr.load(st2.recent_unit_findings(10))
    assert tr.update({("wire:5", "wire_sweep")})[0]["count"] == 3


def test_alerts_saved_with_unit(tmp_path, db_url):
    st = Store(tmp_path, db_url)
    uid = st.add_unit(0, [_shot()], [_f()], ts=1.0)
    st.add_alerts(uid, [{"position": "wire:5", "cls": "wire_sweep", "count": 3, "window": 10}])
    assert st.unit(uid)["alerts"][0]["count"] == 3
    assert st.alerts(limit=5)[0]["unit_id"] == uid
