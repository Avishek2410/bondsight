import pytest
from PIL import Image

from wirebond_inspector.store import Store
from wirebond_inspector.verdict import unit_verdict

CLASSES = ("wire_sweep", "missing_wire")


def _shot(cam="top", flag=None):
    return {"camera": cam, "view": cam, "source": "x", "image": Image.new("RGB", (100, 80)),
            "error": None, "anomaly_flag": flag}


def _f(cam="top", cls="wire_sweep"):
    return {"camera": cam, "cls": cls, "score": 0.9, "box": [10, 10, 30, 30], "position": "wire:5"}


@pytest.fixture
def st(tmp_path, db_url):
    return Store(tmp_path, db_url, CLASSES)


def test_verdict_rule():
    assert unit_verdict([_f()], [_shot()]) == "DEFECT"
    assert unit_verdict([], [_shot(flag=True), _shot("angled", flag=False)]) == "CHECK"
    assert unit_verdict([], [_shot(flag=None)]) == "OK"


def test_verdict_and_source_stored(st):
    u = st.unit(st.add_unit(0, [_shot()], [_f()], ts=1.0))
    assert (u["verdict"], u["source"], u["review_state"]) == ("DEFECT", "camera", "undecided")
    up = st.unit(st.add_unit(None, [_shot("upload")], [], ts=2.0, source="upload", label="a.jpg"))
    assert (up["verdict"], up["source"], up["label"], up["unit_no"]) == ("OK", "upload", "a.jpg", None)
    assert up["shots"][0]["width"] == 100 and up["shots"][0]["height"] == 80


def test_real_decision_decides_unit(st):
    uid = st.add_unit(0, [_shot()], [_f()], ts=1.0)
    fid = st.unit(uid)["findings"][0]["id"]
    st.add_decision(uid, "real", "Rina", finding_id=fid)
    u = st.unit(uid)
    assert u["findings"][0]["decision"] == "real" and u["review_state"] == "decided"
    assert u["decisions"][0]["operator"] == "Rina"


def test_decide_twice_latest_wins(st):
    uid = st.add_unit(0, [_shot()], [_f()], ts=1.0)
    fid = st.unit(uid)["findings"][0]["id"]
    st.add_decision(uid, "real", "Rina", finding_id=fid, ts=1.0)
    st.add_decision(uid, "false_alarm", "Avi", finding_id=fid, ts=2.0)
    u = st.unit(uid)
    assert u["findings"][0]["decision"] == "false_alarm"
    assert [d["verdict"] for d in u["decisions"]] == ["false_alarm", "real"]


def test_foreign_finding_refused(st):
    a = st.add_unit(0, [_shot()], [_f()], ts=1.0)
    b = st.add_unit(1, [_shot()], [_f()], ts=2.0)
    fb = st.unit(b)["findings"][0]["id"]
    with pytest.raises(ValueError, match="does not belong to unit"):
        st.add_decision(a, "real", "Rina", finding_id=fb)


def test_missed_box_checks(st):
    uid = st.add_unit(0, [_shot()], [], ts=1.0)
    st.add_decision(uid, "missed", "Rina", camera="top", cls="missing_wire", box=[5, 5, 40, 40])
    assert st.unit(uid)["review_state"] == "decided"
    with pytest.raises(ValueError, match="inside the picture"):
        st.add_decision(uid, "missed", "Rina", camera="top", cls="missing_wire", box=[5, 5, 400, 40])
    with pytest.raises(ValueError, match="defect type"):
        st.add_decision(uid, "missed", "Rina", camera="top", cls="banana", box=[5, 5, 40, 40])
    with pytest.raises(ValueError, match="camera"):
        st.add_decision(uid, "missed", "Rina", camera="side", cls="missing_wire", box=[5, 5, 40, 40])
    with pytest.raises(ValueError, match="box"):
        st.add_decision(uid, "missed", "Rina", camera="top", cls="missing_wire", box=[40, 5, 5, 40])


def test_operator_required(st):
    uid = st.add_unit(0, [_shot()], [], ts=1.0)
    with pytest.raises(ValueError, match="operator"):
        st.add_decision(uid, "confirm_ok", "   ")
    with pytest.raises(ValueError, match="operator"):
        st.add_decision(uid, "confirm_ok", "x" * 61)


def test_confirm_ok_rules(st):
    ok = st.add_unit(0, [_shot()], [], ts=1.0)
    st.add_decision(ok, "confirm_ok", "Rina")
    assert st.unit(ok)["review_state"] == "decided"
    bad = st.add_unit(1, [_shot()], [_f()], ts=2.0)
    with pytest.raises(ValueError, match="confirm"):
        st.add_decision(bad, "confirm_ok", "Rina")
    fid = st.unit(bad)["findings"][0]["id"]
    st.add_decision(bad, "false_alarm", "Rina", finding_id=fid)
    st.add_decision(bad, "confirm_ok", "Rina")            # every finding was a false alarm: allowed


def test_unknown_verdict_and_unit(st):
    uid = st.add_unit(0, [_shot()], [], ts=1.0)
    with pytest.raises(ValueError, match="verdict"):
        st.add_decision(uid, "maybe", "Rina")
    with pytest.raises(LookupError):
        st.add_decision(9999, "confirm_ok", "Rina")


def test_units_filters(st):
    a = st.add_unit(0, [_shot()], [_f()], ts=1.0)
    b = st.add_unit(1, [_shot()], [], ts=2.0)
    c = st.add_unit(None, [_shot("upload")], [], ts=3.0, source="upload", label="p.jpg")
    st.add_decision(b, "confirm_ok", "Rina")
    assert [u["id"] for u in st.units(undecided=True)] == [c, a]
    assert [u["id"] for u in st.units(verdict="DEFECT")] == [a]
    assert [u["id"] for u in st.units(source="upload")] == [c]
    assert st.units()[1]["review_state"] == "decided"


def test_uploads_do_not_touch_line_numbering(st):
    st.add_unit(0, [_shot()], [_f()], ts=1.0)
    st.add_unit(None, [_shot("upload")], [_f("upload")], ts=2.0, source="upload", label="p.jpg")
    assert st.next_unit_no() == 1
    assert st.recent_unit_findings(10) == [{("wire:5", "wire_sweep")}]
    assert st.heatmap() == {"wire:5": {"wire_sweep": 1}}


def test_decided_units_for_export(st):
    a = st.add_unit(0, [_shot()], [], ts=1.0)
    st.add_unit(1, [_shot()], [_f()], ts=2.0)
    st.add_decision(a, "confirm_ok", "Rina")
    assert [u["id"] for u in st.decided_units()] == [a]


def test_withdraw_wrong_missed_box(st):
    uid = st.add_unit(0, [_shot()], [], ts=1.0)
    bad = st.add_decision(uid, "missed", "Rina", camera="top", cls="missing_wire", box=[5, 5, 40, 40])
    with pytest.raises(ValueError, match="remove"):
        st.add_decision(uid, "confirm_ok", "Rina")             # a standing missed box contradicts "good"
    st.add_decision(uid, "withdraw", "Rina", ref_id=bad)
    u = st.unit(uid)
    assert [d["withdrawn"] for d in u["decisions"] if d["verdict"] == "missed"] == [True]
    assert u["review_state"] == "undecided"
    assert [x["id"] for x in st.units(undecided=True)] == [uid]
    st.add_decision(uid, "confirm_ok", "Rina")
    assert st.unit(uid)["review_state"] == "decided"
    with pytest.raises(ValueError, match="already"):
        st.add_decision(uid, "withdraw", "Rina", ref_id=bad)


def test_withdraw_only_missed_of_this_unit(st):
    a = st.add_unit(0, [_shot()], [_f()], ts=1.0)
    b = st.add_unit(1, [_shot()], [], ts=2.0)
    mb = st.add_decision(b, "missed", "R", camera="top", cls="missing_wire", box=[5, 5, 40, 40])
    real = st.add_decision(a, "real", "R", finding_id=st.unit(a)["findings"][0]["id"])
    with pytest.raises(ValueError, match="missed"):
        st.add_decision(a, "withdraw", "R", ref_id=mb)
    with pytest.raises(ValueError, match="missed"):
        st.add_decision(a, "withdraw", "R", ref_id=real)


def test_latest_decision_is_the_last_saved_even_if_clock_jumped_back(st):
    uid = st.add_unit(0, [_shot()], [_f()], ts=1.0)
    fid = st.unit(uid)["findings"][0]["id"]
    st.add_decision(uid, "real", "R", finding_id=fid, ts=2000.0)
    st.add_decision(uid, "false_alarm", "R", finding_id=fid, ts=1000.0)      # clock stepped back
    assert st.unit(uid)["findings"][0]["decision"] == "false_alarm"


def test_finding_id_is_indexed(st):
    import sqlalchemy as sa
    idx = sa.inspect(st.engine).get_indexes("decisions")
    assert any(i["column_names"] == ["finding_id"] for i in idx)
