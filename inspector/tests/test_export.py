import json

from PIL import Image

from wirebond_inspector.export import export_coco
from wirebond_inspector.store import Store

CLASSES = ("wire_sweep", "missing_wire", "broken_wire")


def _shot(cam="top"):
    return {"camera": cam, "view": cam, "source": "x", "image": Image.new("RGB", (100, 80)), "error": None}


def _f(cls="wire_sweep"):
    return {"camera": "top", "cls": cls, "score": 0.9, "box": [10, 20, 30, 50], "position": "wire:5"}


def test_export_reviewed_units_only(tmp_path):
    st = Store(tmp_path / "data", None, CLASSES)
    real = st.add_unit(0, [_shot()], [_f()], ts=1.0)
    st.add_decision(real, "real", "R", finding_id=st.unit(real)["findings"][0]["id"])
    fa = st.add_unit(1, [_shot()], [_f()], ts=2.0)
    st.add_decision(fa, "false_alarm", "R", finding_id=st.unit(fa)["findings"][0]["id"])
    st.add_decision(fa, "missed", "R", camera="top", cls="broken_wire", box=[60, 10, 90, 40])
    ok = st.add_unit(2, [_shot(), _shot("angled")], [], ts=3.0)
    st.add_decision(ok, "confirm_ok", "R")
    st.add_unit(3, [_shot()], [_f()], ts=4.0)                       # not reviewed: left out

    out = tmp_path / "export"
    summary = export_coco(st, CLASSES, out)
    coco = json.loads((out / "annotations.json").read_text())

    assert summary == {"images": 4, "boxes": 2, "skipped": 0}
    assert [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])] == list(CLASSES)
    files = sorted(i["file_name"] for i in coco["images"])
    assert files == sorted([f"{real}_top.jpg", f"{fa}_top.jpg", f"{ok}_top.jpg", f"{ok}_angled.jpg"])
    assert all((out / "images" / f).is_file() for f in files)
    by_img = {i["id"]: i["file_name"] for i in coco["images"]}
    anns = sorted((by_img[a["image_id"]], a["category_id"], a["bbox"]) for a in coco["annotations"])
    assert anns == sorted([(f"{real}_top.jpg", 0, [10.0, 20.0, 20.0, 30.0]),
                           (f"{fa}_top.jpg", 2, [60.0, 10.0, 30.0, 30.0])])
    assert all(a["area"] > 0 and a["iscrowd"] == 0 for a in coco["annotations"])


def test_export_withdrawn_box_left_out_and_missing_picture_skipped(tmp_path):
    st = Store(tmp_path / "data", None, CLASSES)
    a = st.add_unit(0, [_shot()], [], ts=1.0)
    wrong = st.add_decision(a, "missed", "R", camera="top", cls="broken_wire", box=[1, 1, 9, 9])
    st.add_decision(a, "withdraw", "R", ref_id=wrong)
    st.add_decision(a, "missed", "R", camera="top", cls="broken_wire", box=[60, 10, 90, 40])
    b = st.add_unit(1, [_shot()], [], ts=2.0)
    st.add_decision(b, "confirm_ok", "R")
    (st.dir / st.unit(b)["shots"][0]["image_path"]).unlink()          # picture cleaned up
    s = export_coco(st, CLASSES, tmp_path / "out")
    coco = json.loads((tmp_path / "out" / "annotations.json").read_text())
    assert s == {"images": 1, "boxes": 1, "skipped": 1}
    assert coco["annotations"][0]["bbox"] == [60.0, 10.0, 30.0, 30.0]
