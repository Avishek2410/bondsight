from wirebond_synth.classes import CLASSES, VIEWS, VISIBILITY, class_id


def test_class_table():
    assert CLASSES == ("wire_sweep", "lifted_ball", "lifted_stitch", "missing_wire",
                       "broken_wire", "off_center_ball", "wire_short", "sagging_loop")
    assert class_id("wire_sweep") == 1 and class_id("sagging_loop") == 8
    assert VISIBILITY["sagging_loop"] == {"angled"} and VISIBILITY["lifted_stitch"] == {"angled"}
    assert all(VISIBILITY[c] == set(VIEWS) for c in CLASSES
               if c not in ("sagging_loop", "lifted_stitch"))
