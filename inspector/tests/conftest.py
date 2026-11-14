import json
import os
import shutil
from pathlib import Path

import pytest

PG_URL = os.environ.get("WBI_TEST_PG_URL")      # e.g. postgresql+psycopg://wb@127.0.0.1:55432/wbtest


def _fresh_pg(url: str) -> None:
    """Empty the test PostgreSQL database so every test starts from nothing."""
    import sqlalchemy as sa
    from wirebond_inspector import db
    eng = db.make_engine(url)
    with eng.begin() as c:
        c.execute(sa.text("DROP SCHEMA public CASCADE"))
        c.execute(sa.text("CREATE SCHEMA public"))
    eng.dispose()


@pytest.fixture(params=["sqlite", "postgres"])
def db_url(request):
    """None = the default SQLite file in the data folder; or a fresh PostgreSQL (if WBI_TEST_PG_URL)."""
    if request.param == "sqlite":
        return None
    if not PG_URL:
        pytest.skip("set WBI_TEST_PG_URL (and WBI_DB_PASSWORD) to run the PostgreSQL tests")
    _fresh_pg(PG_URL)
    return PG_URL

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent / "config.example.yaml"
CLASSES = ["wire_sweep", "lifted_ball", "lifted_stitch", "missing_wire", "broken_wire",
           "off_center_ball", "wire_short", "sagging_loop"]


def write_meta(model: Path, tiled: bool = True) -> None:
    meta = {"classes": CLASSES, "img_size": 640, "note": "test"}
    if tiled:
        meta["tiling"] = {"scale": 0.5, "tile": 640, "stride": 320, "nms_iou": 0.5}
    model.with_suffix(".json").write_text(json.dumps(meta))


@pytest.fixture
def example_dir(tmp_path):
    """A copy of config.example.yaml in tmp with a placeholder model next to it."""
    shutil.copy(EXAMPLE, tmp_path / "config.yaml")
    (tmp_path / "models").mkdir()
    model = tmp_path / "models" / "wirebond_dfine_small_m3k.onnx"
    model.write_bytes(b"placeholder")
    write_meta(model)
    return tmp_path


def build_fake_onnx(path: Path, n_queries: int = 300, n_classes: int = 8, batch="N") -> None:
    """A tiny ONNX model with the same inputs/outputs as the real detector.

    Whatever the image, query 0 says: class 3 (missing_wire), score sigmoid(4)=0.98,
    box centred in the image, 1/4 of its width and height. All other queries are ~0.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    logits = np.full((1, n_queries, n_classes), -9.0, np.float32)
    logits[0, 0, 3] = 4.0
    boxes = np.zeros((1, n_queries, 4), np.float32)
    boxes[0, :, :] = [0.5, 0.5, 0.01, 0.01]
    boxes[0, 0] = [0.5, 0.5, 0.25, 0.25]
    inits = [numpy_helper.from_array(logits, "L"), numpy_helper.from_array(boxes, "B"),
             numpy_helper.from_array(np.array([0], np.int64), "s0"),
             numpy_helper.from_array(np.array([1], np.int64), "s1"),
             numpy_helper.from_array(np.array([n_queries, n_classes], np.int64), "qc"),
             numpy_helper.from_array(np.array([n_queries, 4], np.int64), "q4")]
    nodes = [helper.make_node("Shape", ["pixel_values"], ["shp"]),
             helper.make_node("Slice", ["shp", "s0", "s1"], ["n"]),
             helper.make_node("Concat", ["n", "qc"], ["shape_l"], axis=0),
             helper.make_node("Concat", ["n", "q4"], ["shape_b"], axis=0),
             helper.make_node("Expand", ["L", "shape_l"], ["logits"]),
             helper.make_node("Expand", ["B", "shape_b"], ["pred_boxes"])]
    graph = helper.make_graph(
        nodes, "fake",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [batch, 3, 640, 640])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", n_queries, n_classes]),
         helper.make_tensor_value_info("pred_boxes", TensorProto.FLOAT, ["N", n_queries, 4])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.fixture
def fake_model(tmp_path):
    def make(tiled: bool = True, batch="N") -> Path:
        p = tmp_path / f"fake_{'tiled' if tiled else 'whole'}_{batch}.onnx"
        build_fake_onnx(p, batch=batch)
        write_meta(p, tiled)
        return p
    return make
