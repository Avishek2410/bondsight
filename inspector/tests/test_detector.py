import json
import numpy as np
import pytest
from PIL import Image

from wirebond_inspector.detector import OnnxDetector, nms, tile_offsets


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def test_tile_offsets():
    assert tile_offsets(1024, 683, 640, 320) == [(0, 0), (320, 0), (384, 0), (0, 43), (320, 43), (384, 43)]
    assert tile_offsets(300, 200, 640, 320) == [(0, 0)]


def test_nms_keeps_best_per_class():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6], np.float32)
    labels = np.array([0, 0, 1, 0])
    assert sorted(nms(boxes, scores, labels, 0.5).tolist()) == [0, 2, 3]


def test_decode_whole(fake_model):
    det = OnnxDetector(fake_model(tiled=False))
    assert not det.tiled and det.classes[3] == "missing_wire"
    out = det.detect(Image.new("RGB", (800, 600)), 0.4)
    assert len(out) == 1
    f = out[0]
    assert f.cls == "missing_wire" and f.score == pytest.approx(0.982, abs=1e-3)
    assert np.allclose(f.box, (300, 225, 500, 375), atol=1)


def test_tiled_nms_merges_duplicates(fake_model):
    det = OnnxDetector(fake_model(tiled=True))
    assert det.tiled
    # 2048x1366 -> 1024x683 at scale 0.5 -> 6 tiles; vertically overlapping tiles (shift 43 px)
    # report near-identical boxes that must merge; 320 vs 384 px shift (IoU 0.43) stays separate.
    out = det.detect(Image.new("RGB", (2048, 1366)), 0.4)
    assert len(out) == 3
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            assert _iou(out[i].box, out[j].box) <= 0.5
    # tile (0,0) centre in the half-size image is (320,320) -> (640,640) in the original
    assert any(np.allclose(((f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2), (640, 640), atol=45)
               for f in out)


@pytest.mark.parametrize("img,expect_any", [
    (Image.new("L", (300, 200), 90), True),          # small -> whole-image mode
    (Image.new("RGBA", (2592, 1944)), True),
    (Image.fromarray(np.full((1300, 1300), 30000, np.uint16)), True),
], ids=["gray-small", "rgba-5mp", "16bit"])
def test_detect_handles_modes_and_sizes(fake_model, img, expect_any):
    det = OnnxDetector(fake_model(tiled=True))
    out = det.detect(img, 0.4)
    assert bool(out) == expect_any
    for f in out:
        x0, y0, x1, y1 = f.box
        assert 0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height


def test_small_images_are_not_tiled(fake_model):
    # a 640x640 synthetic frame shrunk by 0.5 fits in one tile; the model was trained on such
    # frames WHOLE, so the tiled model must fall back to whole-image mode (box = centre quarter)
    det = OnnxDetector(fake_model(tiled=True))
    out = det.detect(Image.new("RGB", (640, 640)), 0.4)
    assert len(out) == 1
    assert np.allclose(out[0].box, (240, 240, 400, 400), atol=1)


def test_fixed_batch_model_runs_one_tile_at_a_time(fake_model):
    # the real exported model has batch size fixed to 1; the detector must not send 4 tiles at once
    det = OnnxDetector(fake_model(tiled=True, batch=1))
    out = det.detect(Image.new("RGB", (2048, 1366)), 0.4)
    assert len(out) == 3


def test_class_thresholds_from_model_json(fake_model):
    # the fake model's only finding is missing_wire at 0.982
    p = fake_model(tiled=False)
    meta = json.loads(p.with_suffix(".json").read_text())
    img = Image.new("RGB", (800, 600))
    for per_class, cfg_thr, expect in [({"missing_wire": 0.99}, 0.3, 0),   # its own level wins over config
                                       ({"missing_wire": 0.5}, 0.99, 1),   # ... both ways
                                       ({"broken_wire": 0.1}, 0.99, 0),    # unlisted type -> config level
                                       ({}, 0.4, 1)]:                      # old model json: unchanged
        meta["class_thresholds"] = per_class
        p.with_suffix(".json").write_text(json.dumps(meta))
        det = OnnxDetector(p)
        assert len(det.detect(img, cfg_thr)) == expect, (per_class, cfg_thr)
