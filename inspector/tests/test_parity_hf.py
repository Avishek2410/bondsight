"""Check the torch-free detector gives the same answers as the PyTorch one used in training.

Runs only where transformers + the trained model exist (the PC's tools/.venv, which also needs
wirebond_synth importable - the test adds synth/ to the path):
    tools/.venv/Scripts/python -m pytest inspector/tests/test_parity_hf.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "training" / "runs" / "local_v1"
if not (RUNS / "best_model_tiled").is_dir() or not (RUNS / "wirebond_dfine_nano_tiled.onnx").is_file():
    pytest.skip("trained model not present", allow_module_level=True)
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "inspector"), str(ROOT / "synth")]

from reality_check import Detector  # noqa: E402
from wirebond_synth.augment import augment  # noqa: E402
from wirebond_synth.dataset import unit_seed  # noqa: E402
from wirebond_synth.defects import InjectionConfig, plan_unit_defects  # noqa: E402
from wirebond_synth.layout import make_layout  # noqa: E402
from wirebond_synth.render import render_unit  # noqa: E402

from wirebond_inspector.detector import OnnxDetector  # noqa: E402


def unit(u: int) -> Image.Image:
    lay = make_layout(24, 7)
    cfg = InjectionConfig("random", p=0.1)
    img, boxes = render_unit(lay, plan_unit_defects(lay, u, cfg, 7), "top", unit_seed(7, u))
    img, _ = augment(img, boxes, unit_seed(7, u) * 2, 1.0)
    return Image.fromarray(img[:, :, ::-1].copy())


def mosaic() -> Image.Image:
    m = Image.new("RGB", (1920, 1280))
    for i in range(6):
        m.paste(unit(i), ((i % 3) * 640, (i // 3) * 640))
    return m


def _same(ref, got):
    ref = sorted(ref, key=lambda r: -r[2])
    assert [r[1] for r in ref] == [f.cls for f in got]
    for r, f in zip(ref, got):
        assert np.allclose(r[0], f.box, atol=2), (r, f)
        assert abs(r[2] - f.score) < 0.01


@pytest.fixture(scope="module")
def onnx_det():
    return OnnxDetector(RUNS / "wirebond_dfine_nano_tiled.onnx")


def test_whole_path_matches_pytorch(onnx_det):
    img = unit(0)                                     # 640 px -> whole-image mode
    got = onnx_det.detect(img, 0.4)
    assert got, "expected at least one detection to compare"
    _same(Detector(str(RUNS / "best_model_tiled"), tiled=False)(img, 0.4), got)


def test_tiled_path_matches_pytorch(onnx_det):
    img = mosaic()                                    # 1920x1280 -> 960x640 -> 2 tiles
    got = onnx_det.detect(img, 0.4)
    assert got, "expected at least one detection to compare"
    _same(Detector(str(RUNS / "best_model_tiled"), tiled=True)(img, 0.4), got)
