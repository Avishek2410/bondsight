"""Export a trained D-FINE model folder to ONNX (what the Pi runs) and check it matches PyTorch.

    tools/.venv/Scripts/python tools/export_onnx.py training/runs/local_v1/best_model_tiled \
        training/runs/local_v1/wirebond_dfine_nano_tiled.onnx --tiled --note "M2c: NTUST tiles + synthetic"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import DFineForObjectDetection

CLASSES = ["wire_sweep", "lifted_ball", "lifted_stitch", "missing_wire", "broken_wire",
           "off_center_ball", "wire_short", "sagging_loop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("onnx")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--tiled", action="store_true", help="model expects tiled inference (records settings)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    model = DFineForObjectDetection.from_pretrained(a.model).eval()
    dummy = torch.zeros(1, 3, a.img_size, a.img_size)
    torch.onnx.export(model, (dummy,), a.onnx, input_names=["pixel_values"],
                      output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)

    meta = {"classes": CLASSES, "img_size": a.img_size, "note": a.note}
    if a.tiled:
        meta["tiling"] = {"scale": 0.5, "tile": a.img_size, "stride": a.img_size // 2, "nms_iou": 0.5}
    Path(a.onnx).with_suffix(".json").write_text(json.dumps(meta, indent=1))

    px = torch.rand(1, 3, a.img_size, a.img_size)
    with torch.no_grad():
        ref = model(pixel_values=px)
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    lg, bx = sess.run(["logits", "pred_boxes"], {"pixel_values": px.numpy()})
    diff = float(max(np.abs(lg - ref.logits.numpy()).max(), np.abs(bx - ref.pred_boxes.numpy()).max()))
    print(f"saved {a.onnx} ({Path(a.onnx).stat().st_size / 1e6:.1f} MB) | max difference vs PyTorch: {diff:.2e}")


if __name__ == "__main__":
    main()
