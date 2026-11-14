"""Export the feature extractor for the anomaly checker (PatchCore-style) to ONNX.

What: ResNet18 pre-trained on ImageNet (torchvision, BSD-3 licence), cut after layer3.
The ONNX model takes an RGB picture (float 0..1, N x 3 x H x W) and returns one feature
vector per small patch of the picture:
  - ImageNet normalisation is done inside the model,
  - layer2 (128 channels, 1/8 size) and layer3 (256 channels, 1/16 size) are both kept:
    layer2 sees fine detail (a thin wire), layer3 sees a bit more context,
  - each map is smoothed with a 3x3 average so a patch also "knows" its neighbours,
  - layer3 is enlarged to the layer2 size and the two are stacked -> 384 numbers per patch.
Why: these generic features already describe textures and shapes well, so the checker only
needs photos of GOOD units: it remembers their patches and calls anything far from all of
them "unusual". No defect labels, no training on a GPU. Running ONNX keeps the inspector
free of PyTorch (small Docker image, fast on the Pi).

Usage:  tools/.venv/Scripts/python tools/export_anomaly_backbone.py  [--out models/anomaly_backbone.onnx]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PatchFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.layer2, self.layer3 = r.layer2, r.layer3
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)                       # N x 384 x H/8 x W/8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("training/runs/anomaly/anomaly_backbone.onnx"))
    ap.add_argument("--size", type=int, default=640, help="default picture side (the model accepts any multiple of 32)")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    m = PatchFeatures().eval()
    x = torch.rand(1, 3, a.size, a.size)
    torch.onnx.export(m, (x,), str(a.out), input_names=["pixel_values"], output_names=["features"],
                      dynamic_axes={"pixel_values": {0: "n", 2: "h", 3: "w"}, "features": {0: "n", 2: "fh", 3: "fw"}}, opset_version=17,
                      dynamo=False)
    meta = {"backbone": "resnet18 imagenet (torchvision, BSD-3), layer2+layer3, 3x3 avg",
            "img_size": a.size, "feature_dim": 384, "stride": 8}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    import numpy as np
    import onnxruntime as ort
    with torch.no_grad():
        ref = m(x).numpy()
    got = ort.InferenceSession(str(a.out), providers=["CPUExecutionProvider"]).run(None, {"pixel_values": x.numpy()})[0]
    print(f"saved {a.out} ({a.out.stat().st_size / 1e6:.1f} MB) | output {got.shape} | "
          f"max diff vs PyTorch {np.abs(got - ref).max():.2e}")


if __name__ == "__main__":
    main()
