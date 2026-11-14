"""Run the trained wire-bond detector (an .onnx file) on a picture.

What it does, step by step:
1. Read `<model>.json` next to the model: class names, input size, (for the tiled model)
   how to cut the picture into tiles, and (newer models) `class_thresholds`: one confidence
   level per defect type, learned on validation photos. A type listed there uses its own level;
   any other type uses the `threshold` from config.yaml.
2. Tiled model and a big picture (e.g. a 2048 px photo): shrink it by `scale`, cut overlapping
   640x640 tiles, run the model on every tile. Small picture (fits in one tile after shrinking)
   or whole model: just resize the whole picture to 640x640.
3. Turn the model's raw numbers into boxes ("decode"): the model makes 300 guesses, each with
   a score per class; we keep the confident ones.
4. Move each tile's boxes back to where the tile sits in the full picture, and remove
   duplicates where two overlapping tiles found the same defect ("NMS").
Why ONNX Runtime and not PyTorch: it is small, fast on the Pi's CPU, and needs no GPU.
The steps copy what the training notebook and tools/reality_check.py do, because a model only
works well when pictures are prepared the same way as in training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .imageio import to_rgb8

BATCH = 4          # tiles per model call: enough for speed, small enough for the Pi's memory


@dataclass(frozen=True)
class Finding:
    cls: str
    score: float
    box: tuple[float, float, float, float]      # x0, y0, x1, y1 in original-image pixels


def _positions(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    p = list(range(0, length - size + 1, stride))
    if p[-1] + size < length:
        p.append(length - size)
    return p


def tile_offsets(w: int, h: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners of the tiles covering a w x h picture (row by row)."""
    return [(x, y) for y in _positions(h, tile, stride) for x in _positions(w, tile, stride)]


def nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou: float) -> np.ndarray:
    """Indices of boxes to keep: per class, drop a box overlapping a better one by > iou."""
    keep: list[int] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        while idx.size:
            i = idx[0]
            keep.append(int(i))
            rest = idx[1:]
            if not rest.size:
                break
            b, r = boxes[i], boxes[rest]
            iw = np.clip(np.minimum(b[2], r[:, 2]) - np.maximum(b[0], r[:, 0]), 0, None)
            ih = np.clip(np.minimum(b[3], r[:, 3]) - np.maximum(b[1], r[:, 1]), 0, None)
            inter = iw * ih
            union = (b[2] - b[0]) * (b[3] - b[1]) + (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]) - inter
            idx = rest[inter / np.maximum(union, 1e-9) <= iou]
    return np.array(sorted(keep), dtype=np.int64)


class OnnxDetector:
    def __init__(self, model: Path, threads: int = 4):
        model = Path(model)
        meta = json.loads(model.with_suffix(".json").read_text())
        self.classes: tuple[str, ...] = tuple(meta["classes"])
        self.size: int = int(meta["img_size"])
        self.tiling: dict | None = meta.get("tiling")
        self.tiled: bool = self.tiling is not None
        self.class_thresholds: dict[str, float] = {k: float(v) for k, v in
                                                   (meta.get("class_thresholds") or {}).items()}
        self.name: str = model.stem
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        # Some exported models accept exactly one picture per call (batch dimension fixed to 1).
        dim0 = self.session.get_inputs()[0].shape[0]
        self.batch: int = dim0 if isinstance(dim0, int) and dim0 > 0 else BATCH

    def _prep(self, img: Image.Image) -> np.ndarray:
        # same as the training processor: bilinear resize, scale to 0..1, channels first
        a = np.asarray(img.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0
        return a.transpose(2, 0, 1)

    def _run(self, crops: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([self._prep(c) for c in crops])
        logits, boxes = self.session.run(["logits", "pred_boxes"], {"pixel_values": x})
        return logits, boxes

    @staticmethod
    def _decode(logits: np.ndarray, boxes: np.ndarray, w: float, h: float, thr: float):
        """One image's raw output -> (boxes xyxy px, scores, labels) above `thr` (RT-DETR style)."""
        q, c = logits.shape
        prob = 1.0 / (1.0 + np.exp(-logits.reshape(-1).astype(np.float64)))
        top = np.argsort(-prob, kind="stable")[:q]
        top = top[prob[top] >= thr]
        qi, lab = top // c, top % c
        cx, cy, bw, bh = boxes[qi].T
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        return xyxy.astype(np.float32).reshape(-1, 4), prob[top].astype(np.float32), lab.astype(np.int64)

    def detect(self, img: Image.Image, threshold: float) -> list[Finding]:
        per_class = np.array([self.class_thresholds.get(c, threshold) for c in self.classes], np.float32)
        final, threshold = per_class, float(per_class.min())      # decode at the lowest level, filter below
        img = to_rgb8(img)
        W, H = img.size
        t = self.tiling or {}
        scale, tile = float(t.get("scale", 1.0)), int(t.get("tile", self.size))
        w, h = round(W * scale), round(H * scale)
        # Tile only big pictures. A picture that fits in one tile after shrinking (e.g. a 640 px
        # synthetic frame) is looked at whole - that is how such frames were seen in training.
        if not self.tiled or max(w, h) <= tile:
            lg, bx = self._run([img])
            B, S, L = self._decode(lg[0], bx[0], W, H, threshold)
        else:
            stride = int(t["stride"])
            small = img.resize((w, h), Image.LANCZOS)
            offs = tile_offsets(w, h, tile, stride)
            Bs, Ss, Ls = [], [], []
            for i in range(0, len(offs), self.batch):
                chunk = offs[i:i + self.batch]
                crops = [small.crop((x, y, x + tile, y + tile)) for x, y in chunk]   # pads with black
                lg, bx = self._run(crops)
                for (x, y), l1, b1 in zip(chunk, lg, bx):
                    b, s, l = self._decode(l1, b1, tile, tile, threshold)
                    b[:, [0, 2]] += x
                    b[:, [1, 3]] += y
                    Bs.append(b / scale)
                    Ss.append(s)
                    Ls.append(l)
            B, S, L = np.concatenate(Bs), np.concatenate(Ss), np.concatenate(Ls)
            keep = nms(B, S, L, float(t.get("nms_iou", 0.5)))
            B, S, L = B[keep], S[keep], L[keep]
        # keep boxes inside the picture; drop ones that only cover black padding
        B[:, [0, 2]] = B[:, [0, 2]].clip(0, W)
        B[:, [1, 3]] = B[:, [1, 3]].clip(0, H)
        ok = ((B[:, 2] - B[:, 0]) >= 1) & ((B[:, 3] - B[:, 1]) >= 1) & (S >= final[L])
        B, S, L = B[ok], S[ok], L[ok]
        order = np.argsort(-S, kind="stable")
        return [Finding(self.classes[int(L[i])], float(S[i]), tuple(float(v) for v in B[i]))
                for i in order]
