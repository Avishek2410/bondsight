"""Anomaly checker: a second opinion that only knows what a GOOD unit looks like.

Idea (PatchCore, Roth et al. 2022), in plain words:
1. Cut each picture into small overlapping patches (40 x 40 = 1,600 patches for a 320 px
   picture) and describe every patch with 384 numbers from a pre-trained network
   (anomaly_backbone.onnx, see tools/export_anomaly_backbone.py).
2. "Training" = look at pictures of good units and REMEMBER their patch descriptions
   (the memory bank). To keep it small and fast we keep a well-spread subset ("coreset"):
   repeatedly add the patch that is farthest from everything kept so far.
3. Checking a new picture: for each of its patches find the nearest remembered good patch.
   The distance says "how unusual is this spot". The picture's score is the most unusual spot;
   the 40 x 40 distances form a heat map showing WHERE it is unusual.
4. The alarm level (threshold) comes from good pictures that were NOT used for the memory:
   the highest score a good unit reached, times a small safety margin.
Why: the main AI only finds the 8 defect types it was taught. This checker needs no labels
and can flag anything new (a scratch, a foreign particle, a wire in a strange place). It does
not name the defect, so it is a "please check" signal, not a verdict.
One memory per camera, because the top and angled views look different.
Only numpy + ONNX Runtime: runs in the inspector and on the Pi without PyTorch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .imageio import to_rgb8

FORMAT = 1


class FeatureExtractor:
    """Picture -> (rows, cols, 384) patch descriptions, using anomaly_backbone.onnx."""

    def __init__(self, onnx_path: Path, size: int | None = None):
        self.path = Path(onnx_path)
        meta = json.loads(self.path.with_suffix(".json").read_text())
        self.size = int(size or meta["img_size"])      # the network works at any size (multiple of 32)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])

    def __call__(self, img: Image.Image) -> np.ndarray:
        x = np.asarray(to_rgb8(img).resize((self.size, self.size), Image.BILINEAR), np.float32) / 255.0
        f = self.sess.run(None, {"pixel_values": x.transpose(2, 0, 1)[None]})[0][0]   # 384 x h x w
        return np.ascontiguousarray(f.transpose(1, 2, 0))


def nearest_dist(q: np.ndarray, bank: np.ndarray, bank_sq: np.ndarray | None = None,
                 chunk: int = 4096) -> np.ndarray:
    """Distance from every row of q to its nearest row in bank (Euclidean)."""
    bank_sq = (bank * bank).sum(1) if bank_sq is None else bank_sq
    out = np.empty(len(q), np.float32)
    for i in range(0, len(q), chunk):
        a = q[i:i + chunk]
        d2 = (a * a).sum(1)[:, None] + bank_sq[None, :] - 2.0 * (a @ bank.T)
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def coreset(feats: np.ndarray, n: int, seed: int = 0, proj_dim: int = 128) -> np.ndarray:
    """Pick n well-spread rows (greedy farthest-point). Distances use a random projection
    to 128 numbers, which is much faster and keeps the spread almost the same."""
    if n >= len(feats):
        return np.arange(len(feats))
    rng = np.random.default_rng(seed)
    p = feats @ (rng.standard_normal((feats.shape[1], proj_dim)).astype(np.float32) / np.sqrt(proj_dim))
    sq = (p * p).sum(1)                   # |a-b|^2 = |a|^2 + |b|^2 - 2 a.b  (one fast product per step)
    chosen = [int(rng.integers(len(p)))]
    d = sq + sq[chosen[0]] - 2.0 * (p @ p[chosen[0]])
    for _ in range(n - 1):
        i = int(d.argmax())
        chosen.append(i)
        np.minimum(d, sq + sq[i] - 2.0 * (p @ p[i]), out=d)
    return np.array(chosen)


@dataclass
class AnomalyResult:
    score: float               # the most unusual spot in the picture
    flagged: bool              # score above the camera's alarm level
    heat: np.ndarray           # rows x cols map of how unusual each spot is


class AnomalyModel:
    """Memory of good patches for ONE camera, plus its alarm level."""

    def __init__(self, bank: np.ndarray, threshold: float, info: dict | None = None):
        self.bank = np.ascontiguousarray(bank, np.float32)
        self.bank_sq = (self.bank * self.bank).sum(1)
        self.threshold = float(threshold)
        self.info = info or {}

    # ---- building ------------------------------------------------------------------------
    @classmethod
    def fit(cls, extractor: FeatureExtractor, good: list[Image.Image], holdout: list[Image.Image],
            bank_size: int = 10000, margin: float = 1.1, seed: int = 0) -> "AnomalyModel":
        if not good or not holdout:
            raise ValueError("need good pictures for the memory AND separate good pictures for the alarm level")
        feats = np.concatenate([extractor(im).reshape(-1, 384) for im in good])
        keep = coreset(feats, bank_size, seed)
        m = cls(feats[keep], threshold=np.inf)
        scores = [m.score(extractor(im)).score for im in holdout]
        m.threshold = float(max(scores) * margin)
        m.info = {"n_good": len(good), "n_holdout": len(holdout), "bank": int(len(keep)),
                  "holdout_max": float(max(scores)), "holdout_mean": float(np.mean(scores)),
                  "margin": margin}
        return m

    # ---- checking ------------------------------------------------------------------------
    def score(self, feats: np.ndarray) -> AnomalyResult:
        h, w, c = feats.shape
        heat = nearest_dist(feats.reshape(-1, c), self.bank, self.bank_sq).reshape(h, w)
        s = float(heat.max())
        return AnomalyResult(s, s > self.threshold, heat)

    # ---- files ---------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, bank=self.bank.astype(np.float16), threshold=self.threshold,
                            info=json.dumps({**self.info, "format": FORMAT}))

    @classmethod
    def load(cls, path: Path) -> "AnomalyModel":
        z = np.load(path, allow_pickle=False)
        return cls(z["bank"].astype(np.float32), float(z["threshold"]), json.loads(str(z["info"])))


def heat_overlay(img: Image.Image, heat: np.ndarray, threshold: float) -> Image.Image:
    """The picture with unusual spots tinted red (stronger red = more unusual)."""
    base = to_rgb8(img)
    a = np.clip((heat / max(threshold, 1e-6) - 0.7) / 0.6, 0, 1)      # starts at 70% of the alarm level
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR),
                   np.float32)[..., None] / 255.0 * 0.6
    px = np.asarray(base, np.float32)
    red = np.array([255, 0, 0], np.float32)
    return Image.fromarray((px * (1 - a) + red * a).astype(np.uint8))


class CameraChecker:
    """What the inspector uses: the feature extractor plus one memory per camera name.
    A camera without a memory file is simply not checked (status says how to build one)."""

    def __init__(self, backbone: Path, memory_dir: Path, cameras: list[str], size: int | None = None):
        self.extractor = FeatureExtractor(backbone, size)
        self.dir = Path(memory_dir)
        self.models: dict[str, AnomalyModel] = {}
        for name in cameras:
            p = self.dir / f"{name}.npz"
            if p.is_file():
                self.models[name] = AnomalyModel.load(p)

    def check(self, camera: str, img: Image.Image) -> AnomalyResult | None:
        m = self.models.get(camera)
        return None if m is None else m.score(self.extractor(img))

    def status(self, camera: str) -> str:
        m = self.models.get(camera)
        if m is None:
            return f"unusual-spot check: no memory yet ({self.dir / (camera + '.npz')}; run wbinspect-fit-anomaly)"
        return f"unusual-spot check: ready, alarm level {m.threshold:.2f} (from {m.info.get('n_good', '?')} good units)"
