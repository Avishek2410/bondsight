"""Domain randomisation: make clean renders look like varied real camera images, so a model
trained on them learns the wire, not the exact drawing style."""
from __future__ import annotations

import cv2
import numpy as np

from .render import Box, clip_box


def augment(img: np.ndarray, boxes: list[Box], seed: int,
            strength: float = 1.0) -> tuple[np.ndarray, list[Box]]:
    if strength <= 0:
        return img, boxes
    s = strength
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    # 1. small camera misplacement: rotation + shift (boxes move with the image)
    ang = rng.uniform(-3, 3) * s
    tx, ty = rng.uniform(-12, 12, 2) * s
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    new_boxes = []
    for b in boxes:
        c = np.array([[b.x0, b.y0, 1], [b.x1, b.y0, 1], [b.x0, b.y1, 1], [b.x1, b.y1, 1]]) @ M.T
        nb = clip_box(Box(b.cls, *c.min(axis=0), *c.max(axis=0), b.wire), w)
        if nb is not None:
            new_boxes.append(nb)

    f = out.astype(np.float32)
    # 2. uneven illumination: linear gradient in a random direction + vignetting
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = rng.uniform(0, 2 * np.pi)
    ramp = ((xx - w / 2) * np.cos(th) + (yy - h / 2) * np.sin(th)) / (w / 2)
    f *= (1 + rng.uniform(-0.25, 0.25) * s * ramp)[..., None]
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w / 2) ** 2
    f *= (1 - rng.uniform(0, 0.25) * s * r2)[..., None]
    # 3. white balance, brightness, contrast
    f *= (1 + rng.uniform(-0.08, 0.08, 3) * s)[None, None, :]
    f = (f - 128) * (1 + rng.uniform(-0.15, 0.15) * s) + 128 + rng.uniform(-0.15, 0.15) * s * 128
    # 4. focus blur and sensor noise
    sigma = rng.uniform(0, 1.2) * s
    if sigma > 0.05:
        f = cv2.GaussianBlur(f, (0, 0), sigma)
    f += rng.normal(0, rng.uniform(0, 6) * s, f.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8), new_boxes
