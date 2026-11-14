"""Which wire is this defect on?

What: gives every finding a "position" name. If we know the chip's wire map (the layout), the
position is the nearest wire, e.g. "wire:5". If we do not (a real camera with no layout
file), the picture is split into an 8x8 grid and the position is the square, e.g. "cell:r2c6".
Why: "repetitive" defects are the same defect at the same position on chip after chip (a worn
bonding tool, a bad bond program). To count them we need a stable name for "the same place".
In the angled view the wire loop is drawn raised above the chip, so we compare against the
real 3D loop shape seen from that angle, not a flat line.
"""
from __future__ import annotations

import numpy as np
from wirebond_synth.geometry import loop_profile, project

from .detector import Finding

GRID = 8
_T = np.linspace(0.0, 1.0, 25)


def _wire_polyline(w: dict, view: str, size: int) -> np.ndarray:
    pad, lead = w["pad"], w["lead"]
    xs = pad["x"] + (lead["x"] - pad["x"]) * _T
    ys = pad["y"] + (lead["y"] - pad["y"]) * _T
    zs = loop_profile(_T, w["loop_height"])
    return np.array([project(x, y, z, view, size) for x, y, z in zip(xs, ys, zs)])


def _dist_to_polyline(p: np.ndarray, line: np.ndarray) -> float:
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    return float(np.min(np.linalg.norm(a + ab * t[:, None] - p, axis=1)))


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match_score(box, line: np.ndarray) -> float:
    """How well a defect box fits one wire (higher = better).

    Three clues: the box overlaps the wire's outline (IoU), much of the wire lies inside the
    box, and the box centre is close to the wire. Big boxes (a missing or sagging wire) are
    judged mostly by overlap; small boxes (a ball defect) mostly by distance.
    """
    pad = 4.0
    outline = (line[:, 0].min() - pad, line[:, 1].min() - pad, line[:, 0].max() + pad, line[:, 1].max() + pad)
    inside = ((line[:, 0] >= box[0]) & (line[:, 0] <= box[2]) & (line[:, 1] >= box[1]) & (line[:, 1] <= box[3])).mean()
    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return _iou(box, outline) + 0.5 * float(inside) - _dist_to_polyline(centre, line) / 100.0


def position_key(f: Finding, view: str, img_size: tuple[int, int], layout: dict | None) -> str:
    x0, y0, x1, y1 = f.box
    W, H = img_size
    if not layout:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        col = min(GRID - 1, max(0, int(cx / W * GRID)))
        row = min(GRID - 1, max(0, int(cy / H * GRID)))
        return f"cell:r{row}c{col}"
    size = layout["size"]
    sx, sy = size / W, size / H                              # into layout pixels
    box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
    best = max(layout["wires"], key=lambda w: _match_score(box, _wire_polyline(w, view, size)))
    return f"wire:{best['index']}"
