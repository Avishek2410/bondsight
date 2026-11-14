"""Product layout: one die, its bond pads, the package leads, and the wire map pad i -> lead i.

Coordinates are top-view pixels in a square image of side `size`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Pad:
    x: float
    y: float
    w: float          # pad is a square of side w


@dataclass(frozen=True)
class Lead:
    x: float
    y: float
    w: float          # width (x extent)
    h: float          # height (y extent)


@dataclass(frozen=True)
class Wire:
    index: int
    pad: Pad
    lead: Lead
    loop_height: float   # peak loop height above the die, px
    diameter_px: float


@dataclass(frozen=True)
class Layout:
    size: int
    die: tuple[float, float, float, float]   # x0, y0, x1, y1
    wires: tuple[Wire, ...]
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Layout":
        wires = tuple(Wire(w["index"], Pad(**w["pad"]), Lead(**w["lead"]),
                           w["loop_height"], w["diameter_px"]) for w in d["wires"])
        return Layout(d["size"], tuple(d["die"]), wires, d["seed"])


def scale_layout(layout: Layout, z: float) -> Layout:
    """The same product seen z times closer: every length (and the image side) times z."""
    wires = tuple(Wire(w.index, Pad(w.pad.x * z, w.pad.y * z, w.pad.w * z),
                       Lead(w.lead.x * z, w.lead.y * z, w.lead.w * z, w.lead.h * z),
                       w.loop_height * z, w.diameter_px * z) for w in layout.wires)
    return Layout(int(round(layout.size * z)), tuple(v * z for v in layout.die), wires, layout.seed)


def _split_edges(n: int, lengths: list[float]) -> list[int]:
    """Spread n pads over the 4 die edges in proportion to edge length."""
    raw = np.array(lengths) / sum(lengths) * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: n - counts.sum()]:
        counts[i] += 1
    return counts.tolist()


def make_layout(n_wires: int, seed: int, size: int = 640) -> Layout:
    """Build a deterministic product layout with `n_wires` wires (1..n)."""
    if n_wires < 1:
        raise ValueError("n_wires must be >= 1")
    rng = np.random.default_rng(seed)
    c = size / 2
    hw, hh = rng.uniform(0.30, 0.45, 2) * size / 2           # die half-width/height
    die = (c - hw, c - hh, c + hw, c + hh)
    lead_len = 0.05 * size                                    # radial length of a lead
    outer = size / 2 - 0.03 * size - lead_len / 2             # lead-centre ring half-size

    # edges in clockwise order: top, right, bottom, left
    counts = _split_edges(n_wires, [2 * hw, 2 * hh, 2 * hw, 2 * hh])
    pitch = min((2 * hw if k % 2 == 0 else 2 * hh) * 0.85 / max(n, 1)
                for k, n in enumerate(counts) if n)
    pad_w = float(np.clip(pitch * 0.65, 4.0, 0.035 * size))
    inset = pad_w * 0.9

    wires: list[Wire] = []
    for edge, n in enumerate(counts):
        for k in range(n):
            f = -0.85 + 1.7 * (k + 0.5) / n                   # position along edge, -0.85..0.85
            if edge == 0:
                px, py, lx, ly = c + f * hw, c - hh + inset, c + f * outer, c - outer
            elif edge == 1:
                px, py, lx, ly = c + hw - inset, c + f * hh, c + outer, c + f * outer
            elif edge == 2:
                px, py, lx, ly = c - f * hw, c + hh - inset, c - f * outer, c + outer
            else:
                px, py, lx, ly = c - hw + inset, c - f * hh, c - outer, c - f * outer
            along = min(1.7 * outer / n * 0.55, 0.04 * size)
            lw, lh = (along, lead_len) if edge % 2 == 0 else (lead_len, along)
            wires.append(Wire(
                index=len(wires),
                pad=Pad(float(px), float(py), pad_w),
                lead=Lead(float(lx), float(ly), float(lw), float(lh)),
                loop_height=float(rng.uniform(40, 70) * size / 640),
                diameter_px=float(rng.uniform(2.0, 3.5) * size / 640),
            ))
    return Layout(size, tuple(float(v) for v in die), tuple(wires), seed)
