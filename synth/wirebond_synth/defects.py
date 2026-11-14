"""Decide which wires of a unit carry which defect.

Two injection modes model the two kinds of production error:
  random      - each wire fails independently with probability p (common-cause noise)
  repetitive  - one wire position keeps failing with the same defect (assignable cause,
                e.g. a worn capillary or a bad bond-program parameter)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .classes import CLASSES
from .layout import Layout


@dataclass(frozen=True)
class DefectSpec:
    wire: int
    cls: str
    params: dict = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class InjectionConfig:
    mode: str                  # "none" | "random" | "repetitive"
    p: float = 0.0             # random mode: per-wire defect probability
    wire: int | None = None    # repetitive: wire index that keeps failing
    cls: str | None = None     # repetitive: its defect class
    fraction: float = 1.0      # repetitive: share of units (from start_unit) that fail
    start_unit: int = 0        # repetitive: first affected unit
    random_p: float = 0.0      # repetitive: background random defects on other wires
    # mould-flow sweep (any mode): epoxy flowing across the package bends many wires the same
    # way; bend grows with wire length and with how squarely the flow hits the wire
    flow_p: float = 0.0          # probability that a unit had a mould-flow sweep event
    flow_strength: float = 15.0  # max bend (px) of the longest wire crossed squarely
    sweep_reject: float = 0.05   # bend / wire length at or above this = reject (labelled)


# (low, high) for each numeric parameter; `extreme` uses high.
_RANGES = {
    "wire_sweep": {"amplitude_px": (6.0, 25.0)},
    "lifted_ball": {"gap_px": (3.0, 8.0)},
    "lifted_stitch": {"gap_px": (3.0, 8.0)},
    "missing_wire": {},
    "broken_wire": {"t": (0.25, 0.75)},
    "off_center_ball": {"offset": (0.35, 0.6)},     # x pad width
    "wire_short": {},
    "sagging_loop": {"height_factor": (0.3, 0.1)},  # extreme = lowest loop
}


def sample_params(cls: str, rng: np.random.Generator, extreme: bool = False) -> dict:
    """Random parameters for one defect of class `cls`."""
    if cls not in _RANGES:
        raise ValueError(f"unknown class {cls!r}")
    out = {k: float(hi if extreme else rng.uniform(min(lo, hi), max(lo, hi)))
           for k, (lo, hi) in _RANGES[cls].items()}
    if cls == "wire_sweep":
        out["side"] = 1 if extreme else int(rng.choice([-1, 1]))
    if cls == "off_center_ball":
        a = 0.0 if extreme else float(rng.uniform(0, 2 * np.pi))
        out["angle"] = a
    return out


def _validate(layout: Layout, cfg: InjectionConfig) -> None:
    if cfg.mode not in ("none", "random", "repetitive"):
        raise ValueError(f"unknown mode {cfg.mode!r}")
    for name in ("p", "fraction", "random_p", "flow_p"):
        if not 0.0 <= getattr(cfg, name) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 < cfg.sweep_reject < 1.0:
        raise ValueError("sweep_reject must be between 0 and 1 (exclusive)")
    if cfg.flow_strength < 0:
        raise ValueError("flow_strength must be >= 0")
    if cfg.mode == "repetitive":
        if cfg.wire is None or not 0 <= cfg.wire < len(layout.wires):
            raise ValueError(f"wire must be 0..{len(layout.wires) - 1}")
        if cfg.cls not in CLASSES:
            raise ValueError(f"cls must be one of {CLASSES}")


def plan_unit_defects(layout: Layout, unit: int, cfg: InjectionConfig,
                      seed: int) -> list[DefectSpec]:
    """At most one defect per wire; the repetitive defect wins over a random one."""
    _validate(layout, cfg)
    rng = np.random.default_rng([seed, unit])
    out: dict[int, DefectSpec] = {}
    if cfg.mode == "repetitive":
        if unit >= cfg.start_unit and rng.random() < cfg.fraction:
            out[cfg.wire] = DefectSpec(cfg.wire, cfg.cls, sample_params(cfg.cls, rng))
    p = cfg.p if cfg.mode == "random" else cfg.random_p if cfg.mode == "repetitive" else 0.0
    for w in range(len(layout.wires)):
        hit = rng.random() < p
        cls = CLASSES[int(rng.integers(len(CLASSES)))]
        params = sample_params(cls, rng)          # always draw: keeps streams stable
        if hit and w not in out:
            out[w] = DefectSpec(w, cls, params)
    _mould_flow(layout, unit, cfg, seed, out)
    return [out[w] for w in sorted(out)]


def _mould_flow(layout: Layout, unit: int, cfg: InjectionConfig, seed: int,
                out: dict[int, DefectSpec]) -> None:
    """Add a mould-flow sweep event (own random stream, so other defects never change)."""
    if cfg.flow_p <= 0:
        return
    rng = np.random.default_rng([seed, unit, 101])
    if rng.random() >= cfg.flow_p:
        return
    theta = float(rng.uniform(0, 2 * np.pi))
    strength = float(rng.uniform(0.4, 1.0)) * cfg.flow_strength
    flow = np.array([np.cos(theta), np.sin(theta)])
    geo = []
    for w in layout.wires:
        d = np.array([w.lead.x - w.pad.x, w.lead.y - w.pad.y])
        length = float(np.linalg.norm(d))
        geo.append((np.array([-d[1], d[0]]) / length, length))
    longest = max(g[1] for g in geo)
    for w, (perp, length) in zip(layout.wires, geo):
        cross = float(perp @ flow)
        amp = strength * abs(cross) * length / longest
        if w.index in out or amp < 1.0:
            continue
        out[w.index] = DefectSpec(w.index, "wire_sweep", {
            "amplitude_px": amp, "side": 1 if cross >= 0 else -1, "flow_angle": theta,
            "labelled": bool(amp / length >= cfg.sweep_reject)})
