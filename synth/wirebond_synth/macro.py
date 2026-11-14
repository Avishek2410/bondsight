"""Macro (close-up) mode: the same product photographed 2-4x closer, cropped around one wire.

What: scale the whole layout by a zoom z, render it, then cut a 640x640 window centred
(with some jitter) on one wire - the defective one when there is one. Each picture also gets
a random product style: gold ball bonds or aluminium wedge bonds, on a dark substrate, white
ceramic, green board or brown PCB.
Why: real close-up photos (like the MEMS sensor set) show only a few wires, each hundreds of
pixels long, often aluminium. A model trained only on whole-chip views with tiny gold wires
has never seen that, so it found 0% of the MEMS broken wires. This teaches it the close-up look.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .augment import augment
from .classes import VISIBILITY
from .defects import DefectSpec
from .geometry import project
from .layout import Layout, scale_layout
from .render import DEFAULT_STYLE, SS, Box, Style, clip_box, render_unit

ZOOM = (2.0, 4.0)
KEEP = 0.5            # keep a box if at least this share of it is inside the crop

# backgrounds (BGR). Lead finishes are silver so colour alone never says "gold wire".
STYLES: dict[str, Style] = {
    "dark": DEFAULT_STYLE,
    "ceramic": replace(DEFAULT_STYLE, substrate=(205, 208, 210), die_pad=(150, 155, 160),
                       die=(95, 105, 95), die_edge=(60, 66, 60), lead=(190, 195, 200)),
    "green_board": replace(DEFAULT_STYLE, substrate=(60, 95, 45), die_pad=(70, 120, 60),
                           die=(110, 100, 95), lead=(180, 190, 195)),
    "pcb_brown": replace(DEFAULT_STYLE, substrate=(40, 90, 140), die_pad=(55, 60, 70),
                         die=(70, 70, 75), die_edge=(40, 40, 45), lead=(200, 200, 205)),
    "blue_grey": replace(DEFAULT_STYLE, substrate=(120, 105, 95), die_pad=(140, 130, 120),
                         die=(100, 95, 90), lead=(195, 195, 200)),
}
ALUMINIUM = dict(wire=(185, 188, 192), wire_hi=(235, 238, 240), wire_dark=(120, 122, 128),
                 wedge=True)
THIN = 0.6           # whole-chip wires are drawn thicker than life; close up they look thinner
AL_THICK = 1.8       # aluminium wires are thicker than gold ones
DANGLE_P = 0.7       # share of macro broken wires whose free ends curl and droop


def _dangle(defects: list[DefectSpec], rng: np.random.Generator) -> list[DefectSpec]:
    out = []
    for d in defects:
        if d.cls == "broken_wire" and "dangle_px" not in d.params and rng.random() < DANGLE_P:
            d = DefectSpec(d.wire, d.cls, {**d.params, "dangle_px": float(rng.uniform(3, 10)),
                                           "dangle_side": int(rng.choice([-1, 1]))})
        out.append(d)
    return out


def scale_defect(d: DefectSpec, z: float) -> DefectSpec:
    """Pixel-sized defect parameters (names ending in _px) grow with the zoom."""
    return DefectSpec(d.wire, d.cls,
                      {k: v * z if k.endswith("_px") else v for k, v in d.params.items()})


def _thicken(layout: Layout, f: float) -> Layout:
    return replace(layout, wires=tuple(replace(w, diameter_px=w.diameter_px * f)
                                       for w in layout.wires))


def _focus(layout: Layout, defects: list[DefectSpec], view: str,
           rng: np.random.Generator) -> int:
    """The wire to look at: a visible, labelled defect if there is one, else any wire."""
    seen = [d.wire for d in defects
            if view in VISIBILITY[d.cls] and d.params.get("labelled", True)]
    return int(rng.choice(seen)) if seen else int(rng.integers(len(layout.wires)))


def render_macro(layout: Layout, defects: list[DefectSpec], view: str, seed: int,
                 out_size: int = 640, augment_strength: float = 1.0,
                 ) -> tuple[np.ndarray, list[Box], dict]:
    rng = np.random.default_rng([seed, 7])
    z = float(rng.uniform(*ZOOM))
    style_name = str(rng.choice(sorted(STYLES)))
    metal = "al" if rng.random() < 0.5 else "gold"
    focus = _focus(layout, defects, view, rng)

    defects = _dangle(defects, rng)
    big = _thicken(scale_layout(layout, z), THIN)
    st = STYLES[style_name]
    if metal == "al":
        big, st = _thicken(big, AL_THICK), replace(st, **ALUMINIUM)
    img, boxes = render_unit(big, [scale_defect(d, z) for d in defects], view, seed,
                             style=st, shrink=False)

    # crop window: centred on the focus wire's middle (half loop height up), plus jitter
    w = big.wires[focus]
    a = np.array(project(w.pad.x, w.pad.y, w.loop_height * 0.5, view, big.size))
    b = np.array(project(w.lead.x, w.lead.y, 0.0, view, big.size))
    cx, cy = (a + b) / 2 + rng.uniform(-0.2, 0.2, 2) * out_size
    x0 = int(np.clip(cx - out_size / 2, 0, big.size - out_size))
    y0 = int(np.clip(cy - out_size / 2, 0, big.size - out_size))
    crop = img[y0 * SS:(y0 + out_size) * SS, x0 * SS:(x0 + out_size) * SS]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    kept = []
    for bx in boxes:
        moved = Box(bx.cls, bx.x0 - x0, bx.y0 - y0, bx.x1 - x0, bx.y1 - y0, bx.wire)
        c = clip_box(moved, out_size)
        area = (bx.x1 - bx.x0) * (bx.y1 - bx.y0)
        if c is not None and (c.x1 - c.x0) * (c.y1 - c.y0) >= KEEP * area:
            kept.append(c)
    crop, kept = augment(crop, kept, seed * 2 + 1, augment_strength)
    info = {"zoom": round(z, 3), "focus_wire": focus, "metal": metal, "style": style_name,
            "window": [x0, y0]}
    return crop, kept, info
