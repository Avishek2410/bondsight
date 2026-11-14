"""Draw one unit (die, pads, leads, wires) in one camera view and box every visible defect.

Drawing happens at SS x resolution with anti-aliased sub-pixel primitives, then is shrunk
with area averaging, which gives smooth thin wires like a real optical image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .classes import VISIBILITY
from .defects import DefectSpec
from .geometry import WirePath, project, wire_path
from .layout import Layout

SS = 2                 # supersampling factor
SHIFT = 2              # OpenCV sub-pixel bits
_K = SS * (1 << SHIFT)
_S45 = float(np.sin(np.pi / 4))

# BGR colours
SUBSTRATE = (38, 42, 46)
DIE_PAD = (70, 112, 165)        # copper die paddle
DIE = (112, 100, 92)            # silicon
DIE_EDGE = (70, 62, 58)
AL_PAD = (196, 198, 202)        # aluminium bond pad
LEAD = (176, 180, 186)          # silver-plated lead tip
GOLD = (48, 168, 214)
GOLD_HI = (150, 225, 250)
GOLD_DARK = (20, 105, 150)
GAP = (14, 14, 16)


@dataclass(frozen=True)
class Style:
    """Colours (BGR) and bond shape of one kind of product. DEFAULT_STYLE is the original
    look (gold ball bonds on a dark substrate); macro mode picks others (aluminium, white
    ceramic, green board, ...)."""
    substrate: tuple = SUBSTRATE
    die_pad: tuple = DIE_PAD
    die: tuple = DIE
    die_edge: tuple = DIE_EDGE
    bond_pad: tuple = AL_PAD
    lead: tuple = LEAD
    wire: tuple = GOLD
    wire_hi: tuple = GOLD_HI
    wire_dark: tuple = GOLD_DARK
    wedge: bool = False          # True = aluminium wedge bonds: flat feet at both ends, no ball


DEFAULT_STYLE = Style()

LOOP_CLASSES = {"wire_sweep", "broken_wire", "wire_short", "sagging_loop"}
BALL_CLASSES = {"lifted_ball", "off_center_ball"}


@dataclass(frozen=True)
class Box:
    cls: str
    x0: float
    y0: float
    x1: float
    y1: float
    wire: int


def _p(x: float, y: float) -> tuple[int, int]:
    return int(round(x * _K)), int(round(y * _K))


def _rect(img, layout, view, cx, cy, w, h, colour, z=0.0):
    (x0, y0), (x1, y1) = (project(cx - w / 2, cy - h / 2, z, view, layout.size),
                          project(cx + w / 2, cy + h / 2, z, view, layout.size))
    cv2.rectangle(img, _p(x0, y0), _p(x1, y1), colour, -1, cv2.LINE_AA, SHIFT)


def _background(layout: Layout, view: str, rng: np.random.Generator,
                st: Style = DEFAULT_STYLE) -> np.ndarray:
    n = layout.size * SS
    img = np.empty((n, n, 3), np.uint8)
    img[:] = st.substrate
    x0, y0, x1, y1 = layout.die
    m = 0.04 * layout.size
    _rect(img, layout, view, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0 + 2 * m, y1 - y0 + 2 * m,
          st.die_pad)
    # die: side face visible in angled view, then top surface with faint circuit texture
    if view == "angled":
        thick = 0.025 * layout.size * _S45
        (a, b), (c, d) = project(x0, y1, 0, view, layout.size), project(x1, y1, 0, view, layout.size)
        cv2.rectangle(img, _p(a, b), _p(c, d + thick), st.die_edge, -1, cv2.LINE_AA, SHIFT)
    _rect(img, layout, view, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, st.die)
    (dx0, dy0), (dx1, dy1) = (project(x0, y0, 0, view, layout.size),
                              project(x1, y1, 0, view, layout.size))
    X0, Y0, X1, Y1 = (int(v * SS) for v in (dx0 + 8, dy0 + 8, dx1 - 8, dy1 - 8))
    if X1 > X0 and Y1 > Y0:
        tex = rng.normal(0, 1, ((Y1 - Y0) // 8 + 1, (X1 - X0) // 8 + 1)).astype(np.float32)
        tex = cv2.resize(tex, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_NEAREST)
        tex = cv2.GaussianBlur(tex, (0, 0), 1.2) * 6
        img[Y0:Y1, X0:X1] = np.clip(img[Y0:Y1, X0:X1].astype(np.float32) + tex[..., None],
                                    0, 255).astype(np.uint8)
    for w in layout.wires:
        _rect(img, layout, view, w.pad.x, w.pad.y, w.pad.w, w.pad.w, st.bond_pad)
        _rect(img, layout, view, w.lead.x, w.lead.y, w.lead.w, w.lead.h, st.lead)
    return img


def _surface(layout, view, p: WirePath, which: str) -> tuple[float, float]:
    """Where the bond would sit if it were not lifted (the pad/lead surface below it)."""
    x, y = p.ball if which == "ball" else p.stitch
    return (x, y + p.lift_gap * _S45) if view == "angled" else (x, y)


def _draw_wire(img, layout: Layout, view: str, p: WirePath, diameter: float,
               st: Style = DEFAULT_STYLE) -> None:
    if p.ball is None:
        return
    GOLD, GOLD_HI, GOLD_DARK = st.wire, st.wire_hi, st.wire_dark   # noqa: N806 (metal colours)
    sq = _S45 if view == "angled" else 1.0
    # stitch: flattened "fish-tail" ellipse along the wire direction
    ang = float(np.degrees(np.arctan2(p.direction[1], p.direction[0])))
    sx, sy = p.stitch
    if p.lifted == "stitch" and view == "angled":
        bx, by = _surface(layout, view, p, "stitch")
        cv2.rectangle(img, _p(sx - p.stitch_len / 2, sy), _p(sx + p.stitch_len / 2, by),
                      GAP, -1, cv2.LINE_AA, SHIFT)
    cv2.ellipse(img, _p(sx, sy), _p(p.stitch_len / 2, 0.45 * p.ball_r * max(sq, 0.8)),
                ang, 0, 360, GOLD, -1, cv2.LINE_AA, SHIFT)
    # wire
    thick = max(1, int(round(diameter * SS)))
    for seg in p.segments:
        pts = np.array([_p(x, y) for x, y in seg], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, GOLD, thick, cv2.LINE_AA, SHIFT)
        hi = np.array([_p(x, y - 0.25 * diameter) for x, y in seg], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [hi], False, GOLD_HI, max(1, thick // 3), cv2.LINE_AA, SHIFT)
    # ball
    bx, by = p.ball
    r = p.ball_r
    if p.lifted == "ball" and view == "angled":
        _, gy = _surface(layout, view, p, "ball")
        cv2.rectangle(img, _p(bx - r, by), _p(bx + r, gy), GAP, -1, cv2.LINE_AA, SHIFT)
    if st.wedge:        # aluminium wedge bond: a flat foot like the stitch, no ball
        cv2.ellipse(img, _p(bx, by), _p(p.stitch_len / 2, 0.45 * r * max(sq, 0.8)),
                    ang, 0, 360, GOLD_DARK, -1, cv2.LINE_AA, SHIFT)
        cv2.ellipse(img, _p(bx, by), _p(0.4 * p.stitch_len, 0.3 * r * max(sq, 0.8)),
                    ang, 0, 360, GOLD if p.lifted == "ball" else GOLD_HI, -1, cv2.LINE_AA, SHIFT)
        return
    ry = r * (0.85 if view == "angled" else 1.0)
    cv2.ellipse(img, _p(bx, by), _p(r, ry), 0, 0, 360, GOLD_DARK, -1, cv2.LINE_AA, SHIFT)
    top_col = GOLD if p.lifted == "ball" and view == "top" else GOLD_HI
    cv2.ellipse(img, _p(bx, by - 0.1 * r), _p(0.72 * r, 0.72 * ry), 0, 0, 360,
                top_col, -1, cv2.LINE_AA, SHIFT)
    if p.lifted == "ball" and view == "top":            # dimmer, slightly blurred ball
        x0, y0 = int((bx - r - 2) * SS), int((by - r - 2) * SS)
        x1, y1 = int((bx + r + 2) * SS), int((by + r + 2) * SS)
        roi = img[max(y0, 0):y1, max(x0, 0):x1]
        if roi.size:
            roi[:] = cv2.GaussianBlur(roi, (0, 0), 1.5)
            roi[:] = (roi.astype(np.float32) * 0.85).astype(np.uint8)


def _box_for(layout: Layout, view: str, d: DefectSpec, p: WirePath) -> tuple[float, ...]:
    w = layout.wires[d.wire]
    if d.cls == "missing_wire":
        a = project(w.pad.x, w.pad.y, 0, view, layout.size)
        b = project(w.lead.x, w.lead.y, 0, view, layout.size)
        pts, pad = np.array([a, b]), 6.0
    elif d.cls in LOOP_CLASSES:
        pts, pad = np.concatenate(p.segments), 4.0 + w.diameter_px
    elif d.cls in BALL_CLASSES:
        bx, by = p.ball
        pts = np.array([(bx - p.ball_r, by - p.ball_r), (bx + p.ball_r, by + p.ball_r),
                        _surface(layout, view, p, "ball")])
        pad = 4.0
    else:  # lifted_stitch
        sx, sy = p.stitch
        h = p.stitch_len / 2
        pts = np.array([(sx - h, sy - h), (sx + h, sy + h), _surface(layout, view, p, "stitch")])
        pad = 4.0
    lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    return lo[0], lo[1], hi[0], hi[1]


def clip_box(b: Box, size: int) -> Box | None:
    """Clip a box to the image; None if what remains is under 2 px in either direction."""
    x0, y0 = max(b.x0, 0.0), max(b.y0, 0.0)
    x1, y1 = min(b.x1, size - 1.0), min(b.y1, size - 1.0)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return Box(b.cls, x0, y0, x1, y1, b.wire)


def render_unit(layout: Layout, defects: list[DefectSpec], view: str,
                seed: int, style: Style = DEFAULT_STYLE,
                shrink: bool = True) -> tuple[np.ndarray, list[Box]]:
    """Render one unit in `view`. Boxes only for defects visible in that view.
    shrink=False returns the SS-times-larger drawing (macro mode crops it before shrinking)."""
    rng = np.random.default_rng(seed)
    img = _background(layout, view, rng, style)
    by_wire = {d.wire: d for d in defects}
    boxes: list[Box] = []
    for w in layout.wires:
        d = by_wire.get(w.index)
        p = wire_path(layout, w, d, view)
        _draw_wire(img, layout, view, p, w.diameter_px, style)
        # an unlabelled defect (e.g. a within-spec mould-flow bend) is drawn but not boxed
        if d is not None and view in VISIBILITY[d.cls] and d.params.get("labelled", True):
            b = clip_box(Box(d.cls, *_box_for(layout, view, d, p), w.index), layout.size)
            if b is not None:
                boxes.append(b)
    if shrink:
        img = cv2.resize(img, (layout.size, layout.size), interpolation=cv2.INTER_AREA)
    return img, boxes
