"""Wire geometry: a 3D loop from ball (on the pad) to stitch (on the lead), deformed by the
defect, then projected into a camera view.

View `top`    : camera straight down — sees x, y only.
View `angled` : camera tilted 45 deg about the image x-axis — loop height shows as upward shift.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .defects import DefectSpec
from .layout import Layout, Wire

N_POINTS = 64
RISE_T = 0.2                  # fraction of the wire spent rising steeply above the ball
_S45 = float(np.sin(np.pi / 4))
_C45 = float(np.cos(np.pi / 4))


@dataclass
class WirePath:
    segments: list[np.ndarray] = field(default_factory=list)   # (N,2) projected polylines
    ball: tuple[float, float] | None = None                    # projected ball centre
    ball_r: float = 0.0
    stitch: tuple[float, float] | None = None                  # projected stitch centre
    stitch_len: float = 0.0
    direction: tuple[float, float] = (1.0, 0.0)                # projected unit vector ball->stitch
    lift_gap: float = 0.0                                      # px gap under a lifted bond
    lifted: str | None = None                                  # "ball" | "stitch" | None


def project(x: float, y: float, z: float, view: str, size: int) -> tuple[float, float]:
    """Project a 3D point (top-view pixels + height z) into `view`."""
    if view == "top":
        return (x, y)
    if view == "angled":
        cy = size / 2
        return (x, cy + (y - cy) * _C45 - z * _S45)
    raise ValueError(f"unknown view {view!r}")


def loop_profile(t: np.ndarray, height: float) -> np.ndarray:
    """Classic bond loop: steep rise above the ball, long parabolic slope down to the stitch."""
    up = height * np.sin(np.pi / 2 * np.clip(t / RISE_T, 0, 1))
    down = height * (1 - ((t - RISE_T) / (1 - RISE_T)) ** 2)
    return np.where(t < RISE_T, up, down)


def _straight(w: Wire) -> tuple[np.ndarray, np.ndarray]:
    return np.array([w.pad.x, w.pad.y]), np.array([w.lead.x, w.lead.y])


def _short_amplitude(layout: Layout, w: Wire, perp: np.ndarray) -> float:
    """Lateral bow (signed, along perp) that brings this wire's middle onto its nearer neighbour."""
    a, b = _straight(w)
    mid = (a + b) / 2
    best = None
    for j in (w.index - 1, w.index + 1):
        if 0 <= j < len(layout.wires):
            na, nb = _straight(layout.wires[j])
            d = (na + nb) / 2 - mid
            if best is None or np.linalg.norm(d) < np.linalg.norm(best):
                best = d
    return 0.0 if best is None else 0.9 * float(best @ perp)


def wire_path(layout: Layout, wire: Wire, defect: DefectSpec | None, view: str) -> WirePath:
    cls = defect.cls if defect else None
    prm = defect.params if defect else {}
    if cls == "missing_wire":
        return WirePath()

    a, b = _straight(wire)
    # off-centre ball: the ball sits off the wire axis (worn capillary / off-centre free-air
    # ball); the bonder still moves the wire along its programmed path, so only the ball moves
    ball_xy = a.copy()
    if cls == "off_center_ball":
        r = prm["offset"] * wire.pad.w
        ball_xy = a + r * np.array([np.cos(prm["angle"]), np.sin(prm["angle"])])
    d = b - a
    length = float(np.linalg.norm(d))
    u = d / length
    perp = np.array([-u[1], u[0]])

    t = np.linspace(0.0, 1.0, N_POINTS)
    xy = a + t[:, None] * d
    z = loop_profile(t, wire.loop_height)

    lateral = np.zeros_like(t)
    if cls == "wire_sweep":
        lateral = prm["side"] * prm["amplitude_px"] * np.sin(np.pi * t)
    elif cls == "wire_short":
        lateral = _short_amplitude(layout, wire, perp) * np.sin(np.pi * t)
    elif cls == "sagging_loop":
        z = z * prm["height_factor"]
    xy = xy + lateral[:, None] * perp

    gap, lifted = 0.0, None
    if cls == "lifted_ball":
        gap, lifted = prm["gap_px"], "ball"
        z = z + gap * (1 - t)
    elif cls == "lifted_stitch":
        gap, lifted = prm["gap_px"], "stitch"
        z = z + gap * t

    if cls == "broken_wire" and prm.get("dangle_px"):
        # the two free ends near the break curl sideways and droop, like a real snapped wire
        near = np.exp(-np.abs(t - prm["t"]) / 0.12)
        xy = xy + (prm.get("dangle_side", 1) * prm["dangle_px"] * near)[:, None] * perp
        z = np.maximum(z - 0.6 * prm["dangle_px"] * near, 0.0)

    pts = np.array([project(x, y, h, view, layout.size) for (x, y), h in zip(xy, z)])
    if cls == "broken_wire":
        keep = np.abs(t - prm["t"]) >= 0.04
        cut = int(np.argmin(keep)) if not keep.all() else len(t)
        segments = [s for s in (pts[:cut], pts[cut:][keep[cut:]]) if len(s) >= 2]
    else:
        segments = [pts]

    ball_z = gap if lifted == "ball" else 0.0
    stitch_z = gap if lifted == "stitch" else 0.0
    ball = project(ball_xy[0], ball_xy[1], ball_z, view, layout.size)
    stitch = project(b[0], b[1], stitch_z, view, layout.size)
    pd = np.array(stitch) - np.array(ball)
    n = float(np.linalg.norm(pd)) or 1.0
    return WirePath(
        segments=segments,
        ball=ball, ball_r=0.375 * wire.pad.w,
        stitch=stitch, stitch_len=0.375 * wire.pad.w * 2.5,
        direction=(float(pd[0] / n), float(pd[1] / n)),
        lift_gap=gap, lifted=lifted,
    )
