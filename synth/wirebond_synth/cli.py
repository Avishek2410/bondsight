"""Command line: `wbsynth generate ...` and `wbsynth gallery ...`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASSES, VIEWS
from .dataset import generate, write_png
from .defects import DefectSpec, InjectionConfig, sample_params
from .layout import make_layout
from .render import render_unit


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wbsynth", description="Synthetic wire-bond defect images")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write a labelled dataset")
    g.add_argument("--out", required=True)
    g.add_argument("--units", type=int, required=True)
    g.add_argument("--wires", type=int, default=32)
    g.add_argument("--product-seed", type=int, default=1)
    g.add_argument("--mode", choices=["none", "random", "repetitive"], default="random")
    g.add_argument("--p", type=float, default=0.03, help="random: per-wire defect probability")
    g.add_argument("--wire", type=int, help="repetitive: failing wire index")
    g.add_argument("--cls", choices=CLASSES, help="repetitive: defect class")
    g.add_argument("--fraction", type=float, default=0.8)
    g.add_argument("--start-unit", type=int, default=0)
    g.add_argument("--random-p", type=float, default=0.0)
    g.add_argument("--flow-p", type=float, default=0.0,
                   help="probability a unit has a mould-flow sweep event")
    g.add_argument("--flow-strength", type=float, default=15.0,
                   help="max mould-flow bend in px (longest wire, square crossing)")
    g.add_argument("--sweep-reject", type=float, default=0.05,
                   help="bend/length at or above this counts as a wire_sweep reject")
    g.add_argument("--views", default=",".join(VIEWS))
    g.add_argument("--size", type=int, default=640)
    g.add_argument("--augment", type=float, default=1.0)
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--format", choices=["png", "jpg"], default="png")
    g.add_argument("--quality", type=int, default=95, help="JPEG quality (jpg only)")
    g.add_argument("--macro", action="store_true",
                   help="close-up mode: 2-4x zoomed crops around one wire, gold or aluminium")
    gl = sub.add_parser("gallery", help="one example image per defect class")
    gl.add_argument("--out", required=True)
    gl.add_argument("--product-seed", type=int, default=3)
    return ap


def gallery(out: Path, product_seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    layout = make_layout(24, product_seed)
    for cls in (None, *CLASSES):
        d = [] if cls is None else [DefectSpec(2, cls, sample_params(
            cls, np.random.default_rng(0), extreme=True))]
        tiles = []
        for v in VIEWS:
            img, boxes = render_unit(layout, d, v, product_seed)
            img = img.copy()
            for b in boxes:
                cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)),
                              (40, 40, 230), 2)
            cv2.putText(img, f"{cls or 'good'} - {v} view", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(img)
        write_png(out / f"{cls or 'good'}.png", np.hstack(tiles))


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try:
        if a.cmd == "gallery":
            gallery(Path(a.out), a.product_seed)
            print(f"gallery written to {a.out}")
            return 0
        cfg = InjectionConfig(mode=a.mode, p=a.p if a.mode == "random" else 0.0, wire=a.wire,
                              cls=a.cls, fraction=a.fraction, start_unit=a.start_unit,
                              random_p=a.random_p, flow_p=a.flow_p,
                              flow_strength=a.flow_strength, sweep_reject=a.sweep_reject)
        s = generate(Path(a.out), a.units, a.wires, a.product_seed, cfg,
                     tuple(v.strip() for v in a.views.split(",")), a.size, a.augment,
                     a.overwrite, a.format, a.quality, a.macro)
        print(json.dumps(s, indent=1))
        return 0
    except (ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
