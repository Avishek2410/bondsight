"""Write a whole synthetic dataset: images per view, COCO annotations, per-unit sidecars."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2

from .augment import augment
from .classes import CLASSES, VIEWS, class_id
from .defects import InjectionConfig, plan_unit_defects
from .layout import make_layout
from .macro import render_macro
from .render import render_unit

MARKER = "config.json"     # only folders containing this file may be overwritten


def unit_seed(product_seed: int, unit: int) -> int:
    return product_seed * 1_000_003 + unit


def _prepare(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (use --overwrite to replace it)")
        if not (out / MARKER).is_file():
            raise FileExistsError(f"{out} was not created by wbsynth; refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def write_image(path: Path, img, fmt: str = "png", quality: int = 95) -> None:
    """PNG (lossless) or JPEG (much smaller; quality 95 is visually lossless)."""
    if fmt == "png":
        ok, buf = cv2.imencode(".png", img)
    elif fmt == "jpg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    else:
        raise ValueError(f"format must be png or jpg, not {fmt!r}")
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    path.write_bytes(buf.tobytes())


def write_png(path: Path, img) -> None:
    write_image(path, img, "png")


def generate(out: Path, n_units: int, n_wires: int, product_seed: int, cfg: InjectionConfig,
             views: tuple[str, ...] = VIEWS, size: int = 640, augment_strength: float = 1.0,
             overwrite: bool = False, fmt: str = "png", quality: int = 95,
             macro: bool = False) -> dict:
    """macro=True: each picture is a 2-4x close-up crop around one wire (see macro.py)."""
    out = Path(out)
    if n_units < 1:
        raise ValueError("units must be >= 1")
    if fmt not in ("png", "jpg") or not 1 <= quality <= 100:
        raise ValueError("format must be png or jpg, quality 1..100")
    for v in views:
        if v not in VIEWS:
            raise ValueError(f"unknown view {v!r}")
    layout = make_layout(n_wires, product_seed, size)
    plan_unit_defects(layout, 0, cfg, product_seed)          # validate cfg before touching disk
    _prepare(out, overwrite)

    config = {"n_units": n_units, "n_wires": n_wires, "product_seed": product_seed,
              "injection": asdict(cfg), "views": list(views), "size": size,
              "augment_strength": augment_strength, "format": fmt,
              "quality": quality if fmt == "jpg" else None, "macro": macro}
    (out / "layout.json").write_text(json.dumps(layout.to_dict(), indent=1))
    (out / "units").mkdir()
    coco = {v: {"info": {"description": "wirebond-synth", "view": v},
                "images": [], "annotations": [],
                "categories": [{"id": class_id(c), "name": c, "supercategory": "defect"}
                               for c in CLASSES]} for v in views}
    counts = {v: Counter() for v in views}
    for v in views:
        (out / "images" / v).mkdir(parents=True)

    for u in range(n_units):
        defects = plan_unit_defects(layout, u, cfg, product_seed)
        name = f"u{u:05d}"
        sidecar = {"unit": u, "defects": [asdict(d) for d in defects]}
        if macro:
            sidecar["macro"] = {}
        for vi, v in enumerate(views):
            if macro:
                img, boxes, info = render_macro(layout, defects, v,
                                                unit_seed(product_seed, u) * 2 + vi, size,
                                                augment_strength)
                sidecar["macro"][v] = info
            else:
                img, boxes = render_unit(layout, defects, v, unit_seed(product_seed, u))
                img, boxes = augment(img, boxes, unit_seed(product_seed, u) * 2 + vi,
                                     augment_strength)
            write_image(out / "images" / v / f"{name}.{fmt}", img, fmt, quality)
            coco[v]["images"].append({"id": u + 1, "file_name": f"{name}.{fmt}",
                                      "width": size, "height": size})
            for b in boxes:
                w, h = b.x1 - b.x0, b.y1 - b.y0
                coco[v]["annotations"].append({
                    "id": len(coco[v]["annotations"]) + 1, "image_id": u + 1,
                    "category_id": class_id(b.cls),
                    "bbox": [round(float(b.x0), 2), round(float(b.y0), 2),
                             round(float(w), 2), round(float(h), 2)],
                    "area": round(float(w * h), 2), "iscrowd": 0,
                    "attributes": {"wire": b.wire}})
                counts[v][b.cls] += 1
        (out / "units" / f"{name}.json").write_text(json.dumps(sidecar, indent=1))

    (out / "annotations").mkdir()
    for v in views:
        (out / "annotations" / f"{v}.json").write_text(json.dumps(coco[v]))
    (out / MARKER).write_text(json.dumps(config, indent=1))   # written last = complete
    return {"units": n_units, "boxes": {v: dict(counts[v]) for v in views}}
