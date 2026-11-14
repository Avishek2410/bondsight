"""Turn what operators decided into training data for the next, better model.

What: every unit a person has fully reviewed becomes labelled pictures in COCO format (the
format the training notebooks read): DIR/images/<unit>_<camera>.jpg + DIR/annotations.json.
  - a defect marked "real"           -> its box is kept
  - a defect marked "false alarm"    -> no box, but the picture stays: the model learns that this
                                        look-alike is fine (a "hard negative")
  - a "missed" defect a person drew  -> its box is added
  - a unit confirmed good            -> its pictures with no boxes
Units nobody reviewed are left out: the AI's own guesses are never treated as the truth.
A "missed" box the person later removed (withdraw) is left out too. A picture that is no longer
on disk is skipped and counted, so one missing file never spoils the whole export.
Why: the limit of this project is real photos from the real line. This is how a running
station collects them.

    wbinspect-export --config config.yaml --out training_export/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .store import Store


def export_coco(store: Store, classes: tuple[str, ...], out: Path) -> dict:
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cat = {c: i for i, c in enumerate(classes)}
    images, anns, skipped = [], [], 0
    for u in store.decided_units():
        boxes: dict[str, list[tuple[str, list[float]]]] = {}
        for f in u["findings"]:
            if f["decision"] == "real":
                boxes.setdefault(f["camera"], []).append((f["cls"], f["box"]))
        for d in (d for d in u["decisions"] if d["verdict"] == "missed" and not d["withdrawn"]):
            boxes.setdefault(d["camera"], []).append((d["cls"], d["box"]))
        for s in u["shots"]:
            if not s["image_path"]:
                continue
            name = f"{u['id']}_{s['camera']}.jpg"
            try:
                shutil.copyfile(store.dir / s["image_path"], out / "images" / name)
            except OSError:
                skipped += 1
                continue
            img_id = len(images) + 1
            images.append({"id": img_id, "file_name": name, "width": s["width"], "height": s["height"]})
            for cls, (x0, y0, x1, y1) in boxes.get(s["camera"], []):
                if cls not in cat:
                    continue
                w, h = float(x1) - float(x0), float(y1) - float(y0)
                anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cat[cls],
                             "bbox": [float(x0), float(y0), w, h], "area": w * h, "iscrowd": 0})
    coco = {"info": {"description": f"wirebond-inspector review export from {store.dir}",
                     "date_created": time.strftime("%Y-%m-%d %H:%M:%S")},
            "categories": [{"id": i, "name": c} for c, i in cat.items()],
            "images": images, "annotations": anns}
    (out / "annotations.json").write_text(json.dumps(coco, indent=1))
    return {"images": len(images), "boxes": len(anns), "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wbinspect-export", description="Export reviewed units as COCO")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    from .config import ConfigError, load_config
    from .db import DatabaseUnavailable
    try:
        cfg = load_config(a.config)
        classes = tuple(json.loads(cfg.model.with_suffix(".json").read_text())["classes"])
        store = Store(cfg.data_dir, cfg.database or None, classes)
    except (ConfigError, DatabaseUnavailable) as e:
        print(f"wbinspect-export: {e}", file=sys.stderr)
        return 2
    s = export_coco(store, classes, a.out)
    print(f"exported {s['images']} pictures with {s['boxes']} boxes to {a.out}"
          + (f" ({s['skipped']} pictures no longer on disk were skipped)" if s["skipped"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
