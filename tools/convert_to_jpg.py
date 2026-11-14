"""Copy a wbsynth dataset tree (one or many product folders) with images re-encoded as JPEG.

Labels are unchanged except image file names (.png -> .jpg). The PNG source is left as is.

    synth/.venv/Scripts/python tools/convert_to_jpg.py out/train_v1 out/train_v1_jpg --quality 95
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2


def convert_product(src: Path, dst: Path, quality: int) -> int:
    n = 0
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if f.is_dir():
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix.lower() == ".png" and rel.parts[0] == "images":
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError(f"encode failed: {f}")
            out.with_suffix(".jpg").write_bytes(buf.tobytes())
            n += 1
        elif rel.parts[0] == "annotations" and f.suffix == ".json":
            coco = json.loads(f.read_text())
            for im in coco["images"]:
                im["file_name"] = im["file_name"].rsplit(".", 1)[0] + ".jpg"
            out.write_text(json.dumps(coco))
        elif rel.name == "config.json":
            cfg = json.loads(f.read_text())
            cfg.update(format="jpg", quality=quality, converted_from="png")
            out.write_text(json.dumps(cfg, indent=1))
        else:
            shutil.copy2(f, out)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--quality", type=int, default=95)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists() and any(dst.iterdir()):
        print(f"{dst} is not empty; refusing to overwrite", file=sys.stderr)
        return 2
    products = sorted(p.parent for p in src.rglob("config.json"))
    total = 0
    for p in products:
        total += convert_product(p, dst / p.relative_to(src), a.quality)
        print(f"{p.relative_to(src)}: done ({total} images so far)", flush=True)
    print(f"converted {total} images", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
