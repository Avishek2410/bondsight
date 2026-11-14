"""M5 reality check: run the synthetic-trained detector on REAL wire-bond photos.

Scored datasets (labels can be mapped onto our classes):
  NTUST wire-bonding : 'sagged wire' -> wire_sweep/sagging_loop, 'missing wire' -> missing_wire,
                       'good wire' -> must NOT be flagged (false-alarm check)
  WHU mems_60        : '*_wire_bonding_broken' -> broken_wire
Everything else (incl. the curated web photos) is looked at only: detections drawn on a contact sheet.

    tools/.venv/Scripts/python tools/reality_check.py --model training/runs/local_v1/best_model --out out/reality_v1
    tools/.venv/Scripts/python tools/reality_check.py --model training/runs/local_v1/best_model_tiled --tiled --out out/reality_v2

--tiled: shrink the photo to TILE_SCALE, cut 640x640 tiles (stride 320), detect on every tile, map boxes
back to the full photo and merge duplicates with NMS - the same way the M2c model was trained/evaluated.
NOTE: the NTUST numbers include photos the fine-tuned models trained on; the fair NTUST score is the
held-out test split in the training notebook. mems_60 and the look-only sets were never trained on.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, DFineForObjectDetection

CLASSES = ["wire_sweep", "lifted_ball", "lifted_stitch", "missing_wire", "broken_wire",
           "off_center_ball", "wire_short", "sagging_loop"]

# real label -> set of our classes that count as a correct find (None = good wire)
MAPS = {
    "national-taiwan": {"sagged wire": {"wire_sweep", "sagging_loop"},
                        "missing wire": {"missing_wire"}, "good wire": None},
    "mems_60": {"gold_wire_bonding_broken": {"broken_wire"},
                "aluminum_wire_bonding_broken": {"broken_wire"}},
}
LOOKONLY = ["apd-lab", "m11-12-13", "parmi", "reebas", "memsregions"]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def overlap_frac(pred, gt):
    """Fraction of the predicted box lying inside the GT box (labels here are loose)."""
    x0, y0 = max(pred[0], gt[0]), max(pred[1], gt[1])
    x1, y1 = min(pred[2], gt[2]), min(pred[3], gt[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    return inter / area if area > 0 else 0.0


IMG_SIZE, TILE_SCALE, TILE_STRIDE = 640, 0.5, 320


def positions(length, size, stride):
    if length <= size:
        return [0]
    p = list(range(0, length - size + 1, stride))
    if p[-1] + size < length:
        p.append(length - size)
    return p


class Detector:
    def __init__(self, path, tiled=False):
        self.proc = AutoImageProcessor.from_pretrained(path)
        self.model = DFineForObjectDetection.from_pretrained(path).eval()
        self.tiled = tiled

    @torch.no_grad()
    def __call__(self, img: Image.Image, thr: float):
        if self.tiled:
            return self._tiled(img, thr)
        out = self.model(**self.proc(images=img, return_tensors="pt"))
        r = self.proc.post_process_object_detection(out, threshold=thr, target_sizes=[img.size[::-1]])[0]
        return [(b.tolist(), CLASSES[int(l)], float(s)) for b, l, s in zip(r["boxes"], r["labels"], r["scores"])]

    def _tiled(self, img, thr):
        w, h = round(img.width * TILE_SCALE), round(img.height * TILE_SCALE)
        small = img.resize((w, h), Image.LANCZOS)
        offs = [(x, y) for y in positions(h, IMG_SIZE, TILE_STRIDE) for x in positions(w, IMG_SIZE, TILE_STRIDE)]
        B, S, L = [], [], []
        for i in range(0, len(offs), 8):                       # 8 tiles per batch keeps RAM low
            chunk = offs[i:i + 8]
            crops = [small.crop((x, y, x + IMG_SIZE, y + IMG_SIZE)) for x, y in chunk]
            out = self.model(**self.proc(images=crops, return_tensors="pt"))
            res = self.proc.post_process_object_detection(out, threshold=thr,
                                                          target_sizes=[(IMG_SIZE, IMG_SIZE)] * len(crops))
            for (x, y), r in zip(chunk, res):
                b = r["boxes"].clone()
                b[:, [0, 2]] += x; b[:, [1, 3]] += y
                B.append(b / TILE_SCALE); S.append(r["scores"]); L.append(r["labels"])
        B, S, L = torch.cat(B), torch.cat(S), torch.cat(L)
        keep = torchvision.ops.batched_nms(B, S, L, 0.5)
        return [(B[k].tolist(), CLASSES[int(L[k])], float(S[k])) for k in keep]


def load_coco(folder: Path):
    items = []
    for f in sorted(folder.glob("*/_annotations.coco.json")):
        c = json.loads(f.read_text())
        names = {x["id"]: x["name"] for x in c["categories"]}
        by_img = {}
        for a in c["annotations"]:
            x, y, w, h = a["bbox"]
            by_img.setdefault(a["image_id"], []).append(([x, y, x + w, y + h], names[a["category_id"]]))
        for im in c["images"]:
            items.append((f.parent / im["file_name"], by_img.get(im["id"], [])))
    return items


def draw(img, gts, preds, size=480):
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    lw = max(2, img.width // 300)
    for b, name in gts:
        d.rectangle(b, outline=(255, 255, 255), width=lw)
    for b, cls, s in preds:
        d.rectangle(b, outline=(0, 255, 0), width=lw)
        d.text((b[0] + 3, b[1] + 3), f"{cls} {s:.2f}", fill=(0, 255, 0))
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def sheet(tiles, path, cols=4):
    if not tiles:
        return
    s = tiles[0].width
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * s, rows * s))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * s, (i // cols) * s))
    out.save(path, quality=85)


def score(items, mapping, det, thr, out: Path, tag: str):
    found_any = found_cls = total = 0
    good_total = good_flagged = 0
    img_def = img_def_flag = img_ok = img_ok_flag = 0
    per_label = {}
    tiles = []
    for i, (p, gts) in enumerate(items):
        img = Image.open(p).convert("RGB")
        preds = det(img, thr)
        gts = [(b, n) for b, n in gts if n in mapping]
        has_def = any(mapping[n] is not None for _, n in gts)
        if has_def:
            img_def += 1; img_def_flag += bool(preds)
        else:
            img_ok += 1; img_ok_flag += bool(preds)
        for b, n in gts:
            hit = [pr for pr in preds if iou(pr[0], b) >= 0.3 or overlap_frac(pr[0], b) >= 0.6]
            if mapping[n] is None:
                good_total += 1; good_flagged += bool(hit)
                continue
            total += 1
            ok_any, ok_cls = bool(hit), any(pr[1] in mapping[n] for pr in hit)
            found_any += ok_any; found_cls += ok_cls
            pl = per_label.setdefault(n, [0, 0, 0]); pl[0] += 1; pl[1] += ok_any; pl[2] += ok_cls
        if i < 24:
            tiles.append(draw(img, gts, preds))
    sheet(tiles, out / f"{tag}.jpg")
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    return {
        "images": len(items),
        "defect_boxes": total,
        "found_any_class_%": pct(found_any, total),
        "found_right_class_%": pct(found_cls, total),
        "per_label": {k: {"n": v[0], "found_%": pct(v[1], v[0]), "right_class_%": pct(v[2], v[0])}
                      for k, v in per_label.items()},
        "good_wires": good_total,
        "good_wires_false_alarm_%": pct(good_flagged, good_total),
        "defect_images_flagged_%": pct(img_def_flag, img_def),
        "clean_images_flagged_%": pct(img_ok_flag, img_ok),
    }


def look(paths, det, thr, out: Path, tag: str):
    tiles, counts, classes = [], [], {}
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        preds = det(img, thr)
        counts.append(len(preds))
        for _, c, _ in preds:
            classes[c] = classes.get(c, 0) + 1
        if i < 24:
            tiles.append(draw(img, [], preds))
    sheet(tiles, out / f"{tag}.jpg")
    return {"images": len(paths), "mean_detections_per_image": round(float(np.mean(counts)), 2) if counts else 0,
            "images_with_any_detection_%": round(100 * np.mean([c > 0 for c in counts]), 1) if counts else 0,
            "classes": classes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--real", default="real")
    ap.add_argument("--tiled", action="store_true", help="tile the photo like the M2c model")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    det = Detector(a.model, tiled=a.tiled)
    report = {"model": a.model, "tiled": a.tiled, "threshold": a.thr, "scored": {}, "look_only": {}}
    for d in sorted(glob.glob(f"{a.real}/roboflow/*/")):
        name = Path(d).name
        key = next((k for k in MAPS if k in name), None)
        if key:
            report["scored"][name] = r = score(load_coco(Path(d)), MAPS[key], det, a.thr, out, key)
            print(name, json.dumps(r, indent=1), flush=True)
        elif any(k in name for k in LOOKONLY):
            paths = [p for p, _ in load_coco(Path(d))]
            report["look_only"][name] = r = look(paths, det, a.thr, out, name.split("__")[-1])
            print(name, r, flush=True)
    cur = sorted(p for p in glob.glob(f"{a.real}/curated/images/*") if p.lower().endswith((".jpg", ".png")))
    report["look_only"]["curated_web_photos"] = r = look(cur, det, a.thr, out, "curated")
    print("curated", r, flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
