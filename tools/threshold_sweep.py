"""Compare models at many confidence thresholds on the NTUST test photos (tiled).

What: runs each model ONCE per test photo with a very low threshold, keeps every answer and its
confidence, then counts "defects found" and "false alarms" (same rules as the training notebooks)
for each threshold. Prints a table per model and the threshold where each one matches a target
false-alarm rate.
Why: a model that ranks defects better (higher mAP) can look worse at a fixed threshold because
it is simply more confident overall. Comparing at EQUAL false alarms is the fair test.

    tools/.venv/Scripts/python tools/threshold_sweep.py \
        --model M2c=training/runs/local_v1/best_model_tiled \
        --model M2e=training/runs/local_v1/best_model_macro --target-fa 0.30
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from reality_check import Detector, iou

CLASSES = ["wire_sweep", "lifted_ball", "lifted_stitch", "missing_wire", "broken_wire",
           "off_center_ball", "wire_short", "sagging_loop"]
LABEL_MAP = {"sagged wire": "wire_sweep", "missing wire": "missing_wire", "good wire": None}
SEED = 42


def ntust_test(real: str) -> list[dict]:
    """The same 23 test photos as the M2b..M2e notebooks (split by source photo, seed 42)."""
    root = glob.glob(f"{real}/roboflow/national-taiwan*/")[0]
    by_source = defaultdict(list)
    for f in sorted(glob.glob(f"{root}*/_annotations.coco.json")):
        coco = json.load(open(f))
        names = {c["id"]: c["name"] for c in coco["categories"]}
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        for im in coco["images"]:
            defects, good = [], []
            for a in anns[im["id"]]:
                x, y, w, h = a["bbox"]
                m = LABEL_MAP[names[a["category_id"]]]
                (good if m is None else defects).append(([x, y, x + w, y + h], m))
            src = im["file_name"].split(".rf.")[0]
            by_source[src].append({"path": str(Path(f).parent / im["file_name"]),
                                   "defects": defects, "good": [b for b, _ in good]})
    sources = sorted(by_source)
    random.Random(SEED).shuffle(sources)
    return [it for s in sources[:round(len(sources) * 0.2)] for it in by_source[s]]


def score(preds: list[list], items: list[dict], thr: float) -> tuple[float, float, int, int]:
    found = total = flag = gtot = 0
    for p, it in zip(preds, items):
        keep = [(b, c) for b, c, s in p if s >= thr]
        for b, c in it["defects"]:
            total += 1
            found += any(iou(pb, b) >= 0.3 and pc == c for pb, pc in keep)
        for g in it["good"]:
            gtot += 1
            flag += any(iou(pb, g) >= 0.3 for pb, _ in keep)
    return found / max(total, 1), flag / max(gtot, 1), total, gtot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=path (repeatable)")
    ap.add_argument("--real", default="real")
    ap.add_argument("--target-fa", type=float, default=0.30)
    a = ap.parse_args()
    items = ntust_test(a.real)
    thrs = [round(0.20 + 0.05 * i, 2) for i in range(14)]          # 0.20 .. 0.85
    print(f"NTUST test: {len(items)} images")
    for spec in a.model:
        name, path = spec.split("=", 1)
        det = Detector(path, tiled=True)
        preds = [det(Image.open(it["path"]).convert("RGB"), 0.05) for it in items]
        rows = [(t, *score(preds, items, t)) for t in thrs]
        print(f"\n{name}  ({rows[0][3]} defects, {rows[0][4]} good wires)")
        print("  thr   found   false alarms")
        for t, f, fa, _, _ in rows:
            print(f"  {t:.2f}  {100 * f:5.0f}%  {100 * fa:6.0f}%")
        ok = [r for r in rows if r[2] <= a.target_fa]
        if ok:
            t, f, fa, _, _ = ok[0]
            print(f"  -> at <= {100 * a.target_fa:.0f}% false alarms: thr {t:.2f} finds {100 * f:.0f}% "
                  f"(false alarms {100 * fa:.0f}%)")


if __name__ == "__main__":
    main()
