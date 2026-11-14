"""Score collected images for "shows real wire bonds" with zero-shot CLIP, on CPU.

Writes real/relevance.csv (file, score, best_label) for every image under real/*/images and
contact sheets real/review/sheet_XX.jpg of the top-scoring images for a human check.

    tools/.venv/Scripts/python tools/filter_relevance.py --root real
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw

POSITIVE = [
    "a microscope photo of gold bond wires connecting a silicon die to package leads",
    "a close-up photo of wire bonding inside an opened integrated circuit",
    "a macro photo of thin bond wires on a semiconductor chip",
    "a photo of aluminium wedge bonded wires on a transistor die",
    "a decapsulated chip showing bond wires",
    "a photo of ball bonds on bond pads of a die",
]
NEGATIVE = [
    "a photo of a printed circuit board with components",
    "a photo of a packaged black chip without visible wires",
    "a photo of electrical cables or wires in a building",
    "a die shot of a chip surface without any bond wires",
    "a photo of a person", "a photo of a landscape or city",
    "a diagram or drawing", "a photo of a machine or factory equipment",
    "a photo of jewellery or gold chain", "a photo of a light bulb or LED lamp",
]
EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="real")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sheet-top", type=int, default=400)
    a = ap.parse_args()
    root = Path(a.root)
    files = sorted(p for p in root.glob("*/images/*") if p.suffix.lower() in EXT)
    files += sorted(p for p in root.glob("ttu-tornado/*") if p.suffix.lower() in EXT)
    print(f"{len(files)} images", flush=True)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, _, prep = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained)
    tok = open_clip.get_tokenizer(a.model)
    model.eval()
    labels = POSITIVE + NEGATIVE
    with torch.no_grad():
        t = model.encode_text(tok(labels))
        t /= t.norm(dim=-1, keepdim=True)

    rows = []
    for i in range(0, len(files), a.batch):
        chunk, ims = [], []
        for f in files[i:i + a.batch]:
            try:
                ims.append(prep(Image.open(f).convert("RGB")))
                chunk.append(f)
            except Exception:
                pass
        if not ims:
            continue
        with torch.no_grad():
            v = model.encode_image(torch.stack(ims))
            v /= v.norm(dim=-1, keepdim=True)
            p = (100 * v @ t.T).softmax(dim=-1).numpy()
        for f, pr in zip(chunk, p):
            rows.append((f, float(pr[:len(POSITIVE)].sum()), labels[int(pr.argmax())]))
        print(f"scored {min(i + a.batch, len(files))}/{len(files)}", flush=True)

    rows.sort(key=lambda r: -r[1])
    with (root / "relevance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "score", "best_label"])
        for f, s, l in rows:
            w.writerow([f.relative_to(root).as_posix(), f"{s:.4f}", l])

    rev = root / "review"
    rev.mkdir(exist_ok=True)
    top = rows[:a.sheet_top]
    for k in range(0, len(top), 40):
        sheet = Image.new("RGB", (8 * 240, 5 * 200), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for j, (f, s, _) in enumerate(top[k:k + 40]):
            im = Image.open(f).convert("RGB")
            im.thumbnail((236, 176))
            x, y = (j % 8) * 240, (j // 8) * 200
            sheet.paste(im, (x + 2, y + 2))
            d.text((x + 4, y + 180), f"#{k + j} {s:.2f} {f.name[:22]}", fill=(255, 255, 0))
        sheet.save(rev / f"sheet_{k // 40:02d}.jpg", quality=85)
    print(f"kept-candidates >=0.5: {sum(r[1] >= 0.5 for r in rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
