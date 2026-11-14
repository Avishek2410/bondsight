"""Rank collected images by similarity to hand-picked wire-bond examples (few-shot CLIP).

1. Computes a CLIP embedding per image once, cached in real/embeddings.npz (new images are
   added incrementally on later runs).
2. Score = mean cosine similarity to the 5 nearest positive examples
           - max similarity to a broad bank of "not a wire bond" text descriptions.
3. Writes real/ranked.csv and contact sheets real/review2/sheet_XX.jpg (best first).

    tools/.venv/Scripts/python tools/rank_by_examples.py --root real --examples tools/wirebond_examples.txt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw

EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
NOT_WIREBOND = [
    "a photo of a sports ball", "a 3D render of a molecule", "a glass beaker with liquid",
    "a guitar", "a person", "people playing golf", "a city street", "a landscape",
    "a diagram", "an exploded technical drawing", "a car", "an animal", "food",
    "a printed circuit board with soldered components", "a computer motherboard",
    "a laptop or keyboard", "audio equipment knobs", "a logo or badge", "text on paper",
    "a black plastic chip package", "a CPU from above with a heat spreader",
    "a die shot of a processor surface with no wires", "a colourful microchip die photo",
    "a wafer", "electrical cables", "a coil of wire", "jewellery", "a machine in a factory",
    "a microscope", "a light bulb", "a board game", "a building", "a plant or flower",
]


def files_under(root: Path) -> list[Path]:
    fs = [p for p in root.glob("*/images/*") if p.suffix.lower() in EXT]
    fs += [p for p in root.glob("ttu-tornado/*") if p.suffix.lower() in EXT]
    return sorted(fs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="real")
    ap.add_argument("--examples", default="tools/wirebond_examples.txt")
    ap.add_argument("--sheet-top", type=int, default=600)
    a = ap.parse_args()
    root = Path(a.root)

    model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    cache = root / "embeddings.npz"
    emb: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache)  # plain arrays only, no pickle
        emb = dict(zip(z["names"].tolist(), z["vecs"]))
    files = files_under(root)
    todo = [f for f in files if f.relative_to(root).as_posix() not in emb]
    print(f"{len(files)} images, {len(todo)} need embedding", flush=True)
    for i in range(0, len(todo), 32):
        chunk, ims = [], []
        for f in todo[i:i + 32]:
            try:
                ims.append(prep(Image.open(f).convert("RGB")))
                chunk.append(f)
            except Exception:
                pass
        if ims:
            with torch.no_grad():
                v = model.encode_image(torch.stack(ims))
                v = (v / v.norm(dim=-1, keepdim=True)).numpy()
            for f, vec in zip(chunk, v):
                emb[f.relative_to(root).as_posix()] = vec
        print(f"embedded {min(i + 32, len(todo))}/{len(todo)}", flush=True)
    names = sorted(emb)
    V = np.stack([emb[n] for n in names])
    np.savez(cache, names=np.array(names, dtype=str), vecs=V)

    ex = [l.strip() for l in Path(a.examples).read_text().splitlines()
          if l.strip() and not l.startswith("#")]
    idx = [names.index(e) for e in ex if e in names]
    print(f"{len(idx)} of {len(ex)} examples found", flush=True)
    P = V[idx]
    with torch.no_grad():
        T = model.encode_text(tok(NOT_WIREBOND))
        T = (T / T.norm(dim=-1, keepdim=True)).numpy()
    sim_pos = np.sort(V @ P.T, axis=1)[:, -5:].mean(axis=1)
    sim_neg = (V @ T.T).max(axis=1)
    score = sim_pos - 0.5 * sim_neg
    order = np.argsort(-score)

    with (root / "ranked.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "file", "score", "sim_examples", "sim_not"])
        for r, i in enumerate(order):
            w.writerow([r, names[i], f"{score[i]:.4f}", f"{sim_pos[i]:.4f}", f"{sim_neg[i]:.4f}"])

    rev = root / "review2"
    rev.mkdir(exist_ok=True)
    for old in rev.glob("*.jpg"):
        old.unlink()
    top = order[:a.sheet_top]
    for k in range(0, len(top), 40):
        sheet = Image.new("RGB", (8 * 240, 5 * 200), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for j, i in enumerate(top[k:k + 40]):
            im = Image.open(root / names[i]).convert("RGB")
            im.thumbnail((236, 176))
            x, y = (j % 8) * 240, (j // 8) * 200
            sheet.paste(im, (x + 2, y + 2))
            d.text((x + 4, y + 180), f"#{k + j} {score[i]:.2f} {Path(names[i]).name[:20]}",
                   fill=(255, 255, 0))
        sheet.save(rev / f"sheet_{k // 40:02d}.jpg", quality=85)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
