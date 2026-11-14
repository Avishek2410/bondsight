"""Build the M2e notebook (training/m2e_finetune_macro.ipynb): M2c + macro (close-up) synthetic
images + MEMS photos and close-up crops of them, and its smoke-test script (training/_smoke_macro.py)
from ONE list of cells.

M2d (M2c + 43 MEMS photos) still found 0% of the MEMS broken wires: the model had never seen a
close-up of a few big (often aluminium) wires. M2e adds the generator's macro mode
(tools/make_macro_set.sh -> out/macro_v1) and close-up crops of the MEMS training photos.

    tools/.venv/Scripts/python training/build_macro_notebook.py
    SMOKE=1 DATA_ROOT=out/train_v1_jpg MACRO_ROOT=out/macro_v1 REAL_ROOT=real TILE_DIR=out/ntust_tiles_s05 \
        CROP_DIR=<scratch> START_MODEL=training/runs/local_v1/best_model_tiled tools/.venv/Scripts/python training/_smoke_macro.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("md", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


# --------------------------------------------------------------------------------------------
md("""
# Part M2e: Teaching the model what a close-up looks like

**Where we are.** Part M2d added 43 MEMS sensor photos to M2c. It learned which parts of a
MEMS chip are *not* wire flaws (false alarms halved), but it still found **0%** of the
broken wires. The reason: in the MEMS photos a wire is shown **close up**: only a few wires,
each hundreds of pixels long, often thick silver **aluminium** wires. Every picture the model
had trained on showed a whole chip with many thin gold wires. It had simply never seen a
close-up, and 43 photos are too few to learn one.

**The fix here: two new kinds of training pictures.**
1. **Macro synthetic images** (2,400 of them, from the generator's new close-up mode): 2–4×
   zoomed crops around one wire, gold or aluminium, on white, green, brown or dark boards,
   with broken wires whose ends curl and droop like real snapped wires.
2. **Close-up crops of the MEMS training photos**: each broken wire is cut out a few times
   at different zooms, plus some crops with no defect, so the real broken wires are seen
   at close-up size too.

We still mix in the NTUST tiles and the ordinary synthetic images, so the model does not
forget what it already knows.

**Fewer epochs:** in M2d the validation loss was best after epoch 1 and then got worse
(the model started memorising the few MEMS photos). So this time: 3 epochs, smaller
learning rate. The best epoch is kept automatically.

Training takes about **1.5 hours** on your PC.
""")

md("""
## Step 0: Settings

- `START_MODEL`: we continue from **M2c** (still the best model), not from M2d.
- `MACRO_N = 1200`: how many macro (close-up) synthetic images per epoch (out of 2,400).
- `SYNTH_N = 400`: ordinary whole-chip synthetic images, so the old skills are not forgotten.
- `CROPS_PER_DEFECT = 3`, `NEG_CROPS = 2`: close-up crops cut from each MEMS training photo
  (3 around each broken wire, 2 random ones with no broken wire in them).
- `MEMS_REPEAT = 2`: the whole MEMS photos are still shown, twice per epoch.
- Tiling settings (`TILE_SCALE`, `MIN_VISIBLE`) are the same as in M2c.
""")
code("""
import os
SMOKE = os.environ.get("SMOKE") == "1"
EPOCHS = 1 if SMOKE else 3
LR = 3e-5
BATCH = 2 if SMOKE else 16
SYNTH_N = 8 if SMOKE else 400
MACRO_N = 8 if SMOKE else 1200
IMG_SIZE = 640
TILE_SCALE = 0.5
TILE_STRIDE = 320          # tiles overlap by half
MIN_VISIBLE = 0.6
SEED = 42
MEMS_REPEAT = 2
CROPS_PER_DEFECT = 3       # close-up crops around each MEMS broken wire
NEG_CROPS = 2              # close-up crops with no broken wire, per MEMS photo
START_MODEL = os.environ.get("START_MODEL", "best_model_tiled")   # from Part M2c (the current best)
OUT_MODEL = "best_model_macro"
print("smoke test" if SMOKE else "macro fine-tuning", "| epochs:", EPOCHS, "| start from:", START_MODEL)
""")

md("""
## Step 1: Tools
""")
code("""
import json, glob, random, time, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch, torchvision
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, DFineForObjectDetection
from torchmetrics.detection import MeanAveragePrecision

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
print("Running on:", DEVICE)
""")

md("""
## Step 2: Load the data and split it (same split as Part M2b)

Same label translation (`sagged wire` → `wire_sweep`, `missing wire` → `missing_wire`,
`good wire` → no box) and the same split by original photo, so the 23 test photos are the
same ones as in Part M2b and the results are directly comparable.
""")
code("""
DATA_ROOT = os.environ.get("DATA_ROOT", "C:/Users/AVI-TECH/projects/wirebond-ai/out/train_v1_jpg")
REAL_ROOT = os.environ.get("REAL_ROOT", "C:/Users/AVI-TECH/projects/wirebond-ai/real")
TILE_DIR = Path(os.environ.get("TILE_DIR", "C:/Users/AVI-TECH/projects/wirebond-ai/out/ntust_tiles_s05"))
MACRO_ROOT = os.environ.get("MACRO_ROOT", "C:/Users/AVI-TECH/projects/wirebond-ai/out/macro_v1")
CROP_DIR = Path(os.environ.get("CROP_DIR", "C:/Users/AVI-TECH/projects/wirebond-ai/out/mems_crops"))
NTUST = glob.glob(f"{REAL_ROOT}/roboflow/national-taiwan*/")[0]
PRODUCTS = sorted(p.name for p in Path(DATA_ROOT).glob("p*") if (p / "config.json").exists())
CLASSES = json.load(open(f"{DATA_ROOT}/{PRODUCTS[0]}/annotations/top.json"))["categories"]
CLASSES = [c["name"] for c in sorted(CLASSES, key=lambda c: c["id"])]
LABEL_MAP = {"sagged wire": "wire_sweep", "missing wire": "missing_wire", "good wire": None}

by_source = defaultdict(list)
for f in sorted(glob.glob(f"{NTUST}*/_annotations.coco.json")):
    coco = json.load(open(f))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    for im in coco["images"]:
        ours, good = [], []
        for a in anns[im["id"]]:
            m = LABEL_MAP[names[a["category_id"]]]
            if m is None:
                good.append(a["bbox"])
            else:
                ours.append({"bbox": a["bbox"], "category_id": CLASSES.index(m) + 1,
                             "area": a["bbox"][2] * a["bbox"][3]})
        src = im["file_name"].split(".rf.")[0]
        by_source[src].append({"path": str(Path(f).parent / im["file_name"]), "anns": ours,
                               "good": good, "source": src})

sources = sorted(by_source)
random.Random(SEED).shuffle(sources)
n_test, n_val = round(len(sources) * 0.2), round(len(sources) * 0.1)
REAL_SPLIT = {"test": sources[:n_test], "val": sources[n_test:n_test + n_val], "train": sources[n_test + n_val:]}
REAL = {k: [it for s in v for it in by_source[s]] for k, v in REAL_SPLIT.items()}
if SMOKE:
    REAL = {k: v[:3] for k, v in REAL.items()}
print("NTUST:", {k: f"{len(REAL_SPLIT[k])} photos / {len(v)} images" for k, v in REAL.items()})

# ---- MEMS sensor photos (mems_60): broken wires -> broken_wire, other flaws -> no box ----
MEMS_DIR = glob.glob(f"{REAL_ROOT}/roboflow/*mems_60*/")[0]
MEMS_MAP = {"gold_wire_bonding_broken": "broken_wire", "aluminum_wire_bonding_broken": "broken_wire"}
mems_items = []
for f in sorted(glob.glob(f"{MEMS_DIR}*/_annotations.coco.json")):
    coco = json.load(open(f))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    for im in coco["images"]:
        ours, other = [], []
        for a in anns[im["id"]]:
            m = MEMS_MAP.get(names[a["category_id"]])
            if m:
                ours.append({"bbox": a["bbox"], "category_id": CLASSES.index(m) + 1,
                             "area": a["bbox"][2] * a["bbox"][3]})
            else:
                other.append(a["bbox"])          # a real flaw, but not a wire flaw: must NOT be flagged
        mems_items.append({"path": str(Path(f).parent / im["file_name"]), "anns": ours, "good": other,
                           "source": im["file_name"].split(".rf.")[0]})
mems_sources = sorted({it["source"] for it in mems_items})
random.Random(SEED).shuffle(mems_sources)
mt, mv = round(len(mems_sources) * 0.2), round(len(mems_sources) * 0.1)
MEMS_SPLIT = {"test": set(mems_sources[:mt]), "val": set(mems_sources[mt:mt + mv]),
              "train": set(mems_sources[mt + mv:])}
MEMS = {k: [it for it in mems_items if it["source"] in v] for k, v in MEMS_SPLIT.items()}
if SMOKE:
    MEMS = {k: v[:3] for k, v in MEMS.items()}
print("MEMS :", {k: f"{len(v)} photos, {sum(len(i['anns']) for i in v)} broken wires, "
                    f"{sum(len(i['good']) for i in v)} non-wire flaws" for k, v in MEMS.items()})
""")

md("""
## Step 3: Cut the real photos into tiles

For each photo: shrink to half size, then cut 640 × 640 tiles every 320 pixels (so they
overlap). Each defect box is moved into tile coordinates. The tiles are saved to disk once;
the next run reuses them.
""")
code("""
def positions(length, size, stride):
    if length <= size:
        return [0]
    p = list(range(0, length - size + 1, stride))
    if p[-1] + size < length:
        p.append(length - size)
    return p

def tile_offsets(w, h):
    return [(x, y) for y in positions(h, IMG_SIZE, TILE_STRIDE) for x in positions(w, IMG_SIZE, TILE_STRIDE)]

def cut_box(b, s, x, y):
    # scale a COCO [x,y,w,h] box, clip it to the tile, return (clipped box, visible fraction)
    x0, y0, x1, y1 = b[0] * s - x, b[1] * s - y, (b[0] + b[2]) * s - x, (b[1] + b[3]) * s - y
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(IMG_SIZE, x1), min(IMG_SIZE, y1)
    full = (x1 - x0) * (y1 - y0)
    vis = max(0, cx1 - cx0) * max(0, cy1 - cy0) / full if full > 0 else 0
    return [cx0, cy0, cx1 - cx0, cy1 - cy0], vis

def make_tiles(items, split):
    out = []
    folder = TILE_DIR / split
    folder.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for it in items:
        img = Image.open(it["path"]).convert("RGB")
        w, h = round(img.width * TILE_SCALE), round(img.height * TILE_SCALE)
        img = img.resize((w, h), Image.LANCZOS)
        for x, y in tile_offsets(w, h):
            anns, good, cut = [], [], False
            for a in it["anns"]:
                bb, vis = cut_box(a["bbox"], TILE_SCALE, x, y)
                if vis >= MIN_VISIBLE:
                    anns.append({"bbox": bb, "category_id": a["category_id"], "area": bb[2] * bb[3]})
                elif vis > 0.1:
                    cut = True
            for g in it["good"]:
                bb, vis = cut_box(g, TILE_SCALE, x, y)
                if vis >= MIN_VISIBLE:
                    good.append(bb)
            if cut:
                skipped += 1
                continue
            p = folder / f"{Path(it['path']).stem}_{x}_{y}.jpg"
            if not p.exists():
                img.crop((x, y, x + IMG_SIZE, y + IMG_SIZE)).save(p, quality=95)
            out.append({"path": str(p), "anns": anns, "good": good})
    return out, skipped

t = time.time()
TILES = {}
for k in ("train", "val"):
    TILES[k], skipped = make_tiles(REAL[k], k)
    n_def = sum(1 for it in TILES[k] if it["anns"])
    print(f"{k:5s}: {len(TILES[k])} tiles ({n_def} with a defect, {len(TILES[k]) - n_def} defect-free), "
          f"{skipped} skipped because a defect was cut in half")
print(f"({time.time() - t:.0f}s)")
""")

md("""
## Step 3b: Cut close-ups out of the MEMS photos

For every broken wire in a MEMS photo: cut a square around it, 1.2 to 2 times the size of
its box, at a random position that still contains the whole box. Plus a couple of random
squares with **no** broken wire in them (they teach "close-up, but nothing wrong").
A crop that would cut a broken wire in half is thrown away, like with the NTUST tiles.

We do this for the **test** photos too, with a fixed random seed, so Step 7 can also ask:
*given a close-up, does the model find the break?*
""")
code("""
def mems_crops(items, split, seed):
    rng = random.Random(seed)
    folder = CROP_DIR / split
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for it in items:
        img = Image.open(it["path"]).convert("RGB")
        W, H = img.size
        plans = []
        for a in it["anns"]:
            x, y, w, h = a["bbox"]
            for _ in range(CROPS_PER_DEFECT):
                side = min(max(w, h) * rng.uniform(1.2, 2.0), W, H)
                side = max(side, min(320, W, H))
                cx0 = rng.uniform(max(0, x + w - side), max(0, min(x, W - side)) + 1e-6)
                cy0 = rng.uniform(max(0, y + h - side), max(0, min(y, H - side)) + 1e-6)
                plans.append((min(cx0, W - side), min(cy0, H - side), side))
        for _ in range(NEG_CROPS):
            side = rng.uniform(0.35, 0.6) * min(W, H)
            plans.append((rng.uniform(0, W - side), rng.uniform(0, H - side), side))
        for k, (x0, y0, side) in enumerate(plans):
            anns, good, cut = [], [], False
            for a in it["anns"]:
                bb, vis = cut_box([a["bbox"][0] - x0, a["bbox"][1] - y0, *a["bbox"][2:]], IMG_SIZE / side, 0, 0)
                if vis >= MIN_VISIBLE:
                    anns.append({"bbox": bb, "category_id": a["category_id"], "area": bb[2] * bb[3]})
                elif vis > 0.1:
                    cut = True
            for g in it["good"]:
                bb, vis = cut_box([g[0] - x0, g[1] - y0, *g[2:]], IMG_SIZE / side, 0, 0)
                if vis >= MIN_VISIBLE:
                    good.append(bb)
            if cut:
                continue
            p = folder / f"{Path(it['path']).stem}_c{k}.jpg"
            if not p.exists():
                img.crop((round(x0), round(y0), round(x0 + side), round(y0 + side))).resize(
                    (IMG_SIZE, IMG_SIZE), Image.LANCZOS).save(p, quality=95)
            out.append({"path": str(p), "anns": anns, "good": good})
    return out

CROPS = {k: mems_crops(MEMS[k], k, SEED + i) for i, k in enumerate(("train", "val", "test"))}
for k, v in CROPS.items():
    print(f"{k:5s}: {len(v)} MEMS close-ups ({sum(1 for c in v if c['anns'])} with a broken wire)")
""")

md("""
## Step 4: Look at some tiles and MEMS close-ups

Top row: NTUST tiles. Bottom row: MEMS close-ups. Red = defects, green = good wires (NTUST)
or non-wire flaws (MEMS). Notice that many tiles have **no defect at all**. That
is good: they teach the model when to stay quiet.
""")
code("""
def show(item, preds=None, size=450):
    img = Image.open(item["path"]).convert("RGB")
    d = ImageDraw.Draw(img); w = max(2, img.width // 300)
    for g in item["good"]:
        d.rectangle([g[0], g[1], g[0] + g[2], g[1] + g[3]], outline="lime", width=w)
    for a in item["anns"]:
        b = a["bbox"]; d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline="red", width=w)
        d.text((b[0] + 4, b[1] + 4), CLASSES[a["category_id"] - 1], fill="red")
    for b, l, s in (preds or []):
        d.rectangle(b, outline="yellow", width=w)
        d.text((b[0] + 4, b[3] - 14), f"{CLASSES[l]} {s:.2f}", fill="yellow")
    img.thumbnail((size, size))
    return img

grid = Image.new("RGB", (900, 900))
for i, it in enumerate(random.Random(3).sample(TILES["train"], min(2, len(TILES["train"])))
                       + random.Random(3).sample(CROPS["train"], min(2, len(CROPS["train"])))):
    grid.paste(show(it), ((i % 2) * 450, (i // 2) * 450))
grid
""")

md("""
## Step 5: Mix everything together, load the model
""")
code("""
prods = PRODUCTS[:]
random.Random(SEED).shuffle(prods)
n = max(1, len(prods) // 10)
SYN_SPLIT = {"test": prods[:n], "train": prods[2 * n:]}

def load_synth(products, root=DATA_ROOT):
    items = []
    for p in products:
        for view in ("top", "angled"):
            coco = json.load(open(f"{root}/{p}/annotations/{view}.json"))
            anns = defaultdict(list)
            for a in coco["annotations"]:
                anns[a["image_id"]].append(a)
            for im in coco["images"]:
                items.append({"path": f"{root}/{p}/images/{view}/{im['file_name']}",
                              "anns": anns[im["id"]], "good": []})
    return items

SYN_TRAIN = random.Random(SEED).sample(load_synth(SYN_SPLIT["train"]), SYNTH_N)
SYN_TEST = random.Random(SEED).sample(load_synth(SYN_SPLIT["test"]), 8 if SMOKE else 300)
# macro (close-up) synthetic: hold 2 of the 24 designs back for the test
mprods = sorted(p.name for p in Path(MACRO_ROOT).glob("p*") if (p / "config.json").exists())
random.Random(SEED).shuffle(mprods)
MACRO_SPLIT = {"test": mprods[:2], "train": mprods[2:]}
MACRO_TRAIN = random.Random(SEED).sample(load_synth(MACRO_SPLIT["train"], MACRO_ROOT), MACRO_N)
MACRO_TEST = random.Random(SEED).sample(load_synth(MACRO_SPLIT["test"], MACRO_ROOT), 8 if SMOKE else 200)

TRAIN = TILES["train"] + MEMS["train"] * MEMS_REPEAT + CROPS["train"] + SYN_TRAIN + MACRO_TRAIN
random.Random(SEED).shuffle(TRAIN)
print(f"training mix per epoch: {len(TILES['train'])} NTUST tiles + {len(MEMS['train'])} MEMS photos x "
      f"{MEMS_REPEAT} + {len(CROPS['train'])} MEMS close-ups + {len(SYN_TRAIN)} synthetic + "
      f"{len(MACRO_TRAIN)} macro synthetic = {len(TRAIN)}")

processor = AutoImageProcessor.from_pretrained(START_MODEL, size={"height": IMG_SIZE, "width": IMG_SIZE})
model = DFineForObjectDetection.from_pretrained(START_MODEL).to(DEVICE)

class DS(torch.utils.data.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["path"]).convert("RGB")
        target = {"image_id": i, "annotations": [
            {"bbox": a["bbox"], "category_id": a["category_id"] - 1, "area": a["area"], "iscrowd": 0}
            for a in it["anns"]]}
        enc = processor(images=img, annotations=target, return_tensors="pt")
        return enc["pixel_values"][0], enc["labels"][0]

def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

loaders = {"train": torch.utils.data.DataLoader(DS(TRAIN), batch_size=BATCH, shuffle=True, collate_fn=collate),
           "val": torch.utils.data.DataLoader(DS(TILES["val"] + MEMS["val"] + CROPS["val"]), batch_size=BATCH, collate_fn=collate)}
print({k: len(v) for k, v in loaders.items()}, "batches")
""")

md("""
## Step 6: Fine-tune (NTUST tiles + MEMS photos and close-ups + synthetic + macro synthetic)
""")
code("""
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps = EPOCHS * len(loaders["train"])
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 50), pct_start=0.1)
history = {"train": [], "val": []}

def run_epoch(split, train):
    model.train(train)
    total, n = 0.0, 0
    for i, (px, labels) in enumerate(loaders[split]):
        px = px.to(DEVICE); labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]
        with torch.set_grad_enabled(train):
            loss = model(pixel_values=px, labels=labels).loss
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); sched.step()
        total += loss.item(); n += 1
        if SMOKE and i >= 1: break
        if train and i % 25 == 0:
            print(f"   batch {i}/{len(loaders[split])}  loss {loss.item():.3f}", flush=True)
    return total / max(n, 1)

best = math.inf
for ep in range(EPOCHS):
    t = time.time()
    tr = run_epoch("train", True); va = run_epoch("val", False)
    history["train"].append(tr); history["val"].append(va)
    if va < best:
        best = va; model.save_pretrained(OUT_MODEL); processor.save_pretrained(OUT_MODEL)
        print("   new best, saved", flush=True)
    print(f"epoch {ep + 1:2d}/{EPOCHS}  train loss {tr:.3f}  val loss (NTUST tiles + MEMS + close-ups) {va:.3f}  ({time.time() - t:.0f}s)")
""")

code("""
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.5))
plt.plot(range(1, len(history["train"]) + 1), history["train"], "o-", label="train loss (mix)")
plt.plot(range(1, len(history["val"]) + 1), history["val"], "o-", label="validation loss (real)")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=.3); plt.show()
""")

md("""
## Step 7: The honest test: M2c vs M2e

None of these test pictures were used for training:
1. **NTUST** (23 photos, tile by tile): did it keep what M2c already did well?
2. **MEMS whole photos** (12 photos): the same test M2c and M2d failed (0% found).
3. **MEMS close-ups** (crops of the same 12 test photos): given a close-up, does it find the break?
4. **Synthetic** (300) and **macro synthetic** (200, from 2 designs it never saw).

A new column, **"found (in region)"**: the MEMS labels often draw ONE big box around a
whole group of wires, while a model draws a tight box around the one broken wire. Those two
boxes barely overlap, so the strict "found" count can say *missed* even when the model
pointed at the right wire. "In region" counts a defect as found if a `broken_wire` answer
lies mostly (≥ 50%) inside the labelled box. We show both numbers; the strict one stays
the main score.

What we want: M2e clearly better on MEMS (whole and close-up), and not much worse elsewhere.
""")
code("""
THRESH = 0.4

def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

@torch.no_grad()
def predict_whole(m, path, thr):
    img = Image.open(path).convert("RGB")
    r = processor.post_process_object_detection(m(**processor(images=img, return_tensors="pt").to(DEVICE)),
                                                threshold=thr, target_sizes=[img.size[::-1]])[0]
    return r["boxes"].cpu(), r["scores"].cpu(), r["labels"].cpu()

@torch.no_grad()
def predict_tiled(m, path, thr):
    img = Image.open(path).convert("RGB")
    w, h = round(img.width * TILE_SCALE), round(img.height * TILE_SCALE)
    small = img.resize((w, h), Image.LANCZOS)
    offs = tile_offsets(w, h)
    crops = [small.crop((x, y, x + IMG_SIZE, y + IMG_SIZE)) for x, y in offs]
    out = m(**processor(images=crops, return_tensors="pt").to(DEVICE))
    res = processor.post_process_object_detection(out, threshold=thr, target_sizes=[(IMG_SIZE, IMG_SIZE)] * len(crops))
    B, S, L = [], [], []
    for (x, y), r in zip(offs, res):
        b = r["boxes"].cpu().clone()
        b[:, [0, 2]] += x; b[:, [1, 3]] += y
        B.append(b / TILE_SCALE); S.append(r["scores"].cpu()); L.append(r["labels"].cpu())
    B, S, L = torch.cat(B), torch.cat(S), torch.cat(L)
    keep = torchvision.ops.batched_nms(B, S, L, 0.5)
    return B[keep], S[keep], L[keep]

def inside(p, b):
    # share of predicted box p that lies inside labelled box b
    x0, y0, x1, y1 = max(p[0], b[0]), max(p[1], b[1]), min(p[2], b[2]), min(p[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0) / max((p[2] - p[0]) * (p[3] - p[1]), 1e-9)

def evaluate(m, items, predict):
    # 'good' boxes = must NOT be flagged (good wires in NTUST, non-wire flaws in MEMS)
    metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5], class_metrics=True)
    metric.warn_on_many_detections = False
    found = total = good_flag = good_total = region = 0
    for it in items:
        boxes, scores, labels = predict(m, it["path"], 0.05)
        gt_b = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                             for a in it["anns"]], dtype=torch.float32).reshape(-1, 4)
        gt_l = torch.tensor([a["category_id"] - 1 for a in it["anns"]], dtype=torch.long)
        metric.update([{"boxes": boxes, "scores": scores, "labels": labels}], [{"boxes": gt_b, "labels": gt_l}])
        keep = scores >= THRESH
        kb, kl = boxes[keep].tolist(), labels[keep].tolist()
        for b, l in zip(gt_b.tolist(), gt_l.tolist()):
            total += 1; found += any(iou(p, b) >= 0.3 and pl == l for p, pl in zip(kb, kl))
            region += any(pl == l and inside(p, b) >= 0.5 for p, pl in zip(kb, kl))
        for g in it["good"]:
            gb = [g[0], g[1], g[0] + g[2], g[1] + g[3]]
            good_total += 1; good_flag += any(iou(p, gb) >= 0.3 for p in kb)
    res = metric.compute()
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "n/a"
    return {"mAP@0.5": round(res["map_50"].item(), 3), "defects found": pct(found, total),
            "found (in region)": pct(region, total), "false alarms": pct(good_flag, good_total)}

tests = {"NTUST       (tiled)": (REAL["test"], predict_tiled),
         "MEMS        (whole)": (MEMS["test"], predict_whole),
         "MEMS   (close-ups) ": (CROPS["test"], predict_whole),
         "synthetic   (whole)": (SYN_TEST, predict_whole),
         "macro synthetic    ": (MACRO_TEST, predict_whole)}
RESULTS = {}
for name, path in (("M2c", START_MODEL), ("M2e", OUT_MODEL)):
    m = DFineForObjectDetection.from_pretrained(path).to(DEVICE).eval()
    for tname, (items, fn) in tests.items():
        RESULTS[(name, tname.strip())] = r = evaluate(m, items, fn)
        print(f"{name}  {tname}  {r}", flush=True)
    print()
""")

md("""
## Step 8: See it on MEMS test pictures

Top row: whole MEMS test photos. Bottom row: MEMS test close-ups.
Red = real broken wires, green = non-wire flaws (it must stay quiet there).
**Yellow** = M2e's answers.
""")
code("""
new = DFineForObjectDetection.from_pretrained(OUT_MODEL).to(DEVICE).eval()
grid = Image.new("RGB", (900, 900))
with_def = [c for c in CROPS["test"] if c["anns"]] or CROPS["test"]
for i, it in enumerate(random.Random(2).sample(MEMS["test"], min(2, len(MEMS["test"])))
                       + random.Random(2).sample(with_def, min(2, len(with_def)))):
    b, sc, l = predict_whole(new, it["path"], THRESH)
    grid.paste(show(it, list(zip(b.tolist(), l.tolist(), sc.tolist()))), ((i % 2) * 450, (i // 2) * 450))
grid
""")

md("""
## Step 9: Export M2e for the Pi

Saves `wirebond_dfine_nano_macro.onnx` (+ `.json`) next to this notebook and checks that the
ONNX version gives the same numbers as PyTorch. Whether it replaces M2c in the inspector app
is decided from the Step 7 table.
""")
code("""
import onnxruntime as ort
model_cpu = DFineForObjectDetection.from_pretrained(OUT_MODEL).eval()
dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(model_cpu, (dummy,), "wirebond_dfine_nano_macro.onnx", input_names=["pixel_values"],
                  output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
json.dump({"classes": CLASSES, "img_size": IMG_SIZE,
           "note": "M2e: synthetic (whole + macro) + NTUST wire bonding tiles + WHU mems_60 photos and crops (both CC BY 4.0)",
           "tiling": {"scale": TILE_SCALE, "tile": IMG_SIZE, "stride": TILE_STRIDE, "nms_iou": 0.5}},
          open("wirebond_dfine_nano_macro.json", "w"), indent=1)
px = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
with torch.no_grad():
    ref = model_cpu(pixel_values=px)
lg, bx = ort.InferenceSession("wirebond_dfine_nano_macro.onnx", providers=["CPUExecutionProvider"]).run(
    ["logits", "pred_boxes"], {"pixel_values": px.numpy()})
print("saved wirebond_dfine_nano_macro.onnx | max difference vs PyTorch:",
      float(max(np.abs(lg - ref.logits.numpy()).max(), np.abs(bx - ref.pred_boxes.numpy()).max())))
""")

md("""
## Done!

Say "status" (or send a screenshot of **Step 7**). If M2e wins, we drop the new `.onnx` into
the inspector app and rerun the reality check.
""")


# --------------------------------------------------------------------------------------------
def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nbf.write(nb, HERE / "m2e_finetune_macro.ipynb")
    script = ["# AUTO-GENERATED from build_macro_notebook.py - smoke-test copy\n"]
    for k, s in CELLS:
        if k == "code":
            lines = s.splitlines()
            if lines and lines[-1] == "grid":
                lines[-1] = "grid.save('_smoke_macro_grid.jpg')"
            script.append("\n".join(lines) + "\n\n")
    (HERE / "_smoke_macro.py").write_text(
        "".join(script).replace("plt.show()", "plt.savefig('_smoke_macro_curve.png')"), encoding="utf-8")
    print("wrote", HERE / "m2e_finetune_macro.ipynb", "and _smoke_macro.py")


if __name__ == "__main__":
    build()
