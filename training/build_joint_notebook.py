"""Build the joint-training notebook (training/m3j_joint_train.ipynb) and its smoke-test script
(training/_smoke_joint.py) from ONE list of cells.

M2b..M2f were each fine-tuned from the previous model for 3 short CPU epochs, so every round
partly forgot the last one (M2f learned new_photos but lost NTUST). M3j trains ONCE on ALL the data
together, from the general COCO-pretrained D-FINE Nano, for 40 epochs on a free Kaggle GPU, with a
fresh balanced mix every epoch. Compared against M2c (shipped in the dataset as ref/best_model_tiled).

Kaggle: dataset avishek2410/wirebond-train-v2 (private) unzips to /kaggle/input/.../wbv2/ (marker
file WBV2_ROOT). Locally every path can be overridden with environment variables:

    tools/.venv/Scripts/python training/build_joint_notebook.py
    SMOKE=1 WB_ROOT=<staging>/wbv2 WORK_DIR=<scratch> tools/.venv/Scripts/python training/_smoke_joint.py
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
# Part M3j: Train once, on everything, on a GPU

**Why a new approach.** Models M2b → M2c → M2d → M2e → M2f were each made by taking the previous
model and training it a little more (3 short epochs on your PC's CPU) on a new mix. Each round,
the model partly **forgot** what it had learned before. M2f is the clearest example: it learned
to spot lost wires in the new sensor photos (5% → 64%) but got worse on the NTUST photos
(56% → 45% at equal false alarms). This is called **catastrophic forgetting**.

**The fix: joint training.**
- Start again from the general-purpose D-FINE Nano (trained on everyday COCO photos), not from M2c.
- Train on **all** our data together: synthetic, macro synthetic, NTUST, MEMS and new_photos.
- **40 epochs** instead of 3, possible because Kaggle lends us a free **GPU** (about 10–20× faster
  than your PC's CPU).
- Every epoch gets a **fresh, balanced mix**, so no single source can take over (the M2f problem):
  NTUST tiles shown twice, 700 new_photos tiles, MEMS photos and close-ups, 1,500 synthetic and
  600 macro synthetic images.

The tests in Step 7 are the same honest tests as before, so M3j can be compared directly with M2c
(our current best). Expected time on a Kaggle GPU: about **2–3 hours**.
""")

md("""
## Step 0: Settings

- `CHECKPOINT`: the general D-FINE Nano trained on COCO everyday photos (Apache-2.0 licence).
- `EPOCHS = 40`, `LR = 2e-4`: the same learning rate as our very first training (Part M2).
- Per-epoch mix: `NTUST_REPEAT = 2`, `NEWP_PER_EPOCH = 700`, `MEMS_REPEAT = 2` (+ all MEMS
  close-ups), `SYNTH_PER_EPOCH = 1500`, `MACRO_PER_EPOCH = 600`. A new random sample is drawn
  every epoch, so over 40 epochs the model still sees almost every image.
- Tiling (`TILE_SCALE`, `MIN_VISIBLE`) is the same as in M2c.
""")
code("""
import os
SMOKE = os.environ.get("SMOKE") == "1"
EPOCHS = 1 if SMOKE else 40
LR = 2e-4
BATCH = 2 if SMOKE else 16
IMG_SIZE = 640
TILE_SCALE = 0.5
TILE_STRIDE = 320          # tiles overlap by half
MIN_VISIBLE = 0.6
SEED = 42
NTUST_REPEAT = 2
NEWP_PER_EPOCH = 8 if SMOKE else 700
NEWP_EMPTY_KEEP = 0.25     # share of new_photos tiles WITHOUT a defect that are kept
NEWP_VAL = 10
MEMS_REPEAT = 2
CROPS_PER_DEFECT = 3       # close-up crops around each MEMS broken wire
NEG_CROPS = 2              # close-up crops with no broken wire, per MEMS photo
SYNTH_PER_EPOCH = 8 if SMOKE else 1500
MACRO_PER_EPOCH = 8 if SMOKE else 600
CHECKPOINT = os.environ.get("CHECKPOINT", "ustc-community/dfine-nano-coco")   # D-FINE Nano, Apache-2.0
OUT_MODEL = "best_model_joint"
print("smoke test" if SMOKE else "joint training", "| epochs:", EPOCHS, "| start from:", CHECKPOINT)
""")

md("""
## Step 1: Tools
""")
code("""
import subprocess, sys
if os.path.exists("/kaggle/working"):
    # on Kaggle: install the exact library versions this notebook was tested with (about 1 minute)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.2", "torchmetrics==1.9.0",
                    "onnx==1.23.2", "onnxruntime==1.31.1"], check=True)
import json, glob, random, time, math
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
print("Running on:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
""")

md("""
## Step 2: Load the data and split it (same split as Part M2b)

Same label translation (`sagged wire` → `wire_sweep`, `missing wire` → `missing_wire`,
`good wire` → no box) and the same split by original photo, so the 23 test photos are the
same ones as in Part M2b and the results are directly comparable.
""")
code("""
# On Kaggle the private dataset is unzipped under /kaggle/input; the file WBV2_ROOT marks its top folder.
WB_ROOT = os.environ.get("WB_ROOT") or str(Path(sorted(glob.glob("/kaggle/input/**/WBV2_ROOT", recursive=True))[0]).parent)
WORK = Path(os.environ.get("WORK_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "."))
DATA_ROOT = f"{WB_ROOT}/train_v1_jpg"
MACRO_ROOT = f"{WB_ROOT}/macro_v1"
REAL_ROOT = f"{WB_ROOT}/real"
REF_MODEL = f"{WB_ROOT}/ref/best_model_tiled"          # M2c, our current best, for the comparison
TILE_DIR = WORK / "tiles_ntust"                         # tiles are cut here (the input folder is read-only)
NEWP_TILE_DIR = WORK / "tiles_newphotos"
CROP_DIR = WORK / "crops_mems"
print("data:", WB_ROOT, "| work folder:", WORK)
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

# ---- new_photos (WHUrobotics, CC BY 4.0): wire gone from its bond pad -> missing_wire ----
NEWP_DIR = glob.glob(f"{REAL_ROOT}/roboflow/*new_photos*/")[0]
newp_by_split = defaultdict(list)
for f in sorted(glob.glob(f"{NEWP_DIR}*/_annotations.coco.json")):
    split = Path(f).parent.name                       # train / valid / test (Roboflow's split)
    coco = json.load(open(f))
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append({"bbox": a["bbox"], "category_id": CLASSES.index("missing_wire") + 1,
                                    "area": a["bbox"][2] * a["bbox"][3]})
    for im in coco["images"]:
        newp_by_split[split].append({"path": str(Path(f).parent / im["file_name"]), "anns": anns[im["id"]],
                                     "good": [], "source": im["file_name"].split(".rf.")[0]})
train_src = sorted({it["source"] for it in newp_by_split["train"]})
random.Random(SEED).shuffle(train_src)
val_src = set(train_src[:NEWP_VAL])
NEWP = {"train": [it for it in newp_by_split["train"] if it["source"] not in val_src],
        "val": [it for it in newp_by_split["train"] if it["source"] in val_src],
        "test": newp_by_split["valid"] + newp_by_split["test"]}
leak = {it["source"] for it in NEWP["test"]} & {it["source"] for it in NEWP["train"] + NEWP["val"]}
assert not leak, f"test photos also in training: {sorted(leak)[:3]}"
if SMOKE:
    NEWP = {k: v[:3] for k, v in NEWP.items()}
print("new_photos:", {k: f"{len({i['source'] for i in v})} photos / {len(v)} images, "
                         f"{sum(len(i['anns']) for i in v)} lost wires" for k, v in NEWP.items()})
""")

md("""
## Step 3: Cut the real photos into tiles (NTUST and new_photos)

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

def make_tiles(items, split, root=None):
    out = []
    folder = (root or TILE_DIR) / split
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

NEWP_TILES = {}
for k in ("train", "val"):
    tiles, skipped = make_tiles(NEWP[k], k, NEWP_TILE_DIR)
    with_def = [it for it in tiles if it["anns"]]
    empty = [it for it in tiles if not it["anns"]]
    empty = random.Random(SEED).sample(empty, round(len(empty) * NEWP_EMPTY_KEEP))
    NEWP_TILES[k] = with_def + empty
    print(f"new_photos {k:5s}: {len(NEWP_TILES[k])} tiles ({len(with_def)} with a lost wire, {len(empty)} defect-free kept), "
          f"{skipped} skipped because a defect was cut in half")
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
## Step 4: Look at some training pictures

Top left: an NTUST tile. Top right and bottom left: **new_photos tiles** (red = a bond pad
whose wire is gone). Bottom right: a MEMS close-up. Green = good wires (NTUST) or non-wire
flaws (MEMS). Notice that many tiles have **no defect at all**. That
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
newp_def = [t for t in NEWP_TILES["train"] if t["anns"]] or NEWP_TILES["train"]
for i, it in enumerate(random.Random(3).sample(TILES["train"], min(1, len(TILES["train"])))
                       + random.Random(3).sample(newp_def, min(2, len(newp_def)))
                       + random.Random(3).sample(CROPS["train"], min(1, len(CROPS["train"])))):
    grid.paste(show(it), ((i % 2) * 450, (i // 2) * 450))
grid
""")

md("""
## Step 5: Mix everything together, load the general model
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

SYN_TEST = random.Random(SEED).sample(load_synth(SYN_SPLIT["test"]), 8 if SMOKE else 300)
# macro (close-up) synthetic: hold 2 of the 24 designs back for the test
mprods = sorted(p.name for p in Path(MACRO_ROOT).glob("p*") if (p / "config.json").exists())
random.Random(SEED).shuffle(mprods)
MACRO_SPLIT = {"test": mprods[:2], "train": mprods[2:]}
MACRO_TEST = random.Random(SEED).sample(load_synth(MACRO_SPLIT["test"], MACRO_ROOT), 8 if SMOKE else 200)

SYN_POOL = load_synth(SYN_SPLIT["train"])
MACRO_POOL = load_synth(MACRO_SPLIT["train"], MACRO_ROOT)

def epoch_items(ep):
    # a fresh, balanced mix for every epoch (no single source can take over)
    rng = random.Random(SEED + ep)
    items = (TILES["train"] * NTUST_REPEAT
             + rng.sample(NEWP_TILES["train"], min(NEWP_PER_EPOCH, len(NEWP_TILES["train"])))
             + MEMS["train"] * MEMS_REPEAT + CROPS["train"]
             + rng.sample(SYN_POOL, min(SYNTH_PER_EPOCH, len(SYN_POOL)))
             + rng.sample(MACRO_POOL, min(MACRO_PER_EPOCH, len(MACRO_POOL))))
    rng.shuffle(items)
    return items

TRAIN = epoch_items(0)
print(f"training mix per epoch: {len(TILES['train'])} NTUST tiles x {NTUST_REPEAT} + "
      f"{min(NEWP_PER_EPOCH, len(NEWP_TILES['train']))} of {len(NEWP_TILES['train'])} new_photos tiles + "
      f"{len(MEMS['train'])} MEMS photos x {MEMS_REPEAT} + {len(CROPS['train'])} MEMS close-ups + "
      f"{min(SYNTH_PER_EPOCH, len(SYN_POOL))} of {len(SYN_POOL)} synthetic + "
      f"{min(MACRO_PER_EPOCH, len(MACRO_POOL))} of {len(MACRO_POOL)} macro synthetic = {len(TRAIN)}")

id2label = dict(enumerate(CLASSES)); label2id = {v: k for k, v in id2label.items()}
processor = AutoImageProcessor.from_pretrained(CHECKPOINT, size={"height": IMG_SIZE, "width": IMG_SIZE})
model = DFineForObjectDetection.from_pretrained(
    CHECKPOINT, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True).to(DEVICE)

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

WORKERS = 0 if (SMOKE or os.name == "nt") else min(4, os.cpu_count() or 2)

def make_loader(items, shuffle):
    return torch.utils.data.DataLoader(DS(items), batch_size=BATCH, shuffle=shuffle, collate_fn=collate,
                                       num_workers=WORKERS, pin_memory=(DEVICE == "cuda"))

loaders = {"train": make_loader(TRAIN, True),
           "val": make_loader(TILES["val"] + NEWP_TILES["val"] + MEMS["val"] + CROPS["val"], False)}
print({k: len(v) for k, v in loaders.items()}, "batches")
""")

md("""
## Step 6: Train (40 epochs, fresh balanced mix each epoch)
""")
code("""
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps = EPOCHS * len(loaders["train"])
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 50), pct_start=0.1)
scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
history = {"train": [], "val": []}

def run_epoch(split, train):
    model.train(train)
    total, n = 0.0, 0
    for i, (px, labels) in enumerate(loaders[split]):
        px = px.to(DEVICE, non_blocking=True); labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]
        with torch.set_grad_enabled(train), torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            loss = model(pixel_values=px, labels=labels).loss
        if train:
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt); scaler.update()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
        total += loss.item(); n += 1
        if SMOKE and i >= 1: break
        if train and i % 100 == 0:
            print(f"   batch {i}/{len(loaders[split])}  loss {loss.item():.3f}", flush=True)
    return total / max(n, 1)

best = math.inf
t0 = time.time()
for ep in range(EPOCHS):
    t = time.time()
    if ep > 0:
        loaders["train"] = make_loader(epoch_items(ep), True)      # fresh balanced mix
    tr = run_epoch("train", True); va = run_epoch("val", False)
    history["train"].append(tr); history["val"].append(va)
    if va < best:
        best = va; model.save_pretrained(OUT_MODEL); processor.save_pretrained(OUT_MODEL)
        print("   new best, saved", flush=True)
    print(f"epoch {ep + 1:2d}/{EPOCHS}  train loss {tr:.3f}  val loss (NTUST + new_photos + MEMS) {va:.3f}  "
          f"({time.time() - t:.0f}s, total {(time.time() - t0) / 60:.0f} min)", flush=True)
""")

code("""
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.5))
plt.plot(range(1, len(history["train"]) + 1), history["train"], "o-", label="train loss (mix)")
plt.plot(range(1, len(history["val"]) + 1), history["val"], "o-", label="validation loss (real)")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=.3); plt.show()
""")

md("""
## Step 7: The honest test: M2c vs M3j

None of these test pictures were used for training:
1. **new_photos** (30 held-back photos, tile by tile): the new skill. Can it spot a bond pad
   whose wire is gone? M2c has never seen these photos.
2. **NTUST** (23 photos, tile by tile): did it keep what M2c already did well?
3. **MEMS whole photos** and **MEMS close-ups**: still expected to be hard (different defect).
4. **Synthetic** (300) and **macro synthetic** (200): did it forget the eight synthetic classes?

"found (in region)" counts a defect as found when a correct answer lies mostly inside the
labelled box (useful when a label box covers a whole group of wires).

What we want: M3j better than M2c on new_photos **and** at least as good on NTUST. Because a
model can look better or worse just by being more or less trigger-happy, the cell after the table
also compares NTUST **at equal false alarms** (30%), like `tools/threshold_sweep.py`.
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

tests = {"new_photos  (tiled)": (NEWP["test"], predict_tiled),
         "NTUST       (tiled)": (REAL["test"], predict_tiled),
         "MEMS        (whole)": (MEMS["test"], predict_whole),
         "MEMS   (close-ups) ": (CROPS["test"], predict_whole),
         "synthetic   (whole)": (SYN_TEST, predict_whole),
         "macro synthetic    ": (MACRO_TEST, predict_whole)}
RESULTS = {}
for name, path in (("M2c", REF_MODEL), ("M3j", OUT_MODEL)):
    m = DFineForObjectDetection.from_pretrained(path).to(DEVICE).eval()
    for tname, (items, fn) in tests.items():
        RESULTS[(name, tname.strip())] = r = evaluate(m, items, fn)
        print(f"{name}  {tname}  {r}", flush=True)
    print()
""")

md("""
### NTUST at equal false alarms

For each model: run every NTUST test photo once, then try confidence thresholds from 0.20 to 0.85
and report how many defects it finds at the threshold where false alarms are at most 30%.
That threshold is also the one to use in the inspector app.
""")
code("""
def sweep(m, items, target=0.30):
    preds = [predict_tiled(m, it["path"], 0.05) for it in items]
    rows = []
    for thr in [round(0.20 + 0.05 * i, 2) for i in range(14)]:
        found = total = flag = gtot = 0
        for (B, S, L), it in zip(preds, items):
            keep = S >= thr
            kb, kl = B[keep].tolist(), L[keep].tolist()
            for a in it["anns"]:
                b = [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                total += 1; found += any(iou(p, b) >= 0.3 and pl == a["category_id"] - 1 for p, pl in zip(kb, kl))
            for g in it["good"]:
                gb = [g[0], g[1], g[0] + g[2], g[1] + g[3]]
                gtot += 1; flag += any(iou(p, gb) >= 0.3 for p in kb)
        rows.append((thr, found / max(total, 1), flag / max(gtot, 1)))
    ok = [r for r in rows if r[2] <= target]
    return rows, (ok[0] if ok else rows[-1])

for name, path in (("M2c", REF_MODEL), ("M3j", OUT_MODEL)):
    m = DFineForObjectDetection.from_pretrained(path).to(DEVICE).eval()
    rows, (thr, f, fa) = sweep(m, REAL["test"])
    print(f"{name}: at <= 30% false alarms -> threshold {thr:.2f} finds {100 * f:.0f}% (false alarms {100 * fa:.0f}%)")
""")

md("""
## Step 8: See it on held-back new_photos test photos

Red = the real lost-wire labels. **Yellow** = M3j's answers (tile by tile, stitched back).
""")
code("""
new = DFineForObjectDetection.from_pretrained(OUT_MODEL).to(DEVICE).eval()
grid = Image.new("RGB", (900, 900))
for i, it in enumerate(random.Random(2).sample(NEWP["test"], min(4, len(NEWP["test"])))):
    b, sc, l = predict_tiled(new, it["path"], THRESH)
    grid.paste(show(it, list(zip(b.tolist(), l.tolist(), sc.tolist()))), ((i % 2) * 450, (i // 2) * 450))
grid
""")

md("""
## Step 9: Export M3j for the Pi

Saves `wirebond_dfine_nano_joint.onnx` (+ `.json`) and checks that the ONNX version gives the
**same detections** as PyTorch on real test tiles (a random-noise input can give a misleading
number, as we saw with M2f). Download both files from the notebook's **Output** tab.
""")
code("""
import onnxruntime as ort
model_cpu = DFineForObjectDetection.from_pretrained(OUT_MODEL).eval()
onnx_path = str(WORK / "wirebond_dfine_nano_joint.onnx")
torch.onnx.export(model_cpu, (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),), onnx_path, input_names=["pixel_values"],
                  output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
json.dump({"classes": CLASSES, "img_size": IMG_SIZE,
           "note": "M3j: joint training from COCO D-FINE Nano on synthetic (whole + macro) + NTUST wire bonding + "
                   "WHU mems_60 + WHU new_photos (all real sets CC BY 4.0)",
           "tiling": {"scale": TILE_SCALE, "tile": IMG_SIZE, "stride": TILE_STRIDE, "nms_iou": 0.5}},
          open(str(WORK / "wirebond_dfine_nano_joint.json"), "w"), indent=1)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
same = 0
check = (NEWP_TILES["val"] + TILES["val"])[:12]
for it in check:
    px = processor(images=Image.open(it["path"]).convert("RGB"), return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        ref = model_cpu(pixel_values=px)
    lg, bx = sess.run(["logits", "pred_boxes"], {"pixel_values": px.numpy()})
    a = (torch.sigmoid(ref.logits) >= THRESH).nonzero().tolist()
    b = (torch.sigmoid(torch.from_numpy(lg)) >= THRESH).nonzero().tolist()
    same += a == b and np.abs(bx - ref.pred_boxes.numpy()).max() < 0.01
print(f"saved {onnx_path} | same detections as PyTorch on {same}/{len(check)} real tiles")
""")

md("""
## Done!

Say "status" (or send a screenshot of **Step 7**). If M3j wins, we drop the new `.onnx` into
the inspector app and rerun the reality check.
""")


# --------------------------------------------------------------------------------------------
def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nbf.write(nb, HERE / "m3j_joint_train.ipynb")
    script = ["# AUTO-GENERATED from build_joint_notebook.py - smoke-test copy\n"]
    for k, s in CELLS:
        if k == "code":
            lines = s.splitlines()
            if lines and lines[-1] == "grid":
                lines[-1] = "grid.save('_smoke_joint_grid.jpg')"
            script.append("\n".join(lines) + "\n\n")
    (HERE / "_smoke_joint.py").write_text(
        "".join(script).replace("plt.show()", "plt.savefig('_smoke_joint_curve.png')"), encoding="utf-8")
    print("wrote", HERE / "m3j_joint_train.ipynb", "and _smoke_joint.py")


if __name__ == "__main__":
    build()
