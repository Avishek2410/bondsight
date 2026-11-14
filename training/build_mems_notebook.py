"""Build the M2d notebook (training/m2d_finetune_mems.ipynb): M2c + MEMS sensor photos, and its
smoke-test script (training/_smoke_mems.py) from ONE list of cells.

M2c works on NTUST-style photos but fails on package types it never saw (MEMS sensors: broken
wires missed, the dark die window boxed as a defect). M2d continues from M2c and adds the
mems_60 photos (CC BY 4.0): broken wires as `broken_wire`, every other MEMS label (chip injury,
scratch, glue wrinkle) left WITHOUT a box so the model learns those are not wire defects.

    tools/.venv/Scripts/python training/build_mems_notebook.py
    SMOKE=1 DATA_ROOT=out/train_v1_jpg REAL_ROOT=real TILE_DIR=out/ntust_tiles_s05 \
        START_MODEL=training/runs/local_v1/best_model_tiled tools/.venv/Scripts/python training/_smoke_mems.py
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
# Part M2d: Teaching the model a new kind of chip (MEMS sensors)

The reality check showed M2c works on the kind of photo it learned from (NTUST), but on
**MEMS sensor chips** it had never seen, it missed every broken wire and drew boxes around
the dark window in the middle of the chip. A model only knows what it has been shown.

**The fix here:** continue from M2c and add the 61 MEMS photos (licence CC BY 4.0):
- broken wires (gold or aluminium) are labelled `broken_wire`;
- chip injuries, scratches and glue wrinkles get **no box**: they are real flaws, but not
  *wire* flaws, so the model must learn to stay quiet there. Photos like these are called
  **hard negatives**: they look suspicious but are not what we are looking for.

The MEMS photos are split by photo (train / validation / test), like NTUST. Only 12 test
photos is a small test, so treat its numbers as a rough signal, not a precise score.

Training takes about **1.5 hours** on your PC. The recipe below is the same as Part M2c,
with MEMS added.
""")

md("""
## (For reference) How M2c worked: fine-tuning on TILES of real photos

In Part M2b fine-tuning helped (real defects found 0% → 42%, false alarms 45% → 27%), but it
levelled off. One reason: every real photo (about 2000 × 1330 pixels) was **squashed** to
640 × 640. A thin wire became 1–2 pixels wide and the photo was stretched out of shape, so a
slightly bent wire looked almost like a straight one.

**The fix here: tiles.**
1. Shrink each real photo only to **half** size (keeping its shape).
2. Cut it into **overlapping 640 × 640 tiles**, like cutting a big map into pages.
3. Train on the tiles. At test time, run the model on every tile of a photo and **stitch the
   boxes back** onto the full photo.

The model sees wires about **1.6× larger** and undistorted. The cost: 6 tiles per photo means
about 6× more work per photo on the Pi (still about 1 second).

Training takes about **1.5 hours** on your PC.
""")

md("""
## Step 0: Settings

- `START_MODEL`: we continue from the Part M2c model (the current best). `MEMS_REPEAT = 4`: the 43 MEMS training photos are shown 4 times per epoch.
- `TILE_SCALE = 0.5`: photos are shrunk to half size before tiling. At this size 90% of
  real wires (up to about 1,230 pixels long originally) still fit inside one tile.
- `MIN_VISIBLE = 0.6`: a defect cut by a tile edge is kept only if at least 60% of it is
  inside the tile. Tiles where a defect is cut in half are not used for training: half a
  sagged wire is a confusing example (is it a defect or not?).
""")
code("""
import os
SMOKE = os.environ.get("SMOKE") == "1"
EPOCHS = 1 if SMOKE else 4
LR = 5e-5
BATCH = 2 if SMOKE else 16
SYNTH_N = 8 if SMOKE else 800
IMG_SIZE = 640
TILE_SCALE = 0.5
TILE_STRIDE = 320          # tiles overlap by half
MIN_VISIBLE = 0.6
SEED = 42
MEMS_REPEAT = 4            # only 43 MEMS training photos: show each one 4 times per epoch
START_MODEL = os.environ.get("START_MODEL", "best_model_tiled")   # from Part M2c (the current best)
OUT_MODEL = "best_model_mems"
print("smoke test" if SMOKE else "MEMS fine-tuning", "| epochs:", EPOCHS, "| start from:", START_MODEL)
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
## Step 4: Look at some tiles

Red = defects, green = good wires. Notice that many tiles have **no defect at all**. That
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
for i, it in enumerate(random.Random(3).sample(TILES["train"], min(4, len(TILES["train"])))):
    grid.paste(show(it), ((i % 2) * 450, (i // 2) * 450))
grid
""")

md("""
## Step 5: Mix in synthetic images, load the model
""")
code("""
prods = PRODUCTS[:]
random.Random(SEED).shuffle(prods)
n = max(1, len(prods) // 10)
SYN_SPLIT = {"test": prods[:n], "train": prods[2 * n:]}

def load_synth(products):
    items = []
    for p in products:
        for view in ("top", "angled"):
            coco = json.load(open(f"{DATA_ROOT}/{p}/annotations/{view}.json"))
            anns = defaultdict(list)
            for a in coco["annotations"]:
                anns[a["image_id"]].append(a)
            for im in coco["images"]:
                items.append({"path": f"{DATA_ROOT}/{p}/images/{view}/{im['file_name']}",
                              "anns": anns[im["id"]], "good": []})
    return items

SYN_TRAIN = random.Random(SEED).sample(load_synth(SYN_SPLIT["train"]), SYNTH_N)
SYN_TEST = random.Random(SEED).sample(load_synth(SYN_SPLIT["test"]), 8 if SMOKE else 300)
TRAIN = TILES["train"] + MEMS["train"] * MEMS_REPEAT + SYN_TRAIN
random.Random(SEED).shuffle(TRAIN)
print(f"training mix per epoch: {len(TILES['train'])} NTUST tiles + {len(MEMS['train'])} MEMS photos x "
      f"{MEMS_REPEAT} + {len(SYN_TRAIN)} synthetic = {len(TRAIN)}")

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
           "val": torch.utils.data.DataLoader(DS(TILES["val"] + MEMS["val"]), batch_size=BATCH, collate_fn=collate)}
print({k: len(v) for k, v in loaders.items()}, "batches")
""")

md("""
## Step 6: Fine-tune (NTUST tiles + MEMS photos + synthetic)
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
    print(f"epoch {ep + 1:2d}/{EPOCHS}  train loss {tr:.3f}  val loss (NTUST tiles + MEMS) {va:.3f}  ({time.time() - t:.0f}s)")
""")

code("""
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.5))
plt.plot(range(1, len(history["train"]) + 1), history["train"], "o-", label="train loss (mix)")
plt.plot(range(1, len(history["val"]) + 1), history["val"], "o-", label="validation loss (real)")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=.3); plt.show()
""")

md("""
## Step 7: The honest test: M2c vs M2d on three different test sets

None of these test photos were used for training:
1. **NTUST** (23 photos): did adding MEMS keep (or improve) what M2c already did well?
   Big photos, so the model runs **tile by tile**.
2. **MEMS** (12 photos): the new kind of chip. These photos are small enough to be looked at
   **whole** (the same rule the inspector app uses). "False alarms" here = boxes on chip
   injuries, scratches or glue wrinkles, which are not wire flaws.
3. **Synthetic** (300 images): did it forget the eight synthetic defect classes?

What we want: M2d better on MEMS, and not worse on the other two.
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

def evaluate(m, items, predict):
    # 'good' boxes = must NOT be flagged (good wires in NTUST, non-wire flaws in MEMS)
    metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5], class_metrics=True)
    metric.warn_on_many_detections = False
    found = total = good_flag = good_total = 0
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
        for g in it["good"]:
            gb = [g[0], g[1], g[0] + g[2], g[1] + g[3]]
            good_total += 1; good_flag += any(iou(p, gb) >= 0.3 for p in kb)
    res = metric.compute()
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "n/a"
    return {"mAP@0.5": round(res["map_50"].item(), 3), "defects found": pct(found, total),
            "false alarms": pct(good_flag, good_total)}

tests = {"NTUST     (tiled)": (REAL["test"], predict_tiled),
         "MEMS      (whole)": (MEMS["test"], predict_whole),
         "synthetic (whole)": (SYN_TEST, predict_whole)}
for name, path in (("M2c", START_MODEL), ("M2d", OUT_MODEL)):
    m = DFineForObjectDetection.from_pretrained(path).to(DEVICE).eval()
    for tname, (items, fn) in tests.items():
        print(f"{name}  {tname}  {evaluate(m, items, fn)}", flush=True)
    print()
""")

md("""
## Step 8: See it on MEMS test photos

Red = real broken wires, green = non-wire flaws (it must stay quiet there).
**Yellow** = M2d's answers.
""")
code("""
new = DFineForObjectDetection.from_pretrained(OUT_MODEL).to(DEVICE).eval()
grid = Image.new("RGB", (900, 900))
for i, it in enumerate(random.Random(2).sample(MEMS["test"], min(4, len(MEMS["test"])))):
    b, sc, l = predict_whole(new, it["path"], THRESH)
    grid.paste(show(it, list(zip(b.tolist(), l.tolist(), sc.tolist()))), ((i % 2) * 450, (i // 2) * 450))
grid
""")

md("""
## Step 9: Export M2d for the Pi

Saves `wirebond_dfine_nano_mems.onnx` (+ `.json`) next to this notebook and checks that the
ONNX version gives the same numbers as PyTorch. Whether it replaces M2c in the inspector app
is decided from the Step 7 table.
""")
code("""
import onnxruntime as ort
model_cpu = DFineForObjectDetection.from_pretrained(OUT_MODEL).eval()
dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(model_cpu, (dummy,), "wirebond_dfine_nano_mems.onnx", input_names=["pixel_values"],
                  output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
json.dump({"classes": CLASSES, "img_size": IMG_SIZE,
           "note": "M2d: synthetic + NTUST wire bonding tiles + WHU mems_60 (both CC BY 4.0)",
           "tiling": {"scale": TILE_SCALE, "tile": IMG_SIZE, "stride": TILE_STRIDE, "nms_iou": 0.5}},
          open("wirebond_dfine_nano_mems.json", "w"), indent=1)
px = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
with torch.no_grad():
    ref = model_cpu(pixel_values=px)
lg, bx = ort.InferenceSession("wirebond_dfine_nano_mems.onnx", providers=["CPUExecutionProvider"]).run(
    ["logits", "pred_boxes"], {"pixel_values": px.numpy()})
print("saved wirebond_dfine_nano_mems.onnx | max difference vs PyTorch:",
      float(max(np.abs(lg - ref.logits.numpy()).max(), np.abs(bx - ref.pred_boxes.numpy()).max())))
""")

md("""
## Done!

Say "status" (or send a screenshot of **Step 7**). If M2d wins, we drop the new `.onnx` into
the inspector app and rerun the reality check.
""")


# --------------------------------------------------------------------------------------------
def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nbf.write(nb, HERE / "m2d_finetune_mems.ipynb")
    script = ["# AUTO-GENERATED from build_mems_notebook.py - smoke-test copy\n"]
    for k, s in CELLS:
        if k == "code":
            lines = s.splitlines()
            if lines and lines[-1] == "grid":
                lines[-1] = "grid.save('_smoke_mems_grid.jpg')"
            script.append("\n".join(lines) + "\n\n")
    (HERE / "_smoke_mems.py").write_text(
        "".join(script).replace("plt.show()", "plt.savefig('_smoke_mems_curve.png')"), encoding="utf-8")
    print("wrote", HERE / "m2d_finetune_mems.ipynb", "and _smoke_mems.py")


if __name__ == "__main__":
    build()
