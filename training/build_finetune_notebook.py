"""Build the M2b fine-tuning notebook (training/m2b_finetune_real.ipynb) and its smoke-test
script (training/_smoke_finetune.py) from ONE list of cells.

Step 1 of the sim-to-real fix: start from the synthetic-trained model and fine-tune on a mix of
REAL NTUST wire-bond photos (CC BY 4.0) + some synthetic images, then test on held-out real photos.

    tools/.venv/Scripts/python training/build_finetune_notebook.py
    SMOKE=1 DATA_ROOT=out/train_v1_jpg REAL_ROOT=real START_MODEL=training/runs/local_v1/best_model \
        tools/.venv/Scripts/python training/_smoke_finetune.py
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
# Part M2b: Teaching the model about REAL wire bonds (fine-tuning)

In Part M2 the model scored **0.90 mAP on synthetic test images**, but the reality check showed
it **fails on real photos**: it flagged 65% of *good* real wires as defects and never named a
real defect correctly. This is the **sim-to-real gap**.

The standard fix, and what you do here, is **fine-tuning**:
1. Start from the model you already trained (it knows what wires and defects are).
2. Train it a little more on a **mix** of real labelled photos and synthetic images.
3. Test it on **real photos it has never seen**.

**Real data:** the NTUST *wire bonding* dataset (National Taiwan University of Science and
Technology, licence **CC BY 4.0**: we may use and publish it with credit).

This runs on your PC (CPU) and takes about **2 hours**. As before: read each text box, then
press **Shift + Enter** on the code box below it.
""")

md("""
## Step 0: Settings

- `START_MODEL`: the model from Part M2 (we continue from it, we do **not** start again).
- `LR`: smaller than before (5e-5 vs 2e-4). We only want to *adjust* the model, not
  overwrite what it learned. A big learning rate here would make it forget.
- `REAL_REPEAT`: real photos are few, so each one is shown 3 times per epoch.
- `SYNTH_N`: how many synthetic images to mix in, so the model does not forget them
  (forgetting old knowledge when learning new things is called **catastrophic forgetting**).
""")
code("""
import os
SMOKE = os.environ.get("SMOKE") == "1"
EPOCHS = 1 if SMOKE else 8
LR = 5e-5
BATCH = 2 if SMOKE else 16
REAL_REPEAT = 3
SYNTH_N = 8 if SMOKE else 1200
IMG_SIZE = 640
SEED = 42
START_MODEL = os.environ.get("START_MODEL", "best_model")     # from Part M2
OUT_MODEL = "best_model_real"
print("smoke test" if SMOKE else "fine-tuning", "| epochs:", EPOCHS, "| start from:", START_MODEL)
""")

md("""
## Step 1: Tools
""")
code("""
import json, glob, random, time, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, DFineForObjectDetection
from torchmetrics.detection import MeanAveragePrecision

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))   # leave 2 cores for Windows
print("Running on:", DEVICE)
""")

md("""
## Step 2: Find both datasets
""")
code("""
DATA_ROOT = os.environ.get("DATA_ROOT", "C:/Users/AVI-TECH/projects/wirebond-ai/out/train_v1_jpg")
REAL_ROOT = os.environ.get("REAL_ROOT", "C:/Users/AVI-TECH/projects/wirebond-ai/real")
NTUST = glob.glob(f"{REAL_ROOT}/roboflow/national-taiwan*/")[0]
PRODUCTS = sorted(p.name for p in Path(DATA_ROOT).glob("p*") if (p / "config.json").exists())
CLASSES = json.load(open(f"{DATA_ROOT}/{PRODUCTS[0]}/annotations/top.json"))["categories"]
CLASSES = [c["name"] for c in sorted(CLASSES, key=lambda c: c["id"])]
print("synthetic:", len(PRODUCTS), "designs | real:", NTUST)
""")

md("""
## Step 3: Translate the real labels into our defect names

The NTUST photographers used their own label names. We map them onto ours:

| NTUST label | Our class | Why |
|---|---|---|
| `sagged wire` | `wire_sweep` | seen from above, these wires bend **sideways**; that is a sweep |
| `missing wire` | `missing_wire` | same thing |
| `good wire` | *(no box)* | a good wire is **not** a defect; leaving it unboxed teaches the model to stay quiet |

The last row matters most: it is how the model learns that a shiny wire on a glittery pad
is normal.
""")
code("""
LABEL_MAP = {"sagged wire": "wire_sweep", "missing wire": "missing_wire", "good wire": None}

def load_ntust():
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
            source = im["file_name"].split(".rf.")[0]       # e.g. "016_jpg"
            by_source[source].append({"path": str(Path(f).parent / im["file_name"]),
                                      "anns": ours, "good": good, "source": source})
    return by_source

REAL_BY_SOURCE = load_ntust()
print(len(REAL_BY_SOURCE), "different real photos,",
      sum(len(v) for v in REAL_BY_SOURCE.values()), "images incl. augmented copies")
""")

md("""
## Step 4: Split the real photos the *right* way

Roboflow made up to **3 altered copies** (flipped, rotated, cropped) of each original photo.
If one copy went to training and another to testing, the test would be too easy: the model
has basically seen that photo. That is **data leakage** again, the same lesson as splitting
by chip design in Part M2.

So we split by **original photo**: all copies of a photo go to the same side.
""")
code("""
sources = sorted(REAL_BY_SOURCE)
random.Random(SEED).shuffle(sources)
n_test, n_val = round(len(sources) * 0.2), round(len(sources) * 0.1)
REAL_SPLIT = {"test": sources[:n_test], "val": sources[n_test:n_test + n_val],
              "train": sources[n_test + n_val:]}
REAL = {k: [it for s in v for it in REAL_BY_SOURCE[s]] for k, v in REAL_SPLIT.items()}

# synthetic: same design split as Part M2, so the synthetic TEST designs stay unseen
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
if SMOKE:
    REAL = {k: v[:4] for k, v in REAL.items()}

TRAIN = REAL["train"] * REAL_REPEAT + SYN_TRAIN
random.Random(SEED).shuffle(TRAIN)
for k in ("train", "val", "test"):
    v = REAL[k]
    c = Counter(CLASSES[a["category_id"] - 1] for it in v for a in it["anns"])
    print(f"real {k:5s}: {len(REAL_SPLIT[k]):3d} photos, {len(v):3d} images, "
          f"defects {dict(c)}, good wires {sum(len(it['good']) for it in v)}")
print(f"training mix per epoch: {len(REAL['train'])} real x {REAL_REPEAT} + {len(SYN_TRAIN)} synthetic"
      f" = {len(TRAIN)} images")
""")

md("""
## Step 5: Look at the real training data

Red = defects (in our class names). Green = good wires (no box for the model; shown here only
so you can see them).
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
for i, it in enumerate(random.Random(1).sample(REAL["train"], min(4, len(REAL["train"])))):
    grid.paste(show(it), ((i % 2) * 450, (i // 2) * 450))
grid
""")

md("""
## Step 6: Load the model from Part M2 and the data loaders
""")
code("""
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
           "val": torch.utils.data.DataLoader(DS(REAL["val"]), batch_size=BATCH, collate_fn=collate)}
print({k: len(v) for k, v in loaders.items()}, "batches")
""")

md("""
## Step 7: Fine-tune

Same loop as Part M2. Validation now uses **real** photos only, because that is what we care
about. The best version is saved to `best_model_real`.
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
    print(f"epoch {ep + 1:2d}/{EPOCHS}  train loss {tr:.3f}  val loss (real) {va:.3f}  ({time.time() - t:.0f}s)")
""")

code("""
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.5))
plt.plot(range(1, len(history["train"]) + 1), history["train"], "o-", label="train loss (mix)")
plt.plot(range(1, len(history["val"]) + 1), history["val"], "o-", label="validation loss (real)")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=.3); plt.show()
""")

md("""
## Step 8: The honest test, on REAL photos never seen

We compare the **old** model (synthetic only) with the **new** one on the same held-out real
photos. Three numbers:

- **mAP@0.5** for the real defect types (1.0 = perfect).
- **Defects found**: share of real defects with a box of the *right* type on them.
- **False alarms on good wires**: share of good wires the model wrongly flags. Before
  fine-tuning this was about 65%. Lower is better.
""")
code("""
THRESH = 0.4

def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

def predict(m, path, thr):
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        out = m(**processor(images=img, return_tensors="pt").to(DEVICE))
    return processor.post_process_object_detection(out, threshold=thr, target_sizes=[img.size[::-1]])[0]

def evaluate(m, items):
    metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5], class_metrics=True)
    metric.warn_on_many_detections = False
    found = total = good_flag = good_total = 0
    for it in items:
        r = predict(m, it["path"], 0.05)
        gt_b = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                             for a in it["anns"]], dtype=torch.float32).reshape(-1, 4)
        gt_l = torch.tensor([a["category_id"] - 1 for a in it["anns"]], dtype=torch.long)
        metric.update([{k: r[k].cpu() for k in ("boxes", "scores", "labels")}], [{"boxes": gt_b, "labels": gt_l}])
        keep = r["scores"] >= THRESH
        boxes, labels = r["boxes"][keep].tolist(), r["labels"][keep].tolist()
        for b, l in zip(gt_b.tolist(), gt_l.tolist()):
            total += 1; found += any(iou(p, b) >= 0.3 and pl == l for p, pl in zip(boxes, labels))
        for g in it["good"]:
            gb = [g[0], g[1], g[0] + g[2], g[1] + g[3]]
            good_total += 1; good_flag += any(iou(p, gb) >= 0.3 for p in boxes)
    res = metric.compute()
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "n/a"
    return {"mAP@0.5": round(res["map_50"].item(), 3), "defects found": pct(found, total),
            "false alarms on good wires": pct(good_flag, good_total)}

old = DFineForObjectDetection.from_pretrained(START_MODEL).to(DEVICE).eval()
new = DFineForObjectDetection.from_pretrained(OUT_MODEL).to(DEVICE).eval()
print(f"Real test set: {len(REAL['test'])} images from {len(REAL_SPLIT['test'])} photos never seen\\n")
print("BEFORE (synthetic only):", evaluate(old, REAL["test"]))
print("AFTER  (fine-tuned)    :", evaluate(new, REAL["test"]))
""")

md("""
## Step 9: Did it forget the synthetic world?

We re-test on 300 synthetic images from the 3 test designs. A small drop is normal; a big drop
means it forgot (catastrophic forgetting).
""")
code("""
print("BEFORE:", evaluate(old, SYN_TEST))
print("AFTER :", evaluate(new, SYN_TEST))
""")

md("""
## Step 10: See it on real photos

Red = real defects, green = good wires (the true answers). **Yellow** = what the new model says.
""")
code("""
grid = Image.new("RGB", (900, 900))
for i, it in enumerate(random.Random(2).sample(REAL["test"], min(4, len(REAL["test"])))):
    r = predict(new, it["path"], THRESH)
    preds = list(zip(r["boxes"].tolist(), r["labels"].tolist(), r["scores"].tolist()))
    grid.paste(show(it, preds), ((i % 2) * 450, (i // 2) * 450))
grid
""")

md("""
## Step 11: Export the new model for the Pi
""")
code("""
import onnxruntime as ort
model_cpu = DFineForObjectDetection.from_pretrained(OUT_MODEL).eval()
dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(model_cpu, (dummy,), "wirebond_dfine_nano_real.onnx", input_names=["pixel_values"],
                  output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
json.dump({"classes": CLASSES, "img_size": IMG_SIZE, "fine_tuned_on": "NTUST wire bonding (CC BY 4.0)"},
          open("wirebond_dfine_nano_real.json", "w"), indent=1)
sess = ort.InferenceSession("wirebond_dfine_nano_real.onnx", providers=["CPUExecutionProvider"])
px = processor(images=Image.open(REAL["test"][0]["path"]).convert("RGB"), return_tensors="pt")["pixel_values"]
with torch.no_grad():
    ref = model_cpu(pixel_values=px)
lg, bx = sess.run(["logits", "pred_boxes"], {"pixel_values": px.numpy()})
print("saved wirebond_dfine_nano_real.onnx | max difference vs PyTorch:",
      float(max(np.abs(lg - ref.logits.numpy()).max(), np.abs(bx - ref.pred_boxes.numpy()).max())))
""")

md("""
## Done!

Write down the **BEFORE vs AFTER** numbers from Step 8 and the forgetting check from Step 9.
Then send a screenshot and we will run the full reality check again (all 8 real datasets) with
the new model.
""")


# --------------------------------------------------------------------------------------------
def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nbf.write(nb, HERE / "m2b_finetune_real.ipynb")
    script = ["# AUTO-GENERATED from build_finetune_notebook.py - smoke-test copy\n"]
    for k, s in CELLS:
        if k == "code":
            lines = s.splitlines()
            if lines and lines[-1] == "grid":
                lines[-1] = "grid.save('_smoke_ft_grid.jpg')"
            script.append("\n".join(lines) + "\n\n")
    (HERE / "_smoke_finetune.py").write_text(
        "".join(script).replace("plt.show()", "plt.savefig('_smoke_ft_curve.png')"), encoding="utf-8")
    print("wrote", HERE / "m2b_finetune_real.ipynb", "and _smoke_finetune.py")


if __name__ == "__main__":
    build()
