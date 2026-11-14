"""Build the M2 training notebook (training/m2_train_dfine.ipynb) and a plain script copy
(training/_smoke_run.py) from ONE list of cells, so the code Avi runs on Kaggle is exactly the
code smoke-tested on the PC.

    tools/.venv/Scripts/python training/build_notebook.py
    SMOKE=1 DATA_ROOT=out/train_v1_jpg tools/.venv/Scripts/python training/_smoke_run.py
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
# Part M2: Training a wire-bond defect detector

Welcome! In this notebook **you** train an AI model that finds wire-bond defects in images.
You don't need to write code. Read the text above each grey code box, then press
**Shift + Enter** to run it. Each step explains *what* happens and *why*.

**What we are building:** an *object detector*. You give it an image; it draws a box around
every defect and says which defect it is (for example `wire_sweep`) and how sure it is.

**The data:** 9,000 synthetic images we generated ourselves: 30 different chip designs,
each seen from the top and from a 45 degree angle, with 8 defect types labelled.

**Before you start:**
- **On Kaggle:** right side panel → *Accelerator* → **GPU T4 x2**, and add the dataset
  *Wire-Bond Synthetic Defects v1* as Input.
- **On Google Colab:** menu *Runtime → Change runtime type* → **T4 GPU**. The dataset is read
  from your Google Drive (`My Drive/8 PROJECTS/wirebond-ai/wirebond-synth-v1.zip`); Colab will
  ask you to allow Drive access the first time.

This notebook detects where it runs and does the right thing automatically.
""")

md("""
## Step 0: Settings you can change

These are the "knobs" of the experiment. Leave them as they are the first time.
Later, change **one** knob at a time and compare results. That is how real
AI engineers learn what matters.

- `EPOCHS`: how many times the model sees the whole training set.
- `LR` (learning rate): how big each learning step is. Too big = unstable, too small = slow.
- `BATCH`: how many images the model looks at together before each step.
""")
code("""
import os
SMOKE = os.environ.get("SMOKE") == "1"     # tiny quick run used only for testing this notebook
EPOCHS = 1 if SMOKE else 12
LR = 2e-4
BATCH = 2 if SMOKE else 16
IMG_SIZE = 640
CHECKPOINT = "ustc-community/dfine-nano-coco"   # D-FINE Nano, Apache-2.0 licence
SEED = 42
print("smoke test" if SMOKE else "full training", "| epochs:", EPOCHS, "| batch:", BATCH)
""")

md("""
## Step 1: Tools

We load the libraries. `torch` is the deep-learning engine, `transformers` gives us the
ready-made D-FINE detector, `torchmetrics` scores it.
""")
code("""
import importlib.util, subprocess, sys
missing = [p for p, m in (("torchmetrics", "torchmetrics"), ("pycocotools", "pycocotools"), ("scipy", "scipy"))
           if importlib.util.find_spec(m) is None]
if missing:
    print("installing", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

import json, glob, random, time, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, DFineForObjectDetection
from torchmetrics.detection import MeanAveragePrecision

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
""")

md("""
## Step 2: Find the data

On Kaggle the dataset is mounted read-only under `/kaggle/input`. On Colab we connect your
Google Drive and unzip the dataset onto Colab's fast local disk (reading thousands of small
files straight from Drive would be very slow). Then we look for the folders `p01` ... `p30`:
one folder per chip design (a "product").
""")
code("""
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not os.environ.get("DATA_ROOT"):
    from google.colab import drive
    drive.mount("/content/drive")
    if not Path("/content/data/train_v1_jpg").exists():
        zip_path = "/content/drive/MyDrive/8 PROJECTS/wirebond-ai/wirebond-synth-v1.zip"
        print("unzipping dataset to local disk (about 1 minute)...")
        subprocess.run(["unzip", "-q", zip_path, "-d", "/content/data"], check=True)
    os.environ["DATA_ROOT"] = "/content/data/train_v1_jpg"

DATA_ROOT = os.environ.get("DATA_ROOT") or next(
    str(Path(p).parent) for p in sorted(glob.glob("/kaggle/input/**/p01/config.json", recursive=True)))
PRODUCTS = sorted(p.name for p in Path(DATA_ROOT).glob("p*") if (p / "config.json").exists())
CLASSES = json.load(open(f"{DATA_ROOT}/{PRODUCTS[0]}/annotations/top.json"))["categories"]
CLASSES = [c["name"] for c in sorted(CLASSES, key=lambda c: c["id"])]
print(len(PRODUCTS), "products in", DATA_ROOT)
print("Defect classes:", CLASSES)
""")

md("""
## Step 3: Split the data the *right* way

We split by **chip design**, not by image. If images of the same design were in both
training and test, the model could "remember" that design and look better than it really
is. That mistake is called **data leakage**, and it is one of the most common errors in AI.

- **Train** (24 designs): the model learns from these.
- **Validation** (3 designs): we check progress during training.
- **Test** (3 designs): looked at **once**, at the very end, for an honest final score.
""")
code("""
prods = PRODUCTS[:]
random.Random(SEED).shuffle(prods)
n_val = n_test = max(1, len(prods) // 10)
SPLIT = {"test": prods[:n_test], "val": prods[n_test:n_test + n_val], "train": prods[n_test + n_val:]}

def load_items(products):
    items = []
    for p in products:
        for view in ("top", "angled"):
            coco = json.load(open(f"{DATA_ROOT}/{p}/annotations/{view}.json"))
            anns = {}
            for a in coco["annotations"]:
                anns.setdefault(a["image_id"], []).append(a)
            for im in coco["images"]:
                items.append({"path": f"{DATA_ROOT}/{p}/images/{view}/{im['file_name']}",
                              "view": view, "product": p, "anns": anns.get(im["id"], [])})
    return items

DATA = {k: load_items(v) for k, v in SPLIT.items()}
if SMOKE:
    DATA = {k: v[:8] for k, v in DATA.items()}
for k, v in DATA.items():
    print(f"{k:5s}: {len(SPLIT[k]):2d} designs, {len(v):5d} images, "
          f"{sum(len(i['anns']) for i in v):6d} defect boxes")
""")

md("""
## Step 4: Look at the data

Always look at your data before training. Below: random training images with their
**ground-truth** boxes (the correct answers we generated).
""")
code("""
def draw(item, boxes=None, labels=None, scores=None, color="red"):
    img = Image.open(item["path"]).convert("RGB")
    d = ImageDraw.Draw(img)
    if boxes is None:
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                 for a in item["anns"]]
        labels = [a["category_id"] - 1 for a in item["anns"]]
    for i, b in enumerate(boxes):
        d.rectangle(b, outline=color, width=2)
        txt = CLASSES[labels[i]] + (f" {scores[i]:.2f}" if scores is not None else "")
        d.text((b[0] + 2, b[1] + 2), txt, fill=color)
    return img

samples = random.sample(DATA["train"], min(4, len(DATA["train"])))
grid = Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE * 2))
for i, s in enumerate(samples):
    grid.paste(draw(s), ((i % 2) * IMG_SIZE, (i // 2) * IMG_SIZE))
grid.resize((900, 900))
""")

md("""
### How many of each defect?

Notice that `wire_sweep` is far more common than the others (mould-flow events bend many
wires at once). This is called **class imbalance**. Keep an eye on it in the results.
""")
code("""
from collections import Counter
cnt = Counter(CLASSES[a["category_id"] - 1] for it in DATA["train"] for a in it["anns"])
for c in CLASSES:
    print(f"{c:16s} {cnt[c]:6d}  " + "#" * (cnt[c] * 60 // max(cnt.values() or [1])))
""")

md("""
## Step 5: Load a pre-trained model (transfer learning)

We do **not** start from zero. D-FINE Nano was already trained on 118,000 everyday photos
(the COCO dataset), so it already knows edges, shapes and objects. We replace its last
layer so it predicts **our** 8 defect classes instead of cats and cars. This is
**transfer learning**: it needs far less data and time.
""")
code("""
id2label = dict(enumerate(CLASSES)); label2id = {v: k for k, v in id2label.items()}
processor = AutoImageProcessor.from_pretrained(CHECKPOINT, size={"height": IMG_SIZE, "width": IMG_SIZE})
model = DFineForObjectDetection.from_pretrained(
    CHECKPOINT, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} million")
""")

md("""
## Step 6: Feed the data to the model

A *dataset* returns one image + its boxes; a *data loader* groups them into batches and
loads them in the background so the GPU never waits.
""")
code("""
class WireBondDS(torch.utils.data.Dataset):
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

# Windows + Jupyter cannot start background loader processes for classes defined in a notebook
workers = 0 if (SMOKE or os.name == "nt") else min(4, os.cpu_count() or 2)
if DEVICE == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))   # leave 2 cores for Windows
loaders = {k: torch.utils.data.DataLoader(WireBondDS(v), batch_size=BATCH, shuffle=(k == "train"),
                                          num_workers=workers, collate_fn=collate)
           for k, v in DATA.items()}
print({k: len(v) for k, v in loaders.items()}, "batches per split")
""")

md("""
## Step 7: Train!

Each **epoch** the model sees every training image once. For each batch it:
1. makes predictions,
2. measures how wrong they are: the **loss** (lower = better),
3. nudges its millions of numbers a tiny bit to be less wrong (**gradient descent**).

Watch the loss go **down**. If training loss keeps falling but validation loss rises,
the model is **overfitting**: memorising instead of learning.
""")
code("""
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps = EPOCHS * len(loaders["train"])
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 1), pct_start=0.1)
scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
history = {"train": [], "val": []}

def run_epoch(split, train):
    model.train(train)
    total, n = 0.0, 0
    for i, (px, labels) in enumerate(loaders[split]):
        px = px.to(DEVICE); labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]
        with torch.set_grad_enabled(train), torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            loss = model(pixel_values=px, labels=labels).loss
        if train:
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt); scaler.update(); sched.step()
        total += loss.item(); n += 1
        if SMOKE and i >= 1: break
        if train and i % 50 == 0:
            print(f"   batch {i}/{len(loaders[split])}  loss {loss.item():.3f}", flush=True)
    return total / max(n, 1)

best = math.inf
for ep in range(EPOCHS):
    t = time.time()
    tr = run_epoch("train", True); va = run_epoch("val", False)
    history["train"].append(tr); history["val"].append(va)
    if va < best:
        best = va; model.save_pretrained("best_model"); processor.save_pretrained("best_model")
    print(f"epoch {ep + 1:2d}/{EPOCHS}  train loss {tr:.3f}  val loss {va:.3f}  ({time.time() - t:.0f}s)")
""")

md("""
### The learning curve

Both lines should fall. The gap between them tells you about overfitting.
""")
code("""
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.5))
plt.plot(range(1, len(history["train"]) + 1), history["train"], "o-", label="train loss")
plt.plot(range(1, len(history["val"]) + 1), history["val"], "o-", label="validation loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=.3); plt.show()
""")

md("""
## Step 8: Score the model honestly (on the TEST designs)

Now, and only now, we use the 3 test designs the model has never seen.

- **Precision**: of the boxes the model drew, how many were real defects? (few false alarms)
- **Recall**: of the real defects, how many did the model find? (few misses)
- **mAP@0.5**: one number combining both, per class; 1.0 is perfect.

In a factory, **recall** usually matters most: a missed defect ships to a customer;
a false alarm only costs an operator a second look.
""")
code("""
model = DFineForObjectDetection.from_pretrained("best_model").to(DEVICE).eval()
metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5], class_metrics=True)
for it in DATA["test"]:
    img = Image.open(it["path"]).convert("RGB")
    with torch.no_grad():
        out = model(**processor(images=img, return_tensors="pt").to(DEVICE))
    res = processor.post_process_object_detection(out, threshold=0.05, target_sizes=[img.size[::-1]])[0]
    gt_b = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                         for a in it["anns"]], dtype=torch.float32).reshape(-1, 4)
    gt_l = torch.tensor([a["category_id"] - 1 for a in it["anns"]], dtype=torch.long)
    metric.update([{k: res[k].cpu() for k in ("boxes", "scores", "labels")}],
                  [{"boxes": gt_b, "labels": gt_l}])
r = metric.compute()
print(f"Overall mAP@0.5: {r['map_50'].item():.3f}   recall (top-100): {r['mar_100'].item():.3f}\\n")
cls_ids = r["classes"].tolist() if r["classes"].ndim else [int(r["classes"])]
per = r["map_per_class"].tolist() if r["map_per_class"].ndim else [float(r["map_per_class"])]
for c, m in zip(cls_ids, per):
    print(f"  {CLASSES[c]:16s} mAP@0.5 = " + (f"{m:.3f}" if m >= 0 else "n/a (no examples in test set)"))
""")

md("""
## Step 9: See what the model sees

Green = the model's predictions (with confidence). Compare with the real defects.
""")
code("""
THRESH = 0.4
show = random.sample(DATA["test"], min(4, len(DATA["test"])))
grid = Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE * 2))
for i, it in enumerate(show):
    img = Image.open(it["path"]).convert("RGB")
    with torch.no_grad():
        out = model(**processor(images=img, return_tensors="pt").to(DEVICE))
    res = processor.post_process_object_detection(out, threshold=THRESH, target_sizes=[img.size[::-1]])[0]
    pic = draw(it, res["boxes"].tolist(), res["labels"].tolist(), res["scores"].tolist(), color="lime")
    grid.paste(pic, ((i % 2) * IMG_SIZE, (i // 2) * IMG_SIZE))
grid.resize((900, 900))
""")

md("""
## Step 10: Export for the Raspberry Pi

The Pi will not run PyTorch training code; it runs a compact **ONNX** file with ONNX
Runtime on its CPU. We export the trained model and save it to the notebook's *Output*,
from where you can download it.
""")
code("""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)
model_cpu = DFineForObjectDetection.from_pretrained("best_model").eval()
dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(model_cpu, (dummy,), "wirebond_dfine_nano.onnx", input_names=["pixel_values"],
                  output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
json.dump({"classes": CLASSES, "img_size": IMG_SIZE, "checkpoint": CHECKPOINT},
          open("wirebond_dfine_nano.json", "w"), indent=1)
print("saved wirebond_dfine_nano.onnx", round(os.path.getsize("wirebond_dfine_nano.onnx") / 1e6, 1), "MB")
""")

md("""
### Check the exported file gives the same answers

A good engineer never trusts an export blindly. We run the same image through PyTorch and
through ONNX Runtime (what the Pi will use) and compare. The difference should be tiny.
""")
code("""
import onnxruntime as ort
sess = ort.InferenceSession("wirebond_dfine_nano.onnx", providers=["CPUExecutionProvider"])
px = processor(images=Image.open(DATA["test"][0]["path"]).convert("RGB"), return_tensors="pt")["pixel_values"]
with torch.no_grad():
    ref = model_cpu(pixel_values=px)
onnx_logits, onnx_boxes = sess.run(["logits", "pred_boxes"], {"pixel_values": px.numpy()})
print("max difference, class scores:", float(np.abs(onnx_logits - ref.logits.numpy()).max()))
print("max difference, boxes:       ", float(np.abs(onnx_boxes - ref.pred_boxes.numpy()).max()))
t = time.time(); [sess.run(["logits", "pred_boxes"], {"pixel_values": px.numpy()}) for _ in range(5)]
print(f"ONNX speed here: {(time.time() - t) / 5 * 1000:.0f} ms per image (the Pi 5 will be slower)")
""")

md("""
## Done!

You trained, evaluated and exported a defect detector. Note down:
1. the final **test mAP** and the **per-class** numbers;
2. which defects are easiest and which hardest, and *why* you think so;
3. the learning curve shape.

Next we test it on **real** wire-bond images, where we expect it to do worse.
That gap (synthetic → real) is the most important lesson of this project.
""")


# --------------------------------------------------------------------------------------------
def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nbf.write(nb, HERE / "m2_train_dfine.ipynb")
    script = ["# AUTO-GENERATED from build_notebook.py - smoke-test copy of the notebook code\n"]
    for k, s in CELLS:
        if k == "code":
            lines = s.splitlines()
            if lines and lines[-1].startswith("grid.resize"):   # notebook display line
                lines[-1] = "grid.resize((900, 900)).save('_smoke_grid.jpg')"
            script.append("\n".join(lines) + "\n\n")
    (HERE / "_smoke_run.py").write_text("".join(script).replace("plt.show()", "plt.savefig('_smoke_curve.png')"),
                                        encoding="utf-8")
    print("wrote", HERE / "m2_train_dfine.ipynb", "and _smoke_run.py")


if __name__ == "__main__":
    build()
