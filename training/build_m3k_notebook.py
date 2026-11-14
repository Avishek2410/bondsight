"""Build the M3k notebook (training/m3k_train.ipynb) and its smoke-test script (training/_smoke_m3k.py).

M3k = M3j's recipe with four changes, all decided after M3j (see docs/ and the project notes):
  1. 12 epochs instead of 40 (M3j was best at epoch 8, then over-fitted),
  2. TWO models in one run: D-FINE Nano (as M3j) and the bigger D-FINE S, so we can compare,
  3. real photos of GOOD wires as extra "stay quiet" examples (hawk-team wire photos, lyra chip
     photos, Fuji power modules marked Pass), to cut false alarms,
  4. one confidence level per defect type, learned on validation pictures and written into the
     model's .json (the inspector reads `class_thresholds`).
New honest tests: good units of three real products (should stay quiet) and Fuji modules marked
Defect (should be flagged).

The cells are M3j's (training/build_joint_notebook.py), patched below, so both notebooks keep
the same data split and the same tests. Kaggle inputs: wirebond-train-v2 (as M3j) plus
wirebond-train-v3x (new photos, marker WBV3X_ROOT). Locally:
    tools/.venv/Scripts/python training/build_m3k_notebook.py
    SMOKE=1 WB_ROOT=<stage>/wbv2 X_ROOT=<stage>/wbv3x WORK_DIR=<scratch> tools/.venv/Scripts/python training/_smoke_m3k.py
"""
from pathlib import Path

import nbformat as nbf

import build_joint_notebook as joint

HERE = Path(__file__).parent
CELLS = [list(c) for c in joint.CELLS]


def sub(i: int, old: str, new: str) -> None:
    assert CELLS[i][1].count(old) == 1, f"cell {i}: {old[:60]!r} not found exactly once"
    CELLS[i][1] = CELLS[i][1].replace(old, new)


def put(i: int, kind: str, text: str) -> None:
    CELLS[i] = [kind, text.strip("\n")]


def insert_after(i: int, *cells: tuple[str, str]) -> None:
    for k, (kind, text) in enumerate(cells):
        CELLS.insert(i + 1 + k, [kind, text.strip("\n")])


# ---- 0: intro --------------------------------------------------------------------------------
put(0, "md", """
# Part M3k: shorter training, a bigger model, real good wires, one level per defect type

M3j (our current model) was trained once on all our data for 40 epochs. Four lessons since then:

1. **It was best at epoch 8**, then got worse (over-fitting). M3k trains **12 epochs**.
2. Is a **bigger model** better? M3k trains **two** models in this one run: D-FINE **Nano** (same
   size as M3j) and D-FINE **S** (about 3 times bigger, slower on the Pi), so we can compare.
3. **Too many false alarms** (29% of good NTUST wires at threshold 0.3). M3k also learns from
   photos of real **good** wires of three other products (from Roboflow Universe, CC BY 4.0):
   gold wires on a sensor board (hawk-team), power-chip wires (lyra) and power modules marked
   *Pass* (Fuji). On these photos the right answer is "nothing to report". Fuji modules marked
   *Defect* add something we had almost none of: **real bent wires** (`wire_sweep`) with boxes.
4. **One threshold for all eight defect types** is a compromise. M3k learns **one confidence
   level per defect type** on validation pictures and stores it in the model's `.json`; the
   inspector app uses it automatically.

New honest tests in Step 7: good units of the three new products (should stay quiet) and held-back
Fuji power modules marked **Defect** (their bent wires should be found), next to all of M3j's tests.
Expected time on a Kaggle T4 GPU: about **2 to 2.5 hours**.
""")

# ---- 1-2: settings ---------------------------------------------------------------------------
put(1, "md", """
## Step 0: Settings

- `MODELS`: the two general D-FINE models trained on COCO everyday photos (Apache-2.0 licence).
- `EPOCHS = 12`, `LR = 2e-4`. The per-epoch mix is M3j's plus `NEG_PER_EPOCH` tiles of real
  good wires (drawn fresh every epoch).
- Tiling is the same as in M2c and M3j.
""")
sub(2, "EPOCHS = 1 if SMOKE else 40", "EPOCHS = 1 if SMOKE else 12")
sub(2, 'CHECKPOINT = os.environ.get("CHECKPOINT", "ustc-community/dfine-nano-coco")   # D-FINE Nano, Apache-2.0\n'
       'OUT_MODEL = "best_model_joint"\n'
       'print("smoke test" if SMOKE else "joint training", "| epochs:", EPOCHS, "| start from:", CHECKPOINT)',
    'NEG_PER_EPOCH = 8 if SMOKE else 700   # tiles of real GOOD wires per epoch (hawk + chip + Fuji Pass)\n'
    'FUJI_TILES = 2 if SMOKE else 8        # random tiles kept per Fuji Pass training photo\n'
    'MODELS = {"nano": "ustc-community/dfine-nano-coco",      # D-FINE Nano, Apache-2.0 (as M3j)\n'
    '          "small": "ustc-community/dfine-small-coco"}    # D-FINE S, Apache-2.0 (about 3x bigger)\n'
    'CHECKPOINT = MODELS["nano"]           # the picture pre-processing is the same for both\n'
    'OUT_MODELS = {k: f"best_model_m3k_{k}" for k in MODELS}\n'
    'print("smoke test" if SMOKE else "M3k training", "| epochs:", EPOCHS, "| models:", list(MODELS))')

# ---- 6: data: also find the new bundle and load the good-wire photos ---------------------------
CELLS[6][1] += """

# ---- new bundle wirebond-train-v3x: M3j for the comparison + real photos of GOOD wires ----
X_ROOT = os.environ.get("X_ROOT") or str(Path(sorted(glob.glob("/kaggle/input/**/WBV3X_ROOT", recursive=True))[0]).parent)
REF_M3J = f"{X_ROOT}/ref/best_model_joint"

def coco_files(folder):
    # Roboflow export: train/ valid/ test/ each with _annotations.coco.json
    for split in ("train", "valid", "test"):
        f = Path(folder) / split / "_annotations.coco.json"
        if f.exists():
            yield {"valid": "val"}.get(split, split), f

# hawk-team: gold wires on a sensor board, every wire traced. Some photos are of half-bonded or
# damaged boards (many fewer wires), so only COMPLETE boards are used: wire count within 1 of the
# usual count for that board layout. Wire outlines = "good" boxes that must not be flagged.
HAWK = defaultdict(list)
hawk_rows = []
for split, f in coco_files(glob.glob(f"{X_ROOT}/real/roboflow/hawk-team*")[0]):
    coco = json.load(open(f))
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a["bbox"])
    for im in coco["images"]:
        hawk_rows.append((split, (im["width"], im["height"]), str(f.parent / im["file_name"]), anns[im["id"]]))
usual = {size: Counter(len(w) for _, s, _, w in hawk_rows if s == size).most_common(1)[0][0]
         for size in {s for _, s, _, _ in hawk_rows}}
for split, size, path, wires in hawk_rows:
    if abs(len(wires) - usual[size]) <= 1:
        HAWK[split].append({"path": path, "anns": [], "good": wires})
print("hawk-team complete boards:", {k: len(v) for k, v in HAWK.items()},
      f"(of {len(hawk_rows)}; usual wire counts {sorted(usual.values())})")

# lyra: power-chip wires (small grey photos, looked at whole). No defects labelled; the label
# box is the whole chip, so there is nothing to mark as "good" in particular.
CHIP = defaultdict(list)
for split, f in coco_files(glob.glob(f"{X_ROOT}/real/roboflow/lyra-wikbf*")[0]):
    coco = json.load(open(f))
    for im in coco["images"]:
        CHIP[split].append({"path": str(f.parent / im["file_name"]), "anns": [], "good": []})
print("lyra chips:", {k: len(v) for k, v in CHIP.items()})

# Fuji power modules: each photo is marked Pass or Defect; a Defect photo has a box around each
# wire bent out of its lane (leaning on or towards its neighbour) -> our class wire_sweep.
# The bundle has the photos already shrunk to half size (they are 40 MP), so they are tiled at scale 1.
# Split = Roboflow's own: its train part for training, its valid part for the test.
FUJI = []
for it in json.load(open(f"{X_ROOT}/fuji/fuji.json")):
    FUJI.append({"path": f"{X_ROOT}/fuji/{it['file']}", "split": it["split"], "defect": it["defect"], "good": [],
                 "anns": [{"bbox": b, "category_id": CLASSES.index("wire_sweep") + 1, "area": b[2] * b[3]}
                          for b in it["anns"]]})
FUJI_TRAIN = [it for it in FUJI if it["split"] == "train"]
FUJI_TEST = [it for it in FUJI if it["split"] == "test"]
if SMOKE:
    HAWK = {k: v[:2] for k, v in HAWK.items()}; CHIP = {k: v[:2] for k, v in CHIP.items()}
    pick = lambda xs: [i for i in xs if not i["defect"]][:2] + [i for i in xs if i["defect"]][:2]
    FUJI_TRAIN, FUJI_TEST = pick(FUJI_TRAIN), pick(FUJI_TEST)
for k, v in (("train", FUJI_TRAIN), ("test", FUJI_TEST)):
    print(f"Fuji {k}: {sum(not i['defect'] for i in v)} Pass + {sum(i['defect'] for i in v)} Defect photos "
          f"({sum(len(i['anns']) for i in v)} bent wires)")
"""

# ---- 8: tiles: allow a different scale and a cap on tiles per photo ------------------------------
sub(8, "def make_tiles(items, split, root=None):", "def make_tiles(items, split, root=None, scale=None, keep=None, seed=SEED):")
sub(8, "    skipped = 0\n    for it in items:\n        img = Image.open(it[\"path\"]).convert(\"RGB\")\n"
       "        w, h = round(img.width * TILE_SCALE), round(img.height * TILE_SCALE)\n"
       "        img = img.resize((w, h), Image.LANCZOS)\n        for x, y in tile_offsets(w, h):",
    "    skipped = 0\n    s = TILE_SCALE if scale is None else scale\n    rng = random.Random(seed)\n"
    "    for it in items:\n        img = Image.open(it[\"path\"]).convert(\"RGB\")\n"
    "        w, h = round(img.width * s), round(img.height * s)\n"
    "        if s != 1:\n            img = img.resize((w, h), Image.LANCZOS)\n"
    "        offs = tile_offsets(w, h)\n        if keep and len(offs) > keep:\n            offs = rng.sample(offs, keep)\n"
    "        for x, y in offs:")
sub(8, "                bb, vis = cut_box(a[\"bbox\"], TILE_SCALE, x, y)", "                bb, vis = cut_box(a[\"bbox\"], s, x, y)")
sub(8, "                bb, vis = cut_box(g, TILE_SCALE, x, y)", "                bb, vis = cut_box(g, s, x, y)")

# ---- new step 3c after the MEMS close-ups (cell 10) ---------------------------------------------
insert_after(10, ("md", """
## Step 3c: Real GOOD wires (the new "stay quiet" examples) and Fuji's bent wires

hawk-team boards and Fuji Pass modules are cut into tiles like the NTUST photos (Fuji: 8 random
tiles per photo, it has about 80). lyra chip photos are small and used whole.
Fuji Defect photos: every tile with a bent wire (`wire_sweep`, real!) plus 2 tiles without one.
These real bent wires are shown twice per epoch (`FUJI_REPEAT`).
"""), ("code", """
FUJI_REPEAT = 2
NEG_DIR = WORK / "tiles_good"
NEG_TILES = {}
for k in ("train", "val"):
    hawk_t, _ = make_tiles(HAWK.get(k, []), k, NEG_DIR / "hawk")
    NEG_TILES[k] = hawk_t + CHIP.get(k, [])
fuji_t, _ = make_tiles([i for i in FUJI_TRAIN if not i["defect"]], "train", NEG_DIR / "fuji", scale=1.0, keep=FUJI_TILES)
NEG_TILES["train"] += fuji_t
dt, skipped = make_tiles([i for i in FUJI_TRAIN if i["defect"]], "train_defect", NEG_DIR / "fuji", scale=1.0)
FUJI_DEF_TILES = [t for t in dt if t["anns"]] + random.Random(SEED).sample(
    [t for t in dt if not t["anns"]], min(2 * len([i for i in FUJI_TRAIN if i["defect"]]), len([t for t in dt if not t["anns"]])))
print({k: f"{len(v)} good-wire tiles" for k, v in NEG_TILES.items()}, f"(of which {len(fuji_t)} from Fuji Pass modules)")
print(f"Fuji Defect: {len(FUJI_DEF_TILES)} tiles ({sum(len(t['anns']) for t in FUJI_DEF_TILES)} bent wires), "
      f"{skipped} skipped because a bent wire was cut in half")
"""))
def cell(first_line: str) -> int:
    return next(i for i, (_, s) in enumerate(CELLS) if s.splitlines()[0].startswith(first_line))


# ---- step 5: add the good-wire tiles to every epoch's mix; synthetic validation designs --------
c5 = cell("prods = PRODUCTS[:]")
sub(c5, 'SYN_SPLIT = {"test": prods[:n], "train": prods[2 * n:]}',
    'SYN_SPLIT = {"test": prods[:n], "val": prods[n:2 * n], "train": prods[2 * n:]}')
sub(c5, 'MACRO_SPLIT = {"test": mprods[:2], "train": mprods[2:]}',
    'MACRO_SPLIT = {"test": mprods[:2], "val": mprods[2:4], "train": mprods[4:]}')
sub(c5, "SYN_POOL = load_synth(SYN_SPLIT[\"train\"])",
    "# validation pictures for choosing one confidence level per defect type (Step 7b)\n"
    "SYN_VAL = random.Random(SEED).sample(load_synth(SYN_SPLIT[\"val\"]), 8 if SMOKE else 300)\n"
    "MACRO_VAL = random.Random(SEED).sample(load_synth(MACRO_SPLIT[\"val\"], MACRO_ROOT), 8 if SMOKE else 150)\n"
    "SYN_POOL = load_synth(SYN_SPLIT[\"train\"])")
sub(c5, "             + rng.sample(MACRO_POOL, min(MACRO_PER_EPOCH, len(MACRO_POOL))))",
    "             + rng.sample(MACRO_POOL, min(MACRO_PER_EPOCH, len(MACRO_POOL)))\n"
    "             + rng.sample(NEG_TILES[\"train\"], min(NEG_PER_EPOCH, len(NEG_TILES[\"train\"])))\n"
    "             + FUJI_DEF_TILES * FUJI_REPEAT)")
sub(c5, "      f\"{min(MACRO_PER_EPOCH, len(MACRO_POOL))} of {len(MACRO_POOL)} macro synthetic = {len(TRAIN)}\")",
    "      f\"{min(MACRO_PER_EPOCH, len(MACRO_POOL))} of {len(MACRO_POOL)} macro synthetic + \"\n"
    "      f\"{min(NEG_PER_EPOCH, len(NEG_TILES['train']))} of {len(NEG_TILES['train'])} good-wire tiles + \"\n"
    "      f\"{len(FUJI_DEF_TILES)} Fuji bent-wire tiles x {FUJI_REPEAT} = {len(TRAIN)}\")")
sub(c5, "model = DFineForObjectDetection.from_pretrained(\n"
        "    CHECKPOINT, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True).to(DEVICE)\n", "")
sub(c5, '"val": make_loader(TILES["val"] + NEWP_TILES["val"] + MEMS["val"] + CROPS["val"], False)}',
    '"val": make_loader(TILES["val"] + NEWP_TILES["val"] + MEMS["val"] + CROPS["val"] + NEG_TILES["val"], False)}')

# ---- step 6: train each model ------------------------------------------------------------------
c6m = cell("## Step 6:")
put(c6m, "md", """
## Step 6: Train both models (12 epochs each, fresh balanced mix each epoch)

Same training loop as M3j, run once for D-FINE Nano and once for D-FINE S. Each keeps the epoch
with the lowest validation loss. Nano takes about 30 minutes, S about twice as long.
""")
put(c6m + 1, "code", """
def run_epoch(model, opt, sched, scaler, split, train):
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

HISTORY, BEST_EPOCH = {}, {}
for name, ckpt in MODELS.items():
    torch.manual_seed(SEED)
    model = DFineForObjectDetection.from_pretrained(
        ckpt, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True).to(DEVICE)
    loaders["train"] = make_loader(epoch_items(0), True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 50), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    HISTORY[name] = {"train": [], "val": []}
    best, t0 = math.inf, time.time()
    print(f"==== {name}: {ckpt} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters) ====", flush=True)
    for ep in range(EPOCHS):
        t = time.time()
        if ep > 0:
            loaders["train"] = make_loader(epoch_items(ep), True)      # fresh balanced mix
        tr = run_epoch(model, opt, sched, scaler, "train", True)
        va = run_epoch(model, opt, sched, scaler, "val", False)
        HISTORY[name]["train"].append(tr); HISTORY[name]["val"].append(va)
        if va < best:
            best, BEST_EPOCH[name] = va, ep + 1
            model.save_pretrained(OUT_MODELS[name]); processor.save_pretrained(OUT_MODELS[name])
            print("   new best, saved", flush=True)
        print(f"{name} epoch {ep + 1:2d}/{EPOCHS}  train loss {tr:.3f}  val loss {va:.3f}  "
              f"({time.time() - t:.0f}s, total {(time.time() - t0) / 60:.0f} min)", flush=True)
    del model, opt; torch.cuda.empty_cache() if DEVICE == "cuda" else None
print("best epochs:", BEST_EPOCH)
""")
put(c6m + 2, "code", """
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 3.5))
for name, h in HISTORY.items():
    plt.plot(range(1, len(h["train"]) + 1), h["train"], "o-", label=f"{name} train loss (mix)")
    plt.plot(range(1, len(h["val"]) + 1), h["val"], "s--", label=f"{name} validation loss (real)")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.show()
""")

# ---- step 7: four models, new tests ------------------------------------------------------------
c7m = cell("## Step 7:")
put(c7m, "md", """
## Step 7: The honest test: M2c, M3j, M3k Nano, M3k S

None of these test pictures were used for training. First M3j's table (all models at the same
threshold 0.4, so it is directly comparable with earlier parts), with three new rows of real
**good** units (only false alarms count there). Then (7b) one confidence level per defect type,
and (7c) the final comparison with each model at the threshold it would really use.
""")
c7 = c7m + 1
sub(c7, "def predict_tiled(m, path, thr):", "def predict_tiled(m, path, thr, scale=None):")
sub(c7, "    w, h = round(img.width * TILE_SCALE), round(img.height * TILE_SCALE)\n    small = img.resize((w, h), Image.LANCZOS)",
    "    s = TILE_SCALE if scale is None else scale\n"
    "    w, h = round(img.width * s), round(img.height * s)\n"
    "    small = img.resize((w, h), Image.LANCZOS) if s != 1 else img")
sub(c7, "        B.append(b / TILE_SCALE); S.append(r[\"scores\"].cpu()); L.append(r[\"labels\"].cpu())",
    "        B.append(b / s); S.append(r[\"scores\"].cpu()); L.append(r[\"labels\"].cpu())")
sub(c7, '         "macro synthetic    ": (MACRO_TEST, predict_whole)}',
    '         "macro synthetic    ": (MACRO_TEST, predict_whole),\n'
    '         "hawk good boards   ": (HAWK["test"], predict_tiled),\n'
    '         "lyra good chips    ": (CHIP["test"], predict_whole)}\n'
    'MODEL_PATHS = {"M2c": REF_MODEL, "M3j": REF_M3J, "M3k-nano": OUT_MODELS["nano"], "M3k-small": OUT_MODELS["small"]}')
sub(c7, 'for name, path in (("M2c", REF_MODEL), ("M3j", OUT_MODEL)):', "for name, path in MODEL_PATHS.items():")

c7s = cell("def sweep(m, items, target=0.30):")
sub(c7s, 'for name, path in (("M2c", REF_MODEL), ("M3j", OUT_MODEL)):', "for name, path in MODEL_PATHS.items():")

insert_after(c7s, ("md", """
### Step 7b: One confidence level per defect type

For each M3k model and each defect type: run the **validation** pictures (never used for
training, never used for the tests), try levels from 0.20 to 0.80, and keep the one with the best
balance of found defects and false alarms (the F1 score). A type with too few validation examples
keeps the general level 0.3. These levels go into the model's `.json` (Step 9).
"""), ("code", """
LEVELS = [round(0.20 + 0.05 * i, 2) for i in range(13)]
VAL_ITEMS = [(it, predict_whole) for it in TILES["val"] + NEWP_TILES["val"] + MEMS["val"] + CROPS["val"]
             + NEG_TILES["val"] + SYN_VAL + MACRO_VAL]

def class_levels(m, min_examples=8, default=0.3):
    preds = [fn(m, it["path"], 0.05) for it, fn in VAL_ITEMS]
    levels, report = {}, {}
    for c, cname in enumerate(CLASSES):
        n_true = sum(a["category_id"] - 1 == c for it, _ in VAL_ITEMS for a in it["anns"])
        if n_true < min_examples:
            levels[cname], report[cname] = default, f"{n_true} examples -> {default} (too few to choose)"
            continue
        best = (-1, default)
        for thr in LEVELS:
            tp = fp = 0
            for (B, S, L), (it, _) in zip(preds, VAL_ITEMS):
                keep = (S >= thr) & (L == c)
                kb = B[keep].tolist()
                gts = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                       for a in it["anns"] if a["category_id"] - 1 == c]
                hit = [any(iou(p, g) >= 0.3 for g in gts) for p in kb]
                tp += sum(any(iou(p, g) >= 0.3 for p in kb) for g in gts); fp += hit.count(False)
            f1 = 2 * tp / max(2 * tp + fp + (n_true - tp), 1)
            if f1 >= best[0]:
                best = (f1, thr)
        levels[cname], report[cname] = best[1], f"{n_true} examples -> {best[1]:.2f} (F1 {best[0]:.2f})"
    return levels, report

CLASS_LEVELS = {}
for name in ("nano", "small"):
    m = DFineForObjectDetection.from_pretrained(OUT_MODELS[name]).to(DEVICE).eval()
    CLASS_LEVELS[name], rep = class_levels(m)
    print(f"M3k-{name}:"); [print(f"   {k:16s} {v}") for k, v in rep.items()]
"""), ("md", """
### Step 7c: Final comparison, each model at the threshold it would really use

M2c at 0.40 and M3j at 0.30 (their inspector settings), M3k at its per-type levels. Rows:
- **defects found** on NTUST, new_photos, MEMS, synthetic: higher is better,
- **good wires flagged** on NTUST and the hawk boards: lower is better,
- **good units flagged**: share of whole good units (hawk boards, lyra chips, Fuji Pass modules)
  with at least one finding. Lower is better: each one is a false alarm on the factory floor,
- **bent wires found** on the Fuji test modules marked Defect (`wire_sweep` in the right place),
  and **Defect modules flagged**: share of those modules with at least one finding anywhere.
"""), ("code", """
def at_levels(preds, items, levels):
    lv = torch.tensor([levels[c] for c in CLASSES]) if isinstance(levels, dict) else None
    found = total = flag = gtot = units = 0
    for (B, S, L), it in zip(preds, items):
        keep = S >= (lv[L] if lv is not None else levels)
        kb, kl = B[keep].tolist(), L[keep].tolist()
        units += bool(kb)
        for a in it.get("anns", []):
            b = [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
            total += 1; found += any(iou(p, b) >= 0.3 and pl == a["category_id"] - 1 for p, pl in zip(kb, kl))
        for g in it.get("good", []):
            gb = [g[0], g[1], g[0] + g[2], g[1] + g[3]]
            gtot += 1; flag += any(iou(p, gb) >= 0.3 for p in kb)
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"
    return pct(found, total), pct(flag, gtot), pct(units, len(items))

fuji_pass = [it for it in FUJI_TEST if not it["defect"]]
fuji_def = [it for it in FUJI_TEST if it["defect"]]
fuji_tiled = lambda m, p, t: predict_tiled(m, p, t, scale=1.0)     # bundle photos are already half size
FINAL_SETS = {"NTUST": (REAL["test"], predict_tiled), "new_photos": (NEWP["test"], predict_tiled),
              "MEMS close-ups": (CROPS["test"], predict_whole), "synthetic": (SYN_TEST, predict_whole),
              "macro synthetic": (MACRO_TEST, predict_whole), "hawk good boards": (HAWK["test"], predict_tiled),
              "lyra good chips": (CHIP["test"], predict_whole), "Fuji Pass": (fuji_pass, fuji_tiled),
              "Fuji Defect": (fuji_def, fuji_tiled)}
USE = {"M2c": 0.40, "M3j": 0.30, "M3k-nano": CLASS_LEVELS["nano"], "M3k-small": CLASS_LEVELS["small"]}
FINAL = {}
for name, path in MODEL_PATHS.items():
    m = DFineForObjectDetection.from_pretrained(path).to(DEVICE).eval()
    t = time.time()
    for sname, (items, fn) in FINAL_SETS.items():
        FINAL[(name, sname)] = at_levels([fn(m, it["path"], 0.05) for it in items], items, USE[name])
    print(f"{name} done ({time.time() - t:.0f}s)", flush=True)

rows = [("defects found", "NTUST", 0), ("defects found", "new_photos", 0), ("defects found", "MEMS close-ups", 0),
        ("defects found", "synthetic", 0), ("defects found", "macro synthetic", 0),
        ("bent wires found", "Fuji Defect", 0),
        ("good wires flagged", "NTUST", 1), ("good wires flagged", "hawk good boards", 1),
        ("good units flagged", "hawk good boards", 2), ("good units flagged", "lyra good chips", 2),
        ("good units flagged", "Fuji Pass", 2), ("Defect modules flagged", "Fuji Defect", 2)]
print(f"{'':42s}" + "".join(f"{n:>11s}" for n in MODEL_PATHS))
for what, sname, k in rows:
    print(f"{what + ' - ' + sname:42s}" + "".join(f"{FINAL[(n, sname)][k]:>11s}" for n in MODEL_PATHS))
"""))

# ---- step 8: pictures with the Nano model ---------------------------------------------------------
c8 = cell("new = DFineForObjectDetection.from_pretrained(OUT_MODEL)")
sub(c8, "new = DFineForObjectDetection.from_pretrained(OUT_MODEL).to(DEVICE).eval()",
    'new = DFineForObjectDetection.from_pretrained(OUT_MODELS["nano"]).to(DEVICE).eval()')
c8m = cell("## Step 8:")
sub(c8m, "**Yellow** = M3j's answers", "**Yellow** = M3k Nano's answers")

# ---- step 9: export both -------------------------------------------------------------------------
c9m = cell("## Step 9:")
put(c9m, "md", """
## Step 9: Export both M3k models for the Pi

Saves `wirebond_dfine_nano_m3k.onnx` and `wirebond_dfine_small_m3k.onnx` (each with its `.json`,
which now includes `class_thresholds`) and checks that ONNX gives the **same detections** as
PyTorch on real validation tiles. Download them from the notebook's **Output** tab.
""")
put(c9m + 1, "code", """
import onnxruntime as ort
NOTE = ("M3k ({}): 12 epochs from COCO D-FINE {} on synthetic (whole + macro) + NTUST wire bonding + WHU mems_60 + "
        "WHU new_photos + Fuji Wire Inspection (bent wires -> wire_sweep, Pass modules) + good-wire photos "
        "(hawk-team wire, lyra chip); all real sets CC BY 4.0")
for name in MODELS:
    model_cpu = DFineForObjectDetection.from_pretrained(OUT_MODELS[name]).eval()
    onnx_path = str(WORK / f"wirebond_dfine_{name}_m3k.onnx")
    torch.onnx.export(model_cpu, (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),), onnx_path, input_names=["pixel_values"],
                      output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
    json.dump({"classes": CLASSES, "img_size": IMG_SIZE, "note": NOTE.format(name, name.capitalize()),
               "best_epoch": BEST_EPOCH.get(name), "class_thresholds": CLASS_LEVELS[name],
               "tiling": {"scale": TILE_SCALE, "tile": IMG_SIZE, "stride": TILE_STRIDE, "nms_iou": 0.5}},
              open(onnx_path.replace(".onnx", ".json"), "w"), indent=1)
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
    print(f"saved {onnx_path} ({Path(onnx_path).stat().st_size / 1e6:.0f} MB) | same detections as PyTorch on {same}/{len(check)} real tiles")
""")
put(len(CELLS) - 1, "md", """
## Done!

Say "status" (or send a screenshot of **Step 7c**). The winner goes into the inspector app.
""")


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nbf.write(nb, HERE / "m3k_train.ipynb")
    script = ["# AUTO-GENERATED from build_m3k_notebook.py - smoke-test copy\n"]
    for k, s in CELLS:
        if k == "code":
            lines = s.splitlines()
            if lines and lines[-1] == "grid":
                lines[-1] = "grid.save('_smoke_m3k_grid.jpg')"
            script.append("\n".join(lines) + "\n\n")
    (HERE / "_smoke_m3k.py").write_text(
        "".join(script).replace("plt.show()", "plt.savefig('_smoke_m3k_curve.png')"), encoding="utf-8")
    print("wrote", HERE / "m3k_train.ipynb", "and _smoke_m3k.py")


if __name__ == "__main__":
    build()
