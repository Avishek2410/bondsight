# BondSight inspector

A small program that looks at pictures of wire-bonded chips, finds defects with our AI model,
remembers every chip it checked, and warns you when the **same defect keeps appearing on the
same wire**. It runs on a normal PC or on a Raspberry Pi 5 (no GPU needed). No camera is needed
to try it: two "simulated cameras" draw synthetic chips for it.

> Research prototype. The current model (M3k-small) was trained on generated pictures and public
> real photos. It has not been validated on a production line yet. See "Results" below.

## What happens when you press Inspect

1. **Take pictures.** Every camera in `config.yaml` takes one picture of the same chip (unit).
   If one camera fails, the others still work. The failure is noted, and the app never crashes.
2. **Find defects.** The AI (an `.onnx` model run by ONNX Runtime) looks at each picture.
   - Big photos are cut into overlapping 640×640 tiles. Small pictures are looked at whole.
   - It returns boxes, each with a defect class and a confidence score (0–1).
3. **Give each defect a place.** Each box is matched to the wire it sits on (for example
   "wire:5"), using the chip's wire map. With no wire map, a grid square such as
   "cell:r2c6" is used instead.
4. **Save.** The chip's pictures go to `data/images/`; its verdict (OK / DEFECT / CHECK),
   defects and timings go into the database, all or nothing. With Docker that is PostgreSQL;
   without it, a SQLite file `data/inspector2.db` (see "The database" below).
5. **Look for repeats.** If the same defect sat on the same wire in **3 of the last 10 chips**,
   you get a *repetitive defect* alert. That usually means a machine problem, such as a worn
   bonding tool, rather than bad luck.
6. **Show it.** The web page draws the boxes on the pictures and lists the history. A heat map
   shows where defects cluster. For simulated chips it also shows what the generator *really*
   drew, so you can see where the AI is right and wrong.

## The three pages

- **Inspect**: the station screen. A stack light (red DEFECT, amber CHECK, green OK), the camera
  pictures with boxes (Top, Angled, a tilted 3D view of both, and the unusual-spot map), and big
  buttons for the operator's decision: *Real defect*, *False alarm*, *Missed something* (drag a
  box around what the AI missed, then pick the defect type) or *Looks good*. INSPECT NEXT checks
  the next unit. The strip at the bottom lists the latest units.
- **Check a photo**: drop photos (up to 20, 30 MB each) or take one with a tablet camera. They
  go through the same AI. Phone photos are turned upright first. Blurry or tiny photos get a
  warning, and a file that is not a picture gets a plain error without stopping the others.
  Uploaded photos never change the line's unit numbers or repeat alarms.
- **Review**: every unit, newest first, with filters (Not decided, DEFECT, CHECK, OK, Photos,
  Line). Tap one to open it and decide.

The operator's name is asked once and saved with every decision. Reports and Setup come later.

## Decisions become training data

A unit is "decided" when every defect the AI found is marked real or false alarm, or a person
confirmed it good or added a missed defect. Decided units can be exported for training:

```bash
inspector/.venv/Scripts/wbinspect-export --config inspector/config.yaml --out training_export/
```

This writes `images/` and `annotations.json` (COCO, the format the training notebooks read).
Real defects and drawn misses become boxes; false alarms keep their picture with no box, so the
next model learns that look-alike is fine. Units nobody reviewed are left out.

## Online demo (visitor mode)

`demo: true` in the config turns the app into a public demo: no cameras, only *Check a photo*,
sample pictures to click (`samples:` folder; ours are in `demo/samples/`, drawn by our own
generator), and **nothing is stored** (an in-memory database, pictures returned inline). Station
pages (inspect, units, pictures, decisions) answer "not available in the demo".

`tools/build_space.py` assembles a ready-to-upload Hugging Face Space (Docker, port 7860) in
`out/space/`, with the model, `CREDITS.md` and `LICENSE`; it refuses to include `.env`, `real/`
or other private files. It only builds; uploading is a separate, deliberate step.

## The database

Default (no `database:` in the config): a SQLite file `data/inspector2.db`, nothing to install.
In a factory, use PostgreSQL: several people and programs can use it at once and IT can back it up.

```yaml
database: postgresql+psycopg://wb@db-host:5432/wirebond
```

The password is **never** written in the config. Put it in the environment variable
`WBI_DB_PASSWORD` (Docker does this for you, see below). The page shows the address with the
password hidden. If the database cannot be reached, `wbinspect` stops with one plain sentence.
Tables are created and upgraded automatically at start (Alembic migrations), so an upgrade keeps
your data. (The v1 file `data/inspector.db` is not used any more.)

## Run it on your PC (no Docker)

```bash
py -3.11 -m venv inspector/.venv
inspector/.venv/Scripts/python -m pip install -e synth -e "inspector[test]"
copy inspector\config.example.yaml inspector\config.yaml
mkdir inspector\models
curl -L -o inspector/models/wirebond_dfine_small_m3k.onnx https://github.com/Avishek2410/bondsight/releases/download/v0.1.0/wirebond_dfine_small_m3k.onnx
curl -L -o inspector/models/wirebond_dfine_small_m3k.json https://github.com/Avishek2410/bondsight/releases/download/v0.1.0/wirebond_dfine_small_m3k.json
inspector/.venv/Scripts/wbinspect --config inspector/config.yaml
```

The model files come from the [v0.1.0 release](https://github.com/Avishek2410/bondsight/releases/tag/v0.1.0),
together with their credits.

Open http://127.0.0.1:8080 and press **Inspect**. Tests:
`inspector/.venv/Scripts/python -m pytest inspector/tests -q`.

## Run it with Docker (PC or Raspberry Pi)

```bash
mkdir -p docker/models docker/data          # create data/ yourself first, so it is yours, not root's
cp inspector/models/wirebond_dfine_small_m3k.* docker/models/   # downloaded as above
cp docker/config.docker.yaml docker/config.yaml
python3 docker/make_env.py                  # once: random database password in docker/.env (git-ignored)
cd docker && docker compose up -d --build   # PostgreSQL + the inspector; the Pi builds its own arm64 image
```

The page is published on `127.0.0.1:8080`, so only that machine can open it. To view a Pi's page
from your PC, use `ssh -L 8081:127.0.0.1:8080 <pi>` and open http://127.0.0.1:8081. A
multi-arch image for a registry would be built with
`docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile .`.
That is not done yet, because publishing comes later.

PostgreSQL keeps its files in `docker/pgdata/` (git-ignored) and has no port open to the network:
only the inspector container can reach it. Back up `docker/pgdata/` and `docker/data/images/`
together.

## Add a camera

Copy a block under `cameras:` in the config. Every camera needs a unique `name`, a `type` and a
`view` (`top` or `angled`):

| type | what it is | options |
|---|---|---|
| `generator` | draws a synthetic chip (unit N) | `product_seed`, `n_wires`, `injection` |
| `folder` | the next unread image file from a folder | `path`, `pattern` |
| `opencv` | USB camera or microscope (`source: 0`), or network camera (`source: rtsp://…`) | `source`, `width`, `height` |

Raspberry Pi ribbon (CSI) cameras are on the roadmap.

## Swap the model

Put a new `.onnx` and its `.json` (class names, input size, optional `tiling`) in `models/`.
Point `model:` at it, then restart. No rebuild is needed.

## Second opinion: the unusual-spot checker (optional)

The main AI only knows its 8 defect types. The anomaly checker (`anomaly.py`, PatchCore idea)
knows only what GOOD units look like: it remembers small patches of good pictures and scores how
far each patch of a new picture is from all of them. A score above the camera's alarm level =
"unusual". If the AI found nothing but the checker is alarmed, the unit is marked **check**
and the page can show a red map of the unusual spots.

```
tools/.venv/Scripts/python tools/export_anomaly_backbone.py        # ResNet18 (BSD-3) -> ONNX, once
#   (or download anomaly_backbone.onnx + .json from the v0.1.0 release; anomaly-product7.zip there
#    holds ready-made memories for the two simulated cameras in config.example.yaml)
# add to config.yaml:  anomaly: {backbone: <..>/anomaly_backbone.onnx, dir: <..>/anomaly}
inspector/.venv/Scripts/wbinspect-fit-anomaly --config inspector/config.yaml   # memory per camera
```
Generator cameras draw their own good units; real cameras need `--good-dir DIR` with
`DIR/<camera name>/` full of good pictures. Redo it when the product, light or camera changes.

Measured on synthetic product 7, 60 defective + 60 good units, both cameras (tools/anomaly_eval.py,
640 px): AI alone caught 54/60 defective units; AI + checker 57/60; the checker put 0/60 good
units on "check". It catches shape defects best in the angled view (sagging loop, lifted stitch),
which the top view cannot see. Cost: ~0.65 s per picture on a PC, ~1.3 s on the Pi 5
(a unit with two cameras: 1.8 s PC, 2.9 s Pi).

## Speed (M3k-small, measured 2026-11-13, CPU only)

| Machine | One 2048 px photo | One 640 px picture | One unit (2 cameras + checker) |
|---|---|---|---|
| Laptop CPU (i5-12600H) | 2.3 s | | |
| Raspberry Pi 5, Docker | 3.1 s | 0.54 s | 3.5 s |

## Results

On test photos the model never saw in training: 63% of real NTUST defects found, 82% of missing
wires in the WHU close-ups, every bent wire in the Fuji power-module photos, and no complete good
board flagged in the hawk-team set. Some of these test sets come from the same sources as the
training data, so on a new product expect lower numbers until the model has been retrained on it.
All numbers, and how each model got there, are in [docs/results.md](../docs/results.md).
