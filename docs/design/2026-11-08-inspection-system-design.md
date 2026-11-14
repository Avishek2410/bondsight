# Wire-Bond Inspection System — Design

*Status: draft for approval · 2026-11-08 · Author: Dr. Avishek Mondal (with Claude)*

## 1. Intent

Build an open, reproducible AI system that inspects wire bonds in semiconductor packages from
camera images, detects the common defects (focus: **wire sweep** and **lifted bonds**), and tells
**random** defects apart from **repetitive** ones (same wire position failing unit after unit).

Why: wire-bond defect data is private to factories, and commercial AOI is mostly rule-based with
many false calls that operators re-check by eye. A public, end-to-end system — synthetic data →
model → deployable inspector — is useful to learners and is the backbone of the article series.
Writing comes later; the system comes first.

## 2. Goals

1. **Synthetic data generator** — produces labelled wire-bond images for every defect class, from
   two views (top-down and angled), with controllable random vs repetitive defect injection.
2. **Trained detector** — trained on free cloud GPUs (Kaggle / Colab), exported to ONNX.
3. **Inspector runtime** — a Docker image that runs on a Raspberry Pi 5 (CPU only) and on any
   x86 PC, takes images from any number and type of cameras listed in a config file, runs the
   model, and shows results in a web UI.
4. **Repetitive-defect analytics** — tracks defects per wire position across units and raises a
   "repetitive" alert when the same position keeps failing.
5. **Honest reality check** — evaluate on the few public real wire-bond images and report how far
   synthetic training transfers.

## 3. Non-goals (for v1)

- Beating commercial AOI on optical resolution. We aim to beat it on **judgement**: fewer false
  calls, defect classification, repetitive-pattern detection.
- 3D height measurement, X-ray, pull/shear test data.
- An AI accelerator (no AI HAT, no GPU at inference).
- Battery recycling (separate later project).
- Machine control (stopping a bonder, GPIO triggers, MES integration).

## 4. Anti-goals (would make it a failure even if it "works")

- Claiming accuracy numbers on synthetic data as if they were real-factory results.
- A system that only runs on the author's hardware, or only with one camera model.
- Anything requiring paid services or a GPU to *use* it.
- Unexplained "black box" output — every flagged defect must show where and what class.

## 5. Constraints

| Constraint | Detail |
|---|---|
| Edge hardware | Raspberry Pi 5, 8 GB, CPU only, SD-card storage |
| Cameras (start) | 2 × Raspberry Pi 5 MP camera (OV5647, ₹309) on the two CSI ports, with 22→15-pin cables. Prototype quality only. |
| Cameras (upgrade) | Pi HQ Camera (12 MP, C-mount) + microscope lens — same software, config change only |
| Training | Free GPU only (Kaggle / Colab). Dev PC has no GPU. |
| Data | **No real images.** Synthetic first; public real images only for evaluation. |
| Deployment | Docker, one multi-arch image (linux/arm64 + linux/amd64), docker compose |
| Publishing | Public GitHub repo, educational |
| Security | Web UI binds to 127.0.0.1 by default; LAN access is an explicit opt-in in config |

## 6. Decided by the author

- Start with no real images; generate synthetic data.
- Inspection views: one top-down + one angled camera (mirrors real AOI lighting).
- All processing on the Pi 5 CPU, no AI HAT.
- Docker, so it deploys on any hardware with as many and as different cameras as required.
- Buy the ₹309 cameras now for building; HQ microscope camera later.

## 7. Architecture

```
                    OFFLINE (PC / free cloud GPU)
 ┌──────────────────┐     ┌──────────────────┐
 │ synth/ generator │ ──► │ training/ notebook│ ──► model.onnx
 └──────────────────┘     └──────────────────┘
                                                      │
                    RUNTIME (Docker: Pi 5 or any PC)  ▼
 config.yaml ─► ┌──────────────────────────────────────────────┐
                │ inspector                                    │
                │  sources ──► detector (ONNX Runtime, CPU)    │
                │  (csi/usb/   ──► per-unit result ──► SQLite  │
                │   rtsp/folder)        ──► repetitive tracker │
                │  web UI (FastAPI + static HTML)              │
                └──────────────────────────────────────────────┘
```

### 7.1 Synthetic generator (`synth/`)

- Pure Python (NumPy + OpenCV/Pillow), runs on PC, Pi or Colab. No Blender dependency.
- **Product layout**: a die with N pads (configurable, e.g. 16–64), leads around it, a bond map
  (pad i → lead i). A layout is fixed for a "product"; many "units" are rendered from it.
- **Rendering**: top-down view (ring-light look: bright ball tops, specular wire highlights) and
  angled view (side light: loop height and ball/stitch contact visible as shading). Domain
  randomisation: illumination, blur, noise, colour, small rotation/offset, die texture.
- **Defect classes (v1)**:

| Class | Region | Visual cue |
|---|---|---|
| `wire_sweep` | loop | wire bowed sideways from the ball→stitch line |
| `lifted_ball` | ball | ball shifted/raised; dark gap/shadow in angled view |
| `lifted_stitch` | stitch | stitch not flattened; gap in angled view |
| `missing_wire` | all | bare pad and bare lead |
| `broken_wire` | loop | wire gap / loose end |
| `off_center_ball` | ball | ball not centred on pad (capillary wear) |
| `wire_short` | loop | two wires touching or crossing |
| `sagging_loop` | loop | low loop touching die edge (angled view) |

- **Defect injection modes**: `random` (each wire, each unit, independent probability) and
  `repetitive` (a chosen wire position / class fails in a chosen fraction of consecutive units,
  simulating capillary wear or a bad bond program).
- **Output**: images + COCO-format bounding boxes (one annotation file per view; COCO is what
  YOLOX, D-FINE and RT-DETR all train on) + a JSON sidecar per unit (product id, unit id,
  wire index, class, severity parameters). Seeded and reproducible.

### 7.2 Training (`training/`)

- A Jupyter notebook runnable on Kaggle/Colab free GPU: generate data → train a small single-stage
  detector → evaluate → export ONNX (+ quantised INT8 variant for the Pi).
- Detector family: **permissively licensed only** (Apache-2.0/MIT) so the repo stays MIT and
  industry can adopt it. Primary: YOLOX-Tiny/Nano (Apache-2.0). Fallback: D-FINE-N or
  RT-DETRv2-r18 (Apache-2.0, via Hugging Face transformers). Final pick at M2 by accuracy and
  measured Pi 5 latency. **No Ultralytics (AGPL).**
- Metrics: mAP50 and per-class precision/recall on a held-out synthetic set; confusion matrix.

### 7.3 Inspector runtime (`inspector/`)

- **Camera sources** (plugins, chosen per camera in `config.yaml`):
  - `csi` — Pi cameras via libcamera/Picamera2
  - `usb` — any UVC camera/USB microscope via V4L2/OpenCV
  - `rtsp` / `http` — network cameras
  - `folder` — watches a directory of images (used for development with synthetic data)
- **Trigger**: stop-and-shoot. A unit is inspected when the UI "Inspect" button is pressed, an
  HTTP API call arrives, or (folder source) a new image set appears. All cameras capture, each
  image is run through the detector, findings are merged into one per-unit result.
- **Detector**: ONNX Runtime CPU. Target ≤ 1 s per unit (2 images) on Pi 5.
- **Wire mapping**: each finding is assigned to a wire index by nearest wire in the product layout
  (layout file per product), so defects are counted per position.
- **Repetitive tracker**: same wire index + same class in ≥ K of the last M units → "repetitive"
  alert (defaults K=3, M=10; configurable). Everything else is "random".
- **Web UI**: live view, Inspect button, last result with boxes overlaid, unit history, a
  per-wire-position heat map, repetitive alerts, camera list from config. Light/dark mode.
- **Storage**: SQLite + image files in a mounted volume.
- **Config**: `config.yaml` lists products, cameras (type, device, view = top/angled,
  resolution), model path, thresholds, bind address. Adding a camera = adding an entry.

### 7.4 Docker

- One multi-arch image built with `docker buildx` (arm64 + amd64). `docker compose up` runs it.
- CSI on the Pi requires passing `/dev/video*`, `/dev/media*`, `/dev/dma_heap`, `/run/udev`
  (read-only); the compose file has a Pi profile for this. USB cameras: `/dev/video*` only.
- Model file and data directory are volumes, so models can be swapped without rebuilding.

## 8. Repository layout

The existing `wirebond-ai` repo holds both articles and code:

```
articles/  linkedin/       (writing — later)
synth/                     generator package + CLI
training/                  notebook + export script
inspector/                 runtime app (sources, detector, tracker, web)
docker/                    Dockerfile, compose files, Pi profile
docs/design/               this document
samples/                   a few generated example images for the README
```

## 9. Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Generator | Renders both views, all 8 classes, both injection modes; sample gallery looks plausible |
| M2 | Model | Trained on free GPU; mAP50 ≥ 0.85 on held-out synthetic; ONNX exported |
| M3 | Inspector (folder source) | Runs in Docker on the PC **and** the Pi, inspects synthetic units, UI shows results + repetitive alerts — ✅ 2026-11-11 (Pi 0.65 s/unit, 2 cameras) |
| M4 | Real cameras | Two CSI cameras via config on the Pi; capture → inspect works end to end |
| M5 | Reality check | Model evaluated on public real wire-bond images; gap reported honestly |

## 10. Calls made by Claude (please check)

- Generator is **2D procedural** (NumPy/OpenCV), not 3D rendering — fast, dependency-light,
  runs anywhere; 3D can be added later if the realism gap is too large.
- **Bounding-box detection** for v1 (not segmentation or keypoints).
- Capture + inference in **one container** with camera plugins, not one container per camera —
  simpler; multiple machines/stations scale by running more instances.
- Repetitive rule defaults K=3 of M=10.
- Success target mAP50 ≥ 0.85 on synthetic data (a sanity target, not a real-world claim).
- UI on 127.0.0.1 by default.
- "Done" for v1 = M1–M3 (works with no hardware); M4–M5 follow when cameras arrive.

## 11. Decisions log

- **Licence (2026-11-08, author):** code MIT, text CC BY 4.0; permissive-licence detector only
  (option B), no AGPL dependencies.
- **Model (2026-11-10/11):** D-FINE Nano (Apache-2.0) trained on PC CPU, not cloud GPU. The deployed
  model is M2c: synthetic + real NTUST wire-bond photos (CC BY 4.0), fine-tuned on 640 px tiles.
  The inspector tiles only pictures larger than one tile after shrinking by 0.5; synthetic 640 px
  frames are seen whole, as in training. The exported ONNX has a fixed batch of 1.
- **Speed (2026-11-11):** Pi 5 runs 0.20 s per 640 px picture (0.65 s per unit with 2 cameras), so
  the ≤ 1 s/unit target is met. A 2048×1365 real photo takes 1.23 s (6 tiles), so it is NOT met
  for large real photos; reported, not hidden.
- **Cameras (2026-11-11):** M3 ships `generator` (live synthetic units, both views of the same unit),
  `folder` and `opencv` (USB/RTSP/HTTP in one plugin). `csi` moves to M4 with the hardware.
  "Live view" in M3 = the latest captured frames; continuous video preview comes in M4.
- **Wire mapping (2026-11-11):** a finding goes to the wire whose 3D loop outline best matches the box
  (overlap + share of wire inside + centre distance), about 99% correct in the top view and 92%
  in the angled view on true synthetic boxes. No layout: an 8×8 grid cell.
- **Repetitive tracker (2026-11-11):** a (position, class) counts once per unit even if both views see
  it. When every camera fails, no unit is stored.

## 12. Left to the builder

Rendering details and parameter ranges, detector size (nano/small), image resolution, UI layout,
database schema, test strategy.
