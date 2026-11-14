# M3 Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Docker app that takes images from any number of configured cameras, runs the wire-bond
detector on them, stores each inspected unit, raises "repetitive defect" alerts and shows it all
in a web page. It runs on the PC and on the Pi 5 with simulated cameras (no hardware needed).

**Architecture:** New package `inspector/wirebond_inspector`. The pieces are:
- camera plugins (`folder`, `generator`, `opencv` for USB/RTSP/HTTP)
- an ONNX Runtime detector with tiling, with no torch at runtime
- a wire-position mapper
- an SQLite store
- a repetitive tracker
- a FastAPI web app with one static HTML page

`config.yaml` wires it all together. The model and data are Docker volumes.

**Tech Stack:** Python 3.11, numpy, Pillow, onnxruntime, PyYAML, FastAPI + uvicorn, opencv-python-headless, `wirebond-synth` (in-repo, for the generator camera), pytest. Docker image `python:3.11-slim`, multi-arch through buildx.

**Spec:** `docs/design/2026-11-08-inspection-system-design.md` §7.3, §7.4, M3 row of §9.

**Changes since the spec (decided 2026-11-10/11, record them in its decisions log in Task 9):**
- The model is now **M2c** (`training/runs/local_v1/wirebond_dfine_nano_tiled.onnx` + `.json`). It was fine-tuned on real NTUST tiles.
- The model's json has a `tiling` block, so the detector must tile when that block is present.
- Measured Pi 5 speed: 1.23 s per 2048×1365 photo tiled, 0.21 s whole. So the spec's "≤ 1 s per unit (2 images)" is **not met** for tiled real photos; this is reported, not hidden.
- Simulated cameras: `folder` plus a new `generator` source, which renders synthetic units live.
- The `csi` plugin moves to M4 (no cameras attached yet).
- `usb`, `rtsp` and `http` are one `opencv` plugin, because OpenCV opens all three.
- In M3, the UI's "live view" means the latest captured frames of each camera. A continuous video preview comes with real cameras in M4.

## Global Constraints

- Licence: code MIT; **no AGPL dependencies** (no Ultralytics); no paid services or GPU needed to use it.
- No torch or transformers in the inspector image. Inference is onnxruntime only, on the CPU.
- Web UI binds to **127.0.0.1 by default**. LAN access is an explicit opt-in in config. In Docker, the host port is published as `127.0.0.1:8080:8080` unless the config/compose opts in.
- One multi-arch image (linux/arm64 + linux/amd64), started with `docker compose up`. The model file and the data dir are volumes, so swapping models needs no rebuild.
- Every flagged defect shows **where** (a box on the image) and **what** (class + score). No unexplained output.
- Never present synthetic accuracy as real-factory accuracy. The UI footer names the model and states it is a research prototype.
- Repetitive rule defaults: **K=3 of the last M=10 units**, configurable.
- Images are converted to 8-bit RGB on read. Grayscale, RGBA, palette and 16-bit inputs must all work (the reality check crashed on exactly this).
- Learner notes: every module starts with a plain-English docstring saying what it does and why. `inspector/README.md` explains the flow in plain words for someone with very basic Python.
- Class order, which must match the model: `wire_sweep, lifted_ball, lifted_stitch, missing_wire, broken_wire, off_center_ball, wire_short, sagging_loop`. Read it from the model json; never hard-code a second copy.
- Third-party images are never committed. Tests use synthetic or generated images only.

## Review Focus

1. **A camera fails mid-run** (folder empty, USB unplugged, RTSP down). The unit is still recorded, with that camera marked `error: <reason>`; other cameras are still inspected; the UI shows the camera as offline; the app never crashes. *Test: Task 6 `test_inspect_survives_failing_camera`.*
2. **Inspect is pressed again while an inspection is running** (Pi is ~1.2 s per image). The second request gets HTTP 409 "busy", and no half-written unit is stored. *Test: Task 7 `test_inspect_busy_returns_409`.*
3. **Odd image modes and sizes** (grayscale, RGBA, 16-bit PNG, images smaller than a tile, a 2592×1944 OV5647 frame). Detection runs, and the boxes are in original-image pixels inside the image bounds. *Test: Task 2 `test_detect_handles_modes_and_sizes`.*
4. **Restart.** After the app restarts, history and the tracker state come back from SQLite, so an alert that needed 3 of 10 units still fires on the next failing unit. *Test: Task 5 `test_tracker_rebuilds_from_store`.*
5. **Model or config mistakes** (missing model file, json without `classes`, camera with an unknown `type`, duplicate camera names). Startup stops with one clear message naming the file or key. *Test: Task 1 `test_config_errors_are_clear`.*

---

## File Structure

```
inspector/
  pyproject.toml                  package wirebond-inspector, entry point `wbinspect`
  README.md                       plain-English "how it works" + run instructions
  config.example.yaml             2 generator cameras (top + angled), repetitive defect on
  wirebond_inspector/
    __init__.py
    __main__.py                   `python -m wirebond_inspector --config config.yaml`
    config.py                     load + validate config.yaml -> dataclasses
    imageio.py                    to_rgb8(any PIL image) -> RGB 8-bit
    detector.py                   OnnxDetector: preprocess, tiling, decode, NMS
    sources/__init__.py           make_camera(cfg) registry
    sources/base.py               Camera protocol, Frame, CameraError
    sources/folder.py             next unread image from a directory
    sources/generator.py          renders synthetic unit N live (wirebond-synth)
    sources/opencv_cam.py         USB index / rtsp:// / http:// via cv2.VideoCapture
    wiremap.py                    finding -> wire index (layout) or grid cell (no layout)
    store.py                      SQLite: units, shots, findings; image files on disk
    tracker.py                    repetitive rule K of last M
    service.py                    Inspector: capture all cams -> detect -> map -> store -> track
    web/app.py                    FastAPI routes
    web/static/index.html         single page UI (vanilla JS, light/dark)
  tests/
    conftest.py                   tiny fake ONNX model fixture, tmp config
    test_config.py  test_imageio.py  test_detector.py  test_sources.py
    test_wiremap.py test_store_tracker.py test_service.py test_web.py
    test_parity_hf.py             (skips unless transformers + real model present)
docker/
  Dockerfile                      python:3.11-slim, installs synth + inspector
  compose.yaml                    default + `pi` profile (devices for M4)
```

The data dir (volume) holds `inspector.db` and `images/<unit_id>/<camera>.jpg` (JPEG quality 90).

---

### Task 1: Package skeleton + config

**Files:** Create `inspector/pyproject.toml`, `inspector/wirebond_inspector/{__init__,config}.py`, `inspector/config.example.yaml`, `inspector/tests/{conftest,test_config}.py`. Venv: `py -3.11 -m venv inspector/.venv`, then `pip install -e synth -e inspector[test]`.

**Interfaces — Produces:**
```python
@dataclass(frozen=True) class CameraCfg: name: str; type: str; view: str; options: dict
@dataclass(frozen=True) class TrackerCfg: k: int = 3; m: int = 10
@dataclass(frozen=True) class TriggerCfg: mode: str = "manual"; seconds: float = 5.0   # manual | interval
@dataclass(frozen=True) class Config:
    model: Path; threshold: float; cameras: tuple[CameraCfg, ...]; data_dir: Path
    layout: Path | None; tracker: TrackerCfg; trigger: TriggerCfg; bind: str; port: int
class ConfigError(ValueError)
def load_config(path: Path) -> Config
```
Defaults: `threshold 0.4`, `bind "127.0.0.1"`, `port 8080`. `type` is one of `folder | generator | opencv`, and `view` is one of `top | angled`. Relative paths are resolved against the config file's folder.

- [ ] **Step 1: Write the failing tests.** `test_load_example_config` checks that loading `config.example.yaml` gives 2 cameras with views `{"top","angled"}`, `bind == "127.0.0.1"` and `tracker == TrackerCfg(3, 10)`. `test_config_errors_are_clear` is parametrized over: model path missing, camera `type: "csi2"`, two cameras both named `top`, and `threshold: 1.5`. Each case must raise `ConfigError`, and the message must contain the offending key or path.
- [ ] **Step 2:** Run `inspector/.venv/Scripts/python -m pytest inspector/tests/test_config.py -q`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `load_config` with `yaml.safe_load`. Collect every problem and raise one `ConfigError`, one line per problem. The model must exist. Its `.json` (same stem) must contain `classes` and `img_size`; otherwise raise a ConfigError naming the json.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(inspector): package skeleton and validated config`.

### Task 2: Image normalising + ONNX detector with tiling

**Files:** Create `wirebond_inspector/{imageio,detector}.py` and `tests/{test_imageio,test_detector,test_parity_hf}.py`. Extend `conftest.py` with a fixture `fake_model(tmp_path)`. The fixture builds, with the `onnx` helper API (a test-only dependency), a model whose input `pixel_values [N,3,640,640]` maps to constant outputs:
- `logits [N,300,8]`: query 0 is class 3 with logit +4, everything else is −9
- `pred_boxes [N,300,4]`: query 0 = cx .5, cy .5, w .25, h .25

Write the `.json` next to it, with tiling on.

**Interfaces — Produces:**
```python
def to_rgb8(img: PIL.Image.Image) -> PIL.Image.Image          # imageio.py
@dataclass(frozen=True) class Finding: cls: str; score: float; box: tuple[float, float, float, float]  # x0,y0,x1,y1 original px
class OnnxDetector:
    def __init__(self, model: Path, threads: int = 4)          # reads <model>.json: classes, img_size, tiling?
    classes: tuple[str, ...]; tiled: bool; name: str
    def detect(self, img: PIL.Image.Image, threshold: float) -> list[Finding]
def tile_offsets(w: int, h: int, tile: int, stride: int) -> list[tuple[int, int]]
```
The steps must match `tools/reality_check.py` and the training notebook, because that is how the model was trained:
- **Preprocess:** bilinear resize to `img_size × img_size`, scale by 1/255, CHW float32, no mean/std normalisation (the RTDetr processor config has `do_normalize: false`).
- **Decode:** this is RT-DETR focal decoding. Take `sigmoid(logits)`, flatten queries×classes, keep the top-300 scores, label = idx % C, query = idx // C. Convert boxes from cxcywh (normalised) to xyxy times the target size. Keep score ≥ threshold.
- **Tiling** (when `tiling` is in the json):
  - Scale the image by `scale` with LANCZOS.
  - Get the offsets from `tile_offsets` (same `positions()` rule as `reality_check.py`).
  - Crop each tile with PIL `crop`; it pads beyond the edge with black.
  - Run tiles in batches of 4, decode each with target 640×640, shift each tile's boxes by its offset, then divide by `scale`.
  - Merge everything with per-class NMS at IoU `nms_iou`, written in numpy.
  - Clip the final boxes to the image bounds.

- [ ] **Step 1: Write the failing tests:**
  - `test_to_rgb8_modes` (L, RGBA, P, I;16 → mode "RGB", dtype uint8)
  - `test_tile_offsets` (`tile_offsets(1024,683,640,320) == [(0,0),(320,0),(384,0),(0,43),(320,43),(384,43)]`; `tile_offsets(300,200,640,320) == [(0,0)]`)
  - `test_decode_whole` (fake model with tiling off, 800×600 image → one Finding cls `missing_wire`, box ≈ (300,225,500,375) ±1)
  - `test_tiled_nms_merges_duplicates` (tiling on: every tile reports a box at its own centre, so the result has no two same-class boxes with IoU > 0.5)
  - `test_detect_handles_modes_and_sizes` (L 300×200, RGBA 2592×1944, I;16 640×640: no exception; all boxes inside bounds)
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement `imageio.py` and `detector.py`. Set `SessionOptions.intra_op_num_threads = threads`.
- [ ] **Step 4: Parity test.** `test_parity_hf` is skipped unless `transformers` imports and `training/runs/local_v1/best_model_tiled` exists. On 2 synthetic `samples/` images, `OnnxDetector(...tiled.onnx).detect(img, 0.4)` must equal `tools/reality_check.Detector(best_model_tiled, tiled=True)(img, 0.4)`: same classes, and boxes within 2 px. Run it with `tools/.venv/Scripts/python -m pytest inspector/tests/test_parity_hf.py` (tools venv has torch).
- [ ] **Step 5:** All detector tests PASS, and parity PASSES under tools/.venv.
- [ ] **Step 6:** Commit: `feat(inspector): onnx detector with tiling, torch-free`.

### Task 3: Camera plugins

**Files:** Create `sources/{__init__,base,folder,generator,opencv_cam}.py` and `tests/test_sources.py`.

**Interfaces — Consumes:** `CameraCfg`, `to_rgb8`. **Produces:**
```python
class CameraError(RuntimeError)
@dataclass(frozen=True) class Frame: camera: str; view: str; image: PIL.Image.Image; source: str   # file path / url / "gen:u00012"
class Camera(Protocol):
    name: str; view: str
    def capture(self, unit: int) -> Frame        # raises CameraError
    def status(self) -> dict                     # {"name","type","view","ok": bool,"detail": str}
    def layout(self) -> dict | None              # generator only: Layout.to_dict(); others None
def make_camera(cfg: CameraCfg) -> Camera       # registry by cfg.type; unknown -> ConfigError
```
How each plugin behaves:
- **`folder`**: options `path`, `pattern` (default `*.jpg,*.png`). It returns the next file in name order that it hasn't returned before. It keeps state in memory, and `start_after` can be passed in to resume after a restart. When no files are left it raises `CameraError("no new image")`.
- **`generator`**: options `product_seed`, `n_wires` (default 24), `size` (640), plus `injection`, a dict passed to `InjectionConfig(**...)`. It renders unit `unit` with `plan_unit_defects` + `render_unit` + `augment`, using the same seeds as `dataset.generate`. That means two generator cameras with the same `product_seed` and different views show **the same unit**, and their defects agree.
- **`opencv`**: option `source` (int index or URL), plus optional `width`/`height`. It opens lazily, grabs one frame, converts BGR→RGB, and raises `CameraError` on a failed open or read. It reopens on the next capture.

- [ ] **Step 1: Write the failing tests:**
  - `test_folder_returns_files_in_order_then_errors` (3 PNGs in tmp → 3 frames in order, then `CameraError`)
  - `test_generator_views_agree` (top and angled cams, same seed, repetitive `wire: 5, cls: "wire_sweep", fraction: 1` → the generator's own planned defects for unit 0 include wire 5 for both; images are 640×640 RGB)
  - `test_generator_layout_roundtrip` (`Layout.from_dict(cam.layout())` works)
  - `test_opencv_bad_source_raises` (`source: "rtsp://127.0.0.1:1/x"` → `CameraError` within 10 s)
  - `test_make_camera_unknown_type`
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. The generator imports `wirebond_synth`. Add `planned_defects(unit) -> list[DefectSpec]` on the generator only, so the UI can show "truth" for simulated units.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(inspector): folder, generator and opencv camera plugins`.

### Task 4: Wire-position mapping

**Files:** Create `wiremap.py` and `tests/test_wiremap.py`.

**Interfaces — Produces:**
```python
def position_key(f: Finding, view: str, img_size: tuple[int, int], layout: dict | None) -> str
```
- **With a layout:** return `"wire:<index>"`, for the wire whose ball→stitch segment (pad centre → lead centre) is nearest the box centre. For the `angled` view, first project both end points with `wirebond_synth.geometry.project(x, y, 0, "angled", size)`. Scale by `img_size / layout["size"]` when the image is not the layout size.
- **Without a layout:** return `"cell:r<row>c<col>"` on an 8×8 grid over the image, so repeated defects at the same spot are still caught for real cameras.

- [ ] **Step 1: Write the failing tests:**
  - `test_maps_box_on_wire_to_its_index` (take `make_layout(24, seed=7)`; a 10×10 box centred on the midpoint of wire 5's segment → `"wire:5"` in top view)
  - `test_angled_projection` (the same wire, projected → `"wire:5"`)
  - `test_grid_without_layout` (box centre (790, 10) in 800×800 → `"cell:r0c7"`)
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement with point-to-segment distance in numpy.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(inspector): map findings to wire positions`.

### Task 5: SQLite store + repetitive tracker

**Files:** Create `store.py`, `tracker.py` and `tests/test_store_tracker.py`.

**Interfaces — Produces:**
```python
class Store:
    def __init__(self, data_dir: Path)                          # creates inspector.db + images/
    def add_unit(self, unit_no: int, shots: list[dict], findings: list[dict], ts: float) -> int
        # shots: {"camera","view","source","image": PIL|None,"error": str|None}; saved as images/<id>/<camera>.jpg
        # findings: {"camera","cls","score","box":[4],"position":str}; one transaction (all or nothing)
    def unit(self, unit_id: int) -> dict | None                 # {"id","unit_no","ts","shots":[...],"findings":[...],"alerts":[...]}
    def units(self, limit: int = 50, before: int | None = None) -> list[dict]   # newest first, summary + n_findings
    def recent_unit_findings(self, m: int) -> list[set[tuple[str, str]]]        # last m units, oldest first, {(position, cls)}
    def add_alerts(self, unit_id: int, alerts: list[dict]) -> None
    def heatmap(self, last: int = 200) -> dict[str, dict[str, int]]             # position -> cls -> count
    def next_unit_no(self) -> int                               # max(unit_no)+1, 0 when empty
class RepetitiveTracker:
    def __init__(self, k: int = 3, m: int = 10)
    def load(self, history: list[set[tuple[str, str]]]) -> None
    def update(self, unit_findings: set[tuple[str, str]]) -> list[dict]
        # appends, keeps last m; returns [{"position","cls","count","window": m}] for pairs with count >= k that appear in THIS unit
```
Schema: `units(id PK, unit_no, ts)`, `shots(unit_id, camera, view, source, image_path, error)`, `findings(unit_id, camera, cls, score, x0, y0, x1, y1, position)`, `alerts(unit_id, position, cls, count, window)`. A finding appearing in both views of the same unit counts **once** for that unit.

- [ ] **Step 1: Write the failing tests:**
  - `test_add_and_read_unit` (round-trip; the image file exists; a shot with `error` and no image is stored)
  - `test_add_unit_is_atomic` (a finding with a bad box raises, and `units()` stays empty)
  - `test_tracker_k_of_m` (wire:5/wire_sweep in units 0, 4, 8 with K=3, M=10 → alert on unit 8 only; at unit 14 the unit-4 hit has left the window, so no alert)
  - `test_random_defects_do_not_alert` (10 units with different positions → no alerts)
  - `test_tracker_rebuilds_from_store` (store 2 hits, make a new Store + Tracker, `load(store.recent_unit_findings(10))`, the 3rd hit alerts)
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement with the `sqlite3` stdlib, WAL mode, and one connection guarded by a `threading.Lock`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(inspector): sqlite store and repetitive-defect tracker`.

### Task 6: Inspector service

**Files:** Create `service.py` and `tests/test_service.py`.

**Interfaces — Consumes:** Tasks 1–5. **Produces:**
```python
class Busy(RuntimeError)
class Inspector:
    def __init__(self, cfg: Config, detector: OnnxDetector | None = None)   # builds cameras, store, tracker (loaded from store)
    def inspect(self) -> dict            # one unit; raises Busy if one is running; returns store.unit(id) + {"timing_s": {...}}
    def cameras(self) -> list[dict]      # status() of each
    def start_trigger(self) -> None      # interval mode: background thread calls inspect every cfg.trigger.seconds;
                                         # a unit where every camera raised "no new image" is NOT stored (idle, not an error)
    def stop(self) -> None
    model_name: str
```
How `inspect()` works:
- It takes a non-blocking lock; if the lock is held, it raises `Busy`.
- It uses `unit_no = store.next_unit_no()`.
- It captures every camera, one after another, catching `CameraError` per camera.
- For each frame, it runs `detector.detect`, then `position_key` using the layout from the first camera that has one (else the config `layout` file if given, else None). The angled and top findings of the same wire then share `"wire:N"`.
- Then it calls `store.add_unit`, `tracker.update`, `store.add_alerts`, and records how long each camera's detection took.

- [ ] **Step 1: Write the failing tests** (fake model, generator cameras):
  - `test_inspect_stores_unit_with_both_views` (2 shots, unit_no 0 then 1)
  - `test_inspect_survives_failing_camera` (generator + folder on an empty tmp dir → unit stored; the folder shot has `error == "no new image"`; the generator shot has findings)
  - `test_repetitive_alert_end_to_end` (fake detector returning a box on wire 5 for every image; K=3 → alert on the third inspect)
  - `test_interval_idle_when_all_folders_empty` (start/stop the trigger with 0.05 s: no units stored, no exception)
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(inspector): inspection service with per-camera fault isolation`.

### Task 7: Web API + UI

**Files:** Create `web/app.py`, `web/static/index.html`, `__main__.py` and `tests/test_web.py`.

**Interfaces — Consumes:** `Inspector`. **Produces:** `create_app(inspector: Inspector) -> FastAPI`, and the CLI `wbinspect --config PATH [--host H] [--port P]`. The CLI defaults come from the config (`127.0.0.1:8080`); it starts the trigger and runs uvicorn.

Routes, all JSON except `/` and images:
- `GET /` serves the page.
- `GET /api/status` → `{model, threshold, cameras:[status], tracker:{k,m}, units_total}`
- `POST /api/inspect` → the unit dict. It returns **409 `{"detail":"busy"}`** while one is running, and 503 if every camera errored.
- `GET /api/units?limit&before`
- `GET /api/units/{id}` (404 if missing)
- `GET /api/images/{unit_id}/{camera}.jpg`: only paths inside `data_dir/images`; anything else → 404
- `GET /api/heatmap`
- `GET /api/alerts?limit=20`

The page uses vanilla JS with no build step and no external CDN, so it works offline on the Pi. Light/dark comes from `prefers-color-scheme` plus a toggle. Parts of the page:
- camera list with online/offline dots
- **Inspect** button, disabled while a request is running
- the latest unit's images with boxes drawn on a `<canvas>`, each labelled `class score`
- the unit history list (click to view)
- the alerts panel
- a heat map: a table of position × class counts, coloured by count
- a footer showing the model name and "Research prototype — trained mostly on synthetic data; not validated for production use."

The page polls `/api/units?limit=1` every 3 s when interval mode is on.

- [ ] **Step 1: Write the failing tests** (FastAPI `TestClient`, fake model):
  - `test_status`
  - `test_inspect_then_fetch_unit_and_image` (200, image/jpeg)
  - `test_inspect_busy_returns_409` (hold `inspector._lock` and POST → 409)
  - `test_image_path_traversal_blocked` (`/api/images/1/..%2F..%2Finspector.db` → 404)
  - `test_index_served`
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the routes and the page. Follow the frontend-design guidance for the page; it is a small, dense operator screen, not a marketing page.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then run `inspector/.venv/Scripts/wbinspect --config inspector/config.example.yaml` with the real M2c model, press Inspect 5 times in Chrome, and take a screenshot of the boxes, history and heat map. With the example config's repetitive wire, an alert appears by the 3rd–5th unit, provided the model detects it. Record what actually happened, not what was expected.
- [ ] **Step 5:** Commit: `feat(inspector): web api and operator page`.

### Task 8: Docker on PC and Pi

**Files:** Create `docker/Dockerfile`, `docker/compose.yaml`, `.dockerignore`.

- **Dockerfile:** `python:3.11-slim`; apt installs `libgl1 libglib2.0-0` (opencv); copy `synth/` and `inspector/`; `pip install --no-cache-dir ./synth ./inspector`; non-root user `inspector`; `EXPOSE 8080`; `CMD ["wbinspect","--config","/config/config.yaml","--host","0.0.0.0"]`. Binding to 0.0.0.0 *inside* the container is safe because the compose file publishes to `127.0.0.1` only.
- **compose.yaml:**
  - Service `inspector` with volumes `./models:/models:ro`, `./data:/data` and `./config.yaml:/config/config.yaml:ro`.
  - Ports `"127.0.0.1:8080:8080"`.
  - A `pi` profile service that extends it with `devices: [/dev/video0]` (placeholder for M4) and a comment explaining the CSI devices from spec §7.4.
- `.dockerignore` excludes `real/ out/ training/runs/ **/.venv **/__pycache__ samples/`.

- [ ] **Step 1:** Start Docker Desktop on the PC and run `docker build -f docker/Dockerfile -t wirebond-inspector:dev .`. Expected: success, and the image is under 900 MB (`docker images`).
- [ ] **Step 2:** Run compose on the PC with `models/` holding the tiled onnx + json and the example config (model path `/models/...`, data dir `/data`). Check that `curl -s 127.0.0.1:8080/api/status` lists 2 cameras, and that `curl -X POST 127.0.0.1:8080/api/inspect` returns a unit.
- [ ] **Step 3: Pi.** Ask Avi before installing Docker on the Pi; it is a system change. Then run `sudo apt install docker.io docker-compose` (Debian packages; no curl-pipe-sh). Build `linux/arm64` **on the Pi itself** from a copy of the repo without the excluded dirs, using rsync over ssh (`ssh avi24` → `192.168.1.100` if mDNS fails). Run compose with the same smoke checks over an ssh tunnel (`ssh -L 8081:127.0.0.1:8080 avi24`). Record the per-unit `timing_s` from 5 inspects.
- [ ] **Step 4:** Buildx multi-arch push is **not** done in M3 (no registry yet, and publishing needs Avi's confirmation). Document `docker buildx build --platform linux/amd64,linux/arm64` in the README as the later step.
- [ ] **Step 5:** Commit: `feat(docker): inspector image and compose (pc + pi profile)`.

### Task 9: Docs for the learner + spec update

**Files:** Create `inspector/README.md`. Modify `docs/design/2026-11-08-inspection-system-design.md` (§9 M3 row status; §11 decisions log with the changes listed at the top of this plan) and the root `README.md` (one "Run the inspector" section).

- [ ] **Step 1:** Write `inspector/README.md`:
  - a "what happens when you press Inspect" walk-through in 6 plain steps
  - how to add a camera (copy a block in config.yaml)
  - how to swap the model (drop a new `.onnx` + `.json` in `models/`)
  - the measured PC and Pi timings from Tasks 7–8
  - an honest-results section with the real-photo numbers (56% of NTUST defects found on held-out photos, 30% false alarms on good wires, fails on MEMS packages it has never seen), written as plain numbers with no third-party images embedded
- [ ] **Step 2:** Run the full suite: `inspector/.venv/Scripts/python -m pytest inspector/tests -q`, then `synth/.venv/Scripts/python -m pytest synth/tests -q`. Expected: all pass (parity skipped in this venv).
- [ ] **Step 3:** Commit: `docs(inspector): learner walkthrough, spec decisions log`. **No push**; pushing and publishing need Avi's confirmation.
