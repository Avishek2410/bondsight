# M1 Synthetic Wire-Bond Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pip-installable Python package `wirebond_synth` with a CLI that renders labelled
synthetic wire-bond images (top-down and angled views, 8 defect classes, random and repetitive
defect injection) and writes a COCO dataset.

**Architecture:** A product *layout* (die, pads, leads, wires) is generated once per product seed.
For each *unit*, a defect planner decides which wires carry which defects; geometry turns each
wire (plus its defect) into 2D polylines per view; a renderer draws them with OpenCV and emits
bounding boxes for defects visible in that view; augmentation adds lighting/blur/noise; the CLI
writes images, COCO annotation files and a JSON sidecar per unit.

**Tech Stack:** Python 3.11, NumPy, OpenCV (`opencv-python-headless`), pytest. No other runtime deps.

**Spec:** `docs/design/2026-11-08-inspection-system-design.md` (§7.1 is the part this plan implements)

## Global Constraints

- Licence: code MIT. **No AGPL dependencies** (no Ultralytics).
- Runtime deps limited to `numpy`, `opencv-python-headless`. Must run on Windows x86, Linux arm64 (Pi 5), Colab.
- Fully deterministic: same seeds + same args → byte-identical images and labels.
- Class list and order are fixed (COCO category ids = index + 1):
  `wire_sweep, lifted_ball, lifted_stitch, missing_wire, broken_wire, off_center_ball, wire_short, sagging_loop`
- Views: `top`, `angled`. Default image size 640×640, 3-channel uint8 BGR.
- Injection modes: `random` (independent per wire per unit, probability `p`) and `repetitive` (one wire index + class, fails in a given fraction of units from a start unit).
- Repo root: `C:\Users\AVI-TECH\projects\wirebond-ai`; package lives in `synth/`.

## Review Focus

1. **Boxes must stay inside the image and have positive area** even for wires near the image edge or heavily swept wires — clip to `[0, size-1]`, drop boxes < 2 px wide/high. Test: `test_boxes_inside_image_for_all_classes` (Task 4).
2. **Two defects on one unit/wire** (random mode with high `p`, or repetitive + random overlapping) — only one defect per wire per unit; repetitive wins over random. Test: `test_repetitive_overrides_random_on_same_wire` (Task 2).
3. **`wire_short` needs a neighbour** — wire 0/last wire and single-wire layouts must not crash; the short bends toward the nearer existing neighbour. Test: `test_wire_short_on_edge_wire` (Task 3).
4. **Class visible only in one view** (`lifted_stitch`, `sagging_loop` angled-only) — no box emitted in the top view, and the unit is still written. Test: `test_view_visibility_table` (Task 4).
5. **Re-running into an existing output dir** — CLI refuses unless `--overwrite`, so a dataset is never silently mixed. Test: `test_cli_refuses_existing_dir` (Task 6).

---

## File Structure

```
synth/
  pyproject.toml                    package metadata, deps, console script `wbsynth`
  wirebond_synth/
    __init__.py                     version + re-exports
    classes.py                      CLASSES, VIEWS, VISIBILITY table
    layout.py                       Layout/Pad/Lead/Wire dataclasses, make_layout()
    defects.py                      DefectSpec, InjectionConfig, plan_unit_defects()
    geometry.py                     wire_path() per view, incl. defect deformation
    render.py                       render_unit() → image + boxes
    augment.py                      augment() domain randomisation
    dataset.py                      DatasetWriter: images, COCO json, unit sidecars
    cli.py                          argparse CLI `wbsynth generate`, `wbsynth gallery`
    __main__.py                     python -m wirebond_synth
  tests/
    test_layout.py test_defects.py test_geometry.py test_render.py
    test_augment.py test_dataset_cli.py
.gitignore  LICENSE  (repo root)
samples/                            gallery PNGs committed for README
```

---

### Task 1: Repo + package skeleton + class table

**Files:**
- Create: `.gitignore`, `LICENSE` (MIT, "Dr. Avishek Mondal", 2026), `synth/pyproject.toml`, `synth/wirebond_synth/__init__.py`, `synth/wirebond_synth/classes.py`
- Test: `synth/tests/test_classes.py`

**Interfaces:**
- Produces: `CLASSES: tuple[str, ...]` (8 names, order above); `VIEWS = ("top", "angled")`;
  `VISIBILITY: dict[str, frozenset[str]]` — `lifted_stitch` and `sagging_loop` → `{"angled"}`,
  all others → `{"top","angled"}`; `class_id(name: str) -> int` (1-based COCO id).

- [ ] **Step 1:** `git init` in repo root (if not already), create `.gitignore` (`__pycache__/`, `*.egg-info/`, `.venv/`, `out/`, `.pytest_cache/`), `LICENSE`.
- [ ] **Step 2:** Create venv `synth/.venv` and `pyproject.toml` (setuptools, name `wirebond-synth`, version `0.1.0`, `requires-python >=3.10`, deps `numpy>=1.24`, `opencv-python-headless>=4.8`, optional `[test]` = `pytest`, script `wbsynth = wirebond_synth.cli:main`). `pip install -e synth[test]`.
- [ ] **Step 3: Write failing test**
```python
from wirebond_synth.classes import CLASSES, VIEWS, VISIBILITY, class_id
def test_class_table():
    assert CLASSES == ("wire_sweep","lifted_ball","lifted_stitch","missing_wire",
                       "broken_wire","off_center_ball","wire_short","sagging_loop")
    assert class_id("wire_sweep") == 1 and class_id("sagging_loop") == 8
    assert VISIBILITY["sagging_loop"] == {"angled"} and VISIBILITY["lifted_stitch"] == {"angled"}
    assert all(VISIBILITY[c] == set(VIEWS) for c in CLASSES if c not in ("sagging_loop","lifted_stitch"))
```
- [ ] **Step 4:** Run `pytest synth/tests -q` → FAIL (module missing). Implement `classes.py`. Run → PASS.
- [ ] **Step 5:** Commit `chore: package skeleton and defect class table`.

### Task 2: Layout + defect planner

**Files:**
- Create: `synth/wirebond_synth/layout.py`, `synth/wirebond_synth/defects.py`
- Test: `synth/tests/test_layout.py`, `synth/tests/test_defects.py`

**Interfaces:**
- Produces (layout.py), all coordinates in a normalised top-view plane, pixels at `size`:
  - `@dataclass(frozen=True) Pad(x: float, y: float, w: float)`; `Lead(x, y, w, h: float)`
  - `@dataclass(frozen=True) Wire(index: int, pad: Pad, lead: Lead, loop_height: float, diameter_px: float)`
  - `@dataclass(frozen=True) Layout(size: int, die: tuple[float,float,float,float], wires: tuple[Wire, ...], seed: int)` with `to_dict() -> dict` / `from_dict(d) -> Layout`
  - `make_layout(n_wires: int, seed: int, size: int = 640) -> Layout` — die centred, 30–45 % of image width (seeded); pads evenly spaced on the four die edges just inside the die; one lead per pad on a ring outside the die, radially outward; `loop_height` 40–70 px, `diameter_px` 2.0–3.5.
- Produces (defects.py):
  - `@dataclass(frozen=True) DefectSpec(wire: int, cls: str, params: dict)`
  - `@dataclass(frozen=True) InjectionConfig(mode: str, p: float = 0.0, wire: int | None = None, cls: str | None = None, fraction: float = 1.0, start_unit: int = 0, random_p: float = 0.0)` — `mode` ∈ `{"none","random","repetitive"}`; repetitive may also add background random defects via `random_p`.
  - `plan_unit_defects(layout: Layout, unit: int, cfg: InjectionConfig, seed: int) -> list[DefectSpec]` — rng = `np.random.default_rng([seed, unit])`. Params per class drawn here (e.g. `wire_sweep: amplitude_px` 6–25 and `side` ±1; `off_center_ball: dx, dy` 0.35–0.6 × pad w; `broken_wire: t` 0.25–0.75 gap position; `sagging_loop: height_factor` 0.1–0.3; `lifted_ball/lifted_stitch: gap_px` 2–5).

- [ ] **Step 1: Failing tests**
```python
def test_layout_counts_and_determinism():
    a, b = make_layout(32, seed=1), make_layout(32, seed=1)
    assert a == b and len(a.wires) == 32
    x0,y0,x1,y1 = a.die
    for w in a.wires:
        assert x0 <= w.pad.x <= x1 and y0 <= w.pad.y <= y1          # pad on die
        assert not (x0 <= w.lead.x <= x1 and y0 <= w.lead.y <= y1)  # lead off die
    assert Layout.from_dict(a.to_dict()) == a
def test_layout_different_seed_differs():
    assert make_layout(32, 1) != make_layout(32, 2)
def test_none_mode_no_defects(): assert plan_unit_defects(L, 0, InjectionConfig("none"), 7) == []
def test_random_p1_every_wire_once():
    d = plan_unit_defects(L, 0, InjectionConfig("random", p=1.0), 7)
    assert sorted(x.wire for x in d) == list(range(len(L.wires)))
def test_repetitive_fraction():
    cfg = InjectionConfig("repetitive", wire=5, cls="wire_sweep", fraction=0.8, start_unit=10)
    hits = [any(x.wire==5 and x.cls=="wire_sweep" for x in plan_unit_defects(L,u,cfg,7)) for u in range(200)]
    assert not any(hits[:10]) and 0.7 <= sum(hits[10:])/190 <= 0.9
def test_repetitive_overrides_random_on_same_wire():
    cfg = InjectionConfig("repetitive", wire=5, cls="wire_sweep", fraction=1.0, random_p=1.0)
    d = plan_unit_defects(L, 0, cfg, 7)
    assert [x.cls for x in d if x.wire==5] == ["wire_sweep"] and len({x.wire for x in d}) == len(d)
def test_planner_deterministic():
    cfg = InjectionConfig("random", p=0.3)
    assert plan_unit_defects(L,3,cfg,7) == plan_unit_defects(L,3,cfg,7)
```
(`L = make_layout(16, seed=1)` as module fixture.)
- [ ] **Step 2:** Run → FAIL. Implement both modules. Run → PASS.
- [ ] **Step 3:** Commit `feat(synth): product layout and defect planner`.

### Task 3: Geometry — wire paths per view

**Files:**
- Create: `synth/wirebond_synth/geometry.py`
- Test: `synth/tests/test_geometry.py`

**Interfaces:**
- Consumes: `Layout`, `Wire`, `DefectSpec`.
- Produces:
  - `@dataclass WirePath(segments: list[np.ndarray], ball: tuple[float,float] | None, ball_r: float, stitch: tuple[float,float] | None, stitch_len: float, lift_gap: float, lifted: str | None)` — `segments` is a list of (N,2) float polylines (2 segments when broken, 0 when missing).
  - `wire_path(layout: Layout, wire: Wire, defect: DefectSpec | None, view: str) -> WirePath`
  - `project(x: float, y: float, z: float, view: str, size: int) -> tuple[float, float]` — `top`: `(x, y)`; `angled` (45° tilt about the image x-axis, centred): `(x, cy + (y-cy)*cos45 - z*sin45)`.
- Algorithm (not determined by tests, so stated): 3D loop sampled at 64 points along t∈[0,1] from pad to lead; height `z(t)` = `loop_height * (sin(pi*min(t/0.3,1)/2)` rising fast near the ball, then a parabolic descent to the stitch (classic "reverse loop" shape: steep up, long slope down). `wire_sweep` adds a lateral offset perpendicular to the pad→lead line: `side*amplitude*sin(pi*t)`. `sagging_loop` multiplies z by `height_factor`. `wire_short` adds a lateral bow toward the nearer neighbour (wire index ±1; for edge wires the only neighbour) with amplitude = 0.9 × distance to that neighbour's midpoint. `broken_wire` removes points in `[t-0.04, t+0.04]`. `missing_wire` → no segments, ball and stitch `None`. `off_center_ball` shifts ball (and wire start) by `(dx,dy)`.

- [ ] **Step 1: Failing tests**
```python
def max_dev(seg, a, b):  # max perpendicular distance of points from line a-b
    ...
def test_good_wire_top_is_straight():
    p = wire_path(L, L.wires[3], None, "top")
    assert len(p.segments)==1 and max_dev(p.segments[0], p.segments[0][0], p.segments[0][-1]) < 1.0
def test_sweep_amplitude_top():
    d = DefectSpec(3, "wire_sweep", {"amplitude_px": 20.0, "side": 1})
    s = wire_path(L, L.wires[3], d, "top").segments[0]
    assert 18.0 <= max_dev(s, s[0], s[-1]) <= 22.0
def test_broken_has_two_segments():
    assert len(wire_path(L, L.wires[3], DefectSpec(3,"broken_wire",{"t":0.5}), "top").segments) == 2
def test_missing_has_nothing():
    p = wire_path(L, L.wires[3], DefectSpec(3,"missing_wire",{}), "top")
    assert p.segments == [] and p.ball is None and p.stitch is None
def test_sagging_lower_in_angled():
    good = wire_path(L, L.wires[3], None, "angled").segments[0]
    sag = wire_path(L, L.wires[3], DefectSpec(3,"sagging_loop",{"height_factor":0.2}), "angled").segments[0]
    # loop rises toward smaller y in angled view; sag rises less
    assert (good[:,1].min() + 10) < sag[:,1].min()
def test_wire_short_on_edge_wire():
    for i in (0, len(L.wires)-1):
        p = wire_path(L, L.wires[i], DefectSpec(i,"wire_short",{}), "top")
        assert len(p.segments) == 1
```
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS.
- [ ] **Step 3:** Commit `feat(synth): 3D loop geometry and per-view projection`.

### Task 4: Renderer + boxes

**Files:**
- Create: `synth/wirebond_synth/render.py`
- Test: `synth/tests/test_render.py`

**Interfaces:**
- Consumes: `Layout`, `DefectSpec`, `wire_path`, `project`, `VISIBILITY`, `class_id`.
- Produces:
  - `@dataclass(frozen=True) Box(cls: str, x0: float, y0: float, x1: float, y1: float, wire: int)`
  - `render_unit(layout: Layout, defects: list[DefectSpec], view: str, seed: int) -> tuple[np.ndarray, list[Box]]` — returns (size,size,3) uint8 and boxes only for defects whose class is visible in `view`.
- Look (stated so the gallery is plausible): dark-grey substrate with faint texture; die = mid-grey rectangle with darker border and fine circuit-like noise; pads = light aluminium squares; leads = copper/gold-tinted rectangles; wire = gold anti-aliased polyline `diameter_px` thick with a 1-px lighter highlight line; ball = bright filled circle (r ≈ 0.75 × pad w) with a darker rim; stitch = flattened ellipse ("fish-tail") at the lead. Angled view: ball drawn as ellipse, and for `lifted_ball`/`lifted_stitch` a dark gap band `gap_px` tall under the ball/stitch plus a 1-px bright edge; in top view `lifted_ball` = ball 15 % dimmer and slightly blurred.
- Box rule: ball classes → ball bbox padded 4 px; stitch class → stitch bbox padded 4 px; loop classes (`wire_sweep`, `broken_wire`, `wire_short`, `sagging_loop`) → bbox of all wire segments padded 4 px; `missing_wire` → bbox spanning pad and lead centres padded 6 px. Clip to image; drop boxes < 2 px in either dimension.

- [ ] **Step 1: Failing tests**
```python
def test_good_unit_no_boxes():
    img, boxes = render_unit(L, [], "top", seed=1)
    assert img.shape == (640,640,3) and img.dtype == np.uint8 and boxes == []
def test_each_class_box_in_visible_views():   # also test_view_visibility_table
    for c in CLASSES:
        d = plan_one(c)   # helper: a valid DefectSpec for class c on wire 4
        for v in VIEWS:
            _, boxes = render_unit(L, [d], v, seed=1)
            assert ([b.cls for b in boxes] == [c]) == (v in VISIBILITY[c])
def test_boxes_inside_image_for_all_classes():
    big = make_layout(64, seed=9)
    for c in CLASSES:
        for w in (0, 31, 63):
            for v in VIEWS:
                for b in render_unit(big, [plan_one(c, w, extreme=True)], v, 1)[1]:
                    assert 0 <= b.x0 < b.x1 <= 639 and 0 <= b.y0 < b.y1 <= 639
                    assert b.x1-b.x0 >= 2 and b.y1-b.y0 >= 2
def test_render_deterministic():
    a = render_unit(L, [], "angled", 5)[0]; b = render_unit(L, [], "angled", 5)[0]
    assert np.array_equal(a, b)
def test_defect_changes_pixels():
    good = render_unit(L, [], "top", 1)[0]
    bad = render_unit(L, [DefectSpec(4,"wire_sweep",{"amplitude_px":20,"side":1})], "top", 1)[0]
    assert np.abs(good.astype(int)-bad.astype(int)).sum() > 0
```
(`plan_one(c, w=4, extreme=False)` lives in `tests/conftest.py`; `extreme=True` uses the max param of each range.)
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS.
- [ ] **Step 3:** Commit `feat(synth): renderer with per-view defect boxes`.

### Task 5: Augmentation

**Files:**
- Create: `synth/wirebond_synth/augment.py`
- Test: `synth/tests/test_augment.py`

**Interfaces:**
- Produces: `augment(img: np.ndarray, boxes: list[Box], seed: int, strength: float = 1.0) -> tuple[np.ndarray, list[Box]]` — illumination gradient (±25 %), vignetting, gaussian blur σ 0–1.2, gaussian noise σ 0–6, colour/white-balance jitter ±8 %, brightness/contrast ±15 %, translation ±12 px and rotation ±3° (boxes transformed as the bbox of their rotated corners, then clipped/dropped per Task 4 rule). `strength=0` → returns inputs unchanged.

- [ ] **Step 1: Failing tests**
```python
def test_strength_zero_identity():
    img, boxes = render_unit(L, [plan_one("wire_sweep")], "top", 1)
    out, ob = augment(img, boxes, seed=3, strength=0.0)
    assert np.array_equal(out, img) and ob == boxes
def test_augment_keeps_boxes_valid_and_deterministic():
    img, boxes = render_unit(L, [plan_one("broken_wire", 0)], "top", 1)
    a = augment(img, boxes, 3); b = augment(img, boxes, 3)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    for x in a[1]: assert 0 <= x.x0 < x.x1 <= 639 and 0 <= x.y0 < x.y1 <= 639
def test_augment_changes_image():
    img, boxes = render_unit(L, [], "top", 1)
    assert not np.array_equal(augment(img, boxes, 3)[0], img)
```
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS.
- [ ] **Step 3:** Commit `feat(synth): domain-randomisation augmentation`.

### Task 6: Dataset writer + CLI + gallery

**Files:**
- Create: `synth/wirebond_synth/dataset.py`, `synth/wirebond_synth/cli.py`, `synth/wirebond_synth/__main__.py`
- Test: `synth/tests/test_dataset_cli.py`

**Interfaces:**
- Consumes everything above.
- Produces:
  - `generate(out: Path, n_units: int, n_wires: int, product_seed: int, cfg: InjectionConfig, views: tuple[str,...] = VIEWS, size: int = 640, augment_strength: float = 1.0, overwrite: bool = False) -> dict` (returns summary `{"units": n, "boxes": {view: {cls: count}}}`); raises `FileExistsError` if `out` exists and is non-empty and not `overwrite`.
  - Output layout:
    ```
    out/layout.json                       Layout.to_dict()
    out/config.json                       all generate() args (cfg as dict)
    out/images/{view}/u{unit:05d}.png
    out/annotations/{view}.json           COCO: images, annotations (bbox [x,y,w,h], area, iscrowd 0, attributes {"wire": i}), categories id 1..8
    out/units/u{unit:05d}.json            {"unit", "defects": [DefectSpec as dict]}
    ```
  - Per-unit seed for render/augment: `hash`-free, `seed = product_seed * 1_000_003 + unit` (deterministic across platforms).
  - CLI: `wbsynth generate --out DIR --units N [--wires 32] [--product-seed 1] [--mode none|random|repetitive] [--p 0.03] [--wire W --cls C --fraction 0.8 --start-unit 0 --random-p 0.0] [--views top,angled] [--size 640] [--augment 1.0] [--overwrite]`; prints the summary. `wbsynth gallery --out DIR [--product-seed 1]` writes, for each class, one unaugmented top|angled side-by-side PNG `DIR/{cls}.png` plus `DIR/good.png`.

- [ ] **Step 1: Failing tests**
```python
def test_generate_writes_coco(tmp_path):
    s = generate(tmp_path/"d", 6, 16, 1, InjectionConfig("random", p=0.2))
    for v in VIEWS:
        coco = json.loads((tmp_path/"d/annotations"/f"{v}.json").read_text())
        assert len(coco["images"]) == 6 and len(coco["categories"]) == 8
        for a in coco["annotations"]:
            x,y,w,h = a["bbox"]; assert w >= 2 and h >= 2 and 1 <= a["category_id"] <= 8
    assert len(list((tmp_path/"d/units").glob("*.json"))) == 6
def test_generate_deterministic(tmp_path):
    for n in ("a","b"): generate(tmp_path/n, 3, 16, 4, InjectionConfig("random", p=0.3))
    assert (tmp_path/"a/images/top/u00002.png").read_bytes() == (tmp_path/"b/images/top/u00002.png").read_bytes()
def test_cli_refuses_existing_dir(tmp_path):
    generate(tmp_path/"d", 1, 8, 1, InjectionConfig("none"))
    with pytest.raises(FileExistsError): generate(tmp_path/"d", 1, 8, 1, InjectionConfig("none"))
    generate(tmp_path/"d", 1, 8, 1, InjectionConfig("none"), overwrite=True)
def test_cli_main_runs(tmp_path):
    assert main(["generate","--out",str(tmp_path/"c"),"--units","2","--wires","8",
                 "--mode","repetitive","--wire","2","--cls","wire_sweep"]) == 0
```
- [ ] **Step 2:** Run → FAIL. Implement. Run full suite `pytest synth/tests -q` → all PASS.
- [ ] **Step 3: Visual check.** Run `python -m wirebond_synth gallery --out samples` and `wbsynth generate --out out/demo --units 20 --mode random --p 0.05`. Open `samples/*.png`; confirm each class is recognisable to a packaging engineer (wires bow for sweep, gap under ball in angled view for lifted ball, etc.). Fix rendering in Task 4 code if not, re-run tests.
- [ ] **Step 4:** Commit `feat(synth): dataset writer, CLI and sample gallery` (include `samples/`).

---

## Next plans (after M1 is approved and done)

- `docs/plans/…-m2-training.md` — Kaggle/Colab notebook, YOLOX-Tiny (fallback D-FINE-N), ONNX + INT8 export, Pi latency benchmark.
- `docs/plans/…-m3-inspector.md` — Docker runtime, camera plugins (folder first), repetitive tracker, web UI.
