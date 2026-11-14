# Inspector v2, step 2: online demo (visitor mode) — ready to publish, not published

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A "visitor mode" of the inspector that anyone can try in a browser with no install: drop a
photo or click a sample, see what the AI finds; nothing is stored. Packaged as a Hugging Face Space
folder, built locally. Publishing it is a separate step that needs Avi's account and explicit yes.

**Architecture:** One config switch `demo: true`. In demo mode the app needs no cameras, uses an
in-memory SQLite, serves only the Check-a-photo page, forces `check_photos(save=False)` and refuses
every endpoint that would write or read stored units. Sample pictures come from our own synthetic
generator (`samples/*.png`), so no third-party licence is involved. `tools/build_space.py` assembles
`out/space/` (Space README with metadata, Dockerfile on port 7860, demo config, model files, code).

**Tech Stack:** existing inspector (FastAPI, ONNX Runtime), Docker, Hugging Face Spaces (Docker SDK).

**Spec:** `docs/superpowers/specs/2026-11-13-inspector-v2-design.md` (goal 13, anti-goals "the demo
saves visitors' uploads", "third-party photos in the demo's sample set").

## Global Constraints

- Demo never writes a visitor's photo to disk or database (no `images/` files, no unit rows).
- Sample set = our own synthetic images only (`samples/*.png`), credited to this project.
- Same upload limits as the station: 1–20 files, 30 MB, 50 Mpx; uploads one at a time.
- Model: M3k-small (`wirebond_dfine_small_m3k.onnx` + `.json`); CREDITS.md travels with it in the Space.
- No publishing, account creation or push in this plan.
- Page shows a plain note: "Demo: nothing you upload is saved. Research prototype, not for production decisions."

## Review Focus

1. Visitor uploads a photo → nothing appears under `data_dir/images/` and the database stays empty. Test in Task 1.
2. A visitor calls `/api/inspect`, `/api/units`, decisions or images directly → 403 "not available in the demo". Test in Task 1.
3. Config with `demo: true` and no cameras loads; without `demo` it still requires a camera. Test in Task 1.
4. Sample click works with no upload (server reads the sample by name; names outside the sample list → 404, no path traversal). Test in Task 1.
5. Space folder contains the model, CREDITS.md and no secrets / third-party photos. Test in Task 3.

---

### Task 1: Demo mode in config, service and API

**Files:** Modify `config.py`, `service.py`, `web/app.py`; Test `tests/test_demo.py`.

**Interfaces:**
- `Config.demo: bool = False`; `Config.samples: Path | None = None` (YAML `samples:` folder, default none).
- In demo mode `load_config` allows `cameras: []`, forces `database` = `sqlite://` (in memory).
- `Inspector.demo -> bool`. `GET /api/status` adds `"demo": bool`.
- `GET /api/samples` → `[{"name": "wire_sweep.png", "url": "/api/samples/wire_sweep.png"}]` (demo only); `GET /api/samples/{name}` serves the file (name must be in the list).
- `POST /api/check` in demo: always `save=False`; also accepts form field `sample=<name>` instead of files.
- Demo: `/api/inspect`, `/api/units*`, `/api/images/*`, `/api/heatmap`, `/api/alerts` → 403 `"not available in the demo"`.

- [ ] Step 1: failing tests in `tests/test_demo.py` for Review Focus 1–4.
- [ ] Step 2: run → FAIL. Step 3: implement. Step 4: full suite PASS.
- [ ] Step 5: commit `Inspector v2: demo (visitor) mode`.

### Task 2: Demo page

**Files:** Modify `static/index.html`, `app.js`, `app.css`.

- In demo mode (`status.demo`): hide Inspect/Review/Reports/Setup tabs, operator chip, rail cells
  except Model; route always `#check`; show the demo note bar; show a "Try a sample" strip of sample
  thumbnails (click = check that sample); result cards open into a large viewer with boxes and the
  findings list (no decision buttons, since `id` is null).
- [ ] Step 1: build. Step 2: look at it in Chrome with `demo: true` (sample click, upload, bad file).
- [ ] Step 3: suite PASS; commit `Inspector v2: demo page`.

### Task 3: Hugging Face Space folder

**Files:** Create `demo/space/README.md` (Space front matter: `sdk: docker`, `app_port: 7860`, title,
licence MIT, short plain description + credits link), `demo/space/Dockerfile`, `demo/space/config.yaml`,
`tools/build_space.py` (copies synth/, inspector/, samples/*.png, model .onnx/.json, CREDITS.md,
LICENSE into `out/space/`, refuses if any file under `real/` or a `.env` would be included);
Test `tools/test_build_space.py` or an inspector test that runs the builder into tmp and checks
contents (model present, CREDITS.md present, no `.env`, no `real/`).
- [ ] Step 1: failing test. Step 2: implement. Step 3: build `out/space`, run it locally with
  `wbinspect --config out/space/config.yaml` (or docker if available) and check in Chrome.
- [ ] Step 4: commit `Demo: Hugging Face Space folder builder`.
