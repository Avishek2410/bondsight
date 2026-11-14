# Inspector v2, step 1: industrial UI, Check a photo, decisions, Review, PostgreSQL

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the inspector into an industrial-looking three-page app (Inspect, Check a photo,
Review) where every unit gets a verdict, operators record decisions, uploaded photos are inspected
like camera pictures, decided units export as COCO training data, and the data lives in PostgreSQL
(SQLite still works for tests and the later demo).

**Architecture:** `store.py` is rewritten on SQLAlchemy Core (one code path for SQLite and
PostgreSQL) with Alembic migrations run at start-up; the public `Store` methods the service uses
keep their names. `service.py` gains `check_photos()` for uploads (shares detector + checker, never
touches line numbering or the repetitive tracker). FastAPI gains upload/decision/filter endpoints.
The static page is rebuilt framework-free (HTML + one CSS + one JS file) following
`docs/design/ui-industrial-mockup.html`. Docker compose gains a `postgres` service.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.x Core, Alembic, psycopg 3 (binary), python-multipart,
ONNX Runtime, Pillow, plain HTML/CSS/JS, PostgreSQL 16 (Docker `postgres:16-alpine`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-inspector-v2-design.md` (approved 2026-11-13).
Visual reference: `docs/design/ui-industrial-mockup.html`.

**Not in this plan (later steps of the spec):** online demo / visitor mode page, GitHub launch,
camera scan + setup wizard + Pi ribbon-camera driver, Reports, CSV export, products/batches, PIN
roles, audit log, triggers beyond the current ones, retention, central PostgreSQL + offline buffer
+ picture sync. Reports and Setup tabs appear in the header **disabled** with the tooltip text
"Coming in a later version".

## Global Constraints

- Runs on Raspberry Pi 5 (8 GB, CPU only) and Windows/Linux PCs; Docker image stays multi-arch (amd64 + arm64).
- Code MIT; no AGPL dependency. New deps allowed: `sqlalchemy>=2.0`, `alembic>=1.13`, `psycopg[binary]>=3.1`, `python-multipart>=0.0.9`.
- Detector (`detector.py`) and checker (`anomaly.py`) code unchanged: they are verified against training.
- No secret in git, logs, API responses or the page: the database password comes only from env `WBI_DB_PASSWORD` (Docker: `docker/.env`, git-ignored, generated). `/api/status` shows the database URL **without** password.
- Pictures stay files under `data_dir/images/`; the database stores relative paths.
- Visual rules (spec "Visual direction"): light machine-panel palette, colour only for state (DEFECT red `#D2232A`, CHECK amber `#E08A00`, OK green `#1E8A4C`), Archivo condensed, stack light, hazard band on DEFECT, push-buttons that press, INSPECT NEXT yellow `#F2C200`; touch targets ≥ 44 px; no hover-only or right-click actions; `prefers-reduced-motion` respected; no external JS libraries (Google Fonts CSS link allowed, with system-font fallback for offline Pis).
- Plain-language "What / Why" docstring at the top of every new module, like the existing ones.
- Existing v1 SQLite data is test data: v2 uses a new file `inspector2.db`; no migration from v1.
- All existing tests keep passing (adapted where an API is intentionally changed); run with `inspector/.venv/Scripts/python -m pytest -q` from `inspector/`.

## Review Focus

1. **Phone photo with EXIF rotation** → boxes and saved picture must be upright as the person sees it. Test in Task 3.
2. **One bad file in an upload batch** (not an image, empty, over 30 MB) → that file gets its own error line, the others are still checked. Test in Task 3 and Task 4.
3. **Upload while the line is inspecting** → upload works, line unit numbers and the repetitive tracker are untouched. Test in Task 3.
4. **Deciding twice, or on a finding of another unit** → latest decision wins and history is kept; a foreign finding id is refused with 400. Test in Task 2.
5. **PostgreSQL down or wrong password at start** → `wbinspect` exits with code 2 and one plain sentence naming the host, never the password. Test in Task 1.

---

### Task 1: Database layer on SQLAlchemy + Alembic, PostgreSQL or SQLite

**Files:**
- Create: `inspector/wirebond_inspector/db.py` (engine from config, table definitions, migrate on start)
- Create: `inspector/wirebond_inspector/migrations/env.py`, `migrations/script.py.mako`, `migrations/versions/0001_v2_schema.py`
- Modify: `inspector/wirebond_inspector/config.py` (new `database` setting), `store.py` (rewrite over SQLAlchemy), `__main__.py` (clear DB error), `pyproject.toml` (deps + package data `migrations/*`, `migrations/versions/*`)
- Modify: `inspector/tests/conftest.py` (fixture `db_url`), `tests/test_store_tracker.py`, `tests/test_config.py`
- Create: `inspector/tests/test_db.py`

**Interfaces:**
- Produces: `Config.database: str` (SQLAlchemy URL without password; default `sqlite:///<data_dir>/inspector2.db`).
- Produces: `db.make_engine(url: str) -> sqlalchemy.Engine` (adds `WBI_DB_PASSWORD` to a postgresql URL; `WBI_DATABASE_URL` env overrides the config URL entirely), `db.migrate(engine) -> None` (Alembic upgrade head), `db.safe_url(url: str) -> str` (password masked), table objects `db.units, db.shots, db.findings, db.alerts, db.decisions`.
- Produces: `Store(data_dir: Path, url: str | None = None, classes: tuple[str, ...] | None = None)` (`Inspector` passes `cfg.database` and `detector.classes`); unchanged method names `add_unit, add_alerts, unit, units, count, recent_unit_findings, heatmap, alerts, next_unit_no`.
- Produces: `class DatabaseUnavailable(RuntimeError)` in `db.py`, raised by `Store.__init__` with message `cannot reach the database at <host>/<dbname>: <short reason>`.

Schema (migration 0001), all ids integer autoincrement primary keys:
- `units(id, unit_no INTEGER NULL, ts FLOAT NOT NULL, source TEXT NOT NULL DEFAULT 'camera', verdict TEXT NOT NULL, label TEXT NULL)`; `source ∈ {camera, upload}`, `verdict ∈ {OK, DEFECT, CHECK}`, `label` = uploaded file name (uploads) or NULL.
- `shots(id, unit_id FK, camera, view, source, image_path, error, anomaly FLOAT, anomaly_flag BOOL, heat_path, width INT, height INT, hints TEXT)` (`hints` = JSON list of strings).
- `findings(id, unit_id FK, camera, cls, score, x0, y0, x1, y1, position)`.
- `alerts(id, unit_id FK, position, cls, count, window)`.
- `decisions(id, unit_id FK, finding_id FK NULL, verdict TEXT NOT NULL, camera TEXT NULL, cls TEXT NULL, x0, y0, x1, y1 NULL, operator TEXT NOT NULL, ts FLOAT NOT NULL)`; `verdict ∈ {real, false_alarm, missed, confirm_ok}`.
- Indexes on every `unit_id`, on `units.verdict`, `units.source`.

- [ ] **Step 1: Write failing tests** in `tests/test_db.py` and adapt `tests/test_store_tracker.py` to run against a `db_url` fixture parametrised over `["sqlite", "postgres"]` (postgres param skipped unless env `WBI_TEST_PG_URL` is set; each test gets a fresh schema by dropping all tables first):

```python
def test_safe_url_hides_password():
    assert db.safe_url("postgresql+psycopg://wb:secret@pi:5432/wb") == "postgresql+psycopg://wb:***@pi:5432/wb"

def test_config_rejects_password_in_url(example_dir):   # database: postgresql+psycopg://wb:pw@h/db
    ... ConfigError message contains "put the password in WBI_DB_PASSWORD"

def test_default_database_is_sqlite_in_data_dir(example_dir):
    assert cfg.database == f"sqlite:///{(example_dir / 'data' / 'inspector2.db').as_posix()}"

def test_migrate_twice_is_harmless(db_url): Store(tmp, db_url); Store(tmp, db_url)  # no error

def test_unreachable_postgres_gives_plain_error(tmp_path):
    with pytest.raises(db.DatabaseUnavailable) as e:
        Store(tmp_path, "postgresql+psycopg://wb@127.0.0.1:1/wb")
    assert "127.0.0.1" in str(e.value) and "secret" not in str(e.value)   # WBI_DB_PASSWORD=secret set via monkeypatch

def test_main_exits_2_when_database_unreachable(...)   # __main__.main(["--config", cfg]) == 2, stderr one line
```

- [ ] **Step 2: Run** `python -m pytest -q tests/test_db.py tests/test_store_tracker.py` → FAIL (no module `db`).
- [ ] **Step 3: Implement** `db.py`, the migration, the `database` config key (YAML `database:`; validation: must start with `sqlite:` or `postgresql`; reject a URL whose password part is non-empty), and rewrite `Store` on SQLAlchemy Core. SQLite: `connect_args={"check_same_thread": False}`, `PRAGMA journal_mode=WAL` on connect, keep the existing `threading.Lock` around writes. Connection check in `Store.__init__`: `engine.connect()` with a 5 s timeout (`connect_timeout=5` for psycopg), wrap `sqlalchemy.exc.OperationalError` into `DatabaseUnavailable`. `migrate()` uses Alembic `Config` with `script_location` = the package's `migrations` folder and the engine's connection (no `alembic.ini` file). `__main__` catches `DatabaseUnavailable` → print `wbinspect: <message>` to stderr, return 2.
- [ ] **Step 4: Run** the full suite → PASS. Then with Docker: `docker run -d --name wbpg -e POSTGRES_PASSWORD=t -e POSTGRES_USER=wb -e POSTGRES_DB=wbtest -p 127.0.0.1:55432:5432 postgres:16-alpine`, `WBI_TEST_PG_URL=postgresql+psycopg://wb@127.0.0.1:55432/wbtest WBI_DB_PASSWORD=t python -m pytest -q` → postgres params PASS; `docker rm -f wbpg`.
- [ ] **Step 5: Commit** `Inspector v2: SQLAlchemy store, Alembic schema, PostgreSQL or SQLite`.

### Task 2: Verdicts and operator decisions in the store

**Files:**
- Modify: `inspector/wirebond_inspector/store.py`
- Create: `inspector/wirebond_inspector/verdict.py`
- Test: `inspector/tests/test_decisions.py`

**Interfaces:**
- Consumes: Task 1 tables.
- Produces: `verdict.unit_verdict(findings: list[dict], shots: list[dict]) -> str` → `"DEFECT"` if any finding; else `"CHECK"` if any shot has `anomaly_flag` true; else `"OK"`.
- Produces: `Store.add_unit(unit_no: int | None, shots, findings, ts, source: str = "camera", label: str | None = None) -> int` (stores verdict; shots dicts may carry `width`, `height`, `hints`).
- Produces: `Store.add_decision(unit_id: int, verdict: str, operator: str, finding_id: int | None = None, camera: str | None = None, cls: str | None = None, box: list[float] | None = None, ts: float | None = None) -> int`, raising `ValueError` with a plain message on any invalid input.
- Produces: `Store.unit(id)` now returns also `verdict, source, label`, findings with `id` and `decision` (latest verdict or `None`), unit-level `decisions` (all, newest first), and `review_state ∈ {"undecided","decided"}`.
- Produces: `Store.units(limit=50, before=None, verdict=None, source=None, undecided=False)` rows with `verdict, source, label, review_state` added.
- Produces: `Store.decided_units() -> Iterator[dict]` (full `unit()` dicts with `review_state == "decided"`, oldest first) for the export.

Rules: `review_state` is `"decided"` when every finding has at least one `real`/`false_alarm` decision **and** (the unit has findings **or** has a `confirm_ok` or `missed` decision). `real`/`false_alarm` need a `finding_id` of this unit. `missed` needs `camera` (a shot of this unit), `cls` (one of the detector classes, the `classes` given to `Store`; `None` skips the check) and a 4-number box with x0<x1, y0<y1, inside the shot's width/height. `confirm_ok` takes no finding/box and is refused when the unit has findings with no `false_alarm` decision. `operator` 1–60 characters after stripping. Upload units (`unit_no None`) are excluded from `next_unit_no`, `recent_unit_findings` and `heatmap`.

- [ ] **Step 1: Write failing tests** `test_decisions.py` (over `db_url`): verdict rule (3 cases); `add_decision` happy paths for all four verdicts; `test_decide_twice_latest_wins` (real then false_alarm → finding `decision == "false_alarm"`, two rows in `decisions`); `test_foreign_finding_refused` (finding of unit B on unit A → `ValueError` "does not belong to unit"); missed box outside image → `ValueError`; empty operator → `ValueError`; `confirm_ok` on a unit with an undecided finding → `ValueError`; `units(undecided=True)` lists only undecided; uploads don't change `next_unit_no()`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** in `verdict.py` and `store.py`.
- [ ] **Step 4: Run** full suite → PASS (sqlite; postgres if env set).
- [ ] **Step 5: Commit** `Inspector v2: unit verdicts and operator decisions`.

### Task 3: Check a photo in the service

**Files:**
- Create: `inspector/wirebond_inspector/photocheck.py` (decode + quality hints)
- Modify: `inspector/wirebond_inspector/service.py`
- Test: `inspector/tests/test_photocheck.py`

**Interfaces:**
- Consumes: Task 2 `add_unit(..., source="upload", label=...)`, `Inspector.detector`, `Inspector.checker`.
- Produces: `photocheck.load_photo(data: bytes, max_bytes: int = 30_000_000) -> PIL.Image.Image` (applies `ImageOps.exif_transpose`, then `to_rgb8`; raises `PhotoError(str)` for empty, too big, or unreadable data with messages `"empty file"`, `"file is larger than 30 MB"`, `"not a picture the app can read (use JPG, PNG, BMP or TIFF)"`).
- Produces: `photocheck.quality_hints(img) -> list[str]`: `"blurry: the picture looks out of focus"` when variance of the Laplacian of the greyscale image (shrunk so the longer side is ≤ 1024 px) is < 30; `"small: the picture is under 400 px, wires may be too small to judge"` when `min(w, h) < 400`.
- Produces: `Inspector.check_photos(files: list[tuple[str, bytes]], camera: str | None = None, save: bool = True) -> list[dict]`, one result per file in order: on success the unit dict (as `unit()`; when `save=False`: same shape with `id: None`, findings without ids, image returned inline as `image_b64` JPEG) plus `hints`; on failure `{"label": name, "error": message}`. `camera` picks whose unusual-spot memory to use; when `None` or that camera has no memory → the unit's shot gets hint `"unusual-spot check not set up for this photo"` and no anomaly score. The shot's `camera` field is `camera or "upload"`, `view` = that camera's view or `"top"`.
- Uploads do **not** take the inspection lock and never call the tracker.

- [ ] **Step 1: Write failing tests**: EXIF-rotated JPEG (make a 600×400 image with a red square top-left, save with EXIF orientation 6) → loaded image is 400×600 and the red square is at top-right; `load_photo(b"")`, a 31 MB blob, and `b"not an image"` each raise the right message; blurry hint on a flat grey image, none on a sharp checkerboard; `check_photos` with `[("a.jpg", good), ("b.txt", b"x")]` → first result has `id` and `source == "upload"`, second `{"label": "b.txt", "error": ...}`; after an upload, `ins.store.next_unit_no()` and `ins.tracker` state unchanged and a following `ins.inspect()` gets `unit_no == 0`; `save=False` writes nothing to the database or `images/`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Reuse the per-camera detection/anomaly code by extracting `Inspector._analyse(image, camera_name, view, layout) -> (shot_extra: dict, findings: list[dict])` from `inspect()` and calling it from both paths (the extraction must leave `inspect()` behaviour and its tests unchanged).
- [ ] **Step 4: Run** full suite → PASS.
- [ ] **Step 5: Commit** `Inspector v2: check uploaded photos with the same AI`.

### Task 4: API for uploads, decisions, filters

**Files:**
- Modify: `inspector/wirebond_inspector/web/app.py`, `pyproject.toml` (python-multipart)
- Test: `inspector/tests/test_web.py`

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces endpoints (all JSON):
  - `POST /api/check` multipart `files` (1–20 files, else 400 `"send 1 to 20 photos"`), optional form field `camera` → list from `check_photos`. Runs in the thread pool.
  - `POST /api/units/{id}/decisions` body `{"verdict", "operator", "finding_id"?, "camera"?, "cls"?, "box"?}` → 201 `{"id": <decision id>, "unit": <unit dict>}`; `ValueError` → 400 with its message; unknown unit → 404.
  - `GET /api/units?verdict=&source=&undecided=1&limit=&before=` → filtered list.
  - `GET /api/status` adds `"database": safe_url(...)`, `"classes": [...]` and `"station": socket.gethostname()`.
  - `GET /api/images/{id}/{name}` unchanged (uploads use camera name `upload` → `upload.jpg`).

- [ ] **Step 1: Write failing tests** in `test_web.py`: upload two good JPEGs + one text file → 200, three results, third has `error`; 21 files → 400; decide `real` on the first finding → 201 and unit `review_state` changes as per the rules; bad box → 400 with message; `GET /api/units?undecided=1` excludes the decided unit; `/api/status["database"]` contains no password when `WBI_DB_PASSWORD` is set.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** full suite → PASS.
- [ ] **Step 5: Commit** `Inspector v2: API for photo checks, decisions and review filters`.

### Task 5: Training export

**Files:**
- Create: `inspector/wirebond_inspector/export.py`; script `wbinspect-export` in `pyproject.toml`
- Test: `inspector/tests/test_export.py`

**Interfaces:**
- Consumes: `Store.decided_units()`, `Config`.
- Produces: `export.export_coco(store: Store, classes: tuple[str, ...], out: Path) -> dict` returning `{"images": n, "boxes": m}`; CLI `wbinspect-export --config CFG --out DIR`.

Output: `DIR/images/<unit id>_<camera>.jpg` (copied) and `DIR/annotations.json` in COCO format with `categories` = model classes (ids 0..n-1, same order as the model json), one `image` per shot that has a picture, `annotations` = findings whose latest decision is `real` + `missed` boxes (`bbox` = `[x, y, w, h]`), false-alarm findings omitted (picture kept), `confirm_ok` units = pictures with no annotations. Undecided units are skipped. `info.description` names the station data dir and export time.

- [ ] **Step 1: Failing test**: one unit with finding decided `real`, one with a finding decided `false_alarm` plus a `missed` box, one `confirm_ok` unit, one undecided unit → 3 units × shots exported, annotations = 2 boxes with correct `[x, y, w, h]` and category ids, undecided unit absent; files exist.
- [ ] **Step 2–4:** run (FAIL), implement, run full suite (PASS).
- [ ] **Step 5: Commit** `Inspector v2: export reviewed units as COCO training data`.

### Task 6: Industrial UI shell and the Inspect page

**Files:**
- Replace: `inspector/wirebond_inspector/web/static/index.html`
- Create: `inspector/wirebond_inspector/web/static/app.css`, `static/app.js`; serve `/static/*` from `app.py` (StaticFiles)
- Test: `inspector/tests/test_web.py` (page serves, assets serve)

**Interfaces:**
- Consumes: API from Task 4 and existing endpoints.
- Produces: one page with hash routes `#inspect` (default), `#check`, `#review`; Reports/Setup tabs disabled. `app.js` exposes nothing global except `App.go(route)`.

Look and behaviour (match `docs/design/ui-industrial-mockup.html`; take its CSS tokens and components as the starting point):
- Header: nameplate (title "WB-INSPECT", second line = `station` from `/api/status`), tabs, status lamps per camera (green ok / red error from `/api/status`) + Database lamp, operator chip.
- Operator name: asked once in a small panel ("Who is inspecting?") and kept in `localStorage` (wrapped in try/catch; falls back to asking again). Required before any decision button works.
- Info rail: Model, Units (line units total), Defects / Check counts of the last 200 line units, Last cycle (from the last inspect's `timing_s.total`).
- Stack light: lamp for the latest unit's verdict lit (red/amber/green), others dim; lit lamp pulses for 3 s after a new result then stays on.
- Stage: real camera pictures. View buttons Top / Angled / 3D / Unusual. Top/Angled = that shot flat with finding boxes drawn on an overlay (SVG scaled to the image), labels as tags `W05 · WIRE SWEEP · 91%` (position `wire:5` → `W05`; grid positions shown as-is). 3D = both shots as panels in CSS perspective (rotateY ±18°, slight translateZ), tilting smoothly on switch (transition 600 ms). Unusual = heat picture if present, else the note "No unusual spots on this view". Measuring grid + crosshair are decorative overlays only on Top/Angled.
- Side panel: verdict card (DEFECT with hazard band / CHECK / OK) with unit number and time; readout rows (each finding with score, repeat alerts, unusual-spot result); push-buttons per the current unit: with findings → for each finding *Real defect* / *False alarm* (finding chosen by tapping its tag or row, first one pre-selected), *Missed something*; without findings → *Looks good* (`confirm_ok`) and *Missed something*. INSPECT NEXT (yellow) calls `POST /api/inspect`; 409 shows "Busy, one inspection is still running".
- *Missed something*: overlay on the current view; drag (pointer events, works with touch) to draw a box, pick the defect type from big buttons (classes from `/api/status`), Save.
- Check-a-photo slot on the side panel: drop or tap → goes to `#check` with the files.
- Bottom log strip: last 12 units (line + uploads, uploads marked "photo"), colour bar by verdict, tick when decided; tap opens it on the stage.
- Errors shown in a plain bar under the header, never as `alert()`.

- [ ] **Step 1: Failing tests**: `GET /` contains `id="tab-inspect"`, `id="tab-check"`, `id="tab-review"`; `GET /static/app.css` and `/static/app.js` → 200.
- [ ] **Step 2–4:** run (FAIL), build the page, run suite (PASS).
- [ ] **Step 5: Look at it**: start `wbinspect --config inspector/config.yaml`, open http://127.0.0.1:8080 in Chrome, press INSPECT NEXT 5×, check each view, decide on one unit, draw one missed box, check reduced motion and a 768 px wide window (tablet). Screenshot for Avi.
- [ ] **Step 6: Commit** `Inspector v2: industrial UI and Inspect page`.

### Task 7: Check a photo page and Review page

**Files:**
- Modify: `static/index.html`, `static/app.css`, `static/app.js`

Behaviour:
- `#check`: large drop slot + *Choose photos* + *Use camera* (`<input type="file" accept="image/*" capture="environment" multiple>`); optional select "Unusual-spot memory of camera: none / top / angled"; progress line "Checking 3 of 7…"; results as a list of cards (thumbnail with boxes, verdict tag, findings, hints in amber, error lines in red); tapping a card opens it on the Inspect stage with the decision buttons.
- `#review`: filter chips (All / Not decided / DEFECT / CHECK / OK / Photos / Line), list of units newest first with paging ("Load older" uses `before`), each row: verdict bar, unit number or file name, time, findings summary, decided tick; tapping opens it on the stage. Counter "N not decided" in the Review tab label.

- [ ] **Step 1:** Build both pages.
- [ ] **Step 2: Look at it** in Chrome: upload 3 NTUST photos + 1 text file + 1 phone-rotated photo (Review Focus 1–2), decide on two of them, filter Not decided, Load older with > 50 units.
- [ ] **Step 3:** Run full suite → PASS.
- [ ] **Step 4: Commit** `Inspector v2: Check a photo and Review pages`.

### Task 8: Docker with PostgreSQL, PC and Pi

**Files:**
- Modify: `docker/compose.yaml`, `docker/Dockerfile` (deps come via pip), `docker/config.docker.yaml` (`database: postgresql+psycopg://wb@postgres:5432/wirebond`), `.gitignore` (`docker/.env`, `docker/pgdata/`)
- Create: `docker/make_env.py` (writes `docker/.env` with `WBI_DB_PASSWORD=<secrets.token_urlsafe(24)>` and `POSTGRES_PASSWORD=` same value, only if the file does not exist; prints "created docker/.env" without the value)
- Modify: `inspector/README.md` (v2 pages, database, decisions, export, Docker steps), `README.md` (one line)

Compose: service `postgres` (`postgres:16-alpine`, `POSTGRES_USER=wb`, `POSTGRES_DB=wirebond`, `POSTGRES_PASSWORD` from `.env`, volume `./pgdata:/var/lib/postgresql/data`, healthcheck `pg_isready -U wb`, no published port); `inspector` gets `env_file: .env`, `depends_on: postgres: condition: service_healthy`, `restart: unless-stopped`.

- [ ] **Step 1:** Implement; `python docker/make_env.py` twice (second run says it exists, file unchanged).
- [ ] **Step 2: PC check**: `cd docker && docker compose up -d --build`; inspect 3 units via API, decide one; `docker compose restart inspector` → data still there; `docker compose logs inspector` contains no password.
- [ ] **Step 3: Pi** (`ssh <pi>`, with sudo): sync `inspector/` and `docker/` (not `docker/models`, not `docker/data`), `python3 docker/make_env.py`, keep the Pi's `config.yaml` cameras + model + anomaly lines and add the `database:` line, `docker compose up -d --build`, run 5 inspections + one upload via API on the Pi, report speed per unit vs 3.5 s before.
- [ ] **Step 4:** Run the full test suite once more (PASS) and `git status` shows no `.env` / `pgdata`.
- [ ] **Step 5: Commit** `Inspector v2: Docker with PostgreSQL`; update memory file `project_wirebond_ai.md`.
