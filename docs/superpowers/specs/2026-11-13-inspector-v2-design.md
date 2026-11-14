# Inspector v2: from demo to a tool a factory can use

Date: 2026-11-13. Status: APPROVED by Avi 2026-11-13 ("ok go with this, build it").

## Intent and why

The wirebond-ai inspector works today (M3k-small detector + unusual-spot checker, PC and Pi 5,
Docker), but only as a demo: simulated cameras, cameras typed into a YAML file, a single page, a
SQLite file, no way for a person to say "the AI was right/wrong".

Avi wants two things:

1. **Make it public** on GitHub with a series of plain-language articles (LinkedIn + `articles/`),
   so readers can try it.
2. **Make it usable in industry.** The honest finding so far is that the limit is data, not the
   network: the models are trained on synthetic images plus a few hundred public photos of other
   people's products. A factory needs a model fine-tuned on *its own* line. So v2's central job is
   to turn the inspector into its own data collector: every operator decision becomes a labelled
   photo for the next model.

Success = a visitor can try it in a browser with no install; an engineer can plug cameras into a
Pi, set them up without editing files, and run a line; operators' decisions pile up into a training
set; a quality team can trace and report defects across several stations from one PostgreSQL.

## Users

| Who | Needs |
|---|---|
| Operator on the line | A big clear verdict per unit (OK / DEFECT / CHECK) and one-tap decisions. Not known yet whether trained operators or anyone; touchscreen or mouse — **design for both**. |
| Engineer | Find and set up cameras, pick model, products, triggers, reports, export, data clean-up. |
| Visitor (from the articles) | Drop in a photo and see what the AI finds. Nothing to install. |

## Goals

1. **Five pages** in one app: Inspect, Check a photo, Review, Reports, Setup.
2. **Check a photo**: upload one or many photos (drag-drop, file picker, or a phone/tablet camera)
   and get the same AI result as from a camera.
3. **Operator decisions** on every flag: *Real defect* / *False alarm* / *Missed something* (draw a
   box + pick the type). Recorded with who and when.
4. **Training export**: reviewed pictures + corrected labels in the COCO format the training
   notebooks already read.
5. **Camera auto-detection** with a setup wizard (live preview, naming, focus/exposure, sharpness
   meter), saved without anyone editing YAML; health dot per camera, auto-reconnect.
6. **Pi ribbon-cable cameras** (OV5647 now, Pi HQ + microscope lens later) supported — a new
   camera driver, because the current OpenCV source cannot open them on a Pi 5.
7. **Triggers**: on-screen button, timer, GPIO input (foot pedal / part sensor). PLC later.
8. **PostgreSQL** as the database, local or central, with an offline buffer.
9. **Roles with PIN login**: operator, engineer, (admin). Audit log of setting changes.
10. **Products and batches**: every unit tagged with product, batch, operator.
11. **Reports**: defects per hour / per wire / per type, false-alarm rate, operator-found misses;
    CSV/Excel export.
12. **Retention**: delete pictures older than N days; keep defect pictures longer.
13. **Online demo**: a free Hugging Face Space running only *Check a photo* (visitor mode).
14. **GitHub launch + articles** after the demo exists (separate writing work, see last section).

## Non-goals (this version)

- PLC / MES / ERP integration (the PostgreSQL schema should make it possible later, nothing more).
- Training a new model inside the app. The app exports data; training stays in the Kaggle notebooks.
- Cloud hosting of a factory station, user accounts with passwords/SSO, multi-tenant anything.
- X-ray images. Height/3D sensors.
- A mobile app. The web page must work on a phone/tablet browser; that is enough.

## Anti-goals (failure even if it "works")

- The line stops because the network or the central database is down.
- An operator can change cameras, models or thresholds by accident.
- A secret (database password, PIN, API key) lands in git or on GitHub.
- Third-party photos (Roboflow sets etc.) end up in the public repo or the demo's sample set
  without their licence allowing it and credit given (see CREDITS.md / DATA_SOURCES.md).
- The demo saves visitors' uploads.
- The UI needs hovering, right-click or small targets (breaks touchscreens).
- Results that differ between camera and uploaded photo for the same picture.

## Constraints

- Runs on Raspberry Pi 5 (8 GB, CPU only, no AI HAT) and on a Windows/Linux PC; Docker multi-arch.
  Current speeds: M3k-small 3.1 s per 2048 px photo, ~3.5 s per two-camera unit with the checker.
- Licences: code MIT, no AGPL (this rules out Ultralytics and similar). Data CC BY / CC0 / BY-SA /
  GPL only, credited in CREDITS.md.
- Existing code to build on: `inspector/` (FastAPI, ONNX detector with tiling and per-class
  thresholds, camera plugins `folder` / `generator` / `opencv`, wire mapping, K-of-M repetitive
  tracker, unusual-spot checker, SQLite store), 85 tests passing. Keep the detector and checker
  untouched where possible: they are verified against the training code.
- Avi is learning; code and docs carry short plain-language "what / why" notes, as in the current
  inspector.
- Nothing pushed, published or made public without Avi's explicit yes.

## Decisions already made (by Avi)

- **PostgreSQL** is the database. (Avi, 2026-11-13.)
- **Operator + engineer modes with PIN**, touch-friendly, works with a mouse too. (Avi: "maybe
  both" on operator type and input.)
- The five pages, camera auto-detection, upload, decisions → training data, online demo, then
  GitHub + articles. (Avi approved the outline: "no, write it up".)

## Decisions made by Claude (Avi may overrule)

- **SQLite remains supported** for the online demo and the tests; the code talks to both through
  one data layer (e.g. SQLAlchemy). Production default = PostgreSQL.
- **Pictures stay as files on disk**; the database stores paths. Keeps PostgreSQL small.
- **Local or central PostgreSQL** chosen in Setup. With a central server, each station keeps a
  local buffer (queue on disk or local DB) and syncs when the link returns, so inspection never
  waits on the network.
- **The database password is generated at first start** and written to a local, git-ignored
  settings file (`.env`); never in code, logs or chat. PINs stored hashed.
- **Schema migrations** (e.g. Alembic) so upgrades keep data. Existing SQLite data on the PC/Pi can
  be thrown away (it is test data) — no migration from v1 needed.
- **Tables**: units, shots, findings, alerts (as now, plus product/batch/operator/source), products,
  batches, operators (name, role, PIN hash), decisions (finding or unit, verdict, optional box +
  class for misses, operator, time), cameras (saved setup), settings, audit_log.
- **Camera detection** covers: Pi ribbon cameras (libcamera / Picamera2), USB cameras and USB
  microscopes (V4L2 on Linux, DirectShow/MSMF on Windows), network cameras by pasted RTSP URL
  (ONVIF discovery = nice-to-have), and simulated cameras. In Docker the container gets the camera
  devices passed through; Setup explains in plain words when a camera is not visible.
- **Unusual-spot checker in Check a photo** runs only if the chosen product has learned good units
  for that view; otherwise the page says "not set up for this product".
- **Upload quality hints**: warn when a photo is very blurred or the wires are too small to judge.
- **Demo**: Hugging Face Space (free CPU), M3k-small, sample photos limited to images whose licence
  allows redistribution (synthetic images from our own generator are always safe), visitor mode,
  nothing stored.
- **Build order**: (1) Check a photo + decisions + Review + PostgreSQL data layer; (2) online demo,
  then GitHub launch + first articles; (3) camera scan + setup wizard + Pi ribbon-camera driver
  (fully testable once Avi has cameras; can be built against simulated/USB first); (4) Reports,
  export, products/batches, roles/audit, triggers, retention.

## Visual direction (Avi chose, 2026-11-13)

"Industrial, light", from the mockup
`.superpowers/brainstorm/2132-1794542954/content/ui-industrial.html` (the reference for look and
feel; copy it into the repo as `docs/design/ui-industrial-mockup.html` when building).

- Machine-panel look: RAL 7035-style light grey housing, graphite, safety yellow. **Colour only for
  state** (red DEFECT, amber CHECK, green OK), as in ISA-101 HMI practice. Light only for now
  (no dark mode needed).
- Condensed industrial type (Archivo, condensed widths), tabular numbers.
- Header nameplate (station + line), tab strip of the five pages, status lamps per camera +
  database, operator name. Info rail: product, batch, units this shift, defects / check, model,
  last cycle time.
- **Stack light** (red/amber/green andon tower) showing the current unit's verdict, readable from
  across the room.
- **Specimen stage**: the camera picture on a measuring grid with ruler and crosshair; switch
  Top / Angled / 3D / Unusual spots. "3D" = the picture(s) tilted in perspective with smooth motion
  between views (CSS 3D on the real photos; a modelled 3D chip only for products with a wire map —
  later). Defect labels as machine tags ("W05 · WIRE SWEEP · 91%").
- DEFECT panel with a hazard-stripe band; physical-looking push buttons that press down
  (Real defect / False alarm / Missed something), big yellow INSPECT NEXT.
- "Check a photo" drop slot on the Inspect page as well as its own page.
- Bottom log strip of recent units colour-coded by verdict.
- Motion: purposeful only (view tilt, lamp glow on a new verdict, box pop-in, button press);
  respects reduced-motion. Must stay smooth on the Pi 5 browser (CSS transforms, no WebGL needed).

## Answers to the builder review (Claude's calls, Avi may overrule)

1. **Pictures with a central PostgreSQL.** Every row syncs to the central server. Pictures sync too,
   but selectively: every picture with a finding, a CHECK, or an operator decision, plus a small
   random sample of OK pictures (setting, default 2%). So central Review, Reports and training
   export work without visiting stations. Stations keep their own pictures under their own retention;
   the central server has its own (longer) retention. With a local PostgreSQL there is no sync.
2. **Unit verdict rule.** DEFECT if the detector reports any finding; CHECK if no finding but the
   unusual-spot checker alarms on any view; OK otherwise. **Decisions do not hold up the line by
   default**: undecided units wait in a queue on Review. A per-product setting "operator must
   decide before the next unit" exists for lines that want it. *Missed something* is offered on
   every unit, including OK ones. An OK unit counts as confirmed good only after someone reviews it.
3. **Product** = name + wire layout (wire map) + which cameras/views it uses + unusual-spot memory
   per camera + optional model / per-class threshold override (default: the station's model and its
   json levels) + the "must decide" setting. **Teaching the checker the good units is done from
   Setup** (pick N recent OK units or upload good photos → fit). This is allowed: the non-goal is
   training the detector, not fitting the checker's memory.
4. **Pi ribbon cameras inside Docker: yes.** A Pi-specific image based on Debian with the Raspberry
   Pi apt repository (Picamera2 / libcamera from apt), with the camera devices passed through. The
   PC keeps the current slim image. Same app code, two images.
5. **Uploads on a station are saved** as units with source = upload: stored, reviewable, included
   in training export. Reports count line units only by default, with a filter to include uploads.
   The demo (visitor mode) saves nothing.
6. **Training export truth** = reviewed data only. One COCO image per shot. Real defect → box kept.
   False alarm → box dropped; the picture stays (a "hard negative": a look-alike the model must
   learn to ignore). Missed something → the drawn box added. Reviewed OK unit → picture with no
   boxes. Unreviewed units are not exported.
7. **Station vs central.** Each station keeps a local copy of everything it needs to run: operators
   and PIN hashes, products, cameras, settings (edited centrally or locally, synced both ways; last
   change wins, recorded in the audit log). Unit numbers and the repetitive-defect tracker are per
   station (one station = one line position). Central adds station name to every row and reports
   across stations. A station can start, log in and inspect with the link down.

## Left to the builder

- Front-end technology (current UI is a single static page with plain JS; staying framework-free is
  fine, a small framework is fine too if it stays light on the Pi).
- Exact page layouts and visual design, within: touch-friendly, light/dark, readable from a step
  back, big OK / DEFECT / CHECK verdict.
- How the offline buffer is built.
- How many photos one upload batch may hold and the size limit per photo.
- How GPIO is read (library choice) and debouncing.
- Report charts and export formats beyond CSV.

## Writing work (after step 2)

- New GitHub README: what it is, a detection picture, the model story M2 → M3k-small with honest
  numbers and limits, try-it links (demo, Docker in 5 minutes).
- `docs/journey.md`: each model, why, result, lesson — the source for the AI articles.
- Articles: keep the existing 7-part wire-bonding series; add an "building the AI" track. Each as a
  full article in `articles/` plus a short LinkedIn version in `linkedin/`. Same plain language as
  in chat.
- Pre-publish checks: no secrets, no third-party photos, credits complete; choice of GitHub account
  and repo name (Avi decides).
