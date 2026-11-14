"""The web part: a small API plus one page, served by FastAPI.

What: the page in static/index.html asks these addresses for data and shows it:
  GET  /api/status          model, cameras, settings
  POST /api/inspect         inspect one unit now (409 if one is already running)
  POST /api/check           check uploaded photos (form field `files`, 1-20; optional `camera`)
  GET  /api/units           recent units, newest first (?verdict=DEFECT|CHECK|OK &source=camera|upload
                            &undecided=1 &before=<id> &limit=)
  GET  /api/units/{id}      one unit with its defects, decisions, alerts and (simulated) true answer
  POST /api/units/{id}/decisions   record a person's decision (real / false_alarm / missed / confirm_ok /
                            withdraw = take back a missed box, ref_id = its decision id)
  GET  /api/images/{id}/{camera}.jpg   a saved picture ({camera}_heat.jpg = unusual-spot map)
  GET  /api/heatmap         how often each defect appeared at each position
  GET  /api/alerts          recent repetitive-defect alerts
  GET  /static/app.css, /static/app.js   the page's look and logic
Why: keeping the page separate from the API means other programs (a test script, a factory
system) can use the same API without the page.
"""
from __future__ import annotations

import re
import socket
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..db import safe_url
from ..photocheck import MAX_BYTES
from ..service import AllCamerasFailed, Busy, Inspector

MAX_FILES = 20


class DecisionIn(BaseModel):
    verdict: str
    operator: str
    finding_id: int | None = None
    camera: str | None = None
    cls: str | None = None
    box: list[float] | None = None
    ref_id: int | None = None


STATIC = Path(__file__).resolve().parent / "static"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+\.jpg$")
SAMPLE_EXT = {".png", ".jpg", ".jpeg"}
# the online demo (config `demo: true`) only checks photos; these station addresses are closed there
STATION_ONLY = ("/api/inspect", "/api/units", "/api/images", "/api/heatmap", "/api/alerts")


def create_app(ins: Inspector) -> FastAPI:
    app = FastAPI(title="BondSight wire-bond inspector", docs_url="/api/docs", redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")    # app.css, app.js

    demo = bool(getattr(ins.cfg, "demo", False))
    samples_dir = getattr(ins.cfg, "samples", None)

    def sample_names() -> list[str]:
        if not demo or samples_dir is None:
            return []
        return sorted(p.name for p in Path(samples_dir).iterdir() if p.suffix.lower() in SAMPLE_EXT and p.is_file())

    @app.middleware("http")
    async def revalidate_page(request, call_next):
        """Browsers must ask again for the page files, so an update shows without a hard refresh.
        In the demo, station-only addresses answer 403 before anything else runs."""
        if demo and request.url.path.startswith(STATION_ONLY):
            return JSONResponse({"detail": "not available in the demo"}, status_code=403)
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status():
        return {"model": ins.model_name, "threshold": ins.cfg.threshold,
                "cameras": ins.cameras(), "tracker": {"k": ins.cfg.tracker.k, "m": ins.cfg.tracker.m},
                "unusual_check": ins.checker is not None and bool(ins.checker.models),
                "trigger": ins.cfg.trigger.mode, "interval_s": ins.cfg.trigger.seconds,
                "units_total": ins.store.count(), "line_units": ins.store.count(source="camera"),
                "last_error": ins.last_error,
                "database": safe_url(str(ins.store.engine.url)), "station": "online demo" if demo else socket.gethostname(),
                "classes": list(getattr(ins.detector, "classes", []) or []), "demo": demo}

    @app.post("/api/inspect")
    async def inspect():
        try:
            return await run_in_threadpool(ins.inspect)
        except Busy:
            raise HTTPException(409, "busy")
        except AllCamerasFailed as e:
            raise HTTPException(503, f"no camera gave a picture: {e}")

    @app.post("/api/check")
    async def check(files: list[UploadFile] = File(default=[]), camera: str | None = Form(default=None),
                    sample: str | None = Form(default=None)):
        batch = []
        if sample and demo:                   # a demo visitor clicked one of our sample pictures
            if sample not in sample_names():
                raise HTTPException(404, "no such sample")
            batch.append((sample, (Path(samples_dir) / sample).read_bytes()))
        else:
            if not 1 <= len(files) <= MAX_FILES:
                raise HTTPException(400, f"send 1 to {MAX_FILES} photos")
            for f in files:                   # read at most MAX_BYTES + 1: load_photo reports "too big"
                batch.append((f.filename or "photo", await f.read(MAX_BYTES + 1)))
        try:
            return await run_in_threadpool(ins.check_photos, batch, camera or None, not demo)
        except ValueError as e:
            raise HTTPException(400, str(e))

    if demo and samples_dir is not None:
        @app.get("/api/samples")
        def samples():
            return [{"name": n, "url": f"/api/samples/{n}"} for n in sample_names()]

        @app.get("/api/samples/{name}")
        def sample_file(name: str):
            if name not in sample_names():
                raise HTTPException(404, "not found")
            return FileResponse(Path(samples_dir) / name)

    @app.get("/api/units")
    def units(limit: int = 50, before: int | None = None, verdict: str | None = None,
              source: str | None = None, undecided: bool = False):
        return ins.store.units(limit=max(1, min(limit, 500)), before=before, verdict=verdict,
                               source=source, undecided=undecided)

    @app.post("/api/units/{unit_id}/decisions", status_code=201)
    def decide(unit_id: int, d: DecisionIn):
        try:
            did = ins.store.add_decision(unit_id, d.verdict, d.operator, finding_id=d.finding_id,
                                         camera=d.camera, cls=d.cls, box=d.box, ref_id=d.ref_id)
        except LookupError:
            raise HTTPException(404, "no such unit")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"id": did, "unit": ins.unit(unit_id)}

    @app.get("/api/units/{unit_id}")
    def unit(unit_id: int):
        u = ins.unit(unit_id)
        if u is None:
            raise HTTPException(404, "no such unit")
        return u

    @app.get("/api/images/{unit_id}/{name}")
    def image(unit_id: int, name: str):
        root = (ins.store.dir / "images").resolve()
        if not SAFE_NAME.match(name):
            raise HTTPException(404, "not found")
        p = (root / str(unit_id) / name).resolve()
        if root not in p.parents or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p, media_type="image/jpeg")

    @app.get("/api/heatmap")
    def heatmap(last: int = 200):
        return ins.store.heatmap(last=max(1, min(last, 5000)))

    @app.get("/api/alerts")
    def alerts(limit: int = 20):
        return ins.store.alerts(limit=max(1, min(limit, 200)))

    return app
