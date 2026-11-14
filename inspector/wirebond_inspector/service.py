"""The inspection itself: what happens when you press Inspect.

1. Every camera takes a picture of the same unit (a failing camera is noted, not fatal).
2. The AI looks at each picture and returns defects with boxes.
3. Each defect gets a position (which wire, or which grid square).
4. The unit is saved to the database, pictures included.
5. The repetitive tracker checks if the same defect keeps coming back, and saves any alert.
Optional second opinion (config `anomaly:`): the anomaly checker, which only knows what GOOD
units look like, scores each picture. A high score with no defect found = "please check":
something looks unusual that the main AI has no name for. Its red heat map is saved too.
Only one inspection runs at a time; pressing Inspect again meanwhile answers "busy".
In interval mode a background thread presses Inspect every few seconds. When there is simply
nothing new to look at (an empty folder) it waits quietly instead of saving empty units.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time

from .anomaly import CameraChecker, heat_overlay
from .config import Config
from .detector import OnnxDetector
from .photocheck import PhotoError, load_photo, quality_hints
from .sources import CameraError, make_camera
from .store import Store
from .tracker import RepetitiveTracker
from .verdict import unit_verdict
from .wiremap import position_key

log = logging.getLogger("wirebond_inspector")
IDLE = "no new image"


class Busy(RuntimeError):
    """An inspection is already running."""


class AllCamerasFailed(RuntimeError):
    """No camera gave a picture, so there is nothing to inspect (the unit is not saved)."""


class Inspector:
    def __init__(self, cfg: Config, detector: OnnxDetector | None = None,
                 checker: CameraChecker | None = None):
        self.cfg = cfg
        self.detector = detector if detector is not None else OnnxDetector(cfg.model)
        self.model_name = getattr(self.detector, "name", cfg.model.stem)
        self.cams = [make_camera(c) for c in cfg.cameras]
        if checker is None and cfg.anomaly is not None:
            checker = CameraChecker(cfg.anomaly.backbone, cfg.anomaly.dir, [c.name for c in cfg.cameras],
                                    cfg.anomaly.size)
        self.checker = checker
        self.store = Store(cfg.data_dir, cfg.database or None, getattr(self.detector, "classes", None))
        self.tracker = RepetitiveTracker(cfg.tracker.k, cfg.tracker.m)
        self.tracker.load(self.store.recent_unit_findings(cfg.tracker.m))
        self._config_layout = json.loads(cfg.layout.read_text()) if cfg.layout else None
        self._lock = threading.Lock()
        self._upload_slot = threading.Semaphore(1)   # one uploaded photo at a time: the Pi also runs the line
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    # ---- helpers ---------------------------------------------------------------------------
    def _layout(self) -> dict | None:
        for c in self.cams:
            lay = c.layout()
            if lay:
                return lay
        return self._config_layout

    def truth(self, unit_no: int) -> list[dict] | None:
        """For simulated units: the defects the generator really drew (None for real cameras)."""
        for c in self.cams:
            if hasattr(c, "planned_defects"):
                return [{"wire": d.wire, "cls": d.cls} for d in c.planned_defects(unit_no)
                        if d.params.get("labelled", True)]
        return None

    def unit(self, unit_id: int) -> dict | None:
        u = self.store.unit(unit_id)
        if u is not None:     # uploads have no unit number and no "true answer"
            u["truth"] = self.truth(u["unit_no"]) if u["source"] == "camera" else None
        return u

    def cameras(self) -> list[dict]:
        out = [c.status() for c in self.cams]
        if self.checker is not None:
            for st in out:
                st["unusual_check"] = self.checker.status(st["name"])
        return out

    # ---- one picture: AI + optional second opinion ------------------------------------------
    def _analyse(self, shot: dict, memory: str | None, view: str, layout) -> list[dict] | None:
        """Run the AI (and the unusual-spot checker with camera `memory`) on shot["image"].

        Fills shot["anomaly"/"anomaly_flag"/"heat_image"]; returns the findings, or None when the
        AI itself failed (then shot["error"] says why and the picture is still kept).
        """
        img, name = shot["image"], shot["camera"]
        try:
            found = self.detector.detect(img, self.cfg.threshold)
            mine = [{"camera": name, "cls": f.cls, "score": f.score, "box": list(f.box),
                     "position": position_key(f, view, img.size, layout)} for f in found]
        except Exception as e:  # keep the picture, record why it was not checked
            log.exception("detection failed on camera %s", name)
            shot["error"] = f"AI failed on this picture: {type(e).__name__}: {e}"
            return None
        if self.checker is not None and memory is not None:
            try:
                r = self.checker.check(memory, img)
            except Exception:  # the second opinion must never stop an inspection
                log.exception("anomaly check failed on camera %s", name)
                r = None
            if r is not None:
                shot["anomaly"], shot["anomaly_flag"] = round(r.score, 3), r.flagged
                if r.flagged:
                    shot["heat_image"] = heat_overlay(img, r.heat, self.checker.models[memory].threshold)
        return mine

    # ---- uploaded photos ---------------------------------------------------------------------
    def check_photos(self, files: list[tuple[str, bytes]], camera: str | None = None,
                     save: bool = True) -> list[dict]:
        """Check uploaded photos with the same AI. One result per file, in order.

        `camera`: use that camera's view and unusual-spot memory (None = top view, no memory).
        Uploads never change the line's unit numbers or the repetitive tracker, and do not wait
        for a running inspection. save=False (online demo): nothing is stored, the picture comes
        back inline as image_b64.
        """
        cam = None
        if camera is not None:
            cam = next((c for c in self.cams if c.name == camera), None)
            if cam is None:
                raise ValueError(f"camera {camera!r} is unknown (use one of {', '.join(c.name for c in self.cams)})")
        name, view = (cam.name, cam.view) if cam else ("upload", "top")
        layout = self._layout()
        has_memory = self.checker is not None and name in getattr(self.checker, "models", {})
        out = []
        for label, data in files:
            try:
                img = load_photo(data)
            except PhotoError as e:
                out.append({"label": label, "error": str(e)})
                continue
            hints = quality_hints(img)
            if not has_memory:
                hints.append("unusual-spot check not set up for this photo")
            shot = {"camera": name, "view": view, "source": f"upload:{label}", "image": img,
                    "error": None, "hints": hints}
            with self._upload_slot:
                findings = self._analyse(shot, name if has_memory else None, view, layout) or []
            if save:
                uid = self.store.add_unit(None, [shot], findings, ts=time.time(), source="upload", label=label)
                res = self.unit(uid)
            else:
                res = self._unsaved(shot, findings, label)
            res["hints"] = hints
            out.append(res)
        return out

    @staticmethod
    def _unsaved(shot: dict, findings: list[dict], label: str) -> dict:
        buf = io.BytesIO()
        shot["image"].convert("RGB").save(buf, "JPEG", quality=88)
        w, h = shot["image"].size
        return {"id": None, "unit_no": None, "ts": time.time(), "source": "upload", "label": label,
                "verdict": unit_verdict(findings, [shot]), "review_state": "undecided",
                "shots": [{"camera": shot["camera"], "view": shot["view"], "source": shot["source"],
                           "image_path": None, "error": shot["error"], "anomaly": shot.get("anomaly"),
                           "anomaly_flag": shot.get("anomaly_flag"), "heat_path": None,
                           "width": w, "height": h, "hints": shot["hints"]}],
                "findings": [dict(f, id=None, decision=None) for f in findings],
                "alerts": [], "decisions": [], "truth": None,
                "image_b64": base64.b64encode(buf.getvalue()).decode("ascii")}

    # ---- one inspection --------------------------------------------------------------------
    def inspect(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise Busy("an inspection is already running")
        try:
            t_all = time.perf_counter()
            unit_no = self.store.next_unit_no()
            layout = self._layout()
            shots, findings, timing, errors = [], [], {}, []
            for cam in self.cams:
                try:
                    frame = cam.capture(unit_no)
                except Exception as e:  # any camera problem stays with that camera
                    msg = str(e) if isinstance(e, CameraError) else f"{type(e).__name__}: {e}"
                    if not isinstance(e, CameraError):
                        log.exception("camera %s failed", cam.name)
                    shots.append({"camera": cam.name, "view": cam.view, "source": None,
                                  "image": None, "error": msg})
                    errors.append(f"{cam.name}: {msg}")
                    continue
                shot = {"camera": cam.name, "view": cam.view, "source": frame.source,
                        "image": frame.image, "error": None}
                shots.append(shot)
                t = time.perf_counter()
                mine = self._analyse(shot, cam.name, cam.view, layout)
                if mine is None:
                    continue
                timing[cam.name] = round(time.perf_counter() - t, 3)
                findings.extend(mine)
            if len(errors) == len(self.cams):
                raise AllCamerasFailed("; ".join(errors))
            uid = self.store.add_unit(unit_no, shots, findings, ts=time.time())
            alerts = self.tracker.update({(f["position"], f["cls"]) for f in findings})
            if alerts:
                self.store.add_alerts(uid, alerts)
            timing["total"] = round(time.perf_counter() - t_all, 3)
            u = self.unit(uid)
            u["timing_s"] = timing
            return u
        finally:
            self._lock.release()

    # ---- interval trigger ------------------------------------------------------------------
    def start_trigger(self) -> None:
        if self.cfg.trigger.mode != "interval" or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="trigger", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.cfg.trigger.seconds):
            try:
                self.inspect()
                self.last_error = None
            except Busy:
                pass
            except AllCamerasFailed as e:
                idle = all(part.endswith(IDLE) for part in str(e).split("; "))
                self.last_error = None if idle else str(e)
                if not idle:
                    log.warning("inspection skipped: %s", e)
            except Exception as e:  # keep the loop alive whatever happens
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("inspection failed")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
