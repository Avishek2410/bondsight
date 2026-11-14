"""Remember every inspected unit: a database (PostgreSQL or SQLite) plus the pictures on disk.

What: for each unit we save when it was inspected, its verdict (OK / DEFECT / CHECK), each
camera's picture (or why it failed), every defect found (class, score, box, wire position) and
any repetitive-defect alert. Each picture can also carry the unusual-spot checker's verdict
(score, flag, red heat-map picture).
Why: the history, Review page, heat map and repetitive tracker all need the past, and it must
survive a restart or a power cut. The tables themselves are described in db.py.
A unit is saved "all or nothing": if anything goes wrong half way, nothing of it is kept,
not even its pictures.
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import time
from collections.abc import Iterator

import sqlalchemy as sa

from . import db
from .db import alerts as T_alerts
from .db import decisions as T_decisions
from .db import findings as T_findings
from .db import shots as T_shots
from .db import units as T_units
from .verdict import unit_verdict

SOURCES = ("camera", "upload")
VERDICTS = ("real", "false_alarm", "missed", "confirm_ok", "withdraw")


def _decided_expr():
    """SQL for 'a person has finished reviewing this unit' (see Store.add_decision).

    Every defect found has a real/false_alarm decision, and the unit had defects or a person
    confirmed it good / added a missed defect.
    """
    uid = T_units.c.id
    open_findings = (sa.select(sa.func.count()).select_from(T_findings)
                     .where(T_findings.c.unit_id == uid,
                            ~sa.exists().where(T_decisions.c.finding_id == T_findings.c.id,
                                               T_decisions.c.verdict.in_(("real", "false_alarm"))))
                     .scalar_subquery())
    any_finding = sa.exists().where(T_findings.c.unit_id == uid)
    w = T_decisions.alias("w")
    standing_missed = sa.exists().where(
        T_decisions.c.unit_id == uid, T_decisions.c.verdict == "missed",
        ~sa.exists().where(w.c.verdict == "withdraw", w.c.ref_id == T_decisions.c.id))
    confirmed = sa.exists().where(T_decisions.c.unit_id == uid, T_decisions.c.verdict == "confirm_ok")
    return sa.and_(open_findings == 0, sa.or_(any_finding, standing_missed, confirmed))


class Store:
    def __init__(self, data_dir: Path, url: str | None = None, classes: tuple[str, ...] | None = None):
        self.dir = Path(data_dir)
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        self.url = url or db.sqlite_url(self.dir)
        self.classes = tuple(classes) if classes else None
        self.engine = db.connect(self.url)
        self._lock = threading.Lock()          # one writer at a time (SQLite needs it, harmless for PostgreSQL)

    # ---- writing -------------------------------------------------------------------------
    def add_unit(self, unit_no: int | None, shots: list[dict], findings: list[dict], ts: float,
                 source: str = "camera", label: str | None = None) -> int:
        """Save one unit. Uploaded photos use unit_no=None and source="upload" (label = file name)."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}")
        with self._lock:
            folder = None
            try:
                with self.engine.begin() as c:                  # one transaction
                    uid = c.execute(sa.insert(T_units).values(
                        unit_no=unit_no, ts=ts, source=source, label=label,
                        verdict=unit_verdict(findings, shots))).inserted_primary_key[0]
                    folder = self.dir / "images" / str(uid)
                    for f in findings:
                        x0, y0, x1, y1 = (float(v) for v in f["box"])     # must be 4 numbers
                        c.execute(sa.insert(T_findings).values(
                            unit_id=uid, camera=f["camera"], cls=f["cls"], score=float(f["score"]),
                            x0=x0, y0=y0, x1=x1, y1=y1, position=f["position"]))
                    for s in shots:
                        rel = heat = None
                        w = h = None
                        if s.get("image") is not None:
                            folder.mkdir(parents=True, exist_ok=True)
                            rel = f"images/{uid}/{s['camera']}.jpg"
                            s["image"].convert("RGB").save(self.dir / rel, quality=90)
                            w, h = s["image"].size
                        if s.get("heat_image") is not None:
                            folder.mkdir(parents=True, exist_ok=True)
                            heat = f"images/{uid}/{s['camera']}_heat.jpg"
                            s["heat_image"].convert("RGB").save(self.dir / heat, quality=85)
                        flag = s.get("anomaly_flag")
                        c.execute(sa.insert(T_shots).values(
                            unit_id=uid, camera=s["camera"], view=s.get("view"), source=s.get("source"),
                            image_path=rel, error=s.get("error"), anomaly=s.get("anomaly"),
                            anomaly_flag=None if flag is None else bool(flag), heat_path=heat,
                            width=w, height=h, hints=json.dumps(s["hints"]) if s.get("hints") else None))
                return int(uid)
            except Exception:
                if folder is not None and folder.exists():
                    shutil.rmtree(folder, ignore_errors=True)
                raise

    def add_decision(self, unit_id: int, verdict: str, operator: str, finding_id: int | None = None,
                     camera: str | None = None, cls: str | None = None, box: list[float] | None = None,
                     ts: float | None = None, ref_id: int | None = None) -> int:
        """Record what a person decided about a unit. Every decision is kept; the latest one counts.

        real / false_alarm: about one defect the AI found (finding_id).
        missed: the AI missed a defect; the person drew `box` on `camera`'s picture and chose `cls`.
        confirm_ok: the person looked and the unit is good (only if nothing real was found and no
                    "missed" box stands).
        withdraw: take back a wrongly drawn "missed" box (ref_id = that decision's id).
        Raises LookupError for an unknown unit, ValueError (plain message) for anything else wrong.
        """
        operator = (operator or "").strip()
        if not 1 <= len(operator) <= 60:
            raise ValueError("operator: give a name of 1 to 60 characters")
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
        u = self.unit(unit_id)
        if u is None:
            raise LookupError(f"no unit {unit_id}")
        row = {"unit_id": unit_id, "verdict": verdict, "operator": operator,
               "ts": time.time() if ts is None else float(ts)}
        if verdict in ("real", "false_alarm"):
            if finding_id is None or finding_id not in {f["id"] for f in u["findings"]}:
                raise ValueError(f"finding {finding_id} does not belong to unit {unit_id}")
            row["finding_id"] = finding_id
        elif verdict == "missed":
            shot = next((s for s in u["shots"] if s["camera"] == camera and s["image_path"]), None)
            if shot is None:
                raise ValueError(f"camera {camera!r}: this unit has no picture from that camera")
            if self.classes is not None and cls not in self.classes:
                raise ValueError(f"defect type {cls!r} is unknown (use one of {', '.join(self.classes)})")
            try:
                x0, y0, x1, y1 = (float(v) for v in box)
            except (TypeError, ValueError):
                raise ValueError("box: give 4 numbers x0, y0, x1, y1") from None
            if not (x0 < x1 and y0 < y1):
                raise ValueError("box: x0 must be left of x1 and y0 above y1")
            if x0 < 0 or y0 < 0 or x1 > shot["width"] or y1 > shot["height"]:
                raise ValueError("box: must be inside the picture")
            row.update(camera=camera, cls=cls, x0=x0, y0=y0, x1=x1, y1=y1)
        elif verdict == "withdraw":
            target = next((d for d in u["decisions"] if d["id"] == ref_id), None)
            if target is None or target["verdict"] != "missed":
                raise ValueError(f"withdraw: decision {ref_id} is not a missed box of unit {unit_id}")
            if target["withdrawn"]:
                raise ValueError(f"withdraw: missed box {ref_id} was already removed")
            row["ref_id"] = ref_id
        else:   # confirm_ok
            if any(f["decision"] != "false_alarm" for f in u["findings"]):
                raise ValueError("confirm_ok: the AI found defects; mark each one real or false alarm, "
                                 "you can only confirm a unit as good when all of them are false alarms")
            if any(d["verdict"] == "missed" and not d["withdrawn"] for d in u["decisions"]):
                raise ValueError("confirm_ok: a missed defect is marked on this unit; remove that box first "
                                 "if the unit is good")
        with self._lock, self.engine.begin() as c:
            return int(c.execute(sa.insert(T_decisions).values(**row)).inserted_primary_key[0])

    def add_alerts(self, unit_id: int, alerts: list[dict]) -> None:
        if not alerts:
            return
        with self._lock, self.engine.begin() as c:
            c.execute(sa.insert(T_alerts), [{"unit_id": unit_id, "position": a["position"], "cls": a["cls"],
                                              "count": a["count"], "window": a["window"]} for a in alerts])

    # ---- reading -------------------------------------------------------------------------
    def _q(self, stmt) -> list[dict]:
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(stmt)]

    def unit(self, unit_id: int) -> dict | None:
        r = self._q(sa.select(T_units).where(T_units.c.id == unit_id))
        if not r:
            return None
        u = r[0]
        shots = self._q(sa.select(T_shots.c.camera, T_shots.c.view, T_shots.c.source, T_shots.c.image_path,
                                  T_shots.c.error, T_shots.c.anomaly, T_shots.c.anomaly_flag, T_shots.c.heat_path,
                                  T_shots.c.width, T_shots.c.height, T_shots.c.hints)
                        .where(T_shots.c.unit_id == unit_id).order_by(T_shots.c.id))
        for s in shots:
            s["hints"] = json.loads(s["hints"]) if s["hints"] else []
        findings = [{"id": f["id"], "camera": f["camera"], "cls": f["cls"], "score": f["score"],
                     "box": [f["x0"], f["y0"], f["x1"], f["y1"]], "position": f["position"]}
                    for f in self._q(sa.select(T_findings).where(T_findings.c.unit_id == unit_id)
                                     .order_by(T_findings.c.score.desc(), T_findings.c.id))]
        alerts = self._q(sa.select(T_alerts.c.position, T_alerts.c.cls, T_alerts.c.count, T_alerts.c.window)
                         .where(T_alerts.c.unit_id == unit_id).order_by(T_alerts.c.id))
        decisions = []
        for d in self._q(sa.select(T_decisions).where(T_decisions.c.unit_id == unit_id)
                         .order_by(T_decisions.c.id.desc())):       # save order, not the clock
            box = [d.pop(k) for k in ("x0", "y0", "x1", "y1")]
            d["box"] = box if box[0] is not None else None
            d.pop("unit_id")
            decisions.append(d)
        gone = {d["ref_id"] for d in decisions if d["verdict"] == "withdraw"}
        for d in decisions:
            d["withdrawn"] = d["id"] in gone
        latest: dict[int, str] = {}
        for d in decisions:                                   # newest first: first seen = latest
            if d["finding_id"] is not None:
                latest.setdefault(d["finding_id"], d["verdict"])
        for f in findings:
            f["decision"] = latest.get(f["id"])
        decided = all(f["decision"] for f in findings) and (
            bool(findings) or any(d["verdict"] == "confirm_ok" or (d["verdict"] == "missed" and not d["withdrawn"])
                                  for d in decisions))
        return {"id": u["id"], "unit_no": u["unit_no"], "ts": u["ts"], "verdict": u["verdict"],
                "source": u["source"], "label": u["label"],
                "review_state": "decided" if decided else "undecided",
                "shots": shots, "findings": findings, "alerts": alerts, "decisions": decisions}

    def units(self, limit: int = 50, before: int | None = None, verdict: str | None = None,
              source: str | None = None, undecided: bool = False) -> list[dict]:
        uid = T_units.c.id

        def count(t, *extra):
            return sa.select(sa.func.count()).select_from(t).where(t.c.unit_id == uid, *extra).scalar_subquery()

        decided = _decided_expr()
        stmt = (sa.select(uid, T_units.c.unit_no, T_units.c.ts, T_units.c.verdict, T_units.c.source,
                          T_units.c.label,
                          count(T_findings).label("n_findings"), count(T_alerts).label("n_alerts"),
                          count(T_shots, T_shots.c.error.is_not(None)).label("n_errors"),
                          count(T_shots, T_shots.c.anomaly_flag.is_(True)).label("n_unusual"),
                          sa.case((decided, "decided"), else_="undecided").label("review_state"))
                .order_by(uid.desc()).limit(limit))
        if before is not None:
            stmt = stmt.where(uid < before)
        if verdict is not None:
            stmt = stmt.where(T_units.c.verdict == verdict)
        if source is not None:
            stmt = stmt.where(T_units.c.source == source)
        if undecided:
            stmt = stmt.where(~decided)
        return self._q(stmt)

    def decided_units(self) -> Iterator[dict]:
        """Every fully reviewed unit, oldest first (for the training export)."""
        for r in self._q(sa.select(T_units.c.id).where(_decided_expr()).order_by(T_units.c.id)):
            yield self.unit(r["id"])

    def count(self, source: str | None = None) -> int:
        stmt = sa.select(sa.func.count().label("n")).select_from(T_units)
        if source is not None:
            stmt = stmt.where(T_units.c.source == source)
        return self._q(stmt)[0]["n"]

    def _line_ids(self, m: int) -> list[int]:
        rows = self._q(sa.select(T_units.c.id).where(T_units.c.unit_no.is_not(None))
                       .order_by(T_units.c.id.desc()).limit(m))
        return [r["id"] for r in rows]

    def recent_unit_findings(self, m: int) -> list[set[tuple[str, str]]]:
        out = []
        for uid in self._line_ids(m)[::-1]:
            rows = self._q(sa.select(T_findings.c.position, T_findings.c.cls).distinct()
                           .where(T_findings.c.unit_id == uid))
            out.append({(r["position"], r["cls"]) for r in rows})
        return out

    def heatmap(self, last: int = 200) -> dict[str, dict[str, int]]:
        ids = self._line_ids(last)
        if not ids:
            return {}
        rows = self._q(sa.select(T_findings.c.position, T_findings.c.cls,
                                 sa.func.count(sa.distinct(T_findings.c.unit_id)).label("n"))
                       .where(T_findings.c.unit_id.in_(ids)).group_by(T_findings.c.position, T_findings.c.cls))
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            out.setdefault(r["position"], {})[r["cls"]] = r["n"]
        return out

    def alerts(self, limit: int = 20) -> list[dict]:
        return self._q(sa.select(T_alerts.c.unit_id, T_units.c.unit_no, T_units.c.ts, T_alerts.c.position,
                                 T_alerts.c.cls, T_alerts.c.count, T_alerts.c.window)
                       .join(T_units, T_units.c.id == T_alerts.c.unit_id)
                       .order_by(T_alerts.c.id.desc()).limit(limit))

    def next_unit_no(self) -> int:
        m = self._q(sa.select(sa.func.max(T_units.c.unit_no).label("m")))[0]["m"]
        return 0 if m is None else m + 1
