"""Collect the numbers behind every docs figure into out/figdata.json (read by docs/brand/figures.html).

    tools/.venv/Scripts/python tools/figures/export_data.py

Sources: tools/figures/measure.py's cached measurements in out/*.json, the M3k-small
model's per-type levels, and the PC station database for the repeat-alarm strip.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import MIX, SPEED  # noqa: E402


def repeat():
    db = sqlite3.connect(REPO / "inspector" / "data" / "inspector2.db")
    pos, cls = db.execute("SELECT position, cls FROM alerts GROUP BY position, cls ORDER BY COUNT(*) DESC LIMIT 1").fetchone()
    units = db.execute("SELECT id, unit_no FROM units WHERE source='camera' ORDER BY id DESC LIMIT 40").fetchall()[::-1]
    q = lambda tbl, u: bool(db.execute(f"SELECT 1 FROM {tbl} WHERE unit_id=? AND position=? AND cls=?", (u, pos, cls)).fetchone())
    return {"position": pos, "cls": cls, "units": [{"no": n, "hit": q("findings", u), "alarm": q("alerts", u)} for u, n in units]}


def main():
    out = REPO / "out"
    d = {
        "history": json.loads((out / "m3k_history.json").read_text()),
        "sweep": json.loads((out / "sweep.json").read_text()),
        "checker": json.loads((out / "checker_scores.json").read_text()),
        "levels": json.loads((REPO / "training/runs/m3k/wirebond_dfine_small_m3k.json").read_text())["class_thresholds"],
        "mix": MIX,
        "speed": SPEED,
        "repeat": repeat(),
        "gallery": json.loads((out / "gallery_dets.json").read_text()),
        "found": [["M2", "synthetic only", 0], ["M2b", "+ real photos", 42], ["M2c", "+ 640 px tiles", 56],
                  ["M3j", "all data, GPU", 69], ["M3k-small", "BondSight now", 63]],
        "flagged": [["M2c", "", 100], ["M3j", "", 100], ["M3k-small", "BondSight now", 0]],
    }
    (out / "figdata.json").write_text(json.dumps(d))
    print("wrote", out / "figdata.json", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in d.items()})


if __name__ == "__main__":
    main()
