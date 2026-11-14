"""Does the anomaly checker work, and does it add anything to the main AI?

For one synthetic product and each camera view:
  1. memory from N good units, alarm level from M other good units (anomaly.AnomalyModel.fit),
  2. score NEW good units and NEW defective units (random defects, at least one per unit),
  3. report: AUROC (1.0 = perfect separation, 0.5 = guessing), how many defective units it
     flags, how many good units it wrongly flags, and - the key question - of the defective
     units the main AI (M3j) MISSED completely, how many the checker catches.
Writes heat-map examples to --out.

Usage (inspector venv, it has wirebond_synth + the inspector):
  inspector/.venv/Scripts/python tools/anomaly_eval.py --backbone training/runs/anomaly/anomaly_backbone.onnx \
      --detector training/runs/m3j/wirebond_dfine_nano_joint.onnx --out out/anomaly_eval
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wirebond_inspector.anomaly import AnomalyModel, FeatureExtractor, heat_overlay
from wirebond_inspector.detector import OnnxDetector
from wirebond_inspector.sources.generator import GeneratorCamera


def auroc(neg: list[float], pos: list[float]) -> float:
    n, p = np.asarray(neg), np.asarray(pos)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(n) * len(p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=Path, required=True)
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--det-threshold", type=float, default=0.3)
    ap.add_argument("--product", type=int, default=7)
    ap.add_argument("--n-fit", type=int, default=60)
    ap.add_argument("--n-holdout", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--p", type=float, default=0.05, help="per-wire defect chance for defective units")
    ap.add_argument("--out", type=Path, default=Path("out/anomaly_eval"))
    ap.add_argument("--size", type=int, help="picture side for the checker (default: backbone .json)")
    ap.add_argument("--margin", type=float, default=1.1, help="alarm level = highest good score x margin")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ext, det = FeatureExtractor(a.backbone, a.size), OnnxDetector(a.detector)
    report = {}
    for view in ("top", "angled"):
        good_cam = GeneratorCamera("g", view, {"product_seed": a.product, "injection": {"mode": "none"}})
        bad_cam = GeneratorCamera("b", view, {"product_seed": a.product, "injection": {"mode": "random", "p": a.p}})
        t = time.perf_counter()
        fit_imgs = [good_cam.capture(10000 + i).image for i in range(a.n_fit)]
        hold_imgs = [good_cam.capture(20000 + i).image for i in range(a.n_holdout)]
        print(f"{view}: rendered {len(fit_imgs)}+{len(hold_imgs)} good units, building memory...", flush=True)
        model = AnomalyModel.fit(ext, fit_imgs, hold_imgs, margin=a.margin)
        model.save(a.out / f"product{a.product}_{view}.npz")
        t_fit = time.perf_counter() - t

        good_s, bad_s, rows = [], [], []
        for i in range(a.n_test):
            r = model.score(ext(good_cam.capture(30000 + i).image))
            good_s.append(r.score)
        good_det_fa = sum(bool(det.detect(good_cam.capture(30000 + i).image, a.det_threshold))
                          for i in range(a.n_test))
        u, t_score = 40000, []
        while len(bad_s) < a.n_test:
            u += 1
            truth = [d for d in bad_cam.planned_defects(u) if d.params.get("labelled", True)]
            if not truth:
                continue
            img = bad_cam.capture(u).image
            t0 = time.perf_counter()
            r = model.score(ext(img))
            t_score.append(time.perf_counter() - t0)
            found = det.detect(img, a.det_threshold)
            bad_s.append(r.score)
            rows.append({"unit": u, "classes": sorted({d.cls for d in truth}), "anomaly": r.score,
                         "flagged": r.flagged, "detector_found": len(found)})
            if len(rows) <= 6 or (not found and r.flagged and len(rows) < 40):
                heat_overlay(img, r.heat, model.threshold).save(a.out / f"{view}_u{u}_{'missed' if not found else 'seen'}.jpg")
        missed = [x for x in rows if x["detector_found"] == 0]
        caught = [x for x in missed if x["flagged"]]
        res = {
            "auroc": round(auroc(good_s, bad_s), 3),
            "threshold": round(model.threshold, 3),
            "defective flagged": f"{sum(x['flagged'] for x in rows)}/{len(rows)}",
            "good wrongly flagged": f"{sum(s > model.threshold for s in good_s)}/{len(good_s)}",
            "M3j: defective units with NO finding": f"{len(missed)}/{len(rows)}",
            "of those, anomaly checker caught": f"{len(caught)}/{len(missed)}",
            "M3j: good units with a finding": f"{good_det_fa}/{a.n_test}",
            "fit seconds": round(t_fit, 1), "check ms/picture": round(1000 * float(np.mean(t_score)), 1),
            "missed classes": sorted({c for x in missed for c in x["classes"]}),
            "per class flagged": {c: f"{sum(x['flagged'] for x in rows if c in x['classes'])}/"
                                     f"{sum(c in x['classes'] for x in rows)}"
                                  for c in sorted({c for x in rows for c in x["classes"]})},
        }
        report[view] = res
        print(view, json.dumps(res, indent=1))
    report["settings"] = {"size": ext.size, "margin": a.margin}
    (a.out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
