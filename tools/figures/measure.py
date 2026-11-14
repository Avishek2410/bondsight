"""The slow measurements behind the docs figures (cached in out/*.json, read by export_data.py).

    tools/.venv/Scripts/python tools/figures/measure.py sweep      # M3j vs M3k-small on NTUST test photos
    tools/.venv/Scripts/python tools/figures/measure.py checker    # unusual-spot scores on generated units
    inspector/.venv/Scripts/python tools/figures/measure.py gallery   # BondSight's boxes on demo/samples

Sources:
  training curves  out/m3k_history.json (parsed from the Kaggle M3k notebook output)
  training mix     the M3k notebook's per-epoch mix (printed in its Step 5)
  threshold sweep  M3j and M3k-small ONNX models on the 63 NTUST test images (tiled), counted with
                   tools/threshold_sweep.py's rules (a defect is found when a box of the right type
                   overlaps it by IoU >= 0.3; a good wire is flagged when any box overlaps it)
  levels           the per-type confidence levels in the M3k-small model .json
  checker          the station's unusual-spot memories, on freshly generated good and defective units
  repeat alarm     the PC station database (generated units, product 7)
  speed            measured on 2026-11-13 (README numbers)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


MIX = [("Generated whole chips", 1500), ("NTUST tiles (×2)", 1752), ("Fuji bent-wire tiles (×2)", 896),
       ("new_photos tiles", 700), ("Good-wire tiles (hawk, lyra, Fuji Pass)", 700),
       ("Generated close-ups", 600), ("MEMS photos (×2) + close-ups", 219)]


def sweep_data() -> dict:
    sys.path[:0] = [str(REPO / "inspector")]
    import ast
    from PIL import Image
    from wirebond_inspector.detector import OnnxDetector
    # the same test photos and counting rules as tools/threshold_sweep.py, without its PyTorch import
    src = (REPO / "tools" / "threshold_sweep.py").read_text(encoding="utf-8")
    keep = [n for n in ast.parse(src).body if (isinstance(n, ast.FunctionDef) and n.name in ("ntust_test", "score"))
            or (isinstance(n, ast.Assign) and n.targets[0].id in ("LABEL_MAP", "SEED"))
            or isinstance(n, (ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.ImportFrom) and n.module == "reality_check")]
    ns: dict = {}
    exec(compile(ast.Module(keep, []), "threshold_sweep", "exec"), ns)

    def iou(a, b):
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)
    ns["iou"] = iou
    ntust_test, score = ns["ntust_test"], ns["score"]
    items = ntust_test(str(REPO / "real"))
    thrs = [round(0.10 + 0.05 * i, 2) for i in range(16)]
    out = {"n_images": len(items)}
    for name, path in (("M3j", "training/runs/m3j/wirebond_dfine_nano_joint.onnx"),
                       ("M3k-small", "training/runs/m3k/wirebond_dfine_small_m3k.onnx")):
        det = OnnxDetector(REPO / path)
        det.class_thresholds = {}                       # keep every answer; levels are applied below
        preds = [[(list(f.box), f.cls, f.score) for f in det.detect(Image.open(it["path"]).convert("RGB"), 0.05)]
                 for it in items]
        out[name] = [(t, *score(preds, items, t)[:2]) for t in thrs]
        print(name, out[name][::3], flush=True)
    (REPO / "out" / "sweep.json").write_text(json.dumps(out))
    return out


def checker_data() -> dict:
    sys.path[:0] = [str(REPO / "inspector"), str(REPO / "synth")]
    from wirebond_inspector.anomaly import CameraChecker
    from wirebond_inspector.config import CameraCfg, load_config
    from wirebond_inspector.sources import make_camera
    cfg = load_config(REPO / "inspector" / "config.yaml")
    chk = CameraChecker(cfg.anomaly.backbone, cfg.anomaly.dir, ["top", "angled"], cfg.anomaly.size)
    out = {"good": [], "defect": [], "level": max(chk.models[c].threshold for c in ("top", "angled"))}
    for kind, inj in (("good", {"mode": "none"}), ("defect", {"mode": "random", "p": 0.08})):
        cams = [make_camera(CameraCfg(v, "generator", v, {"product_seed": 7, "injection": inj})) for v in ("top", "angled")]
        for u in range(5000, 5040):
            s = []
            for cam in cams:
                if kind == "defect" and not cam.planned_defects(u):
                    break
                r = chk.check(cam.name, cam.capture(u).image)
                s.append(r.score / chk.models[cam.name].threshold)       # 1.0 = that camera's alarm level
            if len(s) == 2:
                out[kind].append(max(s))
        print(kind, len(out[kind]), flush=True)
    (REPO / "out" / "checker_scores.json").write_text(json.dumps(out))
    return out


SPEED = [("One 2048-px photo", 2.3, 3.1), ("One unit, 2 cameras\n+ unusual-spot check", None, 3.5)]


def gallery_data() -> dict:
    """BondSight (M3k-small) on our own demo pictures: the boxes shown in the defect gallery, unedited."""
    sys.path[:0] = [str(REPO / "inspector")]
    from PIL import Image
    from wirebond_inspector.detector import OnnxDetector
    det = OnnxDetector(REPO / "training/runs/m3k/wirebond_dfine_small_m3k.onnx")
    out = {}
    for f in sorted((REPO / "demo" / "samples").glob("*.jpg")):
        found = det.detect(Image.open(f).convert("RGB"), 0.3)
        out[f.stem] = [{"cls": x.cls, "score": round(float(x.score), 3), "box": [round(v, 1) for v in x.box]} for x in found]
    (REPO / "out" / "gallery_dets.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["sweep", "checker"]
    if "sweep" in what:
        sweep_data()
    if "checker" in what:
        checker_data()
    if "gallery" in what:
        gallery_data()
