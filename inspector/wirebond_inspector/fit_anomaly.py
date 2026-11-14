"""Build the anomaly checker's memory for every camera:  wbinspect-fit-anomaly --config config.yaml

What it needs: pictures of GOOD units only, per camera.
  - generator cameras: we draw good units ourselves (same product, defects switched off),
  - folder / real cameras: put good pictures in  <--good-dir>/<camera name>/  (jpg/png).
The pictures are split: most build the memory, the rest (never seen by the memory) set the
alarm level = their highest score x margin. So by construction good units sit below it.
Output: <anomaly.dir>/<camera>.npz (a few MB each). Restart the inspector to use them.
Why a separate command: building the memory takes a minute or two per camera and only needs
redoing when the product, lighting or camera changes - not at every start.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from .anomaly import AnomalyModel, FeatureExtractor
from .config import ConfigError, load_config

PICS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
GOOD_UNIT_BASE = 1_000_000        # unit numbers far away from the ones the inspector uses


def good_pictures(cam, n: int, good_dir: Path | None) -> list[Image.Image]:
    if cam.type == "generator":
        from .sources.generator import GeneratorCamera
        g = GeneratorCamera(cam.name, cam.view, {**cam.options, "injection": {"mode": "none"}})
        return [g.capture(GOOD_UNIT_BASE + i).image for i in range(n)]
    folder = (good_dir / cam.name) if good_dir else None
    if folder is None or not folder.is_dir():
        raise FileNotFoundError(f"put good pictures for camera {cam.name!r} in <--good-dir>/{cam.name}/")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in PICS)[:n]
    return [Image.open(p).convert("RGB") for p in files]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wbinspect-fit-anomaly", description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--good", type=int, default=80, help="good pictures per camera (memory + alarm level)")
    ap.add_argument("--holdout", type=float, default=0.25, help="share kept apart to set the alarm level")
    ap.add_argument("--margin", type=float, default=1.0, help="alarm level = highest held-out good score x margin")
    ap.add_argument("--bank", type=int, default=10000, help="patches kept in each memory")
    ap.add_argument("--good-dir", type=Path, help="folder with one sub-folder of good pictures per real camera")
    ap.add_argument("--camera", action="append", help="only these cameras (repeatable)")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    if cfg.anomaly is None:
        print("config has no `anomaly:` section (backbone + dir) - nothing to build", file=sys.stderr)
        return 2
    ext = FeatureExtractor(cfg.anomaly.backbone, cfg.anomaly.size)
    failed = 0
    for cam in cfg.cameras:
        if a.camera and cam.name not in a.camera:
            continue
        try:
            pics = good_pictures(cam, a.good, a.good_dir)
        except Exception as e:
            print(f"{cam.name}: skipped - {e}")
            failed += 1
            continue
        k = max(1, round(len(pics) * a.holdout))
        if len(pics) - k < 1:
            print(f"{cam.name}: skipped - need at least 2 good pictures, found {len(pics)}")
            failed += 1
            continue
        print(f"{cam.name}: {len(pics) - k} pictures -> memory, {k} -> alarm level ...", flush=True)
        m = AnomalyModel.fit(ext, pics[:-k], pics[-k:], bank_size=a.bank, margin=a.margin)
        out = cfg.anomaly.dir / f"{cam.name}.npz"
        m.info.update({"camera": cam.name, "view": cam.view, "size": ext.size})
        m.save(out)
        print(f"{cam.name}: saved {out}  alarm level {m.threshold:.3f} "
              f"(good held-out: mean {m.info['holdout_mean']:.3f}, max {m.info['holdout_max']:.3f})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
