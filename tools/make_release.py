"""Assemble the model release in out/release/<version>/ (uploaded to GitHub Releases, not to git).

    python tools/make_release.py v0.1.0
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = [
    "training/runs/m3k/wirebond_dfine_small_m3k.onnx",
    "training/runs/m3k/wirebond_dfine_small_m3k.json",
    "training/runs/anomaly/anomaly_backbone.onnx",
    "training/runs/anomaly/anomaly_backbone.json",
    "CREDITS.md",
    "LICENSE",
]


def main(version: str):
    out = REPO / "out" / "release" / version
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(REPO / f, out / Path(f).name)
    with zipfile.ZipFile(out / "anomaly-product7.zip", "w", zipfile.ZIP_DEFLATED) as z:   # memories for the simulated cameras
        for npz in sorted((REPO / "training/runs/anomaly/product7").glob("*.npz")):
            z.write(npz, f"anomaly/{npz.name}")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(out.iterdir())]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    for p in sorted(out.iterdir()):
        print(f"{p.stat().st_size / 1e6:7.1f} MB  {p.name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v0.1.0")
