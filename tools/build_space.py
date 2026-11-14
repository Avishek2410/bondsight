"""Assemble the online demo as a Hugging Face Space folder (Docker SDK). Builds only; never uploads.

What: copies into one folder everything the Space needs:
  README.md + Dockerfile (from demo/space/), config.yaml (demo mode, port 7860), the model
  (.onnx + .json), the demo sample pictures (demo/samples/, our own synthetic images),
  CREDITS.md, LICENSE, and the synth/ + inspector/ code (no virtual environments, tests, data).
Why: the Space must contain the model and its credits, and must never contain a password file
or third-party photos. The builder checks that at the end and fails if anything private slipped in.
Publishing the folder is a separate step that needs the owner's account and explicit yes.

    tools/.venv/Scripts/python tools/build_space.py --out out/space
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARKER = ".wb-space"
DEFAULT_MODEL = REPO / "training" / "runs" / "m3k" / "wirebond_dfine_small_m3k.onnx"
IGNORE = shutil.ignore_patterns(".venv", "__pycache__", "*.egg-info", ".pytest_cache", "data", "tests",
                                "*.pyc", ".env", "pgdata", "samples", "out")
FORBIDDEN = (".env", "real", ".venv", "data", "__pycache__", ".git", "pgdata")


def build(out: Path, model: Path) -> None:
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not (out / MARKER).exists():
        raise SystemExit(f"{out} is not empty and was not made by this builder; choose another --out")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("made by tools/build_space.py\n")
    for f in ("README.md", "Dockerfile"):
        shutil.copyfile(REPO / "demo" / "space" / f, out / f)
    for f in ("CREDITS.md", "LICENSE"):
        shutil.copyfile(REPO / f, out / f)
    for d in ("synth", "inspector"):
        shutil.copytree(REPO / d, out / d, ignore=IGNORE)
    (out / "models").mkdir()
    for p in (model, model.with_suffix(".json")):
        if not p.is_file():
            raise SystemExit(f"model file not found: {p}")
        shutil.copyfile(p, out / "models" / p.name)
    (out / "samples").mkdir()
    for p in sorted((REPO / "demo" / "samples").glob("*")):
        if p.is_file():
            shutil.copyfile(p, out / "samples" / p.name)
    (out / "config.yaml").write_text(
        "# Online demo: visitor mode, nothing saved (see inspector/README.md)\n"
        f"model: models/{model.name}\n"
        "demo: true\nsamples: samples\ndata_dir: /tmp/wb-demo\nport: 7860\n")
    bad = [p.relative_to(out).as_posix() for p in out.rglob("*")
           if any(part in FORBIDDEN for part in p.relative_to(out).parts)]
    if bad:
        shutil.rmtree(out)
        raise SystemExit(f"refusing: private or third-party files would be included: {bad[:5]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "out" / "space")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    a = ap.parse_args()
    build(a.out, a.model)
    size = sum(p.stat().st_size for p in a.out.rglob("*") if p.is_file()) / 1e6
    print(f"Space folder ready: {a.out} ({size:.0f} MB). Not uploaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
