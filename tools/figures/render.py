"""Render docs/brand/figures.html into docs/images/fig-<name>-{light,dark}.webp (headless Chrome, 2x).

    python -m http.server 8765 --bind 127.0.0.1      # from the repo root, in another terminal
    python tools/figures/render.py                   # all figures
    python tools/figures/render.py results sweep     # some

Outside the rounded card the picture is transparent, so each figure sits on GitHub's light or dark page.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "images"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE = "http://127.0.0.1:8765/docs/brand/figures.html"

# name: (width, height) in CSS px; the PNG is twice that
FIGS = {
    "results": (1400, 580), "sweep": (1400, 640), "curves": (1400, 580), "mix": (1400, 560),
    "levels": (1400, 540), "checker": (1400, 560), "repeat": (1400, 380), "speed": (1400, 420),
    "how": (1400, 620), "deploy": (1400, 610), "pipeline": (1400, 560), "dfine": (1400, 470),
    "journey": (1400, 440), "tiling": (1400, 640), "unusual": (1400, 610), "screens": (1400, 1110),
    "glance": (1400, 214), "gallery": (1400, 1200),
}


def render(name: str, theme: str) -> Path:
    w, h = FIGS[name]
    dst = OUT / f"fig-{name}-{theme}.webp"
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "shot.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--user-data-dir={Path(tmp) / 'profile'}", "--default-background-color=00000000", f"--window-size={w},{h}",
                        "--virtual-time-budget=8000", f"--screenshot={png}", f"{BASE}?fig={name}&theme={theme}&w={w}&h={h}"],
                       check=True, capture_output=True, timeout=120)
        Image.open(png).save(dst, "WEBP", quality=90, method=4)   # ~10x smaller than PNG, keeps transparency
    return dst


def main():
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        for theme in ("light", "dark"):
            print(render(n, theme).name, flush=True)


if __name__ == "__main__":
    main()
