"""Render the animated banner docs/brand/hero.html -> docs/images/hero.webp (+ a still hero.jpg).

    python -m http.server 8765 --bind 127.0.0.1      # repo root, in another terminal
    python tools/figures/render_hero.py

One headless Chrome is driven over the DevTools protocol: for each frame the page's frame(t) is called and
the 1500 x 500 banner is captured. Needs the websocket-client package.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
URL = "http://127.0.0.1:8765/docs/brand/hero.html?t=0"
PORT, FPS, LOOP, SCALE = 9333, 15, 5.0, 1.0      # SCALE 1 keeps the animation small; the still is 2x


class Tab:
    def __init__(self, ws_url):
        self.ws, self.n = websocket.create_connection(ws_url, timeout=60), 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})


@contextlib.contextmanager
def chrome_tab(url: str, width: int, height: int, scale: float = 1.0):
    """A headless Chrome with one tab on url, driven over the DevTools protocol."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        chrome = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={PORT}",
                                   f"--user-data-dir={tmp}", f"--window-size={width},{height}", "about:blank"])
        try:
            for _ in range(50):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
                    break
                except OSError:
                    time.sleep(0.2)
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?{url}", method="PUT")
            tab = Tab(json.loads(urllib.request.urlopen(req).read())["webSocketDebuggerUrl"])
            tab.call("Page.enable")
            tab.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale, mobile=False)
            yield tab
        finally:
            chrome.kill()
            chrome.wait(10)


def shot(tab, t, scale):
    tab.call("Emulation.setDeviceMetricsOverride", width=1500, height=500, deviceScaleFactor=scale, mobile=False)
    tab.call("Runtime.evaluate", expression=f"frame({t})")
    png = tab.call("Page.captureScreenshot", format="png", clip={"x": 0, "y": 0, "width": 1500, "height": 500, "scale": 1})["data"]
    return Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB")


def main():
    with chrome_tab(URL, 1500, 500) as tab:
        for _ in range(100):                       # wait for fonts and pictures
            r = tab.call("Runtime.evaluate", expression="document.body.dataset.ready==='1' && [...document.images].every(i=>i.complete)", returnByValue=True)
            if r["result"].get("value"):
                break
            time.sleep(0.1)
        frames = [shot(tab, i / FPS, SCALE) for i in range(int(LOOP * FPS))]
        still = shot(tab, 3.6, 2.0)
    out = REPO / "docs" / "images"
    frames[0].save(out / "hero.webp", save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0,
                   quality=82, method=4)
    still.save(out / "hero.jpg", quality=90)
    print("hero.webp", len(frames), "frames,", (out / "hero.webp").stat().st_size // 1024, "KB; hero.jpg",
          (out / "hero.jpg").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
