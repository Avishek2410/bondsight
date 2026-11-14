"""Record a short animation of the online demo: pick a sample photo, BondSight checks it, the defect is boxed.

    inspector/.venv/Scripts/wbinspect --config <demo config with demo: true, port 8090>   # in another terminal
    python tools/figures/record_demo.py            # -> docs/images/demo.webp

Headless Chrome over the DevTools protocol; a drawn pointer stands in for the mouse. Frames are kept with
their real timing, so the wait for the model is the real wait.
"""
from __future__ import annotations

import time

from PIL import Image

from render_hero import CHROME, REPO, Tab, chrome_tab  # noqa: F401  (shared DevTools helpers)

URL = "http://127.0.0.1:8090/"
W, H = 1280, 800
SAMPLES = ["macro_broken_wire_top.jpg", "wire_sweep_top.jpg"]
CURSOR = """(()=>{const c=document.createElement('div');c.id='rc';c.style.cssText='position:fixed;left:0;top:0;z-index:99999;pointer-events:none;transition:none';
c.innerHTML='<div id=rcr style="position:absolute;left:-18px;top:-18px;width:36px;height:36px;border-radius:50%;border:3px solid #E6B34A;opacity:0"></div><svg width=26 height=30 viewBox="0 0 26 30"><path d="M2 2 L2 24 L8 18 L12 28 L16 26 L12 17 L21 17 Z" fill="#111" stroke="#fff" stroke-width="2" stroke-linejoin="round"/></svg>';
document.body.appendChild(c);window.rcMove=(x,y)=>{c.style.transform=`translate(${x}px,${y}px)`};
window.rcClick=k=>{const r=document.getElementById('rcr');r.style.opacity=(1-k).toFixed(2);r.style.transform=`scale(${0.4+1.2*k})`}})()"""


def window(img: Image.Image) -> Image.Image:
    """Put a frame in a simple browser window (title bar with three dots), like the screenshots figure."""
    from PIL import ImageDraw
    bar, pad = 30, 1
    out = Image.new("RGB", (img.width + 2 * pad, img.height + bar + pad), "#D0D7DE")
    d = ImageDraw.Draw(out)
    d.rectangle([pad, pad, out.width - pad - 1, bar - 1], fill="#EEF0F3")
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        d.ellipse([14 + i * 18, 10, 24 + i * 18, 20], fill=c)
    d.text((84, 9), "bondsight · online demo", fill="#8A939F")
    out.paste(img, (pad, bar))
    return out


def main():
    import io, base64
    frames, stamps = [], []
    with chrome_tab(URL, W, H) as tab:
        ev = lambda js: tab.call("Runtime.evaluate", expression=js, returnByValue=True, awaitPromise=True)["result"].get("value")
        for _ in range(100):
            if ev("!!document.querySelector('#sample-row .sample img') && [...document.images].every(i=>i.complete)"):
                break
            time.sleep(0.1)
        ev(CURSOR)
        t0 = time.time()

        def snap():
            png = tab.call("Page.captureScreenshot", format="jpeg", quality=90, optimizeForSpeed=True)["data"]
            frames.append(Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB"))
            stamps.append(time.time() - t0)

        pos = [W * 0.62, H * 0.82]

        def move(to, secs):
            start, a = time.time(), list(pos)
            while True:
                k = min(1.0, (time.time() - start) / secs)
                e = 1 - (1 - k) ** 3
                pos[:] = [a[0] + (to[0] - a[0]) * e, a[1] + (to[1] - a[1]) * e]
                ev(f"rcMove({pos[0]:.1f},{pos[1]:.1f})")
                snap()
                if k >= 1:
                    return

        def hold(secs):
            end = time.time() + secs
            while time.time() < end:
                snap()

        def click(selector):
            for k in (0.0, 0.35, 0.7, 1.0):
                ev(f"rcClick({k})")
                snap()
            ev(f"document.querySelector({selector!r}).click()")

        def centre(selector):
            r = ev(f"(()=>{{const r=document.querySelector({selector!r}).getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]}})()")
            return r

        ev(f"rcMove({pos[0]},{pos[1]})")
        hold(0.8)
        for i, name in enumerate(SAMPLES):
            if i:
                move(centre("#back"), 0.7)
                click("#back")
                hold(0.6)
            sel = f'#sample-row [data-s="{name}"]'
            ev(f"document.querySelector({sel!r}).scrollIntoView({{block:'center'}})")
            move(centre(sel), 0.9)
            click(sel)
            while not ev("document.querySelector('#verdict .v').textContent.trim()!=='READY' && !document.getElementById('back').hidden"):
                snap()
                if time.time() - t0 > 60:
                    raise RuntimeError("no result from the demo server")
            ev("window.scrollTo(0,0)")
            hold(0.4)
            move(centre("#verdict"), 0.8)
            hold(2.2)
    # frames keep their real timing; trim each to at least 40 ms
    durs = [max(40, int(1000 * (b - a))) for a, b in zip(stamps, stamps[1:])] + [1200]
    small = [window(f.resize((960, 600), Image.LANCZOS)) for f in frames]
    out = REPO / "docs" / "images" / "demo.webp"
    small[0].save(out, save_all=True, append_images=small[1:], duration=durs, loop=0, quality=80, method=4)
    print(out.name, len(frames), "frames,", round(stamps[-1], 1), "s,", out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
