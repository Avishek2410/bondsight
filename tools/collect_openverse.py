"""Collect openly licensed wire-bond photos via Openverse (Flickr and other CC sources).

Wikimedia results are skipped (tools/collect_commons.py covers them). Every image is kept
with its licence and creator in real/openverse/manifest.csv. NC/ND-licensed images are fine
for private evaluation but must never be redistributed or published.

    python tools/collect_openverse.py --out real/openverse
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.openverse.org/v1/images/"
UA = "wirebond-ai-dataset-collector/0.1 (educational research)"
QUERIES = ["wire bonding", "wire bond", "bond wires", "bonding wire", "ball bond",
           "wedge bond", "die shot wires", "decapped chip", "decapped ic", "transistor die",
           "bonding wire microscope", "chip gold wires", "ic die bond", "integrated circuit die",
           "semiconductor die macro", "gold wire chip", "aluminium wire bond", "led die wire",
           "chip package inside", "microchip macro wires", "opened chip", "die photo"]


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            return {}
        except Exception:
            time.sleep(5)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="real/openverse")
    ap.add_argument("--pages", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    for q in QUERIES:
        n0 = len(found)
        for page in range(1, a.pages + 1):
            d = get(API + "?" + urllib.parse.urlencode({"q": q, "page_size": 20, "page": page}))
            res = d.get("results", [])
            for r in res:
                if r.get("source") != "wikimedia" and r.get("url"):
                    found.setdefault(r["id"], r)
            if len(res) < 20 or page >= d.get("page_count", 0):
                break
            time.sleep(1.5)
        print(f"'{q}': +{len(found) - n0} (total {len(found)})", flush=True)

    man = out / "manifest.csv"
    done = {r["id"] for r in csv.DictReader(man.open(encoding="utf-8"))} if man.exists() else set()
    new_file = not man.exists()
    with man.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "id", "title", "license", "license_version",
                                          "creator", "source", "landing_url", "query_tags"])
        if new_file:
            w.writeheader()
        for i, (oid, r) in enumerate(found.items()):
            if oid in done:
                continue
            ext = Path(urllib.parse.urlparse(r["url"]).path).suffix.lower()
            ext = ext if ext in (".jpg", ".jpeg", ".png", ".webp") else ".jpg"
            name = f"{r.get('source', 'x')}_{oid[:12]}{ext}"
            try:
                req = urllib.request.Request(r["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                if len(data) < 5000:
                    continue
                (out / "images" / name).write_bytes(data)
            except Exception as e:
                print(f"skip {oid}: {e}", flush=True)
                continue
            tags = ",".join(t.get("name", "") for t in (r.get("tags") or [])[:15])
            w.writerow({"file": name, "id": oid, "title": re.sub(r"\s+", " ", r.get("title") or ""),
                        "license": r.get("license"), "license_version": r.get("license_version"),
                        "creator": r.get("creator"), "source": r.get("source"),
                        "landing_url": r.get("foreign_landing_url"), "query_tags": tags})
            f.flush()
            if i % 25 == 0:
                print(f"downloaded {i + 1}/{len(found)}", flush=True)
            time.sleep(0.5)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
