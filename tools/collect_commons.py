"""Collect freely licensed real wire-bond photos from Wikimedia Commons.

Scans categories (recursively) and full-text searches, keeps images with a free licence
(CC / public domain) and a minimum size, downloads them into real/commons/ (git-ignored),
and writes real/commons/manifest.csv with title, author, licence and source URL for
attribution.

    python tools/collect_commons.py --out real/commons
"""
from __future__ import annotations

import argparse
import csv
import html
import re
import time
import urllib.parse
import urllib.request
import json
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "wirebond-ai-dataset-collector/0.1 (educational research; github.com/wirebond-ai)"
CATEGORIES = ["Category:Wire bonding", "Category:Bond wires", "Category:Ball bonding",
              "Category:Wedge bonding", "Category:Integrated circuit dies",
              "Category:Decapped integrated circuits", "Category:Transistor dies"]
SEARCHES = ["wire bonding", "wire bond", "bond wires", "bonding wire", "ball bond",
            "wedge bond", "bonded wires die", "decapped chip wires", "die bond wire",
            "gold wire bond", "aluminium wire bond", "wire bonding macro"]
FREE = re.compile(r"^(cc0|cc[- ]by|cc[- ]by[- ]sa|public domain|pd|attribution)", re.I)
EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def api(**params) -> dict:
    params.update(format="json", formatversion="2")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {}


def category_files(cat: str, depth: int, seen: set[str]) -> set[str]:
    if cat in seen or depth < 0:
        return set()
    seen.add(cat)
    files, cont = set(), {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=cat, cmlimit="500",
                cmtype="file|subcat", **cont)
        for m in d.get("query", {}).get("categorymembers", []):
            if m["ns"] == 6:
                files.add(m["title"])
            elif m["ns"] == 14:
                files |= category_files(m["title"], depth - 1, seen)
        if "continue" not in d:
            return files
        cont = d["continue"]


def search_files(q: str, limit: int) -> set[str]:
    files, off = set(), 0
    while off < limit:
        d = api(action="query", list="search", srsearch=q, srnamespace="6", srlimit="500",
                sroffset=str(off))
        hits = d.get("query", {}).get("search", [])
        files |= {h["title"] for h in hits}
        if "continue" not in d or not hits:
            break
        off = d["continue"]["sroffset"]
    return files


def strip(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def info(titles: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(titles), 50):
        d = api(action="query", titles="|".join(titles[i:i + 50]), prop="imageinfo",
                iiprop="url|size|extmetadata", iiurlwidth="1280")  # a standard thumb step
        for p in d.get("query", {}).get("pages", []):
            ii = (p.get("imageinfo") or [{}])[0]
            md = ii.get("extmetadata", {})
            out.append({
                "title": p["title"], "width": ii.get("width", 0), "height": ii.get("height", 0),
                "url": ii.get("thumburl") or ii.get("url", ""),
                "orig": ii.get("url", ""),
                "page": ii.get("descriptionurl", ""),
                "license": strip(md.get("LicenseShortName", {}).get("value", "")),
                "author": strip(md.get("Artist", {}).get("value", ""))[:200],
                "description": strip(md.get("ImageDescription", {}).get("value", ""))[:300],
            })
        time.sleep(0.5)
    return out


def fetch(url: str) -> bytes | None:
    """Download with polite back-off on HTTP 429 (Wikimedia rate limit)."""
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                return None
            time.sleep(20 * (attempt + 1))
        except Exception:
            time.sleep(5)
    return None


TEXT_POS = ["wire bonding", "bond wires on a chip", "transistor die with bond wires",
            "decapped integrated circuit showing wires", "hybrid circuit with wire bonds",
            "die photo of a transistor", "inside of a chip package"]
TEXT_NEG = ["3D molecular model", "chemical compound", "building", "person", "map", "logo",
            "processor die shot", "printed circuit board", "cable", "landscape", "animal"]


def text_rank(rows: list[dict], keep: int) -> list[dict]:
    """Rank candidates by CLIP text similarity of title+description (no downloads)."""
    import open_clip  # needs tools/.venv
    import torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    def enc(texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                v = model.encode_text(tok(texts[i:i + 256]))
                out.append(v / v.norm(dim=-1, keepdim=True))
        return torch.cat(out)

    docs = [f"{r['title'].removeprefix('File:')} {r['description']}"[:300] for r in rows]
    D, P, N = enc(docs), enc(TEXT_POS), enc(TEXT_NEG)
    score = (D @ P.T).max(dim=1).values - 0.5 * (D @ N.T).max(dim=1).values
    order = torch.argsort(score, descending=True).tolist()
    return [rows[i] for i in order[:keep]]


def safe_name(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title.removeprefix("File:"))[:150]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="real/commons")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--search-limit", type=int, default=1500)
    ap.add_argument("--min-side", type=int, default=500)
    ap.add_argument("--rescan", action="store_true", help="ignore cached candidates.json")
    ap.add_argument("--text-rank", type=int, default=0,
                    help="download only the N candidates whose title/description best match")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    cand = out / "candidates.json"
    if cand.exists() and not a.rescan:
        rows = json.loads(cand.read_text(encoding="utf-8"))
        print(f"cached candidates: {len(rows)}", flush=True)
    else:
        titles, seen = set(), set()
        for c in CATEGORIES:
            got = category_files(c, a.depth, seen)
            print(f"{c}: {len(got)}", flush=True)
            titles |= got
        for q in SEARCHES:
            got = search_files(q, a.search_limit)
            print(f"search '{q}': {len(got)}", flush=True)
            titles |= got
        titles = sorted(t for t in titles if t.lower().endswith(EXT))
        print(f"candidate image files: {len(titles)}", flush=True)
        rows = [r for r in info(titles) if FREE.match(r["license"])
                and min(r["width"], r["height"]) >= a.min_side and r["url"]]
        cand.write_text(json.dumps(rows), encoding="utf-8")
    print(f"free licence and >= {a.min_side}px: {len(rows)}", flush=True)
    if a.text_rank:
        rows = text_rank(rows, a.text_rank)
        print(f"kept top {len(rows)} by title/description", flush=True)

    man = out / "manifest.csv"
    done = set()
    if man.exists():
        done = {r["file"] for r in csv.DictReader(man.open(encoding="utf-8"))}
    new = man.exists()
    with man.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "title", "license", "author", "page",
                                          "width", "height", "description"])
        if not new:
            w.writeheader()
        for i, r in enumerate(rows):
            name = safe_name(r["title"])
            if not name.lower().endswith(EXT):
                name += ".jpg"
            if name in done:
                continue
            if (out / "images" / name).exists():
                data = None
            else:
                # originals are static files (not rendered on demand), so they are not
                # throttled like thumbnails; use them when they are a sensible size
                small = r["width"] * r["height"] <= 12_000_000
                orig_ok = r.get("orig", "").lower().endswith((".jpg", ".jpeg", ".png"))
                data = fetch(r["orig"] if small and orig_ok else r["url"])
                if data is None:
                    print(f"skip {name}", flush=True)
                    continue
                (out / "images" / name).write_bytes(data)
            w.writerow({"file": name, **{k: r[k] for k in ("title", "license", "author", "page",
                                                           "width", "height", "description")}})
            f.flush()
            if i % 25 == 0:
                print(f"downloaded {i + 1}/{len(rows)}", flush=True)
            time.sleep(0.5)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
