"""Find and download wire-bond datasets from Roboflow Universe.

Searches Universe with many queries, keeps projects whose name or classes look wire-bond
related, records image counts / classes / licence in real/roboflow/catalog.csv, then
downloads the latest version of each kept project as COCO into real/roboflow/<ws>__<proj>/.

API key is read from ~/.config/roboflow/api_key (never stored in the repo).

    tools/.venv/Scripts/python tools/collect_roboflow.py --out real/roboflow
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

QUERIES = ["wire bond", "wirebond", "wire bonding", "bonding wire", "bond wire", "wire sweep",
           "lifted ball", "lifted stitch", "ball bond", "wedge bond", "stitch bond",
           "wire bond defect", "wire bonding defect", "bond wire defect", "wire x-ray",
           "wire bond inspection", "die attach", "ic package defect", "semiconductor package",
           "semiconductor defect", "chip packaging defect", "leadframe", "qfn", "wire short",
           "broken wire", "missing wire", "bonding", "aoi semiconductor", "capillary",
           "gold wire", "copper wire bond", "ic x-ray", "package x-ray", "mems defect"]
# wire-bond specific words only ("ball", "lead", "die" alone match sports/other domains)
RELEVANT = re.compile(r"wire|bond|stitch|sweep|semicon|mems|capillar", re.I)
UA = "wirebond-ai-collector/0.1"


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    err = "rate limited"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            try:
                return json.loads(e.read())
            except Exception:
                return {"error": str(e)}
        except Exception as e:
            time.sleep(3)
            err = str(e)
    return {"error": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="real/roboflow")
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=5000, help="skip bigger projects")
    ap.add_argument("--catalog-only", action="store_true", help="list projects, download nothing")
    ap.add_argument("--only", default="", help="comma-separated ws/project slugs: skip search, "
                    "download exactly these (one export each)")
    a = ap.parse_args()
    key = (Path.home() / ".config/roboflow/api_key").read_text().strip()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    found: dict[str, dict] = {}
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else []
    for s in only:
        found[s] = {}
    for q in ([] if only else QUERIES):
        for page in range(1, a.pages + 1):
            d = get("https://api.roboflow.com/universe/search?" + urllib.parse.urlencode(
                {"q": q, "page": page, "api_key": key}))
            res = d.get("results", [])
            for r in res:
                slug = urllib.parse.urlparse(r.get("url", "")).path.strip("/")
                if slug.count("/") == 1:
                    found.setdefault(slug, r)
            if len(res) < d.get("page_size", 12):
                break
            time.sleep(0.5)
        print(f"'{q}': total {len(found)}", flush=True)

    rows = []
    for slug in sorted(found):
        d = get(f"https://api.roboflow.com/{slug}?api_key={key}")
        if "project" not in d:
            continue
        pr, vs = d["project"], d.get("versions", [])
        classes = list((pr.get("classes") or {}).keys())
        text = " ".join([slug, pr.get("name", "")] + classes)
        rows.append({"slug": slug, "name": pr.get("name"), "type": pr.get("type"),
                     "images": pr.get("images", 0), "license": pr.get("license"),
                     "classes": "|".join(classes)[:400],
                     "latest": vs[0]["id"].split("/")[-1] if vs else "",
                     "relevant": bool(RELEVANT.search(text))})
        time.sleep(1.0)
    # catalog.csv is rewritten by a search; selected.csv only grows (one row per picked set)
    target = out / ("selected.csv" if only else "catalog.csv")
    prev = []
    if only and target.exists():
        with target.open(newline="", encoding="utf-8") as fh:
            prev = [x for x in csv.DictReader(fh) if x["slug"] not in only]
    with target.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(prev + rows)
    keep = [r for r in rows if (only or r["relevant"]) and r["latest"]
            and 0 < r["images"] <= a.max_images]
    ok_lic = re.compile(r"^(cc by( 4\.0)?|cc by-sa|cc0|public domain|mit)", re.I)
    for r in [r for r in keep if not ok_lic.match(r["license"] or "")]:
        print(f"SKIP (licence not redistributable/unclear): {r['slug']} {r['license']}", flush=True)
    keep = [r for r in keep if ok_lic.match(r["license"] or "")]
    print(f"projects: {len(rows)}, relevant with a version: {len(keep)}", flush=True)
    if a.catalog_only:
        return 0

    for r in keep:
        dest = out / r["slug"].replace("/", "__")
        if (dest / "DONE").exists():
            continue
        fmt = "folder" if r["type"] == "classification" else "coco"
        url = f"https://api.roboflow.com/{r['slug']}/{r['latest']}/{fmt}?api_key={key}"
        d = get(url)
        for _ in range(60):                 # big exports are prepared first ("progress"): wait up to ~20 min
            if (d.get("export") or {}).get("link") or "progress" not in d:
                break
            print(f"   {r['slug']}: export being prepared ({d['progress']})", flush=True)
            time.sleep(20)
            d = get(url)
        link = (d.get("export") or {}).get("link")
        if not link:
            print(f"no export for {r['slug']}: {str(d)[:120]}", flush=True)
            continue
        tmp = out / (dest.name + ".zip.part")      # stream to disk: some exports are several GB
        try:
            with urllib.request.urlopen(urllib.request.Request(link, headers={"User-Agent": UA}),
                                        timeout=600) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh, 1 << 20)
            with zipfile.ZipFile(tmp) as z:
                z.extractall(dest)
            tmp.unlink()
        except Exception as e:
            print(f"download failed {r['slug']}: {e}", flush=True)
            continue
        (dest / "SOURCE.json").write_text(json.dumps(r, indent=1))
        (dest / "DONE").write_text("ok")
        print(f"downloaded {r['slug']} ({r['images']} imgs, {r['license']})", flush=True)
        time.sleep(1)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
