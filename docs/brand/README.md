# BondSight look

Every picture in the docs comes from this folder, so the pages stay one design.

## Palette

Checked for colour-blind separation and contrast in light and dark (pairs that appear together:
gold/blue, blue/copper, blue/red).

| Role | Light | Dark |
|---|---|---|
| Gold wire: BondSight, the AI, "now" | `#B9800A` (bars `#D99E1F` → `#A06C00`) | `#BA8714` (bars `#EDBE5A` → `#B5800F`) |
| Slate: earlier models, history | `#A9B3BF` | `#4C5767` |
| Silicon blue: cameras, real photos, good units | `#2F6FC0` | `#4F8FDB` |
| Copper: generated pictures, people | `#C15B2C` | `#D8703F` |
| Red: **only** defects | `#D93A40` | `#E5484D` |
| Amber: alarms · Green: OK | `#E09100` · `#1E9E63` | `#F2A83B` · `#3CC487` |
| Card / line / ink | `#FFFFFF` / `#E3E6EA` / `#12161B` | `#12171E` / `#252D37` / `#EAEEF3` |
| Banner graphite | `#171A1F` | |

Type: **Archivo** throughout; condensed and heavy (`font-stretch: 72%`, 800–900) for numbers and
the wordmark.

## Files

- `figures.html`: every chart, diagram and annotated photo (`?fig=<name>&theme=light|dark`).
  Charts are drawn with D3 from `/out/figdata.json`.
- `hero.html` → `docs/images/hero.webp` (animated banner, 5 s loop) and `hero.jpg` (still), by
  `tools/figures/render_hero.py`.
- `docs/images/demo.webp`: a real recording of the online demo, by `tools/figures/record_demo.py`
  (needs the demo server on port 8090).
- `assets/`: crops used by the figures (BondSight's own generated pictures).

## Rebuild

```
python -m http.server 8765 --bind 127.0.0.1                    # repo root, keep running
tools/.venv/Scripts/python tools/figures/measure.py            # only when models change (slow)
inspector/.venv/Scripts/python tools/figures/measure.py gallery  # BondSight's boxes for the defect gallery
tools/.venv/Scripts/python tools/figures/export_data.py        # numbers -> out/figdata.json
python tools/figures/render.py                                 # -> docs/images/fig-*-{light,dark}.webp
python tools/figures/render_hero.py                            # -> docs/images/hero.webp + hero.jpg
```

The figures are transparent outside their rounded card, so each one sits on GitHub's light or dark
page; the markdown picks the matching file with `<picture>`.
