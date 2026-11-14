# Data sources

Third-party images are **not** stored in this repository. Rule: only data that may be published and
redistributed (CC BY, CC0/public domain, CC BY-SA, GPL). Each source keeps its own licence.
Which of these trained the published models, and the credit each one needs: [CREDITS.md](CREDITS.md).

## Roboflow Universe datasets

| Dataset | Images | Licence | Classes |
|---|---|---|---|
| [apd-lab-ne-images-3/wirebond](https://universe.roboflow.com/apd-lab-ne-images-3/wirebond) (v1) | 98 | CC BY 4.0 | No-wires |
| [my-own-nggxn/m11-12-13-wire-bonded](https://universe.roboflow.com/my-own-nggxn/m11-12-13-wire-bonded) (v1) | 294 | CC BY 4.0 | Foobar, Glue, Wires |
| [national-taiwan-university-of-science-and-technology-l3dyy/wire-bonding-yaaxf](https://universe.roboflow.com/national-taiwan-university-of-science-and-technology-l3dyy/wire-bonding-yaaxf) (v9) | 113 | CC BY 4.0 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, broken wire, missing wire, good wire, sagged wire |
| [parmi-gkxmb/wire-nnmal](https://universe.roboflow.com/parmi-gkxmb/wire-nnmal) (v3) | 149 | Public Domain | WIRE, DIE, THIN_WIRE, THICK_WIRE |
| [reebas-workspace/wirebond-defect-detection](https://universe.roboflow.com/reebas-workspace/wirebond-defect-detection) (v2) | 41 | CC BY 4.0 | 0, 1 |
| [whurobotics/mems_60_onetypeonebox](https://universe.roboflow.com/whurobotics/mems_60_onetypeonebox) (v1) | 61 | CC BY 4.0 | aluminum_wire_bonding_broken, chip_injury, gold_wire_bonding_broken, chip_scratch, glue_surface_wrinkle |
| [whurobotics/memsregions](https://universe.roboflow.com/whurobotics/memsregions) (v8) | 61 | CC BY 4.0 | aluminum-bond, chip-left-dwon, chip-left-up, gold-bond, circuit, chip, chip-core |
| [wire-xray-2/wire-x-ray-48l](https://universe.roboflow.com/wire-xray-2/wire-x-ray-48l) (v4) | 342 | CC BY 4.0 | conductive-ink, ball-short-metal, double-boad, inner-shift-lead, broken-ball-neck, depress-wire, lifted-stitch, good, da |
| [wire-xray/wire-x-ray](https://universe.roboflow.com/wire-xray/wire-x-ray) (v1) | 500 | CC BY 4.0 | foreign-material, wire-touch, wire sweep, double-boad, broken-ball-neck, good, conductive-ink, ball-short-metal, inner-s |
| [work-qj6em/wirebond-qc](https://universe.roboflow.com/work-qj6em/wirebond-qc) (v2) | 50 | CC BY 4.0 | YOLOv5-YOLOv8, stephole with, stephole without, not stephole |

## Curated real photos (Wikimedia Commons, Flickr via Openverse, TTU-HEP/Tornado)

121 images; per-image author, licence and link in `real/curated/ATTRIBUTION.csv`
(regenerate with `tools/collect_commons.py`, `tools/collect_openverse.py`, `tools/rank_by_examples.py`,
`tools/curated_rejects.txt`). TTU-HEP/Tornado images: GPL-3.0, https://github.com/TTU-HEP/Tornado.

## Reproduce

```bash
tools/.venv/Scripts/python tools/collect_roboflow.py --out real/roboflow --only "<slugs above>"
```
Needs your own Roboflow API key in `~/.config/roboflow/api_key`.
