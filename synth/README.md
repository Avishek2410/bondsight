# wirebond-synth

Synthetic wire-bond images with labelled defects, for training defect detectors when no real
factory images are available.

- Two camera views: `top` (straight down) and `angled` (45°, shows loop height and lifted bonds)
- 8 defect classes: `wire_sweep, lifted_ball, lifted_stitch, missing_wire, broken_wire,
  off_center_ball, wire_short, sagging_loop`
- Two kinds of defect injection:
  - `random`: any wire, any unit (common-cause noise)
  - `repetitive`: the same wire position keeps failing (assignable cause, e.g. a worn capillary)
- Output: PNG images and COCO annotations per view, plus a JSON file per unit saying exactly
  which wire has which defect. Fully reproducible from the seeds.

## Install

```bash
cd synth
python -m venv .venv
.venv/Scripts/activate      # Windows; on Linux/Pi: source .venv/bin/activate
pip install -e ".[test]"
```

## Use

```bash
# one example image per defect class
wbsynth gallery --out samples

# 2,000 units, each wire has a 3 % chance of a random defect
wbsynth generate --out out/train --units 2000 --mode random --p 0.03

# wire 5 keeps sweeping in 80 % of units from unit 100 on, plus 1 % random background defects
wbsynth generate --out out/rep --units 500 --mode repetitive --wire 5 --cls wire_sweep \
    --fraction 0.8 --start-unit 100 --random-p 0.01
```

`--overwrite` only ever replaces a folder that `wbsynth` itself created.

## Test

```bash
pytest tests -q
```

Licence: MIT.
