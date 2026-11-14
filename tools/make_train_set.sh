#!/usr/bin/env bash
# First synthetic training set: 30 product designs x 150 units, random defects + mould-flow sweep.
set -e
OUT=${1:-out/train_v1}
WIRES=(16 20 24 28 32 36 40 48 56 64)
for s in $(seq 1 30); do
  w=${WIRES[$(( (s - 1) % 10 ))]}
  synth/.venv/Scripts/wbsynth generate --out "$OUT/p$(printf %02d $s)" --units 150 --wires $w \
    --product-seed $((1000 + s)) --mode random --p 0.04 --flow-p 0.25 --overwrite > /dev/null
  echo "product $s/30 ($w wires) done"
done
echo "all done"
