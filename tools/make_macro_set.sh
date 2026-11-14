#!/usr/bin/env bash
# Macro (close-up) synthetic set for M2e: 24 product designs x 50 units x 2 views = 2400 images.
# Even designs: one wire keeps breaking (70% of units) + a few random defects, so broken wires
# - the defect the MEMS photos need - are common. Odd designs: random defects of all 8 classes.
set -e
OUT=${1:-out/macro_v1}
WIRES=(8 12 16 20 24 28 32 40)
for s in $(seq 1 24); do
  w=${WIRES[$(( (s - 1) % 8 ))]}
  common=(--out "$OUT/p$(printf %02d $s)" --units 50 --wires $w --product-seed $((2000 + s))
          --macro --format jpg --overwrite)
  if (( s % 2 == 0 )); then
    synth/.venv/Scripts/wbsynth generate "${common[@]}" --mode repetitive --wire $(( (s * 7) % w )) \
      --cls broken_wire --fraction 0.7 --random-p 0.03 > /dev/null
  else
    synth/.venv/Scripts/wbsynth generate "${common[@]}" --mode random --p 0.06 --flow-p 0.15 > /dev/null
  fi
  echo "product $s/24 ($w wires) done"
done
echo "all done"
