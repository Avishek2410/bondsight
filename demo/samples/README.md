# Demo sample pictures

Drawn by this project's own synthetic generator (`synth/`), so they belong to this project
(same licence as the rest of the repo). No third-party photos.

- `<defect>_<view>.jpg`: a whole chip with one wire showing that defect (20 wires, design 9100).
- `macro_<defect>_<view>.jpg`: a close-up (`wbsynth generate --macro`, 16 wires, design 9200).
- `good_*.jpg`, `macro_good_*.jpg`: no defect.

How they were chosen: for every defect type, 4 units were drawn in both views and the M3k-small
model was run on all of them; it found the planted defect in 78 of 136 pictures (57%). For each
type the first picture where it did is used here, so the samples show what the AI *can* do. It
misses more on its own, and much more on real photos; see the main README for honest numbers.
