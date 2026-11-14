# Credits

This project stands on other people's work. Thank you to everyone listed here.

The model files we publish (`wirebond_dfine_small_m3k.onnx`, the earlier `wirebond_dfine_nano_joint.onnx`,
`anomaly_backbone.onnx` and the
files they were made from) were **trained on or built from** the works below. If you share these
model files, share this file with them. Third-party photos themselves are **not** stored in this
repository; download them from the links below.

## Photos used to train the defect AI (models M3j and M3k)

All six are licensed **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. The first three
trained both M3j and M3k; the last three were added for M3k.

| Work | Made by | Link | What we changed |
|---|---|---|---|
| *wire bonding*, version 9 | National Taiwan University of Science and Technology (Roboflow workspace `national-taiwan-university-of-science-and-technology-l3dyy`) | <https://universe.roboflow.com/national-taiwan-university-of-science-and-technology-l3dyy/wire-bonding-yaaxf> | Shrunk to half size and cut into overlapping 640 × 640 tiles; labels renamed (`sagged wire` → `wire_sweep`, `missing wire` → `missing_wire`); `good wire` boxes kept only as known-good areas, not as defects. |
| *mems_60_OneTypeOneBox*, version 1 | WHUrobotics (Roboflow workspace `whurobotics`) | <https://universe.roboflow.com/whurobotics/mems_60_onetypeonebox> | Whole photos plus close-up crops around each broken wire and crops with no defect; both broken-wire labels mapped to `broken_wire`; other labels dropped. |
| *new_photos*, version 3 | WHUrobotics (Roboflow workspace `whurobotics`) | <https://universe.roboflow.com/whurobotics/new_photos> | Shrunk to half size and cut into tiles; label `aluminum wire bonding broken` mapped to `missing_wire` (the boxes mark bond pads whose wire is gone); most tiles without a defect left out. |
| *Wire* | hawk-team (Roboflow workspace `hawk-team`) | <https://universe.roboflow.com/hawk-team/wire-u866o> | Only complete boards kept (wire count within one of the usual count); every wire box used as a known-good area; shrunk to half size and cut into tiles. |
| *chip* | lyra-wikbf (Roboflow workspace `lyra-wikbf`) | <https://universe.roboflow.com/lyra-wikbf/chip-bepjf> | Whole photos used as defect-free examples; original labels not used. |
| *Fuji Wire Inspection* | pg-87jvm (Roboflow workspace `pg-87jvm`) | <https://universe.roboflow.com/pg-87jvm/fuji-wire-inspection> | A subset (382 training, 160 test photos) shrunk to half size; `Defect` boxes (wires bent out of their lane) mapped to `wire_sweep`; `Pass` modules used as defect-free examples. |

The rest of the training pictures are synthetic, drawn by this project's own generator (`synth/`).

## Pre-trained networks we start from

| Work | Made by | Licence | Used for |
|---|---|---|---|
| D-FINE Small, COCO weights (`ustc-community/dfine-small-coco`) | Peng et al., *D-FINE* (2024), weights published by ustc-community on Hugging Face | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Starting point of the current defect AI (M3k-small); we fine-tuned it on the photos above. |
| D-FINE Nano, COCO weights (`ustc-community/dfine-nano-coco`) | Peng et al., *D-FINE* (2024), weights published by ustc-community on Hugging Face | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Starting point of the earlier defect AI (M3j) and of M3k-nano. |
| ResNet18, ImageNet weights (torchvision) | He et al., *Deep Residual Learning* (2016); weights by the PyTorch / torchvision team | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | Feature extractor of the unusual-spot checker (unchanged, cut after layer 3). |

The unusual-spot checker follows the idea of *PatchCore* (Roth et al., "Towards Total Recall in
Industrial Anomaly Detection", CVPR 2022). The code here is our own.

## Photos used only for testing

These were used to measure the models, not to train them, and are not redistributed:

- The other Roboflow Universe sets listed in [DATA_SOURCES.md](DATA_SOURCES.md) (CC BY 4.0 or Public
  Domain), credited there with links.
- 121 photos from Wikimedia Commons, Flickr (via Openverse) and TTU-HEP/Tornado (CC BY, CC BY-SA, CC0,
  GPL-3.0). Author, licence and link for each photo are recorded in `real/curated/ATTRIBUTION.csv`, which
  `tools/collect_commons.py` and `tools/collect_openverse.py` rebuild.

## Suggested credit line

If you use the models, a short credit is enough, for example:

> BondSight wire-bond defect models by Dr. Avishek Mondal (BondSight, MIT). Trained with photos from National
> Taiwan University of Science and Technology, WHUrobotics, hawk-team, lyra-wikbf and pg-87jvm (Roboflow
> Universe, CC BY 4.0), starting from D-FINE (Apache-2.0) and ResNet18 (BSD-3-Clause). Full credits:
> CREDITS.md.
