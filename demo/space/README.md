---
title: BondSight demo
emoji: 🔬
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
license: mit
short_description: Open-source AOI for wire bonds - drop a photo, see the defects.
---

# BondSight: open-source AOI for wire bonds

Drop a photo of wire-bonded chips, or click one of the sample pictures, and BondSight draws a box
around each defect it finds: wire sweep, sagging loop, lifted ball, lifted stitch, off-centre
ball, missing wire, broken wire or wire short. Nothing you upload is saved.

This is a research prototype, so please read the limits before you trust a result. The detector
(D-FINE Small, fine-tuned) learned from pictures drawn by the project's own generator and from a few
hundred public photos. On real photo sets it finds roughly 6 to 8 out of 10 defects. It has never
seen your product, camera or lighting, so expect some misses and some false alarms, and don't use
it for production decisions.

The sample pictures were drawn by the project's generator. The people and projects whose photos
and networks the model learned from are credited in `CREDITS.md` (CC BY 4.0 photos, Apache-2.0
D-FINE, BSD-3 ResNet18). Code under the MIT licence.

Code, model, results and how it works: https://github.com/Avishek2410/bondsight
