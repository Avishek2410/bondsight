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

# BondSight: open-source AOI for wire bonds (online demo)

Drop a photo of wire-bonded chips, or click one of the sample pictures, and the AI draws a box
around each defect it finds (wire sweep, lifted ball or stitch, missing or broken wire, off-centre
ball, wire short, sagging loop). **Nothing you upload is saved.**

BondSight is an open research prototype. The detector (D-FINE Small,
fine-tuned) was trained on synthetic pictures from the project's own generator and on a few hundred
public photos. Honest limits:

- On real photos it finds roughly 6 to 8 out of 10 defects, depending on the photo set.
- It is not trained on your product, camera or lighting, so expect misses and false alarms.
- Do not use it for production decisions.

The sample pictures were drawn by the project's generator. Who made the photos and networks the
model learned from is listed in `CREDITS.md` (CC BY 4.0 photos, Apache-2.0 D-FINE, BSD-3 ResNet18).
Code: MIT.
