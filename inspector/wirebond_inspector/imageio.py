"""Make any picture into plain 8-bit RGB.

What: cameras and image files come in many formats: grayscale, with transparency (RGBA),
palette (P), or 16-bit microscope images. The model only understands 8-bit RGB.
Why: the first reality check crashed on exactly such images, so every picture goes through
`to_rgb8` before the detector sees it. 16-bit values are scaled down (0..65535 -> 0..255),
not cut off, so a bright 16-bit image stays bright.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def to_rgb8(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        a = np.asarray(img).astype(np.float64)
        if img.mode.startswith("I;16") or a.max(initial=0) > 255:
            a = a / 257.0                       # 16-bit range -> 8-bit range
        elif img.mode == "F" and a.max(initial=0) <= 1.0:
            a = a * 255.0                       # float image in 0..1
        a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
        return Image.fromarray(a, "L").convert("RGB")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (0, 0, 0))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")
