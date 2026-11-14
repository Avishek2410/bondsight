"""Get an uploaded photo ready for the AI, and say plainly if it is a poor photo.

What: `load_photo` turns the bytes of an uploaded file into an upright 8-bit RGB picture.
Phones store photos sideways and add a note "rotate me" (EXIF orientation); we apply it, so
the boxes land where the person sees the defect. `quality_hints` warns when the photo is
blurry or so small that wires cannot be judged; the AI still runs, the warning is shown.
Why: an operator who uploads a photo should get either an answer or one clear sentence about
what is wrong with the file, never a crash, and one bad file must not stop the others.
"""
from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .imageio import to_rgb8

MAX_BYTES = 30_000_000
MAX_PIXELS = 50_000_000    # bigger pictures would take minutes and most of the Pi's memory
BLUR_LIMIT = 30.0          # variance of the Laplacian below this = looks out of focus
SMALL_PX = 400


class PhotoError(ValueError):
    """The uploaded file cannot be used (the message says why, in plain words)."""


def load_photo(data: bytes, max_bytes: int = MAX_BYTES) -> Image.Image:
    if not data:
        raise PhotoError("empty file")
    if len(data) > max_bytes:
        raise PhotoError(f"file is larger than {max_bytes // 1_000_000} MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            if img.width * img.height > MAX_PIXELS:      # checked from the header, before decoding
                raise PhotoError(f"picture is {img.width * img.height / 1e6:.0f} megapixels; "
                                 f"the limit is {MAX_PIXELS // 1_000_000} megapixels, please shrink it")
            img.load()
            img = ImageOps.exif_transpose(img)
    except PhotoError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        raise PhotoError("not a picture the app can read (use JPG, PNG, BMP or TIFF)") from None
    return to_rgb8(img)


def quality_hints(img: Image.Image) -> list[str]:
    hints = []
    g = img.convert("L")
    if max(g.size) > 1024:
        s = 1024 / max(g.size)
        g = g.resize((max(1, round(g.width * s)), max(1, round(g.height * s))), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float64)
    if a.shape[0] >= 3 and a.shape[1] >= 3:
        lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
        if lap.var() < BLUR_LIMIT:
            hints.append("blurry: the picture looks out of focus")
    if min(img.size) < SMALL_PX:
        hints.append(f"small: the picture is under {SMALL_PX} px, wires may be too small to judge")
    return hints
