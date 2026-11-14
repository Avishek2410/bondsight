import numpy as np
from PIL import Image

from wirebond_inspector.imageio import to_rgb8


def test_to_rgb8_modes():
    imgs = [Image.new("L", (10, 8), 100), Image.new("RGBA", (10, 8), (1, 2, 3, 0)),
            Image.new("P", (10, 8)), Image.fromarray(np.full((8, 10), 40000, np.uint16))]
    for im in imgs:
        out = to_rgb8(im)
        assert out.mode == "RGB" and out.size == (10, 8)
        assert np.asarray(out).dtype == np.uint8
    # 16-bit 40000 must become bright (~156), not clipped to 255 or wrapped
    assert 150 <= np.asarray(to_rgb8(imgs[3]))[0, 0, 0] <= 160
