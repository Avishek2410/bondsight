import io

import numpy as np
import pytest
from PIL import Image

from wirebond_inspector.photocheck import PhotoError, load_photo, quality_hints
from wirebond_inspector.service import Inspector

from test_service import gen, make_cfg


def _jpeg(img: Image.Image, **kw) -> bytes:
    b = io.BytesIO()
    img.save(b, "JPEG", **kw)
    return b.getvalue()


def _sharp(size=(800, 600)) -> Image.Image:
    a = (np.indices(size[::-1]).sum(0) // 8 % 2 * 255).astype(np.uint8)     # 8 px checkerboard
    return Image.fromarray(a, "L").convert("RGB")


def test_exif_rotated_phone_photo_is_upright():
    img = Image.new("RGB", (600, 400), "white")
    img.paste((255, 0, 0), (0, 0, 100, 100))                     # red square top-left as stored
    exif = Image.Exif()
    exif[0x0112] = 6                                             # "rotate 90 clockwise to view"
    out = load_photo(_jpeg(img, exif=exif))
    assert out.size == (400, 600)
    r, g, b = out.getpixel((350, 50))                            # viewer sees it top-right
    assert r > 200 and g < 60 and b < 60


def test_bad_files_get_plain_messages():
    with pytest.raises(PhotoError, match="empty file"):
        load_photo(b"")
    with pytest.raises(PhotoError, match="larger than 30 MB"):
        load_photo(b"x" * 30_000_001)
    with pytest.raises(PhotoError, match="not a picture the app can read"):
        load_photo(b"not an image at all")


def test_sixteen_bit_and_rgba_become_rgb():
    b = io.BytesIO()
    Image.new("RGBA", (500, 500), (10, 20, 30, 128)).save(b, "PNG")
    assert load_photo(b.getvalue()).mode == "RGB"


def test_quality_hints():
    assert any(h.startswith("blurry") for h in quality_hints(Image.new("RGB", (800, 600), (128, 128, 128))))
    assert quality_hints(_sharp()) == []
    assert any(h.startswith("small") for h in quality_hints(_sharp((300, 300))))


@pytest.fixture
def ins(tmp_path, fake_model):
    return Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top"), gen("angled", "angled")]))


def test_check_photos_mixed_batch(ins):
    res = ins.check_photos([("a.jpg", _jpeg(_sharp())), ("b.txt", b"x")])
    assert res[0]["id"] is not None and res[0]["source"] == "upload" and res[0]["label"] == "a.jpg"
    assert res[0]["findings"][0]["cls"] == "missing_wire"
    assert res[0]["shots"][0]["camera"] == "upload"
    assert "unusual-spot check not set up for this photo" in res[0]["hints"]
    assert res[1]["label"] == "b.txt" and "error" in res[1]


def test_upload_does_not_touch_line(ins):
    ins.check_photos([("a.jpg", _jpeg(_sharp()))])
    assert ins.store.next_unit_no() == 0
    assert ins.tracker.update(set()) == []
    assert ins.inspect()["unit_no"] == 0


def test_upload_with_camera_uses_its_view(ins):
    r = ins.check_photos([("a.jpg", _jpeg(_sharp()))], camera="angled")[0]
    assert r["shots"][0]["camera"] == "angled" and r["shots"][0]["view"] == "angled"


def test_unknown_camera_refused(ins):
    with pytest.raises(ValueError, match="camera"):
        ins.check_photos([("a.jpg", _jpeg(_sharp()))], camera="side")


def test_check_without_saving(ins):
    r = ins.check_photos([("a.jpg", _jpeg(_sharp()))], save=False)[0]
    assert r["id"] is None and r["image_b64"] and r["verdict"] == "DEFECT"
    assert ins.store.count() == 0
    assert not any((ins.store.dir / "images").iterdir())


def test_huge_picture_refused_before_decoding():
    b = io.BytesIO()
    Image.new("L", (9000, 6000)).save(b, "PNG")              # 54 megapixels, tiny file
    with pytest.raises(PhotoError, match="megapixels"):
        load_photo(b.getvalue())


def test_uploads_are_checked_one_at_a_time(ins):
    import threading
    import time as _t
    from wirebond_inspector.detector import Finding

    class Slow:
        name, classes, now, most = "slow", ("wire_sweep",), 0, 0
        lock = threading.Lock()

        def detect(self, img, thr):
            with self.lock:
                Slow.now += 1
                Slow.most = max(Slow.most, Slow.now)
            _t.sleep(0.2)
            with self.lock:
                Slow.now -= 1
            return [Finding("wire_sweep", 0.9, (1, 1, 5, 5))]

    ins.detector = Slow()
    data = _jpeg(_sharp())
    ts = [threading.Thread(target=ins.check_photos, args=([("a.jpg", data)],)) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert Slow.most == 1 and ins.store.count(source="upload") == 3


def test_upload_works_while_line_inspection_runs(ins):
    with ins._lock:                                          # an inspection is "running"
        r = ins.check_photos([("a.jpg", _jpeg(_sharp()))])[0]
    assert r["id"] is not None and ins.store.next_unit_no() == 0
