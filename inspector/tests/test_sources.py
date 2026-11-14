import time

import numpy as np
import pytest
from PIL import Image

from wirebond_inspector.config import CameraCfg, ConfigError
from wirebond_inspector.sources import make_camera
from wirebond_inspector.sources.base import CameraError

INJ = {"mode": "repetitive", "wire": 5, "cls": "wire_sweep", "fraction": 1.0}


def test_folder_returns_files_in_order_then_errors(tmp_path):
    for n in ("b.png", "a.png", "c.jpg"):
        Image.new("L", (20, 10)).save(tmp_path / n)
    (tmp_path / "notes.txt").write_text("not an image")
    cam = make_camera(CameraCfg("f", "folder", "top", {"path": str(tmp_path)}))
    got = [cam.capture(i) for i in range(3)]
    assert [g.source.split("\\")[-1].split("/")[-1] for g in got] == ["a.png", "b.png", "c.jpg"]
    assert all(g.image.mode == "RGB" and g.camera == "f" and g.view == "top" for g in got)
    with pytest.raises(CameraError, match="no new image"):
        cam.capture(3)
    assert cam.status()["ok"] is True          # an empty folder is idle, not broken


def test_folder_resumes_after_start_after(tmp_path):
    for n in ("a.png", "b.png"):
        Image.new("RGB", (8, 8)).save(tmp_path / n)
    cam = make_camera(CameraCfg("f", "folder", "top", {"path": str(tmp_path), "start_after": "a.png"}))
    assert cam.capture(0).source.endswith("b.png")


def test_folder_missing_dir_reports_offline(tmp_path):
    cam = make_camera(CameraCfg("f", "folder", "top", {"path": str(tmp_path / "nope")}))
    with pytest.raises(CameraError):
        cam.capture(0)
    assert cam.status()["ok"] is False


def test_generator_views_agree():
    top = make_camera(CameraCfg("t", "generator", "top", {"product_seed": 7, "injection": INJ}))
    ang = make_camera(CameraCfg("a", "generator", "angled", {"product_seed": 7, "injection": INJ}))
    assert 5 in {d.wire for d in top.planned_defects(0)}
    assert top.planned_defects(0) == ang.planned_defects(0)
    ft, fa = top.capture(0), ang.capture(0)
    assert ft.image.size == fa.image.size == (640, 640) and ft.image.mode == "RGB"
    assert ft.source == "gen:u00000" and fa.view == "angled"
    assert not np.array_equal(np.asarray(ft.image), np.asarray(fa.image))


def test_generator_layout_roundtrip():
    from wirebond_synth.layout import Layout
    cam = make_camera(CameraCfg("t", "generator", "top", {"product_seed": 7, "n_wires": 16}))
    assert len(Layout.from_dict(cam.layout()).wires) == 16


def test_opencv_bad_source_raises():
    cam = make_camera(CameraCfg("c", "opencv", "top", {"source": "rtsp://127.0.0.1:1/x"}))
    t = time.time()
    with pytest.raises(CameraError):
        cam.capture(0)
    assert time.time() - t < 15
    assert cam.status()["ok"] is False


def test_make_camera_unknown_type():
    with pytest.raises(ConfigError, match="csi"):
        make_camera(CameraCfg("x", "csi", "top", {}))


def test_opencv_never_shows_camera_password():
    cam = make_camera(CameraCfg("c", "opencv", "top", {"source": "rtsp://admin:s3cret@127.0.0.1:1/x"}))
    with pytest.raises(CameraError) as e:
        cam.capture(0)
    st = cam.status()
    assert "s3cret" not in str(e.value) and "s3cret" not in st["detail"]
    assert "127.0.0.1:1" in st["detail"]


def test_folder_retries_a_file_that_is_still_being_written(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (9, 9, 9)).save(p)
    good = p.read_bytes()
    p.write_bytes(good[: len(good) // 2])                 # half-written by another program
    cam = make_camera(CameraCfg("f", "folder", "top", {"path": str(tmp_path)}))
    with pytest.raises(CameraError):
        cam.capture(0)
    p.write_bytes(good)                                   # writer finished
    assert cam.capture(1).source.endswith("a.png")


def test_folder_gives_up_on_a_corrupt_file_after_3_tries(tmp_path):
    (tmp_path / "a.png").write_bytes(b"not an image")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    cam = make_camera(CameraCfg("f", "folder", "top", {"path": str(tmp_path)}))
    for i in range(3):
        with pytest.raises(CameraError):
            cam.capture(i)
    assert cam.capture(3).source.endswith("b.png")


def test_opencv_drops_buffered_frames(monkeypatch):
    import numpy as np
    from wirebond_inspector.sources import opencv_cam

    class FakeCap:
        def __init__(self, *a):
            self.grabs, self.props = 0, {}
        def isOpened(self): return True
        def set(self, k, v): self.props[k] = v
        def grab(self):
            self.grabs += 1
            return True
        def retrieve(self): return True, np.full((4, 4, 3), self.grabs, np.uint8)
        def read(self): return self.grab() and self.retrieve()
        def release(self): pass

    caps = []
    monkeypatch.setattr(opencv_cam.cv2, "VideoCapture", lambda *a: caps.append(FakeCap()) or caps[-1])
    cam = make_camera(CameraCfg("c", "opencv", "top", {"source": 0}))
    f = cam.capture(0)
    cap = caps[0]
    assert cap.props.get(opencv_cam.cv2.CAP_PROP_BUFFERSIZE) == 1
    assert cap.grabs >= opencv_cam.FLUSH_FRAMES + 1                  # old frames thrown away
    assert np.asarray(f.image)[0, 0, 0] == cap.grabs                 # newest frame returned


@pytest.mark.parametrize("src", [
    "http://10.0.0.5/snap.jpg?user=admin&pwd=s3cret",
    "rtsp://10.0.0.5:554/live?password=s3cret&channel=1",
    "http://cam.local/cgi?usr=a&token=s3cret",
    "rtsp://admin:s3cret@10.0.0.5:notaport/x",          # unparsable port must not crash or leak
])
def test_redact_hides_credentials_everywhere(src):
    from wirebond_inspector.sources.opencv_cam import redact
    out = redact(src)
    assert "s3cret" not in out
    assert "10.0.0.5" in out or "cam.local" in out
