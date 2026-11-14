import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wirebond_inspector.config import ConfigError, load_config
from wirebond_inspector.service import Inspector
from wirebond_inspector.web.app import create_app


def _jpg() -> bytes:
    a = (np.indices((600, 800)).sum(0) // 8 % 2 * 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a, "L").convert("RGB").save(b, "JPEG")
    return b.getvalue()


@pytest.fixture
def demo(tmp_path, fake_model):
    samples = tmp_path / "samples"
    samples.mkdir()
    Image.new("RGB", (640, 640), (90, 90, 90)).save(samples / "wire_sweep.png")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"model: {fake_model().as_posix()}\ndemo: true\nsamples: samples\ndata_dir: data\n")
    ins = Inspector(load_config(cfg))
    return TestClient(create_app(ins)), ins, tmp_path


def test_demo_config_needs_no_camera_but_station_does(tmp_path, fake_model):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"model: {fake_model().as_posix()}\ndemo: true\n")
    c = load_config(cfg)
    assert c.demo is True and c.cameras == () and c.database == "sqlite://"
    cfg.write_text(f"model: {fake_model().as_posix()}\n")
    with pytest.raises(ConfigError, match="camera"):
        load_config(cfg)


def test_demo_upload_saves_nothing(demo):
    c, ins, tmp = demo
    r = c.post("/api/check", files=[("files", ("a.jpg", _jpg(), "image/jpeg"))])
    assert r.status_code == 200
    res = r.json()[0]
    assert res["id"] is None and res["image_b64"] and res["findings"]
    assert ins.store.count() == 0
    assert not any((tmp / "data" / "images").iterdir())
    s = c.get("/api/status").json()
    assert s["demo"] is True and s["station"] == "online demo"   # never the host machine's name


@pytest.mark.parametrize("method,path", [("post", "/api/inspect"), ("get", "/api/units"), ("get", "/api/units/1"),
                                         ("post", "/api/units/1/decisions"), ("get", "/api/images/1/top.jpg"),
                                         ("get", "/api/heatmap"), ("get", "/api/alerts")])
def test_demo_blocks_station_endpoints(demo, method, path):
    c, _, _ = demo
    kw = {"json": {"verdict": "confirm_ok", "operator": "x"}} if method == "post" and "decisions" in path else {}
    r = getattr(c, method)(path, **kw)
    assert r.status_code == 403 and r.json()["detail"] == "not available in the demo"


def test_demo_samples(demo):
    c, ins, _ = demo
    lst = c.get("/api/samples").json()
    assert lst == [{"name": "wire_sweep.png", "url": "/api/samples/wire_sweep.png"}]
    assert c.get("/api/samples/wire_sweep.png").status_code == 200
    for bad in ("../config.yaml", "..%2Fconfig.yaml", "nope.png"):
        assert c.get(f"/api/samples/{bad}").status_code == 404
    r = c.post("/api/check", data={"sample": "wire_sweep.png"})
    assert r.status_code == 200 and r.json()[0]["label"] == "wire_sweep.png" and ins.store.count() == 0
    assert c.post("/api/check", data={"sample": "../config.yaml"}).status_code == 404


def test_station_has_no_samples_endpoint(tmp_path, fake_model):
    from test_service import gen, make_cfg
    c = TestClient(create_app(Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top")]))))
    assert c.get("/api/samples").status_code == 404
    assert c.get("/api/status").json()["demo"] is False
