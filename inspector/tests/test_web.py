import pytest
from fastapi.testclient import TestClient

from wirebond_inspector.config import CameraCfg
from wirebond_inspector.service import Inspector
from wirebond_inspector.web.app import create_app

from test_service import gen, make_cfg


@pytest.fixture
def client_ins(tmp_path, fake_model):
    ins = Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top"), gen("angled", "angled")]))
    return TestClient(create_app(ins)), ins


def test_status(client_ins):
    c, _ = client_ins
    s = c.get("/api/status").json()
    assert s["model"] and s["threshold"] == 0.4 and s["units_total"] == 0
    assert [x["name"] for x in s["cameras"]] == ["top", "angled"]
    assert s["tracker"] == {"k": 3, "m": 10} and s["trigger"] == "manual"


def test_inspect_then_fetch_unit_and_image(client_ins):
    c, _ = client_ins
    r = c.post("/api/inspect")
    assert r.status_code == 200
    u = r.json()
    assert c.get(f"/api/units/{u['id']}").json()["unit_no"] == 0
    img = c.get(f"/api/images/{u['id']}/top.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert c.get("/api/units?limit=5").json()[0]["id"] == u["id"]
    assert isinstance(c.get("/api/heatmap").json(), dict)
    assert c.get("/api/alerts").json() == []
    assert c.get("/api/units/999").status_code == 404


def test_inspect_busy_returns_409(client_ins):
    c, ins = client_ins
    with ins._lock:
        r = c.post("/api/inspect")
    assert r.status_code == 409 and r.json()["detail"] == "busy"


def test_inspect_all_cameras_failed_returns_503(tmp_path, fake_model):
    empty = tmp_path / "empty"
    empty.mkdir()
    ins = Inspector(make_cfg(tmp_path, fake_model(), [CameraCfg("p", "folder", "top", {"path": str(empty)})]))
    r = TestClient(create_app(ins)).post("/api/inspect")
    assert r.status_code == 503 and "no new image" in r.json()["detail"]


@pytest.mark.parametrize("path", ["/api/images/1/..%2F..%2Finspector.db", "/api/images/1/../../inspector.db",
                                  "/api/images/1/inspector.db", "/api/images/x/top.jpg"])
def test_image_path_traversal_blocked(client_ins, path):
    c, _ = client_ins
    c.post("/api/inspect")
    assert c.get(path).status_code in (404, 422)


def test_index_served(client_ins):
    c, _ = client_ins
    r = c.get("/")
    assert r.status_code == 200 and "Inspect" in r.text and "Research prototype" in r.text


def _jpg(seed=0):
    import io
    import numpy as np
    from PIL import Image
    a = (np.indices((600, 800)).sum(0) // (8 + seed) % 2 * 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a, "L").convert("RGB").save(b, "JPEG")
    return b.getvalue()


def test_check_upload_batch(client_ins):
    c, _ = client_ins
    files = [("files", ("a.jpg", _jpg(), "image/jpeg")), ("files", ("b.jpg", _jpg(1), "image/jpeg")),
             ("files", ("c.txt", b"hello", "text/plain"))]
    r = c.post("/api/check", files=files)
    assert r.status_code == 200
    res = r.json()
    assert [x.get("label") for x in res] == ["a.jpg", "b.jpg", "c.txt"]
    assert res[0]["source"] == "upload" and "error" in res[2]
    assert c.get(f"/api/images/{res[0]['id']}/upload.jpg").status_code == 200


def test_check_upload_limits(client_ins):
    c, _ = client_ins
    assert c.post("/api/check", files=[("files", (f"{i}.jpg", b"x", "image/jpeg")) for i in range(21)]).status_code == 400
    assert c.post("/api/check", data={"camera": "top"}).status_code in (400, 422)
    r = c.post("/api/check", files=[("files", ("a.jpg", _jpg(), "image/jpeg"))], data={"camera": "side"})
    assert r.status_code == 400 and "camera" in r.json()["detail"]


def test_decision_endpoint(client_ins):
    c, _ = client_ins
    u = c.post("/api/inspect").json()
    fid = u["findings"][0]["id"]
    r = c.post(f"/api/units/{u['id']}/decisions", json={"verdict": "real", "operator": "Rina", "finding_id": fid})
    assert r.status_code == 201 and r.json()["unit"]["findings"][0]["decision"] == "real"
    bad = c.post(f"/api/units/{u['id']}/decisions",
                 json={"verdict": "missed", "operator": "Rina", "camera": "top", "cls": "wire_sweep",
                       "box": [0, 0, 99999, 10]})
    assert bad.status_code == 400 and "inside the picture" in bad.json()["detail"]
    assert c.post("/api/units/999/decisions", json={"verdict": "confirm_ok", "operator": "R"}).status_code == 404


def test_units_filters_endpoint(client_ins):
    c, _ = client_ins
    a = c.post("/api/inspect").json()
    b = c.post("/api/inspect").json()
    for u in (a,):
        for f in u["findings"]:
            c.post(f"/api/units/{u['id']}/decisions", json={"verdict": "real", "operator": "R", "finding_id": f["id"]})
    ids = [u["id"] for u in c.get("/api/units?undecided=1").json()]
    assert b["id"] in ids and a["id"] not in ids
    assert all(u["verdict"] == "DEFECT" for u in c.get("/api/units?verdict=DEFECT").json())
    assert c.get("/api/units?source=upload").json() == []


def test_status_shows_database_without_password(client_ins, monkeypatch):
    c, _ = client_ins
    s = c.get("/api/status").json()
    assert s["database"].startswith("sqlite") and s["station"] and "wire_sweep" in s["classes"]


def test_v2_page_and_assets(client_ins):
    c, _ = client_ins
    html = c.get("/").text
    for i in ('id="tab-inspect"', 'id="tab-check"', 'id="tab-review"', "/static/app.css", "/static/app.js"):
        assert i in html
    css, js = c.get("/static/app.css"), c.get("/static/app.js")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]


def test_status_line_units_exclude_uploads(client_ins):
    c, _ = client_ins
    c.post("/api/inspect")
    c.post("/api/check", files=[("files", ("a.jpg", _jpg(), "image/jpeg"))])
    s = c.get("/api/status").json()
    assert s["units_total"] == 2 and s["line_units"] == 1


def test_page_files_are_revalidated(client_ins):
    c, _ = client_ins
    for p in ("/", "/static/app.js", "/static/app.css"):
        assert "no-cache" in c.get(p).headers.get("cache-control", "")
