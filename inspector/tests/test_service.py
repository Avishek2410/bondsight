import time
from pathlib import Path

import pytest
from wirebond_synth.layout import make_layout

from wirebond_inspector.config import CameraCfg, Config, TrackerCfg, TriggerCfg
from wirebond_inspector.detector import Finding
from wirebond_inspector.service import AllCamerasFailed, Busy, Inspector

INJ = {"mode": "repetitive", "wire": 5, "cls": "wire_sweep", "fraction": 1.0}


def make_cfg(tmp_path: Path, model: Path, cameras, k=3, trigger=TriggerCfg()):
    return Config(model=model, threshold=0.4, cameras=tuple(cameras), data_dir=tmp_path / "data",
                  layout=None, tracker=TrackerCfg(k, 10), trigger=trigger, bind="127.0.0.1", port=8080)


def gen(name, view):
    return CameraCfg(name, "generator", view, {"product_seed": 7, "injection": INJ})


class Wire5Detector:
    """Always reports a wire_sweep in the middle of wire 5 (top-view coordinates)."""
    name = "stub"

    def __init__(self):
        w = make_layout(24, 7).wires[5]
        x, y = (w.pad.x + w.lead.x) / 2, (w.pad.y + w.lead.y) / 2
        self.f = Finding("wire_sweep", 0.9, (x - 6, y - 6, x + 6, y + 6))

    def detect(self, img, threshold):
        return [self.f]


def test_inspect_stores_unit_with_both_views(tmp_path, fake_model):
    ins = Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top"), gen("angled", "angled")]))
    u0, u1 = ins.inspect(), ins.inspect()
    assert (u0["unit_no"], u1["unit_no"]) == (0, 1)
    assert {s["camera"] for s in u0["shots"]} == {"top", "angled"}
    assert all(s["image_path"] for s in u0["shots"])
    assert set(u0["timing_s"]) >= {"top", "angled", "total"}
    assert {"wire": 5, "cls": "wire_sweep"} in u0["truth"]
    assert all(f["position"].startswith("wire:") for f in u0["findings"])


def test_inspect_survives_failing_camera(tmp_path, fake_model):
    empty = tmp_path / "empty"
    empty.mkdir()
    cams = [gen("top", "top"), CameraCfg("photos", "folder", "top", {"path": str(empty)})]
    ins = Inspector(make_cfg(tmp_path, fake_model(), cams))
    u = ins.inspect()
    shots = {s["camera"]: s for s in u["shots"]}
    assert shots["photos"]["error"] == "no new image" and shots["photos"]["image_path"] is None
    assert shots["top"]["error"] is None
    assert any(f["camera"] == "top" for f in u["findings"])
    assert {c["name"]: c["ok"] for c in ins.cameras()}["top"] is True


def test_all_cameras_failed_is_not_stored(tmp_path, fake_model):
    empty = tmp_path / "empty"
    empty.mkdir()
    ins = Inspector(make_cfg(tmp_path, fake_model(), [CameraCfg("p", "folder", "top", {"path": str(empty)})]))
    with pytest.raises(AllCamerasFailed, match="no new image"):
        ins.inspect()
    assert ins.store.count() == 0


def test_busy_while_inspecting(tmp_path, fake_model):
    ins = Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top")]))
    with ins._lock:
        with pytest.raises(Busy):
            ins.inspect()


def test_repetitive_alert_end_to_end(tmp_path, fake_model):
    ins = Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top")], k=3), detector=Wire5Detector())
    units = [ins.inspect() for _ in range(3)]
    assert units[0]["alerts"] == [] and units[1]["alerts"] == []
    assert units[2]["alerts"] == [{"position": "wire:5", "cls": "wire_sweep", "count": 3, "window": 10}]
    # restart: history comes back from the database, so the 4th unit alerts with count 4
    ins2 = Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top")], k=3), detector=Wire5Detector())
    assert ins2.inspect()["alerts"][0]["count"] == 4


def test_interval_idle_when_all_folders_empty(tmp_path, fake_model):
    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = make_cfg(tmp_path, fake_model(), [CameraCfg("p", "folder", "top", {"path": str(empty)})],
                   trigger=TriggerCfg("interval", 0.05))
    ins = Inspector(cfg)
    ins.start_trigger()
    time.sleep(0.3)
    ins.stop()
    assert ins.store.count() == 0
    assert ins.last_error is None          # idle is not an error


class BrokenCamera:
    name, view = "broken", "top"

    def capture(self, unit):
        raise RuntimeError("driver exploded")

    def status(self):
        return {"name": self.name, "type": "x", "view": self.view, "ok": False, "detail": ""}

    def layout(self):
        return None


def test_unexpected_camera_error_is_isolated(tmp_path, fake_model):
    ins = Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top")]), detector=Wire5Detector())
    ins.cams.append(BrokenCamera())
    u = ins.inspect()
    shots = {s["camera"]: s for s in u["shots"]}
    assert "driver exploded" in shots["broken"]["error"]
    assert shots["top"]["error"] is None and u["findings"]


def test_detector_error_on_one_camera_is_isolated(tmp_path, fake_model):
    class Det(Wire5Detector):
        def detect(self, img, threshold):
            if self.calls == 1:
                raise ValueError("model blew up")
            self.calls += 1
            return [self.f]
    d = Det()
    d.calls = 0
    ins = Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top"), gen("angled", "angled")]), detector=d)
    u = ins.inspect()
    shots = {s["camera"]: s for s in u["shots"]}
    assert shots["top"]["error"] is None
    assert "model blew up" in shots["angled"]["error"]
    assert shots["angled"]["image_path"]          # the picture is still kept for checking
