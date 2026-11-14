"""Anomaly checker: nearest-distance maths, coreset, fit/score/save/load, overlay.

Uses a tiny fake feature extractor (no ONNX file needed): each picture's pixels ARE its
patch features, so we control exactly what is "normal" and what is "unusual".
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from wirebond_inspector.anomaly import AnomalyModel, coreset, heat_overlay, nearest_dist


class FakeExtractor:
    """Picture -> 8 x 8 grid of 384-number features (grey level + small noise)."""

    def __call__(self, img: Image.Image) -> np.ndarray:
        g = np.asarray(img.convert("L").resize((8, 8)), np.float32) / 255.0
        rng = np.random.default_rng(int(g.sum() * 1000) % 2**32)
        return np.repeat(g[..., None], 384, 2) + rng.normal(0, 0.01, (8, 8, 384)).astype(np.float32)


def grey(v: int, spot: bool = False) -> Image.Image:
    a = np.full((64, 64), v, np.uint8)
    if spot:
        a[24:40, 24:40] = 255                       # one bright square = the "defect"
    return Image.fromarray(a).convert("RGB")


def test_nearest_dist_matches_brute_force():
    rng = np.random.default_rng(0)
    q, bank = rng.normal(size=(50, 16)).astype(np.float32), rng.normal(size=(30, 16)).astype(np.float32)
    want = np.sqrt(((q[:, None] - bank[None]) ** 2).sum(-1)).min(1)
    np.testing.assert_allclose(nearest_dist(q, bank, chunk=7), want, rtol=1e-4, atol=1e-4)


def test_coreset_keeps_spread_points():
    pts = np.concatenate([np.zeros((100, 4)), np.full((1, 4), 10.0), np.full((1, 4), -10.0)]).astype(np.float32)
    keep = coreset(pts, 3, proj_dim=4)
    assert len(set(keep.tolist())) == 3 and {100, 101} <= set(keep.tolist())
    assert len(coreset(pts, 500)) == len(pts)


def test_fit_flags_the_unusual_picture_and_not_the_good_ones():
    ext = FakeExtractor()
    good = [grey(100 + i % 5) for i in range(6)]
    hold = [grey(101), grey(103)]
    m = AnomalyModel.fit(ext, good, hold, bank_size=200)
    assert not m.score(ext(grey(102))).flagged
    bad = m.score(ext(grey(102, spot=True)))
    assert bad.flagged and bad.heat.shape == (8, 8)
    r, c = np.unravel_index(bad.heat.argmax(), bad.heat.shape)
    assert 2 <= r <= 5 and 2 <= c <= 5                      # the hot spot is where the square is


def test_fit_needs_both_sets():
    with pytest.raises(ValueError):
        AnomalyModel.fit(FakeExtractor(), [grey(100)], [])


def test_save_load_roundtrip(tmp_path):
    ext = FakeExtractor()
    m = AnomalyModel.fit(ext, [grey(100), grey(104)], [grey(102)], bank_size=50)
    p = tmp_path / "cam.npz"
    m.save(p)
    m2 = AnomalyModel.load(p)
    assert m2.threshold == pytest.approx(m.threshold) and m2.info["format"] == 1
    f = ext(grey(102, spot=True))
    assert m2.score(f).score == pytest.approx(m.score(f).score, rel=1e-2)


def test_overlay_tints_only_the_hot_spot():
    heat = np.zeros((8, 8), np.float32)
    heat[3:5, 3:5] = 2.0
    out = np.asarray(heat_overlay(grey(100), heat, threshold=1.0)).astype(int)
    assert out.shape == (64, 64, 3)
    assert out[32, 32, 0] > out[32, 32, 1] + 50             # centre is red
    assert abs(out[2, 2, 0] - out[2, 2, 1]) < 3              # corner untouched


# ---- inspector integration ------------------------------------------------------------------
import sqlite3
from types import SimpleNamespace

from wirebond_inspector.anomaly import AnomalyResult
from wirebond_inspector.config import ConfigError, load_config
from wirebond_inspector.service import Inspector
from wirebond_inspector.store import Store

from test_service import gen, make_cfg


class StubChecker:
    """Top camera looks unusual, angled camera looks normal."""

    def __init__(self):
        self.models = {"top": SimpleNamespace(threshold=1.0), "angled": SimpleNamespace(threshold=1.0)}

    def check(self, camera, img):
        heat = np.zeros((8, 8), np.float32)
        heat[2, 2] = 3.0 if camera == "top" else 0.5
        return AnomalyResult(float(heat.max()), camera == "top", heat)

    def status(self, camera):
        return "stub"


class NoFindings:
    name = "stub"

    def detect(self, img, threshold):
        return []


def test_unusual_picture_is_stored_with_heat_map(tmp_path, fake_model):
    ins = Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top"), gen("angled", "angled")]),
                    detector=NoFindings(), checker=StubChecker())
    u = ins.inspect()
    shots = {s["camera"]: s for s in u["shots"]}
    assert shots["top"]["anomaly"] == 3.0 and shots["top"]["anomaly_flag"] is True
    assert shots["top"]["heat_path"] == f"images/{u['id']}/top_heat.jpg"
    assert (ins.store.dir / shots["top"]["heat_path"]).is_file()
    assert shots["angled"]["anomaly_flag"] is False and shots["angled"]["heat_path"] is None
    row = ins.store.units()[0]
    assert row["n_findings"] == 0 and row["n_unusual"] == 1        # = "please check" in the UI
    assert ins.cameras()[0]["unusual_check"] == "stub"


def test_checker_crash_does_not_stop_inspection(tmp_path, fake_model):
    class Broken(StubChecker):
        def check(self, camera, img):
            raise RuntimeError("boom")
    ins = Inspector(make_cfg(tmp_path, fake_model(), [gen("top", "top")]), detector=NoFindings(), checker=Broken())
    u = ins.inspect()
    assert u["shots"][0]["anomaly"] is None and u["shots"][0]["error"] is None


def test_v1_database_is_left_alone_and_v2_has_anomaly_columns(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    db = sqlite3.connect(d / "inspector.db")           # a v1 file: v2 neither reads nor changes it
    db.executescript("CREATE TABLE shots (unit_id INTEGER NOT NULL, camera TEXT);")
    db.commit(); db.close()
    before = (d / "inspector.db").read_bytes()
    Store(d)
    assert (d / "inspector.db").read_bytes() == before
    cols = {r[1] for r in sqlite3.connect(d / "inspector2.db").execute("PRAGMA table_info(shots)")}
    assert {"anomaly", "anomaly_flag", "heat_path"} <= cols


def test_config_anomaly_section(example_dir):
    cfg_path = example_dir / "config.yaml"
    base = cfg_path.read_text()
    assert load_config(cfg_path).anomaly is None                    # optional
    cfg_path.write_text(base + "\nanomaly: {backbone: models/missing.onnx, dir: models/anomaly}\n")
    with pytest.raises(ConfigError, match="anomaly.backbone"):
        load_config(cfg_path)
    (example_dir / "models" / "bb.onnx").write_bytes(b"x")
    (example_dir / "models" / "bb.json").write_text('{"img_size": 480}')
    cfg_path.write_text(base + "\nanomaly: {backbone: models/bb.onnx, dir: models/anomaly, size: 500}\n")
    with pytest.raises(ConfigError, match="anomaly.size"):
        load_config(cfg_path)
    cfg_path.write_text(base + "\nanomaly: {backbone: models/bb.onnx, dir: models/anomaly}\n")
    a = load_config(cfg_path).anomaly
    assert a.backbone == example_dir / "models" / "bb.onnx" and a.dir == example_dir / "models" / "anomaly"
