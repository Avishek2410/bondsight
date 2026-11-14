import json

import pytest
import yaml

from wirebond_inspector.config import ConfigError, TrackerCfg, load_config


def test_load_example_config(example_dir):
    cfg = load_config(example_dir / "config.yaml")
    assert len(cfg.cameras) == 2
    assert {c.view for c in cfg.cameras} == {"top", "angled"}
    assert cfg.bind == "127.0.0.1"
    assert cfg.port == 8080
    assert cfg.tracker == TrackerCfg(3, 10)
    assert cfg.model == example_dir / "models" / "wirebond_dfine_small_m3k.onnx"
    assert cfg.data_dir == example_dir / "data"


def _edit(d, fn):
    p = d / "config.yaml"
    c = yaml.safe_load(p.read_text())
    fn(c)
    p.write_text(yaml.safe_dump(c))
    return p


@pytest.mark.parametrize("change,needle", [
    (lambda c: c.update(model="models/missing.onnx"), "missing.onnx"),
    (lambda c: c["cameras"][0].update(type="csi2"), "csi2"),
    (lambda c: c["cameras"][1].update(name="top"), "top"),
    (lambda c: c.update(threshold=1.5), "threshold"),
    (lambda c: c["cameras"][0].update(view="side"), "side"),
])
def test_config_errors_are_clear(example_dir, change, needle):
    with pytest.raises(ConfigError) as e:
        load_config(_edit(example_dir, change))
    assert needle in str(e.value)


def test_model_json_without_classes(example_dir):
    js = example_dir / "models" / "wirebond_dfine_small_m3k.json"
    js.write_text(json.dumps({"img_size": 640}))
    with pytest.raises(ConfigError) as e:
        load_config(example_dir / "config.yaml")
    assert "wirebond_dfine_small_m3k.json" in str(e.value) and "classes" in str(e.value)


@pytest.mark.parametrize("change,needle", [
    (lambda c: c["cameras"][0].update(name="Top cam"), "Top cam"),                 # space breaks image URLs
    (lambda c: c["cameras"][0].update(name="../x"), "../x"),                       # path escape
    (lambda c: c["cameras"][1].update(name="TOP"), "TOP"),                         # case-insensitive duplicate
    (lambda c: c["cameras"].append({"name": "f", "type": "folder", "view": "top"}), "path"),
    (lambda c: c["cameras"][0]["options"].update(injection={"mode": "random", "bogus": 1}), "injection"),
    (lambda c: c.update(tracker={"k": "three"}), "tracker"),
    (lambda c: c.update(trigger={"seconds": "soon"}), "trigger"),
    (lambda c: c["cameras"].append("just-a-string"), "cameras[2]"),
])
def test_more_config_errors_are_clear(example_dir, change, needle):
    with pytest.raises(ConfigError) as e:
        load_config(_edit(example_dir, change))
    assert needle in str(e.value)


def test_top_level_must_be_a_mapping(example_dir):
    p = example_dir / "config.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_tiling_without_stride_is_rejected(example_dir):
    js = example_dir / "models" / "wirebond_dfine_small_m3k.json"
    m = json.loads(js.read_text())
    del m["tiling"]["stride"]
    js.write_text(json.dumps(m))
    with pytest.raises(ConfigError) as e:
        load_config(example_dir / "config.yaml")
    assert "stride" in str(e.value)


def test_bad_layout_json_is_rejected(example_dir):
    (example_dir / "layout.json").write_text("{not json")
    with pytest.raises(ConfigError) as e:
        load_config(_edit(example_dir, lambda c: c.update(layout="layout.json")))
    assert "layout" in str(e.value)


def test_main_reports_startup_error_without_traceback(example_dir, capsys):
    from wirebond_inspector.__main__ import main
    # the placeholder model file is not a real ONNX model: startup must fail with exit code 2
    assert main(["--config", str(example_dir / "config.yaml")]) == 2
    assert "model" in capsys.readouterr().err.lower()
