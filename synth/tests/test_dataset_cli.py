import json

import pytest

from wirebond_synth.classes import VIEWS
from wirebond_synth.cli import main
from wirebond_synth.dataset import generate
from wirebond_synth.defects import InjectionConfig


def test_generate_writes_coco(tmp_path):
    s = generate(tmp_path / "d", 6, 16, 1, InjectionConfig("random", p=0.2))
    assert s["units"] == 6
    for v in VIEWS:
        coco = json.loads((tmp_path / "d/annotations" / f"{v}.json").read_text())
        assert len(coco["images"]) == 6 and len(coco["categories"]) == 8
        assert len(coco["annotations"]) == sum(s["boxes"][v].values())
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            assert w >= 2 and h >= 2 and 1 <= a["category_id"] <= 8
        for im in coco["images"]:
            assert (tmp_path / "d/images" / v / im["file_name"]).is_file()
    assert len(list((tmp_path / "d/units").glob("*.json"))) == 6
    assert (tmp_path / "d/layout.json").is_file() and (tmp_path / "d/config.json").is_file()


def test_generate_deterministic(tmp_path):
    for n in ("a", "b"):
        generate(tmp_path / n, 3, 16, 4, InjectionConfig("random", p=0.3))
    for f in ("images/top/u00002.png", "annotations/angled.json", "units/u00001.json"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_cli_refuses_existing_dir(tmp_path):
    generate(tmp_path / "d", 1, 8, 1, InjectionConfig("none"))
    with pytest.raises(FileExistsError):
        generate(tmp_path / "d", 1, 8, 1, InjectionConfig("none"))
    generate(tmp_path / "d", 1, 8, 1, InjectionConfig("none"), overwrite=True)


def test_overwrite_never_deletes_foreign_dir(tmp_path):
    foreign = tmp_path / "mine"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        generate(foreign, 1, 8, 1, InjectionConfig("none"), overwrite=True)
    assert (foreign / "precious.txt").read_text() == "keep"


def test_cli_main_runs(tmp_path, capsys):
    assert main(["generate", "--out", str(tmp_path / "c"), "--units", "2", "--wires", "8",
                 "--mode", "repetitive", "--wire", "2", "--cls", "wire_sweep"]) == 0
    assert '"units": 2' in capsys.readouterr().out


def test_cli_bad_args_exit_nonzero(tmp_path):
    assert main(["generate", "--out", str(tmp_path / "x"), "--units", "1",
                 "--mode", "repetitive", "--wire", "99", "--cls", "wire_sweep"]) == 2


def test_gallery(tmp_path):
    assert main(["gallery", "--out", str(tmp_path / "g")]) == 0
    assert len(list((tmp_path / "g").glob("*.png"))) == 9


def test_generate_jpg(tmp_path):
    generate(tmp_path / "j", 2, 8, 1, InjectionConfig("random", p=0.5), fmt="jpg", quality=95)
    coco = json.loads((tmp_path / "j/annotations/top.json").read_text())
    assert all(im["file_name"].endswith(".jpg") for im in coco["images"])
    f = tmp_path / "j/images/top" / coco["images"][0]["file_name"]
    assert f.read_bytes()[:2] == b"\xff\xd8"                       # JPEG magic
    assert json.loads((tmp_path / "j/config.json").read_text())["format"] == "jpg"
    with pytest.raises(ValueError):
        generate(tmp_path / "x", 1, 8, 1, InjectionConfig("none"), fmt="bmp")
