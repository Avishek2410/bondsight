import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_space_folder_has_what_it_needs_and_nothing_private(tmp_path, fake_model):
    model = fake_model()
    out = tmp_path / "space"
    r = subprocess.run([sys.executable, str(REPO / "tools" / "build_space.py"), "--out", str(out),
                        "--model", str(model)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in readme and "app_port: 7860" in readme
    cfg = (out / "config.yaml").read_text()
    assert "demo: true" in cfg and "port: 7860" in cfg and f"models/{model.name}" in cfg
    assert (out / "models" / model.name).is_file() and (out / "models" / model.with_suffix(".json").name).is_file()
    for f in ("Dockerfile", "CREDITS.md", "LICENSE", "inspector/pyproject.toml", "synth/pyproject.toml"):
        assert (out / f).is_file(), f
    assert sorted(p.name for p in (out / "samples").glob("*.jpg")) == \
        sorted(p.name for p in (REPO / "demo" / "samples").glob("*.jpg"))
    names = [p.relative_to(out).as_posix() for p in out.rglob("*")]
    for bad in (".env", "real/", ".venv", "/data/", "__pycache__", ".git/", "pgdata"):
        assert not any(bad in "/" + n + "/" for n in names), bad


def test_builder_refuses_to_overwrite_a_non_space_folder(tmp_path, fake_model):
    out = tmp_path / "x"
    out.mkdir()
    (out / "important.txt").write_text("keep me")
    r = subprocess.run([sys.executable, str(REPO / "tools" / "build_space.py"), "--out", str(out),
                        "--model", str(fake_model())], capture_output=True, text=True)
    assert r.returncode != 0 and (out / "important.txt").read_text() == "keep me"
