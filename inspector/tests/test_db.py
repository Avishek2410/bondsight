import pytest

from wirebond_inspector import db
from wirebond_inspector.__main__ import main
from wirebond_inspector.config import ConfigError, load_config
from wirebond_inspector.store import Store


def test_safe_url_hides_password():
    assert db.safe_url("postgresql+psycopg://wb:secret@pi:5432/wb") == "postgresql+psycopg://wb:***@pi:5432/wb"
    assert db.safe_url("postgresql+psycopg://wb@pi:5432/wb") == "postgresql+psycopg://wb@pi:5432/wb"


def test_config_rejects_password_in_url(example_dir):
    cfg = example_dir / "config.yaml"
    cfg.write_text(cfg.read_text() + "\ndatabase: postgresql+psycopg://wb:pw@h/db\n")
    with pytest.raises(ConfigError) as e:
        load_config(cfg)
    assert "put the password in WBI_DB_PASSWORD" in str(e.value)
    assert ":pw@" not in str(e.value)


def test_config_rejects_unknown_database(example_dir):
    cfg = example_dir / "config.yaml"
    cfg.write_text(cfg.read_text() + "\ndatabase: mysql://wb@h/db\n")
    with pytest.raises(ConfigError, match="database"):
        load_config(cfg)


def test_default_database_is_sqlite_in_data_dir(example_dir):
    cfg = load_config(example_dir / "config.yaml")
    assert cfg.database == f"sqlite:///{(example_dir / 'data' / 'inspector2.db').as_posix()}"


def test_password_comes_from_environment(monkeypatch):
    monkeypatch.setenv("WBI_DB_PASSWORD", "s3cret")
    eng = db.make_engine("postgresql+psycopg://wb@pi:5432/wb")
    assert eng.url.password == "s3cret"


def test_database_url_env_overrides(monkeypatch, tmp_path):
    other = f"sqlite:///{(tmp_path / 'other.db').as_posix()}"
    monkeypatch.setenv("WBI_DATABASE_URL", other)
    assert db.make_engine("sqlite:///nowhere.db").url.database == (tmp_path / "other.db").as_posix()


def test_migrate_twice_is_harmless(tmp_path, db_url):
    Store(tmp_path, db_url)
    st = Store(tmp_path, db_url)
    assert st.count() == 0


def test_sqlite_file_is_inspector2(tmp_path):
    Store(tmp_path)
    assert (tmp_path / "inspector2.db").is_file()


def test_unreachable_postgres_gives_plain_error(tmp_path, monkeypatch):
    monkeypatch.setenv("WBI_DB_PASSWORD", "secret")
    monkeypatch.delenv("WBI_DATABASE_URL", raising=False)
    with pytest.raises(db.DatabaseUnavailable) as e:
        Store(tmp_path, "postgresql+psycopg://wb@127.0.0.1:1/wb")
    msg = str(e.value)
    assert "127.0.0.1" in msg and "secret" not in msg and "\n" not in msg


def test_main_exits_2_when_database_unreachable(tmp_path, fake_model, monkeypatch, capsys):
    monkeypatch.setenv("WBI_DB_PASSWORD", "secret")
    monkeypatch.delenv("WBI_DATABASE_URL", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"model: {fake_model().as_posix()}\n"
                   "database: postgresql+psycopg://wb@127.0.0.1:1/wb\n"
                   "cameras:\n  - {name: top, type: generator, view: top}\n")
    assert main(["--config", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert "127.0.0.1" in err and "secret" not in err and "Traceback" not in err
