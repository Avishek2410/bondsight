"""Where the inspector keeps its records: PostgreSQL in a factory, SQLite for tests and the demo.

What: one description of the tables (below), one way to connect (`make_engine`) and one way to
bring an older database up to date (`migrate`, using Alembic "migrations": small numbered
scripts that each change the tables a little, so an upgrade never loses data).
Why PostgreSQL: several people and programs can read and write at the same time, it can live on a
central server for many stations, and factory IT already knows how to back it up.
Why SQLite too: it is one file and needs no server, which keeps tests and the online demo simple.
Passwords: the database password is never written in config.yaml. It comes from the environment
variable WBI_DB_PASSWORD (Docker puts it there from the git-ignored file docker/.env).
WBI_DATABASE_URL, if set, replaces the database address from the config completely.
"""
from __future__ import annotations

import os
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.engine import Engine, make_url

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


class DatabaseUnavailable(RuntimeError):
    """The database cannot be reached (wrong address, server down, wrong password)."""


metadata = sa.MetaData()

units = sa.Table(
    "units", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("unit_no", sa.Integer, nullable=True),          # None for uploaded photos
    sa.Column("ts", sa.Float, nullable=False),
    sa.Column("source", sa.Text, nullable=False, server_default="camera"),   # camera | upload
    sa.Column("verdict", sa.Text, nullable=False),            # OK | DEFECT | CHECK
    sa.Column("label", sa.Text, nullable=True),               # uploaded file name
)
shots = sa.Table(
    "shots", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("unit_id", sa.Integer, sa.ForeignKey("units.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("camera", sa.Text), sa.Column("view", sa.Text), sa.Column("source", sa.Text),
    sa.Column("image_path", sa.Text), sa.Column("error", sa.Text),
    sa.Column("anomaly", sa.Float), sa.Column("anomaly_flag", sa.Boolean), sa.Column("heat_path", sa.Text),
    sa.Column("width", sa.Integer), sa.Column("height", sa.Integer),
    sa.Column("hints", sa.Text),                               # JSON list of plain-language notes
)
findings = sa.Table(
    "findings", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("unit_id", sa.Integer, sa.ForeignKey("units.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("camera", sa.Text), sa.Column("cls", sa.Text), sa.Column("score", sa.Float),
    sa.Column("x0", sa.Float), sa.Column("y0", sa.Float), sa.Column("x1", sa.Float), sa.Column("y1", sa.Float),
    sa.Column("position", sa.Text),
)
alerts = sa.Table(
    "alerts", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("unit_id", sa.Integer, sa.ForeignKey("units.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("position", sa.Text), sa.Column("cls", sa.Text),
    sa.Column("count", sa.Integer), sa.Column("window", sa.Integer),
)
decisions = sa.Table(
    "decisions", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("unit_id", sa.Integer, sa.ForeignKey("units.id", ondelete="CASCADE"), nullable=False, index=True),
    sa.Column("finding_id", sa.Integer, sa.ForeignKey("findings.id", ondelete="CASCADE"), nullable=True),
    sa.Column("verdict", sa.Text, nullable=False),     # real | false_alarm | missed | confirm_ok | withdraw
    sa.Column("camera", sa.Text), sa.Column("cls", sa.Text),
    sa.Column("x0", sa.Float), sa.Column("y0", sa.Float), sa.Column("x1", sa.Float), sa.Column("y1", sa.Float),
    sa.Column("operator", sa.Text, nullable=False),
    sa.Column("ts", sa.Float, nullable=False),
    sa.Column("ref_id", sa.Integer, nullable=True),            # withdraw -> the "missed" decision it cancels
)
sa.Index("decisions_finding", decisions.c.finding_id)
sa.Index("decisions_ref", decisions.c.ref_id)
sa.Index("units_verdict", units.c.verdict)
sa.Index("units_source", units.c.source)


def sqlite_url(data_dir: Path) -> str:
    return f"sqlite:///{(Path(data_dir) / 'inspector2.db').as_posix()}"


def safe_url(url: str) -> str:
    """The address with any password replaced by *** (safe to show on the page or in logs)."""
    return make_url(url).render_as_string(hide_password=True)


def where(url: str) -> str:
    """Short 'host/database' (or the SQLite file) for error messages."""
    u = make_url(url)
    return u.database or "?" if u.drivername.startswith("sqlite") else f"{u.host}:{u.port or 5432}/{u.database}"


def make_engine(url: str) -> Engine:
    url = os.environ.get("WBI_DATABASE_URL") or url
    u = make_url(url)
    if u.drivername.startswith("sqlite"):
        if not u.database or u.database == ":memory:":   # demo: one shared in-memory database
            return sa.create_engine(u, connect_args={"check_same_thread": False}, poolclass=sa.pool.StaticPool)
        eng = sa.create_engine(u, connect_args={"check_same_thread": False})

        @sa.event.listens_for(eng, "connect")
        def _sqlite_setup(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
        return eng
    if os.environ.get("WBI_DB_PASSWORD") and not u.password:
        u = u.set(password=os.environ["WBI_DB_PASSWORD"])
    return sa.create_engine(u, pool_pre_ping=True, connect_args={"connect_timeout": 5})


def migrate(engine: Engine) -> None:
    """Create the tables, or update them to the newest version (safe to run every start)."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")


def connect(url: str) -> Engine:
    """make_engine + a first connection + migrate; a failure becomes one plain DatabaseUnavailable."""
    engine = make_engine(url)
    try:
        with engine.connect():
            pass
    except sa.exc.OperationalError as e:
        reason = str(getattr(e, "orig", e)).strip().splitlines()[0] if str(getattr(e, "orig", e)).strip() else "no answer"
        pw = engine.url.password
        if pw:
            reason = reason.replace(pw, "***")
        engine.dispose()
        raise DatabaseUnavailable(f"cannot reach the database at {where(str(engine.url))}: {reason}") from None
    migrate(engine)
    return engine
