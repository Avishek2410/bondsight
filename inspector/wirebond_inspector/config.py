"""Read config.yaml and check it before anything starts.

What: turns the YAML file into simple Python objects (dataclasses) the rest of the app uses.
Why: a typo in the config (wrong camera type, missing model file) should stop the app at once
with ONE clear message listing every problem, instead of crashing later in the middle of an
inspection. Relative paths are read relative to the config file, so the config works from
any folder.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CAMERA_TYPES = ("folder", "generator", "opencv")
VIEWS = ("top", "angled")
TRIGGER_MODES = ("manual", "interval")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")      # camera names become picture file names


class ConfigError(ValueError):
    """The config file has one or more problems (the message lists them all)."""


@dataclass(frozen=True)
class CameraCfg:
    name: str
    type: str
    view: str
    options: dict = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class TrackerCfg:
    k: int = 3
    m: int = 10


@dataclass(frozen=True)
class TriggerCfg:
    mode: str = "manual"
    seconds: float = 5.0


@dataclass(frozen=True)
class AnomalyCfg:
    backbone: Path            # anomaly_backbone.onnx (+ .json next to it)
    dir: Path                 # one <camera name>.npz memory per camera (wbinspect-fit-anomaly)
    size: int | None = None   # picture side for the checker; None = the backbone's .json


@dataclass(frozen=True)
class Config:
    model: Path
    threshold: float
    cameras: tuple[CameraCfg, ...]
    data_dir: Path
    layout: Path | None
    tracker: TrackerCfg
    trigger: TriggerCfg
    bind: str
    port: int
    anomaly: AnomalyCfg | None = None     # optional second opinion (see anomaly.py)
    database: str = ""                    # SQLAlchemy address, no password; "" = SQLite file in data_dir
    demo: bool = False                    # visitor mode (online demo): no cameras, nothing saved
    samples: Path | None = None           # demo: folder of sample pictures visitors can click


def _path(base: Path, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else base / p


def _number(value, kind, where: str, problems: list[str], default):
    try:
        return kind(value)
    except (TypeError, ValueError):
        problems.append(f"{where}: must be a number, got {value!r}")
        return default


def _check_model(model: Path, problems: list[str]) -> None:
    meta = model.with_suffix(".json")
    try:
        m = json.loads(meta.read_text())
    except (OSError, ValueError) as e:
        problems.append(f"{meta}: cannot read model info: {e}")
        return
    if not isinstance(m, dict):
        problems.append(f"{meta}: must be a JSON object")
        return
    missing = [k for k in ("classes", "img_size") if k not in m]
    if missing:
        problems.append(f"{meta}: missing key(s) {', '.join(missing)}")
    t = m.get("tiling")
    if t is not None:
        bad = [k for k in ("scale", "stride") if not isinstance((t or {}).get(k), (int, float)) or t[k] <= 0]
        if bad:
            problems.append(f"{meta}: tiling needs positive numbers for {', '.join(bad)}")


def _check_camera_options(where: str, typ: str, opts: dict, problems: list[str]) -> None:
    if typ == "folder" and not opts.get("path"):
        problems.append(f"{where}: folder camera needs options.path")
    if typ == "generator":
        try:
            from wirebond_synth.defects import InjectionConfig
            InjectionConfig(**(opts.get("injection") or {"mode": "random", "p": 0.05}))
        except (TypeError, ValueError) as e:
            problems.append(f"{where}: bad options.injection: {e}")


def load_config(path: Path) -> Config:
    path = Path(path)
    base = path.parent
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"{path}: cannot read config: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the config must be a list of `key: value` settings")
    problems: list[str] = []

    model = _path(base, raw.get("model", ""))
    if not raw.get("model"):
        problems.append("model: missing (path to the .onnx file)")
    elif not model.is_file():
        problems.append(f"model: file not found: {model}")
    else:
        _check_model(model, problems)

    threshold = raw.get("threshold", 0.4)
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        problems.append(f"threshold: must be between 0 and 1, got {threshold!r}")

    cams: list[CameraCfg] = []
    names: set[str] = set()
    demo = raw.get("demo", False) is True             # visitor mode: only "Check a photo", nothing saved
    raw_cams = raw.get("cameras") or []
    if not isinstance(raw_cams, list) or (not raw_cams and not demo):
        problems.append("cameras: at least one camera is needed (a list)")
        raw_cams = []
    for i, c in enumerate(raw_cams):
        if not isinstance(c, dict):
            problems.append(f"cameras[{i}]: must be a block with name/type/view, got {c!r}")
            continue
        name, typ, view = str(c.get("name", "")), c.get("type"), c.get("view", "top")
        where = f"cameras[{i}] ({name or 'no name'})"
        if not name:
            problems.append(f"{where}: name is missing")
        elif not NAME_RE.match(name):
            problems.append(f"{where}: name {name!r} may only use letters, digits, - and _")
        elif name.lower() in names:
            problems.append(f"{where}: duplicate camera name {name!r}")
        names.add(name.lower())
        if typ not in CAMERA_TYPES:
            problems.append(f"{where}: unknown type {typ!r} (use one of {', '.join(CAMERA_TYPES)})")
        if view not in VIEWS:
            problems.append(f"{where}: unknown view {view!r} (use top or angled)")
        opts = c.get("options") or {}
        if not isinstance(opts, dict):
            problems.append(f"{where}: options must be a block of settings")
            opts = {}
        opts = dict(opts)
        _check_camera_options(where, str(typ), opts, problems)
        if typ == "folder" and opts.get("path"):
            opts["path"] = str(_path(base, opts["path"]))
        cams.append(CameraCfg(name, str(typ), str(view), opts))

    t = raw.get("tracker") or {}
    t = t if isinstance(t, dict) else {}
    tracker = TrackerCfg(_number(t.get("k", 3), int, "tracker.k", problems, 3),
                         _number(t.get("m", 10), int, "tracker.m", problems, 10))
    if not 1 <= tracker.k <= tracker.m:
        problems.append(f"tracker: need 1 <= k <= m, got k={tracker.k} m={tracker.m}")

    tr = raw.get("trigger") or {}
    tr = tr if isinstance(tr, dict) else {}
    trigger = TriggerCfg(str(tr.get("mode", "manual")),
                         _number(tr.get("seconds", 5.0), float, "trigger.seconds", problems, 5.0))
    if trigger.mode not in TRIGGER_MODES:
        problems.append(f"trigger.mode: must be manual or interval, got {trigger.mode!r}")
    if trigger.seconds <= 0:
        problems.append("trigger.seconds: must be > 0")

    layout = _path(base, raw["layout"]) if raw.get("layout") else None
    if layout is not None:
        try:
            lay = json.loads(layout.read_text())
            if not isinstance(lay, dict) or "wires" not in lay or "size" not in lay:
                problems.append(f"layout: {layout} must contain `size` and `wires`")
        except (OSError, ValueError) as e:
            problems.append(f"layout: cannot read {layout}: {e}")

    anomaly = None
    an = raw.get("anomaly")
    if an is not None:
        if not isinstance(an, dict) or not an.get("backbone") or not an.get("dir"):
            problems.append("anomaly: needs `backbone:` (the .onnx) and `dir:` (folder of camera memories)")
        else:
            bb = _path(base, an["backbone"])
            if not bb.is_file() or not bb.with_suffix(".json").is_file():
                problems.append(f"anomaly.backbone: {bb} (and its .json) not found")
            size = an.get("size")
            if size is not None and (not isinstance(size, int) or size < 64 or size % 32):
                problems.append(f"anomaly.size: must be a whole number >= 64 and a multiple of 32, got {size!r}")
            anomaly = AnomalyCfg(bb, _path(base, an["dir"]), size)

    port = raw.get("port", 8080)
    if not isinstance(port, int) or not 0 < port < 65536:
        problems.append(f"port: must be 1..65535, got {port!r}")

    data_dir = _path(base, raw.get("data_dir", "data"))
    database = f"sqlite:///{(data_dir / 'inspector2.db').as_posix()}"
    if raw.get("database"):
        database = str(raw["database"]).strip()
        try:
            from sqlalchemy.engine import make_url
            u = make_url(database)
        except Exception:
            u = None
            problems.append("database: not a valid database address "
                            "(example: postgresql+psycopg://wb@localhost:5432/wirebond)")
        if u is not None and not u.drivername.startswith(("sqlite", "postgresql")):
            problems.append(f"database: use a sqlite:/// or postgresql+psycopg:// address, got {u.drivername!r}")
        elif u is not None and u.password:
            problems.append("database: do not write the password in the config; "
                            "put the password in WBI_DB_PASSWORD (Docker: docker/.env)")
    if demo:
        database = "sqlite://"                         # in memory: a visitor's photo is never stored
    samples = _path(base, raw["samples"]) if raw.get("samples") else None
    if samples is not None and not samples.is_dir():
        problems.append(f"samples: folder not found: {samples}")

    if problems:
        raise ConfigError(f"{path}: please fix:\n  - " + "\n  - ".join(problems))
    return Config(model=model, threshold=float(threshold), cameras=tuple(cams),
                  data_dir=data_dir, layout=layout,
                  tracker=tracker, trigger=trigger, bind=str(raw.get("bind", "127.0.0.1")),
                  port=port, anomaly=anomaly, database=database, demo=demo, samples=samples)
