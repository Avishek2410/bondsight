"""Start the inspector:  wbinspect --config config.yaml   (or python -m wirebond_inspector ...)

It reads the config, loads the model, connects the cameras, starts the interval trigger if
configured, and serves the web page. By default only this computer can open the page
(127.0.0.1); set `bind: 0.0.0.0` in the config (or --host) to allow other machines.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wbinspect", description="BondSight wire-bond inspector")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--host", help="override `bind` from the config")
    ap.add_argument("--port", type=int, help="override `port` from the config")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .config import ConfigError, load_config
    try:
        cfg = load_config(a.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2

    import uvicorn

    from .db import DatabaseUnavailable
    from .service import Inspector
    from .web.app import create_app

    try:
        ins = Inspector(cfg)
    except DatabaseUnavailable as e:   # one plain line, never the password
        print(f"wbinspect: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # e.g. the model file is not a valid ONNX model, data folder not writable
        print(f"wbinspect: cannot start: {type(e).__name__}: {e}\n"
              f"  (check `model:` = {cfg.model} and that `data_dir:` = {cfg.data_dir} is writable)",
              file=sys.stderr)
        return 2
    ins.start_trigger()
    host, port = a.host or cfg.bind, a.port or cfg.port
    print(f"BondSight: open http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}")
    try:
        uvicorn.run(create_app(ins), host=host, port=port, log_level="warning")
    finally:
        ins.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
