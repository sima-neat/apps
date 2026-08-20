#!/usr/bin/env python3
"""Start the Neat GenAI Studio Flask UI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from shared.config import DEFAULT_UI_CONFIG, load_ui_config


def _request_shutdown(signum, frame):
    """SIGTERM -> KeyboardInterrupt so run_ui()'s finally (which stops the RAG
    worker) runs on a clean shutdown, matching Ctrl+C behavior."""
    raise KeyboardInterrupt()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_UI_CONFIG)
    return parser


def main() -> int:
    signal.signal(signal.SIGTERM, _request_shutdown)
    args = build_arg_parser().parse_args()
    if not args.config.is_file():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2

    try:
        cfg = load_ui_config(args.config)
    except Exception as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2

    app_dir = Path(__file__).resolve().parent
    os.chdir(app_dir)

    try:
        from flask_app import run_ui

        run_ui(cfg)
    except KeyboardInterrupt:
        print("\nstopping Flask UI...", flush=True)
    except Exception as exc:
        print(f"web app failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
