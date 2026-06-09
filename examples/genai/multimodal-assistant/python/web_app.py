#!/usr/bin/env python3
"""Start the Multimodal Assistant Flask UI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from app_config import DEFAULT_WEB_CONFIG, load_web_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_WEB_CONFIG)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.config.is_file():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2

    try:
        cfg = load_web_config(args.config)
    except Exception as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2

    app_dir = Path(__file__).resolve().parent
    os.chdir(app_dir)

    try:
        from app import run_app

        run_app(cfg)
    except KeyboardInterrupt:
        print("\nstopping Flask UI...", flush=True)
    except Exception as exc:
        print(f"web app failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
