#!/usr/bin/env python3
"""Config-driven entrypoint for the Python multi-camera people tracking example."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from utils.config import load_app_config
from utils.workers import run_app


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "common" / "config.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-camera people detection and tracking example."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "Path to YAML configuration. "
            f"Default: {DEFAULT_CONFIG_PATH}"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_app_config(args.config)
    except FileNotFoundError:
        print(f"Error: config file not found: {args.config}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(f"Error: failed to load config {args.config}: {exc}", file=sys.stderr, flush=True)
        return 2
    return run_app(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
