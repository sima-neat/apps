#!/usr/bin/env python3
"""Entrypoint for the multistream YOLOv8 OptiView example."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from utils.config import load_app_config
from utils.model_family import resolve_model_family


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "common" / "config.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multistream YOLOv8 object detection with OptiView output."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to YAML configuration. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the config and exit without opening RTSP streams or runtime workers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr, flush=True)
        return 2

    try:
        cfg = load_app_config(config_path)
        family = resolve_model_family(cfg.model.path, cfg.model.family)
    except Exception as exc:
        print(f"Error: failed to load config {config_path}: {exc}", file=sys.stderr, flush=True)
        return 2

    if args.validate_config_only:
        print(
            f"Config validated: {config_path} "
            f"(family={family}, workers={cfg.worker_count}, streams={len(cfg.rtsp_urls)})",
            flush=True,
        )
        return 0

    from utils.workers import run_app

    return run_app(cfg, family)


if __name__ == "__main__":
    raise SystemExit(main())
