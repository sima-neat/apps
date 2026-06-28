#!/usr/bin/env python3
"""Start the Detection-to-VLM GenAI server."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def start_genai_server(host: str, port: int, model_name: str, model_path: str):
    import pyneat

    server_options = pyneat.GenAIServerOptions()
    server_options.host = host
    server_options.port = port

    genai_server = pyneat.GenAIServer(server_options)
    served_name = genai_server.add_model(Path(model_path), model_name)
    print(f"added model: {served_name} -> {model_path}", flush=True)
    print(f"serving GenAI API on http://{host}:{port}", flush=True)
    genai_server.start()
    return genai_server


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.config.is_file():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    genai_server_cfg = raw.get("genai_server", {})
    model_cfg = genai_server_cfg.get("model", {})
    host = str(genai_server_cfg.get("host", "0.0.0.0") or "0.0.0.0")
    port = int(genai_server_cfg.get("port", 9998))
    model_name = str(model_cfg.get("name", "") or "")
    model_path = str(model_cfg.get("path", "") or "")

    if not model_name:
        print("config requires genai_server.model.name", file=sys.stderr)
        return 2
    if not model_path:
        print("config requires genai_server.model.path", file=sys.stderr)
        return 2
    if not Path(model_path).is_dir():
        print(f"model directory does not exist: {model_path}", file=sys.stderr)
        return 2

    server = None
    try:
        server = start_genai_server(host, port, model_name, model_path)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping GenAI server...", flush=True)
        return 0
    finally:
        if server is not None:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
