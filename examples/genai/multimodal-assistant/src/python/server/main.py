#!/usr/bin/env python3
"""Start Neat OpenAI hosting for the Multimodal Assistant example."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys
import time

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from shared.config import AppConfig, DEFAULT_SERVER_CONFIG, load_server_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SERVER_CONFIG)
    return parser


def port_accepts_connections(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.5):
            return True
    except OSError:
        return False


def start_openai_server(cfg: AppConfig):
    try:
        import pyneat
    except ImportError as exc:
        raise RuntimeError(
            "pyneat is required. Run this example from the Python environment "
            "provided by the installed Neat Development Environment."
        ) from exc

    options = pyneat.GenAIServerOptions()
    options.host = cfg.openai.host
    options.port = cfg.openai.port

    server = None
    try:
        server = pyneat.GenAIServer(options)
        for model in cfg.chat_models:
            chat_name = server.add_model(model.path, model.name)
            print(f"added chat model: {chat_name} -> {model.path}", flush=True)
        asr_name = server.add_model(cfg.asr_model.path, cfg.asr_model.name)

        print(f"added ASR model: {asr_name} -> {cfg.asr_model.path}", flush=True)
        print(f"available models: {', '.join(server.model_names())}", flush=True)
        print(
            f"serving OpenAI-compatible API on http://{cfg.openai.host}:{cfg.openai.port}",
            flush=True,
        )

        server.start()
        return server
    except BaseException:
        if server is not None:
            server.stop()
        raise


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.config.is_file():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2

    try:
        cfg = load_server_config(args.config)
    except Exception as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2

    missing = [
        str(model.path)
        for model in (*cfg.chat_models, cfg.asr_model)
        if model.path is None or not model.path.is_dir()
    ]
    if missing:
        print("model directory does not exist:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    server = None
    try:
        if port_accepts_connections(cfg.openai.host, cfg.openai.port):
            raise RuntimeError(
                f"port {cfg.openai.port} is already accepting connections. "
                "Stop the old model server before starting a new one."
            )

        server = start_openai_server(cfg)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping OpenAI-compatible API server...", flush=True)
        return 0
    except Exception as exc:
        print(f"server failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
