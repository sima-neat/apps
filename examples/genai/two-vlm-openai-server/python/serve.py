#!/usr/bin/env python3
"""Serve two VLMs through one pyneat OpenAI-compatible server."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from utils.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not args.config.is_file():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2

    missing = [str(model.path) for model in cfg.models if not model.path.is_dir()]
    if missing:
        print("model directory does not exist:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    import pyneat

    options = pyneat.OpenAIServerOptions()
    options.host = cfg.host
    options.port = cfg.port

    server = pyneat.OpenAIServer(options)
    try:
        for model in cfg.models:
            served_name = server.add_model(model.path, model.name)
            print(f"added model: {served_name} -> {model.path}", flush=True)

        print(f"available models: {', '.join(server.model_names())}", flush=True)
        print(f"serving OpenAI-compatible API on http://{cfg.host}:{cfg.port}", flush=True)
        server.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping OpenAI-compatible API server...", flush=True)
        return 0
    except Exception as exc:
        print(f"server failed: {exc}", file=sys.stderr)
        return 2
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
