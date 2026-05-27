#!/usr/bin/env python3
"""Start a pyneat OpenAI-compatible GenAI server."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


DEFAULT_MODELS = [
    ("/media/nvme/llima/models/gemma3-siglip448-a16w4", "gemma3"),
]


def parse_model(value: str) -> tuple[str, str | None]:
    if ":" not in value:
        return value, None
    model_dir, served_name = value.rsplit(":", 1)
    if not model_dir:
        raise argparse.ArgumentTypeError("model path must not be empty")
    return model_dir, served_name or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9998)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        help="Model directory, optionally suffixed with :served_name. Can be repeated.",
    )
    args = parser.parse_args()

    import pyneat

    options = pyneat.OpenAIServerOptions()
    options.host = args.host
    options.port = args.port

    server = pyneat.OpenAIServer(options)
    model_specs = args.model or DEFAULT_MODELS
    for model_dir, served_name in model_specs:
        path = Path(model_dir)
        if served_name:
            actual_name = server.add_model(path, served_name)
        else:
            actual_name = server.add_model(path)
        print(f"added model: {actual_name} -> {path}", flush=True)

    print(f"serving OpenAI-compatible API on http://{args.host}:{args.port}", flush=True)
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping OpenAI-compatible API server...", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
