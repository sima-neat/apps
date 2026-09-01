#!/usr/bin/env python3
"""Multi-stream RTSP YOLO26 object detection - one entry point, two topologies.

    --mode adaptive   (default) ONE GRAPH PER STREAM. A stream is built or torn
                      down while the others keep running, and each stream's
                      delivered resolution is chosen from a shared output
                      bandwidth budget. Per-stream bridges cap reliable
                      metadata at roughly six streams.

    --mode fused      ONE GRAPH for every stream, fanning into a single shared
                      detector, with the source H.264 passed through to Insight
                      untouched (no re-encode). Boxes stay correct at higher
                      stream counts, but adding a stream rebuilds the graph.

The two modes take DIFFERENT config schemas - `adaptive` reads
`streams.sources` plus the `adaptive:` policy sections, `fused` reads a bare
`streams:` list. Passing one mode's config to the other fails validation rather
than running with silently wrong settings.

Everything after --mode is forwarded unchanged to the selected implementation,
so `--config` and `--validate-config-only` behave identically in both.

The C++ port (`src/cpp/main.cpp`) takes the same `--mode` flag, which is what
lets the pipelines UI toggle between languages without changing anything else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODES = ("adaptive", "fused")


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Multi-stream RTSP YOLO26 object detection",
        add_help=False,
    )
    parser.add_argument("--mode", choices=MODES, default="adaptive")
    parser.add_argument("-h", "--help", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, rest = parse_args(argv)

    if args.help:
        print(__doc__)
        print(f"usage: main.py [--mode {{{'|'.join(MODES)}}}] --config CONFIG "
              f"[--validate-config-only]")
        return 0

    # Run as a script the sibling modules are importable already; do it
    # explicitly so `python3 path/to/main.py` works from any directory.
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)

    if args.mode == "fused":
        from fused_app import main as run_selected
    else:
        from adaptive_app import main as run_selected

    return run_selected(rest)


if __name__ == "__main__":
    raise SystemExit(main())
