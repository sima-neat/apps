#!/usr/bin/env python3
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def build_command(argv: list[str]) -> str:
    quoted_args = " ".join(shlex.quote(arg) for arg in argv)
    return f"./tests/test.sh {quoted_args}".rstrip()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    apps_root = root_dir()
    command = build_command(args)
    return subprocess.run(["bash", "-lc", command], cwd=apps_root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
