#!/usr/bin/env python3
from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def remove_stale_build_dir(apps_root: Path) -> None:
    cache_path = apps_root / "build" / "CMakeCache.txt"
    if not cache_path.exists():
        return

    expected = f"CMAKE_HOME_DIRECTORY:INTERNAL={apps_root}"
    try:
        cache_text = cache_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return

    if expected not in cache_text:
        print("[task] removing stale build directory created from a different checkout path", flush=True)
        shutil.rmtree(apps_root / "build", ignore_errors=True)


def build_command(argv: list[str]) -> str:
    quoted_args = " ".join(shlex.quote(arg) for arg in argv)
    return (
        "./build.sh && "
        "source tests/scripts/testing/setup_test_env.sh && "
        f"./tests/test.sh {quoted_args}".rstrip()
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    apps_root = root_dir()
    remove_stale_build_dir(apps_root)
    command = build_command(args)
    return subprocess.run(["bash", "-lc", command], cwd=apps_root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
