#!/usr/bin/env python3
from __future__ import annotations

import subprocess


def main() -> int:
    return subprocess.run(["bash", "/usr/bin/fix_devkit_runtime.sh"]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
