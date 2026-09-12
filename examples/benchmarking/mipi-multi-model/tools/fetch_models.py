#!/usr/bin/env python3
"""Fetch the canonical MPK archives used by the MIPI multi-model example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_SOURCE = Path(__file__).resolve().parents[1] / "src" / "python"
sys.path.insert(0, str(PYTHON_SOURCE))

from model_profiles import (
    PROFILES,
    ModelPackageError,
    fetch_profile,
    profile_named,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*", choices=PROFILES)
    parser.add_argument("--all", action="store_true", help="fetch every profile")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    names = list(PROFILES) if args.all else args.profiles
    if not names:
        parser.error("provide one or more profiles, or use --all")

    failed = False
    for name in names:
        profile = profile_named(name)
        print(f"FETCH {profile.name}: {profile.source}")
        try:
            package = fetch_profile(profile, args.models_dir)
            print(f"OK {profile.name}: {package.path}")
        except ModelPackageError as exc:
            failed = True
            print(f"FAIL {profile.name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
