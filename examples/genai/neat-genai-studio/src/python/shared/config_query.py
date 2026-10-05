#!/usr/bin/env python3
"""Print one Studio config value exactly as the UI reads it (load_ui_config).

run.sh and setup.sh call this so the supervisor and the installer interpret a
customer's config.local.yaml with the same YAML semantics (block or flow style,
any indentation, quoting, booleans) as the application itself:

    python3 src/python/shared/config_query.py --config config.local.yaml web.headless

Booleans print as true/false, unset values as an empty line. Exit status 2 when
the config cannot be loaded (the shell callers then fall back to their own
reader).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import load_ui_config  # noqa: E402

FIELDS = {
    "web.host": lambda c: c.web.host,
    "web.port": lambda c: c.web.port,
    "web.https": lambda c: c.web.https,
    "web.headless": lambda c: c.web.headless,
    "web.cors_origins": lambda c: c.web.cors_origins,
    "supertonic.models_root": lambda c: c.supertonic.models_root,
    "supertonic.venv": lambda c: c.supertonic.venv,
}


def render(value) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "" if value is None else str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("field", choices=sorted(FIELDS))
    args = parser.parse_args(argv)
    try:
        cfg = load_ui_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"config_query: {exc}", file=sys.stderr)
        return 2
    print(render(FIELDS[args.field](cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
