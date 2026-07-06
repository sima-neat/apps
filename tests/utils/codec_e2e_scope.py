"""Write the focused codec-source e2e scope used by CI gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.utils.test_scope import APPS_ROOT, load_scope


TARGETS = {
    "object-detection/single-stream-object-detector": {"python"},
    "segmentation/single-stream-instance-segmenter": {"cpp", "python"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope-file", type=Path, default=APPS_ROOT / "examples")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scope = load_scope(args.scope_file, APPS_ROOT)
    for example_key, entry in scope["examples"].items():
        e2e = entry.get("e2e", {})
        for language in ("cpp", "python"):
            config = e2e.get(language)
            if not isinstance(config, dict):
                continue
            if language in TARGETS.get(example_key, set()):
                config["enabled"] = True
            else:
                config["enabled"] = False
                config["models"] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(scope, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
