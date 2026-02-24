#!/usr/bin/env python3
"""Generate catalog.json from per-example README.md metadata tables."""

import json
import re
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
FIELD_RE = re.compile(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|$")


def parse_metadata(readme: Path) -> dict | None:
    in_table = False
    fields = {}
    for line in readme.read_text().splitlines():
        if line.strip().startswith("## Metadata"):
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("##") or (line.strip() and not line.strip().startswith("|")):
                break
            m = FIELD_RE.match(line)
            if m and m.group(1).strip() != "Field":  # skip header row
                fields[m.group(1).strip()] = m.group(2).strip()
    if not fields:
        return None
    return fields


def main():
    examples = []
    for readme in sorted(EXAMPLES_DIR.glob("*/*/README.md")):
        meta = parse_metadata(readme)
        if meta is None:
            print(f"WARNING: no metadata in {readme}", file=sys.stderr)
            continue

        app_dir = readme.parent
        category = app_dir.parent.name
        app_name = app_dir.name
        tags = [t.strip() for t in meta.get("Tags", "").split(",") if t.strip()]

        examples.append({
            "id": f"{category}/{app_name}",
            "name": readme.read_text().split("\n")[0].lstrip("# ").strip(),
            "category": meta.get("Category", category),
            "tags": tags,
            "status": meta.get("Status", "experimental"),
            "source_path": f"examples/{category}/{app_name}",
            "binary_name": meta.get("Binary Name", app_name),
        })

    catalog = {"schema_version": 1, "examples": examples}
    print(json.dumps(catalog, indent=2))


if __name__ == "__main__":
    main()
