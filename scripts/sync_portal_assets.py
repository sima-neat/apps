#!/usr/bin/env python3
"""Copy portal-consumable example assets into portal/public."""

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PORTAL_PUBLIC = REPO_ROOT / "portal" / "public"
CATALOG_PATH = PORTAL_PUBLIC / "catalog.json"
ASSET_ROOT = PORTAL_PUBLIC / "example-assets"


def portal_asset_destination(rel: Path) -> Path:
    if rel.parts[:2] == ("assets", "portal"):
        return ASSET_ROOT / Path(*rel.parts[2:])
    if rel.parts and rel.parts[0] == "examples":
        return ASSET_ROOT / rel.parent.relative_to("examples") / rel.name
    raise ValueError(f"Unsupported portal asset path: {rel}")


def copy_asset(asset_path: str) -> str | None:
    src = REPO_ROOT / asset_path
    if not src.exists():
        return None

    rel = Path(asset_path)
    dst = portal_asset_destination(rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(PORTAL_PUBLIC))


def main() -> int:
    if not CATALOG_PATH.exists():
        raise SystemExit(f"Missing catalog file: {CATALOG_PATH}")

    data = json.loads(CATALOG_PATH.read_text())

    if ASSET_ROOT.exists():
        shutil.rmtree(ASSET_ROOT)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)

    for example in data.get("examples", []):
        image_path = example.get("image_path")
        if image_path:
            example["image_path"] = copy_asset(image_path)

        synced_asset_paths: list[str] = []
        for asset_path in example.get("asset_paths", []):
            synced_path = copy_asset(asset_path)
            if synced_path and synced_path not in synced_asset_paths:
                synced_asset_paths.append(synced_path)
        if synced_asset_paths:
            example["asset_paths"] = synced_asset_paths

    CATALOG_PATH.write_text(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
