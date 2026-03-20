"""Regression tests for portal image asset discovery and sync."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


APPS_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_portal_cards_always_use_category_asset_cover():
    app_jsx = (APPS_ROOT / "portal" / "src" / "App.jsx").read_text()

    assert 'src={fallbackImage}' in app_jsx
    assert 'src={example.image_path ? `./${example.image_path}` : fallbackImage}' not in app_jsx


def test_generate_catalog_uses_repo_level_portal_assets_for_example():
    module = _load_module(
        "generate_catalog_for_test",
        APPS_ROOT / "scripts" / "generate_catalog.py",
    )

    app_dir = (
        APPS_ROOT
        / "examples"
        / "tracking"
        / "multi-camera-people-detection-and-tracking-optiview"
    )

    assert module.find_image_path(app_dir) == (
        "assets/portal/tracking/"
        "multi-camera-people-detection-and-tracking-optiview/image.png"
    )


def test_generate_catalog_rewrites_readme_preview_images_to_portal_assets(tmp_path):
    module = _load_module(
        "generate_catalog_for_preview_test",
        APPS_ROOT / "scripts" / "generate_catalog.py",
    )

    repo_root = tmp_path / "apps"
    readme = repo_root / "examples" / "tracking" / "demo-example" / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "\n".join(
            [
                "# Demo Example",
                "",
                "## Metadata",
                "| Field | Value |",
                "| --- | --- |",
                "| Category | tracking |",
                "| Difficulty | Intermediate |",
                "| Tags | tracking |",
                "| Languages | C++, Python |",
                "| Status | experimental |",
                "| Binary Name | demo-example |",
                "| Model | yolo_v8m |",
                "",
                "## Concept",
                "Demo concept.",
                "",
                "## Preview",
                "![Demo](../../../assets/portal/tracking/demo-example/image.png)",
            ]
        )
    )

    asset = repo_root / "assets" / "portal" / "tracking" / "demo-example" / "image.png"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"png")

    module.REPO_ROOT = repo_root
    parsed = module.parse_example(readme)

    assert parsed is not None
    preview = next(section for section in parsed["sections"] if section["slug"] == "preview")
    assert preview["markdown"] == "![Demo](./example-assets/tracking/demo-example/image.png)"
    assert parsed["asset_paths"] == ["assets/portal/tracking/demo-example/image.png"]


def test_sync_portal_assets_copies_repo_level_portal_asset(tmp_path):
    module = _load_module(
        "sync_portal_assets_for_test",
        APPS_ROOT / "scripts" / "sync_portal_assets.py",
    )

    repo_root = tmp_path / "apps"
    portal_public = repo_root / "portal" / "public"
    asset_root = portal_public / "example-assets"
    source_image = (
        repo_root
        / "assets"
        / "portal"
        / "tracking"
        / "demo-example"
        / "image.png"
    )
    source_image.parent.mkdir(parents=True, exist_ok=True)
    source_image.write_bytes(b"png")

    catalog_path = portal_public / "catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "id": "tracking/demo-example",
                        "image_path": "assets/portal/tracking/demo-example/image.png",
                    }
                ]
            }
        )
    )

    module.REPO_ROOT = repo_root
    module.PORTAL_PUBLIC = portal_public
    module.CATALOG_PATH = catalog_path
    module.ASSET_ROOT = asset_root

    assert module.main() == 0

    synced_image = asset_root / "tracking" / "demo-example" / "image.png"
    assert synced_image.read_bytes() == b"png"

    updated_catalog = json.loads(catalog_path.read_text())
    assert updated_catalog["examples"][0]["image_path"] == (
        "example-assets/tracking/demo-example/image.png"
    )


def test_sync_portal_assets_copies_readme_linked_assets(tmp_path):
    module = _load_module(
        "sync_portal_assets_for_preview_test",
        APPS_ROOT / "scripts" / "sync_portal_assets.py",
    )

    repo_root = tmp_path / "apps"
    portal_public = repo_root / "portal" / "public"
    asset_root = portal_public / "example-assets"
    source_image = (
        repo_root
        / "assets"
        / "portal"
        / "tracking"
        / "demo-example"
        / "image.png"
    )
    source_image.parent.mkdir(parents=True, exist_ok=True)
    source_image.write_bytes(b"png")

    catalog_path = portal_public / "catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "id": "tracking/demo-example",
                        "image_path": None,
                        "asset_paths": ["assets/portal/tracking/demo-example/image.png"],
                        "sections": [
                            {
                                "title": "Preview",
                                "slug": "preview",
                                "markdown": "![Demo](./example-assets/tracking/demo-example/image.png)",
                            }
                        ],
                    }
                ]
            }
        )
    )

    module.REPO_ROOT = repo_root
    module.PORTAL_PUBLIC = portal_public
    module.CATALOG_PATH = catalog_path
    module.ASSET_ROOT = asset_root

    assert module.main() == 0

    synced_image = asset_root / "tracking" / "demo-example" / "image.png"
    assert synced_image.read_bytes() == b"png"
