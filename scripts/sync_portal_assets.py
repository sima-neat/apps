#!/usr/bin/env python3
"""Copy portal-consumable example assets into portal/public."""

import json
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PORTAL_PUBLIC = REPO_ROOT / "portal" / "public"
CATALOG_PATH = PORTAL_PUBLIC / "catalog.json"
ASSET_ROOT = PORTAL_PUBLIC / "example-assets"
MARKDOWN_IMAGE_RE = re.compile(r"(!\[[^\]]*]\()([^)]+)(\))")
HTML_IMAGE_SRC_RE = re.compile(r'(<img\b[^>]*?\bsrc\s*=\s*["\'])([^"\']+)(["\'])', re.IGNORECASE)


def is_local_asset_ref(value: str) -> bool:
    lower = value.lower()
    return not (
        lower.startswith("http://")
        or lower.startswith("https://")
        or lower.startswith("data:")
        or lower.startswith("#")
        or lower.startswith("/")
    )


def copy_example_asset_path(source_path: str) -> str | None:
    src = REPO_ROOT / source_path
    if not src.exists():
        return None

    rel = Path(source_path)
    dst = ASSET_ROOT / rel.parent.relative_to("examples") / rel.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(PORTAL_PUBLIC))


def resolve_and_copy_ref(example_source_path: str, ref: str) -> str | None:
    if not is_local_asset_ref(ref):
        return None

    app_dir = REPO_ROOT / example_source_path
    src = (app_dir / ref).resolve()
    if not src.exists() or not src.is_file():
        return None

    try:
        src_rel = src.relative_to(REPO_ROOT)
    except ValueError:
        return None

    copied_rel = copy_example_asset_path(str(src_rel))
    if not copied_rel:
        return None
    return f"./{copied_rel}"


def rewrite_section_assets(example: dict) -> None:
    source_path = example.get("source_path", "")
    if not source_path:
        return

    for section in example.get("sections", []):
        markdown = section.get("markdown", "")

        def replace_markdown(match: re.Match[str]) -> str:
            ref = match.group(2).strip()
            rewritten = resolve_and_copy_ref(source_path, ref)
            if not rewritten:
                return match.group(0)
            return f"{match.group(1)}{rewritten}{match.group(3)}"

        def replace_html(match: re.Match[str]) -> str:
            ref = match.group(2).strip()
            rewritten = resolve_and_copy_ref(source_path, ref)
            if not rewritten:
                return match.group(0)
            return f"{match.group(1)}{rewritten}{match.group(3)}"

        updated = MARKDOWN_IMAGE_RE.sub(replace_markdown, markdown)
        updated = HTML_IMAGE_SRC_RE.sub(replace_html, updated)
        section["markdown"] = updated


def main() -> int:
    if not CATALOG_PATH.exists():
        raise SystemExit(f"Missing catalog file: {CATALOG_PATH}")

    data = json.loads(CATALOG_PATH.read_text())

    if ASSET_ROOT.exists():
        shutil.rmtree(ASSET_ROOT)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)

    for example in data.get("examples", []):
        image_path = example.get("image_path")
        if not image_path:
            rewrite_section_assets(example)
            continue

        copied_rel = copy_example_asset_path(image_path)
        if not copied_rel:
            example["image_path"] = None
        else:
            example["image_path"] = copied_rel
        rewrite_section_assets(example)

    CATALOG_PATH.write_text(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
