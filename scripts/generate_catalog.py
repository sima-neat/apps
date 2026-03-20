#!/usr/bin/env python3
"""Generate a portal-ready catalog.json from per-example README.md files."""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
FIELD_RE = re.compile(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|$")
H2_RE = re.compile(r"^##\s+(.+?)\s*$")
IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
IMAGE_CANDIDATES = (
    "assets/card.png",
    "assets/card.jpg",
    "assets/card.jpeg",
    "assets/card.webp",
    "assets/hero.png",
    "assets/hero.jpg",
    "assets/hero.jpeg",
    "assets/hero.webp",
)
PORTAL_IMAGE_CANDIDATES = (
    "image.png",
    "image.jpg",
    "image.jpeg",
    "image.webp",
)


def readme_title(content: str) -> str:
    first_line = content.splitlines()[0] if content.splitlines() else ""
    return first_line.lstrip("# ").strip()


def parse_metadata(content: str) -> dict[str, str] | None:
    in_table = False
    fields: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Metadata"):
            in_table = True
            continue
        if in_table:
            if stripped.startswith("##") or (stripped and not stripped.startswith("|")):
                break
            match = FIELD_RE.match(stripped)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                if key in {"Field", "---"}:
                    continue
                fields[key] = value
    return fields or None


def portal_asset_relative_path(rel: Path) -> str | None:
    if rel.parts[:2] == ("assets", "portal"):
        return str(Path("example-assets") / Path(*rel.parts[2:]))
    if rel.parts and rel.parts[0] == "examples":
        return str(Path("example-assets") / rel.parent.relative_to("examples") / rel.name)
    return None


def rewrite_markdown_images(markdown: str, readme_path: Path) -> tuple[str, list[str]]:
    repo_root_resolved = REPO_ROOT.resolve()
    asset_paths: list[str] = []
    seen_paths: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        alt_text, raw_path = match.groups()
        path_text = raw_path.strip().strip("<>")

        if "://" in path_text or path_text.startswith(("data:", "#")):
            return match.group(0)

        resolved = (readme_path.parent / path_text).resolve()
        try:
            repo_relative = resolved.relative_to(repo_root_resolved)
        except ValueError:
            return match.group(0)

        if not resolved.exists():
            return match.group(0)

        public_path = portal_asset_relative_path(repo_relative)
        if public_path is None:
            return match.group(0)

        repo_relative_str = str(repo_relative)
        if repo_relative_str not in seen_paths:
            asset_paths.append(repo_relative_str)
            seen_paths.add(repo_relative_str)

        return f"![{alt_text}](./{public_path})"

    return IMAGE_LINK_RE.sub(replace, markdown), asset_paths


def parse_sections(content: str, readme_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    sections: list[dict[str, str]] = []
    asset_paths: list[str] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_lines
        if current_title is None:
          return
        body = "\n".join(current_lines).strip()
        body, section_asset_paths = rewrite_markdown_images(body, readme_path)
        for asset_path in section_asset_paths:
            if asset_path not in asset_paths:
                asset_paths.append(asset_path)
        sections.append({
            "title": current_title,
            "slug": slugify(current_title),
            "markdown": body,
        })
        current_title = None
        current_lines = []

    for line in content.splitlines():
        match = H2_RE.match(line)
        if match:
            flush()
            current_title = match.group(1).strip()
            current_lines = []
            continue
        if current_title is not None:
            current_lines.append(line)
    flush()
    return sections, asset_paths


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def first_nonempty_paragraph(markdown: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", markdown) if p.strip()]
    return paragraphs[0] if paragraphs else ""


def normalize_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def find_image_path(app_dir: Path) -> str | None:
    category = app_dir.parent.name
    app_name = app_dir.name

    portal_asset_dir = REPO_ROOT / "assets" / "portal" / category / app_name
    for rel in PORTAL_IMAGE_CANDIDATES:
        path = portal_asset_dir / rel
        if path.exists():
            return str(path.relative_to(REPO_ROOT))

    for rel in IMAGE_CANDIDATES:
        path = app_dir / rel
        if path.exists():
            return str(path.relative_to(REPO_ROOT))
    return None


def section_map(sections: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {section["slug"]: section for section in sections}


def parse_example(readme: Path) -> dict | None:
    content = readme.read_text()
    metadata = parse_metadata(content)
    if metadata is None:
        print(f"WARNING: no metadata in {readme}", file=sys.stderr)
        return None

    app_dir = readme.parent
    category = app_dir.parent.name
    app_name = app_dir.name
    sections, asset_paths = parse_sections(content, readme)
    sections_by_slug = section_map(sections)
    concept = sections_by_slug.get("concept", {"markdown": ""})["markdown"]

    return {
        "id": f"{category}/{app_name}",
        "name": readme_title(content),
        "category": metadata.get("Category", category),
        "difficulty": metadata.get("Difficulty", ""),
        "languages": metadata.get("Languages", ""),
        "tags": normalize_tags(metadata.get("Tags", "")),
        "status": metadata.get("Status", "experimental"),
        "binary_name": metadata.get("Binary Name", app_name),
        "model": metadata.get("Model", ""),
        "source_path": str(app_dir.relative_to(REPO_ROOT)),
        "readme_path": str(readme.relative_to(REPO_ROOT)),
        "image_path": find_image_path(app_dir),
        "asset_paths": asset_paths,
        "summary": first_nonempty_paragraph(concept),
        "metadata": metadata,
        "sections": sections,
    }


def main() -> int:
    examples: list[dict] = []
    categories: dict[str, int] = {}

    for readme in sorted(EXAMPLES_DIR.glob("*/*/README.md")):
        parsed = parse_example(readme)
        if parsed is None:
            continue
        examples.append(parsed)
        categories[parsed["category"]] = categories.get(parsed["category"], 0) + 1

    catalog = {
        "schema_version": 2,
        "generated_from": "scripts/generate_catalog.py",
        "examples_root": str(EXAMPLES_DIR.relative_to(REPO_ROOT)),
        "categories": [
            {"name": name, "count": count}
            for name, count in sorted(categories.items())
        ],
        "examples": examples,
    }
    print(json.dumps(catalog, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
