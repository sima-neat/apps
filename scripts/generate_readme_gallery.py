#!/usr/bin/env python3
"""Generate the example gallery in the root README from the example READMEs.

The gallery is the block between the ``<!-- examples:start -->`` and
``<!-- examples:end -->`` markers in ``README.md``: a count line and a table
with one row per category, showing a thumbnail and every example in that
category with its Concept summary.

    python3 scripts/generate_readme_gallery.py            # rewrite the gallery
    python3 scripts/generate_readme_gallery.py --check    # exit 1 when it is stale
    python3 scripts/generate_readme_gallery.py --thumbs   # regenerate thumbnails (needs Pillow)
"""

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
START_MARKER = "<!-- examples:start -->"
END_MARKER = "<!-- examples:end -->"
GALLERY_RE = re.compile(f"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}", re.DOTALL)
PREVIEW_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
THUMB_SIZE_PX = (320, 200)
THUMB_WIDTH_IN_README_PX = 220
# Matches formatCategoryLabel in portal/src/App.jsx.
CATEGORY_LABELS = {"classification": "Image Classification", "genai": "GenAI"}


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_readmes_for_gallery", REPO_ROOT / "scripts" / "validate_readmes.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


@dataclass(frozen=True)
class Layout:
    """Where a repository keeps the README, the examples, and their images."""

    root: Path

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    @property
    def examples(self) -> Path:
        return self.root / "examples"

    @property
    def previews(self) -> Path:
        return self.root / "portal" / "assets" / "examples"

    @property
    def thumbs(self) -> Path:
        return self.root / "portal" / "assets" / "readme" / "thumbs"


REPO = Layout(REPO_ROOT)


@dataclass(frozen=True)
class Example:
    category: str
    slug: str
    title: str
    summary: str
    languages: str
    status: str

    @property
    def readme_path(self) -> str:
        return f"examples/{self.category}/{self.slug}/README.md"


def read_example(readme: Path) -> Example:
    content = readme.read_text(encoding="utf-8")
    metadata = VALIDATOR.parse_metadata_table(content) or {}
    sections, _ = VALIDATOR.parse_sections(content)
    first_line = content.splitlines()[0] if content else ""
    return Example(
        category=readme.parent.parent.name,
        slug=readme.parent.name,
        title=first_line.lstrip("# ").strip() or readme.parent.name,
        summary=VALIDATOR.first_paragraph(sections.get("Concept", "")),
        languages=metadata.get("Languages", ""),
        status=metadata.get("Status", ""),
    )


def read_examples(layout: Layout = REPO) -> list[Example]:
    return [read_example(readme) for readme in sorted(layout.examples.glob("*/*/README.md"))]


def group_by_category(examples: list[Example]) -> dict[str, list[Example]]:
    grouped: dict[str, list[Example]] = {}
    for example in examples:
        grouped.setdefault(example.category, []).append(example)
    return grouped


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category) or " ".join(
        part.capitalize() for part in category.split("-")
    )


def preview_path(example: Example, layout: Layout = REPO) -> Path | None:
    for suffix in PREVIEW_SUFFIXES:
        candidate = layout.previews / example.category / example.slug / f"image{suffix}"
        if candidate.exists():
            return candidate
    return None


def thumb_path(category: str, layout: Layout = REPO) -> Path:
    return layout.thumbs / f"{category}.webp"


def gallery_image(category: str, examples: list[Example], layout: Layout = REPO) -> str | None:
    """Repo-relative image for a category row: its thumbnail, else the first preview."""
    thumb = thumb_path(category, layout)
    if thumb.exists():
        return thumb.relative_to(layout.root).as_posix()
    for example in examples:
        preview = preview_path(example, layout)
        if preview is not None:
            return preview.relative_to(layout.root).as_posix()
    return None


def plural(n: int, singular: str, plural_form: str) -> str:
    return f"{n} {singular if n == 1 else plural_form}"


def escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_example(example: Example) -> str:
    tags = [language.strip() for language in example.languages.split(",") if language.strip()]
    if example.status and example.status != "stable":
        tags.append(example.status)
    tag_text = f" <sub>{' · '.join(tags)}</sub>" if tags else ""
    link = f"[{escape_cell(example.title)}]({example.readme_path})"
    return f"{link}{tag_text}<br>{escape_cell(example.summary)}"


def render(examples: list[Example], layout: Layout = REPO) -> str:
    grouped = group_by_category(examples)
    lines = [
        START_MARKER,
        f"{plural(len(examples), 'application', 'applications')} "
        f"in {plural(len(grouped), 'category', 'categories')}.",
        "",
        "| Preview | Applications |",
        "| --- | --- |",
    ]
    for category, members in sorted(grouped.items()):
        label = category_label(category)
        image = gallery_image(category, members, layout)
        preview_cell = (
            f'<img src="{image}" width="{THUMB_WIDTH_IN_README_PX}" alt="{label}">' if image else ""
        )
        applications = "<br>".join(render_example(example) for example in members)
        lines.append(f"| {preview_cell} | **{label}**<br>{applications} |")
    lines.append(END_MARKER)
    return "\n".join(lines)


def current_block(readme_text: str) -> str | None:
    match = GALLERY_RE.search(readme_text)
    return match.group(0) if match else None


def splice(readme_text: str, block: str) -> str:
    if current_block(readme_text) is None:
        raise ValueError(f"README.md has no {START_MARKER} / {END_MARKER} markers")
    return GALLERY_RE.sub(lambda _: block, readme_text, count=1)


def write_thumbs(examples: list[Example], layout: Layout = REPO) -> list[Path]:
    """Write one cover-cropped WebP thumbnail per category from its first example preview."""
    from PIL import Image  # Only this command needs Pillow; CI's tests do not.

    layout.thumbs.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for category, members in sorted(group_by_category(examples).items()):
        source = next((preview_path(e, layout) for e in members if preview_path(e, layout)), None)
        if source is None:
            print(f"warning: category {category} has no preview image to crop", file=sys.stderr)
            continue

        width, height = THUMB_SIZE_PX
        with Image.open(source) as image:
            image = image.convert("RGB")
            scale = max(width / image.width, height / image.height)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
            )
            left = (image.width - width) // 2
            top = (image.height - height) // 2
            image = image.crop((left, top, left + width, top + height))
            target = thumb_path(category, layout)
            image.save(target, "WEBP", quality=82, method=6)
        written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when the README gallery is stale"
    )
    parser.add_argument("--thumbs", action="store_true", help="regenerate the category thumbnails")
    args = parser.parse_args()

    examples = read_examples()
    if args.thumbs:
        for path in write_thumbs(examples):
            print(f"wrote {path.relative_to(REPO_ROOT)}")

    readme_text = REPO.readme.read_text(encoding="utf-8")
    block = render(examples)
    if args.check:
        if current_block(readme_text) != block:
            print(
                "README.md gallery is stale: run python3 scripts/generate_readme_gallery.py",
                file=sys.stderr,
            )
            return 1
        print("README.md gallery is current")
        return 0

    updated = splice(readme_text, block)
    if updated == readme_text:
        print("README.md gallery already current")
        return 0
    REPO.readme.write_text(updated, encoding="utf-8")
    print("updated README.md gallery")
    return 0


if __name__ == "__main__":
    sys.exit(main())
