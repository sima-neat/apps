import importlib.util
import re
from pathlib import Path


APPS_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generate_readme_gallery_for_test",
    APPS_ROOT / "scripts" / "generate_readme_gallery.py",
)
GALLERY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GALLERY)

README_TEXT = (APPS_ROOT / "README.md").read_text(encoding="utf-8")
IMAGE_REFERENCE_RE = re.compile(r'(?:<img\b[^>]*?\bsrc="([^"]+)")|(?:!\[[^\]]*\]\(([^)\s]+)\))')
IMAGE_SIGNATURES = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".webp": (b"RIFF",),
}


def local_image_references(markdown: str) -> list[str]:
    references = []
    for match in IMAGE_REFERENCE_RE.finditer(markdown):
        target = match.group(1) or match.group(2)
        if not target.startswith(("http://", "https://", "data:")):
            references.append(target)
    return references


def test_gallery_block_is_current():
    block = GALLERY.current_block(README_TEXT)

    assert block is not None, "README.md has no examples:start / examples:end markers"
    assert block == GALLERY.render(GALLERY.read_examples()), (
        "README.md gallery is stale: run python3 scripts/generate_readme_gallery.py"
    )


def test_gallery_links_every_example():
    block = GALLERY.current_block(README_TEXT) or ""
    example_readmes = sorted(
        path.relative_to(APPS_ROOT).as_posix()
        for path in (APPS_ROOT / "examples").glob("*/*/README.md")
    )

    assert example_readmes
    missing = [readme for readme in example_readmes if f"]({readme})" not in block]
    assert missing == []


def test_readme_images_exist_and_are_valid():
    references = local_image_references(README_TEXT)

    assert "portal/assets/readme/hero.png" in references
    for reference in references:
        path = APPS_ROOT / reference
        assert path.is_file(), f"{reference} is referenced by README.md but missing"
        signatures = IMAGE_SIGNATURES[path.suffix.lower()]
        header = path.read_bytes()[:12]
        assert header.startswith(signatures), f"{reference} is not a valid {path.suffix} image"
        if path.suffix.lower() == ".webp":
            assert header[8:12] == b"WEBP", f"{reference} is not a valid WebP image"


def test_render_row_from_example_readme(tmp_path):
    example_dir = tmp_path / "examples" / "tracking" / "demo-tracker"
    example_dir.mkdir(parents=True)
    (example_dir / "README.md").write_text(
        "# Demo Tracker\n\n"
        "## Metadata\n\n"
        "| Field | Value |\n| --- | --- |\n"
        "| Category | tracking |\n| Languages | C++, Python |\n| Status | experimental |\n\n"
        "## Concept\n\n"
        "Tracks people | vehicles across\nseveral streams.\n\nSecond paragraph.\n",
        encoding="utf-8",
    )
    layout = GALLERY.Layout(tmp_path)
    preview_dir = layout.previews / "tracking" / "demo-tracker"
    preview_dir.mkdir(parents=True)
    (preview_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    examples = GALLERY.read_examples(layout)
    block = GALLERY.render(examples, layout)

    assert examples[0].summary == "Tracks people | vehicles across several streams."
    assert "1 application in 1 category." in block
    assert '<img src="portal/assets/examples/tracking/demo-tracker/image.png"' in block
    assert "**Tracking**<br>" in block
    assert (
        "[Demo Tracker](examples/tracking/demo-tracker/README.md)"
        " <sub>C++ · Python · experimental</sub>"
    ) in block
    assert "Tracks people \\| vehicles across several streams." in block
    assert block.startswith(GALLERY.START_MARKER) and block.endswith(GALLERY.END_MARKER)


def test_category_labels_match_portal():
    assert GALLERY.category_label("classification") == "Image Classification"
    assert GALLERY.category_label("genai") == "GenAI"
    assert GALLERY.category_label("object-detection") == "Object Detection"
