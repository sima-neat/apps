#!/usr/bin/env python3
"""Validate that every example README.md conforms to the required contract.

Scans examples/*/*/README.md files and verifies metadata, portal summaries and
previews, configuration guidance, and supported install and run commands.

Exit code 0 on success, 1 if any README is missing or malformed.
"""

import re
import sys
from pathlib import Path

VALID_CATEGORIES = {
    "anomaly-detection",
    "benchmarking",
    "classification",
    "object-detection",
    "tracking",
    "segmentation",
    "pose-estimation",
    "depth-estimation",
    "face-detection",
    "feature-extraction",
    "genai",
    "throughput",
}

VALID_DIFFICULTIES = {"Beginner", "Intermediate", "Advanced"}

VALID_LANGUAGES = {"C++", "Python"}

VALID_STATUSES = {"stable"}

REQUIRED_METADATA_FIELDS = {
    "Category",
    "Difficulty",
    "Tags",
    "Languages",
    "Status",
    "Binary Name",
    "Model",
}
MODEL_REFERENCE_RE = re.compile(
    r"^(?P<label>[^\[]+?)(?:\s*\[(?P<url>https?://[^\]]+)\])?$"
)

REQUIRED_SECTIONS = {
    "Metadata",
    "Concept",
    "Preview",
    "Prerequisites",
    "Install Apps",
    "Prepare the Model",
    "Configure",
    "Run",
    "Source Files",
    "Development From Source",
}

REQUIRED_SECTION_ORDER = (
    "Metadata",
    "Concept",
    "Preview",
    "Prerequisites",
    "Install Apps",
    "Prepare the Model",
    "Configure",
    "Run",
    "Source Files",
    "Development From Source",
)

INSTALL_COMMAND = "sima-cli neat install apps"
STANDALONE_INSTALL_SCRIPT = "scripts/get-example.sh"
CONTRIBUTING_URL = "https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md"
SOURCE_ONLY_COMMANDS = (
    "git clone ",
    "build.sh",
    "cmake ",
    "ctest ",
    "./build/",
    "tests/test.sh",
)
SUMMARY_MAX_CHARS = 200
SUMMARY_FILLER_PREFIXES = (
    "this example",
    "this application",
    "this app",
    "the example",
)
MARKDOWN_IN_SUMMARY_RE = re.compile(r"[`*_\[\]<>]|!\[")
PREVIEW_IMAGE_RE = re.compile(
    r"^[ \t]{0,3}!\[[^\]]*\]\(([^)]+)\)[ \t]*$", re.MULTILINE
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^>]*)?/?>")
FENCE_START_RE = re.compile(r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})")
PREVIEW_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}


def parse_metadata_table(content: str) -> dict[str, str] | None:
    """Extract field->value pairs from the Metadata markdown table."""
    metadata = {}
    in_table = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Metadata"):
            in_table = True
            continue
        if in_table:
            if stripped.startswith("## "):
                break
            match = re.match(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|", stripped)
            if match:
                key, value = match.group(1).strip(), match.group(2).strip()
                if key in ("---", "Field"):
                    continue
                metadata[key] = value
    return metadata if metadata else None


def parse_sections(content: str) -> tuple[dict[str, str], list[str]]:
    """Return H2 section bodies and their document order."""
    sections: dict[str, list[str]] = {}
    order: list[str] = []
    current: str | None = None
    for line in content.splitlines():
        match = re.match(r"^##\s+(.+)$", line)
        if match:
            current = match.group(1).strip()
            if current not in sections:
                sections[current] = []
                order.append(current)
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}, order


def first_paragraph(section: str) -> str:
    """Return the first nonempty paragraph with whitespace normalized."""
    paragraph: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    return " ".join(" ".join(paragraph).split())


def strip_fenced_code(section: str) -> str:
    """Remove fenced code blocks before checking rendered Markdown content."""
    visible: list[str] = []
    fence_char: str | None = None
    fence_length = 0
    for line in section.splitlines():
        if fence_char is not None:
            closing = re.match(
                rf"^[ \t]{{0,3}}{re.escape(fence_char)}{{{fence_length},}}[ \t]*$",
                line,
            )
            if closing:
                fence_char = None
                fence_length = 0
            continue

        opening = FENCE_START_RE.match(line)
        if opening:
            fence = opening.group("fence")
            fence_char = fence[0]
            fence_length = len(fence)
            continue
        visible.append(line)
    return "\n".join(visible)


def validate_portal_content(
    *,
    sections: dict[str, str],
    metadata: dict[str, str],
    readme_path: Path,
) -> list[str]:
    """Validate the summary and preview consumed by the portal."""
    errors: list[str] = []
    summary = first_paragraph(sections.get("Concept", ""))
    if not summary:
        errors.append("Concept must start with a portal summary")
    else:
        if len(summary) > SUMMARY_MAX_CHARS:
            errors.append(
                f"Concept summary must be at most {SUMMARY_MAX_CHARS} characters "
                f"after whitespace is normalized; found {len(summary)}"
            )
        if summary.lower().startswith(SUMMARY_FILLER_PREFIXES):
            errors.append(
                "Concept summary must start with what the application does, "
                "not filler such as 'This example'"
            )
        if MARKDOWN_IN_SUMMARY_RE.search(summary):
            errors.append("Concept summary must be plain text without Markdown")
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", summary))
        if sentence_count not in (1, 2):
            errors.append("Concept summary must contain one or two sentences")
        binary = metadata.get("Binary Name", "").strip().lower()
        if binary and binary in summary.lower():
            errors.append("Concept summary must not include the executable name")

    preview = HTML_COMMENT_RE.sub("", sections.get("Preview", ""))
    preview = strip_fenced_code(preview)
    if HTML_TAG_RE.search(preview):
        errors.append("Preview must not contain raw HTML")
        return errors
    match = PREVIEW_IMAGE_RE.search(preview)
    if not match:
        errors.append("Preview must contain a Markdown image")
        return errors

    parts = readme_path.parts
    try:
        examples_idx = list(parts).index("examples")
        category = parts[examples_idx + 1]
        example = parts[examples_idx + 2]
    except (ValueError, IndexError):
        return errors

    repo_root = readme_path.parents[3]
    image_path = (readme_path.parent / match.group(1)).resolve()
    expected_dir = (
        repo_root / "portal" / "assets" / "examples" / category / example
    ).resolve()
    if (
        image_path.parent != expected_dir
        or image_path.stem != "image"
        or image_path.suffix.lower() not in PREVIEW_IMAGE_SUFFIXES
    ):
        errors.append(
            "Preview image must use "
            f"portal/assets/examples/{category}/{example}/image.*"
        )
    elif not image_path.is_file():
        errors.append(f"Preview image does not exist: {match.group(1)}")

    return errors


def validate_install_workflow(
    *,
    sections: dict[str, str],
    metadata: dict[str, str],
    readme_path: Path,
) -> list[str]:
    """Validate either the complete bundle or single-example workflow."""
    errors: list[str] = []
    install = sections.get("Install Apps", "")
    run = sections.get("Run", "")
    development = sections.get("Development From Source", "")

    parts = readme_path.parts
    try:
        examples_idx = list(parts).index("examples")
        category = parts[examples_idx + 1]
        example = parts[examples_idx + 2]
    except (ValueError, IndexError):
        return errors

    install_lines = {line.strip() for line in install.splitlines()}
    uses_bundle = INSTALL_COMMAND in install_lines
    uses_standalone = (
        STANDALONE_INSTALL_SCRIPT in install
        and f"bash -s -- {example}" in install
    )
    if not uses_bundle and not uses_standalone:
        errors.append(
            f"Install Apps must use '{INSTALL_COMMAND}' or fetch this example "
            f"with '{STANDALONE_INSTALL_SCRIPT}'"
        )
    if uses_bundle and "cd prebuilt-apps" not in install:
        errors.append("Bundle installation must enter the prebuilt-apps directory")
    expected_app_dir = f"APP_DIR=examples/{category}/{example}"
    if uses_bundle and expected_app_dir not in install:
        errors.append(
            f"Bundle installation must define '{expected_app_dir}'"
        )
    if uses_standalone and f"cd {example}" not in install:
        errors.append(f"Standalone installation must enter the {example} directory")

    for command in SOURCE_ONLY_COMMANDS:
        if command in install or command in run:
            errors.append(
                f"Source-only command '{command.strip()}' must not appear in "
                "Install Apps or Run"
            )

    languages = {
        item.strip()
        for item in metadata.get("Languages", "").split(",")
        if item.strip()
    }
    if "C++" in languages:
        binary = metadata.get("Binary Name", "").strip()
        expected_paths = (
            (
                f"${{APP_DIR}}/src/cpp/pre-built/{binary}",
                f"$APP_DIR/src/cpp/pre-built/{binary}",
            )
            if uses_bundle
            else (f"src/cpp/pre-built/{binary}",)
        )
        if binary and not any(path in run for path in expected_paths):
            errors.append(
                "Run must reference the packaged C++ binary through APP_DIR"
                if uses_bundle
                else f"Run must reference packaged C++ binary '{expected_paths[0]}'"
            )
    if "Python" in languages:
        expected_python_paths = (
            ("${APP_DIR}/src/python/", "$APP_DIR/src/python/")
            if uses_bundle
            else ("src/python/",)
        )
        if not any(path in run for path in expected_python_paths):
            errors.append(
                f"Run must reference a packaged Python entrypoint through "
                f"'{expected_python_paths[0]}'"
            )

    repeated_path = f"examples/{category}/{example}"
    run_without_assignment = "\n".join(
        line for line in run.splitlines() if line.strip() != expected_app_dir
    )
    if uses_bundle and repeated_path in run_without_assignment:
        errors.append("Run must reuse APP_DIR instead of repeating the example path")

    if CONTRIBUTING_URL not in development:
        errors.append("Development From Source must link to CONTRIBUTING.md")

    return errors


def validate_readme(readme_path: Path) -> list[str]:
    """Validate a single README.md. Returns list of error strings."""
    errors = []
    content = readme_path.read_text()

    # Check required sections
    sections, section_order = parse_sections(content)
    for section in REQUIRED_SECTIONS:
        if section not in sections:
            errors.append(f"Missing required section: ## {section}")

    present_required_sections = [
        section for section in section_order if section in REQUIRED_SECTIONS
    ]
    expected_order = [
        section for section in REQUIRED_SECTION_ORDER if section in sections
    ]
    if present_required_sections != expected_order:
        errors.append(
            "Required sections are out of order. Expected: "
            + ", ".join(REQUIRED_SECTION_ORDER)
        )

    if "Get The Apps Repo" in sections:
        errors.append("Obsolete section: ## Get The Apps Repo")

    # Parse and validate metadata
    metadata = parse_metadata_table(content)
    if metadata is None:
        errors.append("Metadata table not found or empty")
        return errors

    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            errors.append(f"Missing required metadata field: {field}")

    category = metadata.get("Category", "")
    if category and category not in VALID_CATEGORIES:
        errors.append(
            f"Invalid Category '{category}'. "
            f"Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
        )

    difficulty = metadata.get("Difficulty", "")
    if difficulty and difficulty not in VALID_DIFFICULTIES:
        errors.append(
            f"Invalid Difficulty '{difficulty}'. "
            f"Must be one of: {', '.join(sorted(VALID_DIFFICULTIES))}"
        )

    languages = [
        item.strip()
        for item in metadata.get("Languages", "").split(",")
        if item.strip()
    ]
    invalid_languages = [
        language for language in languages if language not in VALID_LANGUAGES
    ]
    if invalid_languages:
        errors.append(
            f"Invalid Languages '{metadata.get('Languages', '')}'. "
            f"Must contain only: {', '.join(sorted(VALID_LANGUAGES))}"
        )

    status = metadata.get("Status", "")
    if status and status not in VALID_STATUSES:
        errors.append(
            f"Invalid Status '{status}'. "
            f"Must be one of: {', '.join(sorted(VALID_STATUSES))}"
        )

    model = metadata.get("Model", "").strip()
    if model and not MODEL_REFERENCE_RE.fullmatch(model):
        errors.append(
            "Invalid Model metadata. Use either '<model_label>' or "
            "'<model_label> [https://host/path/model_mpk.tar.gz]'"
        )

    errors.extend(
        validate_install_workflow(
            sections=sections,
            metadata=metadata,
            readme_path=readme_path,
        )
    )
    errors.extend(
        validate_portal_content(
            sections=sections,
            metadata=metadata,
            readme_path=readme_path,
        )
    )

    # Verify the example directory matches its category
    parts = readme_path.parts
    # Expected path: .../examples/<category>/<name>/README.md
    try:
        examples_idx = list(parts).index("examples")
        dir_category = parts[examples_idx + 1]
        if category and dir_category != category:
            errors.append(
                f"Directory category '{dir_category}' does not match "
                f"metadata Category '{category}'"
            )
    except (ValueError, IndexError):
        pass

    return errors


def progress_bar(current: int, total: int, width: int = 20) -> str:
    pct = current * 100 // total
    filled = pct // 5
    bar = "#" * filled + " " * (width - filled)
    return f"[{bar}] {pct:3d}%  ({current}/{total})"


def main() -> int:
    # Find the examples directory relative to this script or cwd
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    examples_dir = repo_root / "examples"

    if not examples_dir.is_dir():
        print(f"ERROR: examples directory not found at {examples_dir}", file=sys.stderr)
        return 1

    # Find all example directories (examples/<category>/<name>/)
    example_dirs = sorted(
        d for d in examples_dir.glob("*/*") if d.is_dir() and not d.name.startswith(".")
    )

    if not example_dirs:
        print("ERROR: No example directories found", file=sys.stderr)
        return 1

    total = len(example_dirs)
    all_errors: list[tuple[str, str]] = []

    for i, example_dir in enumerate(example_dirs, 1):
        readme = example_dir / "README.md"
        rel_path = readme.relative_to(repo_root)
        print(f"\r  [validate-readmes] {progress_bar(i, total)}", end="", flush=True)

        if not readme.exists():
            all_errors.append((str(rel_path), "README.md is missing"))
            continue

        errors = validate_readme(readme)
        for error in errors:
            all_errors.append((str(rel_path), error))
    print()

    if all_errors:
        print()
        for path, error in all_errors:
            print(f"  FAIL  {path}: {error}")
        print()
        print(f"  RESULT: FAILED ({len(all_errors)} errors in {total} READMEs)")
        return 1

    print(f"  RESULT: PASSED ({total} READMEs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
