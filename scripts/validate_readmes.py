#!/usr/bin/env python3
"""Validate that every example README.md conforms to the required contract.

Scans examples/*/*/README.md files and verifies:
- Metadata table exists with all required fields
- Field values match allowed enums (categories, difficulties, languages, statuses)
- Installed Apps commands and required sections are present

Exit code 0 on success, 1 if any README is missing or malformed.
"""

import re
import sys
from pathlib import Path

VALID_CATEGORIES = {
    "benchmarking",
    "classification",
    "object-detection",
    "tracking",
    "segmentation",
    "pose-estimation",
    "depth-estimation",
    "face-detection",
    "genai",
    "throughput",
}

VALID_DIFFICULTIES = {"Beginner", "Intermediate", "Advanced"}

VALID_LANGUAGES = {"C++", "Python"}

VALID_STATUSES = {"experimental", "stable"}

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
    "Prerequisites",
    "Install Apps",
    "Prepare the Model",
    "Run",
    "Source Files",
    "Development From Source",
}

REQUIRED_SECTION_ORDER = (
    "Metadata",
    "Concept",
    "Prerequisites",
    "Install Apps",
    "Prepare the Model",
    "Run",
    "Source Files",
    "Development From Source",
)

INSTALL_COMMAND = "sima-cli neat install apps"
CONTRIBUTING_URL = "https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md"
SOURCE_ONLY_COMMANDS = (
    "git clone ",
    "build.sh",
    "cmake ",
    "ctest ",
    "./build/",
    "tests/test.sh",
)


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


def validate_installed_workflow(
    *,
    sections: dict[str, str],
    metadata: dict[str, str],
    readme_path: Path,
) -> list[str]:
    """Validate commands that must work from the installed bundle root."""
    errors: list[str] = []
    install = sections.get("Install Apps", "")
    run = sections.get("Run", "")
    development = sections.get("Development From Source", "")

    if INSTALL_COMMAND not in {line.strip() for line in install.splitlines()}:
        errors.append(f"Install Apps must use '{INSTALL_COMMAND}'")
    if "cd prebuilt-apps" not in install:
        errors.append("Install Apps must enter the prebuilt-apps directory")

    for command in SOURCE_ONLY_COMMANDS:
        if command in install or command in run:
            errors.append(
                f"Source-only command '{command.strip()}' must not appear in "
                "Install Apps or Run"
            )

    parts = readme_path.parts
    try:
        examples_idx = list(parts).index("examples")
        category = parts[examples_idx + 1]
        example = parts[examples_idx + 2]
    except (ValueError, IndexError):
        return errors

    languages = {
        item.strip()
        for item in metadata.get("Languages", "").split(",")
        if item.strip()
    }
    if "C++" in languages:
        binary = metadata.get("Binary Name", "").strip()
        expected = f"examples/{category}/{example}/src/cpp/pre-built/{binary}"
        if binary and expected not in run:
            errors.append(f"Run must reference packaged C++ binary '{expected}'")
    if "Python" in languages and "src/python/" not in run:
        errors.append("Run must reference a packaged src/python/ entrypoint")

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
        validate_installed_workflow(
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
