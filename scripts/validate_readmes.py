#!/usr/bin/env python3
"""Validate that every example README.md conforms to the required template.

Scans examples/*/*/README.md files and verifies:
- Metadata table exists with all required fields
- Field values match allowed enums (categories, difficulties, statuses)
- Required sections are present

Exit code 0 on success, 1 if any README is missing or malformed.
"""

import re
import sys
from pathlib import Path

VALID_CATEGORIES = {
    "classification",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
    "depth-estimation",
    "face-detection",
    "throughput",
}

VALID_DIFFICULTIES = {"Beginner", "Intermediate", "Advanced"}

VALID_STATUSES = {"experimental", "stable"}

REQUIRED_METADATA_FIELDS = {"Category", "Difficulty", "Tags", "Status", "Binary Name", "Model"}
MODEL_REFERENCE_RE = re.compile(r"^(?P<label>[^\[]+?)(?:\s*\[(?P<url>https?://[^\]]+)\])?$")

REQUIRED_SECTIONS = {"Metadata", "Concept", "Prerequisites", "Run", "Source Files"}


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


def get_sections(content: str) -> set[str]:
    """Return set of H2 section names found in the content."""
    sections = set()
    for line in content.splitlines():
        match = re.match(r"^##\s+(.+)$", line)
        if match:
            sections.add(match.group(1).strip())
    return sections


def validate_readme(readme_path: Path) -> list[str]:
    """Validate a single README.md. Returns list of error strings."""
    errors = []
    content = readme_path.read_text()

    # Check required sections
    sections = get_sections(content)
    for section in REQUIRED_SECTIONS:
        if section not in sections:
            errors.append(f"Missing required section: ## {section}")

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
        d
        for d in examples_dir.glob("*/*")
        if d.is_dir() and not d.name.startswith(".")
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
