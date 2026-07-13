"""Regression tests for source files included in the Apps runtime archive."""

from __future__ import annotations

import re
from pathlib import Path


APPS_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_archive_includes_readme_referenced_stress_helpers():
    build_script = (APPS_ROOT / "build.sh").read_text(encoding="utf-8")
    package_start = build_script.index("package_distribution()")
    package_end = build_script.index("build_portal()", package_start)
    package_body = build_script[package_start:package_end]

    assert "-path '*/stress/*'" in package_body

    for readme in sorted((APPS_ROOT / "examples").glob("*/*/README.md")):
        references = set(re.findall(r"`(stress/[^`\s]+)`", readme.read_text(encoding="utf-8")))
        for reference in references:
            assert (readme.parent / reference).is_file(), (
                f"{readme.relative_to(APPS_ROOT)} references missing runtime helper {reference}"
            )
