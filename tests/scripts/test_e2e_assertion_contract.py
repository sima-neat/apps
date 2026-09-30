"""Repository contract: e2e suites assert usable output, not merely present output.

`AGENTS.md` requires end-to-end tests to prove useful output, and
`tests/utils/output_assertions.py` (Python) with `support/testing/test_process.h`
(C++) are the helpers that do so: saved frames must decode, have plausible
dimensions, and, for a stream, advance. A suite that only counts files and checks
`st_size > 0` waves through a truncated frame and a pipeline frozen on its first
frame, which is what every enabled suite used to do. This guard keeps a new or
rewritten suite from going back to that.

It runs against the source checkout with the other repository contract tests.
"""

from __future__ import annotations

import re
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = APPS_ROOT / "examples"

# Size-only assertions in Python: the `st_size > 0` idiom the shared helpers replaced.
PYTHON_SIZE_ONLY = re.compile(r"st_size\s*>\s*0")
# Size-only assertions in C++: the helper the shared decode checks replaced.
CPP_SIZE_ONLY = re.compile(r"\ball_output_files_nonempty\s*\(")


def _e2e_suites(language: str) -> list[Path]:
    pattern = "*/*/tests/python/test_e2e.py" if language == "python" else "*/*/tests/cpp/test_e2e.cpp"
    return sorted(EXAMPLES.glob(pattern))


def test_the_example_tree_has_e2e_suites_to_check():
    """If this fails, the glob is wrong and the contract below is vacuous."""
    assert len(_e2e_suites("python")) >= 10
    assert len(_e2e_suites("cpp")) >= 10


def test_no_python_e2e_suite_asserts_on_file_size_alone():
    offenders = [
        path.relative_to(APPS_ROOT)
        for path in _e2e_suites("python")
        if PYTHON_SIZE_ONLY.search(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        "these e2e suites assert st_size > 0 instead of proving the output decodes; "
        "use assert_saved_frames_are_usable or assert_streamed_frames_are_usable from "
        f"tests/utils/output_assertions.py: {[str(p) for p in offenders]}"
    )


def test_no_cpp_e2e_suite_asserts_on_file_size_alone():
    offenders = [
        path.relative_to(APPS_ROOT)
        for path in _e2e_suites("cpp")
        if CPP_SIZE_ONLY.search(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        "these e2e suites call all_output_files_nonempty instead of proving the output "
        "decodes; use saved_frames_problem or streamed_frames_problem from "
        f"support/testing/test_process.h: {[str(p) for p in offenders]}"
    )
