"""Examples reach support/ through relative symlinks, never through copies or path code.

A Python example imports the shared helpers through src/python/sima_examples, a relative
symlink to support/python/sima_examples (agents/rules.md). build.sh and get-example.sh copy
the files a link points to, so a link that is absolute, dangling, or points outside support/
breaks the installed bundle or a fetched example.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parents[2]

# Examples that still find support/python in a folder above main.py. Remove an example from
# this list in the pull request that gives it the sima_examples link.
WALK_UP_EXAMPLES = {
    "high-density-multi-stream-object-detector",
    "rfdetr-detection-segmentation",
}

IMPORTS_SIMA_EXAMPLES = re.compile(r"^\s*(?:from|import)\s+sima_examples\b", re.MULTILINE)
PATH_CODE = re.compile(r"\bsys\.path\b")


def symlink_problems(root: Path) -> list[str]:
    """Each symlink under examples/ that is absolute, dangling, or points outside support/."""
    support = (root / "support").resolve()
    problems = []
    for dirpath, dirnames, filenames in os.walk(root / "examples"):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if not path.is_symlink():
                continue
            relative = path.relative_to(root)
            target = os.readlink(path)
            if os.path.isabs(target):
                problems.append(f"{relative} is an absolute link to {target}; make it relative")
            elif not path.exists():
                problems.append(f"{relative} points to {target}, which does not exist")
            elif not path.resolve().is_relative_to(support):
                problems.append(f"{relative} points to {target}, outside support/")
    return problems


def example_problems(root: Path) -> tuple[list[str], int]:
    """Each Python example that copies the helpers, imports them without the link, or keeps
    path code next to it. Also returns how many examples have a correct link."""
    expected = root / "support" / "python" / "sima_examples"
    problems = []
    linked = 0
    for python_dir in sorted(root.glob("examples/*/*/src/python")):
        example = python_dir.parents[1].name
        entry = python_dir / "sima_examples"
        relative = entry.relative_to(root)
        command = f"ln -s {os.path.relpath(expected, python_dir)} {relative}"
        main_py = python_dir / "main.py"
        main_text = main_py.read_text(encoding="utf-8", errors="replace") if main_py.is_file() else ""
        if entry.is_symlink():
            if entry.resolve() != expected.resolve():
                problems.append(f"{relative} must point to support/python/sima_examples: {command}")
                continue
            linked += 1
            if example in WALK_UP_EXAMPLES:
                problems.append(f"{example} has the link; remove it from WALK_UP_EXAMPLES")
            if PATH_CODE.search(main_text):
                problems.append(f"{python_dir.relative_to(root)}/main.py has sys.path code; "
                                "the link makes it unneeded")
        elif entry.is_file():
            problems.append(f"{relative}: git wrote the link as a text file. Clone with "
                            "`git clone -c core.symlinks=true` (Windows: Developer Mode), or set "
                            f"core.symlinks=true and run `git checkout -- {relative}`")
        elif entry.exists():
            problems.append(f"{relative} is a real folder, a copy of the shared helpers; "
                            f"replace it with: {command}")
        elif IMPORTS_SIMA_EXAMPLES.search(main_text) and example not in WALK_UP_EXAMPLES:
            problems.append(f"{python_dir.relative_to(root)}/main.py imports sima_examples "
                            f"without the link; add it with: {command}")
    return problems, linked


def test_examples_link_the_shared_helpers():
    problems, linked = example_problems(APPS_ROOT)
    problems += symlink_problems(APPS_ROOT)
    assert not problems, "\n".join(problems)
    # Guards against a check that passes because it found nothing to check.
    assert linked > 0, "no example has the sima_examples link; this test checks nothing"


# The checks above, on small trees that break one rule each.


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "apps"
    (root / "support" / "python" / "sima_examples").mkdir(parents=True)
    (root / "support" / "python" / "sima_examples" / "config.py").write_text("import yaml\n")
    python_dir = root / "examples" / "category" / "example" / "src" / "python"
    python_dir.mkdir(parents=True)
    (python_dir / "main.py").write_text("from sima_examples.config import Config\n")
    (python_dir / "sima_examples").symlink_to("../../../../../support/python/sima_examples")
    return root, python_dir


def _all_problems(root: Path) -> list[str]:
    problems, _ = example_problems(root)
    return problems + symlink_problems(root)


def test_a_correct_tree_passes(tmp_path):
    root, _ = _tree(tmp_path)
    assert _all_problems(root) == []
    assert example_problems(root)[1] == 1


def test_a_copied_folder_fails(tmp_path):
    root, python_dir = _tree(tmp_path)
    (python_dir / "sima_examples").unlink()
    (python_dir / "sima_examples").mkdir()
    assert any("is a real folder" in p for p in _all_problems(root))


def test_an_absolute_link_fails(tmp_path):
    root, python_dir = _tree(tmp_path)
    (python_dir / "sima_examples").unlink()
    (python_dir / "sima_examples").symlink_to(root / "support" / "python" / "sima_examples")
    assert any("absolute link" in p for p in _all_problems(root))


def test_a_dangling_link_fails(tmp_path):
    root, python_dir = _tree(tmp_path)
    (python_dir / "sima_examples").unlink()
    (python_dir / "sima_examples").symlink_to("../../../../../support/python/missing")
    assert any("does not exist" in p for p in _all_problems(root))


def test_a_link_outside_support_fails(tmp_path):
    root, python_dir = _tree(tmp_path)
    (root / "elsewhere").mkdir()
    (python_dir / "data").symlink_to("../../../../../elsewhere")
    assert any("outside support/" in p for p in _all_problems(root))


def test_an_import_without_the_link_fails(tmp_path):
    root, python_dir = _tree(tmp_path)
    (python_dir / "sima_examples").unlink()
    assert any("without the link" in p for p in _all_problems(root))


def test_path_code_next_to_the_link_fails(tmp_path):
    root, python_dir = _tree(tmp_path)
    (python_dir / "main.py").write_text("import sys\nsys.path.insert(0, '/x')\n")
    assert any("sys.path code" in p for p in _all_problems(root))


def test_a_link_written_as_a_text_file_fails(tmp_path):
    root, python_dir = _tree(tmp_path)
    (python_dir / "sima_examples").unlink()
    (python_dir / "sima_examples").write_text("../../../../../support/python/sima_examples")
    assert any("git wrote the link as a text file" in p for p in _all_problems(root))
