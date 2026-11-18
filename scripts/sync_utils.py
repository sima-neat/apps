#!/usr/bin/env python3
"""Keep the examples' vendored helper files identical to their templates.

Common helpers have one template under support/templates/<lang>/. An example that uses one
holds a copy in its own src/<lang>/utils/ folder, so the example works without the rest of
the repository. A copy is any file under examples/ that starts with these two lines (`//`
instead of `#` in C++):

    # Copied from support/templates/python/config.py;
    # edit the template and run scripts/sync_utils.py.

This script rewrites every copy as those two lines followed by its template. To change a
helper, edit the template and run it. To vendor a helper into an example, create the file
with the two lines and run it.

With --check it writes nothing, names each copy that differs from its template, and exits 1.
It also exits 1 when a copy names a template that does not exist, and when a file in a utils/
folder has a template's name but not the header, so a copy that lost its header is reported.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = Path("support/templates")
COMMENT_BY_SUFFIX = {".py": "#", ".h": "//", ".hpp": "//", ".cpp": "//", ".cc": "//"}
FIX_COMMAND = "python3 scripts/sync_utils.py"


@dataclass(frozen=True)
class Copy:
    """A vendored file and the template it names, both relative to the repository root."""

    path: Path
    template: Path


def header(comment: str, template: Path) -> str:
    return (
        f"{comment} Copied from {template.as_posix()};\n"
        f"{comment} edit the template and run scripts/sync_utils.py.\n"
    )


def find_copies(root: Path) -> list[Copy]:
    """Every file under examples/ whose first line names a template."""
    copies = []
    for path in sorted((root / "examples").rglob("*")):
        comment = COMMENT_BY_SUFFIX.get(path.suffix)
        if comment is None or not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline().rstrip("\n")
        match = re.fullmatch(rf"{re.escape(comment)} Copied from (\S+);", first_line)
        if match:
            copies.append(Copy(path.relative_to(root), Path(match.group(1))))
    return copies


def find_headerless_copies(root: Path, copies: list[Copy]) -> list[str]:
    """Problems for utils/ files named like a template but without the header.

    A copy is found only by its first line, so one that lost the header, or got another
    comment above it, would otherwise drop out of the check without a word.
    """
    known = {copy.path for copy in copies}
    problems = []
    for path in sorted((root / "examples").rglob("*")):
        lang = path.parent.parent.name
        if path.parent.name != "utils" or lang not in ("python", "cpp") or not path.is_file():
            continue
        template = TEMPLATES_DIR / lang / path.name
        if path.relative_to(root) not in known and (root / template).is_file():
            problems.append(
                f"{path.relative_to(root)}: has the name of {template} but does not start with "
                "its 'Copied from' header; restore the header, or rename the file if it is "
                "not a copy"
            )
    return problems


def expected_text(root: Path, copy: Copy) -> str:
    """The two header lines and the template. Raises ValueError when the template is unusable."""
    template_path = (root / copy.template).resolve()
    if not template_path.is_relative_to((root / TEMPLATES_DIR).resolve()):
        raise ValueError(f"names {copy.template}, which is not under {TEMPLATES_DIR}/")
    if template_path.suffix != copy.path.suffix:
        raise ValueError(f"names {copy.template}, which is a different file type")
    if not template_path.is_file():
        raise ValueError(f"names {copy.template}, which does not exist")
    comment = COMMENT_BY_SUFFIX[copy.path.suffix]
    try:
        body = template_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"names {copy.template}, which is not UTF-8 text") from None
    return header(comment, copy.template) + body


def check(root: Path) -> list[str]:
    """One problem line per copy that differs from its template or names an unusable one."""
    copies = find_copies(root)
    problems = find_headerless_copies(root, copies)
    for copy in copies:
        try:
            expected = expected_text(root, copy)
        except ValueError as error:
            problems.append(f"{copy.path}: {error}")
            continue
        if (root / copy.path).read_bytes() != expected.encode("utf-8"):
            problems.append(f"{copy.path}: differs from {copy.template}")
    return problems


def sync(root: Path) -> tuple[list[Path], list[str]]:
    """Rewrite each copy that differs. Returns the rewritten copies and the unusable ones."""
    updated = []
    copies = find_copies(root)
    problems = find_headerless_copies(root, copies)
    for copy in copies:
        try:
            expected = expected_text(root, copy)
        except ValueError as error:
            problems.append(f"{copy.path}: {error}")
            continue
        target = root / copy.path
        if target.read_bytes() != expected.encode("utf-8"):
            target.write_bytes(expected.encode("utf-8"))
            updated.append(copy.path)
    return updated, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report copies that differ from their templates without writing; exit 1 if any",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()

    if args.check:
        problems = check(root)
        for problem in problems:
            print(f"[sync_utils] {problem}", file=sys.stderr)
        if problems:
            print(f"[sync_utils] Fix: {FIX_COMMAND}", file=sys.stderr)
            return 1
        print(f"[sync_utils] {len(find_copies(root))} vendored copies match their templates")
        return 0

    updated, problems = sync(root)
    for path in updated:
        print(f"[sync_utils] updated {path}")
    for problem in problems:
        print(f"[sync_utils] {problem}", file=sys.stderr)
    print(f"[sync_utils] {len(find_copies(root))} vendored copies, {len(updated)} updated")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
