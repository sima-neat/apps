#!/usr/bin/env python3
"""Detect duplicate #include lines in C/C++ source files."""

import pathlib
import re
import sys

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')
CPP_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}


def check_file(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    seen: dict[str, int] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines, start=1):
        match = INCLUDE_RE.match(line)
        if not match:
            continue
        header = match.group(1)
        first = seen.get(header)
        if first is not None:
            errors.append(f"    {path}:{idx}: duplicate '{header}' (first at line {first})")
        else:
            seen[header] = idx
    return errors


def progress_bar(current: int, total: int, width: int = 20) -> str:
    pct = current * 100 // total
    filled = pct // 5
    bar = "#" * filled + " " * (width - filled)
    return f"[{bar}] {pct:3d}%  ({current}/{total})"


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    files = sorted(
        f for f in root.glob("examples/**/*") if f.is_file() and f.suffix in CPP_EXTS
    )

    if not files:
        print("[include-hygiene] no C/C++ files found")
        return 0

    total = len(files)
    all_errors: list[str] = []
    failed_files: list[str] = []

    for i, path in enumerate(files, 1):
        rel = path.relative_to(root)
        print(f"\r  [include-hygiene] {progress_bar(i, total)}", end="", flush=True)
        errors = check_file(rel)
        if errors:
            failed_files.append(str(rel))
            all_errors.extend(errors)
    print()

    if all_errors:
        print()
        for f in failed_files:
            print(f"  FAIL  {f}")
        print()
        for item in all_errors:
            print(item, file=sys.stderr)
        print()
        print(f"  RESULT: FAILED ({len(failed_files)}/{total} files have duplicates)")
        return 1

    print(f"  RESULT: PASSED ({total} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
