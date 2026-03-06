#!/usr/bin/env python3
"""Check CMakeLists.txt files for style issues: tabs, trailing whitespace, indentation."""

import pathlib
import sys


def check_file(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    for idx, line in enumerate(lines, start=1):
        logical = line.rstrip("\n\r")
        if "\t" in logical:
            errors.append(f"    {path}:{idx}: tab character")
        if logical.rstrip(" ") != logical:
            errors.append(f"    {path}:{idx}: trailing whitespace")
        stripped = logical.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(logical) - len(stripped)
        if indent % 2 != 0:
            errors.append(f"    {path}:{idx}: odd indentation ({indent} spaces)")
    return errors


def progress_bar(current: int, total: int, width: int = 20) -> str:
    pct = current * 100 // total
    filled = pct // 5
    bar = "#" * filled + " " * (width - filled)
    return f"[{bar}] {pct:3d}%  ({current}/{total})"


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    cmake_files = sorted(root.glob("examples/**/CMakeLists.txt"))

    if not cmake_files:
        print("[cmake-style] no CMake files found")
        return 0

    total = len(cmake_files)
    all_errors: list[str] = []
    failed_files: list[str] = []

    for i, path in enumerate(cmake_files, 1):
        rel = path.relative_to(root)
        print(f"\r  [cmake-style] {progress_bar(i, total)}", end="", flush=True)
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
        print(f"  RESULT: FAILED ({len(failed_files)}/{total} files have issues)")
        return 1

    print(f"  RESULT: PASSED ({total} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
