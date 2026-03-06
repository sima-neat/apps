#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "ERROR: clang-format is required." >&2
  exit 1
fi

mapfile -t files < <(find examples -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cc' \) | sort)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "[clang-format] no C/C++ files found"
  exit 0
fi

total=${#files[@]}
failed_files=()
i=0
for f in "${files[@]}"; do
  i=$((i + 1))
  pct=$((i * 100 / total))
  bar=$((pct / 5))
  printf "\r  [clang-format] [%-20s] %3d%%  (%d/%d)" "$(printf '#%.0s' $(seq 1 $bar))" "$pct" "$i" "$total"
  if ! clang-format --dry-run --Werror "$f" >/dev/null 2>&1; then
    failed_files+=("$f")
  fi
done
echo ""

if [[ ${#failed_files[@]} -gt 0 ]]; then
  echo ""
  for f in "${failed_files[@]}"; do
    echo "  FAIL  $f"
  done
  echo ""
  echo "  RESULT: FAILED (${#failed_files[@]}/${total} files need formatting)"
  echo "  Fix:    clang-format -i <file>"
  exit 1
fi

echo "  RESULT: PASSED (${total} files)"
