#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mapfile -t files < <(find examples -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cc' -o -name 'CMakeLists.txt' -o -name '*.md' \) | sort)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "[conflicts] no files found"
  exit 0
fi

total=${#files[@]}
fail=0
failed_files=()
i=0
for f in "${files[@]}"; do
  i=$((i + 1))
  pct=$((i * 100 / total))
  bar=$((pct / 5))
  printf "\r  [conflicts] [%-20s] %3d%%  (%d/%d)" "$(printf '#%.0s' $(seq 1 $bar))" "$pct" "$i" "$total"
  markers="$(grep -cE '^(<<<<<<<|=======|>>>>>>>)( .*)?$' "$f" 2>/dev/null || true)"
  if [[ "$markers" -gt 0 ]]; then
    failed_files+=("$f ($markers markers)")
    fail=1
  fi
done
echo ""

if [[ $fail -ne 0 ]]; then
  echo ""
  for entry in "${failed_files[@]}"; do
    echo "  FAIL  ${entry}"
  done
  echo ""
  echo "  RESULT: FAILED (conflict markers found)"
  exit 1
fi

echo "  RESULT: PASSED (${total} files)"
