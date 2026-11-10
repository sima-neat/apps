#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo ""
echo "  NEAT Apps CI Checks"
echo "  ==================="
echo ""

checks=(
  "scripts/check_format.sh"
  "python3 scripts/check_cmake_style.py"
  "python3 scripts/check_duplicate_includes.py"
  "scripts/check_conflicts.sh"
  "scripts/check_artifacts.sh"
  "python3 scripts/validate_readmes.py"
)

total=${#checks[@]}
passed=0
failed=0
failed_names=()

for cmd in "${checks[@]}"; do
  if $cmd; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
    failed_names+=("$cmd")
  fi
  echo ""
done

echo "  ==================="
if [[ $failed -gt 0 ]]; then
  echo "  RESULT: FAILED (${passed}/${total} passed)"
  echo ""
  for name in "${failed_names[@]}"; do
    echo "  FAILED: ${name}"
  done
  exit 1
fi

echo "  RESULT: ALL PASSED (${total}/${total})"
