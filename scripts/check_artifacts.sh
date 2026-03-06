#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MAX_FILE_SIZE_BYTES="${MAX_FILE_SIZE_BYTES:-5242880}"

file_size_bytes() {
  if stat --version >/dev/null 2>&1; then
    stat -c%s "$1"
  else
    stat -f%z "$1"
  fi
}

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "[artifacts] SKIP: not a git repository"
  exit 0
fi

mapfile -t files < <(git ls-files | sort)
total=${#files[@]}

if [[ $total -eq 0 ]]; then
  echo "[artifacts] no tracked files"
  exit 0
fi

fail=0
failed_files=()
i=0
for f in "${files[@]}"; do
  [[ -f "$f" ]] || continue
  i=$((i + 1))
  pct=$((i * 100 / total))
  bar=$((pct / 5))
  printf "\r  [artifacts] [%-20s] %3d%%  (%d/%d)" "$(printf '#%.0s' $(seq 1 $bar))" "$pct" "$i" "$total"

  issue=""
  if [[ "$f" =~ \.(so|a|o|out|dylib|dll|exe|bin)$ ]]; then
    issue="binary artifact"
    fail=1
  fi
  if [[ "$f" == models/*.tar.gz ]]; then
    issue="model file (should be in .gitignore)"
    fail=1
  fi

  size="$(file_size_bytes "$f")"
  if (( size > MAX_FILE_SIZE_BYTES )); then
    size_mb=$((size / 1024 / 1024))
    if [[ -n "$issue" ]]; then
      issue="${issue}, oversized (${size_mb}MB)"
    else
      issue="oversized (${size_mb}MB > $((MAX_FILE_SIZE_BYTES / 1024 / 1024))MB)"
    fi
    fail=1
  fi

  if [[ -n "$issue" ]]; then
    failed_files+=("$f -- $issue")
  fi
done
echo ""

if [[ $fail -ne 0 ]]; then
  echo ""
  for entry in "${failed_files[@]}"; do
    echo "  FAIL  ${entry}"
  done
  echo ""
  echo "  RESULT: FAILED (${#failed_files[@]} issues in ${i} tracked files)"
  exit 1
fi

echo "  RESULT: PASSED (${i} tracked files)"
