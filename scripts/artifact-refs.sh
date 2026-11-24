# shellcheck shell=bash
# Branch key and release tag for artifact names. build.sh resolves Core with
# them and scripts/ci/package-apps.sh versions the Apps archive with them, so
# both source this one file. The caller sets ROOT_DIR to the repository root.

sanitize_branch_key() {
  printf '%s' "$1" | tr '/ ' '--'
}

current_dependency_tag() {
  if [[ "${GITHUB_REF_TYPE:-}" == "tag" && -n "${GITHUB_REF_NAME:-}" ]]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${ROOT_DIR}" describe --tags --exact-match HEAD 2>/dev/null || true
    return 0
  fi
  printf '\n'
}
