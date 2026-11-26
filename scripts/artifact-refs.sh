# shellcheck shell=bash
# Branch key and release tag for artifact names, shared by build.sh and
# scripts/ci/package-apps.sh. The caller sets ROOT_DIR to the repository root.

sanitize_branch_key() {
  printf '%s' "$1" | tr '/ ' '--'
}

current_dependency_tag() {
  if [[ "${GITHUB_REF_TYPE:-}" == "tag" && -n "${GITHUB_REF_NAME:-}" ]]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
    return 0
  fi
  local root="${ROOT_DIR:?artifact-refs.sh: set ROOT_DIR}"
  if command -v git >/dev/null 2>&1 && git -C "${root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${root}" describe --tags --exact-match HEAD 2>/dev/null || true
    return 0
  fi
  printf '\n'
}
