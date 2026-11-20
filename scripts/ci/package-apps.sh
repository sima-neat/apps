#!/usr/bin/env bash
# Packages a finished build into the two CI archives:
#   neat-apps-<branch>-<version>.tar.gz  the customer runtime (prebuilt-apps/)
#   neat-apps-tests.tar.gz               the test overlay CI unpacks over it
# Both land in the repository root. build.sh --all calls this after the
# build; the Vulcan post-build step validates the archives.
#
# Usage: scripts/ci/package-apps.sh [--build-dir <dir>]
#   <dir> is relative to the repository root (default: build, or $BUILD_DIR).
# Needs deps/neat-core.json, which build.sh --all writes when it resolves Core.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPS_DIR="${NEAT_APPS_TEST_DEPS_DIR:-${ROOT_DIR}/deps}"
APPS_MANIFEST="${DEPS_DIR}/manifest.json"
NEAT_CORE_METADATA="${DEPS_DIR}/neat-core.json"
BUILD_DIR="${BUILD_DIR:-build}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

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

git_short_sha() {
  if [[ -n "${NEAT_APPS_ARTIFACT_SHORT_SHA:-}" ]]; then
    printf '%.7s\n' "${NEAT_APPS_ARTIFACT_SHORT_SHA}"
    return 0
  fi
  if [[ -n "${GITHUB_SHA:-}" ]]; then
    printf '%.7s\n' "${GITHUB_SHA}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${ROOT_DIR}" rev-parse --short HEAD
    return 0
  fi
  echo "local"
}

git_branch_key() {
  local raw="local"
  if [[ -n "${NEAT_APPS_ARTIFACT_BRANCH_KEY:-}" ]]; then
    sanitize_branch_key "${NEAT_APPS_ARTIFACT_BRANCH_KEY}"
    return 0
  fi
  if [[ -n "${GITHUB_HEAD_REF:-}" ]]; then
    sanitize_branch_key "${GITHUB_HEAD_REF}"
    return 0
  fi
  if [[ -n "${GITHUB_REF_NAME:-}" ]]; then
    sanitize_branch_key "${GITHUB_REF_NAME}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    raw="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo local)"
  fi
  sanitize_branch_key "${raw}"
}

tag_version() {
  local tag
  tag="$(current_dependency_tag)"
  if [[ -z "${tag}" ]]; then
    return 1
  fi
  if [[ "${tag}" =~ ^v[0-9] ]]; then
    tag="${tag#v}"
  fi
  printf '%s\n' "${tag}"
}

artifact_version() {
  if tag_version; then
    return 0
  fi
  git_short_sha
}

package_distribution() {
  local build_path="${ROOT_DIR}/${BUILD_DIR}"
  local stage_root stage_dir test_stage_dir
  local branch_key version archive_name archive_path apps_ref apps_spec
  local test_archive_path="${ROOT_DIR}/neat-apps-tests.tar.gz"

  if [[ ! -d "${build_path}" ]]; then
    echo "Skipping distribution packaging: build directory not found: ${build_path}"
    return 0
  fi

  branch_key="$(git_branch_key)"
  version="$(artifact_version)"
  archive_name="neat-apps-${branch_key}-${version}.tar.gz"
  archive_path="${ROOT_DIR}/${archive_name}"
  stage_root="$(mktemp -d /tmp/neat-apps-package.XXXXXX)"
  stage_dir="${stage_root}/prebuilt-apps"
  test_stage_dir="${stage_root}/neat-apps-tests"

  mkdir -p \
    "${stage_dir}/examples" \
    "${stage_dir}/assets" \
    "${stage_dir}/deps" \
    "${test_stage_dir}/examples"

  if [[ ! -f "${NEAT_CORE_METADATA}" ]]; then
    echo "ERROR: missing resolved Core metadata: ${NEAT_CORE_METADATA}" >&2
    rm -rf "${stage_root}"
    return 1
  fi
  cp "${NEAT_CORE_METADATA}" "${stage_dir}/deps/neat-core.json"
  apps_ref="${NEAT_APPS_ARTIFACT_BRANCH_KEY:-${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}}"
  if [[ -z "${apps_ref}" ]] && command -v git >/dev/null 2>&1; then
    apps_ref="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  fi
  if [[ -z "${apps_ref}" ]]; then
    echo "ERROR: unable to determine the Apps source ref." >&2
    rm -rf "${stage_root}"
    return 1
  fi
  apps_spec="${GITHUB_SHA:-}"
  if [[ -z "${apps_spec}" ]] && command -v git >/dev/null 2>&1; then
    apps_spec="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [[ -z "${apps_spec}" ]]; then
    echo "ERROR: unable to determine the Apps source commit." >&2
    rm -rf "${stage_root}"
    return 1
  fi
  python3 - "${stage_dir}/deps/neat-apps.json" "${apps_ref}" "${apps_spec}" <<'PY'
import json
import sys

path, ref, spec = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"neat-apps": {"ref": ref, "spec": spec}}, handle, indent=2)
    handle.write("\n")
PY

  while IFS= read -r cpp_dir; do
    local rel target_dir
    rel="${cpp_dir#"${ROOT_DIR}"/examples/}"
    target_dir="${stage_dir}/examples/$(dirname "${rel}")"
    mkdir -p "${target_dir}"
    cp -a "${cpp_dir}" "${target_dir}/"
    rm -f "${target_dir}/cpp/CMakeLists.txt"
    rm -rf "${target_dir}/cpp/pre-built"
  done < <(
    find "${ROOT_DIR}/examples" -mindepth 4 -maxdepth 4 \
      -type d -path '*/src/cpp' | sort
  )

  # Keep production and test executables in separate artifacts while
  # preserving the example layout expected by the test runner.
  while IFS= read -r exe; do
    local rel target_dir
    rel="${exe#${build_path}/}"
    if [[ "${exe}" == *_unit_test || "${exe}" == *_e2e_test ]]; then
      target_dir="${test_stage_dir}/$(dirname "${rel}")/tests/cpp"
    else
      local example_rel
      example_rel="$(dirname "${rel}")"
      example_rel="${example_rel%_cpp}"
      [[ "$(basename "${exe}")" == "$(basename "${example_rel}")" ]] || continue
      [[ -d "${ROOT_DIR}/${example_rel}/src/cpp" ]] || continue
      target_dir="${stage_dir}/${example_rel}/src/cpp/pre-built"
    fi
    mkdir -p "${target_dir}"
    cp "${exe}" "${target_dir}/"
  done < <(
    find "${build_path}/examples" -type f -executable \
      ! -path '*/CMakeFiles/*' \
      ! -name 'cmTC_*' \
      2>/dev/null | sort
  )

  # Runtime source is customer-facing. Test scopes and test source are staged
  # separately and overlaid only in CI.
  while IFS= read -r src_file; do
    local rel target_dir
    rel="${src_file#${ROOT_DIR}/examples/}"
    target_dir="${stage_dir}/examples/$(dirname "${rel}")"
    mkdir -p "${target_dir}"
    cp "${src_file}" "${target_dir}/"
  done < <(
    find "${ROOT_DIR}/examples" -type f \
      ! -path '*/tests/*' \
      ! -path '*/__pycache__/*' \
      ! -name '*.pyc' \
      ! -name '*.pyo' \
      ! -name '*.onnx' \
      ! -name '*.db' \
      ! -name '*.lock' \
      ! -name '*.log' \
      \( -path '*/src/python/*' \
         -o -name 'README.md' \
         -o -path '*/src/common/*' \
         -o -path '*/run.sh' \
         -o -path '*/setup.sh' \
      \) 2>/dev/null | sort
  )

  # Python examples import the helpers they share from support/python. The C++
  # executables are built with support/cpp, shipped as reference source without
  # its build and test files.
  mkdir -p "${stage_dir}/support/cpp"
  cp -a "${ROOT_DIR}/support/README.md" "${stage_dir}/support/"
  cp -a "${ROOT_DIR}/support/python" "${stage_dir}/support/"
  cp -a "${ROOT_DIR}/support/cpp/sima_examples" "${stage_dir}/support/cpp/"
  rm -rf "${stage_dir}/support/cpp/sima_examples/testing"

  while IFS= read -r tests_dir; do
    local rel target_dir
    rel="${tests_dir#${ROOT_DIR}/examples/}"
    target_dir="${test_stage_dir}/examples/${rel}"
    mkdir -p "${target_dir}"
    cp -a "${tests_dir}/." "${target_dir}/"
  done < <(find "${ROOT_DIR}/examples" -type d -name tests -prune | sort)

  mkdir -p "${test_stage_dir}/tests"
  cp -a "${ROOT_DIR}/tests/." "${test_stage_dir}/tests/"
  mkdir -p "${test_stage_dir}/deps" "${test_stage_dir}/scripts"
  cp "${APPS_MANIFEST}" "${test_stage_dir}/deps/manifest.json"
  cp "${ROOT_DIR}/scripts/download_models.sh" "${test_stage_dir}/scripts/download_models.sh"
  cp "${ROOT_DIR}/tests/conftest.py" "${test_stage_dir}/examples/conftest.py"
  local asset_files=()
  if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    mapfile -t asset_files < <(git -C "${ROOT_DIR}" ls-files assets/datasets)
  fi
  if [[ "${#asset_files[@]}" -eq 0 ]]; then
    mapfile -t asset_files < <(
      find "${ROOT_DIR}/assets/datasets" -type f \
        ! -name '.DS_Store' \
        | sed "s#^${ROOT_DIR}/##" \
        | sort
    )
  fi
  for rel in "${asset_files[@]}"; do
    local src_file target_dir
    src_file="${ROOT_DIR}/${rel}"
    [[ -f "${src_file}" ]] || continue
    target_dir="${stage_dir}/$(dirname "${rel}")"
    mkdir -p "${target_dir}"
    cp "${src_file}" "${target_dir}/"
  done

  local test_asset_files=()
  if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    mapfile -t test_asset_files < <(git -C "${ROOT_DIR}" ls-files assets/datasets-test)
  fi
  if [[ "${#test_asset_files[@]}" -eq 0 ]]; then
    mapfile -t test_asset_files < <(
      find "${ROOT_DIR}/assets/datasets-test" -type f \
        ! -name '.DS_Store' \
        | sed "s#^${ROOT_DIR}/##" \
        | sort
    )
  fi
  for rel in "${test_asset_files[@]}"; do
    local src_file target_dir
    src_file="${ROOT_DIR}/${rel}"
    [[ -f "${src_file}" ]] || continue
    target_dir="${test_stage_dir}/$(dirname "${rel}")"
    mkdir -p "${target_dir}"
    cp "${src_file}" "${target_dir}/"
  done

  find "${test_stage_dir}" -type f -name '.env.local' -delete
  find "${stage_dir}" "${test_stage_dir}" -type d \
    \( -name '__pycache__' -o -name '.pytest_cache' \) -prune -exec rm -rf {} +
  find "${stage_dir}" "${test_stage_dir}" -type f \
    \( -name '*.pyc' -o -name '*.pyo' \) -delete

  tar -czf "${archive_path}" -C "${stage_root}" "$(basename "${stage_dir}")"
  tar -czf "${test_archive_path}" -C "${stage_root}" "$(basename "${test_stage_dir}")"
  rm -rf "${stage_root}"

  echo ""
  echo "Distribution packages created:"
  echo "  ${archive_path}"
  echo "  ${test_archive_path}"
}

package_distribution
