#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${NEAT_APPS_TEST_DEPS_DIR:-${ROOT_DIR}/deps}"
APPS_MANIFEST="${DEPS_DIR}/manifest.json"
NEAT_CORE_METADATA="${DEPS_DIR}/neat-core.json"
NEAT_DEBS_DIR="${DEPS_DIR}/debs"
NEAT_INSTALLER_URL="${NEAT_INSTALLER_URL:-https://tools.sima-neat.com/install-neat.sh}"
NEAT_ARTIFACTS_BASE_URL="${NEAT_ARTIFACTS_BASE_URL:-https://artifacts.neat.sima.ai/core}"
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"
NEAT_CORE_INSTALL_DIR=""
NEAT_CORE_INSTALL_DIR_OWNED=0
LATEST_TAG_CACHE_DIR="$(mktemp -d /tmp/neat-apps-latest.XXXXXX)"

cleanup_latest_tag_cache() {
  rm -rf "${LATEST_TAG_CACHE_DIR}"
}
trap cleanup_latest_tag_cache EXIT

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CLEAN=0
BUILD_CPP=ON
INSTALL_CORE=0
ONLY_INSTALL=0
PORTAL_ONLY=0
CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
NEAT_CORE_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Options:
  --build-dir <dir>         Build directory (default: build)
  --debug                   Use Debug build
  --release                 Use Release build (default)
  --clean                   Remove build directory before configure
  --no-cpp                  Skip C++ example build (layout/metadata only)
  --all                     Install NEAT core SDK, build apps, build portal, then package
  --portal-only             Build portal only and exit
  --only-install-neat-core  Install NEAT core SDK and exit (no build)
  --neat-core-version <branch-version>
                            Override deps/manifest.json neat-core for one run
  -h, --help                Show help

Environment:
  SIMA_CLI_BIN              Path to sima-cli binary (default: sima-cli)
  NEAT_APPS_DEPENDENCY_BRANCH
                            Dependency branch for snap manifest neat-core
  NEAT_INSTALLER_URL        Hosted branch installer URL
  NEAT_ARTIFACTS_BASE_URL   Hosted NEAT core artifact index
  NEAT_APPS_CORE_INSTALL_DIR
                            Scratch directory override for Vulcan core installs
                            (default: fresh /tmp/neat-apps-core-install.XXXXXX)
  NEAT_CORE_INSTALL_MODE    legacy or vulcan. Defaults to vulcan when NEAT_VULCAN_ENV is set.
  NEAT_VULCAN_ENV           Vulcan artifact environment for sima-cli core installs (default: production)
  NEAT_SYNC_SYSROOT         Set to ON to align the eLxr SDK sysroot with the selected Core artifact
  CMAKE_TOOLCHAIN_FILE      Optional CMake toolchain file (auto-detected for cross)
  SYSROOT                   Target sysroot (used by the default cross toolchain file)
EOF
}

sanitize_branch_key() {
  printf '%s' "$1" | tr '/ ' '--'
}

core_artifact_branch_key() {
  python3 - "$1" <<'PY'
import sys
import urllib.parse

ref_key = urllib.parse.quote(sys.argv[1], safe="")
print(urllib.parse.quote(ref_key, safe=""))
PY
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
  "${ROOT_DIR}/scripts/ci/validate_apps_runtime_archive.sh" "${archive_path}"
  tar -tzf "${test_archive_path}" neat-apps-tests/tests/test.sh >/dev/null
  tar -tzf "${test_archive_path}" \
    neat-apps-tests/assets/datasets-test/imagenet/goldfish.jpeg >/dev/null
  local test_coco_count
  test_coco_count="$(
    tar -tzf "${test_archive_path}" \
      | grep -Ec '^neat-apps-tests/assets/datasets-test/coco/[^/]+\.jpg$' \
      || true
  )"
  if [[ "${test_coco_count}" -ne 10 ]]; then
    echo "ERROR: expected 10 COCO test images, found ${test_coco_count}." >&2
    return 1
  fi
  rm -rf "${stage_root}"

  echo ""
  echo "Distribution packages created:"
  echo "  ${archive_path}"
  echo "  ${test_archive_path}"
}

build_portal() {
  local portal_dir="${ROOT_DIR}/portal"

  if [[ ! -f "${portal_dir}/package.json" ]]; then
    echo "Skipping portal build: package.json not found under ${portal_dir}"
    return 0
  fi

  if ! command -v npm >/dev/null 2>&1; then
    echo "ERROR: npm is required to build the portal during ./build.sh --all." >&2
    exit 1
  fi

  echo ""
  echo "Building portal..."
  (cd "${portal_dir}" && npm install && npm run build)
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --debug) BUILD_TYPE=Debug; shift ;;
    --release) BUILD_TYPE=Release; shift ;;
    --clean) CLEAN=1; shift ;;
    --no-cpp) BUILD_CPP=OFF; shift ;;
    --all) INSTALL_CORE=1; shift ;;
    --portal-only) PORTAL_ONLY=1; shift ;;
    --only-install-neat-core) INSTALL_CORE=1; ONLY_INSTALL=1; shift ;;
    --neat-core-version) NEAT_CORE_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# NEAT core install
# ---------------------------------------------------------------------------

read_app_manifest() {
  python3 - <<'PY' "$1"
import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
except FileNotFoundError:
    print(f"ERROR: Missing manifest: {path}", file=sys.stderr)
    raise SystemExit(1)
except json.JSONDecodeError as exc:
    print(f"ERROR: Invalid JSON in {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

neat_core = data.get("neat-core")
platform_version = data.get("platform-version")

if not isinstance(platform_version, str) or not platform_version.strip():
    print(f"ERROR: {path} must define platform-version as a non-empty string.", file=sys.stderr)
    raise SystemExit(1)

if isinstance(neat_core, str):
    print("__SNAP__" if not neat_core.strip() else neat_core.strip())
elif isinstance(neat_core, dict):
    policy = str(neat_core.get("policy", "")).strip().lower()
    if policy == "snap":
        print("__SNAP__")
    elif policy:
        print(f"ERROR: unsupported neat-core.policy in {path}: {policy!r}", file=sys.stderr)
        raise SystemExit(1)
    else:
        spec = str(neat_core.get("spec", "")).strip()
        branch = str(neat_core.get("branch", neat_core.get("ref", ""))).strip()
        if not branch:
            print(
                f"ERROR: {path} neat-core object must define policy=snap or branch/ref.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"{branch}:{spec or 'latest'}")
else:
    print(
        f"ERROR: {path} must define neat-core as a string or dependency object.",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(platform_version.strip())
PY
}

current_dependency_branch() {
  if [[ -n "${NEAT_APPS_DEPENDENCY_BRANCH:-}" ]]; then
    printf '%s\n' "${NEAT_APPS_DEPENDENCY_BRANCH}"
    return 0
  fi
  if [[ -n "${GITHUB_HEAD_REF:-}" ]]; then
    printf '%s\n' "${GITHUB_HEAD_REF}"
    return 0
  fi
  if [[ "${GITHUB_REF_TYPE:-}" != "tag" && -n "${GITHUB_REF_NAME:-}" ]]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null
    return 0
  fi
  echo "develop"
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

core_branch_exists() {
  local branch="$1"
  local core_repo="${ROOT_DIR}/../core"

  if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
    return 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    return 1
  fi
  if ! git -C "${core_repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 1
  fi

  if git -C "${core_repo}" show-ref --verify --quiet "refs/heads/${branch}"; then
    return 0
  fi
  if git -C "${core_repo}" show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
    return 0
  fi
  return 1
}

core_artifact_branch_exists() {
  local branch="$1"

  if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
    return 1
  fi

  resolve_latest_version "${branch}" >/dev/null 2>&1
}

core_artifact_version_exists() {
  local branch="$1"
  local version="$2"
  local branch_key

  if [[ -z "${branch}" || -z "${version}" || "${branch}" == "HEAD" ]]; then
    return 1
  fi

  branch_key="$(core_artifact_branch_key "${branch}")"
  download_text "${NEAT_ARTIFACTS_BASE_URL}/${branch_key}/${version}/metadata.json" >/dev/null 2>&1
}

normalize_vulcan_env() {
  local env_name="$1"
  case "${env_name}" in
    production|prod|prd) printf '%s\n' "prod" ;;
    staging|stg) printf '%s\n' "stg" ;;
    development|dev) printf '%s\n' "dev" ;;
    *) printf '%s\n' "${env_name}" ;;
  esac
}

neat_core_installed_matches() {
  local expected_branch="$1"
  local expected_version="$2"
  local expected_env="${3:-}"
  local expected_branch_key expected_env_key status_json

  if ! command -v neat >/dev/null 2>&1; then
    return 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi

  expected_branch_key="$(sanitize_branch_key "${expected_branch}")"
  expected_env_key="$(normalize_vulcan_env "${expected_env}")"
  status_json="$(neat --json 2>/dev/null)" || return 1

  if ! NEAT_STATUS_JSON="${status_json}" python3 - "${expected_branch}" "${expected_branch_key}" "${expected_version}" "${expected_env_key}" <<'PY'
import json
import os
import sys

expected_branch = sys.argv[1]
expected_branch_key = sys.argv[2]
expected_version = sys.argv[3]
expected_env = sys.argv[4]

try:
    data = json.loads(os.environ["NEAT_STATUS_JSON"])
except json.JSONDecodeError:
    raise SystemExit(1)

core = data.get("components", {}).get("core", {})
channel = str(core.get("channel", "")).strip()
tag = str(core.get("tag", "")).strip()
actual_env = str(core.get("provenance", {}).get("vulcanEnvironment", "")).strip()
runtime_version = str(data.get("components", {}).get("runtime", {}).get("version", "")).strip()
gst_plugins_version = str(
    data.get("components", {}).get("gstPlugins", {}).get("version", "")
).strip()

def normalize_env(value: str) -> str:
    if value in {"production", "prod", "prd"}:
        return "prod"
    if value in {"staging", "stg"}:
        return "stg"
    if value in {"development", "dev"}:
        return "dev"
    return value

if channel not in {expected_branch, expected_branch_key}:
    raise SystemExit(1)
if tag != expected_version:
    raise SystemExit(1)
if expected_env and normalize_env(actual_env) != expected_env:
    raise SystemExit(1)

# The Core provenance marker is not sufficient on a persistent runner: an
# interrupted installation can leave the requested Core package beside runtime
# and GStreamer packages from another branch. Those mixed packages made
# `neat --json` report the requested Core tag while BoxDecode still loaded an
# older plugin. Require the installed board-side components to come from the
# same branch before skipping installation.
expected_component_prefix = f"+{expected_branch_key}."
for version in (runtime_version, gst_plugins_version):
    if expected_component_prefix not in version:
        raise SystemExit(1)
PY
  then
    return 1
  fi

  return 0
}

download_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${output}" || return $?
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO "${output}" "${url}" || return $?
    return 0
  fi
  echo "ERROR: neither curl nor wget is installed." >&2
  return 1
}

download_text() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" || return $?
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO- "${url}" || return $?
    return 0
  fi
  echo "ERROR: neither curl nor wget is installed." >&2
  return 1
}

resolve_sima_cli_bin() {
  if [[ -n "${SIMA_CLI_BIN:-}" && -x "${SIMA_CLI_BIN}" ]]; then
    printf '%s\n' "${SIMA_CLI_BIN}"
    return 0
  fi
  if command -v sima-cli >/dev/null 2>&1; then
    command -v sima-cli
    return 0
  fi
  local candidate
  for candidate in \
    /data/sima-cli/.venv/bin/sima-cli \
    "${HOME}/.local/bin/sima-cli" \
    "${HOME}/sima-cli/.venv/bin/sima-cli" \
    /opt/sima-cli/.venv/bin/sima-cli \
    /opt/bin/sima-cli; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

resolve_latest_version() {
  local branch="$1"
  local branch_key cache_key cache_path missing_path latest_url
  local resolved=""

  branch_key="$(core_artifact_branch_key "${branch}")"
  cache_key="$(printf '%s' "${branch_key}" | tr -c 'A-Za-z0-9._-' '_')"
  cache_path="${LATEST_TAG_CACHE_DIR}/${cache_key}.tag"
  missing_path="${LATEST_TAG_CACHE_DIR}/${cache_key}.missing"
  latest_url="${NEAT_ARTIFACTS_BASE_URL}/${branch_key}/latest.tag"

  if [[ -f "${cache_path}" ]]; then
    resolved="$(tr -d '[:space:]' < "${cache_path}")"
    if [[ -n "${resolved}" ]]; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  fi
  if [[ -f "${missing_path}" ]]; then
    echo "ERROR: Could not resolve latest NEAT core artifact from ${latest_url}." >&2
    return 1
  fi

  if resolved="$(download_text "${latest_url}" 2>/dev/null)"; then
    resolved="$(printf '%s' "${resolved}" | tr -d '[:space:]')"
  else
    resolved=""
  fi

  if [[ -n "${resolved}" ]]; then
    printf '%s\n' "${resolved}" > "${cache_path}"
    printf '%s\n' "${resolved}"
    return 0
  fi

  : > "${missing_path}"
  echo "ERROR: Could not resolve latest NEAT core artifact from ${latest_url}." >&2
  return 1
}

parse_core_ref() {
  local ref="$1"
  local label="$2"
  local branch version

  if [[ "${ref}" != *-* ]]; then
    echo "ERROR: ${label} must use branch-version (example: main-latest)." >&2
    return 1
  fi

  branch="${ref%-*}"
  version="${ref##*-}"
  if [[ -z "${branch}" || -z "${version}" ]]; then
    echo "ERROR: ${label} must provide exactly one non-empty branch and version." >&2
    return 1
  fi

  printf '%s\n%s\n' "${branch}" "${version}"
}

first_line() {
  printf '%s\n' "$1" | sed -n '1p'
}

second_line() {
  printf '%s\n' "$1" | sed -n '2p'
}

validate_explicit_core_ref() {
  local branch="$1"
  local version="$2"
  local label="$3"
  local resolved_version

  if [[ "${version}" == "latest" ]]; then
    if ! resolved_version="$(resolve_latest_version "${branch}")"; then
      echo "ERROR: ${label} references '${branch}-latest', but that latest tag is unavailable." >&2
      return 1
    fi
    return 0
  fi

  if core_artifact_version_exists "${branch}" "${version}"; then
    return 0
  fi

  echo "ERROR: ${label} references unavailable NEAT core artifact '${branch}-${version}'." >&2
  return 1
}

resolve_empty_neat_core_branch() {
  local branch
  branch="$(current_dependency_branch)"

  if [[ "${branch}" == "main" || "${branch}" == "develop" ]]; then
    printf '%s\n' "${branch}"
    return 0
  fi

  if core_branch_exists "${branch}" || core_artifact_branch_exists "${branch}"; then
    printf '%s\n' "${branch}"
    return 0
  fi

  echo "No NEAT core artifact found for dependency branch '${branch}'; using develop-latest." >&2
  printf '%s\n' "develop"
}

resolve_neat_core_target() {
  local branch version manifest_core ref_output manifest_output
  if ! manifest_output="$(read_app_manifest "${APPS_MANIFEST}")"; then
    return 1
  fi
  manifest_core="$(first_line "${manifest_output}")"

  if [[ -n "${NEAT_CORE_OVERRIDE}" ]]; then
    if ! ref_output="$(parse_core_ref "${NEAT_CORE_OVERRIDE}" "--neat-core-version")"; then
      return 1
    fi
    branch="$(first_line "${ref_output}")"
    version="$(second_line "${ref_output}")"
    validate_explicit_core_ref "${branch}" "${version}" "--neat-core-version" || return 1
    printf '%s\n%s\n' "${branch}" "${version}"
    return 0
  fi

  if [[ "${manifest_core}" == "__SNAP__" ]]; then
    branch="$(resolve_empty_neat_core_branch)"
    version="latest"
    printf '%s\n%s\n' "${branch}" "${version}"
    return 0
  fi

  if [[ -n "${manifest_core}" ]]; then
    if [[ "${manifest_core}" == *":"* ]]; then
      branch="${manifest_core%%:*}"
      version="${manifest_core#*:}"
    else
      if ! ref_output="$(parse_core_ref "${manifest_core}" "${APPS_MANIFEST} neat-core")"; then
        return 1
      fi
      branch="$(first_line "${ref_output}")"
      version="$(second_line "${ref_output}")"
    fi

    validate_explicit_core_ref "${branch}" "${version}" "${APPS_MANIFEST} neat-core" || return 1
    printf '%s\n%s\n' "${branch}" "${version}"
    return 0
  fi
}

write_neat_core_metadata() {
  local ref="$1"
  local spec="$2"
  mkdir -p "$(dirname "${NEAT_CORE_METADATA}")"
  python3 - "${NEAT_CORE_METADATA}" "${ref}" "${spec}" <<'PY'
import json
import sys

path, ref, spec = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"neat-core": {"ref": ref, "spec": spec}}, handle, indent=2)
    handle.write("\n")
PY
}

load_neat_core_target() {
  local target_output
  if [[ -n "${NEAT_CORE_TARGET_BRANCH:-}" && -n "${NEAT_CORE_TARGET_VERSION:-}" ]]; then
    return 0
  fi
  if ! target_output="$(resolve_neat_core_target)"; then
    exit 1
  fi
  NEAT_CORE_TARGET_BRANCH="$(first_line "${target_output}")"
  NEAT_CORE_TARGET_VERSION="$(second_line "${target_output}")"
}

prepare_vulcan_core_install_dir() {
  NEAT_CORE_INSTALL_DIR_OWNED=0
  if [[ -z "${NEAT_APPS_CORE_INSTALL_DIR:-}" ]]; then
    NEAT_CORE_INSTALL_DIR="$(mktemp -d /tmp/neat-apps-core-install.XXXXXX)"
    NEAT_CORE_INSTALL_DIR_OWNED=1
    return 0
  fi

  NEAT_CORE_INSTALL_DIR="${NEAT_APPS_CORE_INSTALL_DIR}"
  if [[ -z "${NEAT_CORE_INSTALL_DIR}" || "${NEAT_CORE_INSTALL_DIR}" == "/" ]]; then
    echo "ERROR: unsafe NEAT_APPS_CORE_INSTALL_DIR: ${NEAT_CORE_INSTALL_DIR}" >&2
    exit 1
  fi
  rm -rf "${NEAT_CORE_INSTALL_DIR}"
  mkdir -p "${NEAT_CORE_INSTALL_DIR}"
}

cleanup_vulcan_core_install_dir() {
  if [[ "${NEAT_CORE_INSTALL_DIR_OWNED}" == "1" && -n "${NEAT_CORE_INSTALL_DIR}" ]]; then
    rm -rf "${NEAT_CORE_INSTALL_DIR}"
  fi
  NEAT_CORE_INSTALL_DIR=""
  NEAT_CORE_INSTALL_DIR_OWNED=0
}

run_sima_cli_core_install() {
  local sima_cli_bin="$1"
  shift
  local log_path
  log_path="$(mktemp /tmp/neat-apps-sima-cli-install.XXXXXX.log)"
  if ! "${sima_cli_bin}" neat install "$@" 2>&1 | tee "${log_path}"; then
    rm -f "${log_path}"
    return 1
  fi
  if grep -Fq "Installation script exited" "${log_path}"; then
    echo "ERROR: sima-cli reported a NEAT core installer failure." >&2
    rm -f "${log_path}"
    return 1
  fi
  rm -f "${log_path}"
}

is_elxr_sdk() {
  local sdk_version elxr_version
  [[ -f "${ELXR_SDK_RELEASE_FILE}" ]] || return 1
  sdk_version="$(sed -n 's/^SDK Version[[:space:]]*=[[:space:]]*//p' "${ELXR_SDK_RELEASE_FILE}" | head -n1)"
  elxr_version="$(sed -n 's/^eLXr Version[[:space:]]*=[[:space:]]*//p' "${ELXR_SDK_RELEASE_FILE}" | head -n1)"
  [[ -n "${sdk_version}" && -n "${elxr_version}" ]]
}

run_privileged() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "ERROR: root privileges are required to update the SDK sysroot." >&2
    return 1
  fi
}

sync_sysroot_from_core_artifact() (
  local sima_cli_bin="$1"
  local branch="$2"
  local version="$3"
  local vulcan_env="$4"
  [[ "${NEAT_SYNC_SYSROOT:-OFF}" == "ON" ]] || return 0

  if ! is_elxr_sdk; then
    echo "ERROR: NEAT_SYNC_SYSROOT requires an eLxr SDK." >&2
    return 1
  fi
  if ! command -v sysroot >/dev/null 2>&1; then
    echo "ERROR: NEAT_SYNC_SYSROOT requires the sysroot command." >&2
    return 1
  fi

  local resolve_output metadata_url
  if ! resolve_output="$("${sima_cli_bin}" neat install \
      --env "${vulcan_env}" -t minimal --json "core@${branch}:${version}")"; then
    echo "ERROR: Failed to resolve Core metadata for sysroot alignment." >&2
    return 1
  fi
  if ! metadata_url="$(RESOLVE_OUTPUT="${resolve_output}" python3 - "${branch}" "${version}" <<'PY'
import json
import os
import sys

text = os.environ["RESOLVE_OUTPUT"]
start = text.find("{")
if start < 0:
    raise SystemExit("missing JSON object in sima-cli output")
payload = json.loads(text[start:])
if payload.get("repository") != "core":
    raise SystemExit("resolved repository is not core")
if payload.get("ref") != sys.argv[1] or payload.get("resolved_spec") != sys.argv[2]:
    raise SystemExit("resolved Core target does not match the requested target")
metadata_url = payload.get("metadata_url")
if not isinstance(metadata_url, str) or not metadata_url:
    raise SystemExit("resolved Core metadata URL is missing")
print(metadata_url)
PY
  )"; then
    echo "ERROR: Cannot verify resolved Core metadata identity." >&2
    return 1
  fi

  local tmp_dir metadata_file manifest_file
  tmp_dir="$(mktemp -d /tmp/neat-apps-sysroot.XXXXXX)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  metadata_file="${tmp_dir}/metadata-minimal.json"
  manifest_file="${tmp_dir}/internals-manifest.json"
  if ! download_file "${metadata_url}" "${metadata_file}"; then
    echo "ERROR: Failed to download selected Core metadata." >&2
    return 1
  fi

  local resource_info manifest_url expected_checksum
  if ! resource_info="$(python3 - "${metadata_file}" "${metadata_url}" <<'PY'
import json
import re
import sys
from urllib.parse import urljoin

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)
resource = "internals-manifest.json"
if metadata.get("resources", []).count(resource) != 1:
    raise SystemExit("Core metadata must contain exactly one Internals manifest")
checksum = metadata.get("resources-checksum", {}).get(resource)
if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
    raise SystemExit("Core metadata has no valid Internals manifest checksum")
print(urljoin(sys.argv[2], resource) + "\t" + checksum)
PY
  )"; then
    echo "ERROR: Cannot verify the Core Internals manifest resource." >&2
    return 1
  fi
  manifest_url="${resource_info%%$'\t'*}"
  expected_checksum="${resource_info#*$'\t'}"
  if ! download_file "${manifest_url}" "${manifest_file}"; then
    echo "ERROR: Failed to download the Core Internals manifest." >&2
    return 1
  fi

  local receipt
  if ! receipt="$(python3 - "${manifest_file}" "${APPS_MANIFEST}" "${expected_checksum}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != sys.argv[3]:
    raise SystemExit("Internals manifest checksum mismatch")
artifact = json.loads(manifest_path.read_text(encoding="utf-8"))
consumer = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
receipt = artifact["sysroot-version"]
consumer_base = consumer["platform-version"]
if not isinstance(receipt, str) or (
    receipt
    and not re.fullmatch(r"[0-9]+(?:[.][0-9]+){2}(?:~pre[0-9]+)?", receipt)
):
    raise SystemExit("invalid sysroot-version")
if receipt and consumer_base != receipt.split("~pre", 1)[0]:
    raise SystemExit("platform-version does not match the Internals receipt")
print(receipt)
PY
  )"; then
    echo "ERROR: Cannot verify the Core Internals build receipt." >&2
    return 1
  fi
  if [[ -z "${receipt}" ]]; then
    echo "Apps is using the existing SDK sysroot."
    return 0
  fi

  if [[ "${receipt}" == *"~pre"* ]]; then
    echo "Updating SDK sysroot to Core's Internals receipt ${receipt}"
    if ! run_privileged sysroot update "${receipt}"; then
      echo "ERROR: Failed to update SDK sysroot to ${receipt}." >&2
      return 1
    fi
    if ! sysroot status; then
      echo "ERROR: Failed to read SDK sysroot status." >&2
      return 1
    fi
    return 0
  fi

  local status_output image_platform overlay_state overlay_revision
  if ! status_output="$(sysroot status)"; then
    echo "ERROR: Failed to read SDK sysroot status." >&2
    return 1
  fi
  printf '%s\n' "${status_output}"
  image_platform="$(sed -nE \
    's/^SDK image platform:[[:space:]]*([^[:space:]]+).*$/\1/p' \
    <<< "${status_output}" | head -n1)"
  overlay_state="$(sed -nE \
    's/^Sysroot overlay state:[[:space:]]*([^[:space:]]+).*$/\1/p' \
    <<< "${status_output}" | head -n1)"
  overlay_revision="$(sed -nE \
    's/^Sysroot overlay revision:[[:space:]]*([^[:space:]]+).*$/\1/p' \
    <<< "${status_output}" | head -n1)"

  if [[ "${image_platform}" != "${receipt}" ]]; then
    echo "ERROR: SDK image platform ${image_platform:-unknown} does not match required stable platform ${receipt}." >&2
    return 1
  fi
  if [[ "${overlay_state}" != "inactive" || "${overlay_revision}" != "none" ]]; then
    echo "ERROR: Stable platform ${receipt} requires the image-default sysroot, but overlay ${overlay_revision:-unknown} is ${overlay_state:-unknown}." >&2
    echo "ERROR: Recreate the SDK container to remove the prerelease sysroot overlay." >&2
    return 1
  fi
  echo "Using stable SDK sysroot ${image_platform} with no active overlay."
)

ensure_neat_core() {
  local branch install_mode sima_cli_bin="" version vulcan_env
  mkdir -p "${DEPS_DIR}" "${NEAT_DEBS_DIR}"
  load_neat_core_target
  branch="${NEAT_CORE_TARGET_BRANCH}"
  version="${NEAT_CORE_TARGET_VERSION}"
  install_mode="${NEAT_CORE_INSTALL_MODE:-}"
  if [[ -z "${install_mode}" ]]; then
    if [[ -n "${NEAT_VULCAN_ENV:-}" ]]; then
      install_mode="vulcan"
    else
      install_mode="legacy"
    fi
  fi
  if [[ "${install_mode}" == "vulcan" ]]; then
    vulcan_env="${NEAT_VULCAN_ENV:-production}"
  fi

  # Resolve "latest" to an exact tag before checking installed state.
  if [[ "${version}" == "latest" ]]; then
    version="$(resolve_latest_version "${branch}")" || exit 1
  fi
  NEAT_CORE_TARGET_VERSION="${version}"
  write_neat_core_metadata "${branch}" "${version}"

  if [[ "${NEAT_SYNC_SYSROOT:-OFF}" == "ON" ]]; then
    if [[ "${install_mode}" != "vulcan" ]]; then
      echo "ERROR: NEAT_SYNC_SYSROOT requires NEAT_CORE_INSTALL_MODE=vulcan." >&2
      exit 1
    fi
    if ! sima_cli_bin="$(resolve_sima_cli_bin)"; then
      echo "ERROR: NEAT_SYNC_SYSROOT requires sima-cli on PATH or SIMA_CLI_BIN." >&2
      exit 1
    fi
    sync_sysroot_from_core_artifact "${sima_cli_bin}" "${branch}" "${version}" "${vulcan_env}"
  fi

  local expected_tag="${branch}/${version}"
  if neat_core_installed_matches "${branch}" "${version}" "${vulcan_env:-}"; then
    echo "NEAT core already installed (${expected_tag}). Skipping install."
    return 0
  fi

  echo "Installing NEAT core (${expected_tag})..."

  if [[ "${install_mode}" == "vulcan" ]]; then
    local install_status
    if [[ -z "${sima_cli_bin}" ]] && ! sima_cli_bin="$(resolve_sima_cli_bin)"; then
      echo "ERROR: NEAT_CORE_INSTALL_MODE=vulcan requires sima-cli on PATH or SIMA_CLI_BIN." >&2
      exit 1
    fi
    echo "Installing NEAT core from Vulcan env ${vulcan_env} using sima-cli."
    prepare_vulcan_core_install_dir
    echo "  Scratch dir: ${NEAT_CORE_INSTALL_DIR}"
    install_status=0
    (
      cd "${NEAT_CORE_INSTALL_DIR}"
      run_sima_cli_core_install "${sima_cli_bin}" \
        --env "${vulcan_env}" \
        -d . \
        -t minimal \
        "core@${branch}:${version}"
    ) || install_status=$?
    cleanup_vulcan_core_install_dir
    if [[ "${install_status}" -ne 0 ]]; then
      exit "${install_status}"
    fi

    echo "NEAT core installed successfully (${expected_tag})."
    return 0
  fi

  if [[ "${install_mode}" != "legacy" ]]; then
    echo "ERROR: unsupported NEAT_CORE_INSTALL_MODE: ${install_mode}" >&2
    exit 1
  fi

  # Remove stale artifacts from previous installs so the upstream installer
  # downloads fresh copies and is not tripped by corrupted cached files.
  find "${NEAT_DEBS_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  rm -f "${ROOT_DIR}"/install_neat_framework.sh \
        "${ROOT_DIR}"/pyneat-*.whl \
        "${ROOT_DIR}"/sima-neat-*-Linux-core.deb \
        "${ROOT_DIR}"/neat-*.deb \
        "${ROOT_DIR}"/.neat-core-installed

  local installer_path
  installer_path="$(mktemp /tmp/install-neat-from-a-branch.XXXXXX.sh)"
  trap 'rm -f "${installer_path}"' RETURN
  download_file "${NEAT_INSTALLER_URL}" "${installer_path}"
  chmod +x "${installer_path}"
  (cd "${NEAT_DEBS_DIR}" && "${installer_path}" --minimum "${branch}" "${version}")
  rm -f "${installer_path}"
  trap - RETURN
  find "${NEAT_DEBS_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

  echo "NEAT core installed successfully (${expected_tag})."
}

if [[ "${INSTALL_CORE}" -eq 1 ]]; then
  ensure_neat_core
fi

if [[ "${ONLY_INSTALL}" -eq 1 ]]; then
  exit 0
fi

if [[ "${PORTAL_ONLY}" -eq 1 ]]; then
  echo ""
  echo "  NEAT Apps Portal Build"
  echo "  ======================"
  build_portal
  exit 0
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

NEAT_CORE_DISPLAY_BRANCH="(unresolved)"
NEAT_CORE_DISPLAY_VERSION="(unresolved)"
PLATFORM_VERSION="(unresolved)"
if ! manifest_output="$(read_app_manifest "${APPS_MANIFEST}")"; then
  exit 1
fi
PLATFORM_VERSION="$(second_line "${manifest_output}")"
if [[ "${INSTALL_CORE}" -eq 1 ]]; then
  load_neat_core_target
  NEAT_CORE_DISPLAY_BRANCH="${NEAT_CORE_TARGET_BRANCH}"
  NEAT_CORE_DISPLAY_VERSION="${NEAT_CORE_TARGET_VERSION}"
  if [[ "${NEAT_CORE_DISPLAY_VERSION}" == "latest" ]]; then
    NEAT_CORE_DISPLAY_VERSION="$(resolve_latest_version "${NEAT_CORE_DISPLAY_BRANCH}")" || exit 1
  fi
fi

echo ""
echo "  NEAT Apps Build"
echo "  ==============="
echo "  Build directory       : ${BUILD_DIR}"
echo "  Build type            : ${BUILD_TYPE}"
echo "  Build C++ examples    : ${BUILD_CPP}"
echo "  Build portal          : $(if [[ "${INSTALL_CORE}" -eq 1 ]]; then echo "ON"; else echo "OFF"; fi)"
echo "  Install NEAT core     : $(if [[ "${INSTALL_CORE}" -eq 1 ]]; then echo "ON"; else echo "OFF (use --all to enable)"; fi)"
echo "  Platform version      : ${PLATFORM_VERSION}"
if [[ "${INSTALL_CORE}" -eq 1 ]]; then
  echo "  NEAT core target      : ${NEAT_CORE_DISPLAY_BRANCH}:${NEAT_CORE_DISPLAY_VERSION}"
else
  echo "  NEAT core target      : installed SDK core (not resolved during local build)"
fi
echo "  NEAT core override    : ${NEAT_CORE_OVERRIDE:-"(from deps/manifest.json)"}"
echo ""

if [[ "${CLEAN}" -eq 1 ]]; then
  echo "  Cleaning ${BUILD_DIR}..."
  rm -rf "${BUILD_DIR}"
fi

# Auto-enable repo toolchain for cross builds unless caller already set one.
TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-}"
DEFAULT_CROSS_TOOLCHAIN="${ROOT_DIR}/cmake/toolchains/aarch64-modalix.cmake"
if [[ -z "${TOOLCHAIN_FILE}" ]]; then
  if [[ -n "${SYSROOT:-}" || -n "${CROSS_COMPILE:-}" || "${CC:-}" == *aarch64-linux-gnu* || "${CXX:-}" == *aarch64-linux-gnu* ]]; then
    if [[ -f "${DEFAULT_CROSS_TOOLCHAIN}" ]]; then
      TOOLCHAIN_FILE="${DEFAULT_CROSS_TOOLCHAIN}"
    fi
  fi
fi

echo "  Toolchain file         : ${TOOLCHAIN_FILE:-"(none)"}"

if [[ -n "${TOOLCHAIN_FILE}" ]]; then
  cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_TESTING=ON \
    -DSIMANEAT_APPS_BUILD_CPP="${BUILD_CPP}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
else
  cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_TESTING=ON \
    -DSIMANEAT_APPS_BUILD_CPP="${BUILD_CPP}"
fi

if [[ "${BUILD_CPP}" == "ON" ]]; then
  cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 8)"
else
  echo "C++ build disabled; configure completed."
fi

if [[ "${INSTALL_CORE}" -eq 1 ]]; then
  build_portal
  package_distribution
fi
