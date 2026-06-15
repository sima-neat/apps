#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${NEAT_APPS_TEST_DEPS_DIR:-${ROOT_DIR}/deps}"
APPS_MANIFEST="${DEPS_DIR}/manifest.json"
NEAT_DEBS_DIR="${DEPS_DIR}/debs"
NEAT_INSTALLER_URL="${NEAT_INSTALLER_URL:-https://tools.sima-neat.com/install-neat.sh}"
NEAT_ARTIFACTS_BASE_URL="${NEAT_ARTIFACTS_BASE_URL:-https://artifacts.neat.sima.ai/core}"
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

print(urllib.parse.quote(sys.argv[1], safe=""))
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
  local stage_dir="${ROOT_DIR}/neat-apps-runtime"
  local branch_key version archive_name archive_path
  local neat_core_branch neat_core_version

  if [[ ! -d "${build_path}" ]]; then
    echo "Skipping distribution packaging: build directory not found: ${build_path}"
    return 0
  fi

  branch_key="$(git_branch_key)"
  version="$(artifact_version)"
  archive_name="neat-apps-${branch_key}-${version}.tar.gz"
  archive_path="${ROOT_DIR}/${archive_name}"

  rm -rf "${stage_dir}"
  mkdir -p "${stage_dir}/examples" "${stage_dir}/assets"

  load_neat_core_target
  neat_core_branch="${NEAT_CORE_TARGET_BRANCH}"
  neat_core_version="${NEAT_CORE_TARGET_VERSION}"
  if [[ "${neat_core_version}" == "latest" ]]; then
    neat_core_version="$(resolve_latest_version "${neat_core_branch}")"
  fi
  python3 - <<'PY' "${stage_dir}/neat-core.json" "${neat_core_branch}" "${neat_core_version}"
import json
import sys

payload = {
    "neat-core": {
        "branch": sys.argv[2],
        "version": sys.argv[3],
    }
}

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY

  # Copy only built executables from the build tree, preserving the example
  # directory layout so the bundle remains easy to browse and run.
  while IFS= read -r exe; do
    local rel target_dir
    rel="${exe#${build_path}/}"
    target_dir="${stage_dir}/$(dirname "${rel}")"
    if [[ "${exe}" == *_unit_test || "${exe}" == *_e2e_test ]]; then
      target_dir="${target_dir}/tests/cpp"
    fi
    mkdir -p "${target_dir}"
    cp "${exe}" "${target_dir}/"
  done < <(
    find "${build_path}/examples" -type f -executable \
      ! -path '*/CMakeFiles/*' \
      ! -name 'cmTC_*' \
      2>/dev/null | sort
  )

  # Include the Python entrypoints and test-side support files required to run
  # unit/e2e validation in another environment without the full source tree.
  while IFS= read -r src_file; do
    local rel target_dir
    rel="${src_file#${ROOT_DIR}/examples/}"
    target_dir="${stage_dir}/examples/$(dirname "${rel}")"
    mkdir -p "${target_dir}"
    cp "${src_file}" "${target_dir}/"
  done < <(
    find "${ROOT_DIR}/examples" -type f \
      ! -path '*/__pycache__/*' \
      ! -name '*.pyc' \
      ! -name '*.pyo' \
      ! -name '*.onnx' \
      ! -name '*.db' \
      ! -name '*.lock' \
      ! -name '*.log' \
      \( -path '*/src/python/*' \
         -o -path '*/tests/test-scope.yaml' \
         -o -name 'README.md' \
         -o -path '*/tests/python/test_*.py' \
         -o -path '*/src/common/*' \
         -o -path '*/run.sh' \
         -o -path '*/setup.sh' \
      \) 2>/dev/null | sort
  )

  cp -a "${ROOT_DIR}/tests" "${stage_dir}/tests"
  rm -f "${stage_dir}/tests/.env.local"
  mkdir -p "${stage_dir}/deps"
  cp "${APPS_MANIFEST}" "${stage_dir}/deps/manifest.json"
  mkdir -p "${stage_dir}/scripts"
  cp "${ROOT_DIR}/scripts/download_models.sh" "${stage_dir}/scripts/download_models.sh"
  cp "${ROOT_DIR}/tests/conftest.py" "${stage_dir}/examples/conftest.py"
  while IFS= read -r rel; do
    local src_file target_dir
    src_file="${ROOT_DIR}/${rel}"
    [[ -f "${src_file}" ]] || continue
    target_dir="${stage_dir}/$(dirname "${rel}")"
    mkdir -p "${target_dir}"
    cp "${src_file}" "${target_dir}/"
  done < <(git -C "${ROOT_DIR}" ls-files assets)

  find "${stage_dir}" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "${stage_dir}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

  tar -czf "${archive_path}" -C "${ROOT_DIR}" "$(basename "${stage_dir}")"

  echo ""
  echo "Distribution package created:"
  echo "  ${archive_path}"
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
  if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    printf '%s\n' "${GITHUB_BASE_REF}"
    return 0
  fi
  if [[ -n "${GITHUB_REF_NAME:-}" ]]; then
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

protected_manifest_branch_context() {
  if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    printf '%s\n' "${GITHUB_BASE_REF}"
    return 0
  fi
  if [[ -n "${GITHUB_REF_NAME:-}" ]]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null
    return 0
  fi
  echo ""
}

is_protected_manifest_context() {
  local branch
  branch="$(protected_manifest_branch_context)"
  [[ "${branch}" == "main" || "${branch}" == "develop" ]]
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
  local branch tag
  if [[ -z "${NEAT_APPS_DEPENDENCY_BRANCH:-}" ]]; then
    tag="$(current_dependency_tag)"
    if [[ -n "${tag}" ]]; then
      printf '%s\n' "${tag}"
      return 0
    fi
  fi

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
    if is_protected_manifest_context; then
      echo "ERROR: ${APPS_MANIFEST} must keep neat-core as policy=snap on main/develop." >&2
      return 1
    fi

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

ensure_neat_core() {
  local branch exact_tag install_mode sima_cli_bin version vulcan_env
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
    if ! version="$(resolve_latest_version "${branch}")"; then
      exact_tag="$(current_dependency_tag)"
      if [[ -n "${exact_tag}" && "${branch}" == "${exact_tag}" ]]; then
        echo "ERROR: Failed to resolve exact tag-snap NEAT core artifact for tag '${branch}'." >&2
      fi
      exit 1
    fi
  fi

  local expected_tag="${branch}/${version}"
  if neat_core_installed_matches "${branch}" "${version}" "${vulcan_env:-}"; then
    echo "NEAT core already installed (${expected_tag}). Skipping install."
    return 0
  fi

  echo "Installing NEAT core (${expected_tag})..."

  if [[ "${install_mode}" == "vulcan" ]]; then
    local install_status
    if ! sima_cli_bin="$(resolve_sima_cli_bin)"; then
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
    if ! NEAT_CORE_DISPLAY_VERSION="$(resolve_latest_version "${NEAT_CORE_DISPLAY_BRANCH}")"; then
      exact_tag="$(current_dependency_tag)"
      if [[ -n "${exact_tag}" && "${NEAT_CORE_DISPLAY_BRANCH}" == "${exact_tag}" ]]; then
        echo "ERROR: Failed to resolve exact tag-snap NEAT core artifact for tag '${NEAT_CORE_DISPLAY_BRANCH}'." >&2
      fi
      exit 1
    fi
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
