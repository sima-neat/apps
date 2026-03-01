#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEAT_CORE_JSON="${ROOT_DIR}/neat-core.json"
NEAT_CORE_MARKER="${ROOT_DIR}/.neat-core-installed"
NEAT_INSTALLER_URL="${NEAT_INSTALLER_URL:-https://tools.modalix.info/install-neat-from-a-branch.sh}"
NEAT_ARTIFACTS_BASE_URL="${NEAT_ARTIFACTS_BASE_URL:-https://neat-artifacts.modalix.info/neat}"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CLEAN=0
BUILD_CPP=ON
BUILD_PYTHON=OFF
INSTALL_CORE=0
ONLY_INSTALL=0
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
  --python                  Enable Python tooling (placeholder)
  --all                     Install NEAT core SDK (from neat-core.json) then build
  --only-install-neat-core  Install NEAT core SDK and exit (no build)
  --neat-core-version <b:v> Override neat-core.json with branch:version (example: main:latest)
  -h, --help                Show help

Environment:
  SIMA_CLI_BIN              Path to sima-cli binary (default: sima-cli)
  NEAT_INSTALLER_URL        Hosted branch installer URL
  CMAKE_TOOLCHAIN_FILE      Optional CMake toolchain file (auto-detected for cross)
  SYSROOT                   Target sysroot (used by the default cross toolchain file)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --debug) BUILD_TYPE=Debug; shift ;;
    --release) BUILD_TYPE=Release; shift ;;
    --clean) CLEAN=1; shift ;;
    --no-cpp) BUILD_CPP=OFF; shift ;;
    --python) BUILD_PYTHON=ON; shift ;;
    --all) INSTALL_CORE=1; shift ;;
    --only-install-neat-core) INSTALL_CORE=1; ONLY_INSTALL=1; shift ;;
    --neat-core-version) NEAT_CORE_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# NEAT core install
# ---------------------------------------------------------------------------

extract_json_field() {
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['neat_core'][sys.argv[2]])" "$1" "$2"
}

download_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${output}"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO "${output}" "${url}"
    return 0
  fi
  echo "ERROR: neither curl nor wget is installed." >&2
  return 1
}

download_text() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO- "${url}"
    return 0
  fi
  echo "ERROR: neither curl nor wget is installed." >&2
  return 1
}

resolve_neat_core_target() {
  local branch version
  branch="$(extract_json_field "${NEAT_CORE_JSON}" "branch")"
  version="$(extract_json_field "${NEAT_CORE_JSON}" "version")"

  if [[ -n "${NEAT_CORE_OVERRIDE}" ]]; then
    if [[ "${NEAT_CORE_OVERRIDE}" != *:* ]]; then
      echo "ERROR: --neat-core-version must use branch:version (example: main:latest)." >&2
      exit 1
    fi
    branch="${NEAT_CORE_OVERRIDE%%:*}"
    version="${NEAT_CORE_OVERRIDE#*:}"
    if [[ -z "${branch}" || -z "${version}" ]]; then
      echo "ERROR: --neat-core-version must provide both branch and version." >&2
      exit 1
    fi
  fi

  printf '%s\n%s\n' "${branch}" "${version}"
}

ensure_neat_core() {
  if [[ ! -f "${NEAT_CORE_JSON}" ]]; then
    echo "ERROR: Missing ${NEAT_CORE_JSON}" >&2
    exit 1
  fi

  local branch version
  mapfile -t neat_core_target < <(resolve_neat_core_target)
  branch="${neat_core_target[0]}"
  version="${neat_core_target[1]}"

  # Resolve "latest" to the actual commit tag so the marker is precise.
  if [[ "${version}" == "latest" ]]; then
    version="$(download_text "${NEAT_ARTIFACTS_BASE_URL}/${branch}/latest.tag" | tr -d '[:space:]')"
    if [[ -z "${version}" ]]; then
      echo "ERROR: latest.tag is empty for branch: ${branch}" >&2
      exit 1
    fi
  fi

  # Check marker file — skip install if already at this version.
  local expected_tag="${branch}/${version}"
  if [[ -f "${NEAT_CORE_MARKER}" ]]; then
    local current_tag
    current_tag="$(tr -d '[:space:]' < "${NEAT_CORE_MARKER}")"
    if [[ "${current_tag}" == "${expected_tag}" ]]; then
      echo "NEAT core already installed (${expected_tag}). Skipping install."
      return 0
    fi
  fi

  echo "Installing NEAT core (${expected_tag})..."

  # Remove stale artifacts from previous installs so the upstream installer
  # downloads fresh copies and is not tripped by corrupted cached files.
  rm -f "${ROOT_DIR}"/install_neat_framework.sh \
        "${ROOT_DIR}"/pyneat-*.whl \
        "${ROOT_DIR}"/sima-neat-*-Linux-core.deb \
        "${ROOT_DIR}"/neat-*.deb

  local installer_path
  installer_path="$(mktemp /tmp/install-neat-from-a-branch.XXXXXX.sh)"
  trap 'rm -f "${installer_path}"' RETURN
  download_file "${NEAT_INSTALLER_URL}" "${installer_path}"
  chmod +x "${installer_path}"
  "${installer_path}" --minimum "${branch}" "${version}"
  rm -f "${installer_path}"
  trap - RETURN

  # Write marker on success.
  printf '%s\n' "${expected_tag}" > "${NEAT_CORE_MARKER}"
  echo "NEAT core installed successfully (${expected_tag})."
}

if [[ "${INSTALL_CORE}" -eq 1 ]]; then
  ensure_neat_core
fi

if [[ "${ONLY_INSTALL}" -eq 1 ]]; then
  exit 0
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

echo ""
echo "  NEAT Apps Build"
echo "  ==============="
echo "  Build directory       : ${BUILD_DIR}"
echo "  Build type            : ${BUILD_TYPE}"
echo "  Build C++ examples    : ${BUILD_CPP}"
echo "  Install NEAT core     : $(if [[ "${INSTALL_CORE}" -eq 1 ]]; then echo "ON"; else echo "OFF (use --all to enable)"; fi)"
echo "  NEAT core override    : ${NEAT_CORE_OVERRIDE:-"(from neat-core.json)"}"
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

TOOLCHAIN_ARG=()
if [[ -n "${TOOLCHAIN_FILE}" ]]; then
  TOOLCHAIN_ARG=("-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}")
fi

echo "  Toolchain file         : ${TOOLCHAIN_FILE:-"(none)"}"

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DSIMANEAT_APPS_BUILD_CPP="${BUILD_CPP}" \
  -DSIMANEAT_APPS_BUILD_PYTHON="${BUILD_PYTHON}" \
  "${TOOLCHAIN_ARG[@]}"

if [[ "${BUILD_CPP}" == "ON" ]]; then
  cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 8)"
else
  echo "C++ build disabled; configure completed."
fi
