#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEAT_CORE_JSON="${ROOT_DIR}/neat-core.json"
NEAT_CORE_MARKER="${ROOT_DIR}/.neat-core-installed"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CLEAN=0
BUILD_CPP=ON
BUILD_PYTHON=OFF
INSTALL_CORE=0
ONLY_INSTALL=0
CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"

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
  -h, --help                Show help

Environment:
  SIMA_CLI_BIN              Path to sima-cli binary (default: sima-cli)
  CMAKE_PREFIX_PATH         Set if SimaNeatConfig.cmake is not in a default prefix
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

ensure_neat_core() {
  if [[ ! -f "${NEAT_CORE_JSON}" ]]; then
    echo "ERROR: Missing ${NEAT_CORE_JSON}" >&2
    exit 1
  fi

  local branch version install_method
  branch="$(extract_json_field "${NEAT_CORE_JSON}" "branch")"
  version="$(extract_json_field "${NEAT_CORE_JSON}" "version")"
  install_method="$(extract_json_field "${NEAT_CORE_JSON}" "install_method")"

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

  if [[ "${install_method}" == "script" ]]; then
    local install_script
    install_script="$(extract_json_field "${NEAT_CORE_JSON}" "install_script")"
    echo "Using install script method..."
    bash -c "${install_script}"
  else
    local metadata_url
    metadata_url="$(extract_json_field "${NEAT_CORE_JSON}" "metadata_url")"
    if [[ -z "${metadata_url}" ]]; then
      echo "ERROR: neat-core.json must define a non-empty metadata_url." >&2
      exit 1
    fi
    if ! command -v "${CLI_BIN}" >/dev/null 2>&1; then
      echo "ERROR: ${CLI_BIN} not found. Install sima-cli first:" >&2
      echo "  wget -O /tmp/install-neat.sh https://tools.modalix.info/install-neat.sh && bash /tmp/install-neat.sh" >&2
      exit 1
    fi
    echo "Via: ${CLI_BIN} install -m ${metadata_url}"
    "${CLI_BIN}" install -m "${metadata_url}"
  fi

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
echo ""

if [[ "${CLEAN}" -eq 1 ]]; then
  echo "  Cleaning ${BUILD_DIR}..."
  rm -rf "${BUILD_DIR}"
fi

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DSIMANEAT_APPS_BUILD_CPP="${BUILD_CPP}" \
  -DSIMANEAT_APPS_BUILD_PYTHON="${BUILD_PYTHON}"

if [[ "${BUILD_CPP}" == "ON" ]]; then
  cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 8)"
else
  echo "C++ build disabled; configure completed."
fi
