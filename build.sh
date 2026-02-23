#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CLEAN=0
BUILD_CPP=ON
BUILD_PYTHON=OFF

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Options:
  --build-dir <dir>   Build directory (default: build)
  --debug             Use Debug build
  --release           Use Release build (default)
  --clean             Remove build directory before configure
  --no-cpp            Skip C++ example build (layout/metadata only)
  --python            Enable future Python tooling toggle (placeholder)
  -h, --help          Show help

Notes:
  - Install SiMa NEAT core/SDK first with sima-cli.
  - Use CMAKE_PREFIX_PATH if SimaNeatConfig.cmake is not in a default prefix.
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
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${CLEAN}" -eq 1 ]]; then
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
