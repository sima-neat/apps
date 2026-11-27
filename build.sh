#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${NEAT_APPS_TEST_DEPS_DIR:-${ROOT_DIR}/deps}"
APPS_MANIFEST="${DEPS_DIR}/manifest.json"
NEAT_CORE_METADATA="${DEPS_DIR}/neat-core.json"
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CLEAN=0
BUILD_CPP=ON
INSTALL_CORE=0
ONLY_INSTALL=0
ONLY_RESOLVE=0
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
  --all                     Install NEAT core SDK, build apps, then package (scripts/ci/package-apps.sh)
  --only-install-neat-core  Install NEAT core SDK and exit (no build)
  --only-resolve-neat-core  Write deps/neat-core.json, print it as <ref>-<spec>, and exit
                            (no install, no build)
  --neat-core-version <ref>-<spec>
                            Override deps/manifest.json neat-core for one run
  -h, --help                Show help

Environment:
  NEAT_VULCAN_ENV           Core artifact environment: production (default), staging, or dev
  SIMA_NEAT_BASE_URL        Core artifact base URL instead of the environment's (sima-cli reads it)
  SIMA_CLI_BIN              Path to sima-cli binary (default: sima-cli)
  NEAT_APPS_DEPENDENCY_BRANCH
                            Dependency branch for snap manifest neat-core
  NEAT_SYNC_SYSROOT         Set to ON to align the eLxr SDK sysroot with the selected Core artifact
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
    --all) INSTALL_CORE=1; shift ;;
    --only-install-neat-core) INSTALL_CORE=1; ONLY_INSTALL=1; shift ;;
    --only-resolve-neat-core) INSTALL_CORE=1; ONLY_RESOLVE=1; shift ;;
    --neat-core-version)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "ERROR: --neat-core-version takes a <ref>-<spec> value (example: main-latest)." >&2
        exit 1
      fi
      NEAT_CORE_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# NEAT core install
# ---------------------------------------------------------------------------

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

  local tmp_dir metadata_file package_file policy_file
  tmp_dir="$(mktemp -d /tmp/neat-apps-sysroot.XXXXXX)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  metadata_file="${tmp_dir}/metadata-minimal.json"
  package_file="${tmp_dir}/simaai-scheduler.deb"
  policy_file="${tmp_dir}/scheduler/usr/share/SimaScheduler/dependency-policy.json"
  if ! download_file "${metadata_url}" "${metadata_file}"; then
    echo "ERROR: Failed to download selected Core metadata." >&2
    return 1
  fi

  # Core ships the Scheduler package it was built against. That package's
  # dependency policy names the exact SDK sysroot it was built for.
  local resource_info package_url expected_checksum
  if ! resource_info="$(python3 - "${metadata_file}" "${metadata_url}" <<'PY'
import json
import re
import sys
from urllib.parse import urljoin

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)
resource = "simaai-scheduler.deb"
if metadata.get("resources", []).count(resource) != 1:
    raise SystemExit("Core metadata must contain exactly one Scheduler package")
checksum = metadata.get("resources-checksum", {}).get(resource)
if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
    raise SystemExit("Core metadata has no valid Scheduler package checksum")
print(urljoin(sys.argv[2], resource) + "\t" + checksum)
PY
  )"; then
    echo "ERROR: Cannot verify the Core Scheduler package resource." >&2
    return 1
  fi
  package_url="${resource_info%%$'\t'*}"
  expected_checksum="${resource_info#*$'\t'}"
  if ! download_file "${package_url}" "${package_file}"; then
    echo "ERROR: Failed to download the Core Scheduler package." >&2
    return 1
  fi
  local package_checksum
  package_checksum="$(sha256sum "${package_file}")"
  if [[ "${package_checksum%% *}" != "${expected_checksum}" ]]; then
    echo "ERROR: Core Scheduler package checksum mismatch." >&2
    return 1
  fi
  if ! dpkg-deb -x "${package_file}" "${tmp_dir}/scheduler"; then
    echo "ERROR: Failed to extract the Core Scheduler package." >&2
    return 1
  fi

  local sysroot_version
  if ! sysroot_version="$(python3 - "${policy_file}" "${APPS_MANIFEST}" \
      "${ELXR_SDK_RELEASE_FILE}" "${SYSROOT:-/opt/toolchain/aarch64/modalix}" <<'PY'
import json
import re
import sys
from pathlib import Path

policy = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sysroot_version = policy["sysroot-version"]
consumer_base = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["platform-version"]
if not isinstance(sysroot_version, str) or not re.fullmatch(
    r"[0-9]+(?:[.][0-9]+){2}(?:~(?:pre[0-9]+|git[0-9]{12}[.][0-9a-f]{7,40}-[0-9]+))?", sysroot_version
):
    raise SystemExit("invalid sysroot-version")
if consumer_base != sysroot_version.split("~", 1)[0]:
    raise SystemExit("platform-version does not match the Scheduler sysroot-version")
if (policy.get("platform-version") != consumer_base
        or not isinstance(policy.get("elxr_version"), str)
        or not re.fullmatch(re.escape(consumer_base) + r"_(?:master|develop)_B[1-9][0-9]*", policy["elxr_version"])):
    raise SystemExit("Scheduler firmware does not match the Apps platform")
if "~" not in sysroot_version:
    # Match Core's official-release admission; never patch a release SDK in place.
    if policy.get("sdk", {}).get("sysroot-version") != sysroot_version:
        raise SystemExit("Scheduler SDK pin differs from its sysroot-version")
    fields = {}
    for line in Path(sys.argv[3]).read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            if key in fields:
                raise SystemExit(f"Duplicate SDK receipt field: {key}")
            fields[key] = value
    if (fields.get("Platform Version") != sysroot_version
            or fields.get("Platform Base") != sysroot_version
            or fields.get("Platform Channel") != "release"):
        raise SystemExit("Cached SDK does not contain the requested release; refresh its snapshot")
    sysroot = Path(sys.argv[4])
    overlay = sysroot / "var/lib/sima-sdk/sysroot-overlay"
    if overlay.exists() or overlay.is_symlink():
        raise SystemExit("Release SDK contains an overlay; use a fresh release snapshot")
    inventory = sysroot / "var/lib/sima-sdk/sysroot-packages.tsv"
    if inventory.is_symlink():
        raise SystemExit("SDK inventory must be a regular file")
    records = [line.split("\t") for line in inventory.read_text().splitlines()]
    for package in ("simaai-palette-modalix", "simaai-base-files-modalix"):
        versions = [row[2] for row in records if len(row) >= 3 and row[0] == package
                    and row[1] in {"all", "arm64"}]
        if versions != [sysroot_version]:
            raise SystemExit(f"Cached SDK {package} inventory differs: {versions}")
    if not (sysroot / "usr/include").is_dir():
        raise SystemExit("Cached SDK sysroot headers are missing")
    identities = re.findall(r"^SIMA_PACKAGE_VERSION\s*=\s*(\S+)\s*$",
                            (sysroot / "etc/packageinfo").read_text(), re.M)
    if identities != [policy.get("elxr_version")]:
        raise SystemExit(f"Cached SDK firmware identity differs: {identities}")
print(sysroot_version)
PY
  )"; then
    echo "ERROR: Cannot verify the Core Scheduler sysroot-version." >&2
    return 1
  fi

  if [[ "${sysroot_version}" == *~* ]]; then
    echo "Updating SDK sysroot to Core's Scheduler sysroot-version ${sysroot_version}"
    if ! run_privileged sysroot update "${sysroot_version}"; then
      echo "ERROR: Failed to update SDK sysroot to ${sysroot_version}." >&2
      return 1
    fi
  else
    echo "SKIP sysroot installation: SDK receipt, inventory and package identity match official release ${sysroot_version}."
  fi
  if ! sysroot status; then
    echo "ERROR: Failed to read SDK sysroot status." >&2
    return 1
  fi
  cp "${policy_file}" "${DEPS_DIR}/scheduler-sdk-policy.json"
)

recorded_neat_core() {
  python3 - "${NEAT_CORE_METADATA}" <<'PY'
import json
import sys

core = json.load(open(sys.argv[1], encoding="utf-8"))["neat-core"]
print(core["ref"], core["spec"])
PY
}

ensure_neat_core() {
  local core_script="${ROOT_DIR}/scripts/install-neat-core.sh"
  "${core_script}" resolve ${NEAT_CORE_OVERRIDE:+--neat-core-version "${NEAT_CORE_OVERRIDE}"}
  if [[ "${ONLY_RESOLVE}" -eq 1 ]]; then
    # The only stdout line, in the form --neat-core-version takes.
    recorded_neat_core | tr " " -
    exit 0
  fi

  if [[ "${NEAT_SYNC_SYSROOT:-OFF}" == "ON" ]]; then
    local sima_cli_bin branch version
    if ! sima_cli_bin="$(resolve_sima_cli_bin)"; then
      echo "ERROR: NEAT_SYNC_SYSROOT requires sima-cli on PATH or SIMA_CLI_BIN." >&2
      exit 1
    fi
    read -r branch version <<<"$(recorded_neat_core)"
    sync_sysroot_from_core_artifact "${sima_cli_bin}" "${branch}" "${version}" "${NEAT_VULCAN_ENV:-production}"
  fi
  "${core_script}" install
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
if [[ "${INSTALL_CORE}" -eq 1 ]]; then
  echo "  NEAT core target      : $(recorded_neat_core | tr " " :)"
else
  echo "  NEAT core target      : installed SDK core (not resolved during local build)"
fi
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
  bash "${ROOT_DIR}/scripts/ci/package-apps.sh" --build-dir "${BUILD_DIR}"
fi
