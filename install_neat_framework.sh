#!/usr/bin/env bash
set -euo pipefail

# install_neat_framework.sh
#
# Purpose:
# - Install SiMa NEAT wheel into a Python virtual environment (board mode only).
# - Install NEAT runtime .deb packages on Modalix board or into eLxr SDK sysroot.
#
# Behavior:
# - Auto-detects environment:
#   - eLxr SDK when /etc/sdk-release exists, or SYSROOT points to a valid directory.
#   - Modalix board when /etc/buildinfo reports MACHINE=modalix.
# - In Modalix board mode, installs .deb packages with apt (sudo).
#
# Expected working directory:
# - Directory containing:
#   - one .whl file
#   - sima-neat-*-Linux-core.deb
#   - neat-*.deb runtime dependencies
#
# Environment overrides:
# - PYNEAT_VENV_DIR: Python virtualenv path (default: $HOME/pyneat/.venv)
# - SUDO_PASSWORD / DEVKIT_PASSWORD: sudo password (preferred non-interactive override)
# - DEFAULT_SUDO_PASSWORD: fallback password (default: edgeai)
# - SYSROOT: SDK sysroot path override (default: /opt/toolchain/aarch64/modalix)

VENV_DIR="${PYNEAT_VENV_DIR:-$HOME/pyneat/.venv}"
SUDO_PASSWORD="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
DEFAULT_SUDO_PASSWORD="${DEFAULT_SUDO_PASSWORD:-edgeai}"
log() {
  printf '[install_neat_framework] %s\n' "$*"
}

run_sudo() {
  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return 0
  fi

  local pw="${SUDO_PASSWORD}"
  if [[ -z "${pw}" ]]; then
    pw="${DEFAULT_SUDO_PASSWORD}"
  fi

  if printf '%s\n' "${pw}" | sudo -S -v >/dev/null 2>&1; then
    printf '%s\n' "${pw}" | sudo -S "$@"
    return 0
  fi

  if [[ -t 0 ]]; then
    read -r -s -p "Enter sudo password: " pw
    echo
    if [[ -z "${pw}" ]]; then
      echo "sudo password is required." >&2
      exit 1
    fi
    printf '%s\n' "${pw}" | sudo -S -v >/dev/null
    printf '%s\n' "${pw}" | sudo -S "$@"
    return 0
  fi

  echo "Unable to authenticate sudo. Set SUDO_PASSWORD or DEVKIT_PASSWORD." >&2
  exit 1
}

detect_env_mode() {
  if [[ -f /etc/sdk-release ]]; then
    echo "elxr-sdk"
    return 0
  fi
  if [[ -n "${SYSROOT:-}" && -d "${SYSROOT}" ]]; then
    echo "elxr-sdk"
    return 0
  fi
  if [[ -f /etc/buildinfo ]] && grep -qE '^MACHINE[[:space:]]*=[[:space:]]*modalix' /etc/buildinfo; then
    echo "modalix-board"
    return 0
  fi
  echo "modalix-board"
}

install_debs_on_board() {
  log "Detected Modalix board environment; installing DEBs with apt."
  run_sudo apt install -y --allow-downgrades "${DEBS[@]}"
}

install_debs_into_sysroot() {
  local sysroot="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
  if [[ ! -d "${sysroot}" ]]; then
    echo "SYSROOT does not exist: ${sysroot}" >&2
    exit 1
  fi
  if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "dpkg-deb is required for eLxr SDK/sysroot installs." >&2
    exit 1
  fi

  log "Detected eLxr SDK environment; installing DEBs into sysroot: ${sysroot}"
  local deb
  for deb in "${DEBS[@]}"; do
    log "Extracting $(basename "${deb}") into ${sysroot}"
    run_sudo dpkg-deb -x "${deb}" "${sysroot}"
  done
}

mapfile -t DEBS < <(find . -maxdepth 1 -type f \( -name 'sima-neat-*-Linux-core.deb' -o -name 'neat-*.deb' \) | sort)
if [[ "${#DEBS[@]}" -lt 1 ]]; then
  echo "No required DEB files found in current directory." >&2
  exit 1
fi

ENV_MODE="$(detect_env_mode)"
if [[ "${ENV_MODE}" == "elxr-sdk" ]]; then
  install_debs_into_sysroot
else
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip

  WHEEL_FILE="$(ls -1 ./*.whl 2>/dev/null | head -n1 || true)"
  if [[ -z "${WHEEL_FILE}" ]]; then
    echo "No wheel file found in current directory." >&2
    exit 1
  fi
  "${VENV_DIR}/bin/python" -m pip install --force-reinstall "${WHEEL_FILE}"
  install_debs_on_board
fi
