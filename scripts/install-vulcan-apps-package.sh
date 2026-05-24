#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${NEAT_APPS_INSTALL_DIR:-neat-apps}"
RUNTIME_SRC="${NEAT_APPS_RUNTIME_SRC:-neat-apps-runtime}"
RUNTIME_DST="${DEST_DIR}/neat-apps-runtime"
VULCAN_ENV="${NEAT_VULCAN_ENV:-${VULCAN_ENV:-dev}}"

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

extract_neat_core_target() {
  local json_path="$1"
  python3 - <<'PY' "${json_path}"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

neat_core = data.get("neat_core", {})
branch = str(neat_core.get("branch", "")).strip()
version = str(neat_core.get("version", "")).strip()

if not branch or not version:
    raise SystemExit(1)

print(branch)
print(version)
PY
}

if [[ ! -d "${RUNTIME_SRC}" ]]; then
  mapfile -t runtime_candidates < <(find . -mindepth 1 -maxdepth 3 -type d -name "${RUNTIME_SRC}" | sort)
  if [[ "${#runtime_candidates[@]}" -eq 1 ]]; then
    RUNTIME_SRC="${runtime_candidates[0]}"
  else
    echo "ERROR: ${RUNTIME_SRC} was not extracted from the apps package." >&2
    printf 'Found candidates:\n' >&2
    printf '  %s\n' "${runtime_candidates[@]}" >&2
    exit 1
  fi
fi

rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"
mv "${RUNTIME_SRC}" "${RUNTIME_DST}"

NEAT_CORE_JSON_PATH="${RUNTIME_DST}/neat-core.json"
if [[ ! -f "${NEAT_CORE_JSON_PATH}" ]]; then
  echo "ERROR: extracted apps package is missing neat-core.json." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to parse ${NEAT_CORE_JSON_PATH}." >&2
  exit 1
fi

if ! mapfile -t NEAT_CORE_TARGET < <(extract_neat_core_target "${NEAT_CORE_JSON_PATH}"); then
  echo "ERROR: failed to parse NEAT core dependency from ${NEAT_CORE_JSON_PATH}." >&2
  exit 1
fi

SIMA_CLI_RESOLVED="$(resolve_sima_cli_bin)" || {
  echo "ERROR: sima-cli is required to install matching NEAT core." >&2
  exit 1
}

NEAT_CORE_BRANCH="${NEAT_CORE_TARGET[0]}"
NEAT_CORE_VERSION="${NEAT_CORE_TARGET[1]}"

echo
echo "Installing matching NEAT core from Vulcan:"
echo "  Environment: ${VULCAN_ENV}"
echo "  Branch     : ${NEAT_CORE_BRANCH}"
echo "  Version    : ${NEAT_CORE_VERSION}"
"${SIMA_CLI_RESOLVED}" neat install \
  --env "${VULCAN_ENV}" \
  -d . \
  -t all \
  "core@${NEAT_CORE_BRANCH}:${NEAT_CORE_VERSION}"

DOWNLOAD_MODELS_SCRIPT="${RUNTIME_DST}/scripts/download_models.sh"
if [[ -x "${DOWNLOAD_MODELS_SCRIPT}" || -f "${DOWNLOAD_MODELS_SCRIPT}" ]]; then
  echo
  echo "Downloading models referenced by packaged README metadata ..."
  (
    cd "${RUNTIME_DST}"
    chmod +x scripts/download_models.sh
    SIMA_CLI_BIN="${SIMA_CLI_RESOLVED}" bash scripts/download_models.sh
  )
fi

echo
echo "Installed apps runtime under:"
echo "  ${DEST_DIR}"
