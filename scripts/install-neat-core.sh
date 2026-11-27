#!/usr/bin/env bash
# Selects the Neat Core this Apps checkout builds against and installs it with sima-cli.
# build.sh runs "resolve", then "install" unless --only-resolve-neat-core is given.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${NEAT_APPS_TEST_DEPS_DIR:-${ROOT_DIR}/deps}"
APPS_MANIFEST="${DEPS_DIR}/manifest.json"
NEAT_CORE_METADATA="${DEPS_DIR}/neat-core.json"
NEAT_VULCAN_ENV="${NEAT_VULCAN_ENV:-production}"

# shellcheck source=scripts/artifact-refs.sh
source "${ROOT_DIR}/scripts/artifact-refs.sh"

usage() {
  cat <<'EOF'
Usage: scripts/install-neat-core.sh resolve [--neat-core-version <ref>-<spec>]
       scripts/install-neat-core.sh install

  resolve  Selects core@<ref>:<spec> from --neat-core-version or deps/manifest.json,
           resolves "latest" to an exact revision, and writes deps/neat-core.json,
           which scripts/ci/package-apps.sh ships.
  install  Installs the revision that deps/neat-core.json records.

Environment:
  NEAT_VULCAN_ENV     Artifact environment for sima-cli --env: production (default), staging, or dev
  SIMA_NEAT_BASE_URL  Artifact base URL instead of the environment's. sima-cli reads it.
  SIMA_CLI_BIN        sima-cli to run (default: sima-cli on PATH, then the usual install paths)
  NEAT_APPS_DEPENDENCY_BRANCH
                      Apps branch the snap policy matches. An Apps tag comes first. Without
                      the variable, the policy takes GITHUB_HEAD_REF, then a branch
                      GITHUB_REF_NAME, then the checkout's branch, then develop outside a
                      git checkout.
EOF
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
  echo "ERROR: Installing NEAT core requires sima-cli on PATH or SIMA_CLI_BIN." >&2
  return 1
}

# Prints "snap", or the ref and spec on two lines. The manifest must be valid
# even when --neat-core-version replaces its neat-core.
select_core_ref() {
  python3 - "${APPS_MANIFEST}" "$1" <<'PY'
import json
import sys

path, override = sys.argv[1:]


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def split(value, label, colon=True):
    if colon and ":" in value:
        ref, spec = value.split(":", 1)
    elif "-" in value:
        ref, spec = value.rsplit("-", 1)
    else:
        fail(f"{label} must use branch-version (example: main-latest).")
    if not ref or not spec:
        fail(f"{label} must provide exactly one non-empty branch and version.")
    return ref, spec


try:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
except FileNotFoundError:
    fail(f"Missing manifest: {path}")
except json.JSONDecodeError as exc:
    fail(f"Invalid JSON in {path}: {exc}")

platform_version = data.get("platform-version")
if not isinstance(platform_version, str) or not platform_version.strip():
    fail(f"{path} must define platform-version as a non-empty string.")

neat_core = data.get("neat-core")
if isinstance(neat_core, str):
    value = neat_core.strip()
    selection = split(value, f"{path} neat-core") if value else None
elif isinstance(neat_core, dict):
    policy = str(neat_core.get("policy", "")).strip().lower()
    if policy and policy != "snap":
        fail(f"unsupported neat-core.policy in {path}: {policy!r}")
    ref = str(neat_core.get("branch", neat_core.get("ref", ""))).strip()
    if not policy and not ref:
        fail(f"{path} neat-core object must define policy=snap or branch/ref.")
    selection = None if policy else (ref, str(neat_core.get("spec", "")).strip() or "latest")
else:
    fail(f"{path} must define neat-core as a string or dependency object.")

if override:
    selection = split(override, "--neat-core-version", colon=False)
print("snap" if selection is None else "\n".join(selection))
PY
}

current_dependency_branch() {
  if [[ -n "${NEAT_APPS_DEPENDENCY_BRANCH:-}" ]]; then
    printf '%s\n' "${NEAT_APPS_DEPENDENCY_BRANCH}"
  elif [[ -n "${GITHUB_HEAD_REF:-}" ]]; then
    printf '%s\n' "${GITHUB_HEAD_REF}"
  elif [[ "${GITHUB_REF_TYPE:-}" != "tag" && -n "${GITHUB_REF_NAME:-}" ]]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
  elif command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if ! git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null; then
      echo "ERROR: git names no branch for ${ROOT_DIR}. Set NEAT_APPS_DEPENDENCY_BRANCH." >&2
      return 1
    fi
  else
    echo "develop"
  fi
}

# Prints SIMA_CLI_CHECK_FOR_UPDATE for one sima-cli call. sima-cli first asks whether to
# update itself, and the question aborts the call when no terminal can answer it.
update_check() {
  local accept="${SIMA_CLI_AUTO_ACCEPT_UPDATE:-}"
  if [[ -n "${SIMA_CLI_CHECK_FOR_UPDATE:-}" ]]; then
    printf '%s\n' "${SIMA_CLI_CHECK_FOR_UPDATE}"
  elif [[ "${accept,,}" =~ ^(1|true|yes|on)$ || ( "${1:-}" == "terminal" && -t 0 ) ]]; then
    echo 1
  else
    echo 0
  fi
}

core_lookup() {
  SIMA_CLI_CHECK_FOR_UPDATE="$(update_check)" "${SIMA_CLI}" neat install --env "${NEAT_VULCAN_ENV}" \
    -t minimal --json "core@$1:$2" </dev/null 2>"${LOOKUP_ERR}"
}

lookup_failed() {
  cat "${LOOKUP_ERR}" >&2
  echo "ERROR: Cannot resolve NEAT core artifact core@$1:$2 in ${NEAT_VULCAN_ENV}." >&2
  exit 1
}

resolve_neat_core() {
  local selection ref spec fallback=0 output resolved revision metadata_url tag branch=""
  # A failed resolve must not leave an earlier record for install to use.
  rm -f "${NEAT_CORE_METADATA}"
  selection="$(select_core_ref "$1")"
  if [[ "${selection}" == "snap" ]]; then
    # An Apps tag needs the Core release of the same name, so it never falls back.
    tag="$(current_dependency_tag)"
    if [[ -z "${tag}" ]]; then
      branch="$(current_dependency_branch)"
    fi
    ref="${tag:-${branch}}"
    spec="latest"
    [[ -n "${tag}" || "${branch}" == "main" || "${branch}" == "develop" ]] || fallback=1
  else
    ref="$(sed -n 1p <<<"${selection}")"
    spec="$(sed -n 2p <<<"${selection}")"
  fi

  LOOKUP_ERR="$(mktemp "${TMPDIR:-/tmp}/neat-apps-core-lookup.XXXXXX")"
  trap 'rm -f "${LOOKUP_ERR}"' EXIT
  if ! output="$(core_lookup "${ref}" "${spec}")"; then
    # Only a branch Core never published falls back. Other errors fail the build.
    if [[ "${fallback}" -eq 0 ]] || ! grep -Eq 'latest\.tag failed: 40[34]' "${LOOKUP_ERR}"; then
      lookup_failed "${ref}" "${spec}"
    fi
    echo "No NEAT core artifact found for dependency branch '${ref}'; using develop-latest." >&2
    ref="develop"
    output="$(core_lookup "${ref}" "${spec}")" || lookup_failed "${ref}" "${spec}"
  fi

  resolved="$(RESOLVE_OUTPUT="${output}" python3 - "${ref}" "${spec}" <<'PY'
import json
import os
import sys

ref, spec = sys.argv[1:]
text = os.environ["RESOLVE_OUTPUT"]
try:
    payload = json.loads(text[text.index("{"):])
except ValueError:
    raise SystemExit("ERROR: sima-cli printed no Core metadata.")
resolved, url = payload.get("resolved_spec"), payload.get("metadata_url")
if (payload.get("repository") != "core" or payload.get("ref") != ref
        or spec not in ("latest", resolved) or not resolved or not url):
    raise SystemExit("ERROR: Cannot verify resolved Core metadata identity.")
print(resolved, url, sep="\n")
PY
  )"
  revision="$(sed -n 1p <<<"${resolved}")"
  metadata_url="$(sed -n 2p <<<"${resolved}")"
  # sima-cli names an exact revision's URL without fetching it.
  if [[ "${spec}" != "latest" ]] && ! python3 - "${metadata_url}" <<'PY'
import sys
import urllib.error
import urllib.request

try:
    urllib.request.urlopen(sys.argv[1], timeout=60).close()
except urllib.error.HTTPError as exc:
    raise SystemExit(f"HTTP {exc.code}")
except (OSError, ValueError) as exc:
    raise SystemExit(f"cannot reach artifact host: {getattr(exc, 'reason', exc)}")
PY
  then
    echo "ERROR: unavailable NEAT core artifact '${ref}-${spec}': ${metadata_url}" >&2
    return 1
  fi

  mkdir -p "${DEPS_DIR}"
  python3 - "${NEAT_CORE_METADATA}" "${ref}" "${revision}" <<'PY'
import json
import sys

path, ref, spec = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"neat-core": {"ref": ref, "spec": spec}}, handle, indent=2)
    handle.write("\n")
PY
  echo "Selected NEAT core core@${ref}:${revision} (${NEAT_VULCAN_ENV})." >&2
}

install_neat_core() {
  local target status=0
  target="$(python3 - "${NEAT_CORE_METADATA}" <<'PY'
import json
import sys

try:
    core = json.load(open(sys.argv[1], encoding="utf-8"))["neat-core"]
    print(f"core@{core['ref']}:{core['spec']}")
except (OSError, ValueError, KeyError, TypeError) as exc:
    raise SystemExit(f"ERROR: {sys.argv[1]} records no Core ({exc!r}). Run resolve first.")
PY
  )"
  INSTALL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/neat-apps-core-install.XXXXXX")"
  INSTALL_LOG="$(mktemp "${TMPDIR:-/tmp}/neat-apps-sima-cli-install.XXXXXX")"
  trap 'rm -rf "${INSTALL_DIR}" "${INSTALL_LOG}"' EXIT
  echo "Installing NEAT core ${target} (${NEAT_VULCAN_ENV})..."
  (cd "${INSTALL_DIR}" && SIMA_CLI_CHECK_FOR_UPDATE="$(update_check terminal)" \
    "${SIMA_CLI}" neat install --env "${NEAT_VULCAN_ENV}" -d . -t minimal "${target}") 2>&1 | tee "${INSTALL_LOG}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    return "${status}"
  fi
  # sima-cli exits 0 without installing when Core's installer exits 255, and when the
  # metadata runs no installation script.
  if grep -Fq "Installation script exited" "${INSTALL_LOG}" ||
     ! grep -Fq "Installation completed successfully." "${INSTALL_LOG}"; then
    echo "ERROR: sima-cli did not complete the NEAT core install." >&2
    return 1
  fi
  echo "NEAT core installed successfully (${target})."
}

command_name="${1:-}"
[[ $# -eq 0 ]] || shift
override=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --neat-core-version)
      if [[ "${command_name}" != "resolve" || $# -lt 2 || -z "$2" ]]; then
        echo "ERROR: --neat-core-version takes a <ref>-<spec> value and works only with resolve." >&2
        exit 1
      fi
      override="$2"
      shift 2
      ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "${command_name}" in
  resolve) SIMA_CLI="$(resolve_sima_cli_bin)"; resolve_neat_core "${override}" ;;
  install) SIMA_CLI="$(resolve_sima_cli_bin)"; install_neat_core ;;
  -h|--help) usage ;;
  *) usage >&2; exit 1 ;;
esac
