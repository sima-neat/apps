#!/usr/bin/env bash
# Aligns the eLxr SDK sysroot with the Scheduler package that the selected Core ships,
# and writes deps/scheduler-sdk-policy.json, which runner selection reads.
#
# Usage: scripts/ci/sync-sdk-sysroot.sh
# Run it after ./build.sh --only-resolve-neat-core, which writes deps/neat-core.json, and
# before ./build.sh --all, because a sysroot update replaces a Core install in the sysroot.
# Needs an eLxr SDK, sima-cli, curl and git.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPS_DIR="${ROOT_DIR}/deps"
APPS_MANIFEST="${DEPS_DIR}/manifest.json"
NEAT_CORE_METADATA="${DEPS_DIR}/neat-core.json"
POLICY_OUTPUT="${DEPS_DIR}/scheduler-sdk-policy.json"
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"
SYSROOT_DIR="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
SIMA_CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
VULCAN_ENV="${NEAT_VULCAN_ENV:-production}"
CORE_GIT_URL="https://github.com/sima-neat/core.git"

case "${1:-}" in
  "") ;;
  -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "Unknown argument: $1" >&2; exit 1 ;;
esac

# A run that fails after this point leaves no policy, so runner selection never reads a stale one.
rm -f "${POLICY_OUTPUT}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

sdk_release_field() {
  sed -n "s/^$1[[:space:]]*=[[:space:]]*//p" "${ELXR_SDK_RELEASE_FILE}" | head -n1
}

if [[ ! -f "${ELXR_SDK_RELEASE_FILE}" || -z "$(sdk_release_field 'SDK Version')" ||
      -z "$(sdk_release_field 'eLXr Version')" ]]; then
  fail "The SDK sysroot sync needs an eLxr SDK, and ${ELXR_SDK_RELEASE_FILE} names none."
fi
command -v sysroot >/dev/null 2>&1 || fail "The SDK sysroot sync needs the sysroot command."
command -v "${SIMA_CLI_BIN}" >/dev/null 2>&1 || fail "The SDK sysroot sync needs sima-cli on PATH or SIMA_CLI_BIN."

if ! core_target="$(python3 - "${NEAT_CORE_METADATA}" <<'PY'
import json
import re
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as handle:
        core = json.load(handle)["neat-core"]
    ref, spec = core["ref"], core["spec"]
except FileNotFoundError:
    raise SystemExit(f"ERROR: {path} is missing. Run ./build.sh --only-resolve-neat-core first.")
except (ValueError, KeyError, TypeError):
    raise SystemExit(f"ERROR: {path} must hold neat-core.ref and neat-core.spec.")
# Vulcan publishes Core under its commit, so the spec also selects Core's tools/ci.
if (not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref)
        or not isinstance(spec, str) or not re.fullmatch(r"[0-9a-f]{12,40}", spec)):
    raise SystemExit(f"ERROR: {path} must name a Core ref and an exact commit spec, not {core!r}.")
print(ref)
print(spec)
PY
)"; then
  exit 1
fi
ref="$(sed -n 1p <<< "${core_target}")"
spec="$(sed -n 2p <<< "${core_target}")"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/neat-apps-sysroot.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT

if ! resolve_output="$("${SIMA_CLI_BIN}" neat install --env "${VULCAN_ENV}" -t minimal --json \
    "core@${ref}:${spec}")"; then
  fail "sima-cli could not resolve core@${ref}:${spec}."
fi
if ! metadata_url="$(RESOLVE_OUTPUT="${resolve_output}" python3 - "${ref}" "${spec}" <<'PY'
import json
import os
import sys

text = os.environ["RESOLVE_OUTPUT"]
try:
    payload = json.loads(text[text.index("{"):])
except ValueError:
    raise SystemExit("ERROR: sima-cli printed no Core metadata JSON.")
selected = (payload.get("repository"), payload.get("ref"), payload.get("resolved_spec"))
if selected != ("core", sys.argv[1], sys.argv[2]):
    raise SystemExit(f"ERROR: sima-cli resolved {selected}, not core@{sys.argv[1]}:{sys.argv[2]}.")
metadata_url = payload.get("metadata_url")
if not isinstance(metadata_url, str) or not metadata_url:
    raise SystemExit("ERROR: sima-cli printed no Core metadata URL.")
print(metadata_url)
PY
)"; then
  exit 1
fi

metadata_file="${tmp_dir}/metadata-minimal.json"
curl -fsSL "${metadata_url}" -o "${metadata_file}" || fail "Failed to download ${metadata_url}."
if ! package_info="$(python3 - "${metadata_file}" "${metadata_url}" <<'PY'
import json
import re
import sys
from urllib.parse import urljoin

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        metadata = json.load(handle)
except ValueError:
    raise SystemExit(f"ERROR: Core metadata at {sys.argv[2]} is not JSON.")
resources = metadata.get("resources", []) if isinstance(metadata, dict) else None
checksums = metadata.get("resources-checksum", {}) if isinstance(metadata, dict) else None
if not isinstance(resources, list) or not isinstance(checksums, dict):
    raise SystemExit(f"ERROR: Core metadata at {sys.argv[2]} must hold a resources list and a "
                     "resources-checksum map.")
resource = "simaai-scheduler.deb"
count = resources.count(resource)
if count != 1:
    raise SystemExit(f"ERROR: Core metadata lists {resource} {count} times, not once.")
checksum = checksums.get(resource)
if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
    raise SystemExit(f"ERROR: Core metadata has no SHA-256 checksum for {resource}.")
print(urljoin(sys.argv[2], resource))
print(checksum)
PY
)"; then
  exit 1
fi
package_url="$(sed -n 1p <<< "${package_info}")"
expected_checksum="$(sed -n 2p <<< "${package_info}")"
package_file="${tmp_dir}/simaai-scheduler.deb"
curl -fsSL "${package_url}" -o "${package_file}" || fail "Failed to download ${package_url}."
actual_checksum="$(sha256sum "${package_file}")"
if [[ "${actual_checksum%% *}" != "${expected_checksum}" ]]; then
  fail "The Scheduler package's SHA-256 is ${actual_checksum%% *}, but Core metadata says ${expected_checksum}."
fi
dpkg-deb -x "${package_file}" "${tmp_dir}/scheduler" || fail "Failed to extract the Core Scheduler package."
policy_file="${tmp_dir}/scheduler/usr/share/SimaScheduler/dependency-policy.json"
[[ -f "${policy_file}" ]] || fail "The Core Scheduler package has no dependency-policy.json."

# The validator and its published_artifact import come from the Core that Apps installs.
core_dir="${tmp_dir}/core"
# CI has no terminal, so git must fail instead of asking for credentials.
export GIT_TERMINAL_PROMPT=0
core_git() {
  git -C "${core_dir}" "$@"
}
core_commit() {
  core_git rev-parse -q --verify "${spec}^{commit}"
}
git init -q "${core_dir}"
core_git remote add origin "${CORE_GIT_URL}" || fail "Failed to add ${CORE_GIT_URL} as the Core remote."
ref_fetch=0
core_git fetch -q --filter=tree:0 --no-tags origin "${ref}" || ref_fetch=$?
# A Vulcan artifact outlives its branch, and a merged pull request keeps its head ref.
if ! commit="$(core_commit)"; then
  echo "Core ref ${ref} is missing or lacks commit ${spec}. Searching the Core pull request heads."
  pull_fetch=0
  core_git fetch -q --filter=tree:0 --no-tags origin '+refs/pull/*/head:refs/pull/*' || pull_fetch=$?
  if [[ "${pull_fetch}" -ne 0 && "${ref_fetch}" -ne 0 ]]; then
    fail "Could not fetch Core from ${CORE_GIT_URL}."
  elif [[ "${pull_fetch}" -ne 0 ]]; then
    fail "Could not fetch the Core pull request heads from ${CORE_GIT_URL}."
  fi
  commit="$(core_commit)" ||
    fail "Core commit ${spec} is in neither ${ref} nor any Core pull request head at ${CORE_GIT_URL}."
fi
core_git sparse-checkout set --cone tools/ci || fail "Failed to limit the Core checkout to tools/ci."
core_git -c advice.detachedHead=false checkout -q "${commit}" ||
  fail "Failed to check out Core tools/ci at ${commit}."
policy_tool="${core_dir}/tools/ci/scheduler_sdk_policy.py"
[[ -f "${policy_tool}" ]] || fail "Core ${commit} has no tools/ci/scheduler_sdk_policy.py."

policy="$(<"${policy_file}")"
check_policy() {
  python3 "${policy_tool}" --policy-json "${policy}" --core-manifest "${APPS_MANIFEST}" "$@"
}
# Exit status 2 is an argparse usage error, from a Core whose validator takes other options.
validator_failed() {
  if [[ "$1" -eq 2 ]]; then
    fail "Core ${commit}'s scheduler_sdk_policy.py does not accept the arguments this script passes."
  fi
  fail "$2"
}
sysroot_version="$(check_policy --field sysroot-version)" ||
  validator_failed $? "Core's scheduler_sdk_policy.py rejected the Scheduler SDK policy of core@${ref}:${spec}."

if [[ "${sysroot_version}" == *~* ]]; then
  # Core gap: no Core entry point applies a Scheduler SDK policy, so this copies prepare_scheduler_sdk.sh.
  pkg_list="$(tr -d '[:space:]' <<< "${SDK_PKG_LIST:-}" | sed 's/,,*/,/g; s/^,//; s/,$//')"
  update=(env "SDK_PKG_LIST=${pkg_list}" sysroot update --sysroot "${SYSROOT_DIR}" "${sysroot_version}")
  echo "Updating the SDK sysroot to ${sysroot_version}, the version Core's Scheduler was built for."
  if [[ "$(id -u)" -eq 0 ]]; then
    "${update[@]}" || fail "Failed to update the SDK sysroot to ${sysroot_version}."
  elif command -v sudo >/dev/null 2>&1; then
    sudo "${update[@]}" || fail "Failed to update the SDK sysroot to ${sysroot_version}."
  else
    fail "Updating the SDK sysroot to ${sysroot_version} needs root or sudo."
  fi
else
  check_policy --sysroot "${SYSROOT_DIR}" --check-release-sdk ||
    validator_failed $? "The active SDK sysroot is not official release ${sysroot_version}. Use an SDK image of that release."
fi
status="$(sysroot status --sysroot "${SYSROOT_DIR}")" || fail "Failed to read the SDK sysroot status."
echo "${status}"
if [[ "${sysroot_version}" == *~* ]]; then
  # sysroot status exits 0 even when an update left the overlay incomplete. Without an
  # overlay, the sysroot is the one the SDK image shipped, as Core's prepare_scheduler_sdk.sh reads it.
  state="$(sed -n 's/^Sysroot overlay state:[[:space:]]*//p' <<< "${status}")"
  if [[ "${state}" == "inactive" ]]; then
    revision="$(sed -n 's/^SDK image platform:[[:space:]]*//p' <<< "${status}")"
  else
    revision="$(sed -n 's/^Sysroot overlay revision:[[:space:]]*//p' <<< "${status}")"
  fi
  if [[ ( "${state}" != "active" && "${state}" != "inactive" ) || "${revision}" != "${sysroot_version}" ]]; then
    fail "The SDK sysroot is at ${revision:-unknown} (${state:-unknown}) after updating to ${sysroot_version}."
  fi
fi
cp "${policy_file}" "${POLICY_OUTPUT}"
