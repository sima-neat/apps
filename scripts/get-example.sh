#!/usr/bin/env bash
# Fetch one example of the bundle `sima-cli neat install apps@<ref>` installs, laid out as in
# prebuilt-apps/ so the example's README commands run unchanged from the destination folder.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: get-example.sh [--ref <branch|tag>[:<commit>]] [--dest <dir>] [--force] <example>
  <example>  name (single-stream-object-detector) or path (object-detection/<name>)
  --ref      branch or tag the bundle was published from (default: main, latest build);
             --branch is the same option
  --dest     destination folder (default: ./<example name>)
  --force    replace the destination folder, keeping only its models/
Environment: NEAT_APPS_BRANCH (--ref), NEAT_APPS_EXAMPLE_DEST_DIR (--dest),
  NEAT_APPS_ARTIFACTS_URL (default https://artifacts.neat.sima.ai),
  NEAT_APPS_ARCHIVE_URL (a bundle .tar.gz URL or local path, used without the index or a checksum)
USAGE
}

# The absolute path of $1 with symlinks resolved, also when its last parts do not exist yet
# (no realpath -m, which macOS and BusyBox lack). Empty for a missing part named . or ..
resolve() {
  local dir="$1" rest=""
  while [[ ! -d "${dir}" ]]; do
    case "$(basename -- "${dir}")" in .|..) return 0 ;; esac
    rest="/$(basename -- "${dir}")${rest}"; dir="$(dirname -- "${dir}")"
  done
  dir="$(CDPATH='' cd -- "${dir}" && pwd -P)" && printf '%s%s\n' "${dir%/}" "${rest}"
}

# Copy or download $1 to $2. Return 4 when the index answers 403 or 404, which mean "not
# published". Print an error and return 1 on any other failure, a 5xx after retries included.
fetch() {
  local code log help file="${1#file://}"
  local -a opts=()
  if [[ -f "${file}" ]]; then cp "${file}" "$2"
  elif [[ "$1" != *://* || "$1" == file://* ]]; then return 4
  elif command -v curl >/dev/null; then
    code="$(curl -sSL --retry 3 --connect-timeout 20 --speed-time 60 --speed-limit 1000 \
      -w '%{http_code}' -o "$2" "$1")" || return 1
    case "${code}" in 200) ;; 403|404) return 4 ;; *) echo "HTTP ${code}: $1" >&2; return 1 ;; esac
  elif command -v wget >/dev/null; then
    # BusyBox wget rejects or mishandles the GNU timeout and retry options, so only GNU gets them.
    help="$(wget --help 2>&1 || true)"
    [[ "${help}" == *--read-timeout* ]] && opts+=(--tries=3 --read-timeout=60)
    [[ "${help}" == *--retry-on-http-error* ]] && opts+=("--retry-on-http-error=500,502,503,504")
    log="$(wget -S -q ${opts[@]+"${opts[@]}"} -O "$2" "$1" 2>&1)" && return 0
    code="$(sed -n 's#.*HTTP/[0-9.]* \([0-9][0-9][0-9]\).*#\1#p' <<< "${log}" | tail -n 1)"
    case "${code}" in 403|404) return 4 ;; esac
    echo "wget failed${code:+ with HTTP ${code}}: $1" >&2
    tail -n 3 <<< "${log}" >&2
    return 1
  else echo "curl or wget is required." >&2; return 1; fi
}

# Swap the new folder in by rename. The replaced folder waits in the stage until the exit.
swap_in() {
  if [[ -e "${TARGET}" ]]; then
    mv -- "${TARGET}" "${STAGE}/old" || return 1
    if [[ -d "${STAGE}/old/models" && ! -e "${STAGE}/new/models" ]]; then
      mv -- "${STAGE}/old/models" "${STAGE}/new/models" || return 1
    fi
  fi
  mv -- "${STAGE}/new" "${TARGET}"
}

# Put the replaced folder back at the destination, with the models/ it lent the new one.
restore() {
  if [[ -d "${STAGE}/new/models" && ! -e "${STAGE}/old/models" ]]; then
    mv -- "${STAGE}/new/models" "${STAGE}/old/models" || return 1
  fi
  mv -- "${STAGE}/old" "${TARGET}"
}

# A swap that failed or was interrupted leaves the old folder only in the stage, so put it back
# first. If that fails, keep the stage and say where the old folder is.
cleanup() {
  rm -rf "${TMP}"
  [[ -n "${STAGE}" ]] || return 0
  if [[ -e "${STAGE}/old" && ! -e "${TARGET}" ]] && ! restore; then
    echo "The old folder is kept in ${STAGE}/old." >&2
    return 0
  fi
  chmod -R u+rwx "${STAGE}" 2>/dev/null || true
  rm -rf "${STAGE}" || echo "Could not remove ${STAGE}" >&2
}

# All work happens in main, so a download of this script that stops early runs nothing.
main() {
  local ref="${NEAT_APPS_BRANCH:-main}" dest="${NEAT_APPS_EXAMPLE_DEST_DIR:-}" force=0 example=""
  local base="${NEAT_APPS_ARTIFACTS_URL:-https://artifacts.neat.sima.ai}/apps"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --ref|--branch|--dest) [[ $# -ge 2 ]] || { echo "$1 needs a value." >&2; exit 2; }
        if [[ "$1" == --dest ]]; then dest="$2"; else ref="$2"; fi; shift 2 ;;
      --force) force=1; shift ;;
      -h|--help) usage; exit 0 ;;
      -*) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
      *) [[ -z "${example}" ]] || { echo "Only one example at a time." >&2; exit 2; }; example="$1"; shift ;;
    esac
  done
  if [[ -n "${NEAT_APPS_REPO_URL:-}" ]]; then
    echo "NEAT_APPS_REPO_URL is not read: the script fetches the published bundle. Unset it, or set NEAT_APPS_ARCHIVE_URL to a bundle." >&2
    exit 2
  fi
  [[ -n "${example}" ]] || { usage >&2; exit 2; }
  example="${example#examples/}"
  example="${example%/}"
  case "${example}" in
    genai-studio|multimodal-assistant|multimodal-assist|genai/multimodal-assistant) example=neat-genai-studio ;;
  esac
  [[ "${example}" =~ ^([A-Za-z0-9_-]+/)?[A-Za-z0-9_-]+$ ]] || { echo "Not an example name: ${example}" >&2; exit 2; }
  dest="${dest:-$(basename "${example}")}"

  local link="${dest}"
  while [[ "${link}" == */ && "${link}" != / ]]; do link="${link%/}"; done
  [[ ! -L "${link}" ]] || { echo "Refusing a symlink as destination: ${dest}" >&2; exit 1; }
  # A last part . or .. names a folder by its link or its parent, so it is never a destination.
  case "$(basename -- "${link}")" in .|..) echo "Refusing destination: ${dest}" >&2; exit 1 ;; esac
  if [[ -d "${link}" && ( ! -r "${link}" || ! -x "${link}" ) ]]; then
    echo "Refusing to replace ${dest}: it is not readable." >&2; exit 1
  fi
  TARGET="$(resolve "${dest}")"
  # --force deletes the destination: never /, the current folder, $HOME, or a folder above them.
  local keep
  for keep in "$(pwd -P)" "$(resolve "${HOME:-/}")"; do
    if [[ -z "${TARGET}" || "${TARGET}" == / || "${keep%/}/" == "${TARGET}/"* ]]; then
      echo "Refusing destination: ${dest}" >&2; exit 1
    fi
  done
  # A folder right under / is a system folder, such as /opt or /tmp.
  [[ "${TARGET}" == /*/* ]] || { echo "Refusing destination: ${dest}" >&2; exit 1; }
  if [[ -e "${TARGET}" && "${force}" -ne 1 ]]; then
    echo "Destination exists: ${dest} (remove it, or use --force or --dest)." >&2; exit 1
  fi
  # --force replaces one fetched example: a file, an empty folder, or one with
  # deps/neat-apps.json and a single example, never a sima-cli install.
  if [[ -d "${TARGET}" && -n "$(ls -A "${TARGET}")" && ! -f "${TARGET}/deps/neat-apps.json" ]]; then
    echo "Refusing to replace ${dest}: it has no deps/neat-apps.json, so it is not a fetched example." \
      "If an older get-example.sh fetched it, move it away and fetch again." >&2
    exit 1
  fi
  local -a held=("${TARGET}"/examples/*/*/)
  if [[ ${#held[@]} -gt 1 ]]; then
    echo "Refusing to replace ${dest}: it holds more than one example, such as a sima-cli install." >&2
    exit 1
  fi
  if [[ -d "${TARGET}/models" && ! -L "${TARGET}/models" && ! -w "${TARGET}/models" ]]; then
    echo "Make ${dest}/models writable, so --force can keep it." >&2; exit 1
  fi

  TMP="$(mktemp -d)"
  local url="${NEAT_APPS_ARCHIVE_URL:-}" sum="" key spec name status
  if [[ -z "${url}" ]]; then
    # The index keys a ref by its URL-encoded name, encoded once more in the path (/ -> %252F).
    [[ "${ref}" =~ ^[A-Za-z0-9._/-]+(:[0-9a-f]+)?$ ]] || { echo "Unsupported ref: ${ref}" >&2; exit 2; }
    if [[ "${ref}" == *:* && ! "${ref#*:}" =~ ^[0-9a-f]{12,40}$ ]]; then
      echo "Commit in ${ref}: give at least 12 characters of the commit (at most 40)." >&2; exit 2
    fi
    key="${ref%%:*}"; key="${key//\//%252F}"
    if [[ "${ref}" == *:* ]]; then
      spec="${ref#*:}"; spec="${spec:0:12}"  # the index names a build by 12 characters
    else
      status=0; fetch "${base}/${key}/latest.tag" "${TMP}/tag" || status=$?
      if [[ "${status}" -eq 4 ]]; then
        echo "No published Apps bundle for ${ref} (${base}/${key}). Published branches: ${base}/branches.json" >&2; exit 1
      fi
      [[ "${status}" -eq 0 ]] || { echo "Cannot read the index: ${base}/${key}/latest.tag" >&2; exit 1; }
      spec="$(tr -d '[:space:]' < "${TMP}/tag")"
      # A proxy or login page answers with HTML instead of a build name.
      [[ "${spec}" =~ ^[0-9a-f]{12}$ ]] || { echo "Not a build name in ${base}/${key}/latest.tag" >&2; exit 1; }
    fi
    status=0; fetch "${base}/${key}/${spec}/metadata.json" "${TMP}/metadata.json" || status=$?
    if [[ "${status}" -eq 4 ]]; then
      echo "No build ${spec} of ${ref%%:*} (${base}/${key}/${spec}/metadata.json)." >&2; exit 1
    fi
    [[ "${status}" -eq 0 ]] || { echo "Cannot read the index: ${base}/${key}/${spec}/metadata.json" >&2; exit 1; }
    name="$(grep -o '"neat-apps-[^"]*\.tar\.gz"' "${TMP}/metadata.json" | head -n1 | tr -d '"' || true)"
    [[ -n "${name}" ]] && sum="$(sed -n "s/.*\"${name//./\\.}\": *\"\([0-9a-f]\{64\}\)\".*/\1/p" "${TMP}/metadata.json")"
    [[ -n "${sum}" ]] || { echo "No bundle or checksum in ${base}/${key}/${spec}/metadata.json" >&2; exit 1; }
    url="${base}/${key}/${spec}/${name}"
  fi

  echo "Fetching ${url} ..."
  fetch "${url}" "${TMP}/bundle.tar.gz" || { echo "Download failed: ${url}" >&2; exit 1; }
  if [[ -n "${sum}" ]]; then
    local got
    if command -v sha256sum >/dev/null; then got="$(sha256sum "${TMP}/bundle.tar.gz")"
    elif command -v shasum >/dev/null; then got="$(shasum -a 256 "${TMP}/bundle.tar.gz")"
    else echo "sha256sum or shasum is required." >&2; exit 1; fi
    [[ "${got%% *}" == "${sum}" ]] || { echo "Checksum mismatch: ${url}" >&2; exit 1; }
  fi
  tar -tzf "${TMP}/bundle.tar.gz" > "${TMP}/list"
  local root pattern ex
  root="$(awk -F/ 'NR == 1 { print $1 }' "${TMP}/list")"
  # A bare name matches examples/<category>/<name>/ in any category.
  [[ "${example}" == */* ]] && pattern="${example}" || pattern="[^/]+/${example}"
  ex="$(grep -E "^${root}/examples/${pattern}/$" "${TMP}/list" || true)"
  if [[ -z "${ex}" || "${ex}" == *$'\n'* ]]; then
    [[ -z "${ex}" ]] && echo "Example not in the bundle: ${example}. It has:" >&2 \
      || echo "More than one example is named ${example}. Give <category>/<name>:" >&2
    sed -n "s#^${root}/examples/\([^/]*/[^/]*\)/\$#  \1#p" <<< "${ex:-$(cat "${TMP}/list")}" | sort >&2
    exit 1
  fi
  ex="${ex%/}"
  mkdir "${TMP}/x"
  tar -xzf "${TMP}/bundle.tar.gz" -C "${TMP}/x" "${ex}"

  # An example whose code or config names another example as <category>/<name> runs it too.
  # A README only mentions it. BusyBox grep has no --exclude, so README.md is dropped after.
  local others hits status=0
  others="$(sed -n "s#^${root}/examples/\([^/]*/[^/]*\)/\$#\1#p" "${TMP}/list" \
    | grep -vxF "${ex#"${root}"/examples/}" | tr '\n' '|' || true)"
  if [[ -n "${others}" ]]; then
    hits="$(grep -rlE "(^|[^A-Za-z0-9_-])(${others%|})([^A-Za-z0-9_-]|\$)" "${TMP}/x/${ex}")" || status=$?
    [[ "${status}" -le 1 ]] || { echo "grep failed on ${ex}." >&2; exit 1; }
    if [[ -n "$(grep -v '/README\.md$' <<< "${hits}" || true)" ]]; then
      echo "${example} needs the whole bundle: sima-cli neat install apps" >&2; exit 1
    fi
  fi
  # support/, deps/ and this script when the bundle has them, since older bundles lack some,
  # and only the assets/datasets folders the example's files name.
  local -a members=()
  local d
  for d in support deps scripts/get-example.sh; do
    grep -qxF -e "${root}/${d}/" -e "${root}/${d}" "${TMP}/list" && members+=("${root}/${d}")
  done
  while IFS= read -r d; do
    grep -rqIE "assets/datasets/${d//./\\.}([^A-Za-z0-9_.-]|\\.([^A-Za-z0-9_-]|\$)|\$)" "${TMP}/x/${ex}" \
      && members+=("${root}/assets/datasets/${d}")
  done < <(sed -n "s#^${root}/assets/datasets/\([A-Za-z0-9_][A-Za-z0-9_.-]*\)/.*#\1#p" "${TMP}/list" | sort -u)
  [[ ${#members[@]} -eq 0 ]] || tar -xzf "${TMP}/bundle.tar.gz" -C "${TMP}/x" "${members[@]}"
  # A link could point the fetched folder at files outside it.
  local link_found
  link_found="$(find "${TMP}/x" -type l | head -n 1)"
  [[ -z "${link_found}" ]] || { echo "The bundle holds a symlink: ${link_found#"${TMP}/x/"}" >&2; exit 1; }

  # Build the folder next to the destination, so the swap is a rename on one file system.
  mkdir -p "$(dirname -- "${TARGET}")"
  STAGE="$(mktemp -d "$(dirname -- "${TARGET}")/.get-example.XXXXXX")"
  cp -a "${TMP}/x/${root}" "${STAGE}/new" \
    || { echo "Could not copy the example next to ${dest}. It is unchanged." >&2; exit 1; }
  if ! swap_in; then
    # When restore fails here, cleanup tries once more and says where the old folder is.
    if [[ ! -e "${STAGE}/old" ]] || restore; then
      echo "Could not replace ${dest}. It is unchanged." >&2
    else
      echo "Could not replace ${dest}." >&2
    fi
    exit 1
  fi
  echo ""
  echo "Created ${dest} (bundle $(basename "${url}"))."
  printf 'Next: cd %q; APP_DIR=%q\n' "${dest}" "${ex#"${root}"/}"
  echo "      then follow \${APP_DIR}/README.md and run its commands from ${dest}, as from prebuilt-apps/."
}

TMP="" STAGE="" TARGET=""
# Bash also runs it on SIGINT, SIGTERM and SIGHUP, so an interrupted swap is put back.
trap cleanup EXIT
main "$@"
