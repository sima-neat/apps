#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  install_apps_runtime_from_vulcan.sh --target <apps-target> --install-dir <dir> [--env <env>] [--force]

Installs an Apps artifact with sima-cli and prints the installed
neat-apps-runtime directory path.
EOF
}

artifact_env="production"
target=""
install_dir=""
force=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      artifact_env="${2:-}"
      shift 2
      ;;
    --target)
      target="${2:-}"
      shift 2
      ;;
    --install-dir)
      install_dir="${2:-}"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${target}" ]]; then
  echo "--target is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${install_dir}" || "${install_dir}" == "/" ]]; then
  echo "--install-dir must be a non-root directory" >&2
  usage >&2
  exit 1
fi

if ! command -v sima-cli >/dev/null 2>&1; then
  echo "sima-cli not found on test runner." >&2
  exit 1
fi

export SIMA_CLI_AUTO_ACCEPT_UPDATE="${SIMA_CLI_AUTO_ACCEPT_UPDATE:-1}"

rm -rf "${install_dir}"
mkdir -p "${install_dir}"

cmd=(sima-cli neat install --env "${artifact_env}" -d "${install_dir}" "${target}")
if [[ "${force}" == "1" ]]; then
  cmd+=(-f)
fi

echo "Installing ${target} from Vulcan env ${artifact_env}" >&2
"${cmd[@]}" >&2

mapfile -t runtime_dirs < <(find "${install_dir}" -type d -path '*/neat-apps/neat-apps-runtime' | sort)
if [[ "${#runtime_dirs[@]}" -ne 1 ]]; then
  echo "Expected exactly one installed apps runtime, found ${#runtime_dirs[@]}." >&2
  printf '  %s\n' "${runtime_dirs[@]}" >&2 || true
  find "${install_dir}" -maxdepth 4 -mindepth 1 -type d -printf '  %p\n' | sort >&2 || true
  exit 1
fi

printf '%s\n' "${runtime_dirs[0]}"
