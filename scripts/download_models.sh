#!/usr/bin/env bash
# Download model artifacts required by the enabled apps test scope.
#
# Default behavior reads examples/*/*/tests/test-scope.yaml and downloads models for
# enabled Python and C++ e2e tests. Explicit positional arguments are treated as
# modelzoo names for manual one-off downloads.
#
# Usage:
#   ./scripts/download_models.sh
#   ./scripts/download_models.sh --language python --language cpp
#   ./scripts/download_models.sh --scope-file examples --kind e2e
#   ./scripts/download_models.sh resnet_50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
SIMA_CLI_BIN="${SIMA_CLI_BIN:-}"
SCOPE_FILE="${ROOT}/examples"
KIND="e2e"
LANGUAGES=()
POSITIONAL_MODELS=()
SCOPE_PYTHON="${TEST_SCOPE_PYTHON_BIN:-${PYTHON_TEST_BIN:-python3}}"
VERSION_PYTHON="${PYTHON_TEST_BIN:-python3}"
export SIMA_CLI_CHECK_FOR_UPDATE="${SIMA_CLI_CHECK_FOR_UPDATE:-0}"

# Model assets follow the published Model Zoo release, which can lag the platform
# release. modelzoo-version selects it; platform-version is the fallback for
# manifests written before the two versions were separated.
detect_modelzoo_version() {
    "$VERSION_PYTHON" - "$ROOT/deps/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    manifest = json.loads(path.read_text(encoding="utf-8"))
except FileNotFoundError:
    print(
        f"ERROR: Missing manifest: {path}. Set NEAT_APPS_MODELZOO_VERSION to override.",
        file=sys.stderr,
    )
    raise SystemExit(1)

version = manifest.get("modelzoo-version") or manifest.get("platform-version")
if not isinstance(version, str) or not version.strip():
    print(
        f"ERROR: {path} must define modelzoo-version or platform-version as a non-empty string.",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(version.strip())
PY
}

MODELZOO_VERSION="${NEAT_APPS_MODELZOO_VERSION:-}"

ensure_modelzoo_version() {
    if [[ -z "$MODELZOO_VERSION" ]]; then
        MODELZOO_VERSION="$(detect_modelzoo_version)"
    fi
}

usage() {
    cat <<'EOF'
Usage:
  scripts/download_models.sh [options]
  scripts/download_models.sh <modelzoo-name> [...]

Options:
  --scope-file <path>   Test scope source file or directory (default: examples)
  --kind <kind>         Scope kind to resolve (default: e2e)
  --language <lang>     Scope language to include; repeatable (python, cpp)
  NEAT_APPS_MODELZOO_VERSION can override the Model Zoo version.
  -h, --help            Show help
EOF
}

resolve_sima_cli_bin() {
    if [[ -n "${SIMA_CLI_BIN:-}" ]]; then
        if [[ -x "${SIMA_CLI_BIN}" ]]; then
            printf '%s\n' "${SIMA_CLI_BIN}"
            return 0
        fi
        if command -v "${SIMA_CLI_BIN}" >/dev/null 2>&1; then
            command -v "${SIMA_CLI_BIN}"
            return 0
        fi
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

ensure_sima_cli_bin() {
    if SIMA_CLI_BIN="$(resolve_sima_cli_bin)"; then
        return 0
    fi
    echo "sima-cli was not found. Set SIMA_CLI_BIN or install sima-cli." >&2
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scope-file)
            SCOPE_FILE="$2"
            shift 2
            ;;
        --kind)
            KIND="$2"
            shift 2
            ;;
        --language)
            LANGUAGES+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                POSITIONAL_MODELS+=("$1")
                shift
            done
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            POSITIONAL_MODELS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "$MODELS_DIR"

LOCK_DIR="${MODELS_DIR}/.download.lock"
acquire_lock() {
    local waited=0
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        if [[ "$waited" -eq 0 ]]; then
            echo "Waiting for model download lock: $LOCK_DIR"
        fi
        waited=$((waited + 2))
        sleep 2
    done
    trap 'rm -rf "$LOCK_DIR"' EXIT
}

model_exists() {
    local model_name="$1"
    local expected_file="${2:-}"
    if [[ -n "$expected_file" && -f "${MODELS_DIR}/${expected_file}" ]]; then
        return 0
    fi

    local base="${model_name##*/}"
    local pattern
    for pattern in \
        "${base}_mpk.tar.gz" \
        "${base}-mpk.tar.gz" \
        "${base}"*_mpk.tar.gz \
        "${base}"*-mpk.tar.gz \
        "${base}.tar.gz"; do
        compgen -G "${MODELS_DIR}/${pattern}" >/dev/null && return 0
    done

    local alt="${base//_/-}"
    for pattern in "${alt}_mpk.tar.gz" "${alt}-mpk.tar.gz" "${alt}"*_mpk.tar.gz "${alt}"*-mpk.tar.gz; do
        compgen -G "${MODELS_DIR}/${pattern}" >/dev/null && return 0
    done
    return 1
}

download_modelzoo_model() {
    local model_id="$1"
    local model_name="$2"
    local expected_file="$3"
    if model_exists "$model_name" "$expected_file"; then
        echo "[skip] $model_id already exists"
        return 0
    fi
    ensure_sima_cli_bin
    ensure_modelzoo_version

    local before
    before=$(find "$MODELS_DIR" -maxdepth 1 -type f -name '*.tar.gz' | wc -l)
    echo "[download] $model_id (modelzoo: $model_name, modelzoo-version: $MODELZOO_VERSION)"
    (cd "$MODELS_DIR" && "$SIMA_CLI_BIN" modelzoo -v "$MODELZOO_VERSION" get "$model_name")

    local after
    after=$(find "$MODELS_DIR" -maxdepth 1 -type f -name '*.tar.gz' | wc -l)
    if [[ "$after" -gt "$before" || -z "$expected_file" || -f "${MODELS_DIR}/${expected_file}" ]]; then
        echo "[ok] $model_id"
        return 0
    fi

    echo "[error] $model_id: no expected model artifact after download" >&2
    return 1
}

download_url_model() {
    local model_id="$1"
    local url="$2"
    local expected_file="$3"
    local tmpdir
    local downloaded_files=()
    if [[ "$url" == *"{modelzoo_version}"* ]]; then
        ensure_modelzoo_version
        url="${url//\{modelzoo_version\}/$MODELZOO_VERSION}"
    fi

    # An unsubstituted placeholder would otherwise reach sima-cli verbatim and
    # fail as an opaque 404 rather than naming the scope file mistake.
    if [[ "$url" == *"{"*"}"* ]]; then
        echo "[error] $model_id has an unsupported URL placeholder: $url" >&2
        return 1
    fi

    if model_exists "$model_id" "$expected_file"; then
        echo "[skip] $model_id already exists"
        return 0
    fi
    ensure_sima_cli_bin

    tmpdir="$(mktemp -d)"
    echo "[download] $model_id (url)"
    if ! (cd "$tmpdir" && "$SIMA_CLI_BIN" download "$url"); then
        echo "[warn] direct URL download failed, trying legacy syntax" >&2
        (cd "$tmpdir" && "$SIMA_CLI_BIN" download url "$url")
    fi

    while IFS= read -r file; do
        downloaded_files+=("$file")
    done < <(find "$tmpdir" -maxdepth 1 -type f | sort)
    if [[ "${#downloaded_files[@]}" -eq 0 ]]; then
        rm -rf "$tmpdir"
        echo "[error] $model_id: no file downloaded from $url" >&2
        return 1
    fi

    local file
    for file in "${downloaded_files[@]}"; do
        mv "$file" "$MODELS_DIR/"
    done
    rm -rf "$tmpdir"

    if model_exists "$model_id" "$expected_file"; then
        echo "[ok] $model_id"
        return 0
    fi

    echo "[error] $model_id downloaded but filename did not match expected patterns" >&2
    return 1
}

download_huggingface_model() {
    local model_id="$1"
    local repo="$2"
    local path="$3"
    echo "[error] $model_id uses Hugging Face source, but Hugging Face downloads are not implemented yet." >&2
    echo "        repo=${repo:-TBD} path=${path:-TBD}" >&2
    return 1
}

download_vulcan_model() {
    local model_id="$1"
    local target="$2"
    local environment="$3"
    local variant="$4"
    local expected_file="$5"
    local tmpdir installed_path
    local environment_args=()

    if model_exists "$model_id" "$expected_file"; then
        echo "[skip] $model_id already exists"
        return 0
    fi
    ensure_sima_cli_bin

    case "$environment" in
        development) environment_args=(--dev) ;;
        staging) environment_args=(--stg) ;;
        production) environment_args=(--prd) ;;
        *)
            echo "[error] $model_id has unsupported Vulcan environment: $environment" >&2
            return 1
            ;;
    esac

    tmpdir="$(mktemp -d)"
    echo "[download] $model_id (vulcan: $target, environment: $environment)"
    if ! "$SIMA_CLI_BIN" neat install "${environment_args[@]}" \
        --install-dir "$tmpdir" "$target"; then
        rm -rf "$tmpdir"
        echo "[error] $model_id: Vulcan install failed" >&2
        return 1
    fi

    # sima-cli verifies the variant-relative metadata resource, extracts MPKs,
    # and preserves the original archive at the install root. Older clients
    # may leave a non-extracted resource at its metadata-relative path.
    installed_path="$tmpdir/$expected_file"
    if [[ ! -f "$installed_path" ]]; then
        installed_path="$tmpdir/$variant/$expected_file"
    fi
    if [[ ! -f "$installed_path" ]]; then
        echo "[error] $model_id: Vulcan package did not install $variant/$expected_file" >&2
        find "$tmpdir" -maxdepth 3 -type f -print >&2
        rm -rf "$tmpdir"
        return 1
    fi

    if ! mv "$installed_path" "$MODELS_DIR/$expected_file"; then
        rm -rf "$tmpdir"
        echo "[error] $model_id: failed to install $expected_file in $MODELS_DIR" >&2
        return 1
    fi
    rm -rf "$tmpdir"
    echo "[ok] $model_id"
}

split_tsv_row() {
    local row="$1"
    local -n out="$2"
    local normalized="${row//$'\t'/$'\x1f'}"
    IFS=$'\x1f' read -r -a out <<<"$normalized"
}

download_scoped_models() {
    if [[ "${#LANGUAGES[@]}" -eq 0 ]]; then
        LANGUAGES=(python cpp)
    fi

    local args=(--scope-file "$SCOPE_FILE" models --kind "$KIND")
    local lang
    for lang in "${LANGUAGES[@]}"; do
        args+=(--language "$lang")
    done

    local scope_output
    if ! scope_output="$("$SCOPE_PYTHON" "$ROOT/tests/utils/test_scope.py" "${args[@]}")"; then
        echo "[error] failed to resolve scoped models from ${SCOPE_FILE}" >&2
        return 1
    fi

    local rows=()
    if [[ -n "$scope_output" ]]; then
        mapfile -t rows <<<"$scope_output"
    fi
    if [[ "${#rows[@]}" -eq 0 ]]; then
        echo "No scoped models are required for kind=${KIND} language(s): ${LANGUAGES[*]}"
        return 0
    fi

    echo "Models directory: $MODELS_DIR"
    echo "Scope source: $SCOPE_FILE"
    echo "Models to download:"
    printf '  %s\n' "${rows[@]%%$'\t'*}"
    echo ""

    acquire_lock

    local failed=0
    local row model_id source name url expected_file repo path target environment variant fields
    for row in "${rows[@]}"; do
        fields=()
        split_tsv_row "$row" fields
        model_id="${fields[0]:-}"
        source="${fields[1]:-}"
        name="${fields[2]:-}"
        url="${fields[3]:-}"
        expected_file="${fields[4]:-}"
        repo="${fields[5]:-}"
        path="${fields[6]:-}"
        target="${fields[7]:-}"
        environment="${fields[8]:-}"
        variant="${fields[9]:-}"
        case "$source" in
            modelzoo)
                download_modelzoo_model "$model_id" "${name:-$model_id}" "$expected_file" || failed=1
                ;;
            url)
                download_url_model "$model_id" "$url" "$expected_file" || failed=1
                ;;
            huggingface)
                download_huggingface_model "$model_id" "$repo" "$path" || failed=1
                ;;
            vulcan)
                download_vulcan_model \
                    "$model_id" "$target" "$environment" "$variant" "$expected_file" || failed=1
                ;;
            *)
                echo "[error] $model_id has unsupported source: $source" >&2
                failed=1
                ;;
        esac
    done
    return "$failed"
}

download_positional_models() {
    echo "Models directory: $MODELS_DIR"
    echo "Modelzoo models to download: ${POSITIONAL_MODELS[*]}"
    echo ""

    acquire_lock

    local failed=0
    local model
    for model in "${POSITIONAL_MODELS[@]}"; do
        download_modelzoo_model "$model" "$model" "" || failed=1
    done
    return "$failed"
}

if [[ "${#POSITIONAL_MODELS[@]}" -gt 0 ]]; then
    download_positional_models
else
    download_scoped_models
fi
