#!/usr/bin/env bash
# Download all models required by the examples into ./assets/models.
# Parses the "Model" field from each example's README.md metadata table.
# Skips models that are already downloaded.
#
# Usage:
#   ./scripts/download_models.sh          # download all
#   ./scripts/download_models.sh resnet50  # download specific model(s)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT/assets/models}"
EXAMPLES_DIR="$ROOT/examples"
SIMA_CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"

mkdir -p "$MODELS_DIR"

# Extract unique model names from README metadata tables.
get_all_models() {
    grep -rh '| Model |' "$EXAMPLES_DIR" --include="README.md" 2>/dev/null \
        | sed 's/.*| Model | *\([^ |]*\).*/\1/' \
        | sort -u
}

# Check if a model MPK already exists (handles naming variations).
model_exists() {
    local model_name="$1"
    # Try exact match and common variations
    local base="${model_name##*/}"  # strip category/ prefix if any
    for pattern in "${base}_mpk.tar.gz" "${base}-mpk.tar.gz" "${base}.tar.gz"; do
        [ -f "$MODELS_DIR/$pattern" ] && return 0
    done
    # Also check with underscores replaced by different separators
    local alt="${base//_/-}"
    for pattern in "${alt}_mpk.tar.gz" "${alt}-mpk.tar.gz"; do
        [ -f "$MODELS_DIR/$pattern" ] && return 0
    done
    return 1
}

download_model() {
    local model_name="$1"

    if model_exists "$model_name"; then
        echo "[skip] $model_name already exists"
        return 0
    fi

    local before
    before=$(ls "$MODELS_DIR"/*.tar.gz 2>/dev/null | wc -l)

    echo "[download] $model_name"
    (cd "$MODELS_DIR" && "$SIMA_CLI_BIN" modelzoo get "$model_name")

    local after
    after=$(ls "$MODELS_DIR"/*.tar.gz 2>/dev/null | wc -l)

    if [ "$after" -gt "$before" ]; then
        echo "[ok] $model_name"
    else
        echo "[error] $model_name: no new file after download" >&2
        return 1
    fi
}

# If specific models are requested, download those. Otherwise download all.
if [ $# -gt 0 ]; then
    models=("$@")
else
    mapfile -t models < <(get_all_models)
fi

if [ ${#models[@]} -eq 0 ]; then
    echo "No models found in README metadata." >&2
    exit 1
fi

echo "Models directory: $MODELS_DIR"
echo "Models to download: ${models[*]}"
echo ""

failed=0
for model in "${models[@]}"; do
    download_model "$model" || ((failed++))
done

echo ""
ls -lh "$MODELS_DIR"/*.tar.gz 2>/dev/null
echo ""

if [ $failed -gt 0 ]; then
    echo "$failed model(s) failed to download." >&2
    exit 1
fi

echo "All models downloaded to $MODELS_DIR"
