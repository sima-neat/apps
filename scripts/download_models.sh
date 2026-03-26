#!/usr/bin/env bash
# Download all models required by the examples into ./assets/models.
# Parses the "Model" field from each example's README.md metadata table.
# The metadata supports either:
#   | Model | modelzoo_name |
#   | Model | model_label [https://host/path/model_mpk.tar.gz] |
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

declare -A MODEL_URLS=()
declare -a ALL_MODELS=()

# Extract unique model labels and optional URLs from README metadata tables.
load_model_specs() {
    python3 - "$EXAMPLES_DIR" <<'PY'
import re
import sys
from pathlib import Path

examples_dir = Path(sys.argv[1])
field_re = re.compile(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|$")
model_re = re.compile(r"^(?P<label>[^\[]+?)(?:\s*\[(?P<url>https?://[^\]]+)\])?$")


def parse_metadata(readme: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    in_table = False
    for line in readme.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("## Metadata"):
            in_table = True
            continue
        if in_table:
            if stripped.startswith("##") or (stripped and not stripped.startswith("|")):
                break
            match = field_re.match(stripped)
            if not match:
                continue
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key in {"Field", "---"}:
                continue
            fields[key] = value
    return fields


seen: dict[str, str] = {}
for readme in sorted(examples_dir.glob("*/*/README.md")):
    raw = parse_metadata(readme).get("Model", "").strip()
    if not raw:
        continue
    match = model_re.fullmatch(raw)
    if not match:
        print(f"warning: invalid Model metadata in {readme}: {raw}", file=sys.stderr)
        continue
    label = match.group("label").strip()
    url = (match.group("url") or "").strip()
    if label in seen and seen[label] != url:
        print(f"warning: conflicting model URLs for {label}; keeping first value", file=sys.stderr)
        continue
    seen[label] = url

for label in sorted(seen):
    print(f"{label}\t{seen[label]}")
PY
}

while IFS=$'\t' read -r model_label model_url; do
    [ -n "$model_label" ] || continue
    MODEL_URLS["$model_label"]="$model_url"
    ALL_MODELS+=("$model_label")
done < <(load_model_specs)

# Check if a model MPK already exists (handles naming variations).
model_exists() {
    local model_name="$1"
    # Try exact match and common variations
    local base="${model_name##*/}"  # strip category/ prefix if any
    for pattern in \
        "${base}_mpk.tar.gz" \
        "${base}-mpk.tar.gz" \
        "${base}"*_mpk.tar.gz \
        "${base}"*-mpk.tar.gz \
        "${base}.tar.gz"; do
        [ -f "$MODELS_DIR/$pattern" ] && return 0
    done
    # Also check with underscores replaced by different separators
    local alt="${base//_/-}"
    for pattern in "${alt}_mpk.tar.gz" "${alt}-mpk.tar.gz" "${alt}"*_mpk.tar.gz "${alt}"*-mpk.tar.gz; do
        [ -f "$MODELS_DIR/$pattern" ] && return 0
    done
    return 1
}

download_model_from_url() {
    local model_name="$1"
    local model_url="$2"
    local tmpdir
    local downloaded_files=()
    tmpdir="$(mktemp -d)"

    echo "[download] $model_name (direct URL)"
    if ! (cd "$tmpdir" && "$SIMA_CLI_BIN" download "$model_url"); then
        echo "[warn] direct URL download failed, trying legacy syntax" >&2
        (cd "$tmpdir" && "$SIMA_CLI_BIN" download url "$model_url")
    fi

    mapfile -t downloaded_files < <(find "$tmpdir" -maxdepth 1 -type f | sort)
    if [ ${#downloaded_files[@]} -eq 0 ]; then
        rm -rf "$tmpdir"
        echo "[error] $model_name: no file downloaded from $model_url" >&2
        return 1
    fi

    for file in "${downloaded_files[@]}"; do
        mv "$file" "$MODELS_DIR/"
    done
    rm -rf "$tmpdir"

    if model_exists "$model_name"; then
        echo "[ok] $model_name"
        return 0
    fi

    echo "[warn] $model_name downloaded from URL but filename did not match expected patterns" >&2
    return 0
}

download_model() {
    local model_name="$1"
    local model_url="${MODEL_URLS[$model_name]:-}"

    if model_exists "$model_name"; then
        echo "[skip] $model_name already exists"
        return 0
    fi

    if [ -n "$model_url" ]; then
        download_model_from_url "$model_name" "$model_url"
        return $?
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
    models=("${ALL_MODELS[@]}")
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
