#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python_version_ok() {
    "$1" - "$2" <<'PY'
import sys
minimum = tuple(int(part) for part in sys.argv[1].split("."))
raise SystemExit(0 if sys.version_info[:2] >= minimum else 1)
PY
}

python_version() {
    "$1" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
}

find_python() {
    local min_version="$1"
    local candidate
    local candidates=()
    local pyenv_version
    local pyenv_prefix

    if [[ -n "${MOLE_PYTHON:-}" ]]; then
        candidates+=("${MOLE_PYTHON}")
    fi

    candidates+=(python3.13 python3.12 python3.11 python3)

    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            candidate="$(command -v "${candidate}")"
            if python_version_ok "${candidate}" "${min_version}" >/dev/null 2>&1; then
                echo "${candidate}"
                return 0
            fi
        elif [[ -x "${candidate}" ]] && python_version_ok "${candidate}" "${min_version}" >/dev/null 2>&1; then
            echo "${candidate}"
            return 0
        fi
    done

    if command -v pyenv >/dev/null 2>&1; then
        while IFS= read -r pyenv_version; do
            pyenv_prefix="$(pyenv prefix "${pyenv_version}" 2>/dev/null || true)"
            candidate="${pyenv_prefix}/bin/python"
            if [[ -x "${candidate}" ]] && python_version_ok "${candidate}" "${min_version}" >/dev/null 2>&1; then
                echo "${candidate}"
                return 0
            fi
        done < <(pyenv versions --bare 2>/dev/null | sort -Vr)
    fi

    return 1
}

install_ubuntu_python() {
    local min_version="$1"
    local apt_python="python${min_version}"
    local apt_venv="python${min_version}-venv"

    if ! command -v apt-get >/dev/null 2>&1; then
        return 1
    fi

    echo "🐍 Python ${min_version}+ was not found. Attempting to install ${apt_python} and ${apt_venv} with apt."

    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y "${apt_python}" "${apt_venv}"
    else
        apt-get update
        apt-get install -y "${apt_python}" "${apt_venv}"
    fi
}

install_ubuntu_venv() {
    local python_bin="$1"
    local python_major_minor
    local apt_venv

    if ! command -v apt-get >/dev/null 2>&1; then
        return 1
    fi

    python_major_minor="$("${python_bin}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    apt_venv="python${python_major_minor}-venv"

    echo "🐍 Installing missing venv support package: ${apt_venv}"
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y "${apt_venv}"
    else
        apt-get update
        apt-get install -y "${apt_venv}"
    fi
}

MIN_PYTHON_VERSION="3.11"
if ! SYSTEM_PYTHON="$(find_python "${MIN_PYTHON_VERSION}")"; then
    if install_ubuntu_python "${MIN_PYTHON_VERSION}"; then
        SYSTEM_PYTHON="$(find_python "${MIN_PYTHON_VERSION}")" || true
    fi

    if [[ -z "${SYSTEM_PYTHON:-}" ]]; then
        echo "❌ MOLE requires Python ${MIN_PYTHON_VERSION}+."
        echo "   Ubuntu 22.04 defaults to Python 3.10, which is too old for sima_lmm."
        echo "   Install Python 3.11+ and venv support, then re-run this installer."
        echo "   Example: sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv"
        echo "   If using pyenv, run: pyenv global 3.11.7"
        echo "   Or set MOLE_PYTHON=/path/to/python3.11 before running this script."
        exit 1
    fi
fi

VENV_DIR="${HOME}/sima-mole-venv"
echo "🐍 Using Python $(python_version "${SYSTEM_PYTHON}") at ${SYSTEM_PYTHON}"

if [[ -x "${VENV_DIR}/bin/python" ]] && ! python_version_ok "${VENV_DIR}/bin/python" "${MIN_PYTHON_VERSION}"; then
    echo "🔁 Removing existing MOLE virtual environment created with Python $("${VENV_DIR}/bin/python" --version 2>&1)"
    rm -rf "${VENV_DIR}"
fi

echo "🐍 Creating virtual environment at ${VENV_DIR}"
if ! "${SYSTEM_PYTHON}" -m venv "${VENV_DIR}"; then
    if install_ubuntu_venv "${SYSTEM_PYTHON}"; then
        rm -rf "${VENV_DIR}"
        echo "🐍 Retrying virtual environment creation at ${VENV_DIR}"
        if ! "${SYSTEM_PYTHON}" -m venv "${VENV_DIR}"; then
            echo "❌ Failed to create virtual environment with ${SYSTEM_PYTHON} after installing venv support."
            exit 1
        fi
    else
        echo "❌ Failed to create virtual environment with ${SYSTEM_PYTHON}."
        echo "   On Ubuntu, install the matching venv package, for example:"
        echo "   sudo apt-get install -y python3.11-venv"
        exit 1
    fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
PIP_CMD=(python -m pip)

shopt -s nullglob

# Some download clients need `%2B` in HTTP URLs and may persist that encoded
# name on disk; normalize it back to canonical wheel naming for pip.
ENCODED_WHEELS=(./sima_lmm-*%2B*-py3-none-any.whl)
for enc_whl in "${ENCODED_WHEELS[@]}"; do
    decoded_whl="${enc_whl//%2B/+}"
    if [[ "$enc_whl" != "$decoded_whl" ]]; then
        echo "🔁 Normalizing wheel filename ${enc_whl} -> ${decoded_whl}"
        mv -f "$enc_whl" "$decoded_whl"
    fi
done

WHEELS=(./sima_lmm-*-py3-none-any.whl)

if (( ${#WHEELS[@]} == 0 )); then
    echo "❌ No MOLE wheel found in ${SCRIPT_DIR}"
    echo "   Expected: sima_lmm-<ver>-py3-none-any.whl"
    exit 1
fi

for whl in "${WHEELS[@]}"; do
    wheel_with_extras="${whl}[sdk-ext]"
    echo "📦 Installing ${wheel_with_extras} into ${VENV_DIR}"
    INSTALL_ARGS=(install --force-reinstall)
    "${PIP_CMD[@]}" "${INSTALL_ARGS[@]}" "${wheel_with_extras}"
done

echo "✅ MOLE install complete"
echo "🔎 Virtual environment info"
echo "   VIRTUAL_ENV: ${VIRTUAL_ENV:-<unset>}"
echo "   Python: $(python -c 'import sys; print(sys.executable)')"
echo "   Python version: $(python --version 2>&1)"
echo "   Pip: $(python -m pip --version)"
python -m pip show sima-lmm 2>/dev/null || python -m pip show sima_lmm 2>/dev/null || true
