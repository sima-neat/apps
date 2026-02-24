#!/usr/bin/env bash
# CI helper: install NEAT core SDK using build.sh's install logic.
# This is a thin wrapper for CI pipelines that only need the install step.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT_DIR}/build.sh" --only-install-neat-core
