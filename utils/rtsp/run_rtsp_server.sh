#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_rtsp_server.sh [--host-port <port>] [--detach]

Defaults:
  host port: ${RTSP_HOST_PORT:-8555}
  image:     bluenviron/mediamtx
EOF
}

NAME="rtsp_server"
HOST_PORT="${RTSP_HOST_PORT:-8555}"
CONTAINER_PORT="8554"
IMAGE="bluenviron/mediamtx"
DETACH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host-port) HOST_PORT="$2"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if docker ps -a --format '{{.Names}}' | grep -Fxq "${NAME}"; then
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
fi

RUN_ARGS=(--name "${NAME}" --rm -e "MTX_PROTOCOLS=tcp" -p "${HOST_PORT}:${CONTAINER_PORT}")
if [[ "${DETACH}" -eq 1 ]]; then
  RUN_ARGS=(-d "${RUN_ARGS[@]}")
fi

echo "[run_rtsp_server] host_port=${HOST_PORT}"
exec docker run "${RUN_ARGS[@]}" "${IMAGE}"
