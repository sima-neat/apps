#!/usr/bin/env bash

# Copyright 2026 SiMa Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS_ROOT="$(cd "${EXAMPLE_DIR}/../../.." && pwd)"
APP_NAME="pcie-high-density-multi-stream-object-detector"

CONFIG_PATH=""
CARD_HOST="${PCIE_CARD_HOST:-10.0.0.2}"
CARD_USER="${PCIE_CARD_USER:-sima}"
CARD_PORT="${PCIE_CARD_SSH_PORT:-22}"
SSH_IDENTITY="${PCIE_CARD_SSH_KEY:-${HOME}/.ssh/sima_neat_pcie_ed25519}"
CARD_BINARY="${PCIE_CARD_BINARY:-}"
HOST_BINARY="${PCIE_HOST_BINARY:-${APPS_ROOT}/build-host-pcie/${APP_NAME}-host}"
REMOTE_DIR="${PCIE_REMOTE_RUN_DIR:-}"
READINESS_TIMEOUT="${PCIE_CARD_READINESS_TIMEOUT:-120}"
SHUTDOWN_TIMEOUT="${PCIE_SHUTDOWN_TIMEOUT:-30}"
ALLOW_DIRTY_CARD="${PCIE_ALLOW_DIRTY_CARD:-0}"

HOST_PID=""
CARD_STARTED=0
CLEANUP_STARTED=0

usage() {
  cat <<EOF
Usage: ${0##*/} --config PATH [options]

Start the card application, wait for it to become ready, and then start the
host application. Press Ctrl+C to stop the host first and the card second.

Options:
  --config PATH               Local YAML configuration (required)
  --card-host HOST            Card SSH address (default: ${CARD_HOST})
  --card-user USER            Card SSH user (default: ${CARD_USER})
  --card-port PORT            Card SSH port (default: ${CARD_PORT})
  --identity PATH             SSH private key (default: ${SSH_IDENTITY})
  --card-binary PATH          Card-side executable
  --host-binary PATH          Locally built host executable
  --remote-dir PATH           Card runtime directory
  --readiness-timeout SEC     Card startup timeout (default: ${READINESS_TIMEOUT})
  --shutdown-timeout SEC      Per-process graceful shutdown timeout
                               (default: ${SHUTDOWN_TIMEOUT})
  --allow-dirty-card          Start even if the previous card application had
                               to be killed and the card was not rebooted since
  -h, --help                  Show this help

Environment variables with equivalent defaults:
  PCIE_CARD_HOST, PCIE_CARD_USER, PCIE_CARD_SSH_PORT, PCIE_CARD_SSH_KEY,
  PCIE_CARD_BINARY, PCIE_HOST_BINARY, PCIE_REMOTE_RUN_DIR,
  PCIE_CARD_READINESS_TIMEOUT, PCIE_SHUTDOWN_TIMEOUT, PCIE_ALLOW_DIRTY_CARD

A card application that ignores SIGINT is killed with SIGTERM/SIGKILL. That can
leave PCIe endpoint queue state behind on the card, and later sessions may then
stall or crash the card. The launcher records this in the card runtime
directory and refuses to start again until the card has been rebooted.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "${value}" ]] || die "${option} requires a value"
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

shell_quote() {
  printf '%q' "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      require_value "$1" "${2:-}"
      CONFIG_PATH="$2"
      shift 2
      ;;
    --card-host)
      require_value "$1" "${2:-}"
      CARD_HOST="$2"
      shift 2
      ;;
    --card-user)
      require_value "$1" "${2:-}"
      CARD_USER="$2"
      shift 2
      ;;
    --card-port)
      require_value "$1" "${2:-}"
      CARD_PORT="$2"
      shift 2
      ;;
    --identity)
      require_value "$1" "${2:-}"
      SSH_IDENTITY="$2"
      shift 2
      ;;
    --card-binary)
      require_value "$1" "${2:-}"
      CARD_BINARY="$2"
      shift 2
      ;;
    --host-binary)
      require_value "$1" "${2:-}"
      HOST_BINARY="$2"
      shift 2
      ;;
    --remote-dir)
      require_value "$1" "${2:-}"
      REMOTE_DIR="$2"
      shift 2
      ;;
    --readiness-timeout)
      require_value "$1" "${2:-}"
      READINESS_TIMEOUT="$2"
      shift 2
      ;;
    --shutdown-timeout)
      require_value "$1" "${2:-}"
      SHUTDOWN_TIMEOUT="$2"
      shift 2
      ;;
    --allow-dirty-card)
      ALLOW_DIRTY_CARD=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "${CONFIG_PATH}" ]] || {
  usage >&2
  exit 2
}
if [[ -z "${CARD_BINARY}" ]]; then
  CARD_BINARY="/home/${CARD_USER}/prebuilt-apps/examples/object-detection/${APP_NAME}/src/cpp/pre-built/${APP_NAME}"
fi
[[ -f "${CONFIG_PATH}" ]] || die "config does not exist: ${CONFIG_PATH}"
[[ -x "${HOST_BINARY}" ]] || die "host binary is not executable: ${HOST_BINARY}"
[[ -r "${SSH_IDENTITY}" ]] || die "SSH identity is not readable: ${SSH_IDENTITY}"
is_positive_integer "${CARD_PORT}" || die "--card-port must be a positive integer"
is_positive_integer "${READINESS_TIMEOUT}" || die "--readiness-timeout must be a positive integer"
is_positive_integer "${SHUTDOWN_TIMEOUT}" || die "--shutdown-timeout must be a positive integer"
command -v ssh >/dev/null 2>&1 || die "ssh is required"
command -v scp >/dev/null 2>&1 || die "scp is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required"

if [[ -z "${REMOTE_DIR}" ]]; then
  REMOTE_DIR="/home/${CARD_USER}/tmp/pcie-high-density"
fi
if [[ "${REMOTE_DIR}" =~ [[:space:]] || "${CARD_BINARY}" =~ [[:space:]] ]]; then
  die "remote paths must not contain whitespace"
fi

CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"
cd "${APPS_ROOT}"
CARD_TARGET="${CARD_USER}@${CARD_HOST}"
REMOTE_CONFIG="${REMOTE_DIR}/config.yaml"
REMOTE_PID_FILE="${REMOTE_DIR}/card.pid"
REMOTE_LOG="${REMOTE_DIR}/card.log"
REMOTE_DIRTY_FILE="${REMOTE_DIR}/card.dirty"

SSH_OPTIONS=(
  -i "${SSH_IDENTITY}"
  -p "${CARD_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=10
)
SCP_OPTIONS=(
  -i "${SSH_IDENTITY}"
  -P "${CARD_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=10
)

CARD_BINARY_Q="$(shell_quote "${CARD_BINARY}")"
REMOTE_DIR_Q="$(shell_quote "${REMOTE_DIR}")"
REMOTE_CONFIG_Q="$(shell_quote "${REMOTE_CONFIG}")"
REMOTE_PID_FILE_Q="$(shell_quote "${REMOTE_PID_FILE}")"
REMOTE_LOG_Q="$(shell_quote "${REMOTE_LOG}")"
REMOTE_DIRTY_FILE_Q="$(shell_quote "${REMOTE_DIRTY_FILE}")"

remote_exec() {
  ssh "${SSH_OPTIONS[@]}" "${CARD_TARGET}" "$1"
}

wait_for_local_exit() {
  local pid="$1"
  local timeout="$2"
  local deadline=$((SECONDS + timeout))
  while kill -0 "${pid}" 2>/dev/null && [[ "${SECONDS}" -lt "${deadline}" ]]; do
    sleep 1
  done
  ! kill -0 "${pid}" 2>/dev/null
}

stop_host() {
  [[ -n "${HOST_PID}" ]] || return 0
  if ! kill -0 "${HOST_PID}" 2>/dev/null; then
    wait "${HOST_PID}" 2>/dev/null || true
    HOST_PID=""
    return 0
  fi

  echo "Stopping host application..."
  kill -INT -- "-${HOST_PID}" 2>/dev/null || kill -INT "${HOST_PID}" 2>/dev/null || true
  if ! wait_for_local_exit "${HOST_PID}" "${SHUTDOWN_TIMEOUT}"; then
    echo "Host did not stop after SIGINT; sending SIGTERM." >&2
    kill -TERM -- "-${HOST_PID}" 2>/dev/null || kill -TERM "${HOST_PID}" 2>/dev/null || true
    if ! wait_for_local_exit "${HOST_PID}" 3; then
      echo "Host did not stop after SIGTERM; sending SIGKILL." >&2
      kill -KILL -- "-${HOST_PID}" 2>/dev/null || kill -KILL "${HOST_PID}" 2>/dev/null || true
    fi
  fi
  wait "${HOST_PID}" 2>/dev/null || true
  HOST_PID=""
}

stop_card() {
  [[ "${CARD_STARTED}" -eq 1 ]] || return 0
  echo "Stopping card application..."
  remote_exec "
set +e
if [[ -f ${REMOTE_PID_FILE_Q} ]]; then
  pid=\$(cat ${REMOTE_PID_FILE_Q} 2>/dev/null)
  if [[ \"\${pid}\" =~ ^[1-9][0-9]*$ ]] && kill -0 \"\${pid}\" 2>/dev/null; then
    cmdline=\$(tr '\\0' ' ' <\"/proc/\${pid}/cmdline\" 2>/dev/null)
    case \"\${cmdline}\" in
      *${APP_NAME}*)
        kill -INT \"\${pid}\" 2>/dev/null
        remaining=${SHUTDOWN_TIMEOUT}
        while kill -0 \"\${pid}\" 2>/dev/null && [[ \"\${remaining}\" -gt 0 ]]; do
          sleep 1
          remaining=\$((remaining - 1))
        done
        if kill -0 \"\${pid}\" 2>/dev/null; then
          echo 'Card did not stop after SIGINT; sending SIGTERM.' >&2
          echo \"\$(date '+%Y-%m-%d %H:%M:%S') card application PID \${pid} did not stop after SIGINT and was killed\" >${REMOTE_DIRTY_FILE_Q}
          echo 'The card was left in an unclean state; reboot it before starting a new session.' >&2
          kill -TERM \"\${pid}\" 2>/dev/null
          sleep 3
        fi
        if kill -0 \"\${pid}\" 2>/dev/null; then
          echo 'Card did not stop after SIGTERM; sending SIGKILL.' >&2
          kill -KILL \"\${pid}\" 2>/dev/null
        fi
        ;;
      *)
        echo \"Refusing to signal PID \${pid}: it is not ${APP_NAME}.\" >&2
        exit 1
        ;;
    esac
  fi
fi
rm -f ${REMOTE_PID_FILE_Q} ${REMOTE_CONFIG_Q}
"
  CARD_STARTED=0
}

cleanup() {
  local status="$1"
  local card_session_owned="${CARD_STARTED}"
  if [[ "${CLEANUP_STARTED}" -eq 1 ]]; then
    return
  fi
  CLEANUP_STARTED=1
  trap - EXIT INT TERM
  set +e
  stop_host
  stop_card
  if [[ "${card_session_owned}" -eq 1 ]]; then
    echo "Card log retained at ${CARD_TARGET}:${REMOTE_LOG}"
  fi
  return "${status}"
}

on_exit() {
  local status=$?
  cleanup "${status}" || true
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Validating host configuration..."
"${HOST_BINARY}" --config "${CONFIG_PATH}" --validate-config-only

echo "Checking ${CARD_TARGET}..."
remote_exec "
set -eu
command -v nohup >/dev/null
command -v setsid >/dev/null
command -v stdbuf >/dev/null
test -x ${CARD_BINARY_Q} || {
  echo 'Card binary is missing or not executable: ${CARD_BINARY}' >&2
  exit 1
}
mkdir -p ${REMOTE_DIR_Q}
if [[ -f ${REMOTE_DIRTY_FILE_Q} ]]; then
  marked=\$(stat -c %Y ${REMOTE_DIRTY_FILE_Q} 2>/dev/null || echo 0)
  up=\$(cut -d. -f1 /proc/uptime)
  booted=\$(( \$(date +%s) - \${up:-0} ))
  if [[ \"\${booted}\" -gt \"\${marked}\" ]]; then
    rm -f ${REMOTE_DIRTY_FILE_Q}
  elif [[ ${ALLOW_DIRTY_CARD} -eq 1 ]]; then
    echo 'WARNING: starting on a card that was not rebooted after an unclean stop (--allow-dirty-card).' >&2
    rm -f ${REMOTE_DIRTY_FILE_Q}
  else
    echo \"The previous session did not stop cleanly: \$(cat ${REMOTE_DIRTY_FILE_Q})\" >&2
    echo 'Reboot the card before starting a new session (or pass --allow-dirty-card to override).' >&2
    exit 1
  fi
fi
if [[ -f ${REMOTE_PID_FILE_Q} ]]; then
  pid=\$(cat ${REMOTE_PID_FILE_Q} 2>/dev/null || true)
  if [[ \"\${pid}\" =~ ^[1-9][0-9]*$ ]] && kill -0 \"\${pid}\" 2>/dev/null; then
    echo \"Card application is already running with PID \${pid}.\" >&2
    exit 1
  fi
  rm -f ${REMOTE_PID_FILE_Q}
fi
"

echo "Uploading configuration to ${CARD_TARGET}:${REMOTE_CONFIG}..."
scp "${SCP_OPTIONS[@]}" "${CONFIG_PATH}" "${CARD_TARGET}:${REMOTE_CONFIG}"
CARD_STARTED=1

echo "Validating card configuration..."
remote_exec "${CARD_BINARY_Q} --config ${REMOTE_CONFIG_Q} --validate-config-only"

echo "Starting card application..."
remote_exec "
set -eu
: >${REMOTE_LOG_Q}
nohup setsid stdbuf -oL -eL ${CARD_BINARY_Q} --config ${REMOTE_CONFIG_Q} \
  >${REMOTE_LOG_Q} 2>&1 </dev/null &
pid=\$!
echo \"\${pid}\" >${REMOTE_PID_FILE_Q}
sleep 1
if ! kill -0 \"\${pid}\" 2>/dev/null; then
  echo 'Card application exited during startup.' >&2
  tail -n 50 ${REMOTE_LOG_Q} >&2
  rm -f ${REMOTE_PID_FILE_Q}
  exit 1
fi
"

echo "Waiting up to ${READINESS_TIMEOUT}s for the card graph..."
if ! remote_exec "
remaining=${READINESS_TIMEOUT}
while [[ \"\${remaining}\" -gt 0 ]]; do
  pid=\$(cat ${REMOTE_PID_FILE_Q} 2>/dev/null || true)
  if ! [[ \"\${pid}\" =~ ^[1-9][0-9]*$ ]] || ! kill -0 \"\${pid}\" 2>/dev/null; then
    echo 'Card application exited before becoming ready.' >&2
    tail -n 50 ${REMOTE_LOG_Q} >&2
    exit 1
  fi
  if grep -Fq 'Graph ready' ${REMOTE_LOG_Q}; then
    exit 0
  fi
  sleep 1
  remaining=\$((remaining - 1))
done
echo 'Timed out waiting for the card graph.' >&2
tail -n 50 ${REMOTE_LOG_Q} >&2
exit 1
"; then
  die "card application did not become ready"
fi

echo "Card graph is ready. Starting host application..."
echo "Press Ctrl+C to stop the host and card applications."
setsid "${HOST_BINARY}" --config "${CONFIG_PATH}" &
HOST_PID=$!

set +e
wait "${HOST_PID}"
HOST_STATUS=$?
set -e
HOST_PID=""
exit "${HOST_STATUS}"
