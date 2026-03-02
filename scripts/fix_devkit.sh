#!/usr/bin/env bash
set -u

# Recover common DevKit runtime issues:
# - remoteproc cores stuck
# - MLA shared-memory server/client desync
# - stale test/example processes occupying runtime slots

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "[recovery] need root privileges for: $*" >&2
    return 1
  fi
}

run_step() {
  local label="$1"
  shift
  echo "[recovery] ${label}..."
  run_root "$@"
  local rc=$?
  echo "[recovery] ${label} rc=${rc}"
  return "${rc}"
}

echo "[recovery] starting DevKit recovery sequence"

# Stop stale user-space clients that can hold MLA runtime slots.
run_step "stop stale pytest/example processes" pkill -f "/data/workspace/sima-neat/apps/examples/.*/main.py"
run_step "stop stale pytest runner processes" pkill -f "/data/workspace/sima-neat/apps/tests/pytest.ini"

# Remoteproc recovery (ignore individual step failures so sequence can continue).
run_step "remoteproc0 stop" sh -c 'echo stop > /sys/class/remoteproc/remoteproc0/state'
run_step "remoteproc1 stop" sh -c 'echo stop > /sys/class/remoteproc/remoteproc1/state'
run_step "remoteproc1 start" sh -c 'echo start > /sys/class/remoteproc/remoteproc1/state'
run_step "remoteproc0 start" sh -c 'echo start > /sys/class/remoteproc/remoteproc0/state'
run_step "remoteproc status" sh -c 'for rp in /sys/class/remoteproc/remoteproc0 /sys/class/remoteproc/remoteproc1; do echo "$rp: $(cat "$rp/name") state=$(cat "$rp/state")"; done'

# Re-init MLA memory and bounce core services.
run_step "init_mla_memory" /usr/bin/init_mla_memory.sh
run_step "restart simaai-appcomplex.service" systemctl restart simaai-appcomplex.service
run_step "restart simaai-pipeline-manager.service" systemctl restart simaai-pipeline-manager.service
run_step "restart simaai-vdp-cli.service" systemctl restart simaai-vdp-cli.service
run_step "restart simaai-log.service" systemctl restart simaai-log.service
run_step "restart simaai-rpyc-server.service" systemctl restart simaai-rpyc-server.service
run_step "restart rctd.service" systemctl restart rctd.service

# Report key statuses.
run_step "service status summary" systemctl --no-pager --full status \
  simaai-appcomplex.service \
  simaai-pipeline-manager.service \
  simaai-vdp-cli.service \
  simaai-log.service \
  simaai-rpyc-server.service \
  rctd.service

echo "[recovery] done"
