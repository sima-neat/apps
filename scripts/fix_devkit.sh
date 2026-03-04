#!/usr/bin/env bash
set -euo pipefail

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

echo "[fix-devkit] stopping stale app/test processes"
pkill -f "/data/workspace/sima-neat/apps/examples/.*/main.py" || true
pkill -f "/data/workspace/sima-neat/apps/tests/pytest.ini" || true

echo "[fix-devkit] stopping core services"
run_root systemctl stop simaai-pipeline-manager.service || true
run_root systemctl stop simaai-appcomplex.service || true
run_root systemctl stop rctd.service || true
sleep 1

echo "[fix-devkit] resetting remoteproc"
run_root sh -c 'echo stop > /sys/class/remoteproc/remoteproc0/state'
run_root sh -c 'echo stop > /sys/class/remoteproc/remoteproc1/state'
run_root sh -c 'echo start > /sys/class/remoteproc/remoteproc1/state'
run_root sh -c 'echo start > /sys/class/remoteproc/remoteproc0/state'

echo "[fix-devkit] reinitializing MLA memory"
if ! run_root /usr/bin/init_mla_memory.sh; then
  echo "[fix-devkit] MLA init failed, retrying after one more remoteproc reset"
  run_root sh -c 'echo stop > /sys/class/remoteproc/remoteproc0/state'
  run_root sh -c 'echo stop > /sys/class/remoteproc/remoteproc1/state'
  run_root sh -c 'echo start > /sys/class/remoteproc/remoteproc1/state'
  run_root sh -c 'echo start > /sys/class/remoteproc/remoteproc0/state'
  run_root /usr/bin/init_mla_memory.sh
fi

echo "[fix-devkit] restarting core services"
run_root systemctl start simaai-appcomplex.service
run_root systemctl start simaai-pipeline-manager.service
run_root systemctl start rctd.service

sleep 2

appcomplex_state="$(systemctl is-active simaai-appcomplex.service || true)"
pipeline_state="$(systemctl is-active simaai-pipeline-manager.service || true)"
rctd_state="$(systemctl is-active rctd.service || true)"

echo "[fix-devkit] simaai-appcomplex.service: ${appcomplex_state}"
echo "[fix-devkit] simaai-pipeline-manager.service: ${pipeline_state}"
echo "[fix-devkit] rctd.service: ${rctd_state}"

if [[ "${appcomplex_state}" == "active" && "${pipeline_state}" == "active" ]]; then
  echo "[fix-devkit] RESULT: SUCCESS"
  exit 0
fi

echo "[fix-devkit] RESULT: FAILED"
exit 1
