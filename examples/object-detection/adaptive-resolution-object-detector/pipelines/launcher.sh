#!/usr/bin/env bash
# Start/stop the pipeline chooser. Run ON the DevKit.
#   ssh sima@192.168.135.72 'bash <bundle>/launcher.sh start'
#   then open http://192.168.135.72:8080
#
# Kill is by the server's exact PATH, like the per-pipeline ui.sh: the launching
# shell's argv carries launcher.sh, not launcher.py, so this never signals itself.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8080
SERVER="${DIR}/launcher.py"
LOG="${DIR}/logs/launcher_${PORT}.log"
mkdir -p "$(dirname "${LOG}")"

kill_server() {
  pkill -9 -f "${SERVER}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null && { exec 3>&- 3<&-; sleep 0.3; } || return 0
  done
}

case "${1:-start}" in
  start)
    kill_server
    setsid nohup python3 "${SERVER}" > "${LOG}" 2>&1 < /dev/null &
    sleep 2
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
      exec 3>&- 3<&-
      echo "chooser running: http://$(hostname -I | awk '{print $1}'):${PORT}"
    else
      echo "failed to start; last log:"; tail -n 15 "${LOG}"
    fi
    ;;
  stop)   kill_server; echo "stopped :${PORT}" ;;
  status)
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then exec 3>&- 3<&-; echo "running on :${PORT}"
    else echo "not running"; fi ;;
  *) echo "usage: launcher.sh [start|stop|status]" ;;
esac
