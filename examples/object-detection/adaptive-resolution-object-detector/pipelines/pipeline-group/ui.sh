#!/usr/bin/env bash
# Start/stop this pipeline's UI server. Run ON the DevKit.
#
#   ssh sima@192.168.135.72 'bash <bundle>/pipeline-<mode>/ui.sh start'
#   then open http://192.168.135.72:<PORT>   (8090 scale, 8091 live)
#
# Kill is by the exact server PATH (not the port): fuser proved unreliable here,
# and the path is specific per pipeline (pipeline-scale/ vs pipeline-live/), so
# starting one never touches the other. The launching shell's argv contains the
# ui.sh path, not ui_server.py, so this never signals its own shell.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=$(grep -qE '^PIPELINE = "live"' "${DIR}/pipeline.py" && echo 8091 \
     || grep -qE '^PIPELINE = "group"' "${DIR}/pipeline.py" && echo 8092 || echo 8090)
SERVER="${DIR}/ui_server.py"
LOG="${DIR}/../logs/ui_server_${PORT}.log"
mkdir -p "$(dirname "${LOG}")"

kill_server() {
  # -f matches the full command line; the pattern is this pipeline's server path.
  pkill -9 -f "${SERVER}" 2>/dev/null || true
  # Wait until the port is actually released before returning.
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
      echo "UI running: http://$(hostname -I | awk '{print $1}'):${PORT}"
    else
      echo "failed to start; last log:"; tail -n 15 "${LOG}"
    fi
    ;;
  stop)
    kill_server; echo "stopped :${PORT}"
    ;;
  status)
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then exec 3>&- 3<&-; echo "running on :${PORT}"
    else echo "not running"; fi
    ;;
  *)
    echo "usage: ui.sh [start|stop|status]"
    ;;
esac
