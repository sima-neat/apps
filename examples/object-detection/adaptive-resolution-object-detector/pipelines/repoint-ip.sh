#!/usr/bin/env bash
# repoint-ip.sh <new-ip>
#
# Run this ON the SDK container (not the DevKit) any time this machine's own
# network changes - wifi <-> ethernet, a different wifi network, anything -
# and both pipelines stop reaching Insight / the DevKit.
#
# It replays, as one script, everything that had to be found and fixed BY
# HAND during the 2026-07-24 network re-point, in the order that actually
# worked:
#
#   1. repo-wide swap of the container's old IP -> new IP: INSIGHT_HOST,
#      INSIGHT_API, INSIGHT_UI in both pipeline.py/ui_server.py, the RTSP
#      "host:" field baked into generated run.yaml, and any saved camera URL
#      in ui-state.json.
#   2. this container's /etc/environment + /root/.devkit-sync.rc -
#      CONTAINER_HOST_IP / NFS_SERVER_HOST_IP. This is what Insight advertises
#      to browsers for WebRTC; get it wrong and ingest looks perfectly
#      healthy (video + metadata both arriving) while the viewer never shows
#      a frame, because every WebRTC peer is handed a dead address.
#   3. restart Insight so it re-reads that IP and drops stale WebRTC peers.
#   4. DevKit /etc/fstab NFS source - the actual cause when /workspace goes
#      empty on the DevKit and every remote command fails with "No such file".
#   5. DevKit devkit-nfs-watchdog.sh - a per-minute timer with the OLD IP
#      hardcoded. This is the one that is easy to miss: fixing fstab alone
#      looks like it works, then the mount flaps every ~60s because the
#      watchdog keeps "fixing" it back to the dead address. This step is why
#      the mount holds.
#   6. restart the pipeline chooser (:8080) and all three pipeline UI
#      servers (:8090/:8091/:8092) on the DevKit. They all run FROM the NFS
#      share, so they are stopped BEFORE the mount is touched - otherwise they
#      wedge in D state on the detached mount, keep holding their ports, and
#      stall the next reboot for minutes.
#   7. if a detector is currently running with saved streams, rebuild it so
#      its RTSP/Insight URLs are current instead of pointed at a dead host.
#   8. verify every piece end-to-end and print a pass/fail summary.
#
# Usage:
#   ./repoint-ip.sh <host-ip> [board-ip]
#
#   ./repoint-ip.sh 192.168.131.68                  # host moved, same board
#   ./repoint-ip.sh 192.168.131.68 192.168.135.72   # fresh clone: both
#
# FIRST RUN ON A NEW MACHINE: pass BOTH. <host-ip> is this SDK container's
# address as the board and your browser see it; <board-ip> is your DevKit.
# The pipelines ship with whatever addresses the previous owner had baked in,
# and both have to move or nothing connects.
#
# Safe to run repeatedly. Each run DISCOVERS the addresses currently written
# into the REPO (it does not need to be told the old ones) and moves
# everything to the new ones, so it does not matter how many networks you
# have hopped through - or whose machine the pipelines came from.

set -uo pipefail

NEW_IP="${1:-}"
NEW_DEVKIT_IP="${2:-}"
if [[ -z "$NEW_IP" ]]; then
  echo "usage: $0 <host-ip> [board-ip]   e.g. $0 192.168.131.68 192.168.135.72" >&2
  exit 2
fi
for a in "$NEW_IP" ${NEW_DEVKIT_IP:+"$NEW_DEVKIT_IP"}; do
  if ! [[ "$a" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "not an IPv4 address: $a" >&2
    exit 2
  fi
done

# The board address is NOT fixed across users - whoever clones this has their
# own DevKit. Precedence: positional arg, then DEVKIT_IP=, then whatever the
# repo currently says (resolved in the discovery step below).
DEVKIT_IP_ENV="${DEVKIT_IP:-}"

# Insight's port as published on the host. The SDK container regenerates
# /etc/environment on (re)start and this HAS changed across image rebuilds
# (27057 -> 9900 on release-2.1-3b4be39). The pipelines hardcode it, so a
# silent change here breaks every DevKit->Insight call with "Connection
# refused" while every other check in this script still passes.
INSIGHT_PORT="${INSIGHT_PORT:-$(grep -oE '^NEAT_INSIGHT_PORT="?[0-9]+' /etc/environment 2>/dev/null | grep -oE '[0-9]+$')}"
INSIGHT_PORT="${INSIGHT_PORT:-9900}"
DEVKIT_USER="${DEVKIT_USER:-sima}"
# This bundle's own location - never a fixed path, so a clone works wherever
# it lands. DEVKIT_ROOT is the same tree as the DEVKIT sees it: the board
# NFS-mounts the container's workspace, so the absolute path matches on both
# sides. If you clone outside the exported tree, set DEVKIT_ROOT= explicitly.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEVKIT_ROOT="${DEVKIT_ROOT:-$REPO_ROOT}"
# The host directory the DevKit NFS-mounts at /workspace. Discovered from the
# board's own fstab so it does not have to be told; override if that is absent.
NFS_EXPORT="${NFS_EXPORT:-}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8)

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
ok()   { printf '  \033[32m\xe2\x9c\x93\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m\xe2\x9c\x97\033[0m %s\n' "$1"; }

# ---------------------------------------------------------------------------
say "discover the current IP"
# ---------------------------------------------------------------------------
# The REPO is the source of truth for what has to be rewritten - NOT this
# machine's environment. On a fresh clone on someone else's SDK container,
# /etc/environment already holds THEIR correct CONTAINER_HOST_IP. Trusting it
# first makes old == new, the repo-wide swap then matches nothing, and every
# file silently keeps pointing at the machine these pipelines were built on.
# So: read what the files actually say, and fall back to the environment only
# if the repo has no answer.
OLD_IP="$(grep -oE 'INSIGHT_HOST = "[0-9.]+"' "$REPO_ROOT/pipeline-scale/pipeline.py" 2>/dev/null | grep -oE '[0-9.]+' || true)"
if [[ -z "$OLD_IP" ]]; then
  OLD_IP="$(grep -oE '^CONTAINER_HOST_IP="?[0-9.]+' /etc/environment 2>/dev/null | grep -oE '[0-9.]+$' || true)"
fi
if [[ -z "$OLD_IP" ]]; then
  echo "could not discover the current IP from pipeline.py or /etc/environment - aborting" >&2
  exit 1
fi

OLD_DEVKIT_IP="$(grep -oE 'DEVKIT = "[^"]*"' "$REPO_ROOT/pipeline-scale/pipeline.py" 2>/dev/null \
  | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' || true)"
OLD_DEVKIT_IP="${OLD_DEVKIT_IP:-192.168.135.72}"
DEVKIT_IP="${NEW_DEVKIT_IP:-${DEVKIT_IP_ENV:-$OLD_DEVKIT_IP}}"

echo "  host  old: $OLD_IP        new: $NEW_IP"
echo "  board old: $OLD_DEVKIT_IP  new: $DEVKIT_IP"
[[ "$OLD_IP" == "$NEW_IP" ]] && warn "host already pointed at $NEW_IP - running the fix-up/verify steps anyway"
if [[ "$OLD_DEVKIT_IP" == "$DEVKIT_IP" && -z "$NEW_DEVKIT_IP" ]]; then
  warn "board IP left at $DEVKIT_IP - pass it as the 2nd argument if your DevKit differs"
fi

# ---------------------------------------------------------------------------
say "repo-wide IP swap"
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"
SWAP_INCLUDES=(--include="*.py" --include="*.html" --include="*.sh" --include="*.yaml"
               --include="*.yml" --include="*.json" --include="*.md" --include="*.txt")
SWAP_EXCLUDES=(--exclude-dir=.build --exclude-dir=ul_venv --exclude-dir=node_modules
               --exclude-dir=__pycache__)

# Swap one address everywhere, then prove none survived.
swap_repo_ip() {
  local old="$1" new="$2" label="$3"
  if [[ "$old" == "$new" ]]; then
    ok "$label already $new - nothing to change"
    return
  fi
  local files
  mapfile -t files < <(grep -rl "$old" "${SWAP_INCLUDES[@]}" . "${SWAP_EXCLUDES[@]}" 2>/dev/null)
  if [[ "${#files[@]}" -eq 0 ]]; then
    warn "$label: no file references $old - is this repo already repointed?"
    return
  fi
  local f
  for f in "${files[@]}"; do sed -i "s/$old/$new/g" "$f"; done
  ok "$label: $old -> $new in ${#files[@]} file(s)"
  local remain
  remain="$(grep -rl "$old" "${SWAP_INCLUDES[@]}" . "${SWAP_EXCLUDES[@]}" 2>/dev/null | wc -l)"
  if [[ "$remain" -eq 0 ]]; then ok "$label: clean - no references to $old remain"
  else fail "$label: $remain file(s) still reference $old"; fi
}

mapfile -t FILES_ALL < <(grep -rl "INSIGHT" \
  --include="*.py" --include="*.html" --include="*.sh" \
  . --exclude-dir=.build --exclude-dir=ul_venv --exclude-dir=node_modules 2>/dev/null)

swap_repo_ip "$OLD_IP" "$NEW_IP" "host IP"
swap_repo_ip "$OLD_DEVKIT_IP" "$DEVKIT_IP" "board IP"

# Generated run configs carry the host IP in every RTSP URL and in the Insight
# `host:` field. They accumulate addresses from EVERY machine this repo has
# lived on, so a single old->new swap leaves earlier ones behind (that is how
# a third, long-dead IP ends up in a config that still "works" until it
# doesn't). These files are derived - pipeline.py rewrites them on the next
# `up` - so any non-loopback address in them can be moved to the host IP.
# ui-state.json is deliberately NOT touched: it can hold EXTERNAL camera URLs
# that are not this host and must not be rewritten.
STALE_TOTAL=0
for cfg in "$REPO_ROOT"/pipeline-*/*-run.yaml; do
  [[ -f "$cfg" ]] || continue
  mapfile -t stale < <(grep -ohE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' "$cfg" 2>/dev/null \
    | sort -u | grep -vE "^(${NEW_IP//./\\.}|127\.0\.0\.1|0\.0\.0\.0)$" || true)
  [[ "${#stale[@]}" -eq 0 ]] && continue
  cp -f "$cfg" "${cfg}.bak-repoint"
  for old in "${stale[@]}"; do sed -i "s/$old/$NEW_IP/g" "$cfg"; done
  warn "$(basename "$cfg"): stale ${stale[*]} -> $NEW_IP (backup: $(basename "$cfg").bak-repoint)"
  STALE_TOTAL=$((STALE_TOTAL + 1))
done
[[ "$STALE_TOTAL" -eq 0 ]] && ok "generated run configs carry no stale addresses"

for st in "$REPO_ROOT"/pipeline-*/ui-state.json; do
  [[ -f "$st" ]] || continue
  if grep -qE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' "$st" 2>/dev/null &&
     ! grep -q "$NEW_IP" "$st" 2>/dev/null; then
    warn "$(basename "$(dirname "$st")")/ui-state.json holds addresses that are not $NEW_IP -"
    warn "  left alone (may be external cameras). Check it if a saved stream will not start."
  fi
done

# ---------------------------------------------------------------------------
say "Insight port check"
# ---------------------------------------------------------------------------
OLD_PORT="$(grep -oE 'INSIGHT_API = "https://[0-9.]+:([0-9]+)"' "$REPO_ROOT/pipeline-scale/pipeline.py" 2>/dev/null | grep -oE '[0-9]+"$' | tr -d '"')"
if [[ -n "$OLD_PORT" && "$OLD_PORT" != "$INSIGHT_PORT" ]]; then
  warn "pipelines point at Insight :$OLD_PORT but the container publishes :$INSIGHT_PORT - swapping"
  for f in "${FILES_ALL[@]}"; do sed -i "s#:${OLD_PORT}#:${INSIGHT_PORT}#g" "$f"; done
  ok "Insight port -> $INSIGHT_PORT"
else
  ok "Insight port $INSIGHT_PORT matches the pipelines"
fi

# ---------------------------------------------------------------------------
say "this container's env (what Insight advertises for WebRTC)"
# ---------------------------------------------------------------------------
for f in /etc/environment /root/.devkit-sync.rc; do
  if sudo -n test -f "$f" 2>/dev/null; then
    sudo -n cp "$f" "${f}.bak-previous" 2>/dev/null || true
    sudo -n sed -i "s/$OLD_IP/$NEW_IP/g" "$f"
    ok "patched $f"
  else
    warn "$f not found/readable - skipped"
  fi
done
if [[ -f "$HOME/.devkit-sync.rc" ]]; then
  sed -i "s/$OLD_IP/$NEW_IP/g" "$HOME/.devkit-sync.rc" 2>/dev/null && ok "patched $HOME/.devkit-sync.rc"
fi

# ---------------------------------------------------------------------------
say "restart Insight"
# ---------------------------------------------------------------------------
if command -v insight-admin >/dev/null 2>&1; then
  if insight-admin restart >/dev/null 2>&1; then ok "insight-admin restart"; else fail "insight-admin restart failed"; fi
else
  warn "insight-admin not found - skipped"
fi
sleep 2

# ---------------------------------------------------------------------------
say "DevKit: fstab, watchdog, mount, UI servers"
# ---------------------------------------------------------------------------
if ! ssh "${SSH_OPTS[@]}" "${DEVKIT_USER}@${DEVKIT_IP}" true 2>/dev/null; then
  fail "cannot reach the DevKit at ${DEVKIT_IP} - skipped all DevKit-side steps"
else
  ssh "${SSH_OPTS[@]}" "${DEVKIT_USER}@${DEVKIT_IP}" bash -s -- "$OLD_IP" "$NEW_IP" "$DEVKIT_ROOT" "$NFS_EXPORT" <<'REMOTE'
set -u
OLD_IP="$1"; NEW_IP="$2"; DEVKIT_ROOT="$3"; NFS_EXPORT="$4"

# The export path is this board's existing mount source, so a new user does
# not have to know the previous owner's home directory. fstab first, then the
# live mount table, then a last-resort default.
if [[ -z "$NFS_EXPORT" ]]; then
  NFS_EXPORT="$(grep -E "[0-9]:/.*[[:space:]]/workspace[[:space:]]" /etc/fstab 2>/dev/null \
    | head -1 | grep -oE ":/[^[:space:]]+" | cut -c2- || true)"
fi
if [[ -z "$NFS_EXPORT" ]]; then
  NFS_EXPORT="$(findmnt -rn -T /workspace -o SOURCE 2>/dev/null | grep -oE ":/.*" | cut -c2- || true)"
fi
if [[ -z "$NFS_EXPORT" ]]; then
  echo "  WARNING: could not discover the NFS export path - set NFS_EXPORT= and re-run"
  NFS_EXPORT="/workspace"
fi
echo "  NFS export: ${NFS_EXPORT}"

# Reachability from the DEVKIT's own vantage point - the only check that
# matters, since it is the DevKit that mounts from this address. Checked
# BEFORE touching anything, so a bad new IP fails loud here instead of
# hanging systemd's mount unit for ~90s and leaving an unkillable D-state
# mount.nfs process behind (that happened once - see the pkill guard below).
if (exec 3<>"/dev/tcp/${NEW_IP}/2049") 2>/dev/null; then
  exec 3>&- 3<&-
  echo "  NFS (2049) reachable on ${NEW_IP} from the DevKit"
else
  echo "  WARNING: ${NEW_IP}:2049 not reachable from the DevKit - mount will likely fail"
fi

# fstab: only the NFS source host, never touch anything else on the line
if grep -q "workspace" /etc/fstab 2>/dev/null; then
  sudo -n cp /etc/fstab /etc/fstab.bak-previous
  sudo -n sed -i "s#${OLD_IP}:#${NEW_IP}:#" /etc/fstab
  echo "  fstab -> $(grep workspace /etc/fstab)"
fi

# watchdog: the actual root cause of mount flapping if this step is skipped -
# it re-mounts from its own hardcoded src= every ~60s regardless of fstab.
WD=/usr/local/sbin/devkit-nfs-watchdog.sh
if [[ -f "$WD" ]]; then
  sudo -n cp "$WD" "${WD}.bak-previous"
  sudo -n sed -i "s#^src=.*#src=\"${NEW_IP}:${NFS_EXPORT}\"#" "$WD"
  echo "  watchdog src -> $(grep '^src=' "$WD")"
  sudo -n systemctl stop devkit-nfs-watchdog.timer devkit-nfs-watchdog.service 2>/dev/null || true
  if [[ "$OLD_IP" != "$NEW_IP" ]]; then
    # Only ever target mounts referencing the address we are LEAVING. When
    # old == new (re-running against the same IP to heal/verify) this pattern
    # would also match the currently-active, correct mount - killing an NFS
    # mount mid-flight with -9 does not stop it cleanly, it leaves the kernel
    # client wedged in an unkillable D state. Learned this the expensive way.
    # The bracket stops the pattern matching pkill's OWN argv - without it
    # pkill kills itself ("Killed") before reaching any real stale mount.
    sudo -n pkill -9 -f "[m]ount -t nfs.*${OLD_IP}" 2>/dev/null || true
  fi
  sudo -n systemctl start devkit-nfs-watchdog.timer 2>/dev/null || true
fi

# Stop the UI servers BEFORE the mount is touched. They run FROM the NFS
# share, so once the old server is gone they block in uninterruptible D state
# on their next file access - and a D-state process ignores SIGKILL. That is
# what leaves a dead server still holding its port ("Address already in use")
# and what makes a later reboot sit for minutes on
# "Waiting for process: ... (python3)" before force-rebooting. Killing them
# while the mount is still answering avoids creating those zombies at all.
for _p in pipeline-scale pipeline-live pipeline-group; do
  sudo -n pkill -9 -f "${DEVKIT_ROOT}/${_p}/ui_[s]erver.py" 2>/dev/null || true
done
# The chooser lives on the same share and hits the same D-state trap.
sudo -n pkill -9 -f "${DEVKIT_ROOT}/launcher[.]py" 2>/dev/null || true

sudo -n systemctl daemon-reload 2>/dev/null || true
if ps -eo cmd 2>/dev/null | grep -q "^/sbin/mount.nfs.*${NEW_IP}"; then
  echo "  a mount.nfs process for ${NEW_IP} is already running - not starting a second one"
elif findmnt -rn -T /workspace -o SOURCE 2>/dev/null | grep -q "^${NEW_IP}:"; then
  echo "  /workspace already mounted from ${NEW_IP}"
else
  # Mount directly rather than via `systemctl start workspace.mount`. The
  # fstab-generated unit is not always present (the watchdog mounts the share
  # itself, outside systemd, so systemd only ever sees a transient unit and
  # `systemctl start` reports "Unit workspace.mount not found"). Calling mount
  # here makes this step work regardless, and `timeout` bounds it so a target
  # that dies mid-mount cannot wedge us indefinitely.
  MNT_SRC="${NEW_IP}:${NFS_EXPORT}"
  MNT_OPTS="vers=4,proto=tcp,soft,timeo=600,retrans=3,_netdev,nofail"
  sudo -n umount -lf /workspace >/dev/null 2>&1 || true
  if timeout 60 sudo -n mount -t nfs -o "$MNT_OPTS" "$MNT_SRC" /workspace 2>&1; then
    echo "  mounted /workspace from ${NEW_IP}"
  else
    echo "  mount command failed or timed out"
  fi
fi
sleep 2

hold=0
for i in 1 2 3; do
  sleep 5
  ls "$DEVKIT_ROOT" >/dev/null 2>&1 && hold=$((hold+1))
done
echo "  mount held: ${hold}/3 checks over 15s"
if mount | grep -i workspace; then
  # The chooser (:8080) is started alongside the three panels - it is the page
  # the user actually opens, so leaving it down after a re-point would look
  # like the whole thing is broken.
  start_web() {   # <script-path> <port>
    local script="$1" port="$2" out
    [[ -f "$script" ]] || return 0
    out="$(bash "$script" start 2>&1)"
    # A server whose files vanished under it (the old mount was just detached)
    # can sit unkillable for a few seconds still holding its port, so the
    # immediate rebind loses to "Address already in use". That clears on its
    # own - one retry turns a hard failure into a short delay.
    if ! (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
      sleep 8
      out="$(bash "$script" start 2>&1)"
    else
      exec 3>&- 3<&-
    fi
    echo "$out" | tail -1
  }
  start_web "${DEVKIT_ROOT}/launcher.sh" 8080
  for p in pipeline-scale pipeline-live pipeline-group; do
    port=8090; [[ "$p" == "pipeline-live" ]] && port=8091
    [[ "$p" == "pipeline-group" ]] && port=8092
    start_web "${DEVKIT_ROOT}/${p}/ui.sh" "$port"
  done
else
  echo "  WARNING: workspace not in the mount table"
  # A D-state mount.nfs process here means the kernel NFS client is wedged -
  # confirmed twice (2026-07-24, 2026-07-28): no signal touches it, waiting
  # longer does not clear it, retrying the mount only adds more wedged
  # processes. The one fix that has worked is rebooting the DevKit. Detecting
  # this here means finding that out costs one line, not another investigation.
  if ps -eo stat,cmd 2>/dev/null | grep -q "^D.*mount\.nfs"; then
    echo "  DIAGNOSIS: a mount.nfs process is stuck in D state (unkillable) -"
    echo "    this is the known NFS-client wedge, not a bad IP or a config issue."
    echo "    Fix: reboot the DevKit, then re-run this script."
    echo "      ssh ${DEVKIT_USER}@${DEVKIT_IP} 'sudo systemctl reboot'"
    echo "      (wait ~1 min for it to come back, then) bash $0 ${NEW_IP}"
  fi
fi
REMOTE
fi

# ---------------------------------------------------------------------------
say "rebuild any currently-running detector (picks up corrected URLs)"
# ---------------------------------------------------------------------------
for p in pipeline-scale pipeline-live pipeline-group; do
  d="$REPO_ROOT/$p"
  [[ -d "$d" ]] || continue
  port=8090; [[ "$p" == "pipeline-live" ]] && port=8091; [[ "$p" == "pipeline-group" ]] && port=8092
  state="$(curl -sk --max-time 10 "http://${DEVKIT_IP}:${port}/api/state" 2>/dev/null)"
  if [[ -z "$state" ]]; then
    warn "$p: UI not reachable - skipped"
    continue
  fi
  running="$(echo "$state" | python3 -c "import json,sys; print(json.load(sys.stdin).get('app_running'))" 2>/dev/null)"
  count="$(echo "$state" | python3 -c "import json,sys; print(json.load(sys.stdin).get('count',0))" 2>/dev/null)"
  if [[ "$running" == "True" && "${count:-0}" -gt 0 ]]; then
    if ( cd "$d" && python3 - <<'PY'
import sys, time
sys.path.insert(0, ".")
import ui_server, pipeline
streams = ui_server.load_streams()
if streams:
    ui_server.rebuild(streams)
    time.sleep(3)
    print(f"  rebuilt {len(streams)} stream(s) on {pipeline.PIPELINE}")
PY
    ); then ok "$p: rebuilt with corrected URLs"; else warn "$p: rebuild failed - check manually"; fi
  else
    warn "$p: nothing running - skipped"
  fi
done

# ---------------------------------------------------------------------------
say "verify"
# ---------------------------------------------------------------------------
sip="$(curl -sk --max-time 8 https://127.0.0.1:9900/api/server-ip 2>/dev/null)"
echo "  Insight reports: ${sip:-unreachable}"
if echo "$sip" | grep -q "\"$NEW_IP\""; then ok "Insight is advertising $NEW_IP"; else fail "Insight is NOT advertising $NEW_IP"; fi

# The check that matters most and was missing for a long time: Insight as the
# DEVKIT sees it. Verifying it only from inside the container (above) always
# passes, because that path is 127.0.0.1 and never involves the IP or the
# published port at all.
if ssh "${SSH_OPTS[@]}" "${DEVKIT_USER}@${DEVKIT_IP}" \
     "curl -sk --max-time 8 https://${NEW_IP}:${INSIGHT_PORT}/api/server-ip" 2>/dev/null | grep -q "\"$NEW_IP\""; then
  ok "Insight reachable from the DevKit on ${NEW_IP}:${INSIGHT_PORT}"
else
  fail "Insight NOT reachable from the DevKit on ${NEW_IP}:${INSIGHT_PORT} - the pipelines will fail with 'Connection refused'"
fi

ssh "${SSH_OPTS[@]}" "${DEVKIT_USER}@${DEVKIT_IP}" \
  'systemctl is-active workspace.mount' 2>/dev/null | grep -q active \
  && ok "DevKit mount active" || fail "DevKit mount not active"

if curl -sk --max-time 8 "http://${DEVKIT_IP}:8080/api/status" >/dev/null 2>&1; then
  ok "pipeline chooser reachable on :8080"
else
  fail "pipeline chooser not reachable on :8080"
fi

for p in pipeline-scale pipeline-live pipeline-group; do
  port=8090; [[ "$p" == "pipeline-live" ]] && port=8091; [[ "$p" == "pipeline-group" ]] && port=8092
  if curl -sk --max-time 8 "http://${DEVKIT_IP}:${port}/api/state" >/dev/null 2>&1; then
    ok "$p UI reachable on :$port"
  else
    fail "$p UI not reachable on :$port"
  fi
done

echo
echo "Done."
echo "  Pipeline chooser (start here): http://${DEVKIT_IP}:8080/"
echo "  Neat Insight (video output):   https://${NEW_IP}:${INSIGHT_PORT}/"
