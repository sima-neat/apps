#!/usr/bin/env bash
# Local container composition; no model inference or ROS app dispatch here.
set -euo pipefail
mode=${1:-run}
config=${2:?Pass the absolute config.yaml path as the second argument}
common_dir=$(cd "$(dirname "$0")" && pwd)
label=neat-apps.ros-parallel
mapfile -t settings < <(python3 - "$config" <<'PY'
import sys,yaml
from pathlib import Path
c=yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(c['apps_root']).resolve())
print(c['image'])
for app in c['apps']:
    fields=[app['name'],app['path'],app['package']]
    if any(any(ch.isspace() for ch in value) for value in fields):
        raise ValueError('App names, relative paths and packages cannot contain whitespace')
    print('\t'.join(fields))
PY
)
((${#settings[@]} >= 3)) || { echo 'Invalid/empty composition config' >&2; exit 1; }
apps_root=${settings[0]}
image=${settings[1]}
apps() { printf '%s\n' "${settings[@]:2}"; }
stop() {
  while read -r name app package; do
    if [[ $(docker inspect --format '{{index .Config.Labels "neat-apps.ros-parallel"}}' "$name" 2>/dev/null || true) == true ]]; then
      docker stop "$name"
      docker rm "$name"
    fi
  done < <(apps)
}
if [[ $mode == stop ]]; then stop; exit; fi
[[ $mode == start || $mode == run ]] || { echo 'Use run, start or stop' >&2; exit 2; }
[[ $(uname -m) == aarch64 ]] || { echo 'Run on Modalix' >&2; exit 1; }
: "${VIRTUAL_ENV:?Activate the Neat Python environment}"
pyneat_root=$(realpath "$VIRTUAL_ENV")
while read -r name app package; do
  directory="$apps_root/examples/ros2/$app"
  [[ -f "$directory/install/$package/share/$package/local_setup.bash" ]] || {
    echo "Build $directory with colcon build first" >&2; exit 1;
  }
  if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists; stop this composition first" >&2; exit 1
  fi
done < <(apps)
docker build -t "$image" -f "$common_dir/Dockerfile.ros2" "$common_dir"
mounts=(--mount "type=bind,src=$apps_root,dst=$apps_root"
        --mount "type=bind,src=$pyneat_root,dst=$pyneat_root,readonly")
for path in /usr /bin /lib /etc/ld.so.cache /etc/alternatives; do
  mounts+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
devices=()
for device in /dev/mla /dev/cvu /dev/dma_heap/linux,cma /dev/dma_heap/simaai,dms /dev/allegroDecodeIP /dev/allegroIP; do
  devices+=(--device "$device")
done
started=()
cleanup() {
  for name in "${started[@]}"; do
    docker stop "$name" >/dev/null 2>&1 || true
    docker rm "$name" >/dev/null 2>&1 || true
  done
}
trap cleanup ERR
while read -r name app package; do
  directory="$apps_root/examples/ros2/$app"
  docker run --detach --name "$name" --label "$label=true" --network host --ipc host \
    --user "$(id -u):$(id -g)" --tmpfs /tmp:mode=1777 --stop-signal SIGINT \
    "${mounts[@]}" "${devices[@]}" --workdir "$directory" \
    --env "PYNEAT_ROOT=$pyneat_root" --env "ROS_APP_PACKAGE=$package" \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" --env "ROS_APP_NAMESPACE=${name//-/_}" "$image"
  started+=("$name")
done < <(apps)
if [[ $mode == run ]]; then
  trap cleanup EXIT
  trap 'exit 0' INT TERM
  while :; do
    for name in "${started[@]}"; do
      [[ $(docker inspect --format '{{.State.Running}}' "$name") == true ]] || exit 1
    done
    sleep 1 & wait $!
  done
fi
