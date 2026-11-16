#!/usr/bin/env bash
# Run on Modalix after sourcing the installed ROS and Neat environments.
set -euo pipefail
apps_root=$(cd "$(dirname "$0")/../../../../../.." && pwd)
label=neat-apps.ros-parallel
image=neat-ros-apps:local

apps() {
  cat <<'APPS'
neat-ros-segmentation segmentation/single-stream-instance-segmenter
neat-ros-detection object-detection/rfdetr-detection-segmentation
neat-ros-tracking tracking/multi-stream-tracker
neat-ros-pose pose-estimation/multi-stream-pose-estimator
neat-ros-drone tracking/yolo26-tiny-drone-tracker
neat-ros-blazepose pose-estimation/multi-stream-blazepose3d
APPS
}

case "${1:-}" in
  start)
    [[ $(uname -m) == aarch64 ]] || { echo 'Run this command on Modalix.' >&2; exit 1; }
    : "${VIRTUAL_ENV:?Source the Neat Python environment first}"
    : "${ROS_PACKAGE_PREFIX:?Set ROS_PACKAGE_PREFIX to the shared neat_apps installation}"
    pyneat_root=$(realpath "$VIRTUAL_ENV")
    ros_prefix=$(realpath "$ROS_PACKAGE_PREFIX")
    [[ -f "$ros_prefix/share/neat_apps/local_setup.bash" ]] || { echo 'Missing ROS package setup.' >&2; exit 1; }
    while read -r name app; do
      if docker container inspect "$name" >/dev/null 2>&1; then
        echo "Container already exists: $name; stop this example before restarting." >&2
        exit 1
      fi
    done < <(apps)
    docker build --file "$apps_root/examples/ros2/docker/parallel-apps/src/common/Dockerfile.ros2" \
      --tag "$image" "$apps_root/examples/ros2/docker/parallel-apps/src/common"
    mounts=(--mount "type=bind,src=$apps_root,dst=/apps,readonly"
            --mount "type=bind,src=$pyneat_root,dst=$pyneat_root,readonly"
            --mount "type=bind,src=$ros_prefix,dst=$ros_prefix,readonly")
    for path in /usr /bin /lib /etc/ld.so.cache /etc/alternatives; do
      mounts+=(--mount "type=bind,src=$path,dst=$path,readonly")
    done
    devices=()
    for device in /dev/mla /dev/cvu /dev/dma_heap/linux,cma /dev/dma_heap/simaai,dms /dev/allegroDecodeIP /dev/allegroIP; do
      devices+=(--device "$device")
    done
    started=()
    rollback() {
      for name in "${started[@]}"; do docker rm -f "$name" >/dev/null; done
    }
    trap rollback ERR
    while read -r name app; do
      docker run --detach --name "$name" --label "$label=true" --network host --ipc host \
        --user "$(id -u):$(id -g)" --tmpfs /tmp:mode=1777 --stop-signal SIGINT \
        "${mounts[@]}" "${devices[@]}" \
        --env "PYNEAT_ROOT=$pyneat_root" --env "ROS_PACKAGE_PREFIX=$ros_prefix" \
        --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" --env "NEAT_APP=$app" \
        --env "ROS_APP_NAMESPACE=${name//-/_}" "$image"
      started+=("$name")
    done < <(apps)
    ;;
  stop)
    while read -r name app; do
      if [[ $(docker inspect --format '{{index .Config.Labels "neat-apps.ros-parallel"}}' "$name" 2>/dev/null || true) == true ]]; then
        docker stop "$name"
        docker rm "$name"
      fi
    done < <(apps)
    ;;
  *) echo "Usage: $0 start|stop" >&2; exit 2 ;;
esac
