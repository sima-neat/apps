#!/usr/bin/env bash
# Run in the Neat SDK shell, from the installed Apps bundle or an Apps checkout.
#
#   parallel-apps.sh build             build one ARM64 image per app and push it
#                                      to the SDK's container registry
#   parallel-apps.sh deploy [name...]  start the apps on the DevKit, one at a time
#   parallel-apps.sh stop              stop and remove the apps' containers
set -euo pipefail

cd "$(dirname "$0")/../../../../.."
dockerfile=examples/docker/parallel-apps/src/common/Dockerfile

# Container name and example. Each example's binary has the example's name.
apps() {
  cat <<'APPS'
neat-segmentation segmentation/single-stream-instance-segmenter
neat-detection object-detection/rfdetr-detection-segmentation
neat-tracking tracking/multi-stream-tracker
neat-pose pose-estimation/multi-stream-pose-estimator
neat-drone tracking/yolo26-tiny-drone-tracker
neat-blazepose pose-estimation/multi-stream-blazepose3d
APPS
}

fail() {
  echo "error: $*" >&2
  exit 1
}

build() {
  : "${SIMA_CONTAINER_REGISTRY:?run this in the Neat SDK shell}"
  while read -r name example; do
    # The installed bundle ships each binary under src/cpp/pre-built; a source
    # checkout keeps it in the build tree.
    binary="examples/$example/src/cpp/pre-built/${example##*/}"
    [[ -f "$binary" ]] || binary="build/examples/$example/${example##*/}"
    [[ -f "$binary" ]] ||
      fail "no ${example##*/} binary; install Apps with sima-cli neat install apps, or build it with ./build.sh"
    file "$binary" | grep -q aarch64 || fail "$binary is not an ARM64 binary"
    [[ -n "$(ls -A "models/$name" 2>/dev/null)" ]] ||
      fail "models/$name is empty; download the model for $name first"

    docker buildx build --platform linux/arm64 --file "$dockerfile" \
      --build-arg "EXAMPLE=$example" --build-arg "BINARY=$binary" \
      --build-arg "CONTAINER=$name" \
      --tag "$SIMA_CONTAINER_REGISTRY/$name:latest" --push .
  done < <(apps)
}

# dk and the DevKit address come from the SDK shell's startup file, which
# scripts do not load.
load_devkit() {
  [[ -f ~/.devkit-sync.rc ]] ||
    fail "no DevKit connection; run this in the Neat SDK shell set up with sima-cli sdk setup --devkit"
  # shellcheck source=/dev/null
  source ~/.devkit-sync.rc
  devkit="${DEVKIT_SYNC_DEVKIT_USER:-sima}@${DEVKIT_SYNC_DEVKIT_IP:?run this in the Neat SDK shell}"
}

deploy() {
  load_devkit
  mapfile -t names < <(apps | cut -d' ' -f1)
  [[ $# -gt 0 ]] || set -- "${names[@]}"
  for name in "$@"; do
    [[ " ${names[*]} " == *" $name "* ]] || fail "unknown app '$name'; choose from: ${names[*]}"
  done

  # The DevKit user owns the accelerator devices through its group.
  user=$(ssh "$devkit" 'echo "$(id -u):$(id -g)"')

  # Host networking: older DevKit kernels have no NAT for Docker's bridge network.
  # The apps stop cleanly on SIGINT; as PID 1 they would ignore the default SIGTERM.
  # Neat unpacks model packages under /tmp.
  args=(--detach --network host --tmpfs /tmp:mode=1777 --user "$user"
    --stop-signal SIGINT)

  # The DevKit's installed Neat, system libraries, and tools, shared read-only.
  # /bin and /lib are the DevKit's links into /usr.
  for path in /usr /bin /lib /etc/ld.so.cache /etc/alternatives; do
    args+=(--mount "type=bind,src=$path,dst=$path,readonly")
  done

  # Kernel driver devices: MLA, CVU, video decoder and encoder, and their memory heaps.
  for device in /dev/mla /dev/cvu /dev/dma_heap/linux,cma \
    /dev/dma_heap/simaai,dms /dev/allegroDecodeIP /dev/allegroIP; do
    args+=(--device "$device")
  done

  # If one app fails to start, remove the ones already started so they do not
  # keep holding the accelerator and codecs.
  started=()
  for name in "$@"; do
    if ! dk container deploy "$name:latest" --name "$name" "${args[@]}"; then
      echo "error: $name failed to start; removing the apps started by this deploy" >&2
      for app in "${started[@]}"; do
        dk container stop "$app" || true
        dk container remove "$app" || true
      done
      exit 1
    fi
    started+=("$name")
    sleep 8
  done
}

stop() {
  load_devkit
  # dk container list prints docker ps -a, which ends each row with the name.
  existing=$(dk container list | awk 'NR > 1 {print $NF}')
  mapfile -t names < <(apps | cut -d' ' -f1)
  for name in "${names[@]}"; do
    grep -qx "$name" <<<"$existing" || continue
    dk container stop "$name"
    dk container remove "$name"
  done
}

case "${1:-}" in
  build) build ;;
  deploy) shift; deploy "$@" ;;
  stop) stop ;;
  *)
    echo "usage: $0 build | deploy [name...] | stop" >&2
    exit 2
    ;;
esac
