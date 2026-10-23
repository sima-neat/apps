# Parallel Apps in Docker

## Metadata

| Field | Value |
| --- | --- |
| Category | docker |
| Difficulty | Intermediate |
| Tags | docker, containers, compose, multistream, rtsp, insight |
| Languages | C++ |
| Status | stable |
| Binary Name | docker compose |
| Model | Each example's default model |

## Concept

Runs four Neat applications side by side in Docker containers on one Modalix DevKit, each sending live video and results to its own Insight channel.

Each container runs one installed example with its own config: instance segmentation, RF-DETR object detection, multi-class tracking, and pose estimation. The image holds only the example's binary and runtime files. Each container mounts the DevKit's installed Neat and system libraries read-only and gets the accelerator and video codec devices, so it behaves like a normal run of the same example. All four share the DevKit's machine learning accelerator (MLA).

## Preview

![Four Docker containers on four Insight channels](../../../portal/assets/examples/docker/parallel-apps/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix DevKit.
- Docker with the Compose and Buildx plugins on the DevKit, and your user in the `docker` group. Log in again after adding the group:

  ```bash
  sudo apt-get install docker.io docker-compose docker-buildx
  sudo usermod -aG docker "$USER"
  ```

- Four H.264 RTSP sources and an [Insight](https://developer.sima.ai/software/tools/insight/) host reachable from the DevKit.

## Install Apps

Install on the Modalix DevKit. If the Core and Insight selected by Apps are already installed, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps` to skip reinstalling them. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/docker/parallel-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Each container uses its example's default model. Model packages come from the Model Zoo release below, which can differ from the installed platform version.

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-segmentation/yolo26m-seg-bf16-b1.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/rfdetr-small-backbone.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/rfdetr-small-transformer.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26m-det-int8-b1.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-pose/yolo26m-pose-int8-b1.tar.gz"
cd ..
```

The containers see `prebuilt-apps/models` read-only at the same relative path, so model paths in the configs work unchanged. To use another model, follow that example's README.

## Configure

Each container reads its example's own `src/common/config.yaml`. Edit the four configs as their READMEs describe, with one difference: give each example its own Insight channel. Insight maps video port `9000 + N` and metadata port `9100 + N` to channel `N`, and all four configs start on channel 0.

| Example config | Model | RTSP source | Insight ports |
| --- | --- | --- | --- |
| `examples/segmentation/single-stream-instance-segmenter` | `model.path: models/yolo26m-seg-bf16-b1.tar.gz` | `source.url` | `output.insight.video_port: 9000`, `metadata_port: 9100` |
| `examples/object-detection/rfdetr-detection-segmentation` | Ready for RF-DETR small | `source.rtsp_url` | `output.insight.video_port: 9001`, `metadata_port: 9101` |
| `examples/tracking/multi-stream-tracker` | `model.path: models/yolo26m-det-int8-b1.tar.gz` | one URL under `streams` | `output.insight.video_port_base: 9002`, `metadata_port_base: 9102` |
| `examples/pose-estimation/multi-stream-pose-estimator` | `model.path: models/yolo26m-pose-int8-b1.tar.gz` | one URL under `streams` | `output.insight.video_port_base: 9003`, `metadata_port_base: 9103` |

Set `output.insight.host` in all four. Each config is mounted read-only into its container, so a config change takes effect when that container restarts.

## Run

Point Compose at this example's file, then build the four images and start the containers:

```bash
export COMPOSE_FILE=${APP_DIR}/src/common/compose.yaml
docker compose up --build --detach
```

Follow the logs of all four applications:

```bash
docker compose logs --follow
```

Stop and remove the containers:

```bash
docker compose down
```

`docker compose down` sends SIGINT, which the applications handle as a clean shutdown.

## Expected Result

`docker compose ps` lists four running services: `segmentation`, `detection`, `tracking`, and `pose`. Insight shows channels 0 to 3, each with video and overlays from a different application.

The MLA runs one model at a time, so each application's frame rate depends on the combined model time of all four. When the four sources together need more model time than the MLA has, the applications drop frames. Lower a source's frame rate or choose a smaller model in its config.

## Container Settings

`src/common/compose.yaml` defines the four services and the settings they share:

- `network_mode: host`: the applications use the DevKit's own network, so RTSP sources and Insight need no port mapping. Older DevKit kernels also lack NAT for Docker's default bridge network.
- `user: "1000:1000"`: the DevKit's `sima` user. The accelerator and codec devices belong to its group, so the containers run without root.
- `stop_signal: SIGINT`: the applications shut down cleanly on SIGINT. As the container's main process, they ignore Docker's default SIGTERM.
- `tmpfs: /tmp`: Neat unpacks model packages here at startup.
- `devices`: the MLA, the computer vision unit, their memory heaps, and the video decoder and encoder.
- `volumes`: the DevKit's `/usr`, `/bin`, `/lib`, and dynamic linker cache, read-only, plus `prebuilt-apps/models`.
- `configs`: each example's `config.yaml`, mounted read-only at `/apps/config.yaml`.

`src/common/Dockerfile` builds one image per service from the installed bundle. It starts `FROM scratch` and copies only the example's binary and its `src/common` files.

## Troubleshooting

- If a container exits at startup, read its log with `docker compose logs <service>`. The application reports configuration and model errors the same way as a normal run.
- If an application runs on the DevKit but fails to open the MLA in a container, update the DevKit with `sima-cli update`. Older MLA runtimes use system calls that Docker's default security profile blocks.
- If two applications appear on one Insight channel, check that their configs use different Insight ports.

## Source Files

- Container image: `src/common/Dockerfile`
- Services and shared container settings: `src/common/compose.yaml`

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
