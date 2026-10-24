# Parallel Apps in Docker

## Metadata

| Field | Value |
| --- | --- |
| Category | docker |
| Difficulty | Intermediate |
| Tags | docker, containers, multistream, rtsp, insight |
| Languages | C++ |
| Status | stable |
| Binary Name | parallel-apps.sh |
| Model | Each example's default model |

## Concept

Runs four Neat applications side by side in Docker containers on one Modalix DevKit, each sending live video and results to its own Insight channel.

Each container runs one Apps example with its own config: instance segmentation, RF-DETR object detection, multi-class tracking, and pose estimation. You build Apps from source in the Neat SDK, pack each example into an ARM64 image, and push the images to the SDK's container registry. The DevKit pulls them from there. Each image holds only the example's binary, its `src/common` files, its config, and its model. Each container mounts the DevKit's installed Neat and system libraries read-only and gets the accelerator and video codec devices, so it behaves like a normal run of the same example. All four share the DevKit's machine learning accelerator (MLA).

## Preview

![Four Docker containers on four Insight channels](../../../portal/assets/examples/docker/parallel-apps/image.png)

## Prerequisites

- The Neat SDK set up with your Modalix DevKit (`sima-cli sdk setup --devkit`). This starts the SDK's container registry and the `dk` DevKit commands. Run every command below in the Neat SDK shell.
- The same Neat version in the SDK and on the DevKit. The containers run the DevKit's installed Neat with binaries built against the SDK's.
- A DevKit updated with `sima-cli update`. Older MLA runtimes use system calls that Docker's default security profile blocks.
- Docker on the DevKit. If it is missing, `dk container setup` installs it.
- Four H.264 RTSP sources and an [Insight](https://developer.sima.ai/software/tools/insight/) host reachable from the DevKit.

## Build Apps

From the root of the Apps checkout in the Neat SDK shell, build all examples. See the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md) for build options.

```bash
./build.sh
APP_DIR=examples/docker/parallel-apps
```

Run the remaining commands from the Apps checkout root.

## Prepare the Model

Each container gets its own folder under `models/`, named after the container, and its image holds only that folder. Model packages come from the Model Zoo release below, which can differ from the installed platform version.

```bash
export MODELZOO_VERSION="2.1.3"
MODELS="https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix"
mkdir -p models/neat-segmentation models/neat-detection models/neat-tracking models/neat-pose
(cd models/neat-segmentation && sima-cli download "${MODELS}/yolo26-segmentation/yolo26m-seg-bf16-b1.tar.gz")
(cd models/neat-detection && sima-cli download "${MODELS}/rfdetr-small-backbone.tar.gz" && sima-cli download "${MODELS}/rfdetr-small-transformer.tar.gz")
(cd models/neat-tracking && sima-cli download "${MODELS}/yolo26-detection/yolo26m-det-int8-b1.tar.gz")
(cd models/neat-pose && sima-cli download "${MODELS}/yolo26-pose/yolo26m-pose-int8-b1.tar.gz")
```

Inside an image, the container's folder appears as `models/`, so a config names a model as `models/<file>`. To use another model, follow that example's README and place the file in the container's folder.

## Configure

Each image holds its example's own `src/common/config.yaml`. Edit the four configs as their READMEs describe, with one difference: give each example its own Insight channel. Insight maps video port `9000 + N` and metadata port `9100 + N` to channel `N`, and all four configs start on channel 0.

| Container | Example config | Model | RTSP source | Insight ports |
| --- | --- | --- | --- | --- |
| `neat-segmentation` | `examples/segmentation/single-stream-instance-segmenter` | `model.path: models/yolo26m-seg-bf16-b1.tar.gz` | `source.url` | `output.insight.video_port: 9000`, `metadata_port: 9100` |
| `neat-detection` | `examples/object-detection/rfdetr-detection-segmentation` | Ready for RF-DETR small | `source.rtsp_url` | `output.insight.video_port: 9001`, `metadata_port: 9101` |
| `neat-tracking` | `examples/tracking/multi-stream-tracker` | `model.path: models/yolo26m-det-int8-b1.tar.gz` | one URL under `streams` | `output.insight.video_port_base: 9002`, `metadata_port_base: 9102` |
| `neat-pose` | `examples/pose-estimation/multi-stream-pose-estimator` | `model.path: models/yolo26m-pose-int8-b1.tar.gz` | one URL under `streams` | `output.insight.video_port_base: 9003`, `metadata_port_base: 9103` |

Set `output.insight.host` in all four. The config is copied into the image, so after a config change run `build` and `deploy` again.

## Run

Build the four ARM64 images and push them to the SDK's container registry:

```bash
${APP_DIR}/src/common/parallel-apps.sh build
```

Start the four containers on the DevKit, one after another:

```bash
${APP_DIR}/src/common/parallel-apps.sh deploy
```

To start only some of them, name them, for example `${APP_DIR}/src/common/parallel-apps.sh deploy neat-tracking neat-pose`.

Follow one application's log:

```bash
dk container logs neat-tracking --follow
```

Stop and remove the containers:

```bash
${APP_DIR}/src/common/parallel-apps.sh stop
```

`stop` sends SIGINT, which the applications handle as a clean shutdown.

## Expected Result

`dk container list` shows four running containers: `neat-segmentation`, `neat-detection`, `neat-tracking`, and `neat-pose`. Insight shows channels 0 to 3, each with video and overlays from a different application.

The MLA runs one model at a time, so each application's frame rate depends on the combined model time of all four. When the four sources together need more model time than the MLA has, the applications drop frames. Lower a source's frame rate or choose a smaller model in its config.

## Container Settings

`deploy` starts every container with the same settings:

- `--network host`: the applications use the DevKit's own network, so RTSP sources and Insight need no port mapping. Older DevKit kernels also lack NAT for Docker's default bridge network.
- `--user`: the DevKit user's user and group IDs. The accelerator and codec devices belong to its group, so the containers run without root.
- `--stop-signal SIGINT`: the applications shut down cleanly on SIGINT. As the container's main process, they ignore Docker's default SIGTERM.
- `--tmpfs /tmp`: Neat unpacks model packages here at startup.
- `--device`: the MLA, the computer vision unit, their memory heaps, and the video decoder and encoder.
- `--mount`: the DevKit's `/usr`, `/bin`, `/lib`, `/etc/alternatives`, and dynamic linker cache, read-only.

`src/common/Dockerfile` builds one image per container. It starts `FROM scratch` and copies only the example's binary, its `src/common` files, and the container's model folder. Because it runs no commands, it builds for ARM64 on any SDK host without emulation.

## Troubleshooting

- If `build` reports a missing binary, run `./build.sh` first. If it reports an empty model folder, download the model into `models/<container>/`.
- If a container exits at startup, read its log with `dk container logs <container>`. The application reports configuration and model errors the same way as a normal run.
- If an application runs on the DevKit but fails to open the MLA in a container, update the DevKit with `sima-cli update`.
- If an application fails to load a Neat library, install the SDK's Neat version on the DevKit.
- If two applications appear on one Insight channel, check that their configs use different Insight ports.

## Source Files

- Build, deploy, and stop, with the list of containers and examples: `src/common/parallel-apps.sh`
- Container image: `src/common/Dockerfile`

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
