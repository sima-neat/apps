# PCIe High-Density Multi-Stream Object Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream, high-density, pcie, insight |
| Languages | C++ |
| Status | stable |
| Binary Name | pcie-high-density-multi-stream-object-detector |
| Model | yolo26n-det-int8-b1 |

## Concept

Runs YOLO26 object detection across many RTSP streams using a host and Modalix PCIe card, then sends synchronized results to Insight.

## Preview

![PCIe high-density detector preview](../../../portal/assets/examples/object-detection/pcie-high-density-multi-stream-object-detector/image.png)

```text
Host machine                         Modalix PCIe Card
============                         ==================
N x RTSP/H.264  -- encoded video --> N decoders
  +--> Insight video                 --> shared model
                                      --> box decode/NMS
Insight metadata <-- BBOX results -- neatpciesink
```

The precompiled Apps package provides the card application. Build the host application from this
repository, then use `run.sh` to start and stop both sides.

## Prerequisites

- An x86-64 Ubuntu host with the
  [SiMa.ai PCIe driver](https://developer.sima.ai/hardware/getting-started/pcie-mode/driver-installation)
  and matching Neat PCIe host development package.
- A Modalix PCIe Card with matching Neat Core and Internals packages.
- Passwordless SSH access to the card using `~/.ssh/sima_neat_pcie_ed25519`.
- 16, 24, or 48 H.264 RTSP sources and an Insight instance reachable from the host.
- A source checkout of this Apps repository on the host.

## Install Apps

Install the precompiled card application on the Modalix PCIe Card:

```bash
cd "$HOME"
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/pcie-high-density-multi-stream-object-detector
```

The card binary is installed at
`${APP_DIR}/src/cpp/pre-built/pcie-high-density-multi-stream-object-detector`.

To build the card application yourself, see
[Development From Source](#development-from-source).

## Prepare the Model

Download the model on the Modalix PCIe Card:

```bash
mkdir -p /workspace/models
cd /workspace/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz
```

Use this absolute path for `model.path` in the configuration:

```text
/workspace/models/yolo26n-det-int8-b1.tar.gz
```

## Configure

On the host, copy the profile closest to the intended workload:

| Profile | Streams | Input |
| --- | ---: | --- |
| `config.yaml` | 16 | 1280x720 at 30 FPS |
| `config-24x720p20fps.yaml` | 24 | 1280x720 at 20 FPS |
| `config-48x720p10fps.yaml` | 48 | 1280x720 at 10 FPS |

```bash
APP_DIR=examples/object-detection/pcie-high-density-multi-stream-object-detector
cp "$APP_DIR/src/common/config-48x720p10fps.yaml" "$APP_DIR/src/common/config.local.yaml"
```

Edit `config.local.yaml` and set:

- `model.path` to the absolute model path on the card.
- `streams` to exactly 16, 24, or 48 RTSP URLs.
- `output.insight.host` to the Insight address.
- `card.card_id` and `card.queue` if their defaults are not appropriate.

The supplied profiles start RTSP streams 50 ms apart to avoid a simultaneous decoder startup
burst. Set `input.startup_stagger_ms` to `0` to disable the ramp.

The launcher copies this configuration to the card automatically.

## Build the Host Application

The host application is not precompiled. Install the PCIe host package for the host OS:

```bash
# Ubuntu 22.04
sima-cli neat install core/pciehost/ubuntu22/amd64@v0.4.0

# Ubuntu 24.04
sima-cli neat install core/pciehost/ubuntu24/amd64@v0.4.0
```

Build the host application from the Apps repository root:

```bash
APP_DIR=examples/object-detection/pcie-high-density-multi-stream-object-detector
cmake -S "$APP_DIR/src/host" -B build-host-pcie -DCMAKE_BUILD_TYPE=Release
cmake --build build-host-pcie --parallel
```

## Run

Start both applications from the host:

```bash
./${APP_DIR}/run.sh \
  --config ${APP_DIR}/src/common/config.local.yaml
```

The launcher uses the packaged card binary under
`${APP_DIR}/src/cpp/pre-built/pcie-high-density-multi-stream-object-detector`,
waits for the card graph, and then starts the locally built host application.

Press `Ctrl-C` once to stop the host first and the card second. Run `run.sh --help` for nondefault
SSH, binary, timeout, or runtime-directory options.

## Expected Result

- Every active stream continues producing results.
- Insight shows video and matching detections on the corresponding channels.
- The 48-stream, 10 FPS profile reaches approximately 480 aggregate FPS after startup.

## Troubleshooting

The launcher retains the card log after shutdown. Inspect it from the host with:

```bash
ssh -i ~/.ssh/sima_neat_pcie_ed25519 sima@10.0.0.2 \
  tail -n 100 /home/sima/tmp/pcie-high-density/card.log
```

Do not restart only the host against an old card graph. Stop the launcher and start a new session.

## Source Files

- `run.sh`: host-side launcher.
- `src/common/`: shared configurations and parser.
- `src/cpp/`: card graph and result demultiplexer.
- `src/host/`: native host application and build definition.

## Development From Source

To build the card application from source, run on the Modalix PCIe Card:

```bash
cd "$HOME/apps"
APP_DIR=examples/object-detection/pcie-high-density-multi-stream-object-detector
cmake \
  -S ${APP_DIR}/src/cpp \
  -B build-pcie-card \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-pcie-card --parallel
```

Then pass the resulting binary to the launcher:

```bash
./${APP_DIR}/run.sh \
  --config ${APP_DIR}/src/common/config.local.yaml \
  --card-binary /home/sima/apps/build-pcie-card/pcie-high-density-multi-stream-object-detector
```

For the complete repository workflow, see the
[Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
