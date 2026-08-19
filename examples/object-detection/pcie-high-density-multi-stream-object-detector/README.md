# PCIe High-Density Multi-Stream Object Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream, high-density, pcie, insight |
| Languages | C++ |
| Status | experimental |
| Binary Name | pcie-high-density-multi-stream-object-detector |
| Model | yolo26n-det-int8-b1 |

## Concept

This example runs high-density object detection across a host machine and a Modalix PCIe Card. The
host receives H.264 RTSP streams and sends encoded access units over PCIe. The card decodes each
stream, runs one shared object-detection model, and returns compact BBOX results. The host keeps the
encoded video and sends the matched video and detection metadata to Insight.

```text
Host machine                                      Modalix PCIe Card
============                                      ==================

N x RTSP/H.264                                    neatpciesrc.src_N
  +--> Insight video                                --> N decoders
  +--> neatpciehost.sink_N  -- H.264 over PCIe -->  --> shared model
                                                    --> box decode/NMS
Insight metadata  <-- BBOX results over PCIe --   neatpciesink.sink_N
```

The same host and card executables support 16, 24, and 48 streams through configuration. The card
application is included in the precompiled Apps package. The host application is not precompiled;
build it from source on the host machine as described below.

## Prerequisites

On the host machine:

- A supported x86-64 Ubuntu host with the
  [SiMa.ai PCIe driver installed](https://developer.sima.ai/hardware/getting-started/pcie-mode/driver-installation).
- The Neat PCIe host runtime and development package matching the card-side Neat installation.
- A local checkout of this Apps repository.
- Passwordless SSH access to the card. The PCIe host installation provisions the default key at
  `~/.ssh/sima_neat_pcie_ed25519`.
- CMake, a C++20 compiler, `ssh`, and `scp`.

On the Modalix PCIe Card:

- Matching Neat Core and Internals packages.
- Either the precompiled Apps package or an Apps source checkout with a locally built card binary.
- The `yolo26n-det-int8-b1.tar.gz` model package.

The RTSP sources must match the selected profile, and Insight must be reachable from the host.

## Install Apps

The Apps package provides the precompiled **card application only**. Run these commands on the
Modalix PCIe Card. Installing from `/workspace` gives the launcher its default card-binary path:

```bash
cd /workspace
sima-cli neat install apps
cd prebuilt-apps
```

Verify the card binary:

```bash
test -x examples/object-detection/pcie-high-density-multi-stream-object-detector/src/cpp/pre-built/pcie-high-density-multi-stream-object-detector
```

The resulting absolute path is:

```text
/workspace/prebuilt-apps/examples/object-detection/pcie-high-density-multi-stream-object-detector/src/cpp/pre-built/pcie-high-density-multi-stream-object-detector
```

To compile the card application instead, skip this section and follow
[Build the Card Application From Source](#build-the-card-application-from-source).

## Prepare the Model

Run these commands on the Modalix PCIe Card:

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p /workspace/models
cd /workspace/models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz"
```

Set `model.path` in the host-side configuration to the absolute card path:

```yaml
model:
  path: /workspace/models/yolo26n-det-int8-b1.tar.gz
```

The launcher copies the YAML to the card, but it does not copy the model.

## Configure

Run the remaining commands on the host machine from the Apps repository root. Copy the profile
closest to the intended workload:

| Configuration | Streams | Source contract |
| --- | ---: | --- |
| `src/common/config.yaml` | 16 | 1280x720 at 30 FPS |
| `src/common/config-24x720p20fps.yaml` | 24 | 1280x720 at 20 FPS |
| `src/common/config-48x720p10fps.yaml` | 48 | 1280x720 at 10 FPS |

```bash
APP_DIR=examples/object-detection/pcie-high-density-multi-stream-object-detector
cp "$APP_DIR/src/common/config-48x720p10fps.yaml" "$APP_DIR/src/common/config.local.yaml"
```

Edit `config.local.yaml` and set:

- `model.path` to the absolute model path on the card.
- `streams` to exactly 16, 24, or 48 RTSP URLs.
- `output.insight.host` to the Insight address reachable from the host.
- `card.card_id` and `card.queue` when their defaults are not appropriate.

Keep `model.labels` valid in the host Apps checkout. The host uses this file to name returned
classes. Do not change the configured width, height, or FPS unless every RTSP source matches the new
contract.

## Build the Host Application

There is no precompiled host executable. Install the PCIe host development package appropriate for
the host operating system, then build the native host application from the Apps repository.

Ubuntu 22.04:

```bash
sima-cli neat install core/pciehost/ubuntu22/amd64@v0.4.0
```

Ubuntu 24.04:

```bash
sima-cli neat install core/pciehost/ubuntu24/amd64@v0.4.0
```

Build from the Apps repository root:

```bash
APP_DIR=examples/object-detection/pcie-high-density-multi-stream-object-detector
cmake -S "$APP_DIR/src/host" -B build-host-pcie -DCMAKE_BUILD_TYPE=Release
cmake --build build-host-pcie --parallel
```

The launcher uses `build-host-pcie/pcie-high-density-multi-stream-object-detector-host`
by default.

## Run

The host-side launcher owns both process lifecycles. It uploads the selected configuration, starts
the card application with `nohup`, waits for `Graph ready`, and then starts the host application.

From the Apps repository root on the host:

```bash
./examples/object-detection/pcie-high-density-multi-stream-object-detector/run.sh \
  --config examples/object-detection/pcie-high-density-multi-stream-object-detector/src/common/config.local.yaml
```

The default card binary is the packaged executable at
`/workspace/prebuilt-apps/examples/object-detection/pcie-high-density-multi-stream-object-detector/src/cpp/pre-built/pcie-high-density-multi-stream-object-detector`.
For a card binary built from source, override it:

```bash
./examples/object-detection/pcie-high-density-multi-stream-object-detector/run.sh \
  --config examples/object-detection/pcie-high-density-multi-stream-object-detector/src/common/config.local.yaml \
  --card-binary /workspace/apps/build-pcie-card/pcie-high-density-multi-stream-object-detector
```

Use `--card-host`, `--card-user`, or `--identity` when the defaults do not match the installation.
Run `run.sh --help` for all options.

Press `Ctrl-C` once to stop both sides. The launcher sends `SIGINT` to the host first so it can send
EOS and release the PCIe queue, then stops the card. Do not restart only the host against an old
card graph; start a new launcher session instead.

The remote runtime files are:

```text
/home/sima/tmp/pcie-high-density/config.yaml
/home/sima/tmp/pcie-high-density/card.pid
/home/sima/tmp/pcie-high-density/card.log
```

The copied configuration and PID file are removed at shutdown. The card log is retained for
diagnostics.

## Expected Result

- Results continue on every configured stream.
- Insight receives encoded video and matching detection metadata on the corresponding channel.
- A 48-stream, 10 FPS profile advances at approximately 480 aggregate FPS after startup.
- Startup drop and timeout counters may increase while all decoders initialize, but they must
  plateau. Continued growth indicates a stalled or overloaded pipeline.

Insight channel and port mapping is deterministic:

```text
channel index: 0 .. stream_count - 1
video port:    output.insight.video_port_base + channel index
metadata port: output.insight.metadata_port_base + channel index
```

## Troubleshooting

If the launcher cannot find the default card binary, either install Apps under
`/workspace/prebuilt-apps` or pass the absolute source-build path with `--card-binary`.

Inspect retained card output from the host:

```bash
ssh -i ~/.ssh/sima_neat_pcie_ed25519 sima@10.0.0.2 \
  tail -n 100 /home/sima/tmp/pcie-high-density/card.log
```

If startup fails, do not launch the host manually against the partially initialized card graph.
Allow `run.sh` to clean up, then start a new launcher session.

Use one active Insight viewer while validating metadata. Multiple simultaneous viewers can make
box delivery appear intermittent even when the application is advancing normally.

## Source Files

- Lifecycle launcher: `run.sh`
- Shared configurations: `src/common/config*.yaml`
- Shared configuration parser: `src/common/app_config.cpp`
- Card graph: `src/cpp/main.cpp`
- Card result demultiplexer: `src/cpp/stream_demux.cpp`
- Host application: `src/host/main.cpp`
- Host build definition: `src/host/CMakeLists.txt`

The runtime Apps package contains the launcher, shared configurations, card implementation source,
and precompiled card executable. The host source and its CMake build definition are provided by the
Apps source repository and intentionally have no precompiled host artifact.

## Development From Source

To build the card application yourself, use an Apps source checkout on the Modalix PCIe Card with
the matching Neat development packages installed:

```bash
cd /workspace/apps
cmake \
  -S examples/object-detection/pcie-high-density-multi-stream-object-detector/src/cpp \
  -B build-pcie-card \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-pcie-card --parallel
```

The resulting binary is:

```text
/workspace/apps/build-pcie-card/pcie-high-density-multi-stream-object-detector
```

Pass that path to the host launcher with `--card-binary`. The host and card source checkouts must be
from compatible revisions, and the installed Core, Internals, and PCIe host packages must match.

For the complete repository build and contribution workflow, see the
[Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
