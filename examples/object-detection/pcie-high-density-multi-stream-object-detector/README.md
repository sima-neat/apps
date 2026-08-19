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

This example splits the optimized `high-density-multi-stream-object-detector` pipeline across an
x86 PCIe host and a Modalix card. It reuses the high-density example's decoder/model settings and
Insight metadata envelope while keeping generic PCIe transport in Core and Internals.

The application is configuration-driven. The same host and card binaries support 16-, 24-, and
48-stream profiles without source changes.

```text
x86 PCIe host
=============

N x RTSP/H.264
  +--> original encoded Insight video branch
  +--> latest slot before admission --> neatpciehost.sink_N
                                      |
                                      | H.264 access unit
                                      v

Modalix card
============

neatpciesrc.src_N
  --> one SimaDecode context per stream
  --> bounded reliable fair fan-in
  --> one shared object-detection Model
  --> boxdecode/NMS
  --> BBOX demux by stream-id
  --> neatpciesink.sink_N
                                      |
                                      | compact correlated BBOX
                                      v

x86 PCIe host
=============

neatpciehost.src_N
  --> strict BBOX parser
  --> nonblocking MetadataSender
  --> Insight metadata channel N
```

The host bounds both its input queues and its correlation cache. A decoder or
an overloaded card branch may drop an admitted access unit, so missing results
expire from the cache and late results are counted instead of stopping all
streams.

The Modalix executable follows the standard Apps `src/cpp` layout. The native x86 executable lives
under `src/host` and builds independently against the installed PCIe-host package. RTSP,
object-detection, density-profile, and Insight policy remain in Apps; generic PCIe transport and
provisioning remain in the PCIe-host package.

## Prerequisites

- An x86 PCIe host with the SiMa PCIe-host runtime and development packages installed.
- A reachable Modalix PCIe card with matching NEAT Core and Internals packages.
- 16, 24, or 48 H.264 RTSP sources matching the selected profile.
- Insight reachable from the x86 host.
- The `yolo26n-det-int8-b1.tar.gz` model pack.
- A matching Core/Internals release with multi-pad `neatpciesrc`, `neatpciesink`, and
  `neatpciehost` support.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Run the remaining target-side commands from `prebuilt-apps/`.

## Prepare the Model

| Model file | Role | Source |
| --- | --- | --- |
| `yolo26n-det-int8-b1.tar.gz` | Default | Direct artifact |

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz"
cd ..
```

Set `model.path` in the selected configuration to the downloaded model package.

## Configure

The checked-in profile placeholders are:

| Config | Streams | Source contract |
| --- | ---: | --- |
| `src/common/config.yaml` | 16 | 1280x720 at 30 FPS |
| `src/common/config-24x720p20fps.yaml` | 24 | 1280x720 at 20 FPS |
| `src/common/config-48x720p10fps.yaml` | 48 | 1280x720 at 10 FPS |

Populate `streams` with the number of URLs shown for the selected profile, set `model.path`, and
set `output.insight.host`. Both processes consume the same configuration and derive the stream
count from the URL list.

For multiple cards, use one configuration and host/card process pair per card. Set `card.card_id`
for each physical card and assign non-overlapping Insight video and metadata port ranges.

## Run

Start the Modalix/card process first. It waits for the host to create the selected PCIe data queue:

```bash
./examples/object-detection/pcie-high-density-multi-stream-object-detector/src/cpp/pre-built/pcie-high-density-multi-stream-object-detector \
  --config examples/object-detection/pcie-high-density-multi-stream-object-detector/src/common/config.yaml
```

Then start the native host process with the same configuration:

```bash
./pcie-high-density-multi-stream-object-detector-host \
  --config examples/object-detection/pcie-high-density-multi-stream-object-detector/src/common/config.yaml
```

The host waits for a random-access H.264 access unit before admitting each stream, then forwards
encoded access units through a bounded latest-frame PCIe branch. Up to
`pcie.max_inflight_per_stream` inputs are queued independently per stream, and
returned BBOX buffers are matched with their original stream and frame before
metadata is emitted. Missing entries expire after `pcie.result_timeout_ms`;
unknown or late results increment `correlation_misses`. The original encoded
Insight video branch remains independent. The host records the RTP timestamp
actually produced by each `rtph264pay` instance and associates it with the
correlated frame PTS. Detection metadata is sent only when that exact outgoing
RTP timestamp is available; `metadata_without_rtp_timestamp` counts results
whose video frame was not transmitted by the independent latest-video branch.
Each PCIe result branch is bounded and downstream-leaky, so a stalled metadata
consumer discards stale results instead of blocking the shared PCIe receiver.

`pcie.pool_size` independently controls the card-side encoded-input pool for
each stream. Buffer capacity is derived from the first per-stream CAPS and
`pcie.buffer_size` is the maximum/fallback. Shared-detector admission is
configured separately with `inference.max_inflight_per_stream` and
`inference.max_inflight_total`; it must not be increased merely to enlarge the
host transport window.
`inference.internal_queue_depth` configures the final card graph's asynchronous
model-stage handoff depth; the 24-stream profile uses the same depth of two as
the directly connected high-density application.

`input.decoder_input_buffers` also sizes each decoder daemon client's bounded
asynchronous H.264 push queue. Unlike a directly paced RTSP source, PCIe can
deliver several access units in a short burst while the daemon serializes
compressed-input copies across clients. The validated 16-stream profile uses
8 entries; a depth of 2 can overflow during simultaneous startup and cause a
fatal decoder status 506 even though graph admission accepted all streams.
Configuration validation therefore requires `input.decoder_input_buffers` to
be at least `pcie.pool_size`.

For configuration checks without starting PCIe transport, use `--validate-config-only`. The host
also supports `--dump-pipeline`, and the card executable supports `--dump-backend`.

## Source Files

- Modalix/card graph: `src/cpp/main.cpp`
- App-private zero-copy result demultiplexer: `src/cpp/stream_demux.cpp`
- Native x86 host application: `src/host/main.cpp`
- Shared configuration parser: `src/common/app_config.cpp`
- Native host build: `src/host/CMakeLists.txt`
- Default 16-stream profile: `src/common/config.yaml`
- 24-stream profile: `src/common/config-24x720p20fps.yaml`
- 48-stream profile: `src/common/config-48x720p10fps.yaml`

The card path stays graph-native from `neatpciesrc` through decode, one shared model, box decode,
result demultiplexing, and `neatpciesink`. It does not use the legacy demo's appsink/appsrc queues,
application payload copies, overlays, re-encoding, or full-frame return path.

## Development From Source

Build the native host component on the x86 PCIe host after installing the PCIe-host development
package:

```bash
cmake -S examples/object-detection/pcie-high-density-multi-stream-object-detector/src/host \
  -B build/pcie-high-density-host -DCMAKE_BUILD_TYPE=Release
cmake --build build/pcie-high-density-host --parallel
```

Build the Modalix/card component through the normal Apps contributor workflow. See the
[Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
