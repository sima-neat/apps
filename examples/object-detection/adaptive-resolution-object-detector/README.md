# Adaptive Resolution Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream, adaptive-resolution, insight, yolo26 |
| Languages | C++, Python |
| Status | stable |
| Binary Name | adaptive-resolution-object-detector |
| Model | yolo26n-det-int8-b1 |

## Concept

Runs YOLO26 object detection across multiple RTSP streams and sends each stream's video and detection metadata to Insight, with a choice of two graph topologies.

One entry point per language carries both topologies, chosen with `--mode`:

**`--mode adaptive`** (default) builds **one graph per stream**. Streams can be
added or removed while the others keep running: the app polls its config file
and diffs `streams.sources`, and only the affected stream is built or torn down.
Every stream is decoded, detected and delivered at its source's native size.

**`--mode fused`** builds **one graph for all streams**, fanning into a single
shared detector. Adding a stream rebuilds the whole graph, so it is not live —
in exchange, there are no per-stream bridges, which is what keeps detections
correct at higher stream counts.

Both forward the source H.264 to Insight without re-encoding it. The two take
**different config schemas** (see Configure); handing one the other's config
fails validation rather than running with settings you did not ask for.

The C++ and Python entry points take identical flags. That is what lets
[`pipelines/`](pipelines/README.md) switch implementation language from a
browser without changing anything else.

## Preview

![Adaptive resolution object detector preview](../../../portal/assets/examples/object-detection/adaptive-resolution-object-detector/image.png)

## Prerequisites

- Installed Neat Development Environment and Neat Library.
- RTSP sources in Insight, or your own cameras.
- A YOLO26n model pack (below). Model artifacts are user-managed under `assets/models/`.
- On a Modalix DevKit, run `bash /usr/bin/fix_devkit_runtime.sh` first if the
  runtime has been used by earlier ML/video apps.

## Install Apps

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/adaptive-resolution-object-detector
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

```bash
mkdir -p assets/models
cd assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<modelzoo-version>/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz
cd ../..
```

`<modelzoo-version>` is the `modelzoo-version` field in `deps/manifest.json`.

int8 is the default in `config.yaml` and in [`pipelines/`](pipelines/README.md)
because it is the build the published throughput numbers were measured on. bf16
is a drop-in alternative — download
`yolo26n-det-bf16-mla_tess-b1.tar.gz` the same way and point `model.path` at it.


## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input
streams and render each output channel. Start the streams in the Insight Web UI,
copy their RTSP URLs into the config, and use the host and UDP ports reported by
`neat` for the output settings. Stream N publishes to `video_port + N` and
`metadata_port + N`.

Use sources at ~25–30 fps. A much faster clip (for example 150 fps) starts
normally and delivers video, but produces no detections: Insight evicts pending
metadata before the matching frame arrives.

## Configure

The two modes read different files.

**`--mode adaptive`** — `src/common/config.yaml`:

```yaml
model:
  path: assets/models/yolo26n-det-int8-b1.tar.gz
  labels: src/common/coco_label.txt

streams:
  max_streams: 16              # default 8 if omitted
  sources:                     # edit while running to add/remove streams
    - id: cam-1
      rtsp_url: <first-rtsp-url>

output:
  insight:
    host: <insight-host-ip>
    video_port: 9000
    metadata_port: 9100
```

**`--mode fused`** — a bare `streams:` list, up to 64, and `*_port_base` keys:

```yaml
model:
  path: assets/models/yolo26n-det-int8-b1.tar.gz
  labels: src/common/coco_label.txt

streams:
  - <first-rtsp-url>
  - <second-rtsp-url>

inference:
  max_inflight_per_stream: 1
  max_inflight_total: 8

output:
  insight:
    host: <insight-host-ip>
    video_port_base: 9000
    metadata_port_base: 9100
```

## Run

Both languages take the same flags; `--validate-config-only` checks a config
without opening any stream.

### C++

```bash
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 ./${APP_DIR}/src/cpp/pre-built/adaptive-resolution-object-detector \
  --mode adaptive \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 python3 ${APP_DIR}/src/python/main.py \
  --mode adaptive \
  --config ${APP_DIR}/src/common/config.yaml
```

## Pipelines UI

[`pipelines/`](pipelines/README.md) drives both modes from a browser and
switches implementation language without editing anything:

```bash
cd examples/object-detection/adaptive-resolution-object-detector/pipelines
./repoint-ip.sh <host-ip> <board-ip>     # one-time: rewrites both addresses
```

Open `http://<board-ip>:8080/`.

| Pipeline | Runs | Port |
| --- | --- | --- |
| scale | `--mode fused`, one process | 8090 |
| live | `--mode adaptive` | 8091 |
| group | `--mode fused`, several processes, each owning a subset | 8092 |

The Python/C++ toggle on that page applies to all three. They share one MLA and
one set of Insight channels, so selecting one stops the others. C++ appears only
once a binary exists — a fresh clone has none until `./build.sh --clean`.

## Troubleshooting

- Replace the placeholder RTSP URLs and Insight host before running.
- Stop with SIGTERM, not SIGKILL. A killed process leaves decoder and CVU pools
  allocated, and the next run can then fail to allocate at a count that worked.
- Adding a stream in `adaptive` takes 30–90 s to rebuild; it is not instant.
- Video arriving with no boxes: check the source frame rate (see Prepare
  Insight), and that the Insight host and UDP ports match the config.
- Stream ceilings are per-board. Watch the per-channel rate and back off when it
  drops below the source rate.
- `output.debug_dir`, `output.save_every` and `runtime.profile` produce
  diagnostics.

## Source Files

- C++ source: `src/cpp/main.cpp` (entry point), `src/cpp/adaptive_app.h`,
  `src/cpp/fused_app.h`
- Python source: `src/python/main.py` (entry point), `src/python/adaptive_app.py`,
  `src/python/fused_app.py`
- Shared runtime files: `src/common/`
- Browser UI for both modes: `pipelines/`
- Manual test guide: [TESTING.md](TESTING.md)

Each language keeps both topologies behind one entry point because the Apps
build compiles a single `main.cpp` per example; the C++ implementations are
headers in named namespaces so their helpers do not collide.

The packaged C++ source is an implementation reference. Run the executable under
`src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
