# Adaptive Resolution Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream, adaptive-resolution, insight, yolo26 |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | adaptive-resolution-object-detector |
| Model | yolo26n-det-bf16-mla_tess-b1 |

## Concept

This example runs multi-stream RTSP YOLO26 detection and publishes video plus
detection metadata for each stream to Insight. One entry point per language
serves two topologies, selected with `--mode`:

| Mode | Topology | Adding a stream | Use it for |
| --- | --- | --- | --- |
| `adaptive` (default) | one graph **per stream**; a shared output-bandwidth budget picks each stream's delivered resolution | builds live, others keep running | changing stream count without downtime |
| `fused` | **one** graph for all streams into a single shared detector, source H.264 passed through to Insight without a re-encode | rebuilds the graph | higher stream counts |

The per-stream bridges in `adaptive` cap reliable metadata at roughly six
streams; `fused` has no such limit and its ceiling is decoder and pool capacity.

Both languages expose the same flags, which is what lets
[`pipelines/`](pipelines/README.md) switch implementation from a browser without
changing anything else. The tier policy behind `adaptive` is in
[POLICY.md](POLICY.md).

## Prerequisites

- Installed Neat Development Environment and Neat Library.
- RTSP sources in Insight, or your own cameras.
- The model pack (see below). Model artifacts are user-managed under `assets/models/`.
- On a Modalix DevKit, run `bash /usr/bin/fix_devkit_runtime.sh` first if the
  runtime has been used by earlier ML/video apps.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

`adaptive.resolutions` is pinned to `[640]`, so one YOLO26n pack covers every
run. The `model.tiers` block in `config.yaml` is only consulted when you restore
a multi-value `resolutions` list; those per-tier archives are built locally by
`tools/build_yolo26_tiers.sh` and are not needed otherwise.

```bash
mkdir -p assets/models
cd assets/models
# used by src/common/config.yaml
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<modelzoo-version>/models/modalix/yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz
# used by the pipelines/ bundle
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<modelzoo-version>/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz
cd ../..
```

`<modelzoo-version>` is the `modelzoo-version` field in `deps/manifest.json`.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input
streams and render each output channel. Start the required streams in the
Insight Web UI, copy their RTSP URLs into `streams`, and use the host and UDP
port ranges reported by `neat` for the output settings.

Use sources at ~25–30 fps. A very high-rate clip (for example 150 fps) starts
and streams video but produces no detections: Insight's pending-match map is
evicted before frames arrive.

## Configure

Edit `examples/object-detection/adaptive-resolution-object-detector/src/common/config.yaml`.

```yaml
model:
  path: assets/models/yolo26n-det-bf16-mla_tess-b1.tar.gz
  labels: src/common/coco_label.txt

adaptive:
  resolutions: [640]         # single value => fixed model input size

streams:                     # add/remove entries while running to change count
  max_streams: 16
  sources:
    - id: cam-1
      rtsp_url: <first-rtsp-url>

inference:
  min_score: 0.30

output:
  adaptive:
    heights: [2160, 1080, 720, 480]
    budget_megapixels_per_s: 280   # fair-shared across active streams
  insight:
    host: <insight-host-ip>
    video_port: <video-udp-port>
    metadata_port: <metadata-udp-port>
```

`--mode fused` takes a different schema: a bare `streams:` list, as in
[multi-stream-object-detector](../multi-stream-object-detector). Handing a mode
the other's config fails validation rather than running with wrong settings.

## Run

Both languages take the same flags. Run from `prebuilt-apps/`.

### C++

```bash
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 examples/object-detection/adaptive-resolution-object-detector/src/cpp/pre-built/adaptive-resolution-object-detector \
  --mode adaptive \
  --config examples/object-detection/adaptive-resolution-object-detector/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/adaptive-resolution-object-detector/src/python/requirements.txt
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 python3 examples/object-detection/adaptive-resolution-object-detector/src/python/main.py \
  --mode adaptive \
  --config examples/object-detection/adaptive-resolution-object-detector/src/common/config.yaml
```

Use `--mode fused` for the shared-detector topology, and
`--validate-config-only` for a config check that opens no streams.

## Pipelines UI

[`pipelines/`](pipelines/README.md) runs both modes from a browser instead of the
command line, and switches implementation language without editing anything.

```bash
cd examples/object-detection/adaptive-resolution-object-detector/pipelines
./repoint-ip.sh <host-ip> <board-ip>    # one-time: rewrites both addresses
```

Then open `http://<board-ip>:8080/` and pick a pipeline:

| Pipeline | Runs | Port |
| --- | --- | --- |
| scale | `--mode fused`, one process | 8090 |
| live | `--mode adaptive` | 8091 |
| group | `--mode fused`, several processes, each owning a subset | 8092 |

The Python/C++ toggle on that page applies to all three. They share one MLA and
one set of Insight channels, so the chooser starts one and stops the others.
C++ is offered only once a binary exists; a fresh clone has none until
`./build.sh --clean`.

## Troubleshooting

- Replace all placeholder stream URLs and the Insight host before running.
- In `adaptive`, editing `streams.sources` while running adds or removes a
  stream; a rebuild takes 30–90 s, so it is not instant.
- Stop with SIGTERM, not SIGKILL. A killed process leaves decoder and CVU pools
  allocated, and the next run can fail to allocate at a count that worked before.
- No detections while video flows: check the source frame rate (see Prepare
  Insight) and that Insight's host and UDP ports match the config.
- Use `output.debug_dir`, `output.save_every`, and `runtime.profile` for diagnosis.

## Source Files

- C++ reference source: `src/cpp/main.cpp` (entry point), `src/cpp/adaptive_app.h`,
  `src/cpp/fused_app.h`, `src/cpp/adaptive_policy.h`
- Python source: `src/python/main.py` (entry point), `src/python/adaptive_app.py`,
  `src/python/fused_app.py`, `src/python/adaptive_policy.py`
- Shared runtime files: `src/common/`
- Browser UI for both modes: `pipelines/`
- Optional per-tier model builder: `tools/build_yolo26_tiers.sh`

Each language keeps both topologies behind one entry point because the Apps
build compiles a single `main.cpp` per example; the C++ implementations are
headers in named namespaces for that reason.

The packaged C++ source is an implementation reference. Run the executable under
`src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
