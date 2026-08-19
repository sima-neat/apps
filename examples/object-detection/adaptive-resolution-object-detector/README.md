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
| Model | yolo26n-det-int8-b1 |

## Concept

Multi-stream RTSP YOLO26 detection that publishes video plus detection metadata
per stream to Insight. One entry point per language carries two graph
topologies, chosen with `--mode`:

**`--mode adaptive`** (default) builds **one graph per stream**. Streams can be
added or removed while the others keep running: the app polls its config file
and diffs `streams.sources`.

"Adaptive" here means two separate things, and only the first is on by default:

- **Delivered video resolution** *(always on)* — each stream's output height is
  chosen from a shared bandwidth budget divided by the active stream count, so
  the picture steps down gracefully as streams are added rather than the
  pipeline falling over. Changes only on add/remove, never per frame.
- **Model input tier** *(off)* — `adaptive.resolutions: [640]` is a single
  value, so the model input size is fixed. List several ascending sizes matching
  `model.tiers` to let detection accuracy follow scene content too.

[POLICY.md](POLICY.md) documents both axes, with the resolution table and every
knob.

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

One pack is enough. `model.tiers` in the config maps a *model input size* to its
own MLA-compiled archive, and is consulted only when `adaptive.resolutions` has
more than one entry. The shipped config pins `resolutions: [640]`, so tier
switching is off and `model.path` serves every stream. `tools/build_yolo26_tiers.sh`
builds the per-tier archives if you want that behaviour back.

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

adaptive:
  resolutions: [640]           # one entry => fixed model input, no tier switching

streams:
  max_streams: 16              # default 8 if omitted
  sources:                     # edit while running to add/remove streams
    - id: cam-1
      rtsp_url: <first-rtsp-url>

output:
  adaptive:
    heights: [2160, 1080, 720, 480]   # candidates, never upscaled past native
    budget_megapixels_per_s: 280      # starting point; see note below
  insight:
    host: <insight-host-ip>
    video_port: 9000
    metadata_port: 9100
```

`budget_megapixels_per_s` bounds encode/deliver load, not raw decode. Each
stream gets `budget / active_streams` and picks the highest height that fits, so
one stream lands near 4K and sixteen near 480p ([POLICY.md](POLICY.md) has the
full table).

**280 is a starting point, not a measured limit.** There is no hardware number
to query — raise it until frames drop, then back off. The `pipelines/` bundle
sets it high deliberately, so streams are delivered at source resolution and the
budget never binds; the throughput figures quoted for those pipelines were taken
that way.

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
  `src/cpp/fused_app.h`, `src/cpp/adaptive_policy.h`
- Python source: `src/python/main.py` (entry point), `src/python/adaptive_app.py`,
  `src/python/fused_app.py`, `src/python/adaptive_policy.py`
- Shared runtime files: `src/common/`
- Browser UI for both modes: `pipelines/`
- Per-tier model builder: `tools/build_yolo26_tiers.sh`
- Tier policy reference: [POLICY.md](POLICY.md); test plan: [TESTING.md](TESTING.md)

Each language keeps both topologies behind one entry point because the Apps
build compiles a single `main.cpp` per example; the C++ implementations are
headers in named namespaces so their helpers do not collide.

The packaged C++ source is an implementation reference. Run the executable under
`src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
