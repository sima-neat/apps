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
This example runs YOLO26 detection across up to **16 RTSP streams** where **the
stream count and each stream's delivered video resolution adapt at runtime**.
Streams are added or removed on the fly by editing the config while the app runs.
Each stream's **delivered (output) resolution is chosen from a shared bandwidth
budget**: a single stream fills it at ~4K, and as streams are added each one steps
down (≈1080p, 720p, 480p) so the total output bandwidth stays fully utilised
without starving any stream. Per stream it publishes H.264 video plus detection
metadata to Insight, including the current stream count.

The YOLO26n model input size is **fixed** by default (one compiled archive); the
example still supports a second, optional axis that adapts the **model input tier**
(320 / 640 / 960) to scene content under a compute budget — see
[How the Model Tier Adapts](#how-the-model-tier-adapts-optional).

It exercises these Neat Library capabilities: `RtspDecodedInput` (RTSP → NV12) with
post-decode `videoscale` to the delivered resolution, compiled `Model` + `Graph` +
`Run`, `VideoSender` and `MetadataSender` per Insight channel, and multiple
concurrent pipelines rebuilt in place when a stream changes resolution.

## How Output Resolution Adapts (bandwidth budget)
There is no hardware "decoder bandwidth" number to query — Neat only enforces a
per-stream 4K ceiling — so the shared limit is modelled as a **configurable total
output pixel-rate budget** (`output.adaptive.budget_megapixels_per_s`), fair-shared
across active streams. Each stream gets `budget / active_streams` and picks the
**highest delivered height** (`output.adaptive.heights`, clamped to the source's
native size — never upscaled) whose `width × height × fps` fits its share. Widths
are derived per source to preserve aspect ratio.

The decoded frame is scaled once at the decode tail (`use_videoscale` +
`output_caps`), so detection, the H.264 video, and the metadata overlays all share
one resolution and stay aligned. The compiled MLA input size is unchanged (the
model letterboxes whatever frame it receives to its fixed size). **This budget
bounds encode/deliver load, not raw decode**: with single-profile sources the
decoder always decodes at native resolution (only the 4K ceiling applies). To also
cut raw decode you would switch cameras to a lower-resolution RTSP sub-stream.

Because the budget's fair share depends only on the active stream count, output
resolution changes **only when a stream is added or removed** (not per frame), and
only streams that cross a tier boundary rebuild. With `budget_megapixels_per_s: 280`
and 30 fps 16:9 sources:

| Active streams | Delivered per stream |
| --- | --- |
| 1 | 2160p (4K) |
| 2–4 | 1080p |
| 5–10 | 720p |
| 11–16 | 480p |

Tune `budget_megapixels_per_s` to your platform's sustainable encode/deliver
capacity (raise it until frames drop, then back off).

## Preview
![Adaptive resolution object detector preview](../../../assets/portal/object-detection/adaptive-resolution-object-detector/image.png)

## How the Model Tier Adapts (optional)
This second axis is **off by default** (`adaptive.resolutions: [640]` → a single
fixed model input). Enable it by listing several ascending sizes whose keys match
`model.tiers` (e.g. `[320, 640, 960]`); then detection accuracy also adapts to
scene content, independently of the delivered video resolution above.

On the MLA the model input size is fixed at compile time, so a genuine
throughput/accuracy trade-off needs **one compiled archive per resolution tier**.
Each stream runs at a single tier at a time; when the policy commits a switch,
the stream's pipeline is rebuilt against that tier's archive (the Insight channel
is preserved). Hysteresis (`adaptive.hysteresis_frames`) keeps switches
infrequent so the brief rebuild is not disruptive.

- **Step up** when any object is smaller than `min_object_px`, when a detection's
  confidence drops below `confidence_low`, or when the scene is crowded
  (`density_high`).
- **Step down** on simple scenes (few, large, confident objects).
- **Shared budget**: tier cost grows with pixel area (320→1, 640→4, 960→9). Each
  stream's fair share is `budget_units / active_streams`, which caps its maximum
  tier — so adding streams degrades everyone gracefully instead of starving one.

## Insight Setup
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host RTSP
streams, receive video from `VideoSender`, receive detection metadata from
`MetadataSender`, and show rendered overlays plus runtime metrics in the browser.

In the Neat Development Environment, install the sample video assets:

```bash
sima-cli install assets/multi-video-sources
```

To create reproducible RTSP inputs:
1. Run `neat` in the Neat Development Environment and open the reported `Insight Web UI`.
2. In Insight, open `RTSP Source`.
3. Use sample videos or upload your own.
4. Start each stream and copy the RTSP URLs.
5. Put those RTSP URLs into `streams.sources`.

Use the same `neat` output to set `output.insight.host`, `video_port`, and
`metadata_port` from the reported `videoUDP` and `metadataUDP` ranges. Stream N
uses `video_port + N` and `metadata_port + N`.

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- RTSP sources created in Insight or provided by your cameras.
- Model artifacts are user-managed and live under `assets/models/`. Download the
  default single model, and/or build the per-tier archives (see below).
- On Modalix DevKit, run `bash /usr/bin/fix_devkit_runtime.sh` before starting if
  the runtime has been used by earlier ML/video apps.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

This example is pinned to a single YOLO26n input size (`adaptive.resolutions:
[640]` in `src/common/config.yaml`), so one model pack covers every run.

```bash
mkdir -p assets/models
cd assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<modelzoo-version>/models/modalix/yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz
cd ../..
```

The `pipelines/` bundle uses the int8 pack instead - see
[`pipelines/README.md`](pipelines/README.md):

```bash
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<modelzoo-version>/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz
```

`<modelzoo-version>` is the `modelzoo-version` field in `deps/manifest.json`.

## Supported Models
Use the model zoo version wherever `<modelzoo-version>` appears.

The default single model is `yolo26n-det-bf16-mla_tess-b1.tar.gz`, used for every
tier until dedicated tier archives exist (so the app runs immediately, with tier
switches becoming cosmetic).

Download the default model:

```bash
mkdir -p assets/models
cd assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz
cd ../..
```

### Per-resolution tier archives (required for real adaptivity)
The genuine low-res→faster / high-res→better-recall behaviour requires one
MLA-compiled archive per input size. `tools/build_yolo26_tiers.sh` builds YOLO26
at each size and writes:

- `assets/models/yolo26n-320-det-int8-mla_tess-b1.tar.gz`
- `assets/models/yolo26n-640-det-int8-mla_tess-b1.tar.gz`
- `assets/models/yolo26n-960-det-int8-mla_tess-b1.tar.gz`

**Supported input sizes:** any multiple of 32. This example defaults to
**320, 640, 960**; edit `adaptive.resolutions` and `model.tiers` together to use
others. Point each `model.tiers` entry at its archive to enable real tier
switching.

Validated on Modalix: the three tiers compile to genuinely different MLA compute
(≈ **0.32M / 0.67M / 1.24M** MLA cycles for 320 / 640 / 960), so a stream at 320
does roughly half the inference work of 640 and 960 does ~1.85×.

**Model variant note:** the tier builder defaults to **yolo26n** because the
core box-decode surgery tool (`yolo26_boxdecode_surgery.py`) bakes in the
attention channel dims of the YOLO26 *nano* head. The wider s/m/l/x variants are
not supported by that surgery tool; use SiMa's pre-compiled packs for those. The
builder derives the input-dependent P3/P4/P5 grid sizes automatically, so any
multiple-of-32 input size works for yolo26n.

## Configure
Edit `examples/object-detection/adaptive-resolution-object-detector/src/common/config.yaml`.

```yaml
model:
  path: <model-path>                 # Single/fallback archive.
  tiers:                             # One MLA archive per input tier (optional).
    320: assets/models/yolo26m-320-det-int8-mla_tess-b1.tar.gz
    640: assets/models/yolo26m-640-det-int8-mla_tess-b1.tar.gz
    960: assets/models/yolo26m-960-det-int8-mla_tess-b1.tar.gz

adaptive:
  resolutions: [640]                 # Single value => FIXED model input (no tier switching).
                                     # Use [320, 640, 960] (matching model.tiers) to also adapt accuracy.
  confidence_low: 0.40               # (model-tier only) Below this, a stream steps up.
  min_object_px: 24                  # (model-tier only) Smaller objects force a higher tier.
  hysteresis_frames: 15              # (model-tier only) Frames to hold before committing a switch.
  budget_units: 12                   # (model-tier only) Shared MLA compute budget across streams.

streams:
  max_streams: 16
  sources:
    - id: cam-1
      rtsp_url: <first-rtsp-url-copied-from-insight>
    - id: cam-2
      rtsp_url: <second-rtsp-url-copied-from-insight>

output:
  adaptive:
    heights: [2160, 1080, 720, 480]  # Candidate delivered heights (clamped to source native).
    budget_megapixels_per_s: 280     # Total output pixel-rate fair-shared across active streams.
  insight:
    host: <insight-host-ip>          # Host running Insight.
    video_port: <videoUDP start>     # Stream N -> video_port + N.
    metadata_port: <metadataUDP start> # Stream N -> metadata_port + N.
```

## Run

Both languages ship the same two topologies behind one entry point, selected by
`--mode`. Every command below is run from `prebuilt-apps/`.

### C++

```bash
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 examples/object-detection/adaptive-resolution-object-detector/src/cpp/pre-built/adaptive-resolution-object-detector \
  --mode adaptive \
  --config examples/object-detection/adaptive-resolution-object-detector/src/common/config.yaml
```

Use `--mode fused` for the shared-detector topology.

### Python

Activate PyNeat first.

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/adaptive-resolution-object-detector/src/python/requirements.txt
```

### Validate Config Only
A quick smoke test without opening RTSP streams:

```bash
python3 examples/object-detection/adaptive-resolution-object-detector/src/python/main.py \
  --config examples/object-detection/adaptive-resolution-object-detector/src/common/config.yaml \
  --validate-config-only
```

### Adaptive detector - one graph per stream (`--mode adaptive`)
Streams can be added or removed while the others keep running.

```bash
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 python3 examples/object-detection/adaptive-resolution-object-detector/src/python/main.py \
  --mode adaptive \
  --config examples/object-detection/adaptive-resolution-object-detector/src/common/config.yaml
```

### Fused detector - one shared detector for all streams (`--mode fused`)
Higher stream counts, but adding a stream rebuilds the graph. It takes a
different config schema (a bare `streams:` list); the `pipelines/` bundle
generates one, or write it by hand following
[`pipelines/README.md`](pipelines/README.md).

```bash
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 python3 examples/object-detection/adaptive-resolution-object-detector/src/python/main.py \
  --mode fused --config <your-fused-config>.yaml
```

### All three pipelines behind one page
See [`pipelines/README.md`](pipelines/README.md) for the chooser UI that drives
the adaptive detector and the fused detector in three configurations.

## Runtime Control: Add/Remove Streams
While the app is running, edit `streams.sources` in the same `config.yaml` and
save. The app polls the file every `runtime.config_watch_seconds` and diffs by
`id`:
- **Add** a `{id, rtsp_url}` entry → a new Insight video/metadata channel is
  allocated and the stream starts (up to `max_streams`).
- **Remove** an entry → that stream drains, closes, and releases its channel for
  reuse.

No restart is needed. Adding or removing a stream changes every stream's fair
share of the output budget, so the remaining streams step their delivered
resolution **down** (on add) or **up** (on remove); each such change logs
`output <old> -> <new> (rebuilding pipeline)`. Transitions are rate-limited and
staggered per channel, so a fleet-wide step settles over a few seconds rather than
rebuilding all streams at once.

## Rebuild Stability
Both a delivered-resolution change and a model-tier switch rebuild a stream's MLA
pipeline, and the MLA model load happens when the rebuilt pipeline starts. The
same machinery keeps this robust under many streams:
- **Rate limit:** a stream never rebuilds more than once every ~2.5 s — rebuilding
  an MLA pipeline every few frames would thrash the runtime.
- **Serialized builds:** stream pipelines are torn down and rebuilt under a
  global lock so two streams never load MLA models concurrently.
- **Per-channel stagger:** rebuilds are offset by channel so a fleet-wide step
  (e.g. the 16th stream joining) rolls through instead of firing in lockstep.
- **Self-heal:** a transient MLA load error on a rebuild is retried instead of
  aborting the app.
- Keep `hysteresis_frames` at a sane value (default 15) when the model-tier axis
  is enabled, so switches track real scene changes rather than per-frame noise.

Validated on Modalix up to 8 concurrent streams with live add/remove and channel
reuse. Running to the full 16 streams relies on the output budget lowering
per-stream resolution to stay within capacity, and should be re-validated on your
hardware.

## Debugging Notes
- `--validate-config-only` accepts both the rich `streams.sources` form and a
  bare `streams: [url, ...]` list.
- Output-resolution changes log `output <WxH> -> <WxH> (rebuilding pipeline)` and
  happen on stream add/remove; the per-stream startup line shows `native=... output=...`.
- Tier switches log `tier <old> -> <new> (rebuilding pipeline)` (only when the
  optional model-tier axis is enabled); frequent switching means `hysteresis_frames`
  is too low for your scene.
- If a tier archive is missing, the app falls back to `model.path` and logs no
  error — switches then have no speed/accuracy effect.
- `output.debug_dir` + `output.save_every` save periodic annotated frames
  (with an `id · tier · streams` banner) without changing the Insight contract.
- On Modalix DevKit, start with `bash /usr/bin/fix_devkit_runtime.sh`.

## Appendix: Additional Models
Any batch-1 YOLO26 detection pack works as the fallback `model.path` (e.g.
`yolo26n/s/m/l/x-det-bf16-mla_tess-b1.tar.gz`). For real per-tier switching,
compile matching sizes with `tools/build_yolo26_tiers.sh` (yolo26n; see the
model-variant note above).

## Source Files
- Test scope: `tests/test-scope.yaml`
- Python source: `src/python/main.py` (entry point, `--mode adaptive|fused`),
  `src/python/adaptive_app.py`, `src/python/fused_app.py`,
  `src/python/adaptive_policy.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Pipelines bundle: `pipelines/` (see `pipelines/README.md`)
- Model build tool: `tools/build_yolo26_tiers.sh` (optional; only needed to
  produce per-tier packs if you re-enable multi-resolution model switching)

- C++ source: `src/cpp/main.cpp` (entry point, `--mode adaptive|fused`),
  `src/cpp/adaptive_app.h`, `src/cpp/fused_app.h`, `src/cpp/adaptive_policy.h`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`

Both languages expose the same two topologies behind `--mode`, which is what
lets `pipelines/` toggle between them without changing anything else.
- Manual test guide + config generator: `TESTING.md`, `tools/gen_test_config.sh`
- Tier-switching policy + budget reference: `POLICY.md`
- Shared assets: `src/common/`

## Development From Source

To modify or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
