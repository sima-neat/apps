# Multi-Stream Object Tracker

## Metadata

| Field | Value |
| --- | --- |
| Category | tracking |
| Difficulty | Advanced |
| Tags | object-detection, yolo26, rtsp, multistream, insight, tracking, tiny-drone |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multi-stream-people-tracker |
| Model | yolo26m-det-int8-b1 |

## Concept

Track a configured object class across multiple mixed-resolution RTSP inputs.
The default profile tracks people. The optional tiny-drone profile uses a
one-class YOLO26n-P2 INT8 QAT model, two-stage score association, and
constant-velocity matching for boxes that may move far enough to have zero IoU
between frames. Track IDs remain local to each stream, and live video plus
metadata is published to Insight.

## Preview

![Multi-stream people tracker preview](../../../portal/assets/examples/tracking/multi-stream-people-tracker/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- H.264 or H.265 RTSP sources matching `input.codec`, and an [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the target.
- For H.265, the computer running the Insight viewer must support hardware HEVC decoding; Chromium does not provide a software decoder fallback for WebRTC H.265.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model file | Role |
| --- | --- |
| `yolo26m-det-int8-b1.tar.gz` | Default |
| `yolo26n-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26s-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26l-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26x-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-det-bf16-b1.tar.gz` | Supported |

The tiny-drone profile expects
`yolo26n-p2-tiny-drone-int8-qat-b1.tar.gz`. This is a custom, one-class model,
not a current public Model Zoo download. Generate and qualify it with the
`training/yolo26_tiny_drone` recipe in the Models repository before selecting
`src/common/tiny-drone.yaml`. The package must expose four raw bbox heads
followed by four class-logit heads for P2 through P5; Neat derives the four
strides from the model output dimensions. As in the YOLO detector example,
input width, height, normalization, and letterboxing come from model
preprocessing; do not duplicate them in this app configuration.

The standard model packages come from the Model Zoo release below, which can
differ from the installed platform version. Replace `<model-file>` with a file
from the table.

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/<model-file>"
cd ..
```

Set `model.path` in the config to the downloaded package.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input streams and render tracking metadata. Install videos directly from the Insight catalog or through Insight's YouTube support.

In the Insight Web UI, start the required streams and copy their RTSP URLs into `streams`. Use the host and UDP port ranges reported by `neat` for the output settings.

## Configure

Edit `examples/tracking/multi-stream-people-tracker/src/common/config.yaml`.

```yaml
model:
  path: <model-path>

streams:
  - <first-rtsp-url>
  - <second-rtsp-url>

input:
  codec: h264  # h264/avc or h265/hevc

inference:
  frames: 0
  max_inflight_per_stream: 4
  max_inflight_total: 16
  target_class_id: 0
  target_label: person
  min_score: 0.30

tracking:
  high_score_threshold: 0.30
  new_track_threshold: 0.30
  match_iou_threshold: 0.10
  max_center_distance: 2.5
  max_missing_frames: 15

output:
  insight:
    host: <insight-host-ip>
    video_port_base: <videoUDP-start-port>
    metadata_port_base: <metadataUDP-start-port>
```

The legacy `tracking.iou_threshold` key remains supported. When it is used
without `tracking.max_center_distance`, matching remains IoU-only; set the
center-distance value explicitly or use `match_iou_threshold` to enable the
motion-aware OR gate.

For the qualified one-class drone model, copy
`src/common/tiny-drone.yaml`, then set its model path, RTSP URL, and Insight
host. Its low decoder floor is intentional: detections below the high-score
threshold may recover an existing track but cannot create a new one.

## Run

### C++

```bash
./examples/tracking/multi-stream-people-tracker/src/cpp/pre-built/multi-stream-people-tracker \
  --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/tracking/multi-stream-people-tracker/src/python/requirements.txt
python3 examples/tracking/multi-stream-people-tracker/src/python/main.py \
  --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
```

## Troubleshooting

- Start with one stream before scaling to multiple inputs.
- Verify `model.path`, every RTSP URL, and the Insight port ranges.
- Set either inflight limit to `-1` to use the Core default.
- Use `output.debug_dir` and `output.save_every` to save sampled overlays.
- Do not use the tiny-drone thresholds with a generic COCO model; the extra
  low-score candidates add unnecessary postprocessing work.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- C++ tracker helpers: `src/cpp/utils/`
- Python source: `src/python/main.py`
- Python tracker helpers: `src/python/utils/`
- Shared runtime files: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Accuracy Qualification

Evaluate one stream at a time. Ground truth JSONL uses frame dimensions and
`objects`; predictions may be plain `tracks` JSONL or captured Insight
metadata envelopes. Both use pixel-space `[x, y, width, height]` boxes and
numeric frame IDs. Ground truth must contain every evaluated frame, including
frames with an empty `objects` array; predictions outside that annotated set
are rejected so coverage and false-positive rates cannot be diluted. Each
ground-truth object may contain `track_id`, and each prediction may contain
`id`. Non-empty IDs are required on every object/track
when either ID-switch or fragmentation gate is requested, and IDs must be
unique within each frame. Without complete, per-frame-unique IDs, detection
metrics remain available while tracking metrics are reported as unavailable
(`null`) and any requested tracking gate fails closed.

```bash
python3 examples/tracking/multi-stream-people-tracker/src/python/evaluate_tracking.py \
  --ground-truth <ground-truth.jsonl> \
  --predictions <insight-metadata.jsonl> \
  --output <accuracy-report.json> \
  --fps <source-fps> \
  --minimum-frames <required-frames> \
  --minimum-recall <required-recall> \
  --minimum-tiny-recall <required-tiny-recall> \
  --maximum-false-positives-per-minute <allowed-fp-per-minute> \
  --maximum-id-switches <allowed-id-switches> \
  --maximum-fragmentations <allowed-fragmentations>
```

Tiny/small/medium/large buckets are measured after the same 640-pixel model
input scale, so the tiny-object recall gate reflects what the detector sees.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
Source validation includes the evaluator and its JSONL/Insight envelope tests.
