# Multi-Stream Tracker

## Metadata

| Field | Value |
| --- | --- |
| Category | tracking |
| Difficulty | Advanced |
| Tags | object-detection, yolo26, rtsp, multistream, insight, multi-object-tracking, bytetrack |
| Languages | C++, Python |
| Status | stable |
| Binary Name | multi-stream-tracker |
| Model | yolo26m-det-int8-b1 |

## Concept

Tracks up to five object classes across multiple RTSP streams with YOLO26, each class with its own ByteTrack tracker settings. Every track gets a stable ID, sent to Insight with its class label.

## Preview

![Multi-stream tracker preview](../../../portal/assets/examples/tracking/multi-stream-tracker/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix DevKit, or in a Neat SDK container paired with it.
- H.264 or H.265 RTSP sources matching `input.codec`, and an [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the target.
- For H.265, the computer running the Insight viewer must support hardware HEVC decoding; Chromium does not provide a software decoder fallback for WebRTC H.265.

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Only the example itself must run on the DevKit. If the Core and Insight selected by Apps are already installed, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps` to skip reinstalling them. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/tracking/multi-stream-tracker
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

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Replace `<model-file>` with a file from the table.

```bash
export MODELZOO_VERSION="2.1.3"
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

Open `${APP_DIR}/src/common/config.yaml`. Set `model.path` and `model.labels`, add each RTSP URL under `streams`, and set the Insight host and starting video and metadata ports. Set `input.codec` to match the streams.

List the classes to track under `tracking.classes`. The list takes one to five entries. `class` is a name or zero-based class id from the `model.labels` file, which ships as `src/common/coco_label.txt` with one COCO class per line in detector class-id order. Every other key is optional and applies to that class only:

```yaml
tracking:
  classes:
    - class: person
    - class: car
      low_match_iou_threshold: 0.25
      max_missing_frames: 20
      velocity_noise: 0.0125
```

| Key | Default | Effect |
| --- | --- | --- |
| `high_score_threshold` | 0.50 | Detections at or above this score match first. |
| `low_score_threshold` | 0.10 | Detections between the low and high thresholds only keep existing tracks alive. |
| `new_track_threshold` | 0.60 | Minimum score to start a track. |
| `match_iou_threshold` | 0.20 | Minimum IoU between the predicted box and a detection. Lower it for fast objects. |
| `low_match_iou_threshold` | 0.50 | Minimum IoU for low-score matches. |
| `max_missing_frames` | 30 | Frames a lost track waits to be matched again. |
| `min_confirmed_hits` | 2 | Matches before a track is published. |
| `position_noise` | 0.05 | Kalman position noise, relative to box height. |
| `velocity_noise` | 0.00625 | Kalman velocity noise. Raise it for objects that change speed quickly. |

The table lists the built-in defaults, which follow ByteTrack. The shipped `src/common/config.yaml` sets every key for both classes, with score thresholds suited to the default `yolo26m-det-int8-b1` package; a car cannot recover through the built-in `low_match_iou_threshold` of 0.50 at road speed.

The app rejects empty, duplicate (`car` and `2` count as duplicates), unknown, and more than five classes. It also rejects unknown keys. `inference.min_score` must not exceed any class's `low_score_threshold`, or the detector drops the low-score detections the tracker needs.

Set `output.detections_log` to a path prefix to record raw detections per stream for offline tracker comparison.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/multi-stream-tracker \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

## Tracker

Each stream runs one tracker per configured class, so a detection can only match a track of its own class, and tracks never change class. Track IDs are unique within a stream. The C++ and Python trackers are the same algorithm. `tests/common/expected_tracks.json` holds shared replay cases that both unit tests must reproduce exactly; `tools/write_expected_tracks.py` rewrites it.

The tracker follows ByteTrack:

1. A constant-velocity Kalman filter predicts each track's box, using the class's noise settings.
2. High-score detections are matched to tracks by optimal (Hungarian) assignment on IoU with the predicted box. Lost tracks are included, so a track survives short occlusions.
3. Low-score detections can only extend tracks that are still active. This recovers objects whose confidence drops briefly, without creating false tracks.
4. Unmatched high-score detections above `new_track_threshold` start tentative tracks. These are published after `min_confirmed_hits` matches.

### Limits

- A new track is published only after `min_confirmed_hits` matches, and lost tracks are not drawn, so an object appears a frame or two after the detector first sees it.
- There is no appearance model, so an object hidden longer than `max_missing_frames`, or one that crosses a same-class object while both are occluded, gets a new ID.
- Scene cuts start new IDs.

## Troubleshooting

- Start with one stream before scaling to multiple inputs.
- Verify `model.path`, `model.labels`, every RTSP URL, and the Insight port ranges.
- Set either inflight limit to `-1` to use the Core default.
- Use `output.debug_dir` and `output.save_every` to save sampled overlays. They are drawn on the latest decoded frame, which can be a few frames newer than the detections, so fast objects can look offset. Insight matches metadata to video by timestamp and does not have this offset.
- All streams must have the same resolution.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- C++ tracker helpers: `src/cpp/utils/`
- Python source: `src/python/main.py`
- Python tracker helpers: `src/python/utils/`
- Development tools (source checkout only): `tools/`
- Shared runtime files: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
