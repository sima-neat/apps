# Multi-Stream Pose Estimator

## Metadata

| Field | Value |
| --- | --- |
| Category | pose-estimation |
| Difficulty | Advanced |
| Tags | pose-estimation, keypoints, rtsp, multistream, insight, yolo26 |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multi-stream-pose-estimator |
| Model | yolo26m-pose-int8-b1 |

## Concept

This example runs a config-driven multi-stream RTSP pose pipeline and publishes video plus 17-keypoint COCO pose metadata for each stream to Insight.

Every stream feeds one shared pose model. Each decoded branch is admitted through a real-time, keep-latest link, so an overloaded stream drops frames instead of building a backlog.

## Preview

![Multi-stream pose estimator preview](../../../portal/assets/examples/pose-estimation/multi-stream-pose-estimator/image.png)

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
| `yolo26m-pose-int8-b1.tar.gz` | Default |
| `yolo26n-pose-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26s-pose-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-pose-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26l-pose-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26x-pose-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-pose-bf16-b1.tar.gz` | Supported |

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Replace `<model-file>` with a file from the table.

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-pose/<model-file>"
cd ..
```

Set `model.path` in the config to the downloaded package.

These pose packages are published as direct download artifacts and are not indexed in the Model Zoo catalog, so they cannot be resolved by name. Use the download command above.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input streams and render each output channel. Install videos directly from the Insight catalog or through Insight's YouTube support.

In the Insight Web UI, start the required streams and copy their RTSP URLs into `streams`. Use the host and UDP port ranges reported by `neat` for the output settings.

Insight renders the skeleton from the published `pose-estimation` metadata and hides any joint at or below `0.3` confidence. Published metadata always carries all 17 keypoints; `output.min_keypoint_visibility` only controls saved debug overlays.

## Configure

Edit `examples/pose-estimation/multi-stream-pose-estimator/src/common/config.yaml`.

```yaml
model:
  path: <model-path>

streams:
  - <first-rtsp-url>
  - <second-rtsp-url>

input:
  codec: h264  # h264/avc or h265/hevc
  tcp: true
  max_width: 2560
  max_height: 1440

inference:
  frames: 0
  max_inflight_per_stream: 4
  max_inflight_total: 16
  min_score: 0.30
  max_poses: 50

output:
  insight:
    host: <insight-host-ip>
    video_port_base: <videoUDP-start-port>
    metadata_port_base: <metadataUDP-start-port>
  min_keypoint_visibility: 0.30
```

## Run

Validate the config without opening streams:

```bash
./examples/pose-estimation/multi-stream-pose-estimator/src/cpp/pre-built/multi-stream-pose-estimator \
  --config examples/pose-estimation/multi-stream-pose-estimator/src/common/config.yaml \
  --validate-config-only
```

### C++

```bash
./examples/pose-estimation/multi-stream-pose-estimator/src/cpp/pre-built/multi-stream-pose-estimator \
  --config examples/pose-estimation/multi-stream-pose-estimator/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/pose-estimation/multi-stream-pose-estimator/src/python/requirements.txt
python3 examples/pose-estimation/multi-stream-pose-estimator/src/python/main.py \
  --config examples/pose-estimation/multi-stream-pose-estimator/src/common/config.yaml
```

## Output Metadata

Each stream publishes `pose-estimation` metadata. Every pose carries the person box, the person score, and 17 named COCO keypoints:

```json
{"poses": [{"id": "pose_1", "label": "person", "confidence": 0.77,
            "bbox": [163, 403, 130, 315],
            "keypoints": [{"name": "nose", "x": 238, "y": 421, "confidence": 0.99}]}]}
```

Keypoint coordinates are in source-frame pixels, matching the box.

## Troubleshooting

- Replace all placeholder stream URLs and the Insight host before running.
- This example supports up to four active streams.
- Set either inflight limit to `-1` to use the Core default.
- Verify host and UDP port ranges if Insight receives no output.
- If skeletons look sparse, lower `output.min_keypoint_visibility`; occluded joints are reported with low visibility by design.
- Use `output.debug_dir`, `output.save_every`, and profiling output for diagnosis.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared runtime files: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
