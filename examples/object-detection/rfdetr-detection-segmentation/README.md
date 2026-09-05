# RF-DETR Detection and Segmentation

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, instance-segmentation, rfdetr, rtsp, insight |
| Languages | C++, Python |
| Status | stable |
| Binary Name | rfdetr-detection-segmentation |
| Model | RF-DETR Small, Medium, or Segmentation Medium |

## Concept

Run RF-DETR detection or instance segmentation on one H.264, H.265, or MJPEG RTSP stream and view the result in Insight.

The application decodes to NV12 once. EV74 converts, resizes, and normalizes each frame for the selected backbone. A one-frame queue drops stale decoded frames if inference falls behind. Host code then selects the strongest proposals and passes the matching boxes and feature tensor to the transformer. Insight receives the source video and matching detection boxes or segmentation polygons.

## Preview

![RF-DETR detection and segmentation preview](../../../portal/assets/examples/object-detection/rfdetr-detection-segmentation/image.jpg)

## Prerequisites

- [`sima-cli` 2.1.15 or newer](https://developer.sima.ai/software/tools/sima-cli/) on a supported Modalix or DevKit target.
- An H.264, H.265, or MJPEG RTSP source and an [Insight endpoint](https://developer.sima.ai/software/tools/insight/) reachable from the target.

## Install Apps

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/rfdetr-detection-segmentation
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Prepare the model directory, then download the pair for your selected task:

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
```

Small detection is selected by default:

```bash
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/rfdetr-small-backbone.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/rfdetr-small-transformer.tar.gz"
```

For Medium detection:

```bash
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/rfdetr-medium-backbone.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/rfdetr-medium-transformer.tar.gz"
```

For segmentation, download the RF-DETR Segmentation Medium model pair:

```bash
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/rfdetr-seg-medium-backbone.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/rfdetr-seg-medium-transformer.tar.gz"
```

Return to `prebuilt-apps/` after downloading your model pair:

```bash
cd ..
```

## Prepare an RTSP source

1. In Insight, open **Media Sources** and import a video from the catalog or your own file.
2. Assign the video under **Streaming Sources** and start the source.
3. Copy its RTSP URL into `source.rtsp_url` and set `source.codec` to match the video. Use a host and published port reachable from the target, not `localhost`.

## Configure

Edit `$APP_DIR/src/common/config.yaml`:

- Set `model.task` to `detection` or `segmentation`.
- For detection, set `model.detection.variant` to `small` or `medium`.
- Set `source.rtsp_url` and select `source.codec` as `h264`, `h265`, or `mjpeg`.
- Leave `source.width`, `height`, and `fps` at `0` to probe the stream. Width and height are fallbacks; a positive FPS overrides the detected value.
- Set `output.insight.host`, `video_port`, and `metadata_port` to the values reported by Insight.
- Keep `inference.frames: 0` to run continuously, or set a finite result count.

EV74 resizes the decoded frame to the input size required by the selected model.

## Run

### C++

```bash
"$APP_DIR/src/cpp/pre-built/rfdetr-detection-segmentation" --config "$APP_DIR/src/common/config.yaml"
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r "$APP_DIR/src/python/requirements.txt"
python3 "$APP_DIR/src/python/main.py" --config "$APP_DIR/src/common/config.yaml"
```

Insight receives `object-detection` metadata for detection or `segmentation` polygon metadata for segmentation. Stop a continuous run with Ctrl-C.

## Performance

Measured end-to-end throughput on Modalix. The 720p MJPEG row reports Python / C++ results from a 90 FPS source over 300 completed outputs; the other rows summarize validated limits.

| Input Resolution | Codec | Detection Small | Detection Medium | Segmentation Medium |
| --- | --- | ---: | ---: | ---: |
| 720p | H.264, H.265 | Up to 70 FPS | Up to 50 FPS | Up to 39 FPS |
| 720p | MJPEG | 68.9 / 68.9 FPS | 50.1 / 50.0 FPS | 38.6 / 38.7 FPS |
| 1080p | H.264, H.265 | Up to 70 FPS | Up to 50 FPS | Up to 39 FPS |
| 1080p | MJPEG | Up to 70 FPS | Up to 50 FPS | Not verified |
| 4K | H.264 | Up to 60 FPS | Up to 50 FPS | Up to 39 FPS |

## Source Files

- C++ implementation: `src/cpp/main.cpp`
- Python implementation: `src/python/main.py`
- Shared configuration and COCO labels: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
