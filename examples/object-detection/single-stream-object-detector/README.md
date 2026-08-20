# Single-Stream Object Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, yolo26, rtsp, insight |
| Languages | C++, Python |
| Status | stable |
| Binary Name | single-stream-object-detector |
| Model | yolo26m-det-bf16-mla_tess-b1 |

## Concept

Detects objects in one RTSP or MJPEG stream with YOLO26 and sends synchronized H.264 video and detection metadata to Insight.

The decoded frame branches inside a single graph to the detector and to the H.264 sender. Both outputs therefore carry timestamps from the same frame, which is what lets Insight draw each detection on the frame it came from. Setting `output.save_dir` adds a third branch that returns the decoded frame to the application so it can write annotated JPEGs.

## Preview

![Single-stream object detector preview](../../../portal/assets/examples/object-detection/single-stream-object-detector/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- An RTSP H.264, RTSP H.265, RTSP MJPEG, or HTTP MJPEG source and an [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the target.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/single-stream-object-detector
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model file | Role |
| --- | --- |
| `yolo26m-det-bf16-mla_tess-b1.tar.gz` | Default |
| `yolo26n-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26s-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26l-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26x-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-det-bf16-b1.tar.gz` | Supported |
| `yolo26m-det-int8-b1.tar.gz` | Supported |

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Replace `<model-file>` with a file from the table.

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/<model-file>"
cd ..
```

Set `model.path` in the config to the downloaded package.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input stream and render the video and detection metadata. Install videos directly from the Insight catalog or through Insight's YouTube support.

In the Insight Web UI, start the required stream and copy its source URL. Use RTSP for H.264 or H.265; for MJPEG, Insight supports both RTSP and HTTP URLs. Set `source.codec` to `h264`/`avc`, `h265`/`hevc`, or `mjpeg`. Decoded frames are encoded as H.264 for Insight output.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Set `model.path`, the source type, codec, and URL, and the Insight host and video and metadata ports.

The source supports RTSP H.264, H.265, and MJPEG, plus HTTP MJPEG. Keep `inference.frames` at `0` to run until you stop the application.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/single-stream-object-detector \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

## Troubleshooting

- Verify stream reachability if the first frame times out.
- Verify the Insight host and UDP ports if video and detections are absent.
- Check model loading and score thresholds when video arrives without boxes.
- The startup probe adapts the decode path when source resolution changes.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
