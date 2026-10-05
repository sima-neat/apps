# USB Camera Object Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, yolo26, usb-camera, uvc, v4l2, insight |
| Languages | C++, Python |
| Status | stable |
| Binary Name | usb-camera-object-detector |
| Model | yolo_26n_mpk |

## Concept

Captures video from a USB camera, detects objects with YOLO26, and streams video with detection overlays to Insight.

## Preview

![USB camera detections rendered in Insight](../../../portal/assets/examples/object-detection/usb-camera-object-detector/image.png)

## Prerequisites

- A supported Modalix or DevKit target with [sima-cli](https://developer.sima.ai/software/tools/sima-cli/).
- A USB camera that supports MJPEG (`MJPG`) and `v4l2-ctl` for checking its modes.
- A reachable [Insight](https://developer.sima.ai/software/tools/insight/) receiver.

## Install Apps

Run on the target:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/usb-camera-object-detector
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Start with YOLO26n from Model Zoo:

```bash
mkdir -p models
cd models
sima-cli modelzoo --version 2.1.3 --boardtype modalix get yolo_26n
cd ..
```

The downloaded model is `models/yolo_26n_mpk.tar.gz`. Use this path in the config.
Model Zoo version `2.1.3` comes from the Apps dependency manifest.

<details>
<summary>Other supported YOLO26 detection models</summary>

| Model | Role | Source |
| --- | --- | --- |
| `yolo26m-det-bf16-mla_tess-b1.tar.gz` | Supported | Direct artifact |
| `yolo26n-det-bf16-mla_tess-b1.tar.gz` | Supported | Direct artifact |
| `yolo26s-det-bf16-mla_tess-b1.tar.gz` | Supported | Direct artifact |
| `yolo26l-det-bf16-mla_tess-b1.tar.gz` | Supported | Direct artifact |
| `yolo26x-det-bf16-mla_tess-b1.tar.gz` | Supported | Direct artifact |
| `yolo26m-det-bf16-b1.tar.gz` | Supported | Direct artifact |
| `yolo26m-det-int8-b1.tar.gz` | Supported | Direct artifact |

For these direct artifacts, replace `<model-file>` with a filename from the table:

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/<model-file>"
cd ..
```

For example, the medium bf16 and int8 download commands, run from `models/`, are:

```bash
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26m-det-int8-b1.tar.gz"
```

Set `model.path` to the package you downloaded. For `yolo26m-det-int8-b1.tar.gz`,
start with `inference.min_score: 0.15` because its confidence scores are lower.
Classification and segmentation models are not supported.

</details>

## Configure

**1. Find your camera.** Plug it in and run:

```bash
v4l2-ctl --list-devices
```

Find your USB camera by name. Copy a device path listed beneath it, then replace
`<camera-device>` in this command:

```bash
v4l2-ctl --device "<camera-device>" --list-formats-ext
```

Use a node whose output includes `MJPG` (MJPEG). If it lists no image formats,
try the next node under the same camera. Choose a **width, height, and FPS listed
under `MJPG`**. Device numbers can change after reconnecting the camera.

**2. Edit `${APP_DIR}/src/common/config.yaml`.** Set these values:

| Setting | Value to use |
| --- | --- |
| `model.path` | `models/yolo_26n_mpk.tar.gz` |
| `source.device` | The camera device you found above |
| `source.width`, `source.height`, `source.fps` | One MJPEG mode supported by your camera |
| `output.insight.host` | IP address of your Insight receiver |
| `output.insight.video_port` | Insight's video UDP port (default `9000`) |
| `output.insight.metadata_port` | Insight's metadata UDP port (default `9100`) |

The shipped `1920 × 1080 @ 30` values are only defaults. Change all three camera
settings to match your device; 1080p and 90 FPS are not requirements. The app
checks the requested mode at startup and reports an error if it is unsupported.

Keep the other settings unchanged to start. For an upside-down camera, set
`source.flip: rotate-180`.

## Run

Run one implementation at a time. Both use the same config.

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/usb-camera-object-detector \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

Open Insight to view the video and detection overlays. The first run may take
longer while the model is unpacked. Press **Ctrl+C** to stop.

## Troubleshooting

- **Camera cannot open or mode is unsupported:** repeat camera discovery and select an `MJPG` mode shown by `--list-formats-ext`.
- **No video or overlays in Insight:** check the receiver IP address and both UDP ports.
- **Low frame rate or incomplete JPEG errors:** try a lower resolution or FPS supported by the camera. Set `runtime.profile: true` to see the achieved output rate.

Add `--validate-config-only` to either run command to check config values without
opening the camera or loading the model. It does not check camera capabilities.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config and labels: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under
`src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).

Both implementations use Neat `Input`, `JpegParse`, and `SimaDecode` nodes for
hardware MJPEG decoding. `source.override_fragment` is an optional diagnostic
NV12 source; leave it empty for USB capture. End-to-end tests use a fixed image
and require `ffprobe` to validate received video alongside detection metadata.
