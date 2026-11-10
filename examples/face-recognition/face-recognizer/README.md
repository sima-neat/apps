# Face Recognizer

## Metadata

| Field | Value |
| --- | --- |
| Category | face-recognition |
| Difficulty | Advanced |
| Tags | face-detection, face-recognition, scrfd, arcface, rtsp, usb-camera, enrollment, bf16, mla-tessellation |
| Languages | C++, Python |
| Status | stable |
| Binary Name | face-recognizer |
| Model | scrfd_2.5g_bnkps.mla, w600k_r50.surgery |

## Concept

Real-time face detection and recognition on Modalix with SCRFD and ArcFace, both BF16 MLA-tessellated. Every face is recognized on every frame; a 720p video with two faces runs at about 45 FPS.

## Preview

![Face recognizer preview](../../../portal/assets/examples/face-recognition/face-recognizer/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- An H.264 RTSP stream, an H.264 video file, or a USB (UVC) MJPEG camera plugged into the target as input.
- An [Insight](https://developer.sima.ai/software/tools/insight/) host reachable from the target for viewing the live video with face boxes and names (optional).

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/face-recognition/face-recognizer
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Download the pre-compiled model packages from the Model Zoo:

```bash
export MODELZOO_VERSION="3.0.0"
mkdir -p ${APP_DIR}/models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/scrfd_2.5g_bnkps.mla_mpk.tar.gz" -d ${APP_DIR}/models/
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/w600k_r50.surgery_mpk.tar.gz" -d ${APP_DIR}/models/
```

## Enroll Faces

Build a gallery before running recognition. Each invocation is additive — run once per person. When re-enrolling an existing identity, sample counts from the previous enrollment are preserved so centroids stay correctly weighted.

```bash
# From a video clip:
${APP_DIR}/src/cpp/pre-built/face-recognizer --enroll \
    --config ${APP_DIR}/src/common/config.yaml \
    --video  /path/to/alice.mp4 --name "Alice" \
    --gallery ${APP_DIR}/gallery.bin

# From an image folder (one subdirectory per identity):
#   gallery_images/Alice/photo1.jpg, photo2.jpg …
${APP_DIR}/src/cpp/pre-built/face-recognizer --enroll \
    --config ${APP_DIR}/src/common/config.yaml \
    --images gallery_images/ \
    --gallery ${APP_DIR}/gallery.bin
```

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set:

- `input.uri` — an H.264 RTSP stream (`rtsp://<HOST>:<PORT>/<STREAM>`) or an H.264 video file. For a USB camera, leave it empty and set `input.usb` instead (see below).
- `gallery.path` — the gallery built in [Enroll Faces](#enroll-faces) (default `gallery.bin`).
- `output.insight.host` — the Insight host IP to view live results from an RTSP input; leave empty to run headless.

To use a USB camera, set `input.usb.camera` to its stable ID, and `input.usb.width`, `input.usb.height`, and `input.usb.fps` to an MJPEG mode the camera lists. List the connected camera IDs and modes on the target:

```bash
source ~/pyneat/bin/activate
python3 -c "import pyneat; print(pyneat.usb_camera_ids())"
v4l2-ctl --list-devices && v4l2-ctl -d /dev/video<N> --list-formats-ext
```

Insight receives the stream as the source sent it, with no re-encoding: H.264 from an RTSP stream, JPEG (as RTP/JPEG) from a USB camera. RTP/JPEG carries sizes up to 2040 pixels per side in multiples of 8. The same stream is decoded on the hardware decoder for face detection; a USB camera decodes in its own sampling (most webcams send 4:2:2 MJPEG, which decodes to NV16). An unknown ID or an unsupported mode fails at startup with the connected IDs or the camera's MJPEG modes; close other programs that hold the camera.

The remaining settings are optional; see [Tuning](#tuning).

## Run

**C++:**

```bash
./${APP_DIR}/src/cpp/pre-built/face-recognizer \
    --config ${APP_DIR}/src/common/config.yaml
```

**Python** (requires `pyneat`, `numpy`, `opencv-python`, and `PyYAML` on the device):

```bash
python3 ${APP_DIR}/src/python/main.py \
    --config ${APP_DIR}/src/common/config.yaml
```

Python enrollment uses the same flags as C++: `python3 ${APP_DIR}/src/python/main.py --enroll --images gallery_images/ --gallery ${APP_DIR}/gallery.bin`.

With an RTSP or USB camera input and `output.insight.host` set, both implementations send the video and each face's box, name, and similarity to Insight on channel 0, and Insight draws the boxes. Video files run offline and print the recognized names every 100 frames; `--test` prints them for every frame.

**Optional overrides:**

| Flag | Description |
|---|---|
| `--input <uri>` | Use this RTSP URL or video file instead of `input.uri` or `input.usb.camera` |
| `--gallery <path>` | Override `gallery.path` in config |
| `--scrfd-model <path>` | Override `scrfd.model` in config |
| `--arcface-model <path>` | Override `arcface.model` in config |
| `--rtsp-fps <n>` | C++ only: force the RTSP frame rate when the stream does not report one |
| `--max-frames <n>` | Stop after N frames (0 = unlimited) |
| `--test` | Print the result of every frame; disables Insight output |

**Enrollment flags (with `--enroll`):**

| Flag | Description |
|---|---|
| `--video <path>` | Enrollment video; requires `--name` |
| `--name <name>` | Identity label for `--video` mode |
| `--images <dir>` | Enrollment image folder (one subdirectory per identity) |
| `--gallery <path>` | Gallery file to write or append to |
| `--sample-every <n>` | Sample 1 frame every N from video (default: 5) |
| `--min-score <f>` | Minimum SCRFD confidence for enrollment (default: 0.75) |
| `--max-per-person <n>` | Cap images per identity when using `--images` (default: unlimited) |

## Tuning

| Parameter | Default | Notes |
|---|---|---|
| `match.threshold` | `0.55` | Raise to 0.60–0.65 in controlled lighting for fewer false positives |
| `match.margin` | `0.12` | Raise to 0.15–0.20 when multiple people are enrolled to reduce cross-ID errors |
| `scrfd.conf_threshold` | `0.65` | Lower to detect smaller or partially-occluded faces |
| `scrfd.keep_top_k` | `100` | Most faces recognized per frame. Each face adds one ArcFace run (about 10 ms), so lower it to bound the frame time in crowded scenes |

## Source Files

- C++ recognition and enrollment source: `src/cpp/main.cpp`
- Python recognition and enrollment source: `src/python/main.py`
- Shared runtime files: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
