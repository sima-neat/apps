# Face Recognizer

## Metadata

| Field | Value |
| --- | --- |
| Category | face-recognition |
| Difficulty | Advanced |
| Tags | face-detection, face-recognition, scrfd, arcface, rtsp, enrollment, bf16, mla-tessellation |
| Languages | C++, Python |
| Status | stable |
| Binary Name | face-recognizer |
| Model | scrfd_2.5g_bnkps.mla, w600k_r50.surgery |

## Concept

Real-time face detection and recognition on Modalix with SCRFD and ArcFace, both BF16 MLA-tessellated, at about 85–90 FPS on a 720p RTSP stream with three faces in view.

## Preview

![Face recognizer preview](../../../portal/assets/examples/face-recognition/face-recognizer/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- An H.264 RTSP stream or H.264 video file as input.
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
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/scrfd_2.5g_bnkps.mla_mpk.tar.gz" -o ${APP_DIR}/models/
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/w600k_r50.surgery_mpk.tar.gz" -o ${APP_DIR}/models/
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

- `input.uri` — an H.264 RTSP stream (`rtsp://<HOST>:<PORT>/<STREAM>`) or an H.264 video file.
- `gallery.path` — the gallery built in [Enroll Faces](#enroll-faces) (default `gallery.bin`).
- `output.insight.host` — the Insight host IP to view live results from an RTSP input; leave empty to run headless.

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

Python enrollment uses the same flags as C++: `python3 ${APP_DIR}/src/python/main.py --enroll --video <clip> --name <name> --gallery <gallery.bin>`.

With an RTSP input and `output.insight.host` set, both implementations send the decoded video and each face's box, name, and similarity to Insight on channel 0, and Insight draws the boxes. Video files run offline: the recognized names are printed, and the C++ binary can record annotated frames with `--output`. `--test` adds per-frame results and an FPS report.

**Optional overrides:**

| Flag | Description |
|---|---|
| `--input <uri>` | Override `input.uri` in config |
| `--gallery <path>` | Override `gallery.path` in config |
| `--scrfd-model <path>` | Override `scrfd.model` in config |
| `--arcface-model <path>` | Override `arcface.model` in config |
| `--rtsp-fps <n>` | C++ only: force the RTSP frame rate when the stream does not report one |
| `--max-frames <n>` | Stop after N frames (0 = unlimited) |
| `--test` | Print per-frame results and an FPS report; disables Insight output |
| `--output <path>` | C++ only: record annotated frames to `.mp4`, or the first frame to `.jpg`/`.png` |

**Enrollment flags (with `--enroll`):**

| Flag | Description |
|---|---|
| `--video <path>` | Enrollment video; requires `--name` |
| `--name <name>` | Identity label for `--video` mode |
| `--images <dir>` | Enrollment image folder (one subdirectory per identity; C++ only) |
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
| `runtime.recog_interval` | `8` | Frames between re-embeddings (lower = more responsive identity updates) |
| `overlay.draw_landmarks` | `false` | C++ only: set `true` to draw the 5 SCRFD facial keypoints in `--output` recordings |

## Source Files

- C++ recognition source: `src/cpp/main.cpp`
- C++ enrollment source: `src/cpp/enroll.cpp`
- Python recognition and enrollment source: `src/python/main.py`
- Shared runtime files: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
