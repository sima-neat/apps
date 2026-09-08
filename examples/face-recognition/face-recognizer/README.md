# Face Recognizer

## Metadata

| Field | Value |
| --- | --- |
| Category | face-recognition |
| Difficulty | Advanced |
| Tags | face-detection, face-recognition, scrfd, arcface, rtsp, enrollment, bf16, mla-tessellation |
| Languages | C++ |
| Status | stable |
| Binary Name | face-recognizer |
| Model | scrfd_2.5g_bnkps.mla, w600k_r50.surgery |

## Concept

Real-time face detection and recognition on SiMa.ai Modalix hardware using SCRFD face detection and ArcFace embeddings, both running BF16 MLA-tessellated on the MLA at over 40 FPS.

## Preview

![Face recognizer preview](../../../portal/assets/examples/face-recognition/face-recognizer/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- An H.264 RTSP source for the input stream.
- An [Insight](https://developer.sima.ai/software/tools/insight/) host reachable from the target for viewing the annotated output stream (optional).

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
export MODELZOO_VERSION="2.1.3"
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

Open `${APP_DIR}/src/common/config.yaml`. Key settings:

```yaml
gallery:
  path: gallery.bin           # Enrollment data; build with --enroll first

input:
  uri: rtsp://<HOST>:<PORT>/<STREAM>   # RTSP source

output:
  insight:
    host: ""                  # Set to Insight host IP to stream annotated output
    video_port: 29656         # Docker-mapped UDP port for Insight video input
    metadata_port: 23838      # Docker-mapped UDP port for Insight metadata input

match:
  threshold: 0.55             # Cosine similarity cutoff; below → Unknown
  margin:    0.12             # Min gap between best and 2nd-best score

runtime:
  recog_interval: 8           # Re-embed every N frames
```

## Run

```bash
./${APP_DIR}/src/cpp/pre-built/face-recognizer \
    --config ${APP_DIR}/src/common/config.yaml
```

The RTSP input URI, gallery path, model paths, and output options are all read from `config.yaml`.

**Optional overrides:**

| Flag | Description |
|---|---|
| `--input <uri>` | Override `input.uri` in config |
| `--gallery <path>` | Override `gallery.path` in config |
| `--scrfd-model <path>` | Override `scrfd.model` in config |
| `--arcface-model <path>` | Override `arcface.model` in config |
| `--stream-host <ip>` | Send annotated H.264 stream to a custom UDP receiver instead of Insight |
| `--stream-port <n>` | UDP port for custom receiver (default: 5000) |
| `--rtsp-fps <n>` | Force decoder FPS; omit to auto-detect from stream (recommended) |
| `--max-frames <n>` | Stop after N frames (0 = unlimited) |
| `--test` | Print per-frame results and FPS report; headless |
| `--cpu-preproc` | Force CPU NEON preproc instead of EV74 CVU (slower; useful for A/B accuracy comparison) |

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
| `runtime.recog_interval` | `8` | Frames between re-embeddings (lower = more responsive identity updates) |

## Testing

Tests require a source build (see [Development From Source](#development-from-source) below).
Unit tests need no hardware; the E2E test requires models and an input source — all three
env vars must be set or the test fails.

```bash
# Set required prerequisites
export SIMANEAT_APPS_TEST_MODELS_DIR=${APP_DIR}/models
export SIMANEAT_TEST_RTSP_H264_URL=rtsp://<HOST>:<PORT>/<STREAM>
# Optional — enables recognition identity check in addition to detection check:
export SIMANEAT_APPS_TEST_GALLERY_BIN=${APP_DIR}/gallery.bin

# Unit tests — no hardware required
ctest --test-dir build -L unit -R 'face-recognizer' --output-on-failure -V

# E2E test — requires models and input source above
ctest --test-dir build -L e2e -R 'face-recognizer' --output-on-failure -V
```

## Source Files

- C++ recognition source: `src/cpp/main.cpp`
- Enrollment source: `tools/enroll.cpp`
- Shared runtime files: `src/common/`
- Model scripts: `src/common/model/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).

<details>
<summary>Build commands</summary>

Inside the SDK container:

```bash
source /opt/bin/simaai-init-build-env modalix
cd /path/to/apps
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSIMANEAT_APPS_BUILD_CPP=ON
cmake --build build --target face-recognizer -j4
```

Binary: `build/examples/face-recognition/face-recognizer_cpp/face-recognizer`

### Recompile Models from Source

Model preparation and compilation scripts are under
`examples/face-recognition/face-recognizer/src/common/model/`:

```bash
APP=examples/face-recognition/face-recognizer

# Step 1 — Apply graph surgery to ArcFace (required for MLA compatibility)
python3 ${APP}/src/common/model/arcface_to_mla.py \
    --input  /path/to/w600k_r50.onnx \
    --output /tmp/w600k_r50.surgery.onnx

# Step 2 — Prepare SCRFD for MLA (rename outputs, fix input shape)
python3 ${APP}/src/common/model/scrfd_to_mla.py \
    --input  /path/to/scrfd_2.5g_bnkps.onnx \
    --output /tmp/scrfd_2.5g_bnkps.mla.onnx

# Step 3 — Compile both for Modalix (BF16 + MLA-tessellation)
bash ${APP}/src/common/model/compile_models.sh \
    --models-dir /tmp \
    --build-dir /tmp/compiled \
    [--calib-dir /path/to/face_images]
```

</details>
