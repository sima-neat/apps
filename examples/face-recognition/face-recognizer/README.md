# Face Recognizer

## Metadata
| Field | Value |
| --- | --- |
| Category | face-recognition |
| Difficulty | Advanced |
| Tags | face-detection, face-recognition, scrfd, arcface, rtsp, enrollment, bf16, mla-tessellation |
| Languages | C++ |
| Status | experimental |
| Binary Name | face-recognizer |
| Model | scrfd_2.5g_bnkps.mla, w600k_r50.surgery |

## Concept

Real-time face detection and recognition on SiMa.ai Modalix hardware using the Neat Library. Two models run back-to-back on the MLA:

1. **SCRFD 2.5G** — detects faces and 5 facial landmarks per detection (~5.4 ms, BF16 + MLA-tessellated)
2. **ArcFace W600K R50** — produces a 512-d embedding per aligned face crop (~5.5 ms, BF16 + MLA-tessellated)

Embeddings are compared against a pre-enrolled gallery using cosine similarity to produce an identity label for each detected face. The EV74 CVU preproc node handles NV12-to-BF16 conversion in hardware (~0.01 ms), with no CPU preproc for the detection stage.

**Throughput:** ~56.5 FPS on 1280×720 @ 45 FPS RTSP.

## Install

```bash
sima-cli neat install face-recognizer
```

This installs the `face-recognizer` binary and the default `config.yaml` under the package install path.

## Configure

All runtime options are in one `config.yaml`. Edit it before running:

```yaml
scrfd:
  model: models/scrfd_2.5g_bnkps.mla_mpk.tar.gz
  conf_threshold: 0.65    # Minimum face detection confidence

arcface:
  model: models/w600k_r50.surgery_mpk.tar.gz

gallery:
  path: gallery.bin       # Enrollment data; build with --enroll first

input:
  uri: rtsp://<RTSP_HOST>:<RTSP_PORT>/<STREAM_NAME>   # RTSP source

output:
  sink: ""                # "" = headless; "display" = cv::imshow

match:
  threshold: 0.55         # Cosine similarity cutoff; below → Unknown
  margin:    0.12         # Min gap between best and 2nd-best score

runtime:
  recog_interval: 8       # Re-embed every N frames (lower = faster updates)
  timeout_ms:  20000
```

## Download Models

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/scrfd_2.5g_bnkps.mla_mpk.tar.gz" -o models/
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/w600k_r50.surgery_mpk.tar.gz" -o models/
```

## Enroll Faces

Build a gallery before running recognition. Each call is additive — run once per person.

```bash
# From a video clip:
./face-recognizer --enroll \
    --config config.yaml \
    --video  /path/to/alice.mp4 --name "Alice" \
    --gallery gallery.bin

# From an image folder (subdirectory per identity):
#   gallery_images/Alice/photo1.jpg, photo2.jpg …
./face-recognizer --enroll \
    --config config.yaml \
    --images gallery_images/ \
    --gallery gallery.bin
```

## Run

```bash
./face-recognizer --config config.yaml
```

The RTSP input URI, gallery path, model paths, and output options are all read from `config.yaml`. No extra flags are needed for a normal run.

**Optional overrides:**

| Flag | Description |
|---|---|
| `--config <path>` | Config file (default: `src/common/config.yaml`) |
| `--input <uri>` | Override `input.uri` in config |
| `--gallery <path>` | Override `gallery.path` in config |
| `--stream-host <ip>` | Send H.264 overlay stream over UDP to this host |
| `--stream-port <n>` | UDP port for overlay stream (default: 5000) |
| `--max-frames <n>` | Stop after N frames (0 = unlimited) |
| `--test` | Print per-frame results and FPS report; headless |

**Enrollment flags (with `--enroll`):**

| Flag | Description |
|---|---|
| `--video <path>` | Enrollment video; requires `--name` |
| `--name <name>` | Identity label for `--video` mode |
| `--images <dir>` | Enrollment image folder (subdirectory per identity) |
| `--gallery <path>` | Gallery file to write or append to |
| `--sample-every <n>` | Sample 1 frame every N from video (default: 5) |
| `--min-score <f>` | Minimum SCRFD confidence for enrollment (default: 0.75) |

## Tuning

| Parameter | Default | Notes |
|---|---|---|
| `match.threshold` | `0.55` | Raise to 0.60–0.65 in controlled lighting for fewer false positives |
| `match.margin` | `0.12` | Raise to 0.15–0.20 when multiple people are enrolled to reduce cross-ID errors |
| `scrfd.conf_threshold` | `0.65` | Lower to detect smaller or partially-occluded faces |
| `runtime.recog_interval` | `8` | Frames between re-embeddings (lower = more responsive identity updates) |

## Testing

```bash
# Unit tests — no hardware required
ctest --test-dir build -L unit -R 'face-recognizer' --output-on-failure -V

# E2E test — skips gracefully when models or input are absent
export SIMANEAT_APPS_TEST_MODELS_DIR=examples/face-recognition/face-recognizer/models
export SIMANEAT_TEST_RTSP_H264_URL=rtsp://<RTSP_HOST>:<RTSP_PORT>/<STREAM_NAME>
export SIMANEAT_APPS_TEST_GALLERY_BIN=examples/face-recognition/face-recognizer/gallery.bin
ctest --test-dir build -L e2e -R 'face-recognizer' --output-on-failure -V
```

---

## Development from Source

<details>
<summary>Build from the Apps repository</summary>

### Prerequisites

- Neat Development Environment (SDK container)
- SiMa Modalix DevKit accessible over the network
- NFS mount at `/workspace` on the board

### Build

Inside the SDK container:

```bash
source /opt/bin/simaai-init-build-env modalix
cd /path/to/apps
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSIMANEAT_APPS_BUILD_CPP=ON
cmake --build build --target face-recognizer -j4
```

Binary: `build/examples/face-recognition/face-recognizer_cpp/face-recognizer`

### Recompile Models from Source

Model preparation and compilation scripts are under `src/common/model/`:

```bash
# Step 1 — Apply graph surgery to ArcFace (required for MLA compatibility)
python3 src/common/model/arcface_to_mla.py \
    --input  /path/to/w600k_r50.onnx \
    --output /tmp/w600k_r50.surgery.onnx

# Step 2 — Prepare SCRFD for MLA (rename outputs, fix input shape)
python3 src/common/model/scrfd_to_mla.py \
    --input  /path/to/scrfd_2.5g_bnkps.onnx \
    --output /tmp/scrfd_2.5g_bnkps.mla.onnx

# Step 3 — Compile both for Modalix (BF16 + MLA-tessellation)
bash src/common/model/compile_models.sh \
    --models-dir /tmp \
    --output-dir /tmp/compiled \
    [--calib-dir /path/to/face_images]
```

Compiled packages land in `models/` for use with `config.yaml`.

</details>
