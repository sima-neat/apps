# Face Recognizer

## Metadata
| Field | Value |
| --- | --- |
| Category | face-recognition |
| Difficulty | Advanced |
| Tags | face-detection, face-recognition, scrfd, arcface, rtsp, enrollment, bf16, mla-tessellation |
| Languages | C++ |
| Status | experimental |
| Binary Name | face-recognizer, face-enroll |
| Model | scrfd_2.5g_bnkps.mla, w600k_r50.surgery |

## Concept

This example demonstrates a real-time face recognition pipeline on SiMa.ai Modalix hardware using the Neat Library. It combines two models: **SCRFD 2.5G** for face detection and **ArcFace W600K R50** for face recognition.

Both models are compiled with **BF16 activations and MLA tessellation** (`--bf16-activations --mla-tesselation`). Tessellation is compiled into the MLA ELF, bypassing the CVU entirely. At runtime, the Neat Library's `graph.add(model)` API includes the EV74 APU FP32→BF16 cast step automatically from `mpk.json`. The pipeline runs SCRFD and ArcFace inference back-to-back per frame, with optional RTP/H.264 overlay streaming over UDP.

A companion tool, `face-enroll`, builds a gallery of face embeddings from video clips or image folders. The recognizer matches every detected face against the gallery using cosine similarity.

**Pipeline stages:**

1. RTSP decode (Neat HW decoder, NV12 output)
2. CPU letterbox + normalize → FP32 tensor (SCRFD input)
3. SCRFD inference on MLA (BF16, MLA-tessellated): detects faces + 5 landmarks
4. Face alignment and crop per detection
5. ArcFace inference on MLA (BF16, MLA-tessellated): 512-d embedding
6. Cosine similarity match against gallery
7. Overlay render and optional RTP/H.264 UDP stream

**Throughput:** ~45 FPS on a 1280×720 @ 60 FPS RTSP stream (SCRFD ~7 ms, ArcFace ~5.5 ms, NV12→FP32 preproc ~6 ms via NEON 2×2 box filter).

## Model Download and Preparation

The models are not available from the SiMa modelzoo and must be downloaded from public sources and prepared for MLA compilation. All steps run inside the **Neat Development Environment** with the Model Compiler activated.

Preparation scripts are in `examples/face-recognition/scripts/`.

### Step 1 — Activate the Model Compiler

```bash
source /sdk-extensions/model-compiler/bin/activate
```

### Step 2 — Download models and apply graph surgery

`prepare_models.py` downloads both models and applies the graph transformations required for MLA compilation in one step:

```bash
mkdir -p /workspace/face-recog-models
python3 examples/face-recognition/scripts/prepare_models.py \
  --out-dir /workspace/face-recog-models
```

This produces:
- `scrfd_2.5g_bnkps.mla.onnx` — SCRFD with renamed stride outputs and static input shape; postprocess tails removed so the entire model maps to a single MLA segment
- `w600k_r50.surgery.onnx` — ArcFace R50 with BN→Mul+Add, Flatten→Reshape, Gemm→MatMul+Add rewrites for MLA compatibility

**Alternatively**, run the surgery scripts individually for more control:

```bash
# SCRFD: rename outputs, freeze input to 640×640, cut Transpose+Reshape+Sigmoid heads
python3 examples/face-recognition/scripts/scrfd_to_mla.py \
  /workspace/face-recog-models/scrfd_2.5g_bnkps.onnx \
  --out /workspace/face-recog-models/scrfd_2.5g_bnkps.mla.onnx \
  --validate

# ArcFace R50: BN→Mul+Add, Flatten→Reshape, Gemm→MatMul+Add
python3 examples/face-recognition/scripts/arcface_to_mla.py \
  /workspace/face-recog-models/w600k_r50.onnx \
  --out /workspace/face-recog-models/w600k_r50.surgery.onnx \
  --validate
```

### Step 3 — Compile for Modalix (BF16 + MLA tessellation)

`compile_models.sh` compiles both prepared ONNX files with `--bf16-activations --mla-tesselation` and copies the resulting packages directly into `assets/models/`:

```bash
bash examples/face-recognition/scripts/compile_models.sh \
  --models-dir /workspace/face-recog-models \
  --build-dir  /workspace/face-recog-models/build_bf16_mlatess \
  --calib-dir  /workspace/calib_images
```

`--calib-dir` is optional but recommended for accurate BF16 quantization. It should contain representative face images (JPEG/PNG). Without it, random data is used.

After the script completes the compiled packages are placed at:

```
examples/face-recognition/face-recognizer/assets/models/scrfd_2.5g_bnkps.mla_mpk.tar.gz
examples/face-recognition/face-recognizer/assets/models/w600k_r50.surgery_mpk.tar.gz
```

**Why BF16 + MLA tessellation?**
The CVU BF16 tessellate kernels (`CastTess`/`DetessCast`) are unsupported on current Modalix DevKit firmware. Compiling with `--mla-tesselation` moves tessellation inside the MLA ELF. An EV74 APU `cast_0` node (FP32→BF16) is automatically included in the compiled package and invoked via `graph.add(model)` at runtime.

## Agentic Setup

To run the full setup end-to-end with an AI agent (model preparation, compilation,
app build, enrollment, and verification in one go), use the ready-made prompt in
[AGENT_SETUP.md](AGENT_SETUP.md).

Fill in the five placeholders (`<SDK_CONTAINER>`, `<BOARD_IP>`, `<RTSP_HOST>`,
`<RTSP_PORT>`, `<STREAM_NAME>`) and paste the prompt into Claude Code or any
capable coding agent. It executes all steps from scratch, including downloading
models, running graph surgery, BF16+MLA-tessellation compilation, building the
binaries, enrolling two identities from video clips, and running a 200-frame
recognition test to confirm the result.

## Prerequisites

- Neat Development Environment with the Neat Library installed.
- SiMa Modalix DevKit or board accessible over the network.
- Workspace NFS-mounted at `/workspace` on both the host (SDK container) and the board.
- RTSP camera stream available on the network.
- Model artifacts compiled and placed in `assets/models/` (see Model Compilation above).

## Get The Apps Repo

Clone and build the apps repo inside the Neat Development Environment container. The build requires the aarch64 cross-compile environment — source it before building:

```bash
source /opt/bin/simaai-init-build-env modalix
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

> Run all commands that follow from inside the SDK container (e.g. `docker exec <SDK_CONTAINER> bash -c "source /opt/bin/simaai-init-build-env modalix && ..."`).

After building, binaries are at:
- `build/examples/face-recognition/face-recognizer_cpp/face-recognizer`
- `build/examples/face-recognition/face-recognizer_cpp/face-enroll`

## Configure

Edit `examples/face-recognition/face-recognizer/src/common/config.yaml`:

```yaml
scrfd:
  model: examples/face-recognition/face-recognizer/assets/models/scrfd_2.5g_bnkps.mla_mpk.tar.gz
  conf_threshold: 0.60    # Minimum face confidence
  nms_iou:        0.40    # NMS IoU threshold

arcface:
  model: examples/face-recognition/face-recognizer/assets/models/w600k_r50.surgery_mpk.tar.gz

gallery:
  path: examples/face-recognition/face-recognizer/gallery.bin

input:
  uri: rtsp://<RTSP_HOST>:<RTSP_PORT>/<STREAM_NAME>

match:
  threshold: 0.60    # Cosine similarity threshold; below → Unknown
  margin:    0.10    # Min gap between best and 2nd-best score
```

> Model and gallery paths are relative to the `apps/` build root. All `run` commands below `cd` to `/workspace/sima-neat/apps` first so these resolve correctly.

Replace `<RTSP_HOST>`, `<RTSP_PORT>`, and `<STREAM_NAME>` with your stream values.

## Run

All commands below run from the **host machine** and SSH into the board via the Neat Development Environment container. Replace the following placeholders throughout:

> **Working directory:** model paths in `config.yaml` are relative to the `apps/` directory (e.g. `examples/face-recognition/.../scrfd_2.5g_bnkps.mla_mpk.tar.gz`). Every binary invocation below starts with `cd /workspace/sima-neat/apps &&` so those paths resolve correctly.

| Placeholder | Example | Description |
|---|---|---|
| `<BOARD_IP>` | `203.0.113.10` | DevKit / Modalix IP address |
| `<RTSP_HOST>` | `203.0.113.20` | RTSP camera host |
| `<RTSP_PORT>` | `8554` | RTSP port |
| `<STREAM_NAME>` | `facestream` | RTSP stream path |
| `<STREAM_HOST>` | `203.0.113.30` | Host to receive the H.264 overlay stream |
| `<STREAM_PORT>` | `5000` | UDP port for overlay stream |
| `<SDK_CONTAINER>` | `ghcr.io-sima-neat-sdk-release-2.1-latest` | SDK Docker container name |

### Build

Build inside the Neat Development Environment container:

```bash
docker exec <SDK_CONTAINER> bash -c \
  "source /opt/bin/simaai-init-build-env modalix && \
   cmake --build /workspace/sima-neat/apps/build \
   --target face-recognizer face-enroll -j4 2>&1"
```

### Enroll Faces

Enrollment builds a `gallery.bin` of face embeddings. Run once per identity; subsequent runs append to the existing gallery.

**Step 1 — Record an enrollment clip (optional, if no video exists):**

```bash
docker exec <SDK_CONTAINER> bash -c \
  "ssh -o StrictHostKeyChecking=no sima@<BOARD_IP> \
    'ffmpeg -y -rtsp_transport tcp \
     -i rtsp://<RTSP_HOST>:<RTSP_PORT>/<STREAM_NAME> \
     -t 30 -c:v copy /workspace/enroll_clip.mp4 2>&1'"
```

**Step 2 — Enroll from video clip:**

```bash
docker exec <SDK_CONTAINER> bash -c \
  "ssh -o StrictHostKeyChecking=no sima@<BOARD_IP> \
    'cd /workspace/sima-neat/apps && QT_QPA_PLATFORM=offscreen \
     build/examples/face-recognition/face-recognizer_cpp/face-enroll \
     --config examples/face-recognition/face-recognizer/src/common/config.yaml \
     --video /workspace/enroll_clip.mp4 \
     --name \"<PERSON_NAME>\" \
     --gallery examples/face-recognition/face-recognizer/gallery.bin \
     --sample-every 5 --min-score 0.75 2>&1'"
```

Repeat Step 2 with a different `--video` and `--name` for each additional person. Each run prints `[GALLERY] Loaded N existing identities` confirming it appended.

**Enroll from image folder (alternative to video):**

Organize images as:
```
/workspace/gallery_images/
  Alice/
    photo1.jpg
    photo2.jpg
  Bob/
    photo1.jpg
```

```bash
docker exec <SDK_CONTAINER> bash -c \
  "ssh -o StrictHostKeyChecking=no sima@<BOARD_IP> \
    'cd /workspace/sima-neat/apps && QT_QPA_PLATFORM=offscreen \
     build/examples/face-recognition/face-recognizer_cpp/face-enroll \
     --config examples/face-recognition/face-recognizer/src/common/config.yaml \
     --images /workspace/gallery_images/ \
     --gallery examples/face-recognition/face-recognizer/gallery.bin 2>&1'"
```

### Run the Recognition Pipeline

Run the face-recognizer against a live RTSP stream, with the annotated output streamed as RTP/H.264 over UDP:

```bash
docker exec <SDK_CONTAINER> bash -c \
  "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
   sima@<BOARD_IP> \
    'cd /workspace/sima-neat/apps && QT_QPA_PLATFORM=offscreen \
     build/examples/face-recognition/face-recognizer_cpp/face-recognizer \
     --config examples/face-recognition/face-recognizer/src/common/config.yaml \
     --input rtsp://<RTSP_HOST>:<RTSP_PORT>/<STREAM_NAME> \
     --gallery examples/face-recognition/face-recognizer/gallery.bin \
     --stream-host <STREAM_HOST> --stream-port <STREAM_PORT> \
     --max-frames 5000 2>&1'"
```

**Test mode** (prints per-frame matches and FPS report, then exits):

```bash
docker exec <SDK_CONTAINER> bash -c \
  "ssh -o StrictHostKeyChecking=no sima@<BOARD_IP> \
    'cd /workspace/sima-neat/apps && QT_QPA_PLATFORM=offscreen \
     build/examples/face-recognition/face-recognizer_cpp/face-recognizer \
     --config examples/face-recognition/face-recognizer/src/common/config.yaml \
     --input rtsp://<RTSP_HOST>:<RTSP_PORT>/<STREAM_NAME> \
     --gallery examples/face-recognition/face-recognizer/gallery.bin \
     --test --max-frames 5000 2>&1'"
```

**Run directly on the board** (if already SSH'd in):

```bash
cd /workspace/sima-neat/apps
QT_QPA_PLATFORM=offscreen \
build/examples/face-recognition/face-recognizer_cpp/face-recognizer \
  --config examples/face-recognition/face-recognizer/src/common/config.yaml \
  --input rtsp://<RTSP_HOST>:<RTSP_PORT>/<STREAM_NAME> \
  --gallery examples/face-recognition/face-recognizer/gallery.bin \
  --stream-host <STREAM_HOST> --stream-port <STREAM_PORT> \
  --max-frames 5000
```

### View the Overlay Stream

On the receiving host, play the H.264 UDP stream using GStreamer:

```bash
gst-launch-1.0 udpsrc port=<STREAM_PORT> \
  caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" \
  ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Or with ffplay:

```bash
ffplay -fflags nobuffer udp://@:<STREAM_PORT>
```

## CLI Reference

### face-recognizer

| Flag | Description |
|---|---|
| `--config <path>` | Path to config.yaml (default: `src/common/config.yaml`) |
| `--input <uri>` | RTSP URL, video file path, or empty for webcam 0 |
| `--gallery <path>` | Path to gallery.bin |
| `--stream-host <host>` | Destination IP for RTP/H.264 overlay stream |
| `--stream-port <port>` | UDP port for overlay stream (default: 5000) |
| `--max-frames <n>` | Stop after N frames (0 = unlimited) |
| `--test` | Print per-frame results and FPS report; suppress display |
| `--rtsp-fps <n>` | **Optional** override for the decoder's expected source FPS. Omit to auto-detect from the live stream (recommended). Only set this if auto-negotiation fails on your camera. |

### face-enroll

| Flag | Description |
|---|---|
| `--config <path>` | Path to config.yaml |
| `--video <path>` | Enrollment video file |
| `--images <dir>` | Enrollment image folder (subdirectory per identity) |
| `--name <name>` | Identity name (used with `--video`) |
| `--gallery <path>` | Gallery file to write or append to |
| `--sample-every <n>` | Sample 1 frame every N frames from video (default: 5) |
| `--min-score <f>` | Minimum SCRFD confidence to accept a face (default: 0.75) |

## Tuning

| Parameter | Default | Effect |
|---|---|---|
| `scrfd.conf_threshold` | `0.60` | Lower → detect more faces; raise → fewer false detections |
| `match.threshold` | `0.60` | Minimum cosine similarity to accept a match; raise if wrong IDs appear |
| `match.margin` | `0.10` | Min gap between best and 2nd-best score; prevents ambiguous matches with multiple enrolled identities |
| `scrfd.nms_iou` | `0.40` | Raise to `0.45` if double-boxes appear on one face |
| `runtime.supported_fps` | `45` | Max input FPS the pipeline can sustain. The source rate is auto-detected at startup; if it exceeds this, a warning is printed (the decoder drops frames, and a sustained overrun can destabilise long runs). |

Config changes take effect on the next pipeline run — no rebuild required.

**Input frame rate:** the source FPS is auto-detected from the decoder at startup — you do **not** need `--rtsp-fps`. If the detected rate exceeds `runtime.supported_fps` (default 45), the pipeline prints a prominent warning and continues; frames are dropped by the decoder's KeepLatest policy. Only pass `--rtsp-fps <n>` if caps negotiation fails on your specific camera.

## Debugging Notes

- `QT_QPA_PLATFORM=offscreen` is required in headless SSH sessions; omitting it aborts with a Qt xcb display error.
- If the pipeline exits at ~64000 frames, the A65 processcvu has a memory leak on this firmware; add `SIMA_PROCESSCVU_RUN_TARGET=EV74` to the environment. This does not apply to the BF16+MLA-tessellation models (no CVU preproc path).
- If `face-enroll` fails with `QuantTessOptions` errors, the models in `assets/models/` are INT8 packages; re-compile with `--bf16-activations --mla-tesselation` as described above.
- Similarity scores in the range 0.4–0.7 indicate a confident match. Scores below `match.threshold` are labelled Unknown.
- To view SDK model graph details, set `Model::Options.verbose.level = Verbose` in the source and rebuild.

## Testing

Run from the apps repository root after building:

```bash
# Unit tests — no hardware needed, runs on host inside the SDK container
ctest --test-dir build -L unit -R 'face-recognizer' --output-on-failure -V
```

```bash
# E2E test — runs on the board, skips if RTSP or gallery are absent
export SIMANEAT_APPS_TEST_MODELS_DIR=examples/face-recognition/face-recognizer/assets/models
export SIMANEAT_TEST_RTSP_H264_URL=rtsp://<RTSP_HOST>:<RTSP_PORT>/<STREAM_NAME>
export SIMANEAT_APPS_TEST_GALLERY_BIN=examples/face-recognition/face-recognizer/gallery.bin
ctest --test-dir build -L e2e -R 'face-recognizer' --output-on-failure -V
```

To create a test `gallery.bin` from reference images:

```bash
bash examples/face-recognition/face-recognizer/tests/create_test_gallery.sh \
  /path/to/reference_images/ \
  examples/face-recognition/face-recognizer/gallery.bin
```

The script expects `reference_images/<PersonName>/*.jpg` subdirectories and runs `face-enroll` on the board via SSH.

## Source Files

- Test scope: `tests/test-scope.yaml`
- C++ source: `src/cpp/main.cpp`
- C++ shared nodes: `src/cpp/scrfd_decode.cpp`, `src/cpp/align.cpp`, `src/cpp/gallery.cpp`, `src/cpp/match.cpp`, `src/cpp/overlay.cpp`
- C++ headers: `src/cpp/scrfd_decode.h`, `src/cpp/align.h`, `src/cpp/gallery.h`, `src/cpp/match.h`, `src/cpp/overlay.h`
- C++ unit tests: `tests/cpp/test_unit.cpp` (component tests, no hardware)
- C++ e2e test: `tests/cpp/test_e2e.cpp` (full pipeline, skips when assets absent)
- Gallery creation helper: `tests/create_test_gallery.sh`
- Enroll tool: `tools/enroll.cpp`
- Config: `src/common/config.yaml`
- Model preparation: `../scripts/prepare_models.py` — download + graph surgery (SCRFD + ArcFace R50)
- SCRFD surgery: `../scripts/scrfd_to_mla.py` — rename outputs, freeze input, cut postprocess heads
- ArcFace surgery: `../scripts/arcface_to_mla.py` — BN→Mul+Add, Flatten→Reshape, Gemm→MatMul+Add
- Compilation: `../scripts/compile_models.sh` — BF16+MLA-tess compile + copy to assets/
