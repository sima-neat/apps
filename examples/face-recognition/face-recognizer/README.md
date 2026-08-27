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

Both models are compiled with **BF16 activations and MLA tessellation** (`--bf16-activations --mla-tesselation`). At runtime, SCRFD is driven via `InputKind::Image` so the EV74 CVU preproc node baked into the compiled package handles letterbox resize and BT.601 normalization entirely in hardware, feeding BF16 tensors directly to the MLA — no APU FP32→BF16 cast and no CPU preproc. ArcFace uses the same BF16 MLA-tessellation path via `graph.add(model)`. Overlay graphics are rendered directly on the NV12 frame (no full-frame BGR conversion) and streamed as RTP/H.264 via the SiMa hardware encoder over UDP.

A companion tool, `face-enroll`, builds a gallery of face embeddings from video clips or image folders. The recognizer matches every detected face against the gallery using cosine similarity.

**Pipeline stages:**

1. RTSP decode (Neat HW decoder, NV12 output)
2. EV74 CVU preproc: NV12 1280×720 → letterbox resize to 640×640 → normalize [0,1] → BF16 tensor (hardware, ~0.01 ms)
3. SCRFD inference on MLA (BF16, MLA-tessellated): detects faces + 5 landmarks (~5.4 ms)
4. Face alignment and crop from the NV12 source frame per detection
5. ArcFace inference on MLA (BF16, MLA-tessellated): 512-d embedding (~5.5 ms)
6. Cosine similarity match against gallery
7. Overlay rendered directly on NV12 frame (ROI-only BGR round-trip, no full-frame conversion)
8. SiMa HW H.264 encoder → RTP/UDP stream to receiving host

**Throughput:** ~56.5 FPS on a 1280×720 @ 45 FPS RTSP stream (CVU preproc ~0.01 ms, SCRFD ~5.4 ms, ArcFace ~5.5 ms).

## Prepare the Model

| Model file | Role | Source |
|---|---|---|
| `scrfd_2.5g_bnkps.mla_mpk.tar.gz` | Face detection (SCRFD 2.5G, BF16+MLA-tess) | SiMa modelzoo |
| `w600k_r50.surgery_mpk.tar.gz` | Face recognition (ArcFace R50, BF16+MLA-tess) | SiMa modelzoo |

Model zoo version: 2.1.3

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/scrfd_2.5g_bnkps.mla_mpk.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/w600k_r50.surgery_mpk.tar.gz"
cd ..
```

<details>
<summary>Recompile from source (advanced)</summary>

The pre-built packages above are compiled with `--bf16-activations --mla-tesselation`. To recompile from the original ONNX weights, all steps run inside the **Neat Development Environment** with the Model Compiler activated.

**Step 1 — Activate the Model Compiler**

```bash
source /sdk-extensions/model-compiler/bin/activate
```

**Step 2 — Download and apply graph surgery**

```bash
mkdir -p /workspace/face-recog-models
python3 examples/face-recognition/scripts/prepare_models.py \
  --out-dir /workspace/face-recog-models
```

This produces:
- `scrfd_2.5g_bnkps.mla.onnx` — SCRFD with renamed stride outputs, static input shape, postprocess tails removed
- `w600k_r50.surgery.onnx` — ArcFace R50 with BN→Mul+Add, Flatten→Reshape, Gemm→MatMul+Add rewrites

**Step 3 — Compile for Modalix (BF16 + MLA tessellation)**

```bash
bash examples/face-recognition/scripts/compile_models.sh \
  --models-dir /workspace/face-recog-models \
  --build-dir  /workspace/face-recog-models/build_bf16_mlatess \
  --calib-dir  /workspace/calib_images
```

`--calib-dir` is optional but recommended for accurate BF16 quantization (representative face images). After compilation the packages are placed at `models/`.

</details>

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
- Model artifacts compiled and placed in `models/` (see Model Compilation above).

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
  model: models/scrfd_2.5g_bnkps.mla_mpk.tar.gz
  conf_threshold: 0.60    # Minimum face confidence
  nms_iou:        0.40    # NMS IoU threshold

arcface:
  model: models/w600k_r50.surgery_mpk.tar.gz

gallery:
  path: gallery.bin

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
    'cd /workspace/sima-neat/apps && \
     QT_QPA_PLATFORM=offscreen \
     SIMA_PROCESSCVU_RUN_TARGET=EV74 \
     SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY=1 \
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
    'cd /workspace/sima-neat/apps && \
     QT_QPA_PLATFORM=offscreen \
     SIMA_PROCESSCVU_RUN_TARGET=EV74 \
     SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY=1 \
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
SIMA_PROCESSCVU_RUN_TARGET=EV74 \
SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY=1 \
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
gst-launch-1.0 \
  udpsrc port=<STREAM_PORT> buffer-size=2097152 \
    caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" ! \
  rtph264depay ! \
  h264parse ! \
  avdec_h264 ! \
  videoconvert ! \
  fpsdisplaysink sync=false
```

Key elements and why they are required:

| Element / option | Purpose |
|---|---|
| `buffer-size=2097152` | 2 MB OS socket buffer — IDR frames can burst ~400 KB; the default 200 KB socket buffer drops packets, causing decoder stalls |
| `clock-rate=90000` in caps | Correct 90 kHz RTP clock for H.264; without it the RTP jitter buffer miscalculates timestamps and may drop or hold frames |
| `h264parse` | Buffers complete access units and resyncs the decoder at the next IDR after any packet loss; without it `avdec_h264` freezes until the next keyframe and has no clean recovery path |
| `videoconvert` | Explicit YUV→RGB conversion before the display sink; omitting it forces the sink to negotiate directly with `avdec_h264`'s raw I420 output, which produces blurry or colour-shifted frames on some display backends |
| `sync=false` | Disables clock gating on the display; required for live feeds where the pipeline produces frames as fast as they arrive rather than at a fixed wall-clock rate |

Or with ffplay (alternative, no extra flags needed):

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
- `SIMA_PROCESSCVU_RUN_TARGET=EV74` is required for the EV74 CVU preproc path (`InputKind::Image`). Without it the runtime routes CVU ops to the A65, which has a memory leak that crashes long runs at ~64000 frames.
- `SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY=1` is required alongside `EV74` so the pipeline can copy the NV12 buffer from the HW decoder to CPU memory for the overlay step. Omitting it causes a runtime error when `draw_overlay_nv12` tries to access the HW-mapped NV12 buffer.
- If SCRFD produces spurious high-confidence detections (chair detected as a face, landmarks on background objects), the CVU preproc normalize or pad settings are wrong. The `[preproc-meta]` line printed at startup shows what the CVU actually applied — confirm `norm=1` and `pad L/R/T/B=0/0/140/140` (black letterbox bars) for a 1280×720 source. `normalize.enable = AutoFlag::Auto` silently resolves to OFF, scaling inputs by 255×; `pad_value` defaults to 114 (YOLO gray) instead of 0 (SCRFD expects black padding).
- If `face-enroll` fails with `QuantTessOptions` errors, the models in `models/` are INT8 packages; re-compile with `--bf16-activations --mla-tesselation` as described above.
- Similarity scores in the range 0.4–0.7 indicate a confident match. Scores below `match.threshold` are labelled Unknown.
- The pipeline prints encoder push stats on exit (`enc_push_ok` / `enc_push_drop`). Non-zero `enc_push_drop` means the HW encoder queue was full on those frames; increase `enc_run_opt.queue_depth` in `main.cpp` (default 16) if drops persist.
- To view SDK model graph details, set `Model::Options.verbose.level = Verbose` in the source and rebuild.

## Testing

Run from the apps repository root after building:

```bash
# Unit tests — no hardware needed, runs on host inside the SDK container
ctest --test-dir build -L unit -R 'face-recognizer' --output-on-failure -V
```

```bash
# E2E test — runs on the board, skips if RTSP or gallery are absent
export SIMANEAT_APPS_TEST_MODELS_DIR=examples/face-recognition/face-recognizer/models
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
- Compilation: `../scripts/compile_models.sh` — BF16+MLA-tess compile + copy to models/
