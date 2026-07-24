# Face Recognizer — Agentic Setup Prompt

Copy the prompt below and give it to an AI agent (e.g. Claude Code). It covers
every step from scratch: model download, graph surgery, BF16+MLA-tess compilation,
app build, face enrollment from two video clips, and a 200-frame recognition test.
Fill in the five placeholders before pasting.

---

```
You are setting up the full face-recognition pipeline on a SiMa Modalix DevKit.
Work entirely inside the Neat Development Environment SDK container. All commands
that run on the board must go through SSH. Complete every phase in order; do not
skip ahead.

---

## Environment

- SDK container name: <SDK_CONTAINER>   (e.g. ghcr.io-sima-neat-sdk-release-2.1-latest)
- Board:             sima@<BOARD_IP>    (e.g. sima@192.168.135.41)
- Apps root on NFS:  /workspace/sima-neat/apps
- Model output dir:  /workspace/face-recog-models
- Calibration dir:   /workspace/calib_images   (optional; skip if absent)
- Enrollment videos: /workspace/paresh_enroll.mp4   → identity "paresh"
                     /workspace/abhishek_enroll.mp4  → identity "abhishek"
- Gallery output:    /workspace/sima-neat/apps/examples/face-recognition/face-recognizer/gallery.bin

---

## Phase 1 — Model download and graph surgery (inside SDK container)

Activate the Model Compiler, then run prepare_models.py to download both models
from public sources and apply the graph transformations required for MLA
compilation.

  source /sdk-extensions/model-compiler/bin/activate

  mkdir -p /workspace/face-recog-models
  python3 /workspace/sima-neat/apps/examples/face-recognition/scripts/prepare_models.py \
    --out-dir /workspace/face-recog-models

Expected outputs:
  /workspace/face-recog-models/scrfd_2.5g_bnkps.mla.onnx
  /workspace/face-recog-models/w600k_r50.surgery.onnx

If prepare_models.py fails on the download step, download the files manually:
  - SCRFD: https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/main/scrfd_2.5g_bnkps.onnx
    → save as /workspace/face-recog-models/scrfd_2.5g_bnkps.onnx
    then: python3 .../scripts/scrfd_to_mla.py /workspace/face-recog-models/scrfd_2.5g_bnkps.onnx \
            --out /workspace/face-recog-models/scrfd_2.5g_bnkps.mla.onnx --validate
  - ArcFace: extract w600k_r50.onnx from
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
    → save as /workspace/face-recog-models/w600k_r50.onnx
    then: python3 .../scripts/arcface_to_mla.py /workspace/face-recog-models/w600k_r50.onnx \
            --out /workspace/face-recog-models/w600k_r50.surgery.onnx --validate

---

## Phase 2 — Compile for Modalix (BF16 + MLA tessellation, inside SDK container)

  bash /workspace/sima-neat/apps/examples/face-recognition/scripts/compile_models.sh \
    --models-dir /workspace/face-recog-models \
    --build-dir  /workspace/face-recog-models/build_bf16_mlatess \
    --calib-dir  /workspace/calib_images

(If /workspace/calib_images does not exist, omit --calib-dir. Random calibration
data will be used; accuracy may be slightly lower.)

The script activates the Model Compiler venv automatically, compiles both models
with --bf16-weights --bf16-activations --mla-tesselation, and copies the finished
packages to:
  /workspace/sima-neat/apps/examples/face-recognition/face-recognizer/assets/models/
    scrfd_2.5g_bnkps.mla_mpk.tar.gz
    w600k_r50.surgery_mpk.tar.gz

Verify both files exist before proceeding to Phase 3.

---

## Phase 3 — Build the apps (inside SDK container)

  source /opt/bin/simaai-init-build-env modalix
  cmake --build /workspace/sima-neat/apps/build \
    --target face-recognizer face-enroll -j4 2>&1

Verify these binaries exist:
  /workspace/sima-neat/apps/build/examples/face-recognition/face-recognizer_cpp/face-recognizer
  /workspace/sima-neat/apps/build/examples/face-recognition/face-recognizer_cpp/face-enroll

---

## Phase 4 — Enroll faces (runs on the board via SSH)

Enroll "paresh" from /workspace/paresh_enroll.mp4:

  ssh -o StrictHostKeyChecking=no sima@<BOARD_IP> \
    "cd /workspace/sima-neat/apps && QT_QPA_PLATFORM=offscreen \
     build/examples/face-recognition/face-recognizer_cpp/face-enroll \
     --config examples/face-recognition/face-recognizer/src/common/config.yaml \
     --video  /workspace/paresh_enroll.mp4 \
     --name   paresh \
     --gallery examples/face-recognition/face-recognizer/gallery.bin \
     --sample-every 5 --min-score 0.75 2>&1"

Enroll "abhishek" from /workspace/abhishek_enroll.mp4
(appends to the same gallery.bin):

  ssh -o StrictHostKeyChecking=no sima@<BOARD_IP> \
    "cd /workspace/sima-neat/apps && QT_QPA_PLATFORM=offscreen \
     build/examples/face-recognition/face-recognizer_cpp/face-enroll \
     --config examples/face-recognition/face-recognizer/src/common/config.yaml \
     --video  /workspace/abhishek_enroll.mp4 \
     --name   abhishek \
     --gallery examples/face-recognition/face-recognizer/gallery.bin \
     --sample-every 5 --min-score 0.75 2>&1"

The second run must print "[GALLERY] Loaded 1 existing identities" confirming
it appended rather than overwriting.

---

## Phase 5 — Verify (runs on the board via SSH)

Run in test mode for 200 frames to confirm both identities are recognised:

  ssh -o StrictHostKeyChecking=no sima@<BOARD_IP> \
    "cd /workspace/sima-neat/apps && QT_QPA_PLATFORM=offscreen \
     build/examples/face-recognition/face-recognizer_cpp/face-recognizer \
     --config examples/face-recognition/face-recognizer/src/common/config.yaml \
     --input  rtsp://<RTSP_HOST>:<RTSP_PORT>/<STREAM_NAME> \
     --gallery examples/face-recognition/face-recognizer/gallery.bin \
     --test --max-frames 200 2>&1"

Success criteria:
- Pipeline starts without model load errors.
- Per-frame output includes "paresh" and "abhishek" labels with cosine scores >= 0.50.
- Final FPS report shows >= 20 FPS sustained.

If RTSP is unavailable, provide a video file path via --input instead.

---

## Troubleshooting

- "QuantTessOptions" errors at enroll/recognise time → models were compiled as
  INT8; redo Phase 2 with --bf16-activations --mla-tesselation.
- "archive path does not exist" from face-enroll → wrong working directory;
  ensure every binary invocation starts with cd /workspace/sima-neat/apps.
- Pipeline exits at ~64 000 frames → add SIMA_PROCESSCVU_RUN_TARGET=EV74 to
  the environment (not needed for BF16+MLA-tess models).
- QT xcb error in headless SSH → add QT_QPA_PLATFORM=offscreen.
- Low or zero FPS in test mode → check that the RTSP stream is live and that
  both .tar.gz model packages are present in assets/models/.
```
