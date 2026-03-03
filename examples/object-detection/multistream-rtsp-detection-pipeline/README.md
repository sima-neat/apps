# Multistream RTSP Detection Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multistream-rtsp-detection-pipeline |
| Model | yolo_v8m |

## Concept
Multi-camera RTSP object detection pipeline using YOLOv8. The sample demonstrates concurrent stream ingestion, batched runtime behavior, and per-stream annotated output writing.

## Architecture

### Pipeline

```
RTSP Source ──► Frame Queue ──► Model Inference ──► Result Queue ──► Overlay + Save
(producer)                      (infer worker)                      (overlay worker)
```

Each RTSP stream gets its own set of producer, infer, and overlay worker threads connected by bounded queues with keep-latest overflow semantics.

### NEAT API Usage

**RTSP input session** (in `main.py` / `main.cpp`):
- Configure stream options via `pyneat.RtspDecodedInputOptions` / `simaai::neat::nodes::groups::RtspDecodedInputOptions`
- Build a session: `pyneat.Session` → `add(rtsp_decoded_input)` → `add(output)` → `session.build(run_options)` (Python) or `simaai::neat::Session` → `add(RtspDecodedInput)` → `add(Output)` → `session.build(run_opt)` (C++)
- Pull decoded frames: `run.pull_tensor(timeout_ms=...)`

**Model setup** (in `main.py` / `main.cpp`):
- Configure model: `pyneat.ModelOptions` / `simaai::neat::Model::Options` — set `format`, `input_max_width`, `input_max_height`, `input_max_depth`
- Load model: `pyneat.Model(path, options)` / `simaai::neat::Model(path, options)`

**Inference** (in `pipeline.py` / `pipeline.h`):
- Python: lazy-build via `model.build(frame, session_opts, run_opts)`, then `runner.push(frame)` / `runner.pull(timeout_ms=...)` with async pipeline depth
- C++: explicit session with `Input → Preprocess → Infer → SimaBoxDecode → Output` nodes, built via `session.build(dummy_frame, RunMode::Async, run_opts)`, then `run.push(frame)` / `run.pull(timeout_ms, sample)`

## Supported Models
Also works with: `yolo_v8n`, `yolo_v8s`, `yolo_v8l`

Download any variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolo_v8m && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- One or more RTSP camera sources.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolo_v8m && cd ../..`

## Important Behavior
- `--model`, `--output`, and at least one `--rtsp` are required.
- Use repeated `--rtsp` flags for multistream input.
- Output images are written per stream under the output directory.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/object-detection/multistream-rtsp-detection-pipeline/multistream-rtsp-detection-pipeline --model <path> --output <dir> --rtsp <url0> [--rtsp <url1> ...] [options]`
- Required arguments:
  `--model`, `--output`, one or more `--rtsp`
- Optional arguments:
  `--labels-file`, `--frames`, `--fps`, `--tcp`, `--sample-every`, `--save-every`, `--run-queue-depth`, `--overflow-policy`, `--output-memory`, `--min-score`, `--nms-iou`, `--max-det`, `--model-timeout-ms`, `--frame-queue`, `--result-queue`, `--pull-timeout-ms`, `--max-idle-ms`, `--reconnect-miss`, `--debug`, `--profile`, `--profile-every`

### Python
- Invocation:
  `python examples/object-detection/multistream-rtsp-detection-pipeline/main.py --model <path> --output <dir> --rtsp <url0> [--rtsp <url1> ...] [options]`
- Required arguments:
  `--model`, `--output`, one or more `--rtsp`
- Optional arguments:
  `--labels-file`, `--frames`, `--fps`, `--tcp`, `--latency-ms`, `--sample-every`, `--save-every`, `--run-queue-depth`, `--overflow-policy`, `--output-memory`, `--infer-size`, `--min-score`, `--nms-iou`, `--max-det`, `--model-timeout-ms`, `--model-queue-depth`, `--frame-queue`, `--result-queue`, `--pull-timeout-ms`, `--max-idle-ms`, `--reconnect-miss`, `--debug`, `--profile`, `--profile-every`

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/object-detection/multistream-rtsp-detection-pipeline/multistream-rtsp-detection-pipeline
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/object-detection/multistream-rtsp-detection-pipeline
cmake -S . -B build
cmake --build build -j
```

Binary output:
```bash
./build/multistream-rtsp-detection-pipeline
```

## Run

### Start an RTSP test source
If you don't have live RTSP cameras, use the bundled test server to serve a local video file as multiple RTSP streams:
```bash
# In a separate terminal — start 2 looping RTSP streams on port 8554
python utils/rtsp/rtsp_multi_file_server.py /path/to/video.mp4 --streams 2

# With transcoding to match the pipeline's expected resolution
python utils/rtsp/rtsp_multi_file_server.py /path/to/video.mp4 --streams 2 --width 1280 --height 720 --fps 30
```
Streams are mounted at `rtsp://127.0.0.1:8554/stream0`, `rtsp://127.0.0.1:8554/stream1`, etc.

### C++
```bash
./build/examples/object-detection/multistream-rtsp-detection-pipeline/multistream-rtsp-detection-pipeline \
  --model assets/models/yolo_v8m_mpk.tar.gz \
  --output <output_dir> \
  --labels-file examples/object-detection/multistream-rtsp-detection-pipeline/coco_label.txt \
  --frames 100 --tcp --fps 10 --save-every 10 \
  --rtsp rtsp://127.0.0.1:8554/stream0 \
  --rtsp rtsp://127.0.0.1:8554/stream1
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/multistream-rtsp-detection-pipeline/requirements.txt
python examples/object-detection/multistream-rtsp-detection-pipeline/main.py \
  --model assets/models/yolo_v8m_mpk.tar.gz \
  --output <output_dir> \
  --labels-file examples/object-detection/multistream-rtsp-detection-pipeline/coco_label.txt \
  --frames 100 --tcp --fps 10 --save-every 10 \
  --rtsp rtsp://127.0.0.1:8554/stream0 \
  --rtsp rtsp://127.0.0.1:8554/stream1
```

## Debugging Notes
- Start with one stream first, then scale to multiple URLs.
- If streams stall, check pull timeout, queue sizes, and RTSP source health.
- If no detections appear, verify model path and labels file.

## Reference
- C++ source: `main.cpp`
- Python source: `main.py`
- RTSP helper: `utils/rtsp/rtsp_multi_file_server.py`
