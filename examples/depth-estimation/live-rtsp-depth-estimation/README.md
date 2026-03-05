# Live RTSP Depth Estimation

## Metadata
| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation, rtsp |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | live-rtsp-depth-estimation |
| Model | midas_v21_small_256 |

## Concept
Depth estimation from an RTSP stream with optional depth-overlay recording. The example supports MiDaS v2.1 and Depth Anything V2 with auto profile selection based on model filename.

## Architecture

Pipeline:

`RTSP Decode -> Depth Model -> Colorized Depth Overlay -> Video Writer`

Both implementations keep source/session setup, model setup, processing loop, reconnect handling, and teardown as explicit lifecycle stages.

## NEAT API Usage

- RTSP source:
  Python: `RtspDecodedInputOptions` -> `Session.add(rtsp_decoded_input)` -> `build()`
  C++: `RtspDecodedInputOptions` -> `Session.add(RtspDecodedInput)` -> `build(...)`
- Model setup:
  Python: `ModelOptions` + `Model(path, options)` then `model.run(...)`
  C++: `Model::Options` + `Model(path, options)` then runner/session execution
- Pull loop:
  Python: `run.pull_tensor(timeout_ms=...)`
  C++: `run.pull_tensor(...)` / `run.pull(...)` with timeout-driven reconnect policy

## Lifecycle

1. Parse config and infer model profile.
2. Build RTSP decode runtime.
3. Build model runtime.
4. Enter frame loop (pull, infer, overlay, write).
5. Handle reconnect on pull timeout.
6. Close run/writer and print summary.

## Behavior Preserved

- Same model-profile inference from model filename.
- Same pull timeout and reconnect thresholds.
- Same output-file semantics and frame-limit handling.
- Same CLI distinction: C++ uses `--url`, Python uses `--rtsp`.

## Supported Models
Also works with: `depth_anything_v2_vits`

Download into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get midas_v21_small_256 && cd ../..`
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get depth_anything_v2_vits && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- RTSP source reachable from runtime environment.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command (MiDaS): `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get midas_v21_small_256 && cd ../..`
- Download command (Depth Anything V2): `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get depth_anything_v2_vits && cd ../..`

## Important Behavior
- `--model` is required for both C++ and Python.
- Model profile is inferred from filename:
  `midas_v21_small_256` -> `BGR`, default `256x256`; `depth_anything_v2_vits` -> `RGB`, default `518x518`.
- C++ uses `--url`; Python uses `--rtsp`.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/depth-estimation/live-rtsp-depth-estimation/live-rtsp-depth-estimation --model <path> --url <rtsp_url> [options]`
- Required arguments:
  `--model`, `--url` (unless `--self-test`)
- Optional arguments:
  `--output-file`, `--frames`, `--width`, `--height`, `--fps`, `--alpha`, `--tcp`, `--udp`, `--latency`, `--log-every`, `--profile`, `--self-test`, `--video`

### Python
- Invocation:
  `python3 examples/depth-estimation/live-rtsp-depth-estimation/python/main.py --model <path> --rtsp <rtsp_url> [options]`
- Required arguments:
  `--model`, `--rtsp`
- Optional arguments:
  `--output-file`, `--frames`, `--width`, `--height`, `--fps`, `--alpha`, `--tcp`, `--latency-ms`, `--sample-every`, `--log-every`, `--profile`

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/depth-estimation/live-rtsp-depth-estimation/live-rtsp-depth-estimation
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/depth-estimation/live-rtsp-depth-estimation
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/live-rtsp-depth-estimation
```

## Run
### C++
```bash
./build/examples/depth-estimation/live-rtsp-depth-estimation/live-rtsp-depth-estimation \
  --model assets/models/midas_v21_small_256_mpk.tar.gz \
  --url <rtsp_url> \
  --tcp \
  --frames 200 \
  --fps 25 \
  --output-file <output_video_path.mp4>
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/depth-estimation/live-rtsp-depth-estimation/python/requirements.txt
python3 examples/depth-estimation/live-rtsp-depth-estimation/python/main.py \
  --model assets/models/midas_v21_small_256_mpk.tar.gz \
  --rtsp <rtsp_url> \
  --tcp \
  --frames 200 \
  --fps 25 \
  --output-file <output_video_path.mp4>
```

Local RTSP test source:
```bash
python3 utils/rtsp/rtsp_file_server.py assets/videos/neat-video.mp4 --width 1280 --height 720 --fps 25
```

## Debugging Notes
- If no frames arrive, verify RTSP connectivity and transport settings (`--tcp`/`--udp`).
- If output looks wrong, check the model family inferred from the model filename.
- If recording fails, validate output path permissions.

## Source Files
- C++ source: `cpp/main.cpp`
- Python source: `python/main.py`
