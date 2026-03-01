# RTSP Depth Estimation (MiDaS v2.1 / Depth Anything V2)

## Metadata
| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation, rtsp |
| Status | experimental |
| Binary Name | live-rtsp-depth-estimation |
| Model | midas_v21_small_256 |

## Concept
Depth estimation from an RTSP stream with optional depth-overlay recording. The example supports MiDaS v2.1 and Depth Anything V2 with auto profile selection based on model filename.

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
  `python3 examples/depth-estimation/live-rtsp-depth-estimation/main.py --model <path> --rtsp <rtsp_url> [options]`
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
cmake -S . -B build
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
source ~/pyneat/.venv/bin/activate
pip install -r examples/depth-estimation/live-rtsp-depth-estimation/requirements.txt
python3 examples/depth-estimation/live-rtsp-depth-estimation/main.py \
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

## Reference
- C++ source: `main.cpp`
- Python source: `main.py`
