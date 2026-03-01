# RTSP Depth Estimation (MiDaS v2.1 / Depth Anything V2)

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
Depth estimation from an RTSP camera stream using a supported depth model with the Session API.

## Supported Models
Also works with: `depth_anything_v2_vits`

Download either variant into `assets/models/`:
- `sima-cli modelzoo get midas_v21_small_256`
- `sima-cli modelzoo get depth_anything_v2_vits`

The Python and C++ scripts auto-detect the model family from the `--model` filename and use the
correct defaults and tensor unpacking:
- `midas_v21_small_256`: `BGR`, default size `256x256`
- `depth_anything_v2_vits`: `RGB`, default size `518x518`

`--model` is required for both C++ and Python; no automatic model download/lookup is performed.

## Prerequisites
- Installed NEAT SDK
- RTSP camera source (for RTSP mode)
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command (MiDaS): `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get midas_v21_small_256 && cd ../..`
- Download command (Depth Anything V2): `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get depth_anything_v2_vits && cd ../..`

## Run
### C++
```bash
./build/examples/depth-estimation/live-rtsp-depth-estimation/live-rtsp-depth-estimation \
  --model assets/models/midas_v21_small_256_mpk.tar.gz \
  --url <rtsp_url> \
  --tcp \
  --frames <num_frames> \
  --fps <fps>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
cd apps
pip install -r examples/depth-estimation/live-rtsp-depth-estimation/requirements.txt
python3 examples/depth-estimation/live-rtsp-depth-estimation/main.py \
  --model assets/models/midas_v21_small_256_mpk.tar.gz \
  --rtsp <rtsp_url> \
  --tcp \
  --frames <num_frames> \
  --fps <fps> \
  --output-file <output_video_path.mp4>
```

Example with Depth Anything V2 ViT-S (auto-detects `RGB` + `518x518` defaults):

### C++
```bash
./build/examples/depth-estimation/live-rtsp-depth-estimation/live-rtsp-depth-estimation \
  --model assets/models/depth_anything_v2_vits_mpk.tar.gz \
  --url <rtsp_url> \
  --tcp \
  --frames <num_frames> \
  --fps <fps> \
  --output-file <output_video_path.mp4>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
cd apps
pip install -r examples/depth-estimation/live-rtsp-depth-estimation/requirements.txt
python3 examples/depth-estimation/live-rtsp-depth-estimation/main.py \
  --model assets/models/depth_anything_v2_vits_mpk.tar.gz \
  --rtsp <rtsp_url> \
  --tcp \
  --frames <num_frames> \
  --fps <fps> \
  --output-file <output_video_path.mp4>
```

### Test with Local Video
Start a local RTSP server (in a separate terminal):
```bash
cd apps
python3 utils/rtsp/rtsp_file_server.py assets/videos/neat-video.mp4 \
  --width 1280 --height 720 --fps 25
```
Then run the Python example:
```bash
source ~/pyneat/.venv/bin/activate
cd apps
pip install -r examples/depth-estimation/live-rtsp-depth-estimation/requirements.txt
python3 examples/depth-estimation/live-rtsp-depth-estimation/main.py \
  --model assets/models/midas_v21_small_256_mpk.tar.gz \
  --rtsp rtsp://127.0.0.1:8554/stream \
  --tcp \
  --frames 200 \
  --fps 25 \
  --output-file <output_video_path.mp4>
```

Then run the C++ example:
```bash
./build/examples/depth-estimation/live-rtsp-depth-estimation/live-rtsp-depth-estimation \
  --model assets/models/midas_v21_small_256_mpk.tar.gz \
  --url rtsp://127.0.0.1:8554/stream \
  --tcp \
  --frames 200 \
  --fps 25 \
  --output-file <output_video_path.mp4>
```

If `--output-file` is omitted, Python writes `midas_depth_overlay.mp4` and C++ writes
`midas_depth.mp4` in the current working directory.

## Source Files
- C++: `main.cpp`
- Python: `main.py`
