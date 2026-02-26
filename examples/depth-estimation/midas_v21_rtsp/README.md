# MiDaS v2.1 RTSP Depth Estimation

## Metadata
| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation, rtsp |
| Status | experimental |
| Binary Name | midas_v21_rtsp |
| Model | midas_v21_small_256 |

## Concept
Depth estimation from an RTSP camera stream using a MiDaS v2.1 model with the Session API.

## Prerequisites
- Installed NEAT SDK
- RTSP camera source (for RTSP mode)
- Model downloaded: `sima-cli modelzoo get midas_v21_small_256`

## Run
### C++
```bash
./build/examples/depth-estimation/midas_v21_rtsp/midas_v21_rtsp \
  --model models/midas_v21_small_256_mpk.tar.gz \
  --rtsp <rtsp_url> \
  --tcp \
  --frames <num_frames> \
  --fps <fps>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
cd apps
pip install -r examples/depth-estimation/midas_v21_rtsp/requirements.txt
python3 examples/depth-estimation/midas_v21_rtsp/main.py \
  --model models/midas_v21_small_256_mpk.tar.gz \
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
python3 utils/rtsp/rtsp_file_server.py data/videos/neat-video.mp4 \
  --transcode \
  --width 1280 \
  --height 720 \
  --fps 25
```
Then run the Python example:
```bash
source ~/pyneat/.venv/bin/activate
cd apps
pip install -r examples/depth-estimation/midas_v21_rtsp/requirements.txt
python3 examples/depth-estimation/midas_v21_rtsp/main.py \
  --model models/midas_v21_small_256_mpk.tar.gz \
  --rtsp rtsp://127.0.0.1:8554/stream \
  --tcp \
  --frames 300 \
  --fps 25 \
  --output-file <output_video_path.mp4>
```

Then run the C++ example:
```bash
./build/examples/depth-estimation/midas_v21_rtsp/midas_v21_rtsp \
  --model models/midas_v21_small_256_mpk.tar.gz \
  --url rtsp://127.0.0.1:8554/stream \
  --tcp \
  --frames 300 \
  --fps 25 \
  --output-file <output_video_path.mp4>
```

If `--output-file` is omitted, Python writes `midas_depth_overlay.mp4` and C++ writes
`midas_depth.mp4` in the current working directory.

Note: `data/videos/neat-video.mp4` is a high-resolution H.264 source. `--transcode` re-encodes it to a
decoder-friendly RTSP stream for local testing. For other sources that already decode cleanly, passthrough mode
(without `--transcode`) is usually fine.

## Source Files
- C++: `main.cpp`
- Python: `main.py`
