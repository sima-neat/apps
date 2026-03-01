# Depth Anything V2

## Metadata
| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation |
| Status | experimental |
| Binary Name | offline-depth-map-generation |
| Model | depth_anything_v2_vits |

## Concept
Minimal Depth Anything V2 pipeline: loads a depth model and infers depth maps for every image in a folder.

## Prerequisites
- Installed NEAT SDK
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get depth_anything_v2_vits && cd ../..`

## Run
### C++
```bash
./build/examples/depth-estimation/offline-depth-map-generation/offline-depth-map-generation assets/models/depth_anything_v2_vits_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
pip install -r examples/depth-estimation/offline-depth-map-generation/requirements.txt
python examples/depth-estimation/offline-depth-map-generation/main.py assets/models/depth_anything_v2_vits_mpk.tar.gz <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
- Python: `main.py`
