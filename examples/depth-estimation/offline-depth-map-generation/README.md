# Offline Depth Map Generation

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
Offline depth-map generation for image folders. The example runs inference per image and writes visual depth outputs.

## Supported Models
Primary model: `depth_anything_v2_vits`

Download into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get depth_anything_v2_vits && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get depth_anything_v2_vits && cd ../..`

## Important Behavior
- Model path is positional and required.
- Input directory is scanned for common image extensions.
- Output files are written to the provided output directory.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/depth-estimation/offline-depth-map-generation/offline-depth-map-generation <model.tar.gz> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  None.

### Python
- Invocation:
  `python examples/depth-estimation/offline-depth-map-generation/python/main.py <model.tar.gz> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  None.

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/depth-estimation/offline-depth-map-generation/offline-depth-map-generation
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/depth-estimation/offline-depth-map-generation
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/offline-depth-map-generation
```

## Run
### C++
```bash
./build/examples/depth-estimation/offline-depth-map-generation/offline-depth-map-generation \
  assets/models/depth_anything_v2_vits_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/depth-estimation/offline-depth-map-generation/python/requirements.txt
python examples/depth-estimation/offline-depth-map-generation/python/main.py \
  assets/models/depth_anything_v2_vits_mpk.tar.gz <input_dir> <output_dir>
```

## Debugging Notes
- Check model path first if startup fails.
- If no outputs are produced, verify `input_dir` has valid images.
- Check write permissions on `output_dir`.

## Source Files
- C++ source: `cpp/main.cpp`
- Python source: `python/main.py`
