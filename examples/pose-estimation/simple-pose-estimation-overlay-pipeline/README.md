# Simple Pose Estimation Overlay Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | pose-estimation |
| Difficulty | Intermediate |
| Tags | pose-estimation, openpose, skeleton-detection |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | simple-pose-estimation-overlay-pipeline |
| Model | OpenPose |

## Concept
Minimal image-folder pose estimation pipeline. Each image is inferred to extract human pose keypoints, annotated with skeleton overlays (joints and limbs), and written to an output folder. Uses heatmap and Part Affinity Field (PAF) outputs to group keypoints into full-body poses.

## Supported Models
Download variants into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get open_pose && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get open_pose && cd ../..`

## Important Behavior
- C++ and Python both use positional arguments.
- Keypoint detection uses heatmaps with dynamic filtering based on confidence threshold.
- Pose grouping uses greedy bipartite matching on Part Affinity Field vectors.
- Output images are written as `.png` files.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/pose-estimation/simple-pose-estimation-overlay-pipeline/simple-pose-estimation-overlay-pipeline <model.tar.gz> <input_dir> <output_dir> [--profile]`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  `--profile` - Enable performance profiling (reports end-to-end and model-inference timing)

### Python
- Invocation:
  `python examples/pose-estimation/simple-pose-estimation-overlay-pipeline/python/main.py <model.tar.gz> <input_dir> <output_dir> [--profile]`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  `--profile` - Enable performance profiling (reports end-to-end and model-inference timing)

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/pose-estimation/simple-pose-estimation-overlay-pipeline/simple-pose-estimation-overlay-pipeline
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/pose-estimation/simple-pose-estimation-overlay-pipeline
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/simple-pose-estimation-overlay-pipeline
```

## Run
### C++
```bash
./build/examples/pose-estimation/simple-pose-estimation-overlay-pipeline/simple-pose-estimation-overlay-pipeline \
  assets/models/open_pose_mpk.tar.gz \
  assets/test_images \
  output/pose_estimation
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/pose-estimation/simple-pose-estimation-overlay-pipeline/python/requirements.txt
python examples/pose-estimation/simple-pose-estimation-overlay-pipeline/python/main.py \
  assets/models/open_pose_mpk.tar.gz \
  assets/test_images \
  output/pose_estimation
```

### Example with Profiling (Both C++ and Python)
C++:
```bash
./build/examples/pose-estimation/simple-pose-estimation-overlay-pipeline/simple-pose-estimation-overlay-pipeline \
  assets/models/open_pose_mpk.tar.gz \
  assets/test_images \
  output/pose_estimation \
  --profile
```

Python:
```bash
python examples/pose-estimation/simple-pose-estimation-overlay-pipeline/python/main.py \
  assets/models/open_pose_mpk.tar.gz \
  assets/test_images \
  output/pose_estimation \
  --profile
```

## Troubleshooting
- If model load fails, verify `assets/models/open_pose_mpk.tar.gz` exists.
- Ensure input folder contains supported image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- Confirm both input and output directories are writable and paths are correct.
- If output seems unexpected, verify threshold values and other parameters.

## Profiling
Use the `--profile` flag to measure performance:
- Per-image output includes end-to-end time and model inference time
- Summary statistics at the end report average times
- Useful for comparing performance between C++ and Python implementations

## Source Files
- C++ source: `cpp/main.cpp`
- Python source: `python/main.py`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
