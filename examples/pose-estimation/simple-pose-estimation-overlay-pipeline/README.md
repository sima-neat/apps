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
| Model | open_pose |

## Concept
Minimal image-folder pose estimation pipeline. Each image is inferred to extract human pose keypoints, annotated with skeleton overlays (joints and limbs), and written to an output folder. Uses heatmap and Part Affinity Field (PAF) outputs to group keypoints into full-body poses.

## Preview
![Demo screenshot](../../../assets/portal/pose-estimation/simple-pose-estimation-overlay-pipeline/image.png)

## Supported Models
Download variants into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get open_pose && cd ../..`

## Prerequisites
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get pose_estimation/open_pose && cd ../..`

## Important Behavior
- C++ and Python both use positional arguments for model path, input directory, and output directory.
- All detection and grouping parameters are configurable via CLI flags with sensible defaults.
- Keypoint detection uses heatmaps with dynamic filtering based on confidence threshold (default: 0.1, configurable with `--keypoint-score`).
- Pose grouping uses greedy bipartite matching on Part Affinity Field vectors with configurable thresholds (`--paf-score`, `--paf-success-ratio`).
- Output images are written as `.png` files.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/pose-estimation/simple-pose-estimation-overlay-pipeline/simple-pose-estimation-overlay-pipeline <model.tar.gz> <input_dir> <output_dir> [OPTIONS]`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  - `--infer-size SIZE` - Inference input size (default: 640)
  - `--keypoint-score SCORE` - Keypoint confidence threshold (default: 0.1)
  - `--nms-radius RADIUS` - Non-maximum suppression radius (default: 6)
  - `--paf-score SCORE` - Part Affinity Field score threshold (default: 0.05)
  - `--paf-success-ratio RATIO` - PAF success ratio for valid connections (default: 0.8)
  - `--paf-samples N` - Number of PAF samples for line integral (default: 10)
  - `--upsample-factor FACTOR` - Heatmap/PAF upsample factor (default: 4.0)
  - `--timeout MS` - Inference pull timeout in milliseconds (default: 1000)
  - `--profile` - Enable performance profiling (reports end-to-end and model-inference timing)

### Python
- Invocation:
  `python examples/pose-estimation/simple-pose-estimation-overlay-pipeline/python/main.py <model.tar.gz> <input_dir> <output_dir> [OPTIONS]`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  - `--infer-size SIZE` - Inference input size (default: 640)
  - `--keypoint-score SCORE` - Keypoint confidence threshold (default: 0.1)
  - `--nms-radius RADIUS` - Non-maximum suppression radius (default: 6)
  - `--paf-score SCORE` - Part Affinity Field score threshold (default: 0.05)
  - `--paf-success-ratio RATIO` - PAF success ratio for valid connections (default: 0.8)
  - `--paf-samples N` - Number of PAF samples for line integral (default: 10)
  - `--upsample-factor FACTOR` - Heatmap/PAF upsample factor (default: 4.0)
  - `--pull-timeout MS` - Inference pull timeout in milliseconds (default: 5000)
  - `--profile` - Enable performance profiling (reports end-to-end and model-inference timing)

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
source ~/pyneat/bin/activate
pip install -r examples/pose-estimation/simple-pose-estimation-overlay-pipeline/python/requirements.txt
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

### Example with Custom Parameters
C++ (higher inference size, stricter keypoint threshold):
```bash
./build/examples/pose-estimation/simple-pose-estimation-overlay-pipeline/simple-pose-estimation-overlay-pipeline \
  assets/models/open_pose_mpk.tar.gz \
  assets/test_images \
  output/pose_estimation \
  --infer-size 768 \
  --keypoint-score 0.15 \
  --nms-radius 8 \
  --profile
```

Python (lower PAF success ratio for more connections):
```bash
python examples/pose-estimation/simple-pose-estimation-overlay-pipeline/python/main.py \
  assets/models/open_pose_mpk.tar.gz \
  assets/test_images \
  output/pose_estimation \
  --paf-success-ratio 0.7 \
  --paf-score 0.04 \
  --upsample-factor 2.0 \
  --profile
```

## Troubleshooting
- If model load fails, verify `assets/models/open_pose_mpk.tar.gz` exists.
- Ensure input folder contains supported image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- Confirm both input and output directories are writable and paths are correct.
- If too few keypoints are detected, lower `--keypoint-score` threshold (e.g., 0.05 instead of 0.1).
- If too many false keypoints appear, raise `--keypoint-score` threshold (e.g., 0.15 instead of 0.1).
- If poses are incomplete or fragmented, adjust `--paf-score` and `--paf-success-ratio` to be more lenient.
- If inference timeout occurs, increase `--timeout` (C++) or `--pull-timeout` (Python).
- Adjust `--infer-size` if you need different precision levels (larger values use more memory but may improve accuracy).

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
