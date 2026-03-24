# Simple Re-Identification Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | re-identification |
| Difficulty | Intermediate |
| Tags | re-identification, embeddings, similarity, pairwise-comparison |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | simple-re-identification-pipeline |
| Model | reid |

## Concept
This example performs pairwise person re-identification by computing embeddings for two input images and comparing them with a selectable metric. It is designed as a minimal synchronous pipeline that focuses on practical runtime behavior: model warmup, deterministic preprocessing, inference, score computation, threshold-based decision, and artifact output.

Both C++ and Python implementations follow the same user-facing flow and produce the same output artifacts (`comparison.jpg` and/or `result.json`). The sample demonstrates how to use NEAT for embedding extraction and how to build a lightweight post-processing layer for similarity-based identity matching.

## Supported Models
Also works with: other ReID variants compatible with 128x256 RGB input.

Download the default variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get reid && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get reid && cd ../..`

## Important Behavior
- Required positional inputs are exactly two image paths (`image1 image2`).
- Default model path is `assets/models/reid_mpk.tar.gz` (override with `--model`).
- Default output directory is `examples/re-identification/simple-re-identification-pipeline/output_dir` (override with `--output-dir`).
- Metric options: `cosine` (default threshold 0.65) or `euclidean` (default threshold 25.0).
- Output artifact mode is controlled by `--output-type image|json|both` (default: `both`).
- One warmup inference is executed before timing/profile reporting.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/re-identification/simple-re-identification-pipeline/simple-re-identification-pipeline <image1> <image2> [options]`
- Required arguments:
  `<image1> <image2>`
- Optional arguments:
  `--metric <cosine|euclidean>`, `--threshold <float>`, `--output-dir <path>`, `--output-type <image|json|both>`, `--model <path>`, `--profile`

### Python
- Invocation:
  `python3 examples/re-identification/simple-re-identification-pipeline/python/main.py <image1> <image2> [options]`
- Required arguments:
  `<image1> <image2>`
- Optional arguments:
  `--metric <cosine|euclidean>`, `--threshold <float>`, `--output-dir <path>`, `--output-type <image|json|both>`, `--model <path>`, `--profile`

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/re-identification/simple-re-identification-pipeline/simple-re-identification-pipeline
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>
cmake -S examples/re-identification/simple-re-identification-pipeline/cpp -B build/simple-re-identification-pipeline
cmake --build build/simple-re-identification-pipeline -j
```

Binary output:
```bash
./build/simple-re-identification-pipeline/simple-re-identification-pipeline
```

## Run
### C++
```bash
./build/examples/re-identification/simple-re-identification-pipeline/simple-re-identification-pipeline \
  assets/images/neat_reid_examples/id2/744__1.jpg \
  assets/images/neat_reid_examples/id3/795__4.jpg \
  --metric euclidean \
  --threshold 0.75 \
  --output-dir examples/re-identification/simple-re-identification-pipeline/output_dir \
  --profile
```

### Python
```bash
source ~/pyneat/bin/activate
python3 examples/re-identification/simple-re-identification-pipeline/python/main.py \
  assets/images/neat_reid_examples/id2/744__1.jpg \
  assets/images/neat_reid_examples/id3/795__4.jpg \
  --metric euclidean \
  --threshold 0.75 \
  --output-dir examples/re-identification/simple-re-identification-pipeline/output_dir \
  --profile
```

## Debugging Notes
- Confirm the model file exists at `assets/models/reid_mpk.tar.gz` or pass an explicit `--model` path.
- Confirm both input image paths are valid image files.
- Confirm output directory is writable.
- If score/decision seems unexpected, verify metric selection and threshold values.

## Source Files
- C++ source: `cpp/main.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python source: `python/main.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Shared assets: `common/`
