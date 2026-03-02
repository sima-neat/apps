# Simple Image Classification Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | classification, model, mpk |
| Status | experimental |
| Binary Name | simple-image-classification-pipeline |
| Model | resnet_50 |

## Concept
Minimal Model API usage with a ResNet50 MPK. The example loads a compiled model package, runs single-image inference, and prints top-1/top-5 classification output.

## Supported Models
Primary model: `resnet_50`

Download into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get resnet_50 && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get resnet_50 && cd ../..`

## Important Behavior
- `--model` is required in both C++ and Python.
- If `--image` is omitted, the example downloads a sample goldfish image automatically.
- `--min-prob` controls pass/fail threshold for the expected class probability.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/classification/simple-image-classification-pipeline/simple-image-classification-pipeline --model <path> [--image <path>] [--min-prob <float>] [--goldfish-url <url>]`
- Required arguments:
  `--model <path>`
- Optional arguments:
  `--image`, `--min-prob`, `--goldfish-url`

### Python
- Invocation:
  `python examples/classification/simple-image-classification-pipeline/main.py --model <path> [--image <path>] [--min-prob <float>]`
- Required arguments:
  `--model <path>`
- Optional arguments:
  `--image`, `--min-prob`

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/classification/simple-image-classification-pipeline/simple-image-classification-pipeline
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/classification/simple-image-classification-pipeline
cmake -S . -B build
cmake --build build -j
```

Binary output:
```bash
./build/simple-image-classification-pipeline
```

## Run
### C++
```bash
./build/examples/classification/simple-image-classification-pipeline/simple-image-classification-pipeline \
  --model assets/models/resnet_50_mpk.tar.gz
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/classification/simple-image-classification-pipeline/requirements.txt
python examples/classification/simple-image-classification-pipeline/main.py \
  --model assets/models/resnet_50_mpk.tar.gz
```

## Debugging Notes
- If you see a model load error, verify the model file exists at `assets/models/resnet_50_mpk.tar.gz`.
- If image decode fails, pass an explicit `--image` path.
- If top-1 validation fails, try lowering `--min-prob` for debug runs.

## Reference
- C++ source: `main.cpp`
- Python source: `main.py`
