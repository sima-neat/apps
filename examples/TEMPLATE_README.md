# <Example Name>

## Metadata
| Field | Value |
| --- | --- |
| Category | <classification / object-detection / semantic-segmentation / instance-segmentation / depth-estimation / throughput> |
| Difficulty | <Beginner / Intermediate / Advanced> |
| Tags | <comma-separated tags> |
| Status | <experimental / stable> |
| Binary Name | <cmake_target_name> |
| Model | <default_model_name> |

## Concept
<1-2 paragraphs: what this example demonstrates and which NEAT capabilities it exercises.>

## Supported Models
Also works with: `<model_variant_1>`, `<model_variant_2>`

Download any variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get <model_variant_1> && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get <default_model_name> && cd ../..`

## Important Behavior
- This example expects the model path to be provided explicitly at runtime.
- Input and output paths are user-provided.
- Update this section with any example-specific behavior that affects runtime results.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/<category>/<name>/<binary> <args>`
- Required arguments:
  `<required_arg_1> <required_arg_2>`
- Optional arguments:
  `--flag <value>`

### Python
- Invocation:
  `python3 examples/<category>/<name>/main.py <args>`
- Required arguments:
  `<required_arg_1> <required_arg_2>`
- Optional arguments:
  `--flag <value>`

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/<category>/<name>/<binary>
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/<category>/<name>
cmake -S . -B build
cmake --build build -j
```

Binary output:
```bash
./build/<binary>
```

## Run
### C++
```bash
./build/examples/<category>/<name>/<binary> <args>
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/<category>/<name>/requirements.txt
python3 examples/<category>/<name>/main.py <args>
```

## Debugging Notes
- Confirm the model file exists under `assets/models/`.
- Confirm input paths exist and are readable.
- Confirm output directories are writable.

## Reference
- C++ source: `main.cpp`
- Python source: `main.py`
