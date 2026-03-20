# <Example Name>

## Metadata
| Field | Value |
| --- | --- |
| Category | <classification / object-detection / semantic-segmentation / instance-segmentation / depth-estimation / throughput> |
| Difficulty | <Beginner / Intermediate / Advanced> |
| Tags | <comma-separated tags> |
| Languages | C++, Python |
| Status | <experimental / stable> |
| Binary Name | <cmake_target_name> |
| Model | <default_model_name> |

## Concept
<1-2 paragraphs: what this example demonstrates and which NEAT capabilities it exercises.>

## Preview
Optional. If you have a demo screenshot for the portal detail page, place it here immediately after `Concept`.

```md
![Demo screenshot](../../../assets/portal/<category>/<example>/image.png)
```

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
  `python3 examples/<category>/<name>/python/main.py <args>`
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
cd <apps-repo-root>
cmake -S examples/<category>/<name>/cpp -B build/<name>
cmake --build build/<name> -j
```

Binary output:
```bash
./build/<name>/<binary>
```

## Run
### C++
```bash
./build/examples/<category>/<name>/<binary> <args>
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/<category>/<name>/python/requirements.txt
python3 examples/<category>/<name>/python/main.py <args>
```

## Debugging Notes
- Confirm the model file exists under `assets/models/`.
- Confirm input paths exist and are readable.
- Confirm output directories are writable.

## Source Files
- C++ source: `cpp/main.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python source: `python/main.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Shared assets: `common/`
