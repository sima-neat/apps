# Depth Estimator

## Metadata
| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | depth-estimator |
| Model | depth_anything_v2_vits |

## Concept
Depth-map generation for image folders. The example runs inference per image and writes visual depth outputs.

## Preview
Snippet from a pipeline run:

![Depth estimator preview](../../../assets/portal/depth-estimation/depth-estimator/image.png)

## Supported Models
Use the platform version wherever `<platform-version>` appears.

Primary model: `depth_anything_v2_vits`

Download into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo -v <platform-version> get depth_anything_v2_vits && cd ../..`

## Prerequisites
- Installed Neat Development Environment.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo -v <platform-version> get depth_anything_v2_vits && cd ../..`

## Get The Apps Repo
Install the Neat Library first by following the official [Neat Library installation guide](https://developer.sima.ai/software/getting-started/installation/neat-library).

Then clone and build the apps repo:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After this setup, follow the example-specific commands below.

## Important Behavior
- Runtime settings are read from `src/common/config.yaml` by default.
- Input directory is scanned for common image extensions.
- Output files are written to the configured output directory.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/depth-estimation/depth-estimator/depth-estimator [--config <path>]`
- Optional arguments:
  `--config <path>`: YAML config path. Defaults to `src/common/config.yaml`.

### Python
- Invocation:
  `python examples/depth-estimation/depth-estimator/src/python/main.py [--config <path>]`
- Optional arguments:
  `--config <path>`: YAML config path. Defaults to `src/common/config.yaml`.

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/depth-estimation/depth-estimator/depth-estimator
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/depth-estimation/depth-estimator
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/depth-estimator
```

## Run
Edit `examples/depth-estimation/depth-estimator/src/common/config.yaml` to point at the model and image folder.

### C++
```bash
./build/examples/depth-estimation/depth-estimator/depth-estimator
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/depth-estimation/depth-estimator/src/python/requirements.txt
python examples/depth-estimation/depth-estimator/src/python/main.py
```

## Debugging Notes
- Check model path first if startup fails.
- If no outputs are produced, verify `input_dir` has valid images.
- Check write permissions on `output_dir`.

## Source Files
- C++ source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`
