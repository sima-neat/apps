# <Example Name>

## Metadata
| Field | Value |
| --- | --- |
| Category | <benchmarking / classification / object-detection / tracking / face-detection / segmentation / depth-estimation / genai / throughput> |
| Difficulty | <Beginner / Intermediate / Advanced> |
| Tags | <comma-separated tags> |
| Languages | C++, Python |
| Status | <experimental / stable> |
| Binary Name | <cmake_target_name> |
| Model | <default_model_name> [https://example.com/path/to/<default_model_name>_mpk.tar.gz] |

## Concept
<1-2 paragraphs: what this example demonstrates and which Neat Library capabilities it exercises.>

## Preview
Optional. If you have a demo screenshot for the portal detail page, place it here immediately after `Concept`.

```md
![Demo screenshot](../../../assets/portal/<category>/<example>/image.png)
```

## Supported Models
This README is written for SDK `2.1.2`.

Default model: `<default-model>`.

Download the default model:

```bash
mkdir -p assets/models
cd assets/models

sima-cli modelzoo -v 2.1.2 get <default-model>

cd ../..
```

For another supported model, replace `<default-model>` in the download command and config path.

## Prerequisites
- Installed Neat Development Environment.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- If the model is not available through modelzoo, add a direct download URL in the `Model` metadata field using the `[https://...]` suffix.

## Get The Apps Repo
Install the Neat Library first by following the official [Neat Library installation guide](https://developer.sima.ai/software/getting-started/installation/neat-library).

Then clone and build the apps repo:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After this setup, follow the example-specific commands below.

## Configure
Edit `examples/<category>/<name>/src/common/config.yaml`.

```yaml
model:
  path: assets/models/<model-pack>.tar.gz # Model package to load.

io:
  input_dir: assets/test_images           # Folder containing input images.
  output_dir: sandbox/<name>              # Folder for generated outputs.
```

## Run
### C++
```bash
./build/examples/<category>/<name>/<binary> \
  --config examples/<category>/<name>/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/<category>/<name>/src/python/requirements.txt
python3 examples/<category>/<name>/src/python/main.py \
  --config examples/<category>/<name>/src/common/config.yaml
```

## Debugging Notes
- Confirm the model file exists under `assets/models/`.
- Confirm input paths exist and are readable.
- Confirm output directories are writable.

## Appendix: Additional Models
This example can also run with `<model_variant_1>` or `<model_variant_2>`. Replace the default model name in the download command and config path.

## Source Files
- Test scope: `tests/test-scope.yaml`
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/`
