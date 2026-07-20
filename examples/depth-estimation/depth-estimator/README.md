# Depth Estimator

## Metadata

| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation, depth-anything, folder-inference |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | depth-estimator |
| Model | depth_anything_v2_vits |

## Concept

Depth-map generation for image folders. The example runs inference for each image and writes a visual depth map.

## Preview

![Depth estimator preview](../../../portal/assets/examples/depth-estimation/depth-estimator/image.png)

## Prerequisites

- `sima-cli` on a supported Modalix or DevKit target.

## Install Apps

1. Choose a version from the [Neat Apps releases](https://github.com/sima-neat/apps/releases).
2. Install that version and enter the installed bundle:

```bash
sima-cli neat install apps@<release-version>
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model package | Role | Model Zoo name |
| --- | --- | --- |
| `depth_anything_v2_vits_mpk.tar.gz` | Default | `depth_anything_v2_vits` |

The required platform version is recorded in `manifest.json`.

```bash
mkdir -p models
cd models
sima-cli modelzoo -v <platform-version> get depth_anything_v2_vits
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Edit `examples/depth-estimation/depth-estimator/src/common/config.yaml`.

```yaml
model:
  path: <model-path>

io:
  input_dir: assets/datasets/coco
  output_dir: sandbox/depth-estimator
```

## Run

### C++

```bash
./examples/depth-estimation/depth-estimator/src/cpp/pre-built/depth-estimator \
  --config examples/depth-estimation/depth-estimator/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/depth-estimation/depth-estimator/src/python/requirements.txt
python3 examples/depth-estimation/depth-estimator/src/python/main.py \
  --config examples/depth-estimation/depth-estimator/src/common/config.yaml
```

## Troubleshooting

- Verify `model.path` if startup fails.
- Confirm `io.input_dir` contains supported images.
- Confirm `io.output_dir` is writable.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
