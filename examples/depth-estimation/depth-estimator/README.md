# Depth Estimator

## Metadata

| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation, depth-anything, folder-inference |
| Languages | C++, Python |
| Status | stable |
| Binary Name | depth-estimator |
| Model | depth_anything_v2_vits |

## Concept

Creates visual depth maps for a folder of images with Depth Anything V2.

## Preview

![Depth estimator preview](../../../portal/assets/examples/depth-estimation/depth-estimator/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/depth-estimation/depth-estimator
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model package | Role | Model Zoo name |
| --- | --- | --- |
| `depth_anything_v2_vits_mpk.tar.gz` | Default | `depth_anything_v2_vits` |

Model packages come from the Model Zoo release below, which can differ from the installed platform version.

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli modelzoo -v "${MODELZOO_VERSION}" get depth_anything_v2_vits
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set `model.path`, `io.input_dir`, and `io.output_dir`.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/depth-estimator \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
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
