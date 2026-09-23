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

| Model package | Role | Source |
| --- | --- | --- |
| `depth_anything_v2_vits_mpk.tar.gz` | Default | Direct artifact |

This model comes from the direct SDK artifact release below, which can differ from the installed platform version.

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download \
  "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/depth_anything_v2_vits_mpk.tar.gz"
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

## Expected Result

The application prints one line per image and a final count. The C++ binary also
prints `[INFER]` and `[DEPTH]` diagnostic lines between them and quotes the
filenames; both are normal.

```text
[1/21] 000000081061.jpg -> 000000081061.png
...
Done: 21 images processed
```

`io.output_dir` (default `sandbox/depth-estimator`) then holds one PNG depth
visualization per input image. With the packaged `assets/datasets/coco` folder
that is 21 files. Each is a side-by-side image: the resized input on the left,
its depth colormap on the right. The colormap is normalized per image, so
brightness is relative within one image and not comparable between images.

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
