# Image Classifier

## Metadata

| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | classification, model, mpk |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | image-classifier |
| Model | resnet_50 |

## Concept

Classifies one image with ResNet50 and prints the top five predictions with confidence scores.

## Preview

The pipeline classifies this goldfish image:

![Image classifier goldfish input](../../../portal/assets/examples/classification/image-classifier/image.jpeg)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/classification/image-classifier
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model package | Role | Model Zoo name |
| --- | --- | --- |
| `resnet_50_mpk.tar.gz` | Default | `resnet_50` |

Model packages come from the Model Zoo release below, which can differ from the installed platform version.

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli modelzoo -v "${MODELZOO_VERSION}" get resnet_50
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set `model.path`. To classify your own image, set `io.image` to a readable local path. If you leave it empty, the application downloads the sample goldfish image and needs network access.

Change `validation.min_probability` only if you want a different minimum confidence for the result check.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/image-classifier \
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

- Verify `model.path` if model loading fails.
- Set `io.image` to a readable local image if downloading or decoding the fallback image fails.
- Lower `validation.min_probability` when investigating validation failures.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
