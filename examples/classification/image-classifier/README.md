# Image Classifier

## Metadata

| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | classification, model, mpk |
| Languages | C++, Python |
| Status | stable |
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
export MODELZOO_VERSION="2.1.3"
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

## Expected Result

Both implementations print the top-1 index with its probability and the top-5
list, then confirm the top-1 class. They word it differently, so check against
the one you ran. Exact probabilities vary between runs and between the two
implementations; what matters is that class `1` wins by a wide margin.

C++, which prints at the default stream precision:

```text
[model] top1 index=1 score=16.435 prob=0.946191
[model] top5: 1:0.946191 0:0.0495301 392:0.00298376 389:0.000480504 29:0.000206855
[model] top-1 matches expected class 1
```

Python, which prints four decimal places:

```text
top1 index=1 score=16.4350 prob=0.9532
top5: 1:0.9532 0:0.0434 392:0.0023 389:0.0004 29:0.0002
PASS
```

By default `io.image` is `null`, so the application downloads the goldfish image
from `io.fallback_image_url` and needs network access. Class `1` should win with
a probability well above the `validation.min_probability` default of `0.20`.

A failed check reports the reason rather than failing silently. Python prints
one of these to stderr and stops, so you see the first that applies, never both:

```text
FAIL: expected top1=1 (goldfish), got 393
```

```text
FAIL: top1 prob 0.1832 < 0.2
```

C++ raises instead, and the message reaches stderr through the top-level
handler:

```text
Error: model: top-1 mismatch: expected 1 got 393
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
