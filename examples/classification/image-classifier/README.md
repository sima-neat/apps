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

Minimal Model API usage with a compiled ResNet50 package. The example runs single-image inference and prints top-1 and top-5 classification results.

## Preview

The pipeline classifies this goldfish image:

![Image classifier goldfish input](../../../portal/assets/examples/classification/image-classifier/image.jpeg)

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
| `resnet_50_mpk.tar.gz` | Default | `resnet_50` |

The required platform version is recorded in `manifest.json`.

```bash
mkdir -p models
cd models
sima-cli modelzoo -v <platform-version> get resnet_50
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Edit `examples/classification/image-classifier/src/common/config.yaml` to use a different image or threshold.

```yaml
model:
  path: <model-path>

io:
  image: null
  fallback_image_url: https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG

validation:
  min_probability: 0.50
```

With `io.image: null`, the example downloads `fallback_image_url` and therefore requires network access. For offline use, set `io.image` to a readable local image path; the fallback URL is not used when a local path is configured.

## Run

### C++

```bash
./examples/classification/image-classifier/src/cpp/pre-built/image-classifier \
  --config examples/classification/image-classifier/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/classification/image-classifier/src/python/requirements.txt
python3 examples/classification/image-classifier/src/python/main.py \
  --config examples/classification/image-classifier/src/common/config.yaml
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
