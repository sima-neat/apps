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
Minimal Model API usage with a compiled ResNet50 model package. The example loads the package, runs single-image inference, and prints top-1/top-5 classification output.

## Preview
The pipeline is trying to classify this goldfish image.

![Image classifier goldfish input](../../../portal/assets/examples/classification/image-classifier/image.jpeg)

## Supported Models
Use the SDK platform version wherever `<platform-version>` appears.

Primary model: `resnet_50`

Download the model:

```bash
mkdir -p models
cd models
sima-cli modelzoo -v <platform-version> get resnet_50
cd ..
```

The command stores the model under `models/` as a repo-local convention. `model.path` can point to any readable model package path.

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- Model artifacts are user-managed and should be downloaded into `models/`. Download the default model, or set `model.path` to another readable model package.

## Get The Apps Repo
Use the [Neat Development Environment](https://developer.sima.ai/software/getting-started/dev-environment/) with the [Neat Library](https://developer.sima.ai/software/getting-started/neat-library/) installed for setup and compilation.

Clone and build the apps repo inside the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After building, run the example commands below on the Modalix/DevKit board.

## Configure
Edit `examples/classification/image-classifier/src/common/config.yaml` if you want to use a different image or threshold.

```yaml
model:
  path: <model-path>                        # Path to the model package.

io:
  image: null                               # Input image. null uses the sample image.

validation:
  min_probability: 0.50                     # Minimum expected-class probability.
```

## Run
### C++
```bash
./build/examples/classification/image-classifier/image-classifier \
  --config examples/classification/image-classifier/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/classification/image-classifier/src/python/requirements.txt
python3 examples/classification/image-classifier/src/python/main.py \
  --config examples/classification/image-classifier/src/common/config.yaml
```

## Debugging Notes
- If you see a model load error, verify `model.path` points to a readable model package.
- If image decode fails, set `io.image` in the config.
- If top-1 validation fails, try lowering `validation.min_probability` for debug runs.

## Source Files
- C++ source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`
