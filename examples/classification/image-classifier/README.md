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

![Image classifier goldfish input](../../../assets/portal/classification/image-classifier/image.jpeg)

## Supported Models
Use the platform version wherever `<platform-version>` appears.

Primary model: `resnet_50`

Download the model:

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo -v <platform-version> get resnet_50
cd ../..
```

## Prerequisites
- Installed Neat Development Environment.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.

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
Edit `examples/classification/image-classifier/src/common/config.yaml` if you want to use a different image or threshold.

```yaml
model:
  path: assets/models/resnet_50_mpk.tar.gz  # Model package to load.

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
- If you see a model load error, verify the model file exists at `assets/models/resnet_50_mpk.tar.gz`.
- If image decode fails, set `io.image` in the config.
- If top-1 validation fails, try lowering `validation.min_probability` for debug runs.

## Source Files
- C++ source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`
