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
Depth-map generation for image folders. The example runs inference per image and writes visual depth outputs.

## Preview
Snippet from a pipeline run:

![Depth estimator preview](../../../assets/portal/depth-estimation/depth-estimator/image.png)

## Supported Models
This README is written for SDK `2.1.2`.

Primary model: `depth_anything_v2_vits`

Download the model:

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo -v 2.1.2 get depth_anything_v2_vits
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
Edit `examples/depth-estimation/depth-estimator/src/common/config.yaml`.

```yaml
model:
  path: assets/models/depth_anything_v2_vits_mpk.tar.gz # Model package to load.

io:
  input_dir: assets/test_images                         # Folder containing input images.
  output_dir: sandbox/depth-estimator                   # Folder for depth visualizations.
```

## Run
### C++
```bash
./build/examples/depth-estimation/depth-estimator/depth-estimator \
  --config examples/depth-estimation/depth-estimator/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/depth-estimation/depth-estimator/src/python/requirements.txt
python3 examples/depth-estimation/depth-estimator/src/python/main.py \
  --config examples/depth-estimation/depth-estimator/src/common/config.yaml
```

## Debugging Notes
- Check model path first if startup fails.
- If no outputs are produced, verify `input_dir` has valid images.
- Check write permissions on `output_dir`.

## Source Files
- C++ source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`
