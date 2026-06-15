# YOLOv8 Instance Segmenter

## Metadata
| Field | Value |
| --- | --- |
| Category | segmentation |
| Difficulty | Intermediate |
| Tags | segmentation, yolov8, instance-segmentation, folder-inference |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | yolov8-instance-segmenter |
| Model | yolo_v8n_seg |

## Concept
Offline YOLOv8 instance segmentation over image folders using YOLOv8 segmentation outputs and DetessDequant post-processing.

## Preview
Snippet from a pipeline run:

![Instance segmenter preview](../../../assets/portal/segmentation/yolov8-instance-segmenter/image.jpg)

## Supported Models
Use the platform version wherever `<platform-version>` appears.

Also works with: `yolo_v8s_seg`, `yolo_v8m_seg`, `yolo_v8l_seg`

Download one model:

```bash
mkdir -p assets/models
cd assets/models

PLATFORM_VERSION="<platform-version>"
MODEL=yolo_v8n_seg

sima-cli modelzoo -v "${PLATFORM_VERSION}" get "${MODEL}"

cd ../..
```

Set `PLATFORM_VERSION` to your installed SDK platform version, and replace `MODEL` with a supported modelzoo name.

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
Edit `examples/segmentation/yolov8-instance-segmenter/src/common/config.yaml`.

```yaml
model:
  path: assets/models/yolo_v8n_seg_mpk.tar.gz  # Model package to load.

io:
  input_dir: assets/test_images                 # Folder containing input images.
  output_dir: sandbox/yolov8-instance-segmenter # Folder for annotated images.

decode:
  score_threshold: 0.25                         # Minimum instance confidence.
```

## Run
### C++
```bash
./build/examples/segmentation/yolov8-instance-segmenter/yolov8-instance-segmenter \
  --config examples/segmentation/yolov8-instance-segmenter/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/yolov8-instance-segmenter/src/python/requirements.txt
python3 examples/segmentation/yolov8-instance-segmenter/src/python/main.py \
  --config examples/segmentation/yolov8-instance-segmenter/src/common/config.yaml
```

## Debugging Notes
- If startup fails, verify model file path and filename.
- If output is empty, check `decode.score_threshold` and `runtime.infer_size`.
- Ensure output directory is writable.

## Source Files
- C++ source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`
