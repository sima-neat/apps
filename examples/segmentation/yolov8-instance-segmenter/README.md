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
Use the SDK platform version wherever `<platform-version>` appears.

Default model: `yolo_v8n_seg`.

Download the default model:

```bash
mkdir -p assets/models
cd assets/models

sima-cli modelzoo -v <platform-version> get yolo_v8n_seg

cd ../..
```

The command stores the model under `assets/models/` as a repo-local convention. `model.path` can point to any readable model package path.

## Prerequisites
- Installed Neat Development Environment.
- Model artifacts are user-managed. Download the default model, or set `model.path` to another readable model package.

## Get The Apps Repo
Use the [Neat Development Environment](https://developer.sima.ai/software/getting-started/dev-environment/) for setup and compilation. Install the Neat Library first by following the [Neat Library guide](https://developer.sima.ai/software/getting-started/neat-library/).

Clone and build the apps repo in the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After building, run the example commands below on the Modalix/DevKit board.

## Configure
Edit `examples/segmentation/yolov8-instance-segmenter/src/common/config.yaml`.

```yaml
model:
  path: <model-path>                   # Path to the model package.

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

## Appendix: Additional Models
This example also works with `yolo_v8s_seg`, `yolo_v8m_seg`, and `yolo_v8l_seg`.
Replace `yolo_v8n_seg` in the download command and update `model.path`.

## Source Files
- C++ source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`
