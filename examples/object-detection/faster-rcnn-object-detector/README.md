# Faster R-CNN Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | faster-rcnn, object-detection, two-stage-detector |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | faster-rcnn-object-detector |
| Model | Faster R-CNN ResNet-50 FPN, merged backbone+RPN and box-head+predictor |

## Concept
This example runs a two-stage Faster R-CNN detector using two compiled NEAT model packages:

- `backbone_rpn_head_640_640_mpk.tar.gz` for ResNet-50/FPN backbone plus concatenated RPN head
- `box_head_predictor_640_640_mpk.tar.gz` for the ROI box head plus class/regression predictor

The C++ and Python apps perform the same glue logic as the merged quantized pipeline: RPN proposal decoding, ROI Align, final box decoding, class filtering, NMS, and visualization.

## Configure
Edit `examples/object-detection/faster-rcnn-object-detector/src/common/config.yaml`.

```yaml
models:
  backbone_rpn:
    path: assets/models/backbone_rpn_head_640_640_mpk.tar.gz
  head_predictor:
    path: assets/models/box_head_predictor_640_640_mpk.tar.gz

io:
  input_dir: assets/test_images
  output_dir: sandbox/faster-rcnn-object-detector
```

## Run
C++ from the Apps repo root after building:

```bash
./build/examples/object-detection/faster-rcnn-object-detector/faster-rcnn-object-detector \
  --config examples/object-detection/faster-rcnn-object-detector/src/common/config.yaml
```

Python:

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/faster-rcnn-object-detector/src/python/requirements.txt
python3 examples/object-detection/faster-rcnn-object-detector/src/python/main.py \
  --config examples/object-detection/faster-rcnn-object-detector/src/common/config.yaml
```

## Model Packages
Generate the packages from the Faster R-CNN demo ModelSDK scripts, then copy or symlink them into `assets/models/`:

```text
build/backbone_rpn_head_640_640/backbone_rpn_head_640_640_mpk.tar.gz
build/box_head_predictor_640_640/box_head_predictor_640_640_mpk.tar.gz
```

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
