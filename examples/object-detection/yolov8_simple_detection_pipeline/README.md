# YOLOv8n Simple Detection Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Beginner |
| Tags | object-detection, yolov8, folder-inference |
| Status | experimental |
| Binary Name | yolov8_simple_detection_pipeline |
| Model | yolo_v8n |

## Concept
Minimal YOLOv8n image-folder detection pipeline.

Both C++ and Python examples read images from an input folder, run inference one image at a time, overlay bounding boxes with class/confidence labels on the original image, and save the result to an output folder.

## Supported Models
Also works with: `yolo_v8s`, `yolo_v8m`, `yolo_v8l`

Download any variant: `sima-cli modelzoo get yolo_v8s`

## Prerequisites
- Installed NEAT SDK
- Model downloaded: `sima-cli modelzoo get yolo_v8n`

## Run
From the `apps` directory:

### C++
```bash
./build/examples/object-detection/yolov8_simple_detection_pipeline/yolov8_simple_detection_pipeline \
  models/yolo_v8n_mpk.tar.gz \
  examples/object-detection/yolov8_simple_detection_pipeline/coco_label.txt \
  <input_dir> \
  <output_dir>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
pip install -r examples/object-detection/yolov8_simple_detection_pipeline/requirements.txt
python examples/object-detection/yolov8_simple_detection_pipeline/main.py \
  models/yolo_v8n_mpk.tar.gz \
  examples/object-detection/yolov8_simple_detection_pipeline/coco_label.txt \
  <input_dir> \
  <output_dir>
```

## Notes
- Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Labels file is required at runtime (one class label per line)
- You can use: `examples/object-detection/yolov8_simple_detection_pipeline/coco_label.txt`
- Output images are written as `.png` files into `<output_dir>`
- This is intentionally a simple single-folder, single-model pipeline (no RTSP, no multistream)

## Source Files
- C++: `main.cpp`
- Python: `main.py`
