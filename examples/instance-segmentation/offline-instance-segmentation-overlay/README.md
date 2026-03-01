# YOLOv8n Instance Segmentation (DetessDequant)

## Metadata
| Field | Value |
| --- | --- |
| Category | instance-segmentation |
| Difficulty | Intermediate |
| Tags | instance-segmentation |
| Status | experimental |
| Binary Name | offline-instance-segmentation-overlay |
| Model | yolo_v8n_seg |

## Concept
Minimal YOLOv8-seg pipeline using DetessDequant postprocessing (no boxdecode). Processes images and writes segmentation results.

## Supported Models
Also works with: `yolo_v8s_seg`, `yolo_v8m_seg`, `yolo_v8l_seg`

Download any variant: `sima-cli modelzoo get yolo_v8s_seg`

## Prerequisites
- Installed NEAT SDK
- Model downloaded: `sima-cli modelzoo get yolo_v8n_seg`

## Run
### C++
```bash
./build/examples/instance-segmentation/offline-instance-segmentation-overlay/offline-instance-segmentation-overlay models/yolo_v8n_seg_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
pip install -r examples/instance-segmentation/offline-instance-segmentation-overlay/requirements.txt
python3 examples/instance-segmentation/offline-instance-segmentation-overlay/main.py models/yolo_v8n_seg_mpk.tar.gz <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
- Python: `main.py`
