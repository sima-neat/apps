# RetinaFace Face Detection

## Metadata
| Field | Value |
| --- | --- |
| Category | face-detection |
| Difficulty | Beginner |
| Tags | retinaface, face-detection |
| Status | experimental |
| Binary Name | retinaface-face-detection |
| Model | retinaface_mobilenet25 |

## Concept
Minimal RetinaFace face detection example. It runs the compiled RetinaFace model on an input image, decodes outputs, applies confidence filtering + NMS, and writes an annotated image.

## Prerequisites
- Installed NEAT SDK + built apps artifacts.
- Model package available on disk. By default this example uses:
  - `apps/assets/models/retinaface_mobilenet25_mod_0_mpk.tar.gz`

## Run
### C++
```bash
./build/examples/face-detection/retinaface-face-detection_cpp/retinaface-face-detection \
  apps/assets/test_images/image.png \
  --model apps/assets/models/retinaface_mobilenet25_mod_0_mpk.tar.gz \
  --output /tmp/retinaface_out.png \
  --conf 0.4 --nms 0.9
```

### Python
```bash
python apps/examples/face-detection/retinaface-face-detection/python/main.py \
  apps/assets/test_images/image.png \
  --model apps/assets/models/retinaface_mobilenet25_mod_0_mpk.tar.gz \
  --output /tmp/retinaface_out.png \
  --conf 0.4 --nms 0.9
```

## Source Files
- C++ source: `cpp/main.cpp`
- Python source: `python/main.py`

