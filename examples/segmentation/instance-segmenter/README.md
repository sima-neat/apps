# Instance Segmenter

## Metadata
| Field | Value |
| --- | --- |
| Category | segmentation |
| Difficulty | Intermediate |
| Tags | segmentation, yolo26, rtsp, insight |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | instance-segmenter |
| Model | yolo26m-seg-bf16-b1 [https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-segmentation/yolo26m-seg-bf16-b1.tar.gz] |

## Concept
`instance-segmenter` is a single-camera YOLO26 instance segmentation example:

- ingest one RTSP camera stream
- decode the stream into frames
- run YOLO26 instance segmentation
- render mask overlays on the video
- send H.264 video plus segmentation metadata to Insight

The example keeps RTSP ingest, model inference, and Insight output separate so
the segmentation behavior can be debugged independently from transport issues.

## Preview
Snippet from a pipeline run:

![Instance segmenter preview](../../../assets/portal/segmentation/instance-segmenter/image.jpg)

## Supported Models
Supported YOLO26 segmentation models:

- `yolo26n-seg-bf16-mla_tess.tar.gz`
- `yolo26s-seg-bf16-mla_tess.tar.gz`
- `yolo26m-seg-bf16-mla_tess.tar.gz`
- `yolo26l-seg-bf16-mla_tess.tar.gz`
- `yolo26x-seg-bf16-mla_tess.tar.gz`
- `yolo26m-seg-bf16-b1.tar.gz`
- `yolo26m-seg-bf16-mla_tess-b1.tar.gz`
- `yolo26m-seg-int8-b1.tar.gz`

Download the supported variants:

```bash
mkdir -p assets/models/YOLO26-SEGMENTATION
cd assets/models/YOLO26-SEGMENTATION

sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-segmentation/yolo26n-seg-bf16-mla_tess.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-segmentation/yolo26s-seg-bf16-mla_tess.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-segmentation/yolo26m-seg-bf16-mla_tess.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-segmentation/yolo26l-seg-bf16-mla_tess.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-segmentation/yolo26x-seg-bf16-mla_tess.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-segmentation/yolo26m-seg-bf16-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-segmentation/yolo26m-seg-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-segmentation/yolo26m-seg-int8-b1.tar.gz

cd ../../..
```

## Prerequisites
- Installed Neat framework and Insight on the DevKit.
- RTSP camera source, or an Insight/tool-mediasources RTSP stream.
- A YOLO26 segmentation model package downloaded locally.
- `model.path`, `model.labels`, `source.rtsp_url`, and `output.insight.host` set in `src/common/config.yaml`.

## Important Behavior
- C++ and Python read runtime values from `src/common/config.yaml`.
- `model.path` must point to a valid YOLO26 segmentation model package.
- `model.labels` points to the COCO labels file used for overlays and metadata.
- `source.rtsp_url` must point to a live RTSP stream.
- `output.insight.host` must point to the host running the Insight receiver/viewer.
- If `inference.frames` is zero, the sample runs continuously.
- Masks are rendered into the video stream. Metadata contains object labels, confidences, and boxes.
- `output.save_dir` and `output.save_every` can be used to save sampled annotated frames.
- The model path uses model-managed preprocessing with `COCO_YOLO` normalization and `YoloV26Seg` decode.

## Command-Line Options
- `--config <path>`
  Optional. YAML config path. Defaults to `src/common/config.yaml`.
- `--validate-config-only`
  Validate YAML config and exit without opening the RTSP stream.

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/segmentation/instance-segmenter/instance-segmenter
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/segmentation/instance-segmenter
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/instance-segmenter
```

## Run
### C++
```bash
./build/examples/segmentation/instance-segmenter/instance-segmenter \
  --config examples/segmentation/instance-segmenter/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/instance-segmenter/src/python/requirements.txt
python3 examples/segmentation/instance-segmenter/src/python/main.py \
  --config examples/segmentation/instance-segmenter/src/common/config.yaml
```

Example config:

```yaml
model:
  path: assets/models/YOLO26-SEGMENTATION/yolo26m-seg-bf16-b1.tar.gz
  labels: examples/segmentation/instance-segmenter/src/common/coco_label.txt

source:
  rtsp_url: rtsp://<host>:8554/<stream>
  tcp: true
  latency_ms: 100

inference:
  frames: 0
  min_score: 0.55
  nms_iou: 0.60
  max_detections: 50

runtime:
  profile: false
  profile_interval: 100

output:
  save_dir: ""
  save_every: 0
  mask_alpha: 0.55
  mask_threshold: 0.50
  draw_boxes: true
  insight:
    host: <insight-host-ip>
    video_port: 9000
    metadata_port: 9100
```

## Debugging Notes
- If startup fails, verify `model.path` and `source.rtsp_url`.
- If the app times out waiting for RTSP, verify source reachability first.
- If Insight receives no video, verify `output.insight.host` and UDP ports.
- If saved frames are needed for inspection, set `output.save_dir` and `output.save_every`.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared config: `src/common/config.yaml`
- Shared labels: `src/common/coco_label.txt`
