# SSD-MobileNetV2 Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | ssd, mobilenet, object-detection, folder-input, coco |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | ssd-mobilenet-object-detector |
| Model | ssd_mobilenet_v2_heads [https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/ssd_mobilenet_v2_heads_mpk.tar.gz] |

## Concept
This example demonstrates folder-based object detection with a compiled **SSD-MobileNetV2**
model (the TensorFlow Object Detection API model `ssd_mobilenet_v2_coco_2018_03_29`). Each
input image is stretched to the 300×300 model frame, normalized to `[-1, 1]`, run through the
Neat Library pipeline, and then decoded into COCO object boxes on the host.

The compiled pack emits the model's **12 grouped convolution heads** (6 localization + 6
classification, feature maps `19/10/5/3/2/1` → **1917 priors**). The example reassembles those
heads in anchor-major layout (`a*4 + coord` for boxes, `a*91 + class` for scores), decodes each
prior with the recovered `ssd_anchor_generator` priors and the `FasterRcnnBoxCoder`
(scales `10,10,5,5`), applies **sigmoid** scoring, runs **per-class NMS**, maps detections back
onto the original image, and writes annotated output images. This host-side decode is bit-exact
with the reference TensorFlow model.

Unlike DETR (letterbox + ImageNet normalization), this model was trained with a
`fixed_shape_resizer`, so preprocessing is a **direct stretch** to 300×300 and boxes
back-project with a simple per-axis scale.

## Supported Models
Use the SDK platform version wherever `<platform-version>` appears.

Default model: `ssd_mobilenet_v2_heads`.

Download the default model:

```bash
mkdir -p assets/models
cd assets/models

sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/ssd_mobilenet_v2_heads_mpk.tar.gz

cd ../..
```

The command stores the model under `assets/models/` as a repo-local convention. `model.path`
can point to any readable model package path.

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- Model artifacts are user-managed and should be downloaded into `assets/models/`. Download the
  default model, or set `model.path` to another readable model package.
- The only shipped asset is `src/common/coco_labels.txt` — 91 lines (`0=background`,
  `1..90` = MS-COCO ids). The 1917 anchor priors are **generated in code** at startup from
  the per-level prior table (the same table baked into the on-device SSD decode kernel), so
  no anchor asset is required.

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
Edit `examples/object-detection/ssd-mobilenet-object-detector/src/common/config.yaml`.

```yaml
model:
  path: <model-path>         # Path to the model package.
  labels: examples/object-detection/ssd-mobilenet-object-detector/src/common/coco_labels.txt

io:
  input_dir: assets/test_images                       # Folder containing input images.
  output_dir: sandbox/ssd-mobilenet-object-detector   # Folder for annotated images.

decode:
  score_threshold: 0.30      # Minimum sigmoid object confidence.
  nms_iou: 0.60              # Overlap threshold for per-class NMS.
  max_detections: 100        # Max detections to keep per image.

runtime:
  timeout_ms: 20000          # Inference timeout in milliseconds.
  num_runs: 1                # Number of repeated runs (profiling).
  profile: false             # Print profiling summaries.
```

The `labels` path is repo-relative (resolved from the apps root). If the working directory
differs, the app falls back to the copy shipped next to the example under `src/common/`.

## Run
### C++
```bash
./build/examples/object-detection/ssd-mobilenet-object-detector/ssd-mobilenet-object-detector \
  --config examples/object-detection/ssd-mobilenet-object-detector/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/ssd-mobilenet-object-detector/src/python/requirements.txt
python3 examples/object-detection/ssd-mobilenet-object-detector/src/python/main.py \
  --config examples/object-detection/ssd-mobilenet-object-detector/src/common/config.yaml
```

## Debugging Notes
- Confirm `model.path` points to a readable model package.
- If boxes look vertically squished or shifted toward the frame center, the resize mode is wrong.
  This model requires a **stretch** (not letterbox) resize; the example already applies it.
- If detections are missing but the pipeline runs, lower `decode.score_threshold` (int8
  quantization of the classification heads can drop weak detections).
- Set `runtime.profile: true` to print graph vs. postprocessing timing for the first image.

## Appendix: On-Device Decode
This example performs the SSD box decode **on the host**, so it needs no custom kernel. The same
12-head pack can also be decoded **on-device** by the `neatobjectdecode` kernel's SSD variant
(model-managed box decode) for a streaming RTSP → Insight deployment; that path is out of scope
for this folder-input example.

## Source Files
- Test scope: `tests/test-scope.yaml`
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/` (`config.yaml`, `coco_labels.txt`)
