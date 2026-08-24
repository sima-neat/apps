# MIPI Multi-Model

## Metadata

| Field | Value |
| --- | --- |
| Category | benchmarking |
| Difficulty | Intermediate |
| Tags | camera, mipi, zero-copy, yolo26, pose, segmentation, classification, depth |
| Languages | Python |
| Status | stable |
| Binary Name | mipi-multi-model |
| Model | YOLO26, SSD-MobileNet V3, ResNet-50, MiDaS |

## Concept

Runs a selected detection, pose, segmentation, classification, or depth model directly from a zero-copy MIPI camera and prints one compact result per frame. Detection and segmentation profiles can also stream synchronized H.264 video and inference metadata to Insight.

The application loads one profile per process, keeping MLA memory use predictable while making each task independently testable.

## Preview

![MIPI multi-model preview](../../../portal/assets/examples/benchmarking/mipi-multi-model/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- A MIPI camera supported by the installed kernel, libcamera pipeline, and IPA configuration. The IMX568 overlay is one supported setup.
- Neat Library 0.4.0 or later with strict camera DMA-BUF support.

Confirm that libcamera sees the camera before running the application:

```bash
cam -l
```

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/benchmarking/mipi-multi-model
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The default profile uses YOLO26n detection. Every profile consumes a compiled `.tar.gz` package directly.

| Profile | Task | Package source |
| --- | --- | --- |
| `detect` | YOLO26n detection | Model Zoo 2.1.3 `yolo_26n` |
| `pose` | YOLO26n pose | Model Zoo 2.1.3 `yolo_26n_pose` |
| `segment` | YOLO26n instance segmentation | Model Zoo 2.1.3 `yolo_26n_seg` |
| `ssd` | SSD-MobileNet V3 detection | `sima-neat/models` Model Registry |
| `classify` | ResNet-50 classification | Model Zoo 2.1.3 `resnet_50` |
| `depth` | MiDaS v2.1 Small depth | Model Zoo 2.1.3 `midas_v21_small_256` |

Fetch the Model Zoo profiles through `sima-cli`:

```bash
python3 ${APP_DIR}/tools/fetch_models.py \
  detect pose segment classify depth
```

Fetch SSD-MobileNet V3 from the `sima-neat/models` staging registry:

```bash
python3 ${APP_DIR}/tools/fetch_models.py ssd
```

The SSD fetch requires credentials for the staging artifact registry. The helper chooses the Modalix INT8 MLA package and copies it to the stable path used by the profile.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Change only `model.profile`; set `model.path` only when using a package outside `models/`. `runtime.frames` controls how many results are pulled.

The camera path captures 1920x1080 NV12 at 30 FPS. Neat owns MPK parsing, model compatibility, decoding, and the EV74 preprocessing/postprocessing stages.

The graph ends with a manually added, named output so the composition is explicit:

```python
camera_graph.add(pyneat.nodes.camera_input(camera_options, capture_buffer_count=32))
model_graph = model.graph(route)
model_graph.add(pyneat.nodes.output("results"))
```

## Run

List the available profiles:

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py --list-profiles
```

Run the configured profile, or select one on the command line:

```bash
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml \
  --describe

python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml \
  --profile pose
```

Stream YOLO26 detection or segmentation to Insight:

```bash
python3 ${APP_DIR}/src/python/main.py \
  --profile segment \
  --model /path/to/yolo26_seg_mpk.tar.gz \
  --continuous \
  --insight-host 192.168.1.127 \
  --insight-video-port 9000 \
  --insight-metadata-port 9100 \
  --insight-channel 0
```

The Insight graph uses `RealtimeLatestByStream` independently for the model and video branches, preventing either consumer from building a stale camera-frame backlog. Segmentation metadata includes per-instance polygons, class labels, confidence, and bounding boxes.

The application rejects the graph before execution unless the negotiated backend requires zero-copy and contains no CPU camera bridge.

On a bring-up image whose libcamera stack cannot provide the required downstream DMA-BUF pool,
use `--allow-camera-copy` to opt into Neat's camera bridge explicitly. Strict zero-copy remains
the default.

## Expected Result

Each frame prints the selected profile and a task-specific summary, followed by a `PASS` line. For example:

```text
frame=0 profile=detect detections=6
PASS strict-zero-copy MIPI -> YOLO26n object detection -> results
```

## Troubleshooting

- If camera discovery fails, confirm that `cam -l` reports a camera and that the correct overlay is installed.
- If model loading fails, confirm the file is a directly consumable `*_mpk.tar.gz` compatible with the installed Neat runtime. Neat owns MPK parsing and compatibility validation.
- If the strict zero-copy check fails, confirm that Neat, libcamera, and the camera pipeline were built with compatible downstream-owned buffer support.
- Run with `--validate-model-only` to inspect the package without loading Neat or opening the camera.

## Source Files

- Python source: `src/python/main.py`
- Model profile and package handling: `src/python/model_profiles.py`
- Shared runtime configuration: `src/common/config.yaml`
- Model fetch helper: `tools/fetch_models.py`

## Development From Source

To modify or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
