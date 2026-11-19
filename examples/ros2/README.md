# ROS 2 examples

Every Python example is an independent ROS package. Its nodes, launch file,
configuration, labels and implementation helpers live inside its own directory.
Original native examples remain unchanged under `examples/<category>/<example>`.
No example requires building or importing the legacy `neat_apps` package.
The Docker example composes independently built apps; see its own instructions.

## Build on Modalix

Use this directory from an Apps repository checkout on Modalix. ROS examples
build with colcon independently of `build.sh`; the native Apps runtime bundle
does not include their complete build/launch files.

Install ROS 2 with `sima-cli neat install ros2@v3.0.0` and ensure colcon and the
matching Neat Python environment are available. Enter the example you want:

```bash
cd face-detection/face-detector
source /usr/local/ros2/setup.bash
source ~/pyneat/bin/activate
colcon build
source install/setup.bash
```

Edit `src/common/config.yaml` with your model/input paths, then run:

```bash
ros2 launch neat_face_detector app.launch.py \
  config:="$PWD/src/common/config.yaml"
```

These commands run from the example directory. Rebuild after Python/launch
changes; restart after YAML edits. Source its installation in each terminal.
Use absolute model/input paths when copying an example to a different directory.
Build output stays in that example's ignored `build/`, `install/` and `log/`.

## Choose an app

| Example | Package | Main node | Default source |
| --- | --- | --- | --- |
| [anomaly-detection/fastflow-anomaly-detector](anomaly-detection/fastflow-anomaly-detector/README.md) | `neat_fastflow_anomaly_detector` | `anomaly_node.py` | `live` |
| [anomaly-detection/patchcore-anomaly-detector](anomaly-detection/patchcore-anomaly-detector/README.md) | `neat_patchcore_anomaly_detector` | `anomaly_node.py` | `live` |
| [benchmarking/mipi-camera-capture](benchmarking/mipi-camera-capture/README.md) | `neat_mipi_camera_capture` | `capture_node.py` | `live` |
| [benchmarking/mipi-multi-model](benchmarking/mipi-multi-model/README.md) | `neat_mipi_multi_model` | `inference_node.py` | `live` |
| [benchmarking/model-benchmark](benchmarking/model-benchmark/README.md) | `neat_model_benchmark` | `benchmark_node.py` | `none` |
| [classification/image-classification-explorer](classification/image-classification-explorer/README.md) | `neat_image_classification_explorer` | `classification_node.py` | `folder` |
| [depth-estimation/depth-estimator](depth-estimation/depth-estimator/README.md) | `neat_depth_estimator` | `depth_node.py` | `folder` |
| [face-detection/face-detector](face-detection/face-detector/README.md) | `neat_face_detector` | `detection_node.py` | `folder` |
| [face-detection/single-stream-thermal-face-detector](face-detection/single-stream-thermal-face-detector/README.md) | `neat_single_stream_thermal_face_detector` | `detection_node.py` | `live` |
| [face-recognition/face-recognizer](face-recognition/face-recognizer/README.md) | `neat_face_recognizer` | `recognition_node.py` | `live` |
| [feature-extraction/superpoint-feature-extractor](feature-extraction/superpoint-feature-extractor/README.md) | `neat_superpoint_feature_extractor` | `feature_node.py` | `live` |
| [genai/neat-genai-studio](genai/neat-genai-studio/README.md) | `neat_genai_studio` | `studio_node.py` | `none` |
| [multi-model/multi-stream-multi-model](multi-model/multi-stream-multi-model/README.md) | `neat_multi_stream_multi_model` | `inference_node.py` | `live` |
| [object-detection/detr-object-detector](object-detection/detr-object-detector/README.md) | `neat_detr_object_detector` | `detection_node.py` | `folder` |
| [object-detection/high-density-multi-stream-object-detector](object-detection/high-density-multi-stream-object-detector/README.md) | `neat_high_density_multi_stream_object_detector` | `detection_node.py` | `live_graph` |
| [object-detection/multi-stream-obb-detector](object-detection/multi-stream-obb-detector/README.md) | `neat_multi_stream_obb_detector` | `detection_node.py` | `live` |
| [object-detection/multi-stream-object-detector](object-detection/multi-stream-object-detector/README.md) | `neat_multi_stream_object_detector` | `detection_node.py` | `live` |
| [object-detection/pcb-defect-detector](object-detection/pcb-defect-detector/README.md) | `neat_pcb_defect_detector` | `detection_node.py` | `folder` |
| [object-detection/rfdetr-detection-segmentation](object-detection/rfdetr-detection-segmentation/README.md) | `neat_rfdetr_detection_segmentation` | `detection_node.py` | `live` |
| [object-detection/single-stream-object-detector](object-detection/single-stream-object-detector/README.md) | `neat_single_stream_object_detector` | `detection_node.py` | `live` |
| [object-detection/ssd-mobilenet-object-detector](object-detection/ssd-mobilenet-object-detector/README.md) | `neat_ssd_mobilenet_object_detector` | `detection_node.py` | `folder` |
| [object-detection/yolo26-object-detector](object-detection/yolo26-object-detector/README.md) | `neat_yolo26_object_detector` | `detection_node.py` | `folder` |
| [pose-estimation/multi-stream-blazepose3d](pose-estimation/multi-stream-blazepose3d/README.md) | `neat_multi_stream_blazepose3d` | `pose_node.py` | `live` |
| [pose-estimation/multi-stream-pose-estimator](pose-estimation/multi-stream-pose-estimator/README.md) | `neat_multi_stream_pose_estimator` | `pose_node.py` | `live` |
| [segmentation/efficientsam3-promptable-segmenter](segmentation/efficientsam3-promptable-segmenter/README.md) | `neat_efficientsam3_promptable_segmenter` | `segmentation_node.py` | `live` |
| [segmentation/single-stream-instance-segmenter](segmentation/single-stream-instance-segmenter/README.md) | `neat_single_stream_instance_segmenter` | `segmentation_node.py` | `live` |
| [tracking/multi-stream-tracker](tracking/multi-stream-tracker/README.md) | `neat_multi_stream_tracker` | `tracking_node.py` | `live` |
| [tracking/yolo26-tiny-drone-tracker](tracking/yolo26-tiny-drone-tracker/README.md) | `neat_yolo26_tiny_drone_tracker` | `tracking_node.py` | `live` |
| [Detection-to-VLM](genai/detection-to-vlm-assistant/README.md) | `neat_detection_vlm` | `detection_node.py`, `vlm_node.py` | `live` |
| [Docker parallel apps](docker/parallel-apps/README.md) | `neat_parallel_apps` | Container composition script | Configured apps |

## Start a source or wait for a topic

The app's local `launch/app.launch.py` declares its supported modes:

| Mode | Behavior |
| --- | --- |
| `source:=folder` | Read configured images and wait for a result for every admitted image |
| `source:=live` | Own the configured RTSP/HTTP/USB/MIPI/video source and publish images |
| `source:=none` | Start inference only and wait for an external image topic |
| `source:=live_graph` | High-density only: keep decoding and inference together, publish detections |

For example, add `source:=none` to the face detector's launch command and set
`ros.image_topic: /camera/image_raw`. It requires BGR8 images and reliable QoS.
Other apps require their documented formats, often packed NV12. Studio and the
synthetic benchmark use request/report workflows instead of a camera source.

## View results and Insight

Image apps publish JSON on `results` and dense images on additional documented
topics where applicable. Set `ros.visualization.insight.enabled: true` and
`host` to display matching overlays in your running Insight video viewer.
`channel_base`, `channels`, `max_visible_streams` and `fps` control output.
Set `ros.visualization.save.enabled: true` to save matched PNGs and headers.
Both options default off; disable them for throughput measurements.

High-density graph mode keeps encoded-video passthrough and uses
`output.insight` for host/ports plus `output.video_enabled: true`.
Detection-to-VLM adds a second channel showing its actual accepted input.
Studio is available through Insight's GenAI tab; the model benchmark publishes
a report rather than video. The app launches do not start Insight itself.

## Create an app

Start with the nearest example and follow the
[node development guide](NODE_DEVELOPMENT.md). Model initialization,
input preparation, inference submission/completion, decoding and ROS interfaces
are visible in the app's node file. Helpers are ordinary local Python modules.

## Validation status

The 29 packages migrated after the VLM example passed independent `colcon build`,
installed-node imports and launch-description checks on Modalix, without a
`neat_apps` installation. The VLM package retains its previously validated layout.

- **Numerical regression:** 17 standard image apps produced identical results
  before and after the refactor, including dense maps in the focused comparisons.
  PatchCore, the supported OBB route and all four multi-model stream configurations
  also matched. RF-DETR small detection and medium segmentation matched the
  current native Python implementation with identical model artifacts.
- **Timing regression:** three alternating before/after pairs for each of the
  17 standard apps showed no repeatable slowdown. Folder cases used 100–420
  outputs per run; PCB was repeated with 2,000 outputs. Live cases used 360 outputs
  at 30 FPS or 120 outputs at 5 FPS. Source-limited runs verify delivery at that
  input rate, not maximum accelerator capacity. These are comparisons with the
  previous ROS implementation, separate from native-Python parity.
- **Installed workflows:** face detection processed 21 mixed-size images, saved
  all overlays and delivered Insight video datagrams; the four-model launch
  delivered 30 outputs per stream; BlazePose delivered 120 recorded outputs;
  high-density delivered 3,000 recorded outputs covering 20 streams. The model
  benchmark produced its report and accepted another run through its ROS service.
- **Focused checks:** 58 ROS checks passed on Modalix, 83 targeted host checks
  passed, and all 61 repository READMEs passed validation.

**DETR retest after the board update:** Neat 0.6.2 exposes the required
`BilinearAntialiased` resize. The registry ResNet-50 w16a16 package passed native
and ROS execution with exact numerical matches on 21 images. Three alternating
pairs of 903 outputs measured native Python at 35.29–35.78 FPS and asynchronous
ROS at 70.99–77.84 FPS, with no ROS drops or header mismatches. Both workflows
saved all 21 overlays. See the [DETR README](object-detection/detr-object-detector/README.md)
for measurement scope. Other apps' earlier measurements were not repeated on
this updated runtime.

A successful build or a refactor regression check does not establish universal
native parity. The native-Python acceptance target remains no measurable slowdown.

Earlier testing established numerical parity for the tested folder/live cases,
with these remaining limitations: high-density live-graph throughput was below
native in paired 20-stream runs; BlazePose four-camera throughput was below native,
and independently sampled live frames can yield different smoothed histories.
The Core used in those earlier tests blocked the default native OBB decoder
path; its debug-frame branch has separate validation. Studio's tested speech output matched, with lower
request throughput accepted for equivalent output quality. See [ROS2.md](../../ROS2.md).

## Pending validation

- [ ] **ROS CI/CD:** add a dedicated job using `sima-cli neat install ros2@v3.0.0`
  and a compatible Neat Python environment. Build each standalone package and
  test its installed launch workflow. SDK CI currently packages ROS source only;
  native runtime shards do not admit ROS tests.
- [ ] **Physical MIPI testing:** exercise the local camera input node with a
  connected MIPI camera; verify geometry, format, frame identity and throughput.
- [ ] **Physical USB testing:** verify capture and sustained delivery on hardware.
- [ ] **Docker:** validate the container composition on a board with Docker.
- [ ] **PCIe:** add/test ROS integration when a host/device setup is available.
