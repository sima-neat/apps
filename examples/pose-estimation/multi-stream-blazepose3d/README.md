# Multi-Stream BlazePose 3D

## Metadata

| Field | Value |
| --- | --- |
| Category | pose-estimation |
| Difficulty | Advanced |
| Tags | blazepose, yolo26, keypoints, rtsp, multistream, insight |
| Languages | C++, Python |
| Status | stable |
| Binary Name | multi-stream-blazepose3d |
| Model | YOLO26 detection and BlazePose Heavy 3D |

## Concept

Detects people across RTSP streams with shared YOLO26 and BlazePose models, then publishes frame-correlated 2D landmarks and 3D world landmarks to separate Insight channels.

C++ and Python use the same configuration, graph topology, scheduling policy, ROI transform, and metadata schemas.

Each RTSP stream owns an independent source graph and run, so one disconnected source cannot close or stall the others. Each run has one dedicated pull owner. Frames stay NV12 through decode and freshness admission, then each admitted frame is converted once to packed RGB. Application-owned asynchronous queues feed one shared YOLO26 runner and one shared BlazePose runner through public push/pull APIs.

```text
Per stream: RTSP encoded ─┬─> codec passthrough ─> Insight video
                         └─> decode NV12 ─> RealtimeLatestByStream ─> RGB output

Application: latest RGB mailbox per stream ─> shared YOLO26 Model graph
             (Preproc ─> inference ─> BoxDecode) ─> person ROI mailbox per stream
             ─> BlazePose Preproc ROIs ─> owned EV74 ROI inputs
             ─> shared BlazePose runner
             ─> 33 image + world keypoints ─> paired correlated Insight metadata
```

The source runs and both model runners are fixed after startup. Configure one to four cameras, then restart the application after adding, removing, or editing a stream. Separate source runs accept different resolutions without rebuilding either shared model graph.

## Preview

![Multi-stream BlazePose 3D preview](../../../portal/assets/examples/pose-estimation/multi-stream-blazepose3d/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- H.264 or H.265 RTSP sources and an [Insight](https://developer.sima.ai/software/tools/insight/) host reachable from the target.
- A supported YOLO26 detection package and the existing BlazePose Heavy 3D package.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
export APP_DIR=examples/pose-estimation/multi-stream-blazepose3d
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The detector uses the `yolo26m-det-int8-b1.tar.gz` Model Zoo package, which the E2E tests run:

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26m-det-int8-b1.tar.gz"
cd ..
```

Download the published BlazePose GHUM Heavy package from the Model Registry:

```bash
sima-cli models download --stg \
  --id blazepose_ghum_heavy \
  --variant modalix_bf16 \
  --branch feat/blazepose-ghum-heavy \
  --output models

python3 - <<'PY'
from pathlib import Path
import shutil
import tempfile
import zipfile

root = Path("models/blazepose_ghum_heavy/modalix_bf16")
target = root / "blazepose_ghum_heavy_modalix_bf16_mpk.tar.gz"
if not target.is_file():
    packages = list(root.glob("*.zip"))
    if len(packages) != 1:
        raise SystemExit(f"expected one model-package zip under {root}, found {len(packages)}")
    with zipfile.ZipFile(packages[0]) as package:
        members = [item for item in package.infolist() if Path(item.filename).name == target.name]
        if len(members) != 1:
            raise SystemExit(f"expected one {target.name} in {packages[0]}, found {len(members)}")
        temporary = None
        try:
            with package.open(members[0]) as source, tempfile.NamedTemporaryFile(
                dir=root, prefix=f"{target.name}.", delete=False
            ) as destination:
                temporary = Path(destination.name)
                shutil.copyfileobj(source, destination)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
print(target)
PY
```

The extraction step is a no-op when `sima-cli` downloads the model pack
directly, and unwraps the model-package zip returned by newer CLI releases.

The feature-branch reference is temporary while
[`sima-neat/models#136`](https://github.com/sima-neat/models/pull/136) is under
review. Use `--branch develop` after that dependency merges.

## Model Contracts

The public YOLO26 route accepts dynamic HWC RGB images. Model preprocessing resizes and normalizes them for the internal 640×640 tensor; its six raw detection heads are decoded by `YoloV26` BoxDecode into one BBOX output consumed by the application.

The public BlazePose route also accepts dynamic HWC RGB images and preprocesses each selected ROI to 256×256. The existing artifact returns screen landmarks `[1,195]` (39 records of x, y, z, visibility logit, and presence logit), global pose presence `[1,1]`, and world landmarks `[1,117]` (39 x/y/z records). This application publishes the first 33 screen and world records in `pose-estimation`, as required by the application contract, and repeats the world records in a generic `auxiliary-visualization` view for Insight's 3D renderer.

## Prepare Insight

Insight can host the input streams and render each output channel. Install videos from the Insight catalog or through Insight's YouTube support, start the streams in the Insight Web UI, and copy their RTSP URLs into `streams`.

The application sends the original encoded stream plus two correlated metadata messages on the configured channel. Insight draws `pose-estimation` over the video and renders `auxiliary-visualization` world landmarks in the separate 3D Pose panel. Both messages carry the same source PTS timestamp and frame ID. This requires an Insight build with multi-type frame metadata and the `blazepose-3d` auxiliary renderer.

## Configure

Edit `${APP_DIR}/src/common/config.yaml`:

```yaml
models:
  detector_path: models/yolo26m-det-int8-b1.tar.gz
  pose_path: models/blazepose_ghum_heavy/modalix_bf16/blazepose_ghum_heavy_modalix_bf16_mpk.tar.gz

streams:
  - id: entrance
    url: rtsp://camera.example/entrance
    codec: h264
    insight_channel: 0
    width: 1920
    height: 1080
    fps: 30
  - id: warehouse
    url: rtsp://camera.example/warehouse
    codec: h265
    insight_channel: 1
    width: 1280
    height: 720
    fps: 30

output:
  insight:
    host: <insight-host-ip>
```

Each stream needs a unique stable `id` and `insight_channel`. `pose.max_people_per_frame` bounds how many highest-confidence person boxes are sent to BlazePose per admitted frame; its maximum of 10 keeps each JSON metadata datagram below the sender limit.

The active video and metadata UDP ports must be disjoint. If you use sparse or non-zero channel numbers, choose `video_port_base` and `metadata_port_base` so no `base + insight_channel` value overlaps.

Every stream requires `width`, `height`, and `fps`; make them match the RTSP
source's actual caps. Requiring them avoids probing every camera before the
independent source workers start, so an offline channel cannot prevent healthy
channels from running. The FPS is a decoder hint and is not pinned into caps,
so a 29.97 fps (30000/1001) camera works with `fps: 30`.

`pose.temporal_filter_enabled` defaults to `true`. Each stream matches its poses to the previous frame's by person-box overlap and moves the matched image and world landmarks halfway toward the new estimate, so the 2D overlay and 3D view are smoothed together. Disable it when raw model output is required.

## Run

### C++

```bash
"${APP_DIR}/src/cpp/pre-built/multi-stream-blazepose3d" \
  --config "${APP_DIR}/src/common/config.yaml"
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r "${APP_DIR}/src/python/requirements.txt"
python3 "${APP_DIR}/src/python/main.py" \
  --config "${APP_DIR}/src/common/config.yaml"
```

## Output Metadata

Each frame in which YOLO26 selects people produces a correlated pair of messages. Each `pose-estimation` pose carries its YOLO person box, global presence, 33 named image-space keypoints, and 33 named world keypoints. Insight uses the image-space fields for the 2D overlay:

```json
{"stream_id":"entrance","poses":[{"id":"pose_1","label":"person","presence":0.99,"confidence":0.91,
  "bbox":[120,80,240,520],
  "keypoints":[{"name":"nose","x":242,"y":135,"confidence":0.98}],
  "world_keypoints":[{"name":"nose","x":0.01,"y":-0.42,"z":-0.08,"confidence":0.98}]}]}
```

The separate `auxiliary-visualization` message uses the generic Insight schema
and selects the built-in BlazePose renderer:

```json
{"schema_version":1,"id":"world-pose","renderer":"blazepose-3d","title":"3D Pose","stream_id":"entrance",
  "payload":{"poses":[{"id":"pose_1","presence":0.99,"keypoints":[
    {"name":"nose","x":0.01,"y":-0.42,"z":-0.08,"confidence":0.98}
  ]}]}}
```

`MetadataSender` supplies the outer `type`, `timestamp`, and `frame_id` fields.
The two message types are sent while holding the same per-stream metadata lock,
so one channel cannot interleave identities from different frames. A frame
without people publishes an empty pair, which clears the stream's poses in Insight.

The envelope builder accepts any renderer name and JSON object; it does not know
about BlazePose fields. This example's world-pose helper supplies the
`blazepose-3d` payload, while point clouds, meshes, trajectories, or other 3D
data can reuse the same envelope with a separately registered Insight renderer.

The keypoint confidence is the minimum of BlazePose landmark visibility and presence after sigmoid activation. The global pose-presence logit is also sigmoid-activated before it gates each ROI and is published as a probability.

Each frame's stream ID, frame ID and PTS stay with its queued work, so a result is always published on its own stream's channel with its source timestamp. The C++ and Python hardware E2E tests each run up to four H.264 and four H.265 streams. On every configured video port they require both codec configuration and a complete VCL-bearing access unit. Every metadata message must carry the configured stream identity and the advertised bbox, presence, confidence, landmark names, and finite coordinates. Each metadata port must also publish a 2D/3D pair with one `(port, timestamp, frame_id)` identity whose pose IDs, presence values, and world landmarks match exactly, and at least one pair must be non-empty.

## Performance and Scheduling

- Each source run starts and pulls on its own thread, so an offline source's startup timeout cannot delay healthy streams. `RealtimeLatestByStream` bounds admitted decoder-backed frames before the packed-RGB conversion.
- Latest-only detector and pose mailboxes plus round-robin dispatch keep a slow or disconnected stream from blocking the others; newer work replaces queued work instead of accumulating.
- YOLO26 and BlazePose each use one shared model route. Increasing stream count does not create additional model routes.
- Video uses H.264 or H.265 encoded passthrough with latest-only egress, so a slow receiver cannot backpressure analytics. The application does not draw on frames or re-encode them.
- The current public `VideoConvert` node performs the one NV12-to-RGB conversion on A65 after admission. The RGB frame remains holder-backed in application code; Python passes the `Tensor` directly and C++ maps a non-owning `cv::Mat` view.
- YOLO26 preprocessing stays inside the shared `Model::graph()` route; the application pushes each correlated RGB frame directly into that runner.
- The public `stages::Preproc(..., rois)` API receives the fixed-size source RGB frame and all selected BlazePose ROIs in one batched call. Keeping the input dimensions stable lets Neat reuse one preprocessing runner instead of caching a new graph for every changing person-box crop; returned affine metadata still maps landmarks directly into source-frame coordinates. Full RGB frames are not cloned.

At shutdown each stream prints `frames_in` (source frames admitted, which `runtime.frames` limits) and `frames_out` (metadata pairs whose two messages were both queued for Insight). A failed send is logged and does not count.

## Troubleshooting

- Replace all placeholders before running and verify both model paths.
- Reduce `pose.max_people_per_frame` when pose throughput, rather than detection, is the bottleneck.
- H.265 input and video passthrough require an Insight/browser environment that can decode HEVC.
- `[ERR] YOLO26 inference stalled` or `[ERR] BlazePose inference stalled` means a shared model accepted input but returned no output for 5 s; the application stops and exits 1.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared configuration: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
