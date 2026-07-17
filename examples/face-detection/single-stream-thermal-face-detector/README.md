# Single-Stream Thermal Face Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | face-detection |
| Difficulty | Intermediate |
| Tags | face-detection, keypoints, yolov5s-face, thermal, rtsp, insight |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | single-stream-thermal-face-detector |
| Model | yolov5s_face_raw_split |

## Concept
Single-camera RTSP face detection pipeline using yolov5s-face, rendered live in a
[neat-insight](https://developer.sima.ai/software/tools/insight/) viewer. The app
decodes an RTSP stream, runs inference on the MLA, and publishes results to
Insight over UDP:

```
RTSP decode (NV12) --> branch --> video_sender (H264 RTP/UDP -> Insight)
                            \--> model (raw split heads) --> detections
```

For each frame the app publishes a **pose-estimation** overlay carrying the 5
named facial landmarks (`eye_l`, `eye_r`, `nose_tip`, `mouth_l`, `mouth_r`),
which the viewer draws as labeled dots.

Two Insight viewer behaviors shape this choice:
- The viewer renders **one metadata type per channel at a time** — it clears the
  canvas and draws only the most recent message's type. Sending both
  `object-detection` and `pose-estimation` makes them overwrite each other, so
  the pipeline sends a single type.
- The pose overlay joins keypoints whose names match a **COCO body skeleton**
  pair (`nose`–`left_eye`, `nose`–`right_eye`, …). The landmark names above
  deliberately avoid those joint names, so no skeleton lines are drawn across the
  face.

To show boxes instead, send `object-detection` with `{"objects":[{"id","label",
"confidence","bbox":[x,y,w,h]}]}` from `send_metadata` in place of the pose
payload.

Unlike the BBOX-emitting detectors in this repo (yolo26m, yolov8n), yolov5s-face
has a split-head topology with paired box (18-channel) and landmark (30-channel)
outputs at three pyramid levels. The model archive is compiled to emit those six
raw FP32 heads directly:

- `Model` is loaded with an intent-based preprocess (`preprocess.kind = Image`,
  `color_convert.input_format = NV12`, `preset = COCO_YOLO`). The Neat route
  planner attaches the model's on-device CVU (EV74) preprocess, which converts
  NV12->RGB, letterboxes the frame to the 800x800 canvas, normalizes `/255`,
  quantizes to INT8, and tessellates for the MLA.
- `decode_type` is left `Unspecified`, so no fused `SimaBoxDecode` runs. The NEAT
  BBOX wire format carries no landmark slots, so the box + 5-landmark decode runs
  in user space on the host (APU). The C++ and Python decoders implement the same
  math and drive the same on-device graph.

## Preview
Open the Insight video viewer for the app's channel to see live thermal or
visible-light faces with five labeled landmark dots per detected face (eyes,
nose tip, mouth corners).

## Insight Setup
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host an RTSP source, receive video from `VideoSender`, receive landmark metadata from `MetadataSender`, and show rendered overlays plus runtime metrics in the browser.

In the Neat Development Environment, install the sample video assets:

```bash
sima-cli install assets/multi-video-sources
```

This provides 720p and 480p videos that Insight can stream as RTSP sources. This
example expects faces in frame, so upload a thermal or visible-light face clip of
your own when the sample videos do not contain any.

To create a reproducible RTSP input:
1. Run `neat` in the Neat Development Environment and open the reported `Insight Web UI`.
2. In Insight, open `RTSP Source`.
3. Use a sample video or upload your own video.
4. Start the stream and copy the RTSP URL.
5. Put that RTSP URL into `source.rtsp_url`.

Use the same `neat` output to set `output.insight.host`, `video_port`, and `metadata_port` from the reported `videoUDP` and `metadataUDP` ranges.

**Ports when the app runs on a board.** Insight runs inside the SDK container and
builds its URLs from the container's own view, so the RTSP link copied from the UI
carries the container-internal port (`8554`). An app running on a DevKit is
outside that container and must use the SDK host IP plus the **published** host
port instead. Read the mapping from `exposedPorts` in `neat --json` and never
assume the defaults:

| Endpoint | Container port (shown in the UI) | Host port (use this from a board) |
| --- | --- | --- |
| RTSP | 8554 | `rtsp.tcp` → `hostPortStart` |
| Video UDP | 9000 | `videoUDP` → `hostPortStart` |
| Metadata UDP | 9100 | `metadataUDP` → `hostPortStart` |

Keep video and metadata on the same channel (0).

## Supported Models
Use the SDK platform version wherever `<platform-version>` appears.

Primary model: `yolov5s_face_raw_split`

Download the model:

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo -v <platform-version> get yolov5s_face_raw_split
cd ../..
```

The command stores the model under `assets/models/` as a repo-local convention. `model.path` can point to any readable model package path.

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
  Download the model with the command above.
- A running neat-insight instance reachable from the board, plus an RTSP source.
  In the SDK, upload a video through Insight and start a media source to get an
  RTSP URL; see [Use Insight](https://developer.sima.ai/software/tools/insight/).
- The model expects an 800x800 compiled canvas; the on-device preproc accepts
  NV12/BGR/RGB/I420/GRAY input up to 4096x4096.
- Labels file: `examples/face-detection/single-stream-thermal-face-detector/src/common/face_label.txt`

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
Edit `examples/face-detection/single-stream-thermal-face-detector/src/common/config.yaml`.

```yaml
model:
  path: assets/models/yolov5s_face_raw_split_mpk.tar.gz
  labels: examples/face-detection/single-stream-thermal-face-detector/src/common/face_label.txt

source:
  rtsp_url: rtsp://<insight-host-ip>:<rtsp-port>/src0   # RTSP source to detect on.
  tcp: true
  latency_ms: 100

inference:
  frames: 0                  # 0 = run continuously.
  min_score: 0.25
  nms_iou: 0.45
  max_detections: 50

runtime:
  profile: false
  profile_interval: 100

output:
  insight:
    host: <insight-host-ip>  # Host running the Insight receiver/viewer.
    video_port: 9000         # UDP video base port (add channel).
    metadata_port: 9100      # UDP metadata base port (add channel).
```

When Insight was started with non-default host ports (check `neat --json`), set
`source.rtsp_url` to the mapped `rtsp.tcp` port, `output.insight.host` to the SDK
host IP, and `video_port` / `metadata_port` to the mapped `videoUDP` / `metadataUDP`
base ports. Keep video and metadata on the same channel (0).

## Run
### C++
```bash
./build/examples/face-detection/single-stream-thermal-face-detector_cpp/single-stream-thermal-face-detector \
  --config examples/face-detection/single-stream-thermal-face-detector/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/face-detection/single-stream-thermal-face-detector/src/python/requirements.txt
python3 examples/face-detection/single-stream-thermal-face-detector/src/python/main.py \
  --config examples/face-detection/single-stream-thermal-face-detector/src/common/config.yaml
```

Open the Insight video viewer for the channel (e.g. `/api/viewer-url?src=0`) to
watch the annotated stream.

## Debugging Notes
- If the viewer shows video but no overlays, confirm `output.insight.host` and the
  metadata port are reachable and match the viewer's channel. Use Insight's
  `/api/ingest/stats` to check `metadata.messages_received`.
- If the streamed video looks blocky, raise `encoder.bitrate_kbps` in
  `build_pipeline` (the Neat default is 4000). Check the delivered rate with
  `/api/ingest/stats` -> `rtp.bitrate_bps`; the source stream itself is only
  re-encoded here, so a sharp source plus a blocky viewer means the encoder is
  starved.
- If the viewer shows no video, verify the video UDP port and that RTP is arriving
  (`/api/ingest/stats` -> `rtp.packets_received`, `media.seen_sps`).
- If keypoints look mis-anchored, the model may have been retrained with different
  anchors -- update `_STRIDES`/`_anchors()` in `src/python/main.py` and
  `kStrides`/`kAnchors` in `src/cpp/main.cpp` to match.
- Set `runtime.profile: true` to print rolling pull/decode/metadata timing and
  output FPS.
- Use `inference.min_score` and `inference.nms_iou` to tune detection sensitivity.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/face_label.txt`
