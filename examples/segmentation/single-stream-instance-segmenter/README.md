# Single Stream Instance Segmenter

## Metadata
| Field | Value |
| --- | --- |
| Category | segmentation |
| Difficulty | Intermediate |
| Tags | segmentation, yolo26, instance-segmentation, rtsp, insight |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | single-stream-instance-segmenter |
| Model | yolo26m-seg-bf16-b1 |

## Concept
`single-stream-instance-segmenter` is a single-camera YOLO26 instance segmentation example:

- ingest one RTSP camera stream
- decode the stream into frames
- run YOLO26 instance segmentation
- render mask overlays on the video
- send H.264 video plus segmentation metadata to Insight

The example keeps RTSP ingest, model inference, and Insight output separate so
the segmentation behavior can be debugged independently from transport issues.

## Preview
Snippet from a pipeline run:

![Single stream instance segmenter preview](../../../assets/portal/segmentation/single-stream-instance-segmenter/image.png)

## Insight Setup
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host an RTSP source, receive video from `VideoSender`, receive segmentation metadata from `MetadataSender`, and show rendered overlays plus runtime metrics in the browser.

To create a reproducible RTSP input:
1. Run `neat` in the Neat Developer Environment and open the reported `Insight Web UI`.
2. In Insight, open `RTSP Source`.
3. Use a sample video or upload your own video.
4. Start the stream and copy the RTSP URL.
5. Put that RTSP URL into `source.rtsp_url`.

Use the same `neat` output to set `output.insight.host`, `video_port`, and `metadata_port` from the reported `videoUDP` and `metadataUDP` ranges.

## Supported Models
Use the SDK platform version wherever `<platform-version>` appears.

Default model: `yolo26m-seg-bf16-b1.tar.gz`.

Download the default model:

```bash
mkdir -p assets/models/YOLO26-SEGMENTATION
cd assets/models/YOLO26-SEGMENTATION

sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-segmentation/yolo26m-seg-bf16-b1.tar.gz

cd ../../..
```

## Prerequisites
- Installed Neat Library and Insight on the DevKit.
- RTSP source created in Insight or provided by your camera.
- Default YOLO26 segmentation model package downloaded locally.
- `model.path`, `model.labels`, `source.rtsp_url`, and `output.insight.host` set in `src/common/config.yaml`.

## Get The Apps Repo
Install the Neat Library first by following the official [Neat Library installation guide](https://developer.sima.ai/software/getting-started/installation/neat-library).

Then clone and build the apps repo:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After this setup, follow the example-specific commands below.

## Configure
Edit `examples/segmentation/single-stream-instance-segmenter/src/common/config.yaml`.

```yaml
model:
  path: assets/models/YOLO26-SEGMENTATION/yolo26m-seg-bf16-b1.tar.gz # Model package to load.

source:
  rtsp_url: <rtsp-url-copied-from-insight>                          # RTSP stream URL.
  tcp: true                                                          # Use TCP transport for RTSP.

inference:
  frames: 0                                                          # Frame limit. 0 runs continuously.
  min_score: 0.55                                                    # Minimum instance confidence.

output:
  insight:
    host: <insight-host-ip>                                          # Host running Insight.
    video_port: <videoUDP start port from neat>                      # UDP video port.
    metadata_port: <metadataUDP start port from neat>                # UDP metadata port.
```

## Run
### C++
```bash
./build/examples/segmentation/single-stream-instance-segmenter/single-stream-instance-segmenter \
  --config examples/segmentation/single-stream-instance-segmenter/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/single-stream-instance-segmenter/src/python/requirements.txt
python3 examples/segmentation/single-stream-instance-segmenter/src/python/main.py \
  --config examples/segmentation/single-stream-instance-segmenter/src/common/config.yaml
```

## Debugging Notes
- If startup fails, verify `model.path` and `source.rtsp_url`.
- If the app times out waiting for RTSP, verify source reachability first.
- If Insight receives no video, verify `output.insight.host` and UDP ports.
- If saved frames are needed for inspection, set `output.save_dir` and `output.save_every`.

## Appendix: Additional Models
Other supported YOLO26 segmentation models:
- `yolo26n-seg-bf16-mla_tess.tar.gz`
- `yolo26s-seg-bf16-mla_tess.tar.gz`
- `yolo26m-seg-bf16-mla_tess.tar.gz`
- `yolo26l-seg-bf16-mla_tess.tar.gz`
- `yolo26x-seg-bf16-mla_tess.tar.gz`
- `yolo26m-seg-bf16-mla_tess-b1.tar.gz`
- `yolo26m-seg-int8-b1.tar.gz`

Replace the default filename in the download command and `model.path`.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared config: `src/common/config.yaml`
- Shared labels: `src/common/coco_label.txt`
