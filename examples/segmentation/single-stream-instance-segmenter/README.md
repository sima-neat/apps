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

This example ingests one RTSP H.264/H.265/MJPEG or HTTP MJPEG stream, runs YOLO26 instance segmentation, and sends H.264 video plus segmentation metadata to Insight, which draws the masks over the video.

The decoded frame branches inside a single graph to the segmenter and to the H.264 sender. Both outputs therefore carry timestamps from the same frame, which is what lets Insight draw each mask on the frame it came from. Setting `output.save_dir` adds a third branch that returns the decoded frame to the application so it can write annotated JPEGs.

Metadata is sent as `type: "segmentation"` with one `data.segments[]` entry per instance. The model emits masks on a 160x160 grid covering its letterboxed 640x640 input, not the detection box, so each mask is mapped back through that scale and padding, upscaled to the detection rectangle, and only then thresholded. The silhouette is sent as `mask_format: "polygon"` in frame pixels: a polygon keeps the crisp contour the mask has at frame resolution, while run-length encoding would ship a mask crop only a few pixels across and leave Insight to stretch it. Each frame stays inside a 32 KB payload budget; over it, the lowest-confidence segments are dropped and counted in the run summary.

## Preview

![Single stream instance segmenter preview](../../../portal/assets/examples/segmentation/single-stream-instance-segmenter/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- An RTSP H.264, RTSP H.265, RTSP MJPEG, or HTTP MJPEG source and an [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the target.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model file | Role |
| --- | --- |
| `yolo26m-seg-bf16-b1.tar.gz` | Default |
| `yolo26n-seg-bf16-mla_tess.tar.gz` | Supported |
| `yolo26s-seg-bf16-mla_tess.tar.gz` | Supported |
| `yolo26m-seg-bf16-mla_tess.tar.gz` | Supported |
| `yolo26l-seg-bf16-mla_tess.tar.gz` | Supported |
| `yolo26x-seg-bf16-mla_tess.tar.gz` | Supported |
| `yolo26m-seg-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-seg-int8-b1.tar.gz` | Supported |

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Replace `<model-file>` with a file from the table.

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-segmentation/<model-file>"
cd ..
```

Set `model.path` in the config to the downloaded package.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input stream and render segmentation metadata. Install videos directly from the Insight catalog or through Insight's YouTube support.

In the Insight Web UI, start the required stream and copy its source URL. Use RTSP for H.264 or H.265; for MJPEG, Insight supports both RTSP and HTTP URLs. Set `source.codec` to `h264`/`avc`, `h265`/`hevc`, or `mjpeg`. Decoded frames are encoded as H.264 for Insight output.

## Configure

Edit `examples/segmentation/single-stream-instance-segmenter/src/common/config.yaml`.

```yaml
model:
  path: <model-path>
  labels: examples/segmentation/single-stream-instance-segmenter/src/common/coco_label.txt

source:
  type: rtsp
  codec: h264
  url: <rtsp-url>
  tcp: true
  fps: 0

inference:
  frames: 0
  min_score: 0.55

output:
  insight:
    host: <insight-host-ip>
    video_port: <videoUDP-start-port>
    metadata_port: <metadataUDP-start-port>
```

## Run

### C++

```bash
./examples/segmentation/single-stream-instance-segmenter/src/cpp/pre-built/single-stream-instance-segmenter \
  --config examples/segmentation/single-stream-instance-segmenter/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/single-stream-instance-segmenter/src/python/requirements.txt
python3 examples/segmentation/single-stream-instance-segmenter/src/python/main.py \
  --config examples/segmentation/single-stream-instance-segmenter/src/common/config.yaml
```

## Troubleshooting

- Verify `model.path` and the source URL if startup fails.
- Verify stream reachability if the first frame times out.
- Verify the Insight host and UDP ports if no output arrives.
- Set `output.save_dir` and `output.save_every` to save sampled frames.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config and labels: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
