# Multi-Stream People Tracker

## Metadata

| Field | Value |
| --- | --- |
| Category | tracking |
| Difficulty | Advanced |
| Tags | object-detection, yolo26, rtsp, multistream, insight, people-tracking |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multi-stream-people-tracker |
| Model | yolo26m-det-int8-b1 |

## Concept

Track people across multiple RTSP inputs with mixed-resolution support. The pipeline filters detections to the configured person class, assigns stable IDs per stream, and publishes live video and metadata to Insight.

## Preview

![Multi-stream people tracker preview](../../../portal/assets/examples/tracking/multi-stream-people-tracker/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- RTSP sources and an [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the target.

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
| `yolo26m-det-int8-b1.tar.gz` | Default |
| `yolo26n-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26s-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26l-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26x-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-det-bf16-b1.tar.gz` | Supported |

Check the installed platform version, then set `PLATFORM_VERSION` to the displayed `DISTRO_VERSION` value. Replace `<model-file>` with a file from the table.

```bash
cat /etc/buildinfo
export PLATFORM_VERSION="<platform-version>"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${PLATFORM_VERSION}/models/modalix/yolo26-detection/<model-file>"
cd ..
```

Set `model.path` in the config to the downloaded package.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input streams and render tracking metadata. Install videos directly from the Insight catalog or through Insight's YouTube support.

In the Insight Web UI, start the required streams and copy their RTSP URLs into `streams`. Use the host and UDP port ranges reported by `neat` for the output settings.

## Configure

Edit `examples/tracking/multi-stream-people-tracker/src/common/config.yaml`.

```yaml
model:
  path: <model-path>

streams:
  - <first-rtsp-url>
  - <second-rtsp-url>

inference:
  frames: 0
  max_inflight_per_stream: 4
  max_inflight_total: 16
  min_score: 0.30

tracking:
  max_missing_frames: 15

output:
  insight:
    host: <insight-host-ip>
    video_port_base: <videoUDP-start-port>
    metadata_port_base: <metadataUDP-start-port>
```

## Run

### C++

```bash
./examples/tracking/multi-stream-people-tracker/src/cpp/pre-built/multi-stream-people-tracker \
  --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/tracking/multi-stream-people-tracker/src/python/requirements.txt
python3 examples/tracking/multi-stream-people-tracker/src/python/main.py \
  --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
```

## Troubleshooting

- Start with one stream before scaling to multiple inputs.
- Verify `model.path`, every RTSP URL, and the Insight port ranges.
- Set either inflight limit to `-1` to use the Core default.
- Use `output.debug_dir` and `output.save_every` to save sampled overlays.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- C++ tracker helpers: `src/cpp/utils/`
- Python source: `src/python/main.py`
- Python tracker helpers: `src/python/utils/`
- Shared runtime files: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
