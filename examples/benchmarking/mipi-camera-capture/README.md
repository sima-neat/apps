# MIPI Camera Capture

## Metadata

| Field | Value |
| --- | --- |
| Category | benchmarking |
| Difficulty | Beginner |
| Tags | camera, mipi, libcamera, zero-copy, diagnostics |
| Languages | Python |
| Status | experimental |
| Binary Name | mipi-camera-capture |
| Model | None |

## Concept

Capture NV12 frames directly from a MIPI/libcamera camera with the Neat Library `CameraInput` node. The example requests an application-owned 32-buffer capture queue, exercises the strict DMA-BUF path, and records timing and luminance diagnostics in JSON.

The camera-to-Neat path remains zero-copy. Copying occurs only for the explicitly selected snapshots written by the Python application.

## Preview

![MIPI camera capture preview](../../../portal/assets/examples/benchmarking/mipi-camera-capture/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- A MIPI camera supported by the installed kernel, libcamera pipeline, and IPA configuration.
- A Neat Library and libcamera release that supports downstream-owned camera capture buffers.

List the camera names visible to libcamera:

```bash
cam -l
```

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

This camera capture example does not use a model. No model download or compilation is required.

## Configure

Edit `examples/benchmarking/mipi-camera-capture/src/common/config.yaml`:

```yaml
camera:
  name: <camera-name-from-cam-l>
  width: 1920
  height: 1080
  fps_num: 30
  fps_den: 1
  format: NV12
  capture_buffers: 32
  strict_zero_copy: true
  queue_depth: 2

capture:
  duration_seconds: 30
  sample_times_seconds: [3, 15, 27]
  pull_timeout_ms: 2000

output:
  directory: sandbox/mipi-camera-capture
```

Leave `camera.name` empty to select libcamera's default camera. Keep `strict_zero_copy: true` to fail rather than silently copying when the downstream allocator cannot be negotiated.

## Run

```bash
source ~/pyneat/bin/activate
pip install -r examples/benchmarking/mipi-camera-capture/src/python/requirements.txt
python3 examples/benchmarking/mipi-camera-capture/src/python/main.py \
  --config examples/benchmarking/mipi-camera-capture/src/common/config.yaml
```

The output directory contains sampled NV12 frames and `summary.json`. Convert a frame for visual inspection with:

```bash
FRAME="$(find sandbox/mipi-camera-capture -name 'frame_00_*.nv12' -print -quit)"
ffmpeg -f rawvideo -pixel_format nv12 -video_size 1920x1080 \
  -i "${FRAME}" \
  -frames:v 1 sandbox/mipi-camera-capture/frame.png
```

The exact frame filename includes its observed capture time; use the filename printed by the application.

## Expected Result

The application prints `CAPTURE` records for the requested sample times and one final `SUMMARY` record. A healthy run reports frames, zero timeouts, `error: null`, and stable luminance percentiles.

## Troubleshooting

- If camera discovery fails, confirm that `cam -l` reports the configured name.
- If strict allocation negotiation fails, confirm that the installed Neat Library, libcamera plugin, and pipeline handler were built from compatible releases.
- If the requested format is adjusted or rejected, select a resolution, frame rate, and NV12 layout supported by the sensor and ISP.
- Set `strict_zero_copy: false` only when a diagnostic run may use Neat's explicit CPU-copy fallback.

## Source Files

- Python source: `src/python/main.py`
- Shared runtime configuration: `src/common/config.yaml`

## Development From Source

To modify or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
