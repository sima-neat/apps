# Single-Stream Object Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, yolo26, rtsp, insight |
| Languages | C++, Python |
| Status | stable |
| Binary Name | single-stream-object-detector |
| Model | yolo26m (w16a16) |

## Concept

Detects objects in one RTSP stream, HTTP MJPEG stream, or USB camera with YOLO26 and sends synchronized H.264 video and detection metadata to Insight.

The decoded frame branches inside a single graph to the detector and to the H.264 sender. Both outputs therefore carry timestamps from the same frame, which is what lets Insight draw each detection on the frame it came from. Setting `output.save_dir` adds a third branch that returns the decoded frame to the application so it can write annotated JPEGs.

## Preview

![Single-stream object detector preview](../../../portal/assets/examples/object-detection/single-stream-object-detector/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix DevKit, or in a Neat SDK container paired with it.
- An RTSP H.264, RTSP H.265, RTSP MJPEG, or HTTP MJPEG source, or a USB (UVC) camera with MJPEG or YUYV modes plugged into the DevKit, and an [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the target.

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Only the example itself must run on the DevKit. `sima-cli neat install apps` also offers to install the compatible Neat Library and Insight, and skips them when they are up to date. To keep the Neat Library and Insight you have without being asked, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps`. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/single-stream-object-detector
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The [model browser](https://developer.sima.ai/models/) has the YOLO26 detection models in five sizes, from n (nano) to x (extra large). The default is `yolo26m` with the `w16a16` variant. Download it and unpack it with `sima-cli` 2.1.20 or newer:

```bash
sima-cli models download --id yolo26m --variant w16a16
unzip -o models/yolo26m/w16a16/yolo26m_w16a16_default.zip -d models/yolo26m/w16a16
```

The download is a ZIP file with the model package and its license. `unzip` puts the model package, `models/yolo26m/w16a16/yolo26m_w16a16_mpk.tar.gz`, next to the ZIP file. Set `model.path` in the config to that `.tar.gz` file, not to the ZIP file.

| ID | Variants | Role |
| --- | --- | --- |
| `yolo26m` | `w8a8`, `w8a16`, `w16a16` | Default (`w16a16`) |
| `yolo26n` | `w8a8`, `w8a16`, `w16a16` | Supported |
| `yolo26s` | `w8a8`, `w8a16`, `w16a16` | Supported |
| `yolo26l` | `w8a16`, `w16a16` | Supported |
| `yolo26x` | `w8a8`, `w8a16`, `w16a16` | Supported |

To use another model, replace `yolo26m` and `w16a16` in both commands and in `model.path`. The variant is the number format the model was compiled with: `w8a8` stores weights and activations as 8-bit integers, `w8a16` keeps 8-bit integer weights and stores activations in BF16, a 16-bit float format, and `w16a16` uses BF16 for both. The model browser lists the measured accuracy and speed of each size and variant.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input stream and render the video and detection metadata. Install videos directly from the Insight catalog or through Insight's YouTube support.

In the Insight Web UI, start the required stream and copy its source URL. Use RTSP for H.264 or H.265; for MJPEG, Insight supports both RTSP and HTTP URLs. For RTSP, set `source.rtsp.codec` to `h264`, `h265`, or `mjpeg`. Decoded frames are encoded as H.264 for Insight output. Use a host and published port that the target can reach, not `localhost` and not an address only Insight's own machine can resolve. Verify the URL from the target before running; the application prints the resolved source and its dimensions on startup.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Set `model.path` (for the default model, `models/yolo26m/w16a16/yolo26m_w16a16_mpk.tar.gz`), `source.type` and the matching `source.rtsp`, `source.http`, or `source.usb` section (only that section is read), and the Insight host and video and metadata ports.

The source supports RTSP H.264, H.265, and MJPEG, HTTP MJPEG, and USB MJPEG or YUYV. Keep `inference.frames` at `0` to run until you stop the application.

To use a USB camera, set `source.type: usb`, `source.usb.camera` to its stable ID, `source.usb.format` to `mjpeg` (the default) or `yuyv`, and `source.usb.width`, `source.usb.height`, and `source.usb.fps` to a mode the camera lists in that format. List the connected camera IDs and modes on the DevKit:

```bash
source ~/pyneat/bin/activate
python3 -c "import pyneat; print(pyneat.usb_camera_ids())"
v4l2-ctl --list-devices && v4l2-ctl -d /dev/video<N> --list-formats-ext
```

With `mjpeg`, the camera's own JPEG goes to Insight unchanged as RTP/JPEG, with no re-encoding, so view it with an Insight version that shows JPEG channels. The same JPEG is decoded on the hardware decoder only for the model, in the camera's own sampling: most webcams send 4:2:2 MJPEG, which decodes to NV16, and the model reads it directly. RTP/JPEG carries sizes up to 2040 pixels per side in multiples of 8, so 4K modes are rejected at startup. A 1080p webcam's MJPEG is about 66-90 Mbit/s on the network.

With `source.usb.format: yuyv`, the camera sends uncompressed frames. The model reads them directly, and the video branch converts them to NV12 and encodes them as H.264 for Insight, so YUYV modes need even dimensions up to 3840x2160. Uncompressed frames use more USB bandwidth, so cameras usually list YUYV at lower resolutions or frame rates than MJPEG.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/single-stream-object-detector \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

## Expected Result

The application prints the resolved source on startup and a count when the frame
limit is reached:

```text
processed=200 video_sender=<insight-host>:9000
```

The packaged config ships `inference.frames: 0`, which runs continuously. The
closing `processed=` line prints only when a finite limit is reached, and
interrupting the run with Ctrl-C skips it. For a bounded check that ends by
itself, set a positive limit first:

```yaml
inference:
  frames: 200
```

Left at `0`, the live Insight stream is the success signal instead. Either way,
to confirm detections without watching Insight, set `output.save_dir` to a
directory and `output.save_every` to a non-zero interval; the application then
writes annotated frames there as it runs.

## Troubleshooting

- Verify stream reachability if the first frame times out.
- For a USB camera, an unknown ID or an unsupported mode fails at startup with the connected IDs or the camera's modes in the selected format; close other programs that hold the camera.
- Verify the Insight host and UDP ports if video and detections are absent.
- Check model loading and score thresholds when video arrives without boxes.
- If the model fails to load, check that `model.path` points to the `_mpk.tar.gz` file unpacked from the download, not to the ZIP file.
- The startup probe adapts the decode path when source resolution changes.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
