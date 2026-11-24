# Detection-to-VLM Assistant

## Metadata

| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Advanced |
| Tags | object-detection, yolo26, mipi, usb, insight |
| Languages | Python |
| Status | stable |
| Binary Name | detection-to-vlm-assistant |
| Model | yolo26m (w16a16) |

## Concept

Detects objects in one MIPI or USB camera stream with YOLO26 and sends synchronized video and detection metadata to Insight.

```text
How does a frame move through detection-to-vlm-assistant?

╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
┆ RUN, input.type mipi                                                              ┆
┆  ┌────────────────────────────────┐                                               ┆
┆  │ CameraInput                    │                                               ┆
┆  └────────────────────────────────┘                                               ┆
┆             │ NV12 frames                                                         ┆
┆             ▼                                                                     ┆
┆  ┌────────────────────────────────┐  NV12 frames,       ┌──────────────────────┐  ┆
┆  │ branch                         │────────────────────►│ VideoSender          │──┼─────┐
┆  └────────────────────────────────┘  latest, encoded    └──────────────────────┘  ┆     │
┆             │ NV12 frames, latest                                                 ┆     │ video
┆             ▼                                                                     ┆     │
┆  ┌────────────────────────────────┐                                               ┆     │
┆  │ YOLO26 Model                   │                                               ┆     │
┆  └────────────────────────────────┘                                               ┆     │
┆             │ BBOX tensor                                                         ┆     │
┆             ▼                                                                     ┆     │
┆  ┌────────────────────────────────┐                                               ┆     │
┆  │ Output "detections"            │                                               ┆     │
┆  └────────────────────────────────┘                                               ┆     │
╰┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯     │
              ▼ pull, decode_bbox_tensor                                                  ▼
   ┌────────────────────────────────┐                                                ┌─────────┐
   │ MetadataSender                 │───────────────────────────────────────────────►│ Insight │
   └────────────────────────────────┘   boxes as JSON                                └─────────┘

With input.type usb, the camera's own JPEG goes to Insight and only the model reads a
decoded frame:

 ┌────────────────────────────────┐  camera JPEG,       ┌──────────────────────┐
 │ UsbCameraInput                 │────────────────────►│ VideoSender          │
 └────────────────────────────────┘  latest             └──────────────────────┘
            │ camera JPEG
            ▼
 ┌────────────────────────────────┐
 │ SimaDecode                     │
 └────────────────────────────────┘
            │ NV12, or NV16 from a 4:2:2 camera, to the YOLO26 Model
```

A MIPI camera delivers raw NV12, so one branch carries the frames to the model through the camera's buffer and the other encodes them for Insight. The camera stage is checked for strict zero copy before any consumer is connected, because a composed graph describes its execution plan rather than the camera pipeline. A USB camera carries MJPEG, which goes to Insight as received while the decoder feeds the model. Each detection message carries the frame's timestamp, which Insight uses to draw the boxes on the frame they came from.

## Preview

![Detection-to-VLM assistant preview](../../../portal/assets/examples/genai/detection-to-vlm-assistant/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix DevKit, or in a Neat SDK container paired with it.
- A MIPI camera module fitted to the DevKit, or a USB camera attached to it.
- An [Insight](https://developer.sima.ai/software/tools/insight/) host reachable from the target.

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Run the example itself on the DevKit. `sima-cli neat install apps` also offers to install the compatible Neat Library and Insight, and skips them when they are up to date. To keep the Neat Library and Insight you have without being asked, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps`. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
```

```bash
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The [model browser](https://developer.sima.ai/models/) has the YOLO26 detection models in five sizes, from n (nano) to x (extra large). The default detector is `yolo26m` with the `w16a16` variant. Download it and unpack it with `sima-cli` 2.1.20 or newer:

```bash
sima-cli models download --id yolo26m --variant w16a16
```

```bash
unzip -o models/yolo26m/w16a16/yolo26m_w16a16_default.zip -d models/yolo26m/w16a16
```

The download is a ZIP file with the model package and its license. `unzip` puts the model package, `models/yolo26m/w16a16/yolo26m_w16a16_mpk.tar.gz`, next to the ZIP file.

| ID | Variants | Role |
| --- | --- | --- |
| `yolo26m` | `w8a8`, `w8a16`, `w16a16` | Default (`w16a16`) |
| `yolo26n` | `w8a8`, `w8a16`, `w16a16` | Supported |
| `yolo26s` | `w8a8`, `w8a16`, `w16a16` | Supported |
| `yolo26l` | `w8a16`, `w16a16` | Supported |
| `yolo26x` | `w8a8`, `w8a16`, `w16a16` | Supported |

To use another detector, replace `yolo26m` and `w16a16` in both commands and in `model.path`. The variant is the number format the model was compiled with: `w8a8` stores weights and activations as 8-bit integers, `w8a16` keeps 8-bit integer weights and stores activations in BF16, a 16-bit float format, and `w16a16` uses BF16 for both.

Set `model.path` to the unpacked `.tar.gz` package, not to the ZIP file.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) renders the video and the detection metadata. Use the host and UDP port ranges reported by `neat` for the output settings, and a host the target can reach, not `localhost`. The application prints the resolved source and its mode on startup.

## Configure

Open `examples/genai/detection-to-vlm-assistant/src/common/config.yaml`.

- `input.type`: `mipi` or `usb`.
- `input.source`: the MIPI camera buffer name, or a USB camera ID from `pyneat.usb_camera_ids()`.
- `input.width`, `input.height`, `input.fps`: one of the camera's modes. A camera is opened in the mode the config names, so all three are set.
- `model.path`: the unpacked detector package.
- `output.insight.host`, `output.insight.video_port_base`, `output.insight.metadata_port_base`: from `neat`.

## Run

Install the Python dependencies:

```bash
source ~/pyneat/bin/activate
```

```bash
pip install -r examples/genai/detection-to-vlm-assistant/src/python/requirements.txt
```

Start the application:

```bash
python3 examples/genai/detection-to-vlm-assistant/src/python/main.py --config examples/genai/detection-to-vlm-assistant/src/common/config.yaml
```

Stop it with Ctrl-C, or with `SIGTERM`; it closes the run and reports any dropped metadata messages.

## Expected Result

The startup banner names the camera and the Insight destination, and the run reports the frames it processed:

```text
source=camera0 type=mipi stream=1920x1080@30 insight=10.0.0.5 video=9000 metadata=9100
frames=0
```

Insight shows the camera's video with a box and a label on each detected object, drawn on the frame the detection came from.

## Troubleshooting

- `camera input is not forwarding DMA-BUFs`: the MIPI camera stage is copying frames. Check the libcamera package and the camera bridge on the target.
- `input.type must be one of mipi, usb`: the config names another input kind.
- No cameras listed by `cam -l`, or an empty `pyneat.usb_camera_ids()`: no camera is attached, or its driver did not bind.
- Video but no boxes in Insight: check `output.insight.metadata_port_base` against the ports `neat` reports.
- Verify `model.path` points to the detector package, not the ZIP file.

## Source Files

- Application: `src/python/main.py`
- Shared runtime files: `src/common/`

## Development From Source

To modify or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
