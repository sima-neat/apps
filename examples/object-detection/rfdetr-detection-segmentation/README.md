# RF-DETR Detection and Segmentation

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, instance-segmentation, rfdetr, rtsp, insight |
| Languages | C++, Python |
| Status | stable |
| Binary Name | rfdetr-detection-segmentation |
| Model | RF-DETR Nano, Small, Medium, or Large detection; RF-DETR Seg Nano, Small, or Medium segmentation |

## Concept

Run RF-DETR detection or instance segmentation on one H.264, H.265, or MJPEG RTSP stream and view the result in Insight.

```text
How does a frame move through rfdetr-detection-segmentation?

╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
┆ RUN                                                             ┆
┆  ┌──────────────────────┐  encoded, as received ┌─────────────┐ ┆
┆  │ RtspEncodedInput     │──────────────────────►│ VideoSender │─────┐
┆  └──────────────────────┘                       └─────────────┘ ┆   │
┆             │ encoded frames                                    ┆   │ video
┆             ▼                                                   ┆   │
┆  ┌──────────────────────┐                                       ┆   │
┆  │ SimaDecode           │                                       ┆   │
┆  └──────────────────────┘                                       ┆   │
┆             │ NV12, newest 4                                    ┆   │
┆             ▼                                                   ┆   │
┆  ┌──────────────────────┐                                       ┆   │
┆  │ Backbone Model       │                                       ┆   │
┆  └──────────────────────┘                                       ┆   │
┆             │ features, proposals                               ┆   │
┆             ▼                                                   ┆   │
┆  ┌──────────────────────┐                                       ┆   │
┆  │ Output "backbone"    │                                       ┆   │
┆  └──────────────────────┘                                       ┆   │
╰┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯   │
              ▼ pull                                                  │
   ┌────────────────────────────┐                                     │
   │ top proposals (utils/)     │                                     │
   └────────────────────────────┘                                     │
              │ boxes, features                                       │
              ▼                                                       │
   ┌────────────────────────────┐                                     │
   │ Transformer Runner         │                                     │
   └────────────────────────────┘                                     │
              │ logits, boxes, masks                                  │
              ▼                                                       │
   ┌────────────────────────────┐                                     │
   │ results (utils/)           │                                     │
   └────────────────────────────┘                                     │
              │ boxes or polygons                                     │
              ▼                                                       ▼
   ┌────────────────────────────┐  JSON                        ┌──────────────┐
   │ MetadataSender             │─────────────────────────────►│ Insight      │
   └────────────────────────────┘                              └──────────────┘
```

Each RF-DETR model comes as two packages: a backbone and a decoder, which the code calls the transformer. The application decodes the stream to NV12 once. A 4-frame queue replaces stale decoded frames when inference falls behind. EV74 converts, resizes, and normalizes admitted frames for the backbone. Host code selects the backbone's strongest proposals and passes their boxes and the backbone's feature tensor to the transformer. Insight receives the video and the matching detection boxes or segmentation polygons.

The backbone output keeps up to eight results, and the transformer input waits while its four-entry queue is full, so completed backbone work is not dropped. These capacities do not include buffers held by active stages.

TCP sources use RTP timestamps directly to avoid arrival-time corrections during high-rate replay. H.264, H.265, and MJPEG video reach Insight as encoded passthrough, so the video is never encoded again.

## Preview

![RF-DETR detection and segmentation preview](../../../portal/assets/examples/object-detection/rfdetr-detection-segmentation/image.jpg)

## Prerequisites

- [`sima-cli` 2.1.20 or newer](https://developer.sima.ai/software/tools/sima-cli/), the first release that downloads from the model browser by default, on the Modalix DevKit, or in a Neat SDK container paired with it.
- An H.264, H.265, or MJPEG RTSP source and an [Insight endpoint](https://developer.sima.ai/software/tools/insight/) reachable from the target.

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Only the example itself must run on the DevKit. `sima-cli neat install apps` also offers to install the compatible Neat Library and Insight, and skips them when they are up to date. To keep the Neat Library and Insight you have without being asked, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps`. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/rfdetr-detection-segmentation
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The model packages come from the model browser. `model.id` in the config names
the variant, and `model.path` names its backbone package. The application loads
the decoder package from the same folder, with `_backbone_mpk` in the file name
replaced by `_decoder_mpk`. The **Default build** column is the build each
`model.id` was tested with.

| `model.id` | Model browser ID | Default build | Other builds | Input | Queries | Mask |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `det-nano` | `rf_detr_nano` | `w8a8_mixed_w16a16` | `w8a16_w16a16` | 384 | 300 | — |
| `det-small` | `rf_detr_small` | `w8a8_mixed_w16a16` | `w8a16_w16a16` | 512 | 300 | — |
| `det-medium` | `rf_detr_medium` | `w8a8_mixed_w16a16` | `w8a16_w16a16` | 576 | 300 | — |
| `det-large` | `rf_detr_large` | `w8a8_mixed_w16a16` | `w8a16_w16a16` | 704 | 300 | — |
| `seg-nano` | `rf_detr_seg_nano` | `w8a16_w8a8_mixed` | `w8a16_w16a16`, `w8a8_mixed_w16a16`, `w8a8_w16a16` | 312 | 100 | 78×78 |
| `seg-small` | `rf_detr_seg_small` | `w8a16_w8a8_mixed` | `w8a16_w16a16`, `w8a8_mixed_w16a16`, `w8a8_w16a16` | 384 | 100 | 96×96 |
| `seg-medium` | `rf_detr_seg_medium` | `w8a16_w8a8_mixed` | `w8a16_w16a16`, `w8a8_mixed_w16a16`, `w8a8_w16a16` | 432 | 200 | 108×108 |

The application reads the input size, query count, and mask size from the
loaded packages, so every variant runs with the same code.

Download the variant you selected and extract it where it lands. The packaged
config selects `det-small`:

```bash
MODEL=rf_detr_small
BUILD=w8a8_mixed_w16a16
sima-cli models download --id ${MODEL} --variant ${BUILD}
unzip -o models/${MODEL}/${BUILD}/${MODEL}_${BUILD}_default.zip -d models/${MODEL}/${BUILD}
```

For segmentation, `seg-medium` for example:

```bash
MODEL=rf_detr_seg_medium
BUILD=w8a16_w8a8_mixed
sima-cli models download --id ${MODEL} --variant ${BUILD}
unzip -o models/${MODEL}/${BUILD}/${MODEL}_${BUILD}_default.zip -d models/${MODEL}/${BUILD}
```

`sima-cli models download` saves `<model>_<build>_default.zip`. Extracting it
gives `<model>_<build>_backbone_mpk.tar.gz` and `<model>_<build>_decoder_mpk.tar.gz`
in the same folder. Keep both files there.

## Prepare an RTSP source

1. In Insight, open **Media Sources** and import a video from the catalog or your own file.
2. Assign the video under **Streaming Sources** and start the source.
3. Copy its RTSP URL into `input.source` and set `input.codec` to match the video. Use a host and published port reachable from the target, not `localhost`.

## Configure

Edit `$APP_DIR/src/common/config.yaml`:

- Set `model.id` to the variant you downloaded and `model.path` to its backbone
  package. A relative `model.path` starts at the config file, so for `seg-medium`
  it is `../../../../../models/rf_detr_seg_medium/w8a16_w8a8_mixed/rf_detr_seg_medium_w8a16_w8a8_mixed_backbone_mpk.tar.gz`.
- Set `input.source`, and `input.codec` to `h264`, `h265`, or `mjpeg`.
- Leave `input.width`, `input.height`, and `input.fps` at 0 to read the frame
  size and rate from the stream, so a stream of another size needs no config change.
- Set `output.insight.host`, `video_port_base`, and `metadata_port_base` to the
  values reported by Insight.
- For segmentation, set `inference.min_score: 0.3` and
  `inference.max_detections: 24`.
- Keep `inference.frames: 0` to run until Ctrl-C, or set a result count.

Segmentation traces outlines on a 640 grid (`app.mask_grid_size`). Set it to the
model's mask size from the table above (78, 96, or 108) for native contours with
less CPU work.

## Run

### C++

```bash
"$APP_DIR/src/cpp/pre-built/rfdetr-detection-segmentation" --config "$APP_DIR/src/common/config.yaml"
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r "$APP_DIR/src/python/requirements.txt"
python3 "$APP_DIR/src/python/main.py" --config "$APP_DIR/src/common/config.yaml"
```

Insight receives `object-detection` metadata for `det-*` models or `segmentation`
polygon metadata for `seg-*` models. Stop a continuous run with Ctrl-C.

## Expected Result

This application writes nothing to disk; its output is the Insight stream plus a
startup line and a closing line, the same from C++ and Python:

```text
RF-DETR det-small h264: rtsp://<host>:<port>/<stream> (1280x720@30) -> Insight <insight-host-ip> video=9000 metadata=9100
RF-DETR det-small: sent 200 results to Insight
```

The startup line names the model, codec, frame size, and frame rate the
application runs with. The closing line counts the results sent to Insight, and
prints after Ctrl-C too. When the nonblocking metadata sender dropped messages
because the socket buffer was full, a further line counts them.

## Troubleshooting

- `ModelPack: invalid_archive: archive path does not exist` means `model.path`
  resolved from the config file to a missing file, or the decoder package
  (`*_decoder_mpk.tar.gz`) is not next to the backbone. Point `model.path` at the
  backbone and extract both files from the same download into one folder.
- `input.codec: unknown value '<codec>' (expected h264, h265, mjpeg)` means
  `input.codec` names a codec this example does not decode. Use one of those listed.
- `<config> is missing <key>` or `<key> must be <type>` names a key the config
  lacks or sets to the wrong type. Compare it with the packaged config.
- `failed to probe RTSP frame size: <url>` or `failed to probe RTSP frame rate: <url>`
  appears when a value left at 0 cannot be read: the source is unreachable from the
  board, or it does not report that value. Verify the URL from the board itself; the
  URL Insight displays is not always reachable from the target. For a reachable source,
  set the value the message names.

## Performance

End-to-end throughput measured on Modalix with the Python application and no other workload on the board. Each figure is the average rate of inference results, in FPS, over at least 20 seconds of continuous processing of a 720p input stream running at 300 FPS. Every variant runs its default build, and segmentation sets `app.mask_grid_size` to the model's mask size.

| Input resolution | Codec | Nano detector | Small detector | Medium detector | Large detector | Nano segmenter | Small segmenter | Medium segmenter |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 720p | H.264 | 189 | 94 | 72 | 35 | 87 | 74 | 32 |
| 720p | H.265 | 192 | 98 | 72 | 35 | 87 | 73 | 31 |
| 720p | MJPEG | 195 | 98 | 72 | 35 | 87 | 74 | 32 |

The nano detector's MJPEG figure was measured with the video sent to Insight as received. The other MJPEG figures were measured before that change, with `input.fps: 120` and the video encoded again as H.264.

## Source Files

- C++ reference source: `src/cpp/main.cpp`, with the postprocessing in `src/cpp/utils/postprocess.cpp`
- Python implementation: `src/python/main.py`, with the postprocessing in `src/python/utils/postprocess.py`
- Shared configuration and COCO labels: `src/common/`
- Helpers shared with other examples, with no Neat API calls: `support/python/sima_examples/` in `prebuilt-apps/`, and their C++ reference source in `support/cpp/sima_examples/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
