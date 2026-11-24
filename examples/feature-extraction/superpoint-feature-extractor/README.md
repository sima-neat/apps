# SuperPoint Feature Extractor

## Metadata

| Field | Value |
| --- | --- |
| Category | feature-extraction |
| Difficulty | Intermediate |
| Tags | superpoint, feature-extraction, rtsp, insight |
| Languages | C++, Python |
| Status | stable |
| Binary Name | superpoint-feature-extractor |
| Model | superpoint (w8a8) |

## Concept

Finds SuperPoint feature points in one H.264 RTSP stream and sends the video and the points to Insight.

```text
How does a frame move through superpoint-feature-extractor?

╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
┆ RUN                                                                ┆
┆  ┌──────────────────────┐  H.264 as received  ┌──────────────────┐ ┆
┆  │ RtspEncodedInput     │────────────────────►│ VideoSender      │─┼────┐
┆  └──────────────────────┘                     └──────────────────┘ ┆    │
┆             │ encoded frames                                       ┆    │ video
┆             ▼                                                      ┆    │
┆  ┌──────────────────────┐                                          ┆    │
┆  │ SimaDecode           │                                          ┆    │
┆  └──────────────────────┘                                          ┆    │
┆             │ NV12 frames, latest                                  ┆    │
┆             ▼                                                      ┆    │
┆  ┌──────────────────────┐                                          ┆    │
┆  │ Output "frames"      │                                          ┆    │
┆  └──────────────────────┘                                          ┆    │
╰┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯    │
              ▼ pull, Y plane as gray                                     │
╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮                                          │
┆ MODEL RUNNER                 ┆                                          │
┆  ┌──────────────────────┐    ┆                                          │
┆  │ SuperPoint Model     │    ┆                                          │
┆  └──────────────────────┘    ┆                                          │
╰┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯                                          │
              ▼ FEATURE_POINTS_V1 tensors                                 │
   ┌──────────────────────┐                                               │
   │ feature_points       │                                               │
   └──────────────────────┘                                               │
              │ source-frame points                                       │
              ▼                                                           ▼
   ┌──────────────────────┐   points as pose JSON                  ┌──────────────┐
   │ MetadataSender       │───────────────────────────────────────►│ Insight      │
   └──────────────────────┘                                        └──────────────┘
```

The H.264 stream goes to Insight as received. The decoder's NV12 frames go to the application. The model's preprocessing converts no YUV format to gray. So the application hands the model each frame's Y plane, which is the gray image SuperPoint reads. `feature_points` maps the keypoints from the model's 640x480 input back onto the stream's frame.

Each frame's points go to Insight's pose-estimation overlay as the keypoints of one pose. Insight draws them as dots, with no skeleton or box, on the frame whose timestamp they carry. The `frames` output keeps only the newest frame. A slow loop then skips frames instead of holding back the video.

## Preview

![SuperPoint feature overlay](../../../portal/assets/examples/feature-extraction/superpoint-feature-extractor/image.png)

The frame is from the [TUM RGB-D Benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) sequence `freiburg1_desk` by Sturm et al., licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); an earlier version of this example drew the points and their count.

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix DevKit, or in a Neat SDK container paired with it.
- An H.264 RTSP source and an [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the target.
- A SuperPoint model package you compiled, as Prepare the Model describes.

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Only the example itself must run on the DevKit. `sima-cli neat install apps` also offers to install the compatible Neat Library and Insight, and skips them when they are up to date. To keep the Neat Library and Insight you have without being asked, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps`. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/feature-extraction/superpoint-feature-extractor
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Magic Leap licenses SuperPoint under the [SuperPoint Software License Agreement](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE). It restricts the software to noncommercial research use and prohibits its distribution to third parties. Redistribution of a compiled SuperPoint model package is therefore prohibited, and the model browser does not offer one for download. To run this example, compile the model from its recipe in the [models repository](https://github.com/sima-neat/models/tree/main/models/superpoint). Use a machine with the Model Compiler, such as the Neat SDK. Review the license terms to confirm that they permit your intended use.

Follow [Getting started](https://github.com/sima-neat/models/blob/main/docs/getting-started.md) in the models repository with model ID `superpoint` and build profile `w8a8`. That guide converts the checkpoint to ONNX and compiles it into a model package. Two things differ from its YOLO26n walkthrough:

- The `w8a8` build calibrates on HPatches images (`hpatches-cal` in [`model.yaml`](https://github.com/sima-neat/models/blob/main/models/superpoint/model.yaml)). The `w8a16` and `w16a16` build profiles need no calibration images.
- The compile writes `build/superpoint/w8a8/model/superpoint_w8a8_mpk.tar.gz`.

Copy that file to the DevKit, into `models/superpoint/w8a8/` under `prebuilt-apps/`. From `prebuilt-apps/`:

```bash
mkdir -p models/superpoint/w8a8
cp <path>/superpoint_w8a8_mpk.tar.gz models/superpoint/w8a8/
```

The packaged `model.path` already points to it. A relative `model.path` starts at the config file. From `${APP_DIR}/src/common/`, `../../../../../models/` is `prebuilt-apps/models/`. For another build profile, set `model.path` to its package, or to the absolute path of the `.tar.gz` file.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input stream and show the video with the feature points. Install a video from the Insight catalog or through Insight's YouTube support, start it as an RTSP stream, and copy its URL into `input.source`. Use a host and published port that the target can reach, not `localhost`.

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set:

- `input.source`: the H.264 RTSP URL.
- `input.width`, `input.height`, and `input.fps`: leave each at 0 to read it from the stream. Set a value only for a stream that does not report it.
- `output.insight.host`, and the Insight ports `output.insight.video_port_base` and `output.insight.metadata_port_base`.

Keep `inference.frames` at `0` to run until you stop the application.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/superpoint-feature-extractor \
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

The application prints the source, the stream's size and frame rate, and the Insight ports on startup. When it stops, it prints the frames it processed, the average feature points per frame, the descriptor length, and the video endpoint:

```text
source=rtsp://<rtsp-source-ip>:<port>/<stream-name> stream=1280x720@30 insight=<insight-host-ip> video=9000 metadata=9100
frames=200 average_points=600.0 descriptor_dim=256 video_sender=<insight-host-ip>:9000
```

The packaged config ships `inference.frames: 0`, which runs until Ctrl-C. The summary then prints when you stop it. For a run that ends by itself, set `inference.frames: 200`. `average_points` is at most 600, the decoder's limit per frame, and a textured scene reaches it. An `average_points` near zero means the model found almost no features. Investigate it, even though the run exits successfully. A line `<N> metadata messages dropped` follows the summary when the socket was full and dropped some frames' points.

Insight shows the video with a dot on every feature point.

## Troubleshooting

- Confirm `model.path` points to the compiled `superpoint_w8a8_mpk.tar.gz`.
- A missing or incompatible SuperPoint contract fails during model construction.
- Verify the Insight host and UDP ports if the video or the points are absent.
- `frame size <W>x<H> differs from the stream's <W>x<H>` appears when the stream changes resolution during the run. The stream must keep one size. Restart the application after a change.
- `failed to probe RTSP frame size: <url>` or `failed to probe RTSP frame rate: <url>` appears when a value left at 0 cannot be read. Either the source is unreachable from the board, or it does not report that value. Verify the URL from the board itself. For a reachable source, set the value the message names.

## Source Files

- C++ reference source: `src/cpp/main.cpp` and `src/cpp/utils/feature_points.cpp`
- Python source: `src/python/main.py` and `src/python/utils/feature_points.py`
- Shared config: `src/common/config.yaml`
- Helpers the examples share: `support/python/sima_examples/` at the root of the installed bundle, which `main.py` finds in a folder above itself. Their C++ reference source is in `support/cpp/sima_examples/`.

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`. The installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
