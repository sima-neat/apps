# SuperPoint Feature Extractor

## Metadata

| Field | Value |
| --- | --- |
| Category | feature-extraction |
| Difficulty | Intermediate |
| Tags | superpoint, feature-extraction, rtsp, boxdecode |
| Languages | C++, Python |
| Status | stable |
| Binary Name | superpoint-feature-extractor |
| Model | superpoint (w8a8) |

## Concept

Finds SuperPoint feature points in an RTSP stream and sends the annotated video to Insight.

## Preview

![SuperPoint feature overlay](../../../portal/assets/examples/feature-extraction/superpoint-feature-extractor/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix
  DevKit, or in a Neat SDK container paired with it.
- Neat Library with SuperPoint BoxDecode support.
- Insight serving an H.264 RTSP source and receiving the annotated output stream.
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

SiMa.ai cannot provide a compiled SuperPoint package. SuperPoint's [Magic Leap license](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE) allows noncommercial research use only and does not allow redistribution, so the model is not available for download. Compile it yourself from its recipe in the [models repository](https://github.com/sima-neat/models/tree/main/models/superpoint), on a machine with the Model Compiler, such as the Neat SDK.

Follow [Getting started](https://github.com/sima-neat/models/blob/main/docs/getting-started.md) in the models repository with model ID `superpoint` and build profile `w8a8`. That guide converts the checkpoint to ONNX and compiles it into a model package. Two inputs differ from its YOLO26n walkthrough:

- The `w8a8` build calibrates on HPatches images (`hpatches-cal` in [`model.yaml`](https://github.com/sima-neat/models/blob/main/models/superpoint/model.yaml)). The `w8a16` and `w16a16` build profiles need no calibration images.
- The compile writes `build/superpoint/w8a8/model/superpoint_w8a8_mpk.tar.gz`.

Copy that file to the DevKit, into `models/superpoint/w8a8/` under `prebuilt-apps/`. The packaged config already points `model.path` at `models/superpoint/w8a8/superpoint_w8a8_mpk.tar.gz`. For another build profile, set `model.path` to its package.

## Configure

Upload `assets/datasets/tum-rgbd/freiburg1-desk.mp4` to Insight and start it under Media Sources. The clip loops at its native 30 FPS.

Open `${APP_DIR}/src/common/config.yaml`. Set `source.rtsp_url` (for example, `rtsp://<insight-host-ip>:8554/src1`) and `output.insight.host`. `model.path` already points at the package you compiled; change it only if you saved it elsewhere or built another build profile.

Set `runtime.frames` to `0` to run until Ctrl-C. The RTSP source must keep the same resolution for the whole run.

The TUM RGB-D clip's attribution and CC BY 4.0 license are in
`assets/datasets/tum-rgbd/LICENSE.md`.

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

Both implementations stream the overlay to Insight and print the number of processed frames,
average feature count, descriptor dimension, and selected video endpoint.

## Expected Result

This application writes nothing to disk; its output is the Insight stream plus a
single summary line printed when the run ends:

```text
frames=200 average_points=331.6 descriptor_dim=256 video_sender=<insight-host>:9000
```

`frames` should match `runtime.frames`, or the number processed before Ctrl-C when it is `0`.
`descriptor_dim` should be `256`. For the packaged TUM RGB-D sequence
`average_points` runs in the low hundreds. An `average_points` near zero means
the pipeline ran but the model returned almost no features, which is worth
investigating even though the run exits successfully.

## Troubleshooting

- Confirm `model.path` points to the compiled `superpoint_w8a8_mpk.tar.gz`.
- A missing or incompatible typed SuperPoint contract fails during model construction.
- Confirm the H.264 RTSP source is running and keeps a stable resolution.
- Confirm Insight is reachable at the configured host and UDP video port.
- Set `runtime.frames: 30` for a short smoke run.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under
`src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the
[Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
