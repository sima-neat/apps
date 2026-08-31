# SuperPoint Feature Extractor

## Metadata

| Field | Value |
| --- | --- |
| Category | feature-extraction |
| Difficulty | Intermediate |
| Tags | superpoint, feature-extraction, video, boxdecode |
| Languages | C++, Python |
| Status | stable |
| Binary Name | superpoint-feature-extractor |
| Model | superpoint / modalix_int8_tessellation_mla |

## Concept

Finds SuperPoint feature points in a video and streams the annotated video to Insight.

## Preview

Frame from the included TUM RGB-D `freiburg1_desk` sequence:

![SuperPoint feature overlay](../../../portal/assets/examples/feature-extraction/superpoint-feature-extractor/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported
  Modalix or DevKit target.
- Neat Library with SuperPoint BoxDecode support.
- Insight or another RTP receiver for the annotated output stream.
- The SuperPoint model package installed below.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/feature-extraction/superpoint-feature-extractor
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model | Role | Source |
| --- | --- | --- |
| `superpoint_mpk.tar.gz` | Default | Direct artifact |

The model package comes from the Model Zoo release below, which can differ from the installed platform version.

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/superpoint_mpk.tar.gz"
cd ..
```

The downloaded filename matches the path used by the packaged config. The model expects 640x480 grayscale input.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Set `model.path`, `io.input`, and `output.insight.host`. Change the Insight port or channel only if your Insight setup uses different values.

Set `runtime.frames` to `0` to process the full video. The input must keep the same resolution for the whole run.

The included sequence comes from the TUM RGB-D visual-SLAM benchmark. Its camera motion and office
scene provide repeatable local features for SuperPoint to extract. The source, attribution, transformation, and CC BY 4.0 license are documented in
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

## Troubleshooting

- Confirm `model.path` points to the qualified SuperPoint package.
- A missing or incompatible typed SuperPoint contract fails during model construction.
- Confirm the input is a stable-resolution BGR video supported by the target hardware.
- Confirm Insight is reachable at the configured host and UDP video port.
- Set `runtime.frames: 30` for a short smoke run.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`
- C++ tests: `tests/cpp/`
- Python tests: `tests/python/`

The packaged C++ source is an implementation reference. Run the executable under
`src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the
[Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
