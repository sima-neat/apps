# SuperPoint Feature Extractor

## Metadata

| Field | Value |
| --- | --- |
| Category | feature-extraction |
| Difficulty | Intermediate |
| Tags | superpoint, feature-extraction, video, boxdecode |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | superpoint-feature-extractor |
| Model | superpoint / modalix_int8_tessellation_mla |

## Concept

Finds SuperPoint feature points in a video and streams the annotated video to Insight.

The application selects the A65V1 numerical profile explicitly. Tensor roles, output dtypes, and
storage layouts come from the MPK contract rather than being inferred from tensor order or values.

## Preview

Frame from the included TUM RGB-D `freiburg1_desk` sequence:

![SuperPoint feature overlay](../../../portal/assets/examples/feature-extraction/superpoint-feature-extractor/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported
  Modalix or DevKit target.
- Neat Library with SuperPoint BoxDecode support.
- Insight or another RTP receiver for the annotated output stream.
- A qualified SuperPoint MPK as described below.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/feature-extraction/superpoint-feature-extractor
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The [Neat Model Registry](https://github.com/sima-neat/models/issues/24) publishes SuperPoint
through the staging Vulcan artifact registry. Install the current model-matrix package and select
the INT8 variant that keeps tessellation inside the MLA:

```bash
mkdir -p models/superpoint
sima-cli neat install --stg \
  models/superpoint@codex/superpoint-model-matrix:latest \
  --install-dir models/superpoint

cp models/superpoint/superpoint_modalix_int8_tessellation_mla_mpk.tar.gz \
  models/superpoint_mpk.tar.gz
```

Do not download the model from an ad hoc attachment or copy; the registry package supplies the
immutable artifact and verifies its published checksum.

The default CI pipeline test uses `modalix_int8_tessellation_mla`, while the accuracy matrix covers
all four INT8/BF16 and MLA/EV74 tessellation combinations. The INT8 MLA archive was calibrated with
128 deterministic images (80 COCO val2017, 32 HPatches, and 16 TUM RGB-D), contains one MLA
program, and has this SHA-256 checksum:

```text
768f8f2838b335ffa92fd4d2464730b61a4bcdf4190484aac125b7395e271d53
```

Verify the selected package when reproducing the reference qualification:

```bash
sha256sum models/superpoint_mpk.tar.gz
```

The model accepts a normalized 640x480 grayscale input and publishes a 65-channel detector head
and a 256-channel descriptor head. The registry also provides BF16 and EV74-tessellation variants.

Configure `model.path` after the model is published under a different filename.

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
