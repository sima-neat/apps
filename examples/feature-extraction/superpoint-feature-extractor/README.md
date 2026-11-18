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
| Model | superpoint (w16a16) |

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

SuperPoint is licensed by Magic Leap under the [SuperPoint Software License Agreement](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE), which restricts the software to noncommercial research use and prohibits its distribution to third parties. Redistribution of a compiled SuperPoint model package is therefore prohibited, and the model browser does not offer one for download. To run this example, compile the model from its recipe in the [models repository](https://github.com/sima-neat/models/tree/main/models/superpoint) on a machine with the Model Compiler, such as the Neat SDK. Review the license terms to confirm that they permit your intended use.

Follow [Getting started](https://github.com/sima-neat/models/blob/main/docs/getting-started.md) in the models repository with model ID `superpoint` and build profile `w16a16`. That guide converts the checkpoint to ONNX and compiles it into a model package. Three inputs differ from its YOLO26n walkthrough:

- `w16a16` calibrates on nothing, so skip the guide's dataset step and pass no `--dataset`. The whole build needs only the public Hugging Face checkpoint the recipe pins; no SiMa dataset or credential takes part in it.
- The conversion and the compile are the guide's steps 4 and 5 with `superpoint` in place of `yolo26n`:

```bash
scripts/run_model_conversion.sh models/superpoint default build/superpoint/onnx
PYTHONPATH=tools python models/superpoint/scripts/compile.py \
  --model build/superpoint/onnx/model.onnx \
  --output-dir build/superpoint/w16a16 \
  --build-profile w16a16 \
  --compile-invocation w16a16
```

- The compile writes `build/superpoint/w16a16/model/superpoint_w16a16_mpk.tar.gz`.

Copy that file to the DevKit, into `models/superpoint/w16a16/` under `prebuilt-apps/`. The packaged config already points `model.path` at `models/superpoint/w16a16/superpoint_w16a16_mpk.tar.gz`. For another build profile, set `model.path` to its package.

### Choosing a Build Profile

The recipe declares three build profiles. The accuracy and throughput columns are the values
[`models/superpoint/tests/tests.yaml`](https://github.com/sima-neat/models/blob/main/models/superpoint/tests/tests.yaml)
expects on the HPatches validation split and on a Modalix DevKit.

| Build profile | Calibration images | Keypoint repeatability | Homography accuracy | Throughput |
| --- | --- | --- | --- | --- |
| `w16a16` | none | 0.6119 | 0.8057 | 103.0 FPS |
| `w8a16` | none | 0.6042 | 0.8189 | 101.4 FPS |
| `w8a8` | 50 HPatches images | 0.6071 | 0.8226 | 193.0 FPS |

This example is packaged for `w16a16` because it is the profile anyone can build: it needs no
calibration images. Its feature quality is within two points of the other two. `w8a8` is the faster
model, at about 1.9 times the throughput, and needs the calibration images that
[Compiling `w8a8` With Your Own HPatches Images](#compiling-w8a8-with-your-own-hpatches-images)
covers. All three profiles stay ahead of a 30 FPS source such as the packaged TUM RGB-D clip.

### Compiling `w8a8` With Your Own HPatches Images

`w8a8` chooses its INT8 ranges from 50 HPatches images. SiMa CI takes them from the `hpatches-cal`
archive in SiMa's Dataset Registry, whose source is on SiMa's internal Artifactory, so
`dataset-registry` cannot download it without SiMa-internal access. `compile.py` takes the
calibration images as a plain archive path, and the registry archive is built from the public
HPatches sequences release with its files copied unchanged, so you can build the same calibration
input yourself.

1. Download the HPatches image sequences, about 1.3 GB, from the Hugging Face mirror the
   [HPatches dataset repository](https://github.com/hpatches/hpatches-dataset) links, and read its
   terms before using them:

   ```bash
   curl -L -o hpatches-sequences-release.zip \
     https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip
   unzip -q hpatches-sequences-release.zip
   ```

   It holds 116 sequences, named `i_<name>` for an illumination change and `v_<name>` for a
   viewpoint change. Each sequence holds the images `1.ppm` to `6.ppm` and the homographies between
   them.

2. Build the calibration archive. `models/superpoint/scripts/calibration.py` reads a tar archive
   and takes every `.ppm` file two levels under a directory named `cal`, then samples 50 of them
   with a fixed seed. `hpatches-cal` holds ten of the release's sequences, copied unchanged, so the
   same ten under `hpatches/cal/` reproduce the registry archive's calibration images:

   ```bash
   mkdir -p hpatches/cal
   for sequence in i_kurhaus i_lionday i_table i_whitebuilding i_zion \
                   v_apprentices v_calder v_dogman v_graffiti v_home; do
     cp -r "hpatches-sequences-release/$sequence" hpatches/cal/
   done
   COPYFILE_DISABLE=1 tar -czf hpatches-cal.tar.gz hpatches
   ```

   `COPYFILE_DISABLE=1` matters on macOS. The downloaded files carry an extended attribute, and
   macOS `tar` then writes a `._<name>.ppm` companion for each one. The compile reads every
   `.ppm` member under `hpatches/cal/<sequence>/` as an image and stops at the first companion
   with `ValueError: could not decode image`. macOS `tar -tzf` hides those members, so the
   archive looks correct while half of it is unreadable. On Linux the variable does nothing.

   Other sequences also calibrate a working model, as long as the archive holds at least 50 `.ppm`
   files under `hpatches/cal/<sequence>/`. They pick different INT8 ranges, so the package is then
   a different model from the `w8a8` row in the table above.

3. Compile `w8a8` with that archive, after the conversion step above:

   ```bash
   PYTHONPATH=tools python models/superpoint/scripts/compile.py \
     --model build/superpoint/onnx/model.onnx \
     --output-dir build/superpoint/w8a8 \
     --build-profile w8a8 \
     --compile-invocation w8a8 \
     --dataset hpatches-cal.tar.gz
   ```

   Copy `build/superpoint/w8a8/model/superpoint_w8a8_mpk.tar.gz` into
   `models/superpoint/w8a8/` under `prebuilt-apps/` and set `model.path` to it.

The ten sequences above are the ones the registry's calibration archive holds, drawn from the
sequences the 106-sequence HPatches validation split leaves out, so no sequence calibrates and
scores the same model. Whatever images you calibrate with, the compile gates the result: it
compares the quantized model's keypoints with FP32's and fails the build when keypoint recall
falls below 0.8 or descriptor similarity below 0.9.

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

- Confirm `model.path` points to the compiled `superpoint_w16a16_mpk.tar.gz`, or to the package of the build profile you compiled.
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
