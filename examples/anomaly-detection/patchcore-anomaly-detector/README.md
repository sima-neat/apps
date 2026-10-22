# PatchCore Anomaly Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | anomaly-detection |
| Difficulty | Advanced |
| Tags | anomaly-detection, patchcore, wide-resnet50-2, memory-bank, calibration |
| Languages | C++, Python |
| Status | stable |
| Binary Name | patchcore-anomaly-detector |
| Model | patchcore_wide_resnet50_2 |

## Concept

Detects visual anomalies in industrial images or video using a compiled WideResNet-50 patch-feature extractor and a host-side coreset memory bank calibrated from your own known-good images.

## Preview

![Demo screenshot](../../../portal/assets/examples/anomaly-detection/patchcore-anomaly-detector/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- For `source.type: rtsp`, a square RTSP H.264, H.265, or MJPEG source (for example 720x720) reachable from the target.
- For `source.type: video_file` or `rtsp`, [Insight](https://developer.sima.ai/software/tools/insight/) (or another RTP receiver) to view the live annotated stream.

## Install Apps

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/anomaly-detection/patchcore-anomaly-detector
```

Run the remaining commands from `prebuilt-apps/`.

## Set Up Python

Skip this section if you only run the C++ variant. The Python variant runs in the `pyneat` Python environment that the Apps installer sets up with Neat Core (`~/pyneat` by default); it already provides NumPy, OpenCV, and PyYAML. Activate it in every new shell and confirm `pyneat` imports:

```bash
source ~/pyneat/bin/activate
python3 -c "import pyneat"
```

## Prepare the Model

```bash
mkdir -p models
cd models
export MODELZOO_VERSION="2.1.3"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/patchcore_wide_resnet50_2_bf16_mla.tar.gz"
cd ..
```

Set `model.path` in the example config to the downloaded package.

## Prepare the memory bank

Before scoring anything, build a memory bank from a directory of known-good ("nominal") images of your own inspection target. The bundled `assets/datasets/patchcore/` set exists only to make this demo reproducible without your own images, and is not a substitute for your own target's data:

- `nominal/` -- 16 real known-good reference images. `calibration.nominal_images_dir` points here by default; `--calibrate` below builds the memory bank from 13 of them and measures the threshold on the other 3.
- `held_out_normal/` -- 4 further real known-good images, deliberately excluded from calibration, used only by this example's own test suite to check that genuinely normal images the bank never saw still score below the threshold.
- `images/` -- an image-directory scoring input: one normal-looking image (`plain_0.png`) and four images with synthetic scratch defects (`scratch_0.png`-`scratch_3.png`), for exercising `source.type: image_dir` end to end.

To calibrate against your own inspection target instead, point `calibration.nominal_images_dir` at a directory of your own known-good images (a few dozen is a reasonable starting point) before running `--calibrate`. Capture them with the same camera, framing, and lighting you will run with.

The model inspects a square. Images and video frames of any size are center-cropped to their centered square (for example the middle 720x720 of a 1280x720 frame), so the left and right edges of a wide frame are not inspected; center the part. Calibration images are cropped the same way. An RTSP stream must itself be square (for example 720x720): the app exits with an error on a non-square stream, because the in-graph preprocess cannot crop it yet.

`--calibrate` keeps every 5th nominal image out of the bank and measures the threshold on those, so the threshold reflects images the bank has not memorized. Set `calibration.threshold_images_dir` to use a separate set instead.

`calibration.coreset_ratio` sets how many nominal patches the bank keeps. The default 0.0004 keeps 4 of the 10,192 patches from the demo's 13 bank images, which is enough for its uniform texture and keeps scoring fast. On a busy or varied scene a bank this small matches normal clutter poorly, which raises the threshold until real defects can go unflagged; raise the ratio (the PatchCore paper keeps 1-10%) and check that a known defect is caught. Scoring cost grows with the bank: measured on a Modalix DevKit (Platform 3.0), host scoring takes about 1-2 ms, 30 ms, and 400 ms per frame for 5, 125, and 1,254 vectors (C++ and Python are similar).

```bash
./${APP_DIR}/src/cpp/pre-built/patchcore-anomaly-detector --calibrate --config ${APP_DIR}/src/common/config.yaml
```

or, with the Python variant (after [Set Up Python](#set-up-python)):

```bash
python3 ${APP_DIR}/src/python/main.py --calibrate --config ${APP_DIR}/src/common/config.yaml
```

Both write `memory_bank.path` / `memory_bank.meta_path` from `calibration.nominal_images_dir`, and share the same on-disk format -- a bank built with one binary loads and scores in the other. Re-run `--calibrate` whenever `model.path` changes, the inspection target or camera changes, or you change any `calibration` setting.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Set `model.path`, `source.type` and its matching source fields, `memory_bank.path` / `memory_bank.meta_path`, and `calibration.nominal_images_dir` before the first `--calibrate` run. For `source.type: video_file` or `rtsp`, also set `output.insight.host` to the machine running Insight. For `source.type: rtsp` with `codec: h265` or `mjpeg`, also set `source.rtsp.width`/`height` (see Troubleshooting).

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/patchcore-anomaly-detector \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

After [Set Up Python](#set-up-python):

```bash
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

Every processed image/frame prints its score, the configured threshold, and the pass/fail verdict. `image_dir` writes overlays to `output.dir` in which only regions scoring above the calibrated patch threshold are highlighted (yellow to red); normal regions are left unchanged; `video_file`/`rtsp` additionally stream the live overlay to Insight (`output.insight.host`/`video_port`) and, if you set `output.save_every` above 0 (off by default), also save every Nth overlay frame to `output.dir`. Saving has no limit, so on a long live run pick a large N or keep it off.

```
assets/datasets/patchcore/images/plain_0.png: score=16.1795 threshold=19.1899 verdict=normal (mla=2022.1ms host=7.9ms)
assets/datasets/patchcore/images/scratch_0.png: score=32.6038 threshold=19.1899 verdict=ANOMALOUS (mla=32.3ms host=18.1ms)
...
Done: 5 images processed -- overlays written to sandbox/patchcore-anomaly-detector
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'pyneat'`: activate the `pyneat` environment (see [Set Up Python](#set-up-python)).
- `memory bank not found`: run `--calibrate` first.
- `memory bank was built against a different model package`: `model.path` changed since the last `--calibrate`; rebuild the bank.
- `bank_meta.json is missing patch_threshold`: rerun `--calibrate` to regenerate the bank.
- Confirm `source.image_dir` / `source.video_path` / `source.rtsp.url` matches the configured `source.type`.
- `rtsp`: `failed to resolve source geometry` -- the stream wasn't reachable, or (for `codec: h265`/`mjpeg`) didn't expose probeable width/height; set `source.rtsp.width`/`height` explicitly.
- Nothing appears in Insight: confirm `output.insight.host` is reachable and Insight is listening on the configured `video_port`/`channel`.
- `--calibrate` looks stalled on a large nominal set: it prints progress every 10 images; lowering `calibration.coreset_ratio` reduces both build time and per-frame scoring cost.

## Source Files

- C++ reference source: `src/cpp/main.cpp`, `src/cpp/patchcore_memory_bank.h`/`.cpp`
- Python source: `src/python/main.py`, `src/python/patchcore_scoring.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).

## Known limitations

- MVTec AD is not vendored into this repo or used for the shipped memory bank (its license is non-commercial research use); build your own bank from a nominal set captured on your own target.
- A live camera input is out of scope for this example; `source.type` covers image directories, video files, and RTSP/encoded streams only.
