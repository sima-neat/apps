# PatchCore Anomaly Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | anomaly-detection |
| Difficulty | Advanced |
| Tags | anomaly-detection, patchcore, wide-resnet50-2, memory-bank, calibration |
| Languages | C++, Python |
| Status | stable |
| Binary Name | patchcore |
| Model | patchcore_wide_resnet50_2 |

## Concept

Detects visual anomalies in industrial images or video using a compiled WideResNet-50 patch-feature extractor and a host-side coreset memory bank calibrated from your own known-good images.

## Preview

Add an application-specific image under `portal/assets/examples/anomaly-detection/patchcore/`.

```md
![Demo screenshot](../../../portal/assets/examples/anomaly-detection/patchcore/image.png)
```

## Why this shape

Every other Modalix example ends at a model whose output is already the answer -- boxes, keypoints, masks, a depth map. PatchCore is the first case in this repo where the compiled graph produces an intermediate embedding and the decision is made *after* the MLA, against calibration data collected from the deployment itself:

```
image/video/RTSP in -> wide_resnet50_2 layer2+layer3 patch-feature extractor (MLA)
                     -> per-patch nearest-neighbor distance to a coreset memory
                        bank of "normal" reference patch features (host, non-parametric)
                     -> anomaly heatmap + image-level score with neighborhood reweighting
                     -> overlay -> annotated frame out
```

- **MLA stage**: a truncated `wide_resnet50_2` (`torchvision.models.wide_resnet50_2`, `Wide_ResNet50_2_Weights.IMAGENET1K_V1`), tapping `layer2` and `layer3`, locally aggregated with a 3x3 average pool and concatenated into a 1536-dim patch embedding on a 28x28 grid (224x224 input). This is the only backbone this example ships.
- **Host stage**: coreset nearest-neighbor search against `memory_bank.npy`, anomaly-map upsampling with Gaussian smoothing (`scoring.gaussian_sigma`), and the image-level score with the PatchCore paper's neighborhood-reweighting term (`scoring.num_neighbors`) -- see `patchcore_scoring.py`.

Reference: Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (CVPR 2022).

## Generic input sources

`source.type` in the config selects the input. All three sources decode to a host-side BGR frame and score it the same way (`Model.run()`/`Runner.run()` per frame):

- **`image_dir`** -- `cv2.imread`/`cv::imread` a folder of images, one request per image.
- **`video_file`** -- `cv2.VideoCapture`/`cv::VideoCapture` on a local file, one request per frame.
- **`rtsp`** -- a decode-only Neat `Graph` (`RtspDecodedInput`) whose raw decoded frames the host pulls and scores. The model is not embedded in the live graph: a graph-embedded model route has a fixed internal buffer pool too small to survive the model's one-time warm-up stall against a continuously-arriving source, so scoring happens host-side instead, exactly like `video_file`.

## Output: live view vs stored files

- **`image_dir`** -- no live view. Every scored image's heatmap overlay is written to `output.dir`.
- **`video_file`** and **`rtsp`** -- the host draws the full heatmap overlay on each frame and pushes it live to Insight (`output.insight.host`/`video_port`). `output.save_every > 0` additionally writes the same overlay to `output.dir` every Nth frame.

See "Run" below for exactly where to look for each.

## Why the memory bank is built from the compiled model, not from PyTorch

`--calibrate` runs the *compiled, quantized* model on the DevKit to produce the embeddings the memory bank is built from -- not the float PyTorch embeddings from `torchvision`. A bank built in float shifts the distance distribution the scoring stage was calibrated against, and silently degrades score separation once the model is actually running as int8/bf16 on the MLA. Building the bank and scoring queries through the same compiled package keeps the memory bank and the runtime embeddings in the same distribution.

## The `memory_bank.npy` / `bank_meta.json` pair

`--calibrate` writes two files together:

- `memory_bank.npy` -- the coreset: a float32 `(N, 1536)` array, `N` = `calibration.coreset_ratio` of the total nominal patch count, selected by greedy k-center (farthest-point) sampling, not uniform random subsampling. This runs directly in the 1536-dim embedding space rather than the paper's random low-dimensional (Johnson-Lindenstrauss) projection -- simpler, at the cost of `--calibrate` runtime on very large nominal sets; see `greedy_coreset_indices` in `patchcore_scoring.py`.
- `bank_meta.json` -- the model package's sha256 hash, the coreset ratio and nominal-image count used, and the decision threshold with the percentile and image count it was derived from. The threshold is never hard-coded in application source; every scoring run reads it from here.

The pinned model hash means a bank built for one compiled model package fails loudly at load time if pointed at a different package, instead of silently producing meaningless scores:

```
[ERR] memory bank was built against a different model package than the one configured now
      (bank_meta.json model_sha256=..., configured model sha256=...); rebuild the bank with
      --calibrate against the current model.path
```

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- For `source.type: rtsp`, an RTSP H.264, H.265, or MJPEG source reachable from the target.
- For `source.type: video_file` or `rtsp`, [Insight](https://developer.sima.ai/software/tools/insight/) (or another RTP receiver) to view the live annotated stream.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/anomaly-detection/patchcore
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model | Role | Source |
| --- | --- | --- |
| `patchcore_wide_resnet50_2_bf16_mla.tar.gz` | Default | GitHub release (interim) |

This model is not published to the Model Zoo yet, so it is hosted as a GitHub release asset in the interim:

```bash
mkdir -p models
cd models
curl -L -o patchcore_wide_resnet50_2_bf16_mla.tar.gz \
  https://github.com/sima-neat/apps/releases/download/patchcore-model-v1/patchcore_wide_resnet50_2_bf16_mla.tar.gz
cd ..
```

Once this model is published to the Model Zoo, fetch it the same way as every other example instead:

```bash
export MODELZOO_VERSION="2.1.3"
sima-cli modelzoo -v "${MODELZOO_VERSION}" get patchcore_wide_resnet50_2_bf16_mla
```

Set `model.path` in the example config to the downloaded package.

## Prepare the memory bank

Before scoring anything, build a memory bank from a directory of known-good ("nominal") images of your own inspection target -- not `assets/datasets/coco` or any other generic set:

```bash
./${APP_DIR}/src/cpp/pre-built/patchcore --calibrate --config ${APP_DIR}/src/common/config.yaml
```

or, with the Python variant:

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py --calibrate --config ${APP_DIR}/src/common/config.yaml
```

Both write the same `memory_bank.npy` / `bank_meta.json` pair -- the two language variants share the on-disk format and the scoring math (see `support/anomaly_detection/patchcore_memory_bank.h` and `src/python/patchcore_scoring.py`), so a bank built with one binary loads and scores identically in the other. This writes `memory_bank.path` and `memory_bank.meta_path` from `calibration.nominal_images_dir`. Re-run it whenever `model.path` changes (the hash pin in `bank_meta.json` will otherwise refuse to load the bank), the inspection target changes, or you want to move the decision threshold to a different `calibration.threshold_percentile`.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Set `model.path`, `source.type` and its matching source fields, `memory_bank.path` / `memory_bank.meta_path`, and `calibration.nominal_images_dir` before the first `--calibrate` run. For `source.type: video_file` or `rtsp`, also set `output.insight.host` to the machine running Insight (change the ports/channel only if your Insight setup uses non-default values). For `source.type: rtsp` with `codec: h265` or `mjpeg`, also set `source.rtsp.width`/`height` -- see Troubleshooting.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/patchcore \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

Every processed image/frame prints its score, the configured threshold, the pass/fail verdict, and the MLA and host stage timings separately.

**`image_dir`** -- stored output only:
```
assets/datasets/patchcore/images/scratch_0.png: score=30.8839 threshold=19.4000 verdict=ANOMALOUS (mla=42.1ms host=6.7ms)
...
Done: 10 images processed -- overlays written to sandbox/patchcore
```
Open the heatmap overlays directly from `output.dir` (`sandbox/patchcore/scratch_0.png`, etc.) with any image viewer.

**`video_file`** and **`rtsp`** -- live annotated view plus optional stored snapshots:
```
streaming to Insight: 192.168.1.50:9000
frame=1: score=33.9950 threshold=8.9256 verdict=ANOMALOUS (mla=522.2ms host=52.1ms)
...
Done: 143 frames processed  video_sender=192.168.1.50:9000
```
Open the Insight web viewer pointed at `output.insight.host` to watch the heatmap-annotated stream live. If `output.save_every > 0`, the same overlay is also saved to `output.dir` (`frame_1.jpg`, `frame_2.jpg`, ...) every Nth frame. `rtsp` waits indefinitely for each frame, so a network stall pauses rather than ends the run; only a closed source stops it.

## Troubleshooting

- `memory bank not found`: run `--calibrate` first.
- `memory bank was built against a different model package`: `model.path` changed since the last `--calibrate`; rebuild the bank.
- Confirm `source.image_dir` / `source.video_path` / `source.rtsp.url` matches the configured `source.type`.
- `video_file`: `failed to open video source` -- confirm the path is reachable and that the target's OpenCV build supports the container/codec.
- `rtsp`: `failed to resolve source geometry` -- the stream wasn't reachable or didn't expose probeable width/height/fps; for `codec: h265` or `mjpeg`, set `source.rtsp.width`/`height` explicitly -- those caps aren't self-describing the way H.264 SPS is, so Neat needs the hint (H.264 does not need it).
- Nothing appears in Insight: confirm `output.insight.host` is reachable and Insight is listening on the configured `video_port`/`channel`; check for a `[FATAL]`/`Insight video push failed` line in the app's own output first.
- Confirm `output.dir` is writable (`image_dir` always, `video_file`/`rtsp` only when `save_every > 0`).
- `--calibrate` appears to hang on a large nominal set: it prints per-image progress every 10 images, so check that it's advancing rather than stalled. Greedy k-center coreset selection is O(k*n) in the nominal patch count; lowering `calibration.coreset_ratio` reduces both build time and score() cost per frame.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared host scoring stage (C++): `support/anomaly_detection/patchcore_memory_bank.h` / `.cpp`
- Host scoring stage (Python): `src/python/patchcore_scoring.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).

## Model source and versions

- Backbone: `torchvision.models.wide_resnet50_2`, weights `Wide_ResNet50_2_Weights.IMAGENET1K_V1`.
- Coreset ratio: 1% (`calibration.coreset_ratio: 0.01`) if unset; smaller values score faster at the cost of a coarser bank. The shipped `config.yaml` sets `0.006` -- measured on-device, `0.01` gave ~19.6fps on `video_file`/`rtsp` (just under a 20fps target), while `0.006` gave ~27fps with no visible loss of detection quality on the bundled test images. Recalibrate (`--calibrate`) after changing this.
- Neighborhood-reweighting support size: 9 (`scoring.num_neighbors: 9`), matching common PatchCore reference implementations.
- Anomaly-map Gaussian smoothing: sigma 4 output pixels (`scoring.gaussian_sigma: 4.0`).
- Validated with Neat 0.4.0 and Model SDK 2.1.3 on a Modalix board.

## Known limitations

- The C++ and Python scoring stages are separate implementations of the same algorithm (nearest-neighbor distance, the neighborhood-reweighting term, greedy k-center coreset, percentile threshold); they are checked to produce numerically identical scores on fixed test embeddings, but `tests/cpp/test_unit.cpp` itself only covers the CLI surface -- this repo's C++ unit-test target links `support_testing`, not `support_runtime`, so it cannot call into `patchcore_memory_bank.h` directly. The scoring math has direct unit coverage in `tests/python/test_unit.py`.
- `bank_meta.json`'s threshold is derived from whatever `calibration.nominal_images_dir` (and optionally `calibration.threshold_images_dir`) contains -- it is only as good as that nominal set, and is not validated against real defective examples by this example itself.
- MVTec AD is not vendored into this repo or used for the shipped memory bank (its license is non-commercial research use); build your own bank from a nominal set captured on your own target.
- Greedy k-center coreset selection runs in the full 1536-dim embedding space, not the paper's randomly projected lower-dimensional space, so `--calibrate` on very large nominal sets is slower than the paper's reported build times.
- A live camera input is out of scope for this example; `source.type` covers image directories, video files, and RTSP/encoded streams only.
