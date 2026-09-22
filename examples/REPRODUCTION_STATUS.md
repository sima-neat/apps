# Application Reproduction Status

Tracking record for [#517](https://github.com/sima-neat/apps/issues/517): each
application reviewed, the language implementations tried, and whether the
documented workflow succeeded or was blocked.

Every result below comes from following the application's own README on real
hardware, from a clean install, using the commands exactly as written.

## Environment

| Item | Value |
| --- | --- |
| Target | SiMa.ai Modalix DevKit, `aarch64` |
| Kernel | `6.18.3-modalix #4837 SMP PREEMPT Wed Aug 19 18:44:13 UTC 2026` |
| Memory / cores | 11 GB / 16 |
| `sima-cli` | 2.1.17 |
| Apps bundle | `neat-apps-develop-1269006`, installed with `sima-cli neat install apps@develop` |
| Python runtime | `~/pyneat/bin/python` |
| Models | SDK 2.1.3 direct artifacts and Model Zoo 2.1.3 |
| Insight | `neat-insight 0.0.0+main.e3411ae` |
| Decoder | `decoder.service` active |
| Date | 2026-09-22 |

All 16 model packages referenced by the READMEs downloaded successfully with the
documented `sima-cli download` and `sima-cli modelzoo get` commands. No model
acquisition step failed.

### RTSP sources

The READMEs direct users to host RTSP sources in Insight. For a reproducible,
scriptable run the sources here were generated on the target instead:
`mediamtx` plus the packaged video assets, pre-encoded once to conforming
1280x720 H.264 (baseline profile, no B-frames, one IDR per second, 30 fps) and
republished with `ffmpeg -c copy`. A `people` stream built from the packaged
COCO images supplies faces and people where an application needs them.

## Status by application

Legend: OK = documented workflow completed. BLOCKED = could not complete.
n/a = the application does not ship that language.

| Application | C++ | Python | Observed result |
| --- | --- | --- | --- |
| `benchmarking/mipi-camera-capture` | n/a | BLOCKED | No MIPI camera attached; `cam -l` lists no cameras |
| `benchmarking/model-benchmark` | n/a | OK | Report written to `sandbox/model-benchmark/report.json` |
| `classification/image-classifier` | OK | OK | Top-5 predictions printed, validation check passed |
| `depth-estimation/depth-estimator` | OK | OK | 21 depth maps in `sandbox/depth-estimator` |
| `face-detection/face-detector` | OK | OK | 21 annotated images |
| `face-detection/single-stream-thermal-face-detector` | OK | OK | `processed=200`; streamed to Insight |
| `feature-extraction/superpoint-feature-extractor` | OK | OK | `frames=200 average_points=331.6 descriptor_dim=256` |
| `genai/detection-to-vlm-assistant` | n/a | OK | VLM served on `/v1/models`; caption returned |
| `genai/neat-genai-studio` | n/a | BLOCKED (workaround OK) | `./run.sh` aborts on the ASR model ([#521](https://github.com/sima-neat/apps/issues/521)); starts and serves chat once ASR is removed |
| `object-detection/detr-object-detector` | OK | OK | 21 annotated images |
| `object-detection/high-density-multi-stream-object-detector` | BLOCKED | BLOCKED | 8 streams OK; 16 starved 12 of them. See finding D1 and B1 |
| `object-detection/multi-stream-object-detector` | OK | OK | 4 streams, 36 saved frames |
| `object-detection/pcie-high-density-multi-stream-object-detector` | BLOCKED | BLOCKED | Requires a Modalix PCIe Card in an x86-64 Ubuntu host |
| `object-detection/rfdetr-detection-segmentation` | OK | OK | `completed=200 output_fps=30.8` |
| `object-detection/single-stream-object-detector` | OK | OK | 10 saved frames |
| `object-detection/ssd-mobilenet-object-detector` | OK | OK | 21 annotated images |
| `object-detection/yolo26-object-detector` | OK | OK | 21 annotated images |
| `pose-estimation/multi-stream-pose-estimator` | OK | OK | 4 streams, 36 saved frames |
| `segmentation/single-stream-instance-segmenter` | OK | OK | 10 saved frames |
| `segmentation/yolov8-instance-segmenter` | OK | OK | 21 annotated images |
| `tracking/multi-stream-people-tracker` | OK | OK | 4 streams, 36 saved frames |
| `tracking/yolo26-tiny-drone-tracker` | OK | OK | 9 saved frames |

## Documentation findings

**D1. High-density model path contradicts the download step.**
The README tells the user to download the model into `prebuilt-apps/models/`,
then states that relative `model.path` values resolve from the config file. The
natural value `models/yolo26n-det-int8-b1.tar.gz` therefore resolves to
`examples/object-detection/high-density-multi-stream-object-detector/src/common/models/...`
and the run fails:

```
[ERR] [io.parse] ModelPack: invalid_archive: archive path does not exist or is
not a regular file: examples/object-detection/high-density-multi-stream-object-detector/src/common/models/yolo26n-det-int8-b1.tar.gz
```

The resolution rule is documented but the working value never is. An absolute
path loads correctly. Every other application resolves `models/...` from the
Apps root, so this application is also inconsistent with the rest.

**D2. `ffmpeg` and `ffprobe` are not present on the target.**
The `mipi-camera-capture` and `high-density-multi-stream-object-detector`
READMEs instruct the user to run `ffmpeg` and `ffprobe`, but neither binary
exists on a Modalix DevKit and no README supplies an installation step.

**D3. Most READMEs never say how to recognise a successful run.**
Only `mipi-camera-capture` has an `Expected Result` section. Elsewhere the user
is told to set paths and run a command, but not what healthy output looks like,
how many files should appear, or which summary line confirms success.

**D4. Insight-only applications cannot be verified locally.**
`superpoint-feature-extractor`, `rfdetr-detection-segmentation` and
`single-stream-thermal-face-detector` write nothing to disk. Each prints a
summary line that is in practice the success signal
(`frames=200 average_points=331.6`, `completed=200 output_fps=30.8`,
`processed=200`), but no README mentions it, so a user without a working Insight
viewer has no documented way to tell a healthy run from a silent failure.

**D5. Configuration placeholders are inconsistent.**
Some packaged configs ship `path: <model-path>`; others ship a real relative
path. Every README says to set `model.path` regardless, so the reader cannot
tell from the instructions whether an edit is actually required.

**D6. `neat-genai-studio` installs from a different source (observation).**
Its install step fetches `get-example.sh` from the `main` branch, and that script
in turn defaults to fetching the example from `main`, while every other
application installs from the selected Apps release. This may be deliberate, but
it means a reader of a non-`main` README is directed at `main` content. Worth a
maintainer decision rather than a blind change.

**D7. The portal build has an undocumented Python requirement.**
`portal/README.md` lists only Node.js and npm as prerequisites, but `npm run dev`
and `npm run build` both invoke `scripts/generate_catalog.py`. That script and
`scripts/validate_readmes.py` use `X | None` annotations, which Python 3.9
rejects at import:

```
TypeError: unsupported operand type(s) for |: 'types.GenericAlias' and 'NoneType'
```

macOS ships 3.9 as the system `python3`, so the documented portal build fails
out of the box there, and the error reads like a script bug rather than a
version mismatch. `CONTRIBUTING.md` tells contributors to run both scripts
without stating a version either. The portal prerequisites also gave only `apt`
commands, which do not help on the platform where this actually bites.

## Portal comparison

The portal does not hold separately authored instructions. `generate_catalog.py`
parses each example README into `catalog.json`, and the detail page renders every
section except `Metadata` and `Concept`, which become the page header and card
summary. Portal instructions therefore match the repository README by
construction.

This was verified rather than assumed: comparing every `##` heading in all 22
READMEs against the generated catalog found no dropped or invented section, and
the portal was built and browsed locally to confirm the rendered pages. The
`Expected Result` sections added for #517 appear on the rendered pages and in the
page navigation, and placeholders such as `<insight-host>` and `<model-file>`
survive rendering instead of being consumed as HTML.

The installed application matches as well: the READMEs shipped inside
`prebuilt-apps/` are the same files as in the repository.

## Runtime defects

Recorded separately from documentation defects, as #517 asks.

**R1. `nanobind` reference-counting leak on the Python path.**
Tracked as [#522](https://github.com/sima-neat/apps/issues/522).

Five applications print the following on exit:

```
nanobind: leaked 3 instances!
nanobind: leaked 3 keep_alive records!
nanobind: leaked 1 types!
nanobind: leaked 4 functions!
nanobind: this is likely caused by a reference counting issue in the binding code.
```

Affected: `multi-stream-object-detector`, `multi-stream-people-tracker`,
`multi-stream-pose-estimator`, `single-stream-thermal-face-detector`,
`yolo26-tiny-drone-tracker`. Exit status is still 0 and results are correct.

**R2. `setup.sh` installs an ASR model the runtime cannot load.**
Tracked as [#521](https://github.com/sima-neat/apps/issues/521).

Following `get-example.sh` -> `./setup.sh` -> `./run.sh` exactly, the model
server aborts during warmup and takes the studio down with it:

```
server failed: GenAIServer warmup failed for model 'whisper-small-a16w8':
Unsupported legacy Whisper model at '/media/nvme/llima/models/whisper-small-a16w8':
found monolithic encoder 'models--openai--whisper-small_encoder_stage1_mla.elf'.
This LLiMa version requires 12 layered encoder ELF files named
'models--openai--whisper-small_encoder_layer<N>_stage1_mla.elf'.
✘ Model server exited with status 2; shutting down.
```

`setup.sh` downloads this model fresh from `simaai/whisper-small-a16w8` (48
files). Its `elf_files/` directory contains a layered decoder (12 layers) but a
single monolithic encoder ELF, so the published model and the installed LLiMa
disagree about the required layout.

Two parts to this:

1. The published `simaai/whisper-small-a16w8` needs relayering, or `setup.sh`
   needs to select a model matching the installed LLiMa.
2. The control API's completeness check does not detect the mismatch. The
   catalog reports the model as loadable right up until warmup crashes:

   ```json
   {"name": "whisper-small-a16w8", "type": "asr", "complete": true, "incompleteReason": null}
   ```

**Workaround used for this reproduction.** Removing the `genai_server.models.asr`
block from the generated `config.local.yaml` lets the studio start. Everything
else then worked: the Flask UI served on `https://<board-ip>:5000`, the RAG
service started, `/control/status` listed the catalog, `/control/load` loaded
`Qwen2.5-0.5B-Instruct-GPTQ-a16w4` in 3.7 s, `/v1/models` reflected it, and
`/v1/chat/completions` returned a correct reply. Speech-to-text is the only
unverified feature.

## Blocked workflows

**B1. 16-stream high-density profile — unresolved.**
With the model path corrected, 8 streams ran for the full 300 s with no
liveness error. At the documented 16-stream default, 12 of 16 streams never
received a frame:

```
[detector][liveness] reason=initial_stream_detection_timeout streams=16
total_pulls=9486 min_processed=0 max_processed=2372 zero_streams=12
[ERR] timed out waiting for two initial detections from streams: 1,2,4,6,8,9,10,11,12,13,14,15
```

All 16 sources probed correctly at 1280x720@30 and `decoder.service` was
active. Because the only available stream generator runs on the target itself,
a genuine device limit cannot be separated from source-side contention. This
needs an external stream host to resolve.

**B2. `pcie-high-density-multi-stream-object-detector` — hardware unavailable.**
Requires a Modalix PCIe Card hosted in an x86-64 Ubuntu machine with the SiMa.ai
PCIe driver and Neat PCIe host package. Not available.

**B3. `mipi-camera-capture` — hardware unavailable.**
`cam -l` reports no cameras. The README does not name a validated sensor part.
